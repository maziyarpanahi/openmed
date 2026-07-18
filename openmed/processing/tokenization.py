"""Tokenization and hardened dictionary-ingestion utilities for OpenMed."""

import hashlib
import logging
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Tuple

try:
    from transformers import PreTrainedTokenizer

    HF_AVAILABLE = True
except (ImportError, OSError):
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)

_UNSET_MAX_LENGTH_SENTINEL = 1_000_000
_DICTIONARY_READ_CHUNK_BYTES = 64 * 1024
_REGEX_CONSTRUCT_CHARS = frozenset(r".^$*+?{}[]\|()")
_ALLOWED_DICTIONARY_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
DEFAULT_MEDICAL_EXCEPTIONS = [
    "COVID-19",
    "SARS-CoV-2",
    "IL-6",
    "IL-2",
    "TNF-alpha",
    "BCR-ABL1",
    "CAR-T",
    "post-CAR-T",
    "t(8;21)",
    "t(15;17)",
]

_MEDICAL_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:°?[CFK])?"  # numbers/decimals with optional temperature unit (39.8C)
    r"|[A-Za-z]+(?:-[A-Za-z0-9]+)*"  # words, allowing hyphen chains (IL-6-mediated, COVID-19)
    r"|[A-Za-z0-9]+(?:/[A-Za-z0-9µ]+)+"  # ratios like mg/kg, mmHg/...
    r"|[^ \t\r\n]"  # any other non-space char as its own token
)


@dataclass(frozen=True)
class DictionaryLimits:
    """Resource and entry limits for untrusted dictionary files.

    The defaults bound both work and memory while remaining large enough for
    operator-supplied clinical dictionaries. Callers may choose lower limits,
    but disabling a limit is intentionally unsupported.
    """

    max_compressed_bytes: int = 16 * 1024 * 1024
    max_decompressed_bytes: int = 64 * 1024 * 1024
    max_entries: int = 100_000
    max_entry_bytes: int = 4 * 1024
    max_term_characters: int = 256
    max_expansion_ratio: float = 100.0

    def __post_init__(self) -> None:
        """Reject non-positive limits instead of silently removing a guard."""
        values = {
            "max_compressed_bytes": self.max_compressed_bytes,
            "max_decompressed_bytes": self.max_decompressed_bytes,
            "max_entries": self.max_entries,
            "max_entry_bytes": self.max_entry_bytes,
            "max_term_characters": self.max_term_characters,
            "max_expansion_ratio": self.max_expansion_ratio,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"Dictionary limit {name} must be positive")


DEFAULT_DICTIONARY_LIMITS = DictionaryLimits()


@dataclass(frozen=True)
class UserDictionaryEntry:
    """One validated literal entry from a segmenter user dictionary."""

    term: str
    frequency: int | None = None
    pos: str | None = None


class DictionaryIngestionError(ValueError):
    """Base class for fail-closed dictionary ingestion errors."""

    reason = "dictionary_rejected"


class DictionarySourceError(DictionaryIngestionError):
    """Raised when a dictionary source cannot be read safely."""

    reason = "source_error"


class DictionaryArchiveError(DictionaryIngestionError):
    """Raised when a dictionary archive has an unsupported or invalid shape."""

    reason = "archive_error"


class DictionarySizeLimitError(DictionaryIngestionError):
    """Raised when compressed or decompressed bytes exceed a configured cap."""

    reason = "size_limit"


class DictionaryExpansionLimitError(DictionarySizeLimitError):
    """Raised before reading an archive member with excessive expansion."""

    reason = "expansion_ratio"


class DictionaryEntryLimitError(DictionaryIngestionError):
    """Raised as soon as a dictionary crosses its entry-count cap."""

    reason = "entry_limit"

    def __init__(self, observed_count: int) -> None:
        self.observed_count = observed_count
        super().__init__("Dictionary entry count exceeds the configured limit")


class DictionaryEncodingError(DictionaryIngestionError):
    """Raised when a dictionary is not strict UTF-8."""

    reason = "utf8_required"


class DictionaryEntryValidationError(DictionaryIngestionError):
    """Raised when one entry violates a named validation rule."""

    reason = "entry_validation"

    def __init__(self, rule: str, line_number: int) -> None:
        self.rule = rule
        self.line_number = line_number
        super().__init__(
            f"Dictionary entry rejected by rule={rule} at line={line_number}"
        )


@dataclass
class _DictionaryMetadata:
    path_hash: str
    size_bytes: int
    entry_count: int = 0


def load_user_dictionary(
    path: str | Path,
    *,
    limits: DictionaryLimits = DEFAULT_DICTIONARY_LIMITS,
) -> tuple[UserDictionaryEntry, ...]:
    """Load a strict UTF-8 user dictionary through bounded streaming.

    Plain-text files and single-member ``.zip`` archives are supported. Archive
    metadata is checked before member decompression, and both source forms stop
    as soon as a byte, record, or entry limit is crossed. Entries use the jieba
    shape ``term [frequency [POS]]`` but terms are always literals: Unicode
    controls and executable regular-expression constructs are rejected.

    Rejection logs contain only a hash of the source path, byte size, entry
    count, and a machine-readable reason. Dictionary content and raw paths are
    never logged or included in raised exceptions.

    Args:
        path: Path to a plain UTF-8 file or a single-member ZIP archive.
        limits: Positive resource and entry validation limits.

    Returns:
        Validated, immutable dictionary entries.

    Raises:
        DictionaryIngestionError: If the source, archive, encoding, resource
            use, or any entry fails validation.
    """

    source_path = Path(path).expanduser()
    metadata = _DictionaryMetadata(
        path_hash=_dictionary_path_hash(source_path),
        size_bytes=_safe_file_size(source_path),
    )
    try:
        is_archive = source_path.suffix.casefold() == ".zip" or zipfile.is_zipfile(
            source_path
        )
        source_size_limit = (
            limits.max_compressed_bytes if is_archive else limits.max_decompressed_bytes
        )
        if metadata.size_bytes > source_size_limit:
            _reject_dictionary(
                DictionarySizeLimitError(
                    "Dictionary source exceeds the configured size limit"
                ),
                metadata,
            )
        if is_archive:
            return _load_zipped_dictionary(source_path, limits, metadata)
        with source_path.open("rb") as handle:
            return _parse_dictionary_stream(handle, limits, metadata)
    except DictionaryIngestionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile):
        _reject_dictionary(
            DictionarySourceError("Dictionary source is unreadable"), metadata
        )


def validate_user_dictionary_entry(
    line: str,
    *,
    line_number: int = 1,
    limits: DictionaryLimits = DEFAULT_DICTIONARY_LIMITS,
) -> UserDictionaryEntry | None:
    """Validate one dictionary line without logging its content.

    Blank lines and comment-only lines return ``None``. This helper is useful to
    downstream loaders that already own a bounded byte stream; path-based
    callers should prefer :func:`load_user_dictionary` so rejections receive
    PHI-safe file metadata.
    """

    if not isinstance(line, str):
        raise TypeError("Dictionary entry must be text")
    if line_number <= 0:
        raise ValueError("line_number must be positive")

    candidate = line.split("#", 1)[0].strip()
    if not candidate:
        return None
    fields = candidate.split()
    if len(fields) > 3:
        raise DictionaryEntryValidationError("field_count", line_number)

    term = fields[0]
    _validate_dictionary_field(term, "term", line_number, limits)
    if any(char in _REGEX_CONSTRUCT_CHARS for char in term):
        raise DictionaryEntryValidationError("executable_regex_construct", line_number)

    frequency: int | None = None
    if len(fields) >= 2:
        try:
            frequency = int(fields[1], 10)
        except ValueError:
            raise DictionaryEntryValidationError(
                "frequency_integer", line_number
            ) from None
        if frequency <= 0:
            raise DictionaryEntryValidationError("frequency_positive", line_number)
        if frequency > 2_147_483_647:
            raise DictionaryEntryValidationError("frequency_range", line_number)

    pos: str | None = None
    if len(fields) == 3:
        pos = fields[2]
        _validate_dictionary_field(pos, "pos", line_number, limits)
        if len(pos) > 32:
            raise DictionaryEntryValidationError("pos_length", line_number)
        if not all(char.isalnum() or char in {"_", "-"} for char in pos):
            raise DictionaryEntryValidationError("pos_characters", line_number)

    return UserDictionaryEntry(term=term, frequency=frequency, pos=pos)


def _load_zipped_dictionary(
    path: Path,
    limits: DictionaryLimits,
    metadata: _DictionaryMetadata,
) -> tuple[UserDictionaryEntry, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) != 1:
                _reject_dictionary(
                    DictionaryArchiveError(
                        "Dictionary archive must contain exactly one file"
                    ),
                    metadata,
                )
            member = members[0]
            if member.flag_bits & 0x1:
                _reject_dictionary(
                    DictionaryArchiveError(
                        "Encrypted dictionary archives are unsupported"
                    ),
                    metadata,
                )
            if member.compress_type not in _ALLOWED_DICTIONARY_COMPRESSION:
                _reject_dictionary(
                    DictionaryArchiveError(
                        "Dictionary archive compression method is unsupported"
                    ),
                    metadata,
                )
            if member.file_size > limits.max_decompressed_bytes:
                _reject_dictionary(
                    DictionarySizeLimitError(
                        "Dictionary member exceeds the decompressed-size limit"
                    ),
                    metadata,
                )
            expansion_ratio = member.file_size / max(member.compress_size, 1)
            if expansion_ratio >= limits.max_expansion_ratio:
                _reject_dictionary(
                    DictionaryExpansionLimitError(
                        "Dictionary archive expansion ratio is too high"
                    ),
                    metadata,
                )
            with archive.open(member, "r") as handle:
                return _parse_dictionary_stream(handle, limits, metadata)
    except DictionaryIngestionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile):
        _reject_dictionary(
            DictionaryArchiveError("Dictionary archive is invalid"), metadata
        )


def _parse_dictionary_stream(
    handle: Any,
    limits: DictionaryLimits,
    metadata: _DictionaryMetadata,
) -> tuple[UserDictionaryEntry, ...]:
    entries: list[UserDictionaryEntry] = []
    line_number = 0
    total_bytes = 0
    pending = bytearray()

    while True:
        chunk = handle.read(_DICTIONARY_READ_CHUNK_BYTES)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > limits.max_decompressed_bytes:
            _reject_dictionary(
                DictionarySizeLimitError(
                    "Dictionary stream exceeds the decompressed-size limit"
                ),
                metadata,
            )
        pending.extend(chunk)
        while True:
            newline_index = pending.find(b"\n")
            if newline_index < 0:
                if len(pending) > limits.max_entry_bytes:
                    _reject_dictionary(
                        DictionaryEntryValidationError(
                            "entry_byte_length", line_number + 1
                        ),
                        metadata,
                    )
                break
            raw_line = bytes(pending[:newline_index])
            del pending[: newline_index + 1]
            line_number += 1
            _append_dictionary_entry(raw_line, line_number, entries, limits, metadata)

    if pending:
        line_number += 1
        _append_dictionary_entry(bytes(pending), line_number, entries, limits, metadata)

    return tuple(entries)


def _append_dictionary_entry(
    raw_line: bytes,
    line_number: int,
    entries: list[UserDictionaryEntry],
    limits: DictionaryLimits,
    metadata: _DictionaryMetadata,
) -> None:
    if len(raw_line) > limits.max_entry_bytes:
        _reject_dictionary(
            DictionaryEntryValidationError("entry_byte_length", line_number),
            metadata,
        )
    try:
        line = raw_line.rstrip(b"\r").decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _reject_dictionary(
            DictionaryEncodingError("Dictionary must be UTF-8"), metadata
        )

    try:
        entry = validate_user_dictionary_entry(
            line,
            line_number=line_number,
            limits=limits,
        )
    except DictionaryEntryValidationError as exc:
        _reject_dictionary(exc, metadata)
    if entry is None:
        return

    metadata.entry_count += 1
    if metadata.entry_count > limits.max_entries:
        _reject_dictionary(
            DictionaryEntryLimitError(metadata.entry_count),
            metadata,
        )
    entries.append(entry)


def _validate_dictionary_field(
    value: str,
    field: str,
    line_number: int,
    limits: DictionaryLimits,
) -> None:
    if not value:
        raise DictionaryEntryValidationError(f"{field}_empty", line_number)
    if len(value) > limits.max_term_characters:
        raise DictionaryEntryValidationError(f"{field}_length", line_number)
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise DictionaryEntryValidationError("control_character", line_number)


def _dictionary_path_hash(path: Path) -> str:
    try:
        normalized = str(path.resolve(strict=False))
    except OSError:
        normalized = str(path.absolute())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_file_size(path: Path) -> int:
    try:
        return max(0, path.stat().st_size)
    except OSError:
        return 0


def _reject_dictionary(
    error: DictionaryIngestionError,
    metadata: _DictionaryMetadata,
) -> NoReturn:
    logger.warning(
        "dictionary_ingestion_rejected path_hash=%s size_bytes=%d "
        "entry_count=%d reason=%s",
        metadata.path_hash,
        metadata.size_bytes,
        metadata.entry_count,
        error.reason,
    )
    raise error from None


@dataclass(frozen=True)
class SpanToken:
    text: str
    start: int
    end: int


def medical_tokenize(
    text: str,
    *,
    exceptions: Optional[Iterable[str]] = None,
) -> List[SpanToken]:
    """Tokenize clinical text into stable span tokens for output remapping.

    This tokenizer is **not** used to create model input ids. It is used to produce
    user-facing token boundaries and to remap model predictions back onto medical-friendly
    spans.
    """
    exceptions_set = {e for e in (exceptions or []) if e}
    if not exceptions_set:
        return [
            SpanToken(m.group(0), m.start(), m.end())
            for m in _MEDICAL_TOKEN_PATTERN.finditer(text)
        ]

    protected: List[Tuple[int, int]] = []
    for exc in sorted(exceptions_set, key=len, reverse=True):
        start = 0
        while True:
            idx = text.find(exc, start)
            if idx == -1:
                break
            span = (idx, idx + len(exc))
            if any(not (span[1] <= a or span[0] >= b) for a, b in protected):
                start = idx + 1
                continue
            protected.append(span)
            start = idx + len(exc)

    if not protected:
        return [
            SpanToken(m.group(0), m.start(), m.end())
            for m in _MEDICAL_TOKEN_PATTERN.finditer(text)
        ]

    protected.sort()
    tokens: List[SpanToken] = []
    cursor = 0
    for s, e in protected:
        if cursor < s:
            for m in _MEDICAL_TOKEN_PATTERN.finditer(text[cursor:s]):
                tokens.append(
                    SpanToken(m.group(0), m.start() + cursor, m.end() + cursor)
                )
        tokens.append(SpanToken(text[s:e], s, e))
        cursor = e
    if cursor < len(text):
        for m in _MEDICAL_TOKEN_PATTERN.finditer(text[cursor:]):
            tokens.append(SpanToken(m.group(0), m.start() + cursor, m.end() + cursor))

    return [
        t for t in sorted(tokens, key=lambda x: (x.start, x.end)) if t.end > t.start
    ]


def remap_predictions_to_tokens(
    predictions: List[Dict[str, Any]],
    text: str,
    tokens: List[SpanToken],
    *,
    gap: int = 1,
) -> List[Dict[str, Any]]:
    """Remap model predictions (char spans) onto custom tokens and merge contiguous tokens.

    Returns a list of prediction dicts compatible with OutputFormatter.
    """
    if not predictions or not tokens:
        return predictions

    token_labels: List[Optional[str]] = [None] * len(tokens)
    token_scores: List[float] = [0.0] * len(tokens)
    token_meta: List[Optional[Dict[str, Any]]] = [None] * len(tokens)

    for idx, tok in enumerate(tokens):
        best_label: Optional[str] = None
        best_score = -1.0
        best_meta: Optional[Dict[str, Any]] = None
        for pred in predictions:
            start = pred.get("start")
            end = pred.get("end")
            if not (isinstance(start, int) and isinstance(end, int)):
                continue
            if end <= tok.start or start >= tok.end:
                continue
            raw_label = pred.get("entity_group") or pred.get("entity") or ""
            if not raw_label:
                continue
            clean = str(raw_label).replace("B-", "").replace("I-", "")
            score = float(pred.get("score", 0.0) or 0.0)
            if score > best_score:
                best_label = clean
                best_score = score
                meta = pred.get("metadata")
                best_meta = dict(meta) if isinstance(meta, dict) else None
        if best_label is not None:
            token_labels[idx] = best_label
            token_scores[idx] = best_score
            token_meta[idx] = best_meta

    remapped: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        label = token_labels[i]
        if label is None:
            i += 1
            continue
        start = tokens[i].start
        end = tokens[i].end
        scores = [token_scores[i]]
        meta = token_meta[i] or {}
        j = i + 1
        while (
            j < len(tokens)
            and token_labels[j] == label
            and tokens[j].start <= end + gap
        ):
            end = tokens[j].end
            scores.append(token_scores[j])
            if not meta and token_meta[j]:
                meta = token_meta[j] or {}
            j += 1

        remapped.append(
            {
                "start": start,
                "end": end,
                "score": sum(scores) / len(scores),
                "entity_group": label,
                "word": text[start:end],
                "metadata": meta,
            }
        )
        i = j

    return remapped


def _is_reasonable_length(
    value: Optional[int], threshold: int = _UNSET_MAX_LENGTH_SENTINEL
) -> bool:
    if value is None:
        return False
    try:
        # Some tokenizers return special sentinel values like `int(1e30)`
        as_int = int(value)
    except (TypeError, ValueError):
        return False
    if as_int <= 0:
        return False
    return as_int < threshold


def infer_tokenizer_max_length(
    tokenizer: "PreTrainedTokenizer",
    *,
    fallback: Optional[int] = None,
    threshold: int = _UNSET_MAX_LENGTH_SENTINEL,
) -> Optional[int]:
    """Infer a sensible maximum sequence length for a tokenizer.

    Many Hugging Face tokenizers expose very large placeholder values (e.g., ``int(1e30)``,
    ``2**63 - 1``) when the underlying model does not specify an explicit limit. This helper
    collapses the common hints into a single manageable integer suitable for truncation.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        fallback: Optional value to return if no reasonable limit is discovered.
        threshold: Maximum value considered a realistic limit.

    Returns:
        Inferred maximum length or ``None`` if unknown.
    """
    candidates: List[Optional[int]] = []

    max_length = getattr(tokenizer, "model_max_length", None)
    if _is_reasonable_length(max_length, threshold):
        return int(max_length)
    candidates.append(max_length)  # record for debugging

    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    kw_max = init_kwargs.get("model_max_length")
    if _is_reasonable_length(kw_max, threshold):
        return int(kw_max)
    candidates.append(kw_max)

    config = getattr(tokenizer, "config", None)
    if config is not None:
        for attr in (
            "model_max_length",
            "max_position_embeddings",
            "n_positions",
            "seq_length",
        ):
            candidate = getattr(config, attr, None)
            if _is_reasonable_length(candidate, threshold):
                return int(candidate)
            candidates.append(candidate)

    if fallback is not None and _is_reasonable_length(fallback, threshold):
        return int(fallback)

    logger.debug(
        "Tokenizer %s did not expose a reasonable max_length; candidates=%s",
        getattr(tokenizer, "name_or_path", "<unknown>"),
        candidates,
    )
    return None


class TokenizationHelper:
    """Helper class for tokenization operations in medical text."""

    def __init__(self, tokenizer: Optional["PreTrainedTokenizer"] = None):
        """Initialize tokenization helper.

        Args:
            tokenizer: HuggingFace tokenizer instance.
        """
        self.tokenizer = tokenizer

    def tokenize_with_alignment(
        self, text: str, return_word_ids: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text while maintaining word alignment.

        Args:
            text: Input text to tokenize.
            return_word_ids: Whether to return word ID mappings.

        Returns:
            Dictionary containing tokenization results and alignments.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")

        # Tokenize with special tokens
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
        )

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "tokens": self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]),
            "offset_mapping": encoding.get("offset_mapping"),
            "special_tokens_mask": encoding.get("special_tokens_mask"),
        }

        if return_word_ids:
            result["word_ids"] = encoding.word_ids()

        return result

    def align_predictions_to_words(
        self,
        predictions: List[Any],
        word_ids: List[Optional[int]],
        text: str,
        aggregation_strategy: str = "first",
    ) -> List[Tuple[str, Any]]:
        """Align token-level predictions to word-level predictions.

        Args:
            predictions: Token-level predictions.
            word_ids: Word ID mappings from tokenizer.
            text: Original text.
            aggregation_strategy: How to aggregate subword predictions
                                 ("first", "max", "average").

        Returns:
            List of (word, prediction) tuples.
        """
        if len(predictions) != len(word_ids):
            raise ValueError("Predictions and word_ids must have same length")

        words = text.split()
        word_predictions = {}

        for i, (pred, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is None:  # Special tokens
                continue

            if word_id not in word_predictions:
                word_predictions[word_id] = []
            word_predictions[word_id].append(pred)

        # Aggregate predictions for each word
        result = []
        for word_id in sorted(word_predictions.keys()):
            if word_id < len(words):
                word = words[word_id]
                preds = word_predictions[word_id]

                if aggregation_strategy == "first":
                    final_pred = preds[0]
                elif aggregation_strategy == "max":
                    final_pred = (
                        max(preds) if isinstance(preds[0], (int, float)) else preds[0]
                    )
                elif aggregation_strategy == "average":
                    if isinstance(preds[0], (int, float)):
                        final_pred = sum(preds) / len(preds)
                    else:
                        final_pred = preds[0]  # Can't average non-numeric
                else:
                    final_pred = preds[0]

                result.append((word, final_pred))

        return result

    def create_attention_masks(
        self, input_ids: List[List[int]], pad_token_id: int
    ) -> List[List[int]]:
        """Create attention masks for batched inputs.

        Args:
            input_ids: Batched input token IDs.
            pad_token_id: ID of the padding token.

        Returns:
            Attention masks.
        """
        attention_masks = []
        for ids in input_ids:
            mask = [1 if token_id != pad_token_id else 0 for token_id in ids]
            attention_masks.append(mask)
        return attention_masks

    def truncate_sequences(
        self,
        sequences: List[str],
        max_length: int,
        truncation_strategy: str = "longest_first",
    ) -> List[str]:
        """Truncate sequences to fit within max_length.

        Args:
            sequences: List of text sequences.
            max_length: Maximum sequence length.
            truncation_strategy: How to truncate ("longest_first", "do_not_truncate").

        Returns:
            Truncated sequences.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")

        truncated = []
        for seq in sequences:
            tokens = self.tokenizer.tokenize(seq)
            if len(tokens) <= max_length:
                truncated.append(seq)
            else:
                if truncation_strategy == "longest_first":
                    truncated_tokens = tokens[:max_length]
                    truncated_text = self.tokenizer.convert_tokens_to_string(
                        truncated_tokens
                    )
                    truncated.append(truncated_text)
                else:
                    truncated.append(seq)  # Don't truncate

        return truncated

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """Batch encode multiple texts.

        Args:
            texts: List of texts to encode.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.

        Returns:
            Encoded batch.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")

        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
