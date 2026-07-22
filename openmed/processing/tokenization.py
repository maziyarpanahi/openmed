"""Tokenization utilities for OpenMed."""

import hashlib
import importlib
import json
import logging
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openmed.core.decoding.spans import (
    is_grapheme_boundary,
    is_indic_text,
    iter_grapheme_clusters,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

_UNSET_MAX_LENGTH_SENTINEL = 1_000_000
SEGMENTER_RESOURCE_SIZE_BUDGET_BYTES = 64 * 1024
SEGMENTER_RESOURCE_DIRECTORY = "segmenter"
DEFAULT_SEGMENTER_ID = "openmed-cjk-indic-v1"
SEGMENTER_IDS = (
    "openmed-han-v1",
    "openmed-indic-v1",
    DEFAULT_SEGMENTER_ID,
)
_SEGMENTER_RESOURCE_ROOT = Path(__file__).with_name("resources") / "segmenter"
_SEGMENTER_SPECS: dict[str, dict[str, Any]] = {
    "openmed-han-v1": {
        "scripts": ["Han"],
        "license": "MIT",
        "resources": [
            ("han_words.txt", "han_dictionary", "MIT"),
        ],
    },
    "openmed-indic-v1": {
        "scripts": ["Devanagari"],
        "license": "ICU-1.8.1",
        "resources": [
            ("indic_rules.json", "indic_break_rules", "ICU-1.8.1"),
        ],
    },
    DEFAULT_SEGMENTER_ID: {
        "scripts": ["Han", "Devanagari"],
        "license": "MIT AND ICU-1.8.1",
        "resources": [
            ("han_words.txt", "han_dictionary", "MIT"),
            ("indic_rules.json", "indic_break_rules", "ICU-1.8.1"),
        ],
    },
}
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
_NUMERIC_CONNECTORS = frozenset({"/", "-", ".", ",", ":"})


@dataclass(frozen=True)
class SpanToken:
    text: str
    start: int
    end: int


def grapheme_tokenize(text: str) -> List[SpanToken]:
    """Return one non-whitespace token per extended grapheme cluster.

    This output-oriented tokenizer preserves exact source offsets and keeps
    Indic aksharas, emoji ZWJ sequences, and combining-mark sequences intact.
    It does not create model input IDs.

    Args:
        text: Original, unnormalized Unicode text.

    Returns:
        Non-whitespace tokens with exact source code-point offsets.
    """

    return [
        SpanToken(text[start:end], start, end)
        for start, end in iter_grapheme_clusters(text)
        if not text[start:end].isspace()
    ]


def indic_grapheme_tokenize(text: str) -> List[SpanToken]:
    """Return grapheme-aligned tokens for Indic runs in *text*.

    Non-Indic clusters and whitespace are omitted. Offsets always refer to the
    original string, allowing callers to combine this producer with their
    existing tokenization for other scripts.

    Args:
        text: Original, unnormalized Unicode text.

    Returns:
        Indic grapheme tokens with exact source code-point offsets.
    """

    return [token for token in grapheme_tokenize(text) if is_indic_text(token.text)]


def _append_token(tokens: List[SpanToken], text: str, start: int, end: int) -> None:
    if start < end and not text[start:end].isspace():
        tokens.append(SpanToken(text[start:end], start, end))


def _cluster_kind(cluster: str) -> str:
    if cluster.isspace():
        return "space"
    if all(char.isdecimal() for char in cluster):
        return "number"
    if any(unicodedata.category(char).startswith("L") for char in cluster):
        return "word"
    if cluster in _NUMERIC_CONNECTORS:
        return "connector"
    return "punctuation"


def indic_word_tokenize(text: str) -> List[SpanToken]:
    """Tokenize mixed Indic text without bisecting a grapheme cluster.

    Letter clusters are grouped into words, native or ASCII digit runs retain
    internal date/number separators, and punctuation is emitted at its own
    boundary. Returned spans always index ``text`` exactly.
    """

    clusters = list(iter_grapheme_clusters(text))
    tokens: List[SpanToken] = []
    token_start: Optional[int] = None
    token_end = 0
    token_kind: Optional[str] = None

    for cluster_index, (start, end) in enumerate(clusters):
        cluster = text[start:end]
        kind = _cluster_kind(cluster)
        next_kind = (
            _cluster_kind(text[slice(*clusters[cluster_index + 1])])
            if cluster_index + 1 < len(clusters)
            else None
        )

        if kind == "connector" and token_kind == "number" and next_kind == "number":
            token_end = end
            continue

        if kind in {"word", "number"}:
            if token_start is not None and token_kind != kind:
                _append_token(tokens, text, token_start, token_end)
                token_start = None
            if token_start is None:
                token_start = start
                token_kind = kind
            token_end = end
            continue

        if token_start is not None:
            _append_token(tokens, text, token_start, token_end)
            token_start = None
            token_kind = None

        if kind == "punctuation" or kind == "connector":
            _append_token(tokens, text, start, end)

    if token_start is not None:
        _append_token(tokens, text, token_start, token_end)

    return tokens


def _regex_medical_tokens(
    text: str,
    *,
    offset: int = 0,
) -> List[SpanToken]:
    return [
        SpanToken(match.group(0), offset + match.start(), offset + match.end())
        for match in _MEDICAL_TOKEN_PATTERN.finditer(text)
    ]


def _medical_tokens_in_segment(text: str, *, offset: int = 0) -> List[SpanToken]:
    tokens: List[SpanToken] = []
    non_indic_start = 0

    for cluster_start, cluster_end in iter_grapheme_clusters(text):
        cluster = text[cluster_start:cluster_end]
        if not is_indic_text(cluster):
            continue
        if non_indic_start < cluster_start:
            tokens.extend(
                _regex_medical_tokens(
                    text[non_indic_start:cluster_start],
                    offset=offset + non_indic_start,
                )
            )
        if not cluster.isspace():
            tokens.append(
                SpanToken(cluster, offset + cluster_start, offset + cluster_end)
            )
        non_indic_start = cluster_end

    if non_indic_start < len(text):
        tokens.extend(
            _regex_medical_tokens(
                text[non_indic_start:],
                offset=offset + non_indic_start,
            )
        )
    return tokens


@dataclass(frozen=True)
class SegmentSpan:
    """One resource-segmenter span expressed in UTF-8 byte offsets."""

    text: str
    start: int
    end: int


def package_segmenter_resources(
    bundle_dir: str | Path,
    segmenter_id: str = DEFAULT_SEGMENTER_ID,
) -> dict[str, Any]:
    """Copy a compact segmenter resource set into an on-device bundle.

    The returned descriptor is ready to store under the ``segmenter`` key in
    MLX, ONNX, or CoreML manifests. The 64 KiB budget covers only the declared
    data tables; no jieba or ICU runtime is copied into the artifact.
    """

    try:
        spec = _SEGMENTER_SPECS[segmenter_id]
    except KeyError as exc:
        supported = ", ".join(SEGMENTER_IDS)
        raise ValueError(
            f"unsupported segmenter_id {segmenter_id!r}; expected one of {supported}"
        ) from exc

    bundle_path = Path(bundle_dir)
    resource_dir = bundle_path / SEGMENTER_RESOURCE_DIRECTORY
    resource_dir.mkdir(parents=True, exist_ok=True)
    resource_files: list[dict[str, Any]] = []

    for filename, role, license_id in spec["resources"]:
        source = _SEGMENTER_RESOURCE_ROOT / filename
        if not source.is_file():
            raise FileNotFoundError(f"packaged segmenter resource is missing: {source}")
        destination = resource_dir / filename
        shutil.copy2(source, destination)
        payload = destination.read_bytes()
        resource_files.append(
            {
                "path": f"{SEGMENTER_RESOURCE_DIRECTORY}/{filename}",
                "role": role,
                "license": license_id,
                "size_bytes": len(payload),
                "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            }
        )

    total_size = sum(int(item["size_bytes"]) for item in resource_files)
    descriptor = {
        "id": segmenter_id,
        "format_version": 1,
        "scripts": list(spec["scripts"]),
        "license": spec["license"],
        "resource_files": resource_files,
        "total_size_bytes": total_size,
        "size_budget_bytes": SEGMENTER_RESOURCE_SIZE_BUDGET_BYTES,
    }
    validate_segmenter_resources(bundle_path, descriptor)
    return descriptor


def validate_segmenter_resources(
    bundle_dir: str | Path,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one manifest segmenter descriptor and its bundle resources."""

    bundle_path = Path(bundle_dir).resolve()
    segmenter_id = str(descriptor.get("id") or "")
    if segmenter_id not in SEGMENTER_IDS:
        raise ValueError(f"unsupported segmenter resource id: {segmenter_id!r}")

    scripts = descriptor.get("scripts")
    if (
        not isinstance(scripts, list)
        or not scripts
        or not all(isinstance(script, str) and script for script in scripts)
    ):
        raise ValueError("segmenter scripts must be a non-empty string list")
    if not descriptor.get("license"):
        raise ValueError("segmenter descriptor must record a license")
    expected_spec = _SEGMENTER_SPECS[segmenter_id]
    if scripts != expected_spec["scripts"]:
        raise ValueError("segmenter scripts do not match the declared resource id")
    if descriptor.get("license") != expected_spec["license"]:
        raise ValueError("segmenter license does not match the declared resource id")

    resources = descriptor.get("resource_files")
    if not isinstance(resources, list) or not resources:
        raise ValueError("segmenter descriptor must declare resource_files")
    expected_resources = {
        (filename, role): license_id
        for filename, role, license_id in expected_spec["resources"]
    }
    declared_resources = {
        (Path(str(item.get("path") or "")).name, str(item.get("role") or "")): str(
            item.get("license") or ""
        )
        for item in resources
        if isinstance(item, Mapping)
    }
    if declared_resources != expected_resources:
        raise ValueError("segmenter resource roles or licenses do not match its id")

    total_size = 0
    validated_files: list[str] = []
    for resource in resources:
        if not isinstance(resource, Mapping):
            raise ValueError("segmenter resource entries must be objects")
        relative_path = str(resource.get("path") or "")
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError(f"invalid segmenter resource path: {relative_path!r}")
        path = (bundle_path / relative_path).resolve()
        if bundle_path not in path.parents:
            raise ValueError(f"segmenter resource escapes bundle: {relative_path}")
        if not path.is_file():
            raise ValueError(f"segmenter resource is missing: {relative_path}")
        if not resource.get("license"):
            raise ValueError(f"segmenter resource has no license: {relative_path}")

        payload = path.read_bytes()
        actual_size = len(payload)
        declared_size = resource.get("size_bytes")
        if declared_size != actual_size:
            raise ValueError(
                f"segmenter resource size mismatch for {relative_path}: "
                f"declared {declared_size}, actual {actual_size}"
            )
        declared_hash = str(resource.get("sha256") or "")
        actual_hash = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if declared_hash != actual_hash:
            raise ValueError(f"segmenter resource hash mismatch: {relative_path}")
        total_size += actual_size
        validated_files.append(relative_path)

    budget = descriptor.get("size_budget_bytes")
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("segmenter size_budget_bytes must be a positive integer")
    if budget != SEGMENTER_RESOURCE_SIZE_BUDGET_BYTES:
        raise ValueError("segmenter descriptor must use the 64 KiB on-device budget")
    if total_size > budget:
        raise ValueError(
            f"segmenter resources use {total_size} bytes, above the {budget}-byte budget"
        )
    if descriptor.get("total_size_bytes") != total_size:
        raise ValueError(
            "segmenter total_size_bytes does not match the declared resource files"
        )

    return {
        "id": segmenter_id,
        "scripts": list(scripts),
        "files": validated_files,
        "total_size_bytes": total_size,
        "size_budget_bytes": budget,
    }


class ResourceSegmenter:
    """Segment Han and Devanagari text from manifest-declared resource tables."""

    def __init__(
        self,
        bundle_dir: str | Path,
        descriptor: Mapping[str, Any],
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.descriptor = dict(descriptor)
        validate_segmenter_resources(self.bundle_dir, self.descriptor)
        self.scripts = frozenset(str(item) for item in descriptor["scripts"])
        self._han_words: frozenset[str] = frozenset()
        self._max_han_word_length = 1
        self._devanagari_ranges: tuple[tuple[int, int], ...] = ()
        self._viramas: frozenset[int] = frozenset()
        self._joiners: frozenset[int] = frozenset()
        self._load_resources()

    @classmethod
    def from_manifest(
        cls,
        bundle_dir: str | Path,
        manifest: Mapping[str, Any],
    ) -> "ResourceSegmenter | None":
        """Create a segmenter from any OpenMed artifact manifest, if declared."""

        descriptor = manifest.get("segmenter")
        if descriptor is None:
            return None
        if not isinstance(descriptor, Mapping):
            raise ValueError("manifest segmenter descriptor must be an object")
        return cls(bundle_dir, descriptor)

    def segment(
        self,
        text: str,
        *,
        use_accelerated: bool = False,
    ) -> list[SegmentSpan]:
        """Return non-whitespace spans with UTF-8 offsets.

        The stdlib/resource-table implementation is always available. When
        ``use_accelerated`` is true, a locally installed jieba runtime may be
        used for Han runs; it is imported lazily and is never required.
        """

        if not text:
            return []
        han_tokenizer = self._load_optional_jieba() if use_accelerated else None
        return self._segment_stdlib(text, han_tokenizer=han_tokenizer)

    def _load_resources(self) -> None:
        for resource in self.descriptor["resource_files"]:
            path = self.bundle_dir / str(resource["path"])
            role = resource.get("role")
            if role == "han_dictionary":
                words = {
                    line.split()[0]
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                }
                self._han_words = frozenset(words)
                self._max_han_word_length = max(map(len, words), default=1)
            elif role == "indic_break_rules":
                payload = json.loads(path.read_text(encoding="utf-8"))
                rules = payload.get("scripts", {}).get("Devanagari", {})
                self._devanagari_ranges = tuple(
                    (int(start), int(end)) for start, end in rules.get("ranges", [])
                )
                self._viramas = frozenset(
                    int(item) for item in rules.get("viramas", [])
                )
                self._joiners = frozenset(
                    int(item) for item in rules.get("joiners", [])
                )

    def _load_optional_jieba(self) -> Any | None:
        if "Han" not in self.scripts:
            return None
        try:
            module = importlib.import_module("jieba")
            dictionary_path = next(
                self.bundle_dir / str(item["path"])
                for item in self.descriptor["resource_files"]
                if item.get("role") == "han_dictionary"
            )
            return module.Tokenizer(dictionary=str(dictionary_path))
        except (ImportError, AttributeError, OSError, StopIteration, TypeError):
            return None

    def _segment_stdlib(
        self,
        text: str,
        *,
        han_tokenizer: Any | None,
    ) -> list[SegmentSpan]:
        byte_offsets = [0]
        for character in text:
            byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))

        spans: list[SegmentSpan] = []
        cursor = 0
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            if "Han" in self.scripts and _is_han_character(text[cursor]):
                end = cursor + 1
                while end < len(text) and _is_han_character(text[end]):
                    end += 1
                han_spans = self._segment_han_run(
                    text,
                    cursor,
                    end,
                    byte_offsets,
                    han_tokenizer=han_tokenizer,
                )
                spans.extend(han_spans)
                cursor = end
                continue
            if "Devanagari" in self.scripts and self._is_devanagari(text[cursor]):
                end = cursor + 1
                while end < len(text) and (
                    self._is_devanagari(text[end]) or ord(text[end]) in self._joiners
                ):
                    end += 1
                spans.extend(
                    self._segment_devanagari_run(text, cursor, end, byte_offsets)
                )
                cursor = end
                continue

            end = cursor + 1
            while (
                end < len(text)
                and not text[end].isspace()
                and not ("Han" in self.scripts and _is_han_character(text[end]))
                and not (
                    "Devanagari" in self.scripts
                    and (
                        self._is_devanagari(text[end])
                        or ord(text[end]) in self._joiners
                    )
                )
            ):
                end += 1
            spans.append(_segment_span(text, cursor, end, byte_offsets))
            cursor = end
        return spans

    def _segment_han_run(
        self,
        text: str,
        start: int,
        end: int,
        byte_offsets: list[int],
        *,
        han_tokenizer: Any | None,
    ) -> list[SegmentSpan]:
        fallback: list[SegmentSpan] = []
        cursor = start
        while cursor < end:
            upper = min(end, cursor + self._max_han_word_length)
            match_end = cursor + 1
            for candidate_end in range(upper, cursor, -1):
                if text[cursor:candidate_end] in self._han_words:
                    match_end = candidate_end
                    break
            fallback.append(_segment_span(text, cursor, match_end, byte_offsets))
            cursor = match_end

        if han_tokenizer is not None:
            try:
                accelerated = [
                    _segment_span(
                        text,
                        start + int(local_start),
                        start + int(local_end),
                        byte_offsets,
                    )
                    for _, local_start, local_end in han_tokenizer.tokenize(
                        text[start:end]
                    )
                    if int(local_end) > int(local_start)
                ]
                if [(span.start, span.end) for span in accelerated] == [
                    (span.start, span.end) for span in fallback
                ]:
                    return accelerated
            except (AttributeError, TypeError, ValueError):
                pass
        return fallback

    def _segment_devanagari_run(
        self,
        text: str,
        start: int,
        end: int,
        byte_offsets: list[int],
    ) -> list[SegmentSpan]:
        boundaries = [start]
        for cursor in range(start + 1, end):
            codepoint = ord(text[cursor])
            previous = ord(text[cursor - 1])
            previous_previous = ord(text[cursor - 2]) if cursor - 2 >= start else None
            joins_previous = (
                unicodedata.category(text[cursor]).startswith("M")
                or codepoint in self._joiners
                or previous in self._viramas
                or (
                    previous in self._joiners
                    and previous_previous is not None
                    and previous_previous in self._viramas
                )
            )
            if not joins_previous:
                boundaries.append(cursor)
        boundaries.append(end)
        return [
            _segment_span(text, left, right, byte_offsets)
            for left, right in zip(boundaries, boundaries[1:])
        ]

    def _is_devanagari(self, character: str) -> bool:
        codepoint = ord(character)
        return any(start <= codepoint <= end for start, end in self._devanagari_ranges)


def _segment_span(
    text: str,
    start: int,
    end: int,
    byte_offsets: list[int],
) -> SegmentSpan:
    return SegmentSpan(
        text=text[start:end],
        start=byte_offsets[start],
        end=byte_offsets[end],
    )


def _is_han_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    )


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
        return _medical_tokens_in_segment(text)

    protected: List[Tuple[int, int]] = []
    grapheme_boundaries = {0, len(text)}
    grapheme_boundaries.update(end for _, end in iter_grapheme_clusters(text))
    for exc in sorted(exceptions_set, key=len, reverse=True):
        start = 0
        while True:
            idx = text.find(exc, start)
            if idx == -1:
                break
            span = (idx, idx + len(exc))
            if not (
                is_grapheme_boundary(span[0], text)
                and is_grapheme_boundary(span[1], text)
                and span[0] in grapheme_boundaries
                and span[1] in grapheme_boundaries
            ):
                start = idx + 1
                continue
            if any(not (span[1] <= a or span[0] >= b) for a, b in protected):
                start = idx + 1
                continue
            protected.append(span)
            start = idx + len(exc)

    if not protected:
        return _medical_tokens_in_segment(text)

    protected.sort()
    tokens: List[SpanToken] = []
    cursor = 0
    for s, e in protected:
        if cursor < s:
            tokens.extend(_medical_tokens_in_segment(text[cursor:s], offset=cursor))
        tokens.append(SpanToken(text[s:e], s, e))
        cursor = e
    if cursor < len(text):
        tokens.extend(_medical_tokens_in_segment(text[cursor:], offset=cursor))

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
