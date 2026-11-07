"""Portable tokenizer conformance vectors and a value-free verifier.

Bundle digests and final-span parity can show that two runtimes drifted apart
without showing whether the drift began in normalization, special-token
insertion, truncation windows or model execution. This module pins the
intermediate tokenizer contract that every OpenMed runtime must reproduce
before any inference comparison runs:

* a tiny synthetic WordPiece ``tokenizer.json`` built from synthetic text,
* versioned vectors with normalized text, tokens, input ids, attention masks,
  token type ids, special-token masks, source-offset maps, single-pass
  truncation and stride windows,
* a hand-reviewed runtime support file that declares, per runtime adapter,
  which fields it cannot produce and which features knowingly diverge, and
* a digest manifest whose contract reference a model bundle can record.

Vectors come from :mod:`openmed.eval.tokenizer_reference`, a standard-library
implementation of the Rust ``tokenizers`` semantics, so ``--check`` runs need
no optional dependency. Reports and errors contain only case ids, field
names, token indices, feature names, reason codes and digests; never fixture
text, tokens or file paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.eval.tokenizer_reference import ReferenceTokenizer

__all__ = [
    "CONFORMANCE_FIELDS",
    "DEFAULT_FIXTURE_DIR",
    "MANIFEST_FILENAME",
    "SUPPORT_FILENAME",
    "SYNTHETIC_CASES",
    "SYNTHETIC_TOKENIZER_ID",
    "TOKENIZER_CONTRACT_VERSION",
    "TOKENIZER_DIRNAME",
    "VECTORS_FILENAME",
    "WINDOW_FIELDS",
    "ConformanceCase",
    "ConformanceMismatch",
    "RuntimeEncoding",
    "TokenizerConformanceError",
    "TokenizerConformanceReport",
    "build_fixture_files",
    "build_manifest",
    "build_synthetic_tokenizer_files",
    "build_vectors",
    "check_fixture_dir",
    "load_fixture_dir",
    "require_tokenizer_conformance",
    "tokenizer_contract_reference",
    "validate_runtime_support",
    "verify_tokenizer_conformance",
    "write_fixture_dir",
]

TOKENIZER_CONTRACT_VERSION = "openmed.tokenizer-conformance.v1"
VECTORS_SCHEMA = "openmed.tokenizer-conformance.vectors"
SUPPORT_SCHEMA = "openmed.tokenizer-conformance.runtime-support"
MANIFEST_SCHEMA = "openmed.tokenizer-conformance.manifest"
FORMAT_VERSION = 1
SYNTHETIC_TOKENIZER_ID = "synthetic-wordpiece-uncased-v1"
DEFAULT_FIXTURE_DIR = Path("tests/fixtures/parity/tokenizer")
TOKENIZER_DIRNAME = "synthetic-wordpiece-v1"
VECTORS_FILENAME = "tokenizer_conformance_v1.json"
SUPPORT_FILENAME = "tokenizer_runtime_support_v1.json"
MANIFEST_FILENAME = "manifest.json"

CASE_FIELDS = (
    "normalized",
    "tokens",
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "special_tokens_mask",
    "offsets",
)
WINDOW_FIELDS = ("truncated_input_ids", "window_ranges", "window_input_ids")
CONFORMANCE_FIELDS = CASE_FIELDS + WINDOW_FIELDS

CODE_LENGTH_MISMATCH = "TOKENIZER_FIELD_LENGTH_MISMATCH"
CODE_VALUE_MISMATCH = "TOKENIZER_FIELD_VALUE_MISMATCH"
CODE_FIELD_MISSING = "TOKENIZER_FIELD_MISSING"
CODE_RUNTIME_ERROR = "TOKENIZER_RUNTIME_ERROR"
CODE_DIVERGENCE_STALE = "TOKENIZER_DECLARED_DIVERGENCE_STALE"
CODE_RUNTIME_UNDECLARED = "TOKENIZER_RUNTIME_UNDECLARED"
CODE_CONTRACT_MISMATCH = "TOKENIZER_CONTRACT_VERSION_MISMATCH"

_REASON_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_FEATURE_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z")
_CASE_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_RUNTIME_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")

_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
# Synthetic text whose characters seed the character-level vocabulary. It is
# invented clinical-style prose with no names, dates, identifiers or contacts.
SYNTHETIC_VOCAB_CORPUS: tuple[str, ...] = (
    "synthetic patient visited the synthetic clinic for a routine follow up review.",
    "the clinic note was signed after the dose review, and the chart was closed!",
    "cafe naive resume: (draft) note; dose? [review] 'chart' \"note\" - ok.",
    "\u60a3\u8005\u4eca\u65e5\u590d\u8bca\u3002\u533b\u9662\uff1a\u590d\u8bca\u3002"
    "\U00020000",
    "\u0645\u0631\u064a\u0636 \u0641\u064a \u0627\u0644\u0639\u064a\u0627\u062f\u0629",
    "\u05d7\u05d5\u05dc\u05d4",
    "\u0930\u094b\u0917\u0940 \u0915\u094d\u0932\u093f\u0928\u093f\u0915"
    " \u092e\u0947\u0902",
    "\u09b0\u09cb\u0997\u09c0",
    "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf",
    "\U0001f600 \U0001f469\u200d\u2695\ufe0f \U0001f44d\U0001f3fd \U0001f1fa\U0001f1f8",
)
# Explicit synthetic whole words and continuation pieces. Words that the
# vectors split on purpose (``synthetic``, ``visited``) stay out of this list.
SYNTHETIC_WORD_PIECES: tuple[str, ...] = (
    "synth",
    "##etic",
    "patient",
    "visit",
    "##ed",
    "the",
    "clinic",
    "for",
    "routine",
    "follow",
    "up",
    "review",
    "note",
    "dose",
    "chart",
    "cafe",
    "naive",
    "resume",
    "\u0645\u0631\u064a\u0636",
    "\u0627\u0644\u0639\u064a\u0627\u062f\u0629",
    "\u05d7\u05d5\u05dc\u05d4",
    "\u0930\u094b\u0917\u0940",
    "\u0915\u0932\u093f\u0928\u093f\u0915",
)


@dataclass(frozen=True)
class ConformanceCase:
    """One synthetic tokenizer conformance case.

    Attributes:
        id: Stable case id used in every diagnostic.
        text: Synthetic input text.
        features: Controlled feature tags the case exercises.
        window: Optional ``(max_length, stride)`` truncation window request.
    """

    id: str
    text: str
    features: tuple[str, ...]
    window: tuple[int, int] | None = None


_WINDOW_TEXT = (
    "synthetic patient visited the synthetic clinic for a routine follow up review"
)
_CJK_WINDOW_TEXT = "\u60a3\u8005\u4eca\u65e5\u590d\u8bca\u533b\u9662\u590d\u8bca"

SYNTHETIC_CASES: tuple[ConformanceCase, ...] = (
    ConformanceCase(
        "ascii-basic-01",
        "Synthetic patient visited the clinic.",
        (
            "ascii.basic",
            "case.uppercase",
            "punctuation.ascii",
            "wordpiece.continuation",
        ),
    ),
    ConformanceCase(
        "ascii-uppercase-01",
        "SYNTHETIC CLINIC NOTE",
        ("ascii.basic", "case.uppercase", "wordpiece.continuation"),
    ),
    ConformanceCase(
        "ascii-punctuation-01",
        "note:(dose)[review];chart?",
        ("ascii.basic", "punctuation.ascii"),
    ),
    ConformanceCase(
        "unknown-word-01",
        "the zqxj note",
        ("ascii.basic", "unknown.word"),
    ),
    ConformanceCase(
        "unknown-long-word-01",
        "note " + "synth" * 21,
        ("ascii.basic", "unknown.long_word"),
    ),
    ConformanceCase(
        "combining-precomposed-01",
        "Caf\u00e9 na\u00efve r\u00e9sum\u00e9",
        ("combining.precomposed", "strip_accents.latin"),
    ),
    ConformanceCase(
        "combining-decomposed-01",
        "Cafe\u0301 nai\u0308ve",
        (
            "combining.decomposed",
            "offsets.trailing_removed_mark",
            "strip_accents.latin",
        ),
    ),
    ConformanceCase(
        "combining-leading-mark-01",
        "\u0301note dose",
        ("combining.leading_mark", "strip_accents.latin"),
    ),
    ConformanceCase(
        "clean-control-01",
        "dose\u0007note\u000bchart",
        ("clean_text.control",),
    ),
    ConformanceCase(
        "clean-format-01",
        "pa\u200btient \ufeffnote no\u00adte",
        ("clean_text.format", "clean_text.format_inside_word"),
    ),
    ConformanceCase(
        "whitespace-unicode-01",
        "note\u00a0dose\u3000clinic\tvisit\npatient",
        ("whitespace.unicode",),
    ),
    ConformanceCase(
        "whitespace-crlf-01",
        "note\r\ndose\r\nchart",
        ("whitespace.crlf",),
    ),
    ConformanceCase(
        "rtl-arabic-01",
        "\u0645\u0631\u064a\u0636 \u0641\u064a "
        "\u0627\u0644\u0639\u064a\u0627\u062f\u0629",
        ("rtl.arabic",),
    ),
    ConformanceCase(
        "rtl-arabic-harakat-01",
        "\u0645\u064e\u0631\u0650\u064a\u0636",
        ("rtl.arabic", "strip_accents.non_latin"),
    ),
    ConformanceCase(
        "rtl-hebrew-niqqud-01",
        "\u05d7\u05d5\u05b9\u05dc\u05b6\u05d4",
        ("rtl.hebrew", "strip_accents.non_latin"),
    ),
    ConformanceCase(
        "rtl-bidi-marks-01",
        "note \u200f\u0645\u0631\u064a\u0636\u200e \u202bdose\u202c",
        ("clean_text.format", "rtl.arabic", "rtl.bidi_control"),
    ),
    ConformanceCase(
        "cjk-han-01",
        "\u60a3\u8005\u4eca\u65e5\u590d\u8bca",
        ("cjk.han",),
    ),
    ConformanceCase(
        "cjk-mixed-latin-01",
        "synth\u60a3\u8005note",
        ("cjk.han", "cjk.mixed_latin"),
    ),
    ConformanceCase(
        "cjk-supplementary-01",
        "\u60a3\u8005\U00020000\u590d\u8bca",
        ("cjk.han", "cjk.supplementary"),
    ),
    ConformanceCase(
        "cjk-supplementary-02",
        "dose\U00020000note",
        ("cjk.mixed_latin", "cjk.supplementary"),
    ),
    ConformanceCase(
        "cjk-fullwidth-punctuation-01",
        "\u60a3\u8005\uff1a\u590d\u8bca\u3002",
        ("cjk.han", "punctuation.fullwidth"),
    ),
    ConformanceCase(
        "indic-devanagari-01",
        "\u0930\u094b\u0917\u0940 \u0915\u094d\u0932\u093f\u0928\u093f\u0915"
        " \u092e\u0947\u0902",
        (
            "indic.devanagari",
            "indic.spacing_mark",
            "offsets.trailing_removed_mark",
            "strip_accents.non_latin",
        ),
    ),
    ConformanceCase(
        "indic-bengali-01",
        "\u09b0\u09cb\u0997\u09c0",
        (
            "indic.bengali",
            "indic.nfd_expansion",
            "wordpiece.grapheme_cluster_split",
            "wordpiece.mark_only_piece",
        ),
    ),
    ConformanceCase(
        "indic-tamil-01",
        "\u0ba8\u0bcb\u0baf\u0bbe\u0bb3\u0bbf",
        (
            "indic.nfd_expansion",
            "indic.tamil",
            "wordpiece.grapheme_cluster_split",
            "wordpiece.mark_only_piece",
        ),
    ),
    ConformanceCase(
        "emoji-single-01",
        "note \U0001f600 dose",
        ("emoji.single",),
    ),
    ConformanceCase(
        "emoji-zwj-sequence-01",
        "\U0001f469\u200d\u2695\ufe0f visit",
        (
            "clean_text.format",
            "emoji.variation_selector",
            "emoji.zwj_sequence",
            "offsets.trailing_removed_mark",
            "wordpiece.grapheme_cluster_split",
        ),
    ),
    ConformanceCase(
        "emoji-modifier-flag-01",
        "\U0001f44d\U0001f3fd \U0001f1fa\U0001f1f8",
        ("emoji.flag", "emoji.skin_tone", "wordpiece.grapheme_cluster_split"),
    ),
    ConformanceCase("empty-01", "", ("empty",)),
    ConformanceCase("whitespace-only-01", " \t\n ", ("whitespace_only",)),
    ConformanceCase(
        "special-token-in-text-01",
        "note [SEP] dose",
        ("special_tokens.in_text",),
    ),
    ConformanceCase(
        "window-exact-fit-01",
        _WINDOW_TEXT,
        ("window.exact_fit", "wordpiece.continuation"),
        window=(17, 4),
    ),
    ConformanceCase(
        "window-overflow-one-01",
        _WINDOW_TEXT,
        ("truncation.overflow", "window.overflow", "wordpiece.continuation"),
        window=(16, 4),
    ),
    ConformanceCase(
        "window-stride-01",
        _WINDOW_TEXT,
        ("truncation.overflow", "window.stride", "wordpiece.continuation"),
        window=(8, 3),
    ),
    ConformanceCase(
        "window-stride-zero-01",
        _WINDOW_TEXT,
        ("truncation.overflow", "window.stride_zero", "wordpiece.continuation"),
        window=(7, 0),
    ),
    ConformanceCase(
        "window-subword-boundary-01",
        _WINDOW_TEXT,
        (
            "truncation.overflow",
            "window.stride",
            "window.subword_boundary",
            "wordpiece.continuation",
        ),
        window=(7, 1),
    ),
    ConformanceCase(
        "window-cjk-01",
        _CJK_WINDOW_TEXT,
        ("cjk.han", "truncation.overflow", "window.stride"),
        window=(6, 1),
    ),
)


@dataclass(frozen=True)
class RuntimeEncoding:
    """Fields one runtime adapter produced for one case.

    ``None`` means the adapter did not produce the field. A supported field
    that comes back as ``None`` is a conformance failure.
    """

    normalized: str | None = None
    tokens: Sequence[str] | None = None
    input_ids: Sequence[int] | None = None
    attention_mask: Sequence[int] | None = None
    token_type_ids: Sequence[int] | None = None
    special_tokens_mask: Sequence[int] | None = None
    offsets: Sequence[Sequence[int]] | None = None
    truncated_input_ids: Sequence[int] | None = None
    window_ranges: Sequence[Sequence[int]] | None = None
    window_input_ids: Sequence[Sequence[int]] | None = None


@dataclass(frozen=True)
class ConformanceMismatch:
    """One value-free conformance failure.

    Attributes:
        case_id: Case id, or ``"*"`` for runtime-level findings.
        field: Conformance field, or ``"runtime"``.
        index: First diverging token, scalar or window index when known.
        code: Controlled reason code.
        feature: Feature tag for stale divergence declarations.
    """

    case_id: str
    field: str
    index: int | None
    code: str
    feature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mismatch."""
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "field": self.field,
            "index": self.index,
            "code": self.code,
        }
        if self.feature is not None:
            payload["feature"] = self.feature
        return payload


@dataclass(frozen=True)
class TokenizerConformanceReport:
    """Value-free result of checking one runtime against the vectors.

    Attributes:
        contract_version: Tokenizer contract version that was checked.
        runtime: Runtime id from the support file.
        vectors_sha256: Digest of the canonical vectors payload.
        checked: Number of asserted ``(case, field)`` comparisons.
        unsupported_fields: ``(field, reason)`` pairs the runtime declared
            unsupported.
        declared_divergences: ``(case_id, field, reason)`` comparisons that
            diverged as declared.
        mismatches: Undeclared failures.
    """

    contract_version: str
    runtime: str
    vectors_sha256: str
    checked: int
    unsupported_fields: tuple[tuple[str, str], ...]
    declared_divergences: tuple[tuple[str, str, str], ...]
    mismatches: tuple[ConformanceMismatch, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Return whether the runtime conforms, including declarations."""
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, value-free report."""
        return {
            "contract_version": self.contract_version,
            "runtime": self.runtime,
            "vectors_sha256": self.vectors_sha256,
            "status": "pass" if self.passed else "fail",
            "checked": self.checked,
            "unsupported_fields": [
                {"field": name, "reason": reason}
                for name, reason in self.unsupported_fields
            ],
            "declared_divergences": [
                {"case_id": case_id, "field": name, "reason": reason}
                for case_id, name, reason in self.declared_divergences
            ],
            "mismatches": [item.to_dict() for item in self.mismatches],
        }

    def require_passed(self) -> None:
        """Raise when the runtime does not conform.

        Raises:
            TokenizerConformanceError: With the first mismatch's case id,
                field, index and reason code.
        """
        if self.mismatches:
            raise TokenizerConformanceError(self)


class TokenizerConformanceError(RuntimeError):
    """Raised when tokenizer inputs diverge before model execution.

    Attributes:
        report: The failing report.
    """

    def __init__(self, report: TokenizerConformanceReport) -> None:
        first = report.mismatches[0]
        index = "-" if first.index is None else str(first.index)
        super().__init__(
            f"{first.code} runtime={report.runtime} case={first.case_id} "
            f"field={first.field} index={index} "
            f"mismatches={len(report.mismatches)}"
        )
        self.report = report


def build_synthetic_tokenizer_files() -> dict[str, bytes]:
    """Build the synthetic tokenizer assets deterministically.

    The character vocabulary comes from :data:`SYNTHETIC_VOCAB_CORPUS` after
    reference normalization, in both word-initial and ``##`` continuation
    forms, followed by :data:`SYNTHETIC_WORD_PIECES`.

    Returns:
        Mapping of fixture-relative paths to file bytes for ``tokenizer.json``,
        ``tokenizer_config.json`` and a minimal ``config.json``.
    """
    normalizer = {
        "type": "BertNormalizer",
        "clean_text": True,
        "handle_chinese_chars": True,
        "strip_accents": None,
        "lowercase": True,
    }
    seed = ReferenceTokenizer(_tokenizer_payload(normalizer, _special_vocab()))
    characters: set[str] = set()
    for line in SYNTHETIC_VOCAB_CORPUS:
        characters.update(char for char in seed.normalize(line) if not char.isspace())
    vocab = _special_vocab()
    for char in sorted(characters):
        vocab.setdefault(char, len(vocab))
    for char in sorted(characters):
        vocab.setdefault("##" + char, len(vocab))
    for piece in SYNTHETIC_WORD_PIECES:
        vocab.setdefault(piece, len(vocab))
    config = {
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": True,
        "tokenize_chinese_chars": True,
        "strip_accents": None,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 512,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    model_config = {"model_type": "bert", "synthetic_only": True}
    return {
        f"{TOKENIZER_DIRNAME}/config.json": _pretty_json(model_config),
        f"{TOKENIZER_DIRNAME}/tokenizer.json": _pretty_json(
            _tokenizer_payload(normalizer, vocab)
        ),
        f"{TOKENIZER_DIRNAME}/tokenizer_config.json": _pretty_json(config),
    }


def build_vectors(
    tokenizer_json: Mapping[str, Any],
    *,
    tokenizer_sha256: str,
    cases: Sequence[ConformanceCase] = SYNTHETIC_CASES,
) -> dict[str, Any]:
    """Build the conformance vectors from the reference semantics.

    Args:
        tokenizer_json: Parsed synthetic ``tokenizer.json``.
        tokenizer_sha256: Hex digest of the ``tokenizer.json`` bytes.
        cases: Synthetic cases to encode.

    Returns:
        The vectors payload.

    Raises:
        ValueError: If case ids or features are malformed or duplicated.
    """
    reference = ReferenceTokenizer(tokenizer_json)
    seen: set[str] = set()
    encoded_cases = []
    features: set[str] = set()
    for case in cases:
        if _CASE_ID_RE.fullmatch(case.id) is None or case.id in seen:
            raise ValueError("conformance case ids must be unique kebab-case")
        seen.add(case.id)
        if not case.features or any(
            _FEATURE_RE.fullmatch(item) is None for item in case.features
        ):
            raise ValueError("conformance case features must be controlled tags")
        features.update(case.features)
        encoding = reference.encode(case.text)
        expected: dict[str, Any] = {
            "normalized": reference.normalize(case.text),
            "tokens": list(encoding.tokens),
            "input_ids": list(encoding.input_ids),
            "attention_mask": list(encoding.attention_mask),
            "token_type_ids": list(encoding.token_type_ids),
            "special_tokens_mask": list(encoding.special_tokens_mask),
            "offsets": [list(item) for item in encoding.offsets],
        }
        entry: dict[str, Any] = {
            "id": case.id,
            "features": sorted(case.features),
            "text": case.text,
        }
        if case.window is not None:
            max_length, stride = case.window
            windows = reference.windows(case.text, max_length=max_length, stride=stride)
            entry["window"] = {"max_length": max_length, "stride": stride}
            # Single-pass truncation keeps the template and drops overflow.
            expected["truncated_input_ids"] = list(windows[0].input_ids)
            expected["window_ranges"] = [
                [item.content_start, item.content_end] for item in windows
            ]
            expected["window_input_ids"] = [list(item.input_ids) for item in windows]
        entry["expected"] = expected
        encoded_cases.append(entry)
    return {
        "schema": VECTORS_SCHEMA,
        "version": FORMAT_VERSION,
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "synthetic_only": True,
        "offset_unit": "unicode_scalar",
        "reference": {
            "semantics": "huggingface-tokenizers-rust",
            "implementation": "openmed.eval.tokenizer_reference",
        },
        "tokenizer": {
            "id": SYNTHETIC_TOKENIZER_ID,
            "directory": TOKENIZER_DIRNAME,
            "sha256": tokenizer_sha256,
            "special_tokens": {
                token: reference.vocab[token] for token in _SPECIAL_TOKENS
            },
            "template_special_tokens": reference.template_special_count,
        },
        "window_semantics": {
            "direction": "right",
            "capacity": "max_length - template_special_tokens",
            "step": "capacity - stride",
            "ranges": "sequence-token indices, half-open",
            "truncated": "first window, overflow dropped",
        },
        "fields": list(CONFORMANCE_FIELDS),
        "features": sorted(features),
        "cases": encoded_cases,
    }


def validate_runtime_support(
    support: Mapping[str, Any], vectors: Mapping[str, Any]
) -> tuple[str, ...]:
    """Validate the hand-reviewed runtime support declarations.

    Args:
        support: Parsed runtime support payload.
        vectors: Parsed vectors payload the declarations refer to.

    Returns:
        Controlled problem codes; empty when the declarations are valid.
    """
    problems: list[str] = []
    if (
        support.get("schema") != SUPPORT_SCHEMA
        or support.get("version") != FORMAT_VERSION
    ):
        problems.append("support_schema")
    if support.get("contract_version") != vectors.get("contract_version"):
        problems.append("support_contract_version")
    runtimes = support.get("runtimes")
    if not isinstance(runtimes, Mapping) or not runtimes:
        return tuple(problems + ["support_runtimes"])
    features = set(vectors.get("features", ()))
    for runtime, entry in runtimes.items():
        if _RUNTIME_RE.fullmatch(str(runtime)) is None or not isinstance(
            entry, Mapping
        ):
            problems.append("support_runtime_id")
            continue
        for key in ("library", "adapter"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                problems.append(f"support_{key}:{runtime}")
        fields = entry.get("fields")
        if not isinstance(fields, Mapping) or set(fields) != set(CONFORMANCE_FIELDS):
            problems.append(f"support_fields:{runtime}")
            fields = {}
        for status in fields.values():
            if not isinstance(status, Mapping):
                problems.append(f"support_field_status:{runtime}")
            elif status.get("status") != "supported" and (
                status.get("status") != "unsupported"
                or _REASON_RE.fullmatch(str(status.get("reason", ""))) is None
            ):
                problems.append(f"support_field_status:{runtime}")
        divergences = entry.get("divergences", [])
        if not isinstance(divergences, list):
            problems.append(f"support_divergences:{runtime}")
            continue
        for item in divergences:
            if (
                not isinstance(item, Mapping)
                or item.get("feature") not in features
                or not item.get("fields")
                or not set(item.get("fields", ())) <= set(CONFORMANCE_FIELDS)
                or _REASON_RE.fullmatch(str(item.get("reason", ""))) is None
            ):
                problems.append(f"support_divergence:{runtime}")
                continue
            for name in item["fields"]:
                status = fields.get(name)
                if not isinstance(status, Mapping) or status.get("status") != (
                    "supported"
                ):
                    problems.append(f"support_divergence_unsupported_field:{runtime}")
    return tuple(problems)


def verify_tokenizer_conformance(
    vectors: Mapping[str, Any],
    support: Mapping[str, Any],
    runtime: str,
    runner: Callable[[Mapping[str, Any]], RuntimeEncoding],
) -> TokenizerConformanceReport:
    """Check one runtime adapter against the conformance vectors.

    Supported fields must match exactly. Fields the runtime declares
    unsupported are reported, not compared. Comparisons for cases tagged with
    a declared divergence feature are reported as declared divergences when
    they differ; a declaration that no case triggers is stale and fails. A
    supported field the adapter does not produce always fails.

    Args:
        vectors: Parsed vectors payload.
        support: Parsed runtime support payload.
        runtime: Runtime id declared in ``support``.
        runner: Callable that receives one vector case (``text`` plus optional
            ``window``) and returns the adapter's :class:`RuntimeEncoding`.

    Returns:
        A value-free conformance report.
    """
    digest = _canonical_sha256(vectors)
    if (
        vectors.get("contract_version") != TOKENIZER_CONTRACT_VERSION
        or support.get("contract_version") != TOKENIZER_CONTRACT_VERSION
    ):
        return _failed_report(runtime, digest, CODE_CONTRACT_MISMATCH)
    entry = (support.get("runtimes") or {}).get(runtime)
    if not isinstance(entry, Mapping):
        return _failed_report(runtime, digest, CODE_RUNTIME_UNDECLARED)
    statuses = entry.get("fields", {})
    unsupported = tuple(
        (name, str(statuses.get(name, {}).get("reason", "undeclared")))
        for name in CONFORMANCE_FIELDS
        if statuses.get(name, {}).get("status") != "supported"
    )
    unsupported_names = {name for name, _ in unsupported}
    divergences = [
        (str(item["feature"]), tuple(item["fields"]), str(item["reason"]))
        for item in entry.get("divergences", [])
    ]
    observed: set[tuple[str, str]] = set()
    mismatches: list[ConformanceMismatch] = []
    declared: list[tuple[str, str, str]] = []
    checked = 0
    for case in vectors.get("cases", []):
        case_id = str(case["id"])
        try:
            produced = runner(case)
        except Exception:
            # Runtime messages may echo input; keep only the controlled code.
            mismatches.append(
                ConformanceMismatch(case_id, "runtime", None, CODE_RUNTIME_ERROR)
            )
            continue
        expected = case["expected"]
        case_features = set(case.get("features", ()))
        for name in CONFORMANCE_FIELDS:
            if name not in expected or name in unsupported_names:
                continue
            actual = getattr(produced, name)
            if actual is None:
                mismatches.append(
                    ConformanceMismatch(case_id, name, None, CODE_FIELD_MISSING)
                )
                continue
            finding = _compare(expected[name], actual)
            if finding is None:
                checked += 1
                continue
            matching = [
                (feature, reason)
                for feature, fields, reason in divergences
                if feature in case_features and name in fields
            ]
            if matching:
                for feature, _ in matching:
                    observed.add((feature, name))
                declared.append((case_id, name, matching[0][1]))
                continue
            checked += 1
            code, index = finding
            mismatches.append(ConformanceMismatch(case_id, name, index, code))
    for feature, fields, _reason in divergences:
        for name in fields:
            if (feature, name) not in observed:
                mismatches.append(
                    ConformanceMismatch("*", name, None, CODE_DIVERGENCE_STALE, feature)
                )
    return TokenizerConformanceReport(
        contract_version=TOKENIZER_CONTRACT_VERSION,
        runtime=runtime,
        vectors_sha256=digest,
        checked=checked,
        unsupported_fields=unsupported,
        declared_divergences=tuple(declared),
        mismatches=tuple(mismatches),
    )


def require_tokenizer_conformance(
    vectors: Mapping[str, Any],
    support: Mapping[str, Any],
    runtime: str,
    runner: Callable[[Mapping[str, Any]], RuntimeEncoding],
) -> TokenizerConformanceReport:
    """Verify a runtime and raise before any model execution on divergence.

    Call this before comparing inference results so drift is attributed to
    tokenization rather than to model execution.

    Args:
        vectors: Parsed vectors payload.
        support: Parsed runtime support payload.
        runtime: Runtime id declared in ``support``.
        runner: Runtime adapter callable; see
            :func:`verify_tokenizer_conformance`.

    Returns:
        The passing report.

    Raises:
        TokenizerConformanceError: If any undeclared divergence is found.
    """
    report = verify_tokenizer_conformance(vectors, support, runtime, runner)
    report.require_passed()
    return report


def build_manifest(files: Mapping[str, bytes]) -> dict[str, Any]:
    """Build the digest manifest for fixture files.

    Args:
        files: Mapping of fixture-relative paths to bytes, excluding the
            manifest itself.

    Returns:
        Manifest payload with per-file digests and the contract digest.
    """
    entries = [
        {
            "path": path,
            "role": _file_role(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for path, data in sorted(files.items())
    ]
    identity = json.dumps(
        {"contract_version": TOKENIZER_CONTRACT_VERSION, "files": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": MANIFEST_SCHEMA,
        "version": FORMAT_VERSION,
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "synthetic_only": True,
        "files": entries,
        "contract_digest": "sha256:" + hashlib.sha256(identity).hexdigest(),
    }


def tokenizer_contract_reference(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the ``tokenizer_contract`` value a model bundle records.

    Args:
        manifest: Parsed conformance manifest.

    Returns:
        ``{"version": ..., "digest": ...}`` binding a bundle to the tokenizer
        contract its runtime adapters were qualified against.

    Raises:
        ValueError: If the manifest is not a conformance manifest or its
            contract digest does not match its file entries.
    """
    version = manifest.get("contract_version")
    digest = manifest.get("contract_digest")
    files = manifest.get("files")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or version != TOKENIZER_CONTRACT_VERSION
        or not isinstance(digest, str)
        or _DIGEST_RE.fullmatch(digest) is None
        or not isinstance(files, list)
    ):
        raise ValueError("not a tokenizer conformance manifest")
    identity = json.dumps(
        {"contract_version": version, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if digest != "sha256:" + hashlib.sha256(identity).hexdigest():
        raise ValueError("tokenizer conformance manifest digest mismatch")
    return {"version": version, "digest": digest}


def build_fixture_files(support_bytes: bytes) -> dict[str, bytes]:
    """Build every generated fixture file for the given support declarations.

    Args:
        support_bytes: Bytes of the hand-reviewed runtime support file.

    Returns:
        Mapping of fixture-relative paths to bytes, including the manifest.
    """
    files = build_synthetic_tokenizer_files()
    tokenizer_bytes = files[f"{TOKENIZER_DIRNAME}/tokenizer.json"]
    vectors = build_vectors(
        json.loads(tokenizer_bytes),
        tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
    files[VECTORS_FILENAME] = _vectors_json(vectors)
    files[SUPPORT_FILENAME] = support_bytes
    files[MANIFEST_FILENAME] = _pretty_json(build_manifest(files))
    return files


def write_fixture_dir(directory: Path) -> tuple[str, ...]:
    """Regenerate the fixture directory in place.

    The runtime support file is hand-reviewed and is read, never rewritten.

    Args:
        directory: Fixture directory.

    Returns:
        Fixture-relative paths that changed.

    Raises:
        ValueError: If the runtime support declarations are invalid.
    """
    support_bytes = (directory / SUPPORT_FILENAME).read_bytes()
    files = build_fixture_files(support_bytes)
    problems = validate_runtime_support(
        json.loads(support_bytes), json.loads(files[VECTORS_FILENAME])
    )
    if problems:
        raise ValueError("invalid runtime support: " + ",".join(problems))
    changed = []
    for relative, data in sorted(files.items()):
        target = directory / relative
        if target.is_file() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        changed.append(relative)
    return tuple(changed)


def check_fixture_dir(directory: Path) -> tuple[str, ...]:
    """Return controlled problem codes for a stale or tampered fixture set.

    Args:
        directory: Fixture directory.

    Returns:
        Problem codes naming the fixture role, never file contents; empty when
        the committed bytes match a fresh generation.
    """
    support_path = directory / SUPPORT_FILENAME
    if not support_path.is_file():
        return ("missing:runtime_support",)
    support_bytes = support_path.read_bytes()
    files = build_fixture_files(support_bytes)
    problems = []
    for relative, data in sorted(files.items()):
        target = directory / relative
        role = _file_role(relative)
        if not target.is_file():
            problems.append(f"missing:{role}")
        elif target.read_bytes() != data:
            problems.append(f"stale:{role}")
    try:
        support = json.loads(support_bytes)
    except json.JSONDecodeError:
        return tuple(problems + ["support_json"])
    problems.extend(
        validate_runtime_support(support, json.loads(files[VECTORS_FILENAME]))
    )
    return tuple(problems)


def load_fixture_dir(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the vectors, runtime support and manifest payloads.

    Args:
        directory: Fixture directory.

    Returns:
        ``(vectors, support, manifest)`` parsed payloads.
    """
    vectors, support, manifest = (
        json.loads((directory / name).read_text(encoding="utf-8"))
        for name in (VECTORS_FILENAME, SUPPORT_FILENAME, MANIFEST_FILENAME)
    )
    return vectors, support, manifest


def _compare(expected: Any, actual: Any) -> tuple[str, int | None] | None:
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return (CODE_VALUE_MISMATCH, None)
        expected_items: list[Any] = list(expected)
        actual_items: list[Any] = list(actual)
    else:
        expected_items = [_plain(item) for item in expected]
        try:
            actual_items = [_plain(item) for item in actual]
        except TypeError:
            return (CODE_VALUE_MISMATCH, None)
    for index, (left, right) in enumerate(zip(expected_items, actual_items)):
        if left != right:
            return (CODE_VALUE_MISMATCH, index)
    if len(expected_items) != len(actual_items):
        return (CODE_LENGTH_MISMATCH, min(len(expected_items), len(actual_items)))
    return None


def _plain(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Iterable):
        return [_plain(item) for item in value]
    return value


def _failed_report(runtime: str, digest: str, code: str) -> TokenizerConformanceReport:
    return TokenizerConformanceReport(
        contract_version=TOKENIZER_CONTRACT_VERSION,
        runtime=runtime,
        vectors_sha256=digest,
        checked=0,
        unsupported_fields=(),
        declared_divergences=(),
        mismatches=(ConformanceMismatch("*", "runtime", None, code),),
    )


def _special_vocab() -> dict[str, int]:
    return {token: index for index, token in enumerate(_SPECIAL_TOKENS)}


def _tokenizer_payload(
    normalizer: Mapping[str, Any], vocab: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token in _SPECIAL_TOKENS
        ],
        "normalizer": dict(normalizer),
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
            ],
            "special_tokens": {
                token: {"id": token, "ids": [vocab[token]], "tokens": [token]}
                for token in ("[CLS]", "[SEP]")
            },
        },
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": dict(sorted(vocab.items(), key=lambda item: item[1])),
        },
    }


def _file_role(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return {
        "config.json": "tokenizer_model_config",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer_config",
        VECTORS_FILENAME: "vectors",
        SUPPORT_FILENAME: "runtime_support",
        MANIFEST_FILENAME: "manifest",
    }.get(name, "unknown")


def _pretty_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _vectors_json(vectors: Mapping[str, Any]) -> bytes:
    # One case per line keeps fixture diffs reviewable.
    head = {key: value for key, value in vectors.items() if key != "cases"}
    lines = json.dumps(head, indent=2, ensure_ascii=True).rstrip("}").rstrip()
    cases = ",\n".join(
        "    " + json.dumps(case, ensure_ascii=True, separators=(",", ":"))
        for case in vectors["cases"]
    )
    return f'{lines},\n  "cases": [\n{cases}\n  ]\n}}\n'.encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
