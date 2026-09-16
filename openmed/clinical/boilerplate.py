"""Deterministic span annotations for clinical boilerplate and copy-forward text.

The detectors in this module are local, rules-first, and non-destructive. They
return half-open offsets into the original text and never remove or rewrite the
note. The bundled phrase table is synthetic and contains no restricted note
corpus material.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

DEFAULT_BOILERPLATE_TEMPLATE_RESOURCE = "data/boilerplate_templates.json"
BOILERPLATE_DETECTOR_VERSION = "openmed-boilerplate-v1"
COPY_FORWARD_DETECTOR_VERSION = "openmed-copy-forward-v1"

_BOILERPLATE_TYPE = "boilerplate"
_COPY_FORWARD_TYPE = "copy_forward"
_ALLOWED_TEMPLATE_SOURCE_TYPES = frozenset({"synthetic", "public_domain"})
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*")
_CHECKBOX_RE = re.compile(
    r"^(?:[-*\N{BULLET}]\s*)?(?:\[[ xX\N{CHECK MARK}\N{HEAVY CHECK MARK}-]\]|"
    r"[\N{BALLOT BOX}\N{BALLOT BOX WITH CHECK}\N{BALLOT BOX WITH X}])\s+\S"
)
_DOT_PHRASE_RE = re.compile(r"^\.[A-Za-z][\w.-]{2,}(?:\s|$)")
_PLACEHOLDER_RE = re.compile(
    r"(?:\*{3,}|_{3,}|\{\{[^\r\n]{0,80}\}\}|<<[^\r\n]{0,80}>>|"
    r"\[(?:insert|select|choose|enter)[^\]\r\n]{0,80}\])",
    re.IGNORECASE,
)
_MAX_SHINGLE_OCCURRENCES = 64
_MAX_COPY_FORWARD_TOKENS = 100_000


@dataclass(frozen=True)
class BoilerplateTemplate:
    """One validated entry from the bundled unrestricted template corpus."""

    template_id: str
    text: str
    source_type: str


@dataclass(frozen=True)
class BoilerplateSpan:
    """One non-destructive boilerplate annotation over original note offsets."""

    start: int
    end: int
    score: float
    provenance: Mapping[str, Any]
    type: str = field(default=_BOILERPLATE_TYPE, init=False)

    def __post_init__(self) -> None:
        _validate_annotation_fields(self.start, self.end, self.score)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready annotation mapping."""

        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class CopyForwardSpan:
    """One copied span with a caller-safe source reference and source offsets."""

    start: int
    end: int
    copied_from: str
    source_start: int
    source_end: int
    score: float
    provenance: Mapping[str, Any]
    type: str = field(default=_COPY_FORWARD_TYPE, init=False)

    def __post_init__(self) -> None:
        _validate_annotation_fields(self.start, self.end, self.score)
        if not isinstance(self.copied_from, str) or not self.copied_from.strip():
            raise ValueError("copied_from must be a non-empty source reference")
        if (
            isinstance(self.source_start, bool)
            or isinstance(self.source_end, bool)
            or not isinstance(self.source_start, int)
            or not isinstance(self.source_end, int)
        ):
            raise TypeError("source offsets must be integers")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("source offsets must satisfy 0 <= start < end")
        object.__setattr__(self, "copied_from", self.copied_from.strip())
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready annotation mapping."""

        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "copied_from": self.copied_from,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "score": self.score,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class _Token:
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class _BoilerplateCandidate:
    start: int
    end: int
    score: float
    rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class _CopyCandidate:
    start: int
    end: int
    source_reference: str
    source_start: int
    source_end: int
    matched_tokens: int
    match_kind: str


def detect_boilerplate(text: str) -> tuple[BoilerplateSpan, ...]:
    """Annotate template, checkbox, dot-phrase, and placeholder scaffolding.

    Matching uses the committed synthetic phrase corpus plus conservative line
    structure rules. Returned spans use half-open offsets into ``text`` and the
    input string is never changed.

    Args:
        text: Clinical note text to annotate.

    Returns:
        Deterministically ordered boilerplate annotations with rule provenance.

    Raises:
        TypeError: If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return ()

    templates = load_boilerplate_template_corpus()
    template_tokens = tuple(
        (template, tuple(token.normalized for token in _tokens(template.text)))
        for template in templates
    )
    candidates: list[_BoilerplateCandidate] = []

    for line_start, line_end in _content_lines(text):
        surface = text[line_start:line_end]
        stripped = surface.strip()
        if not stripped:
            continue
        content_start = line_start + len(surface) - len(surface.lstrip())
        content_end = line_end - (len(surface) - len(surface.rstrip()))
        line_tokens = _tokens(text, start=content_start, end=content_end)
        normalized_line = tuple(token.normalized for token in line_tokens)

        for template, normalized_template in template_tokens:
            for token_start in _subsequence_starts(
                normalized_line, normalized_template
            ):
                if len(normalized_line) == len(normalized_template):
                    match_start, match_end = content_start, content_end
                else:
                    match_start = line_tokens[token_start].start
                    match_end = line_tokens[
                        token_start + len(normalized_template) - 1
                    ].end
                    terminal = template.text.rstrip()[-1:]
                    if (
                        terminal in ".!?"
                        and text[match_end : match_end + 1] == terminal
                    ):
                        match_end += 1
                candidates.append(
                    _BoilerplateCandidate(
                        start=match_start,
                        end=match_end,
                        score=0.98,
                        rule_ids=(f"template:{template.template_id}",),
                    )
                )

        structural_rules: list[tuple[str, float]] = []
        if _CHECKBOX_RE.search(stripped):
            structural_rules.append(("structure:checkbox", 0.94))
        if _DOT_PHRASE_RE.search(stripped):
            structural_rules.append(("structure:dot_phrase", 0.96))
        if _PLACEHOLDER_RE.search(stripped):
            structural_rules.append(("structure:placeholder", 0.92))
        if structural_rules:
            candidates.append(
                _BoilerplateCandidate(
                    start=content_start,
                    end=content_end,
                    score=max(score for _, score in structural_rules),
                    rule_ids=tuple(rule_id for rule_id, _ in structural_rules),
                )
            )

    return tuple(
        BoilerplateSpan(
            start=candidate.start,
            end=candidate.end,
            score=candidate.score,
            provenance={
                "detector": BOILERPLATE_DETECTOR_VERSION,
                "corpus_version": _template_corpus_version(),
                "rule_ids": candidate.rule_ids,
            },
        )
        for candidate in _merge_boilerplate_candidates(candidates)
    )


def detect_copy_forward(
    text: str,
    *,
    source_documents: Mapping[str, str]
    | Sequence[Mapping[str, Any] | tuple[str, str]]
    | None = None,
    document_id: str | None = None,
    shingle_size: int = 5,
    min_tokens: int = 8,
) -> tuple[CopyForwardSpan, ...]:
    """Detect intra-note and caller-supplied cross-note copied token runs.

    Text is normalized with NFKC and case folding before word shingles are
    compared. Punctuation and whitespace changes therefore preserve a match,
    while returned spans still index the original note. Cross-document inputs
    are references supplied directly by the caller; this function performs no
    patient linking, storage, or network access.

    Args:
        text: Current clinical note text.
        source_documents: Optional prior documents, either a mapping from source
            reference to text, or a sequence of ``(reference, text)`` pairs or
            mappings with ``doc_id`` and ``text`` fields.
        document_id: Optional source reference used for intra-document matches.
        shingle_size: Number of normalized word tokens per shingle.
        min_tokens: Minimum contiguous copied token count to annotate.

    Returns:
        Non-overlapping copy-forward spans ordered by current-note offset.

    Raises:
        TypeError: If text or source documents have unsupported types.
        ValueError: If identifiers or detector options are invalid.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if document_id is not None and (
        not isinstance(document_id, str) or not document_id.strip()
    ):
        raise ValueError("document_id must be a non-empty string when provided")
    _validate_copy_options(shingle_size=shingle_size, min_tokens=min_tokens)

    current_tokens = _tokens(text)
    _validate_token_budget(current_tokens, label="current document")
    if len(current_tokens) < min_tokens:
        return ()

    candidates = _matching_copy_candidates(
        text,
        current_tokens,
        text,
        current_tokens,
        source_reference=document_id.strip() if document_id else "current_document",
        match_kind="intra_document",
        shingle_size=shingle_size,
        min_tokens=min_tokens,
        same_document=True,
    )
    for source_reference, source_text in _coerce_source_documents(source_documents):
        source_tokens = _tokens(source_text)
        _validate_token_budget(
            source_tokens, label=f"source document {source_reference!r}"
        )
        if len(source_tokens) < min_tokens:
            continue
        candidates.extend(
            _matching_copy_candidates(
                text,
                current_tokens,
                source_text,
                source_tokens,
                source_reference=source_reference,
                match_kind="cross_document",
                shingle_size=shingle_size,
                min_tokens=min_tokens,
                same_document=False,
            )
        )

    selected = _select_non_overlapping_copy_candidates(candidates)
    return tuple(
        CopyForwardSpan(
            start=candidate.start,
            end=candidate.end,
            copied_from=candidate.source_reference,
            source_start=candidate.source_start,
            source_end=candidate.source_end,
            score=1.0,
            provenance={
                "detector": COPY_FORWARD_DETECTOR_VERSION,
                "match_kind": candidate.match_kind,
                "normalization": "nfkc_casefold_word_shingles",
                "shingle_size": shingle_size,
                "matched_tokens": candidate.matched_tokens,
            },
        )
        for candidate in selected
    )


@lru_cache(maxsize=1)
def load_boilerplate_template_corpus() -> tuple[BoilerplateTemplate, ...]:
    """Load and validate the bundled synthetic/public template phrase corpus."""

    payload = _load_template_payload()
    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("boilerplate template corpus requires template entries")

    templates: list[BoilerplateTemplate] = []
    seen_ids: set[str] = set()
    for raw_template in raw_templates:
        if not isinstance(raw_template, Mapping):
            raise ValueError("boilerplate template entries must be objects")
        template_id = raw_template.get("id")
        text = raw_template.get("text")
        source_type = raw_template.get("source_type")
        if not isinstance(template_id, str) or not template_id.strip():
            raise ValueError("boilerplate templates require non-empty ids")
        if template_id in seen_ids:
            raise ValueError(f"duplicate boilerplate template id {template_id!r}")
        if not isinstance(text, str) or len(_tokens(text)) < 4:
            raise ValueError(
                f"boilerplate template {template_id!r} requires at least four tokens"
            )
        if source_type not in _ALLOWED_TEMPLATE_SOURCE_TYPES:
            raise ValueError(
                f"boilerplate template {template_id!r} has a restricted source type"
            )
        seen_ids.add(template_id)
        templates.append(
            BoilerplateTemplate(
                template_id=template_id,
                text=text,
                source_type=str(source_type),
            )
        )
    return tuple(templates)


def _load_template_payload() -> dict[str, Any]:
    resource = resources.files("openmed.clinical").joinpath(
        DEFAULT_BOILERPLATE_TEMPLATE_RESOURCE
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("boilerplate template corpus requires schema_version 1")
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("restricted_data") is not False
        or provenance.get("synthetic") is not True
        or not provenance.get("source")
        or not provenance.get("license")
    ):
        raise ValueError(
            "boilerplate template corpus requires unrestricted synthetic provenance"
        )
    corpus_version = payload.get("corpus_version")
    if not isinstance(corpus_version, str) or not corpus_version.strip():
        raise ValueError("boilerplate template corpus requires corpus_version")
    return payload


@lru_cache(maxsize=1)
def _template_corpus_version() -> str:
    return str(_load_template_payload()["corpus_version"])


def _validate_annotation_fields(start: int, end: int, score: float) -> None:
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError("annotation offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError("annotation offsets must satisfy 0 <= start < end")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError("annotation score must be between 0.0 and 1.0")


def _content_lines(text: str) -> tuple[tuple[int, int], ...]:
    lines: list[tuple[int, int]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        content_end = cursor + len(raw_line.rstrip("\r\n"))
        lines.append((cursor, content_end))
        cursor += len(raw_line)
    if not lines and text:
        lines.append((0, len(text)))
    return tuple(lines)


def _normalize_token(surface: str) -> str:
    return unicodedata.normalize("NFKC", surface).casefold()


def _tokens(text: str, *, start: int = 0, end: int | None = None) -> tuple[_Token, ...]:
    safe_end = len(text) if end is None else end
    return tuple(
        _Token(
            normalized=_normalize_token(match.group(0)),
            start=match.start(),
            end=match.end(),
        )
        for match in _TOKEN_RE.finditer(text, start, safe_end)
    )


def _subsequence_starts(
    sequence: tuple[str, ...],
    subsequence: tuple[str, ...],
) -> tuple[int, ...]:
    if not subsequence or len(subsequence) > len(sequence):
        return ()
    width = len(subsequence)
    return tuple(
        index
        for index in range(len(sequence) - width + 1)
        if sequence[index : index + width] == subsequence
    )


def _merge_boilerplate_candidates(
    candidates: Sequence[_BoilerplateCandidate],
) -> tuple[_BoilerplateCandidate, ...]:
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda item: (item.start, item.end, item.rule_ids))
    merged: list[_BoilerplateCandidate] = []
    for candidate in ordered:
        if not merged or candidate.start >= merged[-1].end:
            merged.append(candidate)
            continue
        previous = merged[-1]
        merged[-1] = _BoilerplateCandidate(
            start=min(previous.start, candidate.start),
            end=max(previous.end, candidate.end),
            score=max(previous.score, candidate.score),
            rule_ids=tuple(sorted(set((*previous.rule_ids, *candidate.rule_ids)))),
        )
    return tuple(merged)


def _validate_copy_options(*, shingle_size: int, min_tokens: int) -> None:
    for name, value in (("shingle_size", shingle_size), ("min_tokens", min_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least 2")
    if min_tokens < shingle_size:
        raise ValueError("min_tokens must be greater than or equal to shingle_size")


def _validate_token_budget(tokens: Sequence[_Token], *, label: str) -> None:
    if len(tokens) > _MAX_COPY_FORWARD_TOKENS:
        raise ValueError(f"{label} exceeds the copy-forward token budget")


def _coerce_source_documents(
    source_documents: Mapping[str, str]
    | Sequence[Mapping[str, Any] | tuple[str, str]]
    | None,
) -> tuple[tuple[str, str], ...]:
    if source_documents is None:
        return ()

    raw_items: Sequence[tuple[Any, Any] | Mapping[str, Any]]
    if isinstance(source_documents, Mapping):
        if "doc_id" in source_documents and "text" in source_documents:
            raw_items = (source_documents,)
        else:
            raw_items = tuple(source_documents.items())
    elif isinstance(source_documents, Sequence) and not isinstance(
        source_documents, (str, bytes)
    ):
        raw_items = source_documents
    else:
        raise TypeError("source_documents must be a mapping or sequence")

    documents: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if isinstance(raw_item, Mapping):
            source_reference = raw_item.get("doc_id")
            source_text = raw_item.get("text")
        elif isinstance(raw_item, tuple) and len(raw_item) == 2:
            source_reference, source_text = raw_item
        else:
            raise TypeError(
                f"source document at index {index} must provide a reference and text"
            )
        if not isinstance(source_reference, str) or not source_reference.strip():
            raise ValueError(
                f"source document at index {index} requires a non-empty reference"
            )
        source_reference = source_reference.strip()
        if source_reference in seen:
            raise ValueError(
                f"duplicate source document reference {source_reference!r}"
            )
        if not isinstance(source_text, str):
            raise TypeError(
                f"source document {source_reference!r} text must be a string"
            )
        seen.add(source_reference)
        documents.append((source_reference, source_text))
    return tuple(sorted(documents, key=lambda item: item[0]))


def _shingles(
    text: str,
    tokens: Sequence[_Token],
    size: int,
) -> tuple[tuple[str, ...] | None, ...]:
    return tuple(
        (
            None
            if "\n" in text[tokens[index].start : tokens[index + size - 1].end]
            or "\r" in text[tokens[index].start : tokens[index + size - 1].end]
            else tuple(token.normalized for token in tokens[index : index + size])
        )
        for index in range(len(tokens) - size + 1)
    )


def _shingle_index(
    shingles: Sequence[tuple[str, ...] | None],
) -> dict[tuple[str, ...], tuple[int, ...]]:
    mutable: dict[tuple[str, ...], list[int]] = {}
    for index, shingle in enumerate(shingles):
        if shingle is None:
            continue
        positions = mutable.setdefault(shingle, [])
        if len(positions) <= _MAX_SHINGLE_OCCURRENCES:
            positions.append(index)
    return {
        shingle: tuple(positions)
        for shingle, positions in mutable.items()
        if len(positions) <= _MAX_SHINGLE_OCCURRENCES
    }


def _matching_copy_candidates(
    current_text: str,
    current_tokens: Sequence[_Token],
    source_text: str,
    source_tokens: Sequence[_Token],
    *,
    source_reference: str,
    match_kind: str,
    shingle_size: int,
    min_tokens: int,
    same_document: bool,
) -> list[_CopyCandidate]:
    current_shingles = _shingles(current_text, current_tokens, shingle_size)
    source_shingles = _shingles(source_text, source_tokens, shingle_size)
    source_index = _shingle_index(source_shingles)
    positions_by_diagonal: dict[int, set[int]] = {}

    for current_index, shingle in enumerate(current_shingles):
        if shingle is None:
            continue
        for source_index_value in source_index.get(shingle, ()):
            if same_document and source_index_value + shingle_size > current_index:
                continue
            diagonal = source_index_value - current_index
            positions_by_diagonal.setdefault(diagonal, set()).add(current_index)

    candidates: list[_CopyCandidate] = []
    for diagonal, positions in sorted(positions_by_diagonal.items()):
        ordered_positions = sorted(positions)
        if not ordered_positions:
            continue
        run_start = ordered_positions[0]
        previous = run_start
        for current_index in (*ordered_positions[1:], -1):
            if current_index == previous + 1:
                previous = current_index
                continue
            shingle_count = previous - run_start + 1
            matched_tokens = shingle_count + shingle_size - 1
            if matched_tokens >= min_tokens:
                source_token_start = run_start + diagonal
                source_token_end = source_token_start + matched_tokens
                current_token_end = run_start + matched_tokens
                start, end = _expand_line_span(
                    current_text,
                    current_tokens[run_start].start,
                    current_tokens[current_token_end - 1].end,
                )
                source_start, source_end = _expand_line_span(
                    source_text,
                    source_tokens[source_token_start].start,
                    source_tokens[source_token_end - 1].end,
                )
                candidates.append(
                    _CopyCandidate(
                        start=start,
                        end=end,
                        source_reference=source_reference,
                        source_start=source_start,
                        source_end=source_end,
                        matched_tokens=matched_tokens,
                        match_kind=match_kind,
                    )
                )
            if current_index == -1:
                break
            run_start = current_index
            previous = current_index
    return candidates


def _expand_line_span(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = max(text.rfind("\n", 0, start), text.rfind("\r", 0, start)) + 1
    line_breaks = tuple(
        position
        for separator in ("\n", "\r")
        if (position := text.find(separator, end)) != -1
    )
    line_end = min(line_breaks) if line_breaks else len(text)
    prefix = text[line_start:start]
    suffix = text[end:line_end]
    if not prefix.strip() and not suffix.strip(" \t\r.,;:!?()[]{}"):
        return line_start + len(prefix) - len(prefix.lstrip()), line_end - len(
            text[line_start:line_end]
        ) + len(text[line_start:line_end].rstrip())
    return start, end


def _select_non_overlapping_copy_candidates(
    candidates: Sequence[_CopyCandidate],
) -> tuple[_CopyCandidate, ...]:
    selected: list[_CopyCandidate] = []
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.matched_tokens,
            -(item.end - item.start),
            item.start,
            item.source_reference,
            item.source_start,
        ),
    )
    for candidate in ranked:
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(
        sorted(
            selected,
            key=lambda item: (item.start, item.end, item.source_reference),
        )
    )


__all__ = [
    "BOILERPLATE_DETECTOR_VERSION",
    "COPY_FORWARD_DETECTOR_VERSION",
    "DEFAULT_BOILERPLATE_TEMPLATE_RESOURCE",
    "BoilerplateSpan",
    "BoilerplateTemplate",
    "CopyForwardSpan",
    "detect_boilerplate",
    "detect_copy_forward",
    "load_boilerplate_template_corpus",
]
