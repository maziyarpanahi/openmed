"""Deterministic atomic-claim segmentation for de-identified summaries.

The segmenter preserves exact character offsets into the supplied summary and
splits only boundaries that are stable enough for downstream citation and NLI
checks. Ambiguous compound claims are retained as one span and explicitly
routed to human review instead of being presented as safely atomic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Final, Literal

from openmed.processing import segment_text

SUMMARY_CLAIM_SEGMENTATION_SCHEMA_VERSION: Final[int] = 1

ReviewReason = Literal["unsegmentable_compound", "unsupported_language"]

_REVIEW_REASONS: Final[frozenset[str]] = frozenset(
    {"unsegmentable_compound", "unsupported_language"}
)
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")
_HARD_CLAUSE_BOUNDARY_RE = re.compile(r"[;；\n]+")
_INDEPENDENT_CONNECTOR_RE = re.compile(
    r",\s*(?:and|but|yet)\b|\b(?:and|but|yet)\b|,",
    re.IGNORECASE,
)
_COMPOUND_MARKER_RE = re.compile(
    r"\b(?:and|or|nor|but|yet|although|though|because|while|whereas|"
    r"which|who|whose|that)\b",
    re.IGNORECASE,
)
_FINITE_PREDICATE_RE = re.compile(
    r"\b(?:"
    r"am|is|are|was|were|be|been|being|"
    r"has|have|had|do|does|did|"
    r"can|could|may|might|must|shall|should|will|would|"
    r"report(?:s|ed)?|den(?:y|ies|ied)|take(?:s|n)?|"
    r"receiv(?:e|es|ed)|show(?:s|ed)?|reveal(?:s|ed)?|"
    r"demonstrat(?:e|es|ed)|indicat(?:e|es|ed)|"
    r"improv(?:e|es|ed)|worsen(?:s|ed)?|persist(?:s|ed)?|"
    r"resolv(?:e|es|ed)|remain(?:s|ed)?|develop(?:s|ed)?|"
    r"start(?:s|ed)?|stop(?:s|ped)?|continu(?:e|es|ed)|"
    r"increas(?:e|es|ed)|decreas(?:e|es|ed)|"
    r"present(?:s|ed)?|requir(?:e|es|ed)|recommend(?:s|ed)?|"
    r"plan(?:s|ned)?"
    r")\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class SummaryClaimSegmentationError(ValueError):
    """Raised when a summary cannot be segmented into safe offset spans."""


@dataclass(frozen=True, slots=True)
class SummaryClaimSegment:
    """One exact summary slice prepared for claim-level verification.

    ``start`` and ``end`` are half-open character offsets into the caller's
    post-de-identification summary. The text is available for immediate
    verification, while :meth:`to_dict` deliberately omits it so persisted
    reports remain value-free.
    """

    text: str = field(repr=False)
    start: int
    end: int
    review_required: bool = False
    review_reason: ReviewReason | None = None

    def __post_init__(self) -> None:
        if type(self.text) is not str or not self.text:
            raise SummaryClaimSegmentationError("claim text must be non-empty")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
            or self.end - self.start != len(self.text)
        ):
            raise SummaryClaimSegmentationError("invalid claim output offset")
        if type(self.review_required) is not bool:
            raise SummaryClaimSegmentationError("invalid claim review marker")
        if self.review_reason is not None and self.review_reason not in _REVIEW_REASONS:
            raise SummaryClaimSegmentationError("invalid claim review reason")
        if self.review_required is not (self.review_reason is not None):
            raise SummaryClaimSegmentationError("inconsistent claim review metadata")

    @property
    def offset(self) -> tuple[int, int]:
        """Return the half-open output offset as ``(start, end)``."""

        return self.start, self.end

    def to_dict(self) -> dict[str, object]:
        """Return value-free offsets and controlled review metadata."""

        return {
            "output_offset": {"start": self.start, "end": self.end},
            "review_required": self.review_required,
            "review_reason": self.review_reason,
        }


@dataclass(frozen=True, slots=True)
class SummaryClaimSegmentation:
    """Immutable result containing ordered claim spans and safe metadata."""

    segments: tuple[SummaryClaimSegment, ...]
    language: str = "en"
    schema_version: int = SUMMARY_CLAIM_SEGMENTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SUMMARY_CLAIM_SEGMENTATION_SCHEMA_VERSION
        ):
            raise SummaryClaimSegmentationError("unsupported claim segmentation schema")
        normalized_language = _normalize_language(self.language)
        records = tuple(self.segments)
        if any(type(segment) is not SummaryClaimSegment for segment in records):
            raise SummaryClaimSegmentationError("invalid claim segment collection")
        if any(left.end > right.start for left, right in zip(records, records[1:])):
            raise SummaryClaimSegmentationError("overlapping claim output offsets")
        object.__setattr__(self, "language", normalized_language)
        object.__setattr__(self, "segments", records)

    @property
    def review_required(self) -> bool:
        """Return whether any claim must be reviewed before verification."""

        return any(segment.review_required for segment in self.segments)

    @property
    def review_required_count(self) -> int:
        """Return the number of claims routed to human review."""

        return sum(segment.review_required for segment in self.segments)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic metadata-only representation."""

        return {
            "schema_version": self.schema_version,
            "language": self.language,
            "segment_count": len(self.segments),
            "review_required": self.review_required,
            "review_required_count": self.review_required_count,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def to_json(self) -> str:
        """Return byte-stable JSON containing no claim text."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def segment_summary_claims(
    summary_text: str,
    *,
    language: str = "en",
) -> SummaryClaimSegmentation:
    """Segment a post-de-identification summary into stable atomic claims.

    Sentence boundaries and semicolons are treated as stable. English commas
    and ``and``/``but``/``yet`` boundaries are split only when both sides have
    an explicit subject and finite predicate. Remaining compound markers are
    retained in one exact slice and marked ``unsegmentable_compound``. For
    other languages, sentence-level offsets are still preserved but every
    non-empty segment is marked ``unsupported_language`` because this module
    cannot prove clause atomicity without language-specific rules.

    Args:
        summary_text: Post-de-identification summary text. Raw source notes
            must not be supplied.
        language: BCP-47-like language tag used by the local sentence
            segmenter. English tags enable deterministic clause splitting.

    Returns:
        An immutable ordered segmentation with exact half-open output offsets.

    Raises:
        TypeError: If ``summary_text`` or ``language`` is not a string.
        SummaryClaimSegmentationError: If the language tag or sentence offsets
            are invalid. Error messages never include input text.
    """

    if type(summary_text) is not str:
        raise TypeError("summary_text must be a string")
    normalized_language = _normalize_language(language)
    if not summary_text:
        return SummaryClaimSegmentation(segments=(), language=normalized_language)

    try:
        sentence_spans = segment_text(summary_text, language=normalized_language)
    except Exception:
        raise SummaryClaimSegmentationError(
            "summary sentence segmentation failed"
        ) from None

    english = normalized_language == "en" or normalized_language.startswith("en-")
    segments: list[SummaryClaimSegment] = []
    for sentence in sentence_spans:
        start = sentence.start
        end = sentence.end
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > len(summary_text)
            or sentence.text != summary_text[start:end]
        ):
            raise SummaryClaimSegmentationError("invalid sentence output offset")

        for clause_start, clause_end in _hard_clause_spans(summary_text, start, end):
            clause_spans = (
                _split_independent_english_clauses(
                    summary_text, clause_start, clause_end
                )
                if english
                else ((clause_start, clause_end),)
            )
            for claim_start, claim_end in clause_spans:
                claim_start, claim_end = _trim_bounds(
                    summary_text, claim_start, claim_end
                )
                if claim_start == claim_end:
                    continue
                claim_text = summary_text[claim_start:claim_end]
                review_reason: ReviewReason | None = None
                if not english:
                    review_reason = "unsupported_language"
                elif _COMPOUND_MARKER_RE.search(claim_text) is not None:
                    review_reason = "unsegmentable_compound"
                segments.append(
                    SummaryClaimSegment(
                        text=claim_text,
                        start=claim_start,
                        end=claim_end,
                        review_required=review_reason is not None,
                        review_reason=review_reason,
                    )
                )

    return SummaryClaimSegmentation(
        segments=tuple(segments),
        language=normalized_language,
    )


def _normalize_language(language: object) -> str:
    if type(language) is not str:
        raise TypeError("language must be a string")
    if _LANGUAGE_RE.fullmatch(language) is None:
        raise SummaryClaimSegmentationError("invalid summary language tag")
    return language.replace("_", "-").casefold()


def _hard_clause_spans(
    text: str,
    start: int,
    end: int,
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for boundary in _HARD_CLAUSE_BOUNDARY_RE.finditer(text, start, end):
        claim_start, claim_end = _trim_bounds(text, cursor, boundary.start())
        if claim_start < claim_end:
            spans.append((claim_start, claim_end))
        cursor = boundary.end()
    claim_start, claim_end = _trim_bounds(text, cursor, end)
    if claim_start < claim_end:
        spans.append((claim_start, claim_end))
    return tuple(spans)


def _split_independent_english_clauses(
    text: str,
    start: int,
    end: int,
) -> tuple[tuple[int, int], ...]:
    start, end = _trim_bounds(text, start, end)
    if start == end:
        return ()
    candidate = text[start:end]
    for connector in _INDEPENDENT_CONNECTOR_RE.finditer(candidate):
        connector_start = start + connector.start()
        connector_end = start + connector.end()
        if not _is_top_level(candidate, connector.start()):
            continue
        left_start, left_end = _trim_bounds(text, start, connector_start)
        right_start, right_end = _trim_bounds(text, connector_end, end)
        if not (
            _has_subject_and_predicate(text[left_start:left_end])
            and _has_subject_and_predicate(text[right_start:right_end])
        ):
            continue
        return (
            *_split_independent_english_clauses(text, left_start, left_end),
            *_split_independent_english_clauses(text, right_start, right_end),
        )
    return ((start, end),)


def _has_subject_and_predicate(clause: str) -> bool:
    predicate = _FINITE_PREDICATE_RE.search(clause)
    if predicate is None:
        return False
    return _WORD_RE.search(clause, 0, predicate.start()) is not None


def _is_top_level(text: str, end: int) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in text[:end]:
        if character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
    return not stack


def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


__all__ = [
    "SUMMARY_CLAIM_SEGMENTATION_SCHEMA_VERSION",
    "ReviewReason",
    "SummaryClaimSegment",
    "SummaryClaimSegmentation",
    "SummaryClaimSegmentationError",
    "segment_summary_claims",
]
