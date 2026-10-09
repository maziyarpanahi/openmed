"""Deterministic, local transcript accuracy over fixed in-memory pairs.

No input strings, token values, or caller identifiers are retained in scores.
These descriptive metrics do not qualify a provider or authorize clinical use.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

NORMALIZATION_VERSION = "transcript-nfc32-v1"
TRANSCRIPT_NOTICE = "Non-diagnostic accuracy evidence; reviewer confirmation required."
_LANGUAGES = ("en", "es", "fr", "de")
_WHITESPACE = frozenset(
    "\t\n\v\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004"
    "\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
_LOWER = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ",
    "abcdefghijklmnopqrstuvwxyzàáâãäåæçèéêëìíîïðñòóôõöøùúûüý",
)
_UCD = unicodedata.ucd_3_2_0
_MAX_ALIGNMENT_CELLS = 2_000_000


class ClinicalTermClass(str, Enum):
    """Controlled classes supplied by reference annotators."""

    MEDICATION = "medication"
    DOSE = "dose"
    UNIT = "unit"
    NEGATION = "negation"
    IDENTIFIER = "identifier"


@dataclass(frozen=True, slots=True)
class ReferenceTermSpan:
    """Half-open Python Unicode-scalar offsets into the original reference.

    Args:
        start: Inclusive source offset.
        end: Exclusive source offset.
        term_class: Controlled clinical class; no lexicon is inferred.
    """

    start: int
    end: int
    term_class: ClinicalTermClass

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
            or not isinstance(self.term_class, ClinicalTermClass)
        ):
            raise ValueError("invalid reference annotation")


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Value-free Levenshtein counts; rate is undefined for an empty reference."""

    reference: int
    hypothesis: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        """Return the total number of edit operations."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """Return errors/reference, or None when the denominator is zero."""
        return self.errors / self.reference if self.reference else None


@dataclass(frozen=True, slots=True)
class TermCounts:
    """Annotated span errors, counted once per term rather than per token."""

    term_class: ClinicalTermClass
    size: int
    errors: int


@dataclass(frozen=True, slots=True)
class TranscriptScore:
    """In-memory counts only, before aggregate publication suppression.

    Individual scores must not be published as small-cell-safe reports. Identifier
    recall measures ASR preservation of annotations, not redaction performance.
    """

    language: str
    words: EditCounts
    characters: EditCounts
    terms: tuple[TermCounts, ...]
    revision_tokens: int
    revision_opportunities: int
    final_revision_tokens: int
    final_revision_opportunities: int


def normalize_transcript(text: str, *, language: str = "en") -> str:
    """Normalize text transiently with the versioned conservative policy.

    Args:
        text: Caller-owned in-memory transcript; never logged or persisted.
        language: One of en, es, fr, de. Unknown policies fail closed.

    Returns:
        NFC (fixed Unicode 3.2 database) text with fixed whitespace collapsed.
        Policies use a fixed ASCII/Latin-1 lowercase map. Punctuation, digits
        and diacritics are retained. No transliteration or number rewriting.

    Raises:
        ValueError: Unsupported language policy.
        TypeError: Non-string text.
    """
    _validate_language(language)
    if not isinstance(text, str):
        raise TypeError("transcript must be a string")
    text = _UCD.normalize("NFC", text)
    text = text.translate(_LOWER)
    return " ".join(
        part
        for part in "".join(" " if c in _WHITESPACE else c for c in text).split(" ")
        if part
    )


def _validate_language(language: str) -> None:
    if not isinstance(language, str) or language not in _LANGUAGES:
        raise ValueError("unsupported transcript normalization policy")


def _tokens(text: str, language: str) -> list[tuple[str, int, int]]:
    if not isinstance(text, str):
        raise TypeError("transcript must be a string")
    result: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        if text[start] in _WHITESPACE:
            start += 1
            continue
        end = start + 1
        while end < len(text) and text[end] not in _WHITESPACE:
            end += 1
        value = normalize_transcript(text[start:end], language=language)
        result.append((value, start, end))
        start = end
    return result


def _align(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[EditCounts, list[tuple[int, int]]]:
    # Traceback stores operation bytes, not token values. Stable priority is
    # diagonal (match/substitution), deletion, insertion, including on ties.
    n, m = len(reference), len(hypothesis)
    if (n + 1) * (m + 1) > _MAX_ALIGNMENT_CELLS:
        raise ValueError("transcript alignment budget exceeded")
    trace = [bytearray(m + 1) for _ in range(n + 1)]
    trace[0][1:] = bytes([2]) * m
    previous = list(range(m + 1))
    for i, ref in enumerate(reference, 1):
        current = [i] + [0] * m
        trace[i][0] = 1
        for j, hyp in enumerate(hypothesis, 1):
            costs = (
                previous[j - 1] + (ref != hyp),
                previous[j] + 1,
                current[j - 1] + 1,
            )
            operation = min(range(3), key=costs.__getitem__)
            current[j] = costs[operation]
            trace[i][j] = operation
        previous = current
    substitutions = deletions = insertions = 0
    changes: list[tuple[int, int]] = []
    i, j = n, m
    while i or j:
        operation = trace[i][j]
        if operation == 0:
            i -= 1
            j -= 1
            if reference[i] != hypothesis[j]:
                substitutions += 1
                changes.append((i, i + 1))
        elif operation == 1:
            i -= 1
            deletions += 1
            changes.append((i, i + 1))
        else:
            j -= 1
            insertions += 1
            changes.append((i, i))
    return EditCounts(n, m, substitutions, deletions, insertions), changes


def _revision(previous: Sequence[str], current: Sequence[str]) -> int:
    prefix = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        prefix += 1
    return len(previous) - prefix


def score_transcript(
    reference: str,
    hypothesis: str,
    *,
    language: str = "en",
    spans: Iterable[ReferenceTermSpan] = (),
    partials: Iterable[str] = (),
) -> TranscriptScore:
    """Score a fixed final pair and optional chronological partial snapshots.

    Args:
        reference: Original reference, held in memory only.
        hypothesis: Final hypothesis, held in memory only.
        language: Explicit supported normalization policy.
        spans: Whole-token reference annotations; overlaps across classes are
            allowed, overlaps within a class are rejected.
        partials: Partial hypotheses in emission order for the same utterance.

    Returns:
        Immutable counts with no text. Churn counts previously emitted tokens
        beyond the common prefix; append-only growth is free. Final churn is
        the last partial to final transition, not a quality or clinical gate.

    Raises:
        ValueError: Invalid annotations, policy or alignment budget.
        TypeError: Invalid transcript or annotation types.
    """
    _validate_language(language)
    ref = _tokens(reference, language)
    hyp = _tokens(hypothesis, language)
    ref_words = [t[0] for t in ref]
    hyp_words = [t[0] for t in hyp]
    words, changes = _align(ref_words, hyp_words)
    characters, _ = _align(list("".join(ref_words)), list("".join(hyp_words)))
    annotations = tuple(spans)
    counts = {c: [0, 0] for c in ClinicalTermClass}
    seen: dict[ClinicalTermClass, list[tuple[int, int]]] = {
        c: [] for c in ClinicalTermClass
    }
    for span in annotations:
        if not isinstance(span, ReferenceTermSpan):
            raise TypeError("annotations must be reference term spans")
        indices = [
            i
            for i, (_, start, end) in enumerate(ref)
            if start < span.end and span.start < end
        ]
        if (
            span.end > len(reference)
            or not indices
            or ref[indices[0]][1] != span.start
            or ref[indices[-1]][2] != span.end
        ):
            raise ValueError("annotation must cover complete reference tokens")
        first, last = indices[0], indices[-1] + 1
        if any(first < end and start < last for start, end in seen[span.term_class]):
            raise ValueError("overlapping annotations within a term class")
        seen[span.term_class].append((first, last))
        # Interior insertions affect a term; boundary insertions are unassigned.
        error = any(
            first <= start < last if start != end else first < start < last
            for start, end in changes
        )
        counts[span.term_class][0] += 1
        counts[span.term_class][1] += int(error)
    revision_tokens = opportunities = final_tokens = final_opportunities = 0
    previous: list[str] | None = None
    for partial in partials:
        current = [t[0] for t in _tokens(partial, language)]
        if previous is not None:
            revision_tokens += _revision(previous, current)
            opportunities += len(previous)
        previous = current
    if previous is not None:
        final_tokens = _revision(previous, hyp_words)
        final_opportunities = len(previous)
        revision_tokens += final_tokens
        opportunities += final_opportunities
    return TranscriptScore(
        language,
        words,
        characters,
        tuple(TermCounts(c, *counts[c]) for c in ClinicalTermClass),
        revision_tokens,
        opportunities,
        final_tokens,
        final_opportunities,
    )


def _ratio(rows: Sequence[tuple[int, int]]) -> float | None:
    denominator = sum(r[1] for r in rows)
    return sum(r[0] for r in rows) / denominator if denominator else None


def _interval(
    rows: Sequence[tuple[int, int]], resamples: int, seed: int
) -> list[float] | None:
    # SHA-256 counter sampling avoids platform RNG or iteration-order behavior.
    estimates: list[float] = []
    for sample in range(resamples):
        selected = [
            rows[
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{sample}:{draw}".encode("ascii")).digest(),
                    "big",
                )
                % len(rows)
            ]
            for draw in range(len(rows))
        ]
        estimate = _ratio(selected)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    return [
        estimates[math.floor((len(estimates) - 1) * 0.025)],
        estimates[math.ceil((len(estimates) - 1) * 0.975)],
    ]


def _publish(
    rows: Sequence[tuple[int, int]],
    minimum: int,
    resamples: int,
    seed: int,
    *,
    binary: bool = False,
    cells: Sequence[int] = (),
) -> dict[str, Any]:
    numerator = sum(r[0] for r in rows)
    denominator = sum(r[1] for r in rows)
    contributors = sum(r[1] > 0 or r[0] > 0 for r in rows)
    sensitive_cells = list(cells) + [
        sum(n > 0 for n, _ in rows),
        sum(d > 0 and n == 0 for n, d in rows),
    ]
    if binary:
        sensitive_cells += [numerator, denominator - numerator]
    suppressed = (
        contributors < minimum
        or denominator < minimum
        or any(0 < c < minimum for c in sensitive_cells)
    )
    return {
        "suppressed": suppressed,
        "size": None if suppressed else denominator,
        "contributors": None if suppressed else contributors,
        "errors": None if suppressed else numerator,
        "rate": None if suppressed else _ratio(rows),
        "ci95": None if suppressed else _interval(rows, resamples, seed),
    }


def transcript_accuracy_report(
    scores: Iterable[TranscriptScore],
    *,
    minimum_cell_size: int = 5,
    bootstrap_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Publish micro-averaged counts with pair-cluster bootstrap intervals.

    Args:
        scores: Value-free scores from score_transcript; one independent pair
            per row. Input order does not affect the canonical resampling order.
        minimum_cell_size: Minimum denominator and contributing pairs, >= 2.
            Nonzero small error/correct cells are also suppressed.
        bootstrap_resamples: Number of seeded percentile resamples, >= 100.
        seed: Nonnegative integer for reproducible SHA-256 counter sampling.

    Returns:
        JSON-compatible aggregate evidence containing controlled codes and
        counts only, with entire small cells (including sizes/CIs) suppressed.
        A non-diagnostic notice and explicit reviewer-confirmation requirement
        accompany every report. Identifier recall is annotated-span ASR recall.

    Raises:
        ValueError: Invalid reporting policy or empty scores.
        TypeError: Non-score inputs.
    """
    if type(minimum_cell_size) is not int or minimum_cell_size < 2:
        raise ValueError("minimum cell size must be an integer >= 2")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be an integer >= 100")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    rows = tuple(scores)
    if not rows:
        raise ValueError("at least one transcript score is required")
    if any(not isinstance(s, TranscriptScore) for s in rows):
        raise TypeError("report inputs must be transcript scores")
    # Sort only numeric counts and controlled policy values, never source text.
    rows = tuple(
        sorted(
            rows,
            key=lambda s: (
                s.language,
                tuple(asdict(s.words).values()),
                tuple(asdict(s.characters).values()),
                tuple((t.term_class.value, t.size, t.errors) for t in s.terms),
                s.revision_tokens,
                s.revision_opportunities,
                s.final_revision_tokens,
                s.final_revision_opportunities,
            ),
        )
    )
    policy = (minimum_cell_size, bootstrap_resamples, seed)
    result: dict[str, Any] = {
        "schema_version": "openmed.transcript_accuracy.v1",
        "normalization_version": NORMALIZATION_VERSION,
        "notice": TRANSCRIPT_NOTICE,
        "reviewer_confirmation_required": True,
        "minimum_cell_size": minimum_cell_size,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": seed,
            "unit": "transcript_pair",
            "confidence": 0.95,
        },
        "pairs": len(rows) if len(rows) >= minimum_cell_size else None,
    }
    for key, field in (("wer", "words"), ("cer", "characters")):
        edits = [getattr(s, field) for s in rows]
        cells = [
            sum(getattr(e, attr) for e in edits)
            for attr in ("substitutions", "deletions", "insertions")
        ]
        cells.append(sum(e.reference - e.substitutions - e.deletions for e in edits))
        result[key] = _publish(
            [(e.errors, e.reference) for e in edits], *policy, cells=cells
        )
    result["clinical_terms"] = {}
    for term_class in ClinicalTermClass:
        term_rows = [
            next(t for t in s.terms if t.term_class == term_class) for s in rows
        ]
        published = _publish(
            [(t.errors, t.size) for t in term_rows], *policy, binary=True
        )
        if term_class == ClinicalTermClass.IDENTIFIER:
            published["recall"] = (
                None if published["suppressed"] else 1 - published["rate"]
            )
            published["recall_ci95"] = (
                None
                if published["ci95"] is None
                else [1 - published["ci95"][1], 1 - published["ci95"][0]]
            )
        result["clinical_terms"][term_class.value] = published
    for key, numerator, denominator in (
        ("revision_churn", "revision_tokens", "revision_opportunities"),
        (
            "final_revision_churn",
            "final_revision_tokens",
            "final_revision_opportunities",
        ),
    ):
        result[key] = _publish(
            [(getattr(s, numerator), getattr(s, denominator)) for s in rows],
            *policy,
            binary=True,
        )
    return result
