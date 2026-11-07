"""Negation-scope boundary detection for clinical spans (roadmap v1.9).

The shipped ConText negation layer (:mod:`openmed.clinical.context`) decides
*whether* a negation cue is present, scoping it to the sentence. That is too
coarse: in ``"no chest pain but has fever"`` a sentence-level scope wrongly
negates ``fever``. This module computes the exact token region each negation
cue governs, terminating the scope at contrastive conjunctions, clause breaks,
and scope-terminating cue words, so an entity is negated only when it falls
inside a cue's computed influence span. Coordinated list members stay in scope
(``"denies fever and cough"``) until a conjunct opens a new clause
(``"and reports cough"``) or carries its own cue.

Forward cues (``no``, ``denies``, ``no evidence of`` ...) govern the text to
their right; backward cues (``ruled out``, ``absent`` ...) govern the text to
their left. Pseudo-negation phrases (``cannot be excluded``, ``not ruled out``)
are masked first so they never open a scope. The detector is deterministic and
offline, and reuses the authoritative cue lexicons from
:mod:`openmed.clinical.context` so the two layers cannot drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .context import (
    NEGATION_CUES,
    PSEUDO_NEGATION_CUES,
    _compiled_context_lexicon,
    _contains_clause_opener,
    _has_directional_cue,
    _has_scope_cue,
    _opens_clause,
)

NEGATION_SCOPE_ADVISORY = (
    "Negation scopes are computed deterministically from surface cues and "
    "clause boundaries. They sharpen assertion accuracy but are not a "
    "substitute for clinician review; verify before clinical use."
)

Direction = str  # "forward" | "backward"

FORWARD: Direction = "forward"
BACKWARD: Direction = "backward"

# Post-coordinated cues that govern the text to their LEFT rather than their
# right (kept in sync with the backward cues used by the ConText layer).
_BACKWARD_CUES: frozenset[str] = frozenset(
    {
        "absent",
        "not present",
        "ruled out",
        "resolved",
    }
)

# A negation scope always terminates at sentence punctuation, contrastive
# conjunctions, and classic NegEx clause-break terminators. Commas and the
# coordinators ``and`` / ``or`` are list separators instead: the scope continues
# across them ("no fever, chills, or cough") and ends only where the next
# conjunct opens a new clause or carries its own cue. The separator and clause
# rules come from the English ConText lexicon so both layers agree.
_TERMINATOR_RE = re.compile(
    r"(?:[.!?;]|(?<!\w)(?:"
    r"but|however|although|though|nevertheless|yet|still|"
    r"except|aside from|apart from|secondary to|which|who|because"
    r")(?!\w))",
    re.IGNORECASE,
)
_ENGLISH_RULES = _compiled_context_lexicon("en")
_SEPARATOR_RE = _ENGLISH_RULES.separator_re


def _cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(cues, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


_NEGATION_RE = _cue_pattern(NEGATION_CUES)
_PSEUDO_NEGATION_RE = _cue_pattern(PSEUDO_NEGATION_CUES)


@dataclass(frozen=True)
class NegationScope:
    """The region of text governed by one negation cue.

    ``scope_start`` / ``scope_end`` are half-open character offsets into the
    source text bounding the governed region (excluding the cue itself).
    ``governed`` holds the ``(start, end)`` offsets of the input entity spans
    that fall entirely inside that region.
    """

    cue: str
    cue_start: int
    cue_end: int
    direction: Direction
    scope_start: int
    scope_end: int
    governed: tuple[tuple[int, int], ...]


def _mask_pseudo_negation(text: str) -> str:
    """Blank out pseudo-negation phrases, preserving every character offset."""

    return _PSEUDO_NEGATION_RE.sub(lambda match: " " * len(match.group(0)), text)


def _normalize_cue(surface: str) -> str:
    return " ".join(surface.casefold().split())


def _blank_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Blank entity spans so their words never read as cues or clause openers."""

    chars = list(text)
    for start, end in spans:
        end = min(end, len(chars))
        start = min(max(start, 0), end)
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _separators(check_text: str, start: int, end: int) -> list[tuple[int, int]]:
    if _SEPARATOR_RE is None or start >= end:
        return []
    return [match.span() for match in _SEPARATOR_RE.finditer(check_text, start, end)]


def _first_entity_start(
    entities: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> int | None:
    return min((left for left, _ in entities if start <= left < end), default=None)


def _last_entity_end(
    entities: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> int | None:
    return max((right for left, right in entities if start <= left < end), default=None)


def _conjunct_starts_new_clause(
    check_text: str,
    start: int,
    end: int,
    entities: Sequence[tuple[int, int]],
    *,
    check_trail: bool,
) -> bool:
    """Whether a conjunct is a clause of its own rather than a list member."""

    if _has_scope_cue(check_text[start:end], _ENGLISH_RULES):
        return True
    entity_start = _first_entity_start(entities, start, end)
    if entity_start is None:
        return _opens_clause(check_text[start:end], _ENGLISH_RULES)
    if _contains_clause_opener(check_text[start:entity_start], _ENGLISH_RULES):
        return True
    if not check_trail:
        return False
    entity_end = _last_entity_end(entities, start, end) or end
    return _opens_clause(check_text[entity_end:end], _ENGLISH_RULES)


def _forward_bound(
    text: str,
    check_text: str,
    start: int,
    entities: Sequence[tuple[int, int]],
) -> int:
    """End of a forward scope: a hard terminator or a new-clause conjunct."""

    match = _TERMINATOR_RE.search(text, start)
    clause_end = match.start() if match else len(text)
    separators = _separators(check_text, start, clause_end)
    for index, (separator_start, separator_end) in enumerate(separators):
        conjunct_end = (
            separators[index + 1][0] if index + 1 < len(separators) else clause_end
        )
        if _conjunct_starts_new_clause(
            check_text,
            separator_end,
            conjunct_end,
            entities,
            check_trail=False,
        ):
            return separator_start
    return clause_end


def _backward_bound(
    text: str,
    check_text: str,
    end: int,
    entities: Sequence[tuple[int, int]],
) -> int:
    """Start of a backward scope, walking left across coordinated members."""

    last = 0
    for match in _TERMINATOR_RE.finditer(text, 0, end):
        last = match.end()
    separators = _separators(check_text, last, end)
    if not separators:
        return last

    scope_start = separators[-1][1]
    # The cue's own conjunct may carry a predicate ("cough are absent"); only a
    # subject or verb ahead of its first finding, or a forward cue, splits it.
    entity_start = _first_entity_start(entities, scope_start, end)
    if entity_start is None:
        cue_conjunct_opens = _opens_clause(check_text[scope_start:end], _ENGLISH_RULES)
    else:
        cue_conjunct_opens = _contains_clause_opener(
            check_text[scope_start:entity_start], _ENGLISH_RULES
        )
    if cue_conjunct_opens or _has_directional_cue(
        check_text[scope_start:end], _ENGLISH_RULES, backward=False
    ):
        return scope_start
    for index in range(len(separators) - 1, -1, -1):
        previous_start = separators[index - 1][1] if index else last
        if _conjunct_starts_new_clause(
            check_text,
            previous_start,
            separators[index][0],
            entities,
            check_trail=True,
        ):
            break
        scope_start = previous_start
    return scope_start


def _coerce_span(span: Mapping[str, object]) -> tuple[int, int]:
    try:
        start = int(span["start"])  # type: ignore[arg-type]
        end = int(span["end"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("entity spans require integer 'start' and 'end'") from exc
    if start > end:
        raise ValueError("entity span 'start' must not exceed 'end'")
    return start, end


def detect_negation_scopes(
    text: str,
    spans: Iterable[Mapping[str, object]] = (),
) -> list[NegationScope]:
    """Compute the governed scope of every negation cue in ``text``.

    ``spans`` are optional entity mentions (mappings with integer ``start`` /
    ``end``); each is attached to a scope's ``governed`` tuple when it lies
    fully inside that scope. Pseudo-negation phrases are masked first and never
    open a scope. Scopes are returned ordered by cue offset.
    """

    entities = [_coerce_span(span) for span in spans]
    masked = _mask_pseudo_negation(text)
    check_text = _blank_spans(masked, entities)

    scopes: list[NegationScope] = []
    for match in _NEGATION_RE.finditer(masked):
        cue_start, cue_end = match.start(), match.end()
        surface = text[cue_start:cue_end]
        direction = BACKWARD if _normalize_cue(surface) in _BACKWARD_CUES else FORWARD
        if direction == FORWARD:
            scope_start = cue_end
            scope_end = _forward_bound(text, check_text, cue_end, entities)
        else:
            scope_start = _backward_bound(text, check_text, cue_start, entities)
            scope_end = cue_start

        governed = tuple(
            (start, end)
            for start, end in entities
            if start >= scope_start and end <= scope_end
        )
        scopes.append(
            NegationScope(
                cue=surface,
                cue_start=cue_start,
                cue_end=cue_end,
                direction=direction,
                scope_start=scope_start,
                scope_end=scope_end,
                governed=governed,
            )
        )
    return scopes


def negated_spans(
    text: str,
    spans: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, int], ...]:
    """Return the ``(start, end)`` offsets of spans that fall inside a scope.

    A span is negated only when it lies within some negation cue's computed
    scope, not merely the same sentence. Offsets are returned in first-seen
    order without duplicates.
    """

    scopes = detect_negation_scopes(text, spans)
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for scope in scopes:
        for offset in scope.governed:
            if offset not in seen:
                seen.add(offset)
                ordered.append(offset)
    return tuple(ordered)


__all__ = [
    "NEGATION_SCOPE_ADVISORY",
    "FORWARD",
    "BACKWARD",
    "Direction",
    "NegationScope",
    "detect_negation_scopes",
    "negated_spans",
]
