"""Cue-based experiencer refinement for clinical spans (roadmap v1.9).

The shipped ConText layer resolves the experiencer axis only at the section
level: a finding under a *Family History* heading is attributed to the family.
That is too coarse for free text, where a single sentence can switch subject
("the patient's *mother* has diabetes", "the organ *donor* was CMV-positive").

This module refines the experiencer of a governing clinical span using local
subject cues, distinguishing three subjects:

* ``patient``  -- the default when no other subject is cued.
* ``family``   -- a relative of the patient (mother, father, sibling, ...).
* ``other``    -- a non-patient, non-relative subject (donor, roommate, ...).

Resolution is scoped to the clause containing the span so a subject mentioned in
a previous sentence does not leak across a boundary. A section-level experiencer
prior is used as a fallback when no cue is found, and an explicit cue overrides
the section prior. The resolver is deterministic and offline, and reuses the
``PATIENT_EXPERIENCER`` / ``FAMILY_EXPERIENCER`` constants from
:mod:`openmed.clinical.context`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from .context import (
    AFFIRMED,
    CERTAIN,
    FAMILY_EXPERIENCER,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
    ClinicalContextResult,
)

#: Non-patient, non-relative subject (donor, roommate, ...).
OTHER_EXPERIENCER = "other"

Experiencer = Literal["patient", "family", "other"]

#: Experiencer values after refinement, from default to most specific.
EXPERIENCER_REFINED_VALUES = (
    PATIENT_EXPERIENCER,
    FAMILY_EXPERIENCER,
    OTHER_EXPERIENCER,
)

EXPERIENCER_REFINEMENT_ADVISORY = (
    "Experiencer refinement is a deterministic cue-based heuristic scoped to the "
    "span's clause. It sharpens subject attribution but is not a substitute for "
    "clinician review; verify before clinical use."
)

# Relatives whose mention attributes a nearby finding to the family.
_FAMILY_CUES = (
    "family history",
    "fhx",
    "familial",
    "maternal",
    "paternal",
    "grandmother",
    "grandfather",
    "grandparent",
    "mother",
    "father",
    "sister",
    "brother",
    "sibling",
    "parent",
    "parents",
    "daughter",
    "son",
    "child",
    "aunt",
    "uncle",
    "cousin",
    "niece",
    "nephew",
    "wife",
    "husband",
    "spouse",
    "mom",
    "dad",
)

# Non-patient, non-relative subjects.
_OTHER_CUES = (
    "donor",
    "roommate",
    "housemate",
    "partner",
    "girlfriend",
    "boyfriend",
    "coworker",
    "colleague",
    "neighbor",
    "neighbour",
    "friend",
    "contact",
)

# Clause boundaries that a subject cue may not reach across: sentence
# punctuation plus the ConText engine's contrastive scope terminators, which
# switch the sentence subject ("Mother had breast cancer, but the patient ...").
# The coordinating terminators "and"/"or" are deliberately excluded because they
# usually conjoin findings under the same subject ("mother had X and Y"), where
# the family experiencer must still reach the later finding.
#
# Only the unambiguous contrastive conjunctions are handled here. Broader
# subject-switching markers are intentionally left out for now: "although",
# "though", and "yet" are lower-value, and "while" is ambiguous (temporal
# "while on aspirin" vs contrastive "while the mother ..."), so splitting on it
# risks regressing temporal clauses. Widening this set is a deliberate
# follow-up, not an oversight.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?;]|(?<!\w)(?:but|however|whereas)(?!\w)",
    re.IGNORECASE,
)


def _cue_pattern(cues: tuple[str, ...]) -> re.Pattern[str]:
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(cues, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


_FAMILY_RE = _cue_pattern(_FAMILY_CUES)
_OTHER_RE = _cue_pattern(_OTHER_CUES)


@dataclass(frozen=True)
class ExperiencerAssignment:
    """The resolved experiencer for a span, with provenance.

    ``source`` is ``"cue"`` when a subject cue in the span's clause decided the
    result, ``"section"`` when the section prior was used, or ``"default"`` when
    the span fell back to the patient.
    """

    experiencer: str
    cue: str | None
    cue_offset: tuple[int, int] | None
    source: str


@dataclass(frozen=True)
class RefinedExperiencerAssertion:
    """A span assertion enriched with experiencer provenance."""

    span: Mapping[str, object]
    assertion: ClinicalAssertion
    assignment: ExperiencerAssignment


def _clause_start(text: str, span_start: int) -> int:
    """Offset just after the last clause boundary before ``span_start``."""

    last = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(text, 0, span_start):
        last = match.end()
    return last


def _coerce_span(span: Mapping[str, object]) -> tuple[int, int]:
    try:
        start = int(span["start"])  # type: ignore[arg-type]
        end = int(span["end"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("span requires integer 'start' and 'end'") from exc
    if start > end:
        raise ValueError("span 'start' must not exceed 'end'")
    return start, end


def _nearest_cue(
    text: str,
    clause_start: int,
    span_start: int,
) -> tuple[str, int, int] | None:
    """Return the (experiencer, cue_start, cue_end) of the cue nearest the span.

    Only cues within the span's clause and at or before the span are considered;
    on a tie the more specific ``other`` subject wins over ``family``.
    """

    best: tuple[int, int, str, int, int] | None = None
    for experiencer, pattern, rank in (
        (OTHER_EXPERIENCER, _OTHER_RE, 0),
        (FAMILY_EXPERIENCER, _FAMILY_RE, 1),
    ):
        for match in pattern.finditer(text, clause_start, span_start):
            gap = span_start - match.end()
            key = (gap, rank)
            if best is None or key < best[:2]:
                best = (gap, rank, experiencer, match.start(), match.end())
    if best is None:
        return None
    _gap, _rank, experiencer, cue_start, cue_end = best
    return experiencer, cue_start, cue_end


def resolve_experiencer(
    text: str,
    span: Mapping[str, object],
    *,
    section_experiencer: str | None = None,
) -> ExperiencerAssignment:
    """Resolve the experiencer of ``span`` from local subject cues.

    A subject cue within the span's clause decides the result; otherwise the
    ``section_experiencer`` prior is used, and failing that the span defaults to
    the patient. An explicit cue always overrides the section prior.
    """

    span_start, _span_end = _coerce_span(span)
    clause_start = _clause_start(text, span_start)

    hit = _nearest_cue(text, clause_start, span_start)
    if hit is not None:
        experiencer, cue_start, cue_end = hit
        return ExperiencerAssignment(
            experiencer=experiencer,
            cue=text[cue_start:cue_end].lower(),
            cue_offset=(cue_start, cue_end),
            source="cue",
        )

    if section_experiencer:
        return ExperiencerAssignment(
            experiencer=section_experiencer,
            cue=None,
            cue_offset=None,
            source="section",
        )

    return ExperiencerAssignment(
        experiencer=PATIENT_EXPERIENCER,
        cue=None,
        cue_offset=None,
        source="default",
    )


def refine_experiencer(
    spans: object,
    context_result: object | None = None,
    *,
    text: str | None = None,
    section_experiencer: str | None = None,
) -> list[RefinedExperiencerAssertion] | ExperiencerAssignment:
    """Return clinical assertions enriched with experiencer attribution.

    The issue-facing API accepts an iterable of span mappings plus either one
    shared context result/assertion or a per-span sequence of them, returning a
    ``RefinedExperiencerAssertion`` for each span. For compatibility with the
    single-span resolver introduced in the original PR,
    ``refine_experiencer(text, span)`` still returns an
    ``ExperiencerAssignment``; new code should prefer ``resolve_experiencer``
    for that lower-level operation.
    """

    if isinstance(spans, str):
        if not isinstance(context_result, Mapping):
            raise TypeError("single-span refinement requires a span mapping")
        return resolve_experiencer(
            spans,
            context_result,
            section_experiencer=section_experiencer,
        )

    span_list = list(_iter_span_mappings(spans))
    if not span_list:
        return []

    contexts = _context_sequence(context_result, len(span_list))
    refined: list[RefinedExperiencerAssertion] = []
    for span, context in zip(span_list, contexts, strict=True):
        base_assertion = _coerce_assertion(context)
        document_text = _document_text_for_span(span, text)
        assignment = resolve_experiencer(
            document_text,
            span,
            section_experiencer=section_experiencer or base_assertion.experiencer,
        )
        refined.append(
            RefinedExperiencerAssertion(
                span=span,
                assertion=replace(
                    base_assertion,
                    experiencer=assignment.experiencer,
                ),
                assignment=assignment,
            )
        )
    return refined


def _iter_span_mappings(spans: object) -> list[Mapping[str, object]]:
    if isinstance(spans, Mapping):
        return [spans]
    try:
        items = list(spans)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("spans must be a mapping or iterable of mappings") from exc
    if not all(isinstance(item, Mapping) for item in items):
        raise TypeError("all spans must be mappings")
    return items


def _context_sequence(context_result: object | None, count: int) -> list[object | None]:
    if isinstance(context_result, (ClinicalAssertion, ClinicalContextResult, Mapping)):
        return [context_result] * count
    if context_result is None:
        return [None] * count
    try:
        contexts = list(context_result)  # type: ignore[arg-type]
    except TypeError:
        return [context_result] * count
    if len(contexts) != count:
        raise ValueError("context_result sequence length must match spans")
    return contexts


def _coerce_assertion(context_result: object | None) -> ClinicalAssertion:
    if isinstance(context_result, ClinicalAssertion):
        return context_result
    if isinstance(context_result, ClinicalContextResult):
        return ClinicalAssertion(
            temporality=context_result.temporality,
            certainty=context_result.certainty,
            negation=context_result.negation,
        )
    if isinstance(context_result, Mapping):
        return ClinicalAssertion(
            temporality=str(context_result.get("temporality", RECENT)),
            certainty=str(context_result.get("certainty", CERTAIN)),  # type: ignore[arg-type]
            negation=context_result.get("negation", AFFIRMED),  # type: ignore[arg-type]
            experiencer=context_result.get("experiencer"),  # type: ignore[arg-type]
        )
    return ClinicalAssertion(
        temporality=RECENT,
        certainty=CERTAIN,
        negation=AFFIRMED,
    )


def _document_text_for_span(span: Mapping[str, object], text: str | None) -> str:
    if text is not None:
        return text
    for key in (
        "document_text",
        "context_text",
        "source_text",
        "full_text",
        "note_text",
    ):
        value = span.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("document text is required for experiencer refinement")


__all__ = [
    "OTHER_EXPERIENCER",
    "Experiencer",
    "EXPERIENCER_REFINED_VALUES",
    "EXPERIENCER_REFINEMENT_ADVISORY",
    "ExperiencerAssignment",
    "RefinedExperiencerAssertion",
    "resolve_experiencer",
    "refine_experiencer",
]
