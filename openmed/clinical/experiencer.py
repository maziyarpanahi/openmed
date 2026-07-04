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
from dataclasses import dataclass

from .context import FAMILY_EXPERIENCER, PATIENT_EXPERIENCER

#: Non-patient, non-relative subject (donor, roommate, ...).
OTHER_EXPERIENCER = "other"

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

# Clause boundaries that a cue may not reach across.
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;]")


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


def refine_experiencer(
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


__all__ = [
    "OTHER_EXPERIENCER",
    "EXPERIENCER_REFINED_VALUES",
    "EXPERIENCER_REFINEMENT_ADVISORY",
    "ExperiencerAssignment",
    "refine_experiencer",
]
