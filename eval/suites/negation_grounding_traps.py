"""Curated negation / experiencer grounding trap suite (release gate).

Every case here is a synthetic clinical sentence whose surface concept *matches*
a real code but whose assertion means it must never be emitted as an active
patient ``Condition``. The suite is the hard release gate behind OM-741: if any
denied, hypothetical, or non-patient (family / other experiencer) finding leaks
through :func:`~openmed.clinical.grounding.assertion_grounding.ground_with_context`
as an active patient condition, :func:`active_patient_condition_leaks` returns a
non-empty list and the gate fails.

The same gold set doubles as a status-mapping accuracy fixture: each case
carries its expected :mod:`~openmed.clinical.grounding.assertion_grounding`
status, and :func:`status_mapping_accuracy` scores the resolver against them.

All text is synthetic and algorithmically templated; it contains no real
patient data. Only cue offsets and axis labels ever reach provenance.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from openmed.clinical.exporters.codeable_concept import GroundedSpan
from openmed.clinical.exporters.fhir.condition import to_condition
from openmed.clinical.grounding import Candidate
from openmed.clinical.grounding.assertion_grounding import (
    GROUNDING_HISTORICAL,
    GROUNDING_HYPOTHETICAL,
    GROUNDING_NON_PATIENT,
    GROUNDING_PRESENT,
    GROUNDING_REFUTED,
    GROUNDING_UNCERTAIN,
    AssertedGroundedSpan,
    ground_with_context,
)

#: Axes whose findings must never surface as an active patient condition.
HARD_GATE_STATUSES = frozenset(
    {GROUNDING_REFUTED, GROUNDING_HYPOTHETICAL, GROUNDING_NON_PATIENT}
)

_SUBJECT_REFERENCE = "Patient/trap"


@dataclass(frozen=True)
class TrapCase:
    """One synthetic finding with its expected grounding status.

    ``text`` is the full synthetic sentence, ``surface`` the concept mention
    inside it (located by exact substring), ``system``/``code`` the coded
    concept the mention grounds to, ``axis`` the assertion axis exercised, and
    ``expected_status`` the grounding status the resolver must produce.
    """

    text: str
    surface: str
    system: str
    code: str
    axis: str
    expected_status: str

    @property
    def is_hard_gate(self) -> bool:
        """Whether a leak of this case as an active patient code fails release."""

        return self.expected_status in HARD_GATE_STATUSES

    def grounded_span(self) -> GroundedSpan:
        """Build the pre-grounded span (offsets located inside ``text``)."""

        start = self.text.index(self.surface)
        return GroundedSpan(
            text=self.surface,
            start=start,
            end=start + len(self.surface),
            candidates=(
                Candidate(
                    system=self.system,
                    code=self.code,
                    display=self.surface,
                    score=1.0,
                ),
            ),
        )


# Synthetic finding/code pairs reused across templates. Codes are illustrative
# ICD-10-CM / SNOMED identifiers, not drawn from any licensed distribution.
_FINDINGS = (
    ("pneumonia", "ICD10CM", "J18.9"),
    ("colon cancer", "ICD10CM", "C18.9"),
    ("deep vein thrombosis", "ICD10CM", "I82.90"),
    ("myocardial infarction", "ICD10CM", "I21.9"),
    ("diabetes mellitus", "ICD10CM", "E11.9"),
    ("pulmonary embolism", "ICD10CM", "I26.99"),
)

# axis -> (status, template with a ``{f}`` placeholder for the finding surface).
_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("negation", GROUNDING_REFUTED, "No evidence of {f} on exam."),
    ("negation", GROUNDING_REFUTED, "The patient denies {f}."),
    ("hypothetical", GROUNDING_HYPOTHETICAL, "Return if {f} develops."),
    ("hypothetical", GROUNDING_HYPOTHETICAL, "Start heparin if {f} is suspected."),
    ("experiencer", GROUNDING_NON_PATIENT, "Family history of {f}."),
    ("experiencer", GROUNDING_NON_PATIENT, "Her mother had {f}."),
    ("temporality", GROUNDING_HISTORICAL, "History of {f}, now resolved."),
    ("uncertainty", GROUNDING_UNCERTAIN, "Possible {f} pending imaging."),
    ("present", GROUNDING_PRESENT, "The patient has acute {f}."),
    ("present", GROUNDING_PRESENT, "Assessment: {f} confirmed on exam."),
)


def _build_cases() -> tuple[TrapCase, ...]:
    cases: list[TrapCase] = []
    for axis, status, template in _TEMPLATES:
        for surface, system, code in _FINDINGS:
            cases.append(
                TrapCase(
                    text=template.format(f=surface),
                    surface=surface,
                    system=system,
                    code=code,
                    axis=axis,
                    expected_status=status,
                )
            )
    return tuple(cases)


#: The frozen trap gold set.
TRAP_CASES: tuple[TrapCase, ...] = _build_cases()


def iter_trap_cases() -> Iterator[TrapCase]:
    """Yield every trap case in the frozen gold set."""

    yield from TRAP_CASES


def assert_case(case: TrapCase) -> AssertedGroundedSpan:
    """Ground a single trap case through the assertion-aware path."""

    return ground_with_context(case.text, [case.grounded_span()])[0]


def active_patient_condition_leaks() -> list[TrapCase]:
    """Return hard-gate cases that leak as active patient conditions.

    A case leaks if its grounded span reports itself as an active patient
    condition, or if the FHIR ``Condition`` exporter emits an ``active`` +
    ``confirmed`` patient resource for it. The list must be empty for release.
    """

    leaks: list[TrapCase] = []
    for case in TRAP_CASES:
        if not case.is_hard_gate:
            continue
        asserted = assert_case(case)
        if asserted.is_active_patient_condition:
            leaks.append(case)
            continue
        condition = to_condition(asserted, subject_reference=_SUBJECT_REFERENCE)
        if _is_active_confirmed_patient(condition):
            leaks.append(case)
    return leaks


def status_mapping_accuracy() -> float:
    """Fraction of trap cases whose resolved status matches the gold status."""

    if not TRAP_CASES:
        return 0.0
    correct = sum(
        1
        for case in TRAP_CASES
        if assert_case(case).status.status == case.expected_status
    )
    return correct / len(TRAP_CASES)


def _is_active_confirmed_patient(condition: dict | None) -> bool:
    if condition is None:
        return False
    clinical = _coding_code(condition.get("clinicalStatus"))
    verification = _coding_code(condition.get("verificationStatus"))
    return clinical == "active" and verification == "confirmed"


def _coding_code(concept: dict | None) -> str | None:
    if not concept:
        return None
    codings = concept.get("coding") or []
    if not codings:
        return None
    return codings[0].get("code")
