"""Assertion-aware grounding (roadmap v2.0).

Grounding turns a surface mention into coded concepts, but the base linker path
ignores whether the mention was *asserted*. "No evidence of pneumonia" and
"family history of colon cancer" both match a real concept, yet neither is an
active patient finding: coding a refuted or non-patient mention as present
corrupts every downstream FHIR/OMOP record. This is a correctness and safety
failure, not a ranking nuisance.

This stage reads the ConText decision axes already produced by
:mod:`openmed.clinical.context` (negation, temporality, certainty) and the
experiencer refinement from :mod:`openmed.clinical.experiencer`, composes them
into a per-span :class:`~openmed.clinical.context.ClinicalAssertion`, and derives
a single code-level :class:`AssertionGroundingStatus` that maps onto FHIR
``verificationStatus`` / ``clinicalStatus``:

* ``present``      -- affirmed, current, certain patient finding (active + confirmed).
* ``refuted``      -- a negated finding (``verificationStatus=refuted``; hardest boundary).
* ``hypothetical`` -- a conditional finding, not asserted present (``provisional``).
* ``historical``   -- a resolved past finding (``clinicalStatus=inactive``).
* ``uncertain``    -- a hedged finding (``verificationStatus=provisional``).
* ``non_patient``  -- a family / other-experiencer finding, excluded from patient resources.

The governing rule is safety-critical: a mention whose assertion is denied,
hypothetical, or not-the-patient is **never** emitted as an active patient
concept. Every candidate of a span -- the focus concept and any composite or
post-coordinated qualifier codes -- inherits the same span-level status, so a
refuted focus can never leave a qualifier asserted.

A policy switch controls whether non-present concepts are dropped, suppressed
(kept with candidates stripped), or emitted carrying their status. The default
is status-carrying: a refuted or hypothetical concept is annotated, never
silently presented as present, and never silently discarded.

Provenance records only cue offsets and axis labels plus a content hash; the
raw note text is never logged, so the audit trail carries no PHI. Outputs are
advisory annotations for review and downstream processing, not clinical
decisions and not medical-device instructions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from openmed.core.audit import stable_hash

from ..context import (
    AFFIRMED,
    CERTAIN,
    HYPOTHETICAL,
    PATIENT_EXPERIENCER,
    RECENT,
    ClinicalAssertion,
    resolve_span_context,
    scan_context_cues,
)
from ..experiencer import resolve_experiencer
from .types import Candidate, GroundedSpan

__all__ = [
    "ASSERTION_GROUNDING_ADVISORY",
    "GROUNDING_PRESENT",
    "GROUNDING_REFUTED",
    "GROUNDING_HYPOTHETICAL",
    "GROUNDING_HISTORICAL",
    "GROUNDING_UNCERTAIN",
    "GROUNDING_NON_PATIENT",
    "GROUNDING_ASSERTION_STATUSES",
    "POLICY_STATUS",
    "POLICY_DROP",
    "POLICY_SUPPRESS",
    "GROUNDING_POLICIES",
    "AssertionGroundingStatus",
    "AssertedGroundedSpan",
    "assertion_grounding_status",
    "ground_with_context",
]

ASSERTION_GROUNDING_ADVISORY = (
    "Assertion-aware grounding status is derived deterministically from the "
    "ConText and experiencer axes of each span. Refuted, hypothetical, and "
    "non-patient concepts are never emitted as active patient codes; verify "
    "before clinical use."
)

#: Grounding assertion status labels, ordered from asserted to most withheld.
GROUNDING_PRESENT = "present"
GROUNDING_UNCERTAIN = "uncertain"
GROUNDING_HISTORICAL = "historical"
GROUNDING_HYPOTHETICAL = "hypothetical"
GROUNDING_REFUTED = "refuted"
GROUNDING_NON_PATIENT = "non_patient"

#: All grounding assertion status labels, ordered from asserted to most withheld.
GROUNDING_ASSERTION_STATUSES = (
    GROUNDING_PRESENT,
    GROUNDING_UNCERTAIN,
    GROUNDING_HISTORICAL,
    GROUNDING_HYPOTHETICAL,
    GROUNDING_REFUTED,
    GROUNDING_NON_PATIENT,
)

# FHIR Condition.verificationStatus codes (condition-ver-status value set).
_VERIFICATION_CONFIRMED = "confirmed"
_VERIFICATION_PROVISIONAL = "provisional"
_VERIFICATION_REFUTED = "refuted"

# FHIR Condition.clinicalStatus codes (condition-clinical value set).
_CLINICAL_ACTIVE = "active"
_CLINICAL_INACTIVE = "inactive"

# status label -> (verification_status, clinical_status, patient_subject).
# ``clinical_status`` is ``None`` when the finding is not asserted to hold now
# (refuted, hypothetical) or is not the patient's, so no active/inactive
# clinicalStatus is claimed for it.
_STATUS_MAP: dict[str, tuple[str, str | None, bool]] = {
    GROUNDING_PRESENT: (_VERIFICATION_CONFIRMED, _CLINICAL_ACTIVE, True),
    GROUNDING_UNCERTAIN: (_VERIFICATION_PROVISIONAL, _CLINICAL_ACTIVE, True),
    GROUNDING_HISTORICAL: (_VERIFICATION_CONFIRMED, _CLINICAL_INACTIVE, True),
    GROUNDING_HYPOTHETICAL: (_VERIFICATION_PROVISIONAL, None, True),
    GROUNDING_REFUTED: (_VERIFICATION_REFUTED, None, True),
    GROUNDING_NON_PATIENT: (_VERIFICATION_CONFIRMED, None, False),
}

#: Emit every span carrying its status (default; never silently present).
POLICY_STATUS = "status"
#: Drop non-present spans entirely from the returned sequence.
POLICY_DROP = "drop"
#: Keep non-present spans but strip their candidate codes.
POLICY_SUPPRESS = "suppress"

#: The supported non-present handling policies.
GROUNDING_POLICIES = (POLICY_STATUS, POLICY_DROP, POLICY_SUPPRESS)


@dataclass(frozen=True)
class AssertionGroundingStatus:
    """Code-level assertion status for a grounded span.

    ``status`` is one of :data:`GROUNDING_ASSERTION_STATUSES`.
    ``verification_status`` and ``clinical_status`` are the FHIR
    ``Condition.verificationStatus`` / ``Condition.clinicalStatus`` codes the
    status maps to (``clinical_status`` is ``None`` when no active/inactive
    status is asserted). ``patient_subject`` is ``False`` when the finding
    belongs to a family member or other non-patient subject.
    """

    status: str
    verification_status: str
    clinical_status: str | None
    patient_subject: bool

    @property
    def is_present(self) -> bool:
        """Whether this is an affirmed, current, certain patient finding."""

        return self.status == GROUNDING_PRESENT

    @property
    def is_active_patient_condition(self) -> bool:
        """Whether emitting this as an active patient ``Condition`` is allowed.

        This is the safety invariant of the module: it is ``True`` only for a
        patient-subject finding with an ``active`` ``clinicalStatus`` that is
        not refuted. A denied, hypothetical, or non-patient span can never
        satisfy it, so it can never be coded as an active patient concept.
        """

        return (
            self.patient_subject
            and self.clinical_status == _CLINICAL_ACTIVE
            and self.verification_status != _VERIFICATION_REFUTED
        )


def assertion_grounding_status(
    assertion: ClinicalAssertion,
) -> AssertionGroundingStatus:
    """Derive the code-level grounding status from composed ConText axes.

    The axes are collapsed to a single dominant status by a fixed safety
    precedence: a non-patient experiencer, then a negated polarity, then a
    hypothetical or historical temporality, then an uncertain certainty. The
    experiencer and negation boundaries dominate so a finding that is either
    not the patient's or refuted can never be resolved to ``present``.
    """

    status = _dominant_status(assertion)
    verification, clinical, patient = _STATUS_MAP[status]
    return AssertionGroundingStatus(
        status=status,
        verification_status=verification,
        clinical_status=clinical,
        patient_subject=patient,
    )


def _dominant_status(assertion: ClinicalAssertion) -> str:
    # Fail CLOSED: allowlist each axis to its asserted pole (mirroring the
    # experiencer check) so a non-canonical / unknown label from any producer
    # collapses to a withheld status instead of defaulting to "present". An
    # unset negation (``None``) is a legitimate "no negation composed" state and
    # stays present-compatible; any other non-affirmed label is withheld.
    experiencer = assertion.experiencer or PATIENT_EXPERIENCER
    if experiencer != PATIENT_EXPERIENCER:
        return GROUNDING_NON_PATIENT
    if assertion.negation not in (None, AFFIRMED):
        return GROUNDING_REFUTED
    if assertion.temporality == HYPOTHETICAL:
        return GROUNDING_HYPOTHETICAL
    if assertion.temporality != RECENT:
        return GROUNDING_HISTORICAL
    if assertion.certainty != CERTAIN:
        return GROUNDING_UNCERTAIN
    return GROUNDING_PRESENT


@dataclass(frozen=True)
class AssertedGroundedSpan:
    """A grounded span enriched with its assertion axes and code-level status.

    ``grounded`` is the original :class:`~openmed.clinical.exporters.codeable_concept.GroundedSpan`
    (its candidates are stripped when the span is suppressed by policy).
    ``assertion`` carries the four ConText/experiencer axes and ``status`` the
    derived code-level status shared by every candidate of the span.
    ``provenance`` records the governing cue offsets and axis labels only -- no
    raw note text -- so the audit trail carries no PHI.
    """

    grounded: GroundedSpan
    assertion: ClinicalAssertion
    status: AssertionGroundingStatus
    provenance: Mapping[str, Any]

    @property
    def is_active_patient_condition(self) -> bool:
        """Whether this span may be emitted as an active patient ``Condition``."""

        return self.status.is_active_patient_condition

    def iter_coded_concepts(
        self,
    ) -> Iterator[tuple[Candidate, AssertionGroundingStatus]]:
        """Yield ``(candidate, status)`` for every candidate of the span.

        The focus concept and any composite or post-coordinated qualifier codes
        inherit the identical span-level status, so a refuted or non-patient
        span can never leave one of its codes asserted as present.
        """

        for candidate in self.grounded.candidates:
            yield candidate, self.status


def ground_with_context(
    text: str,
    spans: Iterable[GroundedSpan],
    *,
    policy: str = POLICY_STATUS,
    language: str | None = None,
    section_experiencer: str | None = None,
) -> list[AssertedGroundedSpan]:
    """Attach assertion axes and a code-level status to each grounded span.

    Each span in ``spans`` is resolved against ``text`` for its ConText axes
    (negation, temporality, certainty) and experiencer, composed into a
    :class:`~openmed.clinical.context.ClinicalAssertion`, and tagged with an
    :class:`AssertionGroundingStatus`. The safety invariant holds regardless of
    policy: a denied, hypothetical, or non-patient span never returns a status
    for which :attr:`AssertionGroundingStatus.is_active_patient_condition` is
    true.

    Args:
        text: The source document text the span offsets index into.
        spans: Grounded spans exposing ``text``/``start``/``end``/``candidates``
            (typically :class:`~openmed.clinical.exporters.codeable_concept.GroundedSpan`).
        policy: One of :data:`GROUNDING_POLICIES`. ``"status"`` (default) emits
            every span carrying its status; ``"drop"`` omits non-present spans;
            ``"suppress"`` keeps non-present spans with their candidates stripped.
        language: Optional cue-lexicon language code; unknown codes fall back to
            English for compatibility.
        section_experiencer: Optional section-level experiencer prior applied
            when no in-clause subject cue is found.

    Returns:
        One :class:`AssertedGroundedSpan` per retained span, in input order.

    Raises:
        ValueError: If ``policy`` is not one of :data:`GROUNDING_POLICIES`.
    """

    if policy not in GROUNDING_POLICIES:
        raise ValueError(
            f"policy must be one of {GROUNDING_POLICIES!r}, got {policy!r}"
        )

    results: list[AssertedGroundedSpan] = []
    for grounded in spans:
        assertion = _assert_span(text, grounded, language, section_experiencer)
        status = assertion_grounding_status(assertion)
        provenance = _build_provenance(text, grounded, assertion, status, language)

        if not status.is_present:
            if policy == POLICY_DROP:
                continue
            if policy == POLICY_SUPPRESS:
                grounded = replace(grounded, candidates=())

        results.append(
            AssertedGroundedSpan(
                grounded=grounded,
                assertion=assertion,
                status=status,
                provenance=provenance,
            )
        )
    return results


def _span_mapping(text: str, grounded: GroundedSpan) -> dict[str, Any]:
    return {
        "text": grounded.text,
        "context": text,
        "start": grounded.start,
        "end": grounded.end,
    }


def _assert_span(
    text: str,
    grounded: GroundedSpan,
    language: str | None,
    section_experiencer: str | None,
) -> ClinicalAssertion:
    span = _span_mapping(text, grounded)
    context = resolve_span_context(span, language=language)
    experiencer = resolve_experiencer(
        text,
        {"start": grounded.start, "end": grounded.end},
        section_experiencer=section_experiencer,
    )
    return ClinicalAssertion(
        temporality=context.temporality,
        certainty=context.certainty,
        negation=context.negation,
        experiencer=experiencer.experiencer,
    )


def _build_provenance(
    text: str,
    grounded: GroundedSpan,
    assertion: ClinicalAssertion,
    status: AssertionGroundingStatus,
    language: str | None,
) -> dict[str, Any]:
    """Record axis labels and cue offsets only -- never raw note text (no PHI)."""

    span = {"start": grounded.start, "end": grounded.end}
    scoped = scan_context_cues(text, [span], language=language)
    cues: list[dict[str, Any]] = [
        {"category": hit.category, "start": hit.start, "end": hit.end}
        for hit in scoped.get(span, ())
    ]

    experiencer = resolve_experiencer(
        text,
        {"start": grounded.start, "end": grounded.end},
        section_experiencer=None,
    )
    if experiencer.cue_offset is not None:
        cues.append(
            {
                "category": "experiencer",
                "start": experiencer.cue_offset[0],
                "end": experiencer.cue_offset[1],
            }
        )
    cues.sort(key=lambda cue: (cue["start"], cue["end"], cue["category"]))

    payload: dict[str, Any] = {
        "span_offset": [grounded.start, grounded.end],
        "status": status.status,
        "verification_status": status.verification_status,
        "clinical_status": status.clinical_status,
        "patient_subject": status.patient_subject,
        "axes": assertion.to_dict(),
        "cues": cues,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload
