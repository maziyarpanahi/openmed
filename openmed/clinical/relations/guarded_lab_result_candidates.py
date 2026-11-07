"""Ambiguity-preserving guarded laboratory-result relation candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from openmed.clinical.units import parse_unit

from ._guarded import (
    GuardedEvidenceSpan,
    GuardedSpanInput,
    coerce_guarded_spans,
    nested_value,
    same_sentence,
    span_gap,
    stable_candidate_id,
)

LAB_RESULT_CANDIDATE_ADVISORY = (
    "Laboratory-result links are unconfirmed evidence candidates for human "
    "review and are not diagnostic interpretations."
)

LabUnitStatus = Literal["compatible", "unknown", "missing"]
LabConflictState = Literal["none", "competing"]

_ANALYTE_LABELS = frozenset({"ANALYTE", "LAB", "LAB_NAME", "LAB_TEST", "TEST"})
_VALUE_LABELS = frozenset({"LAB_VALUE", "RESULT", "VALUE"})
_UNIT_LABELS = frozenset({"LAB_UNIT", "UNIT"})
_RANGE_LABELS = frozenset(
    {"REFERENCE_INTERVAL", "REFERENCE_RANGE", "REF_RANGE", "LAB_RANGE"}
)
_SPECIMEN_LABELS = frozenset({"SPECIMEN", "SPECIMEN_TYPE", "SAMPLE"})
_TIME_LABELS = frozenset({"DATE", "OBSERVATION_TIME", "TIME", "TIME_ANCHOR", "TIMEX"})


@dataclass(frozen=True)
class LabResultRelationCandidate:
    """One reviewable analyte/value linkage with optional supporting roles."""

    candidate_id: str
    analyte: GuardedEvidenceSpan
    value: GuardedEvidenceSpan
    unit: GuardedEvidenceSpan | None
    reference_interval: GuardedEvidenceSpan | None
    specimen: GuardedEvidenceSpan | None
    observation_time: GuardedEvidenceSpan | None
    canonical_unit: str | None
    unit_dimension: tuple[tuple[str, int], ...]
    unit_status: LabUnitStatus
    evidence_complete: bool
    conflict_state: LabConflictState = "none"
    competing_candidate_ids: tuple[str, ...] = ()
    review_required: bool = True
    advisory: str = LAB_RESULT_CANDIDATE_ADVISORY

    def __post_init__(self) -> None:
        if not self.review_required:
            raise ValueError("laboratory candidates must require review")
        if self.conflict_state == "none" and self.competing_candidate_ids:
            raise ValueError("non-conflicting candidates cannot name competitors")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic offset-and-hash evidence without raw values."""

        return {
            "candidate_id": self.candidate_id,
            "relation_type": "laboratory_result",
            "analyte": self.analyte.to_dict(),
            "value": self.value.to_dict(),
            "unit": self.unit.to_dict() if self.unit else None,
            "reference_interval": (
                self.reference_interval.to_dict() if self.reference_interval else None
            ),
            "specimen": self.specimen.to_dict() if self.specimen else None,
            "observation_time": (
                self.observation_time.to_dict() if self.observation_time else None
            ),
            "canonical_unit": self.canonical_unit,
            "unit_dimension": dict(self.unit_dimension),
            "unit_status": self.unit_status,
            "evidence_complete": self.evidence_complete,
            "conflict_state": self.conflict_state,
            "competing_candidate_ids": list(self.competing_candidate_ids),
            "review_required": self.review_required,
            "advisory": self.advisory,
        }


def generate_lab_result_candidates(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any]] | None = None,
    *,
    max_distance: int = 96,
) -> tuple[LabResultRelationCandidate, ...]:
    """Generate bounded laboratory-result candidates without choosing ties.

    Every analyte/value pair in the same section and sentence is retained when
    it is within ``max_distance``. Parsed units must be dimension-compatible
    with caller-supplied ``expected_unit``/``expected_units`` metadata and any
    reference-interval unit metadata. Incompatible pairs are rejected; unknown
    or absent units remain explicit incomplete candidates for review.
    """

    if type(max_distance) is not int or max_distance < 0:
        raise ValueError("max_distance must be a non-negative integer")
    inputs, _ = coerce_guarded_spans(text, spans, sections)
    analytes = _with_labels(inputs, _ANALYTE_LABELS)
    values = _with_labels(inputs, _VALUE_LABELS)
    units = _with_labels(inputs, _UNIT_LABELS)
    ranges = _with_labels(inputs, _RANGE_LABELS)
    specimens = _with_labels(inputs, _SPECIMEN_LABELS)
    times = _with_labels(inputs, _TIME_LABELS)

    candidates: list[LabResultRelationCandidate] = []
    for analyte in analytes:
        for value in values:
            if not _eligible_pair(text, analyte, value, max_distance=max_distance):
                continue
            nearby_units = _nearest_options(
                text,
                value,
                units,
                max_distance=min(max_distance, 24),
            ) or (None,)
            reference = _nearest_option(
                text,
                value,
                ranges,
                max_distance=min(max_distance, 64),
            )
            specimen = _nearest_option(
                text,
                analyte,
                specimens,
                max_distance=max_distance,
            )
            observation_time = _nearest_option(
                text,
                value,
                times,
                max_distance=max_distance,
            )
            for unit in nearby_units:
                unit_info = _unit_info(text, unit)
                if not _unit_is_compatible(
                    unit_info,
                    analyte,
                    reference,
                    text=text,
                ):
                    continue
                unit_status, canonical_unit, dimension = unit_info
                candidate_id = stable_candidate_id(
                    "lab-result",
                    analyte.evidence.offsets,
                    value.evidence.offsets,
                    unit.evidence.offsets if unit else None,
                    reference.evidence.offsets if reference else None,
                    specimen.evidence.offsets if specimen else None,
                    observation_time.evidence.offsets if observation_time else None,
                )
                candidates.append(
                    LabResultRelationCandidate(
                        candidate_id=candidate_id,
                        analyte=analyte.evidence,
                        value=value.evidence,
                        unit=unit.evidence if unit else None,
                        reference_interval=(reference.evidence if reference else None),
                        specimen=specimen.evidence if specimen else None,
                        observation_time=(
                            observation_time.evidence if observation_time else None
                        ),
                        canonical_unit=canonical_unit,
                        unit_dimension=dimension,
                        unit_status=unit_status,
                        evidence_complete=all(
                            item is not None
                            for item in (unit, reference, specimen, observation_time)
                        )
                        and unit_status == "compatible",
                    )
                )
    return _mark_competing_candidates(candidates)


def _with_labels(
    inputs: Sequence[GuardedSpanInput], labels: frozenset[str]
) -> tuple[GuardedSpanInput, ...]:
    return tuple(item for item in inputs if item.evidence.label.upper() in labels)


def _eligible_pair(
    text: str,
    left: GuardedSpanInput,
    right: GuardedSpanInput,
    *,
    max_distance: int,
) -> bool:
    return (
        left.evidence.section == right.evidence.section
        and same_sentence(text, left.evidence, right.evidence)
        and span_gap(left.evidence, right.evidence) <= max_distance
    )


def _nearest_options(
    text: str,
    anchor: GuardedSpanInput,
    options: Sequence[GuardedSpanInput],
    *,
    max_distance: int,
) -> tuple[GuardedSpanInput, ...]:
    eligible = tuple(
        option
        for option in options
        if _eligible_pair(text, anchor, option, max_distance=max_distance)
    )
    if not eligible:
        return ()
    nearest_distance = min(
        span_gap(anchor.evidence, item.evidence) for item in eligible
    )
    return tuple(
        item
        for item in eligible
        if span_gap(anchor.evidence, item.evidence) == nearest_distance
    )


def _nearest_option(
    text: str,
    anchor: GuardedSpanInput,
    options: Sequence[GuardedSpanInput],
    *,
    max_distance: int,
) -> GuardedSpanInput | None:
    nearest = _nearest_options(text, anchor, options, max_distance=max_distance)
    return nearest[0] if nearest else None


def _unit_info(
    text: str,
    unit: GuardedSpanInput | None,
) -> tuple[LabUnitStatus, str | None, tuple[tuple[str, int], ...]]:
    if unit is None:
        return "missing", None, ()
    parsed = parse_unit(text[unit.evidence.start : unit.evidence.end])
    if parsed["status"] != "ok":
        return "unknown", None, ()
    return (
        "compatible",
        str(parsed["canonical_unit"]),
        tuple(
            sorted((str(key), int(value)) for key, value in parsed["dimension"].items())
        ),
    )


def _unit_is_compatible(
    unit_info: tuple[LabUnitStatus, str | None, tuple[tuple[str, int], ...]],
    analyte: GuardedSpanInput,
    reference: GuardedSpanInput | None,
    *,
    text: str,
) -> bool:
    status, _, dimension = unit_info
    if status != "compatible":
        return True
    expected = nested_value(
        analyte.data, "expected_units", "allowed_units", "expected_unit"
    )
    expected_units = _unit_values(expected)
    reference_unit = (
        nested_value(reference.data, "unit", "reference_unit") if reference else None
    )
    if reference_unit is not None:
        expected_units = (*expected_units, str(reference_unit))
    for expected_unit in expected_units:
        parsed = parse_unit(expected_unit)
        if (
            parsed["status"] == "ok"
            and tuple(sorted(parsed["dimension"].items())) != dimension
        ):
            return False
    return True


def _unit_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _mark_competing_candidates(
    candidates: Sequence[LabResultRelationCandidate],
) -> tuple[LabResultRelationCandidate, ...]:
    by_value: dict[tuple[int, int], list[LabResultRelationCandidate]] = {}
    by_analyte: dict[tuple[int, int], list[LabResultRelationCandidate]] = {}
    for candidate in candidates:
        by_value.setdefault(candidate.value.offsets, []).append(candidate)
        by_analyte.setdefault(candidate.analyte.offsets, []).append(candidate)
    result: list[LabResultRelationCandidate] = []
    for candidate in candidates:
        related = {
            item.candidate_id
            for item in (
                *by_value[candidate.value.offsets],
                *by_analyte[candidate.analyte.offsets],
            )
            if item.candidate_id != candidate.candidate_id
        }
        result.append(
            replace(
                candidate,
                conflict_state="competing" if related else "none",
                competing_candidate_ids=tuple(sorted(related)),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.analyte.start,
                item.value.start,
                item.unit.start if item.unit else -1,
                item.candidate_id,
            ),
        )
    )


__all__ = [
    "LAB_RESULT_CANDIDATE_ADVISORY",
    "LabConflictState",
    "LabResultRelationCandidate",
    "LabUnitStatus",
    "generate_lab_result_candidates",
]
