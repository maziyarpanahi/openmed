"""Offline synthetic evaluation for SNOMED CT post-coordination.

All identifiers and surfaces in this suite are invented test values. They do
not originate from, reproduce, or require a SNOMED CT edition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from openmed.clinical.grounding.ecl import ECLConstraint, ECLValidator
from openmed.clinical.grounding.postcoordination import (
    ConceptReference,
    Refinement,
    SnomedExpression,
    build_expression,
)

__all__ = [
    "SYNTHETIC_EXPRESSION_GOLD",
    "SyntheticExpressionCase",
    "evaluate_postcoordinated_expressions",
    "synthetic_ecl_validator",
]

_EDITION_URI = "https://synthetic.invalid/sct/edition/20260101"
_ATTRIBUTE_IDS = {
    "laterality": "700001",
    "severity": "700002",
    "morphology": "700003",
    "causative_agent": "700004",
}
_VALUE_ECL = {
    slot: f"<< {720000 + index}" for index, slot in enumerate(_ATTRIBUTE_IDS, start=1)
}
_FOCUS_ECL = "<< 710001"


@dataclass(frozen=True)
class SyntheticExpressionCase:
    """One invented mention with focus, refinements, and exact expression gold."""

    mention: str
    focus: ConceptReference
    refinements: tuple[Refinement, ...]
    expected_expression: str


class _SyntheticResolver:
    def __init__(self, membership: Mapping[str, frozenset[str]]) -> None:
        self._membership = dict(membership)

    def matches(self, concept_id: str, constraint: str, edition_uri: str) -> bool:
        return edition_uri == _EDITION_URI and concept_id in self._membership.get(
            constraint, frozenset()
        )


def _build_gold() -> tuple[SyntheticExpressionCase, ...]:
    cases: list[SyntheticExpressionCase] = []
    slots = tuple(_ATTRIBUTE_IDS)
    for index in range(30):
        focus = ConceptReference(str(810001 + index))
        selected_slots = (slots[index % len(slots)],)
        if index % 3 == 0:
            selected_slots += (slots[(index + 1) % len(slots)],)
        refinements = tuple(
            Refinement(
                slot=slot,
                attribute=ConceptReference(_ATTRIBUTE_IDS[slot]),
                value=ConceptReference(str(820001 + index * len(slots) + offset)),
            )
            for offset, slot in enumerate(selected_slots)
        )
        expected = _serialize_gold(focus, refinements)
        cases.append(
            SyntheticExpressionCase(
                mention=f"synthetic finding {index + 1:02d} with invented modifier",
                focus=focus,
                refinements=refinements,
                expected_expression=expected,
            )
        )
    return tuple(cases)


def synthetic_ecl_validator() -> ECLValidator:
    """Return an in-memory validator over invented concept memberships."""

    membership: dict[str, set[str]] = {
        _FOCUS_ECL: {case.focus.concept_id for case in SYNTHETIC_EXPRESSION_GOLD}
    }
    for slot, ecl in _VALUE_ECL.items():
        membership[ecl] = {
            refinement.value.concept_id
            for case in SYNTHETIC_EXPRESSION_GOLD
            for refinement in case.refinements
            if refinement.slot == slot
        }
    constraints = {
        slot: ECLConstraint(
            slot=slot,
            attribute_id=attribute_id,
            value_domain=_VALUE_ECL[slot],
            focus_domain=_FOCUS_ECL,
        )
        for slot, attribute_id in _ATTRIBUTE_IDS.items()
    }
    return ECLValidator(
        edition_uri=_EDITION_URI,
        constraints=constraints,
        resolver=_SyntheticResolver(
            {key: frozenset(values) for key, values in membership.items()}
        ),
    )


ExpressionPredictor = Callable[[SyntheticExpressionCase], SnomedExpression | str | None]


def evaluate_postcoordinated_expressions(
    predictor: ExpressionPredictor | None = None,
) -> dict[str, Any]:
    """Report exact match, ECL validity, and per-slot attribute F1."""

    validator = synthetic_ecl_validator()
    predict = predictor or (
        lambda case: build_expression(
            case.focus,
            case.refinements,
            validator=validator,
        )
    )
    exact = 0
    valid = 0
    counts = {slot: {"tp": 0, "fp": 0, "fn": 0} for slot in _ATTRIBUTE_IDS}
    for case in SYNTHETIC_EXPRESSION_GOLD:
        prediction = predict(case)
        predicted_text = (
            prediction.to_scg()
            if isinstance(prediction, SnomedExpression)
            else prediction
        )
        if predicted_text == case.expected_expression:
            exact += 1
        if (
            isinstance(prediction, SnomedExpression)
            and validator.validate(prediction).valid
        ):
            valid += 1
        gold_slots = _slot_records(case.refinements)
        predicted_slots = (
            _slot_records(prediction.refinements)
            if isinstance(prediction, SnomedExpression)
            else set()
        )
        for slot in _ATTRIBUTE_IDS:
            expected = {item for item in gold_slots if item[0] == slot}
            observed = {item for item in predicted_slots if item[0] == slot}
            counts[slot]["tp"] += len(expected & observed)
            counts[slot]["fp"] += len(observed - expected)
            counts[slot]["fn"] += len(expected - observed)
    total = len(SYNTHETIC_EXPRESSION_GOLD)
    return {
        "case_count": total,
        "expression_exact_match": exact / total if total else 0.0,
        "validation_rate": valid / total if total else 0.0,
        "attribute_slot_f1": {
            slot: _f1(**slot_counts) for slot, slot_counts in counts.items()
        },
        "metadata": {
            "offline": True,
            "synthetic": True,
            "ships_terminology_content": False,
        },
    }


def _serialize_gold(
    focus: ConceptReference,
    refinements: tuple[Refinement, ...],
) -> str:
    ordered = sorted(
        refinements,
        key=lambda item: (
            item.slot,
            item.attribute.concept_id,
            item.value.concept_id,
        ),
    )
    attributes = ", ".join(
        f"{item.attribute.concept_id} = {item.value.concept_id}" for item in ordered
    )
    return f"{focus.concept_id} : {attributes}"


SYNTHETIC_EXPRESSION_GOLD = _build_gold()


def _slot_records(refinements: tuple[Refinement, ...]) -> set[tuple[str, str, str]]:
    return {
        (item.slot, item.attribute.concept_id, item.value.concept_id)
        for item in refinements
    }


def _f1(*, tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0
