"""Dimension-checked clinical measurement conversion helpers."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from .ucum import (
    MEASUREMENT_NORMALIZATION_ADVISORY,
    Dimension,
    UnitProvenance,
    _parse_unit_expression,
    canonical_unit_for_dimension,
    finite_float,
)

ConversionStatus = Literal[
    "ok",
    "unknown",
    "ambiguous",
    "incommensurable",
    "analyte_context_required",
]

ROUND_TRIP_REL_TOLERANCE = 1e-12
ROUND_TRIP_ABS_TOLERANCE = 1e-12


class MeasurementNormalization(TypedDict, total=False):
    """Public result for measurement parsing and conversion."""

    status: ConversionStatus
    magnitude: float | None
    canonical_magnitude: float | None
    unit: str | None
    canonical_unit: str | None
    dimension: dict[str, int]
    reason: str
    advisory: str
    provenance: dict[str, object]


_NUMERIC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_MEASUREMENT_RE = re.compile(
    rf"^\s*(?P<value>{_NUMERIC})\s*(?P<unit>.+?)\s*$",
    re.IGNORECASE,
)


def _base_result(
    status: ConversionStatus,
    reason: str,
    *,
    provenance: dict[str, object] | None = None,
) -> MeasurementNormalization:
    return {
        "status": status,
        "magnitude": None,
        "canonical_magnitude": None,
        "unit": None,
        "canonical_unit": None,
        "dimension": {},
        "reason": reason,
        "advisory": MEASUREMENT_NORMALIZATION_ADVISORY,
        "provenance": provenance or {},
    }


def _coerce_measurement_parts(
    value: object,
    unit: object | None,
) -> tuple[float | None, object | None, str]:
    if unit is not None:
        return finite_float(value), unit, "value_unit_pair"
    if isinstance(value, str) and (match := _MEASUREMENT_RE.fullmatch(value)):
        return (
            finite_float(match.group("value")),
            match.group("unit").strip(),
            "measurement_string",
        )
    return finite_float(value), unit, "value_without_unit"


def _status_from_unit_status(status: str) -> ConversionStatus:
    return "ambiguous" if status == "ambiguous" else "unknown"


def _dimension_mapping(dimension: Dimension) -> dict[str, int]:
    return dict(dimension)


def _requires_analyte_context(source: Dimension, target: Dimension) -> bool:
    source_map = dict(source)
    target_map = dict(target)
    mass_source = source_map.pop("mass", 0)
    amount_source = source_map.pop("amount", 0)
    eq_source = source_map.pop("equivalent_amount", 0)
    mass_target = target_map.pop("mass", 0)
    amount_target = target_map.pop("amount", 0)
    eq_target = target_map.pop("equivalent_amount", 0)

    same_residual_dimension = source_map == target_map
    mass_amount_pair = (mass_source and amount_target) or (
        amount_source and mass_target
    )
    equivalent_amount_pair = (eq_source and amount_target) or (
        amount_source and eq_target
    )
    mass_equivalent_pair = (mass_source and eq_target) or (eq_source and mass_target)
    return same_residual_dimension and bool(
        mass_amount_pair or equivalent_amount_pair or mass_equivalent_pair
    )


def parse_measurement(
    value: object,
    unit: object | None = None,
) -> MeasurementNormalization:
    """Parse and normalize a value/unit pair into canonical units.

    Args:
        value: Numeric magnitude or a string such as ``"5 mg/dL"``.
        unit: Optional unit string. When omitted, ``value`` must contain both
            the number and unit.

    Returns:
        A structured result containing canonical magnitude, dimension,
        provenance, advisory text, and an explicit status.
    """

    numeric_value, parsed_unit, input_form = _coerce_measurement_parts(value, unit)
    if numeric_value is None:
        return _base_result(
            "unknown",
            "measurement value is not finite numeric",
            provenance={"input_form": input_form},
        )
    if parsed_unit is None:
        return _base_result(
            "unknown",
            "measurement unit is required",
            provenance={"input_form": input_form},
        )

    unit_parse = _parse_unit_expression(parsed_unit)
    if unit_parse.expression is None:
        return _base_result(
            _status_from_unit_status(unit_parse.status),
            unit_parse.reason,
            provenance={
                "input_value": value,
                "input_unit": parsed_unit,
                "unit": unit_parse.provenance,
                "input_form": input_form,
            },
        )

    expression = unit_parse.expression
    canonical_magnitude = expression.to_canonical(numeric_value)
    canonical_unit = canonical_unit_for_dimension(expression.dimension)
    return {
        "status": "ok",
        "magnitude": canonical_magnitude,
        "canonical_magnitude": canonical_magnitude,
        "unit": canonical_unit,
        "canonical_unit": canonical_unit,
        "dimension": _dimension_mapping(expression.dimension),
        "reason": "",
        "advisory": MEASUREMENT_NORMALIZATION_ADVISORY,
        "provenance": {
            "input_value": value,
            "input_unit": parsed_unit,
            "input_form": input_form,
            "unit": expression.provenance,
        },
    }


def normalize_to(
    value: object,
    from_unit: object,
    to_unit: object,
) -> MeasurementNormalization:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` if dimensions match.

    Incommensurable units and analyte-dependent mass-to-amount conversions
    return explicit refusal statuses and never fabricate a numeric result.
    """

    numeric_value = finite_float(value)
    if numeric_value is None:
        return _base_result(
            "unknown",
            "measurement value is not finite numeric",
            provenance={
                "input_value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
            },
        )

    source = _parse_unit_expression(from_unit)
    target = _parse_unit_expression(to_unit)
    provenance: dict[str, object] = {
        "input_value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "source_unit": source.provenance,
        "target_unit": target.provenance,
        "round_trip_relative_tolerance": ROUND_TRIP_REL_TOLERANCE,
        "round_trip_absolute_tolerance": ROUND_TRIP_ABS_TOLERANCE,
    }

    if source.expression is None:
        return _base_result(
            _status_from_unit_status(source.status),
            source.reason,
            provenance=provenance,
        )
    if target.expression is None:
        return _base_result(
            _status_from_unit_status(target.status),
            target.reason,
            provenance=provenance,
        )

    source_expression = source.expression
    target_expression = target.expression
    if source_expression.dimension != target_expression.dimension:
        status: ConversionStatus = "incommensurable"
        reason = "units are not dimensionally commensurable"
        if _requires_analyte_context(
            source_expression.dimension,
            target_expression.dimension,
        ):
            status = "analyte_context_required"
            reason = (
                "mass, substance-amount, or equivalent conversions require "
                "an analyte-specific molar mass or charge context"
            )
        return _base_result(status, reason, provenance=provenance)

    canonical_magnitude = source_expression.to_canonical(numeric_value)
    converted = target_expression.from_canonical(canonical_magnitude)
    return {
        "status": "ok",
        "magnitude": converted,
        "canonical_magnitude": canonical_magnitude,
        "unit": target_expression.normalized_unit,
        "canonical_unit": canonical_unit_for_dimension(source_expression.dimension),
        "dimension": _dimension_mapping(source_expression.dimension),
        "reason": "",
        "advisory": MEASUREMENT_NORMALIZATION_ADVISORY,
        "provenance": {
            **provenance,
            "conversion": "dimension_checked_ucum_subset",
        },
    }


def unit_provenance(result: MeasurementNormalization) -> UnitProvenance | None:
    """Return unit provenance from a measurement result when present."""

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        return None
    unit = provenance.get("unit")
    if isinstance(unit, dict):
        return unit
    return None


__all__ = [
    "ConversionStatus",
    "MeasurementNormalization",
    "ROUND_TRIP_ABS_TOLERANCE",
    "ROUND_TRIP_REL_TOLERANCE",
    "normalize_to",
    "parse_measurement",
    "unit_provenance",
]
