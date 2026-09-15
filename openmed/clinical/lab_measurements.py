"""Deterministic, privacy-safe normalization of laboratory measurements.

The normalizer accepts already extracted synthetic lab rows and combines the
existing UCUM-subset unit parser with reference-range comparison.  It keeps
the typed measurement and source offsets useful for downstream review while
omitting source text and raw input strings from provenance.  Unknown or
ambiguous units are reported explicitly; they are never guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

from .lab_values import AbnormalFlag, derive_abnormal_flag, parse_reference_range
from .lexicons.clinical_norm import parse_locale_number, split_measurement_text
from .units import parse_measurement, parse_unit

LabMeasurementStatus = Literal[
    "ok",
    "unknown_unit",
    "invalid_value",
    "invalid_range",
]
LabUnitStatus = Literal["known", "missing", "unknown", "ambiguous"]
LabRangeStatus = Literal[
    "missing",
    "ok",
    "invalid",
    "unknown_unit",
    "incommensurable",
]

LAB_MEASUREMENT_ADVISORY = (
    "Lab measurement normalization is deterministic, offline assistance for "
    "review. Unknown units are not guessed, and interpretations reflect only "
    "the supplied reference range or originating laboratory flag."
)


class SourceOffsets(TypedDict):
    """A half-open character span into the caller's source document."""

    start: int
    end: int


class LabReferenceRange(TypedDict):
    """A typed reference range with optional canonical bounds."""

    low: float | None
    high: float | None
    low_inclusive: bool
    high_inclusive: bool
    unit: str | None
    canonical_low: float | None
    canonical_high: float | None
    canonical_unit: str | None
    status: LabRangeStatus
    unit_status: LabUnitStatus


class LabMeasurement(TypedDict):
    """One normalized laboratory measurement and privacy-safe provenance."""

    analyte: str | None
    value: float | None
    unit: str | None
    canonical_value: float | None
    canonical_unit: str | None
    dimension: dict[str, int]
    reference_range: LabReferenceRange
    interpretation: AbnormalFlag
    qualifiers: list[str]
    source_offsets: SourceOffsets | None
    status: LabMeasurementStatus
    unit_status: LabUnitStatus
    reason: str
    advisory: str
    provenance: dict[str, object]


class _UnitDetails(TypedDict):
    status: LabUnitStatus
    unit: str | None
    canonical_unit: str | None
    dimension: dict[str, int]


def _empty_reference_range(
    *,
    status: LabRangeStatus = "missing",
    unit_status: LabUnitStatus = "missing",
) -> LabReferenceRange:
    return {
        "low": None,
        "high": None,
        "low_inclusive": True,
        "high_inclusive": True,
        "unit": None,
        "canonical_low": None,
        "canonical_high": None,
        "canonical_unit": None,
        "status": status,
        "unit_status": unit_status,
    }


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _clean_unit_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def _unit_details(value: object, *, language: object | None = None) -> _UnitDetails:
    if value is None:
        return {
            "status": "missing",
            "unit": None,
            "canonical_unit": None,
            "dimension": {},
        }
    if not isinstance(value, str):
        return {
            "status": "unknown",
            "unit": None,
            "canonical_unit": None,
            "dimension": {},
        }

    cleaned = _clean_unit_text(value)
    if cleaned is None:
        return {
            "status": "missing",
            "unit": None,
            "canonical_unit": None,
            "dimension": {},
        }

    parsed = parse_unit(cleaned, language=language)
    parsed_status = parsed.get("status")
    if parsed_status == "ok":
        return {
            "status": "known",
            "unit": parsed.get("unit") or cleaned,
            "canonical_unit": parsed.get("canonical_unit"),
            "dimension": dict(parsed.get("dimension") or {}),
        }

    status: LabUnitStatus = "ambiguous" if parsed_status == "ambiguous" else "unknown"
    return {
        "status": status,
        "unit": cleaned,
        "canonical_unit": None,
        "dimension": {},
    }


def _validate_offsets(value: object) -> SourceOffsets | None:
    if value is None:
        return None

    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        start, end = value
    else:
        raise TypeError("source offsets must be a two-item integer sequence")

    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError("source offsets must be a two-item integer sequence")
    if start < 0 or end < start:
        raise ValueError("source offsets must satisfy 0 <= start <= end")
    return {"start": start, "end": end}


def _mapping_value(
    source: Mapping[str, object],
    names: Sequence[str],
    *,
    default: object = None,
) -> object:
    for name in names:
        if name in source:
            return source[name]
    return default


def _mapping_offsets(source: Mapping[str, object]) -> object:
    for name in ("source_offsets", "offset", "span"):
        if name in source:
            return source[name]
    if "start" in source or "end" in source:
        return {"start": source.get("start"), "end": source.get("end")}
    return None


def _normalize_qualifiers(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = value
    else:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = _clean_optional_text(item)
        if cleaned is None:
            continue
        key = cleaned.casefold()
        if key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _source_value_parts(
    value: object,
    unit: object,
) -> tuple[object, object, str]:
    if isinstance(value, str) and (parts := split_measurement_text(value)) is not None:
        number_text, embedded_unit = parts
        if unit is None:
            return number_text, embedded_unit, "measurement_string"
        return number_text, unit, "value_unit_pair"
    if unit is None:
        return value, None, "value_without_unit"
    return value, unit, "value_unit_pair"


def _finite_numeric(value: object, *, language: object | None = None) -> float | None:
    return parse_locale_number(value, language=language)


def _range_parts(
    source: object,
    *,
    language: object | None = None,
) -> tuple[LabReferenceRange, object, str]:
    if source is None:
        return _empty_reference_range(), None, "missing"

    if isinstance(source, str):
        parsed = parse_reference_range(source, language=language)
        low = parsed.get("low")
        high = parsed.get("high")
        if low is None and high is None:
            status: LabRangeStatus = "invalid" if source.strip() else "missing"
            return _empty_reference_range(status=status), parsed.get("unit"), "text"
        return (
            {
                "low": low,
                "high": high,
                "low_inclusive": bool(parsed.get("low_inclusive", True)),
                "high_inclusive": bool(parsed.get("high_inclusive", True)),
                "unit": None,
                "canonical_low": None,
                "canonical_high": None,
                "canonical_unit": None,
                "status": "ok",
                "unit_status": "missing",
            },
            parsed.get("unit"),
            "text",
        )

    if not isinstance(source, Mapping):
        return _empty_reference_range(status="invalid"), None, "mapping"

    raw_low = source.get("low")
    raw_high = source.get("high")
    low = _finite_numeric(raw_low, language=language)
    high = _finite_numeric(raw_high, language=language)
    if (raw_low is not None and low is None) or (raw_high is not None and high is None):
        return _empty_reference_range(status="invalid"), None, "mapping"
    if low is not None and high is not None and low > high:
        return _empty_reference_range(status="invalid"), None, "mapping"
    if low is None and high is None:
        return _empty_reference_range(status="invalid"), None, "mapping"

    return (
        {
            "low": low,
            "high": high,
            "low_inclusive": (
                source.get("low_inclusive")
                if isinstance(source.get("low_inclusive"), bool)
                else True
            ),
            "high_inclusive": (
                source.get("high_inclusive")
                if isinstance(source.get("high_inclusive"), bool)
                else True
            ),
            "unit": None,
            "canonical_low": None,
            "canonical_high": None,
            "canonical_unit": None,
            "status": "ok",
            "unit_status": "missing",
        },
        _mapping_value(
            source,
            ("unit", "units", "reference_unit"),
        ),
        "mapping",
    )


def _canonical_bound(
    value: float | None,
    unit: str | None,
    *,
    language: object | None = None,
) -> float | None:
    if value is None or unit is None:
        return None
    parsed = parse_measurement(value, unit, language=language)
    if parsed.get("status") != "ok":
        return None
    canonical = parsed.get("canonical_magnitude")
    return canonical if isinstance(canonical, float) else None


def _prepare_range(
    source: object,
    *,
    value_unit: _UnitDetails,
    language: object | None = None,
) -> tuple[LabReferenceRange, str]:
    result, raw_unit, source_kind = _range_parts(source, language=language)
    if result["status"] != "ok":
        return result, source_kind

    range_unit = _unit_details(raw_unit, language=language)
    result["unit_status"] = range_unit["status"]
    result["unit"] = range_unit["unit"]

    if range_unit["status"] in {"unknown", "ambiguous"}:
        result["status"] = "unknown_unit"
        return result, source_kind

    bound_unit: str | None
    if range_unit["status"] == "known":
        bound_unit = range_unit["unit"]
        result["canonical_unit"] = range_unit["canonical_unit"]
    elif value_unit["status"] == "known":
        bound_unit = value_unit["unit"]
        result["canonical_unit"] = value_unit["canonical_unit"]
    else:
        bound_unit = None

    result["canonical_low"] = _canonical_bound(
        result["low"],
        bound_unit,
        language=language,
    )
    result["canonical_high"] = _canonical_bound(
        result["high"],
        bound_unit,
        language=language,
    )

    if (
        range_unit["status"] == "known"
        and value_unit["status"] == "known"
        and range_unit["dimension"] != value_unit["dimension"]
    ):
        result["status"] = "incommensurable"
    return result, source_kind


def _comparison_range(reference_range: LabReferenceRange) -> dict[str, object]:
    result: dict[str, object] = {
        "low": reference_range["low"],
        "high": reference_range["high"],
        "low_inclusive": reference_range["low_inclusive"],
        "high_inclusive": reference_range["high_inclusive"],
    }
    if reference_range["unit"] is not None:
        result["unit"] = reference_range["unit"]
    return result


def _flag_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _interpretation(
    value: float | None,
    reference_range: LabReferenceRange,
    *,
    value_unit: _UnitDetails,
    explicit_flag: object,
    language: object | None = None,
) -> AbnormalFlag:
    flag = _flag_text(explicit_flag)
    if flag is not None:
        # Explicit originating-lab flags remain useful evidence even when the
        # value's unit cannot be normalized. Unknown flags still fail closed.
        return derive_abnormal_flag(
            value,
            _comparison_range(reference_range),
            explicit_flag=flag,
            language=language,
        )

    if value is None or value_unit["status"] != "known":
        return "unknown"
    if reference_range["status"] != "ok":
        return "unknown"

    reference_unit = (
        reference_range["unit"] if reference_range["unit_status"] == "known" else None
    )
    value_unit_text = value_unit["unit"] if reference_unit is not None else None
    return derive_abnormal_flag(
        value,
        _comparison_range(reference_range),
        value_unit=value_unit_text,
        reference_unit=reference_unit,
        language=language,
    )


def _result_status(
    value: float | None,
    value_unit: _UnitDetails,
    reference_range: LabReferenceRange,
) -> tuple[LabMeasurementStatus, str]:
    if value is None:
        return "invalid_value", "measurement value is not finite numeric"
    if value_unit["status"] == "missing":
        return "unknown_unit", "measurement unit is required"
    if value_unit["status"] == "ambiguous":
        return "unknown_unit", "measurement unit is ambiguous"
    if value_unit["status"] == "unknown":
        return "unknown_unit", "measurement unit is unknown"
    if reference_range["status"] == "invalid":
        return "invalid_range", "reference range is invalid"
    if reference_range["status"] == "unknown_unit":
        return "unknown_unit", "reference-range unit is unknown"
    if reference_range["status"] == "incommensurable":
        return "unknown_unit", "measurement and reference-range units differ"
    return "ok", ""


def normalize_lab_measurement(
    measurement: Mapping[str, object] | object,
    unit: object | None = None,
    reference_range: object | None = None,
    *,
    analyte: object | None = None,
    flag: object | None = None,
    qualifiers: object | None = None,
    source_offsets: object | None = None,
    start: int | None = None,
    end: int | None = None,
    language: object | None = None,
) -> LabMeasurement:
    """Normalize one synthetic lab measurement without guessing its unit.

    ``measurement`` may be a numeric value, a string such as ``"120 mg/dL"``,
    or a mapping with ``value``, ``unit``, ``reference_range``, ``flag``,
    ``qualifiers``, and source offset fields.  The positional ``unit`` and
    ``reference_range`` arguments are convenient for scalar callers.

    Source offsets are half-open character positions.  They can be supplied as
    ``source_offsets=(start, end)`` or as ``start``/``end``.  The returned
    record contains offsets and normalized evidence only; it does not copy the
    source measurement string into provenance.

    Args:
        measurement: Numeric value, value-with-unit string, or extracted row.
        unit: Optional unit when ``measurement`` is numeric.
        reference_range: Optional range text or mapping with numeric bounds.
        analyte: Optional analyte label to carry through after whitespace
            normalization.
        flag: Optional originating-laboratory flag such as ``"H"`` or ``"L"``.
        qualifiers: Optional qualifier string or iterable of strings.
        source_offsets: Optional ``(start, end)`` source span.
        start: Optional source-span start, used with ``end``.
        end: Optional source-span end, used with ``start``.
        language: Optional source-language code for localized numbers and unit
            aliases.

    Returns:
        A JSON-compatible typed record.  ``status="unknown_unit"`` and
        ``unit_status`` identify missing, unknown, or ambiguous units; no
        conversion is attempted in that case.
    """

    mapping = measurement if isinstance(measurement, Mapping) else None
    if mapping is not None:
        raw_value = _mapping_value(mapping, ("value", "magnitude", "result"))
        raw_unit = (
            unit if unit is not None else _mapping_value(mapping, ("unit", "units"))
        )
        raw_range = (
            reference_range
            if reference_range is not None
            else _mapping_value(mapping, ("reference_range", "range", "ref_range"))
        )
        raw_analyte = (
            analyte
            if analyte is not None
            else _mapping_value(mapping, ("analyte", "name", "test", "lab_name"))
        )
        raw_flag = (
            flag
            if flag is not None
            else _mapping_value(mapping, ("flag", "abnormal_flag", "interpretation"))
        )
        raw_qualifiers = (
            qualifiers
            if qualifiers is not None
            else _mapping_value(mapping, ("qualifiers", "qualifier"))
        )
        raw_offsets = (
            source_offsets if source_offsets is not None else _mapping_offsets(mapping)
        )
        input_source = "mapping"
    else:
        raw_value = measurement
        raw_unit = unit
        raw_range = reference_range
        raw_analyte = analyte
        raw_flag = flag
        raw_qualifiers = qualifiers
        raw_offsets = source_offsets
        input_source = "arguments"

    if start is not None or end is not None:
        if source_offsets is not None:
            raise ValueError("provide source_offsets or start/end, not both")
        raw_offsets = {"start": start, "end": end}
    offsets = _validate_offsets(raw_offsets)

    numeric_input, unit_input, value_form = _source_value_parts(raw_value, raw_unit)
    value = _finite_numeric(numeric_input, language=language)
    value_unit = _unit_details(unit_input, language=language)
    canonical_value: float | None = None
    if value is not None and value_unit["status"] == "known":
        parsed = parse_measurement(value, value_unit["unit"], language=language)
        if parsed.get("status") == "ok":
            candidate = parsed.get("canonical_magnitude")
            if isinstance(candidate, float):
                canonical_value = candidate

    normalized_range, range_source = _prepare_range(
        raw_range,
        value_unit=value_unit,
        language=language,
    )
    interpretation = _interpretation(
        value,
        normalized_range,
        value_unit=value_unit,
        explicit_flag=raw_flag,
        language=language,
    )
    status, reason = _result_status(value, value_unit, normalized_range)

    analyte_text = _clean_optional_text(raw_analyte)
    safe_flag = _flag_text(raw_flag)
    provenance: dict[str, object] = {
        "input_source": input_source,
        "value_form": value_form,
        "range_source": range_source,
        "offsets_provided": offsets is not None,
        "range_unit_status": normalized_range["unit_status"],
    }
    if safe_flag is not None:
        provenance["explicit_flag_provided"] = True
        provenance["explicit_flag"] = safe_flag.casefold()

    return {
        "analyte": analyte_text,
        "value": value,
        "unit": value_unit["unit"],
        "canonical_value": canonical_value,
        "canonical_unit": value_unit["canonical_unit"],
        "dimension": value_unit["dimension"],
        "reference_range": normalized_range,
        "interpretation": interpretation,
        "qualifiers": _normalize_qualifiers(raw_qualifiers),
        "source_offsets": offsets,
        "status": status,
        "unit_status": value_unit["status"],
        "reason": reason,
        "advisory": LAB_MEASUREMENT_ADVISORY,
        "provenance": provenance,
    }


def normalize_lab_measurements(
    measurements: Iterable[Mapping[str, object] | object],
    *,
    language: object | None = None,
) -> list[LabMeasurement]:
    """Normalize a deterministic sequence of extracted lab measurements.

    Input order is preserved.  A single mapping or value string is accepted as
    a convenience and produces a one-item list.
    """

    if isinstance(measurements, Mapping) or isinstance(measurements, str):
        measurements = (measurements,)
    return [
        normalize_lab_measurement(measurement, language=language)
        for measurement in measurements
    ]


__all__ = [
    "LAB_MEASUREMENT_ADVISORY",
    "LabMeasurement",
    "LabMeasurementStatus",
    "LabRangeStatus",
    "LabReferenceRange",
    "LabUnitStatus",
    "SourceOffsets",
    "normalize_lab_measurement",
    "normalize_lab_measurements",
]
