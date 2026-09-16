"""Offline dose-range checks over caller-supplied reference data.

This module deliberately ships no dosing guidance. A caller supplies a local
reference-range table keyed by drug and route, and the checker compares already
extracted dose amounts after deterministic unit normalization. An out-of-range
value becomes a guarded clinician-review advisory; it is never corrected,
capped, or replaced with a recommendation. Missing references and incompatible
units are explicit ``not_checked`` notes rather than implicit passes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

from .decision_support import GuardedSuggestion, SourceSpan, build_guarded_suggestion
from .lexicons.clinical_norm import split_measurement_text
from .medication_sig import DoseNormalization, normalize_dose

DOSE_RANGE_ADVISORY = (
    "Dose-range comparison is a clinician-review advisory based only on a "
    "caller-supplied reference range; it does not correct, cap, or recommend a "
    "dose."
)

DOSE_RANGE_DATA_NOTICE = (
    "OpenMed includes no production dosing guidance. Callers must supply a "
    "local, license-cleared reference-range table and remain responsible for "
    "its scope, currency, and permitted use."
)

DoseRangeTableSource: TypeAlias = (
    Mapping[Any, Any] | Sequence[Mapping[str, Any]] | str | PathLike[str] | None
)


class DoseRange(TypedDict, total=False):
    """A caller-supplied dose range row."""

    low: object
    high: object
    low_inclusive: bool
    high_inclusive: bool
    unit: str


class DoseRangeTableError(ValueError):
    """Raised when a caller-supplied dose-range table is malformed."""


@dataclass(frozen=True)
class _Bound:
    value: object
    unit: str | None
    canonical_value: float
    canonical_unit: str | None
    dimension: dict[str, int]


@dataclass(frozen=True)
class _ReferenceRange:
    drug: str
    route: str
    low: _Bound | None
    high: _Bound | None
    low_inclusive: bool
    high_inclusive: bool


@dataclass(frozen=True)
class _DoseRecord:
    index: int
    drug: str | None
    route: str | None
    value: object
    unit: object | None
    source_spans: tuple[SourceSpan, ...]


def check_dose_ranges(
    doses: Iterable[Any] | Mapping[str, Any],
    reference_ranges: DoseRangeTableSource,
    *,
    language: object | None = None,
) -> list[GuardedSuggestion]:
    """Flag extracted doses outside caller-supplied drug/route ranges.

    Args:
        doses: Dose-like mappings or objects. Each item should expose a drug
            name, route, and dose amount/unit. ``dose``, ``amount``, ``value``,
            or a normalized-dose mapping are accepted. ``start``/``end`` or
            ``source_spans`` values are used for traceability; deterministic
            synthetic offsets are used when callers do not provide spans.
        reference_ranges: A local mapping, sequence of rows, or local JSON path
            containing caller-owned ranges. The canonical mapping shape is
            ``{"ranges": {"drug": {"route": {"low": ..., "high": ...,
            "unit": "mg"}}}}``. No URL or network source is accepted.
        language: Optional source-language code for localized numbers and unit
            aliases.

    Returns:
        ``GuardedSuggestion`` values for above/below-range doses and explicit
        ``not_checked`` notes. In-range doses produce no suggestion.

    Note:
        A returned flag is an assistive review signal only. This function never
        changes an extracted amount and never emits a replacement dose.

    Raises:
        DoseRangeTableError: If the supplied table cannot be loaded or contains
            an invalid range row.
        TypeError: If ``doses`` is not an iterable of dose-like values.
    """

    if isinstance(doses, (str, bytes)) or doses is None:
        raise TypeError("doses must be an iterable of dose-like values")
    if isinstance(doses, Mapping):
        dose_values = [doses]
    else:
        try:
            dose_values = list(doses)
        except TypeError as exc:
            raise TypeError("doses must be an iterable of dose-like values") from exc

    ranges = _load_reference_ranges(reference_ranges, language=language)
    records = _dose_records(dose_values)
    suggestions: list[GuardedSuggestion] = []

    for record in records:
        reference = _find_reference(ranges, record.drug, record.route)
        if reference is None:
            suggestions.append(
                _not_checked_note(
                    record,
                    "no reference range supplied, not checked",
                    reason="missing_reference_range",
                )
            )
            continue

        normalized = normalize_dose(
            record.value,
            record.unit,
            language=language,
        )
        if not normalized["recognized"]:
            note = (
                "unit mismatch, not checked"
                if normalized.get("unit") is not None
                else "dose could not be normalized, not checked"
            )
            suggestions.append(
                _not_checked_note(
                    record,
                    note,
                    reason="dose_normalization_failed",
                )
            )
            continue

        comparison = _compare_to_reference(normalized, reference)
        if comparison is None:
            suggestions.append(
                _not_checked_note(
                    record,
                    "unit mismatch, not checked",
                    reason="incompatible_units",
                )
            )
            continue

        bound_side, bound = comparison
        if bound_side is not None and bound is not None:
            suggestions.append(
                _range_flag(
                    record,
                    normalized,
                    reference,
                    bound_side,
                    bound,
                )
            )

    return suggestions


def _dose_records(values: Sequence[Any]) -> list[_DoseRecord]:
    records: list[_DoseRecord] = []
    cursor = 0
    for index, value in enumerate(values):
        drug = _text_or_none(
            _field(
                value,
                "drug",
                "drug_name",
                "medication",
                "medication_name",
                "name",
            )
        )
        route = _text_or_none(
            _field(value, "route", "administration_route", "admin_route")
        )
        dose_value, dose_unit, sig_route = _dose_input(value)
        if route is None:
            route = sig_route
        display = drug or f"dose at index {index}"
        source_spans = _source_spans(value)
        if not source_spans:
            end = cursor + max(len(display), 1)
            source_spans = (SourceSpan(start=cursor, end=end, label=f"dose[{index}]"),)
            cursor = end + 1
        records.append(
            _DoseRecord(
                index=index,
                drug=drug,
                route=route,
                value=dose_value,
                unit=dose_unit,
                source_spans=source_spans,
            )
        )
    return records


def _dose_input(value: Any) -> tuple[object, object | None, str | None]:
    normalized = _field(value, "normalized_dose", "dose_normalized")
    raw_dose = normalized
    if raw_dose is None:
        raw_dose = _field(value, "dose", "amount", "value", "magnitude")
    explicit_unit = _field(value, "unit", "units", "dose_unit", "dose_units")

    if raw_dose is None:
        sig = _field(value, "sig", "instructions", "dose_sig")
        if isinstance(sig, str):
            parsed = _parse_sig_fields(sig)
            raw_dose = parsed[0]
            explicit_unit = parsed[1]
            return raw_dose, explicit_unit, parsed[2]

    if isinstance(raw_dose, Mapping):
        nested_value = _field(
            raw_dose,
            "value",
            "amount",
            "dose",
            "magnitude",
        )
        nested_unit = _field(
            raw_dose,
            "unit",
            "units",
            "dose_unit",
            "dose_units",
        )
        if nested_value is not None:
            raw_dose = nested_value
        if nested_unit is not None:
            explicit_unit = nested_unit

    if raw_dose is None:
        raw_dose = value
    return raw_dose, explicit_unit, None


def _parse_sig_fields(sig: str) -> tuple[object, object | None, str | None]:
    from .sig_parser import parse_sig

    parsed = parse_sig(sig)
    return parsed["dose"], parsed["unit"], parsed["route"]


def _source_spans(value: Any) -> tuple[SourceSpan, ...]:
    raw_spans = _field(value, "source_spans", "source_span")
    if raw_spans is not None:
        if isinstance(raw_spans, (SourceSpan, Mapping)):
            candidates = (raw_spans,)
        elif isinstance(raw_spans, Iterable) and not isinstance(
            raw_spans, (str, bytes)
        ):
            candidates = tuple(raw_spans)
        else:
            candidates = (raw_spans,)
        return tuple(SourceSpan.from_obj(span) for span in candidates)

    start = _field(value, "start", "source_start")
    end = _field(value, "end", "source_end")
    if type(start) is int and type(end) is int and 0 <= start < end:
        return (SourceSpan(start=start, end=end),)
    return ()


def _not_checked_note(
    record: _DoseRecord,
    note: str,
    *,
    reason: str,
) -> GuardedSuggestion:
    return build_guarded_suggestion(
        {
            "kind": "dose_range_note",
            "status": "not_checked",
            "drug": record.drug,
            "route": record.route,
            "note": note,
            "review": "clinician_review_required",
        },
        record.source_spans,
        1.0,
        provenance={
            "producer": "openmed.clinical.dosing_check",
            "reason": reason,
            "reference_data": "caller_supplied",
        },
    )


def _range_flag(
    record: _DoseRecord,
    normalized: DoseNormalization,
    reference: _ReferenceRange,
    bound_side: str,
    bound: _Bound,
) -> GuardedSuggestion:
    observed_value = normalized["value"]
    observed_unit = normalized["unit"]
    reference_bound = bound.value
    return build_guarded_suggestion(
        {
            "kind": "dose_range_flag",
            "status": "above_range" if bound_side == "high" else "below_range",
            "direction": "above" if bound_side == "high" else "below",
            "drug": record.drug,
            "route": record.route,
            "observed_value": observed_value,
            "observed_unit": observed_unit,
            "reference_bound": reference_bound,
            "reference_bound_unit": bound.unit,
            "bound": bound_side,
            "reference_range": {
                "low": reference.low.value if reference.low is not None else None,
                "high": (reference.high.value if reference.high is not None else None),
                "low_inclusive": reference.low_inclusive,
                "high_inclusive": reference.high_inclusive,
            },
            "advisory": DOSE_RANGE_ADVISORY,
            "review": "clinician_review_required",
        },
        record.source_spans,
        1.0,
        provenance={
            "producer": "openmed.clinical.dosing_check",
            "comparison": "dimension_checked_unit_normalization",
            "reference_data": "caller_supplied",
            "bound": bound_side,
        },
    )


def _compare_to_reference(
    normalized: DoseNormalization,
    reference: _ReferenceRange,
) -> tuple[str | None, _Bound | None] | None:
    observed = normalized.get("canonical_value")
    if observed is None:
        return None
    dimension = dict(normalized.get("dimension", {}))
    for bound in (reference.low, reference.high):
        if bound is not None and bound.dimension != dimension:
            return None

    if reference.low is not None:
        low = reference.low
        below = observed < low.canonical_value
        if observed == low.canonical_value and not reference.low_inclusive:
            below = True
        if below:
            return "low", low

    if reference.high is not None:
        high = reference.high
        above = observed > high.canonical_value
        if observed == high.canonical_value and not reference.high_inclusive:
            above = True
        if above:
            return "high", high

    return None, None


def _load_reference_ranges(
    source: DoseRangeTableSource,
    *,
    language: object | None,
) -> dict[tuple[str, str], _ReferenceRange]:
    payload: Any = source
    if source is None:
        return {}
    if isinstance(source, (str, PathLike)):
        path_text = str(source)
        if "://" in path_text:
            raise DoseRangeTableError(
                "reference_ranges must be a local mapping or file path, not a URL"
            )
        path = Path(source)
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DoseRangeTableError(
                f"could not load local dose-range table {path}"
            ) from exc

    rows = _range_rows(payload)
    ranges: dict[tuple[str, str], _ReferenceRange] = {}
    for index, row in enumerate(rows):
        reference = _parse_reference_range(row, index=index, language=language)
        key = (reference.drug, reference.route)
        existing = ranges.get(key)
        if existing is not None and existing != reference:
            raise DoseRangeTableError(
                f"dose-range table contains conflicting records for {key!r}"
            )
        ranges[key] = reference
    return ranges


def _range_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        nested = payload.get("ranges", payload.get("reference_ranges"))
        if nested is not None:
            return _range_rows(nested)
        if _has_range_fields(payload):
            return [payload]

        rows: list[Mapping[str, Any]] = []
        for drug, route_values in payload.items():
            if drug in {"metadata", "schema_version"}:
                continue
            if isinstance(drug, (tuple, list)) and len(drug) == 2:
                if isinstance(route_values, Mapping):
                    row = dict(route_values)
                    row.setdefault("drug", drug[0])
                    row.setdefault("route", drug[1])
                    rows.append(row)
                continue
            if not isinstance(route_values, Mapping):
                continue
            if _has_range_fields(route_values):
                row = dict(route_values)
                row.setdefault("drug", drug)
                rows.append(row)
                continue
            for route, range_value in route_values.items():
                if not isinstance(range_value, Mapping):
                    continue
                row = dict(range_value)
                row.setdefault("drug", drug)
                row.setdefault("route", route)
                rows.append(row)
        return rows

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [row for row in payload if isinstance(row, Mapping)]
    raise DoseRangeTableError("dose-range table must be a mapping or row sequence")


def _parse_reference_range(
    row: Mapping[str, Any],
    *,
    index: int,
    language: object | None,
) -> _ReferenceRange:
    drug = _normalize_key(
        _field(row, "drug", "drug_name", "medication", "medication_name", "name")
    )
    route = _normalize_key(_field(row, "route", "administration_route", "admin_route"))
    if not drug or not route:
        raise DoseRangeTableError(
            f"dose-range row {index} requires both drug and route keys"
        )

    nested = _field(row, "range", "reference_range", "bounds")
    range_row: Mapping[str, Any] = nested if isinstance(nested, Mapping) else row
    low_raw = _field(range_row, "low", "minimum", "min", "lower")
    high_raw = _field(range_row, "high", "maximum", "max", "upper")
    if low_raw is None and high_raw is None:
        raise DoseRangeTableError(
            f"dose-range row {index} must supply a low or high bound"
        )

    row_unit = _field(range_row, "unit", "units", "dose_unit", "dose_units")
    if row_unit is None and range_row is not row:
        row_unit = _field(row, "unit", "units", "dose_unit", "dose_units")
    low = _normalize_bound(low_raw, row_unit, language=language)
    high = _normalize_bound(high_raw, row_unit, language=language)
    if low_raw is not None and low is None:
        raise DoseRangeTableError(f"dose-range row {index} has an invalid low bound")
    if high_raw is not None and high is None:
        raise DoseRangeTableError(f"dose-range row {index} has an invalid high bound")
    if low is None and high is None:
        raise DoseRangeTableError(f"dose-range row {index} has no usable bounds")
    if low is not None and high is not None and low.dimension != high.dimension:
        raise DoseRangeTableError(
            f"dose-range row {index} contains incompatible bound units"
        )

    return _ReferenceRange(
        drug=drug,
        route=route,
        low=low,
        high=high,
        low_inclusive=_bool_or_default(_field(range_row, "low_inclusive"), True),
        high_inclusive=_bool_or_default(_field(range_row, "high_inclusive"), True),
    )


def _normalize_bound(
    raw: object,
    unit: object | None,
    *,
    language: object | None,
) -> _Bound | None:
    bound_value = raw
    bound_unit = unit
    if isinstance(raw, Mapping):
        bound_value = _field(raw, "value", "amount", "dose", "magnitude")
        nested_unit = _field(raw, "unit", "units", "dose_unit", "dose_units")
        if nested_unit is not None:
            bound_unit = nested_unit
    elif bound_unit is None and isinstance(raw, str):
        measurement = split_measurement_text(raw)
        if measurement is not None:
            bound_value, bound_unit = measurement

    normalized = normalize_dose(bound_value, bound_unit, language=language)
    canonical_value = normalized.get("canonical_value")
    if not normalized["recognized"] or canonical_value is None:
        return None
    return _Bound(
        value=normalized["value"],
        unit=normalized["unit"],
        canonical_value=canonical_value,
        canonical_unit=normalized["canonical_unit"],
        dimension=dict(normalized["dimension"]),
    )


def _find_reference(
    ranges: Mapping[tuple[str, str], _ReferenceRange],
    drug: str | None,
    route: str | None,
) -> _ReferenceRange | None:
    normalized_drug = _normalize_key(drug)
    normalized_route = _normalize_key(route)
    if not normalized_drug or not normalized_route:
        return None
    return ranges.get((normalized_drug, normalized_route)) or ranges.get(
        (normalized_drug, "*")
    )


def _has_range_fields(value: Mapping[Any, Any]) -> bool:
    return any(
        key in value
        for key in (
            "low",
            "high",
            "minimum",
            "maximum",
            "min",
            "max",
            "lower",
            "upper",
            "range",
            "reference_range",
            "bounds",
        )
    )


def _field(value: Any, *names: str, default: object | None = None) -> object | None:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _text_or_none(value: object | None) -> str | None:
    if isinstance(value, Mapping):
        value = _field(value, "name", "text", "display", "value")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_key(value: object | None) -> str:
    text = _text_or_none(value)
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _bool_or_default(value: object | None, default: bool) -> bool:
    return value if isinstance(value, bool) else default


__all__ = [
    "DOSE_RANGE_ADVISORY",
    "DOSE_RANGE_DATA_NOTICE",
    "DoseRange",
    "DoseRangeTableError",
    "DoseRangeTableSource",
    "check_dose_ranges",
]
