"""Advisory dosing-range checks from caller-supplied reference tables."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from .units import parse_measurement

DoseCheckStatus = Literal["below_range", "above_range"]
DoseCheckNoteReason = Literal["missing_reference_range", "unit_mismatch"]

DOSING_RANGE_CHECK_DISCLAIMER = (
    "Dosing range checks are advisory review flags only. Reference ranges must "
    "be supplied by the user from appropriate local formulary or jurisdictional "
    "sources; OpenMed does not recommend, correct, cap, or prescribe doses."
)


class ExtractedDose(TypedDict, total=False):
    """Normalized medication dose emitted by an upstream extractor."""

    id: str
    drug: str
    route: str
    amount: float
    unit: str


class DoseReferenceRange(TypedDict, total=False):
    """Caller-supplied dose bounds for a drug and route."""

    low: float | None
    high: float | None
    unit: str


class ClinicalSuggestion(TypedDict, total=False):
    """Advisory clinical review flag for an out-of-range extracted dose."""

    type: str
    status: DoseCheckStatus
    drug: str
    route: str
    observed_value: float
    observed_unit: str
    reference_bound: float
    reference_bound_type: Literal["low", "high"]
    reference_unit: str
    message: str
    disclaimer: str


class DoseCheckNote(TypedDict, total=False):
    """Explicit not-checked note when comparison would be unsafe."""

    type: str
    reason: DoseCheckNoteReason
    drug: str
    route: str
    message: str
    disclaimer: str


class DoseRangeCheckResult(TypedDict):
    """Structured result for advisory dose range checks."""

    flags: list[ClinicalSuggestion]
    notes: list[DoseCheckNote]


def check_dose_ranges(
    doses: Sequence[ExtractedDose | Mapping[str, object]],
    reference_ranges: Mapping[object, object],
) -> DoseRangeCheckResult:
    """Flag extracted doses outside user-supplied drug/route bounds.

    The function compares normalized numeric dose amounts and units against
    caller-supplied reference ranges. It emits review flags only; it never
    corrects, caps, recommends, or prescribes a dose. Missing references and
    incompatible or unknown units are returned as explicit not-checked notes.
    """

    flags: list[ClinicalSuggestion] = []
    notes: list[DoseCheckNote] = []

    for dose in doses:
        drug = _clean_string(dose.get("drug"))
        route = _clean_string(dose.get("route"))
        reference = _lookup_reference(reference_ranges, drug, route)
        if reference is None:
            notes.append(_note("missing_reference_range", drug, route))
            continue

        comparison = _canonical_comparison(dose, reference)
        if comparison is None:
            notes.append(_note("unit_mismatch", drug, route))
            continue

        value, low, high = comparison
        if low is not None and value < low:
            flags.append(
                _flag(
                    "below_range",
                    drug,
                    route,
                    dose,
                    reference,
                    bound=reference["low"],
                    bound_type="low",
                )
            )
        elif high is not None and value > high:
            flags.append(
                _flag(
                    "above_range",
                    drug,
                    route,
                    dose,
                    reference,
                    bound=reference["high"],
                    bound_type="high",
                )
            )

    return {"flags": flags, "notes": notes}


def _lookup_reference(
    table: Mapping[object, object],
    drug: str,
    route: str,
) -> DoseReferenceRange | None:
    compound_keys = (
        f"{drug}|{route}",
        f"{drug}:{route}",
        f"{drug}/{route}",
    )
    normalized = {_normalize_key(key): value for key, value in table.items()}
    for key in compound_keys:
        value = normalized.get(_normalize_key(key))
        if isinstance(value, Mapping):
            return _normalize_reference(value)

    drug_entry = normalized.get(_normalize_key(drug))
    if isinstance(drug_entry, Mapping):
        route_entry = {
            _normalize_key(key): value for key, value in drug_entry.items()
        }.get(_normalize_key(route))
        if isinstance(route_entry, Mapping):
            return _normalize_reference(route_entry)
    return None


def _normalize_reference(reference: Mapping[str, object]) -> DoseReferenceRange:
    low = _finite_float(_first_present(reference, "low", "min"))
    high = _finite_float(_first_present(reference, "high", "max"))
    normalized: DoseReferenceRange = {"low": low, "high": high}
    unit = _clean_string(reference.get("unit") or reference.get("units"))
    if unit:
        normalized["unit"] = unit
    return normalized


def _canonical_comparison(
    dose: Mapping[str, object],
    reference: DoseReferenceRange,
) -> tuple[float, float | None, float | None] | None:
    dose_unit = _clean_string(dose.get("unit"))
    reference_unit = _clean_string(reference.get("unit"))
    if not dose_unit or not reference_unit:
        return None

    observed = parse_measurement(_dose_amount(dose), dose_unit)
    if observed["status"] != "ok":
        return None

    dimension = observed["dimension"]
    value = observed["canonical_magnitude"]
    if value is None:
        return None

    low = _canonical_bound(reference.get("low"), reference_unit, dimension)
    high = _canonical_bound(reference.get("high"), reference_unit, dimension)
    if low is False or high is False:
        return None
    return value, low, high


def _canonical_bound(
    value: object,
    unit: str,
    dimension: Mapping[str, int],
) -> float | None | Literal[False]:
    if value is None:
        return None
    parsed = parse_measurement(value, unit)
    if parsed["status"] != "ok" or parsed["dimension"] != dimension:
        return False
    return parsed["canonical_magnitude"]


def _flag(
    status: DoseCheckStatus,
    drug: str,
    route: str,
    dose: Mapping[str, object],
    reference: DoseReferenceRange,
    *,
    bound: object,
    bound_type: Literal["low", "high"],
) -> ClinicalSuggestion:
    observed_value = _finite_float(_dose_amount(dose))
    reference_bound = _finite_float(bound)
    return {
        "type": "dosing_range_review",
        "status": status,
        "drug": drug,
        "route": route,
        "observed_value": observed_value if observed_value is not None else math.nan,
        "observed_unit": _clean_string(dose.get("unit")),
        "reference_bound": reference_bound if reference_bound is not None else math.nan,
        "reference_bound_type": bound_type,
        "reference_unit": _clean_string(reference.get("unit")),
        "message": (
            f"{drug} {route} dose is {status.replace('_', ' ')}; "
            "flagged for clinician review."
        ),
        "disclaimer": DOSING_RANGE_CHECK_DISCLAIMER,
    }


def _note(reason: DoseCheckNoteReason, drug: str, route: str) -> DoseCheckNote:
    if reason == "missing_reference_range":
        message = "no reference range supplied, not checked"
    else:
        message = "unit mismatch, not checked"
    return {
        "type": "dosing_range_not_checked",
        "reason": reason,
        "drug": drug,
        "route": route,
        "message": message,
        "disclaimer": DOSING_RANGE_CHECK_DISCLAIMER,
    }


def _clean_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _dose_amount(dose: Mapping[str, object]) -> object:
    return _first_present(dose, "amount", "value", "dose_amount")


def _first_present(reference: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in reference:
            return reference[key]
    return None


def _normalize_key(value: object) -> str:
    return _clean_string(value).casefold()


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
