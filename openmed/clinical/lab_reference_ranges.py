"""Typed, offline provenance for laboratory reference ranges.

Reference ranges are only comparable when their measurement context is explicit.
This module keeps that context alongside synthetic range bounds and resolves
candidate ranges by exact provenance matching. It deliberately does not convert
units, infer a source instrument, or bridge locale-specific ranges.

The public records contain only range metadata and source fingerprints. Callers
must not pass patient text or raw report content as an analyte or source label.
The output is assistive metadata for review, not a clinical decision or a
substitute for the originating laboratory's report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

REFERENCE_RANGE_SCHEMA_VERSION = 1
LAB_REFERENCE_RANGE_ADVISORY = (
    "Laboratory reference ranges are deterministic assistive metadata for "
    "comparison and review; they are not a clinical decision or a substitute "
    "for the originating laboratory's report."
)

ReferenceRangeState = Literal["known", "unknown", "conflict"]

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCALE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")


class ReferenceRangeStatus(str, Enum):
    """State returned when selecting or comparing a reference range."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    CONFLICTING = "conflict"


def _normalized_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"reference-range {field_name} must be a non-empty string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"reference-range {field_name} must be a non-empty string")
    return normalized


def _normalized_population(value: object) -> str:
    return _normalized_text(value, "population").casefold()


def _normalized_locale(value: object | None) -> str | None:
    if value is None:
        return None
    locale = _normalized_text(value, "locale").replace("_", "-").casefold()
    if _LOCALE_RE.fullmatch(locale) is None:
        raise ValueError("reference-range locale must be a BCP 47-like tag")
    return locale


def _normalized_fingerprint(value: object) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value.casefold()) is None:
        raise ValueError("reference-range source_fingerprint must be a SHA-256 digest")
    return value.casefold()


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"reference-range {field_name} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reference-range {field_name} must be finite numeric"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"reference-range {field_name} must be finite numeric")
    return number


def _canonical_source_value(value: object) -> object:
    """Return a stable JSON-compatible source value without retaining it."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source metadata must contain finite JSON values")
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("source metadata keys must be non-empty strings")
            normalized[key] = _canonical_source_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_source_value(item) for item in value]
    raise TypeError("source metadata must be JSON-compatible")


def fingerprint_source(source: object) -> str:
    """Return a deterministic SHA-256 fingerprint for local source metadata.

    ``source`` may be a string identifier, bytes, or a JSON-compatible mapping
    containing an instrument/version identifier. The source itself is never
    stored in the returned provenance record. Mapping keys are sorted so the
    fingerprint is independent of insertion order.
    """

    canonical = _canonical_source_value(source)
    payload = json.dumps(
        {"schema_version": REFERENCE_RANGE_SCHEMA_VERSION, "source": canonical},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def source_fingerprint(source: object) -> str:
    """Alias for :func:`fingerprint_source` used by callers building records."""

    return fingerprint_source(source)


@dataclass(frozen=True)
class ReferenceRangeProvenance:
    """Measurement context required to compare one reference range.

    ``precision`` is the number of decimal places represented by the bounds.
    ``source_fingerprint`` should identify the local instrument or range source
    without retaining that source's raw identifier. ``locale`` is optional for
    synthetic ranges whose source is explicitly locale-neutral; a locale is
    never inferred during resolution.
    """

    unit: str
    population: str
    precision: int
    source_fingerprint: str
    locale: str | None = None

    def __post_init__(self) -> None:
        unit = _normalized_text(self.unit, "unit")
        population = _normalized_population(self.population)
        if isinstance(self.precision, bool) or not isinstance(self.precision, int):
            raise ValueError("reference-range precision must be a non-negative integer")
        if self.precision < 0:
            raise ValueError("reference-range precision must be a non-negative integer")
        fingerprint = _normalized_fingerprint(self.source_fingerprint)
        locale = _normalized_locale(self.locale)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "source_fingerprint", fingerprint)
        object.__setattr__(self, "locale", locale)

    @property
    def measurement_key(self) -> tuple[str, str, int]:
        """Return the unit/population/precision comparison context."""

        return self.unit, self.population, self.precision

    @property
    def identity_key(self) -> tuple[str, str, int, str, str | None]:
        """Return the complete provenance identity used for exact matching."""

        return (*self.measurement_key, self.source_fingerprint, self.locale)

    def matches(self, other: "ReferenceRangeProvenance") -> bool:
        """Return whether two provenance records are exactly comparable."""

        if not isinstance(other, ReferenceRangeProvenance):
            return False
        return self.identity_key == other.identity_key

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, source-value-free mapping."""

        return {
            "unit": self.unit,
            "population": self.population,
            "precision": self.precision,
            "source_fingerprint": self.source_fingerprint,
            "locale": self.locale,
        }


@dataclass(frozen=True)
class LabReferenceRange:
    """One typed synthetic laboratory reference range."""

    analyte: str
    low: float | None
    high: float | None
    provenance: ReferenceRangeProvenance
    low_inclusive: bool = True
    high_inclusive: bool = True

    def __post_init__(self) -> None:
        analyte = _normalized_text(self.analyte, "analyte")
        low = None if self.low is None else _finite_number(self.low, "low")
        high = None if self.high is None else _finite_number(self.high, "high")
        if low is None and high is None:
            raise ValueError("reference range must include a low or high bound")
        if low is not None and high is not None and low > high:
            raise ValueError("reference range low bound cannot exceed high bound")
        if not isinstance(self.provenance, ReferenceRangeProvenance):
            raise TypeError("reference range provenance must be typed metadata")
        if not isinstance(self.low_inclusive, bool) or not isinstance(
            self.high_inclusive, bool
        ):
            raise ValueError("reference-range inclusivity flags must be booleans")
        object.__setattr__(self, "analyte", analyte)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def analyte_key(self) -> str:
        """Return the stable case-insensitive analyte identity."""

        return self.analyte.casefold()

    @property
    def unit(self) -> str:
        """Return the explicitly recorded measurement unit."""

        return self.provenance.unit

    @property
    def population(self) -> str:
        """Return the normalized reference population."""

        return self.provenance.population

    @property
    def precision(self) -> int:
        """Return the number of decimal places represented by the bounds."""

        return self.provenance.precision

    @property
    def source_fingerprint(self) -> str:
        """Return the source or instrument fingerprint."""

        return self.provenance.source_fingerprint

    @property
    def locale(self) -> str | None:
        """Return the normalized source locale, when explicitly supplied."""

        return self.provenance.locale

    @property
    def bounds_key(self) -> tuple[float | None, float | None, bool, bool]:
        """Return bounds and inclusivity in a hashable comparison shape."""

        return self.low, self.high, self.low_inclusive, self.high_inclusive

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready range with nested and convenient provenance."""

        provenance = self.provenance.to_dict()
        return {
            "schema_version": REFERENCE_RANGE_SCHEMA_VERSION,
            "analyte": self.analyte,
            "low": self.low,
            "high": self.high,
            "low_inclusive": self.low_inclusive,
            "high_inclusive": self.high_inclusive,
            "unit": provenance["unit"],
            "population": provenance["population"],
            "precision": provenance["precision"],
            "source_fingerprint": provenance["source_fingerprint"],
            "locale": provenance["locale"],
            "provenance": provenance,
            "advisory": LAB_REFERENCE_RANGE_ADVISORY,
        }


# This alias keeps the module convenient for callers that use the issue's
# domain term while leaving the existing ``openmed.clinical.lab_values``
# ``ReferenceRange`` TypedDict unchanged.
ReferenceRange = LabReferenceRange
SyntheticReferenceRange = LabReferenceRange


def build_reference_range(
    analyte: str,
    low: object | None,
    high: object | None,
    *,
    unit: str,
    population: str,
    precision: int,
    source: object | None = None,
    source_fingerprint: str | None = None,
    locale: str | None = None,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> LabReferenceRange:
    """Build a typed range from local metadata without retaining raw source.

    Provide either ``source`` (which is fingerprinted locally) or an already
    computed ``source_fingerprint``. Supplying neither is rejected because a
    range without source provenance must not be used for comparison.
    """

    if source is not None and source_fingerprint is not None:
        raise ValueError("provide source or source_fingerprint, not both")
    if source is None and source_fingerprint is None:
        raise ValueError("reference range source provenance is required")
    fingerprint = (
        fingerprint_source(source)
        if source is not None
        else _normalized_fingerprint(source_fingerprint)
    )
    provenance = ReferenceRangeProvenance(
        unit=unit,
        population=population,
        precision=precision,
        source_fingerprint=fingerprint,
        locale=locale,
    )
    return LabReferenceRange(
        analyte=analyte,
        low=low,
        high=high,
        provenance=provenance,
        low_inclusive=low_inclusive,
        high_inclusive=high_inclusive,
    )


create_reference_range = build_reference_range


def reference_range_from_mapping(
    payload: Mapping[str, object],
    *,
    source: object | None = None,
) -> LabReferenceRange:
    """Coerce a local JSON-like range record into the typed representation."""

    if not isinstance(payload, Mapping):
        raise TypeError("reference range payload must be a mapping")
    nested = payload.get("range") or payload.get("reference_range")
    range_payload = nested if isinstance(nested, Mapping) else payload
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, Mapping):
        provenance_payload = payload

    raw_source = source if source is not None else provenance_payload.get("source")
    raw_fingerprint = provenance_payload.get("source_fingerprint")
    if raw_source is not None and raw_fingerprint is not None:
        raise ValueError("reference range source metadata is ambiguous")
    if raw_source is not None:
        raw_fingerprint = fingerprint_source(raw_source)

    required = ("unit", "population", "precision", "source_fingerprint")
    if any(provenance_payload.get(key) is None for key in required):
        raise ValueError("reference range provenance is incomplete")
    analyte = payload.get("analyte") or payload.get("test") or payload.get("name")
    if analyte is None:
        raise ValueError("reference range analyte is required")
    return build_reference_range(
        analyte=str(analyte),
        low=range_payload.get("low"),
        high=range_payload.get("high"),
        unit=str(provenance_payload["unit"]),
        population=str(provenance_payload["population"]),
        precision=provenance_payload["precision"],  # type: ignore[arg-type]
        source_fingerprint=str(raw_fingerprint),
        locale=(
            None
            if provenance_payload.get("locale") is None
            else str(provenance_payload["locale"])
        ),
        low_inclusive=bool(range_payload.get("low_inclusive", True)),
        high_inclusive=bool(range_payload.get("high_inclusive", True)),
    )


coerce_reference_range = reference_range_from_mapping


@dataclass(frozen=True)
class ReferenceRangeResolution:
    """Safe result of exact reference-range selection or comparison."""

    status: ReferenceRangeStatus
    reference_range: LabReferenceRange | None
    reason: str
    candidate_count: int = 0
    advisory: str = LAB_REFERENCE_RANGE_ADVISORY

    @property
    def state(self) -> ReferenceRangeState:
        """Return the string state for simple serializers."""

        return self.status.value  # type: ignore[return-value]

    @property
    def is_known(self) -> bool:
        """Return whether a range was selected without inference."""

        return self.status is ReferenceRangeStatus.KNOWN

    @property
    def is_unknown(self) -> bool:
        """Return whether no exact range was available."""

        return self.status is ReferenceRangeStatus.UNKNOWN

    @property
    def is_conflict(self) -> bool:
        """Return whether explicit candidates disagree."""

        return self.status is ReferenceRangeStatus.CONFLICT

    @property
    def is_conflicting(self) -> bool:
        """Alias for :attr:`is_conflict`."""

        return self.is_conflict

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic resolution without candidate raw text."""

        return {
            "schema_version": REFERENCE_RANGE_SCHEMA_VERSION,
            "status": self.status.value,
            "state": self.state,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "reference_range": (
                None if self.reference_range is None else self.reference_range.to_dict()
            ),
            "advisory": self.advisory,
        }


def _coerce_candidates(
    ranges: Iterable[LabReferenceRange | Mapping[str, object]]
    | LabReferenceRange
    | Mapping[str, object],
) -> tuple[LabReferenceRange, ...]:
    if isinstance(ranges, LabReferenceRange):
        values: Iterable[LabReferenceRange | Mapping[str, object]] = (ranges,)
    elif isinstance(ranges, Mapping):
        values = (ranges,)
    else:
        values = ranges
    coerced = tuple(
        value
        if isinstance(value, LabReferenceRange)
        else reference_range_from_mapping(value)
        for value in values
    )
    return tuple(sorted(coerced, key=_range_sort_key))


def _range_sort_key(
    value: LabReferenceRange,
) -> tuple[object, ...]:
    return (
        value.analyte_key,
        value.provenance.identity_key,
        value.bounds_key,
    )


def _coerce_target_provenance(
    provenance: ReferenceRangeProvenance | Mapping[str, object] | None,
    *,
    unit: str | None,
    population: str | None,
    precision: int | None,
    source_fingerprint: str | None,
    locale: str | None,
) -> ReferenceRangeProvenance | None:
    explicit = (unit, population, precision, source_fingerprint, locale)
    if provenance is not None and any(value is not None for value in explicit):
        raise ValueError("provide provenance or individual provenance fields, not both")
    if isinstance(provenance, ReferenceRangeProvenance):
        return provenance
    if isinstance(provenance, Mapping):
        return ReferenceRangeProvenance(
            unit=provenance.get("unit"),  # type: ignore[arg-type]
            population=provenance.get("population"),  # type: ignore[arg-type]
            precision=provenance.get("precision"),  # type: ignore[arg-type]
            source_fingerprint=provenance.get("source_fingerprint"),  # type: ignore[arg-type]
            locale=provenance.get("locale"),  # type: ignore[arg-type]
        )
    if provenance is not None:
        raise TypeError("provenance must be typed metadata or a mapping")
    if all(value is None for value in explicit):
        return None
    if any(
        value is None for value in (unit, population, precision, source_fingerprint)
    ):
        return None
    return ReferenceRangeProvenance(
        unit=unit,
        population=population,
        precision=precision,
        source_fingerprint=source_fingerprint,
        locale=locale,
    )


def _resolution(
    status: ReferenceRangeStatus,
    reason: str,
    selected: LabReferenceRange | None,
    candidates: tuple[LabReferenceRange, ...],
) -> ReferenceRangeResolution:
    return ReferenceRangeResolution(
        status=status,
        reference_range=selected,
        reason=reason,
        candidate_count=len(candidates),
    )


def resolve_reference_range(
    ranges: Iterable[LabReferenceRange | Mapping[str, object]]
    | LabReferenceRange
    | Mapping[str, object],
    *,
    analyte: str | None = None,
    provenance: ReferenceRangeProvenance | Mapping[str, object] | None = None,
    unit: str | None = None,
    population: str | None = None,
    precision: int | None = None,
    source_fingerprint: str | None = None,
    locale: str | None = None,
) -> ReferenceRangeResolution:
    """Select a range only when its explicit context is unambiguous.

    If ``provenance`` (or its individual fields) is supplied, all five
    comparable fields must match exactly. With no target provenance, a single
    candidate can be returned because its own provenance is explicit; multiple
    distinct candidates return ``conflict`` rather than selecting by order.
    Unit conversion, instrument fallback, population fallback, and locale
    fallback are intentionally unsupported.
    """

    candidates = _coerce_candidates(ranges)
    target = _coerce_target_provenance(
        provenance,
        unit=unit,
        population=population,
        precision=precision,
        source_fingerprint=source_fingerprint,
        locale=locale,
    )
    if analyte is not None:
        analyte_key = _normalized_text(analyte, "analyte").casefold()
        candidates = tuple(
            item for item in candidates if item.analyte_key == analyte_key
        )
    elif len({item.analyte_key for item in candidates}) > 1:
        return _resolution(
            ReferenceRangeStatus.UNKNOWN,
            "analyte context is required when candidates contain multiple analytes",
            None,
            candidates,
        )

    if not candidates:
        return _resolution(
            ReferenceRangeStatus.UNKNOWN,
            "no reference range matched the explicit context",
            None,
            candidates,
        )

    if target is not None:
        exact = tuple(item for item in candidates if item.provenance.matches(target))
        if not exact:
            return _resolution(
                ReferenceRangeStatus.UNKNOWN,
                "no reference range matched the explicit provenance",
                None,
                candidates,
            )
        if len({(item.bounds_key, item.provenance.identity_key) for item in exact}) > 1:
            return _resolution(
                ReferenceRangeStatus.CONFLICT,
                "matching provenance has conflicting range bounds",
                None,
                exact,
            )
        return _resolution(
            ReferenceRangeStatus.KNOWN,
            "exact provenance match",
            exact[0],
            exact,
        )

    signatures = {
        (item.provenance.identity_key, item.bounds_key) for item in candidates
    }
    if len(signatures) > 1:
        return _resolution(
            ReferenceRangeStatus.CONFLICT,
            "multiple explicit ranges have conflicting provenance or bounds",
            None,
            candidates,
        )
    return _resolution(
        ReferenceRangeStatus.KNOWN,
        "single explicit range",
        candidates[0],
        candidates,
    )


def compare_reference_ranges(
    left: LabReferenceRange | Mapping[str, object],
    right: LabReferenceRange | Mapping[str, object],
) -> ReferenceRangeResolution:
    """Compare two ranges without converting or inferring their context."""

    left_range = (
        left
        if isinstance(left, LabReferenceRange)
        else reference_range_from_mapping(left)
    )
    right_range = (
        right
        if isinstance(right, LabReferenceRange)
        else reference_range_from_mapping(right)
    )
    candidates = tuple(sorted((left_range, right_range), key=_range_sort_key))
    if left_range.analyte_key != right_range.analyte_key:
        return _resolution(
            ReferenceRangeStatus.UNKNOWN,
            "analytes do not have matching explicit identities",
            None,
            candidates,
        )
    if left_range.provenance.locale != right_range.provenance.locale:
        return _resolution(
            ReferenceRangeStatus.UNKNOWN,
            "locale provenance differs and cannot be inferred",
            None,
            candidates,
        )
    if left_range.provenance.measurement_key != right_range.provenance.measurement_key:
        return _resolution(
            ReferenceRangeStatus.UNKNOWN,
            "unit, population, or precision provenance differs",
            None,
            candidates,
        )
    if (
        left_range.provenance.source_fingerprint
        != right_range.provenance.source_fingerprint
    ):
        return _resolution(
            ReferenceRangeStatus.CONFLICT,
            "source provenance identifies different range instruments",
            None,
            candidates,
        )
    if left_range.bounds_key != right_range.bounds_key:
        return _resolution(
            ReferenceRangeStatus.CONFLICT,
            "matching provenance has conflicting range bounds",
            None,
            candidates,
        )
    return _resolution(
        ReferenceRangeStatus.KNOWN,
        "matching provenance and bounds",
        left_range,
        candidates,
    )


resolve_lab_reference_range = resolve_reference_range
compare_lab_reference_ranges = compare_reference_ranges


__all__ = [
    "LAB_REFERENCE_RANGE_ADVISORY",
    "REFERENCE_RANGE_SCHEMA_VERSION",
    "LabReferenceRange",
    "ReferenceRange",
    "ReferenceRangeProvenance",
    "ReferenceRangeResolution",
    "ReferenceRangeState",
    "ReferenceRangeStatus",
    "SyntheticReferenceRange",
    "build_reference_range",
    "coerce_reference_range",
    "compare_lab_reference_ranges",
    "compare_reference_ranges",
    "create_reference_range",
    "fingerprint_source",
    "reference_range_from_mapping",
    "resolve_lab_reference_range",
    "resolve_reference_range",
    "source_fingerprint",
]
