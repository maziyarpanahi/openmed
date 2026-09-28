"""Deterministic unit-dimension checks for quantitative relation candidates.

This module checks the dimensions of caller-supplied units without converting
their numeric values.  It is intentionally a review gate: incompatible,
ambiguous, missing, or otherwise unknown units are returned as explicit review
findings instead of being guessed or silently normalized.

The checker builds on the local UCUM-subset parser in
``openmed.clinical.units``.  It does not load terminology services, contact a
network service, or retain candidate values and source text in its results.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import wraps
from itertools import islice
from types import MappingProxyType
from typing import Any, Literal

from openmed.clinical.units import parse_measurement, parse_unit

UNIT_COMPATIBILITY_SCHEMA_VERSION = 1
UNIT_COMPATIBILITY_ADVISORY = (
    "Unit compatibility is a deterministic dimension check only. It does not "
    "convert values, establish clinical equivalence, or replace clinician review."
)

UNIT_COMPATIBLE = "compatible"
UNIT_INCOMPATIBLE = "incompatible"
UNIT_UNKNOWN = "unknown"
UnitCompatibilityStatus = Literal[
    "compatible",
    "incompatible",
    "unknown",
]

QUANTITATIVE_RELATION_KINDS = (
    "dose",
    "rate",
    "concentration",
    "laboratory",
)
QuantitativeRelationKind = Literal[
    "dose",
    "rate",
    "concentration",
    "laboratory",
]

_KNOWN_DIMENSION_NAMES = frozenset(
    {
        "amount",
        "catalytic_activity",
        "entity",
        "equivalent_amount",
        "length",
        "mass",
        "pressure",
        "temperature",
        "time",
        "volume",
    }
)
_KNOWN_RELATION_TYPES = frozenset(
    {
        "concentration",
        "dose",
        "drug_to_dose",
        "drug_to_rate",
        "drug_to_strength",
        "drug_to_concentration",
        "laboratory",
        "lab_result",
        "lab_result_to_value",
        "lab_value_to_unit",
        "rate",
        "laboratory_result",
        "value_to_unit",
    }
)
_MISSING = object()
_SAFE_UNIT_LABEL = re.compile(r"^[A-Za-z0-9%*/^()._\[\]-]{1,32}$")


_REASONS = frozenset(
    {
        "unsupported_relation_kind",
        "invalid_expected_dimension",
        "missing_unit",
        "unknown_left_unit",
        "missing_comparison_unit",
        "unknown_right_unit",
        "expected_dimension_mismatch",
        "dimension_mismatch",
        "relation_dimension_mismatch",
        "compatible_dimensions",
    }
)


def _safe_boundary(function):
    """Do not retain input-bearing exception chains."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TypeError:
            error = TypeError("candidates must be an iterable of relation candidates")
        except Exception:
            error = ValueError("invalid unit compatibility input")
        raise error

    return wrapped


def _bounded_items(value):
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("candidates must be an iterable of relation candidates")
    items = tuple(islice(iter(value), 4097))
    if len(items) > 4096:
        raise ValueError("unit collection exceeds the supported limit")
    return items


@dataclass(frozen=True)
class UnitCompatibilityResult:
    """Value-free result for one quantitative unit-dimension decision.

    ``left_unit`` and ``right_unit`` contain only normalized unit labels when
    those labels were successfully parsed. Numeric values, source text, and
    arbitrary candidate metadata are deliberately absent. ``right_unit`` is
    optional for a single-unit category check; relation-candidate adapters use
    ``candidate_index`` and ``comparison_requested`` to distinguish a missing
    comparison side from that valid single-unit form.
    """

    relation_kind: str
    status: UnitCompatibilityStatus
    review_required: bool
    reason: str
    left_unit: str | None = None
    right_unit: str | None = None
    left_dimension: dict[str, int] | None = None
    right_dimension: dict[str, int] | None = None
    relation_type: str | None = None
    head_offset: tuple[int, int] | None = None
    tail_offset: tuple[int, int] | None = None
    candidate_index: int | None = None
    comparison_requested: bool = False
    advisory: str = UNIT_COMPATIBILITY_ADVISORY

    @_safe_boundary
    def __post_init__(self) -> None:
        """Normalize safe fields and enforce the review-status contract."""

        if self.relation_kind not in (*QUANTITATIVE_RELATION_KINDS, "unknown"):
            raise ValueError("relation_kind must be a supported quantitative kind")
        if self.status not in {
            UNIT_COMPATIBLE,
            UNIT_INCOMPATIBLE,
            UNIT_UNKNOWN,
        }:
            raise ValueError("unsupported unit compatibility status")
        if (
            type(self.review_required) is not bool
            or type(self.comparison_requested) is not bool
        ):
            raise ValueError("review fields must be booleans")
        if self.reason not in _REASONS or self.advisory != UNIT_COMPATIBILITY_ADVISORY:
            raise ValueError("invalid unit metadata")
        expected_review = self.status != UNIT_COMPATIBLE
        if self.review_required != expected_review:
            raise ValueError("review_required must match the compatibility status")
        if self.candidate_index is not None and (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")

        object.__setattr__(self, "left_dimension", _safe_dimension(self.left_dimension))
        object.__setattr__(
            self,
            "right_dimension",
            _safe_dimension(self.right_dimension),
        )
        object.__setattr__(self, "left_unit", _safe_unit_label(self.left_unit))
        object.__setattr__(self, "right_unit", _safe_unit_label(self.right_unit))
        object.__setattr__(
            self,
            "relation_type",
            _safe_relation_type(self.relation_type),
        )
        for side in ("left", "right"):
            unit = getattr(self, side + "_unit")
            dimension = getattr(self, side + "_dimension")
            if unit is not None:
                parsed = _parse_safe_unit(unit, language=None)
                if parsed is None or parsed["dimension"] != dimension:
                    raise ValueError("unit and dimension must agree")
            elif dimension is not None:
                raise ValueError("dimension requires a unit")
        if self.status == UNIT_COMPATIBLE:
            if (
                self.reason != "compatible_dimensions"
                or self.left_unit is None
                or self.relation_kind == "unknown"
                or not _dimension_matches_kind(self.left_dimension, self.relation_kind)
                or (self.comparison_requested and self.right_unit is None)
                or (
                    self.right_unit is not None
                    and self.left_dimension != self.right_dimension
                )
            ):
                raise ValueError("compatible status requires matching known dimensions")
        elif self.reason == "compatible_dimensions":
            raise ValueError("reason does not match status")
        object.__setattr__(self, "head_offset", _safe_offset(self.head_offset))
        object.__setattr__(self, "tail_offset", _safe_offset(self.tail_offset))

    @property
    def compatible(self) -> bool:
        """Return whether the candidate passed the unit-dimension gate."""

        return self.status == UNIT_COMPATIBLE

    @property
    def flagged_for_review(self) -> bool:
        """Return whether a human review is required before using the relation."""

        return self.review_required

    @property
    def decision(self) -> Literal["accept", "review"]:
        """Return the value-free operational disposition for the result."""

        return "accept" if self.compatible else "review"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report entry without values or source text."""

        payload: dict[str, Any] = {
            "schema_version": UNIT_COMPATIBILITY_SCHEMA_VERSION,
            "relation_kind": self.relation_kind,
            "status": self.status,
            "decision": self.decision,
            "review_required": self.review_required,
            "reason": self.reason,
            "left_unit": self.left_unit,
            "right_unit": self.right_unit,
            "left_dimension": _dimension_dict(self.left_dimension),
            "right_dimension": _dimension_dict(self.right_dimension),
            "comparison_requested": self.comparison_requested,
            "advisory": self.advisory,
        }
        if self.relation_type is not None:
            payload["relation_type"] = self.relation_type
        if self.head_offset is not None:
            payload["head_offset"] = list(self.head_offset)
        if self.tail_offset is not None:
            payload["tail_offset"] = list(self.tail_offset)
        if self.candidate_index is not None:
            payload["candidate_index"] = self.candidate_index
        return payload


@dataclass(frozen=True)
class UnitCompatibilityReport:
    """Deterministic batch report for quantitative relation candidates."""

    results: tuple[UnitCompatibilityResult, ...]
    advisory: str = UNIT_COMPATIBILITY_ADVISORY
    schema_version: int = UNIT_COMPATIBILITY_SCHEMA_VERSION

    @_safe_boundary
    def __post_init__(self) -> None:
        """Freeze the report result sequence and validate its schema version."""

        if (
            type(self.schema_version) is not int
            or self.schema_version != UNIT_COMPATIBILITY_SCHEMA_VERSION
        ):
            raise ValueError("unsupported unit compatibility report schema version")
        if self.advisory != UNIT_COMPATIBILITY_ADVISORY:
            raise ValueError("invalid unit advisory")
        values = _bounded_items(self.results)
        if any(type(value) is not UnitCompatibilityResult for value in values):
            raise ValueError("report requires typed unit results")
        object.__setattr__(self, "results", tuple(replace(value) for value in values))

    def __iter__(self):
        """Iterate over result entries in candidate order."""

        return iter(self.results)

    def __len__(self) -> int:
        """Return the number of checked candidates."""

        return len(self.results)

    def __getitem__(self, index):
        """Return one result or a tuple slice."""

        return self.results[index]

    @property
    def compatible(self) -> tuple[UnitCompatibilityResult, ...]:
        """Return candidates whose unit dimensions passed the gate."""

        return tuple(result for result in self.results if result.compatible)

    @property
    def review_required(self) -> tuple[UnitCompatibilityResult, ...]:
        """Return candidates withheld for incompatible or unknown units."""

        return tuple(result for result in self.results if result.review_required)

    @property
    def has_review_findings(self) -> bool:
        """Return whether any candidate requires human review."""

        return bool(self.review_required)

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate counts and value-free result entries."""

        return {
            "schema_version": self.schema_version,
            "advisory": self.advisory,
            "counts": {
                "total": len(self.results),
                "compatible": len(self.compatible),
                "review_required": len(self.review_required),
            },
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self) -> str:
        """Return canonical JSON for the value-free compatibility report."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class _UnitInputs:
    left: object | None
    right: object | None
    comparison_requested: bool


@_safe_boundary
def check_unit_compatibility(
    left_unit: object | None,
    right_unit: object | None = None,
    *,
    relation_kind: str = "laboratory",
    expected_dimension: Mapping[str, int] | None = None,
    require_pair: bool = False,
    language: object | None = None,
    relation_type: str | None = None,
    head_offset: tuple[int, int] | None = None,
    tail_offset: tuple[int, int] | None = None,
    candidate_index: int | None = None,
) -> UnitCompatibilityResult:
    """Check one or two units without converting any numeric value.

    Args:
        left_unit: First unit string, or ``None`` when it was not extracted.
        right_unit: Optional unit to compare against the first unit.
        relation_kind: One of ``dose``, ``rate``, ``concentration``, or
            ``laboratory``. The laboratory kind accepts any known dimension.
        expected_dimension: Optional exact dimension mapping supplied by a
            local caller. Unknown dimension names fail closed.
        require_pair: Require both sides when a relation explicitly joins two
            quantitative attributes.
        language: Optional local language code passed to the unit parser.
        relation_type: Allowlisted relation label retained as safe metadata.
        head_offset: Optional source offsets for value-free review routing.
        tail_offset: Optional source offsets for value-free review routing.
        candidate_index: Optional non-sensitive sequence index.

    Returns:
        A deterministic result. Incompatible and unknown outcomes set
        ``review_required`` and never include a numeric value.
    """

    if type(require_pair) is not bool:
        raise ValueError("require_pair must be a boolean")
    kind = _normalize_relation_kind(relation_kind)
    if kind is None:
        return _result(
            relation_kind="unknown",
            status=UNIT_UNKNOWN,
            reason="unsupported_relation_kind",
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )

    expected = _normalize_dimension(expected_dimension)
    if expected_dimension is not None and expected is None:
        return _result(
            relation_kind=kind,
            status=UNIT_UNKNOWN,
            reason="invalid_expected_dimension",
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )

    left = _parse_safe_unit(left_unit, language=language)
    right = _parse_safe_unit(right_unit, language=language)
    if left is None:
        return _result(
            relation_kind=kind,
            status=UNIT_UNKNOWN,
            reason="missing_unit" if left_unit is None else "unknown_left_unit",
            right=right,
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )
    if require_pair and right_unit is None:
        return _result(
            relation_kind=kind,
            status=UNIT_UNKNOWN,
            reason="missing_comparison_unit",
            left=left,
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=True,
        )
    if right_unit is not None and right is None:
        return _result(
            relation_kind=kind,
            status=UNIT_UNKNOWN,
            reason="unknown_right_unit",
            left=left,
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )

    left_dimension = left["dimension"]
    right_dimension = right["dimension"] if right is not None else None
    if expected is not None and (
        left_dimension != expected
        or (right_dimension is not None and right_dimension != expected)
    ):
        return _result(
            relation_kind=kind,
            status=UNIT_INCOMPATIBLE,
            reason="expected_dimension_mismatch",
            left=left,
            right=right,
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )

    if right_dimension is not None and left_dimension != right_dimension:
        return _result(
            relation_kind=kind,
            status=UNIT_INCOMPATIBLE,
            reason="dimension_mismatch",
            left=left,
            right=right,
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )

    if not _dimension_matches_kind(left_dimension, kind) or (
        right_dimension is not None
        and not _dimension_matches_kind(right_dimension, kind)
    ):
        return _result(
            relation_kind=kind,
            status=UNIT_INCOMPATIBLE,
            reason="relation_dimension_mismatch",
            left=left,
            right=right,
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=require_pair,
        )

    return _result(
        relation_kind=kind,
        status=UNIT_COMPATIBLE,
        reason="compatible_dimensions",
        left=left,
        right=right,
        relation_type=relation_type,
        head_offset=head_offset,
        tail_offset=tail_offset,
        candidate_index=candidate_index,
        comparison_requested=require_pair,
    )


@_safe_boundary
def validate_quantitative_relation(
    candidate: object,
    *,
    relation_kind: str | None = None,
    candidate_index: int | None = None,
) -> UnitCompatibilityResult:
    """Validate units extracted from one quantitative relation candidate.

    The adapter accepts mappings and the repository's relation dataclasses.
    It looks only at unit-bearing fields, normalized measurement fields, and
    endpoint offsets. Candidate values, source text, confidence metadata, and
    arbitrary fields are never copied into the result.
    """

    data = _candidate_data(candidate)
    if relation_kind is not None:
        kind = _normalize_relation_kind(relation_kind)
    else:
        kind = _infer_relation_kind(data)
        if kind is None:
            kind = "laboratory" if _has_unit_field(data) else "unknown"

    relation_type = _safe_relation_type(_first_value(data, "relation_type", "type"))
    head_offset = _offset_from(_first_value(data, "head", "source", "left"))
    tail_offset = _offset_from(
        _first_value(data, "attribute", "tail", "target", "right")
    )
    language = _first_value(data, "language", "source_language")
    inputs = _extract_unit_inputs(data, language=language)
    if kind == "unknown":
        return _result(
            relation_kind="unknown",
            status=UNIT_UNKNOWN,
            reason="unsupported_relation_kind",
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=inputs.comparison_requested,
        )
    if inputs.left is None and inputs.right is None:
        return _result(
            relation_kind=kind,
            status=UNIT_UNKNOWN,
            reason="missing_unit",
            relation_type=relation_type,
            head_offset=head_offset,
            tail_offset=tail_offset,
            candidate_index=candidate_index,
            comparison_requested=inputs.comparison_requested,
        )

    return check_unit_compatibility(
        inputs.left,
        inputs.right,
        relation_kind=kind,
        require_pair=inputs.comparison_requested,
        language=language,
        relation_type=relation_type,
        head_offset=head_offset,
        tail_offset=tail_offset,
        candidate_index=candidate_index,
    )


@_safe_boundary
def validate_quantitative_relations(
    candidates: Iterable[object],
) -> UnitCompatibilityReport:
    """Validate candidates in input order and return a value-free report.

    Each result carries its non-sensitive sequence index, so review systems can
    route a finding back to the caller without copying the candidate itself.
    The function consumes only the supplied iterable and performs no I/O.
    """

    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be an iterable of relation candidates")
    try:
        values = _bounded_items(candidates)
    except TypeError as exc:
        raise TypeError(
            "candidates must be an iterable of relation candidates"
        ) from exc
    return UnitCompatibilityReport(
        tuple(
            validate_quantitative_relation(candidate, candidate_index=index)
            for index, candidate in enumerate(values)
        )
    )


def _candidate_data(candidate: object) -> Mapping[str, object]:
    if isinstance(candidate, Mapping):
        return candidate
    if candidate is None or isinstance(candidate, (str, bytes, int, float, bool)):
        return {}
    try:
        data = vars(candidate)
    except TypeError:
        return {}
    return data if isinstance(data, Mapping) else {}


def _first_value(data: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = data.get(key, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return None


def _has_unit_field(data: Mapping[str, object]) -> bool:
    keys = {
        "unit",
        "units",
        "canonical_unit",
        "value_unit",
        "measurement_unit",
        "dose_unit",
        "rate_unit",
        "concentration_unit",
        "result_unit",
        "reference_unit",
        "expected_unit",
        "source_unit",
        "target_unit",
        "left_unit",
        "right_unit",
        "from_unit",
        "to_unit",
    }
    return any(key in data for key in keys)


def _infer_relation_kind(data: Mapping[str, object]) -> str | None:
    for key in ("relation_kind", "quantitative_kind", "kind"):
        if key in data:
            return _normalize_relation_kind(data[key]) or "unknown"
    for key in (
        "relation_kind",
        "quantitative_kind",
        "kind",
        "attribute_type",
        "relation_type",
        "type",
    ):
        value = data.get(key)
        normalized = _normalize_relation_kind(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_relation_kind(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 256:
        return None
    value = str.__str__(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized in {"dose", "dosage", "strength", "drug_to_dose", "drug_to_strength"}:
        return "dose"
    if normalized in {"rate", "dose_rate", "frequency", "flow_rate"}:
        return "rate"
    if normalized in {"concentration", "density", "mass_concentration"}:
        return "concentration"
    if normalized in {
        "laboratory",
        "lab",
        "lab_result",
        "lab_value",
        "lab_result_to_value",
        "lab_value_to_unit",
        "value_to_unit",
        "observation",
    }:
        return "laboratory"
    if normalized == "drug_to_concentration":
        return "concentration"
    if normalized == "drug_to_rate":
        return "rate"
    if normalized == "laboratory_result":
        return "laboratory"
    return None


def _extract_unit_inputs(
    data: Mapping[str, object], *, language: object | None = None
) -> _UnitInputs:
    primary_keys = (
        "unit",
        "units",
        "canonical_unit",
        "measurement_unit",
        "dose_unit",
        "rate_unit",
        "concentration_unit",
        "result_unit",
    )
    declarations = [
        data[key] for key in primary_keys if key in data and data[key] is not None
    ]
    if any(value != declarations[0] for value in declarations[1:]):
        raise ValueError("conflicting unit aliases")
    if declarations and any(
        left in data or right in data
        for left, right in (
            ("left", "right"),
            ("source", "target"),
            ("observed", "expected"),
        )
    ):
        raise ValueError("conflicting top-level and endpoint units")
    pair_keys = (
        ("left_unit", "right_unit"),
        ("source_unit", "target_unit"),
        ("from_unit", "to_unit"),
        ("value_unit", "reference_unit"),
        ("result_unit", "reference_unit"),
        ("observed_unit", "expected_unit"),
        ("unit", "reference_unit"),
        ("unit", "expected_unit"),
    )
    for left_key, right_key in pair_keys:
        left = data.get(left_key)
        right = data.get(right_key)
        pair_is_explicit = right_key in data or left_key != "unit"
        if pair_is_explicit and (left is not None or right is not None):
            return _UnitInputs(
                left=_unit_from_value(left, language=language),
                right=_unit_from_value(right, language=language),
                comparison_requested=True,
            )

    left = _unit_from_value(
        _first_value(
            data,
            "unit",
            "units",
            "canonical_unit",
            "value_unit",
            "measurement_unit",
            "dose_unit",
            "rate_unit",
            "concentration_unit",
            "result_unit",
        ),
        language=language,
    )
    if left is not None:
        return _UnitInputs(left=left, right=None, comparison_requested=False)

    for left_key, right_key in (
        ("left", "right"),
        ("source", "target"),
        ("value", "reference"),
        ("observed", "expected"),
    ):
        if left_key not in data and right_key not in data:
            continue
        nested_left = _unit_from_value(data.get(left_key), language=language)
        nested_right = _unit_from_value(data.get(right_key), language=language)
        return _UnitInputs(
            left=nested_left,
            right=nested_right,
            comparison_requested=True,
        )

    for key in (
        "normalized",
        "normalized_value",
        "measurement",
        "value",
        "attribute",
        "tail",
        "right",
        "target",
    ):
        nested = data.get(key)
        nested_unit = _unit_from_value(nested, language=language)
        if nested_unit is not None:
            return _UnitInputs(
                left=nested_unit,
                right=None,
                comparison_requested=False,
            )

    for key in ("head", "source", "left"):
        nested_unit = _unit_from_value(data.get(key), language=language)
        if nested_unit is not None:
            return _UnitInputs(
                left=nested_unit,
                right=None,
                comparison_requested=False,
            )
    return _UnitInputs(left=None, right=None, comparison_requested=False)


def _unit_from_value(
    value: object, *, language: object | None = None, depth: int = 0
) -> object | None:
    if depth > 32:
        raise ValueError("unit nesting exceeds the supported limit")
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) > 4096:
            return "__unknown_unit__"
        parsed = parse_unit(value, language=language)
        if parsed["status"] == "ok":
            return value
        measurement = parse_measurement(value, language=language)
        if measurement["status"] == "ok":
            provenance = measurement.get("provenance")
            if isinstance(provenance, Mapping):
                input_unit = provenance.get("input_unit")
                if isinstance(input_unit, str):
                    return input_unit
        return None
    if isinstance(value, Mapping):
        for key in (
            "unit",
            "units",
            "value_unit",
            "measurement_unit",
            "dose_unit",
            "rate_unit",
            "concentration_unit",
            "result_unit",
            "canonical_unit",
            "reference_unit",
            "expected_unit",
        ):
            if key in value and value[key] is not None:
                unit = _unit_from_value(value[key], language=language, depth=depth + 1)
                return unit if unit is not None else "__unknown_unit__"
        for key in ("value", "amount", "dose", "magnitude", "text"):
            if key in value:
                unit = _unit_from_value(value[key], language=language)
                if unit is not None:
                    return unit
        return None
    try:
        unit = getattr(value, "unit", _MISSING)
    except Exception:
        unit = _MISSING
    if unit is not _MISSING and unit is not None:
        parsed_unit = _unit_from_value(unit, language=language, depth=depth + 1)
        return parsed_unit if parsed_unit is not None else "__unknown_unit__"
    for key in ("normalized", "measurement", "value", "amount", "dose", "text"):
        try:
            nested = getattr(value, key, _MISSING)
        except Exception:
            nested = _MISSING
        if nested is not _MISSING:
            parsed_nested = _unit_from_value(nested, language=language, depth=depth + 1)
            if parsed_nested is not None:
                return parsed_nested
    return None


def _parse_safe_unit(
    unit: object | None,
    *,
    language: object | None,
) -> dict[str, Any] | None:
    if not isinstance(unit, str) or len(unit) > 256:
        return None
    parsed = parse_unit(unit, language=language)
    if parsed.get("status") != "ok":
        return None
    dimension = _normalize_dimension(parsed.get("dimension"))
    normalized = parsed.get("unit")
    if dimension is None or not isinstance(normalized, str):
        return None
    return {"unit": normalized, "dimension": dimension}


def _dimension_matches_kind(dimension: Mapping[str, int], kind: str) -> bool:
    if kind == "laboratory":
        return True
    if kind == "dose":
        allowed = {
            "amount",
            "catalytic_activity",
            "entity",
            "equivalent_amount",
            "mass",
            "volume",
        }
        return all(
            name in allowed
            and not (name == "volume" and exponent < 0)
            and not (
                exponent < 0
                and name
                in {
                    "amount",
                    "catalytic_activity",
                    "equivalent_amount",
                    "mass",
                    "volume",
                }
            )
            for name, exponent in dimension.items()
        )
    if kind == "rate":
        if dimension.get("time") != -1:
            return False
        return all(
            name
            in {
                "amount",
                "catalytic_activity",
                "entity",
                "equivalent_amount",
                "mass",
                "volume",
                "time",
            }
            for name in dimension
        )
    if kind == "concentration":
        if not dimension:
            return True
        if dimension.get("volume") != -1 or "time" in dimension:
            return False
        return all(
            name
            in {
                "amount",
                "catalytic_activity",
                "entity",
                "equivalent_amount",
                "mass",
                "volume",
            }
            for name in dimension
        )
    return False


def _result(
    *,
    relation_kind: str,
    status: UnitCompatibilityStatus,
    reason: str,
    left: Mapping[str, Any] | None = None,
    right: Mapping[str, Any] | None = None,
    relation_type: str | None = None,
    head_offset: tuple[int, int] | None = None,
    tail_offset: tuple[int, int] | None = None,
    candidate_index: int | None = None,
    comparison_requested: bool = False,
) -> UnitCompatibilityResult:
    return UnitCompatibilityResult(
        relation_kind=relation_kind,
        status=status,
        review_required=status != UNIT_COMPATIBLE,
        reason=reason,
        left_unit=left.get("unit") if left is not None else None,
        right_unit=right.get("unit") if right is not None else None,
        left_dimension=left.get("dimension") if left is not None else None,
        right_dimension=right.get("dimension") if right is not None else None,
        relation_type=relation_type,
        head_offset=head_offset,
        tail_offset=tail_offset,
        candidate_index=candidate_index,
        comparison_requested=comparison_requested,
    )


def _safe_dimension(value: Mapping[str, int] | None) -> dict[str, int] | None:
    normalized = _normalize_dimension(value)
    return None if normalized is None else MappingProxyType(normalized)


def _normalize_dimension(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, int] = {}
    if len(value) > len(_KNOWN_DIMENSION_NAMES):
        return None
    for name, exponent in value.items():
        if not isinstance(name, str) or name not in _KNOWN_DIMENSION_NAMES:
            return None
        if type(exponent) is not int or abs(exponent) > 32:
            return None
        if exponent:
            normalized[name] = exponent
    return dict(sorted(normalized.items()))


def _dimension_dict(value: Mapping[str, int] | None) -> dict[str, int] | None:
    return None if value is None else dict(sorted(value.items()))


def _safe_unit_label(value: object) -> str | None:
    if not isinstance(value, str) or _SAFE_UNIT_LABEL.fullmatch(value) is None:
        return None
    return value if value.strip() else None


def _safe_relation_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if normalized in _KNOWN_RELATION_TYPES:
        return normalized
    return None


def _safe_offset(value: object) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = value
    elif isinstance(value, Mapping):
        start = value.get("start", value.get("start_char"))
        end = value.get("end", value.get("end_char"))
    else:
        try:
            start = getattr(value, "start")
            end = getattr(value, "end")
        except (AttributeError, TypeError):
            return None
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    return start, end


def _offset_from(value: object) -> tuple[int, int] | None:
    return _safe_offset(value)


# Explicit aliases keep the public API discoverable for callers that describe
# the input as relation candidates rather than quantitative relations.
check_relation_unit_compatibility = validate_quantitative_relation
check_quantitative_relation = validate_quantitative_relation
check_quantitative_relations = validate_quantitative_relations
validate_relation_candidate_units = validate_quantitative_relation
validate_relation_units = validate_quantitative_relations


__all__ = [
    "QUANTITATIVE_RELATION_KINDS",
    "UNIT_COMPATIBILITY_ADVISORY",
    "UNIT_COMPATIBILITY_SCHEMA_VERSION",
    "UNIT_COMPATIBLE",
    "UNIT_INCOMPATIBLE",
    "UNIT_UNKNOWN",
    "QuantitativeRelationKind",
    "UnitCompatibilityReport",
    "UnitCompatibilityResult",
    "UnitCompatibilityStatus",
    "check_quantitative_relation",
    "check_quantitative_relations",
    "check_relation_unit_compatibility",
    "check_unit_compatibility",
    "validate_quantitative_relation",
    "validate_quantitative_relations",
    "validate_relation_candidate_units",
    "validate_relation_units",
]
