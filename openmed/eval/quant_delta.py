"""Quantized-artifact recall delta gates.

The release gate compares quantized artifacts to their full-precision parent on
G1 and G2 labels. A format is blocked only when that format's recall loss meets
or exceeds its threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openmed.core.labels import normalize_label

INT8_RECALL_DELTA_LIMIT = 0.005
INT4_RECALL_DELTA_LIMIT = 0.010

G1_G2_LABELS = frozenset(
    {
        "ACCOUNT_NUMBER",
        "AGE",
        "API_KEY",
        "BUILDING_NUMBER",
        "CREDIT_CARD",
        "DATE",
        "DATE_OF_BIRTH",
        "EMAIL",
        "FIRST_NAME",
        "GPS_COORDINATES",
        "IBAN",
        "ID_NUM",
        "LAST_NAME",
        "LOCATION",
        "MIDDLE_NAME",
        "PERSON",
        "PHONE",
        "SSN",
        "STREET_ADDRESS",
        "TIME",
        "URL",
        "USERNAME",
        "ZIPCODE",
    }
)


@dataclass(frozen=True)
class QuantRecallDeltaResult:
    """Structured quantization recall-delta gate evidence."""

    format: str
    quantized: bool
    passed: bool
    limit: float | None = None
    max_delta: float | None = None
    per_label_delta: Mapping[str, float] = field(default_factory=dict)
    offending_labels: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    labels_evaluated: tuple[str, ...] = ()
    source: str = "not_applicable"

    @property
    def blocking_format(self) -> str | None:
        if self.quantized and not self.passed:
            return self.format
        return None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "quantized": self.quantized,
            "passed": self.passed,
            "limit": self.limit,
            "max_delta": self.max_delta,
            "per_label_delta": dict(self.per_label_delta),
            "offending_labels": {
                label: dict(details) for label, details in self.offending_labels.items()
            },
            "labels_evaluated": list(self.labels_evaluated),
            "source": self.source,
        }
        return payload


def evaluate_quant_recall_delta(
    *,
    format_name: str,
    candidate_recall: Mapping[str, Any],
    parent_recall: Mapping[str, Any] | None = None,
    precomputed_delta: Any = None,
    labels: Sequence[str] | None = None,
) -> QuantRecallDeltaResult:
    """Evaluate quantized recall loss for *format_name*.

    ``precomputed_delta`` may be a scalar, a per-label mapping, or a mapping
    keyed by format. Without precomputed evidence, ``parent_recall`` is compared
    to ``candidate_recall`` on G1+G2 labels.
    """

    normalized_format = _normalise_dimension(format_name)
    limit = limit_for_format(format_name)
    if limit is None:
        return QuantRecallDeltaResult(
            format=format_name,
            quantized=False,
            passed=True,
        )

    selected_labels = {
        normalize_label(str(label)) for label in (labels or sorted(G1_G2_LABELS))
    }
    delta_payload = _select_precomputed_delta(precomputed_delta, normalized_format)
    if delta_payload is not None:
        if isinstance(delta_payload, Mapping):
            per_label = _normalise_delta_map(delta_payload, selected_labels)
            return _result_from_delta_map(
                format_name=format_name,
                limit=limit,
                per_label_delta=per_label,
                source="precomputed_per_label_delta",
            )

        parsed = _normalise_precomputed_delta(delta_payload)
        if parsed is None:
            return _missing_result(format_name, limit)
        return _result_from_delta_map(
            format_name=format_name,
            limit=limit,
            per_label_delta={"OVERALL": parsed},
            source="precomputed_delta",
        )

    if parent_recall is None:
        return _missing_result(format_name, limit)

    parent = _normalise_recall_map(parent_recall)
    candidate = _normalise_recall_map(candidate_recall)
    per_label: dict[str, float] = {}
    for label in sorted(selected_labels & set(parent)):
        parent_value = parent[label]
        candidate_value = candidate.get(label, 0.0)
        per_label[label] = max(parent_value - candidate_value, 0.0)

    if not per_label:
        return _missing_result(format_name, limit)

    return _result_from_delta_map(
        format_name=format_name,
        limit=limit,
        per_label_delta=per_label,
        source="computed_from_parent",
    )


def is_quantized_format(format_name: str) -> bool:
    return limit_for_format(format_name) is not None


def limit_for_format(format_name: str) -> float | None:
    normalized = _normalise_dimension(format_name)
    if "int8" in normalized or "8bit" in normalized or "8-bit" in normalized:
        return INT8_RECALL_DELTA_LIMIT
    if "int4" in normalized or "4bit" in normalized or "4-bit" in normalized:
        return INT4_RECALL_DELTA_LIMIT
    return None


def _result_from_delta_map(
    *,
    format_name: str,
    limit: float,
    per_label_delta: Mapping[str, float],
    source: str,
) -> QuantRecallDeltaResult:
    deltas = {label: float(value) for label, value in sorted(per_label_delta.items())}
    max_delta = max(deltas.values()) if deltas else None
    offending = {
        label: {"delta": delta, "limit": limit}
        for label, delta in deltas.items()
        if delta >= limit
    }
    return QuantRecallDeltaResult(
        format=format_name,
        quantized=True,
        passed=not offending and max_delta is not None,
        limit=limit,
        max_delta=max_delta,
        per_label_delta=deltas,
        offending_labels=offending,
        labels_evaluated=tuple(deltas),
        source=source,
    )


def _missing_result(format_name: str, limit: float) -> QuantRecallDeltaResult:
    return QuantRecallDeltaResult(
        format=format_name,
        quantized=True,
        passed=False,
        limit=limit,
        source="missing_evidence",
    )


def _select_precomputed_delta(value: Any, normalized_format: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return value

    for key, item in value.items():
        if _normalise_dimension(str(key)) == normalized_format:
            return item

    return value


def _normalise_delta_map(
    values: Mapping[str, Any],
    labels: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, value in values.items():
        canonical = (
            "OVERALL"
            if str(label).upper() == "OVERALL"
            else normalize_label(str(label))
        )
        if canonical not in labels and canonical != "OVERALL":
            continue
        parsed = _normalise_precomputed_delta(value)
        if parsed is not None:
            result[canonical] = parsed
    return result


def _normalise_recall_map(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, value in values.items():
        parsed = _optional_float(value)
        if parsed is None:
            continue
        if parsed > 1.0:
            parsed = parsed / 100.0
        canonical = (
            "OVERALL"
            if str(label).upper() == "OVERALL"
            else normalize_label(str(label))
        )
        result[canonical] = parsed
    return result


def _normalise_precomputed_delta(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    delta = abs(parsed)
    if delta > 0.05:
        return delta / 100.0
    return delta


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalise_dimension(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


__all__ = [
    "G1_G2_LABELS",
    "INT4_RECALL_DELTA_LIMIT",
    "INT8_RECALL_DELTA_LIMIT",
    "QuantRecallDeltaResult",
    "evaluate_quant_recall_delta",
    "is_quantized_format",
    "limit_for_format",
]
