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

COREML_RECALL_DELTA_LIMIT = 0.003
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


@dataclass(frozen=True)
class CoreMLSpanParityResult:
    """Structured CoreML variant parity evidence."""

    format: str
    passed: bool
    recall_delta_limit: float
    span_tolerance: int
    max_recall_delta: float | None = None
    per_label_delta: Mapping[str, float] = field(default_factory=dict)
    span_mismatches: tuple[Mapping[str, Any], ...] = ()
    labels_evaluated: tuple[str, ...] = ()
    auto_rejected: bool = False
    rejection_reason: str | None = None
    source: str = "computed"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "passed": self.passed,
            "recall_delta_limit": self.recall_delta_limit,
            "span_tolerance": self.span_tolerance,
            "max_recall_delta": self.max_recall_delta,
            "per_label_delta": dict(self.per_label_delta),
            "span_mismatches": [dict(item) for item in self.span_mismatches],
            "labels_evaluated": list(self.labels_evaluated),
            "auto_rejected": self.auto_rejected,
            "rejection_reason": self.rejection_reason,
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


def evaluate_coreml_span_parity(
    *,
    format_name: str,
    reference_spans: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_spans: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_recall: Mapping[str, Any],
    candidate_recall: Mapping[str, Any],
    recall_delta_limit: float = COREML_RECALL_DELTA_LIMIT,
    span_tolerance: int = 0,
    rejectable: bool = False,
) -> CoreMLSpanParityResult:
    """Evaluate CoreML span parity against a PyTorch reference.

    ``reference_spans`` and ``candidate_spans`` are keyed by fixture id. Each
    span must include a label and character offsets. fp16/int8 callers should
    leave ``rejectable`` false so any mismatch fails the gate. int4 callers set
    it true to produce an explicit auto-rejection report when parity drifts.
    """

    reference = {
        str(fixture_id): _normalise_span_list(spans)
        for fixture_id, spans in reference_spans.items()
    }
    candidate = {
        str(fixture_id): _normalise_span_list(spans)
        for fixture_id, spans in candidate_spans.items()
    }
    span_mismatches = tuple(
        _span_mismatches(reference, candidate, tolerance=span_tolerance)
    )

    parent = _normalise_recall_map(reference_recall)
    child = _normalise_recall_map(candidate_recall)
    labels = sorted(set(parent) | set(child))
    per_label_delta = {
        label: max(parent.get(label, 0.0) - child.get(label, 0.0), 0.0)
        for label in labels
    }
    max_delta = max(per_label_delta.values()) if per_label_delta else None
    recall_violations = {
        label: delta
        for label, delta in per_label_delta.items()
        if delta > recall_delta_limit + 1e-12
    }
    passed = not span_mismatches and not recall_violations and max_delta is not None
    rejection_reason = None
    if rejectable and not passed:
        reasons = []
        if span_mismatches:
            reasons.append("span parity mismatch")
        if recall_violations:
            reasons.append("recall delta exceeds limit")
        if max_delta is None:
            reasons.append("missing recall evidence")
        rejection_reason = "; ".join(reasons)

    return CoreMLSpanParityResult(
        format=format_name,
        passed=passed,
        recall_delta_limit=recall_delta_limit,
        span_tolerance=span_tolerance,
        max_recall_delta=max_delta,
        per_label_delta=per_label_delta,
        span_mismatches=span_mismatches,
        labels_evaluated=tuple(labels),
        auto_rejected=bool(rejectable and not passed),
        rejection_reason=rejection_reason,
    )


def limit_for_format(format_name: str) -> float | None:
    normalized = _normalise_dimension(format_name)
    if "int8" in normalized or "8bit" in normalized or "8-bit" in normalized:
        return INT8_RECALL_DELTA_LIMIT
    if "int4" in normalized or "4bit" in normalized or "4-bit" in normalized:
        return INT4_RECALL_DELTA_LIMIT
    return None


def _normalise_span_list(
    spans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalised: list[dict[str, Any]] = []
    for span in spans:
        start = _optional_int(span.get("start"))
        end = _optional_int(span.get("end"))
        if start is None or end is None:
            continue
        raw_label = (
            span.get("canonical_label")
            or span.get("label")
            or span.get("entity_type")
            or span.get("entity_group")
            or span.get("entity")
            or "OTHER"
        )
        normalised.append(
            {
                "label": normalize_label(str(raw_label)),
                "start": int(start),
                "end": int(end),
                "text": str(span.get("text") or span.get("word") or ""),
            }
        )
    return tuple(
        sorted(normalised, key=lambda item: (item["start"], item["end"], item["label"]))
    )


def _span_mismatches(
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tolerance: int,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    fixture_ids = sorted(set(reference) | set(candidate))
    for fixture_id in fixture_ids:
        expected = list(reference.get(fixture_id, ()))
        observed = list(candidate.get(fixture_id, ()))
        if len(expected) != len(observed):
            mismatches.append(
                {
                    "fixture_id": fixture_id,
                    "reason": "span count mismatch",
                    "expected": expected,
                    "observed": observed,
                }
            )
            continue

        for index, (left, right) in enumerate(zip(expected, observed)):
            if _spans_match(left, right, tolerance=tolerance):
                continue
            mismatches.append(
                {
                    "fixture_id": fixture_id,
                    "span_index": index,
                    "reason": "span mismatch",
                    "expected": dict(left),
                    "observed": dict(right),
                }
            )
    return mismatches


def _spans_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: int,
) -> bool:
    return (
        left.get("label") == right.get("label")
        and abs(int(left.get("start", -1)) - int(right.get("start", -2))) <= tolerance
        and abs(int(left.get("end", -1)) - int(right.get("end", -2))) <= tolerance
    )


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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _normalise_dimension(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


__all__ = [
    "COREML_RECALL_DELTA_LIMIT",
    "CoreMLSpanParityResult",
    "G1_G2_LABELS",
    "INT4_RECALL_DELTA_LIMIT",
    "INT8_RECALL_DELTA_LIMIT",
    "QuantRecallDeltaResult",
    "evaluate_coreml_span_parity",
    "evaluate_quant_recall_delta",
    "is_quantized_format",
    "limit_for_format",
]
