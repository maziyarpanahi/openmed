"""Quantized-artifact recall delta gates.

The release gate compares quantized artifacts to their full-precision parent on
G1 and G2 labels. A format is blocked only when that format's recall loss meets
or exceeds its threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label

COREML_RECALL_DELTA_LIMIT = 0.003
INT8_RECALL_DELTA_LIMIT = 0.005
INT4_RECALL_DELTA_LIMIT = 0.010
DEFAULT_SPECULATIVE_KL_LIMIT = 0.01
DEFAULT_SPECULATIVE_SPEEDUP_LIMIT = 1.6

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


@dataclass(frozen=True)
class OnnxLogitParityResult:
    """Numeric and span parity evidence for optimized ONNX token classifiers."""

    passed: bool
    logits_within_tolerance: bool
    spans_identical: bool
    max_abs_diff: float
    max_rel_diff: float
    token_mismatches: int
    tokens_evaluated: int
    span_count: int = 0
    source: str = "computed_from_logits"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "logits_within_tolerance": self.logits_within_tolerance,
            "spans_identical": self.spans_identical,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "token_mismatches": self.token_mismatches,
            "tokens_evaluated": self.tokens_evaluated,
            "span_count": self.span_count,
            "source": self.source,
        }


@dataclass(frozen=True)
class SpeculativeRedactionParityResult:
    """Structured evidence for speculative redaction decode parity."""

    passed: bool
    greedy_mismatch_count: int
    sampling_kl: float | None
    sampling_kl_limit: float
    median_latency_speedup: float | None
    latency_speedup_limit: float
    max_recall_delta: float
    new_leak_count: int
    tokenizer_fallback_count: int = 0
    tokenizer_fallback_correct: bool = True
    greedy_mismatch_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "greedy_mismatch_count": self.greedy_mismatch_count,
            "greedy_mismatch_indices": list(self.greedy_mismatch_indices),
            "sampling_kl": self.sampling_kl,
            "sampling_kl_limit": self.sampling_kl_limit,
            "median_latency_speedup": self.median_latency_speedup,
            "latency_speedup_limit": self.latency_speedup_limit,
            "max_recall_delta": self.max_recall_delta,
            "new_leak_count": self.new_leak_count,
            "tokenizer_fallback_count": self.tokenizer_fallback_count,
            "tokenizer_fallback_correct": self.tokenizer_fallback_correct,
        }


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


def evaluate_onnx_logit_parity(
    baseline_logits: Any,
    candidate_logits: Any,
    *,
    id2label: Mapping[Any, str] | None = None,
    offsets: Any = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> OnnxLogitParityResult:
    """Compare logits and predicted spans from two token-classification graphs.

    The helper accepts nested Python sequences or array-like values with a
    ``tolist`` method. Span parity is derived from BIO labels when ``id2label``
    and tokenizer offsets are provided; otherwise exact token prediction parity
    is used as the span-safety proxy.
    """

    baseline = _as_nested_list(baseline_logits)
    candidate = _as_nested_list(candidate_logits)
    baseline_values = list(_flatten_numbers(baseline))
    candidate_values = list(_flatten_numbers(candidate))
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        return OnnxLogitParityResult(
            passed=False,
            logits_within_tolerance=False,
            spans_identical=False,
            max_abs_diff=math.inf,
            max_rel_diff=math.inf,
            token_mismatches=0,
            tokens_evaluated=0,
            source="shape_mismatch",
        )

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    logits_within_tolerance = True
    for left, right in zip(baseline_values, candidate_values):
        abs_diff = abs(left - right)
        rel_diff = abs_diff / max(abs(left), abs(right), 1e-12)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        if abs_diff > (atol + rtol * abs(left)):
            logits_within_tolerance = False

    baseline_tokens = _argmax_token_ids(baseline)
    candidate_tokens = _argmax_token_ids(candidate)
    token_pairs = list(
        zip(_flatten_token_ids(baseline_tokens), _flatten_token_ids(candidate_tokens))
    )
    token_mismatches = sum(1 for left, right in token_pairs if left != right)
    tokens_evaluated = len(token_pairs)

    if id2label is not None and offsets is not None:
        baseline_spans = _bio_spans(baseline_tokens, offsets, id2label)
        candidate_spans = _bio_spans(candidate_tokens, offsets, id2label)
        spans_identical = baseline_spans == candidate_spans
        span_count = len(baseline_spans)
    else:
        spans_identical = token_mismatches == 0 and tokens_evaluated > 0
        span_count = 0

    return OnnxLogitParityResult(
        passed=logits_within_tolerance and spans_identical,
        logits_within_tolerance=logits_within_tolerance,
        spans_identical=spans_identical,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        token_mismatches=token_mismatches,
        tokens_evaluated=tokens_evaluated,
        span_count=span_count,
    )


def evaluate_speculative_redaction_parity(
    *,
    reference_greedy_outputs: Sequence[str],
    speculative_greedy_outputs: Sequence[str],
    reference_sampling_counts: Mapping[str, int] | None = None,
    speculative_sampling_counts: Mapping[str, int] | None = None,
    reference_latency_ms: Sequence[float] | None = None,
    speculative_latency_ms: Sequence[float] | None = None,
    reference_recall: Mapping[str, Any] | None = None,
    speculative_recall: Mapping[str, Any] | None = None,
    reference_leak_count: int = 0,
    speculative_leak_count: int = 0,
    tokenizer_fallback_count: int = 0,
    tokenizer_fallback_correct: bool = True,
    sampling_kl_limit: float = DEFAULT_SPECULATIVE_KL_LIMIT,
    latency_speedup_limit: float = DEFAULT_SPECULATIVE_SPEEDUP_LIMIT,
) -> SpeculativeRedactionParityResult:
    """Evaluate parity between plain and speculative generative PII decoding."""

    mismatch_indices = tuple(
        index
        for index, (reference, candidate) in enumerate(
            zip(reference_greedy_outputs, speculative_greedy_outputs)
        )
        if reference != candidate
    )
    length_mismatches = abs(
        len(reference_greedy_outputs) - len(speculative_greedy_outputs)
    )
    if length_mismatches:
        start = min(len(reference_greedy_outputs), len(speculative_greedy_outputs))
        mismatch_indices = (*mismatch_indices, *range(start, start + length_mismatches))

    sampling_kl = None
    if (
        reference_sampling_counts is not None
        and speculative_sampling_counts is not None
    ):
        sampling_kl = token_frequency_kl(
            reference_sampling_counts,
            speculative_sampling_counts,
        )

    median_latency_speedup = None
    reference_median = _median(reference_latency_ms)
    speculative_median = _median(speculative_latency_ms)
    if (
        reference_median is not None
        and speculative_median is not None
        and speculative_median > 0.0
    ):
        median_latency_speedup = reference_median / speculative_median

    max_recall_delta = _max_recall_delta(reference_recall, speculative_recall)
    new_leak_count = max(int(speculative_leak_count) - int(reference_leak_count), 0)

    passed = (
        not mismatch_indices
        and sampling_kl is not None
        and sampling_kl <= sampling_kl_limit
        and median_latency_speedup is not None
        and median_latency_speedup >= latency_speedup_limit
        and max_recall_delta == 0.0
        and new_leak_count == 0
        and tokenizer_fallback_correct
    )

    return SpeculativeRedactionParityResult(
        passed=passed,
        greedy_mismatch_count=len(mismatch_indices),
        greedy_mismatch_indices=mismatch_indices,
        sampling_kl=sampling_kl,
        sampling_kl_limit=sampling_kl_limit,
        median_latency_speedup=median_latency_speedup,
        latency_speedup_limit=latency_speedup_limit,
        max_recall_delta=max_recall_delta,
        new_leak_count=new_leak_count,
        tokenizer_fallback_count=max(int(tokenizer_fallback_count), 0),
        tokenizer_fallback_correct=bool(tokenizer_fallback_correct),
    )


def token_frequency_kl(
    reference_counts: Mapping[str, int],
    candidate_counts: Mapping[str, int],
    *,
    smoothing: float = 1e-12,
) -> float:
    """Return KL(reference || candidate) over fixed-seed token frequencies."""

    keys = set(reference_counts) | set(candidate_counts)
    if not keys:
        return 0.0
    reference_total = sum(max(int(value), 0) for value in reference_counts.values())
    candidate_total = sum(max(int(value), 0) for value in candidate_counts.values())
    if reference_total <= 0 or candidate_total <= 0:
        return math.inf

    vocab_size = len(keys)
    reference_denominator = reference_total + smoothing * vocab_size
    candidate_denominator = candidate_total + smoothing * vocab_size
    divergence = 0.0
    for key in keys:
        p = (max(int(reference_counts.get(key, 0)), 0) + smoothing) / (
            reference_denominator
        )
        q = (max(int(candidate_counts.get(key, 0)), 0) + smoothing) / (
            candidate_denominator
        )
        divergence += p * math.log(p / q)
    return divergence


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


def is_quantized_format(format_name: str) -> bool:
    return limit_for_format(format_name) is not None


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


def _as_nested_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_as_nested_list(item) for item in value]
    if isinstance(value, list):
        return [_as_nested_list(item) for item in value]
    return value


def _flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_numbers(item)
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(number):
        yield number


def _argmax_token_ids(logits: Any) -> list[list[int]]:
    batches = _as_nested_list(logits)
    result: list[list[int]] = []
    if not isinstance(batches, list):
        return result
    for batch in batches:
        if not isinstance(batch, list):
            continue
        tokens: list[int] = []
        for token_logits in batch:
            if not isinstance(token_logits, list) or not token_logits:
                continue
            tokens.append(_argmax(token_logits))
        result.append(tokens)
    return result


def _argmax(values: Sequence[Any]) -> int:
    best_index = 0
    best_value = -math.inf
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = -math.inf
        if number > best_value:
            best_index = index
            best_value = number
    return best_index


def _flatten_token_ids(value: Sequence[Sequence[int]]) -> Iterable[int]:
    for batch in value:
        yield from batch


def _bio_spans(
    token_ids: Sequence[Sequence[int]],
    offsets: Any,
    id2label: Mapping[Any, str],
) -> list[tuple[int, int, str]]:
    offset_batches = _as_nested_list(offsets)
    spans: list[tuple[int, int, str]] = []
    for batch_index, batch_tokens in enumerate(token_ids):
        batch_offsets = (
            offset_batches[batch_index]
            if isinstance(offset_batches, list) and batch_index < len(offset_batches)
            else []
        )
        current: tuple[int, int, str] | None = None
        for token_index, token_id in enumerate(batch_tokens):
            label = _label_for_id(id2label, token_id)
            offset = (
                batch_offsets[token_index]
                if isinstance(batch_offsets, list) and token_index < len(batch_offsets)
                else None
            )
            if not _valid_offset(offset):
                if current is not None:
                    spans.append(current)
                    current = None
                continue

            start = int(offset[0])
            end = int(offset[1])
            prefix, entity = _split_bio_label(label)
            if prefix == "O" or entity is None:
                if current is not None:
                    spans.append(current)
                    current = None
                continue

            if prefix == "B" or current is None or current[2] != entity:
                if current is not None:
                    spans.append(current)
                current = (start, end, entity)
            else:
                current = (current[0], end, current[2])

        if current is not None:
            spans.append(current)
    return spans


def _label_for_id(id2label: Mapping[Any, str], token_id: int) -> str:
    for key in (token_id, str(token_id)):
        if key in id2label:
            return str(id2label[key])
    return str(token_id)


def _valid_offset(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and _optional_float(value[0]) is not None
        and _optional_float(value[1]) is not None
        and int(value[1]) > int(value[0])
    )


def _split_bio_label(label: str) -> tuple[str, str | None]:
    normalized = str(label)
    if normalized == "O":
        return "O", None
    if "-" not in normalized:
        return "B", normalized
    prefix, entity = normalized.split("-", 1)
    if prefix not in {"B", "I"} or not entity:
        return "B", normalized
    return prefix, entity


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


def _median(values: Sequence[float] | None) -> float | None:
    if not values:
        return None
    parsed = sorted(float(value) for value in values)
    midpoint = len(parsed) // 2
    if len(parsed) % 2:
        return parsed[midpoint]
    return (parsed[midpoint - 1] + parsed[midpoint]) / 2.0


def _max_recall_delta(
    reference_recall: Mapping[str, Any] | None,
    speculative_recall: Mapping[str, Any] | None,
) -> float:
    if reference_recall is None and speculative_recall is None:
        return 0.0
    if reference_recall is None or speculative_recall is None:
        return math.inf

    reference = _normalise_recall_map(reference_recall)
    speculative = _normalise_recall_map(speculative_recall)
    labels = set(reference) | set(speculative)
    if not labels:
        return 0.0
    return max(
        max(reference.get(label, 0.0) - speculative.get(label, 0.0), 0.0)
        for label in labels
    )


def _normalise_dimension(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


__all__ = [
    "COREML_RECALL_DELTA_LIMIT",
    "DEFAULT_SPECULATIVE_KL_LIMIT",
    "DEFAULT_SPECULATIVE_SPEEDUP_LIMIT",
    "CoreMLSpanParityResult",
    "G1_G2_LABELS",
    "INT4_RECALL_DELTA_LIMIT",
    "INT8_RECALL_DELTA_LIMIT",
    "OnnxLogitParityResult",
    "QuantRecallDeltaResult",
    "SpeculativeRedactionParityResult",
    "evaluate_coreml_span_parity",
    "evaluate_onnx_logit_parity",
    "evaluate_quant_recall_delta",
    "evaluate_speculative_redaction_parity",
    "is_quantized_format",
    "limit_for_format",
    "token_frequency_kl",
]
