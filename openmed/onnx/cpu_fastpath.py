"""INT8 classification-head inference with SIMD-aware CPU dispatch."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from openmed.onnx.cpu_features import (
    CpuFeatures,
    detect_cpu_features,
    select_cpu_kernel,
)

INT8_MAX = 127
DEFAULT_LOGIT_TOLERANCE = 0.125
DEFAULT_MIN_SPEEDUP = 1.05
_SIMD_BLOCK_WIDTHS = {
    "avx512": 512,
    "avx2": 256,
    "neon": 128,
}


class CpuFastPathVerificationError(ValueError):
    """Raised when the quantized path diverges from the float reference."""


@dataclass(frozen=True)
class BioSpan:
    """One BIO/BIOES span decoded from token-classification decisions."""

    label: str
    score: float
    token_start: int
    token_end: int
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable span record."""

        return {
            "label": self.label,
            "score": self.score,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class QuantizedClassificationHead:
    """Per-output-channel symmetric INT8 classification-head weights."""

    weights: Any
    scales: Any
    bias: Any

    @property
    def label_count(self) -> int:
        """Return the number of classifier outputs."""

        return int(self.weights.shape[0])

    @property
    def hidden_size(self) -> int:
        """Return the classifier input width."""

        return int(self.weights.shape[1])


@dataclass(frozen=True)
class CpuFastPathResult:
    """Quantized logits, token decisions, and fused BIO decode output."""

    kernel: str
    logits: Any
    label_ids: Any
    scores: Any
    spans: tuple[BioSpan, ...]


@dataclass(frozen=True)
class CpuFastPathVerification:
    """Numerical and decision parity evidence against the float path."""

    kernel: str
    tolerance: float
    max_abs_logit_delta: float
    max_abs_score_delta: float
    label_ids_match: bool
    spans_match: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable verification evidence."""

        return {
            "kernel": self.kernel,
            "tolerance": self.tolerance,
            "max_abs_logit_delta": self.max_abs_logit_delta,
            "max_abs_score_delta": self.max_abs_score_delta,
            "label_ids_match": self.label_ids_match,
            "spans_match": self.spans_match,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class CpuFastPathBenchmarkRecord:
    """Scalar-versus-SIMD latency evidence for one detected CPU tier."""

    cpu: CpuFeatures
    sequence_length: int
    hidden_size: int
    label_count: int
    iterations: int
    scalar_latency_ms: float
    fastpath_latency_ms: float
    speedup: float
    minimum_speedup: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark record."""

        return {
            "cpu": self.cpu.to_dict(),
            "shape": {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "label_count": self.label_count,
            },
            "iterations": self.iterations,
            "latency_ms": {
                "scalar": self.scalar_latency_ms,
                "fastpath": self.fastpath_latency_ms,
            },
            "speedup": self.speedup,
            "minimum_speedup": self.minimum_speedup,
            "passed": self.passed,
        }


def quantize_classification_head(
    weights: Any,
    bias: Any | None = None,
) -> QuantizedClassificationHead:
    """Quantize a float classification head with per-label symmetric scales.

    Args:
        weights: Float array shaped ``[labels, hidden_size]``.
        bias: Optional float array shaped ``[labels]``.

    Returns:
        Validated INT8 weights, per-label scales, and float32 bias.
    """

    np = _require_numpy()
    float_weights = _float_matrix(np, weights, name="weights")
    label_count = int(float_weights.shape[0])
    if bias is None:
        float_bias = np.zeros(label_count, dtype=np.float32)
    else:
        float_bias = np.asarray(bias, dtype=np.float32)
        if float_bias.shape != (label_count,):
            raise ValueError(f"bias must have shape ({label_count},)")
        if not np.isfinite(float_bias).all():
            raise ValueError("bias must contain only finite values")
        float_bias = float_bias.copy()

    max_abs = np.max(np.abs(float_weights), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / INT8_MAX, 1.0).astype(np.float32)
    quantized = np.clip(
        np.rint(float_weights / scales[:, None]),
        -INT8_MAX,
        INT8_MAX,
    ).astype(np.int8)
    return QuantizedClassificationHead(
        weights=quantized,
        scales=scales,
        bias=float_bias,
    )


def run_cpu_fastpath(
    hidden_states: Any,
    head: QuantizedClassificationHead,
    offsets: Any,
    id2label: Mapping[int | str, str],
    *,
    threshold: float = 0.0,
    features: CpuFeatures | None = None,
) -> CpuFastPathResult:
    """Run INT8 matmul and fused winner-score/BIO decoding.

    The hidden states are quantized per token. SIMD tiers use aligned NumPy
    tiles with int32 accumulation, while unsupported CPUs use a deterministic
    scalar accumulator with the same arithmetic.
    """

    np = _require_numpy()
    values = _float_matrix(np, hidden_states, name="hidden_states")
    checked_head = _validate_head(np, head)
    checked_offsets = _validate_offsets(np, offsets, int(values.shape[0]))
    labels = _normalize_id2label(id2label, checked_head.label_count)
    _validate_threshold(threshold)
    if int(values.shape[1]) != checked_head.hidden_size:
        raise ValueError(
            "hidden_states width must match classification-head hidden size"
        )

    selected = features or detect_cpu_features()
    kernel = select_cpu_kernel(selected)
    quantized_values, value_scales = _quantize_rows(np, values)
    accumulators = _quantized_matmul(
        np,
        quantized_values,
        checked_head.weights,
        kernel=kernel,
    )
    logits = accumulators.astype(np.float32)
    logits *= value_scales[:, None]
    logits *= checked_head.scales[None, :]
    logits += checked_head.bias[None, :]
    label_ids, scores = _fused_winners(np, logits)
    spans = decode_bio_spans(
        label_ids,
        scores,
        checked_offsets,
        labels,
        threshold=threshold,
    )
    return CpuFastPathResult(
        kernel=kernel,
        logits=logits,
        label_ids=label_ids,
        scores=scores,
        spans=spans,
    )


def run_reference_classification_head(
    hidden_states: Any,
    weights: Any,
    bias: Any | None,
    offsets: Any,
    id2label: Mapping[int | str, str],
    *,
    threshold: float = 0.0,
) -> CpuFastPathResult:
    """Run the float32 matmul and materialized softmax reference path."""

    np = _require_numpy()
    values = _float_matrix(np, hidden_states, name="hidden_states")
    float_weights = _float_matrix(np, weights, name="weights")
    if int(values.shape[1]) != int(float_weights.shape[1]):
        raise ValueError("hidden_states and weights must share hidden size")
    label_count = int(float_weights.shape[0])
    float_bias = (
        np.zeros(label_count, dtype=np.float32)
        if bias is None
        else np.asarray(bias, dtype=np.float32)
    )
    if float_bias.shape != (label_count,):
        raise ValueError(f"bias must have shape ({label_count},)")
    if not np.isfinite(float_bias).all():
        raise ValueError("bias must contain only finite values")
    checked_offsets = _validate_offsets(np, offsets, int(values.shape[0]))
    labels = _normalize_id2label(id2label, label_count)
    _validate_threshold(threshold)

    logits = values @ float_weights.T
    logits += float_bias[None, :]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    label_ids = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    spans = decode_bio_spans(
        label_ids,
        scores,
        checked_offsets,
        labels,
        threshold=threshold,
    )
    return CpuFastPathResult(
        kernel="reference",
        logits=logits,
        label_ids=label_ids,
        scores=scores,
        spans=spans,
    )


def verify_cpu_fastpath(
    hidden_states: Any,
    weights: Any,
    bias: Any | None,
    head: QuantizedClassificationHead,
    offsets: Any,
    id2label: Mapping[int | str, str],
    *,
    threshold: float = 0.0,
    tolerance: float = DEFAULT_LOGIT_TOLERANCE,
    features: CpuFeatures | None = None,
) -> CpuFastPathVerification:
    """Fail closed unless logits, token decisions, and spans match reference.

    This guard is intended for artifact initialization or certification, not
    for every hot-loop invocation.
    """

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    np = _require_numpy()
    fastpath = run_cpu_fastpath(
        hidden_states,
        head,
        offsets,
        id2label,
        threshold=threshold,
        features=features,
    )
    reference = run_reference_classification_head(
        hidden_states,
        weights,
        bias,
        offsets,
        id2label,
        threshold=threshold,
    )
    max_logit_delta = float(np.max(np.abs(fastpath.logits - reference.logits)))
    max_score_delta = float(np.max(np.abs(fastpath.scores - reference.scores)))
    label_ids_match = bool(np.array_equal(fastpath.label_ids, reference.label_ids))
    spans_match = _span_decisions(fastpath.spans) == _span_decisions(reference.spans)
    passed = max_logit_delta <= tolerance and label_ids_match and spans_match
    verification = CpuFastPathVerification(
        kernel=fastpath.kernel,
        tolerance=tolerance,
        max_abs_logit_delta=max_logit_delta,
        max_abs_score_delta=max_score_delta,
        label_ids_match=label_ids_match,
        spans_match=spans_match,
        passed=passed,
    )
    if not passed:
        raise CpuFastPathVerificationError(
            "CPU fast-path verification failed: "
            f"max logit delta {max_logit_delta:.6f} (limit {tolerance:.6f}), "
            f"label_ids_match={label_ids_match}, spans_match={spans_match}"
        )
    return verification


def decode_bio_spans(
    label_ids: Any,
    scores: Any,
    offsets: Any,
    id2label: Mapping[int | str, str],
    *,
    threshold: float = 0.0,
) -> tuple[BioSpan, ...]:
    """Decode token decisions into source-offset BIO/BIOES spans."""

    np = _require_numpy()
    token_ids = np.asarray(label_ids)
    token_scores = np.asarray(scores, dtype=np.float32)
    checked_offsets = np.asarray(offsets)
    if token_ids.ndim != 1:
        raise ValueError("label_ids must be one-dimensional")
    if token_scores.shape != token_ids.shape:
        raise ValueError("scores must have the same shape as label_ids")
    checked_offsets = _validate_offsets(np, checked_offsets, int(token_ids.shape[0]))
    labels = _normalize_id2label(id2label, minimum_size=0)
    _validate_threshold(threshold)

    spans: list[BioSpan] = []
    current: dict[str, Any] | None = None
    for token_index, (label_id, score, raw_offset) in enumerate(
        zip(token_ids, token_scores, checked_offsets)
    ):
        start, end = int(raw_offset[0]), int(raw_offset[1])
        raw_label = labels.get(int(label_id), f"LABEL_{int(label_id)}")
        prefix, label = _split_label(raw_label)
        if start == end or label.upper() == "O":
            current = _flush_span(spans, current, threshold)
            continue

        starts_new = (
            current is None
            or current["label"] != label
            or prefix in {"B", "S", "U"}
            or (prefix not in {"I", "E", "L"} and start > int(current["end"]))
        )
        if starts_new:
            current = _flush_span(spans, current, threshold)
            current = {
                "label": label,
                "token_start": token_index,
                "token_end": token_index + 1,
                "start": start,
                "end": end,
                "scores": [float(score)],
            }
        else:
            current["token_end"] = token_index + 1
            current["end"] = max(int(current["end"]), end)
            current["scores"].append(float(score))

        if prefix in {"E", "L", "S", "U"}:
            current = _flush_span(spans, current, threshold)

    _flush_span(spans, current, threshold)
    return tuple(spans)


def benchmark_cpu_fastpath(
    hidden_states: Any,
    head: QuantizedClassificationHead,
    offsets: Any,
    id2label: Mapping[int | str, str],
    *,
    features: CpuFeatures | None = None,
    iterations: int = 7,
    warmup: int = 2,
    minimum_speedup: float = DEFAULT_MIN_SPEEDUP,
    clock: Callable[[], float] = perf_counter,
) -> CpuFastPathBenchmarkRecord:
    """Benchmark the detected SIMD tier against the scalar INT8 kernel."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if minimum_speedup <= 0.0:
        raise ValueError("minimum_speedup must be positive")

    np = _require_numpy()
    values = _float_matrix(np, hidden_states, name="hidden_states")
    selected = features or detect_cpu_features()
    scalar = CpuFeatures(
        architecture=selected.architecture,
        flags=frozenset(),
    )

    def run_scalar() -> None:
        run_cpu_fastpath(
            values,
            head,
            offsets,
            id2label,
            features=scalar,
        )

    def run_selected() -> None:
        run_cpu_fastpath(
            values,
            head,
            offsets,
            id2label,
            features=selected,
        )

    for _ in range(warmup):
        run_selected()
        run_scalar()
    scalar_latency = _median_latency_ms(run_scalar, iterations, clock)
    fastpath_latency = _median_latency_ms(run_selected, iterations, clock)
    speedup = scalar_latency / fastpath_latency if fastpath_latency > 0.0 else 0.0
    passed = selected.tier != "scalar" and speedup >= minimum_speedup
    return CpuFastPathBenchmarkRecord(
        cpu=selected,
        sequence_length=int(values.shape[0]),
        hidden_size=int(values.shape[1]),
        label_count=head.label_count,
        iterations=iterations,
        scalar_latency_ms=scalar_latency,
        fastpath_latency_ms=fastpath_latency,
        speedup=speedup,
        minimum_speedup=minimum_speedup,
        passed=passed,
    )


def _validate_head(
    np: Any, head: QuantizedClassificationHead
) -> QuantizedClassificationHead:
    weights = np.asarray(head.weights)
    scales = np.asarray(head.scales, dtype=np.float32)
    bias = np.asarray(head.bias, dtype=np.float32)
    if weights.ndim != 2 or not weights.shape[0] or not weights.shape[1]:
        raise ValueError("quantized weights must be a non-empty two-dimensional array")
    if weights.dtype != np.int8:
        raise ValueError("quantized weights must use int8 dtype")
    if scales.shape != (weights.shape[0],):
        raise ValueError("head scales must have one value per label")
    if bias.shape != (weights.shape[0],):
        raise ValueError("head bias must have one value per label")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("head scales must be finite and positive")
    if not np.isfinite(bias).all():
        raise ValueError("head bias must contain only finite values")
    max_hidden_size = np.iinfo(np.int32).max // (INT8_MAX * INT8_MAX)
    if int(weights.shape[1]) > max_hidden_size:
        raise ValueError("classification head is too wide for int32 accumulation")
    return QuantizedClassificationHead(weights=weights, scales=scales, bias=bias)


def _float_matrix(np: Any, values: Any, *, name: str) -> Any:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_offsets(np: Any, offsets: Any, token_count: int) -> Any:
    values = np.asarray(offsets)
    if values.shape != (token_count, 2):
        raise ValueError(f"offsets must have shape ({token_count}, 2)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("offsets must contain integers")
    if np.any(values < 0) or np.any(values[:, 1] < values[:, 0]):
        raise ValueError("offsets must be non-negative ordered pairs")
    return values


def _normalize_id2label(
    id2label: Mapping[int | str, str],
    minimum_size: int,
) -> dict[int, str]:
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("id2label must be a non-empty mapping")
    normalized = {int(key): str(value) for key, value in id2label.items()}
    if minimum_size and any(index not in normalized for index in range(minimum_size)):
        raise ValueError("id2label must define every classification-head output")
    return normalized


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")


def _quantize_rows(np: Any, values: Any) -> tuple[Any, Any]:
    max_abs = np.max(np.abs(values), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / INT8_MAX, 1.0).astype(np.float32)
    quantized = np.clip(
        np.rint(values / scales[:, None]),
        -INT8_MAX,
        INT8_MAX,
    ).astype(np.int8)
    return quantized, scales


def _quantized_matmul(
    np: Any,
    values: Any,
    weights: Any,
    *,
    kernel: str,
) -> Any:
    if kernel == "scalar":
        return _scalar_int8_matmul(np, values, weights)
    block_width = _SIMD_BLOCK_WIDTHS[kernel]
    accumulators = np.zeros((values.shape[0], weights.shape[0]), dtype=np.int32)
    for start in range(0, values.shape[1], block_width):
        stop = min(start + block_width, values.shape[1])
        value_block = values[:, start:stop].astype(np.int32)
        weight_block = weights[:, start:stop].astype(np.int32)
        accumulators += value_block @ weight_block.T
    return accumulators


def _scalar_int8_matmul(np: Any, values: Any, weights: Any) -> Any:
    result = np.empty((values.shape[0], weights.shape[0]), dtype=np.int32)
    for token_index, row in enumerate(values):
        for label_index, weight in enumerate(weights):
            accumulator = 0
            for value, coefficient in zip(row, weight):
                accumulator += int(value) * int(coefficient)
            result[token_index, label_index] = accumulator
    return result


def _fused_winners(np: Any, logits: Any) -> tuple[Any, Any]:
    label_ids = logits.argmax(axis=1)
    winner_logits = np.take_along_axis(logits, label_ids[:, None], axis=1)[:, 0]
    denominator = np.exp(logits - winner_logits[:, None]).sum(axis=1)
    scores = (1.0 / denominator).astype(np.float32)
    return label_ids, scores


def _split_label(raw_label: str) -> tuple[str, str]:
    normalized = str(raw_label).strip()
    if len(normalized) > 2 and normalized[1] in {"-", "_"}:
        prefix = normalized[0].upper()
        if prefix in {"B", "I", "E", "L", "S", "U"}:
            return prefix, normalized[2:]
    return "", normalized


def _flush_span(
    spans: list[BioSpan],
    current: dict[str, Any] | None,
    threshold: float,
) -> None:
    if current is None:
        return None
    span_score = sum(current["scores"]) / len(current["scores"])
    if span_score >= threshold:
        spans.append(
            BioSpan(
                label=str(current["label"]),
                score=span_score,
                token_start=int(current["token_start"]),
                token_end=int(current["token_end"]),
                start=int(current["start"]),
                end=int(current["end"]),
            )
        )
    return None


def _span_decisions(spans: Sequence[BioSpan]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            span.label,
            span.token_start,
            span.token_end,
            span.start,
            span.end,
        )
        for span in spans
    )


def _median_latency_ms(
    callback: Callable[[], None],
    iterations: int,
    clock: Callable[[], float],
) -> float:
    samples = []
    for _ in range(iterations):
        started = clock()
        callback()
        samples.append((clock() - started) * 1000.0)
    return float(median(samples))


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "The CPU classification fast path requires NumPy. "
            "Install with: pip install 'openmed[onnx-runtime]'"
        ) from exc
    return np


def _synthetic_benchmark(args: argparse.Namespace) -> CpuFastPathBenchmarkRecord:
    np = _require_numpy()
    rng = np.random.default_rng(args.seed)
    hidden_states = rng.normal(
        0.0,
        0.5,
        size=(args.sequence_length, args.hidden_size),
    ).astype(np.float32)
    weights = rng.normal(
        0.0,
        0.05,
        size=(args.labels, args.hidden_size),
    ).astype(np.float32)
    head = quantize_classification_head(weights)
    offsets = np.column_stack(
        (
            np.arange(args.sequence_length, dtype=np.int64) * 2,
            np.arange(args.sequence_length, dtype=np.int64) * 2 + 1,
        )
    )
    id2label = {0: "O"}
    id2label.update({index: f"S-ENTITY_{index}" for index in range(1, args.labels)})
    return benchmark_cpu_fastpath(
        hidden_states,
        head,
        offsets,
        id2label,
        iterations=args.iterations,
        warmup=args.warmup,
        minimum_speedup=args.minimum_speedup,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the local CPU INT8 classification-head fast path.",
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--labels", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--minimum-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument("--seed", type=int, default=479)
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="Exit unsuccessfully unless a SIMD tier meets the minimum speedup.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic benchmark CLI and print its JSON record."""

    args = _build_parser().parse_args(argv)
    if args.sequence_length <= 0 or args.hidden_size <= 0 or args.labels <= 1:
        raise SystemExit("benchmark dimensions must be positive and labels > 1")
    record = _synthetic_benchmark(args)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 1 if args.require_speedup and not record.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOGIT_TOLERANCE",
    "DEFAULT_MIN_SPEEDUP",
    "BioSpan",
    "CpuFastPathBenchmarkRecord",
    "CpuFastPathResult",
    "CpuFastPathVerification",
    "CpuFastPathVerificationError",
    "QuantizedClassificationHead",
    "benchmark_cpu_fastpath",
    "decode_bio_spans",
    "main",
    "quantize_classification_head",
    "run_cpu_fastpath",
    "run_reference_classification_head",
    "verify_cpu_fastpath",
]
