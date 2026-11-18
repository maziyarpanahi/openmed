"""Export-time fp16 parity gate for the WebGPU ONNX artifact.

The default ONNX profile converts the fp32 graph to an fp16 graph for WebGPU.
This module decides whether that fp16 graph may be emitted. It compares the
fp16 graph with its fp32 reference on deterministic synthetic inputs through
ONNX Runtime CPU and scores logit deltas, token agreement, span agreement and
per-label recall loss on critical (G1/G2) labels.

ONNX Runtime CPU runs fp16 nodes that lack a CPU fp16 kernel in fp32, so CPU
logit parity alone can miss an activation that overflows fp16 storage on a
WebGPU device. The gate therefore also probes the fp32 reference for every
activation the artifact stores in fp16 and fails when one exceeds the fp16
range.

Gate evidence is value-free: aggregate metrics, sequence lengths, label names,
operator types, counts and SHA-256 digests. It never records input text, token
ids, logits, file paths or exception messages.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.capabilities import raise_missing_backend
from openmed.core.decoding import build_label_info, labels_to_token_spans
from openmed.core.labels import normalize_label
from openmed.eval.quant_delta import (
    COREML_RECALL_DELTA_LIMIT,
    G1_G2_LABELS,
    evaluate_onnx_logit_parity,
)

WEBGPU_FP16_GATE_KEY = "webgpu_fp16_gate"
WEBGPU_FP16_GATE_VERSION = 1
WEBGPU_GATE_POLICY_FAIL = "fail"
WEBGPU_GATE_POLICY_OMIT = "omit"
WEBGPU_GATE_POLICIES = (WEBGPU_GATE_POLICY_FAIL, WEBGPU_GATE_POLICY_OMIT)
WEBGPU_GATE_EXECUTION_PROVIDER = "CPUExecutionProvider"

FP16_MAX_FINITE = 65504.0
# fp32 magnitudes at or beyond this are deliberate masking sentinels such as
# ``finfo(float32).min`` fills. The fp16 converter clamps those constants to the
# fp16 range, so they are counted separately and never treated as overflow.
FP32_SENTINEL_MAGNITUDE = 1e30

DEFAULT_FP16_LOGIT_ATOL = 5e-2
DEFAULT_FP16_LOGIT_RTOL = 2e-2
DEFAULT_FP16_MAX_TOKEN_MISMATCH_RATE = 1e-3
DEFAULT_FP16_CRITICAL_RECALL_DELTA_LIMIT = COREML_RECALL_DELTA_LIMIT
DEFAULT_ACTIVATION_PROBE_MAX_LENGTH = 512
DEFAULT_WEBGPU_GATE_SEED = 3939

# Operator types kept in fp32 on top of the ONNX Runtime converter's own block
# list. Normalization, softmax and reductions accumulate over many elements,
# and ``Pow``/``Sqrt`` carry the squared activations of decomposed norms; these
# are the usual fp16 overflow sites in transformer encoders.
DEFAULT_FP32_KEEP_OPS = (
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "SkipLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "Softmax",
    "ReduceMean",
    "ReduceSum",
    "ReduceSumSquare",
    "ReduceL2",
    "ReduceLogSumExp",
    "Pow",
    "Sqrt",
)

FP32_KEEP_OPS_METADATA_KEY = "openmed.webgpu.fp32_keep_ops"
FP32_KEEP_NODES_METADATA_KEY = "openmed.webgpu.fp32_keep_nodes"

REASON_NO_INPUTS = "no_evaluation_inputs"
REASON_REFERENCE_RUNTIME_ERROR = "reference_runtime_error"
REASON_FP16_RUNTIME_ERROR = "fp16_runtime_error"
REASON_NON_FINITE_LOGITS = "non_finite_logits"
REASON_LOGIT_TOLERANCE = "logit_tolerance_exceeded"
REASON_TOKEN_MISMATCH = "token_mismatch_rate_exceeded"
REASON_CRITICAL_RECALL = "critical_recall_delta_exceeded"
REASON_ACTIVATION_OVERFLOW = "fp16_activation_overflow"
REASON_ACTIVATION_PROBE_FAILED = "activation_probe_failed"
REASON_CORPUS_UNAVAILABLE = "validation_corpus_unavailable"

_WEBGPU_FORMAT_ALIASES = frozenset({"webgpu", "onnx-webgpu", "webgpu-onnx"})
_PROBE_CHUNK_SIZE = 32
_ONNX_FLOAT = 1
_ONNX_FLOAT16 = 10


class WebGpuFp16GateError(RuntimeError):
    """Raised when the fp16 WebGPU gate fails under the ``fail`` policy.

    Attributes:
        report: Value-free gate evidence for the rejected artifact.
    """

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        reasons = ", ".join(str(item) for item in self.report.get("reasons") or [])
        super().__init__(
            "fp16 WebGPU parity gate failed: " + (reasons or "unknown_reason")
        )


@dataclass(frozen=True)
class WebGpuFp16GateConfig:
    """Tolerances, fp32 keep-list and failure policy for the fp16 WebGPU gate.

    Attributes:
        logit_atol: Absolute logit tolerance. A logit passes when
            ``|fp32 - fp16| <= logit_atol + logit_rtol * |fp32|``.
        logit_rtol: Relative logit tolerance.
        max_token_mismatch_rate: Largest share of evaluated tokens whose argmax
            label may differ between the fp32 and fp16 graphs.
        critical_recall_delta_limit: Largest per-label span recall loss allowed
            for G1/G2 labels, measured against the fp32 spans.
        activation_limit: Largest magnitude an activation stored in fp16 may
            reach on the probe inputs.
        activation_probe_max_length: Longest evaluated sequence length used by
            the activation probe.
        keep_fp32_ops: Operator types kept in fp32, in addition to the ONNX
            Runtime converter's default block list.
        keep_fp32_nodes: Node names kept in fp32.
        on_failure: ``"fail"`` raises :class:`WebGpuFp16GateError`; ``"omit"``
            drops the WebGPU artifact and records the reason in the manifest.
        seed: Seed for the deterministic synthetic token inputs.
    """

    logit_atol: float = DEFAULT_FP16_LOGIT_ATOL
    logit_rtol: float = DEFAULT_FP16_LOGIT_RTOL
    max_token_mismatch_rate: float = DEFAULT_FP16_MAX_TOKEN_MISMATCH_RATE
    critical_recall_delta_limit: float = DEFAULT_FP16_CRITICAL_RECALL_DELTA_LIMIT
    activation_limit: float = FP16_MAX_FINITE
    activation_probe_max_length: int = DEFAULT_ACTIVATION_PROBE_MAX_LENGTH
    keep_fp32_ops: tuple[str, ...] = DEFAULT_FP32_KEEP_OPS
    keep_fp32_nodes: tuple[str, ...] = ()
    on_failure: str = WEBGPU_GATE_POLICY_FAIL
    seed: int = DEFAULT_WEBGPU_GATE_SEED

    def __post_init__(self) -> None:
        policy = str(self.on_failure).strip().lower()
        if policy not in WEBGPU_GATE_POLICIES:
            raise ValueError("on_failure must be 'fail' or 'omit'")
        object.__setattr__(self, "on_failure", policy)
        for name in (
            "logit_atol",
            "logit_rtol",
            "max_token_mismatch_rate",
            "critical_recall_delta_limit",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, value)
        if self.max_token_mismatch_rate > 1:
            raise ValueError("max_token_mismatch_rate must be at most 1")
        limit = float(self.activation_limit)
        if not math.isfinite(limit) or limit <= 0 or limit > FP16_MAX_FINITE:
            raise ValueError("activation_limit must be in (0, 65504]")
        object.__setattr__(self, "activation_limit", limit)
        if int(self.activation_probe_max_length) <= 0:
            raise ValueError("activation_probe_max_length must be positive")
        object.__setattr__(
            self, "activation_probe_max_length", int(self.activation_probe_max_length)
        )
        object.__setattr__(self, "keep_fp32_ops", _clean_names(self.keep_fp32_ops))
        object.__setattr__(self, "keep_fp32_nodes", _clean_names(self.keep_fp32_nodes))
        object.__setattr__(self, "seed", int(self.seed))

    def tolerances(self) -> dict[str, Any]:
        """Return the manifest view of the gate tolerances."""

        return {
            "logit_atol": self.logit_atol,
            "logit_rtol": self.logit_rtol,
            "max_token_mismatch_rate": self.max_token_mismatch_rate,
            "critical_recall_delta_limit": self.critical_recall_delta_limit,
            "critical_labels": "G1_G2",
            "span_tolerance_tokens": 0,
            "activation_limit": self.activation_limit,
        }


@dataclass(frozen=True)
class Fp16ParityObservation:
    """fp32 reference and fp16 candidate logits for one evaluated input batch.

    Attributes:
        source: ``"synthetic"`` or ``"validation_texts"``.
        reference_logits: One ``[tokens, labels]`` matrix per row, restricted to
            attended tokens.
        candidate_logits: The fp16 graph's matrices, aligned with
            ``reference_logits``.
        requested_length: Synthetic sequence length, when applicable.
    """

    source: str
    reference_logits: Sequence[Any]
    candidate_logits: Sequence[Any]
    requested_length: int | None = None


def evaluate_webgpu_fp16_gate(
    observations: Sequence[Fp16ParityObservation],
    *,
    gate_config: WebGpuFp16GateConfig | None = None,
    id2label: Mapping[Any, str] | None = None,
    activation_probe: Mapping[str, Any] | None = None,
    runtime_errors: Sequence[Mapping[str, Any]] = (),
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score fp16-vs-fp32 observations and return manifest-ready gate evidence.

    Args:
        observations: Logit pairs produced by running both graphs on the same
            inputs. Rows hold attended tokens only.
        gate_config: Tolerances and policy. Defaults to
            :class:`WebGpuFp16GateConfig`.
        id2label: Model label map used to decode BIO spans.
        activation_probe: Result of the fp16 activation-range probe. A missing
            or failed probe fails the gate closed.
        runtime_errors: Value-free runtime failures, each with ``stage`` set to
            ``"reference"``, ``"candidate"`` or ``"corpus"``.
        evidence: Extra value-free fields to record, such as artifact digests
            and evaluated lengths.

    Returns:
        A JSON-serializable report with ``passed``, ``reasons``, tolerances,
        aggregate deltas, per-label recall deltas and per-input results.
    """

    config = gate_config or WebGpuFp16GateConfig()
    label_map = _normalise_id2label(id2label)
    results: list[dict[str, Any]] = []
    span_reference: Counter[str] = Counter()
    span_candidate: Counter[str] = Counter()
    span_matched: Counter[str] = Counter()
    tokens_evaluated = 0
    token_mismatches = 0
    non_finite_values = 0
    reference_non_finite_values = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    within_tolerance = True
    spans_identical = True

    for observation in observations:
        reference_rows = [_as_matrix(row) for row in observation.reference_logits]
        candidate_rows = [_as_matrix(row) for row in observation.candidate_logits]
        if not reference_rows or len(reference_rows) != len(candidate_rows):
            within_tolerance = False
            spans_identical = False
            results.append(
                {
                    "source": observation.source,
                    "requested_length": observation.requested_length,
                    "passed": False,
                    "error": "shape_mismatch",
                }
            )
            continue

        rows_label_map = label_map or _fallback_label_map(reference_rows)
        label_info = build_label_info(dict(rows_label_map))
        offsets = [
            [[index, index + 1] for index in range(len(row))] for row in reference_rows
        ]
        parity = evaluate_onnx_logit_parity(
            reference_rows,
            candidate_rows,
            id2label=rows_label_map,
            offsets=offsets,
            rtol=config.logit_rtol,
            atol=config.logit_atol,
        )
        row_non_finite = sum(_count_non_finite(row) for row in candidate_rows)
        row_reference_non_finite = sum(_count_non_finite(row) for row in reference_rows)
        observation_tokens = 0
        observation_mismatches = 0
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            reference_ids = [_argmax(token) for token in reference_row]
            candidate_ids = [_argmax(token) for token in candidate_row]
            if len(reference_ids) != len(candidate_ids):
                observation_tokens += max(len(reference_ids), len(candidate_ids))
                observation_mismatches += max(len(reference_ids), len(candidate_ids))
                continue
            observation_tokens += len(reference_ids)
            observation_mismatches += sum(
                1 for left, right in zip(reference_ids, candidate_ids) if left != right
            )
            reference_spans = _decode_spans(reference_ids, label_info)
            candidate_spans = _decode_spans(candidate_ids, label_info)
            for key, count in reference_spans.items():
                span_reference[key[0]] += count
                span_matched[key[0]] += min(count, candidate_spans.get(key, 0))
            for key, count in candidate_spans.items():
                span_candidate[key[0]] += count

        finite_parity = parity.source != "shape_mismatch" and row_non_finite == 0
        observation_within = parity.logits_within_tolerance and finite_parity
        within_tolerance = within_tolerance and observation_within
        spans_identical = spans_identical and parity.spans_identical and finite_parity
        if finite_parity:
            max_abs_diff = max(max_abs_diff, parity.max_abs_diff)
            max_rel_diff = max(max_rel_diff, parity.max_rel_diff)
        tokens_evaluated += observation_tokens
        token_mismatches += observation_mismatches
        non_finite_values += row_non_finite
        reference_non_finite_values += row_reference_non_finite
        results.append(
            {
                "source": observation.source,
                "requested_length": observation.requested_length,
                "rows": len(reference_rows),
                "passed": bool(
                    observation_within
                    and observation_mismatches == 0
                    and parity.spans_identical
                ),
                "logits_within_tolerance": bool(observation_within),
                "max_abs_diff": _finite_or_none(parity.max_abs_diff)
                if finite_parity
                else None,
                "max_rel_diff": _finite_or_none(parity.max_rel_diff)
                if finite_parity
                else None,
                "non_finite_values": row_non_finite,
                "tokens_evaluated": observation_tokens,
                "token_mismatches": observation_mismatches,
                "spans_identical": bool(parity.spans_identical and finite_parity),
                "span_count": parity.span_count,
            }
        )

    per_label: dict[str, dict[str, Any]] = {}
    critical_labels: list[str] = []
    max_critical_delta = 0.0
    for label in sorted(set(span_reference) | set(span_candidate)):
        reference_count = span_reference.get(label, 0)
        matched_count = span_matched.get(label, 0)
        delta = (
            (reference_count - matched_count) / reference_count
            if reference_count
            else 0.0
        )
        critical = normalize_label(label) in G1_G2_LABELS
        if critical:
            critical_labels.append(label)
            max_critical_delta = max(max_critical_delta, delta)
        per_label[label] = {
            "critical": critical,
            "reference_spans": reference_count,
            "candidate_spans": span_candidate.get(label, 0),
            "matched_spans": matched_count,
            "recall_delta": delta,
        }

    total_reference = sum(span_reference.values())
    total_matched = sum(span_matched.values())
    mismatch_rate = token_mismatches / tokens_evaluated if tokens_evaluated else 0.0
    probe = dict(activation_probe) if activation_probe else {"status": "missing"}

    reasons: list[str] = []
    stages = {str(item.get("stage")) for item in runtime_errors}
    if "reference" in stages:
        reasons.append(REASON_REFERENCE_RUNTIME_ERROR)
    if "candidate" in stages:
        reasons.append(REASON_FP16_RUNTIME_ERROR)
    if "corpus" in stages:
        reasons.append(REASON_CORPUS_UNAVAILABLE)
    if not results:
        reasons.append(REASON_NO_INPUTS)
    if non_finite_values:
        reasons.append(REASON_NON_FINITE_LOGITS)
    if results and not within_tolerance:
        reasons.append(REASON_LOGIT_TOLERANCE)
    if mismatch_rate > config.max_token_mismatch_rate:
        reasons.append(REASON_TOKEN_MISMATCH)
    if max_critical_delta > config.critical_recall_delta_limit + 1e-12:
        reasons.append(REASON_CRITICAL_RECALL)
    if probe.get("status") != "computed":
        reasons.append(REASON_ACTIVATION_PROBE_FAILED)
    elif int(probe.get("overflow_tensor_count") or 0) > 0:
        reasons.append(REASON_ACTIVATION_OVERFLOW)

    passed = not reasons
    if passed:
        action = "emitted"
    elif config.on_failure == WEBGPU_GATE_POLICY_OMIT:
        action = "omitted"
    else:
        action = "rejected"

    report: dict[str, Any] = {
        "gate_version": WEBGPU_FP16_GATE_VERSION,
        "passed": passed,
        "reasons": reasons,
        "policy": config.on_failure,
        "action": action,
        "tolerances": config.tolerances(),
    }
    for key, value in dict(evidence or {}).items():
        report.setdefault(str(key), value)
    report.update(
        {
            "logits": {
                "within_tolerance": bool(results) and within_tolerance,
                "max_abs_diff": _finite_or_none(max_abs_diff),
                "max_rel_diff": _finite_or_none(max_rel_diff),
                "non_finite_values": non_finite_values,
                "reference_non_finite_values": reference_non_finite_values,
            },
            "tokens": {
                "evaluated": tokens_evaluated,
                "mismatches": token_mismatches,
                "mismatch_rate": mismatch_rate,
            },
            "spans": {
                "identical": bool(results) and spans_identical,
                "reference": total_reference,
                "candidate": sum(span_candidate.values()),
                "matched": total_matched,
                "agreement": (
                    total_matched / total_reference if total_reference else None
                ),
            },
            "per_label": per_label,
            "critical_labels": critical_labels,
            "max_critical_recall_delta": max_critical_delta,
            "activation_probe": probe,
            "runtime_errors": [dict(item) for item in runtime_errors],
            "results": results,
        }
    )
    return report


def validate_webgpu_fp16_export(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    lengths: Sequence[int],
    config: Mapping[str, Any] | None = None,
    validation_texts: Sequence[str] | None = None,
    sample_texts: Sequence[str] = (),
    tokenizer: Any = None,
    tokenizer_loader: Callable[[], Any] | None = None,
    gate_config: WebGpuFp16GateConfig | None = None,
) -> dict[str, Any]:
    """Run the fp16 WebGPU parity gate on ONNX Runtime CPU.

    Both graphs run on deterministic synthetic token inputs at every requested
    length (one full row and one padded row per length), plus a text corpus
    made of ``sample_texts`` and ``validation_texts``. The fp32 reference is
    also probed for activations the fp16 artifact stores in fp16.

    Args:
        reference_path: The fp32 ONNX graph that was converted.
        candidate_path: The fp16 WebGPU ONNX graph.
        lengths: Sequence lengths to evaluate. Lengths above the model's
            ``max_sequence_length`` (or ``max_position_embeddings``) are skipped
            and recorded.
        config: Exported model configuration (``id2label``, ``vocab_size`` and
            maximum length are read from it).
        validation_texts: Optional corpus already used for optimized-graph
            parity. Texts are tokenized in memory and never recorded. The gate
            fails closed when they cannot be tokenized.
        sample_texts: Synthetic notes evaluated when a tokenizer is available
            and skipped (with ``corpus_status`` recorded) otherwise.
        tokenizer: Tokenizer for the text corpus.
        tokenizer_loader: Called once to obtain a tokenizer when the corpus is
            non-empty and ``tokenizer`` is ``None``.
        gate_config: Tolerances and policy.

    Returns:
        Value-free gate evidence; see :func:`evaluate_webgpu_fp16_gate`.
    """

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise_missing_backend("onnx", feature="fp16 WebGPU parity gate", cause=exc)

    gate_config = gate_config or WebGpuFp16GateConfig()
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    model_config = dict(config or {})
    max_length = _max_sequence_length(model_config)
    requested = _dedupe_ints(lengths)
    evaluated_lengths = [
        length for length in requested if max_length is None or length <= max_length
    ]
    skipped_lengths = [
        length for length in requested if length not in evaluated_lengths
    ]
    required_texts = [str(text) for text in (validation_texts or ()) if str(text)]
    texts = [str(text) for text in sample_texts if str(text)] + required_texts
    if texts and tokenizer is None and tokenizer_loader is not None:
        try:
            tokenizer = tokenizer_loader()
        except Exception:  # noqa: BLE001 - handled as a missing tokenizer.
            tokenizer = None
    corpus_status = "evaluated" if texts and tokenizer is not None else "none"
    if texts and tokenizer is None:
        corpus_status = (
            "tokenizer_unavailable" if required_texts else "skipped_no_tokenizer"
        )

    candidate_model = onnx.load(str(candidate_path), load_external_data=False)
    evidence: dict[str, Any] = {
        "artifact": candidate_path.name,
        "artifact_sha256": _sha256_file(candidate_path),
        "reference": reference_path.name,
        "reference_sha256": _sha256_file(reference_path),
        "runtime": {
            "engine": "onnxruntime",
            "version": str(getattr(ort, "__version__", "unknown")),
            "execution_provider": WEBGPU_GATE_EXECUTION_PROVIDER,
            "graph_optimization": "disabled",
        },
        "seed": gate_config.seed,
        "evaluated_lengths": evaluated_lengths,
        "skipped_lengths": skipped_lengths,
        "corpus_examples": len(texts) if corpus_status == "evaluated" else 0,
        "corpus_status": corpus_status,
        "keep_fp32": _keep_fp32_record(candidate_model),
    }

    runtime_errors: list[dict[str, Any]] = []
    try:
        reference_session = _cpu_session(ort, reference_path)
    except Exception as exc:  # noqa: BLE001 - recorded as a value-free code.
        runtime_errors.append(_runtime_error("reference", None, exc))
        return evaluate_webgpu_fp16_gate(
            [],
            gate_config=gate_config,
            id2label=model_config.get("id2label"),
            runtime_errors=runtime_errors,
            evidence=evidence,
        )
    try:
        candidate_session = _cpu_session(ort, candidate_path)
    except Exception as exc:  # noqa: BLE001 - recorded as a value-free code.
        runtime_errors.append(_runtime_error("candidate", None, exc))
        return evaluate_webgpu_fp16_gate(
            [],
            gate_config=gate_config,
            id2label=model_config.get("id2label"),
            runtime_errors=runtime_errors,
            evidence=evidence,
        )

    session_inputs = reference_session.get_inputs()
    vocab_size = _vocab_size(model_config, reference_path, onnx)
    digest = hashlib.sha256()
    observations: list[Fp16ParityObservation] = []
    probe_feeds: list[tuple[int, dict[str, Any]]] = []
    probe_lengths = _probe_lengths(
        evaluated_lengths, gate_config.activation_probe_max_length
    )
    for length in evaluated_lengths:
        feed, valid_lengths = _synthetic_feed(
            session_inputs,
            sequence_length=length,
            vocab_size=vocab_size,
            seed=gate_config.seed,
            np_module=np,
        )
        for name in sorted(feed):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(feed[name]).tobytes())
        if length in probe_lengths:
            probe_feeds.append(
                (length, {name: value[:1] for name, value in feed.items()})
            )
        observation = _observe(
            reference_session,
            candidate_session,
            feed,
            valid_lengths,
            source="synthetic",
            requested_length=length,
            runtime_errors=runtime_errors,
        )
        if observation is not None:
            observations.append(observation)

    corpus: tuple[dict[str, Any], list[int]] | None = None
    if texts and tokenizer is not None:
        try:
            corpus = _corpus_feed(tokenizer, texts, session_inputs, np_module=np)
        except Exception as exc:  # noqa: BLE001 - recorded as a value-free code.
            evidence["corpus_examples"] = 0
            evidence["corpus_status"] = (
                "tokenizer_error" if required_texts else "skipped_tokenizer_error"
            )
            if required_texts:
                runtime_errors.append(_runtime_error("corpus", None, exc))
    if corpus is not None:
        feed, valid = corpus
        observation = _observe(
            reference_session,
            candidate_session,
            feed,
            valid,
            source="validation_texts",
            requested_length=None,
            runtime_errors=runtime_errors,
        )
        if observation is not None:
            observations.append(observation)
    elif required_texts and tokenizer is None:
        runtime_errors.append(
            {
                "stage": "corpus",
                "requested_length": None,
                "error_type": "tokenizer_unavailable",
            }
        )

    evidence["inputs_sha256"] = digest.hexdigest()
    activation_probe = probe_fp16_activations(
        reference_path,
        candidate_model,
        probe_feeds,
        activation_limit=gate_config.activation_limit,
    )
    return evaluate_webgpu_fp16_gate(
        observations,
        gate_config=gate_config,
        id2label=model_config.get("id2label"),
        activation_probe=activation_probe,
        runtime_errors=runtime_errors,
        evidence=evidence,
    )


def probe_fp16_activations(
    reference_path: str | Path,
    candidate_model: Any,
    feeds: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    activation_limit: float = FP16_MAX_FINITE,
) -> dict[str, Any]:
    """Measure fp32 activations that the fp16 artifact stores in fp16.

    The probe exposes those intermediate tensors on a temporary copy of the
    fp32 reference graph, runs it on ``feeds`` and reports how many exceed
    ``activation_limit``. The temporary graph sits next to the reference so
    external weights resolve, and it is removed before returning.

    Args:
        reference_path: The fp32 ONNX graph.
        candidate_model: The loaded fp16 ``ModelProto``.
        feeds: ``(sequence_length, feed)`` pairs.
        activation_limit: Largest magnitude allowed in fp16 storage.

    Returns:
        Value-free probe evidence with ``status`` set to ``"computed"`` or
        ``"failed"``.
    """

    lengths = [int(length) for length, _ in feeds]
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import helper
    except ImportError as exc:
        raise_missing_backend("onnx", feature="fp16 activation probe", cause=exc)

    if not feeds:
        return {"status": "failed", "lengths": lengths, "error": "no_probe_inputs"}

    reference_path = Path(reference_path)
    temporary_path: Path | None = None
    try:
        reference = onnx.load(str(reference_path), load_external_data=False)
        names, producers = _fp16_stored_tensors(reference, candidate_model, onnx)
        overflow_tensors: set[str] = set()
        sentinel_tensors: set[str] = set()
        max_abs = 0.0
        if names:
            probe_model = onnx.ModelProto()
            probe_model.CopyFrom(reference)
            for name in names:
                probe_model.graph.output.append(
                    helper.make_tensor_value_info(name, _ONNX_FLOAT, None)
                )
            handle, raw_path = tempfile.mkstemp(
                prefix=".openmed-fp16-probe-",
                suffix=".onnx",
                dir=str(reference_path.parent),
            )
            os.close(handle)
            temporary_path = Path(raw_path)
            onnx.save(probe_model, str(temporary_path))
            session = _cpu_session(ort, temporary_path)
            for _, feed in feeds:
                for start in range(0, len(names), _PROBE_CHUNK_SIZE):
                    chunk = names[start : start + _PROBE_CHUNK_SIZE]
                    outputs = session.run(chunk, dict(feed))
                    for name, value in zip(chunk, outputs):
                        array = np.abs(np.asarray(value, dtype=np.float32))
                        finite = array[np.isfinite(array)]
                        if finite.size != array.size:
                            sentinel_tensors.add(name)
                        regular = finite[finite < FP32_SENTINEL_MAGNITUDE]
                        if regular.size != finite.size:
                            sentinel_tensors.add(name)
                        if regular.size:
                            peak = float(regular.max())
                            max_abs = max(max_abs, peak)
                            if peak > activation_limit:
                                overflow_tensors.add(name)
    except Exception as exc:  # noqa: BLE001 - recorded as a value-free code.
        return {
            "status": "failed",
            "lengths": lengths,
            "error": "probe_runtime_error",
            "error_type": type(exc).__name__,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "status": "computed",
        "lengths": lengths,
        "rows": 1,
        "tensors_probed": len(names),
        "limit": float(activation_limit),
        "max_abs_activation": _finite_or_none(max_abs),
        "overflow_tensor_count": len(overflow_tensors),
        "overflow_op_types": sorted({producers[name] for name in overflow_tensors}),
        "sentinel_tensor_count": len(sentinel_tensors - overflow_tensors),
    }


def finalize_fp16_webgpu_graph(
    model: Any,
    *,
    source_node_names: Iterable[str],
    keep_fp32_ops: Sequence[str],
    keep_fp32_nodes: Sequence[str],
) -> dict[str, int]:
    """Repair and annotate ONNX Runtime fp16 converter output for WebGPU.

    The steps are: drop exact duplicate nodes (the converter emits one
    identically named ``Cast`` per fp32-blocked consumer of a shared tensor),
    keep chains of blocked nodes in fp32 (:func:`keep_fp32_chains`), restore
    topological order (:func:`sort_graph_topologically`) and stamp the keep-list
    into the model metadata (:func:`stamp_fp32_keep_metadata`).

    Args:
        model: Converted fp16 ``ModelProto``; modified in place.
        source_node_names: Node names of the fp32 source graph.
        keep_fp32_ops: Operator types kept in fp32 by OpenMed.
        keep_fp32_nodes: Node names kept in fp32.

    Returns:
        Counts of removed duplicate nodes and bypassed ``Cast`` pairs.
    """

    duplicates = _drop_duplicate_nodes(model)
    bypassed = keep_fp32_chains(model, source_node_names=source_node_names)
    sort_graph_topologically(model)
    stamp_fp32_keep_metadata(
        model,
        keep_fp32_ops=keep_fp32_ops,
        keep_fp32_nodes=keep_fp32_nodes,
    )
    return {"duplicate_nodes_removed": duplicates, "fp32_chain_casts_removed": bypassed}


def keep_fp32_chains(model: Any, *, source_node_names: Iterable[str]) -> int:
    """Keep chains of fp32-blocked nodes in fp32 inside a converted graph.

    The ONNX Runtime fp16 converter casts every blocked node's output back to
    fp16, even when the only consumer is another blocked node, so a value such
    as ``Pow -> ReduceMean`` still overflows between the two. This pass removes
    converter-inserted ``Cast(to=float16) -> Cast(to=float)`` pairs and feeds
    the fp32 value straight to the consumer. Casts that exist in the source
    graph are never touched.

    Args:
        model: Converted fp16 ``ModelProto``; modified in place.
        source_node_names: Node names of the fp32 source graph.

    Returns:
        The number of ``Cast`` pairs bypassed.
    """

    graph = getattr(model, "graph", None)
    if graph is None or not hasattr(graph, "node"):
        return 0
    original = {str(name) for name in source_node_names if name}
    nodes = list(graph.node)
    producers = {output: node for node in nodes for output in node.output if output}
    graph_outputs = {output.name for output in graph.output}
    float_types = {
        info.name: info.type.tensor_type.elem_type for info in graph.value_info
    }
    rewired: dict[str, str] = {}
    removed: set[int] = set()
    for node in nodes:
        if not _is_inserted_cast(node, _ONNX_FLOAT, original):
            continue
        upstream = producers.get(node.input[0]) if node.input else None
        if upstream is None or not _is_inserted_cast(upstream, _ONNX_FLOAT16, original):
            continue
        fp32_source = upstream.input[0] if upstream.input else ""
        if not fp32_source or node.output[0] in graph_outputs:
            continue
        if float_types.get(fp32_source, _ONNX_FLOAT) != _ONNX_FLOAT:
            continue
        rewired[node.output[0]] = fp32_source
        removed.add(id(node))

    if not rewired:
        return 0
    for node in nodes:
        for index, name in enumerate(node.input):
            if name in rewired:
                node.input[index] = rewired[name]

    consumed = {
        name for node in nodes if id(node) not in removed for name in node.input
    } | graph_outputs
    for node in nodes:
        if (
            id(node) not in removed
            and _is_inserted_cast(node, _ONNX_FLOAT16, original)
            and node.output[0] not in consumed
        ):
            removed.add(id(node))

    dropped_tensors = {
        output for node in nodes if id(node) in removed for output in node.output
    }
    kept_nodes = [node for node in nodes if id(node) not in removed]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    kept_info = [info for info in graph.value_info if info.name not in dropped_tensors]
    del graph.value_info[:]
    graph.value_info.extend(kept_info)
    return len(rewired)


def sort_graph_topologically(model: Any) -> None:
    """Order a graph's nodes so every input is produced before it is consumed.

    The ONNX Runtime fp16 converter appends the ``Cast`` nodes it inserts for
    fp32-blocked nodes at the end of the graph, which the ONNX checker rejects.
    The sort is stable: nodes keep their original relative order whenever the
    data dependencies allow it.

    Args:
        model: ``ModelProto`` whose top-level graph is reordered in place.
    """

    graph = getattr(model, "graph", None)
    if graph is None or not hasattr(graph, "node"):
        return
    nodes = list(graph.node)
    producer_index = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    waiting_on: list[int] = []
    dependents: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        upstream = {
            producer_index[name]
            for name in node.input
            if name in producer_index and producer_index[name] != index
        }
        waiting_on.append(len(upstream))
        for parent in upstream:
            dependents.setdefault(parent, []).append(index)
    ready = [index for index, count in enumerate(waiting_on) if count == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        order.append(index)
        for child in dependents.get(index, ()):
            waiting_on[child] -= 1
            if waiting_on[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        placed = set(order)
        order.extend(index for index in range(len(nodes)) if index not in placed)
    del graph.node[:]
    graph.node.extend(nodes[index] for index in order)


def stamp_fp32_keep_metadata(
    model: Any,
    *,
    keep_fp32_ops: Sequence[str],
    keep_fp32_nodes: Sequence[str],
) -> None:
    """Record the fp32 keep-list in the converted model's metadata properties.

    Args:
        model: Converted fp16 ``ModelProto``; modified in place.
        keep_fp32_ops: Operator types kept in fp32 by OpenMed.
        keep_fp32_nodes: Node names kept in fp32.
    """

    props = getattr(model, "metadata_props", None)
    if props is None:
        return
    values = {
        FP32_KEEP_OPS_METADATA_KEY: json.dumps(list(_clean_names(keep_fp32_ops))),
        FP32_KEEP_NODES_METADATA_KEY: json.dumps(list(_clean_names(keep_fp32_nodes))),
    }
    for prop in props:
        if prop.key in values:
            prop.value = values.pop(prop.key)
    for key, value in values.items():
        entry = props.add()
        entry.key = key
        entry.value = value


def verify_webgpu_fp16_gate_evidence(
    root: str | Path,
    manifest: Mapping[str, Any],
    *,
    require: bool = False,
) -> Mapping[str, Any] | None:
    """Require a passing, digest-bound fp16 gate record for WebGPU artifacts.

    Args:
        root: Export directory holding the manifest's artifacts.
        manifest: Parsed ``openmed-onnx.json``.
        require: Treat the bundle as declaring WebGPU even when the manifest
            does not, for example because a publisher requested the format.

    Returns:
        The gate record, or ``None`` when no WebGPU artifact is declared.

    Raises:
        ValueError: When the record is missing, failed, incomplete, or does not
            match the artifact bytes. Messages are value-free.
    """

    root = Path(root)
    artifacts = [
        item
        for item in manifest.get("artifacts") or []
        if isinstance(item, Mapping) and is_webgpu_format(item.get("format"))
    ]
    declared = bool(artifacts) or any(
        is_webgpu_format(item) for item in manifest.get("formats") or []
    )
    if not (declared or require):
        return None

    gate = manifest.get(WEBGPU_FP16_GATE_KEY)
    if not isinstance(gate, Mapping):
        raise ValueError("WebGPU artifact requires an fp16 parity gate record")
    if gate.get("passed") is not True:
        raise ValueError("WebGPU artifact fp16 parity gate did not pass")
    if (
        not isinstance(gate.get("tolerances"), Mapping)
        or not isinstance(gate.get("evaluated_lengths"), list)
        or not isinstance(gate.get("artifact_sha256"), str)
    ):
        raise ValueError("WebGPU fp16 parity gate record is incomplete")
    if not artifacts:
        raise ValueError("WebGPU format declared without a gated artifact")

    resolved_root = root.resolve(strict=False)
    for artifact in artifacts:
        relpath = str(artifact.get("path") or "")
        path = (root / relpath).resolve(strict=False)
        if not relpath or resolved_root not in (path, *path.parents):
            raise ValueError("WebGPU artifact path must stay inside the bundle")
        if not path.is_file():
            raise ValueError("WebGPU artifact named in the manifest is missing")
        if _sha256_file(path) != gate["artifact_sha256"]:
            raise ValueError(
                "WebGPU artifact digest does not match its fp16 parity gate record"
            )
    return gate


def is_webgpu_format(value: Any) -> bool:
    """Return whether *value* names the WebGPU ONNX runtime format."""

    return str(value or "").strip().lower().replace("_", "-") in (
        _WEBGPU_FORMAT_ALIASES
    )


def _observe(
    reference_session: Any,
    candidate_session: Any,
    feed: Mapping[str, Any],
    valid_lengths: Sequence[int],
    *,
    source: str,
    requested_length: int | None,
    runtime_errors: list[dict[str, Any]],
) -> Fp16ParityObservation | None:
    try:
        reference = reference_session.run(None, dict(feed))[0]
    except Exception as exc:  # noqa: BLE001 - recorded as a value-free code.
        runtime_errors.append(_runtime_error("reference", requested_length, exc))
        return None
    try:
        candidate = candidate_session.run(None, dict(feed))[0]
    except Exception as exc:  # noqa: BLE001 - recorded as a value-free code.
        runtime_errors.append(_runtime_error("candidate", requested_length, exc))
        return None
    return Fp16ParityObservation(
        source=source,
        requested_length=requested_length,
        reference_logits=[
            reference[row, :valid] for row, valid in enumerate(valid_lengths)
        ],
        candidate_logits=[
            candidate[row, :valid] for row, valid in enumerate(valid_lengths)
        ],
    )


def _drop_duplicate_nodes(model: Any) -> int:
    graph = getattr(model, "graph", None)
    if graph is None or not hasattr(graph, "node"):
        return 0
    seen: set[bytes] = set()
    kept: list[Any] = []
    for node in graph.node:
        key = node.SerializeToString(deterministic=True)
        if key in seen:
            continue
        seen.add(key)
        kept.append(node)
    removed = len(graph.node) - len(kept)
    if removed:
        del graph.node[:]
        graph.node.extend(kept)
        infos: dict[str, Any] = {}
        for info in graph.value_info:
            infos.setdefault(info.name, info)
        if len(infos) != len(graph.value_info):
            unique = list(infos.values())
            del graph.value_info[:]
            graph.value_info.extend(unique)
    return removed


def _runtime_error(
    stage: str, requested_length: int | None, exc: BaseException
) -> dict[str, Any]:
    return {
        "stage": stage,
        "requested_length": requested_length,
        "error_type": type(exc).__name__,
    }


def _cpu_session(ort_module: Any, path: Path) -> Any:
    options = ort_module.SessionOptions()
    options.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort_module.InferenceSession(
        str(path),
        sess_options=options,
        providers=[WEBGPU_GATE_EXECUTION_PROVIDER],
    )


def _synthetic_feed(
    session_inputs: Sequence[Any],
    *,
    sequence_length: int,
    vocab_size: int,
    seed: int,
    np_module: Any,
) -> tuple[dict[str, Any], list[int]]:
    rng = np_module.random.default_rng([int(seed), int(sequence_length)])
    upper = max(int(vocab_size), 1)
    input_ids = rng.integers(0, upper, size=(2, sequence_length), dtype=np_module.int64)
    attention_mask = np_module.ones((2, sequence_length), dtype=np_module.int64)
    padded = max(1, sequence_length - sequence_length // 4)
    attention_mask[1, padded:] = 0
    input_ids[1, padded:] = 0
    feed: dict[str, Any] = {}
    for input_info in session_inputs:
        name = getattr(input_info, "name", "")
        if name == "input_ids":
            feed[name] = input_ids
        elif name == "attention_mask":
            feed[name] = attention_mask
        else:
            feed[name] = np_module.zeros((2, sequence_length), dtype=np_module.int64)
    return feed, [sequence_length, padded]


def _corpus_feed(
    tokenizer: Any,
    texts: Sequence[str],
    session_inputs: Sequence[Any],
    *,
    np_module: Any,
) -> tuple[dict[str, Any], list[int]]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    feed = _feed_for_inputs(session_inputs, encoded, np_module=np_module)
    if not feed:
        raise ValueError("tokenizer output matches no graph input")
    mask = feed.get("attention_mask")
    if mask is None:
        first = next(iter(feed.values()))
        return feed, [int(first.shape[1])] * int(first.shape[0])
    return feed, [int(value) for value in np_module.asarray(mask).sum(axis=1)]


def _feed_for_inputs(
    session_inputs: Sequence[Any],
    encoded: Mapping[str, Any],
    *,
    np_module: Any,
) -> dict[str, Any]:
    feed: dict[str, Any] = {}
    shape: tuple[int, ...] | None = None
    for input_info in session_inputs:
        name = getattr(input_info, "name", "")
        if name in encoded:
            feed[name] = np_module.asarray(encoded[name], dtype=np_module.int64)
            shape = feed[name].shape
    for input_info in session_inputs:
        name = getattr(input_info, "name", "")
        if name not in feed and shape is not None:
            feed[name] = np_module.zeros(shape, dtype=np_module.int64)
    return feed


def _probe_lengths(lengths: Sequence[int], max_length: int) -> set[int]:
    eligible = [length for length in lengths if length <= max_length]
    if not eligible:
        return {min(lengths)} if lengths else set()
    return {min(eligible), max(eligible)}


def _fp16_stored_tensors(
    reference: Any,
    candidate: Any,
    onnx_module: Any,
) -> tuple[list[str], dict[str, str]]:
    producers = {
        output: str(node.op_type)
        for node in reference.graph.node
        for output in node.output
        if output
    }
    graph_outputs = {output.name for output in reference.graph.output}
    reference_types: dict[str, int] = {}
    try:
        inferred = onnx_module.shape_inference.infer_shapes(reference)
        reference_types = {
            info.name: info.type.tensor_type.elem_type
            for info in inferred.graph.value_info
        }
    except Exception:  # noqa: BLE001 - fall back to converter type evidence.
        reference_types = {}
    stored = {
        info.name
        for info in candidate.graph.value_info
        if info.type.tensor_type.elem_type == _ONNX_FLOAT16
    }
    names = sorted(
        name
        for name in stored
        if name in producers
        and name not in graph_outputs
        and reference_types.get(name, _ONNX_FLOAT) == _ONNX_FLOAT
    )
    return names, producers


def _keep_fp32_record(model: Any) -> dict[str, Any]:
    props = {prop.key: prop.value for prop in getattr(model, "metadata_props", [])}
    ops = _json_names(props.get(FP32_KEEP_OPS_METADATA_KEY))
    nodes = _json_names(props.get(FP32_KEEP_NODES_METADATA_KEY))
    recorded = FP32_KEEP_OPS_METADATA_KEY in props
    blocked_ops = set(ops)
    blocked_nodes = set(nodes)
    kept = sum(
        1
        for node in model.graph.node
        if node.op_type in blocked_ops or (node.name and node.name in blocked_nodes)
    )
    return {
        "source": "artifact_metadata" if recorded else "not_recorded",
        "op_types": list(ops),
        "node_names": list(nodes),
        "converter_default_block_list": recorded,
        "kept_node_count": kept,
    }


def _json_names(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except ValueError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return _clean_names(parsed)


def _vocab_size(
    config: Mapping[str, Any], reference_path: Path, onnx_module: Any
) -> int:
    value = config.get("vocab_size")
    if isinstance(value, int) and value > 0:
        return value
    try:
        model = onnx_module.load(str(reference_path), load_external_data=False)
    except Exception:  # noqa: BLE001 - fall back to a single safe token id.
        return 1
    initializers = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        if "input_ids" not in node.input:
            continue
        for name in node.input:
            tensor = initializers.get(name)
            if tensor is not None and len(tensor.dims) == 2 and tensor.dims[0] > 0:
                return int(tensor.dims[0])
    return 1


def _max_sequence_length(config: Mapping[str, Any]) -> int | None:
    for key in ("max_sequence_length", "max_position_embeddings"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _is_inserted_cast(node: Any, target: int, original: set[str]) -> bool:
    if node.op_type != "Cast" or not node.name or node.name in original:
        return False
    if len(node.input) != 1 or len(node.output) != 1:
        return False
    return any(
        attribute.name == "to" and int(attribute.i) == target
        for attribute in node.attribute
    )


def _decode_spans(
    token_ids: Sequence[int],
    label_info: Any,
) -> Counter[tuple[str, int, int]]:
    spans = labels_to_token_spans(dict(enumerate(token_ids)), label_info)
    return Counter(
        (label_info.span_class_names[label], start, end) for label, start, end in spans
    )


def _normalise_id2label(id2label: Mapping[Any, str] | None) -> dict[int, str] | None:
    if not isinstance(id2label, Mapping) or not id2label:
        return None
    try:
        labels = {int(key): str(value) for key, value in id2label.items()}
    except (TypeError, ValueError):
        return None
    if sorted(labels) != list(range(len(labels))):
        return None
    return labels


def _fallback_label_map(rows: Sequence[Sequence[Sequence[float]]]) -> dict[int, str]:
    width = 0
    for row in rows:
        for token in row:
            width = max(width, len(token))
    return {index: f"LABEL_{index}" for index in range(max(width, 1))}


def _as_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(item) for item in token] for token in value]


def _argmax(values: Sequence[float]) -> int:
    best_index = 0
    best_value = -math.inf
    for index, value in enumerate(values):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _count_non_finite(rows: Sequence[Sequence[float]]) -> int:
    return sum(1 for token in rows for value in token if not math.isfinite(value))


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        number = int(value)
        if number <= 0:
            raise ValueError("gate sequence lengths must be positive")
        if number not in result:
            result.append(number)
    return result


def _clean_names(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        name = str(value).strip()
        if name and name not in result:
            result.append(name)
    return tuple(result)


__all__ = [
    "DEFAULT_FP32_KEEP_OPS",
    "FP16_MAX_FINITE",
    "WEBGPU_FP16_GATE_KEY",
    "WEBGPU_GATE_POLICIES",
    "WEBGPU_GATE_POLICY_FAIL",
    "WEBGPU_GATE_POLICY_OMIT",
    "Fp16ParityObservation",
    "WebGpuFp16GateConfig",
    "WebGpuFp16GateError",
    "evaluate_webgpu_fp16_gate",
    "finalize_fp16_webgpu_graph",
    "is_webgpu_format",
    "keep_fp32_chains",
    "probe_fp16_activations",
    "sort_graph_topologically",
    "stamp_fp32_keep_metadata",
    "validate_webgpu_fp16_export",
    "verify_webgpu_fp16_gate_evidence",
]
