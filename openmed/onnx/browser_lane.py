"""Browser execution lane for exported token-classification ONNX graphs.

The lane proves that ONNX Runtime Web executes the graphs produced by
:func:`openmed.onnx.convert.convert`, not only the WGSL classification head.
:func:`prepare_browser_lane` builds a tiny, randomly initialised BERT token
classifier fully offline, exports its fp32 ``model.onnx`` and fp16
``model.webgpu.onnx`` graphs with the repository exporter, derives a negative
control graph that carries an operator the WebGPU provider does not implement,
and writes synthetic token feeds plus Python ONNX Runtime reference logits to
a ``lane-fixture.json`` file.

The Playwright lane under ``tests/browser/ort-lane`` serves those files,
ONNX Runtime Web from the local ``onnxruntime-web`` package, and the OpenMed
browser runtime from one cross-origin-isolated local origin. It runs each case
on the ``webgpu``, threaded ``wasm`` or single-thread ``wasm`` provider and
writes one value-free record per case. :func:`validate_lane_record` checks the
record contract and :func:`evaluate_lane_records` applies the release gate.

Records hold controlled codes, counts, tolerances, measured deltas, versions
and SHA-256 digests only. They never contain logits, token ids, text, URLs,
file paths or raw runtime logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from openmed.core.capabilities import raise_missing_backend
from openmed.eval.quant_delta import evaluate_onnx_logit_parity
from openmed.onnx.openvino_export import run_onnx_reference_logits

LANE_ID = "ort-web-token-classification"
FIXTURE_FORMAT = "openmed-ort-browser-lane-fixture"
FIXTURE_VERSION = 1
RECORD_FORMAT = "openmed-ort-browser-lane-record"
RECORD_VERSION = 1
FIXTURE_FILENAME = "lane-fixture.json"
GRAPHS_DIRNAME = "graphs"
CHECKPOINT_DIRNAME = "checkpoint"
FALLBACK_CONTROL_FILENAME = "model.fallback-control.onnx"
DEFAULT_OUTPUT_DIR = Path("output") / "playwright" / "ort-lane" / "fixture"
DEFAULT_RECORDS_DIR = Path("output") / "playwright" / "ort-lane" / "records"

DEFAULT_SEED = 3940
DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQUENCE_LENGTH = 12
DEFAULT_PADDING_TOKENS = 3
MAX_FEED_ATTEMPTS = 64
CONTROL_IDENTITY_TOLERANCE = 1e-6
CLASSIFIER_INIT_STD = 1.0

SYNTHETIC_LABELS = ("O", "B-NAME", "I-NAME", "B-ID", "I-ID")
SYNTHETIC_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
FIRST_REGULAR_TOKEN_ID = len(SYNTHETIC_SPECIAL_TOKENS)

#: Operator injected into the negative-control graph. ONNX Runtime Web's
#: WebGPU provider registers no kernel for it, so session creation must assign
#: the node to the WASM CPU provider and the lane must report partial support.
FALLBACK_CONTROL_OP_TYPE = "Mod"
_FALLBACK_CONTROL_DIVISOR = 65504.0

PROVIDERS = ("webgpu", "wasm-threaded", "wasm-single")
GRAPH_IDS = ("fp32", "fp16", "fallback-control")
RECORD_STATUSES = ("passed", "failed", "unavailable")
WEBGPU_SUPPORT_LEVELS = ("full", "partial", "unknown", "not_applicable")
ISOLATION_MODES = ("cross-origin-isolated", "none")
ORT_BACKENDS = ("webgpu", "wasm-simd-threads", "wasm-basic")
RUNTIME_BUNDLES = ("webgpu", "wasm")
PLACEMENT_KEYS = ("webgpu", "webgpu-jsep", "cpu", "other")

UNAVAILABLE_REASONS = frozenset(
    {
        "webgpu_api_unavailable",
        "webgpu_adapter_unavailable",
        "webgpu_shader_f16_unavailable",
        "cross_origin_isolation_unavailable",
        "wasm_threads_unavailable",
    }
)
FAILURE_REASONS = frozenset(
    {
        "fixture_digest_mismatch",
        "session_create_failed",
        "inference_failed",
        "backend_mismatch",
        "thread_count_mismatch",
        "logit_shape_mismatch",
        "non_finite_logits",
        "logit_tolerance_exceeded",
        "reference_decode_mismatch",
        "span_mismatch",
        "recall_gate_failed",
        "certification_rejected",
        "fallback_evidence_missing",
        "unexpected_webgpu_support",
        "cross_origin_request",
        "harness_error",
    }
)
REASON_CODES = frozenset({"ok"}) | UNAVAILABLE_REASONS | FAILURE_REASONS

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"\d+(?:\.\d+){0,3}(?:[-.+][0-9A-Za-z.]+)?\Z")
_OP_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")

_RECORD_KEYS = frozenset(
    {
        "format",
        "format_version",
        "lane",
        "case_id",
        "graph",
        "precision",
        "provider",
        "browser",
        "digests",
        "tolerances",
        "outcomes",
    }
)
_PROVIDER_KEYS = frozenset(
    {
        "id",
        "backend",
        "execution_providers",
        "runtime",
        "runtime_version",
        "runtime_bundle",
        "cross_origin_isolated",
        "wasm_threads",
        "worker_count",
    }
)
_BROWSER_KEYS = frozenset({"name", "version"})
_DIGEST_KEYS = frozenset(
    {"graph_sha256", "external_data_sha256", "feeds_sha256", "reference_sha256"}
)
_TOLERANCE_KEYS = frozenset(
    {"max_abs_logit_delta", "span_tolerance_tokens", "max_recall_delta"}
)
_OUTCOME_KEYS = frozenset(
    {
        "status",
        "reason_code",
        "webgpu_support",
        "fallback_op_types",
        "fallback_node_count",
        "node_placement",
        "max_abs_logit_delta",
        "logits_within_tolerance",
        "reference_span_count",
        "candidate_span_count",
        "matched_span_count",
        "recall",
        "span_parity",
        "same_origin_request_count",
        "cross_origin_request_count",
    }
)
_MEASURED_OUTCOMES = (
    "max_abs_logit_delta",
    "logits_within_tolerance",
    "reference_span_count",
    "candidate_span_count",
    "matched_span_count",
    "recall",
    "span_parity",
)


class BrowserLaneError(ValueError):
    """A lane fixture or result record violates the lane contract.

    Messages carry controlled field names and codes only, never values.
    """


@dataclass(frozen=True)
class LaneTolerance:
    """Declared parity tolerance for one lane case.

    Attributes:
        max_abs_logit_delta: Largest allowed absolute logit difference from the
            Python ONNX Runtime fp32 reference.
        span_tolerance_tokens: Allowed token-offset drift per decoded span.
        max_recall_delta: Allowed recall loss against the reference spans.
    """

    max_abs_logit_delta: float
    span_tolerance_tokens: int = 0
    max_recall_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form used by fixtures and records."""

        return {
            "max_abs_logit_delta": self.max_abs_logit_delta,
            "span_tolerance_tokens": self.span_tolerance_tokens,
            "max_recall_delta": self.max_recall_delta,
        }


#: WASM runs the same MLAS kernels as Python ONNX Runtime with a different
#: reduction order, so fp32 logits must agree closely.
WASM_FLOAT32_TOLERANCE = LaneTolerance(max_abs_logit_delta=1e-4)
#: WebGPU shaders reorder fp32 reductions and approximate transcendental ops.
WEBGPU_FLOAT32_TOLERANCE = LaneTolerance(max_abs_logit_delta=1e-3)
#: The fp16 graph is compared with the fp32 reference; spans must still match.
WEBGPU_FLOAT16_TOLERANCE = LaneTolerance(max_abs_logit_delta=5e-2)
#: Every active reference token keeps its top label ahead of the runner-up by
#: more than twice the loosest logit tolerance, so a candidate inside its
#: logit tolerance cannot flip an argmax decision.
MIN_REFERENCE_MARGIN = 2 * WEBGPU_FLOAT16_TOLERANCE.max_abs_logit_delta


@dataclass(frozen=True)
class LaneCase:
    """One provider and graph combination executed by the browser lane.

    Attributes:
        case_id: Stable identifier used for the record file name.
        provider: One of :data:`PROVIDERS`.
        graph: One of :data:`GRAPH_IDS`.
        isolation: Whether the harness page is cross-origin isolated.
        tolerance: Declared parity tolerance.
        expected_webgpu_support: Required support level, or ``None`` to record
            the observed level for the support matrix.
    """

    case_id: str
    provider: str
    graph: str
    isolation: str
    tolerance: LaneTolerance
    expected_webgpu_support: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form consumed by the browser harness."""

        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "graph": self.graph,
            "isolation": self.isolation,
            "tolerances": self.tolerance.to_dict(),
            "expected_webgpu_support": self.expected_webgpu_support,
        }


DEFAULT_CASES: tuple[LaneCase, ...] = (
    LaneCase(
        case_id="wasm-threaded-fp32",
        provider="wasm-threaded",
        graph="fp32",
        isolation="cross-origin-isolated",
        tolerance=WASM_FLOAT32_TOLERANCE,
    ),
    LaneCase(
        case_id="wasm-single-fp32",
        provider="wasm-single",
        graph="fp32",
        isolation="none",
        tolerance=WASM_FLOAT32_TOLERANCE,
    ),
    LaneCase(
        case_id="webgpu-fp32",
        provider="webgpu",
        graph="fp32",
        isolation="cross-origin-isolated",
        tolerance=WEBGPU_FLOAT32_TOLERANCE,
    ),
    LaneCase(
        case_id="webgpu-fp16",
        provider="webgpu",
        graph="fp16",
        isolation="cross-origin-isolated",
        tolerance=WEBGPU_FLOAT16_TOLERANCE,
    ),
    LaneCase(
        case_id="webgpu-fallback-control",
        provider="webgpu",
        graph="fallback-control",
        isolation="cross-origin-isolated",
        tolerance=WEBGPU_FLOAT32_TOLERANCE,
        expected_webgpu_support="partial",
    ),
)


def synthetic_vocabulary() -> tuple[str, ...]:
    """Return the deterministic WordPiece vocabulary of the synthetic model."""

    letters = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    digits = [str(digit) for digit in range(10)]
    return (
        *SYNTHETIC_SPECIAL_TOKENS,
        *letters,
        *digits,
        "-",
        ".",
        ",",
        *(f"##{piece}" for piece in (*letters, *digits)),
    )


def write_synthetic_checkpoint(
    directory: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write a tiny randomly initialised BERT token classifier, offline.

    The checkpoint has two encoder layers, 32 hidden units and the labels in
    :data:`SYNTHETIC_LABELS`. Weights come from a seeded generator, and the
    classifier head is drawn with a wider spread so that argmax decisions have
    clear margins. Nothing is downloaded.

    Args:
        directory: Destination for ``config.json``, weights and tokenizer files.
        seed: Seed for every random draw.

    Returns:
        The checkpoint directory.
    """

    try:
        import torch
        from transformers import (
            BertConfig,
            BertForTokenClassification,
            BertTokenizerFast,
        )
    except ImportError as exc:
        raise_missing_backend("onnx", feature="browser lane fixture", cause=exc)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = synthetic_vocabulary()
    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(str(directory))

    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
        id2label=dict(enumerate(SYNTHETIC_LABELS)),
        label2id={label: index for index, label in enumerate(SYNTHETIC_LABELS)},
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BertForTokenClassification(config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            model.classifier.weight.normal_(
                0.0, CLASSIFIER_INIT_STD, generator=generator
            )
            model.classifier.bias.zero_()
    model.eval()
    model.save_pretrained(str(directory))
    return directory


def build_fallback_control_graph(
    source_path: str | Path,
    target_path: str | Path,
    *,
    output_name: str = "logits",
) -> Path:
    """Append a WebGPU-unsupported identity operator to an exported graph.

    The control applies ``Mod(logits, 65504, fmod=1)``, which returns the
    logits unchanged for every finite fp32 value below the fp16 range, so the
    Python reference still applies. ONNX Runtime Web's WebGPU provider has no
    ``Mod`` kernel; a correct lane reports the graph as partial fallback.

    Args:
        source_path: Exported fp32 ONNX graph (external data is resolved).
        target_path: Destination for the self-contained control graph.
        output_name: Name of the logits output to wrap.

    Returns:
        The control graph path.
    """

    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise_missing_backend("onnx", feature="browser lane control graph", cause=exc)

    model = onnx.load(str(source_path))
    graph = model.graph
    matching = [output for output in graph.output if output.name == output_name]
    if len(matching) != 1:
        raise BrowserLaneError("control graph source must expose one logits output")
    inner_name = f"{output_name}_before_fallback_control"
    for node in graph.node:
        node.output[:] = [
            inner_name if name == output_name else name for name in node.output
        ]
        node.input[:] = [
            inner_name if name == output_name else name for name in node.input
        ]
    divisor_name = "fallback_control_divisor"
    graph.initializer.append(
        numpy_helper.from_array(
            np.array([_FALLBACK_CONTROL_DIVISOR], dtype=np.float32),
            name=divisor_name,
        )
    )
    graph.node.append(
        helper.make_node(
            FALLBACK_CONTROL_OP_TYPE,
            [inner_name, divisor_name],
            [output_name],
            name="fallback_control",
            fmod=1,
        )
    )
    onnx.checker.check_model(model)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))
    return target


def build_synthetic_feeds(
    *,
    seed: int = DEFAULT_SEED,
    attempt: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    padding_tokens: int = DEFAULT_PADDING_TOKENS,
    vocab_size: int | None = None,
) -> dict[str, Any]:
    """Return deterministic synthetic ``int64`` token feeds.

    Each row starts with ``[CLS]`` and ends with ``[SEP]``. The last row is
    right-padded with ``padding_tokens`` masked ``[PAD]`` tokens so the lane
    exercises attention masking.

    Args:
        seed: Base seed.
        attempt: Redraw counter mixed into the seed.
        batch_size: Rows in the batch.
        sequence_length: Tokens per row.
        padding_tokens: Masked tokens at the end of the last row.
        vocab_size: Vocabulary size; defaults to the synthetic vocabulary.

    Returns:
        ``input_ids``, ``attention_mask`` and ``token_type_ids`` arrays.
    """

    import numpy as np

    vocab_size = vocab_size or len(synthetic_vocabulary())
    if batch_size < 1 or sequence_length < 3:
        raise BrowserLaneError("feeds need one row and at least three tokens")
    if not 0 <= padding_tokens <= sequence_length - 3:
        raise BrowserLaneError("padding must leave room for special tokens")
    if vocab_size <= FIRST_REGULAR_TOKEN_ID:
        raise BrowserLaneError("vocabulary has no regular tokens")
    rng = np.random.default_rng([int(seed), int(attempt)])
    input_ids = rng.integers(
        FIRST_REGULAR_TOKEN_ID,
        vocab_size,
        size=(batch_size, sequence_length),
        dtype=np.int64,
    )
    attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)
    for row in range(batch_size):
        active = sequence_length - (padding_tokens if row == batch_size - 1 else 0)
        input_ids[row, 0] = CLS_TOKEN_ID
        input_ids[row, active - 1] = SEP_TOKEN_ID
        input_ids[row, active:] = PAD_TOKEN_ID
        attention_mask[row, active:] = 0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": np.zeros((batch_size, sequence_length), dtype=np.int64),
    }


def decode_token_spans(
    logits: Any,
    id2label: Mapping[Any, str],
    attention_mask: Any = None,
) -> list[dict[str, Any]]:
    """Decode argmax BIO/BILOU token spans exactly like the browser runtime.

    The rules mirror ``decodeWebGpuTokenSpans`` in ``openmedkit-web`` so a
    Python decode of the reference can be compared with the browser decode.

    Args:
        logits: Array-like ``[batch, sequence, labels]`` logits.
        id2label: Label names keyed by integer id.
        attention_mask: Optional ``[batch, sequence]`` mask; masked tokens
            close any open span and are never labelled.

    Returns:
        Spans as ``batch_index``, ``label``, ``start_token`` and ``end_token``
        (exclusive) dictionaries.
    """

    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 3:
        raise BrowserLaneError("logits must have shape [batch, sequence, labels]")
    batch_size, sequence_length, label_count = values.shape
    labels = {int(key): str(value) for key, value in id2label.items()}
    if any(index not in labels for index in range(label_count)):
        raise BrowserLaneError("id2label does not cover every logits label")
    mask = (
        None
        if attention_mask is None
        else np.asarray(attention_mask, dtype=np.int64).reshape(
            batch_size, sequence_length
        )
    )
    spans: list[dict[str, Any]] = []
    for batch in range(batch_size):
        active: dict[str, Any] | None = None

        def close(end: int, row: int = batch) -> None:
            nonlocal active
            if active is not None:
                spans.append(
                    {
                        "batch_index": row,
                        "label": active["label"],
                        "start_token": active["start"],
                        "end_token": end,
                    }
                )
                active = None

        for token in range(sequence_length):
            if mask is not None and mask[batch, token] == 0:
                close(token)
                continue
            kind, label = _parse_token_label(
                labels[int(np.argmax(values[batch, token]))]
            )
            if kind == "outside":
                close(token)
            elif kind == "single":
                close(token)
                spans.append(
                    {
                        "batch_index": batch,
                        "label": label,
                        "start_token": token,
                        "end_token": token + 1,
                    }
                )
            elif kind == "begin":
                close(token)
                active = {"label": label, "start": token}
            elif kind == "inside":
                if active is None or active["label"] != label:
                    close(token)
                    active = {"label": label, "start": token}
            else:
                if active is None or active["label"] != label:
                    close(token)
                    active = {"label": label, "start": token}
                close(token + 1)
        close(sequence_length)
    return spans


def minimum_top2_margin(logits: Any, attention_mask: Any = None) -> float:
    """Return the smallest top-1 minus top-2 logit gap over active tokens.

    Args:
        logits: Array-like ``[batch, sequence, labels]`` logits.
        attention_mask: Optional mask; masked tokens are ignored.

    Returns:
        The minimum margin, or ``inf`` when no token is active.
    """

    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    ordered = np.sort(values, axis=-1)
    margins = ordered[..., -1] - ordered[..., -2]
    if attention_mask is not None:
        mask = np.asarray(attention_mask).reshape(margins.shape).astype(bool)
        margins = margins[mask]
    return float(margins.min()) if margins.size else math.inf


def feeds_digest(feeds: Mapping[str, Any], input_names: Sequence[str]) -> str:
    """Return the SHA-256 of the int64 little-endian feeds in input order.

    Args:
        feeds: Token feeds keyed by graph input name.
        input_names: Graph input names, in the order they are hashed.

    Returns:
        Lowercase hex digest; the browser harness recomputes the same value.
    """

    import numpy as np

    digest = hashlib.sha256()
    for name in input_names:
        digest.update(np.ascontiguousarray(feeds[name], dtype="<i8").tobytes())
    return digest.hexdigest()


def logits_digest(logits: Any) -> str:
    """Return the SHA-256 of float32 little-endian logits.

    Args:
        logits: Array-like logits, hashed in row-major order.

    Returns:
        Lowercase hex digest; the browser harness recomputes the same value.
    """

    import numpy as np

    return hashlib.sha256(
        np.ascontiguousarray(logits, dtype="<f4").tobytes()
    ).hexdigest()


def prepare_browser_lane(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = DEFAULT_SEED,
    checkpoint_dir: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    padding_tokens: int = DEFAULT_PADDING_TOKENS,
    cases: Sequence[LaneCase] = DEFAULT_CASES,
) -> Path:
    """Export the lane graphs and write the browser lane fixture.

    Without ``checkpoint_dir`` the synthetic checkpoint from
    :func:`write_synthetic_checkpoint` is created first. The checkpoint is
    exported with :func:`openmed.onnx.convert.convert` (fp32 plus the WebGPU
    fp16 variant, no graph optimisation), the negative control graph is
    derived, and Python ONNX Runtime CPU computes the fp32 reference logits.
    Feeds are redrawn deterministically until every active token has a clear
    argmax margin and at least one entity span is decoded.

    Args:
        output_dir: Directory that receives ``lane-fixture.json``, ``graphs/``
            and, for the synthetic model, ``checkpoint/``.
        seed: Seed for weights and feeds.
        checkpoint_dir: Optional local token-classification checkpoint to
            export instead of the synthetic one.
        batch_size: Rows in the synthetic batch.
        sequence_length: Tokens per row.
        padding_tokens: Masked tokens at the end of the last row.
        cases: Provider and graph combinations for the browser lane.

    Returns:
        Path to ``lane-fixture.json``.
    """

    try:
        import numpy as np
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise_missing_backend("onnx", feature="browser lane fixture", cause=exc)

    from openmed.onnx.convert import (
        DEFAULT_ONNX_FILENAME,
        DEFAULT_WEBGPU_FILENAME,
        convert,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = checkpoint_dir is None
    checkpoint = (
        write_synthetic_checkpoint(output_dir / CHECKPOINT_DIRNAME, seed=seed)
        if checkpoint_dir is None
        else Path(checkpoint_dir)
    )
    graphs_dir = output_dir / GRAPHS_DIRNAME
    convert(
        str(checkpoint),
        graphs_dir,
        include_webgpu=True,
        include_transformersjs=False,
        include_int8=False,
        optimize_onnx=False,
        max_seq_length=sequence_length,
    )
    fp32_path = graphs_dir / DEFAULT_ONNX_FILENAME
    fp16_path = graphs_dir / DEFAULT_WEBGPU_FILENAME
    control_path = build_fallback_control_graph(
        fp32_path, graphs_dir / FALLBACK_CONTROL_FILENAME
    )

    config = json.loads((graphs_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {str(key): str(value) for key, value in config["id2label"].items()}
    vocab_size = int(config["vocab_size"])
    input_names = _graph_input_names(fp32_path)

    reference: Any = None
    feeds: dict[str, Any] = {}
    for attempt in range(MAX_FEED_ATTEMPTS):
        feeds = build_synthetic_feeds(
            seed=seed,
            attempt=attempt,
            batch_size=batch_size,
            sequence_length=sequence_length,
            padding_tokens=padding_tokens,
            vocab_size=vocab_size,
        )
        candidate = np.asarray(
            run_onnx_reference_logits(fp32_path, feeds), dtype=np.float32
        )
        if minimum_top2_margin(
            candidate, feeds["attention_mask"]
        ) > MIN_REFERENCE_MARGIN and decode_token_spans(
            candidate, id2label, feeds["attention_mask"]
        ):
            reference = candidate
            break
    if reference is None:
        raise BrowserLaneError("no synthetic feed met the reference margin gate")

    _check_python_variant(
        fp16_path,
        feeds,
        reference,
        id2label=id2label,
        tolerance=WEBGPU_FLOAT16_TOLERANCE.max_abs_logit_delta,
        code="fp16 graph exceeded the fp16 tolerance on Python ONNX Runtime",
    )
    _check_python_variant(
        control_path,
        feeds,
        reference,
        id2label=id2label,
        tolerance=CONTROL_IDENTITY_TOLERANCE,
        code="fallback control graph changed the logits on Python ONNX Runtime",
    )

    graphs = {
        "fp32": _graph_entry(fp32_path, output_dir, precision="float32"),
        "fp16": _graph_entry(fp16_path, output_dir, precision="float16"),
        "fallback-control": _graph_entry(
            control_path,
            output_dir,
            precision="float32",
            control_op_type=FALLBACK_CONTROL_OP_TYPE,
        ),
    }
    attention_mask = feeds["attention_mask"]
    fixture = {
        "format": FIXTURE_FORMAT,
        "format_version": FIXTURE_VERSION,
        "lane": LANE_ID,
        "synthetic": synthetic,
        "note": (
            "Synthetic token feeds; the graphs come from the repository exporter "
            "and carry no text, identifiers or restricted weights."
        ),
        "seed": int(seed),
        "toolchain": _toolchain_versions(onnx_module=onnx, ort_module=onnxruntime),
        "model": {
            "family": str(config.get("model_type") or "unknown"),
            "label_count": len(id2label),
            "vocab_size": vocab_size,
        },
        "id2label": id2label,
        "graphs": graphs,
        "feeds": {
            "input_names": list(input_names),
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            **{
                name: [int(value) for value in feeds[name].reshape(-1)]
                for name in input_names
            },
            "sha256": feeds_digest(feeds, input_names),
        },
        "reference": {
            "runtime": "python-onnxruntime-cpu",
            "graph": "fp32",
            "dims": [int(dim) for dim in reference.shape],
            "logits": [float(value) for value in reference.reshape(-1)],
            "sha256": logits_digest(reference),
            "spans": decode_token_spans(reference, id2label, attention_mask),
            "min_top2_margin": minimum_top2_margin(reference, attention_mask),
        },
        "cases": [case.to_dict() for case in cases],
    }
    validate_lane_fixture(fixture)
    fixture_path = output_dir / FIXTURE_FILENAME
    fixture_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixture_path


def validate_lane_fixture(fixture: Mapping[str, Any]) -> None:
    """Check the structural contract of a lane fixture.

    Args:
        fixture: Parsed ``lane-fixture.json`` content.

    Raises:
        BrowserLaneError: If the fixture is not a usable lane fixture.
    """

    if fixture.get("format") != FIXTURE_FORMAT:
        raise BrowserLaneError("fixture format is not the browser lane format")
    if fixture.get("format_version") != FIXTURE_VERSION:
        raise BrowserLaneError("fixture format_version is unsupported")
    graphs = fixture.get("graphs")
    if not isinstance(graphs, Mapping) or set(graphs) != set(GRAPH_IDS):
        raise BrowserLaneError("fixture graphs must cover every graph id")
    for entry in graphs.values():
        if not _SHA256.match(str(entry.get("sha256", ""))):
            raise BrowserLaneError("fixture graph digest is invalid")
        for item in entry.get("external_data", []):
            if not _SHA256.match(str(item.get("sha256", ""))):
                raise BrowserLaneError("fixture external data digest is invalid")
    control_op = graphs["fallback-control"].get("control_op_type")
    if (
        not isinstance(control_op, str)
        or control_op not in graphs["fallback-control"].get("op_types", [])
        or control_op in graphs["fp32"].get("op_types", [])
    ):
        raise BrowserLaneError(
            "fallback control must add an operator absent from the fp32 graph"
        )
    feeds = fixture.get("feeds")
    reference = fixture.get("reference")
    if not isinstance(feeds, Mapping) or not isinstance(reference, Mapping):
        raise BrowserLaneError("fixture must contain feeds and a reference")
    expected_tokens = int(feeds["batch_size"]) * int(feeds["sequence_length"])
    for name in feeds["input_names"]:
        if len(feeds[name]) != expected_tokens:
            raise BrowserLaneError("fixture feed length does not match its shape")
    dims = [int(dim) for dim in reference["dims"]]
    if len(dims) != 3 or math.prod(dims) != len(reference["logits"]):
        raise BrowserLaneError("fixture reference logits do not match their dims")
    if not all(math.isfinite(float(value)) for value in reference["logits"]):
        raise BrowserLaneError("fixture reference logits must be finite")
    seen: set[str] = set()
    for case in fixture.get("cases", []):
        case_id = str(case.get("case_id", ""))
        if not _SAFE_TOKEN.match(case_id) or case_id in seen:
            raise BrowserLaneError("fixture case ids must be unique safe tokens")
        seen.add(case_id)
        if case.get("provider") not in PROVIDERS:
            raise BrowserLaneError("fixture case provider is unknown")
        if case.get("graph") not in GRAPH_IDS:
            raise BrowserLaneError("fixture case graph is unknown")
        if case.get("isolation") not in ISOLATION_MODES:
            raise BrowserLaneError("fixture case isolation is unknown")
        _validate_tolerances(case.get("tolerances"))
    if not seen:
        raise BrowserLaneError("fixture declares no lane cases")


def validate_lane_record(record: Mapping[str, Any]) -> None:
    """Check that a browser lane record is complete, consistent and value-free.

    Every level must contain exactly the documented keys. String values are
    restricted to controlled vocabularies, versions and SHA-256 digests, so a
    record cannot carry text, logits, token ids, URLs or file paths.

    Args:
        record: One parsed per-case record.

    Raises:
        BrowserLaneError: If the record violates the contract.
    """

    _require_keys(record, _RECORD_KEYS, "record")
    if record["format"] != RECORD_FORMAT or record["format_version"] != RECORD_VERSION:
        raise BrowserLaneError("record format or version is unsupported")
    if record["lane"] != LANE_ID:
        raise BrowserLaneError("record lane is unknown")
    _require_token(record["case_id"], "record.case_id")
    if record["graph"] not in GRAPH_IDS:
        raise BrowserLaneError("record.graph is unknown")
    if record["precision"] not in ("float32", "float16"):
        raise BrowserLaneError("record.precision is unknown")

    provider = record["provider"]
    _require_keys(provider, _PROVIDER_KEYS, "record.provider")
    if provider["id"] not in PROVIDERS:
        raise BrowserLaneError("record.provider.id is unknown")
    if provider["backend"] is not None and provider["backend"] not in ORT_BACKENDS:
        raise BrowserLaneError("record.provider.backend is unknown")
    providers = provider["execution_providers"]
    if not isinstance(providers, list) or any(
        item not in ("webgpu", "wasm") for item in providers
    ):
        raise BrowserLaneError("record.provider.execution_providers is invalid")
    if provider["runtime"] != "onnxruntime-web":
        raise BrowserLaneError("record.provider.runtime is unknown")
    _require_version(provider["runtime_version"], "record.provider.runtime_version")
    if provider["runtime_bundle"] not in RUNTIME_BUNDLES:
        raise BrowserLaneError("record.provider.runtime_bundle is unknown")
    _require_bool(provider["cross_origin_isolated"], "record.provider")
    _require_count(provider["wasm_threads"], "record.provider.wasm_threads")
    _require_count(provider["worker_count"], "record.provider.worker_count")

    browser = record["browser"]
    _require_keys(browser, _BROWSER_KEYS, "record.browser")
    if browser["name"] not in ("chromium", "firefox", "webkit"):
        raise BrowserLaneError("record.browser.name is unknown")
    _require_version(browser["version"], "record.browser.version")

    digests = record["digests"]
    _require_keys(digests, _DIGEST_KEYS, "record.digests")
    for key in ("graph_sha256", "feeds_sha256", "reference_sha256"):
        if not isinstance(digests[key], str) or not _SHA256.match(digests[key]):
            raise BrowserLaneError(f"record.digests.{key} is not a SHA-256 digest")
    external = digests["external_data_sha256"]
    if external is not None and (
        not isinstance(external, str) or not _SHA256.match(external)
    ):
        raise BrowserLaneError("record.digests.external_data_sha256 is invalid")

    tolerances = record["tolerances"]
    _validate_tolerances(tolerances)
    _validate_outcomes(
        record["outcomes"], provider_id=provider["id"], tolerances=tolerances
    )


def load_lane_records(records_dir: str | Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` record in *records_dir*, sorted by file name.

    Args:
        records_dir: Directory written by the Playwright lane.

    Returns:
        Parsed records; validate them with :func:`validate_lane_record`.

    Raises:
        BrowserLaneError: If a record file is not a JSON object.
    """

    records: list[dict[str, Any]] = []
    for path in sorted(Path(records_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise BrowserLaneError("lane record must be a JSON object")
        records.append(payload)
    return records


def evaluate_lane_records(
    records: Iterable[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    *,
    allow_unavailable: Collection[str] = (),
) -> dict[str, Any]:
    """Apply the browser lane release gate to a set of records.

    The gate requires one valid record per fixture case, with the fixture's
    digests and tolerances. Every case must pass, except that an
    ``unavailable`` record is accepted when its reason code is listed in
    ``allow_unavailable``; a ``failed`` record is never accepted. A case with
    ``expected_webgpu_support`` must report that level, and the negative
    control must name its injected operator among the fallback operators.

    Args:
        records: Parsed per-case records.
        fixture: Parsed lane fixture the records were produced from.
        allow_unavailable: Unavailable reason codes to accept, for example
            ``webgpu_shader_f16_unavailable`` on an adapter without
            ``shader-f16``.

    Returns:
        A value-free summary with ``passed``, per-case ``statuses`` and
        ``webgpu_support``, and the controlled ``failures`` that blocked the
        gate.

    Raises:
        BrowserLaneError: If the fixture is invalid or an allowed reason code
            is not an unavailable reason.
    """

    validate_lane_fixture(fixture)
    allowed = frozenset(allow_unavailable)
    if not allowed <= UNAVAILABLE_REASONS:
        raise BrowserLaneError("allow_unavailable lists a non-unavailable reason")
    cases = {case["case_id"]: case for case in fixture["cases"]}
    failures: list[str] = []
    statuses: dict[str, str] = {}
    support: dict[str, str] = {}
    by_case: dict[str, Mapping[str, Any]] = {}
    for record in records:
        try:
            validate_lane_record(record)
        except BrowserLaneError:
            failures.append("record_invalid")
            continue
        case_id = str(record["case_id"])
        if case_id not in cases:
            failures.append(f"{case_id}:record_unknown_case")
            continue
        if case_id in by_case:
            failures.append(f"{case_id}:record_duplicated")
            continue
        by_case[case_id] = record

    for case_id, case in cases.items():
        found = by_case.get(case_id)
        if found is None:
            failures.append(f"{case_id}:record_missing")
            continue
        outcomes = found["outcomes"]
        status = str(outcomes["status"])
        statuses[case_id] = status
        support[case_id] = str(outcomes["webgpu_support"])
        if (
            found["provider"]["id"] != case["provider"]
            or found["graph"] != case["graph"]
        ):
            failures.append(f"{case_id}:record_case_mismatch")
        graph = fixture["graphs"][case["graph"]]
        if (
            found["digests"]["graph_sha256"] != graph["sha256"]
            or found["digests"]["external_data_sha256"] != _external_data_digest(graph)
            or found["digests"]["feeds_sha256"] != fixture["feeds"]["sha256"]
            or found["digests"]["reference_sha256"] != fixture["reference"]["sha256"]
        ):
            failures.append(f"{case_id}:digest_mismatch")
        if found["tolerances"] != case["tolerances"]:
            failures.append(f"{case_id}:tolerance_mismatch")
        if status == "failed" or (
            status == "unavailable" and outcomes["reason_code"] not in allowed
        ):
            failures.append(f"{case_id}:{outcomes['reason_code']}")
        expected = case.get("expected_webgpu_support")
        if status == "passed" and expected and outcomes["webgpu_support"] != expected:
            failures.append(f"{case_id}:unexpected_webgpu_support")
        control_op = graph.get("control_op_type")
        if (
            status == "passed"
            and control_op
            and control_op not in outcomes["fallback_op_types"]
        ):
            failures.append(f"{case_id}:control_op_not_reported")
    return {
        "lane": LANE_ID,
        "passed": not failures,
        "statuses": statuses,
        "webgpu_support": support,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``prepare`` or ``check`` lane command.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(
        prog="python -m openmed.onnx.browser_lane",
        description="Prepare or check the ONNX Runtime Web browser lane.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="export graphs and references")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="local token-classification checkpoint to export instead",
    )
    check = commands.add_parser("check", help="validate and gate lane records")
    check.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / FIXTURE_FILENAME,
    )
    check.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    check.add_argument(
        "--allow-unavailable",
        action="append",
        default=[],
        choices=sorted(UNAVAILABLE_REASONS),
        help="accept unavailable records with this reason code (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.command == "prepare":
        fixture_path = prepare_browser_lane(
            args.output_dir, seed=args.seed, checkpoint_dir=args.checkpoint
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        summary = {
            "lane": LANE_ID,
            "graphs": {
                key: value["sha256"] for key, value in fixture["graphs"].items()
            },
            "reference_sha256": fixture["reference"]["sha256"],
            "cases": [case["case_id"] for case in fixture["cases"]],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    summary = evaluate_lane_records(
        load_lane_records(args.records_dir),
        fixture,
        allow_unavailable=args.allow_unavailable,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def _parse_token_label(value: str) -> tuple[str, str]:
    if value.upper() == "O":
        return "outside", ""
    match = re.match(r"([BIESUL])[-_](.+)\Z", value, flags=re.IGNORECASE)
    if match is None:
        return "single", value
    prefix = match.group(1).upper()
    label = match.group(2)
    if prefix == "B":
        return "begin", label
    if prefix == "I":
        return "inside", label
    if prefix in {"E", "L"}:
        return "end", label
    return "single", label


def _graph_input_names(path: Path) -> tuple[str, ...]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    return tuple(
        item.name for item in model.graph.input if item.name not in initializers
    )


def _graph_entry(
    path: Path,
    root: Path,
    *,
    precision: str,
    control_op_type: str | None = None,
) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    locations: list[str] = []
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for item in tensor.external_data:
            if item.key == "location" and item.value not in locations:
                locations.append(item.value)
    external = []
    for location in locations:
        data_path = path.parent / location
        if Path(location).name != location or not data_path.is_file():
            raise BrowserLaneError("external data must sit beside its graph")
        external.append(
            {
                "location": location,
                "file": data_path.relative_to(root).as_posix(),
                "sha256": _file_sha256(data_path),
                "bytes": data_path.stat().st_size,
            }
        )
    entry: dict[str, Any] = {
        "file": path.relative_to(root).as_posix(),
        "precision": precision,
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "external_data": external,
        "op_types": sorted({node.op_type for node in model.graph.node}),
    }
    if control_op_type is not None:
        entry["control_op_type"] = control_op_type
    return entry


def _external_data_digest(graph: Mapping[str, Any]) -> str | None:
    external = graph.get("external_data") or []
    if not external:
        return None
    if len(external) == 1:
        return str(external[0]["sha256"])
    digest = hashlib.sha256()
    for item in external:
        digest.update(str(item["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _check_python_variant(
    path: Path,
    feeds: Mapping[str, Any],
    reference: Any,
    *,
    id2label: Mapping[str, str],
    tolerance: float,
    code: str,
) -> None:
    import numpy as np

    candidate = np.asarray(run_onnx_reference_logits(path, feeds), dtype=np.float32)
    parity = evaluate_onnx_logit_parity(
        reference,
        candidate,
        id2label=id2label,
        rtol=0.0,
        atol=tolerance,
    )
    spans_match = decode_token_spans(
        candidate, id2label, feeds["attention_mask"]
    ) == decode_token_spans(reference, id2label, feeds["attention_mask"])
    if not (parity.logits_within_tolerance and spans_match):
        raise BrowserLaneError(code)


def _toolchain_versions(*, onnx_module: Any, ort_module: Any) -> dict[str, str]:
    versions = {
        "onnx": str(getattr(onnx_module, "__version__", "unknown")),
        "python_onnxruntime": str(getattr(ort_module, "__version__", "unknown")),
    }
    for name in ("torch", "transformers"):
        module = sys.modules.get(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tolerances(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise BrowserLaneError("tolerances must be an object")
    _require_keys(value, _TOLERANCE_KEYS, "tolerances")
    delta = value["max_abs_logit_delta"]
    recall = value["max_recall_delta"]
    span = value["span_tolerance_tokens"]
    if not _is_number(delta) or not 0 <= float(delta) < 1:
        raise BrowserLaneError("tolerances.max_abs_logit_delta is invalid")
    if not _is_number(recall) or not 0 <= float(recall) <= 1:
        raise BrowserLaneError("tolerances.max_recall_delta is invalid")
    if isinstance(span, bool) or not isinstance(span, int) or span < 0:
        raise BrowserLaneError("tolerances.span_tolerance_tokens is invalid")


def _validate_outcomes(
    outcomes: Any,
    *,
    provider_id: str,
    tolerances: Mapping[str, Any],
) -> None:
    _require_keys(outcomes, _OUTCOME_KEYS, "record.outcomes")
    status = outcomes["status"]
    reason = outcomes["reason_code"]
    if status not in RECORD_STATUSES:
        raise BrowserLaneError("record.outcomes.status is unknown")
    if reason not in REASON_CODES:
        raise BrowserLaneError("record.outcomes.reason_code is unknown")
    if (status == "passed") != (reason == "ok"):
        raise BrowserLaneError("record.outcomes.reason_code contradicts status")
    if status == "unavailable" and reason not in UNAVAILABLE_REASONS:
        raise BrowserLaneError("record.outcomes.reason_code contradicts status")
    if status == "failed" and reason not in FAILURE_REASONS:
        raise BrowserLaneError("record.outcomes.reason_code contradicts status")

    support = outcomes["webgpu_support"]
    if support not in WEBGPU_SUPPORT_LEVELS:
        raise BrowserLaneError("record.outcomes.webgpu_support is unknown")
    if (provider_id == "webgpu") == (support == "not_applicable"):
        raise BrowserLaneError("record.outcomes.webgpu_support contradicts provider")
    ops = outcomes["fallback_op_types"]
    if (
        not isinstance(ops, list)
        or any(not isinstance(op, str) or not _OP_TYPE.match(op) for op in ops)
        or ops != sorted(set(ops))
    ):
        raise BrowserLaneError("record.outcomes.fallback_op_types is invalid")
    if support == "partial" and not ops:
        raise BrowserLaneError("partial support must name fallback operators")
    if support == "full" and ops:
        raise BrowserLaneError("full support cannot name fallback operators")
    _require_count(outcomes["fallback_node_count"], "record.outcomes")
    if bool(ops) != (outcomes["fallback_node_count"] > 0):
        raise BrowserLaneError("record.outcomes.fallback_node_count is inconsistent")
    placement = outcomes["node_placement"]
    if not isinstance(placement, Mapping) or not set(placement) <= set(PLACEMENT_KEYS):
        raise BrowserLaneError("record.outcomes.node_placement is invalid")
    for count in placement.values():
        _require_count(count, "record.outcomes.node_placement")
    _require_count(outcomes["same_origin_request_count"], "record.outcomes")
    _require_count(outcomes["cross_origin_request_count"], "record.outcomes")

    for key in ("max_abs_logit_delta", "recall"):
        value = outcomes[key]
        if value is not None and (not _is_number(value) or value < 0):
            raise BrowserLaneError(f"record.outcomes.{key} is invalid")
    for key in ("logits_within_tolerance", "span_parity"):
        if outcomes[key] is not None:
            _require_bool(outcomes[key], "record.outcomes")
    for key in ("reference_span_count", "candidate_span_count", "matched_span_count"):
        if outcomes[key] is not None:
            _require_count(outcomes[key], "record.outcomes")

    if status == "unavailable" and any(
        outcomes[key] is not None for key in _MEASURED_OUTCOMES
    ):
        raise BrowserLaneError("unavailable records cannot carry measurements")
    if status == "passed":
        if any(outcomes[key] is None for key in _MEASURED_OUTCOMES):
            raise BrowserLaneError("passed records must carry every measurement")
        if (
            outcomes["logits_within_tolerance"] is not True
            or outcomes["span_parity"] is not True
            or float(outcomes["max_abs_logit_delta"])
            > float(tolerances["max_abs_logit_delta"])
            or outcomes["cross_origin_request_count"] != 0
            or support == "unknown"
        ):
            raise BrowserLaneError("passed record contradicts its measurements")


def _require_keys(value: Any, keys: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise BrowserLaneError(f"{label} must be an object")
    present = set(value)
    if present != keys:
        missing = sorted(keys - present)
        extra = len(present - keys)
        raise BrowserLaneError(
            f"{label} keys are invalid (missing {missing}, {extra} unexpected)"
        )


def _require_token(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_TOKEN.match(value):
        raise BrowserLaneError(f"{label} is not a controlled token")


def _require_version(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _VERSION.match(value):
        raise BrowserLaneError(f"{label} is not a version string")


def _require_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        raise BrowserLaneError(f"{label} flag must be boolean")


def _require_count(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BrowserLaneError(f"{label} count must be a non-negative integer")


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
