"""AWQ recipe and fail-closed recall gate for grounding embedders."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from openmed.eval.metrics import LatencyMetrics, compute_latency_summary
from openmed.eval.report import BenchmarkReport
from openmed.processing.tokenizer_cache import get_tokenizer_with_loader

from .calibration import (
    QUANTIZATION_CALIBRATION_SOURCE,
    calibration_texts_sha256,
    load_awq_calibration_texts,
)
from .quantize_awq import AWQ_FORMAT, QUANT_CONFIG_FILENAME, quantize_awq

GROUNDING_AWQ_PROFILE = "openmed-awq-grounding"
GROUNDING_AWQ_PROFILE_VERSION = 1
GROUNDING_BENCHMARK_FILENAME = "grounding_awq_benchmark.json"
GROUNDING_BENCHMARK_SUITE = "grounding-awq-retrieval"
DEFAULT_GROUNDING_TOP_K = 3
DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE = 0.05
DEFAULT_GROUNDING_MAX_LENGTH = 128

# These deliberately fictional concepts exercise retrieval without bundling or
# calibrating against a restricted clinical terminology.
SYNTHETIC_GROUNDING_QUERIES: tuple[str, ...] = (
    "aster pyrexia",
    "beryl cough",
    "corin skin flare",
    "dax ankle sprain",
    "elin glucose panel",
)
SYNTHETIC_GROUNDING_PASSAGES: tuple[str, ...] = (
    "Aster fever is a synthetic condition used only for retrieval testing.",
    "Beryl cough pattern is a fictional respiratory concept.",
    "Corin skin flare is a synthetic dermatology phrase.",
    "Dax ankle strain is a fabricated mobility concept.",
    "Elin sugar panel is a synthetic observation concept.",
    "Faren breath score is a fictional observation scale.",
    "Halo pain rating is a synthetic assessment concept.",
    "Iona sleep coaching is a fabricated treatment concept.",
)

Clock = Callable[[], float]


class GroundingEmbedder(Protocol):
    """Minimal embedding contract consumed by the grounding recall gate."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one numeric embedding per input text."""


class GroundingAwqRejected(ValueError):
    """Raised when an AWQ grounding artifact lacks passing recall evidence."""

    def __init__(
        self,
        message: str,
        *,
        gate: GroundingRecallGate | None = None,
        benchmark_report_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.gate = gate
        self.benchmark_report_path = benchmark_report_path


@dataclass(frozen=True)
class GroundingRecallGate:
    """Top-k retrieval overlap between an fp16 and AWQ grounding embedder."""

    top_k: int
    query_count: int
    passage_count: int
    per_query_overlap: tuple[float, ...]
    mean_top_k_overlap: float
    recall_delta: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the gate as deterministic benchmark metrics."""

        return {
            "metric": "top_k_overlap",
            "top_k": self.top_k,
            "query_count": self.query_count,
            "passage_count": self.passage_count,
            "per_query_overlap": list(self.per_query_overlap),
            "mean_top_k_overlap": self.mean_top_k_overlap,
            "recall_delta": self.recall_delta,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GroundingRecallCertification:
    """Recall gate plus fp16 and AWQ embedding latency measurements."""

    gate: GroundingRecallGate
    fp16_latency: LatencyMetrics
    awq_latency: LatencyMetrics


@dataclass(frozen=True)
class AwqGroundingQuantizationResult:
    """Certified AWQ grounding artifact and its evidence paths."""

    output_dir: Path
    quant_config_path: Path
    benchmark_report_path: Path
    source_model_id: str
    source_revision: str
    calibration_sample_count: int
    calibration_sha256: str
    recall_gate: GroundingRecallGate


class HuggingFaceGroundingEmbedder:
    """Mean-pooled Hugging Face model adapter for grounding retrieval."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int = DEFAULT_GROUNDING_MAX_LENGTH,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return L2-normalized mean-pooled hidden states for ``texts``."""

        normalized_texts = _normalize_texts(texts, name="texts")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Grounding embedding requires PyTorch. Install with: "
                "pip install openmed[awq]"
            ) from exc

        inputs = self.tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = _model_input_device(self.model)
        if device is not None:
            mover = getattr(inputs, "to", None)
            if callable(mover):
                inputs = mover(device)
            elif isinstance(inputs, Mapping):
                inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }

        eval_model = getattr(self.model, "eval", None)
        if callable(eval_model):
            eval_model()
        with torch.inference_mode():
            outputs = self.model(
                **dict(inputs),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if not hidden_states:
                    raise ValueError("embedding model did not return hidden states")
                hidden = hidden_states[-1]

            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(
                    hidden.shape[:2],
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return pooled.detach().cpu().tolist()


def certify_grounding_recall(
    fp16_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    awq_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    *,
    queries: Sequence[str] = SYNTHETIC_GROUNDING_QUERIES,
    passages: Sequence[str] = SYNTHETIC_GROUNDING_PASSAGES,
    top_k: int = DEFAULT_GROUNDING_TOP_K,
    recall_delta_tolerance: float = DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE,
    clock: Clock = perf_counter,
) -> GroundingRecallCertification:
    """Compare AWQ retrieval top-k results with the fp16 parent.

    The gate measures, for each query, the fraction of fp16 top-k passage IDs
    retained by the AWQ artifact. The recall delta is ``1 - mean overlap``.

    Raises:
        ValueError: If fixtures, tolerances, or embedding outputs are invalid.
    """

    normalized_queries = _normalize_texts(queries, name="queries")
    normalized_passages = _normalize_texts(passages, name="passages")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not 0.0 <= recall_delta_tolerance <= 1.0:
        raise ValueError("recall_delta_tolerance must be between 0 and 1")
    effective_top_k = min(top_k, len(normalized_passages))
    all_texts = [*normalized_queries, *normalized_passages]

    fp16_embeddings, fp16_latency = _timed_embeddings(
        fp16_embedder,
        all_texts,
        clock=clock,
    )
    awq_embeddings, awq_latency = _timed_embeddings(
        awq_embedder,
        all_texts,
        clock=clock,
    )
    if len(fp16_embeddings[0]) != len(awq_embeddings[0]):
        raise ValueError("fp16 and AWQ embeddings must use the same vector dimension")
    query_count = len(normalized_queries)
    fp16_queries = fp16_embeddings[:query_count]
    fp16_passages = fp16_embeddings[query_count:]
    awq_queries = awq_embeddings[:query_count]
    awq_passages = awq_embeddings[query_count:]

    per_query_overlap = []
    for fp16_query, awq_query in zip(fp16_queries, awq_queries):
        fp16_top_k = set(
            _top_k_indices(fp16_query, fp16_passages, top_k=effective_top_k)
        )
        awq_top_k = set(_top_k_indices(awq_query, awq_passages, top_k=effective_top_k))
        per_query_overlap.append(len(fp16_top_k & awq_top_k) / effective_top_k)

    mean_overlap = sum(per_query_overlap) / len(per_query_overlap)
    recall_delta = max(1.0 - mean_overlap, 0.0)
    gate = GroundingRecallGate(
        top_k=effective_top_k,
        query_count=query_count,
        passage_count=len(normalized_passages),
        per_query_overlap=tuple(per_query_overlap),
        mean_top_k_overlap=mean_overlap,
        recall_delta=recall_delta,
        tolerance=recall_delta_tolerance,
        passed=recall_delta <= recall_delta_tolerance + 1e-12,
    )
    return GroundingRecallCertification(
        gate=gate,
        fp16_latency=fp16_latency,
        awq_latency=awq_latency,
    )


def quantize_awq_grounding(
    model_name: str,
    out_dir: str | Path,
    *,
    revision: str | None = None,
    calibration_limit: int | None = None,
    certification_queries: Sequence[str] = SYNTHETIC_GROUNDING_QUERIES,
    certification_passages: Sequence[str] = SYNTHETIC_GROUNDING_PASSAGES,
    top_k: int = DEFAULT_GROUNDING_TOP_K,
    recall_delta_tolerance: float = DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE,
    group_size: int = 128,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    device_map: str | Mapping[str, Any] | None = "auto",
    max_calib_seq_len: int = 512,
    max_length: int = DEFAULT_GROUNDING_MAX_LENGTH,
    fp16_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any] | None = None,
    awq_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any] | None = None,
    clock: Clock = perf_counter,
) -> AwqGroundingQuantizationResult:
    """Quantize and certify a 4-bit AWQ grounding embedder.

    Calibration is intentionally restricted to OpenMed's committed synthetic
    calibration loader. A failed or missing top-k certificate leaves the
    artifact unloadable through :func:`load_awq_grounding_embedder`.

    Raises:
        GroundingAwqRejected: If AWQ top-k overlap exceeds the allowed delta.
        ImportError: If optional AWQ, Transformers, or torch dependencies are
            unavailable.
        ValueError: If recipe or certification inputs are invalid.
    """

    calibration_texts = load_awq_calibration_texts(limit=calibration_limit)
    calibration_sha256 = calibration_texts_sha256(calibration_texts)
    quantized = quantize_awq(
        model_name,
        calibration_texts,
        out_dir,
        w_bit=4,
        group_size=group_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        device_map=device_map,
        max_calib_seq_len=max_calib_seq_len,
    )
    output_dir = Path(quantized.output_dir)
    benchmark_report_path = output_dir / GROUNDING_BENCHMARK_FILENAME

    parent = fp16_embedder or _load_fp16_grounding_embedder(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        device_map=device_map,
        max_length=max_length,
    )
    candidate = awq_embedder or _load_awq_grounding_embedder_unchecked(
        output_dir,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        max_length=max_length,
    )
    certification = certify_grounding_recall(
        parent,
        candidate,
        queries=certification_queries,
        passages=certification_passages,
        top_k=top_k,
        recall_delta_tolerance=recall_delta_tolerance,
        clock=clock,
    )
    fixture_sha256 = grounding_fixture_sha256(
        certification_queries,
        certification_passages,
    )
    _update_quant_config_for_grounding(
        Path(quantized.quant_config_path),
        certification=certification,
        benchmark_report_path=benchmark_report_path,
        fixture_sha256=fixture_sha256,
    )
    artifact_size_bytes = _artifact_size_bytes(
        output_dir,
        exclude_names={GROUNDING_BENCHMARK_FILENAME},
    )
    report = _grounding_benchmark_report(
        source_model_id=model_name,
        source_revision=quantized.source_revision,
        calibration_sample_count=len(calibration_texts),
        calibration_sha256=calibration_sha256,
        fixture_sha256=fixture_sha256,
        artifact_size_bytes=artifact_size_bytes,
        certification=certification,
    )
    report.write_json(benchmark_report_path)

    if not certification.gate.passed:
        raise GroundingAwqRejected(
            "AWQ grounding artifact rejected: top-k recall delta "
            f"{certification.gate.recall_delta:.6f} exceeds tolerance "
            f"{certification.gate.tolerance:.6f}",
            gate=certification.gate,
            benchmark_report_path=benchmark_report_path,
        )

    return AwqGroundingQuantizationResult(
        output_dir=output_dir,
        quant_config_path=Path(quantized.quant_config_path),
        benchmark_report_path=benchmark_report_path,
        source_model_id=model_name,
        source_revision=quantized.source_revision,
        calibration_sample_count=len(calibration_texts),
        calibration_sha256=calibration_sha256,
        recall_gate=certification.gate,
    )


def load_awq_grounding_embedder(
    artifact_dir: str | Path,
    *,
    trust_remote_code: bool = False,
    device_map: str | Mapping[str, Any] | None = "auto",
    max_length: int = DEFAULT_GROUNDING_MAX_LENGTH,
) -> HuggingFaceGroundingEmbedder:
    """Load a certified AWQ artifact for the grounding retrieval path.

    The loader fails closed when recipe metadata, the benchmark report, or the
    passing top-k certificate is missing or inconsistent.
    """

    artifact_path = Path(artifact_dir)
    _validate_certified_artifact(artifact_path)
    return _load_awq_grounding_embedder_unchecked(
        artifact_path,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        max_length=max_length,
    )


def grounding_fixture_sha256(
    queries: Sequence[str],
    passages: Sequence[str],
) -> str:
    """Return a stable digest for ordered synthetic retrieval fixtures."""

    normalized_queries = _normalize_texts(queries, name="queries")
    normalized_passages = _normalize_texts(passages, name="passages")
    records = [
        *[f"query\0{text}" for text in normalized_queries],
        *[f"passage\0{text}" for text in normalized_passages],
    ]
    return calibration_texts_sha256(records)


def _grounding_benchmark_report(
    *,
    source_model_id: str,
    source_revision: str,
    calibration_sample_count: int,
    calibration_sha256: str,
    fixture_sha256: str,
    artifact_size_bytes: int,
    certification: GroundingRecallCertification,
) -> BenchmarkReport:
    return BenchmarkReport(
        suite=GROUNDING_BENCHMARK_SUITE,
        model_name=source_model_id,
        device="awq",
        fixture_count=certification.gate.query_count,
        generated_at=_utc_now(),
        metrics={
            "retrieval": certification.gate.to_dict(),
            "latency": {
                "fp16": certification.fp16_latency.to_dict(),
                "awq": certification.awq_latency.to_dict(),
            },
            "resources": {
                "model_size_bytes": artifact_size_bytes,
                "model_size_mib": artifact_size_bytes / (1024 * 1024),
            },
        },
        metadata={
            "format": GROUNDING_AWQ_PROFILE,
            "format_version": GROUNDING_AWQ_PROFILE_VERSION,
            "source_revision": source_revision,
            "certified": certification.gate.passed,
            "calibration": {
                "source": QUANTIZATION_CALIBRATION_SOURCE,
                "sample_count": calibration_sample_count,
                "sha256": calibration_sha256,
            },
            "grounding_fixture_sha256": fixture_sha256,
        },
    )


def _update_quant_config_for_grounding(
    quant_config_path: Path,
    *,
    certification: GroundingRecallCertification,
    benchmark_report_path: Path,
    fixture_sha256: str,
) -> None:
    with quant_config_path.open("r", encoding="utf-8") as handle:
        quant_config = json.load(handle)
    quant_config["task"] = "feature-extraction"
    quant_config["grounding"] = {
        "profile": GROUNDING_AWQ_PROFILE,
        "profile_version": GROUNDING_AWQ_PROFILE_VERSION,
        "certified": certification.gate.passed,
        "metric": "top_k_overlap",
        "top_k": certification.gate.top_k,
        "mean_top_k_overlap": certification.gate.mean_top_k_overlap,
        "recall_delta": certification.gate.recall_delta,
        "recall_delta_tolerance": certification.gate.tolerance,
        "grounding_fixture_sha256": fixture_sha256,
        "benchmark_report_path": benchmark_report_path.name,
    }
    with quant_config_path.open("w", encoding="utf-8") as handle:
        json.dump(quant_config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_certified_artifact(artifact_dir: Path) -> None:
    quant_config_path = artifact_dir / QUANT_CONFIG_FILENAME
    benchmark_path = artifact_dir / GROUNDING_BENCHMARK_FILENAME
    if not quant_config_path.is_file():
        raise GroundingAwqRejected(
            f"AWQ grounding artifact is missing {QUANT_CONFIG_FILENAME}"
        )
    if not benchmark_path.is_file():
        raise GroundingAwqRejected(
            f"AWQ grounding artifact is missing {GROUNDING_BENCHMARK_FILENAME}"
        )

    try:
        with quant_config_path.open("r", encoding="utf-8") as handle:
            quant_config = json.load(handle)
        report = BenchmarkReport.read_json(benchmark_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GroundingAwqRejected(
            "AWQ grounding certification metadata is invalid"
        ) from exc

    grounding = quant_config.get("grounding")
    metrics = report.metrics if isinstance(report.metrics, Mapping) else {}
    retrieval = metrics.get("retrieval")
    metadata = report.metadata if isinstance(report.metadata, Mapping) else {}
    calibration = metadata.get("calibration")
    recall_delta = _finite_float(
        retrieval.get("recall_delta") if isinstance(retrieval, Mapping) else None
    )
    tolerance = _finite_float(
        retrieval.get("tolerance") if isinstance(retrieval, Mapping) else None
    )
    mean_overlap = _finite_float(
        retrieval.get("mean_top_k_overlap") if isinstance(retrieval, Mapping) else None
    )
    calibration_sha256 = quant_config.get("calibration_sha256")
    valid = (
        quant_config.get("format") == AWQ_FORMAT
        and quant_config.get("task") == "feature-extraction"
        and quant_config.get("w_bit") == 4
        and quant_config.get("calibration_source") == QUANTIZATION_CALIBRATION_SOURCE
        and isinstance(calibration_sha256, str)
        and len(calibration_sha256) == 64
        and isinstance(quant_config.get("calibration_sample_count"), int)
        and quant_config.get("calibration_sample_count", 0) > 0
        and isinstance(grounding, Mapping)
        and grounding.get("profile") == GROUNDING_AWQ_PROFILE
        and grounding.get("profile_version") == GROUNDING_AWQ_PROFILE_VERSION
        and grounding.get("certified") is True
        and grounding.get("benchmark_report_path") == benchmark_path.name
        and report.suite == GROUNDING_BENCHMARK_SUITE
        and report.model_name == quant_config.get("source_model_id")
        and metadata.get("format") == GROUNDING_AWQ_PROFILE
        and metadata.get("format_version") == GROUNDING_AWQ_PROFILE_VERSION
        and metadata.get("source_revision") == quant_config.get("source_revision")
        and metadata.get("certified") is True
        and isinstance(calibration, Mapping)
        and calibration.get("source") == QUANTIZATION_CALIBRATION_SOURCE
        and calibration.get("sha256") == calibration_sha256
        and calibration.get("sample_count")
        == quant_config.get("calibration_sample_count")
        and isinstance(retrieval, Mapping)
        and retrieval.get("passed") is True
        and recall_delta is not None
        and tolerance is not None
        and mean_overlap is not None
        and 0.0 <= recall_delta <= tolerance <= 1.0
        and math.isclose(
            recall_delta,
            max(1.0 - mean_overlap, 0.0),
            abs_tol=1e-12,
        )
        and grounding.get("grounding_fixture_sha256")
        == metadata.get("grounding_fixture_sha256")
        and grounding.get("recall_delta") == recall_delta
        and grounding.get("recall_delta_tolerance") == tolerance
    )
    if not valid:
        raise GroundingAwqRejected(
            "AWQ grounding artifact does not have consistent passing recall evidence",
            benchmark_report_path=benchmark_path,
        )


def _load_fp16_grounding_embedder(
    model_name: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
    device_map: str | Mapping[str, Any] | None,
    max_length: int,
) -> HuggingFaceGroundingEmbedder:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Grounding certification requires torch and Transformers. "
            "Install with: pip install openmed[awq]"
        ) from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = get_tokenizer_with_loader(
        model_name,
        AutoTokenizer.from_pretrained,
        **kwargs,
    )
    model_kwargs = {**kwargs, "torch_dtype": torch.float16}
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    return HuggingFaceGroundingEmbedder(model, tokenizer, max_length=max_length)


def _load_awq_grounding_embedder_unchecked(
    artifact_dir: Path,
    *,
    trust_remote_code: bool,
    device_map: str | Mapping[str, Any] | None,
    max_length: int,
) -> HuggingFaceGroundingEmbedder:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "AWQ grounding loading requires the optional AWQ dependencies. "
            "Install with: pip install openmed[awq]"
        ) from exc

    tokenizer = get_tokenizer_with_loader(
        str(artifact_dir),
        AutoTokenizer.from_pretrained,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(
        tokenizer, "eos_token", None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "safetensors": True,
        # Fused modules preallocate a fixed batch-size-one cache by default and
        # are optimized for generation. Grounding needs variable batches and
        # the underlying hidden states for mean pooling.
        "fuse_layers": False,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoAWQForCausalLM.from_quantized(str(artifact_dir), **model_kwargs)
    return HuggingFaceGroundingEmbedder(model, tokenizer, max_length=max_length)


def _timed_embeddings(
    embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    texts: Sequence[str],
    *,
    clock: Clock,
) -> tuple[tuple[tuple[float, ...], ...], LatencyMetrics]:
    started_at = clock()
    encode = getattr(embedder, "encode", None)
    raw_embeddings = encode(texts) if callable(encode) else embedder(texts)
    elapsed_ms = max((clock() - started_at) * 1000.0, 0.0)
    embeddings = _normalize_embeddings(raw_embeddings, expected_count=len(texts))
    per_item_ms = elapsed_ms / len(texts)
    latency = compute_latency_summary([per_item_ms] * len(texts))
    return embeddings, latency


def _normalize_embeddings(
    embeddings: Any,
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        rows = list(embeddings)
    except TypeError as exc:
        raise ValueError("embedder output must be a sequence of vectors") from exc
    if len(rows) != expected_count:
        raise ValueError(
            "embedder output count does not match inputs: "
            f"{len(rows)} != {expected_count}"
        )

    normalized: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError("embedding vectors must contain numeric values")
        try:
            vector = tuple(float(value) for value in row)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vectors must contain numeric values") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vectors must be non-empty and finite")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("embedding vectors must use one consistent dimension")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("embedding vectors must have non-zero norm")
        normalized.append(tuple(value / norm for value in vector))
    return tuple(normalized)


def _top_k_indices(
    query: Sequence[float],
    passages: Sequence[Sequence[float]],
    *,
    top_k: int,
) -> tuple[int, ...]:
    scores = [
        (sum(left * right for left, right in zip(query, passage)), index)
        for index, passage in enumerate(passages)
    ]
    return tuple(
        index
        for _score, index in sorted(scores, key=lambda row: (-row[0], row[1]))[:top_k]
    )


def _normalize_texts(texts: Sequence[str], *, name: str) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    normalized = [
        text.strip() for text in texts if isinstance(text, str) and text.strip()
    ]
    if len(normalized) != len(texts) or not normalized:
        raise ValueError(f"{name} must contain only non-empty strings")
    return normalized


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            return None
    return None


def _artifact_size_bytes(
    path: Path,
    *,
    exclude_names: set[str] | None = None,
) -> int:
    excluded = exclude_names or set()
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and item.name not in excluded
    )


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "AwqGroundingQuantizationResult",
    "DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE",
    "DEFAULT_GROUNDING_TOP_K",
    "GROUNDING_AWQ_PROFILE",
    "GROUNDING_BENCHMARK_FILENAME",
    "GroundingAwqRejected",
    "GroundingEmbedder",
    "GroundingRecallCertification",
    "GroundingRecallGate",
    "HuggingFaceGroundingEmbedder",
    "SYNTHETIC_GROUNDING_PASSAGES",
    "SYNTHETIC_GROUNDING_QUERIES",
    "certify_grounding_recall",
    "grounding_fixture_sha256",
    "load_awq_grounding_embedder",
    "quantize_awq_grounding",
]
