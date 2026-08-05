"""Q4_K_M GGUF export and fail-closed grounding certification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from openmed.eval.metrics import compute_latency_summary
from openmed.eval.quant_delta import INT4_RECALL_DELTA_LIMIT
from openmed.eval.report import BenchmarkReport
from openmed.gguf.convert import convert as convert_gguf
from openmed.onnx.gguf_embed_runtime import LlamaCppEmbeddingRuntime

GGUF_INT4_FORMAT = "gguf-int4"
GGUF_INT4_PROFILE = "openmed-gguf-int4-grounding"
GGUF_INT4_PROFILE_VERSION = 1
GGUF_INT4_FILENAME = "model-q4_k_m.gguf"
Q4_K_M_FILENAME = GGUF_INT4_FILENAME
GGUF_INT4_BENCHMARK_FILENAME = "gguf-grounding-benchmark.json"
GGUF_GROUNDING_BENCHMARK_FILENAME = GGUF_INT4_BENCHMARK_FILENAME
GGUF_GROUNDING_BENCHMARK_SUITE = "grounding-gguf-q4_k_m"
DEFAULT_GROUNDING_TOP_K = 3
DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE = INT4_RECALL_DELTA_LIMIT
DEFAULT_EXPORT_TIMEOUT_SECONDS = 3600.0
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 120.0

_QUANTIZER_NAMES = ("llama-quantize", "quantize", "llama_quantize")

# These fictional concepts exercise retrieval without bundling or calibrating
# against a restricted clinical terminology.
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


class GgufInt4ExportError(RuntimeError):
    """Raised when the external GGUF quantizer cannot create an artifact."""


class GgufInt4Rejected(ValueError):
    """Raised when a Q4_K_M grounding artifact fails certification."""

    def __init__(
        self,
        message: str,
        *,
        gate: "GgufGroundingRecallGate | None" = None,
        benchmark_report_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.gate = gate
        self.benchmark_report_path = benchmark_report_path


@dataclass(frozen=True)
class GgufGroundingRecallGate:
    """G4-style top-k overlap and determinism evidence for Q4_K_M."""

    top_k: int
    query_count: int
    passage_count: int
    per_query_overlap: tuple[float, ...]
    mean_top_k_overlap: float
    recall_delta: float
    tolerance: float
    deterministic: bool
    passed: bool
    rejection_reason: str | None = None
    format: str = GGUF_INT4_FORMAT
    gate: str = "G4"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable certification evidence."""

        return {
            "format": self.format,
            "gate": self.gate,
            "metric": "top_k_overlap",
            "top_k": self.top_k,
            "query_count": self.query_count,
            "passage_count": self.passage_count,
            "per_query_overlap": list(self.per_query_overlap),
            "mean_top_k_overlap": self.mean_top_k_overlap,
            "recall_delta": self.recall_delta,
            "tolerance": self.tolerance,
            "deterministic": self.deterministic,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class GgufGroundingCertification:
    """Recall gate and latency evidence for the fp16 and Q4_K_M runners."""

    gate: GgufGroundingRecallGate
    fp16_latency: Any
    int4_latency: Any


@dataclass(frozen=True)
class GgufInt4ExportResult:
    """Published GGUF artifacts and their grounding certification evidence."""

    output_dir: Path
    manifest_path: Path
    benchmark_report_path: Path
    fp16_path: Path
    q8_0_path: Path
    q4_k_m_path: Path
    certification: GgufGroundingCertification

    @property
    def artifact_path(self) -> Path:
        """Return the certified Q4_K_M artifact path."""

        return self.q4_k_m_path

    @property
    def recall_gate(self) -> GgufGroundingRecallGate:
        """Return the G4 grounding gate."""

        return self.certification.gate


def certify_gguf_grounding(
    fp16_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    int4_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    *,
    queries: Sequence[str] = SYNTHETIC_GROUNDING_QUERIES,
    passages: Sequence[str] = SYNTHETIC_GROUNDING_PASSAGES,
    top_k: int = DEFAULT_GROUNDING_TOP_K,
    recall_delta_tolerance: float = DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE,
    clock: Clock = perf_counter,
) -> GgufGroundingCertification:
    """Compare Q4_K_M retrieval rankings with the fp16 GGUF parent.

    The gate measures the fraction of fp16 top-k passage indexes retained by
    Q4_K_M and also runs the candidate twice. A missing, malformed, or
    nondeterministic vector set fails closed through the raised ``ValueError``
    or a gate whose ``passed`` field is false.
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
    int4_embeddings, int4_latency = _timed_embeddings(
        int4_embedder,
        all_texts,
        clock=clock,
    )
    repeated_int4 = _collect_embeddings(int4_embedder, all_texts)

    if len(fp16_embeddings[0]) != len(int4_embeddings[0]):
        raise ValueError(
            "fp16 and Q4_K_M embeddings must use the same vector dimension"
        )
    if len(int4_embeddings[0]) != len(repeated_int4[0]):
        raise ValueError("repeated Q4_K_M embeddings changed vector dimension")
    deterministic = int4_embeddings == repeated_int4

    query_count = len(normalized_queries)
    fp16_queries = fp16_embeddings[:query_count]
    fp16_passages = fp16_embeddings[query_count:]
    int4_queries = int4_embeddings[:query_count]
    int4_passages = int4_embeddings[query_count:]
    per_query_overlap = []
    for fp16_query, int4_query in zip(fp16_queries, int4_queries):
        fp16_top_k = set(
            _top_k_indices(fp16_query, fp16_passages, top_k=effective_top_k)
        )
        int4_top_k = set(
            _top_k_indices(int4_query, int4_passages, top_k=effective_top_k)
        )
        per_query_overlap.append(len(fp16_top_k & int4_top_k) / effective_top_k)

    mean_overlap = sum(per_query_overlap) / len(per_query_overlap)
    recall_delta = max(1.0 - mean_overlap, 0.0)
    rejection_reason = None
    if not deterministic:
        rejection_reason = "Q4_K_M embedding vectors are not deterministic"
    elif recall_delta > recall_delta_tolerance + 1e-12:
        rejection_reason = "recall delta exceeds G4 tolerance"
    gate = GgufGroundingRecallGate(
        top_k=effective_top_k,
        query_count=query_count,
        passage_count=len(normalized_passages),
        per_query_overlap=tuple(per_query_overlap),
        mean_top_k_overlap=mean_overlap,
        recall_delta=recall_delta,
        tolerance=recall_delta_tolerance,
        deterministic=deterministic,
        passed=deterministic and recall_delta <= recall_delta_tolerance + 1e-12,
        rejection_reason=rejection_reason,
    )
    return GgufGroundingCertification(
        gate=gate,
        fp16_latency=fp16_latency,
        int4_latency=int4_latency,
    )


def export_gguf_int4(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    converter_path: str | Path | None = None,
    quantizer_path: str | Path | None = None,
    llama_cpp_dir: str | Path | None = None,
    embedding_binary: str | Path | None = None,
    embedding_command: Sequence[str] | None = None,
    python_executable: str | Path = sys.executable,
    source_model_id: str | None = None,
    timeout_seconds: float | None = DEFAULT_EXPORT_TIMEOUT_SECONDS,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    overwrite: bool = False,
    fp16_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any] | None = None,
    int4_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any] | None = None,
    queries: Sequence[str] = SYNTHETIC_GROUNDING_QUERIES,
    passages: Sequence[str] = SYNTHETIC_GROUNDING_PASSAGES,
    top_k: int = DEFAULT_GROUNDING_TOP_K,
    recall_delta_tolerance: float = DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE,
    embedding_context_size: int | None = 512,
    embedding_batch_size: int | None = 32,
    embedding_extra_args: Sequence[str] = (),
    clock: Clock = perf_counter,
) -> GgufInt4ExportResult:
    """Export and certify a Q4_K_M grounding GGUF bundle.

    F16 and Q8_0 are produced by the existing OM-195 exporter in a staging
    directory. The local ``llama-quantize`` executable then creates Q4_K_M
    from the staged F16 artifact. The final bundle is published only after
    top-k recall and deterministic-vector checks pass.

    ``fp16_embedder`` and ``int4_embedder`` may be injected test doubles or
    callable adapters. When either is omitted, ``embedding_binary`` (or
    ``embedding_command``) is required so this function can construct the
    corresponding local llama.cpp subprocess runners after quantization.
    """

    if fp16_embedder is None or int4_embedder is None:
        if (
            embedding_binary is None
            and embedding_command is None
            and llama_cpp_dir is None
            and os.environ.get("LLAMA_CPP_DIR") is None
        ):
            raise GgufInt4Rejected(
                "Q4_K_M grounding export requires fp16 and int4 recall evidence "
                "or a local llama.cpp embedding executable"
            )

    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")
    if embedding_timeout_seconds <= 0:
        raise ValueError("embedding_timeout_seconds must be positive")

    quantizer = _resolve_quantizer_path(
        quantizer_path,
        llama_cpp_dir=llama_cpp_dir,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target_names = {
        "config.json",
        "model-f16.gguf",
        "model-q8_0.gguf",
        GGUF_INT4_FILENAME,
        "openmed-gguf.json",
        GGUF_INT4_BENCHMARK_FILENAME,
    }
    _check_output_conflicts(destination, target_names, overwrite=overwrite)

    conversion_kwargs: dict[str, Any] = {
        "python_executable": python_executable,
        "source_model_id": source_model_id,
        "timeout_seconds": timeout_seconds,
        "overwrite": False,
    }
    if converter_path is not None:
        conversion_kwargs["converter_path"] = converter_path
    else:
        conversion_kwargs["llama_cpp_dir"] = llama_cpp_dir

    with tempfile.TemporaryDirectory(
        prefix=".openmed-gguf-int4-",
        dir=destination,
    ) as staging_value:
        staging = Path(staging_value)
        conversion = convert_gguf(model_path, staging, **conversion_kwargs)
        fp16_path = _find_artifact(conversion.artifacts, "model-f16.gguf")
        q8_path = _find_artifact(conversion.artifacts, "model-q8_0.gguf")
        q4_path = staging / GGUF_INT4_FILENAME
        _run_quantizer(
            quantizer,
            input_path=fp16_path,
            output_path=q4_path,
            timeout_seconds=timeout_seconds,
        )

        resolved_fp16 = fp16_embedder
        resolved_int4 = int4_embedder
        if resolved_fp16 is None:
            resolved_fp16 = _build_runtime(
                fp16_path,
                embedding_binary=embedding_binary,
                embedding_command=embedding_command,
                llama_cpp_dir=llama_cpp_dir,
                timeout_seconds=embedding_timeout_seconds,
                context_size=embedding_context_size,
                batch_size=embedding_batch_size,
                extra_args=embedding_extra_args,
            )
        if resolved_int4 is None:
            resolved_int4 = _build_runtime(
                q4_path,
                embedding_binary=embedding_binary,
                embedding_command=embedding_command,
                llama_cpp_dir=llama_cpp_dir,
                timeout_seconds=embedding_timeout_seconds,
                context_size=embedding_context_size,
                batch_size=embedding_batch_size,
                extra_args=embedding_extra_args,
            )

        try:
            certification = certify_gguf_grounding(
                resolved_fp16,
                resolved_int4,
                queries=queries,
                passages=passages,
                top_k=top_k,
                recall_delta_tolerance=recall_delta_tolerance,
                clock=clock,
            )
        except ValueError as exc:
            raise GgufInt4Rejected(
                "Q4_K_M grounding certification evidence is invalid"
            ) from exc

        benchmark_path = staging / GGUF_INT4_BENCHMARK_FILENAME
        source_id = source_model_id or Path(model_path).expanduser().name
        manifest = _read_json(conversion.manifest_path)
        source_revision = str(manifest.get("source_revision") or "local")
        fixture_sha256 = grounding_fixture_sha256(queries, passages)
        report = _grounding_benchmark_report(
            source_model_id=source_id,
            source_revision=source_revision,
            fixture_sha256=fixture_sha256,
            artifact_size_bytes=q4_path.stat().st_size,
            certification=certification,
        )
        report.write_json(benchmark_path)

        if not certification.gate.passed:
            raise GgufInt4Rejected(
                "Q4_K_M grounding artifact rejected: "
                f"{certification.gate.rejection_reason or 'G4 gate failed'}",
                gate=certification.gate,
                benchmark_report_path=benchmark_path,
            )

        _update_manifest(
            conversion.manifest_path,
            q4_path=q4_path,
            benchmark_path=benchmark_path,
            certification=certification,
            fixture_sha256=fixture_sha256,
        )
        _publish_staged_bundle(
            staging,
            destination,
            target_names,
        )

    return GgufInt4ExportResult(
        output_dir=destination,
        manifest_path=destination / "openmed-gguf.json",
        benchmark_report_path=destination / GGUF_INT4_BENCHMARK_FILENAME,
        fp16_path=destination / "model-f16.gguf",
        q8_0_path=destination / "model-q8_0.gguf",
        q4_k_m_path=destination / GGUF_INT4_FILENAME,
        certification=certification,
    )


def quantize_gguf_int4(
    model_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> GgufInt4ExportResult:
    """Alias for :func:`export_gguf_int4` with quantization-oriented naming."""

    return export_gguf_int4(model_path, output_dir, **kwargs)


def quantize_gguf_int4_grounding(
    model_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> GgufInt4ExportResult:
    """Alias for :func:`export_gguf_int4` for grounding-oriented callers."""

    return export_gguf_int4(model_path, output_dir, **kwargs)


def certify_gguf_grounding_recall(
    fp16_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    int4_embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    **kwargs: Any,
) -> GgufGroundingCertification:
    """Alias for :func:`certify_gguf_grounding`."""

    return certify_gguf_grounding(fp16_embedder, int4_embedder, **kwargs)


def validate_gguf_int4_artifact(artifact_dir: str | Path) -> None:
    """Validate the local manifest and passing G4 report before loading."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "openmed-gguf.json"
    report_path = root / GGUF_INT4_BENCHMARK_FILENAME
    artifact_path = root / GGUF_INT4_FILENAME
    if not artifact_path.is_file():
        raise GgufInt4Rejected(
            f"GGUF grounding artifact is missing {GGUF_INT4_FILENAME}"
        )
    if not manifest_path.is_file():
        raise GgufInt4Rejected("GGUF grounding artifact is missing openmed-gguf.json")
    if not report_path.is_file():
        raise GgufInt4Rejected(
            f"GGUF grounding artifact is missing {GGUF_INT4_BENCHMARK_FILENAME}"
        )

    try:
        manifest = _read_json(manifest_path)
        report = BenchmarkReport.read_json(report_path)
        retrieval = report.metrics.get("retrieval")
        metadata = report.metadata
        quantization = manifest.get("quantization")
        certification = manifest.get("certification")
        recall_delta = _finite_float(
            retrieval.get("recall_delta") if isinstance(retrieval, Mapping) else None
        )
        tolerance = _finite_float(
            retrieval.get("tolerance") if isinstance(retrieval, Mapping) else None
        )
        manifest_delta = _finite_float(
            quantization.get("quant_recall_delta")
            if isinstance(quantization, Mapping)
            else None
        )
        manifest_tolerance = _finite_float(
            quantization.get("recall_delta_limit")
            if isinstance(quantization, Mapping)
            else None
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GgufInt4Rejected(
            "GGUF grounding certification metadata is invalid"
        ) from exc

    valid = (
        manifest.get("format") == "openmed-gguf"
        and isinstance(quantization, Mapping)
        and quantization.get("scheme") == "Q4_K_M"
        and quantization.get("certified") is True
        and quantization.get("recall_delta_path") == report_path.name
        and isinstance(certification, Mapping)
        and certification.get("gate") == "G4"
        and certification.get("passed") is True
        and certification.get("report_path") == report_path.name
        and manifest.get("certified") is True
        and report.suite == GGUF_GROUNDING_BENCHMARK_SUITE
        and isinstance(retrieval, Mapping)
        and retrieval.get("format") == GGUF_INT4_FORMAT
        and retrieval.get("gate") == "G4"
        and retrieval.get("passed") is True
        and retrieval.get("deterministic") is True
        and recall_delta is not None
        and tolerance is not None
        and 0.0 <= recall_delta <= tolerance <= 1.0
        and manifest_delta is not None
        and manifest_tolerance is not None
        and math.isclose(manifest_delta, recall_delta, abs_tol=1e-12)
        and math.isclose(manifest_tolerance, tolerance, abs_tol=1e-12)
        and quantization.get("recall_delta_path") == report_path.name
        and metadata.get("format") == GGUF_INT4_PROFILE
        and metadata.get("format_version") == GGUF_INT4_PROFILE_VERSION
        and metadata.get("certified") is True
        and metadata.get("deterministic") is True
    )
    if not valid:
        raise GgufInt4Rejected(
            "GGUF grounding artifact does not have consistent passing G4 evidence",
            benchmark_report_path=report_path,
        )


def load_gguf_grounding_embedder(
    artifact_dir: str | Path,
    executable: str | Path | None = None,
    *,
    llama_cpp_dir: str | Path | None = None,
    command: Sequence[str] | None = None,
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    context_size: int | None = 512,
    batch_size: int | None = 32,
    extra_args: Sequence[str] = (),
) -> LlamaCppEmbeddingRuntime:
    """Load a Q4_K_M grounding runtime only after validating its G4 report."""

    root = Path(artifact_dir).expanduser().resolve()
    validate_gguf_int4_artifact(root)
    return LlamaCppEmbeddingRuntime(
        root / GGUF_INT4_FILENAME,
        executable,
        llama_cpp_dir=llama_cpp_dir,
        command=command,
        timeout_seconds=timeout_seconds,
        context_size=context_size,
        batch_size=batch_size,
        extra_args=extra_args,
    )


def grounding_fixture_sha256(
    queries: Sequence[str],
    passages: Sequence[str],
) -> str:
    """Return a stable digest for ordered synthetic retrieval fixtures."""

    normalized_queries = _normalize_texts(queries, name="queries")
    normalized_passages = _normalize_texts(passages, name="passages")
    digest = hashlib.sha256()
    for kind, values in (
        ("query", normalized_queries),
        ("passage", normalized_passages),
    ):
        for value in values:
            digest.update(f"{kind}\0{value}\n".encode("utf-8"))
    return digest.hexdigest()


def _build_runtime(
    model_path: Path,
    *,
    embedding_binary: str | Path | None,
    embedding_command: Sequence[str] | None,
    llama_cpp_dir: str | Path | None,
    timeout_seconds: float,
    context_size: int | None,
    batch_size: int | None,
    extra_args: Sequence[str],
) -> LlamaCppEmbeddingRuntime:
    return LlamaCppEmbeddingRuntime(
        model_path,
        embedding_binary,
        llama_cpp_dir=(
            llama_cpp_dir
            if embedding_command is None and embedding_binary is None
            else None
        ),
        command=embedding_command,
        timeout_seconds=timeout_seconds,
        context_size=context_size,
        batch_size=batch_size,
        extra_args=extra_args,
    )


def _resolve_quantizer_path(
    quantizer_path: str | Path | None,
    *,
    llama_cpp_dir: str | Path | None,
) -> Path:
    if quantizer_path is not None:
        resolved = Path(quantizer_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"llama.cpp quantizer not found: {resolved}")
        return resolved

    checkout_value = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if checkout_value is None:
        raise FileNotFoundError(
            "llama.cpp quantizer is not configured; pass quantizer_path, "
            "llama_cpp_dir, or set LLAMA_CPP_DIR"
        )
    checkout = Path(checkout_value).expanduser().resolve()
    for name in _QUANTIZER_NAMES:
        for candidate in (checkout / name, checkout / "build" / "bin" / name):
            if candidate.is_file():
                return candidate
    names = " or ".join(_QUANTIZER_NAMES)
    raise FileNotFoundError(f"{names} not found in llama.cpp checkout: {checkout}")


def _run_quantizer(
    quantizer: Path,
    *,
    input_path: Path,
    output_path: Path,
    timeout_seconds: float | None,
) -> None:
    command = [str(quantizer), str(input_path), str(output_path), "Q4_K_M"]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise GgufInt4ExportError(
            f"Q4_K_M quantization exceeded {timeout_seconds} seconds"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise GgufInt4ExportError("llama.cpp Q4_K_M quantization failed") from exc
    except OSError as exc:
        raise GgufInt4ExportError("could not start llama.cpp Q4_K_M quantizer") from exc
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise GgufInt4ExportError(
            f"Q4_K_M quantization did not write {output_path.name}"
        )


def _timed_embeddings(
    embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    texts: Sequence[str],
    *,
    clock: Clock,
) -> tuple[tuple[tuple[float, ...], ...], Any]:
    started_at = clock()
    embeddings = _collect_embeddings(embedder, texts)
    elapsed_ms = max((clock() - started_at) * 1000.0, 0.0)
    per_item_ms = elapsed_ms / len(texts)
    return embeddings, compute_latency_summary([per_item_ms] * len(texts))


def _collect_embeddings(
    embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    texts: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    encode = getattr(embedder, "encode", None)
    raw_embeddings = encode(texts) if callable(encode) else embedder(texts)
    return _normalize_embeddings(raw_embeddings, expected_count=len(texts))


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
    try:
        values = list(texts)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of non-empty strings") from exc
    normalized = [text.strip() for text in values if isinstance(text, str)]
    if (
        len(normalized) != len(values)
        or not normalized
        or any(not text for text in normalized)
    ):
        raise ValueError(f"{name} must contain only non-empty strings")
    return normalized


def _grounding_benchmark_report(
    *,
    source_model_id: str,
    source_revision: str,
    fixture_sha256: str,
    artifact_size_bytes: int,
    certification: GgufGroundingCertification,
) -> BenchmarkReport:
    return BenchmarkReport(
        suite=GGUF_GROUNDING_BENCHMARK_SUITE,
        model_name=source_model_id,
        device="llama.cpp",
        fixture_count=certification.gate.query_count,
        generated_at=_utc_now(),
        metrics={
            "retrieval": certification.gate.to_dict(),
            "latency": {
                "fp16": certification.fp16_latency.to_dict(),
                "int4": certification.int4_latency.to_dict(),
            },
            "resources": {
                "model_size_bytes": artifact_size_bytes,
                "model_size_mib": artifact_size_bytes / (1024 * 1024),
            },
        },
        metadata={
            "format": GGUF_INT4_PROFILE,
            "format_version": GGUF_INT4_PROFILE_VERSION,
            "source_revision": source_revision,
            "certified": certification.gate.passed,
            "deterministic": certification.gate.deterministic,
            "quantization": {
                "scheme": "Q4_K_M",
                "source": "model-f16.gguf",
                "gate": "G4",
            },
            "grounding_fixture_sha256": fixture_sha256,
        },
    )


def _update_manifest(
    manifest_path: Path,
    *,
    q4_path: Path,
    benchmark_path: Path,
    certification: GgufGroundingCertification,
    fixture_sha256: str,
) -> None:
    manifest = _read_json(manifest_path)
    gate = certification.gate
    metadata = {
        "profile": GGUF_INT4_PROFILE,
        "profile_version": GGUF_INT4_PROFILE_VERSION,
        "gate": "G4",
        "certified": gate.passed,
        "deterministic": gate.deterministic,
        "recall_delta": gate.recall_delta,
        "recall_delta_limit": gate.tolerance,
        "recall_delta_path": benchmark_path.name,
        "grounding_fixture_sha256": fixture_sha256,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    q4_record = {
        "format": "gguf",
        "path": q4_path.relative_to(manifest_path.parent).as_posix(),
        "precision": "q4_k_m",
        "quantization": "Q4_K_M",
        "metadata": metadata,
    }
    replaced = False
    normalized_artifacts: list[dict[str, Any]] = []
    for item in artifacts:
        if isinstance(item, Mapping) and item.get("path") == GGUF_INT4_FILENAME:
            normalized_artifacts.append(q4_record)
            replaced = True
        elif isinstance(item, Mapping):
            normalized_artifacts.append(dict(item))
    if not replaced:
        normalized_artifacts.append(q4_record)

    manifest["artifacts"] = normalized_artifacts
    manifest["quantization"] = {
        "scheme": "Q4_K_M",
        "source_artifact": "model-f16.gguf",
        "certified": gate.passed,
        "quant_recall_delta": gate.recall_delta,
        "recall_delta_limit": gate.tolerance,
        "recall_delta_path": benchmark_path.name,
    }
    manifest["certified"] = gate.passed
    manifest["quant_recall_delta"] = gate.recall_delta
    manifest["recall_delta_path"] = benchmark_path.name
    manifest["certification"] = {
        "gate": "G4",
        "metric": "top_k_overlap",
        "passed": gate.passed,
        "deterministic": gate.deterministic,
        "limit": gate.tolerance,
        "report_path": benchmark_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_staged_bundle(
    staging: Path,
    destination: Path,
    target_names: set[str],
) -> None:
    for name in sorted(target_names):
        source = staging / name
        if not source.is_file():
            raise GgufInt4ExportError(f"staged GGUF bundle is missing {name}")
    for name in sorted(target_names):
        os.replace(staging / name, destination / name)


def _check_output_conflicts(
    output_dir: Path,
    filenames: Sequence[str] | set[str],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    conflicts = sorted(name for name in filenames if (output_dir / name).exists())
    if conflicts:
        raise FileExistsError(
            "GGUF int4 output already exists ("
            + ", ".join(conflicts)
            + "); pass overwrite=True to replace it"
        )


def _find_artifact(artifacts: Sequence[Any], name: str) -> Path:
    for artifact in artifacts:
        path = getattr(artifact, "path", None)
        if isinstance(path, Path) and path.name == name:
            return path
    raise GgufInt4ExportError(f"OM-195 GGUF export did not produce {name}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_EMBEDDING_TIMEOUT_SECONDS",
    "DEFAULT_EXPORT_TIMEOUT_SECONDS",
    "DEFAULT_GROUNDING_RECALL_DELTA_TOLERANCE",
    "DEFAULT_GROUNDING_TOP_K",
    "GGUF_GROUNDING_BENCHMARK_FILENAME",
    "GGUF_GROUNDING_BENCHMARK_SUITE",
    "GGUF_INT4_BENCHMARK_FILENAME",
    "GGUF_INT4_FILENAME",
    "GGUF_INT4_FORMAT",
    "GGUF_INT4_PROFILE",
    "GGUF_INT4_PROFILE_VERSION",
    "GgufGroundingCertification",
    "GgufGroundingRecallGate",
    "GgufInt4ExportError",
    "GgufInt4ExportResult",
    "GgufInt4Rejected",
    "GroundingEmbedder",
    "Q4_K_M_FILENAME",
    "SYNTHETIC_GROUNDING_PASSAGES",
    "SYNTHETIC_GROUNDING_QUERIES",
    "certify_gguf_grounding",
    "certify_gguf_grounding_recall",
    "export_gguf_int4",
    "grounding_fixture_sha256",
    "load_gguf_grounding_embedder",
    "quantize_gguf_int4",
    "quantize_gguf_int4_grounding",
    "validate_gguf_int4_artifact",
]
