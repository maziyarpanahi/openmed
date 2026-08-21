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
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from openmed.eval.metrics import compute_latency_summary
from openmed.eval.quant_delta import INT4_RECALL_DELTA_LIMIT
from openmed.eval.report import BenchmarkReport
from openmed.gguf.convert import convert as convert_gguf
from openmed.onnx.gguf_embed_runtime import (
    MAX_EMBEDDING_DIMENSION,
    MAX_EMBEDDING_TEXT_CHARS,
    LlamaCppEmbeddingRuntime,
)

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
MAX_GROUNDING_QUERIES = 256
MAX_GROUNDING_PASSAGES = 4096
MAX_GROUNDING_TOTAL_CHARS = 4 * 1024 * 1024
MAX_CERTIFICATION_JSON_BYTES = 4 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024

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

    normalized_queries = _normalize_texts(
        queries,
        name="queries",
        maximum=MAX_GROUNDING_QUERIES,
    )
    normalized_passages = _normalize_texts(
        passages,
        name="passages",
        maximum=MAX_GROUNDING_PASSAGES,
    )
    if (
        sum(len(text) for text in normalized_queries)
        + sum(len(text) for text in normalized_passages)
        > MAX_GROUNDING_TOTAL_CHARS
    ):
        raise ValueError("queries and passages exceed the grounding text-size limit")
    validated_top_k = _positive_int(top_k, name="top_k")
    validated_tolerance = _fraction(
        recall_delta_tolerance,
        name="recall_delta_tolerance",
    )

    effective_top_k = min(validated_top_k, len(normalized_passages))
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
    elif recall_delta > validated_tolerance + 1e-12:
        rejection_reason = "recall delta exceeds G4 tolerance"
    gate = GgufGroundingRecallGate(
        top_k=effective_top_k,
        query_count=query_count,
        passage_count=len(normalized_passages),
        per_query_overlap=tuple(per_query_overlap),
        mean_top_k_overlap=mean_overlap,
        recall_delta=recall_delta,
        tolerance=validated_tolerance,
        deterministic=deterministic,
        passed=deterministic and recall_delta <= validated_tolerance + 1e-12,
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

    validated_timeout = _optional_positive_finite_float(
        timeout_seconds,
        name="timeout_seconds",
    )
    validated_embedding_timeout = _positive_finite_float(
        embedding_timeout_seconds,
        name="embedding_timeout_seconds",
    )

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
        "timeout_seconds": validated_timeout,
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
            timeout_seconds=validated_timeout,
        )

        resolved_fp16 = fp16_embedder
        resolved_int4 = int4_embedder
        if resolved_fp16 is None:
            resolved_fp16 = _build_runtime(
                fp16_path,
                embedding_binary=embedding_binary,
                embedding_command=embedding_command,
                llama_cpp_dir=llama_cpp_dir,
                timeout_seconds=validated_embedding_timeout,
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
                timeout_seconds=validated_embedding_timeout,
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
        artifact_size_bytes = q4_path.stat().st_size
        artifact_sha256 = _sha256_file(q4_path)
        report = _grounding_benchmark_report(
            source_model_id=source_id,
            source_revision=source_revision,
            fixture_sha256=fixture_sha256,
            artifact_size_bytes=artifact_size_bytes,
            artifact_sha256=artifact_sha256,
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
            artifact_size_bytes=artifact_size_bytes,
            artifact_sha256=artifact_sha256,
        )
        _publish_staged_bundle(
            staging,
            destination,
            target_names,
            overwrite=overwrite,
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
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise GgufInt4Rejected(
            f"GGUF grounding artifact is missing {GGUF_INT4_FILENAME}"
        )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise GgufInt4Rejected("GGUF grounding artifact is missing openmed-gguf.json")
    if not report_path.is_file() or report_path.is_symlink():
        raise GgufInt4Rejected(
            f"GGUF grounding artifact is missing {GGUF_INT4_BENCHMARK_FILENAME}"
        )

    try:
        manifest = _read_json(manifest_path)
        report = _read_json(report_path)
        metrics = _require_mapping(report.get("metrics"), name="report metrics")
        retrieval = _require_mapping(
            metrics.get("retrieval"),
            name="retrieval evidence",
        )
        resources = _require_mapping(
            metrics.get("resources"),
            name="resource evidence",
        )
        latency = _require_mapping(
            metrics.get("latency"),
            name="latency evidence",
        )
        metadata = _require_mapping(report.get("metadata"), name="report metadata")
        quantization = manifest.get("quantization")
        certification = manifest.get("certification")
        quantization = _require_mapping(
            quantization,
            name="manifest quantization",
        )
        certification = _require_mapping(
            certification,
            name="manifest certification",
        )
        artifact_record = _q4_artifact_record(manifest)
        artifact_metadata = _require_mapping(
            artifact_record.get("metadata"),
            name="Q4_K_M artifact metadata",
        )
        recall_delta, tolerance, query_count, passage_count = (
            _validate_retrieval_evidence(retrieval)
        )
        _validate_latency_evidence(
            latency,
            expected_count=query_count + passage_count,
        )
        manifest_delta = _strict_finite_float(
            quantization.get("quant_recall_delta"),
            name="manifest recall delta",
        )
        manifest_tolerance = _strict_finite_float(
            quantization.get("recall_delta_limit"),
            name="manifest recall tolerance",
        )
        top_level_delta = _strict_finite_float(
            manifest.get("quant_recall_delta"),
            name="top-level manifest recall delta",
        )
        actual_size = artifact_path.stat().st_size
        with artifact_path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ValueError("Q4_K_M artifact does not have a GGUF header")
        actual_sha256 = _sha256_file(artifact_path)
        resource_size = _positive_int(
            resources.get("model_size_bytes"),
            name="report model_size_bytes",
        )
        resource_mib = _strict_finite_float(
            resources.get("model_size_mib"),
            name="report model_size_mib",
        )
        resource_sha256 = _sha256_value(
            resources.get("artifact_sha256"),
            name="report artifact_sha256",
        )
        record_size = _positive_int(
            artifact_record.get("size_bytes"),
            name="artifact size_bytes",
        )
        record_sha256 = _sha256_value(
            artifact_record.get("sha256"),
            name="artifact sha256",
        )
        artifact_metadata_size = _positive_int(
            artifact_metadata.get("size_bytes"),
            name="artifact metadata size_bytes",
        )
        artifact_metadata_sha256 = _sha256_value(
            artifact_metadata.get("sha256"),
            name="artifact metadata sha256",
        )
        artifact_metadata_delta = _strict_finite_float(
            artifact_metadata.get("recall_delta"),
            name="artifact metadata recall delta",
        )
        artifact_metadata_tolerance = _strict_finite_float(
            artifact_metadata.get("recall_delta_limit"),
            name="artifact metadata recall tolerance",
        )
        manifest_size = _positive_int(
            quantization.get("artifact_size_bytes"),
            name="manifest artifact_size_bytes",
        )
        manifest_sha256 = _sha256_value(
            quantization.get("artifact_sha256"),
            name="manifest artifact_sha256",
        )
        fixture_sha256 = _sha256_value(
            metadata.get("grounding_fixture_sha256"),
            name="report grounding fixture hash",
        )
        artifact_fixture_sha256 = _sha256_value(
            artifact_metadata.get("grounding_fixture_sha256"),
            name="artifact grounding fixture hash",
        )
        certification_fixture_sha256 = _sha256_value(
            certification.get("grounding_fixture_sha256"),
            name="certification grounding fixture hash",
        )
        certification_sha256 = _sha256_value(
            certification.get("artifact_sha256"),
            name="certification artifact hash",
        )
        certification_limit = _strict_finite_float(
            certification.get("limit"),
            name="certification limit",
        )
        report_quantization = _require_mapping(
            metadata.get("quantization"),
            name="report quantization",
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as exc:
        raise GgufInt4Rejected(
            "GGUF grounding certification metadata is invalid"
        ) from exc

    valid = (
        manifest.get("format") == "openmed-gguf"
        and _is_exact_int(manifest.get("format_version"), 1)
        and artifact_record.get("format") == "gguf"
        and artifact_record.get("path") == GGUF_INT4_FILENAME
        and artifact_record.get("precision") == "q4_k_m"
        and artifact_record.get("quantization") == "Q4_K_M"
        and quantization.get("scheme") == "Q4_K_M"
        and quantization.get("source_artifact") == "model-f16.gguf"
        and quantization.get("certified") is True
        and quantization.get("recall_delta_path") == report_path.name
        and certification.get("gate") == "G4"
        and certification.get("metric") == "top_k_overlap"
        and certification.get("passed") is True
        and certification.get("deterministic") is True
        and certification.get("report_path") == report_path.name
        and manifest.get("certified") is True
        and manifest.get("recall_delta_path") == report_path.name
        and report.get("suite") == GGUF_GROUNDING_BENCHMARK_SUITE
        and report.get("device") == "llama.cpp"
        and _is_exact_int(report.get("fixture_count"), query_count)
        and math.isclose(
            manifest_delta,
            recall_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            manifest_tolerance,
            tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            top_level_delta,
            recall_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            certification_limit,
            tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and metadata.get("format") == GGUF_INT4_PROFILE
        and _is_exact_int(
            metadata.get("format_version"),
            GGUF_INT4_PROFILE_VERSION,
        )
        and metadata.get("certified") is True
        and metadata.get("deterministic") is True
        and report_quantization.get("scheme") == "Q4_K_M"
        and artifact_metadata.get("profile") == GGUF_INT4_PROFILE
        and _is_exact_int(
            artifact_metadata.get("profile_version"),
            GGUF_INT4_PROFILE_VERSION,
        )
        and artifact_metadata.get("gate") == "G4"
        and artifact_metadata.get("certified") is True
        and artifact_metadata.get("deterministic") is True
        and artifact_metadata.get("recall_delta_path") == report_path.name
        and math.isclose(
            artifact_metadata_delta,
            recall_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            artifact_metadata_tolerance,
            tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and report_quantization.get("source") == "model-f16.gguf"
        and report_quantization.get("gate") == "G4"
        and record_size
        == artifact_metadata_size
        == manifest_size
        == resource_size
        == actual_size
        and math.isclose(
            resource_mib,
            actual_size / (1024 * 1024),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and record_sha256
        == artifact_metadata_sha256
        == manifest_sha256
        == certification_sha256
        == resource_sha256
        == actual_sha256
        and fixture_sha256 == artifact_fixture_sha256 == certification_fixture_sha256
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

    normalized_queries = _normalize_texts(
        queries,
        name="queries",
        maximum=MAX_GROUNDING_QUERIES,
    )
    normalized_passages = _normalize_texts(
        passages,
        name="passages",
        maximum=MAX_GROUNDING_PASSAGES,
    )
    if (
        sum(len(text) for text in normalized_queries)
        + sum(len(text) for text in normalized_passages)
        > MAX_GROUNDING_TOTAL_CHARS
    ):
        raise ValueError("queries and passages exceed the grounding text-size limit")
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
    if not output_path.is_file() or output_path.stat().st_size < 4:
        raise GgufInt4ExportError(
            f"Q4_K_M quantization did not write {output_path.name}"
        )
    with output_path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise GgufInt4ExportError(
            f"Q4_K_M quantization wrote an invalid GGUF header to {output_path.name}"
        )


def _timed_embeddings(
    embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    texts: Sequence[str],
    *,
    clock: Clock,
) -> tuple[tuple[tuple[float, ...], ...], Any]:
    started_at = _finite_clock_sample(clock(), name="clock start")
    embeddings = _collect_embeddings(embedder, texts)
    finished_at = _finite_clock_sample(clock(), name="clock finish")
    elapsed_ms = max((finished_at - started_at) * 1000.0, 0.0)
    if not math.isfinite(elapsed_ms):
        raise ValueError("embedding latency must be finite")
    per_item_ms = elapsed_ms / len(texts)
    return embeddings, compute_latency_summary([per_item_ms] * len(texts))


def _collect_embeddings(
    embedder: GroundingEmbedder | Callable[[Sequence[str]], Any],
    texts: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    encode = getattr(embedder, "encode", None)
    if callable(encode):
        raw_embeddings = encode(texts)
    elif callable(embedder):
        raw_embeddings = embedder(texts)
    else:
        raise ValueError("embedder must be callable or provide encode(texts)")
    return _normalize_embeddings(raw_embeddings, expected_count=len(texts))


def _normalize_embeddings(
    embeddings: Any,
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    if expected_count <= 0 or expected_count > (
        MAX_GROUNDING_QUERIES + MAX_GROUNDING_PASSAGES
    ):
        raise ValueError("expected embedding count exceeds the grounding limit")
    if isinstance(embeddings, (str, bytes, Mapping)):
        raise ValueError("embedder output must be a sequence of vectors")
    try:
        rows = list(islice(iter(embeddings), expected_count + 1))
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
        if isinstance(row, (str, bytes, Mapping)):
            raise ValueError("embedding vectors must contain numeric values")
        try:
            raw_values = list(islice(iter(row), MAX_EMBEDDING_DIMENSION + 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding vectors must contain numeric values") from exc
        if len(raw_values) > MAX_EMBEDDING_DIMENSION:
            raise ValueError("embedding vector exceeds the dimension limit")
        if any(isinstance(value, (str, bytes, bool)) for value in raw_values):
            raise ValueError("embedding vectors must contain numeric values")
        try:
            vector = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding vectors must contain numeric values") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vectors must be non-empty and finite")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("embedding vectors must use one consistent dimension")
        scale = max(abs(value) for value in vector)
        if scale == 0.0:
            raise ValueError("embedding vectors must have non-zero norm")
        scaled_norm = math.sqrt(math.fsum((value / scale) ** 2 for value in vector))
        if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
            raise ValueError("embedding vector norm must be finite and non-zero")
        normalized.append(tuple((value / scale) / scaled_norm for value in vector))
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


def _normalize_texts(
    texts: Sequence[str],
    *,
    name: str,
    maximum: int,
) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    try:
        values = list(islice(iter(texts), maximum + 1))
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of non-empty strings") from exc
    if len(values) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} items")
    normalized = [text.strip() for text in values if isinstance(text, str)]
    if (
        len(normalized) != len(values)
        or not normalized
        or any(not text for text in normalized)
    ):
        raise ValueError(f"{name} must contain only non-empty strings")
    if any("\0" in text for text in normalized):
        raise ValueError(f"{name} must not contain NUL characters")
    if any(len(text) > MAX_EMBEDDING_TEXT_CHARS for text in normalized):
        raise ValueError(
            f"each {name} item must contain at most "
            f"{MAX_EMBEDDING_TEXT_CHARS} characters"
        )
    return normalized


def _grounding_benchmark_report(
    *,
    source_model_id: str,
    source_revision: str,
    fixture_sha256: str,
    artifact_size_bytes: int,
    artifact_sha256: str,
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
                "artifact_sha256": artifact_sha256,
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
    artifact_size_bytes: int,
    artifact_sha256: str,
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
        "size_bytes": artifact_size_bytes,
        "sha256": artifact_sha256,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    q4_record = {
        "format": "gguf",
        "path": q4_path.relative_to(manifest_path.parent).as_posix(),
        "precision": "q4_k_m",
        "quantization": "Q4_K_M",
        "size_bytes": artifact_size_bytes,
        "sha256": artifact_sha256,
        "metadata": metadata,
    }
    replaced = False
    normalized_artifacts: list[dict[str, Any]] = []
    for item in artifacts:
        if isinstance(item, Mapping) and item.get("path") == GGUF_INT4_FILENAME:
            if not replaced:
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
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_sha256": artifact_sha256,
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
        "grounding_fixture_sha256": fixture_sha256,
        "artifact_sha256": artifact_sha256,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_staged_bundle(
    staging: Path,
    destination: Path,
    target_names: set[str],
    *,
    overwrite: bool,
) -> None:
    for name in sorted(target_names):
        source = staging / name
        if not source.is_file() or source.is_symlink():
            raise GgufInt4ExportError(f"staged GGUF bundle is missing {name}")
    existing = [
        name for name in sorted(target_names) if os.path.lexists(destination / name)
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "GGUF int4 output appeared during export ("
            + ", ".join(existing)
            + "); rerun with overwrite=True to replace it"
        )
    directory_conflicts = [name for name in existing if (destination / name).is_dir()]
    if directory_conflicts:
        raise IsADirectoryError(
            "refusing to replace output directories: " + ", ".join(directory_conflicts)
        )

    rollback = staging / ".rollback"
    rollback.mkdir()
    moved_existing: list[str] = []
    published: list[str] = []
    try:
        for name in existing:
            os.replace(destination / name, rollback / name)
            moved_existing.append(name)
        for name in sorted(target_names):
            os.replace(staging / name, destination / name)
            published.append(name)
    except OSError as exc:
        for name in reversed(published):
            published_path = destination / name
            if os.path.lexists(published_path):
                os.replace(published_path, staging / name)
        for name in reversed(moved_existing):
            backup_path = rollback / name
            if os.path.lexists(backup_path):
                os.replace(backup_path, destination / name)
        raise GgufInt4ExportError(
            "could not publish the staged GGUF bundle; previous outputs restored"
        ) from exc


def _check_output_conflicts(
    output_dir: Path,
    filenames: Sequence[str] | set[str],
    *,
    overwrite: bool,
) -> None:
    directory_conflicts = sorted(
        name
        for name in filenames
        if os.path.lexists(output_dir / name) and (output_dir / name).is_dir()
    )
    if directory_conflicts:
        raise IsADirectoryError(
            "refusing to replace output directories: " + ", ".join(directory_conflicts)
        )
    if overwrite:
        return
    conflicts = sorted(name for name in filenames if os.path.lexists(output_dir / name))
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


def _q4_artifact_record(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(item, Mapping) for item in artifacts
    ):
        raise ValueError("manifest artifacts must be a list of objects")
    matches = [item for item in artifacts if item.get("path") == GGUF_INT4_FILENAME]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one Q4_K_M artifact")
    return matches[0]


def _validate_retrieval_evidence(
    retrieval: Mapping[str, Any],
) -> tuple[float, float, int, int]:
    if (
        retrieval.get("format") != GGUF_INT4_FORMAT
        or retrieval.get("gate") != "G4"
        or retrieval.get("metric") != "top_k_overlap"
        or retrieval.get("passed") is not True
        or retrieval.get("deterministic") is not True
        or retrieval.get("rejection_reason") is not None
    ):
        raise ValueError("retrieval evidence does not contain a passing G4 gate")

    top_k = _positive_int(retrieval.get("top_k"), name="retrieval top_k")
    query_count = _positive_int(
        retrieval.get("query_count"),
        name="retrieval query_count",
    )
    passage_count = _positive_int(
        retrieval.get("passage_count"),
        name="retrieval passage_count",
    )
    if query_count > MAX_GROUNDING_QUERIES:
        raise ValueError("retrieval query_count exceeds the grounding limit")
    if passage_count > MAX_GROUNDING_PASSAGES:
        raise ValueError("retrieval passage_count exceeds the grounding limit")
    if top_k > passage_count:
        raise ValueError("retrieval top_k exceeds passage_count")

    overlaps = retrieval.get("per_query_overlap")
    if not isinstance(overlaps, list) or len(overlaps) != query_count:
        raise ValueError("per_query_overlap must match query_count")
    normalized_overlaps = [
        _fraction(value, name="per-query overlap") for value in overlaps
    ]
    mean_overlap = _fraction(
        retrieval.get("mean_top_k_overlap"),
        name="mean top-k overlap",
    )
    expected_mean = math.fsum(normalized_overlaps) / query_count
    if not math.isclose(
        mean_overlap,
        expected_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("mean top-k overlap does not match per-query evidence")

    recall_delta = _fraction(
        retrieval.get("recall_delta"),
        name="recall delta",
    )
    tolerance = _fraction(retrieval.get("tolerance"), name="recall tolerance")
    expected_delta = max(1.0 - mean_overlap, 0.0)
    if not math.isclose(
        recall_delta,
        expected_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("recall delta does not match overlap evidence")
    if recall_delta > tolerance + 1e-12:
        raise ValueError("passing recall evidence exceeds its tolerance")
    return recall_delta, tolerance, query_count, passage_count


def _validate_latency_evidence(
    latency: Mapping[str, Any],
    *,
    expected_count: int,
) -> None:
    for precision in ("fp16", "int4"):
        record = _require_mapping(
            latency.get(precision),
            name=f"{precision} latency",
        )
        if not _is_exact_int(record.get("count"), expected_count):
            raise ValueError(f"{precision} latency count does not match fixtures")
        p50 = _strict_finite_float(record.get("p50_ms"), name=f"{precision} p50")
        p95 = _strict_finite_float(record.get("p95_ms"), name=f"{precision} p95")
        p99 = _strict_finite_float(record.get("p99_ms"), name=f"{precision} p99")
        if not 0.0 <= p50 <= p95 <= p99:
            raise ValueError(f"{precision} latency percentiles are invalid")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _strict_finite_float(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _fraction(value: Any, *, name: str) -> float:
    parsed = _strict_finite_float(value, name=name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > (1 << 63) - 1:
        raise ValueError(f"{name} exceeds the signed 64-bit limit")
    return value


def _is_exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _positive_finite_float(value: Any, *, name: str) -> float:
    parsed = _strict_finite_float(value, name=name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive_finite_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite_float(value, name=name)


def _finite_clock_sample(value: Any, *, name: str) -> float:
    return _strict_finite_float(value, name=name)


def _sha256_value(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CERTIFICATION_JSON_BYTES:
        raise ValueError(f"{path.name} exceeds the certification JSON size limit")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


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
