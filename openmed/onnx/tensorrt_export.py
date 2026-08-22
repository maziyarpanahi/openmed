"""TensorRT engine export, certification, and benchmark helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from openmed.eval.quant_delta import (
    QuantRecallDeltaResult,
    evaluate_quant_recall_delta,
)
from openmed.eval.report import BenchmarkReport
from openmed.onnx.openvino_export import token_spans_from_logits
from openmed.onnx.tensorrt_session import TensorRTTokenClassificationSession
from openmed.torch.calibration import (
    calibration_texts_sha256,
    load_quantization_calibration_texts,
)

TENSORRT_PROFILE_NAME = "tensorrt"
TENSORRT_ENGINE_FORMAT = "tensorrt-engine"
TENSORRT_FP16_FORMAT = "tensorrt-fp16"
TENSORRT_INT8_FORMAT = "tensorrt-int8"
TENSORRT_ENGINE_FILENAME = "model.engine"
TENSORRT_BENCHMARK_REPORT = "tensorrt-benchmark.report.json"
TENSORRT_BUILD_METADATA_SUFFIX = ".build.json"
SYNTHETIC_NOTE = "Jane Doe visited Boston Clinic on 2024-01-15."
DEFAULT_LOGIT_TOLERANCE = 1e-3
DEFAULT_WORKSPACE_SIZE_BYTES = 1 << 30
SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16", "int8"})
MAX_BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 16_384
MAX_PROFILE_TOKENS = 1_048_576
MAX_WORKSPACE_SIZE_BYTES = 64 << 30
MAX_ONNX_FILE_BYTES = 16 << 30
MAX_IN_MEMORY_ONNX_BYTES = 512 << 20
MAX_SERIALIZED_ENGINE_BYTES = 16 << 30
MAX_CALIBRATION_SAMPLES = 4_096
MAX_CALIBRATION_TEXT_BYTES = 64 << 10
MAX_CALIBRATION_TOTAL_BYTES = 16 << 20
MAX_CALIBRATION_INPUT_BYTES = 256 << 20
MAX_CALIBRATION_CACHE_BYTES = 64 << 20
MAX_VERIFICATION_TEXT_BYTES = 64 << 10
MAX_VERIFICATION_LOGIT_ELEMENTS = 8 * 1_024 * 1_024
MAX_VERIFICATION_SPANS = 65_536
MAX_RECALL_LABELS = 4_096
MAX_RECALL_EVIDENCE_BYTES = 256 << 10
MAX_BUILD_METADATA_BYTES = 1 << 20
MAX_BENCHMARK_RECORDS = 128
MAX_BENCHMARK_ITERATIONS = 10_000
MAX_BENCHMARK_REPORT_BYTES = 16 << 20
MAX_REPORT_METADATA_BYTES = 64 << 10
MODEL_OPT_TIMEOUT_SECONDS = 30 * 60
MAX_MODEL_OPT_ERROR_BYTES = 16 << 10


class TensorRTBuildError(RuntimeError):
    """Raised when TensorRT cannot parse or build an engine."""


class TensorRTVerificationError(ValueError):
    """Raised when TensorRT output fails the ONNX reference check."""


class TensorRTReproducibilityError(ValueError):
    """Raised when pinned build-input or engine hashes do not match."""


class TensorRTQuantizationRejected(ValueError):
    """Raised when an INT8 build lacks passing G4 recall-delta evidence."""

    def __init__(self, message: str, gate: QuantRecallDeltaResult) -> None:
        super().__init__(message)
        self.gate = gate


@dataclass(frozen=True)
class TensorRTShapeProfile:
    """Minimum, optimum, and maximum token-classification input shapes."""

    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1
    min_sequence_length: int = 8
    opt_sequence_length: int = 128
    max_sequence_length: int = 512

    def __post_init__(self) -> None:
        _validate_shape_range(
            "batch size",
            self.min_batch_size,
            self.opt_batch_size,
            self.max_batch_size,
            limit=MAX_BATCH_SIZE,
        )
        _validate_shape_range(
            "sequence length",
            self.min_sequence_length,
            self.opt_sequence_length,
            self.max_sequence_length,
            limit=MAX_SEQUENCE_LENGTH,
        )
        if self.max_batch_size * self.max_sequence_length > MAX_PROFILE_TOKENS:
            raise ValueError(
                "TensorRT maximum profile exceeds the token budget: "
                f"{self.max_batch_size} * {self.max_sequence_length} > "
                f"{MAX_PROFILE_TOKENS}"
            )

    @property
    def minimum(self) -> tuple[int, int]:
        """Return the minimum batch and sequence shape."""

        return self.min_batch_size, self.min_sequence_length

    @property
    def optimum(self) -> tuple[int, int]:
        """Return the optimum batch and sequence shape."""

        return self.opt_batch_size, self.opt_sequence_length

    @property
    def maximum(self) -> tuple[int, int]:
        """Return the maximum batch and sequence shape."""

        return self.max_batch_size, self.max_sequence_length

    def to_dict(self) -> dict[str, list[int]]:
        """Return JSON-serializable optimization-profile metadata."""

        return {
            "min": list(self.minimum),
            "opt": list(self.optimum),
            "max": list(self.maximum),
        }


@dataclass(frozen=True)
class TensorRTExportVerification:
    """Synthetic ONNX/TensorRT parity evidence."""

    sample_text_sha256: str
    sample_text_length: int
    tolerance: float
    max_abs_logit_delta: float
    reference_token_spans: tuple[dict[str, Any], ...]
    tensorrt_token_spans: tuple[dict[str, Any], ...]
    passed: bool = True

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable verification metadata."""

        return {
            "sample_text_sha256": self.sample_text_sha256,
            "sample_text_length": self.sample_text_length,
            "tolerance": self.tolerance,
            "max_abs_logit_delta": self.max_abs_logit_delta,
            "reference_token_spans": [
                dict(span) for span in self.reference_token_spans
            ],
            "tensorrt_token_spans": [dict(span) for span in self.tensorrt_token_spans],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TensorRTBuildResult:
    """TensorRT engine paths, hashes, and certification evidence."""

    engine_path: Path
    metadata_path: Path
    source_onnx_path: Path
    build_onnx_path: Path
    family: str
    precision: str
    shape_profile: TensorRTShapeProfile
    tensorrt_version: str
    source_onnx_sha256: str
    build_onnx_sha256: str
    build_input_sha256: str
    engine_sha256: str
    calibration_sha256: str | None = None
    calibration_input_sha256: str | None = None
    calibration_cache_input_sha256: str | None = None
    recall_delta_gate: QuantRecallDeltaResult | None = None
    verification: TensorRTExportVerification | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable engine build metadata."""

        payload: dict[str, Any] = {
            "profile": TENSORRT_PROFILE_NAME,
            "format": _format_for_precision(self.precision),
            "family": self.family,
            "precision": self.precision,
            "shape_profile": self.shape_profile.to_dict(),
            "tensorrt_version": self.tensorrt_version,
            "source_onnx_sha256": self.source_onnx_sha256,
            "build_onnx_sha256": self.build_onnx_sha256,
            "build_input_sha256": self.build_input_sha256,
            "engine_sha256": self.engine_sha256,
        }
        if self.calibration_sha256 is not None:
            payload["calibration_sha256"] = self.calibration_sha256
        if self.calibration_input_sha256 is not None:
            payload["calibration_input_sha256"] = self.calibration_input_sha256
        if self.calibration_cache_input_sha256 is not None:
            payload["calibration_cache_input_sha256"] = (
                self.calibration_cache_input_sha256
            )
        if self.recall_delta_gate is not None:
            payload["gate"] = "G4"
            payload["recall_delta_gate"] = self.recall_delta_gate.to_dict()
        if self.verification is not None:
            payload["synthetic_verification"] = self.verification.to_metadata()
        return payload


@dataclass(frozen=True)
class TensorRTBenchmarkRecord:
    """One TensorRT device-tier latency and throughput record."""

    device_tier: str
    device: str
    precision: str
    latency_ms: float
    throughput_items_per_second: float
    sample_count: int
    batch_size: int = 1
    sequence_length: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        device_tier = _validate_bounded_name("device_tier", self.device_tier)
        device = _validate_bounded_name("device", self.device)
        normalized_precision = str(self.precision).strip().lower()
        if normalized_precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"unsupported TensorRT precision {self.precision!r}; expected one of "
                + ", ".join(sorted(SUPPORTED_PRECISIONS))
            )
        latency_ms = _validate_positive_finite("latency_ms", self.latency_ms)
        throughput = _validate_positive_finite(
            "throughput_items_per_second",
            self.throughput_items_per_second,
        )
        _validate_positive_int("sample_count", self.sample_count, limit=1_000_000)
        _validate_positive_int("batch_size", self.batch_size, limit=MAX_BATCH_SIZE)
        if self.sequence_length is not None:
            _validate_positive_int(
                "sequence_length",
                self.sequence_length,
                limit=MAX_SEQUENCE_LENGTH,
            )
        metadata = dict(self.metadata)
        _validate_json_size("benchmark metadata", metadata, MAX_REPORT_METADATA_BYTES)
        object.__setattr__(self, "device_tier", device_tier)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", normalized_precision)
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "throughput_items_per_second", throughput)
        object.__setattr__(self, "metadata", metadata)

    def to_metrics(self) -> dict[str, Any]:
        """Return the standard benchmark metrics block for this device."""

        payload: dict[str, Any] = {
            "device_tier": self.device_tier,
            "device": self.device,
            "precision": self.precision,
            "latency": {
                "p50_ms": self.latency_ms,
                "p95_ms": self.latency_ms,
                "count": self.sample_count,
            },
            "throughput": {
                "items_per_second": self.throughput_items_per_second,
            },
            "batch_size": self.batch_size,
        }
        if self.sequence_length is not None:
            payload["sequence_length"] = self.sequence_length
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class _CalibrationSpec:
    samples: tuple[Mapping[str, Any], ...]
    cache_path: Path | None
    cache_bytes: bytes | None


def build_tensorrt_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    family: str,
    precision: str = "fp16",
    shape_profile: TensorRTShapeProfile | None = None,
    workspace_size_bytes: int = DEFAULT_WORKSPACE_SIZE_BYTES,
    calibration_tokenizer: Any | None = None,
    calibration_texts: Iterable[str] | None = None,
    calibration_cache_path: str | Path | None = None,
    candidate_recall: Mapping[str, Any] | None = None,
    parent_recall: Mapping[str, Any] | None = None,
    precomputed_delta: Any = None,
    labels: Sequence[str] | None = None,
    sample_inputs: Mapping[str, Any] | None = None,
    reference_logits: Any | None = None,
    id2label: Mapping[str | int, str] | None = None,
    sample_text: str = SYNTHETIC_NOTE,
    tolerance: float = DEFAULT_LOGIT_TOLERANCE,
    expected_build_input_sha256: str | None = None,
    expected_engine_sha256: str | None = None,
    trt_module: Any | None = None,
    session_factory: Any = TensorRTTokenClassificationSession,
) -> TensorRTBuildResult:
    """Build and optionally certify a TensorRT token-classification engine.

    INT8 builds consume the shared deterministic synthetic calibration loader and
    are rejected before engine construction unless per-family G4 recall evidence
    passes. TensorRT 10 and earlier use entropy calibration. TensorRT 11 and later
    use Model Optimizer to add explicit ONNX Q/DQ nodes before engine construction.

    Args:
        onnx_path: Source ONNX graph produced by the OpenMed exporter.
        output_path: Destination for the device-specific serialized engine.
        family: Token-classification model family recorded in build metadata.
        precision: One of ``fp32``, ``fp16``, or ``int8``.
        shape_profile: Variable batch/sequence optimization range.
        workspace_size_bytes: TensorRT builder workspace limit.
        calibration_tokenizer: Tokenizer for shared INT8 calibration texts.
        calibration_texts: Optional synthetic calibration override.
        calibration_cache_path: Optional TensorRT 10 calibration cache path.
        candidate_recall: Per-label INT8 recall evidence.
        parent_recall: Full-precision per-label recall evidence.
        precomputed_delta: Precomputed per-label or overall recall delta.
        labels: Optional label subset for the G4 gate.
        sample_inputs: Synthetic tokenized note inputs used for parity checking.
        reference_logits: ONNX Runtime logits for the synthetic note.
        id2label: Token label mapping used for span comparison.
        sample_text: Synthetic note whose digest and UTF-8 length are recorded.
        tolerance: Maximum absolute logit delta for parity.
        expected_build_input_sha256: Optional pinned build-input hash.
        expected_engine_sha256: Optional pinned serialized engine hash.
        trt_module: Optional TensorRT module injection for testing.
        session_factory: Optional inference-session injection for testing.

    Returns:
        Engine paths, reproducibility hashes, and optional certification evidence.

    Raises:
        TensorRTQuantizationRejected: If INT8 calibration or G4 evidence is absent.
        TensorRTReproducibilityError: If a pinned hash does not match.
        TensorRTVerificationError: If synthetic ONNX/TensorRT parity fails.
        TensorRTBuildError: If TensorRT parsing or engine construction fails.
    """

    source_onnx = _validate_input_file(
        "ONNX model",
        Path(onnx_path),
        maximum_bytes=MAX_ONNX_FILE_BYTES,
    )
    engine_path = Path(output_path)
    metadata_path = engine_path.with_suffix(
        engine_path.suffix + TENSORRT_BUILD_METADATA_SUFFIX
    )
    family_name = _validate_bounded_name("family", family)
    _validate_output_path("TensorRT engine", engine_path)
    _validate_output_path("TensorRT build metadata", metadata_path)
    _require_distinct_paths(
        {
            "source ONNX": source_onnx,
            "TensorRT engine": engine_path,
            "TensorRT build metadata": metadata_path,
        }
    )
    normalized_precision = str(precision).strip().lower()
    if normalized_precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"unsupported TensorRT precision {precision!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_PRECISIONS))
        )
    _validate_positive_int(
        "workspace_size_bytes",
        workspace_size_bytes,
        limit=MAX_WORKSPACE_SIZE_BYTES,
    )
    tolerance = _validate_nonnegative_finite("tolerance", tolerance)
    verification_requested = _verification_requested(
        sample_inputs,
        reference_logits,
        id2label,
    )
    if verification_requested:
        if sample_inputs is None or reference_logits is None or id2label is None:
            raise ValueError(
                "sample_inputs, reference_logits, and id2label are all required "
                "for TensorRT synthetic verification"
            )
        _validate_verification_inputs(sample_inputs, id2label, sample_text)
    if normalized_precision != "int8" and calibration_cache_path is not None:
        raise ValueError("calibration_cache_path is only valid for INT8 builds")

    if shape_profile is not None and not isinstance(
        shape_profile,
        TensorRTShapeProfile,
    ):
        raise ValueError("shape_profile must be a TensorRTShapeProfile")
    profile = shape_profile or TensorRTShapeProfile()
    calibration_spec = None
    calibration_sha256 = None
    calibration_input_sha256 = None
    calibration_cache_input_sha256 = None
    recall_gate = None
    build_onnx = source_onnx
    calibration_text_values: list[str] | None = None
    calibration_samples: tuple[Mapping[str, Any], ...] | None = None

    if normalized_precision == "int8":
        _validate_recall_evidence(
            candidate_recall=candidate_recall,
            parent_recall=parent_recall,
            precomputed_delta=precomputed_delta,
            labels=labels,
        )
        recall_gate = evaluate_quant_recall_delta(
            format_name=TENSORRT_INT8_FORMAT,
            candidate_recall=candidate_recall or {},
            parent_recall=parent_recall,
            precomputed_delta=precomputed_delta,
            labels=labels,
        )
        if not recall_gate.passed:
            raise TensorRTQuantizationRejected(
                "TensorRT INT8 engine rejected by G4 recall-delta gate",
                recall_gate,
            )
        if not callable(calibration_tokenizer):
            raise TensorRTQuantizationRejected(
                "TensorRT INT8 export requires a callable tokenizer for shared "
                "calibration",
                recall_gate,
            )
        calibration_text_values = _normalize_calibration_texts(calibration_texts)
        calibration_sha256 = calibration_texts_sha256(calibration_text_values)
        calibration_samples, calibration_input_sha256 = _prepare_calibration_samples(
            calibration_tokenizer,
            calibration_text_values,
            profile,
        )

    trt = trt_module if trt_module is not None else _tensorrt_api()
    tensorrt_version = _validate_bounded_name(
        "TensorRT version",
        str(getattr(trt, "__version__", "unknown")),
    )

    if normalized_precision == "int8":
        assert calibration_text_values is not None
        assert calibration_samples is not None
        if _supports_legacy_int8_calibration(trt):
            cache_path = (
                Path(calibration_cache_path)
                if calibration_cache_path is not None
                else None
            )
            if cache_path is not None:
                _require_distinct_paths(
                    {
                        "source ONNX": source_onnx,
                        "TensorRT engine": engine_path,
                        "TensorRT build metadata": metadata_path,
                        "TensorRT calibration cache": cache_path,
                    }
                )
            cache_bytes = _read_calibration_cache(cache_path)
            if cache_bytes is not None:
                calibration_cache_input_sha256 = hashlib.sha256(cache_bytes).hexdigest()
            calibration_spec = _CalibrationSpec(
                samples=calibration_samples,
                cache_path=cache_path,
                cache_bytes=cache_bytes,
            )
        else:
            if calibration_cache_path is not None:
                raise ValueError(
                    "calibration_cache_path is unsupported for explicit TensorRT "
                    "quantization"
                )
            build_onnx = engine_path.with_suffix(".int8.onnx")
            _validate_output_path("TensorRT INT8 ONNX", build_onnx)
            _require_distinct_paths(
                {
                    "source ONNX": source_onnx,
                    "TensorRT engine": engine_path,
                    "TensorRT build metadata": metadata_path,
                    "TensorRT INT8 ONNX": build_onnx,
                }
            )
            _quantize_onnx_with_modelopt(
                source_onnx,
                build_onnx,
                samples=calibration_samples,
            )

    elif normalized_precision == "fp16" and not _has_builder_flag(trt, "FP16"):
        build_onnx = engine_path.with_suffix(".fp16.onnx")
        _validate_output_path("TensorRT FP16 ONNX", build_onnx)
        _require_distinct_paths(
            {
                "source ONNX": source_onnx,
                "TensorRT engine": engine_path,
                "TensorRT build metadata": metadata_path,
                "TensorRT FP16 ONNX": build_onnx,
            }
        )
        _autocast_onnx_with_modelopt(source_onnx, build_onnx)

    _validate_input_file(
        "effective ONNX model",
        build_onnx,
        maximum_bytes=MAX_ONNX_FILE_BYTES,
    )
    source_onnx_sha256 = sha256_file(source_onnx)
    build_onnx_sha256 = sha256_file(build_onnx)
    build_spec = {
        "schema_version": 2,
        "family": family_name,
        "precision": normalized_precision,
        "shape_profile": profile.to_dict(),
        "workspace_size_bytes": workspace_size_bytes,
        "tensorrt_version": tensorrt_version,
        "source_onnx_sha256": source_onnx_sha256,
        "build_onnx_sha256": build_onnx_sha256,
        "calibration_sha256": calibration_sha256,
        "calibration_input_sha256": calibration_input_sha256,
        "calibration_cache_input_sha256": calibration_cache_input_sha256,
        "recall_delta_gate": recall_gate.to_dict() if recall_gate else None,
    }
    build_input_sha256 = sha256_json(build_spec)
    _check_expected_hash(
        "build input",
        build_input_sha256,
        expected_build_input_sha256,
    )

    engine_bytes = _serialize_tensorrt_engine(
        build_onnx,
        precision=normalized_precision,
        shape_profile=profile,
        workspace_size_bytes=workspace_size_bytes,
        calibration_spec=calibration_spec,
        trt=trt,
    )
    if not isinstance(engine_bytes, (bytes, bytearray, memoryview)):
        raise TensorRTBuildError("TensorRT returned a non-bytes serialized engine")
    engine_bytes = bytes(engine_bytes)
    if not engine_bytes:
        raise TensorRTBuildError("TensorRT returned an empty serialized engine")
    if len(engine_bytes) > MAX_SERIALIZED_ENGINE_BYTES:
        raise TensorRTBuildError(
            "TensorRT serialized engine exceeds the supported size limit: "
            f"{len(engine_bytes)} > {MAX_SERIALIZED_ENGINE_BYTES} bytes"
        )
    engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
    _check_expected_hash("engine", engine_sha256, expected_engine_sha256)
    if sha256_file(source_onnx) != source_onnx_sha256:
        raise TensorRTBuildError("source ONNX graph changed during the TensorRT build")
    if build_onnx != source_onnx and sha256_file(build_onnx) != build_onnx_sha256:
        raise TensorRTBuildError(
            "effective ONNX graph changed during the TensorRT build"
        )

    verification = None
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{engine_path.name}.",
        suffix=".staging",
        dir=engine_path.parent,
        delete=False,
    ) as staging_file:
        staging_file.write(engine_bytes)
        staging_path = Path(staging_file.name)

    metadata_staging_path: Path | None = None
    try:
        if verification_requested:
            assert sample_inputs is not None
            assert reference_logits is not None
            assert id2label is not None
            session = session_factory(staging_path)
            candidate_logits = session.run(**dict(sample_inputs))
            verification = certify_tensorrt_reference(
                reference_logits=reference_logits,
                tensorrt_logits=candidate_logits,
                id2label=id2label,
                attention_mask=sample_inputs.get("attention_mask"),
                sample_text=sample_text,
                tolerance=tolerance,
            )
        if sha256_file(staging_path) != engine_sha256:
            raise TensorRTBuildError(
                "serialized TensorRT engine changed during staged verification"
            )
        result = TensorRTBuildResult(
            engine_path=engine_path,
            metadata_path=metadata_path,
            source_onnx_path=source_onnx,
            build_onnx_path=build_onnx,
            family=family_name,
            precision=normalized_precision,
            shape_profile=profile,
            tensorrt_version=tensorrt_version,
            source_onnx_sha256=source_onnx_sha256,
            build_onnx_sha256=build_onnx_sha256,
            build_input_sha256=build_input_sha256,
            engine_sha256=engine_sha256,
            calibration_sha256=calibration_sha256,
            calibration_input_sha256=calibration_input_sha256,
            calibration_cache_input_sha256=calibration_cache_input_sha256,
            recall_delta_gate=recall_gate,
            verification=verification,
        )
        metadata = result.to_metadata()
        _validate_json_size(
            "TensorRT build metadata",
            metadata,
            MAX_BUILD_METADATA_BYTES,
        )
        metadata_staging_path = _stage_json(metadata_path, metadata)
        _publish_artifact_pair(
            staging_path,
            engine_path,
            metadata_staging_path,
            metadata_path,
        )
    finally:
        staging_path.unlink(missing_ok=True)
        if metadata_staging_path is not None:
            metadata_staging_path.unlink(missing_ok=True)
    return result


def certify_tensorrt_reference(
    *,
    reference_logits: Any,
    tensorrt_logits: Any,
    id2label: Mapping[str | int, str],
    attention_mask: Any | None = None,
    sample_text: str = SYNTHETIC_NOTE,
    tolerance: float = DEFAULT_LOGIT_TOLERANCE,
) -> TensorRTExportVerification:
    """Check TensorRT logits and decoded spans against an ONNX reference."""

    import numpy as np

    tolerance = _validate_nonnegative_finite("tolerance", tolerance)
    sample_text_sha256, sample_text_length = _verification_text_evidence(sample_text)
    reference = np.asarray(reference_logits)
    candidate = np.asarray(tensorrt_logits)
    if reference.shape != candidate.shape:
        raise TensorRTVerificationError(
            "TensorRT logits shape does not match ONNX reference: "
            f"{candidate.shape} != {reference.shape}"
        )
    if (
        reference.ndim != 3
        or not reference.size
        or reference.shape[0] != 1
        or reference.shape[1] > MAX_SEQUENCE_LENGTH
    ):
        raise TensorRTVerificationError(
            "TensorRT verification logits must be a non-empty rank-three, "
            "single-sample array within the sequence-length limit"
        )
    _validate_id2label(id2label, expected_count=int(reference.shape[2]))
    if reference.size > MAX_VERIFICATION_LOGIT_ELEMENTS:
        raise TensorRTVerificationError(
            "TensorRT verification logits exceed the element limit: "
            f"{reference.size} > {MAX_VERIFICATION_LOGIT_ELEMENTS}"
        )
    if reference.dtype.kind not in "iuf" or candidate.dtype.kind not in "iuf":
        raise TensorRTVerificationError(
            "TensorRT verification logits must be real numeric arrays"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise TensorRTVerificationError(
            "TensorRT verification logits must contain only finite values"
        )
    if attention_mask is not None:
        mask = np.asarray(attention_mask)
        if mask.shape != reference.shape[:2]:
            raise TensorRTVerificationError(
                "TensorRT attention mask shape does not match logits: "
                f"{mask.shape} != {reference.shape[:2]}"
            )
        if not np.issubdtype(mask.dtype, np.number) or not np.all(np.isfinite(mask)):
            raise TensorRTVerificationError(
                "TensorRT attention mask must contain only finite numeric values"
            )
        if not np.all((mask == 0) | (mask == 1)):
            raise TensorRTVerificationError(
                "TensorRT attention mask must contain only zero or one"
            )
    max_abs_delta = float(
        np.max(
            np.abs(
                reference.astype(np.float64, copy=False)
                - candidate.astype(np.float64, copy=False)
            )
        )
    )
    reference_spans = token_spans_from_logits(
        reference,
        id2label,
        attention_mask=attention_mask,
    )
    candidate_spans = token_spans_from_logits(
        candidate,
        id2label,
        attention_mask=attention_mask,
    )
    if (
        len(reference_spans) > MAX_VERIFICATION_SPANS
        or len(candidate_spans) > MAX_VERIFICATION_SPANS
    ):
        raise TensorRTVerificationError(
            "TensorRT verification produced too many decoded token spans"
        )
    if max_abs_delta > tolerance:
        raise TensorRTVerificationError(
            "TensorRT logits exceeded tolerance "
            f"{tolerance}: max abs delta {max_abs_delta}"
        )
    if candidate_spans != reference_spans:
        raise TensorRTVerificationError(
            "TensorRT decoded token spans do not match ONNX reference"
        )
    return TensorRTExportVerification(
        sample_text_sha256=sample_text_sha256,
        sample_text_length=sample_text_length,
        tolerance=tolerance,
        max_abs_logit_delta=max_abs_delta,
        reference_token_spans=reference_spans,
        tensorrt_token_spans=candidate_spans,
    )


def verify_tensorrt_engine_hash(
    engine_path: str | Path,
    expected_sha256: str,
) -> str:
    """Verify a serialized engine against a pinned SHA-256 digest.

    Args:
        engine_path: Serialized engine to hash.
        expected_sha256: Required lowercase or uppercase SHA-256 digest.

    Returns:
        The normalized matching digest.

    Raises:
        TensorRTReproducibilityError: If the digest differs.
    """

    actual = sha256_file(engine_path)
    _check_expected_hash("engine", actual, expected_sha256)
    return actual


def measure_tensorrt_latency(
    session: TensorRTTokenClassificationSession,
    sample_inputs: Mapping[str, Any],
    *,
    device_tier: str,
    device: str,
    precision: str,
    iterations: int = 5,
) -> TensorRTBenchmarkRecord:
    """Measure synchronized latency and throughput for one TensorRT device."""

    _validate_positive_int(
        "iterations",
        iterations,
        limit=MAX_BENCHMARK_ITERATIONS,
    )
    latencies = []
    for _ in range(iterations):
        started = perf_counter()
        session.run(**dict(sample_inputs))
        latencies.append((perf_counter() - started) * 1000.0)
    latency_ms = median(latencies)
    batch_size = _batch_size(sample_inputs)
    return TensorRTBenchmarkRecord(
        device_tier=device_tier,
        device=device,
        precision=precision,
        latency_ms=latency_ms,
        throughput_items_per_second=(
            batch_size * 1000.0 / latency_ms if latency_ms else 0.0
        ),
        sample_count=iterations,
        batch_size=batch_size,
        sequence_length=_sequence_length(sample_inputs),
        metadata={"profile": TENSORRT_PROFILE_NAME},
    )


def build_tensorrt_benchmark_report(
    *,
    model_name: str,
    records: Sequence[TensorRTBenchmarkRecord],
    suite: str = "tensorrt-runtime",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkReport:
    """Build a benchmark report with device-tier latency and throughput."""

    if not records:
        raise ValueError("at least one TensorRT benchmark record is required")
    if len(records) > MAX_BENCHMARK_RECORDS:
        raise ValueError(
            "too many TensorRT benchmark records: "
            f"{len(records)} > {MAX_BENCHMARK_RECORDS}"
        )
    model_name = _validate_bounded_name("model_name", model_name)
    suite = _validate_bounded_name("suite", suite)
    report_metadata = dict(metadata or {})
    _validate_json_size(
        "benchmark report metadata",
        report_metadata,
        MAX_REPORT_METADATA_BYTES,
    )
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    generated = _validate_bounded_name("generated_at", generated)
    device_names = [record.device for record in records]
    metrics: dict[str, Any] = {}
    for record in records:
        key = f"{record.device_tier}:{record.device}"
        if key in metrics:
            raise ValueError(f"duplicate TensorRT benchmark device key: {key}")
        metrics[key] = record.to_metrics()
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device="tensorrt:" + ",".join(device_names),
        fixture_count=sum(record.sample_count for record in records),
        generated_at=generated,
        metadata={"profile": TENSORRT_PROFILE_NAME, **report_metadata},
        metrics={"devices": metrics},
    )


def write_tensorrt_benchmark_report(
    output_path: str | Path,
    *,
    model_name: str,
    records: Sequence[TensorRTBenchmarkRecord],
    suite: str = "tensorrt-runtime",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a TensorRT device-tier benchmark report as JSON."""

    report = build_tensorrt_benchmark_report(
        model_name=model_name,
        records=records,
        suite=suite,
        generated_at=generated_at,
        metadata=metadata,
    )
    report_path = Path(output_path)
    _validate_output_path("TensorRT benchmark report", report_path)
    report_payload = report.to_dict()
    _validate_json_size(
        "TensorRT benchmark report",
        report_payload,
        MAX_BENCHMARK_REPORT_BYTES,
    )
    staging_path = _stage_json(report_path, report_payload)
    try:
        staging_path.replace(report_path)
    finally:
        staging_path.unlink(missing_ok=True)
    return report_path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON mapping."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_tensorrt_engine(
    onnx_path: Path,
    *,
    precision: str,
    shape_profile: TensorRTShapeProfile,
    workspace_size_bytes: int,
    calibration_spec: _CalibrationSpec | None,
    trt: Any,
) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    init_plugins = getattr(trt, "init_libnvinfer_plugins", None)
    if init_plugins is not None and init_plugins(logger, "") is False:
        raise TensorRTBuildError("TensorRT plugin initialization failed")
    builder = trt.Builder(logger)
    if builder is None:
        raise TensorRTBuildError("TensorRT could not create a builder")
    network = builder.create_network(_network_creation_flags(trt))
    if network is None:
        raise TensorRTBuildError("TensorRT could not create a network")
    parser = trt.OnnxParser(network, logger)
    if parser is None:
        raise TensorRTBuildError("TensorRT could not create an ONNX parser")
    parsed = _parse_onnx(parser, onnx_path)
    if not parsed:
        error_count = min(max(int(getattr(parser, "num_errors", 0)), 0), 32)
        errors = [str(parser.get_error(index))[:1_024] for index in range(error_count)]
        detail = "; ".join(errors) or "unknown parser error"
        raise TensorRTBuildError(f"TensorRT could not parse ONNX graph: {detail}")

    config = builder.create_builder_config()
    if config is None:
        raise TensorRTBuildError("TensorRT could not create a builder configuration")
    _set_workspace_limit(config, trt, workspace_size_bytes)
    optimization_profile = _add_optimization_profile(
        builder,
        network,
        config,
        shape_profile,
    )

    if precision == "fp16" and _has_builder_flag(trt, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and calibration_spec is not None:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _create_entropy_calibrator(
            trt,
            network,
            calibration_spec,
        )
        set_calibration_profile = getattr(config, "set_calibration_profile", None)
        if set_calibration_profile is not None:
            set_calibration_profile(optimization_profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise TensorRTBuildError("TensorRT failed to build a serialized engine")
    return _coerce_bounded_bytes(
        "TensorRT serialized engine",
        serialized,
        limit=MAX_SERIALIZED_ENGINE_BYTES,
    )


def _add_optimization_profile(
    builder: Any,
    network: Any,
    config: Any,
    profile: TensorRTShapeProfile,
) -> Any:
    optimization_profile = builder.create_optimization_profile()
    if optimization_profile is None:
        raise TensorRTBuildError("TensorRT could not create an optimization profile")
    if network.num_inputs < 1:
        raise TensorRTBuildError("TensorRT ONNX graph has no inputs")
    if network.num_inputs > 16:
        raise TensorRTBuildError("TensorRT ONNX graph has too many inputs")
    seen_names: set[str] = set()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor is None:
            raise TensorRTBuildError(
                f"TensorRT could not resolve network input at index {index}"
            )
        tensor_name = _validate_bounded_name(
            "TensorRT input name",
            str(tensor.name),
        )
        if tensor_name in seen_names:
            raise TensorRTBuildError(f"duplicate TensorRT input name: {tensor_name}")
        seen_names.add(tensor_name)
        input_shape = tuple(int(dim) for dim in tensor.shape)
        if getattr(tensor, "is_shape_tensor", False):
            raise TensorRTBuildError(
                f"TensorRT shape-tensor input {tensor.name!r} is not supported"
            )
        if len(input_shape) != 2:
            raise TensorRTBuildError(
                "TensorRT token-classification inputs must have rank two; "
                f"{tensor_name!r} has shape {input_shape}"
            )
        for dimension, limit in zip(
            input_shape,
            (MAX_BATCH_SIZE, MAX_SEQUENCE_LENGTH),
        ):
            if dimension != -1 and (dimension < 1 or dimension > limit):
                raise TensorRTBuildError(
                    f"TensorRT input {tensor_name!r} has unsupported shape "
                    f"{input_shape}"
                )
        minimum = _merge_profile_shape(input_shape, profile.minimum)
        optimum = _merge_profile_shape(input_shape, profile.optimum)
        maximum = _merge_profile_shape(input_shape, profile.maximum)
        if maximum[0] * maximum[1] > MAX_PROFILE_TOKENS:
            raise TensorRTBuildError(
                f"TensorRT input {tensor_name!r} exceeds the profile token budget"
            )
        if (
            optimization_profile.set_shape(
                tensor_name,
                minimum,
                optimum,
                maximum,
            )
            is False
        ):
            raise TensorRTBuildError(
                f"TensorRT rejected optimization profile for input {tensor.name!r}"
            )
    if config.add_optimization_profile(optimization_profile) < 0:
        raise TensorRTBuildError("TensorRT rejected the optimization profile")
    return optimization_profile


def _create_entropy_calibrator(
    trt: Any,
    network: Any,
    calibration_spec: _CalibrationSpec,
) -> Any:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError(
            "NumPy and CUDA-enabled PyTorch are required for TensorRT calibration"
        ) from exc
    if not torch.cuda.is_available():
        raise TensorRTBuildError("TensorRT INT8 calibration requires a CUDA device")

    input_dtypes = {
        network.get_input(index).name: np.dtype(
            trt.nptype(network.get_input(index).dtype)
        )
        for index in range(network.num_inputs)
    }

    class SharedCalibrationEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._samples = iter(calibration_spec.samples)
            self._device_inputs: dict[str, Any] = {}

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            try:
                encoded = next(self._samples)
            except StopIteration:
                return None
            if (
                not 1 <= len(names) <= 16
                or len(set(names)) != len(names)
                or any(name not in input_dtypes for name in names)
            ):
                raise TensorRTBuildError(
                    "TensorRT requested an invalid calibration input set"
                )
            host_inputs: dict[str, Any] = {}
            for name in names:
                if name not in encoded:
                    raise TensorRTBuildError(
                        f"calibration tokenizer did not produce required input {name!r}"
                    )
                host_inputs[name] = _coerce_calibration_array(
                    encoded[name],
                    input_dtypes[name],
                    name=name,
                )
            self._device_inputs = {}
            pointers = []
            for name in names:
                tensor = torch.as_tensor(
                    host_inputs[name],
                    device="cuda",
                ).contiguous()
                self._device_inputs[name] = tensor
                pointers.append(int(tensor.data_ptr()))
            return pointers

        def read_calibration_cache(self) -> bytes | None:
            return calibration_spec.cache_bytes

        def write_calibration_cache(self, cache: Any) -> None:
            cache_path = calibration_spec.cache_path
            if cache_path is None:
                return
            cache_bytes = _coerce_bounded_bytes(
                "TensorRT calibration cache",
                cache,
                limit=MAX_CALIBRATION_CACHE_BYTES,
            )
            _write_bytes_atomic(cache_path, cache_bytes)

    return SharedCalibrationEntropyCalibrator()


def _quantize_onnx_with_modelopt(
    source_path: Path,
    output_path: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
) -> Path:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openmed-tensorrt-calibration-") as tmp:
        calibration_dir = Path(tmp)
        for index, arrays in enumerate(samples):
            np.savez(calibration_dir / f"{index:05d}.npz", **dict(arrays))
        staging_path = _reserve_staging_path(output_path, suffix=".onnx")
        try:
            command = [
                sys.executable,
                "-m",
                "modelopt.onnx.quantization",
                "--onnx_path",
                str(source_path),
                "--quantize_mode",
                "int8",
                "--calibration_method",
                "entropy",
                "--calibration_data",
                str(calibration_dir),
                "--output_path",
                str(staging_path),
            ]
            _run_modelopt(command, purpose="INT8 calibration")
            _validate_input_file(
                "Model Optimizer INT8 ONNX",
                staging_path,
                maximum_bytes=MAX_ONNX_FILE_BYTES,
            )
            staging_path.replace(output_path)
        finally:
            staging_path.unlink(missing_ok=True)
    return output_path


def _autocast_onnx_with_modelopt(source_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = _reserve_staging_path(output_path, suffix=".onnx")
    try:
        command = [
            sys.executable,
            "-m",
            "modelopt.onnx.autocast",
            "--onnx_path",
            str(source_path),
            "--output_path",
            str(staging_path),
        ]
        _run_modelopt(command, purpose="FP16 autocast")
        _validate_input_file(
            "Model Optimizer FP16 ONNX",
            staging_path,
            maximum_bytes=MAX_ONNX_FILE_BYTES,
        )
        staging_path.replace(output_path)
    finally:
        staging_path.unlink(missing_ok=True)
    return output_path


def _run_modelopt(command: Sequence[str], *, purpose: str) -> None:
    with tempfile.TemporaryFile(mode="w+b") as output:
        try:
            subprocess.run(
                list(command),
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=MODEL_OPT_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise TensorRTBuildError(
                f"TensorRT {purpose} requires NVIDIA Model Optimizer"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TensorRTBuildError(
                f"TensorRT {purpose} timed out after {MODEL_OPT_TIMEOUT_SECONDS} seconds"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = _read_log_tail(output)
            raise TensorRTBuildError(
                f"TensorRT {purpose} failed in NVIDIA Model Optimizer: {detail}"
            ) from exc


def _parse_onnx(parser: Any, onnx_path: Path) -> bool:
    parse_from_file = getattr(parser, "parse_from_file", None)
    if parse_from_file is not None:
        return bool(parse_from_file(str(onnx_path)))
    if onnx_path.stat().st_size > MAX_IN_MEMORY_ONNX_BYTES:
        raise TensorRTBuildError(
            "this TensorRT parser requires in-memory ONNX loading, and the graph "
            f"exceeds {MAX_IN_MEMORY_ONNX_BYTES} bytes"
        )
    with onnx_path.open("rb") as handle:
        payload = handle.read(MAX_IN_MEMORY_ONNX_BYTES + 1)
    if len(payload) > MAX_IN_MEMORY_ONNX_BYTES:
        raise TensorRTBuildError("ONNX graph changed while it was being read")
    return bool(parser.parse(payload))


def _network_creation_flags(trt: Any) -> int:
    flags = getattr(trt, "NetworkDefinitionCreationFlag", None)
    explicit_batch = getattr(flags, "EXPLICIT_BATCH", None)
    if explicit_batch is None:
        return 0
    return 1 << int(explicit_batch)


def _set_workspace_limit(config: Any, trt: Any, workspace_size_bytes: int) -> None:
    set_memory_pool_limit = getattr(config, "set_memory_pool_limit", None)
    memory_pool_type = getattr(trt, "MemoryPoolType", None)
    if set_memory_pool_limit is not None and memory_pool_type is not None:
        set_memory_pool_limit(memory_pool_type.WORKSPACE, workspace_size_bytes)
    else:
        config.max_workspace_size = workspace_size_bytes


def _merge_profile_shape(
    network_shape: tuple[int, int],
    requested_shape: tuple[int, int],
) -> tuple[int, int]:
    merged = tuple(
        requested if network == -1 else network
        for network, requested in zip(network_shape, requested_shape)
    )
    return merged[0], merged[1]


def _normalize_calibration_texts(texts: Iterable[str] | None) -> list[str]:
    source = load_quantization_calibration_texts() if texts is None else texts
    if isinstance(source, (str, bytes)):
        raise _calibration_rejection(
            "TensorRT INT8 calibration must be an iterable of text samples"
        )
    raw_texts = list(itertools.islice(iter(source), MAX_CALIBRATION_SAMPLES + 1))
    if len(raw_texts) > MAX_CALIBRATION_SAMPLES:
        raise _calibration_rejection(
            "TensorRT INT8 calibration exceeds the sample-count limit"
        )

    normalized: list[str] = []
    total_bytes = 0
    for text in raw_texts:
        if not isinstance(text, str):
            raise _calibration_rejection(
                "TensorRT INT8 calibration samples must be strings"
            )
        value = text.strip()
        if not value:
            continue
        byte_count = len(value.encode("utf-8"))
        if byte_count > MAX_CALIBRATION_TEXT_BYTES:
            raise _calibration_rejection(
                "TensorRT INT8 calibration sample exceeds the UTF-8 byte limit"
            )
        total_bytes += byte_count
        if total_bytes > MAX_CALIBRATION_TOTAL_BYTES:
            raise _calibration_rejection(
                "TensorRT INT8 calibration set exceeds the aggregate byte limit"
            )
        normalized.append(value)
    if not normalized:
        raise _calibration_rejection(
            "TensorRT INT8 export requires calibration samples"
        )
    return normalized


def _calibration_rejection(message: str) -> TensorRTQuantizationRejected:
    gate = evaluate_quant_recall_delta(
        format_name=TENSORRT_INT8_FORMAT,
        candidate_recall={},
    )
    return TensorRTQuantizationRejected(message, gate)


def _prepare_calibration_samples(
    tokenizer: Any,
    texts: Sequence[str],
    profile: TensorRTShapeProfile,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required for TensorRT INT8 calibration") from exc
    if not callable(tokenizer):
        raise TensorRTBuildError("TensorRT calibration tokenizer must be callable")

    allowed_names = {"input_ids", "attention_mask", "token_type_ids"}
    required_names = {"input_ids", "attention_mask"}
    prepared: list[Mapping[str, Any]] = []
    expected_names: tuple[str, ...] | None = None
    digest = hashlib.sha256()
    digest.update(b"openmed-tensorrt-calibration-input-v1\0")
    total_input_bytes = 0
    for index, text in enumerate(texts):
        try:
            encoded = tokenizer(
                [text],
                max_length=profile.opt_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
        except Exception as exc:
            raise TensorRTBuildError(
                f"calibration tokenizer failed for sample {index}"
            ) from exc
        if not isinstance(encoded, Mapping):
            raise TensorRTBuildError(
                "calibration tokenizer must return a mapping of input arrays"
            )
        names = tuple(sorted(set(encoded).intersection(allowed_names)))
        if not required_names.issubset(names):
            raise TensorRTBuildError(
                "calibration tokenizer must produce input_ids and attention_mask"
            )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise TensorRTBuildError(
                "calibration tokenizer returned inconsistent input names"
            )

        arrays: dict[str, Any] = {}
        for name in names:
            array = np.asarray(encoded[name])
            if array.ndim != 2 or tuple(array.shape) != (
                1,
                profile.opt_sequence_length,
            ):
                raise TensorRTBuildError(
                    "calibration tokenizer returned an invalid shape for "
                    f"{name!r}: {tuple(array.shape)}"
                )
            if not np.issubdtype(array.dtype, np.integer):
                raise TensorRTBuildError(
                    f"calibration tokenizer returned a non-integer {name!r} array"
                )
            minimum = int(array.min())
            maximum = int(array.max())
            if name == "attention_mask" and (minimum < 0 or maximum > 1):
                raise TensorRTBuildError(
                    "calibration attention_mask must contain only zero or one"
                )
            if name in {"input_ids", "token_type_ids"} and minimum < 0:
                raise TensorRTBuildError(
                    f"calibration token input {name!r} must not contain negative values"
                )
            array = np.ascontiguousarray(array)
            total_input_bytes += int(array.nbytes)
            if total_input_bytes > MAX_CALIBRATION_INPUT_BYTES:
                raise TensorRTBuildError(
                    "tokenized calibration inputs exceed the aggregate byte limit"
                )
            arrays[name] = array
            header = json.dumps(
                {
                    "index": index,
                    "name": name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(header)
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
            digest.update(b"\0")
        prepared.append(arrays)
    return tuple(prepared), digest.hexdigest()


def _read_calibration_cache(path: Path | None) -> bytes | None:
    if path is None:
        return None
    _validate_output_path("TensorRT calibration cache", path)
    if not path.exists():
        return None
    _validate_input_file(
        "TensorRT calibration cache",
        path,
        maximum_bytes=MAX_CALIBRATION_CACHE_BYTES,
    )
    with path.open("rb") as handle:
        cache = handle.read(MAX_CALIBRATION_CACHE_BYTES + 1)
    if not cache or len(cache) > MAX_CALIBRATION_CACHE_BYTES:
        raise TensorRTBuildError(
            "TensorRT calibration cache is empty or exceeds the size limit"
        )
    return cache


def _coerce_calibration_array(array: Any, dtype: Any, *, name: str) -> Any:
    import numpy as np

    source = np.asarray(array)
    target = np.dtype(dtype)
    if source.dtype.kind not in "biu" or target.kind not in "biu":
        raise TensorRTBuildError(
            f"TensorRT calibration input {name!r} must use an integer dtype"
        )
    minimum = int(source.min())
    maximum = int(source.max())
    if target.kind == "b":
        representable = minimum >= 0 and maximum <= 1
    else:
        limits = np.iinfo(target)
        representable = minimum >= int(limits.min) and maximum <= int(limits.max)
    if not representable:
        raise TensorRTBuildError(
            f"TensorRT calibration input {name!r} cannot be represented by "
            "the engine dtype"
        )
    return np.ascontiguousarray(source, dtype=target)


def _supports_legacy_int8_calibration(trt: Any) -> bool:
    version = str(getattr(trt, "__version__", "")).strip()
    major_text = version.split(".", 1)[0]
    try:
        major_version = int(major_text)
    except ValueError:
        major_version = None
    if major_version is not None and major_version >= 11:
        return False
    return hasattr(trt, "IInt8EntropyCalibrator2") and _has_builder_flag(trt, "INT8")


def _has_builder_flag(trt: Any, name: str) -> bool:
    flags = getattr(trt, "BuilderFlag", None)
    return flags is not None and hasattr(flags, name)


def _format_for_precision(precision: str) -> str:
    if precision == "fp16":
        return TENSORRT_FP16_FORMAT
    if precision == "int8":
        return TENSORRT_INT8_FORMAT
    return TENSORRT_ENGINE_FORMAT


def _verification_requested(*values: Any) -> bool:
    return any(value is not None for value in values)


def _check_expected_hash(
    name: str,
    actual: str,
    expected: str | None,
) -> None:
    if expected is None:
        return
    normalized = str(expected).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"expected {name} hash must be a 64-character SHA-256")
    if actual != normalized:
        raise TensorRTReproducibilityError(
            f"TensorRT {name} hash mismatch: {actual} != {normalized}"
        )


def _validate_shape_range(
    name: str,
    minimum: int,
    optimum: int,
    maximum: int,
    *,
    limit: int,
) -> None:
    if any(type(value) is not int for value in (minimum, optimum, maximum)):
        raise ValueError(f"{name} values must be integers")
    if minimum < 1 or minimum > optimum or optimum > maximum:
        raise ValueError(
            f"invalid {name} range: expected 1 <= min <= opt <= max, got "
            f"{minimum} <= {optimum} <= {maximum}"
        )
    if maximum > limit:
        raise ValueError(f"{name} maximum exceeds the supported limit {limit}")


def _validate_positive_int(name: str, value: Any, *, limit: int) -> int:
    if type(value) is not int or value < 1 or value > limit:
        raise ValueError(f"{name} must be an integer between 1 and {limit}")
    return value


def _validate_nonnegative_finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _validate_positive_finite(name: str, value: Any) -> float:
    normalized = _validate_nonnegative_finite(name, value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _coerce_bounded_bytes(name: str, value: Any, *, limit: int) -> bytes:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise TensorRTBuildError(f"{name} is not a bytes-like payload") from exc
    if not 1 <= view.nbytes <= limit:
        raise TensorRTBuildError(f"{name} must contain between 1 and {limit} bytes")
    return bytes(view)


def _validate_bounded_name(name: str, value: Any, *, maximum_bytes: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized.encode("utf-8")) > maximum_bytes:
        raise ValueError(
            f"{name} must contain between 1 and {maximum_bytes} UTF-8 bytes"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{name} must not contain control characters")
    return normalized


def _validate_recall_evidence(
    *,
    candidate_recall: Mapping[str, Any] | None,
    parent_recall: Mapping[str, Any] | None,
    precomputed_delta: Any,
    labels: Sequence[str] | None,
) -> None:
    for name, evidence in (
        ("candidate_recall", candidate_recall),
        ("parent_recall", parent_recall),
    ):
        if evidence is None:
            continue
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if len(evidence) > MAX_RECALL_LABELS:
            raise ValueError(f"{name} exceeds the recall-label limit")
        for label, value in evidence.items():
            _validate_bounded_name(f"{name} label", label)
            if isinstance(value, bool):
                raise ValueError(f"{name} values must be finite recall numbers")
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} values must be finite recall numbers"
                ) from exc
            if not math.isfinite(score) or not 0.0 <= score <= 100.0:
                raise ValueError(f"{name} values must be between 0 and 100")
        _validate_json_size(name, evidence, MAX_RECALL_EVIDENCE_BYTES)

    if labels is not None:
        if isinstance(labels, (str, bytes)) or len(labels) > MAX_RECALL_LABELS:
            raise ValueError("labels must be a bounded sequence of label names")
        for label in labels:
            _validate_bounded_name("recall label", label)

    if precomputed_delta is not None:
        if isinstance(precomputed_delta, Mapping) and len(precomputed_delta) > (
            MAX_RECALL_LABELS
        ):
            raise ValueError("precomputed_delta exceeds the recall-label limit")
        _validate_json_size(
            "precomputed_delta",
            precomputed_delta,
            MAX_RECALL_EVIDENCE_BYTES,
        )


def _validate_json_size(name: str, value: Any, limit: int) -> None:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data") from exc
    if len(encoded) > limit:
        raise ValueError(f"{name} exceeds the JSON byte limit {limit}")


def _verification_text_evidence(sample_text: Any) -> tuple[str, int]:
    if not isinstance(sample_text, str) or not sample_text:
        raise TensorRTVerificationError(
            "TensorRT verification sample_text must be a non-empty string"
        )
    encoded = sample_text.encode("utf-8")
    if len(encoded) > MAX_VERIFICATION_TEXT_BYTES:
        raise TensorRTVerificationError(
            "TensorRT verification sample_text exceeds the UTF-8 byte limit"
        )
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _validate_id2label(
    id2label: Mapping[str | int, str],
    *,
    expected_count: int | None = None,
) -> None:
    if not isinstance(id2label, Mapping) or not id2label or len(id2label) > 4_096:
        raise TensorRTVerificationError(
            "TensorRT id2label must contain between 1 and 4096 entries"
        )
    normalized_keys: set[int] = set()
    for key, label in id2label.items():
        if isinstance(key, bool):
            raise TensorRTVerificationError(
                "TensorRT id2label keys must be non-negative integer IDs"
            )
        try:
            normalized_key = int(key)
        except (TypeError, ValueError) as exc:
            raise TensorRTVerificationError(
                "TensorRT id2label keys must be non-negative integer IDs"
            ) from exc
        if normalized_key < 0 or str(normalized_key) != str(key):
            raise TensorRTVerificationError(
                "TensorRT id2label keys must be canonical non-negative integer IDs"
            )
        if normalized_key in normalized_keys:
            raise TensorRTVerificationError(
                "TensorRT id2label contains duplicate normalized IDs"
            )
        normalized_keys.add(normalized_key)
        try:
            _validate_bounded_name("TensorRT label", label, maximum_bytes=256)
        except ValueError as exc:
            raise TensorRTVerificationError(str(exc)) from exc
    if expected_count is not None:
        if len(normalized_keys) != expected_count or normalized_keys != set(
            range(expected_count)
        ):
            raise TensorRTVerificationError(
                "TensorRT id2label must cover every logits label index exactly once"
            )


def _validate_verification_inputs(
    sample_inputs: Mapping[str, Any],
    id2label: Mapping[str | int, str],
    sample_text: str,
) -> None:
    if not isinstance(sample_inputs, Mapping):
        raise ValueError("sample_inputs must be a mapping")
    names = set(sample_inputs)
    allowed = {"input_ids", "attention_mask", "token_type_ids"}
    if not {"input_ids", "attention_mask"}.issubset(names) or not names.issubset(
        allowed
    ):
        raise ValueError(
            "sample_inputs must contain input_ids and attention_mask, with optional "
            "token_type_ids only"
        )
    _validate_id2label(id2label)
    _verification_text_evidence(sample_text)


def _validate_input_file(name: str, path: Path, *, maximum_bytes: int) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    size = path.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise ValueError(
            f"{name} must contain between 1 and {maximum_bytes} bytes: {path}"
        )
    return path


def _validate_output_path(name: str, path: Path) -> None:
    if path.exists() and not path.is_file():
        raise ValueError(f"{name} path is not a file: {path}")
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise ValueError(f"{name} parent is not a directory: {parent}")


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    identities: dict[Path, str] = {}
    for name, path in paths.items():
        identity = path.resolve(strict=False)
        if identity in identities:
            raise ValueError(
                f"{name} path collides with {identities[identity]}: {path}"
            )
        identities[identity] = name


def _batch_size(inputs: Mapping[str, Any]) -> int:
    value = inputs.get("input_ids")
    if value is None:
        return 1
    try:
        return int(value.shape[0])
    except AttributeError:
        try:
            return len(value)
        except TypeError:
            return 1


def _sequence_length(inputs: Mapping[str, Any]) -> int | None:
    value = inputs.get("input_ids")
    if value is None:
        return None
    try:
        return int(value.shape[1])
    except (AttributeError, IndexError):
        try:
            return len(value[0])
        except (TypeError, IndexError):
            return None


def _reserve_staging_path(path: Path, *, suffix: str = ".staging") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=suffix,
        dir=path.parent,
        delete=False,
    ) as staging_file:
        staging_path = Path(staging_file.name)
    staging_path.unlink()
    return staging_path


def _stage_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".staging",
            dir=path.parent,
            delete=False,
        ) as staging_file:
            staging_path = Path(staging_file.name)
            json.dump(
                payload,
                staging_file,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            staging_file.write("\n")
    except BaseException:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
        raise
    assert staging_path is not None
    return staging_path


def _publish_artifact_pair(
    engine_staging_path: Path,
    engine_path: Path,
    metadata_staging_path: Path,
    metadata_path: Path,
) -> None:
    targets = (engine_path, metadata_path)
    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        for target in targets:
            if target.exists() or target.is_symlink():
                backup = _reserve_staging_path(target, suffix=".backup")
                target.replace(backup)
                backups[target] = backup
        engine_staging_path.replace(engine_path)
        published.add(engine_path)
        metadata_staging_path.replace(metadata_path)
        published.add(metadata_path)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                backup.replace(target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    _validate_output_path("TensorRT calibration cache", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".staging",
            dir=path.parent,
            delete=False,
        ) as staging_file:
            staging_file.write(payload)
            staging_path = Path(staging_file.name)
        assert staging_path is not None
        staging_path.replace(path)
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


def _read_log_tail(output: Any) -> str:
    output.flush()
    size = output.tell()
    output.seek(max(0, size - MAX_MODEL_OPT_ERROR_BYTES))
    detail = output.read(MAX_MODEL_OPT_ERROR_BYTES).decode("utf-8", errors="replace")
    return detail.strip() or "unknown error"


def _tensorrt_api() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT is required for engine export. Install it for the target "
            "NVIDIA platform; OpenMed does not bundle TensorRT or CUDA."
        ) from exc
    return trt


__all__ = [
    "DEFAULT_LOGIT_TOLERANCE",
    "DEFAULT_WORKSPACE_SIZE_BYTES",
    "SUPPORTED_PRECISIONS",
    "SYNTHETIC_NOTE",
    "TENSORRT_BENCHMARK_REPORT",
    "TENSORRT_BUILD_METADATA_SUFFIX",
    "TENSORRT_ENGINE_FILENAME",
    "TENSORRT_ENGINE_FORMAT",
    "TENSORRT_FP16_FORMAT",
    "TENSORRT_INT8_FORMAT",
    "TENSORRT_PROFILE_NAME",
    "TensorRTBenchmarkRecord",
    "TensorRTBuildError",
    "TensorRTBuildResult",
    "TensorRTExportVerification",
    "TensorRTQuantizationRejected",
    "TensorRTReproducibilityError",
    "TensorRTShapeProfile",
    "TensorRTVerificationError",
    "build_tensorrt_benchmark_report",
    "build_tensorrt_engine",
    "certify_tensorrt_reference",
    "measure_tensorrt_latency",
    "sha256_file",
    "sha256_json",
    "verify_tensorrt_engine_hash",
    "write_tensorrt_benchmark_report",
]
