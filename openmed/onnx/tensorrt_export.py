"""TensorRT engine export, certification, and benchmark helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from numbers import Integral
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
        for field_name in (
            "min_batch_size",
            "opt_batch_size",
            "max_batch_size",
            "min_sequence_length",
            "opt_sequence_length",
            "max_sequence_length",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            object.__setattr__(self, field_name, int(value))
        _validate_shape_range(
            "batch size",
            self.min_batch_size,
            self.opt_batch_size,
            self.max_batch_size,
        )
        _validate_shape_range(
            "sequence length",
            self.min_sequence_length,
            self.opt_sequence_length,
            self.max_sequence_length,
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
    tolerance: float
    max_abs_logit_delta: float
    reference_token_spans: tuple[dict[str, Any], ...]
    tensorrt_token_spans: tuple[dict[str, Any], ...]
    passed: bool = True

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable verification metadata."""

        return {
            "sample_text_sha256": self.sample_text_sha256,
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
    recall_delta_gate: QuantRecallDeltaResult | None = None
    verification: TensorRTExportVerification | None = None

    def to_metadata(self, root: str | Path | None = None) -> dict[str, Any]:
        """Return JSON-serializable engine build metadata."""

        base = Path(root) if root is not None else self.engine_path.parent
        payload: dict[str, Any] = {
            "profile": TENSORRT_PROFILE_NAME,
            "format": _format_for_precision(self.precision),
            "family": self.family,
            "precision": self.precision,
            "engine_path": _relative_or_absolute(self.engine_path, base),
            "source_onnx_path": _relative_or_absolute(self.source_onnx_path, base),
            "build_onnx_path": _relative_or_absolute(self.build_onnx_path, base),
            "shape_profile": self.shape_profile.to_dict(),
            "tensorrt_version": self.tensorrt_version,
            "source_onnx_sha256": self.source_onnx_sha256,
            "build_onnx_sha256": self.build_onnx_sha256,
            "build_input_sha256": self.build_input_sha256,
            "engine_sha256": self.engine_sha256,
        }
        if self.calibration_sha256 is not None:
            payload["calibration_sha256"] = self.calibration_sha256
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
    p95_latency_ms: float | None = None
    batch_size: int = 1
    sequence_length: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("device_tier", "device"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value.strip())
        normalized_precision = str(self.precision).strip().lower()
        if normalized_precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"unsupported TensorRT benchmark precision {self.precision!r}"
            )
        object.__setattr__(self, "precision", normalized_precision)
        for field_name, allow_zero in (
            ("latency_ms", False),
            ("throughput_items_per_second", True),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must be a finite number")
            normalized = float(value)
            if (
                not math.isfinite(normalized)
                or normalized < 0
                or (not allow_zero and normalized == 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{field_name} must be finite and {qualifier}")
            object.__setattr__(self, field_name, normalized)
        p95_latency = (
            self.latency_ms if self.p95_latency_ms is None else self.p95_latency_ms
        )
        if isinstance(p95_latency, bool):
            raise TypeError("p95_latency_ms must be a finite number")
        normalized_p95 = float(p95_latency)
        if not math.isfinite(normalized_p95) or normalized_p95 <= 0:
            raise ValueError("p95_latency_ms must be finite and positive")
        if normalized_p95 < self.latency_ms:
            raise ValueError("p95_latency_ms must not be below p50 latency_ms")
        object.__setattr__(self, "p95_latency_ms", normalized_p95)
        for field_name in ("sample_count", "batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, int(value))
        if self.sequence_length is not None:
            value = self.sequence_length
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("sequence_length must be an integer")
            if int(value) < 1:
                raise ValueError("sequence_length must be positive")
            object.__setattr__(self, "sequence_length", int(value))

    def to_metrics(self) -> dict[str, Any]:
        """Return the standard benchmark metrics block for this device."""

        payload: dict[str, Any] = {
            "device_tier": self.device_tier,
            "device": self.device,
            "precision": self.precision,
            "latency": {
                "p50_ms": self.latency_ms,
                "p95_ms": self.p95_latency_ms,
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
    tokenizer: Any
    texts: tuple[str, ...]
    cache_path: Path | None
    allow_cache_read: bool = False


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
        sample_text: Synthetic note used for parity; only its SHA-256 is stored.
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

    source_onnx = Path(onnx_path)
    engine_path = Path(output_path)
    metadata_path = engine_path.with_suffix(
        engine_path.suffix + TENSORRT_BUILD_METADATA_SUFFIX
    )
    if not source_onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source_onnx}")
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must not be empty")
    normalized_family = family.strip()
    normalized_precision = str(precision).strip().lower()
    if normalized_precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"unsupported TensorRT precision {precision!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_PRECISIONS))
        )
    normalized_workspace_size = _normalize_workspace_size(workspace_size_bytes)
    normalized_tolerance = _normalize_tolerance(tolerance)
    _validate_output_paths(source_onnx, engine_path, metadata_path)

    profile = shape_profile or TensorRTShapeProfile()
    if not isinstance(profile, TensorRTShapeProfile):
        raise TypeError("shape_profile must be a TensorRTShapeProfile")
    calibration_spec = None
    calibration_sha256 = None
    recall_gate = None
    build_onnx = source_onnx
    calibration_text_values: list[str] | None = None

    if normalized_precision == "int8":
        calibration_text_values = _normalize_calibration_texts(calibration_texts)
        calibration_sha256 = calibration_texts_sha256(calibration_text_values)
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
        if calibration_tokenizer is None:
            raise TensorRTQuantizationRejected(
                "TensorRT INT8 export requires a tokenizer for shared calibration",
                recall_gate,
            )

    trt = trt_module or _tensorrt_api()

    if normalized_precision == "int8":
        assert calibration_text_values is not None
        if _supports_legacy_int8_calibration(trt):
            normalized_cache_path = (
                Path(calibration_cache_path)
                if calibration_cache_path is not None
                else None
            )
            if normalized_cache_path is not None:
                _validate_publish_target(
                    normalized_cache_path,
                    name="TensorRT calibration cache",
                )
            calibration_spec = _CalibrationSpec(
                tokenizer=calibration_tokenizer,
                texts=tuple(calibration_text_values),
                cache_path=normalized_cache_path,
                allow_cache_read=(
                    expected_build_input_sha256 is not None
                    and expected_engine_sha256 is not None
                ),
            )
        else:
            build_onnx = engine_path.with_suffix(".int8.onnx")
            _validate_derived_onnx_path(
                source_onnx,
                build_onnx,
                engine_path,
                metadata_path,
            )
            _quantize_onnx_with_modelopt(
                source_onnx,
                build_onnx,
                tokenizer=calibration_tokenizer,
                texts=calibration_text_values,
                shape_profile=profile,
            )

    elif normalized_precision == "fp16" and not _has_builder_flag(trt, "FP16"):
        build_onnx = engine_path.with_suffix(".fp16.onnx")
        _validate_derived_onnx_path(
            source_onnx,
            build_onnx,
            engine_path,
            metadata_path,
        )
        _autocast_onnx_with_modelopt(source_onnx, build_onnx)

    source_onnx_sha256 = sha256_file(source_onnx)
    build_onnx_sha256 = sha256_file(build_onnx)
    tensorrt_version = str(getattr(trt, "__version__", "unknown"))
    build_spec = {
        "schema_version": 1,
        "family": normalized_family,
        "precision": normalized_precision,
        "shape_profile": profile.to_dict(),
        "workspace_size_bytes": normalized_workspace_size,
        "tensorrt_version": tensorrt_version,
        "source_onnx_sha256": source_onnx_sha256,
        "build_onnx_sha256": build_onnx_sha256,
        "calibration_sha256": calibration_sha256,
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
        workspace_size_bytes=normalized_workspace_size,
        calibration_spec=calibration_spec,
        trt=trt,
    )
    if not isinstance(engine_bytes, (bytes, bytearray, memoryview)):
        raise TensorRTBuildError("TensorRT returned an invalid serialized engine")
    engine_bytes = bytes(engine_bytes)
    if not engine_bytes:
        raise TensorRTBuildError("TensorRT returned an empty serialized engine")
    engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
    _check_expected_hash("engine", engine_sha256, expected_engine_sha256)

    verification = None
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{engine_path.name}.",
        suffix=".staging",
        dir=engine_path.parent,
        delete=False,
    ) as staging_file:
        staging_file.write(engine_bytes)
        staging_file.flush()
        os.fsync(staging_file.fileno())
        staging_path = Path(staging_file.name)

    metadata_staging_path: Path | None = None
    try:
        if _verification_requested(sample_inputs, reference_logits, id2label):
            if sample_inputs is None or reference_logits is None or id2label is None:
                raise ValueError(
                    "sample_inputs, reference_logits, and id2label are all required "
                    "for TensorRT synthetic verification"
                )
            session = session_factory(staging_path)
            candidate_logits = session.run(**dict(sample_inputs))
            verification = certify_tensorrt_reference(
                reference_logits=reference_logits,
                tensorrt_logits=candidate_logits,
                id2label=id2label,
                attention_mask=sample_inputs.get("attention_mask"),
                sample_text=sample_text,
                tolerance=normalized_tolerance,
            )
        result = TensorRTBuildResult(
            engine_path=engine_path,
            metadata_path=metadata_path,
            source_onnx_path=source_onnx,
            build_onnx_path=build_onnx,
            family=normalized_family,
            precision=normalized_precision,
            shape_profile=profile,
            tensorrt_version=tensorrt_version,
            source_onnx_sha256=source_onnx_sha256,
            build_onnx_sha256=build_onnx_sha256,
            build_input_sha256=build_input_sha256,
            engine_sha256=engine_sha256,
            calibration_sha256=calibration_sha256,
            recall_delta_gate=recall_gate,
            verification=verification,
        )
        metadata_staging_path = _stage_json(
            metadata_path,
            result.to_metadata(engine_path.parent),
        )
        _publish_engine_bundle(
            engine_staging_path=staging_path,
            engine_path=engine_path,
            metadata_staging_path=metadata_staging_path,
            metadata_path=metadata_path,
        )
        return result
    finally:
        staging_path.unlink(missing_ok=True)
        if metadata_staging_path is not None:
            metadata_staging_path.unlink(missing_ok=True)


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

    normalized_tolerance = _normalize_tolerance(tolerance)
    try:
        reference = np.asarray(reference_logits)
        candidate = np.asarray(tensorrt_logits)
    except (TypeError, ValueError) as exc:
        raise TensorRTVerificationError(
            "TensorRT parity logits must be numeric arrays"
        ) from exc
    if reference.shape != candidate.shape:
        raise TensorRTVerificationError(
            "TensorRT logits shape does not match ONNX reference: "
            f"{candidate.shape} != {reference.shape}"
        )
    if reference.ndim not in {2, 3} or (
        reference.ndim == 3 and reference.shape[0] != 1
    ):
        raise TensorRTVerificationError(
            "TensorRT synthetic parity requires rank-two logits or one rank-three batch"
        )
    if reference.size == 0:
        raise TensorRTVerificationError("TensorRT parity logits must not be empty")
    if not _is_finite_numeric_array(reference) or not _is_finite_numeric_array(
        candidate
    ):
        raise TensorRTVerificationError(
            "TensorRT parity logits must contain only finite numeric values"
        )
    max_abs_delta = float(np.max(np.abs(reference - candidate)))
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
    if max_abs_delta > normalized_tolerance:
        raise TensorRTVerificationError(
            "TensorRT logits exceeded tolerance "
            f"{normalized_tolerance}: max abs delta {max_abs_delta}"
        )
    if candidate_spans != reference_spans:
        raise TensorRTVerificationError(
            "TensorRT decoded token spans do not match ONNX reference"
        )
    return TensorRTExportVerification(
        sample_text_sha256=_sha256_text(sample_text),
        tolerance=normalized_tolerance,
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

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    latencies = []
    for _ in range(iterations):
        started = perf_counter()
        session.run(**dict(sample_inputs))
        latencies.append((perf_counter() - started) * 1000.0)
    latency_ms = median(latencies)
    p95_latency_ms = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    batch_size = _batch_size(sample_inputs)
    return TensorRTBenchmarkRecord(
        device_tier=device_tier,
        device=device,
        precision=precision,
        latency_ms=latency_ms,
        p95_latency_ms=p95_latency_ms,
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
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    device_names = [record.device for record in records]
    metric_keys = [f"{record.device_tier}:{record.device}" for record in records]
    if len(metric_keys) != len(set(metric_keys)):
        raise ValueError("TensorRT benchmark device-tier records must be unique")
    metrics = {key: record.to_metrics() for key, record in zip(metric_keys, records)}
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device="tensorrt:" + ",".join(device_names),
        fixture_count=sum(record.sample_count for record in records),
        generated_at=generated,
        metadata={"profile": TENSORRT_PROFILE_NAME, **dict(metadata or {})},
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
    return report.write_json(output_path)


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
    if init_plugins is not None:
        init_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(_network_creation_flags(trt))
    parser = trt.OnnxParser(network, logger)
    parsed = _parse_onnx(parser, onnx_path)
    if not parsed:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        detail = "; ".join(errors) or "unknown parser error"
        raise TensorRTBuildError(f"TensorRT could not parse ONNX graph: {detail}")

    config = builder.create_builder_config()
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
            shape_profile,
        )
        set_calibration_profile = getattr(config, "set_calibration_profile", None)
        if set_calibration_profile is not None:
            set_calibration_profile(optimization_profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise TensorRTBuildError("TensorRT failed to build a serialized engine")
    return bytes(serialized)


def _add_optimization_profile(
    builder: Any,
    network: Any,
    config: Any,
    profile: TensorRTShapeProfile,
) -> Any:
    optimization_profile = builder.create_optimization_profile()
    if network.num_inputs < 1:
        raise TensorRTBuildError("TensorRT ONNX graph has no inputs")
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        input_shape = tuple(int(dim) for dim in tensor.shape)
        if getattr(tensor, "is_shape_tensor", False):
            raise TensorRTBuildError(
                f"TensorRT shape-tensor input {tensor.name!r} is not supported"
            )
        if len(input_shape) != 2:
            raise TensorRTBuildError(
                "TensorRT token-classification inputs must have rank two; "
                f"{tensor.name!r} has shape {input_shape}"
            )
        if any(dimension == 0 or dimension < -1 for dimension in input_shape):
            raise TensorRTBuildError(
                f"TensorRT input {tensor.name!r} has invalid shape {input_shape}"
            )
        minimum = _merge_profile_shape(input_shape, profile.minimum)
        optimum = _merge_profile_shape(input_shape, profile.optimum)
        maximum = _merge_profile_shape(input_shape, profile.maximum)
        if (
            optimization_profile.set_shape(
                tensor.name,
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
    profile: TensorRTShapeProfile,
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

    input_specs: dict[str, tuple[Any, tuple[int, int]]] = {}
    for index in range(network.num_inputs):
        network_input = network.get_input(index)
        network_shape = tuple(int(dim) for dim in network_input.shape)
        if len(network_shape) != 2:
            raise TensorRTBuildError(
                "TensorRT calibration inputs must have rank two; "
                f"{network_input.name!r} has shape {network_shape}"
            )
        expected_shape = _merge_profile_shape(network_shape, profile.optimum)
        input_specs[network_input.name] = (
            np.dtype(trt.nptype(network_input.dtype)),
            expected_shape,
        )
    if not input_specs:
        raise TensorRTBuildError("TensorRT calibration graph has no inputs")
    calibration_batch_sizes = {shape[0] for _, shape in input_specs.values()}
    calibration_sequence_lengths = {shape[1] for _, shape in input_specs.values()}
    if len(calibration_batch_sizes) != 1:
        raise TensorRTBuildError(
            "TensorRT calibration inputs must use one common batch size"
        )
    if len(calibration_sequence_lengths) != 1:
        raise TensorRTBuildError(
            "TensorRT calibration inputs must use one common sequence length"
        )
    calibration_batch_size = next(iter(calibration_batch_sizes))
    calibration_sequence_length = next(iter(calibration_sequence_lengths))

    class SharedCalibrationEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._texts = iter(calibration_spec.texts)
            self._device_inputs: dict[str, Any] = {}

        def get_batch_size(self) -> int:
            return calibration_batch_size

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            batch = []
            for _ in range(calibration_batch_size):
                try:
                    batch.append(next(self._texts))
                except StopIteration:
                    break
            if not batch:
                return None
            batch.extend([batch[-1]] * (calibration_batch_size - len(batch)))
            encoded = calibration_spec.tokenizer(
                batch,
                max_length=calibration_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            self._device_inputs = {}
            pointers = []
            for name in names:
                if name not in input_specs:
                    raise TensorRTBuildError(
                        f"TensorRT requested unknown calibration input {name!r}"
                    )
                if name not in encoded:
                    raise TensorRTBuildError(
                        f"calibration tokenizer did not produce required input {name!r}"
                    )
                dtype, expected_shape = input_specs[name]
                try:
                    raw_array = np.asarray(encoded[name])
                except (TypeError, ValueError) as exc:
                    raise TensorRTBuildError(
                        f"calibration tokenizer produced invalid values for {name!r}"
                    ) from exc
                if raw_array.shape != expected_shape:
                    raise TensorRTBuildError(
                        "calibration tokenizer produced an unsafe shape for "
                        f"{name!r}: {raw_array.shape} != {expected_shape}"
                    )
                if not _is_finite_numeric_array(raw_array):
                    raise TensorRTBuildError(
                        f"calibration tokenizer produced non-finite values for {name!r}"
                    )
                array = _safe_calibration_cast(raw_array, dtype=dtype, name=name)
                tensor = torch.as_tensor(array, device="cuda").contiguous()
                self._device_inputs[name] = tensor
                pointers.append(int(tensor.data_ptr()))
            return pointers

        def read_calibration_cache(self) -> bytes | None:
            cache_path = calibration_spec.cache_path
            if (
                calibration_spec.allow_cache_read
                and cache_path is not None
                and cache_path.is_file()
                and not cache_path.is_symlink()
            ):
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache: Any) -> None:
            cache_path = calibration_spec.cache_path
            if cache_path is None:
                return
            _write_bytes_atomic(cache_path, bytes(cache))

    return SharedCalibrationEntropyCalibrator()


def _quantize_onnx_with_modelopt(
    source_path: Path,
    output_path: Path,
    *,
    tokenizer: Any,
    texts: Sequence[str],
    shape_profile: TensorRTShapeProfile,
) -> Path:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required for TensorRT INT8 calibration") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openmed-tensorrt-calibration-") as tmp:
        calibration_dir = Path(tmp)
        for index, text in enumerate(texts):
            encoded = tokenizer(
                [text],
                max_length=shape_profile.opt_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            arrays = {}
            for name, value in encoded.items():
                if name not in {"input_ids", "attention_mask", "token_type_ids"}:
                    continue
                try:
                    array = np.asarray(value)
                except (TypeError, ValueError) as exc:
                    raise TensorRTBuildError(
                        f"calibration tokenizer produced invalid values for {name!r}"
                    ) from exc
                expected_shape = (1, shape_profile.opt_sequence_length)
                if array.shape != expected_shape:
                    raise TensorRTBuildError(
                        "calibration tokenizer produced an unsafe shape for "
                        f"{name!r}: {array.shape} != {expected_shape}"
                    )
                if not np.issubdtype(array.dtype, np.integer):
                    raise TensorRTBuildError(
                        f"calibration tokenizer must produce integer values for {name!r}"
                    )
                if bool(np.any(array < 0)):
                    raise TensorRTBuildError(
                        f"calibration tokenizer produced negative values for {name!r}"
                    )
                arrays[name] = np.ascontiguousarray(array)
            if "input_ids" not in arrays or "attention_mask" not in arrays:
                raise TensorRTBuildError(
                    "calibration tokenizer must produce input_ids and attention_mask"
                )
            np.savez(calibration_dir / f"{index:05d}.npz", **arrays)
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
            str(output_path),
        ]
        _run_modelopt(command, purpose="INT8 calibration")
    if not output_path.is_file():
        raise TensorRTBuildError(
            f"Model Optimizer did not create quantized ONNX: {output_path}"
        )
    return output_path


def _autocast_onnx_with_modelopt(source_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "modelopt.onnx.autocast",
        "--onnx_path",
        str(source_path),
        "--output_path",
        str(output_path),
    ]
    _run_modelopt(command, purpose="FP16 autocast")
    if not output_path.is_file():
        raise TensorRTBuildError(
            f"Model Optimizer did not create FP16 ONNX: {output_path}"
        )
    return output_path


def _run_modelopt(command: Sequence[str], *, purpose: str) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TensorRTBuildError(
            f"TensorRT {purpose} requires NVIDIA Model Optimizer"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        raise TensorRTBuildError(
            f"TensorRT {purpose} failed in NVIDIA Model Optimizer: {detail}"
        ) from exc


def _parse_onnx(parser: Any, onnx_path: Path) -> bool:
    parse_from_file = getattr(parser, "parse_from_file", None)
    if parse_from_file is not None:
        return bool(parse_from_file(str(onnx_path)))
    return bool(parser.parse(onnx_path.read_bytes()))


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
    return (
        requested_shape[0] if network_shape[0] == -1 else network_shape[0],
        requested_shape[1] if network_shape[1] == -1 else network_shape[1],
    )


def _normalize_calibration_texts(texts: Iterable[str] | None) -> list[str]:
    raw_texts = load_quantization_calibration_texts() if texts is None else list(texts)
    normalized = [str(text).strip() for text in raw_texts if str(text).strip()]
    if not normalized:
        gate = evaluate_quant_recall_delta(
            format_name=TENSORRT_INT8_FORMAT,
            candidate_recall={},
        )
        raise TensorRTQuantizationRejected(
            "TensorRT INT8 export requires calibration samples",
            gate,
        )
    return normalized


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
        raise TensorRTReproducibilityError(
            f"TensorRT expected {name} hash must be a 64-character SHA-256 digest"
        )
    if actual != normalized:
        raise TensorRTReproducibilityError(
            f"TensorRT {name} hash mismatch: {actual} != {normalized}"
        )


def _normalize_workspace_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("workspace_size_bytes must be an integer")
    normalized = int(value)
    if normalized < 1 or normalized > sys.maxsize:
        raise ValueError(f"workspace_size_bytes must be between 1 and {sys.maxsize}")
    return normalized


def _normalize_tolerance(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("tolerance must be a finite non-negative number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("tolerance must be a finite non-negative number") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    return normalized


def _validate_shape_range(name: str, minimum: int, optimum: int, maximum: int) -> None:
    if minimum < 1 or minimum > optimum or optimum > maximum or maximum > (2**31 - 1):
        raise ValueError(
            f"invalid {name} range: expected 1 <= min <= opt <= max <= "
            f"{2**31 - 1}, got "
            f"{minimum} <= {optimum} <= {maximum}"
        )


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


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_finite_numeric_array(value: Any) -> bool:
    import numpy as np

    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        return False
    return bool(np.all(np.isfinite(array)))


def _safe_calibration_cast(value: Any, *, dtype: Any, name: str) -> Any:
    import numpy as np

    array = np.asarray(value)
    if bool(np.any(array < 0)):
        raise TensorRTBuildError(
            f"calibration tokenizer produced negative values for {name!r}"
        )
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        if np.issubdtype(array.dtype, np.floating) and not bool(
            np.all(array == np.trunc(array))
        ):
            raise TensorRTBuildError(
                f"calibration tokenizer produced fractional values for {name!r}"
            )
        integer_limits = np.iinfo(target_dtype)
        if bool(np.any(array < integer_limits.min)) or bool(
            np.any(array > integer_limits.max)
        ):
            raise TensorRTBuildError(
                f"calibration tokenizer values exceed the dtype range for {name!r}"
            )
    elif np.issubdtype(target_dtype, np.floating):
        floating_limits = np.finfo(target_dtype)
        if bool(np.any(array < floating_limits.min)) or bool(
            np.any(array > floating_limits.max)
        ):
            raise TensorRTBuildError(
                f"calibration tokenizer values exceed the dtype range for {name!r}"
            )
    else:
        raise TensorRTBuildError(
            f"TensorRT calibration input {name!r} has a non-numeric dtype"
        )
    try:
        converted = np.ascontiguousarray(array, dtype=target_dtype)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TensorRTBuildError(
            f"calibration tokenizer values could not be converted for {name!r}"
        ) from exc
    if not bool(np.all(np.isfinite(converted))):
        raise TensorRTBuildError(
            f"calibration tokenizer values exceed the finite range for {name!r}"
        )
    return converted


def _sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("sample_text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path_identity(path: Path) -> str:
    try:
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"could not resolve artifact path: {path}") from exc


def _same_artifact_path(left: Path, right: Path) -> bool:
    if _path_identity(left) == _path_identity(right):
        return True
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            return False
    return False


def _validate_publish_target(path: Path, *, name: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"{name} must be a regular file path: {path}")


def _validate_output_paths(
    source_path: Path,
    engine_path: Path,
    metadata_path: Path,
) -> None:
    _validate_publish_target(engine_path, name="TensorRT engine output")
    _validate_publish_target(metadata_path, name="TensorRT metadata output")
    paths = (source_path, engine_path, metadata_path)
    if any(
        _same_artifact_path(left, right)
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        raise ValueError("source ONNX, engine output, and metadata output must differ")


def _validate_derived_onnx_path(
    source_path: Path,
    build_path: Path,
    engine_path: Path,
    metadata_path: Path,
) -> None:
    _validate_publish_target(build_path, name="derived ONNX output")
    if any(
        _same_artifact_path(build_path, other)
        for other in (source_path, engine_path, metadata_path)
    ):
        raise ValueError(
            "derived ONNX output must differ from source, engine, and metadata paths"
        )


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
            staging_file.flush()
            os.fsync(staging_file.fileno())
        assert staging_path is not None
        return staging_path
    except BaseException:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
        raise


def _reserve_backup_path(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".backup",
        dir=path.parent,
        delete=False,
    ) as backup_file:
        return Path(backup_file.name)


def _atomic_replace(source: Path, destination: Path) -> None:
    source.replace(destination)


def _publish_engine_bundle(
    *,
    engine_staging_path: Path,
    engine_path: Path,
    metadata_staging_path: Path,
    metadata_path: Path,
) -> None:
    pairs = (
        (engine_staging_path, engine_path),
        (metadata_staging_path, metadata_path),
    )
    for _, target in pairs:
        _validate_publish_target(target, name="TensorRT artifact output")

    backups: dict[Path, Path] = {}
    reserved_backups: set[Path] = set()
    published: list[Path] = []
    try:
        for _, target in pairs:
            if target.exists():
                backup = _reserve_backup_path(target)
                reserved_backups.add(backup)
                _atomic_replace(target, backup)
                backups[target] = backup
        for staging, target in pairs:
            _atomic_replace(staging, target)
            published.append(target)
    except BaseException as exc:
        rollback_errors = []
        for target in reversed(published):
            if target in backups:
                continue
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for _, target in reversed(pairs):
            restored_backup = backups.get(target)
            if restored_backup is not None and restored_backup.exists():
                try:
                    _atomic_replace(restored_backup, target)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
        for backup in reserved_backups - set(backups.values()):
            backup.unlink(missing_ok=True)
        if rollback_errors:
            raise TensorRTBuildError(
                "TensorRT artifact publication failed and rollback was incomplete; "
                "recovery backups were retained"
            ) from exc
        raise
    else:
        for backup in reserved_backups:
            backup.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    staging_path = _stage_json(path, payload)
    try:
        _atomic_replace(staging_path, path)
    finally:
        staging_path.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_publish_target(path, name="calibration cache output")
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".staging",
            dir=path.parent,
            delete=False,
        ) as staging_file:
            staging_path = Path(staging_file.name)
            staging_file.write(payload)
            staging_file.flush()
            os.fsync(staging_file.fileno())
        _atomic_replace(staging_path, path)
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


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
