"""TensorRT engine export, certification, and benchmark helpers."""

from __future__ import annotations

import hashlib
import json
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

    sample_text: str
    tolerance: float
    max_abs_logit_delta: float
    reference_token_spans: tuple[dict[str, Any], ...]
    tensorrt_token_spans: tuple[dict[str, Any], ...]
    passed: bool = True

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable verification metadata."""

        return {
            "sample_text": self.sample_text,
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
    batch_size: int = 1
    sequence_length: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
    tokenizer: Any
    texts: tuple[str, ...]
    cache_path: Path | None


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
        sample_text: Synthetic note description stored with parity evidence.
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
    if not source_onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source_onnx}")
    if not family.strip():
        raise ValueError("family must not be empty")
    normalized_precision = str(precision).strip().lower()
    if normalized_precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"unsupported TensorRT precision {precision!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_PRECISIONS))
        )
    if workspace_size_bytes <= 0:
        raise ValueError("workspace_size_bytes must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    profile = shape_profile or TensorRTShapeProfile()
    trt = trt_module or _tensorrt_api()
    calibration_spec = None
    calibration_sha256 = None
    recall_gate = None
    build_onnx = source_onnx

    if normalized_precision == "int8":
        texts = _normalize_calibration_texts(calibration_texts)
        calibration_sha256 = calibration_texts_sha256(texts)
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

        if _supports_legacy_int8_calibration(trt):
            calibration_spec = _CalibrationSpec(
                tokenizer=calibration_tokenizer,
                texts=tuple(texts),
                cache_path=(
                    Path(calibration_cache_path)
                    if calibration_cache_path is not None
                    else None
                ),
            )
        else:
            build_onnx = engine_path.with_suffix(".int8.onnx")
            _quantize_onnx_with_modelopt(
                source_onnx,
                build_onnx,
                tokenizer=calibration_tokenizer,
                texts=texts,
                shape_profile=profile,
            )

    elif normalized_precision == "fp16" and not _has_builder_flag(trt, "FP16"):
        build_onnx = engine_path.with_suffix(".fp16.onnx")
        _autocast_onnx_with_modelopt(source_onnx, build_onnx)

    source_onnx_sha256 = sha256_file(source_onnx)
    build_onnx_sha256 = sha256_file(build_onnx)
    tensorrt_version = str(getattr(trt, "__version__", "unknown"))
    build_spec = {
        "schema_version": 1,
        "family": family.strip(),
        "precision": normalized_precision,
        "shape_profile": profile.to_dict(),
        "workspace_size_bytes": workspace_size_bytes,
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
        workspace_size_bytes=workspace_size_bytes,
        calibration_spec=calibration_spec,
        trt=trt,
    )
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
        staging_path = Path(staging_file.name)

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
                tolerance=tolerance,
            )
        staging_path.replace(engine_path)
    finally:
        staging_path.unlink(missing_ok=True)

    metadata_path = engine_path.with_suffix(
        engine_path.suffix + TENSORRT_BUILD_METADATA_SUFFIX
    )
    result = TensorRTBuildResult(
        engine_path=engine_path,
        metadata_path=metadata_path,
        source_onnx_path=source_onnx,
        build_onnx_path=build_onnx,
        family=family.strip(),
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
    _write_json_atomic(metadata_path, result.to_metadata(engine_path.parent))
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

    reference = np.asarray(reference_logits)
    candidate = np.asarray(tensorrt_logits)
    if reference.shape != candidate.shape:
        raise TensorRTVerificationError(
            "TensorRT logits shape does not match ONNX reference: "
            f"{candidate.shape} != {reference.shape}"
        )
    max_abs_delta = (
        float(np.max(np.abs(reference - candidate))) if reference.size else 0.0
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
        sample_text=sample_text,
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

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
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
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    device_names = [record.device for record in records]
    metrics = {
        f"{record.device_tier}:{record.device}": record.to_metrics()
        for record in records
    }
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

    input_dtypes = {
        network.get_input(index).name: np.dtype(
            trt.nptype(network.get_input(index).dtype)
        )
        for index in range(network.num_inputs)
    }

    class SharedCalibrationEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._texts = iter(calibration_spec.texts)
            self._device_inputs: dict[str, Any] = {}

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            try:
                text = next(self._texts)
            except StopIteration:
                return None
            encoded = calibration_spec.tokenizer(
                [text],
                max_length=profile.opt_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            self._device_inputs = {}
            pointers = []
            for name in names:
                if name not in encoded:
                    raise TensorRTBuildError(
                        f"calibration tokenizer did not produce required input {name!r}"
                    )
                array = np.ascontiguousarray(encoded[name], dtype=input_dtypes[name])
                tensor = torch.as_tensor(array, device="cuda").contiguous()
                self._device_inputs[name] = tensor
                pointers.append(int(tensor.data_ptr()))
            return pointers

        def read_calibration_cache(self) -> bytes | None:
            cache_path = calibration_spec.cache_path
            if cache_path is not None and cache_path.is_file():
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache: Any) -> None:
            cache_path = calibration_spec.cache_path
            if cache_path is None:
                return
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(cache))

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
            arrays = {
                name: np.ascontiguousarray(value)
                for name, value in encoded.items()
                if name in {"input_ids", "attention_mask", "token_type_ids"}
            }
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
    return tuple(
        requested if network == -1 else network
        for network, requested in zip(network_shape, requested_shape)
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
    if actual != normalized:
        raise TensorRTReproducibilityError(
            f"TensorRT {name} hash mismatch: {actual} != {normalized}"
        )


def _validate_shape_range(name: str, minimum: int, optimum: int, maximum: int) -> None:
    if minimum < 1 or minimum > optimum or optimum > maximum:
        raise ValueError(
            f"invalid {name} range: expected 1 <= min <= opt <= max, got "
            f"{minimum} <= {optimum} <= {maximum}"
        )


def _batch_size(inputs: Mapping[str, Any]) -> int:
    value = inputs.get("input_ids")
    try:
        return int(value.shape[0])
    except AttributeError:
        try:
            return len(value)
        except TypeError:
            return 1


def _sequence_length(inputs: Mapping[str, Any]) -> int | None:
    value = inputs.get("input_ids")
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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".staging",
        dir=path.parent,
        delete=False,
    ) as staging_file:
        json.dump(payload, staging_file, indent=2, sort_keys=True)
        staging_file.write("\n")
        staging_path = Path(staging_file.name)
    staging_path.replace(path)


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
