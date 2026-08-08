"""ONNX and browser-targeted conversion helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from openmed.core.capabilities import is_backend_available as _is_backend_available
from openmed.core.capabilities import require_backend as _require_backend


def is_onnx_available() -> bool:
    """Return True when the ``onnx`` extra is importable, without importing it."""

    return _is_backend_available("onnx")


def ensure_onnx_available() -> None:
    """Raise an actionable error when the ``onnx`` extra is not installed."""

    _require_backend("onnx", feature="ONNX export and inference")


def is_openvino_available() -> bool:
    """Return True when the ``openvino`` extra is importable, without importing."""

    return _is_backend_available("openvino")


def ensure_openvino_available() -> None:
    """Raise an actionable error when the ``openvino`` extra is not installed."""

    _require_backend("openvino", feature="OpenVINO export and inference")


__all__ = [
    "ensure_onnx_available",
    "ensure_openvino_available",
    "is_onnx_available",
    "is_openvino_available",
    "ANDROID_ONNX_FORMAT",
    "ANDROID_ONNX_OPSET",
    "ANDROID_PROFILE_NAME",
    "OPENVINO_FORMAT",
    "OPENVINO_INT8_FORMAT",
    "OPENVINO_PROFILE_NAME",
    "AndroidProfileValidation",
    "BufferReleaseError",
    "ExportArtifact",
    "INT8_ONNX_FILENAME",
    "LayerGroupSpec",
    "LocalWeightsRequired",
    "MAPLE_ARCHITECTURE",
    "MAPLE_BUNDLE_FILENAME",
    "MAPLE_BUNDLE_RUNTIMES",
    "MAPLE_MLX_MODEL",
    "MAPLE_MLX_REVISION",
    "MAPLE_MLX_SOURCE_RECEIPT_FILENAME",
    "MAPLE_ONNX_2BIT_QUANTIZATION",
    "MAPLE_ONNX_2BIT_RUNTIME_MIN_VERSION",
    "MAPLE_ONNX_EXPORT_REQUIREMENTS",
    "MAPLE_ONNX_GRAPH_FILENAME",
    "MAPLE_ONNX_QUANTIZATION",
    "MAPLE_ONNX_RECEIPT_FILENAME",
    "MAPLE_ONNX_RUNTIME_MIN_VERSION",
    "MAPLE_ORT_GENAI_VERSION",
    "MAPLE_QMOE_SMOKE_BLOCK_SIZE",
    "MAPLE_QMOE_SMOKE_MIN_ORT_VERSION",
    "MAPLE_SOURCE_MODEL",
    "MAPLE_SOURCE_REVISION",
    "MAPLE_VOCAB_SIZE",
    "MAPLE_WEB_EXTERNAL_SHARD_BYTES",
    "MapleBundleBuild",
    "MapleBundleError",
    "MapleBundleFile",
    "MapleOnnxExportError",
    "MapleOnnxExportPlan",
    "MapleOnnxExportResult",
    "MapleOnnxSourceInspection",
    "MapleMlxSourceInspection",
    "MapleQMoESmokeError",
    "MapleQMoESmokeResult",
    "ONNX_INT8_FORMAT",
    "OnnxEntity",
    "OnnxModel",
    "OnnxConversionResult",
    "OpenVinoBenchmarkRecord",
    "OpenVinoDeviceSelection",
    "OpenVinoExportResult",
    "OpenVinoExportVerification",
    "OpenVinoQuantizationRejected",
    "OpenVinoQuantizationResult",
    "OpenVinoTokenClassificationSession",
    "OpenVinoVerificationError",
    "build_openvino_benchmark_report",
    "certify_openvino_reference",
    "apply_int8_recall_certification",
    "OnnxOptimizationConfig",
    "PeakRamProbe",
    "PeakRamReport",
    "ORT_ANDROID_FORMAT",
    "OrtMobileConversionResult",
    "ShapeBucketConfig",
    "RamBudget",
    "RamBudgetExceeded",
    "RamProbeUnavailable",
    "ShardFormatError",
    "StreamedLayerGroup",
    "StreamingLoadReport",
    "StreamingWeightLoader",
    "build_maple_onnx_bundle",
    "build_maple_onnx_export_bundle",
    "create_maple_onnx_bundle_manifest",
    "convert",
    "convert_android_onnx_to_ort",
    "export_android_fp16",
    "export_onnx",
    "export_openvino_ir",
    "export_maple_onnx",
    "export_transformersjs_bundle",
    "export_webgpu",
    "measure_openvino_latency",
    "quantize_openvino_int8",
    "resolve_openvino_device",
    "run_onnx_reference_logits",
    "token_spans_from_logits",
    "int8_artifact_metadata",
    "load_onnx_model",
    "download_maple_onnx_source",
    "inspect_maple_onnx_source",
    "inspect_maple_mlx_source",
    "maple_onnx_export_requirements",
    "optimize_onnx_graph",
    "quantize_android_int8",
    "quantize_dynamic_int8",
    "plan_maple_onnx_export",
    "repack_maple_onnx_qmoe_2bit",
    "run_maple_qmoe_smoke",
    "validate_android_profile",
    "validate_maple_onnx_bundle",
    "validate_maple_onnx_graph",
    "validate_maple_qmoe_runtime",
    "validate_optimized_onnx_export",
    "validate_transformersjs_bundle",
    "validate_transformersjs_contract",
    "write_openvino_benchmark_report",
    "write_int8_recall_delta_report",
    "write_maple_qmoe_smoke_model",
    "write_maple_onnx_export_manifest",
    "write_export_manifest",
]


def __getattr__(name: str) -> Any:
    """Load conversion helpers lazily so ``python -m`` stays warning-free."""

    if name in __all__:
        if name in {
            "MAPLE_ARCHITECTURE",
            "MAPLE_BUNDLE_FILENAME",
            "MAPLE_BUNDLE_RUNTIMES",
            "MAPLE_SOURCE_MODEL",
            "MAPLE_SOURCE_REVISION",
            "MAPLE_VOCAB_SIZE",
            "MapleBundleBuild",
            "MapleBundleError",
            "MapleBundleFile",
            "build_maple_onnx_bundle",
            "create_maple_onnx_bundle_manifest",
            "validate_maple_onnx_bundle",
        }:
            module = import_module("openmed.onnx.maple_bundle")
            return getattr(module, name)
        if name in {
            "MAPLE_MLX_MODEL",
            "MAPLE_MLX_REVISION",
            "MAPLE_MLX_SOURCE_RECEIPT_FILENAME",
            "MAPLE_ONNX_2BIT_QUANTIZATION",
            "MAPLE_ONNX_2BIT_RUNTIME_MIN_VERSION",
            "MAPLE_ONNX_EXPORT_REQUIREMENTS",
            "MAPLE_ONNX_GRAPH_FILENAME",
            "MAPLE_ONNX_QUANTIZATION",
            "MAPLE_ONNX_RECEIPT_FILENAME",
            "MAPLE_ONNX_RUNTIME_MIN_VERSION",
            "MAPLE_ORT_GENAI_VERSION",
            "MAPLE_WEB_EXTERNAL_SHARD_BYTES",
            "MapleOnnxExportError",
            "MapleOnnxExportPlan",
            "MapleOnnxExportResult",
            "MapleOnnxSourceInspection",
            "MapleMlxSourceInspection",
            "build_maple_onnx_export_bundle",
            "download_maple_onnx_source",
            "export_maple_onnx",
            "inspect_maple_onnx_source",
            "inspect_maple_mlx_source",
            "maple_onnx_export_requirements",
            "plan_maple_onnx_export",
            "repack_maple_onnx_qmoe_2bit",
            "validate_maple_onnx_graph",
            "write_maple_onnx_export_manifest",
        }:
            module = import_module("openmed.onnx.maple_export")
            return getattr(module, name)
        if name in {
            "MAPLE_QMOE_SMOKE_BLOCK_SIZE",
            "MAPLE_QMOE_SMOKE_MIN_ORT_VERSION",
            "MapleQMoESmokeError",
            "MapleQMoESmokeResult",
            "run_maple_qmoe_smoke",
            "validate_maple_qmoe_runtime",
            "write_maple_qmoe_smoke_model",
        }:
            module = import_module("openmed.onnx.maple_qmoe_smoke")
            return getattr(module, name)
        if name in {
            "PeakRamProbe",
            "PeakRamReport",
            "RamBudget",
            "RamBudgetExceeded",
            "RamProbeUnavailable",
        }:
            module = import_module("openmed.onnx.ram_budget")
            return getattr(module, name)
        if name in {
            "BufferReleaseError",
            "LayerGroupSpec",
            "LocalWeightsRequired",
            "ShardFormatError",
            "StreamedLayerGroup",
            "StreamingLoadReport",
            "StreamingWeightLoader",
        }:
            module = import_module("openmed.onnx.streaming_loader")
            return getattr(module, name)
        if name in {"OnnxEntity", "OnnxModel", "load_onnx_model"}:
            module = import_module("openmed.onnx.inference")
            return getattr(module, name)
        if name.startswith(("ANDROID_", "Android", "export_android")) or (
            name == "validate_android_profile"
        ):
            module = import_module("openmed.onnx.android_profile")
            return getattr(module, name)
        if name in {
            "OpenVinoDeviceSelection",
            "OpenVinoTokenClassificationSession",
            "resolve_openvino_device",
        }:
            module = import_module("openmed.onnx.openvino_session")
            return getattr(module, name)
        if name.startswith(("OPENVINO_", "OpenVino")) or name in {
            "build_openvino_benchmark_report",
            "certify_openvino_reference",
            "export_openvino_ir",
            "measure_openvino_latency",
            "quantize_openvino_int8",
            "run_onnx_reference_logits",
            "token_spans_from_logits",
            "write_openvino_benchmark_report",
        }:
            module = import_module("openmed.onnx.openvino_export")
            return getattr(module, name)
        if name.startswith(("export_transformersjs", "validate_transformersjs")):
            module = import_module("openmed.onnx.transformersjs")
            return getattr(module, name)
        if name in {
            "INT8_ONNX_FILENAME",
            "ONNX_INT8_FORMAT",
            "apply_int8_recall_certification",
            "int8_artifact_metadata",
            "quantize_android_int8",
            "quantize_dynamic_int8",
            "write_int8_recall_delta_report",
        }:
            module = import_module("openmed.onnx.quantize_int8")
            return getattr(module, name)
        if (
            name.startswith("ORT_")
            or name.startswith("OrtMobile")
            or name == "convert_android_onnx_to_ort"
        ):
            module = import_module("openmed.onnx.ort_mobile")
            return getattr(module, name)
        module = import_module("openmed.onnx.convert")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
