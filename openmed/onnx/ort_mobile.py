"""ONNX Runtime Mobile conversion helpers for Android ONNX artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.onnx.android_profile import (
    ANDROID_PROFILE_NAME,
    AndroidProfileValidation,
    validate_android_profile,
)

logger = logging.getLogger(__name__)

ORT_ANDROID_FORMAT = "ort-android"
ORT_MODEL_FILENAME = "model.ort"
ORT_REQUIRED_OPERATORS_CONFIG_FILENAME = "model.required_operators_and_types.config"
ORT_REQUIRED_OPERATORS_CONFIG_TYPE = "required_operators_and_types"
ORT_OPTIMIZATION_STYLE = "Fixed"
ORT_TARGET_PLATFORM = "arm"
ORT_CONVERSION_UNAVAILABLE_REASON = (
    "ONNX Runtime ORT mobile conversion tooling is unavailable; install the "
    "onnx extra with `pip install openmed[onnx]` to emit .ort artifacts."
)
ORT_CONVERSION_FAILED_REASON = (
    "ONNX Runtime could not serialize this graph to ORT format; the ONNX "
    "artifacts remain available for Android execution."
)


class OrtMobileConversionUnavailable(ImportError):
    """Raised when optional ONNX Runtime mobile conversion tooling is missing."""


@dataclass(frozen=True)
class OrtMobileConversionResult:
    """Paths and metadata from an Android ORT mobile conversion attempt."""

    source_onnx_path: Path
    ort_path: Path | None = None
    op_config_path: Path | None = None
    validation: AndroidProfileValidation | None = None
    skipped: bool = False
    skip_reason: str | None = None

    def to_metadata(self, root: Path) -> dict[str, Any]:
        """Return JSON-serializable manifest metadata for an ORT artifact."""

        metadata: dict[str, Any] = {
            "format": ORT_ANDROID_FORMAT,
            "profile": ANDROID_PROFILE_NAME,
            "source_onnx_path": _relative_manifest_path(self.source_onnx_path, root),
            "optimization_style": ORT_OPTIMIZATION_STYLE,
            "target_platform": ORT_TARGET_PLATFORM,
            "op_config_type": ORT_REQUIRED_OPERATORS_CONFIG_TYPE,
            "enable_type_reduction": True,
        }
        if self.ort_path is not None:
            metadata["ort_path"] = _relative_manifest_path(self.ort_path, root)
        if self.op_config_path is not None:
            metadata["op_config_path"] = _relative_manifest_path(
                self.op_config_path,
                root,
            )
        if self.validation is not None:
            metadata["operators"] = list(self.validation.operators)
            metadata["opset"] = self.validation.opset
        return metadata


def convert_android_onnx_to_ort(
    onnx_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    validation: AndroidProfileValidation | None = None,
) -> OrtMobileConversionResult:
    """Convert an Android-profile ONNX model into ORT mobile format."""

    source_path = Path(onnx_path)
    artifact_dir = Path(output_dir) if output_dir is not None else source_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        ort_tools = _load_ort_tools()
    except OrtMobileConversionUnavailable as exc:
        reason = str(exc)
        logger.info("Skipping Android ORT mobile conversion: %s", reason)
        return OrtMobileConversionResult(
            source_onnx_path=source_path,
            skipped=True,
            skip_reason=reason,
        )

    validation = validation or validate_android_profile(source_path)
    fixed_style = getattr(ort_tools.OptimizationStyle, ORT_OPTIMIZATION_STYLE)
    ort_path = artifact_dir / source_path.with_suffix(".ort").name
    op_config_path = artifact_dir / (
        source_path.stem + ".required_operators_and_types.config"
    )
    try:
        ort_tools.convert_onnx_models_to_ort(
            source_path,
            output_dir=artifact_dir,
            optimization_styles=[fixed_style],
            target_platform=ORT_TARGET_PLATFORM,
            save_optimized_onnx_model=False,
            allow_conversion_failures=False,
            enable_type_reduction=True,
        )
    except Exception as exc:
        ort_path.unlink(missing_ok=True)
        op_config_path.unlink(missing_ok=True)
        logger.warning(
            "Skipping Android ORT mobile conversion after %s; %s",
            exc.__class__.__name__,
            ORT_CONVERSION_FAILED_REASON,
        )
        return OrtMobileConversionResult(
            source_onnx_path=source_path,
            validation=validation,
            skipped=True,
            skip_reason=ORT_CONVERSION_FAILED_REASON,
        )

    _require_generated_file(ort_path, "ORT mobile model")
    _require_generated_file(op_config_path, "ORT required-operators config")

    return OrtMobileConversionResult(
        source_onnx_path=source_path,
        ort_path=ort_path,
        op_config_path=op_config_path,
        validation=validation,
    )


def _load_ort_tools() -> Any:
    try:
        from onnxruntime.tools import convert_onnx_models_to_ort as ort_tools
    except ImportError as exc:
        raise OrtMobileConversionUnavailable(ORT_CONVERSION_UNAVAILABLE_REASON) from exc

    required = (
        "OptimizationStyle",
        "convert_onnx_models_to_ort",
    )
    if any(not hasattr(ort_tools, name) for name in required):
        raise OrtMobileConversionUnavailable(ORT_CONVERSION_UNAVAILABLE_REASON)
    if not hasattr(ort_tools.OptimizationStyle, ORT_OPTIMIZATION_STYLE):
        raise OrtMobileConversionUnavailable(ORT_CONVERSION_UNAVAILABLE_REASON)
    return ort_tools


def _require_generated_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"ONNX Runtime did not produce the {description}: {path}")


def _relative_manifest_path(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
