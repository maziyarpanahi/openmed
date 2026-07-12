"""Android ONNX Runtime Mobile profile helpers for token classifiers.

The Android profile exports token-classification graphs with a fixed ONNX
opset, stable tensor names, and named dynamic axes. Batch and sequence axes stay
dynamic so one artifact can serve variable Android inference batches, while the
fixed opset keeps the graph predictable for the later ONNX Runtime Mobile
``.ort`` conversion step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ANDROID_PROFILE_NAME = "android"
ANDROID_ONNX_FORMAT = "onnx-android"
ANDROID_ONNX_OPSET = 18
ANDROID_FP16_FILENAME = "model_fp16.onnx"
ANDROID_EXECUTION_PROVIDERS = ("NNAPI", "XNNPACK")

ANDROID_REQUIRED_INPUTS = ("input_ids", "attention_mask")
ANDROID_OPTIONAL_INPUTS = ("token_type_ids",)
ANDROID_OUTPUTS = ("logits",)
ANDROID_INPUT_DTYPE = "int64"
ANDROID_LOGITS_DTYPE = "float32"

_MOBILE_SAFE_OPS = frozenset(
    {
        "Add",
        "And",
        "ArgMax",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Dropout",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "GatherND",
        "Gelu",
        "Gemm",
        "Identity",
        "LayerNormalization",
        "Less",
        "Log",
        "MatMul",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)

_ONNX_DTYPE_NAMES = {
    1: "float32",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AndroidProfileValidation:
    """Validated graph contract and mobile compatibility warnings."""

    path: Path
    opset: int
    inputs: tuple[dict[str, Any], ...]
    outputs: tuple[dict[str, Any], ...]
    operators: tuple[str, ...]
    unsupported_ops: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable artifact metadata for the export manifest."""

        return {
            "profile": ANDROID_PROFILE_NAME,
            "opset": self.opset,
            "execution_providers": list(ANDROID_EXECUTION_PROVIDERS),
            "inputs": [dict(item) for item in self.inputs],
            "outputs": [dict(item) for item in self.outputs],
            "operators": list(self.operators),
            "unsupported_ops": list(self.unsupported_ops),
            "warnings": list(self.warnings),
        }


def export_android_fp16(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    keep_io_types: bool = True,
    expected_opset: int = ANDROID_ONNX_OPSET,
    validate: bool = True,
) -> Path:
    """Create an Android fp16-weight ONNX variant and validate its graph."""

    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required for Android fp16 export. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    fp16_model = convert_float_to_float16(
        str(onnx_path),
        keep_io_types=keep_io_types,
    )
    _copy_missing_model_metadata(fp16_model, model)
    onnx.save(fp16_model, str(output_path))
    if validate:
        validate_android_profile(output_path, expected_opset=expected_opset)
    return output_path


def validate_android_profile(
    model_path: str | Path,
    *,
    expected_opset: int = ANDROID_ONNX_OPSET,
) -> AndroidProfileValidation:
    """Validate Android-profile tensor contract, opset, and mobile ops."""

    path = Path(model_path)
    model = _load_and_check_onnx(path)
    opset = _default_opset(model)
    if opset != expected_opset:
        raise ValueError(
            "Android ONNX profile requires opset "
            f"{expected_opset}, found {opset} in {path}"
        )

    inputs, outputs = _validate_tensor_contract(model)
    operators = _operators(model)
    unsupported_ops = tuple(op for op in operators if op not in _MOBILE_SAFE_OPS)
    warnings = tuple(
        f"{op} is not in the Android ONNX Runtime Mobile NNAPI/XNNPACK safe set"
        for op in unsupported_ops
    )
    for warning in warnings:
        logger.warning("Android ONNX profile warning for %s: %s", path, warning)

    return AndroidProfileValidation(
        path=path,
        opset=opset,
        inputs=inputs,
        outputs=outputs,
        operators=operators,
        unsupported_ops=unsupported_ops,
        warnings=warnings,
    )


def _load_and_check_onnx(path: Path) -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate Android ONNX artifacts. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(str(path))
    return model


def _copy_missing_model_metadata(target: Any, source: Any) -> None:
    if not getattr(target, "ir_version", None) and getattr(source, "ir_version", None):
        target.ir_version = source.ir_version

    for field in ("producer_name", "producer_version", "domain", "doc_string"):
        if not getattr(target, field, None) and getattr(source, field, None):
            setattr(target, field, getattr(source, field))
    if not getattr(target, "model_version", None) and getattr(
        source, "model_version", None
    ):
        target.model_version = source.model_version

    target_opsets = getattr(target, "opset_import", None)
    source_opsets = getattr(source, "opset_import", None)
    if target_opsets is not None and source_opsets is not None and not target_opsets:
        target_opsets.extend(source_opsets)

    target_graph = getattr(target, "graph", None)
    source_graph = getattr(source, "graph", None)
    if (
        target_graph is not None
        and source_graph is not None
        and not getattr(target_graph, "name", None)
        and getattr(source_graph, "name", None)
    ):
        target_graph.name = source_graph.name


def _default_opset(model: Any) -> int:
    for opset in getattr(model, "opset_import", ()):
        if getattr(opset, "domain", "") in {"", "ai.onnx"}:
            return int(getattr(opset, "version"))
    raise ValueError("Android ONNX profile graph is missing the default opset")


def _validate_tensor_contract(
    model: Any,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    graph = model.graph
    initializer_names = {getattr(item, "name", "") for item in graph.initializer}
    graph_inputs = [
        item
        for item in graph.input
        if getattr(item, "name", "") not in initializer_names
    ]
    input_by_name = {item.name: item for item in graph_inputs}
    output_by_name = {item.name: item for item in graph.output}

    missing_inputs = [
        name for name in ANDROID_REQUIRED_INPUTS if name not in input_by_name
    ]
    if missing_inputs:
        raise ValueError(
            "Android ONNX profile model is missing inputs: " + ", ".join(missing_inputs)
        )

    allowed_inputs = set(ANDROID_REQUIRED_INPUTS) | set(ANDROID_OPTIONAL_INPUTS)
    unexpected_inputs = sorted(set(input_by_name) - allowed_inputs)
    if unexpected_inputs:
        raise ValueError(
            "Android ONNX profile model has unexpected inputs: "
            + ", ".join(unexpected_inputs)
        )

    expected_input_order = [
        *ANDROID_REQUIRED_INPUTS,
        *[name for name in ANDROID_OPTIONAL_INPUTS if name in input_by_name],
    ]
    actual_input_order = [item.name for item in graph_inputs]
    if actual_input_order != expected_input_order:
        raise ValueError(
            "Android ONNX profile inputs must be ordered as "
            f"{', '.join(expected_input_order)}"
        )

    if list(output_by_name) != list(ANDROID_OUTPUTS):
        raise ValueError(
            "Android ONNX profile model must expose only "
            f"{', '.join(ANDROID_OUTPUTS)} output"
        )

    inputs = []
    for name in expected_input_order:
        value_info = input_by_name[name]
        _validate_dtype(value_info, ANDROID_INPUT_DTYPE)
        _validate_axes(value_info, ("batch", "sequence"))
        inputs.append(_value_info_contract(value_info, ("batch", "sequence")))

    logits = output_by_name["logits"]
    _validate_dtype(logits, ANDROID_LOGITS_DTYPE)
    _validate_axes(logits, ("batch", "sequence", "labels"))
    outputs = (_value_info_contract(logits, ("batch", "sequence", "labels")),)
    return tuple(inputs), outputs


def _validate_dtype(value_info: Any, expected_dtype: str) -> None:
    actual = _dtype_name(value_info)
    if actual != expected_dtype:
        raise ValueError(
            f"{value_info.name} must have dtype {expected_dtype}, found {actual}"
        )


def _validate_axes(value_info: Any, axes: Sequence[str]) -> None:
    dims = _tensor_dims(value_info)
    if len(dims) != len(axes):
        raise ValueError(
            f"{value_info.name} must have rank {len(axes)} for axes " + ", ".join(axes)
        )

    for index, axis in enumerate(axes):
        dim = _dim_contract(dims[index])
        if axis in {"batch", "sequence"}:
            if dim["kind"] != "dynamic" or dim.get("name") != axis:
                raise ValueError(f"{value_info.name} axis {axis} must be dynamic")
        elif axis == "labels" and dim["kind"] != "static":
            raise ValueError(f"{value_info.name} axis labels must be static")


def _value_info_contract(value_info: Any, axes: Sequence[str]) -> dict[str, Any]:
    return {
        "name": value_info.name,
        "dtype": _dtype_name(value_info),
        "axes": list(axes),
        "shape": [_dim_contract(dim) for dim in _tensor_dims(value_info)],
    }


def _tensor_dims(value_info: Any) -> Sequence[Any]:
    try:
        return value_info.type.tensor_type.shape.dim
    except AttributeError as exc:
        raise ValueError(f"{value_info.name} must have tensor shape metadata") from exc


def _dtype_name(value_info: Any) -> str:
    try:
        elem_type = int(value_info.type.tensor_type.elem_type)
    except AttributeError as exc:
        raise ValueError(f"{value_info.name} must have tensor dtype metadata") from exc
    return _ONNX_DTYPE_NAMES.get(elem_type, f"onnx:{elem_type}")


def _dim_contract(dim: Any) -> dict[str, Any]:
    dim_param = getattr(dim, "dim_param", "")
    dim_value = getattr(dim, "dim_value", 0)
    if dim_param:
        return {"kind": "dynamic", "name": str(dim_param)}
    if dim_value:
        return {"kind": "static", "value": int(dim_value)}
    return {"kind": "dynamic", "name": None}


def _operators(model: Any) -> tuple[str, ...]:
    operators = []
    seen = set()
    for node in getattr(model.graph, "node", ()):
        op_type = str(getattr(node, "op_type", ""))
        domain = str(getattr(node, "domain", ""))
        key = op_type if domain in {"", "ai.onnx"} else f"{domain}:{op_type}"
        if key and key not in seen:
            seen.add(key)
            operators.append(key)
    return tuple(operators)
