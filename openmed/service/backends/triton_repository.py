"""Build Triton model repositories from exported ONNX token classifiers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRITON_DATA_TYPES = {
    "TYPE_BOOL",
    "TYPE_UINT8",
    "TYPE_UINT16",
    "TYPE_UINT32",
    "TYPE_UINT64",
    "TYPE_INT8",
    "TYPE_INT16",
    "TYPE_INT32",
    "TYPE_INT64",
    "TYPE_FP16",
    "TYPE_FP32",
    "TYPE_FP64",
    "TYPE_BF16",
    "TYPE_STRING",
}
_ONNX_TO_TRITON_DATA_TYPE = {
    "BOOL": "TYPE_BOOL",
    "UINT8": "TYPE_UINT8",
    "UINT16": "TYPE_UINT16",
    "UINT32": "TYPE_UINT32",
    "UINT64": "TYPE_UINT64",
    "INT8": "TYPE_INT8",
    "INT16": "TYPE_INT16",
    "INT32": "TYPE_INT32",
    "INT64": "TYPE_INT64",
    "FLOAT16": "TYPE_FP16",
    "FLOAT": "TYPE_FP32",
    "DOUBLE": "TYPE_FP64",
    "BFLOAT16": "TYPE_BF16",
    "STRING": "TYPE_STRING",
}


@dataclass(frozen=True)
class TritonTensorSpec:
    """One input or output entry in a Triton ``ModelConfig``."""

    name: str
    data_type: str
    dims: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Triton tensor names must not be empty")
        if self.data_type not in _TRITON_DATA_TYPES:
            raise ValueError(f"Unsupported Triton data type: {self.data_type}")
        if not self.dims:
            raise ValueError(f"Triton tensor {self.name!r} must have at least one dim")
        if any(dim == 0 or dim < -1 for dim in self.dims):
            raise ValueError(
                f"Triton tensor {self.name!r} has invalid dims {self.dims!r}"
            )


@dataclass(frozen=True)
class TritonModelConfig:
    """Minimal ONNX Runtime model configuration accepted by Triton."""

    name: str
    max_batch_size: int
    inputs: tuple[TritonTensorSpec, ...]
    outputs: tuple[TritonTensorSpec, ...]
    backend: str = "onnxruntime"

    def __post_init__(self) -> None:
        _validate_model_name(self.name)
        if self.backend != "onnxruntime":
            raise ValueError("ONNX repositories must use the onnxruntime backend")
        if (
            isinstance(self.max_batch_size, bool)
            or not isinstance(self.max_batch_size, int)
            or self.max_batch_size < 0
        ):
            raise ValueError("max_batch_size must be a non-negative integer")
        if not self.inputs:
            raise ValueError("Triton model config requires at least one input")
        if not self.outputs:
            raise ValueError("Triton model config requires at least one output")
        names = [item.name for item in (*self.inputs, *self.outputs)]
        if len(names) != len(set(names)):
            raise ValueError("Triton model input and output names must be unique")

    def to_pbtxt(self) -> str:
        """Render this configuration as deterministic protobuf text."""

        lines = [
            f"name: {json.dumps(self.name)}",
            f"backend: {json.dumps(self.backend)}",
            f"max_batch_size: {self.max_batch_size}",
        ]
        lines.extend(_render_tensor_block("input", self.inputs))
        lines.extend(_render_tensor_block("output", self.outputs))
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TritonRepositoryResult:
    """Paths and schema for one generated Triton model version."""

    repository_dir: Path
    model_dir: Path
    version_dir: Path
    model_path: Path
    config_path: Path
    config: TritonModelConfig


def write_triton_model_repository(
    onnx_path: str | Path,
    repository_dir: str | Path,
    *,
    model_name: str,
    version: int = 1,
    max_batch_size: int = 8,
) -> TritonRepositoryResult:
    """Write one exported ONNX model into a Triton model repository.

    The ONNX graph is inspected before any repository files are written. Dynamic
    graph axes become ``-1`` in ``config.pbtxt`` and the graph's leading batch
    axis is omitted from tensor dims when ``max_batch_size`` is positive.

    Existing model versions are never overwritten. A new version may be added
    only when its derived tensor schema matches the existing ``config.pbtxt``.
    """

    source_path = Path(onnx_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {source_path}")
    _validate_model_name(model_name)
    version_name = _validate_version(version)
    config, external_files = _inspect_onnx_model(
        source_path,
        model_name=model_name,
        max_batch_size=max_batch_size,
    )

    repository = Path(repository_dir).expanduser().resolve()
    model_dir = repository / model_name
    version_dir = model_dir / version_name
    config_path = model_dir / "config.pbtxt"
    rendered_config = config.to_pbtxt()

    if version_dir.exists():
        raise FileExistsError(f"Triton model version already exists: {version_dir}")
    if (
        config_path.exists()
        and config_path.read_text(encoding="utf-8") != rendered_config
    ):
        raise ValueError(
            f"Existing Triton model config does not match ONNX schema: {config_path}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    temporary_version = Path(
        tempfile.mkdtemp(prefix=f".{version_name}.openmed-", dir=model_dir)
    )
    try:
        target_model = temporary_version / "model.onnx"
        shutil.copy2(source_path, target_model)
        for relative_path in external_files:
            source_external = source_path.parent / relative_path
            target_external = temporary_version / relative_path
            target_external.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_external, target_external)
        os.replace(temporary_version, version_dir)
    except Exception:
        shutil.rmtree(temporary_version, ignore_errors=True)
        raise

    if not config_path.exists():
        temporary_config = model_dir / f".{config_path.name}.tmp"
        temporary_config.write_text(rendered_config, encoding="utf-8")
        os.replace(temporary_config, config_path)

    return validate_triton_model_repository(
        repository,
        model_name=model_name,
        version=version,
    )


def validate_triton_model_repository(
    repository_dir: str | Path,
    *,
    model_name: str,
    version: int = 1,
) -> TritonRepositoryResult:
    """Validate a generated layout and config against its copied ONNX graph."""

    repository = Path(repository_dir).expanduser().resolve()
    _validate_model_name(model_name)
    version_name = _validate_version(version)
    model_dir = repository / model_name
    version_dir = model_dir / version_name
    model_path = version_dir / "model.onnx"
    config_path = model_dir / "config.pbtxt"

    if not model_path.is_file():
        raise ValueError(
            f"Triton ONNX layout requires {model_name}/{version_name}/model.onnx"
        )
    if not config_path.is_file():
        raise ValueError(f"Triton model config is missing: {config_path}")

    existing_text = config_path.read_text(encoding="utf-8")
    max_batch_size = _read_max_batch_size(existing_text)
    expected_config, external_files = _inspect_onnx_model(
        model_path,
        model_name=model_name,
        max_batch_size=max_batch_size,
    )
    if existing_text != expected_config.to_pbtxt():
        raise ValueError("config.pbtxt does not match the copied ONNX graph schema")
    for relative_path in external_files:
        if not (version_dir / relative_path).is_file():
            raise ValueError(f"ONNX external tensor data is missing: {relative_path}")

    return TritonRepositoryResult(
        repository_dir=repository,
        model_dir=model_dir,
        version_dir=version_dir,
        model_path=model_path,
        config_path=config_path,
        config=expected_config,
    )


def _inspect_onnx_model(
    model_path: Path,
    *,
    model_name: str,
    max_batch_size: int,
) -> tuple[TritonModelConfig, tuple[Path, ...]]:
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
        raise ValueError("max_batch_size must be a non-negative integer")
    if max_batch_size < 0:
        raise ValueError("max_batch_size must be a non-negative integer")

    onnx, model = _load_onnx_model(model_path)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [
        item for item in model.graph.input if item.name not in initializer_names
    ]
    graph_outputs = list(model.graph.output)
    inputs = tuple(
        _tensor_spec_from_onnx(onnx, value, max_batch_size=max_batch_size)
        for value in graph_inputs
    )
    outputs = tuple(
        _tensor_spec_from_onnx(onnx, value, max_batch_size=max_batch_size)
        for value in graph_outputs
    )

    input_names = {item.name for item in inputs}
    if "input_ids" not in input_names or "attention_mask" not in input_names:
        raise ValueError(
            "ONNX token classifier must expose input_ids and attention_mask inputs"
        )
    if "logits" not in {item.name for item in outputs}:
        raise ValueError("ONNX token classifier must expose a logits output")

    config = TritonModelConfig(
        name=model_name,
        max_batch_size=max_batch_size,
        inputs=inputs,
        outputs=outputs,
    )
    external_files = _external_tensor_files(onnx, model, model_path.parent)
    return config, external_files


def _load_onnx_model(model_path: Path) -> tuple[Any, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "ONNX repository export requires the ONNX extra. "
            "Install with: pip install 'openmed[onnx]'"
        ) from exc

    model = onnx.load_model(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)
    return onnx, model


def _tensor_spec_from_onnx(
    onnx: Any,
    value_info: Any,
    *,
    max_batch_size: int,
) -> TritonTensorSpec:
    tensor_type = value_info.type.tensor_type
    element_name = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    try:
        data_type = _ONNX_TO_TRITON_DATA_TYPE[element_name]
    except KeyError as exc:
        raise ValueError(
            f"ONNX tensor {value_info.name!r} uses unsupported type {element_name}"
        ) from exc

    graph_dims = tuple(_onnx_dimension_value(dim) for dim in tensor_type.shape.dim)
    if max_batch_size > 0:
        if len(graph_dims) < 2:
            raise ValueError(
                f"Batched ONNX tensor {value_info.name!r} must have rank at least 2"
            )
        if graph_dims[0] != -1:
            raise ValueError(
                f"Batched ONNX tensor {value_info.name!r} must have a dynamic "
                "leading batch axis"
            )
        dims = graph_dims[1:]
    else:
        dims = graph_dims
    return TritonTensorSpec(
        name=str(value_info.name),
        data_type=data_type,
        dims=dims,
    )


def _onnx_dimension_value(dimension: Any) -> int:
    has_field = getattr(dimension, "HasField", None)
    if callable(has_field) and has_field("dim_value"):
        value = int(dimension.dim_value)
        return value if value > 0 else -1
    return -1


def _external_tensor_files(
    onnx: Any,
    model: Any,
    source_dir: Path,
) -> tuple[Path, ...]:
    files: set[Path] = set()
    tensors = list(model.graph.initializer)
    tensors.extend(
        value
        for sparse in getattr(model.graph, "sparse_initializer", ())
        for value in (sparse.values, sparse.indices)
    )
    for tensor in tensors:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = next(
            (entry.value for entry in tensor.external_data if entry.key == "location"),
            None,
        )
        if not location:
            raise ValueError("ONNX external tensor data is missing its location")
        relative_path = _safe_external_path(location)
        source_path = source_dir / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"ONNX external tensor data not found: {source_path}"
            )
        files.add(relative_path)
    return tuple(sorted(files, key=lambda item: item.as_posix()))


def _safe_external_path(location: str) -> Path:
    path = Path(location)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe ONNX external tensor location: {location!r}")
    return path


def _render_tensor_block(
    field_name: str,
    tensors: Sequence[TritonTensorSpec],
) -> list[str]:
    lines = [f"{field_name} ["]
    for index, tensor in enumerate(tensors):
        suffix = "," if index < len(tensors) - 1 else ""
        dims = ", ".join(str(dim) for dim in tensor.dims)
        lines.extend(
            [
                "  {",
                f"    name: {json.dumps(tensor.name)}",
                f"    data_type: {tensor.data_type}",
                f"    dims: [ {dims} ]",
                f"  }}{suffix}",
            ]
        )
    lines.append("]")
    return lines


def _read_max_batch_size(config_text: str) -> int:
    match = re.search(r"(?m)^max_batch_size:\s*(\d+)\s*$", config_text)
    if match is None:
        raise ValueError("config.pbtxt is missing max_batch_size")
    return int(match.group(1))


def _validate_model_name(model_name: str) -> None:
    if not isinstance(model_name, str) or not _MODEL_NAME_PATTERN.fullmatch(model_name):
        raise ValueError(
            "model_name must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens"
        )


def _validate_version(version: int) -> str:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")
    return str(version)


__all__ = [
    "TritonModelConfig",
    "TritonRepositoryResult",
    "TritonTensorSpec",
    "validate_triton_model_repository",
    "write_triton_model_repository",
]
