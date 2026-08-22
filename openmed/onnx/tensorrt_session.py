"""TensorRT inference sessions for token-classification engines."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

MAX_TENSORRT_ENGINE_BYTES = 16 << 30
MAX_TENSORRT_IO_TENSORS = 64
MAX_RUNTIME_BATCH_SIZE = 64
MAX_RUNTIME_SEQUENCE_LENGTH = 16_384
MAX_RUNTIME_INPUT_ELEMENTS = 1_048_576
MAX_RUNTIME_INPUT_BYTES = 64 << 20
MAX_RUNTIME_TOTAL_INPUT_BYTES = 64 << 20
MAX_RUNTIME_OUTPUT_ELEMENTS = 128 * 1_024 * 1_024
MAX_RUNTIME_OUTPUT_BYTES = 1 << 30
MAX_RUNTIME_TOTAL_OUTPUT_ELEMENTS = 128 * 1_024 * 1_024
MAX_RUNTIME_TOTAL_OUTPUT_BYTES = 1 << 30


class TensorRTSessionError(RuntimeError):
    """Raised when a TensorRT engine cannot be loaded or executed safely."""


class TensorRTTokenClassificationSession:
    """Load a trusted TensorRT engine and return token-classification logits.

    TensorRT engines contain executable GPU code and are specific to the build
    platform. Only load engines built locally or obtained from a trusted source.

    Args:
        engine_path: Path to a serialized TensorRT engine.
        device: CUDA device used for input and output buffers.
        trt_module: Optional TensorRT module injection for testing.
        torch_module: Optional PyTorch module injection for testing.

    Raises:
        ImportError: If TensorRT or PyTorch is unavailable.
        TensorRTSessionError: If CUDA or engine deserialization is unavailable.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: str = "cuda",
        trt_module: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.device = _normalize_cuda_device(device)
        self.trt = trt_module if trt_module is not None else _tensorrt_api()
        self.torch = torch_module if torch_module is not None else _torch_api()

        if not self.torch.cuda.is_available():
            raise TensorRTSessionError(
                "TensorRT inference requires a CUDA-capable PyTorch runtime"
            )

        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        init_plugins = getattr(self.trt, "init_libnvinfer_plugins", None)
        if init_plugins is not None and init_plugins(self.logger, "") is False:
            raise TensorRTSessionError("TensorRT plugin initialization failed")

        self.runtime = self.trt.Runtime(self.logger)
        if self.runtime is None:
            raise TensorRTSessionError("TensorRT could not create a runtime")
        engine_bytes = _read_engine_bytes(self.engine_path)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise TensorRTSessionError(
                f"TensorRT could not deserialize engine: {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TensorRTSessionError("TensorRT could not create an execution context")

    def run(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Any | None = None,
    ) -> Any:
        """Run one batch and return the engine's logits as a NumPy array.

        Args:
            input_ids: Rank-two token ID batch.
            attention_mask: Rank-two attention-mask batch.
            token_type_ids: Optional rank-two token-type batch.

        Returns:
            The ``logits`` output, or the first output when the engine does not
            name one ``logits``.

        Raises:
            TensorRTSessionError: If required inputs are missing or execution
                fails.
        """

        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        if _uses_named_io(self.engine):
            outputs = self._run_named_io(inputs)
        else:
            outputs = self._run_legacy_bindings(inputs)
        return _extract_logits(outputs)

    def _run_named_io(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        io_count = _validate_io_count(self.engine.num_io_tensors)
        tensor_names = [self.engine.get_tensor_name(index) for index in range(io_count)]
        _validate_io_names(tensor_names)
        input_mode = self.trt.TensorIOMode.INPUT
        input_names = [
            name
            for name in tensor_names
            if self.engine.get_tensor_mode(name) == input_mode
        ]
        output_names = [name for name in tensor_names if name not in input_names]
        if not input_names or not output_names:
            raise TensorRTSessionError(
                "TensorRT token-classification engine must expose inputs and outputs"
            )
        stream = self.torch.cuda.Stream(device=self.device)
        with self.torch.cuda.stream(stream):
            prepared = self._prepare_inputs(input_names, inputs)
            for name, tensor in prepared.items():
                if self.context.set_input_shape(name, tuple(tensor.shape)) is False:
                    raise TensorRTSessionError(
                        f"TensorRT rejected runtime input shape for {name}: "
                        f"{tuple(tensor.shape)}"
                    )
                if (
                    self.context.set_tensor_address(name, int(tensor.data_ptr()))
                    is False
                ):
                    raise TensorRTSessionError(
                        f"TensorRT rejected the device address for input {name}"
                    )

            output_shapes: dict[str, tuple[int, ...]] = {}
            total_output_elements = 0
            total_output_bytes = 0
            for name in output_names:
                shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                if not shape or any(dim < 0 for dim in shape):
                    raise TensorRTSessionError(
                        f"TensorRT could not resolve output shape for {name}: {shape}"
                    )
                elements, byte_count = _validate_output_allocation(
                    name,
                    shape,
                    _as_numpy_dtype(self._numpy_dtype(name, named_io=True), name=name),
                )
                total_output_elements += elements
                total_output_bytes += byte_count
                output_shapes[name] = shape
            _validate_total_output_allocation(
                total_output_elements,
                total_output_bytes,
            )

            outputs: dict[str, Any] = {}
            for name, shape in output_shapes.items():
                tensor = self._empty_tensor(name, shape, named_io=True)
                outputs[name] = tensor
                if (
                    self.context.set_tensor_address(name, int(tensor.data_ptr()))
                    is False
                ):
                    raise TensorRTSessionError(
                        f"TensorRT rejected the device address for output {name}"
                    )

            succeeded = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not succeeded:
            raise TensorRTSessionError("TensorRT asynchronous execution failed")
        stream.synchronize()
        return {name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()}

    def _run_legacy_bindings(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        binding_count = _validate_io_count(self.engine.num_bindings)
        binding_names = [
            self.engine.get_binding_name(index) for index in range(binding_count)
        ]
        _validate_io_names(binding_names)
        input_names = [
            name
            for index, name in enumerate(binding_names)
            if self.engine.binding_is_input(index)
        ]
        output_names = [name for name in binding_names if name not in input_names]
        if not input_names or not output_names:
            raise TensorRTSessionError(
                "TensorRT token-classification engine must expose inputs and outputs"
            )
        bindings = [0] * binding_count
        stream = self.torch.cuda.Stream(device=self.device)
        with self.torch.cuda.stream(stream):
            prepared = self._prepare_inputs(input_names, inputs)
            for name, tensor in prepared.items():
                index = self.engine.get_binding_index(name)
                if self.context.set_binding_shape(index, tuple(tensor.shape)) is False:
                    raise TensorRTSessionError(
                        f"TensorRT rejected runtime input shape for {name}: "
                        f"{tuple(tensor.shape)}"
                    )
                bindings[index] = int(tensor.data_ptr())

            output_shapes: dict[str, tuple[int, ...]] = {}
            total_output_elements = 0
            total_output_bytes = 0
            for name in output_names:
                index = self.engine.get_binding_index(name)
                shape = tuple(int(dim) for dim in self.context.get_binding_shape(index))
                if not shape or any(dim < 0 for dim in shape):
                    raise TensorRTSessionError(
                        f"TensorRT could not resolve output shape for {name}: {shape}"
                    )
                elements, byte_count = _validate_output_allocation(
                    name,
                    shape,
                    _as_numpy_dtype(self._numpy_dtype(name, named_io=False), name=name),
                )
                total_output_elements += elements
                total_output_bytes += byte_count
                output_shapes[name] = shape
            _validate_total_output_allocation(
                total_output_elements,
                total_output_bytes,
            )

            outputs: dict[str, Any] = {}
            for name, shape in output_shapes.items():
                index = self.engine.get_binding_index(name)
                tensor = self._empty_tensor(name, shape, named_io=False)
                outputs[name] = tensor
                bindings[index] = int(tensor.data_ptr())

            succeeded = self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=stream.cuda_stream,
            )
        if not succeeded:
            raise TensorRTSessionError("TensorRT asynchronous execution failed")
        stream.synchronize()
        return {name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()}

    def _prepare_inputs(
        self,
        required_names: list[str],
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        missing = [name for name in required_names if name not in inputs]
        if missing:
            raise TensorRTSessionError(
                "TensorRT engine requires missing input(s): " + ", ".join(missing)
            )

        arrays: dict[str, Any] = {}
        expected_shape: tuple[int, int] | None = None
        for name in required_names:
            dtype = self._numpy_dtype(
                name,
                named_io=_uses_named_io(self.engine),
            )
            array = _as_contiguous_array(inputs[name], dtype, name=name)
            if expected_shape is None:
                expected_shape = tuple(array.shape)
            elif tuple(array.shape) != expected_shape:
                raise TensorRTSessionError(
                    "TensorRT token inputs must share the same batch and sequence shape"
                )
            arrays[name] = array

        if sum(int(array.nbytes) for array in arrays.values()) > (
            MAX_RUNTIME_TOTAL_INPUT_BYTES
        ):
            raise TensorRTSessionError(
                "TensorRT inputs exceed the aggregate runtime byte limit"
            )
        return {
            name: self.torch.as_tensor(array, device=self.device).contiguous()
            for name, array in arrays.items()
        }

    def _empty_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        named_io: bool,
    ) -> Any:
        import numpy as np

        numpy_dtype = _as_numpy_dtype(
            self._numpy_dtype(name, named_io=named_io),
            name=name,
        )
        _validate_output_allocation(name, shape, numpy_dtype)
        torch_dtype = self.torch.from_numpy(np.empty((), dtype=numpy_dtype)).dtype
        return self.torch.empty(
            shape,
            dtype=torch_dtype,
            device=self.device,
        )

    def _numpy_dtype(self, name: str, *, named_io: bool) -> Any:
        if named_io:
            dtype = self.engine.get_tensor_dtype(name)
        else:
            dtype = self.engine.get_binding_dtype(self.engine.get_binding_index(name))
        return self.trt.nptype(dtype)


def _as_contiguous_array(value: Any, dtype: Any, *, name: str) -> Any:
    import numpy as np

    raw = np.asarray(value)
    if raw.ndim != 2:
        raise TensorRTSessionError(
            f"TensorRT input {name!r} must have rank two; received {raw.shape}"
        )
    batch_size, sequence_length = (int(dimension) for dimension in raw.shape)
    if not 1 <= batch_size <= MAX_RUNTIME_BATCH_SIZE:
        raise TensorRTSessionError(
            f"TensorRT input {name!r} exceeds the batch-size limit"
        )
    if not 1 <= sequence_length <= MAX_RUNTIME_SEQUENCE_LENGTH:
        raise TensorRTSessionError(
            f"TensorRT input {name!r} exceeds the sequence-length limit"
        )
    if raw.size > MAX_RUNTIME_INPUT_ELEMENTS:
        raise TensorRTSessionError(f"TensorRT input {name!r} exceeds the element limit")
    if raw.dtype.kind not in "biu":
        raise TensorRTSessionError(
            f"TensorRT token input {name!r} must contain integer or boolean values"
        )
    target_dtype = _as_numpy_dtype(dtype, name=name)
    if target_dtype.kind not in "biu":
        raise TensorRTSessionError(
            f"TensorRT token input {name!r} has an unsupported engine dtype"
        )
    minimum = int(raw.min())
    maximum = int(raw.max())
    if name == "attention_mask" and (minimum < 0 or maximum > 1):
        raise TensorRTSessionError(
            "TensorRT attention_mask must contain only zero or one"
        )
    if name in {"input_ids", "token_type_ids"} and minimum < 0:
        raise TensorRTSessionError(
            f"TensorRT token input {name!r} must not contain negative values"
        )
    if target_dtype.kind == "b":
        if minimum < 0 or maximum > 1:
            raise TensorRTSessionError(
                f"TensorRT input {name!r} cannot be represented by the engine dtype"
            )
    else:
        limits = np.iinfo(target_dtype)
        if minimum < int(limits.min) or maximum > int(limits.max):
            raise TensorRTSessionError(
                f"TensorRT input {name!r} cannot be represented by the engine dtype"
            )
    array = np.ascontiguousarray(raw, dtype=target_dtype)
    if array.nbytes > MAX_RUNTIME_INPUT_BYTES:
        raise TensorRTSessionError(f"TensorRT input {name!r} exceeds the byte limit")
    return array


def _extract_logits(outputs: Mapping[str, Any]) -> Any:
    if "logits" in outputs:
        logits = outputs["logits"]
    else:
        try:
            logits = next(iter(outputs.values()))
        except StopIteration as exc:
            raise TensorRTSessionError(
                "TensorRT inference returned no outputs"
            ) from exc
    return _validate_logits(logits)


def _validate_logits(logits: Any) -> Any:
    import numpy as np

    array = np.asarray(logits)
    if array.ndim != 3 or not array.size:
        raise TensorRTSessionError(
            "TensorRT token-classification logits must be a non-empty rank-three array"
        )
    if (
        array.size > MAX_RUNTIME_OUTPUT_ELEMENTS
        or array.nbytes > MAX_RUNTIME_OUTPUT_BYTES
    ):
        raise TensorRTSessionError("TensorRT logits exceed the runtime output limit")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise TensorRTSessionError("TensorRT logits must contain finite numeric values")
    return logits


def _uses_named_io(engine: Any) -> bool:
    return getattr(engine, "num_io_tensors", None) is not None


def _validate_io_count(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_TENSORRT_IO_TENSORS:
        raise TensorRTSessionError(
            "TensorRT engine has an invalid or excessive I/O tensor count"
        )
    return value


def _validate_io_names(names: list[Any]) -> None:
    normalized: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 256:
            raise TensorRTSessionError("TensorRT engine contains an invalid I/O name")
        if any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise TensorRTSessionError("TensorRT engine contains an invalid I/O name")
        normalized.append(name)
    if len(set(normalized)) != len(normalized):
        raise TensorRTSessionError("TensorRT engine contains duplicate I/O names")


def _as_numpy_dtype(dtype: Any, *, name: str) -> Any:
    import numpy as np

    try:
        normalized = np.dtype(dtype)
    except TypeError as exc:
        raise TensorRTSessionError(
            f"TensorRT tensor {name!r} has an unsupported dtype"
        ) from exc
    if normalized.kind not in "biuf":
        raise TensorRTSessionError(f"TensorRT tensor {name!r} has an unsupported dtype")
    return normalized


def _validate_output_allocation(
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> tuple[int, int]:
    if not 1 <= len(shape) <= 4 or any(
        type(dimension) is not int or dimension < 1 for dimension in shape
    ):
        raise TensorRTSessionError(
            f"TensorRT returned an invalid output shape for {name!r}: {shape}"
        )
    elements = math.prod(shape)
    if elements > MAX_RUNTIME_OUTPUT_ELEMENTS:
        raise TensorRTSessionError(
            f"TensorRT output {name!r} exceeds the element limit"
        )
    byte_count = elements * int(dtype.itemsize)
    if byte_count > MAX_RUNTIME_OUTPUT_BYTES:
        raise TensorRTSessionError(f"TensorRT output {name!r} exceeds the byte limit")
    return elements, byte_count


def _validate_total_output_allocation(elements: int, byte_count: int) -> None:
    if (
        elements > MAX_RUNTIME_TOTAL_OUTPUT_ELEMENTS
        or byte_count > MAX_RUNTIME_TOTAL_OUTPUT_BYTES
    ):
        raise TensorRTSessionError(
            "TensorRT outputs exceed the aggregate runtime allocation limit"
        )


def _read_engine_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {path}")
    size = path.stat().st_size
    if not 1 <= size <= MAX_TENSORRT_ENGINE_BYTES:
        raise TensorRTSessionError(
            "TensorRT engine is empty or exceeds the supported size limit"
        )
    with path.open("rb") as handle:
        payload = handle.read(MAX_TENSORRT_ENGINE_BYTES + 1)
    if len(payload) != size or len(payload) > MAX_TENSORRT_ENGINE_BYTES:
        raise TensorRTSessionError("TensorRT engine changed while it was being read")
    return payload


def _normalize_cuda_device(device: Any) -> str:
    if not isinstance(device, str):
        raise TensorRTSessionError("TensorRT device must be a CUDA device string")
    normalized = device.strip().lower()
    if re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]{0,3}))?", normalized) is None:
        raise TensorRTSessionError(
            "TensorRT device must be 'cuda' or 'cuda:<non-negative index>'"
        )
    return normalized


def _tensorrt_api() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT is required for engine inference. Install TensorRT for the "
            "target NVIDIA platform."
        ) from exc
    return trt


def _torch_api() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch with CUDA support is required for TensorRT buffer management"
        ) from exc
    return torch


__all__ = [
    "MAX_RUNTIME_BATCH_SIZE",
    "MAX_RUNTIME_INPUT_BYTES",
    "MAX_RUNTIME_INPUT_ELEMENTS",
    "MAX_RUNTIME_OUTPUT_BYTES",
    "MAX_RUNTIME_OUTPUT_ELEMENTS",
    "MAX_RUNTIME_SEQUENCE_LENGTH",
    "MAX_RUNTIME_TOTAL_INPUT_BYTES",
    "MAX_RUNTIME_TOTAL_OUTPUT_BYTES",
    "MAX_RUNTIME_TOTAL_OUTPUT_ELEMENTS",
    "MAX_TENSORRT_ENGINE_BYTES",
    "MAX_TENSORRT_IO_TENSORS",
    "TensorRTSessionError",
    "TensorRTTokenClassificationSession",
]
