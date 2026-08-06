"""TensorRT inference sessions for token-classification engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


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
        self.device = device
        self.trt = trt_module or _tensorrt_api()
        self.torch = torch_module or _torch_api()

        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        if not self.torch.cuda.is_available():
            raise TensorRTSessionError(
                "TensorRT inference requires a CUDA-capable PyTorch runtime"
            )

        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        init_plugins = getattr(self.trt, "init_libnvinfer_plugins", None)
        if init_plugins is not None:
            init_plugins(self.logger, "")

        self.runtime = self.trt.Runtime(self.logger)
        engine_bytes = self.engine_path.read_bytes()
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

        if hasattr(self.engine, "num_io_tensors"):
            outputs = self._run_named_io(inputs)
        else:
            outputs = self._run_legacy_bindings(inputs)
        return _extract_logits(outputs)

    def _run_named_io(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        tensor_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        ]
        input_mode = self.trt.TensorIOMode.INPUT
        input_names = [
            name
            for name in tensor_names
            if self.engine.get_tensor_mode(name) == input_mode
        ]
        output_names = [name for name in tensor_names if name not in input_names]
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

            outputs: dict[str, Any] = {}
            for name in output_names:
                shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                if not shape or any(dim < 0 for dim in shape):
                    raise TensorRTSessionError(
                        f"TensorRT could not resolve output shape for {name}: {shape}"
                    )
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
        binding_names = [
            self.engine.get_binding_name(index)
            for index in range(self.engine.num_bindings)
        ]
        input_names = [
            name
            for index, name in enumerate(binding_names)
            if self.engine.binding_is_input(index)
        ]
        output_names = [name for name in binding_names if name not in input_names]
        bindings = [0] * self.engine.num_bindings
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

            outputs: dict[str, Any] = {}
            for name in output_names:
                index = self.engine.get_binding_index(name)
                shape = tuple(int(dim) for dim in self.context.get_binding_shape(index))
                if not shape or any(dim < 0 for dim in shape):
                    raise TensorRTSessionError(
                        f"TensorRT could not resolve output shape for {name}: {shape}"
                    )
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

        prepared: dict[str, Any] = {}
        for name in required_names:
            dtype = self._numpy_dtype(
                name, named_io=hasattr(self.engine, "num_io_tensors")
            )
            array = _as_contiguous_array(inputs[name], dtype)
            prepared[name] = self.torch.as_tensor(
                array,
                device=self.device,
            ).contiguous()
        return prepared

    def _empty_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        named_io: bool,
    ) -> Any:
        import numpy as np

        numpy_dtype = self._numpy_dtype(name, named_io=named_io)
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


def _as_contiguous_array(value: Any, dtype: Any) -> Any:
    import numpy as np

    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _extract_logits(outputs: Mapping[str, Any]) -> Any:
    if "logits" in outputs:
        return outputs["logits"]
    try:
        return next(iter(outputs.values()))
    except StopIteration as exc:
        raise TensorRTSessionError("TensorRT inference returned no outputs") from exc


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
    "TensorRTSessionError",
    "TensorRTTokenClassificationSession",
]
