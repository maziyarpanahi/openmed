"""Remote token-classification inference over the KServe V2 protocol.

OpenMed keeps tokenization, offset handling, and entity decoding in-process.
Only numeric model tensors cross the explicitly configured network boundary.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import quote, urlsplit, urlunsplit

from openmed.core.offline import is_local_only, raise_offline_error
from openmed.onnx.inference import _decode_entities

_PROTOCOLS = {"http", "grpc"}
_NUMPY_TO_KSERVE = {
    "bool": "BOOL",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "float16": "FP16",
    "float32": "FP32",
    "float64": "FP64",
}
_KSERVE_TO_NUMPY = {value: key for key, value in _NUMPY_TO_KSERVE.items()}


class RemoteInferenceTransport(Protocol):
    """Transport contract used by :class:`RemoteInferencePipeline`."""

    def infer(
        self,
        inputs: Mapping[str, Any],
        *,
        output_name: str,
    ) -> Any:
        """Return one output tensor for a batch of named input tensors."""

        ...


@dataclass(frozen=True)
class RemoteInferenceSettings:
    """Connection and model-routing settings for remote inference.

    Args:
        endpoint: KServe V2 server base URL or gRPC target.
        model_name: Server-side model name.
        protocol: ``"http"`` or ``"grpc"``.
        model_version: Optional server-side model version.
        timeout_seconds: Per-request timeout.
        verify_tls: Verify HTTPS certificates. gRPC TLS always requires
            verification and rejects ``False``.
    """

    endpoint: str
    model_name: str
    protocol: str = "http"
    model_version: str | None = None
    timeout_seconds: float = 30.0
    verify_tls: bool = True

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip() if isinstance(self.endpoint, str) else ""
        if not endpoint:
            raise ValueError("remote_inference_endpoint must not be empty")
        model_name = self.model_name.strip() if isinstance(self.model_name, str) else ""
        if not model_name:
            raise ValueError("remote_inference_model_name must not be empty")
        if not isinstance(self.protocol, str):
            raise TypeError("remote_inference_protocol must be a string")
        protocol = self.protocol.strip().lower()
        if protocol not in _PROTOCOLS:
            raise ValueError("remote_inference_protocol must be 'http' or 'grpc'")
        if isinstance(self.timeout_seconds, bool):
            raise TypeError("remote_inference_timeout_seconds must be a real number")
        timeout_seconds = float(self.timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "remote_inference_timeout_seconds must be positive and finite"
            )
        if not isinstance(self.verify_tls, bool):
            raise TypeError("remote_inference_verify_tls must be a boolean")
        if protocol == "http":
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    "HTTP remote inference endpoints must start with http:// or "
                    "https:// and include a host"
                )
            if parsed.query or parsed.fragment:
                raise ValueError(
                    "HTTP remote inference endpoints must not include a query or "
                    "fragment"
                )
        elif "://" in endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"grpc", "grpcs"}:
                raise ValueError(
                    "gRPC remote inference endpoints must use grpc://, grpcs://, "
                    "or host:port"
                )
            if not parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(
                    "gRPC remote inference endpoints must include a target and "
                    "must not include a query or fragment"
                )

        model_version = self.model_version
        if model_version is not None:
            model_version = str(model_version).strip()
            if not model_version:
                raise ValueError("remote_inference_model_version must not be empty")

        object.__setattr__(self, "endpoint", endpoint.rstrip("/"))
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)

    @classmethod
    def from_config(cls, config: Any) -> "RemoteInferenceSettings":
        """Build settings from an :class:`~openmed.OpenMedConfig`."""

        endpoint = getattr(config, "remote_inference_endpoint", None)
        model_name = getattr(config, "remote_inference_model_name", None)
        if not endpoint:
            raise ValueError(
                "backend='remote' requires remote_inference_endpoint in OpenMedConfig"
            )
        if not model_name:
            raise ValueError(
                "backend='remote' requires remote_inference_model_name in OpenMedConfig"
            )
        return cls(
            endpoint=str(endpoint),
            model_name=str(model_name),
            protocol=str(getattr(config, "remote_inference_protocol", "http")),
            model_version=getattr(config, "remote_inference_model_version", None),
            timeout_seconds=getattr(config, "remote_inference_timeout_seconds", 30.0),
            verify_tls=getattr(config, "remote_inference_verify_tls", True),
        )


class KServeV2HttpTransport:
    """JSON implementation of the KServe V2 HTTP inference endpoint."""

    def __init__(
        self,
        settings: RemoteInferenceSettings,
        *,
        client: Any | None = None,
    ) -> None:
        if settings.protocol != "http":
            raise ValueError("KServeV2HttpTransport requires protocol='http'")
        _ensure_remote_allowed()
        self.settings = settings
        self._owns_client = client is None
        if client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "KServe HTTP inference requires the Triton extra. "
                    "Install with: pip install 'openmed[triton]'"
                ) from exc
            client = httpx.Client(
                timeout=settings.timeout_seconds,
                verify=settings.verify_tls,
            )
        self.client = client

    def infer(
        self,
        inputs: Mapping[str, Any],
        *,
        output_name: str,
    ) -> Any:
        """Send tensor data as KServe V2 JSON and return one named output."""

        np = _load_numpy()
        payload_inputs = []
        for name, value in inputs.items():
            array = np.ascontiguousarray(value)
            payload_inputs.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "datatype": _numpy_datatype(array.dtype),
                    "data": array.reshape(-1).tolist(),
                }
            )
        payload = {
            "inputs": payload_inputs,
            "outputs": [{"name": output_name}],
        }
        response = self.client.post(self._infer_url(), json=payload)
        status_code = int(getattr(response, "status_code", 0))
        if not 200 <= status_code < 300:
            raise RuntimeError(
                f"Remote inference request failed with HTTP status {status_code}"
            )
        try:
            response_payload = response.json()
        except Exception:
            # JSON decoder messages can include response-body fragments. Keep
            # potentially sensitive server content out of the public error.
            raise RuntimeError("Remote inference response is not valid JSON") from None
        if not isinstance(response_payload, Mapping):
            raise RuntimeError("Remote inference response must be a JSON object")
        output = _find_output(response_payload.get("outputs"), output_name)
        shape = _tensor_shape(output.get("shape"))
        data = output.get("data")
        datatype = output.get("datatype")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise RuntimeError("Remote inference output is missing JSON tensor data")
        if datatype not in _KSERVE_TO_NUMPY:
            raise RuntimeError(
                f"Remote inference output uses unsupported datatype {datatype!r}"
            )
        try:
            return np.asarray(data, dtype=_KSERVE_TO_NUMPY[datatype]).reshape(shape)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Remote inference output data does not match its declared shape"
            ) from exc

    def close(self) -> None:
        """Close a transport-owned HTTP client."""

        if not self._owns_client:
            return
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _infer_url(self) -> str:
        parsed = urlsplit(self.settings.endpoint)
        base_path = parsed.path.rstrip("/")
        model_path = f"{base_path}/v2/models/{quote(self.settings.model_name, safe='')}"
        if self.settings.model_version is not None:
            model_path += "/versions/" + quote(
                self.settings.model_version,
                safe="",
            )
        model_path += "/infer"
        return urlunsplit((parsed.scheme, parsed.netloc, model_path, "", ""))


class TritonGrpcTransport:
    """Wire-compatible transport for the KServe V2 gRPC inference API."""

    def __init__(
        self,
        settings: RemoteInferenceSettings,
        *,
        rpc: Any | None = None,
    ) -> None:
        if settings.protocol != "grpc":
            raise ValueError("TritonGrpcTransport requires protocol='grpc'")
        _ensure_remote_allowed()
        self.settings = settings
        from openmed.service.proto.generated import kserve_v2_pb2

        self.messages = kserve_v2_pb2
        self.channel = None
        if rpc is None:
            try:
                import grpc
            except ImportError as exc:
                raise ImportError(
                    "KServe gRPC inference requires the Triton extra. "
                    "Install with: pip install 'openmed[triton]'"
                ) from exc
            target, use_tls = _grpc_target(settings.endpoint)
            if use_tls:
                if not settings.verify_tls:
                    raise ValueError(
                        "gRPC TLS verification cannot be disabled; configure a "
                        "trusted endpoint certificate"
                    )
                self.channel = grpc.secure_channel(
                    target,
                    grpc.ssl_channel_credentials(),
                )
            else:
                self.channel = grpc.insecure_channel(target)
            rpc = self.channel.unary_unary(
                "/inference.GRPCInferenceService/ModelInfer",
                request_serializer=kserve_v2_pb2.ModelInferRequest.SerializeToString,
                response_deserializer=kserve_v2_pb2.ModelInferResponse.FromString,
            )
        self.rpc = rpc

    def infer(
        self,
        inputs: Mapping[str, Any],
        *,
        output_name: str,
    ) -> Any:
        """Send named NumPy tensors through the standard unary gRPC method."""

        np = _load_numpy()
        request = self.messages.ModelInferRequest(
            model_name=self.settings.model_name,
            model_version=self.settings.model_version or "",
        )
        for name, value in inputs.items():
            array = np.ascontiguousarray(value)
            request.inputs.add(
                name=name,
                shape=list(array.shape),
                datatype=_numpy_datatype(array.dtype),
            )
            request.raw_input_contents.append(array.tobytes(order="C"))
        request.outputs.add(name=output_name)

        response = self.rpc(request, timeout=self.settings.timeout_seconds)
        return _grpc_output_array(np, response, output_name)

    def close(self) -> None:
        """Close the underlying gRPC channel when one was created."""

        close = getattr(self.channel, "close", None)
        if callable(close):
            close()


class RemoteInferencePipeline:
    """Pipeline-compatible remote token classifier with local decoding."""

    def __init__(
        self,
        settings: RemoteInferenceSettings,
        tokenizer_source: str | Path,
        *,
        tokenizer: Any | None = None,
        id2label: Mapping[str | int, str] | None = None,
        transport: RemoteInferenceTransport | None = None,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        output_name: str = "logits",
        aggregation_strategy: str | None = "simple",
    ) -> None:
        _ensure_remote_allowed()
        if aggregation_strategy not in {None, "simple"}:
            raise ValueError(
                "The remote inference backend supports aggregation_strategy='simple'"
            )
        self._np = _load_numpy()
        self.settings = settings
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("Remote inference output_name must not be empty")
        self.output_name = output_name
        self.aggregation_strategy = aggregation_strategy
        source = str(tokenizer_source)
        if tokenizer is None or id2label is None:
            loaded_tokenizer, loaded_labels = _load_tokenizer_metadata(
                source,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
            tokenizer = tokenizer or loaded_tokenizer
            id2label = id2label or loaded_labels
        self.tokenizer = tokenizer
        self.id2label = {int(key): str(value) for key, value in id2label.items()}
        expected_label_ids = set(range(len(self.id2label)))
        if not self.id2label or set(self.id2label) != expected_label_ids:
            raise ValueError(
                "Remote inference requires a contiguous, non-empty id2label map"
            )
        self.transport = transport or _create_transport(settings)

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        threshold: float = 0.0,
        max_length: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        """Tokenize locally, infer remotely, and return entity dictionaries."""

        del batch_size, num_workers
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported remote pipeline arguments: {names}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        single_input = isinstance(text, str)
        texts = [text] if single_input else list(text)
        if not texts or any(not isinstance(item, str) or not item for item in texts):
            raise ValueError("text must contain one or more non-empty strings")

        tokenizer_kwargs: dict[str, Any] = {
            "padding": True,
            "return_offsets_mapping": True,
            "return_tensors": "np",
            "truncation": True,
        }
        if max_length is not None:
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            tokenizer_kwargs["max_length"] = max_length
        encoded = dict(self.tokenizer(texts, **tokenizer_kwargs))
        offsets = encoded.pop("offset_mapping", None)
        if offsets is None:
            raise RuntimeError("tokenizer did not return offset_mapping")
        model_input_names = tuple(
            getattr(self.tokenizer, "model_input_names", encoded.keys())
        )
        inputs = {
            name: self._np.asarray(encoded[name])
            for name in model_input_names
            if name in encoded
        }
        if "input_ids" not in inputs or "attention_mask" not in inputs:
            raise RuntimeError(
                "tokenizer must provide input_ids and attention_mask tensors"
            )

        logits = self._np.asarray(
            self.transport.infer(inputs, output_name=self.output_name)
        )
        offset_values = self._np.asarray(offsets)
        if logits.ndim != 3:
            raise RuntimeError(
                "Remote token-classification logits must have rank 3 "
                "[batch, sequence, labels]"
            )
        if offset_values.ndim != 3 or offset_values.shape[-1] != 2:
            raise RuntimeError("Tokenizer offsets must have shape [batch, sequence, 2]")
        if logits.shape[0] != len(texts) or offset_values.shape[0] != len(texts):
            raise RuntimeError("Remote inference batch size does not match input text")
        if logits.shape[1] != offset_values.shape[1]:
            raise RuntimeError(
                "Remote logits sequence length does not match tokenizer offsets"
            )
        if logits.shape[2] != len(self.id2label):
            raise RuntimeError(
                "Remote logits label dimension does not match tokenizer metadata"
            )

        predictions = [
            [
                {
                    "entity_group": entity.label,
                    "score": entity.score,
                    "word": entity.text,
                    "start": entity.start,
                    "end": entity.end,
                }
                for entity in _decode_entities(
                    self._np,
                    logits[index],
                    offset_values[index],
                    self.id2label,
                    input_text,
                    threshold=threshold,
                )
            ]
            for index, input_text in enumerate(texts)
        ]
        return predictions[0] if single_input else predictions

    def close(self) -> None:
        """Close the configured transport when it exposes ``close``."""

        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


def create_remote_inference_pipeline(
    model_name: str,
    *,
    config: Any,
    task: str = "token-classification",
    aggregation_strategy: str | None = "simple",
    use_fast_tokenizer: bool = True,
    **kwargs: Any,
) -> RemoteInferencePipeline:
    """Create a remote pipeline from config without changing pipeline calls.

    Remote execution is deliberately incompatible with ``local_only`` and
    ``OPENMED_OFFLINE`` because either setting promises a no-egress runtime.
    """

    if task not in {"ner", "token-classification"}:
        raise ValueError("The remote inference backend supports token-classification")
    if not use_fast_tokenizer:
        raise ValueError("Remote inference requires a fast tokenizer for offsets")
    if is_local_only(config):
        raise RuntimeError(
            "Remote inference is disabled by local_only or OPENMED_OFFLINE"
        )
    settings = RemoteInferenceSettings.from_config(config)
    tokenizer_source = getattr(
        config,
        "remote_inference_tokenizer",
        None,
    ) or _resolve_tokenizer_source(model_name)
    return RemoteInferencePipeline(
        settings,
        tokenizer_source,
        cache_dir=getattr(config, "cache_dir", None),
        token=getattr(config, "hf_token", None),
        local_files_only=False,
        aggregation_strategy=aggregation_strategy,
        **kwargs,
    )


def remote_inference_dependencies_available(config: Any) -> bool:
    """Return whether dependencies for the configured protocol are importable."""

    if not _modules_available("numpy", "transformers"):
        return False
    protocol = str(getattr(config, "remote_inference_protocol", "http")).lower()
    if protocol == "grpc":
        return _modules_available("grpc", "google.protobuf")
    return _modules_available("httpx")


def _load_tokenizer_metadata(
    source: str,
    *,
    cache_dir: str | Path | None,
    token: str | None,
    local_files_only: bool,
) -> tuple[Any, dict[int, str]]:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Remote inference requires tokenizer support. "
            "Install with: pip install 'openmed[triton]'"
        ) from exc

    load_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
        "token": token,
    }
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **load_kwargs)
    if not getattr(tokenizer, "is_fast", True):
        raise ValueError("Remote inference requires a fast tokenizer")
    model_config = AutoConfig.from_pretrained(source, **load_kwargs)
    id2label = getattr(model_config, "id2label", None)
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("Tokenizer model config must provide a non-empty id2label map")
    return tokenizer, {int(key): str(value) for key, value in id2label.items()}


def _resolve_tokenizer_source(model_name: str) -> str:
    from openmed.core.model_registry import get_model_info

    model_info = get_model_info(model_name)
    return model_info.model_id if model_info is not None else model_name


def _create_transport(settings: RemoteInferenceSettings) -> RemoteInferenceTransport:
    if settings.protocol == "grpc":
        return TritonGrpcTransport(settings)
    return KServeV2HttpTransport(settings)


def _find_output(outputs: Any, output_name: str) -> Mapping[str, Any]:
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise RuntimeError("Remote inference response is missing outputs")
    for output in outputs:
        if isinstance(output, Mapping) and output.get("name") == output_name:
            return output
    raise RuntimeError(f"Remote inference response is missing output {output_name!r}")


def _grpc_output_array(np: Any, response: Any, output_name: str) -> Any:
    outputs = list(response.outputs)
    output_index = next(
        (index for index, item in enumerate(outputs) if item.name == output_name),
        None,
    )
    if output_index is None:
        raise RuntimeError(
            f"Remote inference response is missing output {output_name!r}"
        )
    output = outputs[output_index]
    if output.datatype not in _KSERVE_TO_NUMPY:
        raise RuntimeError(
            f"Remote inference output uses unsupported datatype {output.datatype!r}"
        )
    dimensions = _tensor_shape(output.shape)
    try:
        if len(response.raw_output_contents) == len(outputs):
            return np.frombuffer(
                response.raw_output_contents[output_index],
                dtype=_KSERVE_TO_NUMPY[output.datatype],
            ).reshape(dimensions)

        values = _grpc_typed_contents(output.contents, output.datatype)
        return np.asarray(
            values,
            dtype=_KSERVE_TO_NUMPY[output.datatype],
        ).reshape(dimensions)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Remote inference output data does not match its declared shape"
        ) from exc


def _grpc_typed_contents(contents: Any, datatype: str) -> Any:
    field_by_datatype = {
        "BOOL": "bool_contents",
        "INT8": "int_contents",
        "INT16": "int_contents",
        "INT32": "int_contents",
        "INT64": "int64_contents",
        "UINT8": "uint_contents",
        "UINT16": "uint_contents",
        "UINT32": "uint_contents",
        "UINT64": "uint64_contents",
        "FP32": "fp32_contents",
        "FP64": "fp64_contents",
    }
    try:
        field_name = field_by_datatype[datatype]
    except KeyError as exc:
        raise RuntimeError(
            f"Remote gRPC output {datatype!r} requires raw tensor contents"
        ) from exc
    return getattr(contents, field_name)


def _tensor_shape(shape: Any) -> tuple[int, ...]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise RuntimeError("Remote inference output is missing a valid shape")
    try:
        dimensions = tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Remote inference output has an invalid shape") from exc
    if not dimensions or any(item < 0 for item in dimensions):
        raise RuntimeError("Remote inference output has an invalid shape")
    return dimensions


def _numpy_datatype(dtype: Any) -> str:
    name = str(dtype)
    try:
        return _NUMPY_TO_KSERVE[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported remote tensor dtype: {name}") from exc


def _grpc_target(endpoint: str) -> tuple[str, bool]:
    parsed = urlsplit(endpoint)
    if parsed.scheme in {"grpc", "grpcs"}:
        target = parsed.netloc + parsed.path
        if not target:
            raise ValueError("gRPC remote inference endpoint must include a target")
        return target.rstrip("/"), parsed.scheme == "grpcs"
    target = endpoint.rstrip("/")
    if not target:
        raise ValueError("gRPC remote inference endpoint must include a target")
    return target, False


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Remote inference requires NumPy. Install with: "
            "pip install 'openmed[triton]'"
        ) from exc
    return np


def _modules_available(*names: str) -> bool:
    try:
        return all(importlib.util.find_spec(name) is not None for name in names)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _ensure_remote_allowed() -> None:
    if is_local_only():
        raise_offline_error("KServe V2 remote inference")


__all__ = [
    "KServeV2HttpTransport",
    "RemoteInferencePipeline",
    "RemoteInferenceSettings",
    "RemoteInferenceTransport",
    "TritonGrpcTransport",
    "create_remote_inference_pipeline",
    "remote_inference_dependencies_available",
]
