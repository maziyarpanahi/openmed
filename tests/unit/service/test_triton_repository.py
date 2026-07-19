"""Tests for Triton repository export and remote inference parity."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openmed.core.config import OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.onnx import inference as onnx_inference
from openmed.service.backends.remote_inference import (
    KServeV2HttpTransport,
    RemoteInferencePipeline,
    RemoteInferenceSettings,
    TritonGrpcTransport,
)
from openmed.service.backends.triton_repository import (
    TritonModelConfig,
    TritonTensorSpec,
    validate_triton_model_repository,
    write_triton_model_repository,
)

ID2LABEL = {0: "O", 1: "B-PERSON", 2: "E-PERSON"}


class SyntheticTokenizer:
    """Fast-tokenizer fixture with deterministic text offsets."""

    is_fast = True
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __call__(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        assert all(item == "Alice Nguyen" for item in texts)
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["return_tensors"] == "np"
        batch_size = len(texts)
        return {
            "input_ids": np.tile(
                np.array([[101, 11, 12, 102]], dtype=np.int64),
                (batch_size, 1),
            ),
            "attention_mask": np.ones((batch_size, 4), dtype=np.int64),
            "token_type_ids": np.zeros((batch_size, 4), dtype=np.int64),
            "offset_mapping": np.tile(
                np.array([[[0, 0], [0, 5], [6, 12], [0, 0]]], dtype=np.int64),
                (batch_size, 1, 1),
            ),
        }


class SyntheticSession:
    """Local ONNX session fixture returning BIO person logits."""

    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="token_type_ids", type="tensor(int64)"),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, feed):
        assert output_names == ["logits"]
        return [_synthetic_logits(feed["input_ids"].shape[0])]


class SyntheticOrt:
    InferenceSession = SyntheticSession


class SyntheticTransport:
    """Remote transport fixture returning the same logits as local ONNX."""

    def __init__(self) -> None:
        self.inputs = None

    def infer(self, inputs, *, output_name):
        assert output_name == "logits"
        self.inputs = inputs
        return _synthetic_logits(inputs["input_ids"].shape[0])


def _synthetic_logits(batch_size: int) -> np.ndarray:
    one = np.array(
        [
            [9.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return np.tile(one[None, :, :], (batch_size, 1, 1))


def _write_local_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"synthetic")
    (artifact / "config.json").write_text(
        json.dumps({"id2label": ID2LABEL}),
        encoding="utf-8",
    )
    return artifact


def _write_synthetic_onnx(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    inputs = [
        onnx.helper.make_tensor_value_info(
            name,
            onnx.TensorProto.INT64,
            ["batch", "sequence"],
        )
        for name in ("input_ids", "attention_mask", "token_type_ids")
    ]
    output = onnx.helper.make_tensor_value_info(
        "logits",
        onnx.TensorProto.FLOAT,
        ["batch", "sequence", 3],
    )
    value = onnx.helper.make_tensor(
        "synthetic_logits",
        onnx.TensorProto.FLOAT,
        [1, 1, 3],
        [1.0, 0.0, 0.0],
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["logits"], value=value)],
        "synthetic-token-classifier",
        inputs,
        [output],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    onnx.save_model(model, path)


def test_model_config_renders_documented_minimal_schema() -> None:
    config = TritonModelConfig(
        name="openmed_pii",
        max_batch_size=8,
        inputs=(
            TritonTensorSpec("input_ids", "TYPE_INT64", (-1,)),
            TritonTensorSpec("attention_mask", "TYPE_INT64", (-1,)),
        ),
        outputs=(TritonTensorSpec("logits", "TYPE_FP32", (-1, 3)),),
    )

    assert config.to_pbtxt() == (
        'name: "openmed_pii"\n'
        'backend: "onnxruntime"\n'
        "max_batch_size: 8\n"
        "input [\n"
        "  {\n"
        '    name: "input_ids"\n'
        "    data_type: TYPE_INT64\n"
        "    dims: [ -1 ]\n"
        "  },\n"
        "  {\n"
        '    name: "attention_mask"\n'
        "    data_type: TYPE_INT64\n"
        "    dims: [ -1 ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "logits"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ -1, 3 ]\n"
        "  }\n"
        "]\n"
    )


def test_model_config_parses_with_official_triton_protobuf() -> None:
    model_config_pb2 = pytest.importorskip("tritonclient.grpc.model_config_pb2")
    from google.protobuf import text_format

    config = TritonModelConfig(
        name="openmed_pii",
        max_batch_size=8,
        inputs=(
            TritonTensorSpec("input_ids", "TYPE_INT64", (-1,)),
            TritonTensorSpec("attention_mask", "TYPE_INT64", (-1,)),
        ),
        outputs=(TritonTensorSpec("logits", "TYPE_FP32", (-1, 3)),),
    )

    parsed = text_format.Parse(config.to_pbtxt(), model_config_pb2.ModelConfig())

    assert parsed.name == "openmed_pii"
    assert parsed.backend == "onnxruntime"
    assert parsed.max_batch_size == 8
    assert [item.name for item in parsed.input] == [
        "input_ids",
        "attention_mask",
    ]
    assert parsed.output[0].dims == [-1, 3]


def test_repository_layout_and_config_validate_for_synthetic_export(
    tmp_path: Path,
) -> None:
    source = tmp_path / "export" / "model.onnx"
    source.parent.mkdir()
    _write_synthetic_onnx(source)

    result = write_triton_model_repository(
        source,
        tmp_path / "repository",
        model_name="openmed_pii",
        version=1,
        max_batch_size=8,
    )

    assert result.model_path == (
        tmp_path / "repository" / "openmed_pii" / "1" / "model.onnx"
    )
    assert result.model_path.is_file()
    assert result.config.inputs[0] == TritonTensorSpec("input_ids", "TYPE_INT64", (-1,))
    assert result.config.outputs == (TritonTensorSpec("logits", "TYPE_FP32", (-1, 3)),)
    assert (
        validate_triton_model_repository(
            tmp_path / "repository",
            model_name="openmed_pii",
        )
        == result
    )


def test_repository_writer_never_overwrites_a_model_version(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"synthetic")
    config = TritonModelConfig(
        name="openmed_pii",
        max_batch_size=8,
        inputs=(
            TritonTensorSpec("input_ids", "TYPE_INT64", (-1,)),
            TritonTensorSpec("attention_mask", "TYPE_INT64", (-1,)),
        ),
        outputs=(TritonTensorSpec("logits", "TYPE_FP32", (-1, 3)),),
    )
    with (
        patch(
            "openmed.service.backends.triton_repository._inspect_onnx_model",
            return_value=(config, ()),
        ),
        patch(
            "openmed.service.backends.triton_repository.validate_triton_model_repository",
            side_effect=lambda repository, **_: SimpleNamespace(repository=repository),
        ),
    ):
        write_triton_model_repository(
            source,
            tmp_path / "repository",
            model_name="openmed_pii",
        )
        with pytest.raises(FileExistsError, match="already exists"):
            write_triton_model_repository(
                source,
                tmp_path / "repository",
                model_name="openmed_pii",
            )


def test_remote_mock_produces_local_onnx_equivalent_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _write_local_artifact(tmp_path)
    tokenizer = SyntheticTokenizer()
    monkeypatch.setattr(
        onnx_inference,
        "_load_runtime_dependencies",
        lambda: (np, SyntheticOrt, SyntheticTokenizer, None),
    )
    local_model = onnx_inference.OnnxModel(
        artifact,
        tokenizer=tokenizer,
        session=SyntheticSession(),
        variant="fp32",
    )
    transport = SyntheticTransport()
    remote_model = RemoteInferencePipeline(
        RemoteInferenceSettings(
            endpoint="http://triton.example:8000",
            model_name="openmed_pii",
        ),
        artifact,
        tokenizer=tokenizer,
        id2label=ID2LABEL,
        transport=transport,
    )

    local_entities = local_model("Alice Nguyen")
    remote_entities = remote_model("Alice Nguyen")

    assert [
        (item.label, item.start, item.end, item.text) for item in local_entities
    ] == [
        (
            item["entity_group"],
            item["start"],
            item["end"],
            item["word"],
        )
        for item in remote_entities
    ]
    assert set(transport.inputs) == {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }


def test_kserve_http_transport_uses_v2_json_tensor_schema() -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "outputs": [
                    {
                        "name": "logits",
                        "shape": [1, 2, 2],
                        "datatype": "FP32",
                        "data": [0.0, 1.0, 2.0, 3.0],
                    }
                ]
            }

    class Client:
        call = None

        def post(self, url, *, json):
            self.call = (url, json)
            return Response()

    client = Client()
    transport = KServeV2HttpTransport(
        RemoteInferenceSettings(
            endpoint="https://example.test/inference",
            model_name="openmed_pii",
            model_version="1",
        ),
        client=client,
    )

    result = transport.infer(
        {"input_ids": np.array([[101, 102]], dtype=np.int64)},
        output_name="logits",
    )

    assert client.call[0] == (
        "https://example.test/inference/v2/models/openmed_pii/versions/1/infer"
    )
    assert client.call[1] == {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 2],
                "datatype": "INT64",
                "data": [101, 102],
            }
        ],
        "outputs": [{"name": "logits"}],
    }
    assert result.shape == (1, 2, 2)


def test_triton_grpc_transport_reshapes_client_output() -> None:
    from openmed.service.proto.generated import kserve_v2_pb2

    class Rpc:
        call = None

        def __call__(self, request, *, timeout):
            self.call = (request, timeout)
            response = kserve_v2_pb2.ModelInferResponse(
                model_name="openmed_pii",
                model_version="1",
            )
            response.outputs.add(
                name="logits",
                shape=[1, 2, 3],
                datatype="FP32",
            )
            response.raw_output_contents.append(
                np.ones((1, 2, 3), dtype=np.float32).tobytes()
            )
            return response

    rpc = Rpc()
    transport = TritonGrpcTransport(
        RemoteInferenceSettings(
            endpoint="grpcs://triton.example:8001",
            model_name="openmed_pii",
            protocol="grpc",
            model_version="1",
        ),
        rpc=rpc,
    )

    result = transport.infer(
        {"input_ids": np.array([[101, 102]], dtype=np.int64)},
        output_name="logits",
    )

    request, timeout = rpc.call
    assert request.model_name == "openmed_pii"
    assert request.model_version == "1"
    assert request.inputs[0].name == "input_ids"
    assert request.inputs[0].shape == [1, 2]
    assert request.inputs[0].datatype == "INT64"
    assert (
        request.raw_input_contents[0]
        == np.array([[101, 102]], dtype=np.int64).tobytes()
    )
    assert request.outputs[0].name == "logits"
    assert timeout == 30.0
    assert result.shape == (1, 2, 3)


def test_triton_grpc_transport_round_trips_over_standard_kserve_rpc() -> None:
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from openmed.service.proto.generated import kserve_v2_pb2

    seen = {}

    def model_infer(request, context):
        del context
        seen["request"] = request
        response = kserve_v2_pb2.ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
        )
        response.outputs.add(
            name="logits",
            shape=[1, 2, 3],
            datatype="FP32",
        )
        response.raw_output_contents.append(
            np.arange(6, dtype=np.float32).reshape(1, 2, 3).tobytes()
        )
        return response

    handler = grpc.unary_unary_rpc_method_handler(
        model_infer,
        request_deserializer=kserve_v2_pb2.ModelInferRequest.FromString,
        response_serializer=kserve_v2_pb2.ModelInferResponse.SerializeToString,
    )
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "inference.GRPCInferenceService",
                {"ModelInfer": handler},
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    transport = TritonGrpcTransport(
        RemoteInferenceSettings(
            endpoint=f"127.0.0.1:{port}",
            model_name="openmed_pii",
            protocol="grpc",
            model_version="1",
        )
    )
    try:
        result = transport.infer(
            {"input_ids": np.array([[101, 102]], dtype=np.int64)},
            output_name="logits",
        )
    finally:
        transport.close()
        server.stop(0).wait()

    assert seen["request"].model_name == "openmed_pii"
    assert seen["request"].inputs[0].datatype == "INT64"
    assert result.tolist() == [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]


def test_model_loader_selects_remote_backend_from_config() -> None:
    config = OpenMedConfig(
        backend="remote",
        remote_inference_endpoint="http://triton.example:8000",
        remote_inference_model_name="openmed_pii",
        remote_inference_tokenizer="/models/openmed-pii-export",
    )
    sentinel = object()
    with (
        patch("openmed.core.models.HF_AVAILABLE", True),
        patch(
            "openmed.core.backends.RemoteInferenceBackend.is_available",
            return_value=True,
        ),
        patch(
            "openmed.service.backends.remote_inference.create_remote_inference_pipeline",
            return_value=sentinel,
        ) as create_pipeline,
    ):
        result = ModelLoader(config).create_pipeline(
            "OpenMed/openmed-pii",
            aggregation_strategy="simple",
        )

    assert result is sentinel
    create_pipeline.assert_called_once_with(
        "OpenMed/openmed-pii",
        config=config,
        task="token-classification",
        aggregation_strategy="simple",
        use_fast_tokenizer=True,
    )


def test_remote_config_round_trips_through_dict() -> None:
    config = OpenMedConfig.from_dict(
        {
            "backend": "remote",
            "remote_inference_endpoint": "https://kserve.example",
            "remote_inference_protocol": "http",
            "remote_inference_model_name": "openmed_pii",
            "remote_inference_model_version": "2",
            "remote_inference_tokenizer": "/models/openmed-pii-export",
            "remote_inference_timeout_seconds": 12,
            "remote_inference_verify_tls": True,
        }
    )

    assert OpenMedConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
