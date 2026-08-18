"""Triton repository generation and mocked KServe V2 parity tests.

All texts, model tensors, and graph artifacts in this module are synthetic.
No inference server or network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openmed.core.backends import OnnxTokenClassificationPipeline
from openmed.core.config import (
    OpenMedConfig,
    load_config_from_file,
    save_config_to_file,
)
from openmed.core.models import ModelLoader
from openmed.core.offline import OFFLINE_ENV_VAR, OfflineModeError
from openmed.onnx import inference as onnx_inference
from openmed.service.backends.remote_inference import (
    KServeV2HttpTransport,
    RemoteInferencePipeline,
    RemoteInferenceSettings,
    TritonGrpcTransport,
    create_remote_inference_pipeline,
)
from openmed.service.backends.triton_repository import (
    TritonModelConfig,
    TritonTensorSpec,
    validate_triton_model_repository,
    write_triton_model_repository,
)

ID2LABEL = {0: "O", 1: "B-PERSON", 2: "E-PERSON"}
SYNTHETIC_TEXT = "Alice Nguyen"


class SyntheticTokenizer:
    """Fast-tokenizer fixture with deterministic synthetic offsets."""

    is_fast = True
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __call__(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        assert all(item == SYNTHETIC_TEXT for item in texts)
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


class SyntheticTransport:
    """Remote fixture that returns the same logits as local ONNX."""

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
    (artifact / "model.onnx").write_bytes(b"synthetic ONNX fixture")
    (artifact / "config.json").write_text(
        json.dumps({"id2label": ID2LABEL}),
        encoding="utf-8",
    )
    return artifact


def _write_synthetic_onnx(
    path: Path,
    *,
    label_count: int = 3,
    external_data: bool = False,
) -> None:
    onnx = pytest.importorskip("onnx")
    inputs = [
        onnx.helper.make_tensor_value_info(
            name,
            onnx.TensorProto.INT64,
            ["batch", "sequence"],
        )
        for name in ("input_ids", "attention_mask")
    ]
    output = onnx.helper.make_tensor_value_info(
        "logits",
        onnx.TensorProto.FLOAT,
        ["batch", "sequence", label_count],
    )
    axes = onnx.numpy_helper.from_array(
        np.array([2], dtype=np.int64),
        name="axes",
    )
    nodes = [
        onnx.helper.make_node("Cast", ["input_ids"], ["ids_float"], to=1),
        onnx.helper.make_node("Unsqueeze", ["ids_float", "axes"], ["ids_3d"]),
        onnx.helper.make_node(
            "Concat",
            ["ids_3d"] * label_count,
            ["logits"],
            axis=2,
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "synthetic-token-classifier",
        inputs,
        [output],
        initializer=[axes],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    if external_data:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, path)


def _documented_triton_model_config_type():
    """Build the emitted subset of NVIDIA's public ModelConfig schema."""

    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="synthetic_triton_model_config.proto",
        package="triton",
        syntax="proto3",
    )
    data_type = file_descriptor.enum_type.add(name="DataType")
    for number, name in enumerate(("TYPE_INVALID", "TYPE_INT64", "TYPE_FP32")):
        data_type.value.add(name=name, number=number)

    for message_name in ("ModelInput", "ModelOutput"):
        message = file_descriptor.message_type.add(name=message_name)
        message.field.add(
            name="name",
            number=1,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )
        message.field.add(
            name="data_type",
            number=2,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
            type_name=".triton.DataType",
        )
        message.field.add(
            name="dims",
            number=3,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
        )

    model_config = file_descriptor.message_type.add(name="ModelConfig")
    fields = (
        ("name", 1, "string", None, False),
        ("input", 2, "message", ".triton.ModelInput", True),
        ("output", 3, "message", ".triton.ModelOutput", True),
        ("max_batch_size", 4, "int32", None, False),
        ("backend", 17, "string", None, False),
    )
    field_types = {
        "string": descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        "message": descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        "int32": descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
    }
    for name, number, kind, type_name, repeated in fields:
        field = model_config.field.add(
            name=name,
            number=number,
            label=(
                descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
                if repeated
                else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
            ),
            type=field_types[kind],
        )
        if type_name is not None:
            field.type_name = type_name

    descriptor = descriptor_pool.DescriptorPool().Add(file_descriptor)
    return message_factory.GetMessageClass(
        descriptor.message_types_by_name["ModelConfig"]
    )


def test_model_config_matches_documented_triton_schema() -> None:
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

    rendered = config.to_pbtxt()
    parsed = text_format.Parse(rendered, _documented_triton_model_config_type()())

    assert rendered == (
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
    assert parsed.name == "openmed_pii"
    assert parsed.backend == "onnxruntime"
    assert parsed.max_batch_size == 8
    assert [item.name for item in parsed.input] == [
        "input_ids",
        "attention_mask",
    ]
    assert list(parsed.output[0].dims) == [-1, 3]


def test_repository_layout_and_schema_validate_for_synthetic_export(
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
    assert result.config.inputs[0] == TritonTensorSpec(
        "input_ids",
        "TYPE_INT64",
        (-1,),
    )
    assert result.config.outputs == (TritonTensorSpec("logits", "TYPE_FP32", (-1, 3)),)
    assert (
        validate_triton_model_repository(
            tmp_path / "repository",
            model_name="openmed_pii",
        )
        == result
    )


def test_external_data_uses_documented_multifile_onnx_layout(tmp_path: Path) -> None:
    source = tmp_path / "export" / "model.onnx"
    source.parent.mkdir()
    _write_synthetic_onnx(source, external_data=True)

    result = write_triton_model_repository(
        source,
        tmp_path / "repository",
        model_name="openmed_pii",
    )

    assert result.model_path == (
        tmp_path / "repository" / "openmed_pii" / "1" / "model.onnx" / "model.onnx"
    )
    assert (result.model_path.parent / "weights.bin").is_file()
    assert (
        validate_triton_model_repository(
            tmp_path / "repository",
            model_name="openmed_pii",
        ).model_path
        == result.model_path
    )


def test_repository_rejects_external_data_path_escape(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    source = tmp_path / "export" / "model.onnx"
    source.parent.mkdir()
    _write_synthetic_onnx(source, external_data=True)
    model = onnx.load_model(source, load_external_data=False)
    location = next(
        item
        for item in model.graph.initializer[0].external_data
        if item.key == "location"
    )
    location.value = "../escaped.bin"
    source.write_bytes(model.SerializeToString())
    (tmp_path / "escaped.bin").write_bytes(b"synthetic external data")

    with pytest.raises(ValueError, match="Unsafe ONNX external tensor location"):
        write_triton_model_repository(
            source,
            tmp_path / "repository",
            model_name="openmed_pii",
        )
    assert not (tmp_path / "repository").exists()


def test_repository_writer_never_overwrites_or_mixes_schemas(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    _write_synthetic_onnx(source)
    repository = tmp_path / "repository"
    write_triton_model_repository(
        source,
        repository,
        model_name="openmed_pii",
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_triton_model_repository(
            source,
            repository,
            model_name="openmed_pii",
        )

    incompatible = tmp_path / "incompatible.onnx"
    _write_synthetic_onnx(incompatible, label_count=4)
    with pytest.raises(ValueError, match="does not match ONNX schema"):
        write_triton_model_repository(
            incompatible,
            repository,
            model_name="openmed_pii",
            version=2,
        )
    assert not (repository / "openmed_pii" / "2").exists()


def test_repository_writer_rejects_model_directory_symlink(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    _write_synthetic_onnx(source)
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (repository / "openmed_pii").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="must not be symbolic links"):
        write_triton_model_repository(
            source,
            repository,
            model_name="openmed_pii",
        )
    assert list(outside.iterdir()) == []


def test_mocked_remote_response_matches_local_onnx_pipeline_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _write_local_artifact(tmp_path)
    tokenizer = SyntheticTokenizer()
    monkeypatch.setattr(
        onnx_inference,
        "_load_runtime_dependencies",
        lambda: (np, None, None, None),
    )
    local_model = onnx_inference.OnnxModel(
        artifact,
        tokenizer=tokenizer,
        session=SyntheticSession(),
        variant="fp32",
    )
    local_pipeline = OnnxTokenClassificationPipeline(local_model)
    transport = SyntheticTransport()
    remote_pipeline = RemoteInferencePipeline(
        RemoteInferenceSettings(
            endpoint="http://triton.example:8000",
            model_name="openmed_pii",
        ),
        artifact,
        tokenizer=tokenizer,
        id2label=ID2LABEL,
        transport=transport,
    )

    local_entities = local_pipeline(SYNTHETIC_TEXT)
    remote_entities = remote_pipeline(SYNTHETIC_TEXT)

    assert remote_entities == local_entities
    assert [(item["start"], item["end"], item["word"]) for item in remote_entities] == [
        (0, 12, "Alice Nguyen")
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
    assert result.tolist() == [[[0.0, 1.0], [2.0, 3.0]]]


def test_http_errors_expose_status_without_response_body() -> None:
    class Response:
        status_code = 503

        @staticmethod
        def json():
            return {"raw_note": "synthetic-sensitive-value"}

    class Client:
        @staticmethod
        def post(url, *, json):
            del url, json
            return Response()

    transport = KServeV2HttpTransport(
        RemoteInferenceSettings(
            endpoint="https://example.test",
            model_name="openmed_pii",
        ),
        client=Client(),
    )

    with pytest.raises(RuntimeError, match="HTTP status 503") as exc_info:
        transport.infer(
            {"input_ids": np.array([[101]], dtype=np.int64)},
            output_name="logits",
        )
    assert "synthetic-sensitive-value" not in str(exc_info.value)


def test_triton_grpc_transport_uses_wire_compatible_raw_tensors() -> None:
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
    assert (
        kserve_v2_pb2.ModelInferRequest.DESCRIPTOR.fields_by_name[
            "raw_input_contents"
        ].number
        == 7
    )
    assert (
        kserve_v2_pb2.ModelInferResponse.DESCRIPTOR.fields_by_name[
            "raw_output_contents"
        ].number
        == 6
    )


def test_remote_pipeline_rejects_malformed_logits() -> None:
    class MalformedTransport:
        @staticmethod
        def infer(inputs, *, output_name):
            del inputs, output_name
            return np.zeros((1, 4, 2), dtype=np.float32)

    pipeline = RemoteInferencePipeline(
        RemoteInferenceSettings(
            endpoint="http://triton.example:8000",
            model_name="openmed_pii",
        ),
        "synthetic",
        tokenizer=SyntheticTokenizer(),
        id2label=ID2LABEL,
        transport=MalformedTransport(),
    )

    with pytest.raises(RuntimeError, match="label dimension"):
        pipeline(SYNTHETIC_TEXT)


def test_model_loader_selects_remote_backend_from_config() -> None:
    config = OpenMedConfig(
        backend="remote",
        remote_inference_endpoint="http://triton.example:8000",
        remote_inference_model_name="openmed_pii",
        remote_inference_tokenizer="/models/openmed-pii-export",
    )
    sentinel = object()
    with (
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


def test_remote_config_round_trips_and_normalizes_protocol() -> None:
    config = OpenMedConfig.from_dict(
        {
            "backend": "remote",
            "remote_inference_endpoint": "https://kserve.example",
            "remote_inference_protocol": " HTTP ",
            "remote_inference_model_name": "openmed_pii",
            "remote_inference_model_version": "2",
            "remote_inference_tokenizer": "/models/openmed-pii-export",
            "remote_inference_timeout_seconds": "12.5",
            "remote_inference_verify_tls": True,
        }
    )

    assert config.remote_inference_protocol == "http"
    assert config.remote_inference_timeout_seconds == 12.5
    assert OpenMedConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_remote_timeout_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        OpenMedConfig(remote_inference_timeout_seconds=timeout)


def test_remote_config_round_trips_through_toml(tmp_path: Path) -> None:
    config = OpenMedConfig(
        backend="remote",
        remote_inference_endpoint="https://kserve.example",
        remote_inference_model_name="openmed_pii",
        remote_inference_timeout_seconds=12.5,
    )
    config_path = save_config_to_file(config, tmp_path / "openmed.toml")

    loaded = load_config_from_file(config_path)

    assert "remote_inference_timeout_seconds = 12.5" in config_path.read_text(
        encoding="utf-8"
    )
    assert loaded.remote_inference_endpoint == "https://kserve.example"
    assert loaded.remote_inference_model_name == "openmed_pii"
    assert loaded.remote_inference_timeout_seconds == 12.5


def test_remote_backend_is_disabled_by_no_egress_config() -> None:
    config = OpenMedConfig(
        backend="remote",
        local_only=True,
        remote_inference_endpoint="http://127.0.0.1:8000",
        remote_inference_model_name="openmed_pii",
    )

    with pytest.raises(RuntimeError, match="disabled by local_only"):
        create_remote_inference_pipeline("synthetic", config=config)


def test_direct_transport_is_disabled_by_offline_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")

    with pytest.raises(OfflineModeError, match="KServe V2 remote inference"):
        KServeV2HttpTransport(
            RemoteInferenceSettings(
                endpoint="http://127.0.0.1:8000",
                model_name="openmed_pii",
            ),
            client=object(),
        )


def test_triton_extra_contains_clients_but_no_server_distribution() -> None:
    import tomllib

    project_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = pyproject["project"]["optional-dependencies"]["triton"]
    normalized = " ".join(dependencies).lower()

    assert "grpcio" in normalized
    assert "httpx" in normalized
    assert "tritonserver" not in normalized
    assert "tritonclient" not in normalized
    assert "kserve" not in normalized


def test_grpc_tls_cannot_disable_certificate_verification() -> None:
    with pytest.raises(ValueError, match="cannot be disabled"):
        TritonGrpcTransport(
            RemoteInferenceSettings(
                endpoint="grpcs://triton.example:8001",
                model_name="openmed_pii",
                protocol="grpc",
                verify_tls=False,
            )
        )
