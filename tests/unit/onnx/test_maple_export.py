"""Offline tests for the pinned Maple ONNX export path."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.onnx.maple_bundle import MAPLE_SOURCE_MODEL, MAPLE_SOURCE_REVISION
from openmed.onnx.maple_export import (
    MAPLE_MLX_MODEL,
    MAPLE_MLX_REVISION,
    MAPLE_MLX_SOURCE_RECEIPT_FILENAME,
    MAPLE_ONNX_2BIT_QUANTIZATION,
    MAPLE_ONNX_EXPORT_REQUIREMENTS,
    MAPLE_ONNX_GRAPH_FILENAME,
    MAPLE_ONNX_QUANTIZATION,
    MAPLE_ONNX_RECEIPT_FILENAME,
    MAPLE_ONNX_RUNTIME_MIN_VERSION,
    MAPLE_QMOE_BLOCK_SIZE,
    MAPLE_SOURCE_RECEIPT_FILENAME,
    MapleOnnxExportError,
    _ExternalDataShardWriter,
    _make_maple_builder_type,
    _maple_2bit_initializer_contracts,
    _MapleMlxTernarySource,
    _validate_maple_2bit_repack_targets,
    _validate_maple_2bit_replacement,
    build_maple_onnx_export_bundle,
    inspect_maple_mlx_source,
    inspect_maple_onnx_source,
    main,
    maple_onnx_export_requirements,
    plan_maple_onnx_export,
    validate_maple_onnx_graph,
    write_maple_onnx_export_manifest,
)


def _maple_config() -> dict[str, object]:
    return {
        "architectures": ["MapleForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_maple.MapleConfig",
            "AutoModelForCausalLM": "modeling_maple.MapleForCausalLM",
        },
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "vocab_size": 151_936,
        "sliding_window": 512,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 131_072,
        "layer_types": [
            "full_attention" if (index + 1) % 4 == 0 else "sliding_attention"
            for index in range(24)
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_fake_source(root: Path, *, verified: bool = True) -> Path:
    root.mkdir()
    _write_json(root / "config.json", _maple_config())
    for filename in ("configuration_maple.py", "modeling_maple.py", "fa3.py"):
        (root / filename).write_text("# pinned synthetic code\n", encoding="utf-8")
    _write_json(root / "tokenizer.json", {"model": {"type": "BPE"}})
    _write_json(root / "tokenizer_config.json", {"model_max_length": 131_072})
    shard_payloads = {
        "model-00001-of-00002.safetensors": b"first-shard",
        "model-00002-of-00002.safetensors": b"second-shard",
    }
    for filename, payload in shard_payloads.items():
        (root / filename).write_bytes(payload)
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": sum(map(len, shard_payloads.values()))},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
            },
        },
    )
    if verified:
        _write_json(
            root / MAPLE_SOURCE_RECEIPT_FILENAME,
            {
                "schema_version": 1,
                "source_model": MAPLE_SOURCE_MODEL,
                "source_revision": MAPLE_SOURCE_REVISION,
            },
        )
    return root


def _write_fake_mlx_source(root: Path, *, verified: bool = True) -> Path:
    root.mkdir()
    config = _maple_config()
    config.update(
        {
            "model_type": "maple",
            "quantization": {
                "bits": 2,
                "group_size": 128,
                "mode": "affine",
                "lm_head": {"bits": 4, "group_size": 64},
            },
        }
    )
    _write_json(root / "config.json", config)
    (root / "maple.py").write_text("# pinned synthetic runtime\n", encoding="utf-8")
    _write_json(root / "tokenizer.json", {"model": {"type": "BPE"}})
    _write_json(root / "tokenizer_config.json", {"model_max_length": 131_072})
    for index in range(1, 4):
        (root / f"model-{index:05d}-of-00003.safetensors").write_bytes(
            f"shard-{index}".encode()
        )
    if verified:
        _write_json(
            root / MAPLE_MLX_SOURCE_RECEIPT_FILENAME,
            {
                "schema_version": 1,
                "source_model": MAPLE_MLX_MODEL,
                "source_revision": MAPLE_MLX_REVISION,
            },
        )
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inspects_pinned_bf16_source_without_loading_weights(tmp_path):
    source = _write_fake_source(tmp_path / "source")

    inspection = inspect_maple_onnx_source(source)

    assert inspection.source_revision == MAPLE_SOURCE_REVISION
    assert inspection.revision_verified is True
    assert inspection.total_weight_bytes == len(b"first-shardsecond-shard")
    assert inspection.weight_shards == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )


def test_inspects_pinned_lossless_two_bit_mlx_source(tmp_path):
    source = _write_fake_mlx_source(tmp_path / "mlx")

    inspection = inspect_maple_mlx_source(source)

    assert inspection.source_model == MAPLE_MLX_MODEL
    assert inspection.source_revision == MAPLE_MLX_REVISION
    assert inspection.revision_verified is True
    assert inspection.weight_shards == tuple(
        f"model-{index:05d}-of-00003.safetensors" for index in range(1, 4)
    )


def test_rejects_unverified_or_wrongly_quantized_mlx_source(tmp_path):
    unverified = _write_fake_mlx_source(tmp_path / "unverified-mlx", verified=False)
    with pytest.raises(MapleOnnxExportError, match="MLX source revision is unverified"):
        inspect_maple_mlx_source(unverified)

    wrong = _write_fake_mlx_source(tmp_path / "wrong-mlx")
    config = _maple_config()
    config.update({"model_type": "maple", "quantization": {"bits": 4}})
    _write_json(wrong / "config.json", config)
    with pytest.raises(MapleOnnxExportError, match="2-bit affine group-128"):
        inspect_maple_mlx_source(wrong)


def test_rejects_unverified_or_quantized_source(tmp_path):
    unverified = _write_fake_source(tmp_path / "unverified", verified=False)
    with pytest.raises(MapleOnnxExportError, match="revision is unverified"):
        inspect_maple_onnx_source(unverified)
    assert (
        inspect_maple_onnx_source(
            unverified, require_verified_revision=False
        ).revision_verified
        is False
    )

    quantized = _write_fake_source(tmp_path / "quantized")
    config = _maple_config()
    config["quantization"] = {"group_size": 64, "bits": 2}
    _write_json(quantized / "config.json", config)
    with pytest.raises(MapleOnnxExportError, match="BF16 source"):
        inspect_maple_onnx_source(quantized)


def test_rejects_source_index_traversal_and_wrong_size(tmp_path):
    source = _write_fake_source(tmp_path / "source")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["model.norm.weight"] = "../private.safetensors"
    _write_json(index_path, index)
    with pytest.raises(MapleOnnxExportError, match="unsafe artifact path"):
        inspect_maple_onnx_source(source)

    safe_source = _write_fake_source(tmp_path / "wrong-size")
    index_path = safe_source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["total_size"] = 10_000
    _write_json(index_path, index)
    with pytest.raises(MapleOnnxExportError, match="shard sizes"):
        inspect_maple_onnx_source(safe_source)


@pytest.mark.parametrize(
    ("target", "provider", "io_dtype"),
    (("mobile", "cpu", "float32"), ("webgpu", "webgpu", "float16")),
)
def test_plans_unified_four_bit_export(
    tmp_path, target: str, provider: str, io_dtype: str
):
    source = _write_fake_source(tmp_path / "source")

    plan = plan_maple_onnx_export(source, tmp_path / "output", target=target)

    assert plan.execution_provider == provider
    assert plan.io_dtype == io_dtype
    assert plan.graph_filename == MAPLE_ONNX_GRAPH_FILENAME
    assert plan.quantization == MAPLE_ONNX_QUANTIZATION
    assert plan.qmoe_block_size == MAPLE_QMOE_BLOCK_SIZE
    assert plan.runtime_min_version == MAPLE_ONNX_RUNTIME_MIN_VERSION


def test_refuses_to_plan_over_an_existing_export_directory(tmp_path):
    source = _write_fake_source(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        plan_maple_onnx_export(source, output)


def test_exposes_pinned_export_requirements_and_cli(capsys):
    assert maple_onnx_export_requirements() == MAPLE_ONNX_EXPORT_REQUIREMENTS
    assert "accelerate==1.11.0" in MAPLE_ONNX_EXPORT_REQUIREMENTS
    assert "onnxruntime==1.25.1" in MAPLE_ONNX_EXPORT_REQUIREMENTS
    assert "onnxruntime-genai==0.12.0" in MAPLE_ONNX_EXPORT_REQUIREMENTS
    assert "transformers==4.57.1" in MAPLE_ONNX_EXPORT_REQUIREMENTS

    assert main(["requirements"]) == 0
    output = capsys.readouterr().out
    assert "onnxruntime==1.25.1" in output
    assert "torch==2.9.1" in output


def test_maple_builder_adapter_selects_alternating_rope_and_fused_qmoe():
    class DataType:
        FLOAT = 1

    class FakeBase:
        def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, options):
            self.attention_attrs = {
                "q_norm": False,
                "k_norm": False,
                "use_packed_matmul": True,
                "use_rope_in_attn": True,
            }
            self.moe_attrs = {}
            self.window_size = -1
            self.nodes = []
            self.values = {
                "hidden": SimpleNamespace(dtype=DataType.FLOAT, shape=[1, 1, 128])
            }
            self.lowered_layernorms = []

        def _make_simplified_layer_norm(self, name, root, weight, output, dtype, shape):
            self.lowered_layernorms.append((name, root, weight, output, dtype, shape))

        def make_node(
            self,
            op_type,
            inputs,
            outputs,
            *,
            name,
            domain="",
            **kwargs,
        ):
            self.nodes.append(
                SimpleNamespace(
                    domain=domain,
                    inputs=inputs,
                    kwargs=kwargs,
                    name=name,
                    op_type=op_type,
                    outputs=outputs,
                )
            )

    config = SimpleNamespace(
        num_experts=256,
        layer_types=_maple_config()["layer_types"],
        sliding_window=512,
    )
    ir = SimpleNamespace(DataType=DataType)
    adapter_type = _make_maple_builder_type(FakeBase, ir, SimpleNamespace())
    adapter = adapter_type(config, DataType.FLOAT, 4, "cpu", "cache", {})

    assert config.num_local_experts == 256
    assert adapter.model_type == "Qwen2ForCausalLM"
    assert adapter.attention_attrs["q_norm"] is True
    assert adapter.attention_attrs["k_norm"] is True
    assert adapter.attention_attrs["use_packed_matmul"] is False
    assert adapter.moe_attrs["expert_weight_bits"] == 4
    assert adapter.moe_attrs["swiglu_fusion"] == 1
    assert adapter.moe_attrs["swiglu_limit"] == 7.0

    calls: list[tuple[str, int, bool]] = []
    adapter.make_attention_input_proj = lambda *_args, **_kwargs: calls.append(
        ("input", adapter.window_size, adapter.attention_attrs["use_rope_in_attn"])
    )
    adapter.make_maple_attention_qk = lambda *_args, **_kwargs: calls.append(
        ("qk", adapter.window_size, adapter.attention_attrs["use_rope_in_attn"])
    )
    adapter.make_attention_output_proj = lambda *_args, **_kwargs: calls.append(
        ("output", adapter.window_size, adapter.attention_attrs["use_rope_in_attn"])
    )
    adapter.make_attention(0, object(), "root")
    adapter.make_attention(3, object(), "root")

    assert calls[:3] == [
        ("input", 512, True),
        ("qk", 512, True),
        ("output", 512, True),
    ]
    assert calls[3:] == [
        ("input", -1, False),
        ("qk", -1, False),
        ("output", -1, False),
    ]
    assert adapter.window_size == -1
    assert adapter.attention_attrs["use_rope_in_attn"] is True

    adapter.make_node(
        "SimplifiedLayerNormalization",
        inputs=["hidden", "weight"],
        outputs=["normalized"],
        name="/model/layers.0/attn/q_norm/SimplifiedLayerNormalization",
    )
    assert adapter.lowered_layernorms == [
        (
            "/model/layers.0/attn/q_norm/SimplifiedLayerNormalization",
            "hidden",
            "weight",
            "normalized",
            DataType.FLOAT,
            [1, 1, 128],
        )
    ]
    assert adapter.nodes == []


def test_mlx_ternary_codes_map_exactly_to_ort_two_bit_zero_point():
    np = pytest.importorskip("numpy")
    source = object.__new__(_MapleMlxTernarySource)
    source._np = np
    # MLX packs sixteen little-endian codes per uint32. The published model
    # uses only 0/1/2 for -alpha/0/+alpha.
    mlx_codes = np.array([0, 1, 2, 0] * 4, dtype=np.uint32)
    packed = np.array(
        [sum(int(code) << (2 * index) for index, code in enumerate(mlx_codes))],
        dtype=np.uint32,
    )
    tensor = SimpleNamespace(numpy=lambda: packed)

    shifted = source._shift_codes(tensor)
    ort_codes = np.array(
        [(int(shifted[0]) >> (2 * index)) & 0x3 for index in range(4)],
        dtype=np.int8,
    )

    assert ort_codes.tolist() == [1, 2, 3, 1]
    assert (ort_codes - 2).tolist() == [-1, 0, 1, -1]


def test_mlx_ternary_repack_rejects_reserved_code_three():
    np = pytest.importorskip("numpy")
    source = object.__new__(_MapleMlxTernarySource)
    source._np = np
    tensor = SimpleNamespace(numpy=lambda: np.array([3], dtype=np.uint32))

    with pytest.raises(MapleOnnxExportError, match="reserved two-bit code 3"):
        source._shift_codes(tensor)


def test_mlx_ternary_attention_uses_rowwise_two_bit_contract():
    np = pytest.importorskip("numpy")
    source = object.__new__(_MapleMlxTernarySource)
    source._np = np
    packed = np.zeros((2048, 128), dtype=np.uint32)

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

        def float(self):
            return self

    source._tensor_slice = lambda key, *_args: (
        FakeTensor(np.ones((2048,), dtype=np.float32))
        if key.endswith("row_alpha")
        else FakeTensor(packed)
    )

    qweight = io.BytesIO()
    assert source._write_attention(0, "q", "Q4", qweight) == (2048, 1, 512)
    assert len(qweight.getvalue()) == 2048 * 512
    assert qweight.getvalue()[:4] == b"UUUU"

    scales = io.BytesIO()
    assert source._write_attention(0, "q", "scales", scales) == (2048, 1)
    assert len(scales.getvalue()) == 2048 * 2


def _fake_maple_2bit_repack_model():
    contracts = _maple_2bit_initializer_contracts()
    initializers = [
        SimpleNamespace(
            name=name,
            external_data=[SimpleNamespace(key="length", value=str(length))],
        )
        for name, (_dimensions, length) in contracts.items()
    ]
    nodes = []
    for layer_id in range(24):
        nodes.append(
            SimpleNamespace(
                op_type="QMoE",
                name=f"/model/layers.{layer_id}/moe",
                input=[
                    "hidden",
                    "router",
                    f"model.layers.{layer_id}.moe.fc1.qweight",
                    f"model.layers.{layer_id}.moe.fc1.scales",
                    "",
                    f"model.layers.{layer_id}.moe.fc2.qweight",
                    f"model.layers.{layer_id}.moe.fc2.scales",
                ],
            )
        )
        for projection in "qkvo":
            prefix = f"model.layers.{layer_id}.attn.{projection}_proj.MatMul"
            nodes.append(
                SimpleNamespace(
                    op_type="MatMulNBits",
                    name=(f"/model/layers.{layer_id}/attn/{projection}_proj/MatMul_Q4"),
                    input=[
                        "hidden",
                        f"{prefix}.weight_Q4",
                        f"{prefix}.weight_scales",
                    ],
                )
            )
    return SimpleNamespace(graph=SimpleNamespace(node=nodes, initializer=initializers))


def test_two_bit_repack_requires_exact_graph_initializer_names():
    model = _fake_maple_2bit_repack_model()
    contracts = _validate_maple_2bit_repack_targets(model)
    assert len(contracts) == 288

    model.graph.node[0].input[2] += ".renamed"
    with pytest.raises(MapleOnnxExportError, match="initializer inputs changed"):
        _validate_maple_2bit_repack_targets(model)


def test_two_bit_repack_rejects_missing_or_malformed_replacements():
    model = _fake_maple_2bit_repack_model()
    missing = model.graph.initializer.pop()
    with pytest.raises(MapleOnnxExportError, match="initializers are missing"):
        _validate_maple_2bit_repack_targets(model)

    contracts = _maple_2bit_initializer_contracts()
    name = "model.layers.0.attn.q_proj.MatMul.weight_Q4"
    dimensions, length = contracts[name]
    with pytest.raises(MapleOnnxExportError, match="invalid 2-bit replacement"):
        _validate_maple_2bit_replacement(
            name,
            dimensions,
            length - 1,
            contracts,
        )
    assert missing.name in contracts


def test_browser_external_data_writer_bounds_each_sidecar(tmp_path):
    with _ExternalDataShardWriter(tmp_path, maximum_bytes=10) as writer:
        first_name, first, first_offset = writer.reserve(6)
        first.write(b"abcd")
        writer.finish_initializer(first_offset, 4)
        second_name, second, second_offset = writer.reserve(7)
        second.write(b"1234567")
        writer.finish_initializer(second_offset, 7)

    assert first_name == "model.onnx.data.000"
    assert second_name == "model.onnx.data.001"
    assert (tmp_path / first_name).read_bytes() == b"abcd"
    assert (tmp_path / second_name).read_bytes() == b"1234567"
    with _ExternalDataShardWriter(tmp_path / "other", maximum_bytes=10) as writer:
        with pytest.raises(MapleOnnxExportError, match="shard limit"):
            writer.reserve(11)


def _write_fake_completed_export(
    root: Path,
    *,
    target: str = "mobile",
    quantization: str = MAPLE_ONNX_QUANTIZATION,
) -> Path:
    root.mkdir()
    payloads = {
        MAPLE_ONNX_GRAPH_FILENAME: b"synthetic-onnx-graph",
        "model.onnx.data": b"synthetic-external-weights",
        "tokenizer.json": b'{"model":{"type":"BPE"}}',
        "tokenizer_config.json": b'{"model_max_length":131072}',
        "genai_config.json": b'{"model":{"type":"maple"}}',
        "config.json": json.dumps(_maple_config()).encode(),
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    receipt = {
        "schema_version": 1,
        "source_model": MAPLE_SOURCE_MODEL,
        "source_revision": MAPLE_SOURCE_REVISION,
        "target": target,
        "quantization": quantization,
        "graph": {"path": MAPLE_ONNX_GRAPH_FILENAME},
        "files": [
            {
                "path": name,
                "size_bytes": (root / name).stat().st_size,
                "sha256": _sha256(root / name),
            }
            for name in payloads
        ],
    }
    _write_json(root / MAPLE_ONNX_RECEIPT_FILENAME, receipt)
    return root


def test_packages_unified_export_and_integrity_receipt(tmp_path, monkeypatch):
    export = _write_fake_completed_export(tmp_path / "export")
    monkeypatch.setattr(
        "openmed.onnx.maple_export.validate_maple_onnx_graph",
        lambda *_args, **_kwargs: {
            "maple_graph_contract_completed": True,
            "qmoe_expert_weight_bits": 4,
        },
    )

    result = build_maple_onnx_export_bundle(export, tmp_path / "maple.zip")

    assert result.manifest["graphs"]["prefill_path"] == MAPLE_ONNX_GRAPH_FILENAME
    assert result.manifest["graphs"]["decode_path"] == MAPLE_ONNX_GRAPH_FILENAME
    assert result.manifest["quantization"] == "qmoe-4bit-blockwise-128"
    assert result.manifest["runtime"] == "onnxruntime-mobile"
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist().count(MAPLE_ONNX_GRAPH_FILENAME) == 1
        assert "model.onnx.data" in archive.namelist()
        assert MAPLE_ONNX_RECEIPT_FILENAME in archive.namelist()


def test_packages_two_bit_webgpu_export(tmp_path, monkeypatch):
    export = _write_fake_completed_export(
        tmp_path / "export",
        target="webgpu",
        quantization=MAPLE_ONNX_2BIT_QUANTIZATION,
    )
    monkeypatch.setattr(
        "openmed.onnx.maple_export.validate_maple_onnx_graph",
        lambda *_args, **_kwargs: {
            "maple_graph_contract_completed": True,
            "qmoe_expert_weight_bits": 2,
        },
    )

    result = build_maple_onnx_export_bundle(export, tmp_path / "maple-qmoe2.zip")

    assert result.manifest["runtime"] == "onnxruntime-web"
    assert result.manifest["quantization"] == MAPLE_ONNX_2BIT_QUANTIZATION


def test_writes_unpacked_two_bit_web_manifest(tmp_path, monkeypatch):
    export = _write_fake_completed_export(
        tmp_path / "export",
        target="webgpu",
        quantization=MAPLE_ONNX_2BIT_QUANTIZATION,
    )
    monkeypatch.setattr(
        "openmed.onnx.maple_export.validate_maple_onnx_graph",
        lambda *_args, **_kwargs: {"qmoe_expert_weight_bits": 2},
    )

    destination = write_maple_onnx_export_manifest(export)
    manifest = json.loads(destination.read_text(encoding="utf-8"))

    assert destination.name == "maple-bundle.json"
    assert manifest["runtime"] == "onnxruntime-web"
    assert manifest["quantization"] == MAPLE_ONNX_2BIT_QUANTIZATION
    assert {item["path"] for item in manifest["files"]} >= {
        MAPLE_ONNX_GRAPH_FILENAME,
        "model.onnx.data",
        "tokenizer.json",
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        write_maple_onnx_export_manifest(export)


def test_rejects_completed_export_with_tampered_artifact(tmp_path, monkeypatch):
    export = _write_fake_completed_export(tmp_path / "export")
    artifact = export / "model.onnx.data"
    artifact.write_bytes(b"x" * artifact.stat().st_size)
    monkeypatch.setattr(
        "openmed.onnx.maple_export.validate_maple_onnx_graph",
        lambda *_args, **_kwargs: {
            "maple_graph_contract_completed": True,
            "qmoe_expert_weight_bits": 4,
        },
    )

    with pytest.raises(MapleOnnxExportError, match="artifact hash changed"):
        build_maple_onnx_export_bundle(export, tmp_path / "maple.zip")


def _write_structural_maple_graph(
    path: Path,
    *,
    expert_bits: int = 4,
    target: str = "mobile",
    unsupported_normalization: bool = False,
) -> None:
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    io_type = TensorProto.FLOAT16 if target == "webgpu" else TensorProto.FLOAT
    batch = "batch_size"
    sequence = "sequence_length"
    past = "past_sequence_length"
    total = "total_sequence_length"
    inputs = [
        helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, [batch, sequence]
        ),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, [batch, total]
        ),
        helper.make_tensor_value_info(
            "hidden_states", io_type, [batch, sequence, 2048]
        ),
        helper.make_tensor_value_info("query", io_type, [batch, sequence, 2048]),
        helper.make_tensor_value_info("key", io_type, [batch, sequence, 512]),
        helper.make_tensor_value_info("value", io_type, [batch, sequence, 512]),
        helper.make_tensor_value_info(
            "logits_input", io_type, [batch, sequence, 151_936]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", io_type, [batch, sequence, 151_936])
    ]
    nodes = [helper.make_node("Identity", ["logits_input"], ["logits"])]
    initializers = [
        numpy_helper.from_array(
            np.zeros((2048, 256), dtype=np.float32), name="router_weight"
        ),
        numpy_helper.from_array(
            np.array([-1, 256], dtype=np.int64), name="router_shape"
        ),
        numpy_helper.from_array(np.full((1,), 0x88, dtype=np.uint8), name="fc1"),
        numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="fc1_scale"),
        numpy_helper.from_array(np.full((1,), 0x88, dtype=np.uint8), name="fc2"),
        numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="fc2_scale"),
        numpy_helper.from_array(np.array([0], dtype=np.int32), name="seqlens"),
        numpy_helper.from_array(np.array([1], dtype=np.int32), name="total_length"),
    ]
    if unsupported_normalization:
        initializers.append(
            numpy_helper.from_array(
                np.ones((2048,), dtype=np.float32), name="rms_weight"
            )
        )
        nodes.append(
            helper.make_node(
                "SimplifiedLayerNormalization",
                ["hidden_states", "rms_weight"],
                ["unsupported_rms"],
                domain="com.microsoft",
            )
        )
    for layer_id in range(24):
        attention_bits = expert_bits
        attention_block = 128 if attention_bits == 4 else 2048
        for projection, output_size in (
            ("q", 2048),
            ("k", 512),
            ("v", 512),
            ("o", 2048),
        ):
            nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    ["hidden_states", "fc1", "fc1_scale"],
                    [f"attention_projection.{layer_id}.{projection}"],
                    name=(
                        f"/model/layers.{layer_id}/attn/"
                        f"{projection}_proj/MatMul_Q{attention_bits}"
                    ),
                    domain="com.microsoft",
                    K=2048,
                    N=output_size,
                    accuracy_level=4,
                    bits=attention_bits,
                    block_size=attention_block,
                )
            )
        router = f"/model/layers.{layer_id}/moe/router/MatMul"
        router_output = f"router.{layer_id}"
        router_flat = f"router_flat.{layer_id}"
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    ["hidden_states", "router_weight"],
                    [router_output],
                    name=router,
                ),
                helper.make_node(
                    "Reshape", [router_output, "router_shape"], [router_flat]
                ),
                helper.make_node(
                    "QMoE",
                    [
                        "hidden_states",
                        router_flat,
                        "fc1",
                        "fc1_scale",
                        "",
                        "fc2",
                        "fc2_scale",
                    ],
                    [f"moe.{layer_id}"],
                    name=f"/model/layers.{layer_id}/moe",
                    domain="com.microsoft",
                    activation_alpha=1.0,
                    activation_beta=0.0,
                    activation_type="swiglu",
                    block_size=128 if expert_bits == 4 else 0,
                    expert_weight_bits=expert_bits,
                    k=8,
                    normalize_routing_weights=1,
                    swiglu_fusion=1,
                    swiglu_limit=7.0,
                    use_sparse_mixer=0,
                ),
            ]
        )
        past_key = f"past_key_values.{layer_id}.key"
        past_value = f"past_key_values.{layer_id}.value"
        present_key = f"present.{layer_id}.key"
        present_value = f"present.{layer_id}.value"
        inputs.extend(
            [
                helper.make_tensor_value_info(past_key, io_type, [batch, 4, past, 128]),
                helper.make_tensor_value_info(
                    past_value, io_type, [batch, 4, past, 128]
                ),
            ]
        )
        outputs.extend(
            [
                helper.make_tensor_value_info(
                    present_key, io_type, [batch, 4, total, 128]
                ),
                helper.make_tensor_value_info(
                    present_value, io_type, [batch, 4, total, 128]
                ),
            ]
        )
        local = (layer_id + 1) % 4 != 0
        nodes.append(
            helper.make_node(
                "GroupQueryAttention",
                [
                    "query",
                    "key",
                    "value",
                    past_key,
                    past_value,
                    "seqlens",
                    "total_length",
                    "",
                    "",
                    "",
                    "",
                ],
                [f"attention.{layer_id}", present_key, present_value],
                name=f"/model/layers.{layer_id}/attn/GroupQueryAttention",
                domain="com.microsoft",
                do_rotary=int(local),
                kv_num_heads=4,
                local_window_size=512 if local else -1,
                num_heads=16,
                rotary_interleaved=0,
                scale=1 / (128**0.5),
                softcap=0.0,
            )
        )
    graph = helper.make_graph(
        nodes,
        "synthetic_maple_contract",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )


def test_validates_unified_graph_contract_with_external_data(tmp_path):
    graph = tmp_path / MAPLE_ONNX_GRAPH_FILENAME
    _write_structural_maple_graph(graph)

    validation = validate_maple_onnx_graph(graph, target="mobile")

    assert validation["onnx_checker_completed"] is True
    assert validation["qmoe_nodes"] == 24
    assert validation["group_query_attention_nodes"] == 24
    assert validation["external_data_files"] == ["model.onnx.data"]


def test_validates_two_bit_ternary_qmoe_graph_contract(tmp_path):
    graph = tmp_path / MAPLE_ONNX_GRAPH_FILENAME
    _write_structural_maple_graph(graph, expert_bits=2, target="webgpu")

    validation = validate_maple_onnx_graph(graph, target="webgpu")

    assert validation["qmoe_expert_weight_bits"] == 2
    assert validation["qmoe_block_size"] == 0


def test_rejects_nonportable_normalization_contrib_ops(tmp_path):
    graph = tmp_path / MAPLE_ONNX_GRAPH_FILENAME
    _write_structural_maple_graph(graph, unsupported_normalization=True)

    with pytest.raises(MapleOnnxExportError, match="portable ONNX"):
        validate_maple_onnx_graph(graph, target="mobile")
