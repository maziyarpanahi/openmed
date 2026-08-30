"""Executable tests for Maple's synthetic ONNX Runtime QMoE probe."""

from __future__ import annotations

import importlib.metadata

import pytest

from openmed.onnx.maple_qmoe_smoke import (
    MAPLE_QMOE_SMOKE_BLOCK_SIZE,
    run_maple_qmoe_smoke,
    write_maple_qmoe_smoke_model,
)


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(item) for item in version.split(".")[:3])


def test_writes_fused_four_bit_maple_qmoe_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    path = write_maple_qmoe_smoke_model(tmp_path / "qmoe.onnx")

    model = onnx.load(path)
    node = model.graph.node[0]
    attributes = {
        item.name: onnx.helper.get_attribute_value(item) for item in node.attribute
    }
    assert node.op_type == "QMoE"
    assert node.domain == "com.microsoft"
    assert len(node.input) == 7
    assert node.input[4] == ""
    assert attributes["expert_weight_bits"] == 4
    assert attributes["block_size"] == MAPLE_QMOE_SMOKE_BLOCK_SIZE
    assert attributes["swiglu_fusion"] == 1
    assert attributes["activation_type"] == b"swiglu"

    with pytest.raises(FileExistsError, match="overwrite"):
        write_maple_qmoe_smoke_model(path)


def test_executes_fused_four_bit_qmoe_on_supported_cpu_runtime(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if _version_tuple(importlib.metadata.version("onnxruntime")) < (1, 25, 1):
        pytest.skip("installed ONNX Runtime predates the block-wise QMoE probe")
    path = write_maple_qmoe_smoke_model(tmp_path / "qmoe.onnx")

    result = run_maple_qmoe_smoke(path)

    assert result.output_shape == (1, 1, 64)
    assert result.maximum_absolute_output == 0.0
