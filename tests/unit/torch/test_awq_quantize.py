"""Tests for the AWQ quantization recipe."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def _awq_module():
    return importlib.import_module("openmed.torch.quantize_awq")


def test_calibration_loader_returns_deterministic_non_empty_samples() -> None:
    calibration = importlib.import_module("openmed.torch.calibration")

    first = calibration.load_awq_calibration_texts(limit=3)
    second = calibration.load_awq_calibration_texts(limit=3)

    assert first == second
    assert len(first) == 3
    assert all(sample.strip() for sample in first)
    assert calibration.calibration_texts_sha256(first) == (
        calibration.calibration_texts_sha256(second)
    )


def test_quantize_awq_missing_autoawq_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _awq_module()
    monkeypatch.setitem(sys.modules, "awq", None)

    with pytest.raises(ImportError) as excinfo:
        module.quantize_awq("OpenMed/test-model", ["synthetic note"], tmp_path)

    message = str(excinfo.value)
    assert "autoawq" in message.lower()
    assert "pip install openmed[awq]" in message


def test_quantize_awq_runs_recipe_and_writes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _awq_module()
    calls: dict[str, object] = {}

    class FakeConfig:
        _commit_hash = "abc123"

        def to_dict(self) -> dict[str, object]:
            return {
                "model_type": "bert",
                "task": "token-classification",
                "id2label": {"0": "O", "1": "NAME"},
            }

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["config"] = {"model_name": model_name, "kwargs": kwargs}
            return FakeConfig()

    class FakeTokenizer:
        def save_pretrained(self, output_dir: Path) -> None:
            Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["tokenizer"] = {"model_name": model_name, "kwargs": kwargs}
            return FakeTokenizer()

    class FakeAwqModel:
        def quantize(self, tokenizer, **kwargs) -> None:
            calls["quantize"] = {"tokenizer": tokenizer, "kwargs": kwargs}

        def save_quantized(self, output_dir: str) -> None:
            calls["save_quantized"] = output_dir
            Path(output_dir, "model.safetensors").write_bytes(b"awq")

    class FakeAutoAWQForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["model"] = {"model_name": model_name, "kwargs": kwargs}
            return FakeAwqModel()

    monkeypatch.setattr(
        module,
        "_require_autoawq",
        lambda: FakeAutoAWQForCausalLM,
    )
    monkeypatch.setattr(
        module,
        "_require_transformers",
        lambda: (FakeAutoConfig, FakeAutoTokenizer),
    )

    result = module.quantize_awq(
        "OpenMed/test-model",
        [" synthetic calibration one ", "synthetic calibration two"],
        tmp_path,
        group_size=64,
        revision="main",
        local_files_only=True,
    )

    quant_config = json.loads(result.quant_config_path.read_text(encoding="utf-8"))
    quantize_kwargs = calls["quantize"]["kwargs"]  # type: ignore[index]

    assert result.output_dir == tmp_path
    assert result.source_revision == "main"
    assert result.calibration_sample_count == 2
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "tokenizer.json").exists()
    assert quantize_kwargs["quant_config"] == {
        "zero_point": True,
        "q_group_size": 64,
        "w_bit": 4,
        "version": "GEMM",
    }
    assert quantize_kwargs["calib_data"] == [
        "synthetic calibration one",
        "synthetic calibration two",
    ]
    assert quantize_kwargs["max_calib_samples"] == 2
    assert quantize_kwargs["max_calib_seq_len"] == 512
    assert quant_config["format"] == "openmed-awq"
    assert quant_config["source_model_id"] == "OpenMed/test-model"
    assert quant_config["source_revision"] == "main"
    assert quant_config["w_bit"] == 4
    assert quant_config["group_size"] == 64
    assert quant_config["q_group_size"] == 64
    assert quant_config["calibration_sample_count"] == 2
    assert quant_config["label_count"] == 2
    assert "synthetic calibration one" not in result.quant_config_path.read_text(
        encoding="utf-8"
    )


def test_write_quant_config_uses_config_commit_hash(tmp_path: Path) -> None:
    module = _awq_module()

    class FakeConfig:
        _commit_hash = "resolved-sha"

        def to_dict(self) -> dict[str, str]:
            return {"model_type": "deberta-v2"}

    source_revision = module._resolve_source_revision(
        config=FakeConfig(),
        model_name="OpenMed/test-model",
        explicit_revision=None,
    )

    config_path = module.write_quant_config(
        tmp_path,
        source_model_id="OpenMed/test-model",
        source_revision=source_revision,
        w_bit=4,
        group_size=128,
        calibration_texts=["one", "two"],
        autoawq_quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        },
        config=FakeConfig(),
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["source_revision"] == "resolved-sha"
    assert data["family"] == "deberta-v2"
    assert data["calibration_sample_count"] == 2
