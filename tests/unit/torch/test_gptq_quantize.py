"""Tests for the GPTQ quantization recipe."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def _gptq_module():
    return importlib.import_module("openmed.torch.quantize_gptq")


def _awq_module():
    return importlib.import_module("openmed.torch.quantize_awq")


def test_awq_and_gptq_use_same_shared_calibration_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    awq = _awq_module()
    gptq = _gptq_module()
    calls: list[int | None] = []

    assert (
        awq.load_quantization_calibration_texts
        is gptq.load_quantization_calibration_texts
    )

    def fake_loader(limit: int | None = None) -> list[str]:
        calls.append(limit)
        return [" shared synthetic calibration "]

    monkeypatch.setattr(awq, "load_quantization_calibration_texts", fake_loader)
    monkeypatch.setattr(gptq, "load_quantization_calibration_texts", fake_loader)

    assert awq._normalize_calibration_texts(None) == ["shared synthetic calibration"]
    assert gptq._normalize_calibration_texts(None) == ["shared synthetic calibration"]
    assert calls == [None, None]


def test_quantize_gptq_missing_autogptq_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _gptq_module()
    monkeypatch.setitem(sys.modules, "auto_gptq", None)

    with pytest.raises(ImportError) as excinfo:
        module.quantize_gptq("OpenMed/test-model", ["synthetic note"], tmp_path)

    message = str(excinfo.value)
    assert "auto-gptq" in message.lower()
    assert "pip install openmed[gptq]" in message


def test_quantize_gptq_runs_recipe_and_writes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _gptq_module()
    calls: dict[str, object] = {}

    class FakeConfig:
        _commit_hash = "abc123"

        def to_dict(self) -> dict[str, object]:
            return {
                "model_type": "llama",
                "task": "text-generation",
                "id2label": {"0": "O", "1": "NAME"},
            }

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["config"] = {"model_name": model_name, "kwargs": kwargs}
            return FakeConfig()

    class FakeTokenizer:
        def __call__(self, text: str, **kwargs):
            tokenized = calls.setdefault("tokenized", [])
            tokenized.append({"text": text, "kwargs": kwargs})
            return {"input_ids": [len(text)], "attention_mask": [1]}

        def save_pretrained(self, output_dir: Path) -> None:
            Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls["tokenizer"] = {"model_name": model_name, "kwargs": kwargs}
            return FakeTokenizer()

    class FakeBaseQuantizeConfig:
        def __init__(self, **kwargs) -> None:
            calls["base_quantize_config"] = kwargs
            self.kwargs = kwargs

    class FakeGptqModel:
        def quantize(self, examples, **kwargs) -> None:
            calls["quantize"] = {"examples": examples, "kwargs": kwargs}

        def save_quantized(self, output_dir: str, **kwargs) -> None:
            calls["save_quantized"] = {"output_dir": output_dir, "kwargs": kwargs}
            Path(output_dir, "model.safetensors").write_bytes(b"gptq")

    class FakeAutoGPTQForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, quantize_config, **kwargs):
            calls["model"] = {
                "model_name": model_name,
                "quantize_config": quantize_config,
                "kwargs": kwargs,
            }
            return FakeGptqModel()

    monkeypatch.setattr(
        module,
        "_require_autogptq",
        lambda: (FakeAutoGPTQForCausalLM, FakeBaseQuantizeConfig),
    )
    monkeypatch.setattr(
        module,
        "_require_transformers",
        lambda: (FakeAutoConfig, FakeAutoTokenizer),
    )

    result = module.quantize_gptq(
        "OpenMed/test-model",
        [" synthetic calibration one ", "synthetic calibration two"],
        tmp_path,
        group_size=64,
        desc_act=True,
        revision="main",
        local_files_only=True,
        max_calib_seq_len=256,
        calib_batch_size=2,
    )

    quant_config = json.loads(result.quant_config_path.read_text(encoding="utf-8"))
    quantize_call = calls["quantize"]
    save_call = calls["save_quantized"]

    assert result.output_dir == tmp_path
    assert result.source_revision == "main"
    assert result.calibration_sample_count == 2
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "tokenizer.json").exists()
    assert calls["base_quantize_config"] == {
        "bits": 4,
        "group_size": 64,
        "desc_act": True,
    }
    assert quantize_call["examples"] == [
        {"input_ids": [25], "attention_mask": [1]},
        {"input_ids": [25], "attention_mask": [1]},
    ]
    assert quantize_call["kwargs"] == {"batch_size": 2}
    assert save_call["kwargs"] == {"use_safetensors": True}
    assert quant_config["format"] == "openmed-gptq"
    assert quant_config["source_model_id"] == "OpenMed/test-model"
    assert quant_config["source_revision"] == "main"
    assert quant_config["bits"] == 4
    assert quant_config["group_size"] == 64
    assert quant_config["desc_act"] is True
    assert quant_config["calibration_sample_count"] == 2
    assert quant_config["label_count"] == 2
    assert quant_config["autogptq_quant_config"] == {
        "bits": 4,
        "desc_act": True,
        "group_size": 64,
    }
    assert "synthetic calibration one" not in result.quant_config_path.read_text(
        encoding="utf-8"
    )


def test_write_quant_config_uses_config_commit_hash(tmp_path: Path) -> None:
    module = _gptq_module()

    class FakeConfig:
        _commit_hash = "resolved-sha"

        def to_dict(self) -> dict[str, str]:
            return {"model_type": "mistral"}

    source_revision = module._resolve_source_revision(
        config=FakeConfig(),
        model_name="OpenMed/test-model",
        explicit_revision=None,
    )

    config_path = module.write_quant_config(
        tmp_path,
        source_model_id="OpenMed/test-model",
        source_revision=source_revision,
        bits=4,
        group_size=128,
        desc_act=False,
        calibration_texts=["one", "two"],
        autogptq_quant_config={
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
        },
        config=FakeConfig(),
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["source_revision"] == "resolved-sha"
    assert data["family"] == "mistral"
    assert data["bits"] == 4
    assert data["group_size"] == 128
    assert data["desc_act"] is False
    assert data["calibration_sample_count"] == 2
