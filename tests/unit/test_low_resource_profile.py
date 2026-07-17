"""Tests for the CPU-only low-resource execution profile."""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from openmed.core.backends import (
    HuggingFaceBackend,
    OnnxBackend,
    OnnxTokenClassificationPipeline,
    get_backend,
)
from openmed.core.config import LOW_RESOURCE_PII_MODEL, OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.core.pipeline import Pipeline


class _FakeEntity:
    label = "NAME"
    score = 0.98
    text = "Jordan Example"
    start = 8
    end = 22


class _FakeOnnxModel:
    tokenizer = object()
    variant = "int8"

    def predict(self, text, *, threshold=0.0, max_length=None):
        del text, threshold, max_length
        return [_FakeEntity()]


def test_low_resource_profile_has_bounded_runtime_defaults():
    config = OpenMedConfig.from_profile("low_resource")

    assert config.backend == "onnx"
    assert config.device == "cpu"
    assert config.batch_size == 1
    assert config.num_workers == 1
    assert config.lazy_model_loading is True
    assert config.onnx_variant == "int8"
    assert config.onnx_intra_op_num_threads == 2
    assert config.pii_model == LOW_RESOURCE_PII_MODEL
    assert config.timeout == 900


def test_environment_profile_applies_runtime_defaults(monkeypatch):
    monkeypatch.setenv("OPENMED_PROFILE", "low_resource")

    config = OpenMedConfig()

    assert config.profile == "low_resource"
    assert config.backend == "onnx"
    assert config.onnx_variant == "int8"
    assert config.pii_model == LOW_RESOURCE_PII_MODEL


def test_pipeline_uses_profile_default_pii_model():
    config = OpenMedConfig.from_profile("low_resource")

    pipeline = Pipeline(config=config, model_detector=lambda *args, **kwargs: None)

    assert pipeline.model_name == LOW_RESOURCE_PII_MODEL


def test_backend_detection_honors_profile_without_hf_fallback():
    config = OpenMedConfig.from_profile("low_resource")

    with (
        patch.object(OnnxBackend, "is_available", return_value=True),
        patch.object(HuggingFaceBackend, "is_available") as hf_available,
    ):
        backend = get_backend(config=config)

    assert isinstance(backend, OnnxBackend)
    hf_available.assert_not_called()


def test_missing_onnx_runtime_has_actionable_error():
    config = OpenMedConfig.from_profile("low_resource")

    with patch.object(OnnxBackend, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match=r"openmed\[onnx-runtime\]"):
            get_backend(config=config)


def test_onnx_pipeline_uses_standard_entity_schema():
    pipeline = OnnxTokenClassificationPipeline(_FakeOnnxModel())

    result = pipeline("Patient Jordan Example arrived.", batch_size=1)

    assert result == [
        {
            "entity_group": "NAME",
            "score": 0.98,
            "word": "Jordan Example",
            "start": 8,
            "end": 22,
        }
    ]
    assert pipeline.variant == "int8"


def test_onnx_backend_requests_int8_cpu_session():
    config = OpenMedConfig.from_profile("low_resource")
    fake_ort = ModuleType("onnxruntime")
    fake_ort.SessionOptions = lambda: SimpleNamespace()

    with (
        patch.dict(sys.modules, {"onnxruntime": fake_ort}),
        patch(
            "openmed.onnx.inference.load_onnx_model",
            return_value=_FakeOnnxModel(),
        ) as load_model,
    ):
        pipeline = OnnxBackend(config).create_pipeline(config.pii_model)

    assert pipeline.variant == "int8"
    kwargs = load_model.call_args.kwargs
    assert kwargs["variant"] == "int8"
    assert kwargs["providers"] == ("CPUExecutionProvider",)
    assert kwargs["session_options"].intra_op_num_threads == 2
    assert kwargs["session_options"].inter_op_num_threads == 1


def test_low_resource_backend_selection_never_imports_torch_subprocess():
    script = """
import sys
from unittest.mock import patch
from openmed.core.backends import OnnxBackend, get_backend
from openmed.core.config import load_config_with_profile

config = load_config_with_profile()
with patch.object(OnnxBackend, "is_available", return_value=True):
    backend = get_backend(config=config)
assert isinstance(backend, OnnxBackend)
assert config.onnx_variant == "int8"
assert "torch" not in sys.modules
"""
    env = os.environ.copy()
    env["OPENMED_PROFILE"] = "low_resource"
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_low_resource_cache_release_does_not_import_torch():
    loader = object.__new__(ModelLoader)
    loader.config = OpenMedConfig.from_profile("low_resource")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("low_resource attempted to import torch")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        loader._release_cached_memory()


def test_low_resource_workflow_enforces_cgroup_and_regression_gate():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/low-resource.yml").read_text(encoding="utf-8")

    assert "--memory 4g --memory-swap 4g --cpus 2" in workflow
    assert "OPENMED_PROFILE: low_resource" in workflow
    assert "--require-cgroup-limit-gib 4" in workflow
    assert "--max-peak-rss-mib 2560" in workflow
    assert "--max-regression-percent 10" in workflow
    assert "docs/benchmarks/low-resource-baseline.json" in workflow
