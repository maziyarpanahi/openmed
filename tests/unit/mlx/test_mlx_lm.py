"""Tests for OpenMed MLX-LM language-model support."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_resolve_laneformer_source_downloads_openmed_mlx_repo(monkeypatch):
    from openmed.mlx.lm import resolve_mlx_language_model

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/laneformer-mlx"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resolved = resolve_mlx_language_model("kogai/laneformer-2b-it")

    assert resolved == "/tmp/laneformer-mlx"
    assert calls[0]["repo_id"] == "OpenMed/laneformer-2b-it-q4-mlx"
    assert calls[0]["repo_type"] == "model"
    assert "model*.safetensors" in calls[0]["allow_patterns"]
    assert "laneformer.py" not in calls[0]["allow_patterns"]
    assert "*.py" in calls[0]["allow_patterns"]


def test_resolve_default_downloads_openmed_mlx_repo(monkeypatch):
    from openmed.mlx.lm import LANEFORMER_MLX_MODEL, resolve_mlx_language_model

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/laneformer-mlx"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resolved = resolve_mlx_language_model(LANEFORMER_MLX_MODEL)

    assert resolved == "/tmp/laneformer-mlx"
    assert calls[0]["repo_id"] == "OpenMed/laneformer-2b-it-q4-mlx"


def test_resolve_local_mlx_lm_artifact_does_not_download(tmp_path, monkeypatch):
    from openmed.mlx.lm import resolve_mlx_language_model

    artifact = tmp_path / "laneformer"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")

    def fail_download(**_kwargs):
        raise AssertionError("local artifact should not be downloaded")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_download),
    )

    assert resolve_mlx_language_model(str(artifact)) == str(artifact)


def test_language_model_generate_uses_mlx_lm(monkeypatch, tmp_path):
    from openmed.mlx.lm import OpenMedMLXLanguageModel

    artifact = tmp_path / "laneformer"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")

    calls = []
    fake_model = object()
    fake_tokenizer = object()

    def fake_load(path):
        calls.append(("load", path))
        return fake_model, fake_tokenizer

    def fake_generate(model, tokenizer, **kwargs):
        calls.append(("generate", model, tokenizer, kwargs))
        return "response"

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=fake_load, generate=fake_generate),
    )

    runner = OpenMedMLXLanguageModel(str(artifact))
    result = runner.generate("hello", max_tokens=8, temp=0.0)

    assert result == "response"
    assert calls[0] == ("load", str(artifact))
    assert calls[1][0] == "generate"
    assert calls[1][3]["prompt"] == "hello"
    assert calls[1][3]["max_tokens"] == 8
    assert "temp" not in calls[1][3]
    assert "top_p" not in calls[1][3]


def test_language_model_generate_uses_sampler_for_sampling(monkeypatch, tmp_path):
    from openmed.mlx.lm import OpenMedMLXLanguageModel

    artifact = tmp_path / "laneformer"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")

    calls = []
    fake_model = object()
    fake_tokenizer = object()
    fake_sampler = object()

    fake_mlx_lm = ModuleType("mlx_lm")
    fake_sample_utils = ModuleType("mlx_lm.sample_utils")

    def fake_load(path):
        calls.append(("load", path))
        return fake_model, fake_tokenizer

    def fake_make_sampler(**kwargs):
        calls.append(("make_sampler", kwargs))
        return fake_sampler

    def fake_generate(model, tokenizer, **kwargs):
        calls.append(("generate", model, tokenizer, kwargs))
        return "sampled"

    fake_mlx_lm.load = fake_load
    fake_mlx_lm.generate = fake_generate
    fake_sample_utils.make_sampler = fake_make_sampler

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    runner = OpenMedMLXLanguageModel(str(artifact))
    result = runner.generate("hello", max_tokens=8, temp=0.7, top_p=0.9)

    assert result == "sampled"
    assert calls[1] == ("make_sampler", {"temp": 0.7, "top_p": 0.9})
    assert calls[2][0] == "generate"
    assert calls[2][3]["sampler"] is fake_sampler


def test_top_level_generate_text_is_exported(monkeypatch, tmp_path):
    import openmed

    artifact = tmp_path / "laneformer"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(
            load=lambda _path: (object(), object()),
            generate=lambda *_args, **_kwargs: "ok",
        ),
    )

    assert (
        openmed.generate_text("hello", model_name=str(artifact), max_tokens=1) == "ok"
    )
    assert "generate_text" in openmed.__all__
