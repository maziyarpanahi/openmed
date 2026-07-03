"""Tests for OpenMed MLX-LM language-model support."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<eos>": 0, "A": 1, "B": 2, "C": 3, "D": 4}
        self.inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        self.eos_token_id = self.vocab["<eos>"]

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, **_kwargs):
        return [self.vocab[char] for char in text]

    def decode(self, token_ids, **_kwargs):
        return "".join(
            self.inverse_vocab[int(token_id)]
            for token_id in token_ids
            if int(token_id) != self.eos_token_id
        )


class ScriptedCausalModel:
    def __init__(self, next_by_prefix, *, vocab_size=5, default=0):
        self.next_by_prefix = {
            tuple(prefix): int(token_id) for prefix, token_id in next_by_prefix.items()
        }
        self.vocab_size = vocab_size
        self.default = default

    def __call__(self, input_ids):
        token_ids = input_ids[0]
        rows = []
        for index in range(len(token_ids)):
            prefix = tuple(int(token_id) for token_id in token_ids[: index + 1])
            next_token = self.next_by_prefix.get(prefix, self.default)
            logits = [-10.0] * self.vocab_size
            logits[next_token] = 10.0
            rows.append(logits)
        return [rows]


def _artifact(tmp_path: Path, name: str) -> Path:
    artifact = tmp_path / name
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    return artifact


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


def test_resolve_default_draft_model_downloads_separate_artifact(monkeypatch):
    from openmed.mlx.lm import (
        LANEFORMER_DRAFT_MLX_MODEL,
        resolve_mlx_draft_language_model,
    )

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/laneformer-draft"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resolved = resolve_mlx_draft_language_model("laneformer-2b-it")

    assert resolved == "/tmp/laneformer-draft"
    assert calls[0]["repo_id"] == LANEFORMER_DRAFT_MLX_MODEL
    assert calls[0]["repo_type"] == "model"


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


def test_language_model_speculative_greedy_matches_target_with_rollback(
    monkeypatch,
    tmp_path,
):
    from openmed.mlx.lm import OpenMedMLXLanguageModel, SpeculativeDecodeResult

    target_artifact = _artifact(tmp_path, "target")
    draft_artifact = _artifact(tmp_path, "draft")
    tokenizer = FakeTokenizer()
    target_model = ScriptedCausalModel(
        {
            (1,): 2,
            (1, 2): 3,
            (1, 2, 3): 0,
        }
    )
    draft_model = ScriptedCausalModel(
        {
            (1,): 2,
            (1, 2): 3,
            (1, 2, 3): 4,
        }
    )

    def fake_load(path):
        if path == str(target_artifact):
            return target_model, tokenizer
        if path == str(draft_artifact):
            return draft_model, tokenizer
        raise AssertionError(f"unexpected load path: {path}")

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=fake_load))

    runner = OpenMedMLXLanguageModel(
        str(target_artifact),
        draft_model_name=str(draft_artifact),
    )
    result = runner.generate(
        "A",
        max_tokens=4,
        speculative=True,
        max_speculative_tokens=3,
        return_metrics=True,
    )

    assert isinstance(result, SpeculativeDecodeResult)
    assert result.text == "BC"
    assert result.metrics.enabled is True
    assert result.metrics.drafted_tokens == 3
    assert result.metrics.accepted_tokens == 2
    assert result.metrics.rollback_count == 1
    assert result.metrics.target_batches == 1
    assert result.metrics.acceptance_rate == 2 / 3


def test_language_model_speculative_tokenizer_mismatch_falls_back(
    monkeypatch,
    tmp_path,
):
    from openmed.mlx.lm import OpenMedMLXLanguageModel

    target_artifact = _artifact(tmp_path, "target")
    draft_artifact = _artifact(tmp_path, "draft")
    target_tokenizer = FakeTokenizer()
    draft_tokenizer = FakeTokenizer({"<eos>": 0, "A": 1, "B": 3, "C": 2, "D": 4})
    target_model = ScriptedCausalModel({(1,): 2})
    draft_model = ScriptedCausalModel({(1,): 2})
    calls = []

    def fake_load(path):
        if path == str(target_artifact):
            return target_model, target_tokenizer
        if path == str(draft_artifact):
            return draft_model, draft_tokenizer
        raise AssertionError(f"unexpected load path: {path}")

    def fake_generate(*_args, **kwargs):
        calls.append(kwargs)
        return "plain"

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=fake_load, generate=fake_generate),
    )

    runner = OpenMedMLXLanguageModel(
        str(target_artifact),
        draft_model_name=str(draft_artifact),
    )
    result = runner.generate("A", speculative=True, return_metrics=True)

    assert result.text == "plain"
    assert result.metrics.enabled is False
    assert result.metrics.fallback_reason == "tokenizer_mismatch"
    assert calls[0]["prompt"] == "A"


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
