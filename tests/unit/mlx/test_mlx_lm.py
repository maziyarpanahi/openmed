"""Tests for OpenMed MLX-LM language-model support."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


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


@pytest.mark.parametrize(
    ("total_tokens", "budget_exceeded", "evictions", "chunk_ranges"),
    [
        (16, False, 0, [(0, 5), (5, 10), (10, 15), (15, 16)]),
        (17, True, 1, [(0, 5), (5, 10), (10, 15), (15, 17)]),
        (9, False, 0, [(0, 5), (5, 9)]),
    ],
)
def test_paged_kv_cache_plan_covers_page_boundaries(
    total_tokens, budget_exceeded, evictions, chunk_ranges
):
    from openmed.mlx.lm import PagedKVCacheConfig

    config = PagedKVCacheConfig(
        memory_budget_bytes=16,
        page_size_tokens=4,
        chunk_size_tokens=5,
        bytes_per_token=1,
    )

    plan = config.plan(total_tokens)

    assert plan.total_pages == 4
    assert plan.resident_window_tokens == 16
    assert plan.budget_exceeded is budget_exceeded
    assert plan.evictions == evictions
    assert [(chunk.start, chunk.end) for chunk in plan.chunk_ranges] == chunk_ranges
    assert plan.stats().memory_budget_bytes == 16


def test_paged_kv_cache_page_table_stores_and_evicts_at_boundaries():
    from openmed.mlx.lm import OpenMedPagedKVCache, PagedKVCacheConfig

    cache = OpenMedPagedKVCache(
        PagedKVCacheConfig(
            memory_budget_bytes=8,
            page_size_tokens=4,
            bytes_per_token=1,
        )
    )

    for token in range(8):
        cache.store("note", token, f"k{token}", f"v{token}")

    assert set(cache.page_table("note")) == {0, 1}
    assert cache.get("note", 3) == ("k3", "v3")
    assert cache.get("note", 4) == ("k4", "v4")
    assert cache.get("note", 7) == ("k7", "v7")

    cache.store("note", 8, "k8", "v8")

    assert cache.get("note", 8) == ("k8", "v8")
    assert cache.stats().total_pages == 2
    assert cache.stats().resident_pages == 2
    assert cache.stats().evictions == 1
    assert cache.resident_token_positions("note") == (4, 5, 6, 7, 8)
    with pytest.raises(KeyError):
        cache.get("note", 0)


def test_paged_kv_cache_plan_keeps_four_x_long_note_under_budget():
    from openmed.mlx.lm import PagedKVCacheConfig

    config = PagedKVCacheConfig(
        memory_budget_bytes=16,
        page_size_tokens=4,
        chunk_size_tokens=4,
        bytes_per_token=1,
    )

    plan = config.plan(config.exact_context_tokens * 4 + 1)
    stats = plan.stats()

    assert plan.budget_exceeded is True
    assert plan.evictions > 0
    assert stats.peak_resident_pages * stats.bytes_per_page <= 16


def test_language_model_generate_can_plan_paged_cache_without_output_drift(
    monkeypatch, tmp_path
):
    from openmed.mlx.lm import OpenMedMLXLanguageModel, PagedKVCacheConfig

    artifact = tmp_path / "laneformer"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")

    calls = []
    fake_model = object()

    class FakeTokenizer:
        def encode(self, text):
            return text.split()

    def fake_load(path):
        calls.append(("load", path))
        return fake_model, FakeTokenizer()

    def fake_generate(model, tokenizer, **kwargs):
        calls.append(("generate", model, tokenizer, kwargs))
        return "stable redaction"

    class FakeMetrics:
        def __init__(self):
            self.records = []

        def record_mlx_paged_kv_cache(self, **kwargs):
            self.records.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=fake_load, generate=fake_generate),
    )

    runner = OpenMedMLXLanguageModel(str(artifact))
    prompt = " ".join(f"tok{i}" for i in range(10))
    dense = runner.generate(prompt, max_tokens=2)

    metrics = FakeMetrics()
    paged = runner.generate(
        prompt,
        max_tokens=2,
        paged_kv_cache=PagedKVCacheConfig(
            memory_budget_bytes=16,
            page_size_tokens=4,
            chunk_size_tokens=3,
            bytes_per_token=1,
        ),
        metrics=metrics,
    )

    assert paged == dense
    generate_kwargs = calls[-1][3]
    assert generate_kwargs["prefill_step_size"] == 3
    assert generate_kwargs["max_kv_size"] == 16
    assert runner.last_paged_kv_cache_plan is not None
    assert runner.last_paged_kv_cache_plan.exact is True
    assert metrics.records == [
        {
            "total_pages": 4,
            "resident_pages": 3,
            "evictions": 0,
            "peak_pages": 3,
            "memory_budget_bytes": 16,
        }
    ]
