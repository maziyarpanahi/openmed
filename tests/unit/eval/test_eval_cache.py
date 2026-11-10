"""Unit tests for eval benchmark report caching."""

from __future__ import annotations

import json
import re

import pytest

from openmed.eval.cache import (
    build_report_key,
    cache_path,
    clear,
    eval_code_hash,
    hash_fixture_set,
    invalidate,
    load,
    load_or_compute,
    store,
)
from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.report import BenchmarkReport


def test_load_or_compute_reuses_cached_report_without_recomputing(tmp_path):
    report_key = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash="fixtures-v1",
        code_hash="code-v1",
    )
    calls = 0

    def compute() -> BenchmarkReport:
        nonlocal calls
        calls += 1
        return _report(marker="fresh")

    first = load_or_compute(report_key, compute, cache_dir=tmp_path)
    second = load_or_compute(
        report_key,
        lambda: pytest.fail("compute_fn should not run on a cache hit"),
        cache_dir=tmp_path,
    )

    assert calls == 1
    assert second.to_dict() == first.to_dict()


def test_fixture_or_code_hash_change_uses_distinct_cache_entry(tmp_path):
    calls: list[str] = []

    def compute(marker: str) -> BenchmarkReport:
        calls.append(marker)
        return _report(marker=marker)

    base_key = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash="fixtures-v1",
        code_hash="code-v1",
    )
    fixture_changed = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash="fixtures-v2",
        code_hash="code-v1",
    )
    code_changed = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash="fixtures-v1",
        code_hash="code-v2",
    )

    assert (
        load_or_compute(
            base_key,
            lambda: compute("base"),
            cache_dir=tmp_path,
        ).metrics["marker"]
        == "base"
    )
    assert (
        load_or_compute(
            base_key,
            lambda: pytest.fail("same key should hit the cache"),
            cache_dir=tmp_path,
        ).metrics["marker"]
        == "base"
    )
    assert (
        load_or_compute(
            fixture_changed,
            lambda: compute("fixture"),
            cache_dir=tmp_path,
        ).metrics["marker"]
        == "fixture"
    )
    assert (
        load_or_compute(
            code_changed,
            lambda: compute("code"),
            cache_dir=tmp_path,
        ).metrics["marker"]
        == "code"
    )

    assert calls == ["base", "fixture", "code"]


def test_cached_payload_round_trips_through_benchmark_report(tmp_path):
    report_key = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash="fixtures-v1",
        code_hash="code-v1",
    )
    report = _report(marker="round-trip")

    store(report_key, report, cache_dir=tmp_path)
    path = cache_path(report_key, cache_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == report.to_dict()
    assert load(report_key, cache_dir=tmp_path).to_dict() == report.to_dict()
    assert BenchmarkReport.from_dict(payload).to_dict() == report.to_dict()


def test_cache_entry_contains_report_json_not_fixture_text(tmp_path):
    fixture_text = "Synthetic patient Alpha has identifier ID-00042."
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "synthetic-note",
            "text": fixture_text,
            "gold_spans": [{"start": 18, "end": 23, "label": "PERSON"}],
        }
    )
    report_key = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash=hash_fixture_set([fixture]),
        code_hash="code-v1",
    )
    report = _report(marker="privacy")

    path = store(report_key, report, cache_dir=tmp_path)
    raw_payload = path.read_text(encoding="utf-8")

    assert fixture_text not in raw_payload
    assert json.loads(raw_payload) == report.to_dict()


def test_invalidate_and_clear_remove_cache_entries(tmp_path):
    first_key = build_report_key(
        model_name="privacy-filter",
        suite="golden",
        fixture_set_hash="fixtures-v1",
        code_hash="code-v1",
    )
    second_key = build_report_key(
        model_name="privacy-filter",
        suite="shield",
        fixture_set_hash="fixtures-v1",
        code_hash="code-v1",
    )
    store(first_key, _report(marker="first"), cache_dir=tmp_path)
    store(second_key, _report(marker="second"), cache_dir=tmp_path)

    assert invalidate(first_key, cache_dir=tmp_path) is True
    assert invalidate(first_key, cache_dir=tmp_path) is False
    assert load(first_key, cache_dir=tmp_path) is None
    assert clear(cache_dir=tmp_path) == 1
    assert load(second_key, cache_dir=tmp_path) is None


def test_run_benchmark_cache_is_opt_in(tmp_path):
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John",
            "gold_spans": [{"start": 8, "end": 12, "label": "PERSON"}],
        }
    )
    calls = 0

    def runner(fixture, model_name, device):
        nonlocal calls
        calls += 1
        return [{"start": 8, "end": 12, "label": "PERSON"}]

    run_benchmark([fixture], suite="golden", model_name="test-model", runner=runner)
    run_benchmark([fixture], suite="golden", model_name="test-model", runner=runner)

    assert calls == 2

    calls = 0
    first = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
        cache_dir=tmp_path,
        cache_code_hash="code-v1",
    )
    second = run_benchmark(
        [fixture],
        suite="golden",
        model_name="test-model",
        runner=runner,
        cache_dir=tmp_path,
        cache_code_hash="code-v1",
    )

    assert calls == 1
    assert second.to_dict() == first.to_dict()


def test_eval_code_hash_returns_stable_sha256_digest():
    digest = eval_code_hash(("openmed.eval.metrics",))

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == eval_code_hash(("openmed.eval.metrics",))


def _report(*, marker: str) -> BenchmarkReport:
    return BenchmarkReport(
        suite="golden",
        model_name="privacy-filter",
        device="cpu",
        fixture_count=2,
        generated_at="2026-06-11T00:00:00Z",
        metrics={
            "leakage": {"overall": 0.0, "leaked_chars": 0, "total_chars": 12},
            "marker": marker,
        },
        metadata={"fixture_ids": ["fixture-1", "fixture-2"]},
    )
