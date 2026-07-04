"""Unit tests for per-device perf benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.perf import (
    PerfDocument,
    load_perf_documents,
    lookup_tier_budget,
    run_perf_benchmark,
)

MIB = 1024 * 1024


def _sequence(values):
    iterator = iter(values)
    last = values[-1]

    def next_value():
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return next_value


def test_run_perf_benchmark_reports_throughput_latency_resources_and_size(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"openmed-model")
    docs = [
        PerfDocument(document_id="note-a", text="Synthetic one-page note A."),
        PerfDocument(document_id="note-b", text="Synthetic one-page note B."),
    ]

    report = run_perf_benchmark(
        model_path,
        device="cpu",
        tier="base",
        docs=docs,
        runner=lambda model, document, device: {"id": document.document_id},
        clock=_sequence([0.0, 0.0, 0.01, 0.01, 0.03, 0.03]),
        rss_sampler=_sequence([100 * MIB, 110 * MIB, 120 * MIB]),
        generated_at="2026-07-03T00:00:00Z",
    )

    payload = report.to_dict()
    assert payload["docs_per_second"] == pytest.approx(2 / 0.03)
    assert payload["latency"]["p50_ms"] == pytest.approx(10.0)
    assert payload["latency"]["p95_ms"] == pytest.approx(20.0)
    assert payload["resources"]["peak_rss_bytes"] == 120 * MIB
    assert payload["resources"]["model_size_bytes"] == len(b"openmed-model")
    assert payload["slo_results"]["p95_latency_ms"]["passed"] is True
    assert json.loads(report.to_json())["model_name"] == str(model_path)


def test_exceeding_budget_flags_non_gating_slo_failures() -> None:
    report = run_perf_benchmark(
        "fixture-model",
        device="cpu",
        tier="base",
        docs=["Synthetic one-page note."],
        runner=lambda model, document, device: None,
        clock=_sequence([0.0, 0.0, 0.5, 0.5]),
        rss_sampler=_sequence([100 * MIB, 901 * MIB]),
    )

    assert report.slo_results["p95_latency_ms"].passed is False
    assert report.slo_results["peak_rss_mib"].passed is False


def test_tier_budget_lookup_matches_section_6_2_table() -> None:
    phone = lookup_tier_budget("phone")
    base = lookup_tier_budget("base")
    server = lookup_tier_budget("server")

    assert phone.canonical_tier == "Tiny"
    assert phone.ram_mb_max == 350
    assert phone.p50_ms_max == 60
    assert phone.p95_ms_max == 150
    assert base.canonical_tier == "Base"
    assert base.ram_mb_max == 900
    assert base.p50_ms_max == 150
    assert base.p95_ms_max == 400
    assert server.canonical_tier == "Accurate-XLarge"
    assert server.ram_mb_max == 8192
    assert server.p50_ms_max == 400
    assert server.p95_ms_max == 1200


def test_default_synthetic_workload_is_committed_and_loadable() -> None:
    documents = load_perf_documents()

    assert [document.document_id for document in documents] == [
        "synthetic-note-001",
        "synthetic-note-002",
    ]
    assert all(document.text for document in documents)
