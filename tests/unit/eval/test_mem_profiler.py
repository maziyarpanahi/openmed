"""Unit tests for the inference-path memory profiler."""

from __future__ import annotations

import json

import pytest

from openmed.cli import main_module
from openmed.eval.memprofile import (
    PROFILE_PHASES,
    AllocatorStat,
    MemoryProfile,
    PhaseMemory,
    profile_memory,
    synthetic_memprofile_loader,
)
from openmed.eval.metrics import ResourceMetrics
from openmed.eval.perf import PerfDocument

MIB = 1024 * 1024


def _mock_loader(model):
    """Return a per-document callable that allocates without touching a model.

    Simulates the load/inference path so no heavy model download is needed while
    still exercising ``tracemalloc``. Allocations are retained on the callable so
    they remain live when the phase snapshot is taken (mirroring the model state,
    KV caches, and buffers a real inference path keeps resident).
    """
    loaded_state = ["x" * 4096 for _ in range(64)]

    def run_document(document: PerfDocument):
        scratch = ["y" * 4096 for _ in range(64)]
        # Retain the working set so it is resident at snapshot time.
        run_document._resident.append(scratch)
        return {"document_id": document.document_id, "chars": len(scratch)}

    run_document._loaded = loaded_state  # keep the load allocation alive
    run_document._resident: list = []
    return run_document


def test_profile_memory_reports_per_phase_peak_rss_and_top_allocators() -> None:
    docs = [
        PerfDocument(document_id="note-a", text="Synthetic one-page note A."),
        PerfDocument(document_id="note-b", text="Synthetic one-page note B."),
    ]

    profile = profile_memory(
        "mock-model",
        docs=docs,
        loader=_mock_loader,
        rss_sampler=_sequence([100 * MIB, 130 * MIB, 130 * MIB, 150 * MIB, 150 * MIB]),
        generated_at="2026-07-05T00:00:00Z",
    )

    assert isinstance(profile, MemoryProfile)
    assert profile.model_name == "mock-model"
    assert profile.document_count == 2

    for phase in profile.phases:
        assert isinstance(phase, PhaseMemory)
        assert phase.peak_rss_bytes is not None
        assert phase.peak_rss_bytes >= 0
        # tracemalloc must surface at least one allocator per phase.
        assert phase.top_allocators
        assert all(isinstance(stat, AllocatorStat) for stat in phase.top_allocators)


def test_profile_memory_phases_are_load_first_forward_then_steady_state() -> None:
    profile = profile_memory(
        "mock-model",
        docs=["Synthetic one-page note."],
        loader=_mock_loader,
        rss_sampler=lambda: 120 * MIB,
        generated_at="2026-07-05T00:00:00Z",
    )

    observed_order = tuple(phase.phase for phase in profile.phases)
    assert observed_order == PROFILE_PHASES
    assert observed_order == ("load", "first-forward", "steady-state-batch")
    # The serialized payload preserves the phase order as well.
    assert profile.to_dict()["phase_order"] == list(PROFILE_PHASES)


def test_profile_memory_rss_capture_uses_shared_resource_helper() -> None:
    profile = profile_memory(
        "mock-model",
        docs=["Synthetic one-page note."],
        loader=_mock_loader,
        rss_sampler=lambda: 200 * MIB,
        generated_at="2026-07-05T00:00:00Z",
    )

    load_phase = profile.phase("load")
    resources = load_phase.resources
    # The per-phase resource summary is the shared eval/metrics type, and the
    # MiB conversion matches ResourceMetrics rather than an ad-hoc computation.
    assert isinstance(resources, ResourceMetrics)
    assert resources.peak_rss_bytes == load_phase.peak_rss_bytes
    assert (
        load_phase.to_dict()["peak_rss_mib"]
        == ResourceMetrics(peak_rss_bytes=load_phase.peak_rss_bytes).peak_rss_mib
    )


def test_profile_memory_output_contains_no_document_text() -> None:
    secret = "PATIENT ACME-12345 SSN 555-66-7777"
    profile = profile_memory(
        "mock-model",
        docs=[PerfDocument(document_id="note-secret", text=secret)],
        loader=_mock_loader,
        rss_sampler=lambda: 120 * MIB,
        generated_at="2026-07-05T00:00:00Z",
    )

    serialized = profile.to_json() + profile.to_markdown()
    # No raw PHI: neither the note text nor identifying substrings leak into the
    # allocator provenance, which only carries file/line and byte counts.
    assert secret not in serialized
    assert "ACME-12345" not in serialized
    assert "555-66-7777" not in serialized


def test_profile_memory_defaults_to_committed_synthetic_workload() -> None:
    profile = profile_memory(
        "synthetic-one-page-note-runner",
        loader=synthetic_memprofile_loader,
        rss_sampler=lambda: 90 * MIB,
        generated_at="2026-07-05T00:00:00Z",
    )

    assert profile.document_count >= 1
    assert tuple(phase.phase for phase in profile.phases) == PROFILE_PHASES
    # The default workload runs offline without any heavy model download.
    assert profile.metadata["document_ids"]


def test_profile_memory_rejects_non_callable_loader_result() -> None:
    with pytest.raises(TypeError):
        profile_memory(
            "mock-model",
            docs=["Synthetic note."],
            loader=lambda model: "not-callable",
            rss_sampler=lambda: 120 * MIB,
        )


def test_cli_profile_memory_runs_synthetic_workload_without_crashing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(["profile", "memory"])

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["model_name"] == "synthetic-one-page-note-runner"
    assert payload["phase_order"] == list(PROFILE_PHASES)
    assert set(payload["phases"]) == set(PROFILE_PHASES)
    for phase_name in PROFILE_PHASES:
        phase = payload["phases"][phase_name]
        assert phase["peak_rss_bytes"] is None or phase["peak_rss_bytes"] >= 0
        assert isinstance(phase["top_allocators"], list)


def test_cli_profile_memory_writes_markdown_to_file(tmp_path, capsys) -> None:
    output = tmp_path / "profile.md"
    result = main_module.main(
        ["profile", "memory", "--format", "markdown", "--output", str(output)]
    )

    assert result == 0
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "# Memory Profile" in text
    assert "## Phase Breakdown" in text
    assert str(output) in capsys.readouterr().out


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
