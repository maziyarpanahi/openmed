"""Unit tests for the inference-path memory profiler."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc

import pytest

from openmed.cli import main_module
from openmed.eval import memprofile as memprofile_module
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
    assert "note-secret" not in serialized


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
    assert profile.metadata["document_hashes"]
    assert all(
        value.startswith("sha256:") for value in profile.metadata["document_hashes"]
    )


def test_default_loader_reuses_exact_preloaded_non_hf_pipeline(monkeypatch) -> None:
    events: list[str] = []

    class FakeLoader:
        config = None

        def create_pipeline(self, model_name, **kwargs):
            events.append(f"create:{model_name}")

            def run_pipeline(text, **call_kwargs):
                events.append(f"run:{text}")
                return []

            return run_pipeline

        def get_max_sequence_length(self, model_name, *, tokenizer=None):
            events.append(f"max-length:{model_name}")
            return 128

    monkeypatch.setattr("openmed.core.models.ModelLoader", FakeLoader)

    runnable = memprofile_module._default_loader("fixture-model")
    assert events == ["create:fixture-model", "max-length:fixture-model"]

    result = runnable(PerfDocument("note-a", "Synthetic note.", language="en"))

    assert result.text == "Synthetic note."
    assert events.count("create:fixture-model") == 1
    assert events.count("max-length:fixture-model") == 1
    assert events.count("run:['Synthetic note.']") == 1


def test_default_loader_preloads_and_reuses_privacy_filter_pipeline(
    monkeypatch,
) -> None:
    events: list[str] = []

    def fake_create_privacy_filter_pipeline(model_name, config=None):
        events.append(f"create:{model_name}")

        def run_pipeline(texts, **kwargs):
            events.append(f"run:{len(texts)}")
            return [
                [
                    {
                        "entity_group": "PERSON",
                        "score": 0.99,
                        "start": 0,
                        "end": 9,
                        "word": "Synthetic",
                    }
                ]
                for _ in texts
            ]

        return run_pipeline

    monkeypatch.setattr(
        "openmed.core.backends.create_privacy_filter_pipeline",
        fake_create_privacy_filter_pipeline,
    )

    runnable = memprofile_module._default_loader("openai/privacy-filter")
    assert events == ["create:openai/privacy-filter"]

    result = runnable(PerfDocument("note-a", "Synthetic note.", language="en"))

    assert events == ["create:openai/privacy-filter", "run:1"]
    assert result.text == "Synthetic note."
    assert [(entity.text, entity.label) for entity in result.entities] == [
        ("Synthetic", "PERSON")
    ]


def test_measure_phase_samples_phase_local_current_rss() -> None:
    values = iter((100, 140, 120))
    last = 120

    def sample() -> int:
        nonlocal last
        try:
            last = next(values)
        except StopIteration:
            pass
        return last

    def work() -> bytearray:
        result = bytearray(4096)
        time.sleep(0.02)
        return result

    _, phase = memprofile_module._measure_phase(
        "fixture",
        work,
        sample_rss=sample,
        top_allocators=5,
    )

    assert phase.baseline_rss_bytes == 100
    assert phase.peak_rss_bytes == 140
    assert phase.rss_delta_bytes == 40
    assert phase.to_dict()["rss_semantics"] == "current-sampled"


def test_profile_memory_preserves_preexisting_tracemalloc_state() -> None:
    tracemalloc.start()
    try:
        sentinel = bytearray(8192)
        assert tracemalloc.get_object_traceback(sentinel) is not None

        profile = profile_memory(
            "mock-model",
            docs=["Synthetic note."],
            loader=_mock_loader,
            rss_sampler=lambda: 100 * MIB,
            generated_at="2026-07-05T00:00:00Z",
        )

        assert tracemalloc.is_tracing()
        assert tracemalloc.get_object_traceback(sentinel) is not None
        assert all(phase.tracemalloc_preexisting for phase in profile.phases)
    finally:
        tracemalloc.stop()


def test_preexisting_tracemalloc_peak_uses_non_destructive_phase_sampling() -> None:
    tracemalloc.start()
    try:
        prior_high_water = bytearray(4 * MIB)
        del prior_high_water
        _, peak_before = tracemalloc.get_traced_memory()
        assert peak_before >= 4 * MIB

        def transient_work() -> None:
            transient = bytearray(MIB)
            time.sleep(0.02)
            del transient
            time.sleep(0.01)

        _, phase = memprofile_module._measure_phase(
            "fixture",
            transient_work,
            sample_rss=lambda: 100 * MIB,
            top_allocators=5,
        )

        _, peak_after = tracemalloc.get_traced_memory()
        assert phase.tracemalloc_preexisting is True
        assert phase.traced_peak_bytes >= MIB - 16 * 1024
        assert phase.traced_peak_bytes < peak_before
        assert phase.to_dict()["traced_peak_semantics"] == "current-sampled-delta"
        assert peak_after >= peak_before
    finally:
        tracemalloc.stop()


def test_caller_metadata_cannot_override_protected_hashes_or_leak_phi() -> None:
    secret = "PATIENT-SECRET-METADATA"
    profile = profile_memory(
        "mock-model",
        docs=[PerfDocument(document_id="note-safe", text="Synthetic note.")],
        loader=_mock_loader,
        rss_sampler=lambda: 100 * MIB,
        generated_at="2026-07-05T00:00:00Z",
        metadata={
            "document_hashes": [secret],
            "raw_note": {"value": secret},
            "source": "cli",
        },
    )

    expected_hash = "sha256:" + hashlib.sha256(b"note-safe").hexdigest()
    serialized = profile.to_json() + profile.to_markdown()

    assert profile.metadata["document_hashes"] == [expected_hash]
    assert profile.metadata["caller_metadata"]["entry_count"] == 3
    assert profile.metadata["caller_metadata"]["sha256"].startswith("sha256:")
    assert profile.metadata["provenance"] == {"source": "cli"}
    assert secret not in serialized


def test_local_model_path_is_hashed_in_report(tmp_path) -> None:
    local_model = tmp_path / "PATIENT-SECRET-model"
    local_model.mkdir()

    profile = profile_memory(
        str(local_model),
        docs=["Synthetic note."],
        loader=_mock_loader,
        rss_sampler=lambda: 100 * MIB,
        generated_at="2026-07-05T00:00:00Z",
    )
    serialized = profile.to_json() + profile.to_markdown()

    assert profile.model_name.startswith("local-model:sha256:")
    assert str(local_model) not in serialized
    assert local_model.name not in serialized


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
