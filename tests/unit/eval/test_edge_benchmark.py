"""Tests for the offline edge-SBC benchmark result contract."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from openmed.core.offline import OfflineModeError
from openmed.eval.edge_benchmark import (
    EdgeRuntime,
    load_edge_documents,
    measure_install_size,
    run_edge_benchmark,
)
from openmed.eval.footprint_gate import gate_footprint
from openmed.eval.perf import PerfDocument

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility path
    import tomli as tomllib

MIB = 1024 * 1024
ROOT = Path(__file__).resolve().parents[3]


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


def _documents() -> list[PerfDocument]:
    return [
        PerfDocument(
            document_id="synthetic-a",
            text="Synthetic note alpha",
            metadata={"source": "synthetic"},
        ),
        PerfDocument(
            document_id="synthetic-b",
            text="Synthetic note beta has details",
            metadata={"source": "synthetic"},
        ),
    ]


def _runtime(calls: list[str] | None = None) -> EdgeRuntime:
    def infer(text: str) -> None:
        if calls is not None:
            calls.append(text)

    return EdgeRuntime(
        name="unit-onnx-runtime",
        backend="onnxruntime",
        backend_version="1.23.0",
        execution_provider="CPUExecutionProvider",
        artifact_sha256="a" * 64,
        inference=infer,
    )


def test_edge_benchmark_records_cold_start_tokens_ram_and_install_size() -> None:
    calls: list[str] = []
    report = run_edge_benchmark(
        profile="raspberry-pi-5",
        documents=_documents(),
        runtime_loader=lambda: _runtime(calls),
        repeat=1,
        install_size_bytes=250 * MIB,
        clock=_sequence([0.0, 0.2, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5]),
        rss_sampler=_sequence([100 * MIB, 120 * MIB, 130 * MIB, 140 * MIB]),
        generated_at="2026-08-18T00:00:00Z",
    )

    payload = report.to_dict()
    assert payload["cold_start_ms"] == pytest.approx(200.0)
    assert payload["token_count"] == 8
    assert payload["tokens_per_second"] == pytest.approx(8 / 0.3)
    assert payload["peak_rss_bytes"] == 140 * MIB
    assert payload["install_size_bytes"] == 250 * MIB
    assert payload["steady_state_latency_ms"] == {
        "p50": 100.0,
        "p95": 200.0,
        "p99": 200.0,
    }
    assert payload["runtime"]["execution_provider"] == "CPUExecutionProvider"
    assert payload["network_guard"] == "socket-blocked"
    assert payload["offline"] is True
    assert len(calls) == 3  # one cold start plus two steady-state notes


def test_result_record_excludes_synthetic_text_ids_and_local_paths(
    tmp_path: Path,
) -> None:
    report = run_edge_benchmark(
        profile="jetson-nano",
        documents=_documents(),
        runtime_loader=_runtime,
        repeat=1,
        install_size_bytes=1,
        clock=_sequence([0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3]),
        rss_sampler=lambda: 10,
        generated_at="2026-08-18T00:00:00Z",
    )
    output = report.write_json(tmp_path / "edge.json")
    serialized = output.read_text(encoding="utf-8")

    assert json.loads(serialized)["workload"]["synthetic"] is True
    assert "Synthetic note alpha" not in serialized
    assert "synthetic-a" not in serialized
    assert str(tmp_path) not in serialized


def test_custom_corpus_filename_is_not_copied_into_report(tmp_path: Path) -> None:
    corpus_path = tmp_path / "Alice Patient private notes.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "id": "synthetic-note",
                "language": "en",
                "text": "Synthetic note for an invented person.",
                "metadata": {"source": "synthetic"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_edge_benchmark(
        profile="jetson-nano",
        corpus_path=corpus_path,
        runtime_loader=_runtime,
        repeat=1,
        install_size_bytes=1,
        clock=_sequence([0.0, 0.1, 0.1, 0.1, 0.2, 0.2]),
        rss_sampler=lambda: 10,
        generated_at="2026-08-18T00:00:00Z",
    )

    serialized = report.to_json()
    assert report.workload_name == "caller-supplied-synthetic"
    assert corpus_path.name not in serialized
    assert "Alice Patient" not in serialized


def test_benchmark_socket_guard_blocks_loader_network_calls() -> None:
    def network_loader() -> EdgeRuntime:
        socket.create_connection(("example.invalid", 443), timeout=0.01)
        return _runtime()

    with pytest.raises(OfflineModeError, match="blocks outbound network"):
        run_edge_benchmark(
            profile="jetson-nano",
            documents=_documents(),
            runtime_loader=network_loader,
            repeat=1,
            install_size_bytes=1,
        )


def test_non_synthetic_or_empty_workloads_fail_closed() -> None:
    real = PerfDocument(
        document_id="record",
        text="Not an approved fixture",
        metadata={"source": "clinical"},
    )

    with pytest.raises(ValueError, match="source=synthetic"):
        run_edge_benchmark(
            profile="jetson-nano",
            documents=[real],
            runtime_loader=_runtime,
            install_size_bytes=1,
        )
    with pytest.raises(ValueError, match="at least one"):
        run_edge_benchmark(
            profile="jetson-nano",
            documents=[],
            runtime_loader=_runtime,
            install_size_bytes=1,
        )


def test_committed_workload_is_explicitly_synthetic() -> None:
    documents = load_edge_documents()

    assert len(documents) >= 2
    assert all(document.metadata["source"] == "synthetic" for document in documents)


def test_archived_arm64_proxy_results_are_aggregate_and_within_budget() -> None:
    result_dir = ROOT / "docs" / "benchmarks" / "edge-sbc-results"
    paths = sorted(result_dir.glob("2026-08-18-*-arm64-proxy.json"))

    assert len(paths) == 2
    for path in paths:
        serialized = path.read_text(encoding="utf-8")
        payload = json.loads(serialized)
        verdict = gate_footprint(payload, profile=payload["profile"])

        assert payload["machine"]["architecture"] == "arm64"
        assert payload["workload"]["synthetic"] is True
        assert payload["network_guard"] == "socket-blocked"
        assert "Patient Alex Rivera" not in serialized
        assert "Jordan Lee" not in serialized
        assert verdict.passed is True


def test_install_size_counts_regular_files_without_symlink_duplicates(
    tmp_path: Path,
) -> None:
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "a.py").write_bytes(b"123")
    (tmp_path / "package" / "__pycache__").mkdir()
    (tmp_path / "package" / "__pycache__" / "a.pyc").write_bytes(b"ignored")
    (tmp_path / "metadata").write_bytes(b"12345")
    (tmp_path / "alias").symlink_to(tmp_path / "metadata")

    assert measure_install_size(tmp_path) == 8


def test_require_aarch64_rejects_other_architectures(monkeypatch) -> None:
    monkeypatch.setattr(
        "openmed.eval.edge_benchmark.platform.machine", lambda: "x86_64"
    )

    with pytest.raises(RuntimeError, match="aarch64/arm64"):
        run_edge_benchmark(
            profile="raspberry-pi-5",
            documents=_documents(),
            runtime_loader=_runtime,
            install_size_bytes=1,
            require_aarch64=True,
        )


def test_edge_sbc_extra_is_local_only_and_excludes_heavy_frameworks() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]

    edge = extras["edge-sbc"]
    normalized = {requirement.split(">=", 1)[0] for requirement in edge}
    assert normalized == {"numpy", "onnxruntime", "tokenizers"}
    assert all("torch" not in requirement for requirement in edge)
    assert all("transformers" not in requirement for requirement in edge)
    assert all("huggingface-hub" not in requirement for requirement in edge)
    assert "huggingface-hub>=0.30" in extras["onnx-runtime"]
