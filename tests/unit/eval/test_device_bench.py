"""Offline tests for the batch-aware device benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval import device_bench as device_bench_module
from openmed.eval.device_bench import (
    DeviceBenchResult,
    load_device_benchmark_archive,
    run_device_benchmark,
    synthetic_device_bench_runner,
    write_device_benchmark_archive,
)
from openmed.eval.tiers import TIERS

MIB = 1024 * 1024


def _stepping_clock(step: float = 0.01):
    current = 0.0

    def clock() -> float:
        nonlocal current
        value = current
        current += step
        return value

    return clock


def _steady_rss(values: list[int]):
    iterator = iter(values)
    last = values[-1]

    def sample() -> int:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return sample


def _run_result(*, metadata=None) -> DeviceBenchResult:
    return run_device_benchmark(
        "OpenMed/synthetic-device-model",
        device="cpu",
        tier="base",
        model_format="int8",
        docs=["Synthetic note alpha.", "Synthetic note beta."],
        sequence_lengths=[4, 8],
        batch_sizes=[1, 2],
        runner=synthetic_device_bench_runner,
        clock=_stepping_clock(),
        rss_sampler=_steady_rss([100 * MIB, 110 * MIB, 120 * MIB]),
        metadata=metadata,
    )


def test_device_benchmark_reports_throughput_percentiles_and_rss() -> None:
    result = _run_result()

    assert result.sequence_lengths == (4, 8)
    assert result.batch_sizes == (1, 2)
    assert result.batch_count == 6
    assert result.document_count == 8
    assert result.docs_per_second == pytest.approx(8 / 0.06)
    assert result.p50_ms == pytest.approx(10.0)
    assert result.p95_ms == pytest.approx(10.0)
    assert result.peak_rss_mib == pytest.approx(120.0)
    assert result.tier_budget["p50_ms_max"] == TIERS["Base"]["p50_ms_max"]
    assert result.to_dict()["metrics"] == {
        "docs_per_second": result.docs_per_second,
        "p50_ms": result.p50_ms,
        "p95_ms": result.p95_ms,
        "peak_rss_mib": result.peak_rss_mib,
    }


def test_device_benchmark_archive_is_keyed_and_reproducible(tmp_path: Path) -> None:
    first = _run_result()
    archive_path = write_device_benchmark_archive(first, tmp_path)
    first_bytes = archive_path.read_bytes()

    second = _run_result()
    write_device_benchmark_archive(second, tmp_path)
    assert archive_path.read_bytes() == first_bytes

    payload = load_device_benchmark_archive(archive_path)
    assert payload["format"] == "INT8"
    assert payload["device"] == "cpu"
    assert payload["tier"] == "Base"
    assert set(payload["results"]) == {"OpenMed/synthetic-device-model"}
    assert payload["results"]["OpenMed/synthetic-device-model"]["archive_key"] == (
        "OpenMed/synthetic-device-model|INT8|cpu"
    )
    assert payload["results"]["OpenMed/synthetic-device-model"]["key"] == {
        "device": "cpu",
        "format": "INT8",
        "repo_id": "OpenMed/synthetic-device-model",
        "tier": "Base",
    }
    assert json.loads(first_bytes.decode("utf-8")) == payload


def test_device_benchmark_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        run_device_benchmark(
            "synthetic-model",
            device="gpu",
            runner=synthetic_device_bench_runner,
            docs=["Synthetic note."],
        )


def test_device_benchmark_rejects_unbounded_matrix_inputs() -> None:
    with pytest.raises(ValueError, match="repeats must be between"):
        run_device_benchmark(
            "synthetic-model",
            repeats=101,
            runner=synthetic_device_bench_runner,
            docs=["Synthetic note."],
        )

    with pytest.raises(ValueError, match="positive integers"):
        run_device_benchmark(
            "synthetic-model",
            sequence_lengths=[32_769],
            runner=synthetic_device_bench_runner,
            docs=["Synthetic note."],
        )


def test_device_benchmark_rejects_non_finite_clock_and_invalid_rss() -> None:
    clock_values = iter([0.0, float("nan")])
    with pytest.raises(ValueError, match="clock must return a finite number"):
        run_device_benchmark(
            "synthetic-model",
            sequence_lengths=[4],
            batch_sizes=[1],
            runner=synthetic_device_bench_runner,
            docs=["Synthetic note."],
            clock=lambda: next(clock_values),
            rss_sampler=lambda: None,
        )

    with pytest.raises(ValueError, match="rss_sampler"):
        run_device_benchmark(
            "synthetic-model",
            sequence_lengths=[4],
            batch_sizes=[1],
            runner=synthetic_device_bench_runner,
            docs=["Synthetic note."],
            rss_sampler=lambda: True,
        )


def test_device_benchmark_redacts_and_freezes_metadata() -> None:
    result = _run_result(
        metadata={
            "patient_note": "Ada Example, MRN 12345",
            "repeats": 999,
            "runtime": {"worker_count": 2},
        }
    )

    payload = result.to_dict()
    assert "patient_note" not in payload["metadata"]
    assert len(payload["metadata"]["patient_note_sha256"]) == 64
    assert payload["metadata"]["repeats"] == 1
    assert "Ada Example" not in result.to_json()
    with pytest.raises(TypeError):
        result.metadata["runtime"]["worker_count"] = 3


def test_device_benchmark_does_not_archive_local_model_paths(tmp_path: Path) -> None:
    model_path = tmp_path / "private-model"
    result = run_device_benchmark(
        model_path,
        docs=["Synthetic note."],
        sequence_lengths=[4],
        batch_sizes=[1],
        runner=synthetic_device_bench_runner,
    )

    assert result.repo_id.startswith("local-sha256-")
    assert str(tmp_path) not in result.to_json()


def test_device_benchmark_rejects_non_timestamp_generated_at() -> None:
    with pytest.raises(ValueError, match="ISO-8601"):
        run_device_benchmark(
            "synthetic-model",
            docs=["Synthetic note."],
            generated_at="Ada Example",
            sequence_lengths=[4],
            batch_sizes=[1],
            runner=synthetic_device_bench_runner,
        )


def test_device_benchmark_rejects_metadata_cycles() -> None:
    metadata: dict[str, object] = {}
    metadata["nested"] = metadata

    with pytest.raises(ValueError, match="cycles"):
        _run_result(metadata=metadata)


def test_load_device_corpus_applies_a_pre_parse_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text('{"documents": []}', encoding="utf-8")
    monkeypatch.setattr(device_bench_module, "_MAX_CORPUS_BYTES", 8)

    with pytest.raises(ValueError, match="size limit"):
        device_bench_module.load_device_corpus(corpus_path)


def test_device_benchmark_archive_rejects_tampered_identity(tmp_path: Path) -> None:
    archive_path = write_device_benchmark_archive(_run_result(), tmp_path)
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    payload["archive_key"] = "tampered"
    archive_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="archive key mismatch"):
        load_device_benchmark_archive(archive_path)


def test_device_benchmark_archive_rejects_unknown_raw_fields(tmp_path: Path) -> None:
    archive_path = write_device_benchmark_archive(_run_result(), tmp_path)
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    payload["results"]["OpenMed/synthetic-device-model"]["text"] = "raw note"
    archive_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid result"):
        load_device_benchmark_archive(archive_path)


def test_device_benchmark_archive_write_is_atomic(tmp_path: Path) -> None:
    archive_path = write_device_benchmark_archive(_run_result(), tmp_path)

    assert archive_path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [archive_path.name]
