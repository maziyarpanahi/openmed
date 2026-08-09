"""Offline tests for the batch-aware device benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def _run_result() -> DeviceBenchResult:
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
