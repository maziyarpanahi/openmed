"""Tests for the offline ARM SMS latency benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from openmed.eval.arm_latency import (
    MAX_SMS_CHARACTERS,
    ArmLatencyBudget,
    LatencyDocument,
    load_arm_latency_budget,
    load_latency_documents,
    run_arm_latency_benchmark,
)


class _FakeInt8Model:
    variant = "int8"

    def __init__(self, model_path: Path = Path("model_int8.onnx")) -> None:
        self.model_path = model_path

    def predict(self, text: str) -> list[object]:
        return []


def _clock(values: list[float]):
    iterator = iter(values)
    return lambda: next(iterator)


def _empty_int8_model(tmp_path: Path) -> _FakeInt8Model:
    model_path = tmp_path / "model_int8.onnx"
    model_path.write_bytes(b"")
    return _FakeInt8Model(model_path)


def _budget(*, reference_p95_ms: float = 200.0) -> ArmLatencyBudget:
    return ArmLatencyBudget(
        name="unit-arm-budget",
        reference_device="Raspberry Pi 5 8GB",
        reference_cpu="4-core Arm Cortex-A76",
        reference_p95_ms=reference_p95_ms,
        regression_tolerance=0.2,
        model_id="OpenMed/unit-int8",
        model_revision="a" * 40,
        artifact_sha256=hashlib.sha256(b"").hexdigest(),
        quantization="int8",
    )


def _documents() -> list[LatencyDocument]:
    return [
        LatencyDocument("sms-a", "Synthetic patient A called 555-0100."),
        LatencyDocument("sms-b", "Synthetic patient B has a review tomorrow."),
    ]


def test_committed_corpus_is_synthetic_and_sms_scale() -> None:
    documents = load_latency_documents()

    assert len(documents) >= 10
    assert all(document.synthetic for document in documents)
    assert max(len(document.text) for document in documents) <= MAX_SMS_CHARACTERS


def test_committed_budget_applies_exact_twenty_percent_tolerance() -> None:
    budget = load_arm_latency_budget()

    assert budget.reference_p95_ms == 1500.0
    assert (
        budget.artifact_sha256
        == "48a0b2e9269933bef0cf8913239d07996fa2afb107cd223ced95c8decd24ae6b"
    )
    assert budget.maximum_p95_ms == pytest.approx(1800.0)
    assert budget.evaluate(1800.0).passed is True
    assert budget.evaluate(1800.001).passed is False


def test_report_emits_aggregate_latency_throughput_rss_and_metadata(
    tmp_path: Path,
) -> None:
    rss = iter([100 * 1024**2, 110 * 1024**2, 120 * 1024**2])
    report = run_arm_latency_benchmark(
        _empty_int8_model(tmp_path),
        model_id="OpenMed/unit-int8",
        model_revision="a" * 40,
        documents=_documents(),
        budget=_budget(),
        clock=_clock([0.0, 0.0, 0.1, 0.1, 0.3, 0.3]),
        rss_sampler=lambda: next(rss),
        warmup_runs=0,
        repeat=1,
        generated_at="2026-07-18T00:00:00Z",
    )

    payload = report.to_dict()
    assert payload["latency_ms"] == {"p50": 100.0, "p95": 200.0}
    assert payload["throughput_texts_per_second"] == pytest.approx(6.666667)
    assert payload["peak_rss_mib"] == 120.0
    assert payload["model"]["quantization"] == "int8"
    assert payload["model"]["artifact_sha256"] == hashlib.sha256(b"").hexdigest()
    assert payload["machine"]["cpu_count"]
    assert payload["corpus"]["synthetic"] is True
    assert payload["passed"] is True
    assert "Synthetic patient" not in report.to_json()


def test_intentionally_slowed_fixture_trips_gate(tmp_path: Path) -> None:
    report = run_arm_latency_benchmark(
        _empty_int8_model(tmp_path),
        model_id="OpenMed/unit-int8",
        model_revision="a" * 40,
        documents=[LatencyDocument("slow", "Synthetic slowed fixture")],
        budget=_budget(reference_p95_ms=1500.0),
        clock=_clock([0.0, 0.0, 1.801, 1.801]),
        rss_sampler=lambda: None,
        warmup_runs=0,
        repeat=1,
    )

    assert report.p95_ms == pytest.approx(1801.0)
    assert report.verdict.maximum_p95_ms == pytest.approx(1800.0)
    assert report.passed is False


def test_non_int8_graph_is_rejected() -> None:
    model = _FakeInt8Model()
    model.variant = "fp32"
    model.model_path = Path("model.onnx")

    with pytest.raises(ValueError, match="model_int8.onnx"):
        run_arm_latency_benchmark(
            model,
            model_id="OpenMed/unit-int8",
            model_revision="a" * 40,
            documents=_documents(),
            budget=_budget(),
            warmup_runs=0,
            repeat=1,
        )


def test_wrong_int8_artifact_hash_fails_closed(tmp_path: Path) -> None:
    model_path = tmp_path / "model_int8.onnx"
    model_path.write_bytes(b"unexpected model bytes")

    with pytest.raises(ValueError, match="SHA-256"):
        run_arm_latency_benchmark(
            _FakeInt8Model(model_path),
            model_id="OpenMed/unit-int8",
            model_revision="a" * 40,
            documents=_documents(),
            budget=_budget(),
            warmup_runs=0,
            repeat=1,
        )


def test_unpinned_artifact_hash_fails_closed(tmp_path: Path) -> None:
    budget = _budget()
    unpinned_budget = ArmLatencyBudget(
        name=budget.name,
        reference_device=budget.reference_device,
        reference_cpu=budget.reference_cpu,
        reference_p95_ms=budget.reference_p95_ms,
        regression_tolerance=budget.regression_tolerance,
        model_id=budget.model_id,
        model_revision=budget.model_revision,
        artifact_sha256="",
        quantization=budget.quantization,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        run_arm_latency_benchmark(
            _empty_int8_model(tmp_path),
            model_id="OpenMed/unit-int8",
            model_revision="a" * 40,
            documents=_documents(),
            budget=unpinned_budget,
            warmup_runs=0,
            repeat=1,
        )


def test_empty_corpus_and_unpinned_revision_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        run_arm_latency_benchmark(
            _FakeInt8Model(),
            model_id="OpenMed/unit-int8",
            model_revision="a" * 40,
            documents=[],
            budget=_budget(),
            warmup_runs=0,
            repeat=1,
        )

    with pytest.raises(ValueError, match="model id"):
        run_arm_latency_benchmark(
            _FakeInt8Model(),
            model_id="OpenMed/different-model",
            model_revision="a" * 40,
            documents=_documents(),
            budget=_budget(),
            warmup_runs=0,
            repeat=1,
        )

    with pytest.raises(ValueError, match="model revision"):
        run_arm_latency_benchmark(
            _FakeInt8Model(),
            model_id="OpenMed/unit-int8",
            model_revision="b" * 40,
            documents=_documents(),
            budget=_budget(),
            warmup_runs=0,
            repeat=1,
        )


@pytest.mark.parametrize(
    "document",
    [
        LatencyDocument("", "Synthetic text"),
        LatencyDocument("real", "Synthetic text", synthetic=False),
        LatencyDocument("long", "x" * (MAX_SMS_CHARACTERS + 1)),
    ],
)
def test_invalid_or_non_synthetic_documents_fail_closed(
    document: LatencyDocument,
) -> None:
    with pytest.raises(ValueError):
        run_arm_latency_benchmark(
            _FakeInt8Model(),
            model_id="OpenMed/unit-int8",
            model_revision="a" * 40,
            documents=[document],
            budget=_budget(),
            warmup_runs=0,
            repeat=1,
        )
