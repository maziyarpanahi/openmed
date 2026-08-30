"""Focused tests for the offline comparator benchmark harness."""

from __future__ import annotations

import json
import random
import socket
import threading
from dataclasses import replace

import pytest

import openmed.eval.comparator as comparator_module
from openmed.eval.comparator import (
    STATUS_NOT_AVAILABLE,
    STATUS_SCORED,
    ComparatorAdapter,
    ComparatorAdapterUnavailable,
    ComparatorBudget,
    ComparatorExecutionError,
    ComparatorFixture,
    ComparatorFixtureError,
    ComparatorReport,
    load_comparator_fixtures,
    run_comparator_benchmark,
)
from openmed.eval.harness import BenchmarkFixture


def test_comparator_scores_required_metrics_with_shared_budget() -> None:
    fixture = _fixture()
    report = run_comparator_benchmark(
        [fixture],
        [
            ComparatorAdapter(name="exact", runner=_exact_runner),
            ComparatorAdapter(name="weak", runner=_person_only_runner),
        ],
        suite="synthetic-deidentification",
        budget=ComparatorBudget(max_latency_ms=5.0, max_memory_bytes=200),
        clock=_clock([0.0, 0.002, 0.0, 0.002]),
        memory_sampler=_sampler([100, 120, 100, 120]),
        generated_at="2026-08-09T00:00:00Z",
    )

    assert isinstance(report, ComparatorReport)
    assert report.fixture_count == 1
    assert [row.adapter for row in report.results] == ["exact", "weak"]

    exact = report.result("exact")
    assert exact.status == STATUS_SCORED
    assert exact.metrics is not None
    assert exact.metrics.precision == 1.0
    assert exact.metrics.recall == 1.0
    assert exact.metrics.critical_leakage == 0.0
    assert exact.metrics.critical_leakage_count == 0
    assert exact.metrics.latency.p95_ms == pytest.approx(2.0)
    assert exact.metrics.memory.peak_bytes == 120
    assert exact.metrics.within_budget is True

    weak = report.result("weak")
    assert weak.metrics is not None
    assert weak.metrics.precision == 1.0
    assert weak.metrics.recall == pytest.approx(0.5)
    assert weak.metrics.critical_leakage == 1.0
    assert weak.metrics.critical_leakage_count == 1
    assert weak.metrics.critical_span_count == 1


def test_comparator_report_is_deterministic_and_source_safe() -> None:
    fixture = _fixture()

    def run_once():
        return run_comparator_benchmark(
            [fixture],
            [ComparatorAdapter(name="exact", runner=_exact_runner)],
            metadata={"fixture_note": fixture.text, "owner": "synthetic"},
            clock=_clock([1.0, 1.005]),
            memory_sampler=_sampler([50, 75]),
        )

    first = run_once()
    second = run_once()
    first_json = first.to_json()
    second_json = second.to_json()

    assert first_json == second_json
    assert first.reproducibility_hash == second.reproducibility_hash
    assert fixture.text not in first_json
    assert fixture.text not in first.to_markdown()
    assert fixture.to_dict()["fixture_digest"] not in fixture.to_dict().get("text", "")
    payload = json.loads(first_json)
    assert payload["fixture_digests"]
    assert payload["results"][0]["metrics"]["memory"]["peak_bytes"] == 75


def test_reproducibility_hash_covers_fixture_and_adapter_configuration() -> None:
    fixture = _fixture()
    renamed_fixture = replace(fixture, fixture_id="synthetic-002")

    def run_once(*, model_name: str, metadata: dict[str, int]):
        return run_comparator_benchmark(
            [fixture],
            [
                ComparatorAdapter(
                    name="exact",
                    runner=_exact_runner,
                    model_name=model_name,
                )
            ],
            metadata=metadata,
        )

    baseline = run_once(model_name="local-a", metadata={"revision": 1})
    changed_model = run_once(model_name="local-b", metadata={"revision": 1})
    changed_metadata = run_once(model_name="local-a", metadata={"revision": 2})

    assert fixture.digest != renamed_fixture.digest
    assert baseline.reproducibility_hash != changed_model.reproducibility_hash
    assert baseline.reproducibility_hash != changed_metadata.reproducibility_hash


def test_report_rejects_forged_hash_and_duplicate_adapter_rows() -> None:
    report = run_comparator_benchmark(
        [_fixture()],
        [ComparatorAdapter(name="exact", runner=_exact_runner)],
    )

    with pytest.raises(ValueError, match="invalid comparator report"):
        replace(report, reproducibility_hash="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="invalid comparator report"):
        replace(
            report,
            results=(report.results[0], report.results[0]),
            reproducibility_hash="",
        )


def test_missing_or_network_adapters_are_reported_without_running() -> None:
    called = False

    def should_not_run(text: str, language: str):
        nonlocal called
        called = True
        return _exact_runner(text, language)

    report = run_comparator_benchmark(
        [_fixture()],
        [
            ComparatorAdapter(name="missing"),
            ComparatorAdapter(
                name="remote",
                runner=should_not_run,
                requires_network=True,
            ),
            ComparatorAdapter(
                name="declared-missing",
                runner=should_not_run,
                unavailable_reason="local optional dependency is absent",
            ),
        ],
    )

    assert [row.status for row in report.results] == [
        STATUS_NOT_AVAILABLE,
        STATUS_NOT_AVAILABLE,
        STATUS_NOT_AVAILABLE,
    ]
    assert all(row.metrics is None for row in report.results)
    assert called is False
    assert all(
        row.reason == "adapter is not available for this offline run"
        for row in report.results
    )


def test_adapter_errors_do_not_echo_raw_values() -> None:
    secret = "SYNTHETIC-ONLY-IDENTIFIER-0001"

    def broken_runner(text: str, language: str):
        raise RuntimeError(f"unexpected adapter payload: {secret}")

    with pytest.raises(ComparatorExecutionError) as excinfo:
        run_comparator_benchmark(
            [_fixture()],
            [ComparatorAdapter(name="broken", runner=broken_runner)],
        )

    assert secret not in str(excinfo.value)
    assert "unexpected adapter payload" not in str(excinfo.value)


def test_unavailable_exception_is_not_a_failed_benchmark() -> None:
    def unavailable_runner(text: str, language: str):
        raise ComparatorAdapterUnavailable("optional local package is absent")

    report = run_comparator_benchmark(
        [_fixture()],
        [ComparatorAdapter(name="optional", runner=unavailable_runner)],
    )

    assert report.result("optional").status == STATUS_NOT_AVAILABLE
    assert report.result("optional").metrics is None


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"synthetic": True},
        {"phi_free": True},
        {"synthetic": "true", "phi_free": True},
    ],
)
def test_mapping_fixtures_require_explicit_boolean_safety_flags(flags) -> None:
    payload = {
        "fixture_id": "synthetic-mapping",
        "text": "Synthetic text only.",
        **flags,
    }

    with pytest.raises(ComparatorFixtureError):
        ComparatorFixture.from_mapping(payload)

    payload.update(synthetic=True, phi_free=True)
    assert ComparatorFixture.from_mapping(payload).synthetic is True

    with pytest.raises(ComparatorFixtureError):
        ComparatorFixture(
            fixture_id="synthetic-defaults",
            text="Synthetic text only.",
        )


def test_existing_benchmark_fixture_requires_explicit_safety_metadata() -> None:
    fixture = BenchmarkFixture(
        fixture_id="synthetic-existing",
        text="Synthetic text only.",
        gold_spans=(),
    )

    with pytest.raises(ComparatorFixtureError):
        ComparatorFixture.from_benchmark_fixture(fixture)

    trusted = BenchmarkFixture(
        fixture_id=fixture.fixture_id,
        text=fixture.text,
        gold_spans=(),
        metadata={"synthetic": True, "phi_free": True},
    )
    assert ComparatorFixture.from_benchmark_fixture(trusted).phi_free is True


def test_fixture_loader_is_bounded_and_fails_closed(tmp_path, monkeypatch) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "fixture_id": "synthetic-file",
                    "text": "Synthetic text only.",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ComparatorFixtureError):
        load_comparator_fixtures(fixture_path)

    fixture_path.write_text(
        json.dumps(
            [
                {
                    "fixture_id": "synthetic-file",
                    "text": "Synthetic text only.",
                    "synthetic": True,
                    "phi_free": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    assert len(load_comparator_fixtures(fixture_path)) == 1

    monkeypatch.setattr(comparator_module, "_MAX_FIXTURE_FILE_BYTES", 8)
    with pytest.raises(ComparatorFixtureError, match="exceeds size limit"):
        load_comparator_fixtures(fixture_path)


def test_fixture_and_adapter_reprs_exclude_raw_values() -> None:
    secret = "SYNTHETIC-RAW-VALUE-DO-NOT-REPORT"
    fixture = ComparatorFixture(
        fixture_id="synthetic-safe-repr",
        text=secret,
        metadata={"note": secret},
        synthetic=True,
        phi_free=True,
    )
    adapter = ComparatorAdapter(
        name="safe-adapter",
        runner=_exact_runner,
        unavailable_reason=secret,
        metadata={"note": secret},
    )

    assert secret not in repr(fixture)
    assert secret not in repr(adapter)
    assert "text_length" in repr(fixture)
    assert "runner_available=True" in repr(adapter)


def test_runner_is_forced_offline_even_without_network_declaration() -> None:
    original_create_connection = socket.create_connection

    def network_runner(text: str, language: str):
        del text, language
        socket.create_connection(("127.0.0.1", 9), timeout=0.001)
        return ()

    with pytest.raises(
        ComparatorExecutionError,
        match="comparator adapter execution failed",
    ):
        run_comparator_benchmark(
            [_fixture()],
            [ComparatorAdapter(name="local-only", runner=network_runner)],
        )

    assert socket.create_connection is original_create_connection


def test_report_identifiers_reject_markup_without_echoing_it() -> None:
    secret = "SYNTHETIC-REPORT-INJECTION"

    with pytest.raises(ValueError) as adapter_error:
        ComparatorAdapter(name=f"safe\n| {secret}", runner=_exact_runner)
    assert secret not in str(adapter_error.value)

    with pytest.raises(ValueError) as suite_error:
        run_comparator_benchmark(
            [_fixture()],
            [ComparatorAdapter(name="safe", runner=_exact_runner)],
            suite=f"safe\n# {secret}",
        )
    assert secret not in str(suite_error.value)

    with pytest.raises(ValueError) as report_error:
        ComparatorReport(
            suite="safe",
            fixture_count=1,
            results=(),
            fixture_digests=(secret,),
            critical_labels=("SSN",),
            budget=ComparatorBudget(),
        )
    assert secret not in str(report_error.value)

    report = run_comparator_benchmark(
        [_fixture()],
        [ComparatorAdapter(name="safe", runner=_exact_runner)],
    )
    with pytest.raises(KeyError) as lookup_error:
        report.result(f"safe\n{secret}")
    assert secret not in str(lookup_error.value)


def test_prediction_span_collection_is_bounded(monkeypatch) -> None:
    fixture = _fixture()
    monkeypatch.setattr(comparator_module, "_MAX_SPAN_COUNT", 1)

    with pytest.raises(ComparatorExecutionError):
        run_comparator_benchmark(
            [fixture],
            [ComparatorAdapter(name="too-many", runner=_exact_runner)],
        )


def test_falsey_callable_runner_is_preserved() -> None:
    class FalseyRunner:
        def __bool__(self) -> bool:
            return False

        def __call__(self, text: str, language: str):
            return _exact_runner(text, language)

    report = run_comparator_benchmark(
        [_fixture()],
        [ComparatorAdapter(name="falsey-runner", runner=FalseyRunner())],
    )

    assert report.result("falsey-runner").status == STATUS_SCORED


def test_legacy_runner_cannot_read_gold_spans_or_arbitrary_metadata() -> None:
    observed = {}

    def legacy_runner(fixture, model_name, device):
        observed["gold_spans"] = fixture.gold_spans
        observed["metadata"] = fixture.metadata
        observed["model_name"] = model_name
        observed["device"] = device
        return _exact_runner(fixture.text, fixture.language)

    report = run_comparator_benchmark(
        [_fixture()],
        [
            ComparatorAdapter(
                name="legacy",
                runner=legacy_runner,
                model_name="local-model",
            )
        ],
    )

    assert report.result("legacy").status == STATUS_SCORED
    assert observed == {
        "gold_spans": (),
        "metadata": {"synthetic": True, "phi_free": True},
        "model_name": "local-model",
        "device": "cpu",
    }


def test_modern_varargs_runner_receives_text_and_language() -> None:
    observed = {}

    def modern_runner(text, language, *extra):
        observed["text"] = text
        observed["language"] = language
        observed["extra"] = extra
        return _exact_runner(text, language)

    fixture = _fixture()
    report = run_comparator_benchmark(
        [fixture],
        [ComparatorAdapter(name="modern", runner=modern_runner)],
    )

    assert report.result("modern").status == STATUS_SCORED
    assert observed == {
        "text": fixture.text,
        "language": fixture.language,
        "extra": (),
    }


def test_resource_budget_rejects_ambiguous_runtime_values() -> None:
    with pytest.raises(ValueError):
        ComparatorBudget(max_latency_ms=True)
    with pytest.raises(ValueError):
        ComparatorBudget(max_memory_bytes="100")


def test_critical_labels_and_metadata_fail_closed_when_ambiguous(
    monkeypatch,
) -> None:
    monkeypatch.setattr(comparator_module, "_MAX_CRITICAL_LABEL_COUNT", 1)
    adapter = ComparatorAdapter(name="exact", runner=_exact_runner)

    with pytest.raises(ValueError, match="invalid critical label configuration"):
        run_comparator_benchmark(
            [_fixture()],
            [adapter],
            critical_labels=("SSN", "ID_NUM"),
        )
    with pytest.raises(ValueError, match="invalid critical label configuration"):
        run_comparator_benchmark(
            [_fixture()],
            [adapter],
            critical_labels="SSN",
        )
    with pytest.raises(ValueError, match="invalid comparator metadata"):
        run_comparator_benchmark(
            [_fixture()],
            [adapter],
            critical_labels=("SSN",),
            metadata={1: "integer", "1": "string"},  # type: ignore[dict-item]
        )


def test_random_context_restores_state_and_serializes_threads() -> None:
    original_state = random.getstate()
    expected = random.Random(17)

    with comparator_module._random_context(17):
        first = random.random()
        with comparator_module._random_context(29):
            random.random()
        second = random.random()

    assert [first, second] == [expected.random(), expected.random()]
    assert random.getstate() == original_state

    started = threading.Event()
    entered = threading.Event()

    def enter_context() -> None:
        started.set()
        with comparator_module._random_context(31):
            entered.set()

    thread = threading.Thread(target=enter_context, daemon=True)
    with comparator_module._RANDOM_CONTEXT_LOCK:
        thread.start()
        assert started.wait(1.0)
        assert not entered.wait(0.05)
    assert entered.wait(1.0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def _fixture() -> ComparatorFixture:
    text = "Synthetic note for Rowan Vale; code 000-00-0000."
    person_start = text.index("Rowan Vale")
    code_start = text.index("000-00-0000")
    return ComparatorFixture(
        fixture_id="synthetic-001",
        text=text,
        gold_spans=(
            {
                "start": person_start,
                "end": person_start + len("Rowan Vale"),
                "label": "PERSON",
            },
            {
                "start": code_start,
                "end": code_start + len("000-00-0000"),
                "label": "SSN",
            },
        ),
        metadata={"synthetic": True, "phi_free": True},
    )


def _exact_runner(text: str, language: str):
    del language
    person_start = text.index("Rowan Vale")
    code_start = text.index("000-00-0000")
    return (
        {
            "start": person_start,
            "end": person_start + len("Rowan Vale"),
            "label": "PERSON",
        },
        {
            "start": code_start,
            "end": code_start + len("000-00-0000"),
            "label": "SSN",
        },
    )


def _person_only_runner(text: str, language: str):
    del language
    person_start = text.index("Rowan Vale")
    return (
        {
            "start": person_start,
            "end": person_start + len("Rowan Vale"),
            "label": "PERSON",
        },
    )


def _clock(values: list[float]):
    iterator = iter(values)
    last = values[-1]

    def tick() -> float:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return tick


def _sampler(values: list[int]):
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
