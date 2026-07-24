from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from openmed.eval import i18n_throughput
from openmed.eval.release_gates import evaluate_i18n_throughput_gate


def _baseline_store(value: float = 1_000.0) -> dict[str, object]:
    entries = {}
    for language in i18n_throughput.I18N_THROUGHPUT_LANGUAGES:
        key = f"i18n-throughput::{language}::pattern-only"
        entries[key] = {
            "key": key,
            "family": "i18n-throughput",
            "tier": language,
            "format": "pattern-only",
            "metrics": {
                "segmentation_chars_per_second": value,
                "deidentify_spans_per_second": value,
            },
            "metadata": {"regression_threshold": 0.2},
            "reproducibility_hash": "sha256:" + ("0" * 64),
        }
    return {"schema_version": 1, "entries": entries}


def _throughput_report(value: float = 1_000.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": i18n_throughput.I18N_THROUGHPUT_ARTIFACT,
        "languages": {
            language: {
                "segmentation_chars_per_second": value,
                "deidentify_spans_per_second": value,
            }
            for language in i18n_throughput.I18N_THROUGHPUT_LANGUAGES
        },
    }


@pytest.mark.parametrize(
    "language",
    i18n_throughput.I18N_THROUGHPUT_LANGUAGES,
)
def test_committed_corpora_are_large_deterministic_faker_fixtures(
    language: str,
) -> None:
    fixture = i18n_throughput.load_synthetic_corpus(language)

    assert len(fixture["text"]) >= i18n_throughput.I18N_THROUGHPUT_MIN_CHARS
    assert fixture["metadata"]["synthetic"] is True
    assert fixture["metadata"]["generated_only"] is True
    assert fixture["metadata"]["generator"] == "Faker"
    assert fixture["metadata"]["sha256"].startswith("sha256:")


def test_benchmark_emits_cold_start_and_steady_state_metrics() -> None:
    corpus = "合成患者电话13000000000。" * 10

    result = i18n_throughput.benchmark_language(
        "zh",
        corpus,
        iterations=2,
        segmenter_factory=lambda: lambda text: list(text),
        deidentifier_factory=lambda: lambda text: 3,
    )

    assert result["segmentation_cold_start_ms"] >= 0
    assert result["segmentation_chars_per_second"] > 0
    assert result["deidentify_cold_start_ms"] >= 0
    assert result["deidentify_spans_per_second"] > 0
    assert result["iterations"] == 2


def test_throughput_gate_allows_exactly_twenty_percent_drop() -> None:
    report = _throughput_report(800.0)

    check = evaluate_i18n_throughput_gate(report, _baseline_store())

    assert check.passed is True


def test_sleep_injected_segmenter_triggers_language_metric_gate() -> None:
    corpus = "患" * 1_000

    def delayed_segmenter(text: str) -> list[str]:
        time.sleep(0.01)
        return [text]

    zh_metrics = i18n_throughput.benchmark_language(
        "zh",
        corpus,
        iterations=1,
        segmenter_factory=lambda: delayed_segmenter,
        deidentifier_factory=lambda: lambda text: 1,
    )
    report = _throughput_report(1_000.0)
    report["languages"]["zh"] = zh_metrics
    baseline = _baseline_store()
    baseline["entries"]["i18n-throughput::zh::pattern-only"]["metrics"][
        "segmentation_chars_per_second"
    ] = 1_000_000.0

    check = evaluate_i18n_throughput_gate(report, baseline)

    assert check.passed is False
    assert "zh.segmentation_chars_per_second" in check.reason


def test_benchmark_output_and_logs_never_include_corpus_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "合成患者日志哨兵13000000000"
    fixture = {
        "language": "zh",
        "path": "synthetic.json",
        "text": raw_text,
        "metadata": {
            "synthetic": True,
            "generator": "Faker",
            "faker_locale": "zh_CN",
            "seed": 1,
            "char_count": len(raw_text),
            "sha256": "sha256:" + ("1" * 64),
        },
    }
    monkeypatch.setattr(
        i18n_throughput,
        "load_synthetic_corpus",
        lambda language, fixture_dir: fixture,
    )
    monkeypatch.setattr(
        i18n_throughput,
        "benchmark_language",
        lambda language, corpus, iterations: {
            "char_count": len(corpus),
            "segmentation_cold_start_ms": 1.0,
            "segmentation_chars_per_second": 2.0,
            "deidentify_cold_start_ms": 3.0,
            "deidentify_spans_per_second": 4.0,
        },
    )

    with caplog.at_level(logging.DEBUG):
        report = i18n_throughput.run_benchmark(
            fixture_dir=tmp_path,
            languages=("zh",),
            iterations=1,
        )

    serialized = json.dumps(report, ensure_ascii=False)
    rendered_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert raw_text not in serialized
    assert raw_text not in rendered_logs


def test_release_gate_cli_writes_machine_readable_throughput_verdict(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "gate.json"
    candidate.write_text(json.dumps(_throughput_report()), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline_store()), encoding="utf-8")

    from openmed.eval import release_gates

    exit_code = release_gates.main(
        [
            "--throughput-candidate",
            str(candidate),
            "--baseline-store",
            str(baseline),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "openmed.eval.i18n_throughput_gate"
    assert payload["decision"] == "RELEASABLE"


def test_release_workflow_runs_throughput_gate_only_on_release_surface() -> None:
    release_workflow = Path(".github/workflows/release-gates.yml").read_text(
        encoding="utf-8"
    )
    pr_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m openmed.eval.i18n_throughput" in release_workflow
    assert "--throughput-candidate i18n-throughput-report.json" in release_workflow
    assert "steps.i18n_throughput.outcome == 'success'" in release_workflow
    assert "steps.i18n_throughput.outcome != 'success'" in release_workflow
    assert "python -m openmed.eval.i18n_throughput" not in pr_workflow
