"""Tests for synthetic cross-corpus generalization reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.eval import generalization as generalization_module
from openmed.eval.generalization import GeneralizationReport, cross_corpus_report
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.report import BenchmarkReport


def test_cross_corpus_report_surfaces_planted_domain_shift() -> None:
    in_domain = (_fixture("in-domain", prediction="gold"),)
    shifted = (_fixture("shifted", prediction="miss"),)

    report = cross_corpus_report(
        "synthetic-generalization-model",
        in_domain,
        {"shifted-corpus": shifted},
        runner=_runner,
    )

    assert isinstance(report, GeneralizationReport)
    assert report.in_domain_suite == "in-domain"
    assert report.out_of_domain_suites == ("shifted-corpus",)
    assert report.deltas["shifted-corpus"]["leakage_rate"] == pytest.approx(1.0)
    assert report.deltas["shifted-corpus"]["recall"] == pytest.approx(-1.0)
    assert report.deltas["shifted-corpus"]["f1"] == pytest.approx(-1.0)
    assert report.metrics["f1"]["shifted-corpus"]["gap"] == pytest.approx(1.0)
    assert report.headline_gap == pytest.approx(1.0)

    payload = report.to_dict()
    assert payload["artifact_type"] == "openmed.eval.generalization"
    assert payload["delta_by_metric"]["f1"]["shifted-corpus"] == pytest.approx(-1.0)
    assert "Synthetic" not in report.to_json()
    assert report.to_markdown() == report.to_markdown()


def test_identical_in_and_out_corpora_have_zero_generalization_gap() -> None:
    fixtures = (_fixture("identical", prediction="gold"),)

    report = cross_corpus_report(
        "synthetic-generalization-model",
        fixtures,
        {"same-corpus-copy": fixtures},
        runner=_runner,
    )

    assert report.headline_gap == pytest.approx(0.0)
    assert report.deltas["same-corpus-copy"] == {
        "leakage_rate": pytest.approx(0.0),
        "recall": pytest.approx(0.0),
        "f1": pytest.approx(0.0),
    }
    assert report.generalization_gap == pytest.approx(0.0)


def test_generalization_cli_writes_report_for_local_synthetic_suites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    in_path = tmp_path / "in-domain.jsonl"
    out_path = tmp_path / "out-domain.jsonl"
    output_path = tmp_path / "generalization.json"
    rows = [
        {
            "id": "synthetic-note",
            "text": "Synthetic subject",
            "gold_spans": [{"start": 9, "end": 16, "label": "PERSON"}],
        }
    ]
    fixture_payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    in_path.write_text(fixture_payload, encoding="utf-8")
    out_path.write_text(fixture_payload, encoding="utf-8")

    def fake_run_benchmark(fixtures, **kwargs):
        source = kwargs["metadata"]["source_corpus"]
        shifted = source == "out-domain"
        return BenchmarkReport(
            suite=kwargs["suite"],
            model_name=kwargs["model_name"],
            device=kwargs["device"],
            fixture_count=len(fixtures),
            metrics={
                "leakage": {"overall": 0.0},
                "character_recall": {"rate": 0.0 if shifted else 1.0},
                "exact_span_f1": {"f1": 0.0 if shifted else 1.0},
            },
            generated_at=kwargs.get("generated_at"),
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(generalization_module, "run_benchmark", fake_run_benchmark)

    result = main_module.main(
        [
            "benchmark",
            "generalization",
            "--model",
            "synthetic-generalization-model",
            "--in-domain",
            str(in_path),
            "--out-of-domain",
            str(out_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    report = GeneralizationReport.read_json(output_path)
    assert report.headline_gap == pytest.approx(2.0 / 3.0)
    assert report.out_of_domain_suites == ("out-domain",)


def _fixture(fixture_id: str, *, prediction: str) -> BenchmarkFixture:
    return BenchmarkFixture.from_mapping(
        {
            "id": fixture_id,
            "text": "Synthetic subject",
            "gold_spans": [{"start": 9, "end": 16, "label": "PERSON"}],
            "metadata": {"prediction": prediction, "synthetic": True},
        }
    )


def _runner(fixture: BenchmarkFixture, model_name: str, device: str):
    assert model_name == "synthetic-generalization-model"
    assert device == "cpu"
    if fixture.metadata["prediction"] == "gold":
        return [
            {
                "start": span.start,
                "end": span.end,
                "label": span.label,
            }
            for span in fixture.gold_spans
        ]
    return []
