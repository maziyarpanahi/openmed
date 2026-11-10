"""Unit tests for the site/demographic extraction-fairness audit.

The audit measures whether extraction recall/F1 is equitable across synthetic
site, note-type, and demographic surrogate groups. It is distinct from the de-id
leakage fairness path, which is covered by ``test_fairness_report.py`` and must
stay unchanged.
"""

from __future__ import annotations

import json

import pytest

from openmed.eval import (
    EXTRACTION_FAIRNESS_ASSISTIVE_NOTE,
    extraction_fairness_report,
    load_extraction_fairness_fixtures,
)
from openmed.eval.harness import BenchmarkFixture


def _full_recall_runner(fixture, model_name, device):
    return [
        {"start": span.start, "end": span.end, "label": span.label, "text": span.text}
        for span in fixture.gold_spans
    ]


def _drop_group_runner(axis: str, value: str):
    def runner(fixture, model_name, device):
        if fixture.metadata.get(axis) == value:
            return []
        return _full_recall_runner(fixture, model_name, device)

    return runner


def test_extraction_fairness_report_returns_per_group_metrics_and_gap() -> None:
    fixtures = load_extraction_fairness_fixtures()

    report = extraction_fairness_report(
        "extract-model", fixtures, runner=_full_recall_runner
    )

    assert report.fixture_count == len(fixtures)
    # One group key per axis value present in the corpus.
    assert set(report.per_group) == {
        "site=site_alpha",
        "site=site_beta",
        "note_type=discharge_summary",
        "note_type=progress_note",
        "demographic_group=surrogate_a",
        "demographic_group=surrogate_b",
    }
    for metrics in report.per_group.values():
        assert metrics.entity_f1 == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.critical_finding_recall == pytest.approx(1.0)
    assert report.extraction_f1_gap == pytest.approx(0.0)
    assert report.recall_gap == pytest.approx(0.0)
    assert report.critical_finding_recall_gap == pytest.approx(0.0)


def test_extraction_fairness_report_flags_underperforming_demographic_group() -> None:
    fixtures = load_extraction_fairness_fixtures()

    report = extraction_fairness_report(
        "extract-model",
        fixtures,
        runner=_drop_group_runner("demographic_group", "surrogate_b"),
    )

    weak = report.per_group["demographic_group=surrogate_b"]
    strong = report.per_group["demographic_group=surrogate_a"]
    assert weak.entity_f1 == pytest.approx(0.0)
    assert weak.recall == pytest.approx(0.0)
    assert weak.critical_finding_recall == pytest.approx(0.0)
    assert strong.entity_f1 == pytest.approx(1.0)

    assert report.extraction_f1_gap == pytest.approx(1.0)
    assert report.worst_group == "demographic_group=surrogate_b"
    assert report.best_group == "demographic_group=surrogate_a"
    assert report.worst_group_by_metric == {
        "entity_f1": "demographic_group=surrogate_b",
        "recall": "demographic_group=surrogate_b",
        "critical_finding_recall": "demographic_group=surrogate_b",
    }
    # The demographic axis carries the disparity; the site axis is only partial.
    assert report.by_axis["demographic_group"]["extraction_f1_gap"] == pytest.approx(
        1.0
    )
    assert report.by_axis["site"]["extraction_f1_gap"] == pytest.approx(0.0)


def test_extraction_fairness_report_flags_underperforming_site_group() -> None:
    fixtures = load_extraction_fairness_fixtures()

    report = extraction_fairness_report(
        "extract-model",
        fixtures,
        runner=_drop_group_runner("site", "site_beta"),
    )

    assert report.per_group["site=site_beta"].entity_f1 == pytest.approx(0.0)
    assert report.per_group["site=site_alpha"].entity_f1 == pytest.approx(1.0)
    assert report.by_axis["site"]["worst_group"] == "site=site_beta"
    assert report.by_axis["site"]["extraction_f1_gap"] == pytest.approx(1.0)


def test_extraction_fairness_report_states_assistive_synthetic_provenance() -> None:
    fixtures = load_extraction_fairness_fixtures()

    report = extraction_fairness_report(
        "extract-model", fixtures, runner=_full_recall_runner
    )
    payload = report.to_dict()

    assert report.assistive is True
    assert report.note == EXTRACTION_FAIRNESS_ASSISTIVE_NOTE
    assert "synthetic surrogate" in report.note
    assert "not from patient attributes inferred" in report.note
    assert payload["assistive"] is True
    assert payload["artifact_type"] == "openmed.extraction_fairness_audit"
    # Deterministic, PHI-free serialization.
    assert json.loads(report.to_json()) == payload
    assert (
        extraction_fairness_report(
            "extract-model", fixtures, runner=_full_recall_runner
        ).to_json()
        == report.to_json()
    )


def test_extraction_fairness_gate_metric_matches_report() -> None:
    fixtures = load_extraction_fairness_fixtures()
    report = extraction_fairness_report(
        "extract-model",
        fixtures,
        runner=_drop_group_runner("demographic_group", "surrogate_b"),
    )

    metric = report.gate_metric()

    assert metric["extraction_f1_gap"] == pytest.approx(1.0)
    assert metric["worst_group"] == "demographic_group=surrogate_b"
    assert set(metric["per_group"]) == set(report.per_group)
    assert metric["per_group"]["demographic_group=surrogate_b"][
        "entity_f1"
    ] == pytest.approx(0.0)
    assert metric["assistive"] is True


def test_scorecard_extraction_fairness_section_is_additive() -> None:
    from openmed.eval.scorecard import ModelScorecard

    fixtures = load_extraction_fairness_fixtures()
    audit = extraction_fairness_report(
        "m", fixtures, runner=_drop_group_runner("demographic_group", "surrogate_b")
    )

    without = ModelScorecard(model_name="m", reports=())
    assert "extraction_fairness" not in without.to_dict()
    assert "Extraction Fairness" not in without.to_markdown()

    withf = ModelScorecard(model_name="m", reports=(), extraction_fairness=audit)
    payload = withf.to_dict()
    assert payload["extraction_fairness"]["extraction_f1_gap"] == pytest.approx(1.0)
    assert payload["extraction_fairness"]["worst_group"] == (
        "demographic_group=surrogate_b"
    )
    markdown = withf.to_markdown()
    assert "## Extraction Fairness (Synthetic Surrogates)" in markdown
    assert "not from patient attributes inferred" in markdown


def test_evidence_bundle_extraction_fairness_section_is_additive(tmp_path) -> None:
    from openmed.eval.evidence_bundle import bundle_gate_evidence

    gate_report = {
        "repo_id": "OpenMed/unit-model",
        "decision": "releasable",
        "gate_results": [{"gate": "G14", "passed": True, "reason": "ok"}],
    }
    fixtures = load_extraction_fairness_fixtures()
    audit = extraction_fairness_report(
        "m", fixtures, runner=_drop_group_runner("demographic_group", "surrogate_b")
    )

    without = bundle_gate_evidence(dict(gate_report), tmp_path / "a")
    assert "extraction_fairness" not in without.manifest

    withf = bundle_gate_evidence(
        dict(gate_report), tmp_path / "b", extraction_fairness=audit
    )
    section = withf.manifest["extraction_fairness"]
    assert section["extraction_f1_gap"] == pytest.approx(1.0)
    assert section["worst_group"] == "demographic_group=surrogate_b"
    assert section["assistive"] is True


def test_extraction_fairness_report_handles_inline_fixtures() -> None:
    fixtures = [
        BenchmarkFixture.from_mapping(
            {
                "id": "inline-1",
                "language": "en",
                "text": "Synthetic note names Robin Vega.",
                "gold_spans": [
                    {"start": 21, "end": 31, "label": "PERSON", "text": "Robin Vega"}
                ],
                "metadata": {"site": "site_x", "note_type": "progress_note"},
            }
        )
    ]

    report = extraction_fairness_report("m", fixtures, runner=_full_recall_runner)

    assert set(report.per_group) == {"site=site_x", "note_type=progress_note"}
    assert report.extraction_f1_gap == pytest.approx(0.0)
