"""Synthetic summary gate failures; adjudication rows are test fixtures only."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.eval.summary_gate import evaluate_summary_gate, load_summary_eval_dataset

BASELINE = Path(__file__).resolve().parents[3] / "gates/baseline.json"


def sample():
    text = "Dehydration improved."
    return dict(
        deidentified=DeidentificationResult(
            "Juniper Example: " + text,
            "[NAME]: " + text,
            [
                PIIEntity(
                    text="Juniper Example",
                    label="NAME",
                    start=0,
                    end=15,
                    confidence=1.0,
                    redacted_text="[NAME]",
                )
            ],
            "mask",
            datetime(2026, 1, 1),
        ),
        summary=text,
        source_facts=[("problem", "dehydration")],
        summary_facts=[("problem", "dehydration")],
        source_evidence=[{"evidence_id": "e", "start": 8, "end": 28}],
        claims=[
            {
                "claim_id": "c",
                "claim_class": "finding",
                "evidence_ids": ["e"],
                "citations": [{"evidence_id": "e"}],
            }
        ],
        support_evidence=[
            {
                "evidence_id": "e",
                "claim_id": "c",
                "relation": "supported",
                "approved": True,
            }
        ],
        adjudications=[{"claim_id": "c", "evidence_id": "e", "label": "supports"}],
        thresholds=json.loads(BASELINE.read_text())["summary"],
    )


def test_pass_and_value_free_report():
    checks = evaluate_summary_gate(**sample())
    assert len(checks) == 5
    assert all(check.passed for check in checks), [c.to_dict() for c in checks]
    report = json.dumps([check.to_dict() for check in checks])
    assert "Juniper" not in report and "Dehydration" not in report


@pytest.mark.parametrize(
    "change,gate",
    [
        ({"summary": "Juniper Example improved."}, "summary_leakage"),
        ({"summary_facts": []}, "summary_clinical_fact_recall"),
        ({"adjudications": None}, "summary_citation_support"),
        ({"support_evidence": []}, "summary_unsupported_claim_rate"),
        (
            {
                "claims": [
                    {
                        "claim_id": "c",
                        "claim_class": "finding",
                        "evidence_ids": ["missing"],
                        "citations": [{"evidence_id": "missing"}],
                    }
                ]
            },
            "summary_fact_coverage",
        ),
        ({"source_facts": []}, "summary_evidence"),
    ],
)
def test_gate_rejects(change, gate):
    values = sample()
    values.update(change)
    checks = {c.gate: c for c in evaluate_summary_gate(**values)}
    assert gate in checks, {k: v.to_dict() for k, v in checks.items()}
    assert not checks[gate].passed


@pytest.mark.parametrize("value", [True, float("nan"), -1, 2])
def test_invalid_thresholds_fail_closed(value):
    values = sample()
    values["thresholds"]["fact_coverage_min"] = value
    assert not evaluate_summary_gate(**values)[0].passed


def test_missing_dua_corpus_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENMED_MIMIC_IV_BHC_PATH", raising=False)
    with pytest.raises(Exception, match="credential"):
        load_summary_eval_dataset("mimic-iv-bhc")


def test_seeded_gold_and_actual_extractive_report_are_reproducible():
    from openmed.eval.clinical_fixtures import generate_fixture
    from openmed.eval.summary_benchmark import (
        run_summary_benchmark,
        verify_summary_report,
    )

    root = BASELINE.parent.parent
    fixture_dir = root / "tests/fixtures/eval/summaries"
    seeds = json.loads((fixture_dir / "seeds.json").read_text())["seeds"]
    gold = json.loads((fixture_dir / "discharge-gold.json").read_text())
    assert [
        generate_fixture("discharge_summary", seed=s).to_dict(include_text=True)
        for s in seeds
    ] == gold
    report = run_summary_benchmark(
        "extractive", seeds=seeds, thresholds=sample()["thresholds"]
    ).to_dict()
    assert report == json.loads(
        (root / "eval/suites/summaries/extractive.json").read_text()
    )
    assert report["metrics"]["generated_count"] == 3
    assert not verify_summary_report(report)
    for fixture in gold:
        assert fixture["text"] not in json.dumps(report)


def test_report_verification_rejects_missing_mutated_and_empty_evidence(tmp_path):
    from openmed.eval.summary_benchmark import main, verify_summary_report

    assert not verify_summary_report({})
    report = json.loads(
        (BASELINE.parent.parent / "eval/suites/summaries/mlx.json").read_text()
    )
    assert not verify_summary_report(report)
    report["fixture_count"] = 0
    assert not verify_summary_report(report)
    assert main(["--verify", str(tmp_path / "absent.json")]) == 1


def test_daily_workflow_always_consumes_both_reports():
    import yaml

    workflow = yaml.safe_load(
        (BASELINE.parent.parent / ".github/workflows/release-gates.yml").read_text()
    )
    job = workflow["jobs"]["summary-evidence"]
    command = job["steps"][-1]["run"]
    assert "--verify" in command
    assert "eval/suites/summaries/extractive.json" in command
    assert "eval/suites/summaries/mlx.json" in command
    assert "continue-on-error" not in job["steps"][-1]
