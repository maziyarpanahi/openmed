"""Focused tests for the G12 gold-corpus agreement quality release gate."""

from __future__ import annotations

import pytest

from openmed.eval import release_gates
from openmed.eval.release_gates import ReleaseGate
from openmed.eval.report import GoldCorpusQualityReport


def _quality_report(mean_kappa: float) -> GoldCorpusQualityReport:
    return GoldCorpusQualityReport(
        n_documents=2,
        overall_agreement=mean_kappa,
        per_label={"PERSON": 1.0},
        relation_agreement={},
        relation_types={},
        adjudication_coverage=1.0,
        low_agreement_examples=(),
    )


def _g12(report) -> release_gates.GateCheck:
    return next(check for check in report.gate_results if check.gate == "G12")


def test_g12_passes_at_configured_agreement_floor() -> None:
    check = release_gates.evaluate_gold_corpus_agreement_gate(
        _quality_report(release_gates.G12_MIN_AGREEMENT_KAPPA)
    )

    assert check.passed is True
    assert check.reason == "ok"
    assert check.details == {
        "evidence_present": True,
        "floor": release_gates.G12_MIN_AGREEMENT_KAPPA,
        "mean_agreement_kappa": release_gates.G12_MIN_AGREEMENT_KAPPA,
        "metric_source": "candidate.overall_agreement",
        "threshold": release_gates.G12_MIN_AGREEMENT_KAPPA,
    }


def test_g12_fails_below_configured_agreement_floor() -> None:
    observed = release_gates.G12_MIN_AGREEMENT_KAPPA - 0.001

    check = release_gates.evaluate_gold_corpus_agreement_gate(
        {"gold_corpus_quality": {"mean_agreement_kappa": observed}}
    )

    assert check.passed is False
    assert check.details["mean_agreement_kappa"] == pytest.approx(observed)
    assert check.details["metric_source"] == (
        "candidate.gold_corpus_quality.mean_agreement_kappa"
    )
    assert check.details["threshold"] == release_gates.G12_MIN_AGREEMENT_KAPPA


def test_g12_missing_evidence_is_deterministic_not_applicable() -> None:
    first = release_gates.evaluate_gold_corpus_agreement_gate({})
    second = release_gates.evaluate_gold_corpus_agreement_gate({})

    assert first == second
    assert first.passed is True
    assert first.reason == "not provided"
    assert first.details == {
        "evidence_present": False,
        "floor": release_gates.G12_MIN_AGREEMENT_KAPPA,
        "metric_source": None,
        "threshold": release_gates.G12_MIN_AGREEMENT_KAPPA,
    }


@pytest.mark.parametrize(
    "value", (None, True, "0.9", float("nan"), -1.01, 1.01, 10**400)
)
def test_g12_fails_closed_for_malformed_supplied_evidence(value: object) -> None:
    check = release_gates.evaluate_gold_corpus_agreement_gate(
        {"gold_quality_report": {"overall_agreement": value}}
    )

    assert check.passed is False
    assert check.details["evidence_present"] is True
    assert check.details["metric_source"] == (
        "candidate.gold_quality_report.overall_agreement"
    )


def test_release_gate_emits_g12_from_nested_candidate_metrics(tmp_path) -> None:
    candidate = {
        "suite": "gold-quality",
        "model_name": "unit-model",
        "device": "cpu",
        "fixture_count": 2,
        "metrics": {
            "gold_corpus_quality": {
                "overall_agreement": release_gates.G12_MIN_AGREEMENT_KAPPA - 0.1
            }
        },
        "metadata": {
            "repo_id": "OpenMed/unit-model",
            "family": "PII",
            "tier": "Tiny",
            "format": "mlx-fp",
            "eval_set_hash": "sha256:eval",
            "leakage_fixture_hash": "sha256:leakage",
        },
    }

    result = ReleaseGate(
        baseline_path=tmp_path / "missing-baselines.json",
        signing_key="unit-g12-key",
    ).preview(candidate, {})

    check = _g12(result)
    assert check.passed is False
    assert check.details["metric_source"] == (
        "metrics.gold_corpus_quality.overall_agreement"
    )
