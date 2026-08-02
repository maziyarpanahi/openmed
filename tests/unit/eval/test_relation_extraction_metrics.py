"""Tests for relation-extraction metrics, harness, and scorecards."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.eval.datasets import DRUGPROT, corpus_from_rows
from openmed.eval.harness import (
    RELATION_SCORECARD_ARTIFACT,
    RelationGateFailure,
    RelationScorecard,
    run_relation_benchmark,
    run_relation_suite,
    run_suite,
)
from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import (
    EvalRelation,
    compute_relaxed_relation_f1,
    compute_strict_relation_f1,
)
from openmed.eval.scorecard import ModelScorecard
from openmed.eval.suites.relations import DEFAULT_RELATION_GOLD_PATH

RELATION_SIGNING_KEY = "synthetic-relation-scorecard-key"


def test_strict_and_relaxed_relation_f1_cover_match_errors() -> None:
    gold = [
        _relation("INHIBITOR", 0, 7, 17, 21),
        _relation("ACTIVATOR", 22, 31, 42, 46),
        _relation("ANTAGONIST", 50, 56, 60, 64),
    ]
    predicted = [
        _relation("INHIBITOR", 0, 7, 17, 21),
        _relation("ACTIVATOR", 23, 31, 42, 46),
        _relation("INHIBITOR", 50, 56, 60, 64),
        _relation("ANTAGONIST", 50, 56, 65, 70),
    ]

    strict = compute_strict_relation_f1(gold, predicted)
    relaxed = compute_relaxed_relation_f1(gold, predicted)

    assert strict.true_positives == 1
    assert strict.false_positives == 3
    assert strict.false_negatives == 2
    assert strict.precision == pytest.approx(1 / 4)
    assert strict.recall == pytest.approx(1 / 3)
    assert strict.f1 == pytest.approx(2 / 7)
    assert relaxed.true_positives == 2
    assert relaxed.false_positives == 2
    assert relaxed.false_negatives == 1
    assert relaxed.f1 == pytest.approx(4 / 7)


def test_relation_harness_scorecard_uses_in_memory_drugprot_rows() -> None:
    corpus = corpus_from_rows(
        [("DPX", "Aspirin inhibits TP53", "Metformin activates EGFR")],
        [
            ("DPX", "T1", "CHEMICAL", "0", "7", "Aspirin"),
            ("DPX", "T2", "GENE", "17", "21", "TP53"),
            ("DPX", "T3", "CHEMICAL", "22", "31", "Metformin"),
            ("DPX", "T4", "GENE", "42", "46", "EGFR"),
        ],
        [
            ("DPX", "INHIBITOR", "Arg1:T1", "Arg2:T2"),
            ("DPX", "ACTIVATOR", "Arg1:T3", "Arg2:T4"),
        ],
        source_path="<memory>",
    )
    fixtures = corpus.to_relation_fixtures()

    def runner(fixture, model_name, device):
        assert fixture.fixture_id == "DPX"
        assert model_name == "relation-model"
        assert device == "cpu"
        return fixture.relations

    report = run_relation_benchmark(
        fixtures,
        suite=DRUGPROT,
        model_name="relation-model",
        runner=runner,
        ci_resamples=20,
        ci_seed=11,
    )

    metrics = report.metrics["relation_extraction"]
    assert metrics["strict"]["f1"] == 1.0
    assert metrics["strict"]["confidence_interval"]["lower"] == 1.0
    assert metrics["relaxed"]["f1"] == 1.0
    assert set(metrics["per_relation_type"]) == {"ACTIVATOR", "INHIBITOR"}

    scorecard = ModelScorecard.from_reports([report])
    row = scorecard.to_dict()["device_tiers"][0]
    assert row["relation_strict_f1"] == 1.0
    assert row["relation_relaxed_f1"] == 1.0
    assert row["relation_per_type_f1"]["INHIBITOR"]["strict"] == 1.0
    markdown = scorecard.to_markdown()
    assert "Strict RE-F1" in markdown
    assert "INHIBITOR: strict 100.00%, relaxed 100.00%" in markdown


def test_relation_suite_emits_signed_scorecard_and_model_card_evidence(
    tmp_path,
) -> None:
    json_path = tmp_path / "relation-scorecard.json"
    markdown_path = tmp_path / "relation-scorecard.md"

    scorecard = run_relation_suite(
        model_name="synthetic-relation-model",
        runner=lambda fixture, _model, _device: fixture.gold_relations,
        output_json=json_path,
        output_markdown=markdown_path,
        generated_at="2026-08-02T00:00:00Z",
        signing_key=RELATION_SIGNING_KEY,
        ci_resamples=20,
        ci_seed=19,
    )

    restored = RelationScorecard.from_json(json_path.read_text(encoding="utf-8"))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert scorecard.verify(RELATION_SIGNING_KEY)
    assert restored.verify(RELATION_SIGNING_KEY)
    assert payload["artifact_type"] == RELATION_SCORECARD_ARTIFACT
    assert payload["gate_passed"] is True
    assert payload["metrics"]["strict"]["f1"] == 1.0
    assert payload["metrics"]["relaxed"]["f1"] == 1.0
    assert set(payload["metrics"]["by_type"]) == {
        "ASSERTED_ABSENT",
        "TEMPORALLY_BEFORE",
        "TREATS",
    }
    assert payload["metrics"]["trap_leaks"]["assertion"]["leak_count"] == 0
    assert payload["metrics"]["trap_leaks"]["temporal"]["leak_count"] == 0
    assert payload["metrics"]["consistency"]["assertion"] == {
        "evaluated_relation_count": 1,
        "leak_count": 0,
        "score": 1.0,
    }
    assert payload["metrics"]["consistency"]["temporal"] == {
        "evaluated_relation_count": 2,
        "leak_count": 0,
        "score": 1.0,
    }
    assert payload["provenance"]["fixture_set_hash"].startswith("sha256:")
    assert len(payload["provenance"]["fixture_hashes"]) == 3
    assert "Strict" in markdown
    assert "Zero-Tolerance Trap Summary" in markdown
    assert "| `assertion` | 1 | 0 | 100.00% | yes |" in markdown

    evidence = scorecard.model_card_evidence()["relation_scorecard"]
    assert evidence["repro_hash"] == scorecard.repro_hash
    assert evidence["signature"]["key_id"] == "relation-scorecard"
    benchmark_report = scorecard.to_benchmark_report()
    assert benchmark_report.metadata["fixture_hashes"]
    assert benchmark_report.metadata["relation_traps"]["total"] == 2
    assert benchmark_report.metadata["relation_trap_leaks"]["assertion"] == {
        "evaluated_relation_count": 1,
        "leak_count": 0,
        "leaked_relation_hashes": [],
        "trap_count": 1,
    }
    model_scorecard = ModelScorecard.from_reports([benchmark_report])
    row = model_scorecard.to_dict()["device_tiers"][0]
    assert row["relation_strict_f1"] == 1.0
    assert "Aspirin treats fever" not in scorecard.to_json() + markdown


@pytest.mark.parametrize(
    ("fixture_id", "conflicting_type", "trap_kind", "expected_score"),
    [
        ("relation-assertion-negated", "ASSERTED_PRESENT", "assertion", 0.0),
        ("relation-document-temporal", "TEMPORALLY_AFTER", "temporal", 0.5),
    ],
)
def test_relation_suite_runner_writes_signed_failure_before_propagating(
    tmp_path,
    monkeypatch,
    fixture_id,
    conflicting_type,
    trap_kind,
    expected_score,
) -> None:
    json_path = tmp_path / "failed-relation-scorecard.json"
    markdown_path = tmp_path / "failed-relation-scorecard.md"
    monkeypatch.setenv("OPENMED_RELATION_SCORECARD_KEY", RELATION_SIGNING_KEY)

    def conflicting_runner(fixture, _model_name, _device):
        predictions = list(fixture.gold_relations)
        if fixture.fixture_id == fixture_id:
            predictions[0] = replace(
                predictions[0],
                relation_type=conflicting_type,
            )
        return predictions

    with pytest.raises(RelationGateFailure) as captured:
        run_suite(
            DEFAULT_RELATION_GOLD_PATH,
            suite="relations",
            model_name="synthetic-conflicting-relations",
            runner=conflicting_runner,
            output_json=json_path,
            output_markdown=markdown_path,
            ci_resamples=20,
            ci_seed=23,
        )

    failure = captured.value.scorecard
    restored = RelationScorecard.from_json(json_path.read_text(encoding="utf-8"))
    assert failure.verify(RELATION_SIGNING_KEY)
    assert restored.verify(RELATION_SIGNING_KEY)
    assert failure.gate_passed is False
    assert failure.gate_result["reason"] == (
        "zero-tolerance assertion or temporal trap leak"
    )
    assert failure.metrics["trap_leaks"][trap_kind]["leak_count"] == 1
    assert failure.metrics["consistency"][trap_kind]["score"] == expected_score
    assert "| Relation gate | failed |" in markdown_path.read_text(encoding="utf-8")


def _relation(
    relation_type: str,
    arg1_start: int,
    arg1_end: int,
    arg2_start: int,
    arg2_end: int,
) -> EvalRelation:
    return EvalRelation(
        relation_type=relation_type,
        head=EvalSpan(start=arg1_start, end=arg1_end, label="OTHER"),
        tail=EvalSpan(start=arg2_start, end=arg2_end, label="OTHER"),
    )
