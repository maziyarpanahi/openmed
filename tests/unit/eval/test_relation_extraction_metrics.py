"""Tests for relation-extraction metrics, harness, and scorecards."""

from __future__ import annotations

import pytest

from openmed.eval.datasets import DRUGPROT, corpus_from_rows
from openmed.eval.harness import run_relation_benchmark
from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import (
    EvalRelation,
    compute_relaxed_relation_f1,
    compute_strict_relation_f1,
)
from openmed.eval.scorecard import ModelScorecard


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
