"""Offline counterfactual tests using only repository-authored synthetic text."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sdoh import SDOHFinding
from openmed.eval.sdoh_counterfactuals import (
    SDOHCounterfactualPair,
    SDOHCounterfactualReport,
    evaluate_sdoh_counterfactuals,
    generate_sdoh_counterfactual_pairs,
    require_sdoh_counterfactual_invariance,
)


def _finding(*, category: str = "employment", score: float = 0.8) -> SDOHFinding:
    return SDOHFinding(
        category=category,
        value="employed",
        status="employed",
        extent=None,
        temporality="recent",
        span=(0, 1),
        score=score,
    )


def test_pairs_preserve_determinant_evidence_and_are_reproducible() -> None:
    pairs = generate_sdoh_counterfactual_pairs(9, seed=11)

    assert pairs == generate_sdoh_counterfactual_pairs(9, seed=11)
    assert pairs != generate_sdoh_counterfactual_pairs(9, seed=12)
    for pair in pairs:
        assert pair.synthetic is True
        assert pair.baseline_text != pair.counterfactual_text
        assert (
            pair.baseline_text.split("Social History:\n", 1)[1]
            == (pair.counterfactual_text.split("Social History:\n", 1)[1])
        )


def test_real_local_extractor_is_invariant_on_synthetic_pairs() -> None:
    pairs = generate_sdoh_counterfactual_pairs(9, seed=11)

    report = require_sdoh_counterfactual_invariance(pairs)

    assert report.pair_count == 9
    assert report.invariant_pairs == 9
    assert report.invariance_rate == 1.0
    assert report.mismatch_by_category == ()
    assert json.loads(json.dumps(report.to_dict())) == report.to_dict()
    assert "Social History" not in str(report.to_dict())


def test_changed_label_and_confidence_are_counted_without_text() -> None:
    pairs = generate_sdoh_counterfactual_pairs(2, seed=2)

    def biased_extractor(text: str) -> list[SDOHFinding]:
        if "age 76" in text:
            return [_finding(category="housing")]
        return [_finding()]

    report = evaluate_sdoh_counterfactuals(pairs, extractor=biased_extractor)
    assert report.label_mismatch_pairs == 2
    assert report.confidence_mismatch_pairs == 0
    assert report.invariant_pairs == 0
    assert report.mismatch_by_category == (("employment", 2), ("other", 2))
    for pair in pairs:
        assert pair.baseline_text not in str(report.to_dict())
        assert pair.counterfactual_text not in str(report.to_dict())

    with pytest.raises(ValueError, match="invariance gate failed"):
        require_sdoh_counterfactual_invariance(pairs, extractor=biased_extractor)

    def score_biased_extractor(text: str) -> list[SDOHFinding]:
        return [_finding(score=0.9 if "age 76" in text else 0.8)]

    score_report = evaluate_sdoh_counterfactuals(
        pairs, extractor=score_biased_extractor
    )
    assert score_report.label_mismatch_pairs == 0
    assert score_report.confidence_mismatch_pairs == 2
    assert score_report.mismatch_by_category == (("employment", 2),)


def test_rejects_nonsynthetic_inputs_and_sanitizes_extractor_errors() -> None:
    pair = SDOHCounterfactualPair("private baseline", "private variant", False)
    with pytest.raises(ValueError, match="must be synthetic"):
        evaluate_sdoh_counterfactuals((pair,))

    def leaking_extractor(_text: str) -> list[SDOHFinding]:
        raise RuntimeError("private baseline")

    safe_pair = SDOHCounterfactualPair("private baseline", "private variant")
    with pytest.raises(ValueError, match="extraction failed") as error:
        evaluate_sdoh_counterfactuals((safe_pair,), extractor=leaking_extractor)
    assert "private baseline" not in str(error.value)


def test_empty_evaluation_and_invalid_generator_count() -> None:
    report = evaluate_sdoh_counterfactuals(())
    assert report.pair_count == 0
    assert report.invariance_rate == 1.0
    with pytest.raises(ValueError, match="count"):
        generate_sdoh_counterfactual_pairs(-1)
    with pytest.raises(ValueError, match="category"):
        SDOHCounterfactualReport(1, 0, 0, (("private-identifier", 1),))
