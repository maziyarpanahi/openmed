"""Language strata uncertainty and underpower negative controls."""

import pytest

from openmed.eval.language_uncertainty import (
    LanguageSlice,
    StratumCounts,
    language_uncertainty_report,
)


def item(counts, language="hi", label="CRITICAL"):
    return LanguageSlice(
        language, label, {name: StratumCounts(*value) for name, value in counts.items()}
    )


def test_tiny_perfect_slice_is_not_ranked():
    result = language_uncertainty_report([item({"tiny": (5, 5)})])["slices"][0]
    assert result["estimate"] == 1
    assert result["stratified_confidence_interval"][0] < 1
    assert result["ranking_allowed"] is False


def test_large_common_stratum_does_not_hide_small_critical_stratum():
    result = language_uncertainty_report(
        [item({"common": (9000, 10000), "rare": (2, 2)})]
    )["slices"][0]
    assert result["evidence"] == "insufficient_evidence"
    assert result["strata"][1]["minimum_detectable_effect"] > 0.1


def test_adequately_powered_strata_can_be_ranked():
    result = language_uncertainty_report(
        [item({"one": (450, 500), "two": (450, 500)})]
    )["slices"][0]
    assert result["ranking_allowed"] is True
    assert result["estimate"] == 0.9
    assert (
        result["stratified_confidence_interval"][0]
        < 0.9
        < result["stratified_confidence_interval"][1]
    )


def test_empty_stratum_has_no_interval_or_fake_perfect_rate():
    result = language_uncertainty_report([item({"empty": (0, 0)})])["slices"][0]
    assert result["estimate"] is None
    assert result["strata"][0]["confidence_interval"] is None
    assert result["ranking_allowed"] is False


def test_normal_mde_known_conservative_control():
    result = language_uncertainty_report([item({"one": (250, 500)})])["slices"][0]
    assert result["strata"][0]["minimum_detectable_effect"] == pytest.approx(
        0.088593928521275, rel=1e-6
    )


def test_language_alias_and_order_invariance():
    first = item({"a": (450, 500)}, language="HI-IN")
    second = item({"a": (450, 500)}, language="te")
    assert first.language == "hi"
    assert language_uncertainty_report([first, second]) == language_uncertainty_report(
        [second, first]
    )
    with pytest.raises(ValueError, match="duplicate"):
        language_uncertainty_report([first, item({"a": (1, 1)})])


@pytest.mark.parametrize(
    "args",
    [
        {"confidence": 1},
        {"power": float("nan")},
        {"desired_effect": 0},
        {"minimum_stratum_trials": True},
    ],
)
def test_invalid_policy_fails(args):
    with pytest.raises(ValueError):
        language_uncertainty_report([], **args)


def test_invalid_counts_and_mutable_mapping_snapshot():
    with pytest.raises(ValueError):
        StratumCounts(2, 1)
    counts = {"a": StratumCounts(5, 5)}
    record = LanguageSlice("hi", "CRITICAL", counts)
    counts["a"] = StratumCounts(0, 5)
    assert record.strata["a"].successes == 5
