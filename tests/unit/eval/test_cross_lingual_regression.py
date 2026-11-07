"""Paired multilingual release gating and non-offsetting regression controls."""

import pytest

from openmed.eval.cross_lingual_regression import (
    PairedTransferCell,
    cross_lingual_regression_gate,
    require_cross_lingual_release,
)


def cell(
    language="hi",
    *,
    base=100,
    candidate=100,
    lost=0,
    gained=0,
    critical=True,
    trials=100,
):
    return PairedTransferCell(
        language, "ID", "0" * 64, trials, base, candidate, lost, gained, critical
    )


def options(rows):
    return dict(
        expected_baseline_cells={
            (row.language, row.label): ("0" * 64, row.critical) for row in rows
        },
        baseline_checkpoint_digest="1" * 64,
        candidate_checkpoint_digest="2" * 64,
    )


def gate(rows):
    return cross_lingual_regression_gate(rows, **options(rows))


def test_positive_target_transfer_cannot_offset_supported_language_critical_decline():
    result = gate(
        [cell(base=50, candidate=70, gained=20), cell("te", candidate=99, lost=1)]
    )
    assert result["release_allowed"] is False
    assert result["cells"][0]["classification"] == "positive"
    assert result["cells"][1]["classification"] == "inconclusive"
    assert result["cells"][1]["critical_regression"] is True


def test_statistically_negative_cell_is_separate_from_inconclusive():
    result = gate([cell(candidate=80, lost=20)])
    assert result["cells"][0]["classification"] == "negative"
    assert result["cells"][0]["paired_p_value"] == pytest.approx(2**-19)


def test_unchanged_last_green_can_qualify_without_positive_transfer_claim():
    result = gate([cell()])
    assert result["release_allowed"] is True
    assert result["cells"][0]["classification"] == "inconclusive"


def test_missing_language_is_a_release_failure():
    row = cell()
    args = options([row])
    args["expected_baseline_cells"][("te", "ID")] = ("0" * 64, True)
    assert cross_lingual_regression_gate([row], **args)["release_allowed"] is False


def test_mismatched_cohort_and_unexpected_cell_fail_closed():
    row = cell()
    args = options([row])
    args["expected_baseline_cells"][("hi", "ID")] = ("f" * 64, True)
    assert cross_lingual_regression_gate([row], **args)["failed_cells"] == [
        ["hi", "ID"]
    ]
    args["expected_baseline_cells"] = {}
    with pytest.raises(ValueError):
        cross_lingual_regression_gate([row], **args)


def test_small_or_empty_slices_cannot_qualify():
    assert gate([cell(base=5, candidate=5, trials=5)])["release_allowed"] is False
    assert gate([cell(base=0, candidate=0, trials=0)])["release_allowed"] is False


def test_promotion_api_actually_refuses_regression():
    rows = [cell(candidate=99, lost=1)]
    with pytest.raises(ValueError, match="qualification failed"):
        require_cross_lingual_release(rows, **options(rows))


def test_inconsistent_paired_counts_are_rejected():
    with pytest.raises(ValueError):
        cell(base=100, candidate=90, lost=0)


def test_duplicate_cell_and_aliases_fail():
    rows = [cell(), cell("HI-IN")]
    with pytest.raises(ValueError, match="duplicate"):
        gate(rows)


def test_report_is_order_invariant():
    rows = [cell(), cell("te")]
    assert gate(rows) == gate(rows[::-1])


def test_large_discordance_is_explicitly_inconclusive_without_unbounded_work():
    result = gate([cell(base=10000, candidate=5000, lost=5000, trials=10000)])
    assert result["cells"][0]["paired_p_value"] is None
    assert result["cells"][0]["statistical_evidence"] == "computation_budget_exceeded"
    assert result["cells"][0]["classification"] == "inconclusive"
    assert result["release_allowed"] is False


def test_candidate_cannot_remove_trusted_baseline_critical_designation():
    from dataclasses import replace

    original = cell(candidate=99, lost=1)
    result = cross_lingual_regression_gate(
        [replace(original, critical=False)], **options([original])
    )
    assert result["release_allowed"] is False
    assert result["failed_cells"] == [["hi", "ID"]]
    assert result["cells"][0]["baseline_aligned"] is False


def test_empty_baseline_policy_is_rejected_before_comparison():
    args = options([cell()])
    args["expected_baseline_cells"] = {}
    with pytest.raises(ValueError, match="baseline policy"):
        cross_lingual_regression_gate([cell()], **args)
