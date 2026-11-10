"""Tests for the structured-table membership-inference self-test."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    MembershipInferenceResult,
    membership_inference_self_test,
    run_membership_inference_self_test,
)


def _table(prefix: str, *, marker: str | None = None) -> list[dict[str, object]]:
    rows = []
    for index in range(8):
        row: dict[str, object] = {
            "record_id": f"{prefix}-{index}",
            "age": 30 + index % 4,
            "site": ("north", "south")[index % 2],
            "cohort": ("a", "b")[index % 2],
        }
        if marker is not None:
            row["release_marker"] = marker
        rows.append(row)
    return rows


def test_well_anonymized_tables_stay_near_chance_and_pass_budget() -> None:
    result = membership_inference_self_test(
        _table("synthetic"),
        _table("heldout"),
        advantage_budget=0.05,
    )

    assert isinstance(result, MembershipInferenceResult)
    assert result.auc == pytest.approx(0.5)
    assert result.accuracy == pytest.approx(0.5)
    assert result.advantage == pytest.approx(0.0)
    assert result.passed is True
    assert result.member_count == 8
    assert result.heldout_count == 8


def test_leaky_table_has_elevated_advantage_and_riskiest_ids() -> None:
    result = membership_inference_self_test(
        _table("synthetic", marker="leaked-cohort"),
        _table("heldout", marker="redacted"),
        advantage_budget=0.05,
        top_k=3,
    )

    assert result.auc > 0.9
    assert result.advantage > 0.4
    assert result.passed is False
    assert len(result.riskiest_records) == 3
    assert all(
        item["record_id"].startswith("synthetic-") for item in result.riskiest_records
    )
    assert all(set(item) == {"record_id", "score"} for item in result.riskiest_records)


def test_release_proximity_path_detects_member_rows_without_raw_output() -> None:
    released = _table("release", marker="leaked-cohort")
    members = _table("member", marker="leaked-cohort")
    heldout = _table("heldout", marker="redacted")

    result = run_membership_inference_self_test(
        released,
        member_records=members,
        heldout_records=heldout,
    )
    serialized = result.to_json()

    assert result.mode == "release_proximity"
    assert result.advantage > 0.4
    assert "release_marker" not in serialized
    assert "leaked-cohort" not in serialized
    assert "member-0" in serialized
    assert "heldout-0" in serialized


def test_serialized_report_contains_only_safe_record_references() -> None:
    result = membership_inference_self_test(
        _table("synthetic", marker="leaked-cohort"),
        _table("heldout", marker="redacted"),
    )

    payload = result.to_dict()
    assert json.loads(result.to_json()) == payload
    assert "release_marker" not in json.dumps(payload)
    assert "leaked-cohort" not in json.dumps(payload)
    assert all(
        set(item) == {"record_id", "score"} for item in payload["riskiest_records"]
    )


def test_advantage_budget_and_input_validation_are_fail_closed() -> None:
    leaky = (
        _table("synthetic", marker="leaked"),
        _table("heldout", marker="safe"),
    )

    assert membership_inference_self_test(*leaky, advantage_budget=0.5).passed is True
    with pytest.raises(ValueError, match="between 0 and 0.5"):
        membership_inference_self_test(*leaky, advantage_budget=0.51)
    with pytest.raises(ValueError, match="non-negative"):
        membership_inference_self_test(*leaky, top_k=-1)
    with pytest.raises(ValueError, match="heldout records"):
        membership_inference_self_test(_table("synthetic"))
