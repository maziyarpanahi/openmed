"""Synthetic offline tests for SDOH category completeness audits."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.clinical.sdoh_completeness import (
    MISSING_PROCESSING_RESULT,
    SDOHCategoryResult,
    SDOHCategoryState,
    audit_sdoh_completeness,
)


def test_audit_reports_every_configured_category_and_all_states() -> None:
    audit = audit_sdoh_completeness(
        ("housing", "food", "transportation", "utilities"),
        (
            SDOHCategoryResult(
                category="housing",
                state=SDOHCategoryState.PROCESSED,
                finding_count=1,
            ),
            SDOHCategoryResult(
                category="food",
                state=SDOHCategoryState.SKIPPED,
                reason_code="policy_disabled",
            ),
            SDOHCategoryResult(
                category="transportation",
                state=SDOHCategoryState.UNSUPPORTED,
                reason_code="extractor_unavailable",
            ),
            SDOHCategoryResult(
                category="utilities",
                state=SDOHCategoryState.FAILED,
                reason_code="extractor_error",
            ),
        ),
    )
    payload = audit.to_dict()

    assert [item["category"] for item in payload["categories"]] == [
        "food",
        "housing",
        "transportation",
        "utilities",
    ]
    assert payload["state_counts"] == {
        "processed": 1,
        "skipped": 1,
        "unsupported": 1,
        "failed": 1,
    }


def test_processed_zero_findings_is_unmentioned_not_negative() -> None:
    audit = audit_sdoh_completeness(
        ("housing",),
        (
            SDOHCategoryResult(
                category="housing",
                state=SDOHCategoryState.PROCESSED,
            ),
        ),
    )

    category = audit.to_dict()["categories"][0]
    assert category["finding_count"] == 0
    assert category["absence_interpretation"] == "unmentioned_not_negative"
    assert "negative" not in category["state"]


def test_missing_processing_result_is_explicit_failure() -> None:
    audit = audit_sdoh_completeness(("food", "housing"), ())

    assert all(item.state is SDOHCategoryState.FAILED for item in audit.categories)
    assert audit.to_dict()["reason_counts"] == {MISSING_PROCESSING_RESULT: 2}


def test_counts_and_reason_codes_are_value_free() -> None:
    audit = audit_sdoh_completeness(
        ("housing",),
        (
            SDOHCategoryResult(
                category="housing",
                state=SDOHCategoryState.SKIPPED,
                reason_code="consent_unavailable",
            ),
        ),
    )
    rendered = json.dumps(audit.to_dict(), sort_keys=True)

    assert "raw" not in rendered.lower()
    assert "consent_unavailable" in rendered


def test_unconfigured_category_is_rejected_without_echoing_code() -> None:
    result = SDOHCategoryResult(
        category="private_marker",
        state=SDOHCategoryState.PROCESSED,
    )

    with pytest.raises(ValueError, match="unconfigured category") as error:
        audit_sdoh_completeness(("housing",), (result,))

    assert "private_marker" not in str(error.value)


def test_audit_is_input_order_independent() -> None:
    categories = ("housing", "food")
    results = (
        SDOHCategoryResult("housing", SDOHCategoryState.PROCESSED, 1),
        SDOHCategoryResult("food", SDOHCategoryState.PROCESSED, 0),
    )

    assert audit_sdoh_completeness(categories, results).to_dict() == (
        audit_sdoh_completeness(reversed(categories), reversed(results)).to_dict()
    )


def test_completeness_audit_performs_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    assert audit_sdoh_completeness(("housing",), ()).categories
