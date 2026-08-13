"""Tests for policy-aware, value-free redaction summary diffs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.risk import (
    diff_redaction_results,
    diff_redaction_summaries,
    fingerprint_policy,
    render_redaction_diff,
)


def _summary(
    policy: str,
    *,
    actions: dict[str, int],
    categories: dict[str, int],
    counts: dict[str, int],
) -> dict[str, object]:
    return {
        "policy": policy,
        "action_counts": actions,
        "category_counts": categories,
        "counts": counts,
    }


def test_diff_classifies_action_category_and_count_changes() -> None:
    before = _summary(
        "clinical_minimal_redaction",
        actions={"keep": 1, "mask": 2},
        categories={"LOCATION": 2, "PERSON": 1},
        counts={"redacted": 2, "total": 3},
    )
    after = _summary(
        "strict_no_leak",
        actions={"mask": 3, "redact": 1},
        categories={"LOCATION": 1, "PHONE": 3},
        counts={"redacted": 4, "total": 4},
    )

    diff = diff_redaction_summaries(before, after)
    payload = diff.to_dict()

    actions = {change["key"]: change for change in payload["action_changes"]}
    assert actions["keep"]["classification"] == "removed"
    assert actions["mask"] == {
        "key": "mask",
        "before": 2,
        "after": 3,
        "delta": 1,
        "classification": "increased",
    }
    assert actions["redact"]["classification"] == "added"

    categories = {change["key"]: change for change in payload["category_changes"]}
    assert categories["LOCATION"]["classification"] == "decreased"
    assert categories["PERSON"]["classification"] == "removed"
    assert categories["PHONE"]["classification"] == "added"

    counts = {change["key"]: change for change in payload["count_changes"]}
    assert counts["redacted"]["delta"] == 2
    assert counts["total"]["classification"] == "increased"
    assert payload["policy_changed"] is True
    assert payload["policy_fingerprints"]["before"] == fingerprint_policy(
        "clinical_minimal_redaction"
    )
    assert payload["policy_fingerprints"]["after"] == fingerprint_policy(
        "strict_no_leak"
    )


def test_diff_is_deterministic_and_renders_only_aggregate_values() -> None:
    before = {
        "policy_fingerprint": "sha256:" + "a" * 64,
        "summary": {
            "action_counts": {"replace": 2, "keep": 1},
            "category_counts": {
                "PERSON": 2,
                "synthetic-sensitive-category": 1,
            },
            "counts": {
                "total": 3,
                "synthetic-sensitive-metric": 1,
            },
        },
    }
    after = {
        "policy_fingerprint": "sha256:" + "b" * 64,
        "summary": {
            "action_counts": {"keep": 1, "replace": 3},
            "category_counts": {
                "synthetic-sensitive-category": 2,
                "PERSON": 2,
            },
            "counts": {
                "synthetic-sensitive-metric": 2,
                "total": 4,
            },
        },
    }

    first = diff_redaction_summaries(before, after)
    second = diff_redaction_summaries(
        {
            **before,
            "summary": {
                **before["summary"],
                "action_counts": {"keep": 1, "replace": 2},
                "category_counts": {
                    "synthetic-sensitive-category": 1,
                    "PERSON": 2,
                },
                "counts": {
                    "synthetic-sensitive-metric": 1,
                    "total": 3,
                },
            },
        },
        after,
    )

    assert first.to_dict() == second.to_dict()
    serialized = first.to_json()
    rendered = render_redaction_diff(first)
    assert json.loads(serialized) == first.to_dict()
    assert "synthetic-sensitive-category" not in serialized
    assert "synthetic-sensitive-metric" not in serialized
    assert "synthetic-sensitive-category" not in rendered
    assert "synthetic-sensitive-metric" not in rendered
    assert "category:sha256:" in serialized
    assert "count:sha256:" in serialized


def test_nested_category_records_derive_action_and_category_counts() -> None:
    before = {
        "policy": "baseline-policy",
        "categories": [
            {
                "category": "PERSON",
                "detection_count": 2,
                "applied_action_counts": {"mask": 2},
            },
            {
                "category": "LOCATION",
                "detection_count": 1,
                "applied_action_counts": {"keep": 1},
            },
        ],
    }
    after = {
        "policy": "candidate-policy",
        "categories": [
            {
                "category": "PERSON",
                "detection_count": 2,
                "applied_action_counts": {"redact": 2},
            },
            {
                "category": "LOCATION",
                "detection_count": 1,
                "applied_action_counts": {"mask": 1},
            },
        ],
    }

    diff = diff_redaction_results(before, after)
    assert diff.category_changes == ()
    assert {change.key for change in diff.action_changes} == {
        "keep",
        "mask",
        "redact",
    }
    assert diff.policy_changed is True


def test_local_json_paths_and_malformed_counts_are_handled_safely(
    tmp_path: Path,
) -> None:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(
        json.dumps(
            _summary("strict_no_leak", actions={"mask": 1}, categories={}, counts={})
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            _summary("strict_no_leak", actions={"mask": 2}, categories={}, counts={})
        ),
        encoding="utf-8",
    )

    diff = diff_redaction_summaries(before_path, after_path)
    assert diff.action_changes[0].delta == 1

    with pytest.raises(ValueError, match="non-negative integers"):
        diff_redaction_summaries(
            {"action_counts": {"mask": -1}},
            {"action_counts": {"mask": 0}},
        )
