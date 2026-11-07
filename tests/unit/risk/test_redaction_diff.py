"""Tests for policy-aware, value-free redaction summary diffs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast, overload

import pytest

from openmed.risk import (
    CountChange,
    RedactionDiff,
    diff_redaction_results,
    diff_redaction_summaries,
    fingerprint_policy,
    render_redaction_diff,
)


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-exception")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-exception")

    def __len__(self) -> int:
        return 1


class _EndlessCategories(Sequence[object]):
    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        if isinstance(index, slice):
            return [{"category": "PERSON", "count": 1}]
        return {"category": "PERSON", "count": 1}

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[object]:
        while True:
            yield {"category": "PERSON", "count": 1}


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
    before_summary = cast(dict[str, object], before["summary"])
    second = diff_redaction_summaries(
        {
            **before,
            "summary": {
                **before_summary,
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


def test_rejects_value_bearing_or_ambiguous_summary_fields() -> None:
    invalid_inputs = (
        {"text": "synthetic-sensitive-value"},
        {
            "categories": [
                {
                    "category": "PERSON",
                    "count": 1,
                    "surface": "synthetic-sensitive-value",
                }
            ]
        },
        {"action_counts": {"mask": {"text": "synthetic-sensitive-value"}}},
        {"action_counts": {"mask": 1}, "actions": {"mask": 1}},
        {
            "policy_fingerprint": "sha256:" + "a" * 64,
            "policy": "strict_no_leak",
        },
    )

    for invalid in invalid_inputs:
        with pytest.raises(ValueError) as exc_info:
            diff_redaction_summaries(invalid, {})
        assert "synthetic-sensitive-value" not in str(exc_info.value)


def test_sanitizes_custom_container_failures_and_bounds_iteration() -> None:
    with pytest.raises(ValueError) as exc_info:
        diff_redaction_summaries(_ExplodingMapping(), {})
    assert "synthetic-sensitive-exception" not in str(exc_info.value)

    with pytest.raises(ValueError, match="container limits"):
        diff_redaction_summaries({"categories": _EndlessCategories()}, {})

    nested_counts: dict[str, object] = {"total": 1}
    for _ in range(18):
        nested_counts = {"nested": nested_counts}
    with pytest.raises(ValueError, match="nesting limits"):
        diff_redaction_summaries({"counts": nested_counts}, {})


def test_json_input_is_bounded_and_rejects_duplicate_keys(tmp_path: Path) -> None:
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"counts":{"total":1,"total":2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        diff_redaction_summaries(duplicate_path, {})

    oversized_path = tmp_path / "oversized.json"
    oversized_path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="file limits"):
        diff_redaction_summaries(oversized_path, {})


def test_public_result_records_enforce_value_free_invariants() -> None:
    with pytest.raises(ValueError, match="aggregate identifier"):
        CountChange(
            key="synthetic-sensitive-value",
            before=1,
            after=2,
            delta=1,
            classification="increased",
        )
    with pytest.raises(ValueError, match="delta"):
        CountChange(
            key="mask",
            before=1,
            after=2,
            delta=2,
            classification="increased",
        )

    category_change = CountChange(
        key="PERSON",
        before=1,
        after=2,
        delta=1,
        classification="increased",
    )
    with pytest.raises(ValueError, match="wrong dimension"):
        RedactionDiff(
            before_policy_fingerprint=None,
            after_policy_fingerprint=None,
            action_changes=(category_change,),
            category_changes=(),
            count_changes=(),
        )
