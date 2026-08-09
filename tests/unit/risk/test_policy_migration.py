"""Tests for deterministic, privacy-safe policy migration checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.risk import (
    MigrationClassification,
    PolicyMigrationAcknowledgementRequired,
    check_policy_migration,
    compare_policy_versions,
)


def _policy(*, action: str = "mask") -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "synthetic-policy",
        "default_action": action,
        "safety_sweep_mandatory": True,
        "actions": {
            "EMAIL": action,
            "PHONE": action,
        },
    }


def test_equal_policy_is_compatible_and_deterministic() -> None:
    policy = _policy()

    first = compare_policy_versions(policy, copy.deepcopy(policy))
    second = compare_policy_versions(policy, copy.deepcopy(policy))

    assert first.classification == MigrationClassification.COMPATIBLE
    assert first.approved is True
    assert first.changes == ()
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_stronger_redaction_action_is_classified_as_stricter() -> None:
    before = _policy(action="mask")
    after = _policy(action="redact")

    report = compare_policy_versions(before, after)

    assert report.classification == "stricter"
    assert report.requires_acknowledgement is False
    assert report.approved is True
    assert {change.path_key for change in report.changes} == {
        "actions.EMAIL",
        "actions.PHONE",
        "default_action",
    }
    assert all(
        change.classification == MigrationClassification.STRICTER
        for change in report.changes
    )


def test_weaker_action_requires_report_bound_human_acknowledgement() -> None:
    before = _policy(action="redact")
    after = _policy(action="mask")

    report = compare_policy_versions(before, after)

    assert report.classification == MigrationClassification.INCOMPATIBLE
    assert report.requires_acknowledgement is True
    assert report.approved is False
    assert report.acknowledgement_token is not None
    with pytest.raises(PolicyMigrationAcknowledgementRequired) as exc_info:
        check_policy_migration(before, after)
    assert exc_info.value.report.to_dict() == report.to_dict()

    acknowledged = check_policy_migration(
        before,
        after,
        acknowledgement_token=report.acknowledgement_token,
    )
    assert acknowledged.acknowledged is True
    assert acknowledged.approved is True


def test_weaker_safety_setting_is_gated() -> None:
    before = _policy()
    after = copy.deepcopy(before)
    after["safety_sweep_mandatory"] = False

    report = compare_policy_versions(before, after)

    assert report.classification == "incompatible"
    assert report.weakened_changes[0].path_key == "safety_sweep_mandatory"
    assert report.weakened_changes[0].kind == "boolean"


def test_metadata_change_is_compatible_without_copying_arbitrary_values() -> None:
    before = _policy()
    after = copy.deepcopy(before)
    metadata = {"review_note": "synthetic-review-marker"}
    before["metadata"] = metadata
    after["metadata"] = {"review_note": "synthetic-review-marker-updated"}

    report = compare_policy_versions(before, after)
    serialized = json.dumps(report.to_dict(), sort_keys=True)

    assert report.classification == "compatible"
    assert "synthetic-review-marker" not in serialized
    assert "synthetic-review-marker-updated" not in serialized


def test_unknown_rule_values_are_reported_without_raw_values() -> None:
    before = {
        "version": 1,
        "rules": {"EMAIL": {"pattern": "synthetic-pattern-a"}},
    }
    after = {
        "version": 2,
        "rules": {"EMAIL": {"pattern": "synthetic-pattern-b"}},
    }

    report = compare_policy_versions(before, after)
    serialized = report.to_json()

    assert report.classification == MigrationClassification.INCOMPATIBLE
    assert "synthetic-pattern-a" not in serialized
    assert "synthetic-pattern-b" not in serialized


def test_local_json_paths_are_supported_without_network_access(tmp_path: Path) -> None:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(_policy()), encoding="utf-8")
    after_path.write_text(json.dumps(_policy(action="redact")), encoding="utf-8")

    report = compare_policy_versions(before_path, after_path)

    assert report.classification == "stricter"
    assert report.before_digest.startswith("sha256:")
    assert report.after_digest.startswith("sha256:")


def test_invalid_input_errors_do_not_include_input_value() -> None:
    with pytest.raises(ValueError) as exc_info:
        compare_policy_versions("not-a-policy-value", {"default_action": "mask"})

    assert "not-a-policy-value" not in str(exc_info.value)
