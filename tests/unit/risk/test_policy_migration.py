"""Tests for deterministic, privacy-safe policy migration checks."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from openmed.risk import (
    MigrationClassification,
    PolicyChange,
    PolicyMigrationAcknowledgementRequired,
    PolicyMigrationError,
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

    assert report.acknowledgement_token is not None
    with pytest.raises(PolicyMigrationAcknowledgementRequired):
        report.with_acknowledgement("openmed-ack:" + "0" * 64)
    assert report.with_acknowledgement(report.acknowledgement_token).approved is True


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


@pytest.mark.parametrize("key", ["schema_version", "version"])
def test_schema_version_changes_fail_closed(key: str) -> None:
    before = _policy()
    after = copy.deepcopy(before)
    before[key] = 1
    after[key] = 2

    report = compare_policy_versions(before, after)

    assert report.classification == MigrationClassification.INCOMPATIBLE
    assert any(
        change.reason == "policy schema version changed" for change in report.changes
    )
    assert report.requires_acknowledgement is True


def test_boolean_and_numeric_type_changes_fail_closed() -> None:
    before = _policy()
    before["privacy"] = {"enabled": True, "threshold": 0.5}
    after = copy.deepcopy(before)
    after["privacy"] = {"enabled": "false", "threshold": "0.5"}

    report = compare_policy_versions(before, after)
    reasons = {change.reason for change in report.changes}

    assert report.classification == MigrationClassification.INCOMPATIBLE
    assert "boolean protection setting changed type" in reasons
    assert "numeric protection threshold changed type" in reasons


def test_numeric_protection_type_changes_fail_closed() -> None:
    before = {"schema_version": 1, "privacy": {"threshold": 1}}
    after = {"schema_version": 1, "privacy": {"threshold": 1.0}}

    report = compare_policy_versions(before, after)

    assert report.classification == MigrationClassification.INCOMPATIBLE
    assert report.changes[0].reason == "numeric protection threshold changed type"


def test_unknown_numeric_protection_values_are_redacted() -> None:
    before = {
        "schema_version": 1,
        "privacy": {"sensitivity_marker": 123456789},
    }
    after = {
        "schema_version": 1,
        "privacy": {"sensitivity_marker": 123456790},
    }

    report = compare_policy_versions(before, after)
    serialized = report.to_json()

    assert report.classification == MigrationClassification.INCOMPATIBLE
    assert "123456789" not in serialized
    assert "123456790" not in serialized


def test_recall_floor_and_confidence_directions_are_classified_safely() -> None:
    before = {
        "schema_version": 1,
        "privacy": {
            "min_confidence": 0.9,
            "recall_floors": {"EMAIL": 0.95},
        },
    }
    stricter = copy.deepcopy(before)
    stricter["privacy"] = {
        "min_confidence": 0.8,
        "recall_floors": {"EMAIL": 0.99},
    }
    weaker = copy.deepcopy(before)
    weaker["privacy"] = {
        "min_confidence": 0.95,
        "recall_floors": {"EMAIL": 0.9},
    }

    stricter_report = compare_policy_versions(before, stricter)
    weaker_report = compare_policy_versions(before, weaker)

    assert stricter_report.classification == MigrationClassification.STRICTER
    assert weaker_report.classification == MigrationClassification.INCOMPATIBLE


@pytest.mark.parametrize(
    "document",
    [
        '{"schema_version":1,"schema_version":2}',
        '{"schema_version":NaN}',
    ],
)
def test_noncanonical_json_is_rejected(document: str) -> None:
    with pytest.raises(PolicyMigrationError):
        compare_policy_versions(document, _policy())


def test_policy_input_complexity_is_bounded() -> None:
    oversized = {"metadata": "x" * 65_537}
    oversized_total = {f"field_{index}": "x" * 65_536 for index in range(17)}
    nested: dict[str, object] = {"value": True}
    for _ in range(34):
        nested = {"privacy": nested}

    with pytest.raises(PolicyMigrationError):
        compare_policy_versions(oversized, _policy())
    with pytest.raises(PolicyMigrationError):
        compare_policy_versions(oversized_total, _policy())
    with pytest.raises(PolicyMigrationError):
        compare_policy_versions(nested, _policy())


def test_policy_file_size_is_bounded(tmp_path: Path) -> None:
    oversized_path = tmp_path / "oversized.json"
    oversized_path.write_bytes(b"{" + b" " * 1_048_576 + b"}")

    with pytest.raises(PolicyMigrationError):
        compare_policy_versions(oversized_path, _policy())


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("synthetic-sensitive-value")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-value")

    def __len__(self) -> int:
        raise RuntimeError("synthetic-sensitive-value")


def test_hostile_mapping_errors_are_value_free() -> None:
    with pytest.raises(PolicyMigrationError) as exc_info:
        compare_policy_versions(_ExplodingMapping(), _policy())

    assert "synthetic-sensitive-value" not in str(exc_info.value)


def test_direct_change_construction_rejects_arbitrary_report_values() -> None:
    digest = "sha256:" + "0" * 64

    with pytest.raises(PolicyMigrationError):
        PolicyChange(
            path=("metadata",),
            kind="metadata",
            classification=MigrationClassification.COMPATIBLE,
            before_present=True,
            after_present=True,
            before="synthetic-sensitive-value",
            after="<redacted>",
            before_digest=digest,
            after_digest=digest,
            reason="non-behavioural policy metadata changed",
        )


def test_report_json_indentation_is_bounded() -> None:
    report = compare_policy_versions(_policy(), _policy())

    with pytest.raises(PolicyMigrationError):
        report.to_json(indent=1_000_000)


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
