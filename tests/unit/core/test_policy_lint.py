from __future__ import annotations

import copy
import json
from pathlib import Path

from openmed.core.labels import CLINICAL_CONCEPT, DIRECT_IDENTIFIER
from openmed.core.policy import CANONICAL_POLICY_NAMES, load_policy
from openmed.core.policy_lint import (
    POLICY_DUPLICATE_ACTION_LABEL,
    POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP,
    POLICY_SCHEMA_INVALID,
    POLICY_STRICT_POSTURE_REQUIRES_SWEEP,
    POLICY_UNKNOWN_ACTION_LABEL,
    POLICY_UNKNOWN_ACTION_VALUE,
    POLICY_UNKNOWN_THRESHOLD_PROFILE,
    POLICY_UNREACHABLE_POLICY_LABEL_ACTION,
    lint_policy,
)


def _profile_payload(name: str = "hipaa_safe_harbor") -> dict[str, object]:
    return copy.deepcopy(load_policy(name).to_dict())


def _codes(report: dict[str, object], key: str) -> set[str]:
    return {str(finding["code"]) for finding in report[key]}


def _finding_paths(report: dict[str, object], code: str) -> set[str]:
    return {
        str(finding["path"])
        for finding in report["errors"] + report["warnings"]
        if finding["code"] == code
    }


def test_lint_policy_flags_unknown_action_label_and_threshold_profile() -> None:
    payload = _profile_payload()
    actions = payload["actions"]
    assert isinstance(actions, dict)
    actions["NOT_A_LABEL"] = "mask"
    payload["threshold_profile"] = "missing_threshold_profile"

    report = lint_policy(payload)

    assert report["valid"] is False
    assert POLICY_UNKNOWN_ACTION_LABEL in _codes(report, "errors")
    assert POLICY_UNKNOWN_THRESHOLD_PROFILE in _codes(report, "errors")
    assert "$.actions.NOT_A_LABEL" in _finding_paths(
        report, POLICY_UNKNOWN_ACTION_LABEL
    )
    assert "$.threshold_profile" in _finding_paths(
        report, POLICY_UNKNOWN_THRESHOLD_PROFILE
    )


def test_lint_policy_flags_unknown_action_values_at_stable_paths() -> None:
    payload = _profile_payload()
    actions = payload["actions"]
    policy_label_actions = payload["policy_label_actions"]
    assert isinstance(actions, dict)
    assert isinstance(policy_label_actions, dict)
    payload["default_action"] = "remove"
    actions["ID_NUM"] = "drop"
    policy_label_actions[DIRECT_IDENTIFIER] = "scrub"

    report = lint_policy(payload)

    assert report["valid"] is False
    assert POLICY_UNKNOWN_ACTION_VALUE in _codes(report, "errors")
    assert _finding_paths(report, POLICY_UNKNOWN_ACTION_VALUE) == {
        "$.actions.ID_NUM",
        "$.default_action",
        "$.policy_label_actions.DIRECT_IDENTIFIER",
    }


def test_lint_policy_detects_duplicate_action_labels(tmp_path: Path) -> None:
    raw_profile = json.dumps(_profile_payload(), indent=2)
    raw_profile = raw_profile.replace(
        '"PERSON": "mask"',
        '"PERSON": "mask",\n    "PERSON": "hash"',
        1,
    )
    profile_path = tmp_path / "policy.json"
    profile_path.write_text(raw_profile, encoding="utf-8")

    report = lint_policy(profile_path)

    assert POLICY_DUPLICATE_ACTION_LABEL in _codes(report, "errors")
    assert "$.actions.PERSON" in _finding_paths(report, POLICY_DUPLICATE_ACTION_LABEL)


def test_lint_policy_rejects_non_object_json(tmp_path: Path) -> None:
    profile_path = tmp_path / "policy.json"
    profile_path.write_text("[]", encoding="utf-8")

    report = lint_policy(profile_path)

    assert POLICY_SCHEMA_INVALID in _codes(report, "errors")
    assert "$" in _finding_paths(report, POLICY_SCHEMA_INVALID)


def test_lint_policy_warns_when_hipaa_posture_keeps_direct_identifier() -> None:
    payload = _profile_payload()
    policy_label_actions = payload["policy_label_actions"]
    assert isinstance(policy_label_actions, dict)
    policy_label_actions[DIRECT_IDENTIFIER] = "keep"

    report = lint_policy(payload)

    assert report["valid"] is True
    assert report["error_count"] == 0
    assert POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP in _codes(report, "warnings")
    assert "$.policy_label_actions.DIRECT_IDENTIFIER" in _finding_paths(
        report, POLICY_HIPAA_DIRECT_IDENTIFIER_KEEP
    )


def test_lint_policy_warns_when_strict_posture_omits_safety_sweep() -> None:
    payload = _profile_payload("strict_no_leak")
    payload["safety_sweep_mandatory"] = False

    report = lint_policy(payload)

    assert report["error_count"] == 0
    assert POLICY_STRICT_POSTURE_REQUIRES_SWEEP in _codes(report, "warnings")
    assert "$.safety_sweep_mandatory" in _finding_paths(
        report, POLICY_STRICT_POSTURE_REQUIRES_SWEEP
    )


def test_lint_policy_warns_on_shadowed_policy_label_action() -> None:
    payload = _profile_payload("clinical_minimal_redaction")
    policy_label_actions = payload["policy_label_actions"]
    assert isinstance(policy_label_actions, dict)
    policy_label_actions[CLINICAL_CONCEPT] = "mask"

    report = lint_policy(payload)

    assert report["error_count"] == 0
    assert POLICY_UNREACHABLE_POLICY_LABEL_ACTION in _codes(report, "warnings")
    assert "$.policy_label_actions.CLINICAL_CONCEPT" in _finding_paths(
        report, POLICY_UNREACHABLE_POLICY_LABEL_ACTION
    )


def test_bundled_policy_profiles_lint_clean() -> None:
    for name in CANONICAL_POLICY_NAMES:
        report = lint_policy(name)

        assert report["error_count"] == 0, name
        assert report["warning_count"] == 0, name


def test_lint_report_is_json_serializable_without_raw_metadata() -> None:
    payload = _profile_payload()
    payload["metadata"] = {
        "note": "Patient Jane Doe was seen on 2026-06-01",
    }

    report = lint_policy(payload)
    serialized = json.dumps(report, sort_keys=True)

    assert "Jane Doe" not in serialized
    assert "2026-06-01" not in serialized
