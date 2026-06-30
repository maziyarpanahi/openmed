from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from openmed.cli import main_module
from openmed.core.policy import load_policy
from openmed.core.policy_diff import diff_policies, render


def _field_change(diff_dict: dict[str, object], section: str, field: str) -> dict:
    changes = diff_dict[section]
    assert isinstance(changes, list)
    for change in changes:
        if change["field"] == field:
            return change
    raise AssertionError(f"missing {field} in {section}")


def _action_change(diff_dict: dict[str, object], label: str) -> dict:
    changes = diff_dict["changed_label_actions"]
    assert isinstance(changes, list)
    for change in changes:
        if change["label"] == label:
            return change
    raise AssertionError(f"missing action change for {label}")


def test_diff_policies_reports_action_direction_and_runtime_flags() -> None:
    diff = diff_policies("clinical_minimal_redaction", "strict_no_leak")
    payload = render(diff, fmt="dict")

    location = _action_change(payload, "LOCATION")
    assert location == {
        "label": "LOCATION",
        "from": "keep",
        "to": "mask",
        "direction": "stronger",
    }

    strict_flag = _field_change(payload, "runtime_flag_changes", "strict_no_leak")
    assert strict_flag == {"field": "strict_no_leak", "from": False, "to": True}

    sweep_flag = _field_change(
        payload,
        "runtime_flag_changes",
        "safety_sweep_mandatory",
    )
    assert sweep_flag == {
        "field": "safety_sweep_mandatory",
        "from": False,
        "to": True,
    }

    threshold = _field_change(payload, "setting_changes", "threshold_profile")
    assert threshold == {
        "field": "threshold_profile",
        "from": "balanced",
        "to": "strict_no_leak",
    }

    reverse = render(
        diff_policies("strict_no_leak", "clinical_minimal_redaction"),
        fmt="dict",
    )
    assert _action_change(reverse, "LOCATION")["direction"] == "weaker"

    gdpr = render(
        diff_policies("clinical_minimal_redaction", "gdpr_pseudonymization"),
        fmt="dict",
    )
    reversible = _field_change(gdpr, "runtime_flag_changes", "reversible_id")
    assert reversible == {"field": "reversible_id", "from": False, "to": True}


def test_diffing_same_policy_profile_is_empty() -> None:
    profile = load_policy("strict_no_leak")
    diff = diff_policies(profile, profile)

    assert diff.is_empty is True
    assert render(diff) == (
        "Policy diff: strict_no_leak -> strict_no_leak\nNo policy changes."
    )
    assert all(
        not value
        for key, value in render(diff, fmt="dict").items()
        if key not in {"base", "candidate"}
    )


def test_diff_policies_accepts_paths_and_reports_added_removed_labels(
    tmp_path: Path,
) -> None:
    base_payload = load_policy("clinical_minimal_redaction").to_dict()
    base_payload["name"] = "custom_baseline"

    candidate_payload = deepcopy(base_payload)
    candidate_payload["name"] = "custom_candidate"
    candidate_payload["actions"]["LOCATION"] = "mask"
    candidate_payload["actions"]["NEW_IDENTIFIER"] = "hash"
    candidate_payload["actions"].pop("OTHER")
    candidate_payload["forced_cascade_tiers"] = ["R0", "R1", "R2"]

    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(base_payload), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    payload = render(diff_policies(base_path, candidate_path), fmt="dict")

    assert _action_change(payload, "LOCATION")["direction"] == "stronger"
    assert payload["added_labels"] == [{"label": "NEW_IDENTIFIER", "action": "hash"}]
    assert payload["removed_labels"] == [{"label": "OTHER", "action": "keep"}]
    assert _field_change(payload, "setting_changes", "forced_cascade_tiers") == {
        "field": "forced_cascade_tiers",
        "from": ["R0", "R1"],
        "to": ["R0", "R1", "R2"],
    }


def test_rendered_policy_diff_is_json_serializable() -> None:
    payload = render(
        diff_policies("clinical_minimal_redaction", "strict_no_leak"),
        fmt="dict",
    )

    assert json.loads(json.dumps(payload)) == payload


def test_policy_diff_cli_prints_summary_and_json(
    capsys,
) -> None:
    result = main_module.main(
        ["policy", "diff", "clinical_minimal_redaction", "strict_no_leak"]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Policy diff: clinical_minimal_redaction -> strict_no_leak" in captured.out
    assert "LOCATION: keep -> mask (stronger)" in captured.out
    assert "Runtime flag changes:" in captured.out

    result = main_module.main(
        [
            "policy",
            "diff",
            "clinical_minimal_redaction",
            "strict_no_leak",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload["base"] == "clinical_minimal_redaction"
    assert payload["candidate"] == "strict_no_leak"
