"""Tests for audit-report diffs across de-identification runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from openmed.core.audit import hash_text
from openmed.risk import diff_audit_reports


def _span(
    text: str,
    label: str,
    *,
    action: str = "mask",
    confidence: float = 0.95,
    threshold: float = 0.7,
) -> dict[str, object]:
    start = 8
    end = start + len(text)
    return {
        "start": start,
        "end": end,
        "label": label,
        "canonical_label": label,
        "sources": ["unit"],
        "confidence": confidence,
        "threshold": threshold,
        "action": action,
        "surrogate": f"[{label}]",
        "text_hash": hash_text(text),
        "context": {"before": "Patient ", "after": " arrived."},
    }


def _report() -> dict[str, object]:
    return {
        "policy": "hipaa_safe_harbor",
        "spans": [_span("John Doe", "PERSON")],
        "thresholds": {"unit-model": {"PERSON": {"en": 0.7}}},
        "residual_risk": {
            "projected_leakage": 0.05,
            "risk_report": {
                "leakage_rate": 0.0,
                "reid_rate": 0.0,
                "k_min": 2,
            },
        },
        "signature": {
            "key_id": "release",
            "algorithm": "HMAC-SHA256",
            "value": "not-verified-by-diff",
        },
    }


def test_self_diff_has_zero_span_changes_and_ignores_signature() -> None:
    report = _report()

    diff = diff_audit_reports(report, report)
    payload = diff.to_dict()

    assert payload["summary"]["added_spans"] == 0
    assert payload["summary"]["removed_spans"] == 0
    assert payload["summary"]["changed_spans"] == 0
    assert payload["threshold_changes"] == []
    assert payload["residual_risk_delta"] == []


def test_label_change_is_reported_as_single_change_not_add_remove() -> None:
    before = _report()
    after = copy.deepcopy(before)
    after_span = after["spans"][0]
    assert isinstance(after_span, dict)
    after_span["label"] = "FIRST_NAME"
    after_span["canonical_label"] = "FIRST_NAME"

    diff = diff_audit_reports(before, after)
    payload = diff.to_dict()

    assert payload["summary"]["added_spans"] == 0
    assert payload["summary"]["removed_spans"] == 0
    assert payload["summary"]["changed_spans"] == 1
    assert payload["summary"]["label_changed_spans"] == 1
    assert payload["label_changed_spans"][0]["before_label"] == "PERSON"
    assert payload["label_changed_spans"][0]["after_label"] == "FIRST_NAME"


def test_policy_action_change_is_reported_for_matched_span() -> None:
    before = _report()
    after = copy.deepcopy(before)
    after_span = after["spans"][0]
    assert isinstance(after_span, dict)
    after_span["action"] = "redact"

    payload = diff_audit_reports(before, after).to_dict()

    assert payload["summary"]["policy_action_changed_spans"] == 1
    assert payload["policy_action_changed_spans"][0]["before_action"] == "mask"
    assert payload["policy_action_changed_spans"][0]["after_action"] == "redact"


def test_threshold_matrix_change_reports_label_language_pair() -> None:
    before = _report()
    after = copy.deepcopy(before)
    thresholds = after["thresholds"]
    assert isinstance(thresholds, dict)
    thresholds["unit-model"]["PERSON"]["en"] = 0.82

    diff = diff_audit_reports(before, after)
    payload = diff.to_dict()

    assert payload["summary"]["threshold_changes"] == 1
    change = payload["threshold_changes"][0]
    assert change["path"] == ["unit-model", "PERSON", "en"]
    assert change["model_id"] == "unit-model"
    assert change["label"] == "PERSON"
    assert change["language"] == "en"
    assert change["before"] == 0.7
    assert change["after"] == 0.82
    assert "### Threshold Changes" in diff.to_markdown()
    assert "unit-model.PERSON.en" in diff.to_markdown()


def test_residual_risk_delta_reports_numeric_leaf_changes() -> None:
    before = _report()
    after = copy.deepcopy(before)
    residual = after["residual_risk"]
    assert isinstance(residual, dict)
    residual["projected_leakage"] = 0.13
    risk_report = residual["risk_report"]
    assert isinstance(risk_report, dict)
    risk_report["reid_rate"] = 0.04

    payload = diff_audit_reports(before, after).to_dict()

    assert payload["summary"]["residual_risk_changes"] == 2
    deltas = {tuple(item["path"]): item for item in payload["residual_risk_delta"]}
    assert deltas[("projected_leakage",)]["delta"] == 0.08
    assert deltas[("risk_report", "reid_rate")]["delta"] == 0.04


def test_to_dict_is_json_serializable_and_deterministic_for_paths(
    tmp_path: Path,
) -> None:
    before = _report()
    after = copy.deepcopy(before)
    after["spans"].append(_span("555-1212", "PHONE", confidence=0.99))

    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")

    first = diff_audit_reports(before_path, after_path).to_dict()
    second = diff_audit_reports(before_path, after_path).to_dict()

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["summary"]["added_spans"] == 1
    assert first["added_spans"][0]["label"] == "PHONE"
