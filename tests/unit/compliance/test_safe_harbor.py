"""Tests for HIPAA Safe Harbor attestation reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.compliance import (
    SAFE_HARBOR_CATEGORY_LABELS,
    SAFE_HARBOR_CATEGORY_ORDER,
    SafeHarborAttestation,
    generate_safe_harbor_attestation,
)
from openmed.core.audit import AuditReport, AuditSpan, hash_text
from openmed.core.labels import HIPAA_SAFE_HARBOR_CLASSES, LABEL_TO_HIPAA

SYNTHETIC_NAME = "Alex Rivers"
SYNTHETIC_FAX = "555-010-7788"


def _span(
    *,
    start: int,
    surface: str,
    label: str,
    action: str = "mask",
    safe_harbor_class: str | None = None,
) -> AuditSpan:
    evidence = {"synthetic": True}
    if safe_harbor_class is not None:
        evidence["hipaa_safe_harbor_class"] = safe_harbor_class
    return AuditSpan(
        start=start,
        end=start + len(surface),
        label=label,
        canonical_label=label,
        sources=["synthetic"],
        confidence=1.0,
        threshold=0.7,
        action=action,
        surrogate=f"[{label}]",
        text_hash=hash_text(surface),
        evidence=evidence,
    )


def _audit_report() -> AuditReport:
    spans = [
        _span(start=0, surface=SYNTHETIC_NAME, label="PERSON"),
        _span(start=20, surface="Jordan Lake", label="PERSON"),
        _span(
            start=40,
            surface=SYNTHETIC_FAX,
            label="PHONE",
            safe_harbor_class="FAX_NUMBER",
        ),
        _span(start=60, surface="123-45-6789", label="SSN", action="hash"),
    ]
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask"},
        detectors=[],
        safety_sweep={},
        spans=spans,
        thresholds={},
        residual_risk={},
        openmed_version="test",
        manifest_hash=hash_text("synthetic-manifest"),
        document_length=100,
        input_hash=hash_text("synthetic-input"),
        deidentified_text_hash=hash_text("synthetic-output"),
    )


def _categories(report: SafeHarborAttestation) -> dict[str, object]:
    return {category.category: category for category in report.categories}


def test_category_mapping_enumerates_all_18_core_classes() -> None:
    assert len(SAFE_HARBOR_CATEGORY_ORDER) == 18
    assert set(SAFE_HARBOR_CATEGORY_ORDER) == HIPAA_SAFE_HARBOR_CLASSES
    assert set(SAFE_HARBOR_CATEGORY_LABELS) == HIPAA_SAFE_HARBOR_CLASSES

    reverse_pairs = {
        (label, category)
        for category, labels in SAFE_HARBOR_CATEGORY_LABELS.items()
        for label in labels
    }
    assert reverse_pairs == set(LABEL_TO_HIPAA.items())


def test_attestation_reports_counts_actions_and_policy_mapping() -> None:
    report = generate_safe_harbor_attestation(_audit_report())
    categories = _categories(report)

    assert isinstance(report, SafeHarborAttestation)
    assert len(report.categories) == 18
    assert report.total_detection_count == 4

    names = categories["NAME"]
    assert names.detection_count == 2
    assert names.applied_action_counts == {"mask": 2}
    assert names.policy_actions["PERSON"] == "mask"

    fax = categories["FAX_NUMBER"]
    assert fax.detection_count == 1
    assert fax.applied_action_counts == {"mask": 1}

    ssn = categories["SOCIAL_SECURITY_NUMBER"]
    assert ssn.detection_count == 1
    assert ssn.applied_action_counts == {"hash": 1}


def test_uncovered_categories_are_residual_risk_items() -> None:
    report = generate_safe_harbor_attestation(_audit_report())
    categories = _categories(report)
    uncovered = {
        category
        for category, labels in SAFE_HARBOR_CATEGORY_LABELS.items()
        if not labels
    }

    assert uncovered
    assert report.requires_expert_determination is True
    assert set(report.residual_risk_categories) == uncovered
    for category in uncovered:
        item = categories[category]
        assert item.residual_risk is True
        assert "expert review" in item.residual_risk_reason


def test_attestation_serialization_contains_no_raw_phi() -> None:
    report = generate_safe_harbor_attestation(_audit_report())
    rendered = report.to_json()
    payload = json.loads(rendered)

    assert SYNTHETIC_NAME not in rendered
    assert SYNTHETIC_FAX not in rendered
    assert "surrogate" not in rendered
    assert "context" not in rendered
    assert payload["source_report_hash"].startswith("sha256:")
    assert payload["attestation_hash"].startswith("sha256:")
    assert payload["summary"]["category_count"] == 18


def test_attestation_rejects_untrusted_action_text() -> None:
    untrusted = "raw synthetic patient value"
    with pytest.raises(ValueError, match="unsupported action") as error:
        generate_safe_harbor_attestation(
            {
                "policy": "hipaa_safe_harbor",
                "spans": [
                    {
                        "canonical_label": "EMAIL",
                        "action": untrusted,
                    }
                ],
            }
        )
    assert untrusted not in str(error.value)


@pytest.mark.parametrize(
    ("span", "message"),
    [
        (
            {"canonical_label": "synthetic patient value", "action": "mask"},
            "unsupported label",
        ),
        (
            {
                "canonical_label": "EMAIL",
                "action": "mask",
                "safe_harbor_class": "synthetic patient value",
            },
            "unknown Safe Harbor category",
        ),
    ],
)
def test_attestation_errors_do_not_echo_untrusted_fields(
    span: dict[str, str],
    message: str,
) -> None:
    untrusted = "synthetic patient value"
    with pytest.raises(ValueError, match=message) as error:
        generate_safe_harbor_attestation(
            {"policy": "hipaa_safe_harbor", "spans": [span]}
        )
    assert untrusted not in str(error.value)


def test_attestation_rejects_tampered_audit_report() -> None:
    audit_report = _audit_report()
    audit_report.repro_hash = hash_text("stale-audit-payload")

    with pytest.raises(ValueError, match="reproducibility hash does not match"):
        generate_safe_harbor_attestation(audit_report)


def test_attestation_rejects_forged_mapping_repro_hash() -> None:
    forged_hash = hash_text("unrelated audit payload")

    with pytest.raises(ValueError, match="reproducibility hash does not match"):
        generate_safe_harbor_attestation(
            {
                "policy": "hipaa_safe_harbor",
                "spans": [],
                "repro_hash": forged_hash,
            }
        )


def test_keep_action_adds_residual_risk_to_a_covered_category() -> None:
    report = generate_safe_harbor_attestation(
        {
            "policy": "hipaa_safe_harbor",
            "spans": [{"canonical_label": "EMAIL", "action": "keep"}],
        }
    )
    email = _categories(report)["EMAIL_ADDRESS"]

    assert email.residual_risk is True
    assert "keep action" in email.residual_risk_reason


def test_cli_emits_attestation_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "safe-harbor-attestation.json"
    audit_path.write_text(_audit_report().to_json() + "\n", encoding="utf-8")

    result = main_module.main(
        [
            "compliance",
            "safe-harbor",
            str(audit_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert str(output_path) in capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["report_type"] == "hipaa_safe_harbor_attestation"
    assert payload["summary"]["category_count"] == 18
    assert SYNTHETIC_NAME not in json.dumps(payload)


def test_cli_stdout_uses_json_envelope_when_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(_audit_report().to_json() + "\n", encoding="utf-8")

    result = main_module.main(["compliance", "safe-harbor", str(audit_path), "--json"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "compliance safe-harbor"
    assert payload["data"]["report_type"] == "hipaa_safe_harbor_attestation"


def test_cli_failure_does_not_echo_untrusted_audit_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    untrusted = "synthetic patient value"
    audit = _audit_report().to_dict()
    audit["spans"][0]["action"] = untrusted
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    result = main_module.main(["compliance", "safe-harbor", str(audit_path), "--json"])

    assert result == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "attestation_failed"
    assert untrusted not in output


def test_cli_rejects_invalid_audit_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_path = tmp_path / "invalid.json"
    audit_path.write_text("{}", encoding="utf-8")

    result = main_module.main(["compliance", "safe-harbor", str(audit_path)])

    assert result == 1
    assert "Failed to generate Safe Harbor attestation." in capsys.readouterr().err
