"""Tests for audit and risk CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text


def _audit_report() -> AuditReport:
    original = "Patient John Doe called 555-1234 from North Clinic."
    deidentified = "Patient [NAME] called [PHONE] from [CLINIC]."
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={
            "method": "mask",
            "confidence_threshold": 0.7,
            "language": "en",
        },
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="unit-test-model",
                model_format="transformers",
                commit="abc123",
            )
        ],
        safety_sweep={
            "source": "safety_sweep",
            "patterns_version": "safety-sweep-v1",
            "spans_added": 0,
        },
        spans=[
            AuditSpan(
                start=8,
                end=16,
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.95,
                threshold=0.7,
                action="mask",
                surrogate="[NAME]",
                text_hash=hash_text("John Doe"),
                evidence={"raw_label": "NAME", "model_id": "unit-test-model"},
                context={"before": "Patient ", "after": " called 555-1234."},
            ),
            AuditSpan(
                start=24,
                end=32,
                label="PHONE",
                canonical_label="PHONE",
                sources=["regex"],
                confidence=0.99,
                threshold=0.7,
                action="mask",
                surrogate="[PHONE]",
                text_hash=hash_text("555-1234"),
                evidence={"raw_label": "PHONE"},
                context={"before": "called ", "after": " from North Clinic."},
            ),
        ],
        thresholds={"PERSON": 0.7, "PHONE": 0.7},
        residual_risk={
            "projected_leakage": 0.05,
            "risk_report_record_score": 0.0,
            "risk_report": {
                "leakage_rate": 0.0,
                "reid_rate": 0.0,
                "k_min": 0,
                "singleton_records": [],
                "quasi_identifiers": [],
            },
        },
        openmed_version="1.7.0",
        manifest_hash="sha256:manifest",
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text(deidentified),
    )


def _write_report(path: Path, report: AuditReport) -> None:
    path.write_text(report.to_json() + "\n", encoding="utf-8")


def test_audit_verify_passes_for_signed_report_with_cli_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "audit.json"
    _write_report(report_path, _audit_report().sign("release-key", key_id="unit"))

    result = main_module.main(
        ["audit", "verify", str(report_path), "--key", "release-key"]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Audit report verification: PASS" in output
    assert "Reproducibility hash: PASS" in output
    assert "HMAC signature: PASS" in output


def test_audit_verify_uses_environment_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "audit.json"
    _write_report(report_path, _audit_report().sign("release-key", key_id="unit"))
    monkeypatch.setenv("OPENMED_AUDIT_KEY", "release-key")

    result = main_module.main(["audit", "verify", str(report_path)])

    assert result == 0
    assert "HMAC signature: PASS" in capsys.readouterr().out


def test_audit_verify_fails_for_tampered_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "audit.json"
    signed = _audit_report().sign("release-key", key_id="unit")
    payload = json.loads(signed.to_json())
    payload["policy"] = "tampered"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = main_module.main(
        ["audit", "verify", str(report_path), "--key", "release-key"]
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "Audit report verification: FAIL" in output
    assert "Reproducibility hash: FAIL" in output
    assert "HMAC signature: FAIL" in output


def test_audit_show_prints_phi_safe_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "audit.json"
    _write_report(report_path, _audit_report().sign("release-key", key_id="unit"))

    result = main_module.main(["audit", "show", str(report_path)])

    assert result == 0
    output = capsys.readouterr().out
    assert "Audit report summary" in output
    assert "PERSON: 1" in output
    assert "PHONE: 1" in output
    assert "mask: 2" in output
    assert "Projected leakage: 0.050" in output
    assert "John Doe" not in output
    assert "555-1234" not in output
    assert "North Clinic" not in output


def test_risk_text_prints_phi_safe_quasi_identifier_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    note_path = tmp_path / "note.txt"
    note_path.write_text(
        "Assessment: 94 years old seen at North Clinic on 2024-02-03 "
        "with rare condition.",
        encoding="utf-8",
    )

    result = main_module.main(["risk", "text", str(note_path)])

    assert result == 0
    output = capsys.readouterr().out
    assert "Text risk summary" in output
    assert "Leakage rate:" in output
    assert "Minimum k: 1" in output
    assert "Singleton records: 1" in output
    assert "Quasi-identifier categories:" in output
    assert "age: 1" in output
    assert "provider_institution: 1" in output
    assert "North Clinic" not in output
    assert "2024-02-03" not in output


def test_risk_table_prints_phi_safe_singleton_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "\n".join(
            [
                "record_id,age,city,visit_date,condition",
                "a,72,Riverton,2025-03-01,routine follow up",
                "b,72,Riverton,2025-03-01,routine follow up",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = main_module.main(["risk", "table", str(csv_path)])

    assert result == 0
    output = capsys.readouterr().out
    assert "Table risk summary" in output
    assert "Leakage rate:" in output
    assert "Minimum k: 2" in output
    assert "Singleton records: 0" in output
    assert "Quasi-identifier categories:" in output
    assert "age: 2" in output
    assert "date:" in output
    assert "geography: 2" in output
    assert "Riverton" not in output
    assert "2025-03-01" not in output
