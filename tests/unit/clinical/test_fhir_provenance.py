"""Tests for FHIR Provenance and AuditEvent export from audit reports."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.exporters.fhir import (
    GovernedWriteAction,
    GovernedWriteAuditAttempt,
    GovernedWriteAuditError,
    GovernedWriteOutcome,
    GovernedWriteReviewerRole,
    to_audit_event,
    to_bundle,
    to_governed_write_audit_event,
    to_provenance,
)
from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text


def _signed_report() -> AuditReport:
    original = "Patient John Doe called 555-1234."
    redacted = "Patient [NAME] called [PHONE]."
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
                context={"before": "called ", "after": "."},
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
                "quasi_identifiers": ["PERSON", "PHONE"],
            },
        },
        openmed_version="1.7.0",
        manifest_hash="sha256:manifest",
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text(redacted),
    ).sign("release-key", key_id="unit-test")


def _detail_map(resource: dict) -> dict[str, str]:
    return {
        item["type"]: item["valueString"] for item in resource["entity"][0]["detail"]
    }


def test_to_provenance_references_targets_and_repro_hash():
    report = _signed_report()

    provenance = to_provenance(
        report,
        ["Condition/cond1", {"reference": "Observation/obs1", "display": "ignored"}],
    )

    assert provenance["resourceType"] == "Provenance"
    assert provenance["target"] == [
        {"reference": "Condition/cond1"},
        {"reference": "Observation/obs1"},
    ]
    assert provenance["recorded"].endswith("Z")
    assert provenance["activity"]["coding"][0]["code"] == "de-identify"

    agent = provenance["agent"][0]
    assert agent["who"]["identifier"]["value"] == "openmed"
    assert agent["who"]["display"] == "openmed 1.7.0"

    entity_identifier = provenance["entity"][0]["what"]["identifier"]
    assert entity_identifier["value"] == report.repro_hash


def test_to_audit_event_describes_deidentification_outcome_and_risk_details():
    report = _signed_report()

    audit_event = to_audit_event(report)
    details = _detail_map(audit_event)

    assert audit_event["resourceType"] == "AuditEvent"
    assert audit_event["type"]["code"] == "de-identification"
    assert {coding["code"] for coding in audit_event["subtype"]} == {
        "de-identify",
        "transform",
    }
    assert audit_event["action"] == "E"
    assert audit_event["outcome"] == "0"
    assert audit_event["agent"][0]["who"]["identifier"]["value"] == "openmed"
    assert audit_event["source"]["observer"]["display"] == "openmed 1.7.0"

    assert details["openmed.repro_hash"] == report.repro_hash
    assert details["openmed.span_labels"] == "PERSON,PHONE"
    assert details["openmed.span_count"] == "2"
    assert details["openmed.residual_risk.projected_leakage"] == "0.05"
    assert details["openmed.residual_risk.risk_report.reid_rate"] == "0.0"
    assert details["openmed.residual_risk.risk_report.quasi_identifiers_count"] == "2"


def test_resources_do_not_embed_raw_phi_or_span_text():
    report = _signed_report()

    payload = json.dumps(
        [
            to_provenance(report, ["Condition/cond1"]),
            to_audit_event(report),
        ],
        sort_keys=True,
    )

    assert "John Doe" not in payload
    assert "555-1234" not in payload
    assert "Patient " not in payload
    assert "called " not in payload
    assert "[NAME]" not in payload
    assert "[PHONE]" not in payload
    assert hash_text("John Doe") in payload
    assert "PERSON" in payload


def test_resources_assemble_cleanly_into_bundle():
    report = _signed_report()
    condition = {
        "resourceType": "Condition",
        "id": "cond1",
        "code": {"text": "redacted condition"},
    }
    provenance = to_provenance(report, ["Condition/cond1"])
    audit_event = to_audit_event(report)

    bundle = to_bundle([condition, provenance, audit_event], doc_id="doc-1")

    full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
    condition_url = bundle["entry"][0]["fullUrl"]
    bundled_provenance = bundle["entry"][1]["resource"]

    assert bundled_provenance["target"] == [{"reference": condition_url}]
    assert len(bundle["entry"]) == 3
    assert all(entry["fullUrl"] in full_urls for entry in bundle["entry"])


def test_to_provenance_requires_target_references():
    with pytest.raises(ValueError, match="at least one target reference"):
        to_provenance(_signed_report(), [])


_SOFTWARE = "urn:uuid:00000000-0000-4000-8000-000000000001"
_TARGET = "urn:uuid:00000000-0000-4000-8000-000000000002"
_OTHER_TARGET = "urn:uuid:00000000-0000-4000-8000-000000000003"
_RECEIPT = "sha256:" + "a" * 64
_INSTANT = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
_PRIVATE_SENTINEL = "SYNTHETIC-PROTECTED-0042-DO-NOT-EMIT"


def _attempt(**overrides: Any) -> GovernedWriteAuditAttempt:
    fields: dict[str, Any] = {
        "action": GovernedWriteAction.CREATE,
        "outcome": GovernedWriteOutcome.SUCCESS,
        "software_agent_ref": _SOFTWARE,
        "target_refs": (_TARGET,),
        "reviewer_role": GovernedWriteReviewerRole.CLINICAL,
        "approval_receipt_digest": _RECEIPT,
    }
    fields.update(overrides)
    return GovernedWriteAuditAttempt(**fields)


@pytest.mark.parametrize(
    "action,action_code",
    [(GovernedWriteAction.CREATE, "C"), (GovernedWriteAction.UPDATE, "U")],
)
@pytest.mark.parametrize(
    "outcome,outcome_code",
    [
        (GovernedWriteOutcome.SUCCESS, "0"),
        (GovernedWriteOutcome.SERVER_REJECTED, "4"),
        (GovernedWriteOutcome.COMMIT_UNKNOWN, "8"),
        (GovernedWriteOutcome.POLICY_DENIED, "4"),
        (GovernedWriteOutcome.ADMISSION_REFUSED, "4"),
    ],
)
def test_governed_audit_action_and_outcome_mapping(
    action: GovernedWriteAction,
    action_code: str,
    outcome: GovernedWriteOutcome,
    outcome_code: str,
) -> None:
    event = to_governed_write_audit_event(
        _attempt(action=action, outcome=outcome), clock=lambda: _INSTANT
    )
    assert event["action"] == action_code
    assert event["outcome"] == outcome_code
    assert event["subtype"][0]["code"] == outcome.value


def test_governed_unknown_commit_preserves_uncertainty_explicitly() -> None:
    event = to_governed_write_audit_event(
        _attempt(outcome=GovernedWriteOutcome.COMMIT_UNKNOWN), clock=lambda: _INSTANT
    )
    assert event["outcome"] == "8"
    assert event["type"]["code"] == "governed-write-attempt"
    assert event["subtype"][0]["code"] == "commit_unknown"
    assert "commit_absent" not in json.dumps(event)
    assert "write_failed" not in json.dumps(event)


@pytest.mark.parametrize(
    "outcome",
    [GovernedWriteOutcome.POLICY_DENIED, GovernedWriteOutcome.ADMISSION_REFUSED],
)
def test_governed_refusal_needs_no_materialized_target_or_invented_review(
    outcome: GovernedWriteOutcome, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    def no_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("audit projection must never contact a server")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    attempt = GovernedWriteAuditAttempt("update", outcome, _SOFTWARE)
    event = to_governed_write_audit_event(attempt, clock=lambda: _INSTANT)
    assert event["action"] == "U" and event["outcome"] == "4"
    assert len(event["agent"]) == 1
    assert "entity" not in event
    assert "role" not in event["agent"][0]


def test_governed_audit_is_deterministic_for_fixed_clock_and_metadata() -> None:
    attempt = _attempt(target_refs=(_OTHER_TARGET, _TARGET, _TARGET))
    first = to_governed_write_audit_event(attempt, clock=lambda: _INSTANT)
    second = to_governed_write_audit_event(attempt, clock=lambda: _INSTANT)
    reordered = to_governed_write_audit_event(
        _attempt(target_refs=(_TARGET, _OTHER_TARGET)), clock=lambda: _INSTANT
    )
    assert first == second == reordered
    assert len(first["id"]) <= 64
    assert first["recorded"] == "2026-01-02T03:04:05.123456Z"
    first["agent"][0]["who"]["reference"] = "changed locally"
    assert second["agent"][0]["who"]["reference"] == _SOFTWARE
    assert second["source"]["observer"]["reference"] == _SOFTWARE
    assert attempt.target_refs == (_TARGET, _OTHER_TARGET)
    with pytest.raises(FrozenInstanceError):
        attempt.action = GovernedWriteAction.UPDATE  # type: ignore[misc]


def test_governed_clock_is_called_once_and_normalized_to_utc() -> None:
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        return _INSTANT.astimezone(timezone(timedelta(hours=5, minutes=30)))

    event = to_governed_write_audit_event(_attempt(), clock=clock)
    assert calls == 1
    assert event["recorded"] == "2026-01-02T03:04:05.123456Z"


@pytest.mark.parametrize(
    "clock",
    [None, lambda: datetime(2026, 1, 2), lambda: _PRIVATE_SENTINEL, lambda: None],
)
def test_governed_invalid_clock_is_controlled(clock: Any) -> None:
    with pytest.raises(GovernedWriteAuditError) as error:
        to_governed_write_audit_event(_attempt(), clock=clock)
    assert error.value.code == "invalid_clock"
    assert _PRIVATE_SENTINEL not in str(error.value)


def test_governed_clock_failure_does_not_chain_private_exception() -> None:
    def clock() -> datetime:
        raise RuntimeError(_PRIVATE_SENTINEL)

    with pytest.raises(GovernedWriteAuditError) as error:
        to_governed_write_audit_event(_attempt(), clock=clock)
    assert error.value.code == "clock_failed"
    assert error.value.__context__ is None
    assert _PRIVATE_SENTINEL not in repr(error.value)


@pytest.mark.parametrize(
    "fields,code",
    [
        ({"action": _PRIVATE_SENTINEL}, "invalid_action"),
        ({"outcome": _PRIVATE_SENTINEL}, "invalid_outcome"),
        (
            {"software_agent_ref": "Device/" + _PRIVATE_SENTINEL},
            "invalid_software_agent_ref",
        ),
        (
            {"software_agent_ref": "https://synthetic.invalid/Device/0042"},
            "invalid_software_agent_ref",
        ),
        ({"target_refs": ("Patient/0042",)}, "invalid_target_refs"),
        (
            {"target_refs": (_TARGET + "?token=" + _PRIVATE_SENTINEL,)},
            "invalid_target_refs",
        ),
        ({"target_refs": (_PRIVATE_SENTINEL,)}, "invalid_target_refs"),
        ({"target_refs": (_TARGET.upper(),)}, "invalid_target_refs"),
        (
            {"target_refs": (_TARGET.replace("-4000-", "-5000-"),)},
            "invalid_target_refs",
        ),
        ({"target_refs": (_TARGET + "\u200b",)}, "invalid_target_refs"),
        ({"target_refs": [_TARGET] * 129}, "invalid_target_refs"),
        ({"target_refs": _TARGET}, "invalid_target_refs"),
        ({"reviewer_role": _PRIVATE_SENTINEL}, "invalid_reviewer_role"),
        ({"reviewer_role": "role:local/" + _PRIVATE_SENTINEL}, "invalid_reviewer_role"),
        ({"approval_receipt_digest": _PRIVATE_SENTINEL}, "invalid_receipt_digest"),
        ({"approval_receipt_digest": "a" * 64}, "invalid_receipt_digest"),
        ({"approval_receipt_digest": None}, "incomplete_review_evidence"),
        ({"reviewer_role": None}, "incomplete_review_evidence"),
        (
            {"reviewer_role": None, "approval_receipt_digest": None},
            "incomplete_dispatch_evidence",
        ),
        ({"target_refs": ()}, "incomplete_dispatch_evidence"),
    ],
)
def test_governed_invalid_metadata_refuses_without_value_echo(
    fields: dict[str, Any], code: str
) -> None:
    with pytest.raises(GovernedWriteAuditError) as error:
        _attempt(**fields)
    assert error.value.code == code
    assert _PRIVATE_SENTINEL not in repr(error.value)
    assert "Patient/0042" not in str(error.value)
    assert error.value.__context__ is None


def test_governed_resources_carry_only_controlled_metadata() -> None:
    event = to_governed_write_audit_event(_attempt(), clock=lambda: _INSTANT)
    assert set(event) == {
        "resourceType",
        "id",
        "type",
        "subtype",
        "action",
        "recorded",
        "outcome",
        "agent",
        "source",
        "entity",
    }
    assert event["agent"][0]["who"] == {"reference": _SOFTWARE, "type": "Device"}
    assert event["agent"][1]["role"][0]["coding"][0]["code"] == "clinical_reviewer"
    assert "who" not in event["agent"][1]
    assert event["entity"][0]["what"] == {"reference": _TARGET}
    assert event["entity"][1]["detail"] == [
        {"type": "openmed.approval_receipt_digest", "valueString": _RECEIPT}
    ]
    for forbidden in (
        "diagnostics",
        "outcomeDesc",
        "display",
        "text",
        "name",
        "altId",
        "query",
        "payload",
        "credential",
        "endpoint",
    ):
        assert '"' + forbidden + '"' not in json.dumps(event)


@pytest.mark.parametrize(
    "field",
    [
        "action",
        "outcome",
        "reviewer_role",
        "approval_receipt_digest",
        "target_refs",
        "software_agent_ref",
    ],
)
def test_governed_event_id_binds_classified_metadata(field: str) -> None:
    before = _attempt()
    alternatives = {
        "action": GovernedWriteAction.UPDATE,
        "outcome": GovernedWriteOutcome.SERVER_REJECTED,
        "reviewer_role": GovernedWriteReviewerRole.PRIVACY,
        "approval_receipt_digest": "sha256:" + "b" * 64,
        "target_refs": (_OTHER_TARGET,),
        "software_agent_ref": _OTHER_TARGET,
    }
    original = to_governed_write_audit_event(before, clock=lambda: _INSTANT)
    changed = to_governed_write_audit_event(
        replace(before, **{field: alternatives[field]}), clock=lambda: _INSTANT
    )
    assert original["id"] != changed["id"]
    later = to_governed_write_audit_event(
        before, clock=lambda: _INSTANT + timedelta(microseconds=1)
    )
    assert original["id"] != later["id"]


@pytest.mark.integration
def test_governed_audit_projects_into_existing_local_fhir_bundle() -> None:
    from openmed.interop.fhir import validate_resource

    event = to_governed_write_audit_event(_attempt(), clock=lambda: _INSTANT)
    outcome = validate_resource(event)
    assert not any(
        issue["severity"] in {"fatal", "error"} for issue in outcome["issue"]
    )
    bundle = to_bundle(
        (event,), doc_id="synthetic-audit-bundle", bundle_type="collection"
    )
    assert bundle["entry"][0]["resource"] == event
    assert bundle["type"] == "collection"


def test_governed_projection_refuses_untyped_input_before_clock() -> None:
    def clock() -> datetime:
        pytest.fail("untyped input must stop before invoking the clock")

    with pytest.raises(GovernedWriteAuditError, match="invalid_attempt"):
        to_governed_write_audit_event({"payload": _PRIVATE_SENTINEL}, clock=clock)  # type: ignore[arg-type]


def test_governed_documented_refusal_runs_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from openmed.core.models import ModelLoader
    from openmed.core.offline import HF_OFFLINE_ENV_VARS, network_blocked_if_offline

    def reject_model(*args: Any, **kwargs: Any) -> None:
        pytest.fail("audit projection must not load a model")

    monkeypatch.setattr(ModelLoader, "load_model", reject_model)
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")
    guide = Path(__file__).resolve().parents[3] / "docs/fhir-interop.md"
    blocks = re.findall(
        r"(?ms)^```python\n(.*?)^```", guide.read_text(encoding="utf-8")
    )
    runnable = [block for block in blocks if block.startswith("# Runnable:")]
    assert runnable
    with network_blocked_if_offline(local_only=True):
        for source in runnable:
            exec(compile(source, "write_audit_example.py", "exec"), {})
    captured = capsys.readouterr()
    assert captured.out == "{'action': 'C', 'outcome': '4'}\n"
    assert captured.err == ""
