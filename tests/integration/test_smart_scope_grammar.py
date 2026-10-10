"""Synthetic scope-set intake to privacy-safe offline audit evidence."""

import json
import socket

import pytest

from openmed.interop.smart_scope_audit import audit_smart_scopes
from openmed.interop.smart_scope_grammar import compare_smart_scopes


@pytest.mark.integration
def test_mixed_ehr_scope_intake_is_offline_and_fails_closed(monkeypatch) -> None:
    def forbid_network(*args, **kwargs):
        pytest.fail("scope comparison attempted a network call")

    monkeypatch.setattr(socket, "socket", forbid_network)
    required = "openid fhirUser launch/patient offline_access patient/Observation.read"
    declared = (
        "openid fhirUser launch/patient offline_access "
        "patient/Observation.rs?category=urn:synthetic|PRIVATE_MARKER "
        "patient/Observation.write unknown-PRIVATE_MARKER"
    )
    shared = compare_smart_scopes(required_scopes=required, declared_scopes=declared)
    report = audit_smart_scopes(
        workflow_id="synthetic", required_scopes=required, declared_scopes=declared
    )
    assert report.status == "invalid"
    assert [scope.value for scope in shared.missing_scopes] == [
        "patient/Observation.rs"
    ]
    assert any("d" in scope.operations for scope in shared.excessive_scopes)
    evidence = json.dumps(report.to_dict(), sort_keys=True)
    assert "PRIVATE_MARKER" not in evidence
    assert "urn:synthetic" not in evidence
    assert "constraint_digest" in evidence
    assert report.findings[0].index == 6
    # Negative control: v1 read, unlike a granular grant, covers all observations.
    safe = audit_smart_scopes(
        workflow_id="synthetic", required_scopes=required, declared_scopes=required
    )
    assert safe.status == "pass"


@pytest.mark.integration
def test_current_preflight_consumes_the_shared_mixed_scope_grammar() -> None:
    from openmed.interop.smart_scope_audit import audit_smart_scope_preflight

    scopes = "openid fhirUser launch/patient offline_access patient/Observation.read"
    report = audit_smart_scope_preflight(
        required_scopes=scopes, requested_scopes=scopes.split()
    )
    assert report.is_least_privilege
    assert not report.findings


@pytest.mark.integration
def test_current_preflight_invalid_scope_returns_value_free_findings() -> None:
    from openmed.interop.smart_scope_audit import audit_smart_scope_preflight

    report = audit_smart_scope_preflight(
        required_scopes=["patient/Observation.r"],
        requested_scopes=["unknown-SYNTHETIC_PRIVATE_MARKER"],
    )
    assert not report.is_least_privilege
    assert report.findings
    assert "SYNTHETIC_PRIVATE_MARKER" not in json.dumps(report.to_dict())


@pytest.mark.integration
def test_current_custody_retains_constraints_without_exporting_private_values() -> None:
    from datetime import datetime, timedelta, timezone

    from openmed.interop.fhir.smart_custody import SmartCustodyError, SmartTokenCustody

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sent = []
    custody = SmartTokenCustody(lambda *args: sent.append(args), clock=lambda: now)
    constrained = "patient/Observation.rs?category=SYNTHETIC_PRIVATE_MARKER"
    handle = custody.store(
        access_token="synthetic-token",
        audience="https://example.invalid/fhir",
        expires_at=now + timedelta(minutes=5),
        scopes=[constrained, "openid", "fhirUser", "offline_access"],
    )
    with pytest.raises(SmartCustodyError, match="insufficient_scope"):
        custody.dispatch(
            handle,
            audience="https://example.invalid/fhir",
            required_scopes=["patient/Observation.read"],
        )
    assert not sent
    receipt = custody.dispatch(
        handle, audience="https://example.invalid/fhir", required_scopes=[constrained]
    )
    assert len(sent) == 1
    for public in [repr(receipt), json.dumps(receipt.to_dict())]:
        assert "SYNTHETIC_PRIVATE_MARKER" not in public
        assert "category" not in public
        assert "constraint_digest=" in public
    from openmed.interop.smart_scope_audit import parse_smart_scope_preflight

    parsed = parse_smart_scope_preflight(constrained)
    assert "SYNTHETIC_PRIVATE_MARKER" not in repr(parsed)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("required", "declared", "missing", "excessive"),
    [
        ("patient/Observation.read", "patient/Observation.rs", False, False),
        ("patient/Observation.cu", "patient/Observation.write", False, True),
        ("patient/Observation.rs", "patient/Observation.rs?category=lab", True, False),
        ("patient/Observation.rs?category=lab", "patient/Observation.rs", False, True),
        (
            "patient/Observation.rs?category=lab",
            "patient/Observation.rs?category=exam",
            True,
            True,
        ),
        (
            "patient/Observation.rs?category=urn%3Asynthetic%7Clab",
            "patient/Observation.rs?category=urn:synthetic|lab",
            False,
            False,
        ),
        ("patient/Observation.rs", "openid fhirUser offline_access", True, True),
    ],
)
def test_current_preflight_conservative_granular_and_identity_controls(
    required, declared, missing, excessive
) -> None:
    from openmed.interop.smart_scope_audit import audit_smart_scope_preflight

    report = audit_smart_scope_preflight(
        required_scopes=required, requested_scopes=declared
    )
    assert bool(report.missing_scopes) is missing
    assert bool(report.excessive_scopes) is excessive
    assert not report.findings
    assert report.is_least_privilege is (not missing and not excessive)
    assert "category=" not in json.dumps(report.to_dict())
