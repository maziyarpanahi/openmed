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
