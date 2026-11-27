"""Synthetic FHIR reference-server matrix and opt-in local container probe."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from openmed.interop.fhir.compensation_report import build_compensation_report
from openmed.interop.fhir.concurrency_guard import (
    FHIRWriteConflict,
    require_no_server_conflict,
)
from openmed.interop.fhir.conformance import (
    CASES,
    MATRIX_VERSION,
    SERVERS,
    ConformanceResult,
    Outcome,
    ReferenceProfile,
    classify,
    run_reference_server,
)
from openmed.interop.fhir.smart_custody import SmartCustodyError, SmartTokenCustody
from openmed.interop.fhir.subscription_checkpoint import SubscriptionCheckpoint
from openmed.interop.smart_scope_audit import audit_smart_scope_preflight
from tests.fixtures.fhir.reference_servers import MATRIX
from tests.fixtures.fhir_capabilities import build_capability_statement


@pytest.mark.parametrize("server", SERVERS)
def test_versioned_matrix_is_complete_and_value_free(server: str) -> None:
    profile = ReferenceProfile(server, "8.0.0", "http://127.0.0.1:8080/fhir")
    assert MATRIX[server] == {
        "schema_version": MATRIX_VERSION,
        "fhir_version": "4.0.1",
        "cases": CASES,
    }
    assert {
        classify(profile, case, supported=True, passed=True).outcome for case in CASES
    } == {Outcome.PASS}
    assert (
        classify(
            profile,
            "etag_conflict",
            supported=True,
            passed=False,
            expected_variance=True,
        ).outcome
        is Outcome.EXPECTED_VARIANCE
    )
    assert (
        classify(profile, "etag_conflict", supported=True, passed=False).outcome
        is Outcome.FAIL
    )
    assert (
        classify(
            profile, "batch_partial_failure", supported=False, passed=False
        ).outcome
        is Outcome.UNSUPPORTED
    )
    assert set(
        ConformanceResult(
            server, profile.version, "etag_conflict", Outcome.PASS, "observed"
        ).to_dict()
    ) == {"matrix_version", "server", "version", "case", "outcome", "reason_code"}


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/fhir",
        "http://example.org/fhir",
        "http://user:pw@localhost/fhir",
    ],
)
def test_nonlocal_or_credentialed_profile_is_rejected_without_echo(url: str) -> None:
    with pytest.raises(ValueError, match="local HTTP endpoint") as error:
        ReferenceProfile("hapi", "8.0.0", url)
    assert url not in str(error.value)


def test_transport_failure_never_serializes_sensitive_request_data() -> None:
    profile = ReferenceProfile("medplum", "test-version", "http://localhost:8080/fhir")
    marker = "synthetic-private-marker"

    def transport(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError(marker)

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        results = run_reference_server(profile, client)
    assert len(results) == len(CASES)
    assert all(item.outcome is Outcome.FAIL for item in results)
    assert marker not in str([item.to_dict() for item in results])
    assert profile.base_url not in repr(profile)


@pytest.mark.integration
def test_synthetic_reference_transport_covers_read_search_and_write_safety() -> None:
    resources: dict[str, dict] = {}
    statement = build_capability_statement(
        resource_type="Patient",
        resource_interactions=("read", "search-type", "update", "create"),
        conditional_create=True,
        system_interactions=("transaction", "batch"),
    )
    statement["software"] = {"name": "HAPI FHIR", "version": "8.0.0"}

    def transport(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/metadata"):
            return httpx.Response(200, json=statement)
        if path.endswith("/fhir/") and request.method == "POST":
            bundle = json.loads(request.content)
            if bundle["type"] == "transaction":
                return httpx.Response(400)
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "batch-response",
                    "entry": [
                        {"response": {"status": "201 Created"}},
                        {"response": {"status": "400 Bad Request"}},
                    ],
                },
            )
        if path.endswith("/Patient"):
            if request.method == "POST":
                patient = json.loads(request.content)
                key = patient["identifier"][0]["value"]
                if key in resources:
                    return httpx.Response(200, json=resources[key])
                patient["id"] = key
                resources[key] = patient
                return httpx.Response(201, json=patient)
            if request.url.params.get("page") == "2":
                return httpx.Response(
                    200,
                    json={
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "entry": [{"resource": list(resources.values())[1]}],
                    },
                )
            ids = set(request.url.params.get("_id", "").split(","))
            entries = [
                {"resource": resource}
                for key, resource in resources.items()
                if key in ids
            ]
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "entry": entries[:1],
                    "link": [
                        {
                            "relation": "next",
                            "url": "http://127.0.0.1:8080/fhir/Patient?page=2",
                        }
                    ]
                    if len(entries) > 1
                    else [],
                },
            )
        key = path.rsplit("/", 1)[-1]
        if request.method == "PUT":
            if request.headers.get("If-Match") == 'W/"openmed-stale"':
                return httpx.Response(412)
            resource = json.loads(request.content)
            resource["meta"] = {"versionId": "1", "lastUpdated": "2026-01-01T00:00:00Z"}
            resources[key] = resource
            return httpx.Response(201)
        if request.method == "GET" and key in resources:
            return httpx.Response(200, json=resources[key])
        if request.method == "DELETE":
            resources.pop(key, None)
            return httpx.Response(204)
        return httpx.Response(404)

    profile = ReferenceProfile("hapi", "8.0.0", "http://127.0.0.1:8080/fhir")
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        results = run_reference_server(profile, client)
    assert [item.case for item in results] == list(CASES)
    assert {item.case for item in results if item.outcome is Outcome.PASS} == {
        "read_search_pagination",
        "capability_preflight",
        "etag_conflict",
        "conditional_write",
        "transaction_atomicity",
        "batch_partial_failure",
    }
    if resources:
        pytest.fail("synthetic cleanup incomplete")
    serialized = str([item.to_dict() for item in results])
    assert "127.0.0.1" not in serialized
    assert "openmed-test-" not in serialized


@pytest.mark.integration
def test_failure_cases_cover_scope_refresh_revocation_duplicate_and_partial_failure(
    tmp_path,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sent: list[bool] = []
    custody = SmartTokenCustody(
        lambda _audience, header: sent.append(header.startswith("Bearer ")),
        clock=lambda: now,
    )
    audience = "https://synthetic.invalid/fhir"
    handle = custody.store(
        access_token="synthetic-old-token",
        audience=audience,
        expires_at=now + timedelta(minutes=1),
        scopes=["patient/Patient.r"],
        refresh_token="synthetic-refresh-token",
    )
    assert custody.dispatch(
        handle, audience=audience, required_scopes=["patient/Patient.r"]
    ).scopes == ("patient/Patient.r",)
    assert not audit_smart_scope_preflight(
        required_scopes=["patient/Patient.r"],
        requested_scopes=["patient/Patient.rs"],
    ).is_least_privilege
    custody.revoke(handle)
    with pytest.raises(SmartCustodyError) as revoked:
        custody.dispatch(
            handle, audience=audience, required_scopes=["patient/Patient.r"]
        )
    assert revoked.value.reason_code == "unknown_handle"
    refreshed = custody.store(
        access_token="synthetic-new-token",
        audience=audience,
        expires_at=now + timedelta(minutes=2),
        scopes=["patient/Patient.r"],
    )
    custody.dispatch(
        refreshed, audience=audience, required_scopes=["patient/Patient.r"]
    )
    assert sent == [True, True]

    delivery = dict(
        subscription_id="synthetic-subscription",
        notification_id="synthetic-notification",
        resource_id="synthetic-resource",
        resource_version="1",
        sequence=0,
    )
    with SubscriptionCheckpoint(
        tmp_path / "checkpoint.sqlite", secret=b"synthetic-private-test-secret-32-bytes"
    ) as checkpoint:
        assert checkpoint.claim(**delivery).status == "claimed"
        checkpoint.commit(**delivery)
        assert checkpoint.claim(**delivery).status == "duplicate"

    intended = {
        "resourceType": "Bundle",
        "type": "batch",
        "entry": [
            {"request": {"method": "POST", "url": "Patient"}},
            {"request": {"method": "PUT", "url": "Patient/synthetic"}},
        ],
    }
    received = {
        "resourceType": "Bundle",
        "type": "batch-response",
        "entry": [
            {"response": {"status": "201 Created", "location": "Patient/synthetic"}},
            {"response": {"status": "412 Precondition Failed"}},
        ],
    }
    packet = build_compensation_report(intended, received)
    assert packet.has_partial_failure and packet.approval_required
    assert [effect.status_code for effect in packet.effects] == [201, 412]
    with pytest.raises(FHIRWriteConflict) as conflict:
        require_no_server_conflict(412)
    assert conflict.value.reason_code == "precondition_failed"


@pytest.mark.integration
def test_opt_in_local_reference_server_profile() -> None:
    if os.environ.get("OPENMED_FHIR_REFERENCE_TEST") != "1":
        pytest.skip("set OPENMED_FHIR_REFERENCE_TEST=1 for isolated local containers")
    server = os.environ.get("OPENMED_FHIR_REFERENCE_SERVER", "")
    version = os.environ.get("OPENMED_FHIR_REFERENCE_VERSION", "")
    base_url = os.environ.get("OPENMED_FHIR_REFERENCE_URL", "")
    profile = ReferenceProfile(server, version, base_url)
    with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
        results = run_reference_server(profile, client)
    assert {result.case for result in results} == set(CASES)
    assert all(
        result.to_dict()["matrix_version"] == MATRIX_VERSION for result in results
    )
    assert all(result.outcome is not Outcome.FAIL for result in results)
