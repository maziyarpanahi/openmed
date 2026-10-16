"""Offline vectors for the injected FHIR execution boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Lock
from types import SimpleNamespace as Vector

import pytest

from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.interop.fhir.write_client import (
    FHIRHTTPResponse,
    FHIRWriteClient,
    FHIRWriteError,
    FHIRWriteLimits,
    FHIRWriteOutcome,
    FHIRWriteStatus,
)
from openmed.interop.fhir_capability_preflight import (
    FHIRCapabilityMetadata,
    FHIRResourceCapability,
)

KEY = "fhir-cw-v1-" + "a" * 64
HANDLE = "smart_" + "H" * 43
SECRET = b"synthetic-commitment-key-32-bytes!"
NOW = datetime(2026, 10, 9, 8, 0, tzinfo=timezone.utc)
AUDIENCE = "https://synthetic.invalid/fhir"
PRIVATE = "synthetic-private-note-987654"
ROLE = "role:org.openmed/clinical-reviewer@1.0.0"


def _manifest(**changes):
    record = Vector(
        resource_handle="res_" + "a" * 32,
        field_path="status",
        evidence=(Vector(digest="b" * 64, start=0, end=4),),
        step_digests=("c" * 64,),
        policy_digest="d" * 64,
        approval_receipt_digest="e" * 64,
        target_digest="f" * 64,
        plan_key=KEY,
    )
    record.__dict__.update(changes)
    return Vector(action_digest="1" * 64, records=(record,))


def _plan(kind="create", **changes):
    result = Vector(
        kind=kind,
        resource_type="Observation",
        predicate=Vector(canonical_query="identifier=urn%3Asynthetic%7C123"),
        idempotency_key=KEY,
    )
    result.__dict__.update(changes)
    return result


def _capabilities():
    return FHIRCapabilityMetadata(
        "4.0.1",
        tuple(
            FHIRResourceCapability(t, frozenset({"create", "update"}), True, True)
            for t in ("Observation", "Provenance")
        ),
        frozenset({"transaction"}),
    )


class _Ledger:
    """Synthetic process-local test ledger, never a production durability claim."""

    def __init__(self):
        self.claims = {}
        self.results = {}
        self.lock = Lock()
        self.ack = True

    def claim(self, key, action):
        with self.lock:
            if key in self.claims:
                return False
            self.claims[key] = action
            return True

    def lookup(self, key):
        return self.results.get(key)

    def finish(self, key, action, result):
        assert self.claims[key] == action
        self.results[key] = result
        return self.ack


def _response(status=201, **changes):
    value = dict(
        status_code=status,
        body=json.dumps(
            {
                "resourceType": "Observation",
                "id": "synthetic-1",
                "meta": {"versionId": "2"},
                "note": [{"text": PRIVATE}],
            }
        ).encode(),
        headers=(("Content-Type", "application/fhir+json"),),
    )
    value.update(changes)
    return FHIRHTTPResponse(**value)


def _setup(**overrides):
    env = Vector(
        now=NOW,
        requests=[],
        manifest=_manifest(),
        allow=True,
        broker=None,
        credential_ok=True,
        calls=0,
        clock_calls=0,
    )

    def transport(request):
        env.requests.append(request)
        return env.response

    def factory(sender):
        def dispatch(handle, *, audience, required_scopes):
            assert handle == HANDLE and audience == AUDIENCE
            assert tuple(required_scopes) in {
                ("user/Observation.c",),
                ("user/Observation.u",),
            }
            if not env.credential_ok:
                raise ValueError(PRIVATE)
            sender(audience, "Bearer SyntheticPrivateToken")

        env.broker = Vector(dispatch=dispatch, sender=sender)
        return env.broker

    def authorize(prepared, receipt):
        env.calls += 1
        return env.allow

    def clock():
        env.clock_calls += 1
        return env.now

    env.response = _response()
    config = dict(
        audience=AUDIENCE,
        secret=SECRET,
        capabilities=_capabilities(),
        transport=transport,
        custody_factory=factory,
        ledger=_Ledger(),
        authorize=authorize,
        verify_lineage=lambda p: env.manifest,
        clock=clock,
    )
    config.update(overrides)
    env.client = FHIRWriteClient(**config)
    env.ledger = config["ledger"]
    return env


def _prepared(env, kind="create", resource=None, **changes):
    return env.client.prepare_conditional(
        _plan(kind),
        resource
        or {
            "resourceType": "Observation",
            "status": "final",
            "note": [{"text": PRIVATE}],
        },
        credential_handle=HANDLE,
        lineage=env.manifest,
        precondition=Vector(if_match='W/"1"') if kind == "update" else None,
        **changes,
    )


def _authorization(prepared):
    key = b"synthetic-local-human-approval-key"
    instant = int(NOW.timestamp())
    token = ApprovalTokenSigner(key, clock=lambda: instant).issue(
        action_digest=prepared.action_digest,
        reviewer_role=ROLE,
        expires_at=instant + 60,
    )
    return ApprovalTokenVerifier(
        key, InMemoryApprovalNonceStore(), clock=lambda: instant
    ).consume_authorization(
        token, action_digest=prepared.action_digest, reviewer_role=ROLE
    )


@pytest.mark.parametrize(
    "kind,method,path,header",
    [
        ("create", "POST", "Observation", "If-None-Exist"),
        ("update", "PUT", "Observation?identifier=urn%3Asynthetic%7C123", "If-Match"),
    ],
)
def test_exact_wire_request_and_safe_authorization(kind, method, path, header):
    env = _setup()
    proposed = _prepared(env, kind)
    outcome = env.client.submit(proposed, _authorization(proposed))
    assert outcome.status is FHIRWriteStatus.COMMITTED
    request = env.requests[0]
    assert request.method == method and request.url == AUDIENCE + "/" + path
    assert dict(request.headers)[header] == (
        'W/"1"' if kind == "update" else "identifier=urn%3Asynthetic%7C123"
    )
    assert dict(request.headers)["Idempotency-Key"] == KEY
    assert dict(request.headers)["Authorization"] == "Bearer SyntheticPrivateToken"
    assert request.body == proposed._body
    assert request.retries == 0 and request.follow_redirects is False
    assert request.timeout_seconds == 10 and request.max_response_bytes == 262144
    assert env.calls == 3
    rendered = (
        repr(proposed)
        + repr(request)
        + repr(env.response)
        + repr(outcome)
        + json.dumps(outcome.to_dict())
        + json.dumps(proposed.to_dict())
    )
    for private in (
        PRIVATE,
        "SyntheticPrivateToken",
        AUDIENCE,
        "urn%3Asynthetic",
        HANDLE,
        'W/"1"',
    ):
        assert private not in rendered


def test_no_mutation_of_snapshot_or_detached_review_copy():
    env = _setup()
    resource = {
        "resourceType": "Observation",
        "status": "final",
        "note": [{"text": PRIVATE}],
    }
    proposed = _prepared(env, resource=resource)
    resource["note"][0]["text"] = "changed"
    proposed.payload["note"][0]["text"] = "changed-again"
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.COMMITTED
    )
    assert PRIVATE.encode() in env.requests[0].body


@pytest.mark.parametrize(
    "field,value",
    [
        ("_body", b"{}"),
        ("_path", "Patient"),
        ("_method", "DELETE"),
        ("_handle", "smart_" + "Z" * 43),
        ("_headers", ()),
        ("_scopes", ("user/Patient.write",)),
        ("_lineage", b"{}"),
        ("_types", ("Patient",)),
        ("resource_count", 2),
        ("idempotency_key", "fhir-cw-v1-" + "b" * 64),
        ("payload_digest", "sha256:" + "a" * 64),
    ],
)
def test_tampered_review_proposal_never_dispatches(field, value):
    env = _setup()
    proposed = _prepared(env, "update")
    with pytest.raises(FHIRWriteError, match="invalid_plan"):
        env.client.submit(replace(proposed, **{field: value}), _authorization(proposed))
    assert not env.requests and not env.ledger.claims


@pytest.mark.parametrize("allow", [False, None, 1, "yes", [], {}])
def test_authorization_requires_exact_true(allow):
    env = _setup()
    env.allow = allow
    proposed = _prepared(env)
    result = env.client.submit(proposed, _authorization(proposed))
    assert result.status is FHIRWriteStatus.REJECTED
    assert result.reason_code == "authorization_rejected"
    assert not env.requests and not env.ledger.claims


@pytest.mark.parametrize("offset", [-1, 60, 61])
def test_receipt_future_or_expired(offset):
    env = _setup()
    env.now += timedelta(seconds=offset)
    proposed = _prepared(env)
    result = env.client.submit(proposed, _authorization(proposed))
    assert result.reason_code == "authorization_expired" and not env.requests


def test_expiry_during_lineage_verification_is_rechecked():
    env = _setup()
    proposed = _prepared(env)

    def lineage(_):
        env.now += timedelta(seconds=60)
        return env.manifest

    env.client._verify_lineage = lineage
    assert (
        env.client.submit(proposed, _authorization(proposed)).reason_code
        == "authorization_expired"
    )
    assert not env.requests


@pytest.mark.parametrize("at", [2, 3])
def test_revocation_between_reservation_custody_and_send(at):
    env = _setup()
    proposed = _prepared(env)

    def authorize(*_):
        env.calls += 1
        return env.calls < at

    env.client._authorize = authorize
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.REJECTED
    )
    assert not env.requests


def test_credential_denial_and_out_of_band_sender_do_not_send():
    env = _setup()
    proposed = _prepared(env)
    env.credential_ok = False
    assert (
        env.client.submit(proposed, _authorization(proposed)).reason_code
        == "credential_rejected"
    )
    with pytest.raises(FHIRWriteError, match="credential_rejected"):
        env.broker.sender(AUDIENCE, "Bearer SyntheticPrivateToken")
    assert not env.requests


@pytest.mark.parametrize(
    "exception", [TimeoutError(PRIVATE), ConnectionError(PRIVATE), ValueError(PRIVATE)]
)
def test_post_submission_exception_is_unknown_without_retry(exception):
    env = _setup()
    proposed = _prepared(env)

    def fail(request):
        env.requests.append(request)
        raise exception

    env.client._transport = fail
    receipt = _authorization(proposed)
    first = env.client.submit(proposed, receipt)
    assert first.status is FHIRWriteStatus.UNKNOWN and first.reconciliation_required
    assert env.client.submit(proposed, receipt) == first
    assert env.client.reconcile(proposed) == first
    assert len(env.requests) == 1 and PRIVATE not in json.dumps(first.to_dict())


def test_duplicate_and_changed_payload_same_key():
    env = _setup()
    original = _prepared(env)
    receipt = _authorization(original)
    result = env.client.submit(original, receipt)
    assert env.client.submit(original, receipt) == result
    changed = _prepared(
        env, resource={"resourceType": "Observation", "status": "amended"}
    )
    assert changed.action_digest != original.action_digest
    assert (
        env.client.submit(changed, _authorization(changed)).reason_code
        == "idempotency_mismatch"
    )
    assert len(env.requests) == 1


def test_reserved_key_after_crash_is_unknown_and_not_replayed():
    env = _setup()
    proposed = _prepared(env)
    env.ledger.claim(KEY, proposed.action_digest)
    assert (
        env.client.submit(proposed, _authorization(proposed)).reason_code
        == "attempt_in_progress"
    )
    assert env.client.reconcile(proposed).status is FHIRWriteStatus.UNKNOWN
    assert not env.requests


@pytest.mark.parametrize("method", ["claim", "lookup", "finish"])
def test_failed_storage_acknowledgements(method):
    env = _setup()
    proposed = _prepared(env)

    def fail(*_):
        raise OSError(PRIVATE)

    if method == "lookup":
        env.ledger.claim(KEY, proposed.action_digest)
    setattr(env.ledger, method, fail)
    result = env.client.submit(proposed, _authorization(proposed))
    assert result.reason_code == (
        "receipt_uncertain" if method == "finish" else "ledger_unavailable"
    )
    assert len(env.requests) == (1 if method == "finish" else 0)
    assert PRIVATE not in repr(result)


def test_lost_finish_ack_preserves_durable_evidence_for_read_only_reconcile():
    env = _setup()
    proposed = _prepared(env)
    env.ledger.ack = False
    result = env.client.submit(proposed, _authorization(proposed))
    assert result.status is FHIRWriteStatus.UNKNOWN
    assert env.client.reconcile(proposed).status is FHIRWriteStatus.COMMITTED
    assert len(env.requests) == 1


@pytest.mark.parametrize(
    "status,expected,reason",
    [
        (400, "rejected", "server_rejected"),
        (401, "rejected", "server_rejected"),
        (403, "rejected", "server_rejected"),
        (422, "rejected", "server_rejected"),
        (409, "conflict", "server_conflict"),
        (412, "conflict", "precondition_failed"),
        (428, "conflict", "precondition_required"),
        (301, "unknown_commit", "redirect_received"),
        (302, "unknown_commit", "redirect_received"),
        (307, "unknown_commit", "redirect_received"),
        (308, "unknown_commit", "redirect_received"),
        (408, "unknown_commit", "unexpected_response"),
        (429, "unknown_commit", "unexpected_response"),
        (500, "unknown_commit", "unexpected_response"),
        (503, "unknown_commit", "unexpected_response"),
        (202, "unknown_commit", "unexpected_response"),
    ],
)
def test_closed_status_classification(status, expected, reason):
    env = _setup()
    proposed = _prepared(env)
    env.response = _response(status)
    result = env.client.submit(proposed, _authorization(proposed))
    assert result.status.value == expected and result.reason_code == reason
    assert len(env.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b"[]",
        b"null",
        b"not-json",
        b"\xff",
        b'{"resourceType":"Patient","resourceType":"Observation"}',
        b'{"resourceType":"Observation","id":"x","meta":{"versionId":"1"},"x":NaN}',
        json.dumps(
            {"resourceType": "OperationOutcome", "issue": [{"diagnostics": PRIVATE}]}
        ).encode(),
        json.dumps(
            {"resourceType": "Patient", "id": "x", "meta": {"versionId": "1"}}
        ).encode(),
        json.dumps({"resourceType": "Observation", "id": "x"}).encode(),
    ],
)
def test_unexpected_resource_acknowledgements_are_unknown(body):
    env = _setup()
    proposed = _prepared(env)
    env.response = _response(body=body)
    result = env.client.submit(proposed, _authorization(proposed))
    assert (
        result.reason_code == "unexpected_response" and result.reconciliation_required
    )


@pytest.mark.parametrize(
    "headers",
    [
        (),
        (("Content-Type", "text/html"),),
        (
            ("Content-Type", "application/fhir+json"),
            ("content-type", "application/fhir+json"),
        ),
        (("Content-Type", "application/fhir+json\r\n"),),
        (("bad header", "private"),),
        (("X-Large", "x" * 32769),),
    ],
)
def test_unexpected_headers_do_not_claim_commit(headers):
    env = _setup()
    proposed = _prepared(env)
    env.response = _response(headers=headers)
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.UNKNOWN
    )


@pytest.mark.parametrize(
    "location,etag",
    [
        ("Observation/synthetic-1/_history/2", 'W/"2"'),
        (AUDIENCE + "/Observation/synthetic-1/_history/2", 'W/"2"'),
    ],
)
def test_minimal_resource_acknowledgement(location, etag):
    env = _setup()
    proposed = _prepared(env)
    env.response = FHIRHTTPResponse(
        204, headers=(("Location", location), ("ETag", etag))
    )
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.COMMITTED
    )


@pytest.mark.parametrize(
    "location,etag",
    [
        ("https://untrusted.invalid/Observation/x/_history/2", 'W/"2"'),
        ("Observation/x", 'W/"2"'),
        ("Observation/x/_history/2", 'W/"3"'),
        ("Patient/x/_history/2", 'W/"2"'),
        ("Observation/x/_history/2", '"2"'),
    ],
)
def test_wrong_minimal_ack_is_unknown(location, etag):
    env = _setup()
    proposed = _prepared(env)
    env.response = FHIRHTTPResponse(
        204, headers=(("Location", location), ("ETag", etag))
    )
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.UNKNOWN
    )


def test_oversized_response_and_json_depth_are_bounded():
    env = _setup(limits=FHIRWriteLimits(max_response_bytes=300))
    proposed = _prepared(env)
    env.response = _response(body=b"x" * 301)
    assert (
        env.client.submit(proposed, _authorization(proposed)).reason_code
        == "response_limit_exceeded"
    )
    env = _setup(limits=FHIRWriteLimits(max_json_depth=6))
    proposed = _prepared(env)
    env.response = _response(body=b'{"deep": [[[[[[[[1]]]]]]]]}')
    assert (
        env.client.submit(proposed, _authorization(proposed)).reason_code
        == "unexpected_response"
    )


@pytest.mark.parametrize(
    "resource",
    [
        {},
        {"resourceType": "Patient"},
        {"resourceType": "Observation", "x": float("nan")},
        {"resourceType": "Observation", "x": object()},
        {"resourceType": "Observation", 1: "x"},
        {"resourceType": "Observation", "id": PRIVATE + "/unsafe"},
    ],
)
def test_invalid_payload_never_reaches_sender(resource):
    env = _setup()
    with pytest.raises(FHIRWriteError):
        env.client.prepare_conditional(
            _plan(), resource, credential_handle=HANDLE, lineage=env.manifest
        )
    assert not env.requests


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?identifier=x",
        "identifier=x\r\nAuthorization:y",
        "identifier=%ZZ",
        "identifier=",
        "identifier=%0D",
        "=x",
        "identifier=" + "x" * 4097,
    ],
)
def test_invalid_predicate_is_value_free(query):
    env = _setup()
    with pytest.raises(FHIRWriteError, match="invalid_plan") as caught:
        env.client.prepare_conditional(
            _plan(predicate=Vector(canonical_query=query)),
            {"resourceType": "Observation"},
            credential_handle=HANDLE,
            lineage=env.manifest,
        )
    assert query not in str(caught.value) or query == ""


@pytest.mark.parametrize(
    "precondition",
    [
        None,
        Vector(if_match='"1"'),
        Vector(if_match='W/"1"\r\nX:yes'),
        Vector(if_match=1),
        Vector(if_match='W/""'),
    ],
)
def test_update_requires_atomic_version_header(precondition):
    env = _setup()
    with pytest.raises(FHIRWriteError, match="invalid_plan"):
        env.client.prepare_conditional(
            _plan("update"),
            {"resourceType": "Observation"},
            credential_handle=HANDLE,
            lineage=env.manifest,
            precondition=precondition,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"policy_digest": PRIVATE},
        {"target_digest": PRIVATE},
        {"approval_receipt_digest": PRIVATE},
        {"resource_handle": PRIVATE},
        {"field_path": "bad/key"},
        {"plan_key": "fhir-cw-v1-" + "b" * 64},
        {"evidence": ()},
        {"step_digests": ()},
    ],
)
def test_incomplete_lineage_never_prepares(change):
    env = _setup()
    env.manifest = _manifest(**change)
    with pytest.raises(FHIRWriteError, match="invalid_lineage"):
        _prepared(env)


def test_changed_lineage_gate_refuses_before_claim():
    env = _setup()
    proposed = _prepared(env)
    env.manifest = _manifest(policy_digest="3" * 64)
    assert (
        env.client.submit(proposed, _authorization(proposed)).reason_code
        == "invalid_lineage"
    )
    assert not env.requests and not env.ledger.claims


@pytest.mark.parametrize(
    "audience",
    [
        "http://synthetic.invalid/fhir",
        "https://x.invalid/fhir/",
        "https://u:p@x.invalid/fhir",
        "https://x.invalid/fhir?a=b",
        "https://x.invalid/fhir#frag",
        "https://x.invalid/a/../fhir",
        "https://x.invalid/%2e%2e/fhir",
        "https://x.invalid\r\n/fhir",
        "https://x.invalid:bad/fhir",
        "file:///tmp/fhir",
    ],
)
def test_invalid_operator_endpoint_is_sanitized(audience):
    with pytest.raises(FHIRWriteError, match="invalid_configuration") as caught:
        _setup(audience=audience)
    assert audience not in str(caught.value)


@pytest.mark.parametrize(
    "change",
    [
        {"max_request_bytes": 0},
        {"max_response_bytes": True},
        {"max_entries": 1001},
        {"max_json_depth": 65},
        {"max_json_nodes": 100001},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
    ],
)
def test_invalid_limits_are_closed(change):
    with pytest.raises(FHIRWriteError, match="invalid_configuration"):
        FHIRWriteLimits(**change)


def test_capability_denial_and_different_audience_binding():
    env = _setup(capabilities=FHIRCapabilityMetadata("4.0.1", (), frozenset()))
    with pytest.raises(FHIRWriteError, match="unsupported_capability"):
        _prepared(env)
    original = _setup()
    proposed = _prepared(original)
    other = _setup(audience="https://other.invalid/fhir")
    with pytest.raises(FHIRWriteError, match="invalid_plan"):
        other.client.submit(proposed, _authorization(proposed))


def _transaction():
    payload = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": "urn:uuid:11111111-1111-4111-8111-111111111111",
                "resource": {"resourceType": "Observation", "status": "final"},
                "request": {
                    "method": "PUT",
                    "url": "Observation?identifier=urn%3Asynthetic%7C123",
                    "ifMatch": 'W/"1"',
                },
            },
            {
                "fullUrl": "urn:uuid:22222222-2222-4222-8222-222222222222",
                "resource": {
                    "resourceType": "Provenance",
                    "target": [
                        {"reference": "urn:uuid:11111111-1111-4111-8111-111111111111"}
                    ],
                },
                "request": {"method": "POST", "url": "Provenance"},
            },
        ],
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    return Vector(
        serialized=body,
        bundle_digest="sha256:" + hashlib.sha256(body).hexdigest(),
        entry_count=2,
    )


def test_transaction_preserves_exact_bytes_and_checks_all_scopes():
    env = _setup()
    transaction = _transaction()
    scopes = []

    def dispatch(handle, *, audience, required_scopes):
        scopes.extend(required_scopes)
        env.broker.sender(audience, "Bearer SyntheticPrivateToken")

    env.broker.dispatch = dispatch
    prepared = env.client.prepare_transaction(
        transaction, idempotency_key=KEY, credential_handle=HANDLE, lineage=env.manifest
    )
    response = {
        "resourceType": "Bundle",
        "type": "transaction-response",
        "entry": [
            {
                "response": {
                    "status": "200 OK",
                    "location": "Observation/x/_history/2",
                    "etag": 'W/"2"',
                }
            },
            {
                "response": {
                    "status": "201 Created",
                    "location": "Provenance/p/_history/1",
                    "etag": 'W/"1"',
                }
            },
        ],
    }
    env.response = FHIRHTTPResponse(
        200, json.dumps(response).encode(), (("Content-Type", "application/fhir+json"),)
    )
    assert (
        env.client.submit(prepared, _authorization(prepared)).status
        is FHIRWriteStatus.COMMITTED
    )
    assert env.requests[0].url == AUDIENCE and env.requests[0].method == "POST"
    assert env.requests[0].body == transaction.serialized
    assert scopes == ["user/Observation.u", "user/Provenance.c"]


@pytest.mark.parametrize(
    "problem",
    ["digest", "count", "missing_version", "wrong_type", "extra_method", "batch"],
)
def test_malformed_transactions_refused(problem):
    env = _setup()
    transaction = _transaction()
    payload = json.loads(transaction.serialized)
    if problem == "digest":
        transaction.bundle_digest = "sha256:" + "0" * 64
    elif problem == "count":
        transaction.entry_count = 3
    else:
        if problem == "missing_version":
            del payload["entry"][0]["request"]["ifMatch"]
        elif problem == "wrong_type":
            payload["entry"][0]["request"]["url"] = "Patient/x"
        elif problem == "extra_method":
            payload["entry"][0]["request"]["method"] = "DELETE"
        elif problem == "batch":
            payload["type"] = "batch"
        transaction.serialized = json.dumps(payload).encode()
        transaction.bundle_digest = (
            "sha256:" + hashlib.sha256(transaction.serialized).hexdigest()
        )
    with pytest.raises(FHIRWriteError):
        env.client.prepare_transaction(
            transaction,
            idempotency_key=KEY,
            credential_handle=HANDLE,
            lineage=env.manifest,
        )
    assert not env.requests


def test_transaction_mixed_or_incomplete_acknowledgement_is_unknown():
    env = _setup()
    transaction = _transaction()
    prepared = env.client.prepare_transaction(
        transaction, idempotency_key=KEY, credential_handle=HANDLE, lineage=env.manifest
    )
    env.broker.dispatch = lambda h, *, audience, required_scopes: env.broker.sender(
        audience, "Bearer SyntheticPrivateToken"
    )
    env.response = _response(
        200,
        body=json.dumps(
            {
                "resourceType": "Bundle",
                "type": "transaction-response",
                "entry": [
                    {"response": {"status": "200 OK"}},
                    {"response": {"status": "409 Conflict"}},
                ],
            }
        ).encode(),
    )
    assert (
        env.client.submit(prepared, _authorization(prepared)).status
        is FHIRWriteStatus.UNKNOWN
    )


def test_invalid_response_objects_and_malformed_outcomes_are_closed():
    env = _setup()
    proposed = _prepared(env)
    env.response = {"status": 201, "private": PRIVATE}
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.UNKNOWN
    )
    with pytest.raises(FHIRWriteError):
        FHIRWriteOutcome(
            FHIRWriteStatus.COMMITTED,
            PRIVATE,
            proposed.action_digest,
            proposed.payload_digest,
            1,
        )


@pytest.mark.parametrize(
    "query",
    [
        "identifier=a&identifier=b",
        "identifier=a%2Cb",
        "_count=1",
        "identifier=" + "x" * 2049,
    ],
)
def test_predicates_follow_existing_single_value_plan_contract(query):
    env = _setup()
    with pytest.raises(FHIRWriteError, match="invalid_plan"):
        env.client.prepare_conditional(
            _plan(predicate=Vector(canonical_query=query)),
            {"resourceType": "Observation"},
            credential_handle=HANDLE,
            lineage=env.manifest,
        )


def test_acknowledgement_must_match_supplied_resource_id():
    env = _setup()
    proposed = _prepared(
        env,
        "update",
        resource={
            "resourceType": "Observation",
            "id": "expected-synthetic",
            "status": "final",
        },
    )
    assert (
        env.client.submit(proposed, _authorization(proposed)).status
        is FHIRWriteStatus.UNKNOWN
    )


def test_package_exports_are_lazy_and_identical():
    from openmed.interop import fhir
    from openmed.interop.fhir import write_client

    for name in write_client.__all__:
        assert name in fhir.__all__
        assert getattr(fhir, name) is getattr(write_client, name)


@pytest.mark.parametrize("proof", ["receipt", "mapping", "other_action"])
def test_unverified_metadata_and_different_action_cannot_authorize_dispatch(proof):
    env = _setup()
    proposed = _prepared(env)
    authority = _authorization(proposed)
    if proof == "receipt":
        candidate = authority.receipt
    elif proof == "mapping":
        candidate = authority.receipt.to_dict()
    else:
        changed = _prepared(
            env, resource={"resourceType": "Observation", "status": "amended"}
        )
        candidate = _authorization(changed)
    result = env.client.submit(proposed, candidate)
    assert result.reason_code == "authorization_rejected"
    assert not env.requests and not env.ledger.claims
