"""Offline HTTP controls over a synthetic, custodied governance service."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import (
    ApprovalReceipt,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.workflows.recovery import CompensationLimit, EffectKind, EffectRecord
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import DEFAULT_ROUTE_SCOPES, hash_api_key
from openmed.service.governed_workflows import (
    MAX_WORKFLOW_REQUEST_BYTES,
    WORKFLOW_ERROR_STATUSES,
    WorkflowHTTPPolicy,
    WorkflowReceiptVerification,
    WorkflowReference,
    WorkflowServiceError,
    WorkflowView,
    workflow_receipt_digest,
)
from openmed.service.logging import ACCESS_LOGGER_NAME
from openmed.service.schemas import workflow_http_schemas
from openmed.service.workflow_routes import WORKFLOW_ROUTE_SCOPES

ACTION = "sha256:" + "a" * 64
STATE = "sha256:" + "b" * 64
NEXT_STATE = "sha256:" + "c" * 64
KEY = "synthetic-workflow-key"
PRIVATE = "Synthetic private note /private/test-only credential-marker"
BASE = "http://127.0.0.1"


def reference():
    return WorkflowReference(
        RunId.parse("run_" + "1" * 32),
        WorkflowId.parse("workflow:test.example/review@1.0.0"),
        ACTION,
        STATE,
        "req_" + "2" * 32,
    )


class FakeLoader:
    def __init__(self, config):
        self.config = config

    def loaded_models(self):
        return {}


class CustodiedService:
    """Synthetic authority and atomic dedup fixture, never a shipped engine."""

    def __init__(self):
        signing_key = b"synthetic-offline-approval-key-32!!"
        token = ApprovalTokenSigner(signing_key, clock=lambda: 100).issue(
            action_digest=ACTION,
            reviewer_role="role:test.example/reviewer",
            expires_at=200,
            nonce_source=lambda size: b"x" * size,
        )
        self.authorization = ApprovalTokenVerifier(
            signing_key, InMemoryApprovalNonceStore(), clock=lambda: 100
        ).consume_authorization(
            token, action_digest=ACTION, reviewer_role="role:test.example/reviewer"
        )
        self.receipt = self.authorization.receipt
        effect = EffectRecord.create(
            ordinal=0,
            run_id=reference().run_id,
            action_id=ActionId.parse("act_" + "3" * 32),
            tool_id=ToolId.parse("tool:test.example/fhir@1.0.0"),
            kind=EffectKind.FHIR_WRITE,
            operation_digest=ACTION,
            approval_required=True,
            compensation_limit=CompensationLimit.PROPOSE_ONLY,
        )
        self.view = WorkflowView(
            reference(), STATE, ActionPhase.WAITING_REVIEW, (effect,)
        )
        self.receipt_consumed_at = self.authorization.consumed_at
        self.receipt_expires_at = self.authorization.expires_at
        self.authority_valid = True
        self.lock = threading.Lock()
        self.ledger = {}
        self.mutations = 0
        self.calls = []
        self.execute = Mock(side_effect=AssertionError("effect execution forbidden"))
        self.compensate = Mock(side_effect=AssertionError("compensation forbidden"))

    def _owner(self, principal, ref):
        if principal.subject != "operator":
            raise WorkflowServiceError("workflow_forbidden")
        if (
            ref.run_id != self.view.reference.run_id
            or ref.workflow_id != self.view.reference.workflow_id
            or ref.action_digest != ACTION
        ):
            raise WorkflowServiceError("workflow_conflict")

    def _read(self, operation, principal, ref):
        with self.lock:
            self._owner(principal, ref)
            self.calls.append(operation)
            return self.view

    def preflight(self, principal, ref):
        return self._read("preflight", principal, ref)

    def preview(self, principal, ref):
        return self._read("preview", principal, ref)

    def status(self, principal, ref):
        return self._read("status", principal, ref)

    def _verify(self, principal, ref, receipt, now):
        self._owner(principal, ref)
        if (
            not self.authority_valid
            or self.authorization.reviewer_role != "role:test.example/reviewer"
            or receipt != self.receipt
        ):
            raise WorkflowServiceError("workflow_receipt_unverified")
        if now < self.receipt_consumed_at:
            raise WorkflowServiceError("workflow_receipt_future")
        if now >= self.receipt_expires_at:
            raise WorkflowServiceError("workflow_receipt_expired")

    def verify_receipt(self, principal, ref, receipt, *, now):
        with self.lock:
            self.calls.append("verify")
            self._verify(principal, ref, receipt, now)
            return WorkflowReceiptVerification(
                ref, workflow_receipt_digest(receipt), True
            )

    def _mutate(self, operation, principal, ref, receipt=None, now=100):
        with self.lock:
            self._owner(principal, ref)
            fingerprint = (
                operation,
                ref.to_dict(),
                receipt.to_dict() if receipt else None,
            )
            if ref.request_id in self.ledger:
                stored, ack = self.ledger[ref.request_id]
                if stored != fingerprint:
                    raise WorkflowServiceError("workflow_conflict")
                return ack
            if ref.expected_state_digest != self.view.state_digest:
                raise WorkflowServiceError("workflow_conflict")
            if self.view.phase in (ActionPhase.ABORTED, ActionPhase.COMPLETED):
                raise WorkflowServiceError("workflow_terminal")
            if receipt is not None:
                self._verify(principal, ref, receipt, now)
            self.view = replace(
                self.view,
                state_digest=NEXT_STATE,
                receipt_digest=workflow_receipt_digest(receipt)
                if receipt
                else self.view.receipt_digest,
                cancellation_requested=operation == "cancel",
            )
            self.ledger[ref.request_id] = (fingerprint, self.view)
            self.mutations += 1
            return self.view

    def submit_receipt(self, principal, ref, receipt, *, now):
        self.calls.append("submit")
        return self._mutate("review", principal, ref, receipt, now)

    def cancel(self, principal, ref):
        self.calls.append("cancel")
        return self._mutate("cancel", principal, ref)


@pytest.fixture(autouse=True)
def isolated_service_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENMED_SERVICE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    configure_keys(monkeypatch, ["workflow:*"])


def configure_keys(monkeypatch, scopes):
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "key_id": "fixture",
                    "key_hash": hash_api_key(KEY),
                    "principal": "operator",
                    "scopes": scopes,
                },
                {
                    "key_id": "other",
                    "key_hash": hash_api_key("synthetic-other-key"),
                    "principal": "other-operator",
                    "scopes": ["workflow:*"],
                },
            ]
        ),
    )


def client(service=None, *, policy=None, clock=None):
    return TestClient(
        create_app(
            workflow_service=service,
            workflow_policy=policy or WorkflowHTTPPolicy(True, True, True),
            workflow_clock=clock or (lambda: 100),
        ),
        base_url=BASE,
    )


def post(http, operation, payload=None, *, key=KEY):
    return http.post(
        "/v1/workflows/" + operation,
        json=reference().to_dict() if payload is None else payload,
        headers={"X-API-Key": key} if key else {},
    )


def review_payload(service):
    return {**reference().to_dict(), "receipt": service.receipt.to_dict()}


@pytest.mark.parametrize("operation", ["preflight", "preview", "status"])
def test_reads_use_custody_without_effects(operation):
    service = CustodiedService()
    with client(service) as http:
        response = post(http, operation)
    assert response.status_code == 200
    assert response.json() == service.view.to_dict()
    assert response.headers["cache-control"] == "no-store"
    assert service.calls == [operation]
    assert service.mutations == 0
    service.execute.assert_not_called()
    service.compensate.assert_not_called()


@pytest.mark.parametrize(
    "key,status",
    [(None, 401), ("wrong-synthetic-key", 401), ("synthetic-other-key", 403)],
)
def test_authenticated_caller_and_run_custody_required(key, status):
    service = CustodiedService()
    with client(service) as http:
        response = post(http, "status", key=key)
    assert response.status_code == status
    assert service.mutations == 0


@pytest.mark.parametrize("operation", list(WORKFLOW_ROUTE_SCOPES))
def test_scopes_cannot_be_replaced_by_body_authority(monkeypatch, operation):
    configure_keys(monkeypatch, ["unrelated:read"])
    service = CustodiedService()
    with client(service) as http:
        response = http.post(
            operation,
            json={**review_payload(service), "role": "admin"},
            headers={"X-API-Key": KEY},
        )
    assert response.status_code == 403
    assert not service.calls


@pytest.mark.parametrize("mode", ["disabled", "exempt", "scope-override"])
def test_weak_global_auth_policy_cannot_open_routes(monkeypatch, mode):
    service = CustodiedService()
    app = create_app(workflow_service=service, workflow_policy=WorkflowHTTPPolicy(True))
    config = app.state.auth.config
    if mode == "disabled":
        app.state.auth.config = replace(config, enabled=False)
    elif mode == "exempt":
        app.state.auth.config = replace(
            config, exempt_paths=config.exempt_paths | {"/v1/workflows/status"}
        )
    else:
        configure_keys(monkeypatch, [])
        app = create_app(
            workflow_service=service, workflow_policy=WorkflowHTTPPolicy(True)
        )
        app.state.auth.config = replace(
            app.state.auth.config, route_scopes={("POST", "/v1/workflows/status"): ()}
        )
    with TestClient(app, base_url=BASE) as http:
        response = post(http, "status")
    assert response.status_code == (403 if mode == "scope-override" else 401)
    assert not service.calls


def test_server_owned_enable_and_mutation_opt_ins():
    service = CustodiedService()
    with client(service, policy=WorkflowHTTPPolicy()) as http:
        assert post(http, "status").status_code == 503
    with client(service, policy=WorkflowHTTPPolicy(True)) as http:
        assert post(http, "review-receipts", review_payload(service)).status_code == 403
        assert post(http, "cancel").status_code == 403
    with client() as http:
        assert post(http, "status").status_code == 503
    assert not service.calls


def test_actual_consumed_receipt_duplicate_ack_and_cancellation_intent():
    service = CustodiedService()
    with client(service) as http:
        first = post(http, "review-receipts", review_payload(service))
        duplicate = post(http, "review-receipts", review_payload(service))
        assert first.status_code == duplicate.status_code == 200
        assert first.json() == duplicate.json()
        assert service.mutations == 1
        # Reusing the same id for a different mutation is rejected atomically.
        assert post(http, "cancel").status_code == 409
        cancel = {
            **reference().to_dict(),
            "expected_state_digest": NEXT_STATE,
            "request_id": "req_" + "4" * 32,
        }
        cancelled = post(http, "cancel", cancel)
        again = post(http, "cancel", cancel)
        assert cancelled.status_code == again.status_code == 200
        assert cancelled.json() == again.json()
        assert cancelled.json()["cancellation_requested"] is True
        assert cancelled.json()["committed_effect_count"] == 0
        assert cancelled.json()["phase"] == ActionPhase.WAITING_REVIEW.value
    assert service.mutations == 2
    service.execute.assert_not_called()
    service.compensate.assert_not_called()


@pytest.mark.parametrize(
    "change,code",
    [
        ({"reviewer_role": "role:test.example/admin"}, "workflow_invalid_input"),
        ({"token_digest": NEXT_STATE}, "workflow_receipt_unverified"),
        ({"action_digest": NEXT_STATE}, "workflow_conflict"),
        ({"consumed_at": 101}, "workflow_invalid_input"),
        ({"expires_at": 100, "consumed_at": 99}, "workflow_invalid_input"),
        ({"expires_at": True}, "workflow_invalid_input"),
        ({"approved": True}, "workflow_invalid_input"),
    ],
)
def test_caller_receipt_claims_never_grant_authority(change, code):
    service = CustodiedService()
    payload = review_payload(service)
    payload["receipt"].update(change)
    with client(service) as http:
        response = post(http, "review-receipts", payload)
    assert response.json()["error"]["code"] == code
    assert service.mutations == 0
    service.execute.assert_not_called()


def test_expiry_after_read_only_verification_prevents_submission():
    service = CustodiedService()
    times = iter([199, 200])
    with client(service, clock=lambda: next(times)) as http:
        response = post(http, "review-receipts", review_payload(service))
    assert response.json()["error"]["code"] == "workflow_receipt_expired"
    assert service.calls == ["verify", "submit"]
    assert service.mutations == 0


@pytest.mark.parametrize(
    "change",
    [
        {"action_digest": NEXT_STATE},
        {"expected_state_digest": NEXT_STATE},
        {"run_id": "run_" + "9" * 32},
        {"workflow_id": "workflow:test.example/other"},
        {"expected_state_digest": None},
        {"request_id": None},
    ],
)
def test_changed_action_state_or_missing_mutation_metadata_denied(change):
    service = CustodiedService()
    payload = review_payload(service)
    payload.update(change)
    with client(service) as http:
        response = post(http, "review-receipts", payload)
    assert response.status_code in (409, 422)
    assert service.mutations == 0


@pytest.mark.parametrize("proof", [None, False, "wrong-digest", "wrong-reference"])
def test_trusted_proof_must_match_exact_receipt_and_reference(proof):
    service = CustodiedService()
    if proof is None:
        value = {"verified": True}
    elif proof is False:
        value = WorkflowReceiptVerification(
            reference(), workflow_receipt_digest(service.receipt), False
        )
    elif proof == "wrong-digest":
        value = WorkflowReceiptVerification(reference(), NEXT_STATE, True)
    else:
        value = WorkflowReceiptVerification(
            replace(reference(), request_id="req_" + "8" * 32),
            workflow_receipt_digest(service.receipt),
            True,
        )
    service.verify_receipt = Mock(return_value=value)
    with client(service) as http:
        response = post(http, "review-receipts", review_payload(service))
    assert response.json()["error"]["code"] == "workflow_receipt_unverified"
    assert service.mutations == 0


@pytest.mark.parametrize("operation", ["status", "cancel"])
def test_service_exception_does_not_echo_private_data(operation):
    service = CustodiedService()
    setattr(service, operation, Mock(side_effect=RuntimeError(PRIVATE)))
    with client(service) as http:
        response = post(http, operation)
    assert response.status_code == 503
    assert PRIVATE not in response.text
    assert response.json()["error"]["code"] == (
        "workflow_mutation_unknown"
        if operation == "cancel"
        else "workflow_service_failed"
    )


@pytest.mark.parametrize("kind", ["dict", "tampered", "wrong-run", "no-cancel-ack"])
def test_invalid_service_result_is_refused(kind):
    service = CustodiedService()
    result = service.view
    operation = "status"
    if kind == "dict":
        result = {**result.to_dict(), "clinical_text": PRIVATE}
    elif kind == "tampered":
        object.__setattr__(result, "state_digest", PRIVATE)
    elif kind == "wrong-run":
        result = replace(
            result,
            reference=replace(reference(), run_id=RunId.parse("run_" + "7" * 32)),
            effects=(),
        )
    else:
        operation = "cancel"
    setattr(service, operation, Mock(return_value=result))
    with client(service) as http:
        response = post(http, operation)
    assert response.status_code == (409 if kind == "wrong-run" else 502)
    assert PRIVATE not in response.text


def test_content_unknown_fields_and_correlations_do_not_enter_access_logs(caplog):
    service = CustodiedService()
    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER_NAME):
        with client(service) as http:
            response = http.post(
                "/v1/workflows/preview",
                json={**reference().to_dict(), PRIVATE: PRIVATE},
                headers={"X-API-Key": KEY, "X-Request-ID": PRIVATE},
            )
            assert response.status_code == 422
            assert PRIVATE not in response.text
            assert PRIVATE not in response.headers["x-request-id"]
            assert post(http, "preview").status_code == 200
    records = [r.getMessage() for r in caplog.records if r.name == ACCESS_LOGGER_NAME]
    assert records
    for record in records:
        assert PRIVATE not in record and KEY not in record
        assert ACTION not in record and STATE not in record
        assert "operation_digest" not in record and "reviewer_role" not in record
        assert "effects" not in record


def test_strict_body_limit_queries_duplicates_and_content_type():
    service = CustodiedService()
    with client(service) as http:
        headers = {"X-API-Key": KEY, "Content-Type": "application/json"}
        large = http.post(
            "/v1/workflows/status",
            content=b"x" * (MAX_WORKFLOW_REQUEST_BYTES + 1),
            headers=headers,
        )
        assert large.status_code == 413
        assert (
            http.post(
                "/v1/workflows/status", content='{"a":1,"a":2}', headers=headers
            ).status_code
            == 422
        )
        assert (
            http.post(
                "/v1/workflows/status", content='{"a":NaN}', headers=headers
            ).status_code
            == 422
        )
        assert (
            http.post(
                "/v1/workflows/status?role=admin",
                json=reference().to_dict(),
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            http.post(
                "/v1/workflows/status",
                content=json.dumps(reference().to_dict()),
                headers={"X-API-Key": KEY},
            ).status_code
            == 422
        )
    assert not service.calls


def test_openapi_matches_native_contracts_and_fixed_route_scopes():
    schemas = workflow_http_schemas()
    service = CustodiedService()
    for name, payload in [
        ("reference", reference().to_dict()),
        ("mutation", reference().to_dict()),
        ("review", review_payload(service)),
        ("view", service.view.to_dict()),
    ]:
        Draft202012Validator.check_schema(schemas[name])
        Draft202012Validator(schemas[name]).validate(payload)
    assert list(
        Draft202012Validator(schemas["review"]).iter_errors(
            {**review_payload(service), "approved": True}
        )
    )
    spec = create_app().openapi()
    assert spec["components"]["schemas"]["GovernedWorkflowView"] == schemas["view"]
    for path, scope in WORKFLOW_ROUTE_SCOPES.items():
        endpoint = spec["paths"][path]["post"]
        assert not endpoint.get("parameters")
        assert endpoint["x-required-scope"] == scope
        assert DEFAULT_ROUTE_SCOPES[("POST", path)] == (scope,)
        assert {str(v) for v in WORKFLOW_ERROR_STATUSES.values()} <= endpoint[
            "responses"
        ].keys()
        assert endpoint["security"]


@pytest.mark.parametrize("operation", ["status", "cancel"])
def test_timeout_has_no_retry_and_reports_unknown_mutation(operation):
    service = CustodiedService()
    release = threading.Event()
    started = threading.Event()

    def stalled(principal, ref):
        started.set()
        release.wait(1)
        return service.view

    method = Mock(side_effect=stalled)
    setattr(service, operation, method)
    try:
        with client(service, policy=WorkflowHTTPPolicy(True, True, True, 0.01)) as http:
            response = post(http, operation)
            assert started.is_set()
            assert response.status_code == 503
            assert response.json()["error"]["code"] == (
                "workflow_mutation_unknown"
                if operation == "cancel"
                else "workflow_unavailable"
            )
            method.assert_called_once()
            release.set()
    finally:
        release.set()


def test_receipt_commit_rechecks_custody_after_read_only_proof():
    service = CustodiedService()
    verify = service.verify_receipt

    def revoke_after_proof(principal, ref, receipt, *, now):
        proof = verify(principal, ref, receipt, now=now)
        service.authority_valid = False
        return proof

    service.verify_receipt = revoke_after_proof
    with client(service) as http:
        response = post(http, "review-receipts", review_payload(service))
    assert response.json()["error"]["code"] == "workflow_receipt_unverified"
    assert service.mutations == 0


def test_cancellation_terminal_run_is_refused_without_compensation():
    service = CustodiedService()
    service.view = replace(service.view, phase=ActionPhase.ABORTED)
    with client(service) as http:
        response = post(http, "cancel")
    assert response.json()["error"]["code"] == "workflow_terminal"
    assert service.mutations == 0
    service.compensate.assert_not_called()


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("receipt_expires_at", 100, "workflow_receipt_expired"),
        ("receipt_consumed_at", 101, "workflow_receipt_future"),
    ],
)
def test_receipt_time_is_verified_in_trusted_service_custody(field, value, code):
    service = CustodiedService()
    setattr(service, field, value)
    with client(service) as http:
        response = post(http, "review-receipts", review_payload(service))
    assert response.json()["error"]["code"] == code
    assert service.mutations == 0 and "submit" not in service.calls


@pytest.mark.parametrize("kind", ["mutated", "foreign"])
def test_untrusted_service_diagnostic_cannot_hide_unknown_mutation(kind):
    service = CustodiedService()

    class ForeignError(WorkflowServiceError):
        def __init__(self, _code):
            ValueError.__init__(self, "Foreign workflow diagnostic")

        def __getattribute__(self, name):
            if name == "code":
                raise UnicodeDecodeError("utf-8", PRIVATE.encode(), 0, 1, "invalid")
            return super().__getattribute__(name)

    error = (
        WorkflowServiceError("workflow_service_failed")
        if kind == "mutated"
        else ForeignError("workflow_service_failed")
    )
    if kind == "mutated":
        error.code = PRIVATE

    def uncertain(*args, **kwargs):
        service.mutations += 1
        raise error

    service.cancel = uncertain
    with client(service) as http:
        response = post(http, "cancel")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "workflow_mutation_unknown"
    assert PRIVATE not in response.text
    assert service.mutations == 1
