"""Synthetic MCP dispatch, consent custody and action-binding controls."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("mcp")

from openmed.agent.action_phases import ActionPhase
from openmed.agent.artifact_reference import ArtifactKind, ArtifactReference
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.reviewer_handoff import RequestedDecision, ReviewerHandoffPacket
from openmed.mcp.authorization_conformance import (
    ConformanceViolation,
    validate_governed_tool_registration,
)
from openmed.mcp.consent_receipts import (
    ConsentReceiptIssuer,
    ConsentReceiptPolicy,
    ConsentReceiptVerifier,
    MappingConsentKeyProvider,
)
from openmed.mcp.governed_workflows import (
    GOVERNED_MCP_OPERATIONS,
    GovernanceDecision,
    GovernanceStatus,
    GovernedMCPError,
    GovernedMCPRequest,
    GovernedMCPReview,
    GovernedMCPView,
    ReceiptObservation,
    default_governed_consent_policy,
)
from openmed.mcp.server import create_mcp_server
from openmed.mcp.tool_registry import TOOL_REGISTRY

ACTION = "sha256:" + "a" * 64
STATE = "sha256:" + "b" * 64
NEXT_STATE = "sha256:" + "c" * 64
OTHER = "sha256:" + "d" * 64
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SENTINEL = "PRIVATE PATIENT MRN-9032 bearer-secret-sentinel"


def _request(**changes):
    return replace(
        GovernedMCPRequest(
            RunId("run_" + "a" * 32),
            WorkflowId("workflow:org.example/synthetic@1.0.0"),
            ACTION,
            STATE,
        ),
        **changes,
    )


def _view(**changes):
    return replace(
        GovernedMCPView(
            _request().run_id,
            _request().workflow_id,
            ACTION,
            STATE,
            ActionPhase.READY,
            GovernanceStatus.READY,
            GovernanceDecision.ALLOWED,
            GovernanceDecision.ALLOWED,
            GovernanceDecision.ALLOWED,
            1,
        ),
        **changes,
    )


def _packet(**changes):
    values = {
        "run_id": _request().run_id,
        "workflow_id": _request().workflow_id,
        "reason_code": "human_gate",
        "requested_decision": RequestedDecision.DECIDE_NEXT_STEP,
        "evidence_references": (
            ArtifactReference(
                "art_" + "a" * 32,
                ArtifactKind.EVIDENCE,
                "openmed.agent.action.v1",
                ACTION.removeprefix("sha256:"),
                128,
            ),
        ),
        "issued_at": NOW,
        "expires_at": NOW + timedelta(minutes=10),
        "validation_time": NOW,
    }
    values.update(changes)
    return ReviewerHandoffPacket(**values)


class _Service:
    def __init__(self):
        self.view = _view()
        self.calls = []
        self.handoffs = 0
        self.clinical_effects = 0
        self.review_packet = _packet()
        self.race = False

    def preflight(self, request):
        self.calls.append("preflight")
        return self.view

    def preview(self, request):
        self.calls.append("preview")
        return self.view

    def status(self, request):
        self.calls.append("status")
        return self.view

    def request_review(self, request):
        self.calls.append("request_review")
        if self.race:
            self.view = replace(self.view, state_digest=NEXT_STATE)
        if request.expected_state_digest != self.view.state_digest:
            raise GovernedMCPError("governance_conflict")
        self.handoffs += 1
        self.view = replace(
            self.view,
            phase=ActionPhase.WAITING_REVIEW,
            status=GovernanceStatus.REVIEW_REQUIRED,
            state_digest=NEXT_STATE,
        )
        return GovernedMCPReview(self.view, self.review_packet)


def _consent(*, changes=None, required=True):
    keys = MappingConsentKeyProvider({"synthetic": "synthetic-review-key"})
    policy = ConsentReceiptPolicy(
        ConsentReceiptVerifier(keys, clock=lambda: NOW.timestamp()),
        client="synthetic-client",
        resource="synthetic-resource",
        scope="mcp.governance.review",
        require_receipt=required,
    )
    issuer = ConsentReceiptIssuer(
        keys,
        key_id="synthetic",
        clock=lambda: NOW.timestamp(),
        receipt_id_factory=lambda: "synthetic-review-receipt",
    )
    receipt = issuer.issue(
        "synthetic-client",
        "openmed_workflow_request_review",
        "synthetic-resource",
        "mcp.governance.review",
        {"request": _request().to_dict()},
        ttl_seconds=60,
    )
    if changes:
        receipt = replace(receipt, **changes)
    return policy, lambda tool, arguments: receipt


def _server(service=None, **kwargs):
    return create_mcp_server(
        governance_service=service, governance_clock=lambda: NOW, **kwargs
    )


def _call(server, tool, *, request_data=None, extras=None):
    result = asyncio.run(
        server.call_tool(
            tool,
            {
                "request": _request().to_dict()
                if request_data is None
                else request_data,
                **(extras or {}),
            },
        )
    )
    assert json.loads(result.content[0].text) == result.structuredContent
    assert SENTINEL not in json.dumps(result.model_dump(mode="json"))
    return result


def test_registration_advertises_exact_contracts_without_authority_fields():
    tools = {t.name: t for t in asyncio.run(_server().list_tools())}
    for name, operation in GOVERNED_MCP_OPERATIONS.items():
        tool = tools[name]
        spec = TOOL_REGISTRY.get(name)
        assert tool.inputSchema == spec.input_schema
        assert tool.outputSchema == spec.mcp_output_schema()
        assert tool.annotations.model_dump(exclude_none=True) == spec.annotations()
        assert tool.annotations.readOnlyHint is (operation != "request_review")
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.openWorldHint is False
        assert set(tool.inputSchema["properties"]) == {"request"}
        assert "approval" not in json.dumps(tool.inputSchema)
        assert "reviewer" not in json.dumps(tool.inputSchema)


@pytest.mark.parametrize("operation", ["preflight", "preview", "status"])
def test_read_commands_only_observe_exact_action(operation):
    service = _Service()
    result = _call(_server(service), "openmed_workflow_" + operation)
    assert (
        result.isError is False and result.structuredContent == service.view.to_dict()
    )
    assert (
        service.calls == [operation]
        and service.handoffs == service.clinical_effects == 0
    )


@pytest.mark.parametrize("name", GOVERNED_MCP_OPERATIONS)
def test_default_has_no_live_governance_adapter(name):
    result = _call(_server(), name)
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_unavailable"
    )


def test_review_needs_human_channel_consent_and_creates_no_approval():
    service = _Service()
    policy, provider = _consent()
    server = _server(
        service,
        governance_consent_policy=policy,
        governance_consent_receipt_provider=provider,
    )
    result = _call(server, "openmed_workflow_request_review")
    assert result.isError is False
    payload = result.structuredContent
    assert payload["status"] == "review_required"
    assert payload["review_request_digest"].startswith("sha256:")
    assert payload["receipt_verification"] == "absent"
    assert service.handoffs == 1 and service.clinical_effects == 0
    assert service.calls == ["status", "request_review"]
    assert "synthetic-review-key" not in json.dumps(payload)
    assert "synthetic-review-receipt" not in json.dumps(payload)
    assert "reason_code" not in payload and "packet" not in payload
    replay = _call(server, "openmed_workflow_request_review")
    assert (
        replay.isError
        and replay.structuredContent["error"]["code"] == "governance_conflict"
    )
    assert service.handoffs == 1


def test_review_without_any_consent_never_dispatches_handoff():
    service = _Service()
    result = _call(_server(service), "openmed_workflow_request_review")
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "consent_required"
    )
    assert service.calls == ["status"] and service.handoffs == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"tool": "openmed_workflow_preview"},
        {"client": "different-client"},
        {"resource": "different-resource"},
        {"argument_digest": OTHER},
        {"signature": "hmac-sha256:" + "0" * 64},
        {
            "issued_at": int(NOW.timestamp()) - 120,
            "expires_at": int(NOW.timestamp()) - 60,
        },
    ],
)
def test_edited_expired_or_misbound_consent_cannot_create_review(changes):
    service = _Service()
    policy, provider = _consent(changes=changes)
    result = _call(
        _server(
            service,
            governance_consent_policy=policy,
            governance_consent_receipt_provider=provider,
        ),
        "openmed_workflow_request_review",
    )
    assert result.isError and service.handoffs == 0
    assert service.calls == ["status"]


@pytest.mark.parametrize(
    "field",
    [
        "approval_token",
        "receipt",
        "consent_receipt",
        "reviewer_role",
        "reviewer_identity",
        "credential",
        "authorization",
        "access_token",
    ],
)
@pytest.mark.parametrize("nested", [True, False])
def test_mcp_rejects_inline_authority_without_echo_or_logs(field, nested, caplog):
    service = _Service()
    request_data = _request().to_dict()
    extras = {}
    (request_data if nested else extras)[field] = SENTINEL
    with caplog.at_level(logging.DEBUG):
        result = _call(
            _server(service),
            "openmed_workflow_request_review",
            request_data=request_data,
            extras=extras,
        )
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "invalid_arguments"
    )
    assert service.calls == [] and service.handoffs == 0
    assert SENTINEL not in caplog.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unknown"),
        ("run_id", SENTINEL),
        ("workflow_id", SENTINEL),
        ("action_digest", SENTINEL),
        ("expected_state_digest", False),
    ],
)
def test_invalid_metadata_is_refused_before_service(field, value):
    service = _Service()
    request_data = {**_request().to_dict(), field: value}
    result = _call(
        _server(service), "openmed_workflow_preflight", request_data=request_data
    )
    assert result.isError and service.calls == []


@pytest.mark.parametrize("request_data", [None, [], "private", {}, True])
def test_direct_request_parser_refuses_other_shapes(request_data):
    with pytest.raises(GovernedMCPError, match="Governed workflow request refused"):
        GovernedMCPRequest.from_dict(request_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", RunId("run_" + "f" * 32)),
        ("workflow_id", WorkflowId("workflow:org.example/other@1.0.0")),
        ("action_digest", OTHER),
    ],
)
def test_cross_run_action_and_workflow_results_fail_closed(field, value):
    service = _Service()
    service.view = replace(service.view, **{field: value})
    result = _call(_server(service), "openmed_workflow_status")
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_conflict"
    )


@pytest.mark.parametrize(
    "value", [False, {"status": "approved", "token": SENTINEL}, None]
)
def test_untyped_service_results_never_cross_mcp(value):
    service = _Service()
    service.preview = lambda request: value
    result = _call(_server(service), "openmed_workflow_preview")
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_invalid_result"
    )


def test_service_prints_and_failures_do_not_corrupt_stdio_or_echo_values(capsys):
    service = _Service()

    def broken(request):
        print(SENTINEL)
        raise RuntimeError(SENTINEL)

    service.preview = broken
    result = _call(_server(service), "openmed_workflow_preview")
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_failed"
    )
    assert SENTINEL not in capsys.readouterr().out


@pytest.mark.parametrize("state", [OTHER, None])
def test_review_requires_current_state_before_consent_or_mutation(state):
    service = _Service()
    result = _call(
        _server(service),
        "openmed_workflow_request_review",
        request_data=_request(expected_state_digest=state).to_dict(),
    )
    assert result.isError and service.handoffs == 0


@pytest.mark.parametrize(
    "policy", [None, replace(default_governed_consent_policy(), require_receipt=False)]
)
def test_conformance_rejects_missing_or_disabled_receipt_policy(policy):
    with pytest.raises(ConformanceViolation) as error:
        validate_governed_tool_registration(TOOL_REGISTRY.latest_specs(), policy)
    assert error.value.category == "unapproved_state_change"


def test_registration_refuses_permissive_governance_policy():
    policy, _ = _consent(required=False)
    with pytest.raises(ConformanceViolation):
        _server(_Service(), governance_consent_policy=policy)


@pytest.mark.parametrize("name", GOVERNED_MCP_OPERATIONS)
def test_conformance_detects_annotation_drift(name):
    specs = [
        replace(s, read_only_hint=not s.read_only_hint) if s.name == name else s
        for s in TOOL_REGISTRY.latest_specs()
    ]
    with pytest.raises(ConformanceViolation):
        validate_governed_tool_registration(specs, default_governed_consent_policy())


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposed_effect_count", True),
        ("proposed_effect_count", 1025),
        ("committed_effect_count", 2),
        ("receipt_verification", "verified"),
        ("grant", "allowed"),
        ("review_request_digest", SENTINEL),
        ("status", GovernanceStatus.COMPLETED),
        ("grant", GovernanceDecision.DENIED),
    ],
)
def test_view_invariants_refuse_false_authority_or_invalid_counts(field, value):
    with pytest.raises(GovernedMCPError):
        _view(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", RunId("run_" + "f" * 32)),
        ("evidence_references", ()),
        ("workflow_id", WorkflowId("workflow:org.example/other@1.0.0")),
    ],
)
def test_handoff_acknowledgement_must_reference_exact_action(field, value):
    service = _Service()
    service.review_packet = _packet(**{field: value})
    policy, provider = _consent()
    result = _call(
        _server(
            service,
            governance_consent_policy=policy,
            governance_consent_receipt_provider=provider,
        ),
        "openmed_workflow_request_review",
    )
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_invalid_result"
    )
    assert service.handoffs == 1 and service.clinical_effects == 0


def test_service_atomic_state_race_refuses_handoff():
    service = _Service()
    service.race = True
    policy, provider = _consent()
    result = _call(
        _server(
            service,
            governance_consent_policy=policy,
            governance_consent_receipt_provider=provider,
        ),
        "openmed_workflow_request_review",
    )
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_conflict"
    )
    assert service.handoffs == service.clinical_effects == 0


@pytest.mark.parametrize(
    "status,phase",
    [
        (GovernanceStatus.COMPLETED, ActionPhase.COMPLETED),
        (GovernanceStatus.CANCELLED, ActionPhase.ABORTED),
        (GovernanceStatus.DENIED, ActionPhase.PREFLIGHT),
        (GovernanceStatus.UNAVAILABLE, ActionPhase.PREFLIGHT),
    ],
)
def test_terminal_and_refused_states_never_reach_consent_provider(status, phase):
    service = _Service()
    service.view = replace(service.view, status=status, phase=phase)

    def forbidden(*args):
        raise AssertionError("Refused workflow requested consent")

    result = _call(
        _server(service, governance_consent_receipt_provider=forbidden),
        "openmed_workflow_request_review",
    )
    assert result.isError and service.calls == ["status"] and service.handoffs == 0


def test_receipt_verification_observation_never_resumes_an_action():
    service = _Service()
    service.view = replace(
        service.view, receipt_verification=ReceiptObservation.VERIFIED
    )
    result = _call(_server(service), "openmed_workflow_status")
    assert (
        not result.isError
        and result.structuredContent["receipt_verification"] == "verified"
    )
    assert service.calls == ["status"] and service.clinical_effects == 0


@pytest.mark.parametrize("operation", ["preflight", "preview", "status"])
def test_missing_and_unimplemented_capabilities_are_unavailable(operation):
    service = _Service()

    def missing(request):
        raise NotImplementedError(SENTINEL)

    setattr(service, operation, missing)
    result = _call(_server(service), "openmed_workflow_" + operation)
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_unavailable"
    )


def test_real_signed_expired_consent_never_creates_handoff():
    service = _Service()
    policy, _ = _consent()
    keys = MappingConsentKeyProvider({"synthetic": "synthetic-review-key"})
    issuer = ConsentReceiptIssuer(
        keys, key_id="synthetic", clock=lambda: NOW.timestamp() - 120
    )
    receipt = issuer.issue(
        "synthetic-client",
        "openmed_workflow_request_review",
        "synthetic-resource",
        "mcp.governance.review",
        {"request": _request().to_dict()},
        ttl_seconds=60,
    )
    result = _call(
        _server(
            service,
            governance_consent_policy=policy,
            governance_consent_receipt_provider=lambda *args: receipt,
        ),
        "openmed_workflow_request_review",
    )
    assert result.isError and service.calls == ["status"] and service.handoffs == 0


def test_provider_cannot_rebind_consent_by_mutating_argument_copy():
    service = _Service()
    policy, _ = _consent()
    keys = MappingConsentKeyProvider({"synthetic": "synthetic-review-key"})
    issuer = ConsentReceiptIssuer(
        keys, key_id="synthetic", clock=lambda: NOW.timestamp()
    )

    def changed(tool, arguments):
        arguments["request"]["action_digest"] = OTHER
        return issuer.issue(
            "synthetic-client",
            tool,
            "synthetic-resource",
            "mcp.governance.review",
            arguments,
            ttl_seconds=60,
        )

    result = _call(
        _server(
            service,
            governance_consent_policy=policy,
            governance_consent_receipt_provider=changed,
        ),
        "openmed_workflow_request_review",
    )
    assert result.isError and service.calls == ["status"] and service.handoffs == 0


@pytest.mark.parametrize(
    "now", [NOW + timedelta(minutes=11), NOW - timedelta(minutes=1)]
)
def test_expired_and_future_handoff_acknowledgements_require_reconciliation(now):
    service = _Service()
    policy, provider = _consent()
    # Change only the trusted handoff-validation clock; consent has its own
    # service-owned clock. An acknowledgement can fail after handoff creation.
    server = create_mcp_server(
        governance_service=service,
        governance_consent_policy=policy,
        governance_consent_receipt_provider=provider,
        governance_clock=lambda: now,
    )
    result = _call(server, "openmed_workflow_request_review")
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "governance_invalid_result"
    )
    assert service.handoffs == 1 and service.clinical_effects == 0


@pytest.mark.parametrize(
    "clock", [lambda: None, lambda: datetime(2026, 1, 1), lambda: False]
)
def test_invalid_trusted_clock_fails_before_any_handoff(clock):
    service = _Service()
    result = _call(
        create_mcp_server(governance_service=service, governance_clock=clock),
        "openmed_workflow_request_review",
    )
    assert result.isError and service.calls == [] and service.handoffs == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"committed_effect_count": 1},
        {"proposed_effect_count": 2},
        {"receipt_verification": ReceiptObservation.VERIFIED},
        {"review_request_digest": OTHER},
    ],
)
def test_review_acknowledgement_cannot_claim_execution_or_other_handoff(changes):
    service = _Service()
    original = service.request_review

    def altered(request):
        result = original(request)
        return GovernedMCPReview(replace(result.view, **changes), result.packet)

    service.request_review = altered
    policy, provider = _consent()
    result = _call(
        _server(
            service,
            governance_consent_policy=policy,
            governance_consent_receipt_provider=provider,
        ),
        "openmed_workflow_request_review",
    )
    assert result.isError and service.handoffs == 1 and service.clinical_effects == 0


def test_legacy_optional_consent_cannot_weaken_governance_review_boundary():
    service = _Service()
    policy, _ = _consent(required=False)
    result = _call(
        _server(service, consent_policy=policy), "openmed_workflow_request_review"
    )
    assert (
        result.isError
        and result.structuredContent["error"]["code"] == "consent_required"
    )
    assert service.handoffs == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "9.9.9"},
        {"input_schema": {"type": "object", "additionalProperties": True}},
        {"output_schema": {"type": "object", "additionalProperties": True}},
        {"parameters": ()},
        {"open_world_hint": True},
        {"idempotent_hint": False},
    ],
)
def test_actual_registration_rejects_substituted_governance_catalog(
    monkeypatch, changes
):
    original = TOOL_REGISTRY.latest_specs()
    changed = [
        replace(s, **changes) if s.name == "openmed_workflow_preflight" else s
        for s in original
    ]
    monkeypatch.setattr(TOOL_REGISTRY, "latest_specs", lambda: changed)
    service = _Service()
    with pytest.raises(ConformanceViolation) as error:
        _server(service)
    assert error.value.category == "unapproved_state_change"
    assert service.calls == [] and service.handoffs == 0
