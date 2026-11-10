"""Offline tests for access-scope minimization evidence."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.compliance import (
    REASON_OVERBROAD_REQUEST,
    REASON_UNAPPROVED_ESCALATION,
    REASON_UNAPPROVED_REQUEST,
    REASON_UNAPPROVED_USE,
    REASON_UNDECLARED_USE,
    REASON_WILDCARD_REQUEST,
    REASON_WILDCARD_USE,
    AccessScope,
    AccessScopePolicy,
    AccessScopeValidationError,
    AccessScopeViolationError,
    enforce_access_scope,
    evaluate_access_scope,
    render_access_scope_evidence,
)


def test_minimal_concrete_scopes_allow_and_serialize_deterministically() -> None:
    first = evaluate_access_scope(
        requested={"records:read"},
        used={"records:read"},
        approved={"records:*"},
    )
    second = evaluate_access_scope(
        requested=["records:read"],
        used=["records:read"],
        approved=["records:*"],
    )

    assert first.allowed is True
    assert first.decision == "allow"
    assert first.reasons == ()
    assert first.to_json() == second.to_json()
    assert first.counts.wildcard_approved == 1
    assert first.to_dict()["summary"] == {
        "approved_count": 1,
        "escalated_used_count": 0,
        "escalation_rule_count": 0,
        "undeclared_used_count": 0,
        "unapproved_escalation_rule_count": 0,
        "unapproved_requested_count": 0,
        "unapproved_used_count": 0,
        "unused_requested_count": 0,
        "used_count": 1,
        "wildcard_approved_count": 1,
        "wildcard_escalation_rule_count": 0,
        "wildcard_requested_count": 0,
        "wildcard_rule_count": 0,
        "wildcard_used_count": 0,
        "requested_count": 1,
    }


def test_undeclared_use_and_unused_request_are_blocked() -> None:
    result = evaluate_access_scope(
        requested={"records:read", "records:write"},
        used={"billing:read", "records:read"},
        approved={"billing:read", "records:read", "records:write"},
    )

    assert result.allowed is False
    assert result.reasons == (REASON_UNDECLARED_USE, REASON_OVERBROAD_REQUEST)
    assert result.undeclared_scopes == (AccessScope("billing", "read"),)
    assert result.overbroad_requested == (AccessScope("records", "write"),)
    assert result.to_dict()["violations"] == [
        {"reason": REASON_UNDECLARED_USE, "count": 1},
        {"reason": REASON_OVERBROAD_REQUEST, "count": 1},
    ]

    with pytest.raises(AccessScopeViolationError) as error:
        enforce_access_scope(
            requested={"records:read", "records:write"},
            used={"billing:read", "records:read"},
            approved={"billing:read", "records:read", "records:write"},
        )
    assert "billing" not in str(error.value)
    assert "records" not in error.value.evaluation.to_json()


def test_requested_and_used_scopes_must_be_approved() -> None:
    result = evaluate_access_scope(
        requested={"records:read"},
        used={"records:read"},
        approved={"records:write"},
    )

    assert result.reasons == (REASON_UNAPPROVED_REQUEST, REASON_UNAPPROVED_USE)
    assert result.counts.unapproved_requested == 1
    assert result.counts.unapproved_used == 1


def test_wildcard_requests_require_an_explicit_rule() -> None:
    blocked = evaluate_access_scope(
        requested={"records:*"},
        used={"records:read"},
        approved={"records:*"},
    )
    allowed = evaluate_access_scope(
        requested={"records:*"},
        used={"records:read"},
        approved={"records:*"},
        wildcard_rules={"records:*"},
    )

    assert blocked.reasons == (REASON_WILDCARD_REQUEST,)
    assert allowed.allowed is True
    assert allowed.counts.wildcard_requested == 1
    assert allowed.to_json().count("records") == 0
    assert render_access_scope_evidence(allowed).count("records") == 0


def test_escalation_must_be_explicit_and_approved() -> None:
    undeclared = evaluate_access_scope(
        requested={"records:read"},
        used={"records:export", "records:read"},
        approved={"records:export", "records:read"},
    )
    explicit = evaluate_access_scope(
        requested={"records:read"},
        used={"records:export", "records:read"},
        approved={"records:export", "records:read"},
        escalation_rules={"records:export"},
    )
    unapproved = evaluate_access_scope(
        requested={"records:read"},
        used={"records:export", "records:read"},
        approved={"records:read"},
        escalation_rules={"records:export"},
    )

    assert undeclared.reasons == (REASON_UNDECLARED_USE,)
    assert explicit.allowed is True
    assert explicit.escalated_scopes == (AccessScope("records", "export"),)
    assert explicit.counts.escalated_used == 1
    assert unapproved.reasons == (
        REASON_UNAPPROVED_USE,
        REASON_UNAPPROVED_ESCALATION,
    )


def test_wildcard_actual_use_is_blocked_even_when_approved() -> None:
    result = evaluate_access_scope(
        requested={"records:read"},
        used={"records:*"},
        approved={"records:*"},
    )

    assert result.reasons == (REASON_OVERBROAD_REQUEST, REASON_WILDCARD_USE)
    assert result.counts.wildcard_used == 1


def test_policy_object_is_equivalent_to_explicit_rules() -> None:
    direct = evaluate_access_scope(
        requested={"records:*"},
        used={"records:read"},
        approved={"records:*"},
        wildcard_rules={"records:*"},
        escalation_rules={"records:export"},
    )
    policy = evaluate_access_scope(
        requested={"records:*"},
        used={"records:read"},
        approved={"records:*"},
        policy=AccessScopePolicy(
            wildcard_rules={"records:*"},
            escalation_rules={"records:export"},
        ),
    )

    assert direct.to_json() == policy.to_json()


def test_mapping_input_is_normalized_and_invalid_values_are_not_echoed() -> None:
    result = evaluate_access_scope(
        requested={"records": {"read", "write"}},
        used={"records:read"},
        approved={"records": {"read", "write"}},
    )
    assert result.counts.requested == 2
    assert result.reasons == (REASON_OVERBROAD_REQUEST,)

    raw_value = "Synthetic private value 17"
    with pytest.raises(AccessScopeValidationError) as error:
        AccessScope.from_string(raw_value)
    assert raw_value not in str(error.value)


def test_evaluation_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("access-scope evaluation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    result = evaluate_access_scope(
        requested={"records:read"},
        used={"records:read"},
        approved={"records:read"},
    )

    assert json.loads(result.to_json()) == result.to_dict()
