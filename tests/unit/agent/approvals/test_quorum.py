"""Synthetic offline quorum policy safety controls."""

from dataclasses import replace

import pytest

from openmed.agent.approvals.quorum import (
    ApprovalQuorumDecision,
    ApprovalQuorumError,
    ApprovalQuorumEvaluator,
    ApprovalQuorumPolicy,
)
from openmed.agent.approvals.tokens import ApprovalReceipt, ApprovalTokenValidationError

CLINICIAN = "role:org.example/clinician@1.0.0"
PHARMACIST = "role:org.example/pharmacist@1.0.0"
REQUESTER = "role:org.example/trainee@1.0.0"
OTHER = "role:org.example/observer@1.0.0"
ACTION = "sha256:" + "a" * 64
CHANGED = "sha256:" + "b" * 64


def receipt(role=CLINICIAN, token=1, **kwargs):
    return ApprovalReceipt(
        action_digest=kwargs.get("action_digest", ACTION),
        reviewer_role=role,
        token_digest=f"sha256:{token:064x}",
        consumed_at=kwargs.get("consumed_at", 10),
        expires_at=kwargs.get("expires_at", 100),
    )


def policy(**kwargs):
    return ApprovalQuorumPolicy(
        action_class="high-impact-write",
        required_count=kwargs.get("required_count", 2),
        allowed_reviewer_roles=kwargs.get(
            "allowed_reviewer_roles", (CLINICIAN, PHARMACIST, REQUESTER)
        ),
        distinct_roles=kwargs.get("distinct_roles", True),
        excluded_requester_roles=kwargs.get("excluded_requester_roles", ()),
    )


def evaluate(receipts, **kwargs):
    return ApprovalQuorumEvaluator([kwargs.pop("policy", policy())]).evaluate(
        action_class="high-impact-write",
        action_digest=ACTION,
        requester_role=REQUESTER,
        receipts=receipts,
        now=kwargs.pop("now", 20),
        **kwargs,
    )


def test_two_distinct_roles_satisfy_and_order_is_deterministic():
    receipts = [receipt(), receipt(PHARMACIST, 2)]
    decision = evaluate(receipts)
    assert decision.satisfied
    assert decision.approved_count == 2
    assert evaluate(reversed(receipts)) == decision
    assert decision.reviewer_roles == tuple(sorted((CLINICIAN, PHARMACIST)))


@pytest.mark.parametrize(
    "receipts,expected",
    [
        ([], 0),
        ([receipt(), receipt(CLINICIAN, 2)], 1),
        ([receipt(), receipt(REQUESTER, 2)], 1),
        ([receipt(), receipt(PHARMACIST, 2, action_digest=CHANGED)], 1),
        ([receipt(), receipt(PHARMACIST, 2, expires_at=20)], 1),
        ([receipt(), receipt(PHARMACIST, 2, consumed_at=21)], 1),
        ([receipt(), receipt(OTHER, 2)], 1),
        ([receipt(), receipt()], 0),
        ([receipt(), receipt(PHARMACIST)], 0),
    ],
)
def test_ineligible_or_replayed_receipts_cannot_satisfy(receipts, expected):
    decision = evaluate(receipts)
    assert not decision.satisfied
    assert decision.approved_count == expected


def test_external_replay_exclusions():
    decision = evaluate(
        [receipt(), receipt(PHARMACIST, 2)],
        replayed_receipt_digests=(receipt().token_digest,),
    )
    assert decision.approved_count == 1


def test_explicit_requester_categories_are_excluded():
    decision = evaluate(
        [receipt(), receipt(PHARMACIST, 2), receipt(REQUESTER, 3)],
        policy=policy(excluded_requester_roles=(CLINICIAN,)),
    )
    assert decision.approved_count == 1


def test_non_distinct_policy_counts_unique_receipts_not_duplicate_tokens():
    configured = policy(distinct_roles=False)
    assert evaluate([receipt(), receipt(CLINICIAN, 2)], policy=configured).satisfied
    assert not evaluate([receipt(), receipt()], policy=configured).satisfied


def test_policy_digest_commits_all_rules_and_normalizes_role_order():
    configured = policy()
    assert (
        replace(
            configured,
            allowed_reviewer_roles=tuple(reversed(configured.allowed_reviewer_roles)),
        ).digest
        == configured.digest
    )
    for changed in (
        replace(configured, required_count=1),
        replace(configured, distinct_roles=False),
        replace(configured, excluded_requester_roles=(REQUESTER,)),
        replace(configured, action_class="other-action"),
    ):
        assert changed.digest != configured.digest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required_count": 0},
        {"required_count": True},
        {"required_count": 4},
        {"distinct_roles": 1},
        {"allowed_reviewer_roles": ()},
        {"allowed_reviewer_roles": (CLINICIAN, CLINICIAN)},
        {"allowed_reviewer_roles": [CLINICIAN]},
    ],
)
def test_invalid_or_impossible_policies_fail_closed(kwargs):
    with pytest.raises(ApprovalQuorumError):
        policy(**kwargs)


def test_missing_or_duplicate_action_class_fails_closed():
    with pytest.raises(ApprovalQuorumError, match="duplicate_action_class"):
        ApprovalQuorumEvaluator([policy(), policy()])
    with pytest.raises(ApprovalQuorumError, match="unknown_action_class"):
        ApprovalQuorumEvaluator([]).policy("private payload")


@pytest.mark.parametrize("field", ["action_digest", "requester_role", "now"])
def test_invalid_inputs_do_not_echo_values(field):
    kwargs = dict(
        action_class="high-impact-write",
        action_digest=ACTION,
        requester_role=REQUESTER,
        now=20,
        receipts=[],
    )
    kwargs[field] = "Synthetic Patient ID 123 /private/credential"
    with pytest.raises(ApprovalTokenValidationError) as caught:
        ApprovalQuorumEvaluator([policy()]).evaluate(**kwargs)
    assert "Patient" not in str(caught.value)
    assert "credential" not in str(caught.value)


def test_decision_contains_only_roles_counts_and_digests():
    decision = evaluate([receipt(), receipt(PHARMACIST, 2)])
    assert set(decision.to_dict()) == {
        "action_digest",
        "policy_digest",
        "requester_role",
        "reviewer_roles",
        "receipt_digests",
        "required_count",
        "approved_count",
    }
    assert "expires_at" not in repr(decision)
    with pytest.raises(ApprovalTokenValidationError):
        replace(decision, action_digest="Synthetic Patient ID 123")
    with pytest.raises(ApprovalQuorumError, match="invalid_counts"):
        replace(decision, approved_count=100)
    assert isinstance(decision, ApprovalQuorumDecision)


def test_raw_payloads_and_oversized_sets_are_refused():
    with pytest.raises(ApprovalQuorumError, match="invalid_receipt"):
        evaluate([{"payload": "Synthetic Patient ID 123"}])
    with pytest.raises(ApprovalQuorumError, match="too_many_receipts"):
        evaluate(receipt(token=i) for i in range(10_001))
