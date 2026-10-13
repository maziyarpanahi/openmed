"""Synthetic governance declarations and independent policy failure controls."""

from dataclasses import replace
from decimal import Decimal

import pytest

from openmed.compliance.governance.federation_charter import (
    CharterHistory,
    FederationCharter,
    FederationRoundDeclaration,
    validate_federation_round,
)


def charter(**changes):
    defaults = dict(
        version=1,
        effective_round=0,
        permitted_tasks=("adapter_training",),
        permitted_roles=("participant", "reviewer", "coordinator"),
        decision_quorum=3,
        minimum_reviewers=2,
        epsilon_ceiling=Decimal("2"),
        delta_ceiling=Decimal("0.00001"),
        incident_response="hold_pending_review",
        appeals="independent_review",
        termination="erase_round_artifacts",
        maximum_round=20,
    )
    return FederationCharter(**(defaults | changes))


def declaration(policy, **changes):
    defaults = dict(
        sequence=0,
        charter_version=policy.version,
        charter_digest=policy.digest,
        task="adapter_training",
        roles=("participant", "reviewer"),
        decision_votes=3,
        reviewer_votes=2,
        epsilon=Decimal("1"),
        delta=Decimal("0.000001"),
    )
    return FederationRoundDeclaration(**(defaults | changes))


def test_valid_and_exact_boundary():
    c = charter()
    report = validate_federation_round(
        CharterHistory((c,)),
        declaration(c, epsilon=c.epsilon_ceiling, delta=c.delta_ceiling),
    )
    assert report["outcome"] == "accept"
    assert report["reasons"] == []


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"task": "unapproved_training"}, "task_not_permitted"),
        (
            {"roles": ("participant", "reviewer", "incident_responder")},
            "role_not_permitted",
        ),
        ({"roles": ("participant",)}, "required_role_missing"),
        ({"decision_votes": 2}, "decision_quorum_missing"),
        ({"reviewer_votes": 1}, "decision_quorum_missing"),
        ({"epsilon": Decimal("2.1")}, "privacy_ceiling_exceeded"),
        ({"delta": Decimal("0.00002")}, "privacy_ceiling_exceeded"),
        ({"incident_open": True}, "incident_review_required"),
        ({"appeal_open": True}, "appeal_review_required"),
        ({"terminated": True}, "federation_terminated"),
        ({"sequence": 21}, "federation_terminated"),
        ({"charter_digest": "sha256:" + "0" * 64}, "inactive_charter"),
        ({"charter_version": 2}, "inactive_charter"),
    ],
)
def test_independent_round_gate(changes, reason):
    c = charter()
    report = validate_federation_round(CharterHistory((c,)), declaration(c, **changes))
    assert report["outcome"] == "hold"
    assert reason in report["reasons"]


def test_activation_and_amendment_custody():
    old = charter()
    new = replace(old, version=2, effective_round=4, epsilon_ceiling=Decimal("0.5"))
    initial = CharterHistory((old,))
    amended = initial.append(new)
    assert initial.amendments == (old,)
    assert amended.active(3) == old
    assert amended.active(4) == new
    assert validate_federation_round(amended, declaration(old, sequence=4))[
        "reasons"
    ] == ["inactive_charter", "privacy_ceiling_exceeded"]
    assert (
        validate_federation_round(
            amended, declaration(new, sequence=4, epsilon=Decimal("0.4"))
        )["outcome"]
        == "accept"
    )
    exported = amended.to_dict()
    assert exported["amendments"][1]["previous_digest"] == old.digest
    exported["amendments"][0]["charter"]["permitted_tasks"].append("unapproved")
    assert amended.to_dict()["amendments"][0]["charter"]["permitted_tasks"] == [
        "adapter_training"
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 0},
        {"permitted_tasks": ("SENSITIVE SENTINEL",)},
        {"permitted_tasks": ("x", "x")},
        {"permitted_roles": ("patient",)},
        {"minimum_reviewers": 4},
        {"epsilon_ceiling": Decimal("NaN")},
        {"delta_ceiling": Decimal("1")},
        {"epsilon_ceiling": 1.0},
        {"delta_ceiling": Decimal("1e-1000")},
        {"incident_response": "ignore"},
        {"appeals": "no_review"},
        {"termination": "keep_forever"},
        {"maximum_round": -1},
    ],
)
def test_bad_contract_controlled(changes):
    with pytest.raises(ValueError, match="^federation_charter_invalid_contract$"):
        charter(**changes)


@pytest.mark.parametrize(
    "second", [dict(version=3, effective_round=1), dict(version=2, effective_round=0)]
)
def test_nonsequential_amendments_rejected(second):
    with pytest.raises(ValueError):
        CharterHistory((charter(), charter(**second)))


def test_frozen_inputs_and_canonical_order():
    tasks = ["adapter_training", "evaluation"]
    c = charter(permitted_tasks=tasks)
    digest = c.digest
    tasks.append("new_task")
    assert c.digest == digest
    assert charter(permitted_tasks=("evaluation", "adapter_training")).digest == digest
    assert replace(c, decision_quorum=4).digest != digest


def test_all_failures_retained_and_values_omitted():
    c = charter()
    report = validate_federation_round(
        CharterHistory((c,)),
        declaration(
            c, task="synthetic_secret_code", incident_open=True, terminated=True
        ),
    )
    assert report["reasons"] == [
        "federation_terminated",
        "incident_review_required",
        "task_not_permitted",
    ]
    assert "synthetic_secret_code" not in str(report)


def test_vote_consistency_and_strict_boolean():
    c = charter()
    for changes in (
        {"reviewer_votes": 4},
        {"incident_open": 1},
        {"decision_votes": True},
    ):
        with pytest.raises(ValueError):
            declaration(c, **changes)


def test_history_json_roundtrip():
    original = CharterHistory((charter(), charter(version=2, effective_round=4)))
    restored = CharterHistory.from_dict(original.to_dict())
    assert restored == original
    assert restored.active(4).digest == original.active(4).digest


@pytest.mark.parametrize("field", ["digest", "previous_digest", "charter"])
def test_amendment_corruption_rejected(field):
    payload = CharterHistory(
        (charter(), charter(version=2, effective_round=4))
    ).to_dict()
    if field == "charter":
        payload["amendments"][1][field]["decision_quorum"] = 4
    else:
        payload["amendments"][1][field] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="^federation_charter_invalid_contract$"):
        CharterHistory.from_dict(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"unknown_sensitive_field": "SENTINEL"},
        {"epsilon_ceiling": "NaN"},
        {"delta_ceiling": "1e-99999"},
    ],
)
def test_closed_machine_readable_policy(changes):
    with pytest.raises(ValueError, match="^federation_charter_invalid_contract$"):
        FederationCharter.from_dict(charter().to_dict() | changes)


def test_semantically_equal_budgets_have_one_commitment():
    a = charter(epsilon_ceiling=Decimal("1"), delta_ceiling=Decimal("0.01"))
    b = charter(epsilon_ceiling=Decimal("1.00"), delta_ceiling=Decimal("0.0100"))
    assert a == b
    assert a.to_dict() == b.to_dict()
    assert a.digest == b.digest
    assert charter(epsilon_ceiling=Decimal("10.0")).to_dict()["epsilon_ceiling"] == "10"


@pytest.mark.parametrize("changes", [{"appeals": []}, {"termination": []}])
def test_unhashable_policy_inputs_controlled(changes):
    with pytest.raises(ValueError, match="^federation_charter_invalid_contract$"):
        charter(**changes)
