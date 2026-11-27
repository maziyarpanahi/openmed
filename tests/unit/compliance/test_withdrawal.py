import json
from dataclasses import dataclass, field

import pytest

from openmed.compliance.withdrawal import (
    ParticipantCode,
    ParticipantPlan,
    WithdrawalCoordinator,
    WithdrawalPlan,
)
from openmed.core.audit import stable_hash

SUBJECT = stable_hash("synthetic-subject")


@dataclass
class Store:
    records: list = field(default_factory=lambda: ["synthetic-record"])
    verifiable: bool = True

    def plan(self, subject):
        return ParticipantPlan(len(self.records), stable_hash([subject, self.records]))

    def apply(self, subject, selection):
        assert self.plan(subject) == selection
        count = len(self.records)
        self.records.clear()
        return count

    def verify(self, subject):
        return self.verifiable and not self.records


def test_dry_run_then_apply_two_stores():
    first, second = Store(), Store()
    coordinator = WithdrawalCoordinator(
        {ParticipantCode.RESULT_CACHE: first, ParticipantCode.SERVICE_JOBS: second}
    )
    plan = coordinator.withdraw(SUBJECT)
    assert isinstance(plan, WithdrawalPlan)
    assert first.records and second.records
    receipt = coordinator.withdraw(SUBJECT, confirmation=plan.confirmation_token)
    assert receipt.state == "success"
    assert [p.applied for p in receipt.participants] == [1, 1]
    assert not first.records and not second.records
    encoded = json.dumps(receipt.to_dict())
    assert SUBJECT in encoded and "synthetic" not in encoded


def test_changed_membership_even_at_same_count_rejects_before_mutation():
    first, second = Store(), Store()
    coordinator = WithdrawalCoordinator(
        {ParticipantCode.SERVICE_JOBS: first, ParticipantCode.SURROGATE_VAULT: second}
    )
    token = coordinator.plan(SUBJECT).confirmation_token
    second.records[:] = ["other-record"]
    with pytest.raises(ValueError, match="confirmation"):
        coordinator.withdraw(SUBJECT, confirmation=token)
    assert first.records and second.records


def test_unverified_store_yields_partial_and_other_store_is_applied():
    stores = {
        ParticipantCode.RESULT_CACHE: Store(verifiable=False),
        ParticipantCode.JOURNEY_STORE: Store(),
    }
    coordinator = WithdrawalCoordinator(stores)
    receipt = coordinator.withdraw(
        SUBJECT, confirmation=coordinator.plan(SUBJECT).confirmation_token
    )
    assert receipt.state == "partial"
    assert {p.code: p.verified for p in receipt.participants} == {
        ParticipantCode.RESULT_CACHE: False,
        ParticipantCode.JOURNEY_STORE: True,
    }


def test_raw_subject_and_training_store_rejected():
    with pytest.raises(ValueError, match="supported"):
        WithdrawalCoordinator({"training": Store()})
    coordinator = WithdrawalCoordinator({ParticipantCode.RESULT_CACHE: Store()})
    with pytest.raises(ValueError, match="^invalid subject reference digest$"):
        coordinator.withdraw("synthetic-subject")
