"""Offline tests of subject-scoped job erasure and durable worker fencing."""

import json

import pytest
from pydantic import ValidationError
from test_contract_schemas import JOB_RECORD_SCHEMA, _validator
from test_job_payload_lifecycle import (
    TEXT_SENTINEL,
    Clock,
    _encrypted_store,
    _queue,
    _request,
)

from openmed.compliance.withdrawal import ParticipantCode, WithdrawalCoordinator
from openmed.service.jobs import (
    LocalJobStore,
    ServiceJobWithdrawalParticipant,
    job_response_payload,
)

SUBJECT = "hmac-sha256:" + "a" * 64


def test_confirmed_withdrawal_fences_queued_and_completed_jobs(monkeypatch, tmp_path):
    clock = Clock()
    store = _encrypted_store(tmp_path, clock)
    queue = _queue(monkeypatch, tmp_path, clock=clock, payload_store=store)
    try:
        request = _request(subject_ref_digest=SUBJECT, webhook=None)
        completed = queue.submit(request)
        queue._run_job(queue._executor.work[-1])
        assert queue.get(completed["id"])["status"] == "done"
        queued = queue.submit(request, idempotency_key="withdrawal-test")
        stale_work = queue._executor.work[-1]
        untagged = queue.submit(_request(webhook=None))
        other = queue.submit(
            _request(subject_ref_digest="sha256:" + "b" * 64, webhook=None)
        )
        participant = ServiceJobWithdrawalParticipant(queue)
        with pytest.raises(ValueError):
            participant.plan(None)
        coordinator = WithdrawalCoordinator({ParticipantCode.SERVICE_JOBS: participant})
        plan = coordinator.plan(SUBJECT)
        assert plan.participants[0][1].count == 2
        receipt = coordinator.withdraw(SUBJECT, confirmation=plan.confirmation_token)
        assert receipt.state == "success"
        assert TEXT_SENTINEL not in json.dumps(receipt.to_dict())
        queue._run_job(stale_work)
        queue.reconcile()
        for job in (queued, completed):
            record = LocalJobStore(tmp_path / "jobs.json", clock=clock).get(job["id"])
            assert record["status"] == "withdrawn"
            public = job_response_payload(record)
            _validator(JOB_RECORD_SCHEMA).validate(public)
            assert public["documents"] == public["spans"] == []
            assert public["webhook_delivery"] is None
            assert SUBJECT not in json.dumps(public)
            assert not store.has_job_payloads(job["id"])
        assert (
            queue.submit(request, idempotency_key="withdrawal-test")["id"]
            == queued["id"]
        )
        for job in (other, untagged):
            assert queue.get(job["id"])["status"] == "queued"
            assert store.has_job_payloads(job["id"])
    finally:
        queue.shutdown()


def test_pinned_payload_keeps_withdrawal_partial(monkeypatch, tmp_path):
    queue = _queue(monkeypatch, tmp_path, clock=Clock())
    try:
        with pytest.raises(ValidationError):
            _request(subject_ref_digest=TEXT_SENTINEL, webhook=None)
        job = queue.submit(_request(subject_ref_digest=SUBJECT, webhook=None))
        coordinator = WithdrawalCoordinator(
            {ParticipantCode.SERVICE_JOBS: ServiceJobWithdrawalParticipant(queue)}
        )
        with queue.payloads.pinned(job["payloads"]["input"]["ref"]):
            plan = coordinator.plan(SUBJECT)
            assert (
                coordinator.withdraw(
                    SUBJECT, confirmation=plan.confirmation_token
                ).state
                == "partial"
            )
            assert queue.get(job["id"])["status"] == "withdrawn"
        plan = coordinator.plan(SUBJECT)
        assert (
            coordinator.withdraw(SUBJECT, confirmation=plan.confirmation_token).state
            == "success"
        )
    finally:
        queue.shutdown()
