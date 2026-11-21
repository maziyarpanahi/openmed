import json

import pytest

from openmed.compliance.corrective_action import (
    CorrectiveAction,
    CorrectiveActionLedger,
)
from openmed.core.audit import stable_hash
from openmed.core.audit_chain import AuditChain


def action(**overrides):
    values = dict(
        record_id="CAPA-2026-0001",
        source="release_gate",
        severity="high",
        capabilities=("deidentify",),
        versions=("4.3.0",),
        local_note="private synthetic finding",
    )
    return CorrectiveAction(**(values | overrides))


def test_full_lifecycle_and_tamper_detection():
    ledger = CorrectiveActionLedger()
    record = ledger.open(action())
    key = record.record_id
    ledger.transition(
        key, "investigating", before_evidence_digest=stable_hash("before")
    )
    ledger.transition(
        key,
        "action_planned",
        root_cause="implementation",
        corrections=["patch"],
        preventive_actions=["test_coverage"],
    )
    ledger.transition(
        key,
        "implemented",
        release_tags=["v4.3.1"],
        artifact_digests=[stable_hash("fix")],
    )
    with pytest.raises(ValueError):
        ledger.transition(key, "verified")
    assert ledger.records()[0].state == "implemented"
    ledger.transition(key, "verified", after_evidence_digest=stable_hash("after"))
    ledger.transition(key, "closed")
    payload = ledger.to_dict()
    assert AuditChain.from_dict(payload["audit_chain"]).verify().valid
    assert "private" not in json.dumps(payload) and "local_note" not in str(payload)
    assert CorrectiveActionLedger.from_dict(payload).to_dict() == payload
    payload["records"][0]["severity"] = "low"
    with pytest.raises(ValueError, match="invalid_corrective_action_ledger"):
        CorrectiveActionLedger.from_dict(payload)


def test_skip_and_duplicate_rejected_without_mutation():
    ledger = CorrectiveActionLedger()
    record = ledger.open(action())
    initial = ledger.to_dict()
    with pytest.raises(ValueError, match="invalid_corrective_action_transition"):
        ledger.transition(record.record_id, "closed")
    with pytest.raises(ValueError):
        ledger.open(action())
    assert ledger.to_dict() == initial


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "private note"},
        {"capabilities": ["private-input"]},
        {"root_cause": "private note"},
        {"corrections": ["private note"]},
        {"versions": ["raw-private"]},
    ],
)
def test_reject_uncontrolled_fields(changes):
    with pytest.raises(ValueError, match="^invalid_corrective_action$"):
        action(**changes)
