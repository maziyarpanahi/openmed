"""Synthetic offline tests for the value-free action ledger."""

import json

import pytest

from openmed.agent.audit.action_ledger import (
    ActionEntry,
    ActionLedger,
    ActionLedgerError,
    ActionState,
    verify_action_ledger,
)
from openmed.agent.correlation import ActionId, RunId

RUN = RunId("run_" + "1" * 32)
ACTION = ActionId("act_" + "2" * 32)
OTHER = ActionId("act_" + "3" * 32)
GRANT = "sha256:" + "a" * 64
TOOL = "sha256:" + "b" * 64
REF = "sha256:" + "c" * 64
ROLE = "role:openmed.local/operator"


def record(ledger, state, action_id=ACTION, **changes):
    fields = dict(
        run_id=RUN,
        action_id=action_id,
        state=state,
        actor_role=ROLE,
        grant_digest=GRANT,
        tool_digest=TOOL,
        resource_refs=(REF,),
    )
    fields.update(changes)
    return ledger.record(**fields)


def test_chain_is_durable_deterministic_and_value_free(tmp_path):
    ledger = ActionLedger(tmp_path / "ledger")
    entries = [
        record(ledger, state)
        for state in (
            ActionState.PROPOSED,
            ActionState.APPROVED,
            ActionState.ATTEMPTED,
            ActionState.COMMITTED,
        )
    ]
    assert ActionLedger(tmp_path / "ledger").load() == tuple(entries)
    assert entries[1].previous_digest == entries[0].entry_digest
    assert ledger.export_evidence() == ledger.export_evidence()
    assert ledger.export_evidence()["head_digest"] == entries[-1].entry_digest
    assert len(list((tmp_path / "ledger").glob("entry-*.json"))) == 4
    assert all("payload" not in json.dumps(entry.to_dict()) for entry in entries)


def test_interleaved_actions_and_rejection(tmp_path):
    ledger = ActionLedger(tmp_path / "ledger")
    record(ledger, ActionState.PROPOSED)
    record(ledger, ActionState.PROPOSED, OTHER)
    record(ledger, ActionState.REJECTED)
    record(ledger, ActionState.ATTEMPTED, OTHER)
    assert len(ledger.load()) == 4
    with pytest.raises(ActionLedgerError, match="invalid_transition"):
        record(ledger, ActionState.COMMITTED)


@pytest.mark.parametrize(
    "states,code",
    [
        ([ActionState.COMMITTED], "missing_proposal"),
        ([ActionState.PROPOSED, ActionState.COMMITTED], "invalid_transition"),
        (
            [ActionState.PROPOSED, ActionState.APPROVED, ActionState.APPROVED],
            "invalid_transition",
        ),
    ],
)
def test_invalid_transitions_fail_closed(tmp_path, states, code):
    ledger = ActionLedger(tmp_path / "ledger")
    for state in states[:-1]:
        record(ledger, state)
    with pytest.raises(ActionLedgerError, match=code):
        record(ledger, states[-1])


def test_action_binding_and_run_binding(tmp_path):
    ledger = ActionLedger(tmp_path / "ledger")
    record(ledger, ActionState.PROPOSED)
    with pytest.raises(ActionLedgerError, match="action_metadata_changed"):
        record(ledger, ActionState.APPROVED, tool_digest=GRANT)
    with pytest.raises(ActionLedgerError, match="run_mismatch"):
        record(ledger, ActionState.APPROVED, run_id=RunId("run_" + "4" * 32))


def test_tampering_gap_and_unknown_fields_are_rejected(tmp_path):
    directory = tmp_path / "ledger"
    ledger = ActionLedger(directory)
    record(ledger, ActionState.PROPOSED)
    record(ledger, ActionState.APPROVED)
    second = directory / "entry-00000000000000000001.json"
    data = json.loads(second.read_text())
    data["actor_role"] = "role:openmed.local/reviewer"
    second.write_text(json.dumps(data))
    with pytest.raises(ActionLedgerError, match="entry_digest_mismatch"):
        ledger.load()
    second.rename(directory / "entry-00000000000000000002.json")
    with pytest.raises(ActionLedgerError, match="sequence_gap"):
        ledger.load()
    (directory / "entry-00000000000000000002.json").unlink()
    first = directory / "entry-00000000000000000000.json"
    data = json.loads(first.read_text())
    data["clinical_value"] = "PRIVATE_VALUE"
    first.write_text(json.dumps(data))
    with pytest.raises(ActionLedgerError) as caught:
        ledger.load()
    assert "PRIVATE_VALUE" not in str(caught.value)


def test_raw_values_and_malformed_refs_never_reach_artifacts(tmp_path):
    ledger = ActionLedger(tmp_path / "ledger")
    for changes in (
        {"actor_role": "PRIVATE_VALUE"},
        {"grant_digest": "PRIVATE_VALUE"},
        {"resource_refs": ("Patient/PRIVATE_VALUE",)},
    ):
        with pytest.raises(ActionLedgerError) as caught:
            record(ledger, ActionState.PROPOSED, **changes)
        assert "PRIVATE_VALUE" not in str(caught.value)
    assert ledger.load() == ()


def test_conflicting_or_unsafe_files_fail_closed(tmp_path):
    directory = tmp_path / "ledger"
    ledger = ActionLedger(directory)
    entry = record(ledger, ActionState.PROPOSED)
    with pytest.raises(ActionLedgerError, match="sequence_gap"):
        ledger.append(entry)
    (directory / "unexpected.json").write_text("{}")
    with pytest.raises(ActionLedgerError, match="unexpected_ledger_file"):
        ledger.load()


def test_forged_predecessor_and_duplicate_json_fields_fail_closed(tmp_path):
    directory = tmp_path / "ledger"
    ledger = ActionLedger(directory)
    first = record(ledger, ActionState.PROPOSED)
    forged = ActionEntry.create(
        run_id=RUN,
        action_id=ACTION,
        sequence=1,
        state=ActionState.APPROVED,
        actor_role=ROLE,
        grant_digest=GRANT,
        tool_digest=TOOL,
        resource_refs=(REF,),
        previous_digest=GRANT,
    )
    with pytest.raises(ActionLedgerError, match="chain_mismatch"):
        verify_action_ledger((first, forged))
    with pytest.raises(ActionLedgerError, match="chain_mismatch"):
        ledger.append(forged)

    path = directory / "entry-00000000000000000000.json"
    serialized = first.to_json()
    path.write_text(serialized.replace('"sequence":0', '"sequence":0,"sequence":0'))
    with pytest.raises(ActionLedgerError, match="duplicate_entry_field"):
        ledger.load()
