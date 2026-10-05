"""Synthetic offline safety controls for persistent effect admission."""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from openmed.agent.admission import (
    AdmissionError,
    AdmissionReason,
    AdmissionRole,
    AdmissionState,
    EffectAdmissionController,
    SQLiteAdmissionStore,
    dispatch_with_admission,
)
from openmed.agent.identifiers import WorkflowId

KEY = b"synthetic-admission-key-32-bytes-only"
WORKFLOW = WorkflowId("workflow:org.example/synthetic-fhir")
OTHER = WorkflowId("workflow:org.example/synthetic-omop")


def store_at(tmp_path: Path) -> SQLiteAdmissionStore:
    return SQLiteAdmissionStore(tmp_path / "ledger.db", tmp_path / "anchor.db", KEY)


@pytest.fixture
def controller(tmp_path: Path) -> EffectAdmissionController:
    store = store_at(tmp_path)
    store.initialize(now=100)
    return EffectAdmissionController(store)


@pytest.mark.parametrize("workflow", [WORKFLOW, OTHER])
def test_no_configuration_never_invokes_effect(workflow: WorkflowId) -> None:
    effect = Mock()
    with pytest.raises(AdmissionError, match="admission_disabled"):
        dispatch_with_admission(effect, workflow_id=workflow)
    effect.assert_not_called()


def test_workflow_enable_admits_only_that_scope(
    controller: EffectAdmissionController,
) -> None:
    assert controller.status().state is AdmissionState.DISABLED
    controller.enable(workflow_id=WORKFLOW, now=101)
    effect = Mock(return_value="opaque-result")
    assert (
        dispatch_with_admission(effect, workflow_id=WORKFLOW, admission=controller)
        == "opaque-result"
    )
    with pytest.raises(AdmissionError, match="admission_disabled"):
        dispatch_with_admission(effect, workflow_id=OTHER, admission=controller)
    assert effect.call_count == 1


def test_global_stop_overrides_every_workflow_and_requires_fresh_enable(
    controller: EffectAdmissionController,
    tmp_path: Path,
) -> None:
    controller.enable(workflow_id=WORKFLOW, now=101)
    preview_generation = controller.require_admitted(WORKFLOW).generation
    controller.stop(now=102)
    restarted = EffectAdmissionController(store_at(tmp_path))
    for workflow in (WORKFLOW, OTHER):
        with pytest.raises(AdmissionError, match="admission_stopped"):
            restarted.require_admitted(workflow, generation=preview_generation)
    with pytest.raises(AdmissionError, match="admission_stopped"):
        restarted.enable(workflow_id=WORKFLOW, now=103)
    restarted.enable(now=104)
    with pytest.raises(AdmissionError, match="stale_generation"):
        restarted.require_admitted(WORKFLOW, generation=preview_generation)
    assert restarted.require_admitted(WORKFLOW).generation > preview_generation


def test_workflow_stop_does_not_stop_other_scope(
    controller: EffectAdmissionController,
) -> None:
    controller.enable(now=101)
    controller.stop(workflow_id=WORKFLOW, now=102)
    assert controller.status(WORKFLOW).state is AdmissionState.STOPPED
    assert controller.status(OTHER).state is AdmissionState.ENABLED
    controller.enable(workflow_id=WORKFLOW, now=103)
    assert controller.status(WORKFLOW).state is AdmissionState.ENABLED


def test_receipts_are_signed_chained_and_content_free(
    controller: EffectAdmissionController, tmp_path: Path
) -> None:
    receipt = controller.enable(workflow_id=WORKFLOW, now=101)
    signature = receipt.pop("signature")
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    assert (
        signature == "hmac-sha256:" + hmac.new(KEY, encoded, hashlib.sha256).hexdigest()
    )
    assert receipt["role"] == "operator"
    assert receipt["time"] == 101
    assert receipt["generation"] == 2
    assert receipt["reason_code"] == "explicit_enable"
    assert receipt["previous_digest"].startswith("sha256:")
    encoded_ledger = (tmp_path / "ledger.db").read_bytes()
    assert KEY not in encoded_ledger
    assert str(tmp_path).encode() not in encoded_ledger
    assert "payload" not in receipt
    assert set(controller.status(WORKFLOW).to_dict()) == {
        "scope",
        "reason_code",
        "generation",
        "receipt_digest",
    }


@pytest.mark.parametrize("file", ["ledger.db", "anchor.db"])
@pytest.mark.parametrize("damage", ["missing", "corrupted", "rollback"])
def test_missing_corrupted_or_rolled_back_state_fails_closed(
    tmp_path: Path,
    file: str,
    damage: str,
) -> None:
    store = store_at(tmp_path)
    store.initialize(now=100)
    original = (tmp_path / file).read_bytes()
    controller = EffectAdmissionController(store)
    controller.enable(now=101)
    controller.stop(now=102)
    target = tmp_path / file
    if damage == "missing":
        target.unlink()
    elif damage == "corrupted":
        target.write_bytes(b"synthetic-private-content; not a database")
    else:
        target.write_bytes(original)
    for check in (controller, EffectAdmissionController(store_at(tmp_path))):
        assert check.status(WORKFLOW).state is AdmissionState.STOPPED
        with pytest.raises(AdmissionError, match="untrusted_state"):
            check.require_admitted(WORKFLOW)
        with pytest.raises(AdmissionError, match="untrusted_state"):
            check.enable(now=103)
    assert controller.status().generation >= 3
    if file == "ledger.db":
        assert EffectAdmissionController(store_at(tmp_path)).status().generation == 3


@pytest.mark.parametrize(
    "field",
    [
        "scope",
        "role",
        "reason_code",
        "time",
        "generation",
        "signature",
        "previous_digest",
        "state",
        "schema_version",
        "payload",
    ],
)
def test_receipt_tampering_is_rejected_without_echoing_content(
    controller: EffectAdmissionController,
    tmp_path: Path,
    field: str,
) -> None:
    controller.enable(now=101)
    sentinel = "Synthetic Patient; token-secret; /private/source"
    with sqlite3.connect(tmp_path / "ledger.db") as db:
        receipt = json.loads(
            db.execute("SELECT document FROM receipts WHERE generation=2").fetchone()[0]
        )
        receipt[field] = sentinel
        db.execute(
            "UPDATE receipts SET document=? WHERE generation=2", (json.dumps(receipt),)
        )
    status = controller.status(WORKFLOW)
    assert status.reason_code == "untrusted_state"
    assert sentinel not in json.dumps(status.to_dict())
    with pytest.raises(AdmissionError) as caught:
        controller.require_admitted(WORKFLOW)
    assert sentinel not in str(caught.value)


def test_truncated_ledger_cannot_bypass_anchor(
    controller: EffectAdmissionController, tmp_path: Path
) -> None:
    controller.enable(now=101)
    controller.stop(now=102)
    with sqlite3.connect(tmp_path / "ledger.db") as db:
        db.execute("DELETE FROM receipts WHERE generation=3")
    assert (
        EffectAdmissionController(store_at(tmp_path)).status().reason_code
        == "untrusted_state"
    )


def test_initialization_never_resets_existing_state(
    controller: EffectAdmissionController, tmp_path: Path
) -> None:
    controller.stop(now=101)
    with pytest.raises(AdmissionError, match="already_initialized"):
        store_at(tmp_path).initialize(now=102)
    assert controller.status().state is AdmissionState.STOPPED


def test_failed_transition_rolls_back_ledger_and_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = store_at(tmp_path)
    store.initialize(now=100)
    append = store._append

    def interrupt(db: sqlite3.Connection, receipt: dict) -> None:
        append(db, receipt)
        raise OSError("synthetic-private-path")

    monkeypatch.setattr(store, "_append", interrupt)
    with pytest.raises(AdmissionError, match="untrusted_state"):
        EffectAdmissionController(store).enable(now=101)
    restarted = EffectAdmissionController(store_at(tmp_path))
    assert restarted.status().generation == 1
    assert restarted.status().state is AdmissionState.DISABLED


def test_concurrent_operators_have_unique_increasing_generations(
    tmp_path: Path,
) -> None:
    store_at(tmp_path).initialize(now=100)

    def enable(index: int) -> int:
        receipt = EffectAdmissionController(store_at(tmp_path)).enable(now=101 + index)
        return receipt["generation"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        generations = list(pool.map(enable, range(12)))
    assert sorted(generations) == list(range(2, 14))
    assert store_at(tmp_path).status().generation == 13


@pytest.mark.parametrize("clock", [-1, True, "Synthetic Patient", 2**64])
def test_invalid_clock_does_not_change_state(
    controller: EffectAdmissionController, clock: object
) -> None:
    with pytest.raises(AdmissionError, match="invalid_time"):
        controller.enable(now=clock)
    assert controller.status().generation == 1


def test_invalid_control_metadata_is_rejected(
    controller: EffectAdmissionController,
) -> None:
    with pytest.raises(AdmissionError, match="invalid_scope"):
        controller.enable(workflow_id="Synthetic Patient")
    with pytest.raises(AdmissionError, match="invalid_transition"):
        controller.stop(reason=AdmissionReason.EXPLICIT_ENABLE)
    with pytest.raises(AdmissionError, match="invalid_transition"):
        controller.enable(role="Synthetic Patient")
    assert controller.status().generation == 1


def test_anchor_and_ledger_must_be_distinct(tmp_path: Path) -> None:
    with pytest.raises(AdmissionError, match="invalid_anchor"):
        SQLiteAdmissionStore(tmp_path / "ledger.db", tmp_path / "ledger.db", KEY)


def test_symlink_state_is_rejected(
    controller: EffectAdmissionController, tmp_path: Path
) -> None:
    controller.enable(now=101)
    shutil.move(tmp_path / "ledger.db", tmp_path / "moved.db")
    (tmp_path / "ledger.db").symlink_to(tmp_path / "moved.db")
    assert controller.status().reason_code == "untrusted_state"


@pytest.mark.parametrize(
    "document",
    ["[" * 2000 + "]" * 2000, "x" * 16385, '{"generation":1,"generation":2}'],
)
def test_malformed_nested_or_oversized_receipt_fails_closed(
    controller: EffectAdmissionController, tmp_path: Path, document: str
) -> None:
    with sqlite3.connect(tmp_path / "ledger.db") as db:
        db.execute("UPDATE receipts SET document=?", (document,))
    assert controller.status().reason_code == "untrusted_state"


def test_signed_invalid_transition_still_fails_closed(
    controller: EffectAdmissionController, tmp_path: Path
) -> None:
    with sqlite3.connect(tmp_path / "ledger.db") as db:
        receipt = json.loads(db.execute("SELECT document FROM receipts").fetchone()[0])
        receipt["state"] = "enabled"
        unsigned = {key: value for key, value in receipt.items() if key != "signature"}
        encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        receipt["signature"] = (
            "hmac-sha256:" + hmac.new(KEY, encoded, hashlib.sha256).hexdigest()
        )
        db.execute("UPDATE receipts SET document=?", (json.dumps(receipt),))
        digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
    with sqlite3.connect(tmp_path / "anchor.db") as db:
        db.execute("UPDATE head SET digest=?", (digest,))
    assert controller.status().reason_code == "untrusted_state"


def test_unavailable_storage_and_short_keys_are_controlled(tmp_path: Path) -> None:
    with pytest.raises(AdmissionError, match="invalid_key"):
        SQLiteAdmissionStore(tmp_path / "ledger.db", tmp_path / "anchor.db", b"secret")
    store = SQLiteAdmissionStore(
        tmp_path / "missing" / "ledger.db", tmp_path / "anchor.db", KEY
    )
    with pytest.raises(AdmissionError, match="untrusted_state"):
        store.initialize(now=100)
    assert store.status().reason_code == "untrusted_state"
