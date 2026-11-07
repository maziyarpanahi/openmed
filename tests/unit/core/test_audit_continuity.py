"""Tests for audit continuity across key rotation, retention, and restore."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openmed.core.audit import AuditReport, AuditSpan, hash_text
from openmed.core.audit_chain import AuditChain
from openmed.core.audit_continuity import (
    CONTINUITY_REASON_CODES,
    AuditContinuityError,
    AuditContinuityRecoveryHook,
    AuditContinuityStore,
    ContinuityAnchorRef,
    ContinuityEvent,
    ContinuityManifest,
    record_restore_transition,
    tombstone_segment_digest,
    verify_continuity,
    verify_tombstone_segment,
)
from openmed.core.audit_key_rotation import AuditKeyRotationSigner
from openmed.risk.audit_retention import (
    AuditRetentionPolicy,
    RetentionRule,
    scrub_audit_artifacts,
)

ROOT = Path(__file__).resolve().parents[3]
KEYS = {
    f"audit-epoch-{index}": f"synthetic-continuity-key-{index:02d}-0123456789abcdef".encode(
        "ascii"
    )
    for index in range(12)
}
POLICY = AuditRetentionPolicy(rules={"audit": RetentionRule.days(30)})
BACKUP_DIGEST = "sha256:" + "b" * 64
SENTINEL_PREFIX = "Quorvexa Sentinel"


def _sentinel(index: int) -> str:
    return f"{SENTINEL_PREFIX} {index:04d}"


def _report(index: int) -> AuditReport:
    name = _sentinel(index)
    original = f"Patient {name} attended the synthetic clinic."
    start = original.index(name)
    return AuditReport(
        policy="synthetic_policy",
        resolved_profile={"method": "mask"},
        detectors=[],
        safety_sweep={},
        spans=[
            AuditSpan(
                start=start,
                end=start + len(name),
                label="NAME",
                canonical_label="PERSON",
                sources=["regex"],
                confidence=1.0,
                threshold=0.5,
                action="mask",
                surrogate="[PERSON]",
                text_hash=hash_text(name),
            )
        ],
        thresholds={},
        residual_risk={},
        openmed_version="synthetic",
        manifest_hash=hash_text("synthetic manifest"),
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text(
            "Patient [PERSON] attended the synthetic clinic."
        ),
    )


def _store(root: Path, **kwargs: Any) -> AuditContinuityStore:
    return AuditContinuityStore(root / "audit-chain.json", key_provider=KEYS, **kwargs)


def _initialized(root: Path, **kwargs: Any) -> AuditContinuityStore:
    store = _store(root, **kwargs)
    store.initialize(key_id="audit-epoch-0", chain_id="chain-synthetic")
    return store


def _append(store: AuditContinuityStore, start: int, count: int) -> int:
    for index in range(start, start + count):
        store.append(_report(index))
    return start + count


def _documents(store: AuditContinuityStore) -> tuple[dict[str, Any], dict[str, Any]]:
    chain = json.loads(store.chain_path.read_text(encoding="utf-8"))
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    return chain, manifest


def _verify(chain: Any, manifest: Any, **kwargs: Any):
    return verify_continuity(chain, manifest, key_provider=KEYS, **kwargs)


def _resign(
    manifest: dict[str, Any],
    index: int,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    *,
    keys: dict[str, str] | None = None,
    provider: dict[str, bytes] = KEYS,
) -> None:
    """Mutate one event and recompute its digest and signatures in place."""

    event = manifest["events"][index]
    if mutate is not None:
        mutate(event)
    digest = ContinuityEvent.from_dict(event).compute_digest(manifest["chain_id"])
    event["digest"] = digest
    for signature in event["signatures"]:
        key_id = (keys or {}).get(signature["role"], signature["key_id"])
        signature["key_id"] = key_id
        signature["value"] = AuditKeyRotationSigner(
            key_id=key_id, key_provider=provider
        ).sign_digest(digest, purpose="audit-continuity", role=signature["role"])
    if index == len(manifest["events"]) - 1:
        manifest["head_digest"] = digest


def _lifecycle(root: Path) -> tuple[AuditContinuityStore, int]:
    """Append, rotate three times, and prune twice; return the next report id."""

    store = _initialized(root)
    next_id = 0
    for epoch in range(1, 4):
        next_id = _append(store, next_id, 4)
        store.anchor()
        store.rotate(f"audit-epoch-{epoch}")
    next_id = _append(store, next_id, 2)
    store.prune_through(4, policy=POLICY)
    store.prune_through(9, policy_fingerprint=POLICY.fingerprint)
    store.anchor()
    return store, next_id


# Lifecycle -----------------------------------------------------------------


def test_chain_stays_verifiable_across_rotation_pruning_backup_and_restore(
    tmp_path: Path,
) -> None:
    live = tmp_path / "live"
    store, next_id = _lifecycle(live)
    assert store.verify().code == "valid"

    backup = tmp_path / "backup"
    shutil.copytree(live, backup)
    next_id = _append(store, next_id, 3)
    store.rotate("audit-epoch-4")
    next_id = _append(store, next_id, 1)
    store.anchor()
    lost_head = store.load()[1].head_ref()

    shutil.rmtree(live)
    shutil.copytree(backup, live)
    event = record_restore_transition(
        live / "audit-chain.json",
        key_provider=KEYS,
        backup_digest=BACKUP_DIGEST,
        restore_id="restore-synthetic",
    )
    assert event.kind == "restore"
    next_id = _append(store, next_id, 2)
    store.rotate("audit-epoch-5")
    _append(store, next_id, 2)
    store.anchor()

    report = store.verify(
        trusted_anchors=[lost_head], allowed_policy_fingerprints=[POLICY.fingerprint]
    )
    assert report.valid, report.code
    assert report.code == "valid"
    counts = report.counts
    assert counts["rotation_count"] == 4
    assert counts["retention_count"] == 2
    assert counts["restore_count"] == 1
    assert counts["pruned_entry_count"] == 10
    assert counts["retained_entry_count"] == counts["next_sequence"] - 10
    assert counts["unanchored_entry_count"] == 0
    assert counts["superseded_trusted_anchor_count"] == 1
    assert report.policy_fingerprints == (POLICY.fingerprint,)

    epochs = report.epochs
    assert [epoch.epoch for epoch in epochs] == list(range(len(epochs)))
    assert [epoch.opened_by for epoch in epochs] == [
        "genesis",
        "rotation",
        "rotation",
        "rotation",
        "restore",
        "rotation",
    ]
    assert epochs[0].start_sequence == 0
    for previous, current in zip(epochs, epochs[1:], strict=False):
        assert previous.state == "retired"
        assert previous.end_sequence == current.start_sequence
    assert epochs[-1].state == "active"
    assert epochs[-1].key_id == "audit-epoch-5"


def test_restore_after_live_manifest_survives_reapplies_pruning(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path / "live")
    next_id = _append(store, 0, 8)
    store.anchor()
    old_chain = store.chain_path.read_text(encoding="utf-8")

    next_id = _append(store, next_id, 4)
    store.prune_through(5, retention_report=_retention_report())
    store.rotate("audit-epoch-1")
    _append(store, next_id, 2)
    store.anchor()

    # Only the chain file is rolled back; the manifest is the live one.
    store.chain_path.write_text(old_chain, encoding="utf-8")
    assert store.verify().code == "incomplete_transition"
    store.record_restore_transition(backup_digest=BACKUP_DIGEST)

    chain, _ = store.load()
    assert chain.base_sequence == 6
    assert chain.next_sequence == 8
    report = store.verify()
    assert report.valid, report.code
    assert report.counts["superseded_event_count"] >= 1
    assert report.epochs[-1].opened_by == "restore"
    retained = store.chain_path.read_text(encoding="utf-8")
    for index in range(6):
        assert _sentinel(index) not in retained
        assert hash_text(_sentinel(index)) not in retained


def test_restore_refuses_a_chain_that_forked_from_the_manifest(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path / "live")
    _append(store, 0, 3)
    store.anchor()
    forked = AuditChain()
    for index in range(100, 104):
        forked.append(_report(index))
    forked.write(store.chain_path)

    with pytest.raises(AuditContinuityError) as error:
        store.record_restore_transition(backup_digest=BACKUP_DIGEST)

    assert error.value.code == "anchor_mismatch"


def test_append_refuses_a_swapped_or_truncated_chain(tmp_path: Path) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 3)
    store.anchor()
    original = store.chain_path.read_text(encoding="utf-8")

    forked = AuditChain()
    for index in range(100, 104):
        forked.append(_report(index))
    forked.write(store.chain_path)
    with pytest.raises(AuditContinuityError) as swapped:
        store.append(_report(9))

    shorter = AuditChain.from_json(original)
    shorter = AuditChain(shorter.entries[:2])
    shorter.write(store.chain_path)
    with pytest.raises(AuditContinuityError) as truncated:
        store.append(_report(9))

    store.chain_path.unlink()
    with pytest.raises(AuditContinuityError) as missing:
        store.append(_report(9))

    assert swapped.value.code == "anchor_mismatch"
    assert truncated.value.code == "sequence_gap"
    assert missing.value.code == "sequence_gap"


def test_recovery_hook_records_the_restore_epoch_from_a_service_context(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 2)
    store.anchor()

    class _Context:
        recovery_point_id = "rp_" + "0" * 32
        manifest_digest = "sha256:" + "c" * 64

    hook = AuditContinuityRecoveryHook(
        store.chain_path, key_provider=KEYS, key_id="audit-epoch-1"
    )
    hook.open_recovery_epoch(_Context())

    _, manifest = store.load()
    restore = manifest.head_event
    assert restore.kind == "restore"
    assert dict(restore.body) == {
        "backup_digest": _Context.manifest_digest,
        "key_id": "audit-epoch-1",
        "restore_id": _Context.recovery_point_id,
    }
    assert manifest.current_epoch.key_id == "audit-epoch-1"
    assert store.verify().valid
    assert "AuditContinuityRecoveryHook()" == repr(hook)


def _retention_report():
    artifacts = [
        {"artifact_id": "synthetic-a", "created_at": "2026-01-01", "counts": {"a": 1}},
        {"artifact_id": "synthetic-b", "created_at": "2026-03-01", "counts": {"a": 2}},
    ]
    return scrub_audit_artifacts(
        [{**item, "disposition": "audit"} for item in artifacts],
        POLICY,
        as_of="2026-03-10",
    )


def test_tombstone_records_the_approving_retention_report(tmp_path: Path) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 3)
    retention = _retention_report()
    event = store.prune_through(1, retention_report=retention, policy=POLICY)

    tombstone = event.tombstone
    assert tombstone is not None
    assert tombstone.policy_fingerprint == POLICY.fingerprint
    assert tombstone.retention_report_digest == retention.integrity_digest
    assert tombstone.record_count == 2
    assert (tombstone.first_sequence, tombstone.last_sequence) == (0, 1)
    store.anchor()
    assert store.verify().valid


def test_pruning_requires_one_consistent_policy_approval(tmp_path: Path) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 3)
    other = AuditRetentionPolicy(rules={"audit": RetentionRule.days(7)})

    with pytest.raises(AuditContinuityError) as missing:
        store.prune_through(1)
    with pytest.raises(AuditContinuityError) as mismatch:
        store.prune_through(1, policy=POLICY, policy_fingerprint=other.fingerprint)
    with pytest.raises(AuditContinuityError) as out_of_range:
        store.prune_through(9, policy=POLICY)

    assert missing.value.code == "invalid_argument"
    assert mismatch.value.code == "policy_mismatch"
    assert out_of_range.value.code == "invalid_argument"
    assert store.load()[0].base_sequence == 0


def test_rotation_refuses_reused_keys_and_is_idempotent_for_the_active_key(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path)
    first = store.rotate("audit-epoch-1")
    again = store.rotate("audit-epoch-1")

    with pytest.raises(AuditContinuityError) as reused:
        store.rotate("audit-epoch-0")
    with pytest.raises(AuditContinuityError) as unknown:
        store.rotate("audit-missing-key")

    assert again == first
    assert reused.value.code == "retired_signer_key"
    assert unknown.value.code == "signer_key_unavailable"
    assert store.load()[1].event_count == 2


def test_store_refuses_operations_before_initialization(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(AuditContinuityError) as error:
        store.append(_report(0))
    assert error.value.code == "not_initialized"
    store.initialize(key_id="audit-epoch-0")
    with pytest.raises(AuditContinuityError) as again:
        store.initialize(key_id="audit-epoch-0")
    assert again.value.code == "already_initialized"
    assert store.load()[1].chain_id.startswith("chain-")


def test_existing_chain_is_adopted_by_the_genesis_event(tmp_path: Path) -> None:
    chain = AuditChain()
    for index in range(3):
        chain.append(_report(index))
    chain.write(tmp_path / "audit-chain.json")

    store = _initialized(tmp_path)
    genesis = store.load()[1].head_event

    assert genesis.next_sequence == 3
    assert genesis.head_hash == chain.head_hash
    assert store.verify().valid


def test_anchor_interval_signs_the_head_without_explicit_anchors(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path, anchor_interval=3)
    _append(store, 0, 7)

    report = store.verify(allow_unanchored_tail=True)
    assert report.valid
    assert report.counts["anchor_count"] == 2
    assert report.counts["unanchored_entry_count"] == 1
    assert store.verify().code == "unanchored_tail"


# Tamper detection -------------------------------------------------------------


@pytest.fixture
def anchored(tmp_path: Path) -> AuditContinuityStore:
    store, _ = _lifecycle(tmp_path)
    assert store.verify().valid
    return store


def _index_of(manifest: dict[str, Any], kind: str, *, last: bool = False) -> int:
    indices = [
        position
        for position, event in enumerate(manifest["events"])
        if event["kind"] == kind
    ]
    return indices[-1] if last else indices[0]


def test_deleted_record_is_a_sequence_gap(anchored: AuditContinuityStore) -> None:
    chain, manifest = _documents(anchored)
    del chain["entries"][1]

    assert _verify(chain, manifest).code == "sequence_gap"


def test_truncated_chain_tail_is_a_sequence_gap(
    anchored: AuditContinuityStore,
) -> None:
    chain, manifest = _documents(anchored)
    del chain["entries"][-1]
    chain["entry_count"] -= 1
    chain["head_hash"] = chain["entries"][-1]["entry_hash"]

    assert _verify(chain, manifest).code == "sequence_gap"


def test_rebuilt_chain_without_a_record_is_an_anchor_mismatch(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 4)
    store.anchor()
    _, manifest = _documents(store)
    rebuilt = AuditChain()
    for index in (0, 2, 3, 4):
        rebuilt.append(_report(index))

    report = _verify(rebuilt, manifest)

    assert report.code == "anchor_mismatch"
    assert report.event_index == 1


def test_unexplained_pruning_is_a_sequence_gap(tmp_path: Path) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 4)
    store.anchor()
    chain, manifest = store.load()
    chain.prune_through(1)

    assert _verify(chain, manifest).code == "sequence_gap"


def test_missing_duplicated_and_reordered_anchors_fail_with_stable_codes(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    def missing(manifest: dict[str, Any]) -> None:
        del manifest["events"][_index_of(manifest, "anchor")]

    def duplicated(manifest: dict[str, Any]) -> None:
        position = _index_of(manifest, "anchor")
        manifest["events"].insert(position, dict(manifest["events"][position]))

    def reordered(manifest: dict[str, Any]) -> None:
        position = _index_of(manifest, "anchor")
        events = manifest["events"]
        events[position], events[position + 1] = events[position + 1], events[position]

    def modified(manifest: dict[str, Any]) -> None:
        position = _index_of(manifest, "anchor")
        manifest["events"][position]["head_hash"] = "sha256:" + "0" * 64

    def truncated_header(manifest: dict[str, Any]) -> None:
        manifest["events"].pop()

    cases = {
        "missing": (missing, "event_gap"),
        "duplicated": (duplicated, "event_duplicate"),
        "reordered": (reordered, "event_reordered"),
        "modified": (modified, "anchor_mismatch"),
        "truncated": (truncated_header, "event_gap"),
    }
    for name, (mutate, code) in cases.items():
        with subtests.test(case=name):
            chain, manifest = _documents(anchored)
            mutate(manifest)
            report = _verify(chain, manifest)
            assert report.code == code
            assert not report.valid


def test_resigned_anchor_without_the_key_is_rejected(
    anchored: AuditContinuityStore,
) -> None:
    chain, manifest = _documents(anchored)
    position = _index_of(manifest, "anchor")
    attacker = {"audit-attacker": b"synthetic-attacker-key-material-0123456789"}

    def move_head(event: dict[str, Any]) -> None:
        event["head_hash"] = "sha256:" + "1" * 64

    _resign(manifest, position, move_head, provider={**KEYS, **attacker})
    manifest["events"][position]["signatures"][0]["value"] = "0" * 64

    assert _verify(chain, manifest).code == "signature_invalid"


def test_unknown_and_retired_signer_material_fail_closed(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    attacker = {"audit-attacker": b"synthetic-attacker-key-material-0123456789"}
    with subtests.test(case="unknown"):
        chain, manifest = _documents(anchored)
        position = _index_of(manifest, "anchor", last=True)
        _resign(
            manifest,
            position,
            keys={"signer": "audit-attacker"},
            provider={**KEYS, **attacker},
        )
        assert _verify(chain, manifest).code == "unknown_signer_key"

    with subtests.test(case="retired"):
        chain, manifest = _documents(anchored)
        position = _index_of(manifest, "anchor", last=True)
        _resign(manifest, position, keys={"signer": "audit-epoch-0"})
        assert _verify(chain, manifest).code == "retired_signer_key"

    with subtests.test(case="unavailable"):
        chain, manifest = _documents(anchored)
        partial = {key: value for key, value in KEYS.items() if key != "audit-epoch-0"}
        report = verify_continuity(chain, manifest, key_provider=partial)
        assert report.code == "signer_key_unavailable"
        assert report.event_index == 0


def test_trusted_anchor_lets_retired_keys_be_dropped(
    anchored: AuditContinuityStore,
) -> None:
    chain, manifest = anchored.load()
    checkpoint = manifest.head_ref()
    next_id = _append(anchored, 100, 2)
    anchored.anchor()
    assert next_id == 102
    chain, manifest = anchored.load()
    current_only = {"audit-epoch-3": KEYS["audit-epoch-3"]}

    without = verify_continuity(chain, manifest, key_provider=current_only)
    with_checkpoint = verify_continuity(
        chain, manifest, key_provider=current_only, trusted_anchors=[checkpoint]
    )

    assert without.code == "signer_key_unavailable"
    assert with_checkpoint.valid, with_checkpoint.code
    assert with_checkpoint.counts["trusted_anchor_count"] == 1
    assert with_checkpoint.counts["unverified_signature_count"] > 0


def test_trusted_anchor_detects_truncation_and_forks(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    _, manifest_obj = anchored.load()
    head = manifest_obj.head_ref()
    earlier = ContinuityAnchorRef(index=2, digest=manifest_obj.events[2].digest)

    with subtests.test(case="truncated"):
        chain, manifest = _documents(anchored)
        manifest["events"].pop()
        manifest["event_count"] -= 1
        manifest["head_digest"] = manifest["events"][-1]["digest"]
        report = _verify(chain, manifest, trusted_anchors=[head])
        assert report.code == "anchor_missing"

    with subtests.test(case="fork"):
        chain, manifest = _documents(anchored)
        wrong = ContinuityAnchorRef(index=earlier.index, digest="sha256:" + "2" * 64)
        report = _verify(chain, manifest, trusted_anchors=[wrong.to_dict()])
        assert report.code == "fork"

    with subtests.test(case="matching"):
        chain, manifest = _documents(anchored)
        report = _verify(chain, manifest, trusted_anchors=[earlier, head])
        assert report.valid
        assert report.counts["trusted_anchor_count"] == 2


def test_reordered_and_missing_epochs_are_detected(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    with subtests.test(case="summary reordered"):
        chain, manifest = _documents(anchored)
        epochs = manifest["epochs"]
        epochs[1], epochs[2] = epochs[2], epochs[1]
        assert _verify(chain, manifest).code == "epoch_mismatch"

    with subtests.test(case="rotation events reordered"):
        chain, manifest = _documents(anchored)
        rotations = [
            position
            for position, event in enumerate(manifest["events"])
            if event["kind"] == "rotation"
        ]
        events = manifest["events"]
        first, second = rotations[0], rotations[1]
        events[first], events[second] = events[second], events[first]
        assert _verify(chain, manifest).code == "event_reordered"

    with subtests.test(case="rotation renumbered"):
        chain, manifest = _documents(anchored)
        position = _index_of(manifest, "rotation")
        manifest["events"][position]["epoch"] = 7
        assert _verify(chain, manifest).code == "epoch_mismatch"

    with subtests.test(case="rotation removed"):
        chain, manifest = _documents(anchored)
        del manifest["events"][_index_of(manifest, "rotation")]
        assert _verify(chain, manifest).code == "event_gap"


def test_unsigned_epoch_transition_is_rejected(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    for role in ("outgoing", "incoming"):
        with subtests.test(role=role):
            chain, manifest = _documents(anchored)
            event = manifest["events"][_index_of(manifest, "rotation")]
            event["signatures"] = [
                item for item in event["signatures"] if item["role"] != role
            ]
            assert _verify(chain, manifest).code == "unsigned_epoch_transition"


def test_forged_tombstones_are_rejected(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    with subtests.test(case="edited record count"):
        chain, manifest = _documents(anchored)
        position = _index_of(manifest, "retention")
        manifest["events"][position]["body"]["record_count"] = 1
        assert _verify(chain, manifest).code == "tombstone_mismatch"

    with subtests.test(case="edited segment digest"):
        chain, manifest = _documents(anchored)
        position = _index_of(manifest, "retention")
        manifest["events"][position]["body"]["segment_digest"] = "sha256:" + "3" * 64
        assert _verify(chain, manifest).code == "tombstone_mismatch"

    with subtests.test(case="insider tombstone with the wrong boundary"):
        store = _initialized(anchored.chain_path.parent / "insider")
        _append(store, 0, 4)
        store.prune_through(1, policy=POLICY)
        chain, manifest = _documents(store)
        position = _index_of(manifest, "retention", last=True)

        def move_boundary(event: dict[str, Any]) -> None:
            event["body"]["end_hash"] = "sha256:" + "4" * 64

        _resign(manifest, position, move_boundary)
        assert _verify(chain, manifest).code == "tombstone_mismatch"

    with subtests.test(case="insider tombstone with a coverage gap"):
        store = _initialized(anchored.chain_path.parent / "gap")
        _append(store, 0, 4)
        store.prune_through(1, policy=POLICY)
        chain, manifest = _documents(store)
        position = _index_of(manifest, "retention", last=True)

        def skip_first(event: dict[str, Any]) -> None:
            event["body"]["first_sequence"] = 1
            event["body"]["record_count"] = 1

        _resign(manifest, position, skip_first)
        assert _verify(chain, manifest).code == "sequence_gap"


def test_policy_versions_must_match_when_required(
    anchored: AuditContinuityStore,
) -> None:
    other = AuditRetentionPolicy(rules={"audit": RetentionRule.days(7)})

    assert (
        anchored.verify(allowed_policy_fingerprints=[other.fingerprint]).code
        == "policy_mismatch"
    )
    assert anchored.verify(allowed_policy_fingerprints=[POLICY.fingerprint]).valid


def test_manifest_shape_version_and_chain_binding_fail_closed(
    anchored: AuditContinuityStore,
    subtests,
) -> None:
    def unknown_field(manifest: dict[str, Any]) -> None:
        manifest["events"][0][_sentinel(9)] = "unsafe"

    def future_version(manifest: dict[str, Any]) -> None:
        manifest["version"] = 99

    def other_chain(manifest: dict[str, Any]) -> None:
        manifest["chain_genesis_hash"] = "sha256:" + "5" * 64

    def changed_chain_id(manifest: dict[str, Any]) -> None:
        manifest["chain_id"] = "chain-other"

    cases = {
        "unknown field": (unknown_field, "malformed_manifest"),
        "future version": (future_version, "unsupported_version"),
        "other chain format": (other_chain, "chain_mismatch"),
        "changed chain id": (changed_chain_id, "chain_mismatch"),
    }
    for name, (mutate, code) in cases.items():
        with subtests.test(case=name):
            chain, manifest = _documents(anchored)
            mutate(manifest)
            report = _verify(chain, manifest)
            assert report.code == code
            assert _sentinel(9) not in report.to_json()

    chain, manifest = _documents(anchored)
    assert (
        _verify(chain, manifest, expected_chain_id="chain-other").code
        == "chain_mismatch"
    )


def test_every_reason_code_is_controlled() -> None:
    assert "valid" in CONTINUITY_REASON_CODES
    assert all(code.replace("_", "").isalpha() for code in CONTINUITY_REASON_CODES)
    assert str(AuditContinuityError("not-a-code")) == (
        "audit continuity failed: malformed_manifest"
    )


# Crash recovery and concurrency -------------------------------------------------


def _interrupt_next_chain_write(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AuditChain.write

    def interrupted(self: AuditChain, path: Any) -> Path:
        monkeypatch.setattr(AuditChain, "write", original)
        raise OSError("simulated crash before the chain rewrite")

    monkeypatch.setattr(AuditChain, "write", interrupted)


def test_interrupted_pruning_is_completed_deterministically_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _initialized(tmp_path / "primary")
    _append(store, 0, 6)
    store.anchor()
    _interrupt_next_chain_write(monkeypatch)
    with pytest.raises(AuditContinuityError) as crashed:
        store.prune_through(3, policy=POLICY)
    assert crashed.value.code == "io_failed"
    assert store.verify().code == "incomplete_transition"

    replica = tmp_path / "replica"
    shutil.copytree(tmp_path / "primary", replica)
    restarted = _store(tmp_path / "primary")
    other = _store(replica)
    first = restarted.recover()
    second = other.recover()

    assert first.retention_completed and first.pruned_entry_count == 4
    assert second == first
    assert restarted.recover().retention_completed is False
    assert restarted.chain_path.read_bytes() == other.chain_path.read_bytes()
    assert restarted.verify().valid
    assert _sentinel(0) not in restarted.chain_path.read_text(encoding="utf-8")


def test_interrupted_pruning_with_tampered_pending_entries_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 4)
    _interrupt_next_chain_write(monkeypatch)
    with pytest.raises(AuditContinuityError):
        store.prune_through(1, policy=POLICY)
    chain = AuditChain()
    for index in (50, 51, 52, 53):
        chain.append(_report(index))
    chain.write(store.chain_path)

    with pytest.raises(AuditContinuityError) as refused:
        store.recover()

    assert refused.value.code == "tombstone_mismatch"
    assert store.verify().code == "tombstone_mismatch"


def test_concurrent_writers_recover_an_interrupted_transition_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 5)
    _interrupt_next_chain_write(monkeypatch)
    with pytest.raises(AuditContinuityError):
        store.prune_through(2, policy=POLICY)

    errors: list[BaseException] = []

    def writer(offset: int) -> None:
        try:
            _append(_store(tmp_path), offset, 4)
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(100 * n,)) for n in range(1, 4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    store.anchor()
    report = store.verify()
    assert report.valid, report.code
    assert report.counts["pruned_entry_count"] == 3
    assert report.counts["next_sequence"] == 5 + 12


def test_rotation_concurrent_with_appends_has_one_ordered_transition(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 2)
    started = threading.Barrier(5)
    errors: list[BaseException] = []
    transitions: list[ContinuityEvent] = []

    def writer(offset: int) -> None:
        try:
            started.wait()
            _append(_store(tmp_path), offset, 6)
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    def rotator() -> None:
        try:
            started.wait()
            transitions.append(_store(tmp_path).rotate("audit-epoch-1"))
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(100 * n,)) for n in range(1, 4)]
    threads += [threading.Thread(target=rotator) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(transitions) == 2 and transitions[0] == transitions[1]
    store.anchor()
    chain, manifest = store.load()
    rotations = [event for event in manifest.events if event.kind == "rotation"]
    assert len(rotations) == 1
    boundary = rotations[0].next_sequence
    assert 2 <= boundary <= 20
    assert rotations[0].head_hash == chain.entry_at(boundary - 1).entry_hash
    epochs = manifest.epochs
    assert [(epoch.start_sequence, epoch.end_sequence) for epoch in epochs] == [
        (0, boundary),
        (boundary, None),
    ]
    report = store.verify()
    assert report.valid, report.code
    assert report.counts["next_sequence"] == 20
    assert report.counts["unanchored_entry_count"] == 0


_CHILD_WRITER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from openmed.core.audit import AuditReport, hash_text
    from openmed.core.audit_continuity import AuditContinuityStore

    keys = {
        f"audit-epoch-{index}": (
            f"synthetic-continuity-key-{index:02d}-0123456789abcdef"
        ).encode("ascii")
        for index in range(12)
    }
    store = AuditContinuityStore(Path(sys.argv[1]), key_provider=keys)
    worker = sys.argv[2]
    empty = hash_text("")
    for index in range(int(sys.argv[3])):
        store.append(
            AuditReport(
                policy="synthetic_policy",
                resolved_profile={"method": "mask"},
                detectors=[],
                safety_sweep={},
                spans=[],
                thresholds={},
                residual_risk={},
                openmed_version="synthetic",
                manifest_hash=empty,
                document_length=0,
                input_hash=hash_text(f"synthetic-{worker}-{index}"),
                deidentified_text_hash=empty,
            )
        )
    """
)


def test_rotation_is_serialized_with_appends_from_other_processes(
    tmp_path: Path,
) -> None:
    store = _initialized(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CHILD_WRITER,
                str(store.chain_path),
                f"worker{worker}",
                "8",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for worker in range(2)
    ]
    # Rotate once the children are writing so the rollover interleaves with
    # their appends; the outcome must not depend on that timing.
    while all(child.poll() is None for child in children):
        if store.load()[0].next_sequence >= 2:
            break
        time.sleep(0.005)
    store.rotate("audit-epoch-1")
    store.rotate("audit-epoch-2")
    for child in children:
        _, stderr = child.communicate(timeout=120)
        assert child.returncode == 0, stderr.decode("utf-8", "replace")[-500:]

    store.anchor()
    report = store.verify()
    assert report.valid, report.code
    assert report.counts["next_sequence"] == 16
    assert report.counts["rotation_count"] == 2
    windows = [(epoch.start_sequence, epoch.end_sequence) for epoch in report.epochs]
    assert windows[0][0] == 0 and windows[-1][1] is None
    for previous, current in zip(windows, windows[1:], strict=False):
        assert previous[1] == current[0]


def test_failed_writes_leave_no_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 2)

    def failing_replace(source: Any, target: Any) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(AuditContinuityError) as error:
        store.anchor()
    monkeypatch.undo()

    assert error.value.code == "io_failed"
    assert (
        sorted(path.name for path in tmp_path.iterdir() if path.name.endswith(".tmp"))
        == []
    )
    stale = tmp_path / ".audit-chain.json.leftover.tmp"
    stale.write_text("{}", encoding="utf-8")
    assert store.recover().temporary_files_removed == 1
    assert store.verify(allow_unanchored_tail=True).valid


# Privacy and offline behavior ---------------------------------------------------


def test_pruned_sentinels_are_absent_from_retained_state_and_evidence(
    anchored: AuditContinuityStore,
) -> None:
    report = anchored.verify()
    evidence = report.to_json(indent=2)
    retained = "".join(
        path.read_text(encoding="utf-8")
        for path in anchored.chain_path.parent.iterdir()
        if path.is_file()
    )

    for index in range(10):
        for value in (_sentinel(index), hash_text(_sentinel(index))):
            assert value not in retained
            assert value not in evidence
    # Retained entries keep their own value-free commitments.
    assert hash_text(_sentinel(10)) in retained
    for material in KEYS.values():
        assert material.decode("ascii") not in retained
        assert material.decode("ascii") not in evidence


def test_report_is_bounded_deterministic_and_value_free(
    anchored: AuditContinuityStore,
) -> None:
    first = anchored.verify()
    second = anchored.verify()
    payload = first.to_dict()

    assert first.to_json() == second.to_json()
    assert payload["format"] == "openmed.audit-continuity.verification"
    assert payload["version"] == 1
    assert payload["report_digest"] == first.report_digest
    assert set(payload["counts"]) == set(first.counts)
    assert all(type(value) is int and value >= 0 for value in first.counts.values())
    assert set(payload) == {
        "format",
        "version",
        "manifest_version",
        "valid",
        "code",
        "event_index",
        "chain_id",
        "manifest_digest",
        "chain_head_hash",
        "counts",
        "epochs",
        "policy_fingerprints",
        "report_digest",
    }


def test_errors_never_echo_identifiers_or_paths(tmp_path: Path) -> None:
    store = _initialized(tmp_path / _sentinel(1).replace(" ", "-"))
    with pytest.raises(AuditContinuityError) as error:
        store.rotate(_sentinel(2))
    message = str(error.value)
    assert _sentinel(1).replace(" ", "-") not in message
    assert _sentinel(2) not in message
    assert error.value.code == "invalid_argument"


def test_manifest_round_trips_and_rejects_duplicate_keys(
    anchored: AuditContinuityStore,
) -> None:
    _, manifest = anchored.load()
    text = manifest.to_json()
    assert ContinuityManifest.from_json(text).to_json() == text
    duplicated = text[:-1] + ',"chain_id":"chain-other"}'
    with pytest.raises(AuditContinuityError) as error:
        ContinuityManifest.from_json(duplicated)
    assert error.value.code == "malformed_manifest"


def test_verification_is_fully_offline(
    anchored: AuditContinuityStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, manifest = anchored.load()

    def deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden during verification")

    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)

    assert verify_continuity(chain, manifest, key_provider=KEYS).valid


def test_segment_digest_helpers_check_a_retained_backup_copy(tmp_path: Path) -> None:
    store = _initialized(tmp_path)
    _append(store, 0, 4)
    chain_before, _ = store.load()
    segment = chain_before.entries[:2]
    event = store.prune_through(1, policy=POLICY)
    tombstone = event.tombstone
    assert tombstone is not None

    assert tombstone.segment_digest == tombstone_segment_digest(
        "chain-synthetic", segment
    )
    assert verify_tombstone_segment(tombstone, segment, chain_id="chain-synthetic")
    assert not verify_tombstone_segment(
        tombstone, chain_before.entries[1:3], chain_id="chain-synthetic"
    )
    assert not verify_tombstone_segment(tombstone, segment, chain_id="chain-other")


# Property: random lifecycles verify and any event edit is detected -------------

_OPERATIONS = st.lists(
    st.sampled_from(["append", "append", "append", "anchor", "rotate", "prune"]),
    min_size=1,
    max_size=14,
)


@settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(operations=_OPERATIONS, edit=st.integers(min_value=0, max_value=10_000))
def test_random_lifecycles_verify_and_edits_are_detected(
    operations: list[str],
    edit: int,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = _initialized(Path(directory))
        next_id = 0
        rotations = 0
        for operation in operations:
            if operation == "append":
                store.append(_report(next_id))
                next_id += 1
            elif operation == "anchor":
                store.anchor()
            elif operation == "rotate" and rotations < 10:
                rotations += 1
                store.rotate(f"audit-epoch-{rotations}")
            elif operation == "prune":
                chain, _ = store.load()
                if chain.entries:
                    store.prune_through(chain.base_sequence, policy=POLICY)
        store.anchor()
        report = store.verify()
        assert report.valid, report.code

        chain_doc, manifest = _documents(store)
        position = edit % len(manifest["events"])
        manifest["events"][position]["head_hash"] = "sha256:" + "6" * 64
        tampered = _verify(chain_doc, manifest)
        assert not tampered.valid
        assert tampered.code in CONTINUITY_REASON_CODES
