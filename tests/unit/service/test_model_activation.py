"""Unit tests for transactional model activation (synthetic bundles, offline)."""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.core.model_integrity import write_local_artifact_manifest
from openmed.service.model_activation import (
    ActivationEvent,
    ActivationOutcome,
    ActivationPolicy,
    ActivationReason,
    GenerationContext,
    HubArtifactSource,
    LocalBundleSource,
    ModelActivationError,
    ModelActivationManager,
    ModelReference,
    StagedArtifact,
    WarmPoolGenerationLoader,
    verify_receipt_chain,
)
from openmed.service.warm_pool import WarmPool

MODEL_ID = "openmed-test/synthetic-activation-model"
REPRO = "sha256:" + "ab" * 32


class FakeClock:
    """Deterministic aware UTC clock."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def write_bundle(
    directory: Path,
    payload: bytes = b"weights-v1",
    *,
    model_id: str = MODEL_ID,
) -> Path:
    """Write a tiny synthetic model bundle with a relocatable manifest."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type": "synthetic"}\n')
    (directory / "model.safetensors").write_bytes(payload)
    write_local_artifact_manifest(
        directory, model_id=model_id, reproducibility_hash=REPRO
    )
    return directory


class FakeSource:
    """Network-free source that writes a synthetic bundle into the workspace."""

    def __init__(
        self,
        payload: bytes = b"weights-v1",
        *,
        model_id: str = MODEL_ID,
        mutate: Any = None,
        error: BaseException | None = None,
        gate: threading.Event | None = None,
        entered: threading.Event | None = None,
    ) -> None:
        self.payload = payload
        self.model_id = model_id
        self.mutate = mutate
        self.error = error
        self.gate = gate
        self.entered = entered
        self.calls = 0

    def acquire(self, reference: ModelReference, workspace: Path) -> Path:
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.error is not None:
            raise self.error
        bundle = write_bundle(
            workspace / "bundle", self.payload, model_id=self.model_id
        )
        if self.mutate is not None:
            self.mutate(bundle, workspace)
        return bundle


class FakeHandle:
    def __init__(self, context: GenerationContext, payload: bytes) -> None:
        self.generation = context.generation
        self.artifact_digest = context.artifact_digest
        self.payload = payload
        self.warmed = False


class FakeLoader:
    """Loader double that reads the generation payload and records calls."""

    def __init__(self, *, fail_load: bool = False, fail_warm: bool = False) -> None:
        self.fail_load = fail_load
        self.fail_warm = fail_warm
        self.loaded: list[int] = []
        self.unloaded: list[int] = []
        self.lock = threading.Lock()

    def load(self, context: GenerationContext) -> FakeHandle:
        if self.fail_load:
            raise RuntimeError("synthetic load failure")
        payload = (context.artifact_dir / "model.safetensors").read_bytes()
        with self.lock:
            self.loaded.append(context.generation)
        return FakeHandle(context, payload)

    def warm(self, handle: FakeHandle, context: GenerationContext) -> None:
        if self.fail_warm:
            raise RuntimeError("synthetic warm failure")
        handle.warmed = True

    def unload(self, handle: FakeHandle, context: GenerationContext) -> None:
        with self.lock:
            self.unloaded.append(context.generation)


def checkpoint(version: str, parent: str | None = None) -> dict[str, Any]:
    """Synthetic registry checkpoint metadata that is rollback-compatible."""

    return {
        "model_id": f"synthetic/checkpoint-{version}",
        "family": "pii",
        "version": version,
        "semver_constraint": ">=1.0.0",
        "lineage": [] if parent is None else [f"synthetic/checkpoint-{parent}"],
        "policy_fingerprint": "policy-v1",
        "tokenizer_ids": ["tokenizer-v1"],
        "evidence_schema_versions": ["1"],
    }


def reference(revision: str = "r1", **pins: Any) -> ModelReference:
    return ModelReference(MODEL_ID, revision=revision, **pins)


def make_manager(
    root: Path,
    *,
    loader: FakeLoader | None = None,
    policy: ActivationPolicy | None = None,
    clock: FakeClock | None = None,
    **kwargs: Any,
) -> ModelActivationManager:
    manager = ModelActivationManager(
        root,
        loader=loader or FakeLoader(),
        policy=policy or ActivationPolicy(),
        clock=clock or FakeClock(),
        **kwargs,
    )
    manager.start()
    return manager


def activate_v1(manager: ModelActivationManager, slot: str = "pii-default"):
    return manager.activate(
        slot,
        reference("r1"),
        source=FakeSource(b"weights-v1"),
        compatibility=checkpoint("1.0.0"),
    )


def activate_v2(manager: ModelActivationManager, slot: str = "pii-default", **kw):
    return manager.activate(
        slot,
        reference("r2"),
        source=kw.pop("source", FakeSource(b"weights-v2")),
        compatibility=kw.pop("compatibility", checkpoint("1.1.0", parent="1.0.0")),
        **kw,
    )


def generation_dirs(root: Path) -> list[str]:
    base = root / "generations"
    if not base.is_dir():
        return []
    return sorted(
        f"{slot.name}/{gen.name}" for slot in base.iterdir() for gen in slot.iterdir()
    )


def require_symlinks(tmp_path: Path) -> None:
    """Skip when the platform or account cannot create symbolic links."""

    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable on this platform")
    probe.unlink()


# ---------------------------------------------------------------------------
# Happy path, idempotency, generations
# ---------------------------------------------------------------------------


def test_first_activation_commits_generation_one(tmp_path: Path) -> None:
    loader = FakeLoader()
    manager = make_manager(tmp_path, loader=loader)

    result = activate_v1(manager)

    assert result.outcome is ActivationOutcome.ACTIVATED
    assert result.ok
    assert result.transaction_id and result.transaction_id.startswith("txn-")
    assert (result.active_generation, result.last_green_generation) == (1, 1)
    assert result.candidate_generation == 1
    assert result.model_ref_digest == reference("r1").digest
    assert result.artifact_digest and result.artifact_digest.startswith("sha256:")
    assert loader.loaded == [1]
    with manager.lease("pii-default") as lease:
        assert lease.generation == 1
        assert lease.handle.payload == b"weights-v1"
        assert lease.handle.warmed
    assert list((tmp_path / "staging").iterdir()) == []
    status = manager.status("pii-default")
    assert status["active_generation"] == 1
    assert status["last_green_generation"] == 1
    assert status["pending_transaction"] is None
    assert manager.verify_receipts()
    manager.close()


def test_new_generation_supersedes_and_retires_previous(tmp_path: Path) -> None:
    loader = FakeLoader()
    manager = make_manager(tmp_path, loader=loader)
    activate_v1(manager)

    result = activate_v2(manager)

    assert result.outcome is ActivationOutcome.ACTIVATED
    assert (result.active_generation, result.last_green_generation) == (2, 2)
    assert loader.unloaded == [1]
    assert len(generation_dirs(tmp_path)) == 1
    with manager.lease("pii-default") as lease:
        assert (lease.generation, lease.handle.payload) == (2, b"weights-v2")
    events = [row["event"] for row in manager.receipts()]
    assert events == ["activated", "activated"]
    assert manager.receipts()[-1]["from_generation"] == 1
    assert manager.receipts()[-1]["compatibility_decision"] == "compatible"
    manager.close()


def test_repeating_a_committed_reference_is_idempotent(tmp_path: Path) -> None:
    loader = FakeLoader()
    manager = make_manager(tmp_path, loader=loader)
    first = activate_v1(manager)
    source = FakeSource(b"weights-v1")

    again = manager.activate("pii-default", reference("r1"), source=source)

    assert again.outcome is ActivationOutcome.ALREADY_ACTIVE
    assert again.ok and again.replayed
    assert again.transaction_id == first.transaction_id
    assert again.active_generation == 1
    assert source.calls == 0
    assert loader.loaded == [1]
    manager.close()


def test_idempotency_key_replays_and_detects_conflicts(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    first = manager.activate(
        "pii-default",
        reference("r1"),
        source=FakeSource(),
        compatibility=checkpoint("1.0.0"),
        idempotency_key="deploy-1",
    )
    replay = manager.activate(
        "pii-default",
        reference("r1"),
        source=FakeSource(),
        idempotency_key="deploy-1",
    )
    conflict = manager.activate(
        "pii-default",
        reference("r9"),
        source=FakeSource(),
        idempotency_key="deploy-1",
    )

    assert first.outcome is ActivationOutcome.ACTIVATED
    assert replay.transaction_id == first.transaction_id
    assert replay.outcome is ActivationOutcome.ACTIVATED and replay.replayed
    assert conflict.outcome is ActivationOutcome.CONFLICT
    assert conflict.reason is ActivationReason.TRANSACTION_CONFLICT
    manager.close()


def test_pinned_digest_must_match_and_repeat_honours_pin(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    first = activate_v1(manager)

    same = manager.activate(
        "pii-default",
        reference("r1", expected_artifact_digest=first.artifact_digest),
        source=FakeSource(),
    )
    assert same.outcome is ActivationOutcome.ALREADY_ACTIVE

    wrong = manager.activate(
        "pii-default",
        reference("r1", expected_artifact_digest="sha256:" + "0" * 64),
        source=FakeSource(),
        compatibility=checkpoint("1.0.0"),
    )
    assert wrong.outcome is ActivationOutcome.QUARANTINED
    assert wrong.reason is ActivationReason.PIN_MISMATCH
    assert wrong.active_generation == 1
    manager.close()


# ---------------------------------------------------------------------------
# Verification failures quarantine and are never loaded
# ---------------------------------------------------------------------------


def _tamper_digest(bundle: Path, workspace: Path) -> None:
    (bundle / "model.safetensors").write_bytes(b"tampered-after-manifest")


def _drop_manifest(bundle: Path, workspace: Path) -> None:
    (bundle / ".openmed-integrity.json").unlink()


def _corrupt_manifest(bundle: Path, workspace: Path) -> None:
    (bundle / ".openmed-integrity.json").write_text("{not json")


def _external_root(bundle: Path, workspace: Path) -> None:
    sidecar = bundle / ".openmed-integrity.json"
    payload = json.loads(sidecar.read_text())
    payload["artifact_root"] = str(workspace.parent.parent)
    sidecar.write_text(json.dumps(payload))


def _unlisted_file(bundle: Path, workspace: Path) -> None:
    (bundle / "extra.bin").write_bytes(b"not in the manifest")


def _escaping_link(bundle: Path, workspace: Path) -> None:
    outside = workspace.parent.parent.parent / "outside-secret.txt"
    outside.write_text("outside the workspace")
    (bundle / "tokenizer.json").symlink_to(outside)


def _missing_listed_file(bundle: Path, workspace: Path) -> None:
    (bundle / "config.json").unlink()


@pytest.mark.parametrize(
    ("mutate", "source_kwargs", "expected"),
    [
        (_tamper_digest, {}, ActivationReason.DIGEST_MISMATCH),
        (_drop_manifest, {}, ActivationReason.MANIFEST_MISSING),
        (_corrupt_manifest, {}, ActivationReason.MANIFEST_INVALID),
        (_external_root, {}, ActivationReason.MANIFEST_INVALID),
        (None, {"model_id": "other/model"}, ActivationReason.MANIFEST_MISMATCH),
        (_unlisted_file, {}, ActivationReason.ARTIFACT_UNLISTED),
        (_escaping_link, {}, ActivationReason.ARTIFACT_LINK),
        (_missing_listed_file, {}, ActivationReason.ARTIFACT_MISSING),
    ],
)
def test_invalid_artifacts_are_quarantined_and_never_loaded(
    tmp_path: Path, mutate: Any, source_kwargs: dict[str, Any], expected: Any
) -> None:
    if mutate is _escaping_link:
        require_symlinks(tmp_path)
    root = tmp_path / "root"
    loader = FakeLoader()
    manager = make_manager(root, loader=loader)
    activate_v1(manager)

    result = activate_v2(
        manager, source=FakeSource(b"weights-v2", mutate=mutate, **source_kwargs)
    )

    assert result.outcome is ActivationOutcome.QUARANTINED
    assert result.reason is expected
    assert not result.ok
    assert (result.active_generation, result.last_green_generation) == (1, 1)
    assert loader.loaded == [1]
    assert list((root / "staging").iterdir()) == []
    quarantined = list((root / "quarantine").iterdir())
    assert len(quarantined) == 1
    metadata = json.loads((quarantined[0] / "quarantine.json").read_text())
    assert metadata["reason"] == expected.value
    with manager.lease("pii-default") as lease:
        assert lease.handle.payload == b"weights-v1"
    manager.close()


def test_size_bounds_and_policy_validators_quarantine(tmp_path: Path) -> None:
    seen: list[StagedArtifact] = []

    def reject(staged: StagedArtifact) -> bool:
        seen.append(staged)
        return False

    manager = make_manager(
        tmp_path / "a", policy=ActivationPolicy(max_artifact_bytes=8)
    )
    result = activate_v1(manager)
    assert result.reason is ActivationReason.ARTIFACT_TOO_LARGE
    manager.close()

    manager = make_manager(
        tmp_path / "b", policy=ActivationPolicy(validators=(reject,))
    )
    result = activate_v1(manager)
    assert result.outcome is ActivationOutcome.QUARANTINED
    assert result.reason is ActivationReason.POLICY_REJECTED
    assert seen and seen[0].files == ("config.json", "model.safetensors")
    assert "artifact_dir" not in repr(seen[0])
    manager.close()


def test_incompatible_rollback_target_is_refused_unless_waived(tmp_path: Path) -> None:
    manager = make_manager(tmp_path / "strict")
    activate_v1(manager)
    unrelated = checkpoint("1.1.0")  # lineage does not reach the last green
    result = activate_v2(manager, compatibility=unrelated)
    assert result.outcome is ActivationOutcome.QUARANTINED
    assert result.reason is ActivationReason.ROLLBACK_INCOMPATIBLE
    assert result.active_generation == 1
    missing = activate_v2(manager, compatibility=None)
    assert missing.reason is ActivationReason.ROLLBACK_INCOMPATIBLE
    manager.close()

    waived = make_manager(
        tmp_path / "waived",
        policy=ActivationPolicy(require_rollback_compatibility=False),
    )
    activate_v1(waived)
    result = activate_v2(waived, compatibility=unrelated)
    assert result.outcome is ActivationOutcome.ACTIVATED
    assert waived.receipts()[-1]["compatibility_decision"] == "waived"
    waived.close()


def test_invalid_compatibility_metadata_is_refused(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    result = manager.activate(
        "pii-default",
        reference(),
        source=FakeSource(),
        compatibility={"model_id": object()},
    )
    assert result.outcome is ActivationOutcome.REFUSED
    assert result.reason is ActivationReason.COMPATIBILITY_METADATA_INVALID
    manager.close()


# ---------------------------------------------------------------------------
# Load, warm, commit, and disk failures roll back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader_kwargs", "expected"),
    [
        ({"fail_load": True}, ActivationReason.LOAD_FAILED),
        ({"fail_warm": True}, ActivationReason.WARM_FAILED),
    ],
)
def test_failed_load_or_warm_preserves_active(
    tmp_path: Path, loader_kwargs: dict[str, bool], expected: Any
) -> None:
    loader = FakeLoader()
    manager = make_manager(tmp_path, loader=loader)
    activate_v1(manager)
    for name, value in loader_kwargs.items():
        setattr(loader, name, value)

    result = activate_v2(manager)

    assert result.outcome is ActivationOutcome.ROLLED_BACK
    assert result.reason is expected
    assert result.active_generation == 1
    assert len(generation_dirs(tmp_path)) == 1
    if expected is ActivationReason.WARM_FAILED:
        assert loader.unloaded == [2]
    with manager.lease("pii-default") as lease:
        assert lease.generation == 1
    manager.close()


class ArmedFault:
    """Raise ``error`` once at ``target`` after ``trigger`` while enabled."""

    def __init__(self, trigger: str | None, target: str, error: BaseException):
        self.trigger = trigger
        self.target = target
        self.error = error
        self.enabled = False
        self.armed = trigger is None

    def __call__(self, point: str) -> None:
        if not self.enabled:
            return
        if point == self.trigger:
            self.armed = True
        if self.armed and point == self.target:
            self.enabled = False
            raise self.error


@pytest.mark.parametrize("target", ["state.write", "commit.published"])
def test_failed_pointer_commit_restores_previous_generation(
    tmp_path: Path, target: str
) -> None:
    hook = ArmedFault("commit.before", target, OSError(errno.ENOSPC, "full"))
    manager = make_manager(tmp_path, fault_hook=hook)
    activate_v1(manager)
    hook.enabled = True

    result = activate_v2(manager)

    assert result.outcome is ActivationOutcome.ROLLED_BACK
    assert result.reason is ActivationReason.DISK_EXHAUSTED
    assert (result.active_generation, result.last_green_generation) == (1, 1)
    assert manager.status("pii-default")["pending_transaction"] is None
    with manager.lease("pii-default") as lease:
        assert lease.generation == 1
    assert len(generation_dirs(tmp_path)) == 1
    manager.close()


def test_disk_exhaustion_during_acquire_quarantines(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    activate_v1(manager)
    result = activate_v2(
        manager, source=FakeSource(error=OSError(errno.ENOSPC, "no space"))
    )
    assert result.outcome is ActivationOutcome.QUARANTINED
    assert result.reason is ActivationReason.DISK_EXHAUSTED
    assert result.active_generation == 1
    manager.close()


def test_first_journal_write_failure_refuses_without_side_effects(
    tmp_path: Path,
) -> None:
    hook = ArmedFault(None, "journal.write", OSError(errno.ENOSPC, "full"))
    manager = make_manager(tmp_path, fault_hook=hook)
    hook.enabled = True
    result = activate_v1(manager)
    assert result.outcome is ActivationOutcome.REFUSED
    assert result.reason is ActivationReason.DISK_EXHAUSTED
    assert list((tmp_path / "journal").glob("txn-*.json")) == []
    assert list((tmp_path / "staging").iterdir()) == []
    manager.close()


def test_source_returning_outside_workspace_is_quarantined(tmp_path: Path) -> None:
    outside = write_bundle(tmp_path / "elsewhere")

    class EscapingSource:
        def acquire(self, ref: ModelReference, workspace: Path) -> Path:
            return outside

    manager = make_manager(tmp_path / "root")
    result = manager.activate("pii-default", reference(), source=EscapingSource())
    assert result.reason is ActivationReason.SOURCE_OUTSIDE_WORKSPACE
    assert (outside / "model.safetensors").exists()
    manager.close()


# ---------------------------------------------------------------------------
# Health window, last green, rollback
# ---------------------------------------------------------------------------


def test_failed_health_window_rolls_back_to_last_green(tmp_path: Path) -> None:
    clock = FakeClock()
    loader = FakeLoader()
    manager = make_manager(
        tmp_path,
        loader=loader,
        clock=clock,
        policy=ActivationPolicy(health_window_seconds=60),
    )
    first = activate_v1(manager)
    assert first.outcome is ActivationOutcome.PENDING_HEALTH
    clock.advance(61)
    assert manager.poll_health("pii-default").outcome is ActivationOutcome.ACTIVATED

    pending = activate_v2(manager)
    assert pending.outcome is ActivationOutcome.PENDING_HEALTH
    assert (pending.active_generation, pending.last_green_generation) == (2, 1)
    assert pending.health_deadline is not None
    status = manager.status("pii-default")
    assert status["resident_generations"] == [1, 2]
    with manager.lease("pii-default") as lease:
        assert lease.generation == 2

    rolled = manager.record_health("pii-default", healthy=False)

    assert rolled.outcome is ActivationOutcome.ROLLED_BACK
    assert rolled.reason is ActivationReason.HEALTH_WINDOW_FAILED
    assert (rolled.active_generation, rolled.last_green_generation) == (1, 1)
    with manager.lease("pii-default") as lease:
        assert (lease.generation, lease.handle.payload) == (1, b"weights-v1")
    assert loader.unloaded == [2]
    assert len(generation_dirs(tmp_path)) == 1
    assert manager.receipts()[-1]["event"] == "rolled_back"
    assert manager.receipts()[-1]["to_generation"] == 1
    assert manager.verify_receipts()
    manager.close()


def test_passing_health_window_promotes_and_retires(tmp_path: Path) -> None:
    clock = FakeClock()
    loader = FakeLoader()
    policy = ActivationPolicy(health_window_seconds=30, min_healthy_reports=2)
    manager = make_manager(tmp_path, loader=loader, clock=clock, policy=policy)
    activate_v1(manager)
    clock.advance(31)
    manager.record_health("pii-default", healthy=True)
    manager.record_health("pii-default", healthy=True)
    assert manager.poll_health("pii-default").outcome is ActivationOutcome.ACTIVATED

    activate_v2(manager)
    assert manager.poll_health("pii-default").outcome is (
        ActivationOutcome.PENDING_HEALTH
    )
    assert manager.record_health("pii-default", healthy=True).ok
    assert manager.record_health("pii-default", healthy=True).ok
    clock.advance(31)
    result = manager.poll_health("pii-default")

    assert result.outcome is ActivationOutcome.ACTIVATED
    assert (result.active_generation, result.last_green_generation) == (2, 2)
    assert loader.unloaded == [1]
    assert manager.status("pii-default")["resident_generations"] == [2]
    manager.close()


def test_unproven_health_window_rolls_back(tmp_path: Path) -> None:
    clock = FakeClock()
    manager = make_manager(
        tmp_path,
        clock=clock,
        policy=ActivationPolicy(
            health_window_seconds=10,
            min_healthy_reports=1,
            require_rollback_compatibility=False,
        ),
    )
    activate_v1(manager)
    clock.advance(11)
    manager.record_health("pii-default", healthy=True)
    manager.poll_health("pii-default")
    activate_v2(manager)
    clock.advance(11)

    result = manager.poll_health("pii-default")

    assert result.outcome is ActivationOutcome.ROLLED_BACK
    assert result.reason is ActivationReason.HEALTH_WINDOW_UNPROVEN
    assert result.active_generation == 1
    manager.close()


def test_first_activation_health_failure_leaves_slot_empty(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, policy=ActivationPolicy(health_window_seconds=5))
    activate_v1(manager)
    result = manager.record_health("pii-default", healthy=False)
    assert result.outcome is ActivationOutcome.ROLLED_BACK
    assert result.active_generation is None
    with pytest.raises(ModelActivationError) as caught:
        manager.lease("pii-default")
    assert caught.value.reason is ActivationReason.NO_ACTIVE_GENERATION
    assert generation_dirs(tmp_path) == []
    manager.close()


def test_explicit_rollback_and_unavailable_rollback(tmp_path: Path) -> None:
    clock = FakeClock()
    manager = make_manager(
        tmp_path, clock=clock, policy=ActivationPolicy(health_window_seconds=5)
    )
    activate_v1(manager)
    clock.advance(6)
    manager.poll_health("pii-default")

    unavailable = manager.rollback("pii-default")
    assert unavailable.outcome is ActivationOutcome.REFUSED
    assert unavailable.reason is ActivationReason.ROLLBACK_UNAVAILABLE

    activate_v2(manager)
    result = manager.rollback("pii-default")
    assert result.outcome is ActivationOutcome.ROLLED_BACK
    assert result.reason is ActivationReason.ROLLBACK_REQUESTED
    assert result.active_generation == 1
    assert manager.record_health("pii-default", healthy=True).reason is (
        ActivationReason.NO_HEALTH_WINDOW
    )
    manager.close()


def test_tampered_last_green_blocks_rollback_without_deleting(tmp_path: Path) -> None:
    clock = FakeClock()
    manager = make_manager(
        tmp_path, clock=clock, policy=ActivationPolicy(health_window_seconds=5)
    )
    activate_v1(manager)
    clock.advance(6)
    manager.poll_health("pii-default")
    activate_v2(manager)
    gen1 = next((tmp_path / "generations").iterdir()) / "00000001"
    (gen1 / "artifact" / "model.safetensors").write_bytes(b"tampered-on-disk")

    result = manager.record_health("pii-default", healthy=False)

    assert result.outcome is ActivationOutcome.REFUSED
    assert result.reason is ActivationReason.ROLLBACK_TARGET_UNVERIFIED
    assert (result.active_generation, result.last_green_generation) == (2, 1)
    assert gen1.is_dir()
    with manager.lease("pii-default") as lease:
        assert lease.generation == 2
    clock.advance(6)
    retried = manager.poll_health("pii-default")
    assert retried.reason is ActivationReason.ROLLBACK_TARGET_UNVERIFIED
    assert retried.last_green_generation == 1
    assert gen1.is_dir()
    manager.close()

    restarted = ModelActivationManager(
        tmp_path, loader=FakeLoader(), clock=clock, policy=manager.policy
    )
    results = restarted.start()
    assert [result.reason for result in results] == [
        ActivationReason.ROLLBACK_TARGET_UNVERIFIED
    ]
    assert gen1.is_dir()
    assert restarted.status("pii-default")["serving_generation"] == 2
    restarted.close()


def test_open_health_window_conflicts_with_other_references(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, policy=ActivationPolicy(health_window_seconds=5))
    pending = activate_v1(manager)

    same = manager.activate("pii-default", reference("r1"), source=FakeSource())
    other = activate_v2(manager)

    assert same.outcome is ActivationOutcome.PENDING_HEALTH
    assert same.transaction_id == pending.transaction_id and same.replayed
    assert other.outcome is ActivationOutcome.CONFLICT
    assert other.reason is ActivationReason.SLOT_CONFLICT
    manager.close()


def test_last_green_can_be_unloaded_during_window_and_reloaded(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    loader = FakeLoader()
    policy = ActivationPolicy(health_window_seconds=5, keep_last_green_resident=False)
    manager = make_manager(tmp_path, loader=loader, clock=clock, policy=policy)
    activate_v1(manager)
    clock.advance(6)
    manager.poll_health("pii-default")
    activate_v2(manager)
    assert manager.status("pii-default")["resident_generations"] == [2]

    result = manager.record_health("pii-default", healthy=False)

    assert result.active_generation == 1
    assert loader.loaded == [1, 2, 1]
    manager.close()


# ---------------------------------------------------------------------------
# Generation leases and the warm pool
# ---------------------------------------------------------------------------


def test_in_flight_lease_finishes_on_its_generation(tmp_path: Path) -> None:
    loader = FakeLoader()
    manager = make_manager(tmp_path, loader=loader)
    activate_v1(manager)
    old = manager.lease("pii-default")

    activate_v2(manager)

    with manager.lease("pii-default") as new:
        assert new.generation == 2
    assert old.generation == 1
    assert old.handle.payload == b"weights-v1"
    assert loader.unloaded == []
    assert len(generation_dirs(tmp_path)) == 2
    old.release()
    old.release()
    assert loader.unloaded == [1]
    assert len(generation_dirs(tmp_path)) == 1
    manager.close()


class _PoolLoader:
    """Minimal ModelLoader double for the real WarmPool."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.unloaded: list[str] = []

    def create_pipeline(self, model_name: str, **kwargs: Any) -> object:
        self.created.append(model_name)
        return {"pipeline_for": Path(model_name).parent.name}

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        self.unloaded.append(model_name)
        return {"model_name": model_name, "models": 0, "tokenizers": 0, "pipelines": 1}


def test_warm_pool_loader_keys_handles_per_generation(tmp_path: Path) -> None:
    backing = _PoolLoader()
    pool = WarmPool(lambda: backing)
    probes: list[Any] = []
    loader = WarmPoolGenerationLoader(pool, warm_probe=probes.append)
    manager = ModelActivationManager(tmp_path, loader=loader, clock=FakeClock())
    manager.start()
    activate_v1(manager)
    in_flight = manager.lease("pii-default")
    pool_key = backing.created[0]
    pool.begin_request(pool_key)

    activate_v2(manager)

    assert len(backing.created) == 2 and backing.created[0] != backing.created[1]
    assert len(probes) == 2
    with manager.lease("pii-default") as lease:
        assert lease.handle == {"pipeline_for": "00000002"}
    assert in_flight.handle == {"pipeline_for": "00000001"}
    assert backing.unloaded == []
    pool.finish_request(pool_key, keep_alive=None)
    in_flight.release()
    assert backing.unloaded == [pool_key]
    manager.close()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_activation_of_one_slot_is_busy(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    gate = threading.Event()
    entered = threading.Event()
    results: dict[str, Any] = {}

    def run() -> None:
        results["first"] = manager.activate(
            "pii-default",
            reference("r1"),
            source=FakeSource(gate=gate, entered=entered),
            compatibility=checkpoint("1.0.0"),
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(10)
    busy = activate_v2(manager)
    health = manager.poll_health("pii-default")
    gate.set()
    worker.join(10)

    assert busy.outcome is ActivationOutcome.BUSY
    assert busy.reason is ActivationReason.SLOT_BUSY
    assert health.outcome is ActivationOutcome.BUSY
    assert results["first"].outcome is ActivationOutcome.ACTIVATED
    assert manager.status("pii-default")["active_generation"] == 1
    manager.close()


def test_different_slots_activate_concurrently(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    slots = [f"slot-{index}" for index in range(6)]
    results: dict[str, Any] = {}

    def run(slot: str) -> None:
        results[slot] = activate_v1(manager, slot)

    threads = [threading.Thread(target=run, args=(slot,)) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert {result.outcome for result in results.values()} == {
        ActivationOutcome.ACTIVATED
    }
    for slot in slots:
        assert manager.status(slot)["active_generation"] == 1
    assert manager.verify_receipts()
    assert len(manager.receipts()) == len(slots)
    manager.close()


def test_readers_never_observe_mixed_generations(tmp_path: Path) -> None:
    manager = make_manager(
        tmp_path, policy=ActivationPolicy(require_rollback_compatibility=False)
    )
    manager.activate("pii-default", reference("g1"), source=FakeSource(b"weights-1"))
    stop = threading.Event()
    problems: list[str] = []
    observed: set[int] = set()

    def reader() -> None:
        while not stop.wait(0.0005):
            with manager.lease("pii-default") as lease:
                handle = lease.handle
                observed.add(lease.generation)
                if handle.generation != lease.generation:
                    problems.append("generation")
                if handle.artifact_digest != lease.artifact_digest:
                    problems.append("digest")
                if handle.payload != f"weights-{lease.generation}".encode():
                    problems.append("payload")

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    for index in range(2, 8):
        result = manager.activate(
            "pii-default",
            reference(f"g{index}"),
            source=FakeSource(f"weights-{index}".encode()),
        )
        assert result.outcome is ActivationOutcome.ACTIVATED
    stop.set()
    for thread in threads:
        thread.join(10)

    assert problems == []
    assert observed and observed <= set(range(1, 8))
    assert manager.status("pii-default")["active_generation"] == 7
    with manager.lease("pii-default") as lease:
        assert lease.handle.payload == b"weights-7"
    manager.close()


def test_activation_root_is_owned_by_one_manager(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    other = ModelActivationManager(tmp_path, loader=FakeLoader())
    with pytest.raises(ModelActivationError) as caught:
        other.start()
    assert caught.value.reason is ActivationReason.ROOT_BUSY
    manager.close()
    assert other.start() == ()
    other.close()


# ---------------------------------------------------------------------------
# Durable artifacts, receipts, privacy
# ---------------------------------------------------------------------------


def test_receipt_chain_detects_tampering(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    activate_v1(manager)
    activate_v2(manager)
    manager.close()
    log = tmp_path / "receipts.jsonl"
    assert verify_receipt_chain(log)

    rows = log.read_text().splitlines()
    tampered = json.loads(rows[0])
    tampered["to_generation"] = 99
    log.write_text("\n".join([json.dumps(tampered), *rows[1:]]) + "\n")
    assert not verify_receipt_chain(log)


def test_torn_receipt_append_is_repaired_on_start(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    activate_v1(manager)
    manager.close()
    log = tmp_path / "receipts.jsonl"
    with log.open("a") as handle:
        handle.write('{"partial": ')

    manager = make_manager(tmp_path)
    activate_v2(manager)
    assert manager.verify_receipts()
    assert [row["sequence"] for row in manager.receipts()] == [1, 2]
    manager.close()


def test_lifecycle_evidence_is_value_free(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="openmed.service.model_activation")
    sentinels = {
        "SENTINEL-MODEL-zq7",
        "SENTINEL-REVISION-zq7",
        "SENTINEL-TOKEN-zq7",
        "sentinel-url.invalid",
        "SENTINEL-PAYLOAD-zq7",
    }
    root = tmp_path / "SENTINEL-PATH-zq7"
    events: list[ActivationEvent] = []
    clock = FakeClock()
    manager = make_manager(
        root,
        clock=clock,
        event_sink=events.append,
        policy=ActivationPolicy(health_window_seconds=5),
    )
    model_id = "openmed-test/SENTINEL-MODEL-zq7"
    good = ModelReference(model_id, revision="SENTINEL-REVISION-zq7")
    results = [
        manager.activate(
            "pii-default",
            good,
            source=FakeSource(b"SENTINEL-PAYLOAD-zq7", model_id=model_id),
            compatibility=checkpoint("1.0.0"),
        )
    ]
    error = RuntimeError(
        "download failed https://sentinel-url.invalid/x?token=SENTINEL-TOKEN-zq7"
    )
    clock.advance(6)
    results.append(manager.poll_health("pii-default"))
    results.append(
        manager.activate(
            "pii-default",
            ModelReference(model_id, revision="r2"),
            source=FakeSource(error=error),
        )
    )
    results.append(
        manager.activate(
            "pii-default",
            ModelReference(model_id, revision="r3"),
            source=FakeSource(b"SENTINEL-PAYLOAD-zq7", model_id=model_id),
            compatibility=checkpoint("1.1.0", parent="1.0.0"),
        )
    )
    results.append(manager.record_health("pii-default", healthy=False))
    manager.close()
    restarted = make_manager(root, event_sink=events.append)
    restarted.close()

    evidence = [json.dumps(result.to_dict()) for result in results]
    evidence += [json.dumps(event.to_dict()) for event in events]
    evidence += [repr(result) for result in results]
    evidence.append(caplog.text)
    for path in [
        root / "state.json",
        root / "receipts.jsonl",
        *(root / "journal").glob("*.json"),
        *(root / "quarantine").glob("*/quarantine.json"),
    ]:
        evidence.append(path.read_text())
    blob = "\n".join(evidence)
    for sentinel in sentinels:
        assert sentinel not in blob
    assert str(tmp_path) not in blob
    assert "SENTINEL-PATH" not in blob
    assert results[2].reason is ActivationReason.ACQUIRE_FAILED
    assert repr(good).count("SENTINEL") == 0


def test_activation_needs_no_network_beyond_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    clock = FakeClock()
    manager = make_manager(
        tmp_path, clock=clock, policy=ActivationPolicy(health_window_seconds=5)
    )
    activate_v1(manager)
    clock.advance(6)
    manager.poll_health("pii-default")
    activate_v2(manager)
    assert manager.record_health("pii-default", healthy=False).ok is False
    manager.close()
    assert make_manager(tmp_path).status("pii-default")["active_generation"] == 1


def test_local_bundle_source_copies_without_touching_the_bundle(
    tmp_path: Path,
) -> None:
    bundle = write_bundle(tmp_path / "bundle", b"air-gapped")
    before = sorted(path.name for path in bundle.iterdir())
    manager = make_manager(tmp_path / "root")

    result = manager.activate(
        "pii-default", reference(), source=LocalBundleSource(bundle)
    )

    assert result.outcome is ActivationOutcome.ACTIVATED
    assert sorted(path.name for path in bundle.iterdir()) == before
    assert "bundle" not in repr(LocalBundleSource(bundle))
    manager.close()


def test_hub_source_stages_a_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_symlinks(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_prefetch(name: str, **kwargs: Any) -> str:
        calls.append({"name": name, **kwargs})
        cache = Path(kwargs["cache_dir"])
        blobs = cache / "models--x" / "blobs"
        snapshot = cache / "models--x" / "snapshots" / ("c" * 40)
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        (blobs / "aaa").write_bytes(b"hub-weights")
        (blobs / "bbb").write_text('{"model_type": "synthetic"}')
        (snapshot / "model.safetensors").symlink_to(Path("../../blobs/aaa"))
        (snapshot / "config.json").symlink_to(Path("../../blobs/bbb"))
        return str(snapshot)

    monkeypatch.setattr("openmed.core.hf_hub.prefetch_model", fake_prefetch)
    loader = FakeLoader()
    manager = make_manager(tmp_path / "root", loader=loader)

    result = manager.activate(
        "pii-default", reference("main"), source=HubArtifactSource(retries=1)
    )

    assert result.outcome is ActivationOutcome.ACTIVATED
    assert calls[0]["name"] == MODEL_ID
    assert calls[0]["revision"] == "main"
    assert "staging" in calls[0]["cache_dir"]
    with manager.lease("pii-default") as lease:
        assert lease.handle.payload == b"hub-weights"
    generation = tmp_path / "root" / "generations"
    artifact = next(generation.iterdir()) / "00000001" / "artifact"
    assert not any(path.is_symlink() for path in artifact.rglob("*"))
    manager.close()


# ---------------------------------------------------------------------------
# Startup safety and argument validation
# ---------------------------------------------------------------------------


def test_corrupt_state_fails_closed_without_changes(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    activate_v1(manager)
    manager.close()
    state = tmp_path / "state.json"
    state.write_text("{corrupt")
    before = sorted(str(path) for path in tmp_path.rglob("*"))

    with pytest.raises(ModelActivationError) as caught:
        make_manager(tmp_path)

    assert caught.value.reason is ActivationReason.STATE_UNREADABLE
    assert str(caught.value) == "state_unreadable"
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before
    again = ModelActivationManager(tmp_path, loader=FakeLoader())
    with pytest.raises(ModelActivationError):
        again.start()


def test_corrupt_journal_is_quarantined_on_start(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    activate_v1(manager)
    manager.close()
    bogus = tmp_path / "journal" / ("txn-" + "f" * 32 + ".json")
    bogus.write_text('{"schema_version": "unknown"}')

    manager = make_manager(tmp_path)

    assert not bogus.exists()
    reasons = [
        json.loads(path.read_text())["reason"]
        for path in (tmp_path / "quarantine").glob("*/quarantine.json")
    ]
    assert reasons == ["journal_unreadable"]
    assert manager.status("pii-default")["active_generation"] == 1
    manager.close()


def test_start_refuses_to_serve_tampered_active_generation(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    activate_v1(manager)
    manager.close()
    gen = next((tmp_path / "generations").iterdir()) / "00000001"
    (gen / "artifact" / "model.safetensors").write_bytes(b"tampered")

    manager = ModelActivationManager(tmp_path, loader=FakeLoader())
    results = manager.start()

    assert [result.reason for result in results] == [ActivationReason.ACTIVE_UNVERIFIED]
    assert gen.is_dir()
    with pytest.raises(ModelActivationError):
        manager.lease("pii-default")
    manager.close()


def test_operations_require_start_and_valid_arguments(tmp_path: Path) -> None:
    manager = ModelActivationManager(tmp_path, loader=FakeLoader())
    with pytest.raises(ModelActivationError) as caught:
        manager.activate("pii-default", reference(), source=FakeSource())
    assert caught.value.reason is ActivationReason.NOT_STARTED
    manager.start()
    for slot in ("", "Has Space", "../escape", "a::b::c::d", "https://x"):
        with pytest.raises(ModelActivationError):
            manager.activate(slot, reference(), source=FakeSource())
    with pytest.raises(ModelActivationError):
        manager.activate("pii-default", "not-a-reference")  # type: ignore[arg-type]
    with pytest.raises(ModelActivationError):
        ModelReference("")
    with pytest.raises(ModelActivationError):
        ModelReference(MODEL_ID, expected_artifact_digest="md5:abc")
    assert manager.activate(
        "PII::None::PyTorch", reference(), source=FakeSource()
    ).slot == ("pii::none::pytorch")
    manager.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"health_window_seconds": -1},
        {"health_window_seconds": float("inf")},
        {"health_window_seconds": True},
        {"min_healthy_reports": -1},
        {"max_artifact_files": 0},
        {"max_artifact_bytes": 0},
    ],
)
def test_policy_rejects_malformed_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ActivationPolicy(**kwargs)


def test_orphaned_staging_is_bounded_and_reconciled(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, policy=ActivationPolicy(max_quarantine_entries=3))
    activate_v1(manager)
    manager.close()
    for index in range(6):
        junk = tmp_path / "staging" / f"junk-{index}"
        junk.mkdir()
        (junk / "partial.bin").write_bytes(os.urandom(16))
    slot_dir = next((tmp_path / "generations").iterdir())
    (slot_dir / "00000042").mkdir()
    (tmp_path / "generations" / "deadbeefdeadbeef" / "00000001").mkdir(parents=True)
    (tmp_path / ".state.json.abc.tmp").write_text("{}")

    manager = make_manager(tmp_path, policy=ActivationPolicy(max_quarantine_entries=3))

    assert list((tmp_path / "staging").iterdir()) == []
    assert len(list((tmp_path / "quarantine").iterdir())) == 3
    assert generation_dirs(tmp_path) == [f"{slot_dir.name}/00000001"]
    assert not (tmp_path / ".state.json.abc.tmp").exists()
    with manager.lease("pii-default") as lease:
        assert lease.handle.payload == b"weights-v1"
    manager.close()
