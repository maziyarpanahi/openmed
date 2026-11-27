"""Retention classes and purge orchestration for private-learning artifacts."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from openmed.risk.deletion_plan import DeletionArtifact, plan_deletion_impact
from openmed.training.federated_artifact_purge import (
    FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION,
    FederatedArtifactPurge,
    FederatedPurgeBlockedError,
    FederatedPurgeError,
    FederatedPurgeReceipt,
    FederatedPurgeSide,
    FederatedPurgeStatus,
    FederatedPurgeTrigger,
    FederatedRetentionPolicy,
    PrivateLearningArtifactClass,
    RetentionDisposition,
    emit_federated_round_evidence,
    finalize_federated_round,
)
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundState,
    FederatedRoundTransitionError,
)

A = PrivateLearningArtifactClass
D = RetentionDisposition
Side = FederatedPurgeSide
Trigger = FederatedPurgeTrigger
Status = FederatedPurgeStatus

_SITE = "site-alpha-general-hospital"
_TENSOR_TEXT = "0.123456789,-0.987654321"
_ROUND = f"{_SITE}/round-0007"
_COMMITMENT = "sha256:" + "c" * 64
_SENTINELS = (_SITE, _TENSOR_TEXT, "round-0007", "/", "\\", ".pt", "tmp")


def _digest(label: str) -> str:
    return f"sha256:{sha256(label.encode('utf-8')).hexdigest()}"


class _Crash(BaseException):
    """Simulated process crash that escapes ordinary exception handling."""


@dataclass
class _FakeDeleter:
    """In-memory deleter that resolves digests through a private registry."""

    paths: dict[str, Path] = field(default_factory=dict)
    fail: set[str] = field(default_factory=set)
    crash_after: int | None = None
    calls: list[DeletionArtifact] = field(default_factory=list)

    def add(self, purge: FederatedArtifactPurge, path: Path, cls: A, **kw: Any) -> str:
        digest = purge.register(str(path), cls, **kw)
        self.paths[digest] = path
        return digest

    def __call__(self, artifact: DeletionArtifact) -> None:
        if self.crash_after is not None and len(self.calls) >= self.crash_after:
            raise _Crash()
        self.calls.append(artifact)
        if artifact.artifact_hash in self.fail:
            raise OSError(f"cannot remove {self.paths[artifact.artifact_hash]}")
        self.paths[artifact.artifact_hash].unlink(missing_ok=True)

    @property
    def called_digests(self) -> list[str]:
        return [artifact.artifact_hash for artifact in self.calls]


def _write(root: Path, name: str) -> Path:
    path = root / _SITE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{_SITE}:{_TENSOR_TEXT}", encoding="utf-8")
    return path


def _assert_safe(text: str) -> None:
    for sentinel in _SENTINELS:
        assert sentinel not in text


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    yield


def _coordinator(
    tmp_path: Path,
    *,
    policy: FederatedRetentionPolicy | None = None,
    sink: list[str] | None = None,
) -> tuple[FederatedArtifactPurge, _FakeDeleter, dict[str, str]]:
    purge = FederatedArtifactPurge(
        policy or FederatedRetentionPolicy(Side.COORDINATOR),
        round_reference=_ROUND,
        state_sink=None if sink is None else sink.append,
    )
    deleter = _FakeDeleter()
    names: dict[str, str] = {}
    for index in range(3):
        names[f"payload{index}"] = deleter.add(
            purge,
            _write(tmp_path, f"client-{index}-update.pt"),
            A.CLIENT_UPDATE_PAYLOAD,
        )
    names["masked"] = deleter.add(
        purge, _write(tmp_path, "masked-input.pt"), A.MASKED_INPUT
    )
    names["shares"] = deleter.add(
        purge, _write(tmp_path, "shares.bin"), A.SHARE_MATERIAL
    )
    names["transcript"] = deleter.add(
        purge, _write(tmp_path, "transcript.log"), A.TRANSCRIPT
    )
    names["commitment"] = deleter.add(
        purge,
        _write(tmp_path, "aggregate-commitment.json"),
        A.COMMITMENT,
        dependencies=[names["payload0"], names["payload1"], names["payload2"]],
    )
    names["evidence"] = deleter.add(
        purge,
        _write(tmp_path, "verification-evidence.json"),
        A.EVIDENCE,
        dependencies=[names["commitment"]],
    )
    return purge, deleter, names


def _client(
    tmp_path: Path,
    *,
    policy: FederatedRetentionPolicy | None = None,
) -> tuple[FederatedArtifactPurge, _FakeDeleter, dict[str, str]]:
    purge = FederatedArtifactPurge(
        policy or FederatedRetentionPolicy(Side.CLIENT),
        round_reference=_ROUND,
    )
    deleter = _FakeDeleter()
    names = {
        "update": deleter.add(
            purge, _write(tmp_path, "local-update.pt"), A.CLIENT_UPDATE_PAYLOAD
        ),
        "masked": deleter.add(purge, _write(tmp_path, "masked.pt"), A.MASKED_INPUT),
        "cache": deleter.add(
            purge, _write(tmp_path, "tokenized-cache.arrow"), A.LOCAL_CACHE
        ),
        "checkpoint": deleter.add(
            purge, _write(tmp_path, "resume-state.pt"), A.RESUME_CHECKPOINT
        ),
        "evidence": deleter.add(
            purge, _write(tmp_path, "submission-evidence.json"), A.EVIDENCE
        ),
    }
    return purge, deleter, names


# Retention policy -------------------------------------------------------------


@pytest.mark.parametrize("side", list(Side))
def test_default_policy_declares_every_artifact_class(side: Side) -> None:
    policy = FederatedRetentionPolicy(side)

    assert set(policy.dispositions) == set(A)
    assert policy.disposition_for(A.COMMITMENT) is D.PRESERVE
    assert policy.disposition_for(A.EVIDENCE) is D.PRESERVE
    for cls in (
        A.CLIENT_UPDATE_PAYLOAD,
        A.MASKED_INPUT,
        A.SHARE_MATERIAL,
        A.LOCAL_CACHE,
        A.RESUME_CHECKPOINT,
        A.TRANSCRIPT,
    ):
        assert policy.disposition_for(cls) is D.MANDATORY_PURGE
    assert policy.to_dict()["side"] == side.value


def test_policy_digest_is_stable_and_bound_to_declarations() -> None:
    first = FederatedRetentionPolicy(Side.COORDINATOR)
    second = FederatedRetentionPolicy(Side.COORDINATOR)
    relaxed = FederatedRetentionPolicy(
        Side.COORDINATOR,
        {
            **first.dispositions,
            A.LOCAL_CACHE: D.BEST_EFFORT_PURGE,
        },
    )

    assert first.policy_digest == second.policy_digest
    assert first.policy_digest.startswith("sha256:")
    assert FederatedRetentionPolicy(Side.CLIENT).policy_digest != first.policy_digest
    assert relaxed.policy_digest != first.policy_digest


@pytest.mark.parametrize("cls", [A.COMMITMENT, A.EVIDENCE])
@pytest.mark.parametrize("disposition", [D.MANDATORY_PURGE, D.BEST_EFFORT_PURGE])
def test_policy_refuses_to_purge_commitments_or_evidence(
    cls: A, disposition: D
) -> None:
    dispositions = dict(FederatedRetentionPolicy(Side.COORDINATOR).dispositions)
    dispositions[cls] = disposition

    with pytest.raises(FederatedPurgeError) as caught:
        FederatedRetentionPolicy(Side.COORDINATOR, dispositions)
    assert caught.value.code == "retention_policy_invalid"


@pytest.mark.parametrize(
    "cls", [A.CLIENT_UPDATE_PAYLOAD, A.MASKED_INPUT, A.SHARE_MATERIAL]
)
@pytest.mark.parametrize("disposition", [D.PRESERVE, D.BEST_EFFORT_PURGE])
def test_policy_requires_mandatory_purge_for_update_material(
    cls: A, disposition: D
) -> None:
    dispositions = dict(FederatedRetentionPolicy(Side.CLIENT).dispositions)
    dispositions[cls] = disposition

    with pytest.raises(FederatedPurgeError) as caught:
        FederatedRetentionPolicy(Side.CLIENT, dispositions)
    assert caught.value.code == "retention_policy_invalid"


@pytest.mark.parametrize("cls", [A.LOCAL_CACHE, A.RESUME_CHECKPOINT, A.TRANSCRIPT])
def test_policy_refuses_to_retain_local_training_artifacts(cls: A) -> None:
    dispositions = dict(FederatedRetentionPolicy(Side.CLIENT).dispositions)
    dispositions[cls] = D.PRESERVE

    with pytest.raises(FederatedPurgeError) as caught:
        FederatedRetentionPolicy(Side.CLIENT, dispositions)
    assert caught.value.code == "retention_policy_invalid"


@pytest.mark.parametrize(
    "dispositions",
    [
        {A.CLIENT_UPDATE_PAYLOAD: D.MANDATORY_PURGE},
        {"not_a_class": D.PRESERVE},
        {cls: "purge" for cls in A},
    ],
)
def test_policy_requires_a_complete_typed_declaration(dispositions: Any) -> None:
    with pytest.raises(FederatedPurgeError) as caught:
        FederatedRetentionPolicy(Side.CLIENT, dispositions)
    assert caught.value.code == "retention_policy_invalid"


def test_policy_rejects_unknown_side() -> None:
    with pytest.raises(FederatedPurgeError) as caught:
        FederatedRetentionPolicy("coordinator")  # type: ignore[arg-type]
    assert caught.value.code == "retention_policy_invalid"


# Coordinator after commitment -------------------------------------------------


@pytest.mark.integration
def test_coordinator_purges_client_payloads_after_commitment(tmp_path: Path) -> None:
    purge, deleter, names = _coordinator(tmp_path)

    assert purge.status is Status.AWAITING_TRIGGER
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    assert purge.status is Status.PENDING
    receipt = purge.run(deleter)

    for key in ("payload0", "payload1", "payload2", "masked", "shares", "transcript"):
        assert not deleter.paths[names[key]].exists()
    assert deleter.paths[names["commitment"]].exists()
    assert deleter.paths[names["evidence"]].exists()
    assert purge.status is Status.COMPLETE
    assert isinstance(receipt, FederatedPurgeReceipt)
    assert receipt.status is Status.COMPLETE
    assert receipt.side is Side.COORDINATOR
    assert receipt.trigger is Trigger.COMMITMENT_RECORDED
    assert receipt.commitment_digest == _COMMITMENT
    assert receipt.registered_count == 8
    assert receipt.purge_target_count == 6
    assert receipt.purged_count == 6
    assert receipt.pending_count == 0
    assert receipt.failed_count == 0
    assert receipt.preserved_count == 2
    assert receipt.preserved_dependent_count == 2
    assert receipt.purged_by_class == {
        "client_update_payload": 3,
        "masked_input": 1,
        "share_material": 1,
        "transcript": 1,
    }
    assert receipt.preserved_by_class == {"commitment": 1, "evidence": 1}


def test_receipt_lists_only_digests_and_counts(tmp_path: Path) -> None:
    purge, deleter, _ = _coordinator(tmp_path)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    payload = purge.run(deleter).to_dict()

    assert set(payload) == {
        "schema_version",
        "side",
        "trigger",
        "status",
        "policy_digest",
        "round_digest",
        "commitment_digest",
        "manifest_digest",
        "plan_digest",
        "purged_set_digest",
        "preserved_set_digest",
        "registered_count",
        "purge_target_count",
        "mandatory_target_count",
        "purged_count",
        "pending_count",
        "failed_count",
        "best_effort_failed_count",
        "preserved_count",
        "preserved_dependent_count",
        "purged_by_class",
        "preserved_by_class",
        "raw_values_included",
    }
    assert payload["schema_version"] == FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION
    assert payload["raw_values_included"] is False
    for key, value in payload.items():
        if key.endswith("_digest"):
            assert isinstance(value, str) and value.startswith("sha256:")
            assert len(value) == len("sha256:") + 64
        elif key.endswith("_count"):
            assert type(value) is int
        elif key.endswith("_by_class"):
            assert all(type(count) is int for count in value.values())
            assert set(value) <= {cls.value for cls in A}


def test_receipt_plan_digest_comes_from_the_deletion_planner(tmp_path: Path) -> None:
    purge, deleter, names = _coordinator(tmp_path)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    receipt = purge.run(deleter)

    preserved = {names["commitment"], names["evidence"]}
    manifest = [
        {
            "artifact_hash": artifact.artifact_hash,
            "kind": artifact.kind,
            "retention_class": artifact.retention_class,
            "dependencies": list(artifact.dependencies),
        }
        for artifact in purge.manifest
    ]
    targets = [digest for digest in names.values() if digest not in preserved]
    expected = plan_deletion_impact(manifest, targets)

    assert receipt.plan_digest == expected.plan_digest
    assert receipt.manifest_digest == expected.manifest_digest
    assert {artifact.kind for artifact in deleter.calls} == {
        "client_update_payload",
        "masked_input",
        "share_material",
        "transcript",
    }
    assert {artifact.retention_class for artifact in deleter.calls} == {
        "mandatory_purge"
    }


def test_coordinator_abort_also_purges_payloads(tmp_path: Path) -> None:
    purge, deleter, names = _coordinator(tmp_path)
    purge.record_trigger(Trigger.ROUND_ABORTED)
    receipt = purge.run(deleter)

    assert receipt.trigger is Trigger.ROUND_ABORTED
    assert receipt.commitment_digest is None
    assert receipt.status is Status.COMPLETE
    assert not deleter.paths[names["payload0"]].exists()
    assert deleter.paths[names["commitment"]].exists()


# Client after submission or abort ---------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("trigger", [Trigger.UPDATE_SUBMITTED, Trigger.ROUND_ABORTED])
def test_client_purges_local_caches_and_checkpoints(
    tmp_path: Path, trigger: Trigger
) -> None:
    purge, deleter, names = _client(tmp_path)
    purge.record_trigger(trigger)
    receipt = purge.run(deleter)

    for key in ("update", "masked", "cache", "checkpoint"):
        assert not deleter.paths[names[key]].exists()
    assert deleter.paths[names["evidence"]].exists()
    assert receipt.side is Side.CLIENT
    assert receipt.trigger is trigger
    assert receipt.purged_count == 4
    assert receipt.preserved_count == 1
    assert receipt.purged_by_class == {
        "client_update_payload": 1,
        "local_cache": 1,
        "masked_input": 1,
        "resume_checkpoint": 1,
    }


def test_client_submission_may_bind_a_submission_commitment(tmp_path: Path) -> None:
    purge, deleter, _ = _client(tmp_path)
    purge.record_trigger(Trigger.UPDATE_SUBMITTED, commitment_digest=_COMMITMENT)

    assert purge.run(deleter).commitment_digest == _COMMITMENT


@pytest.mark.parametrize(
    ("side", "trigger", "commitment"),
    [
        (Side.COORDINATOR, Trigger.UPDATE_SUBMITTED, None),
        (Side.CLIENT, Trigger.COMMITMENT_RECORDED, _COMMITMENT),
        (Side.COORDINATOR, Trigger.COMMITMENT_RECORDED, None),
        (Side.COORDINATOR, Trigger.COMMITMENT_RECORDED, "sha256:not-a-digest"),
        (Side.COORDINATOR, Trigger.COMMITMENT_RECORDED, _SITE),
        (Side.COORDINATOR, "commitment_recorded", _COMMITMENT),
    ],
)
def test_trigger_must_match_the_side_and_carry_a_valid_commitment(
    side: Side, trigger: Any, commitment: str | None
) -> None:
    purge = FederatedArtifactPurge(
        FederatedRetentionPolicy(side), round_reference=_ROUND
    )
    with pytest.raises(FederatedPurgeError) as caught:
        purge.record_trigger(trigger, commitment_digest=commitment)
    assert caught.value.code == "trigger_invalid"
    assert purge.status is Status.AWAITING_TRIGGER
    _assert_safe(str(caught.value))


def test_trigger_is_recorded_once(tmp_path: Path) -> None:
    purge, _, _ = _client(tmp_path)
    purge.record_trigger(Trigger.UPDATE_SUBMITTED)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.record_trigger(Trigger.ROUND_ABORTED)
    assert caught.value.code == "trigger_already_recorded"


def test_run_is_refused_before_a_trigger(tmp_path: Path) -> None:
    purge, deleter, names = _client(tmp_path)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.run(deleter)
    assert caught.value.code == "trigger_not_recorded"
    assert deleter.calls == []
    assert deleter.paths[names["update"]].exists()
    with pytest.raises(FederatedPurgeError) as caught:
        purge.receipt()
    assert caught.value.code == "trigger_not_recorded"


def test_registration_closes_once_the_trigger_is_recorded(tmp_path: Path) -> None:
    purge, _, _ = _client(tmp_path)
    purge.record_trigger(Trigger.UPDATE_SUBMITTED)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.register("late-cache", A.LOCAL_CACHE)
    assert caught.value.code == "registration_closed"


@pytest.mark.parametrize(
    ("reference", "cls", "dependencies", "code"),
    [
        ("dup", A.LOCAL_CACHE, (), "artifact_duplicate"),
        ("new", A.LOCAL_CACHE, ("never-registered",), "artifact_unknown"),
        ("new", "local_cache", (), "artifact_invalid"),
        ("", A.LOCAL_CACHE, (), "artifact_invalid"),
        (b"bytes", A.LOCAL_CACHE, (), "artifact_invalid"),
        ("new", A.LOCAL_CACHE, "dup", "artifact_invalid"),
    ],
)
def test_registration_rejects_invalid_artifacts(
    reference: Any, cls: Any, dependencies: Any, code: str
) -> None:
    purge = FederatedArtifactPurge(
        FederatedRetentionPolicy(Side.CLIENT), round_reference=_ROUND
    )
    purge.register("dup", A.LOCAL_CACHE)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.register(reference, cls, dependencies=dependencies)
    assert caught.value.code == code


def test_register_returns_the_canonical_digest_and_accepts_digests() -> None:
    purge = FederatedArtifactPurge(
        FederatedRetentionPolicy(Side.CLIENT), round_reference=_ROUND
    )
    existing = _digest("synthetic-cache")

    assert purge.register("opaque-ref", A.LOCAL_CACHE) == _digest("opaque-ref")
    assert purge.register(existing, A.TRANSCRIPT) == existing
    assert {artifact.artifact_hash for artifact in purge.manifest} == {
        _digest("opaque-ref"),
        existing,
    }


# Finalization and evidence gate -----------------------------------------------


def _evaluating() -> FederatedRoundLifecycle:
    lifecycle = FederatedRoundLifecycle()
    for state in (
        FederatedRoundState.PREFLIGHT,
        FederatedRoundState.COLLECTING,
        FederatedRoundState.AGGREGATING,
        FederatedRoundState.EVALUATING,
    ):
        lifecycle = lifecycle.transition_to(state)
    return lifecycle


def test_evidence_and_finalization_refused_before_the_trigger(tmp_path: Path) -> None:
    purge, _, _ = _coordinator(tmp_path)
    emitted: list[Any] = []

    with pytest.raises(FederatedPurgeBlockedError) as caught:
        emit_federated_round_evidence(purge, emitted.append)
    assert caught.value.code == "purge_awaiting_trigger"
    with pytest.raises(FederatedPurgeBlockedError) as caught:
        finalize_federated_round(_evaluating(), purge)
    assert caught.value.code == "purge_awaiting_trigger"
    assert emitted == []


def test_evidence_and_finalization_refused_while_purge_pending(tmp_path: Path) -> None:
    purge, _, _ = _coordinator(tmp_path)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    emitted: list[Any] = []

    with pytest.raises(FederatedPurgeBlockedError) as caught:
        emit_federated_round_evidence(purge, emitted.append)
    assert caught.value.code == "purge_pending"
    with pytest.raises(FederatedPurgeBlockedError) as caught:
        finalize_federated_round(_evaluating(), purge)
    assert caught.value.code == "purge_pending"
    with pytest.raises(FederatedPurgeBlockedError):
        purge.require_purge_complete()
    assert emitted == []


def test_evidence_and_finalization_refused_after_a_failed_purge(
    tmp_path: Path,
) -> None:
    purge, deleter, names = _coordinator(tmp_path)
    deleter.fail.add(names["payload1"])
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    receipt = purge.run(deleter)
    emitted: list[Any] = []

    assert receipt.status is Status.FAILED
    assert receipt.failed_count == 1
    assert receipt.purged_count == 5
    assert purge.status is Status.FAILED
    assert deleter.paths[names["payload1"]].exists()
    assert not deleter.paths[names["payload2"]].exists()
    with pytest.raises(FederatedPurgeBlockedError) as caught:
        emit_federated_round_evidence(purge, emitted.append)
    assert caught.value.code == "purge_failed"
    with pytest.raises(FederatedPurgeBlockedError) as caught:
        finalize_federated_round(_evaluating(), purge)
    assert caught.value.code == "purge_failed"
    assert emitted == []


@pytest.mark.integration
def test_completed_purge_allows_evidence_and_finalization(tmp_path: Path) -> None:
    purge, deleter, _ = _coordinator(tmp_path)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    receipt = purge.run(deleter)
    emitted: list[Any] = []

    evidence = emit_federated_round_evidence(purge, emitted.append)
    promoted = finalize_federated_round(_evaluating(), purge)

    assert emitted == [evidence]
    assert evidence["receipt"] == receipt.to_dict()
    assert evidence["receipt_digest"] == receipt.receipt_digest
    assert evidence["commitment_digest"] == _COMMITMENT
    assert promoted.state is FederatedRoundState.PROMOTED
    purge.require_purge_complete()


def test_finalization_negative_controls(tmp_path: Path) -> None:
    aborted, deleter, _ = _coordinator(tmp_path / "aborted")
    aborted.record_trigger(Trigger.ROUND_ABORTED)
    aborted.run(deleter)
    with pytest.raises(FederatedPurgeError) as caught:
        finalize_federated_round(_evaluating(), aborted)
    assert caught.value.code == "round_finalization_invalid"

    client, client_deleter, _ = _client(tmp_path / "client")
    client.record_trigger(Trigger.UPDATE_SUBMITTED)
    client.run(client_deleter)
    with pytest.raises(FederatedPurgeError) as caught:
        finalize_federated_round(_evaluating(), client)
    assert caught.value.code == "round_finalization_invalid"
    assert emit_federated_round_evidence(client, lambda _payload: None)

    committed, committed_deleter, _ = _coordinator(tmp_path / "committed")
    committed.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    committed.run(committed_deleter)
    with pytest.raises(FederatedRoundTransitionError):
        finalize_federated_round(FederatedRoundLifecycle(), committed)
    with pytest.raises(FederatedPurgeError) as caught:
        finalize_federated_round("evaluating", committed)  # type: ignore[arg-type]
    assert caught.value.code == "round_finalization_invalid"


def test_failed_evidence_emitter_is_reported_with_a_fixed_code(
    tmp_path: Path,
) -> None:
    purge, deleter, _ = _client(tmp_path)
    purge.record_trigger(Trigger.ROUND_ABORTED)
    purge.run(deleter)

    def _emitter(_payload: Any) -> None:
        raise RuntimeError(f"upload to {_SITE} failed")

    with pytest.raises(FederatedPurgeError) as caught:
        emit_federated_round_evidence(purge, _emitter)
    assert caught.value.code == "evidence_emitter_failed"
    assert caught.value.__cause__ is None
    _assert_safe(str(caught.value))


# Restart and idempotency ------------------------------------------------------


@pytest.mark.integration
def test_restart_resumes_a_failed_purge_idempotently(tmp_path: Path) -> None:
    sink: list[str] = []
    purge, deleter, names = _coordinator(tmp_path, sink=sink)
    deleter.fail.update({names["payload1"], names["shares"]})
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    failed = purge.run(deleter)
    assert failed.status is Status.FAILED
    assert failed.failed_count == 2

    restored = FederatedArtifactPurge.from_state_json(
        sink[-1], policy=FederatedRetentionPolicy(Side.COORDINATOR)
    )
    assert restored.status is Status.FAILED
    resumed_deleter = _FakeDeleter(paths=deleter.paths)
    resumed = restored.run(resumed_deleter)

    assert sorted(resumed_deleter.called_digests) == sorted(
        [names["payload1"], names["shares"]]
    )
    assert resumed.status is Status.COMPLETE
    assert resumed.plan_digest == failed.plan_digest
    assert resumed.purged_count == 6

    again = _FakeDeleter(paths=deleter.paths)
    assert restored.run(again) == resumed
    assert again.calls == []

    uninterrupted, clean_deleter, _ = _coordinator(tmp_path / "clean")
    uninterrupted.record_trigger(
        Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT
    )
    clean = uninterrupted.run(clean_deleter)
    assert clean.purged_count == resumed.purged_count
    assert clean.status is resumed.status


@pytest.mark.integration
def test_restart_after_a_crash_resumes_from_the_last_checkpoint(
    tmp_path: Path,
) -> None:
    sink: list[str] = []
    purge, deleter, names = _coordinator(tmp_path, sink=sink)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    deleter.crash_after = 2

    with pytest.raises(_Crash):
        purge.run(deleter)
    finished_before_crash = list(deleter.called_digests)
    assert len(finished_before_crash) == 2

    restored = FederatedArtifactPurge.from_state_json(
        sink[-1], policy=FederatedRetentionPolicy(Side.COORDINATOR)
    )
    assert restored.status is Status.PENDING
    with pytest.raises(FederatedPurgeBlockedError):
        emit_federated_round_evidence(restored, lambda _payload: None)
    resumed_deleter = _FakeDeleter(paths=deleter.paths)
    receipt = restored.run(resumed_deleter)

    assert set(resumed_deleter.called_digests).isdisjoint(finished_before_crash)
    assert len(resumed_deleter.calls) == 4
    assert receipt.status is Status.COMPLETE
    assert receipt.purged_count == 6
    for key in ("payload0", "payload1", "payload2", "masked", "shares", "transcript"):
        assert not deleter.paths[names[key]].exists()


def test_state_round_trips_before_the_trigger(tmp_path: Path) -> None:
    sink: list[str] = []
    purge, _, _ = _coordinator(tmp_path, sink=sink)

    state = purge.to_state_json()
    restored = FederatedArtifactPurge.from_state_json(
        state, policy=FederatedRetentionPolicy(Side.COORDINATOR)
    )
    assert restored.to_state_json() == state
    assert restored.status is Status.AWAITING_TRIGGER
    assert sink and sink[-1] == state


def test_restore_accepts_dependents_that_sort_before_their_dependencies() -> None:
    policy = FederatedRetentionPolicy(Side.COORDINATOR)
    purge = FederatedArtifactPurge(policy, round_reference=_ROUND)
    payload = purge.register("sha256:" + "f" * 64, A.CLIENT_UPDATE_PAYLOAD)
    purge.register("sha256:" + "1" * 64, A.COMMITMENT, dependencies=[payload])

    restored = FederatedArtifactPurge.from_state_json(
        purge.to_state_json(), policy=policy
    )
    assert restored.to_state_json() == purge.to_state_json()


def test_keyed_artifact_digests_survive_a_restart() -> None:
    policy = FederatedRetentionPolicy(Side.CLIENT)
    purge = FederatedArtifactPurge(policy, round_reference=_ROUND)
    keyed = "hmac-sha256:" + "a" * 64

    assert purge.register(keyed, A.LOCAL_CACHE) == keyed
    restored = FederatedArtifactPurge.from_state_json(
        purge.to_state_json(), policy=policy
    )
    assert [artifact.artifact_hash for artifact in restored.manifest] == [keyed]


def test_restore_keeps_artifacts_purged_before_the_trigger(tmp_path: Path) -> None:
    sink: list[str] = []
    purge = FederatedArtifactPurge(
        FederatedRetentionPolicy(Side.CLIENT),
        round_reference=_ROUND,
        state_sink=sink.append,
    )
    deleter = _FakeDeleter()
    cache = deleter.add(purge, _write(tmp_path, "cache.arrow"), A.LOCAL_CACHE)
    update = deleter.add(purge, _write(tmp_path, "update.pt"), A.CLIENT_UPDATE_PAYLOAD)
    purge.purge_artifact(cache, deleter)

    restored = FederatedArtifactPurge.from_state_json(
        sink[-1], policy=FederatedRetentionPolicy(Side.CLIENT)
    )
    restored.record_trigger(Trigger.UPDATE_SUBMITTED)
    resumed = _FakeDeleter(paths=deleter.paths)
    receipt = restored.run(resumed)

    assert resumed.called_digests == [update]
    assert receipt.purged_count == 2
    assert receipt.status is Status.COMPLETE


def test_state_sink_failure_stops_the_run_with_a_fixed_code(tmp_path: Path) -> None:
    broken: list[bool] = []

    def _sink(_state: str) -> None:
        if broken:
            raise OSError(f"disk full under {_SITE}")

    purge = FederatedArtifactPurge(
        FederatedRetentionPolicy(Side.CLIENT),
        round_reference=_ROUND,
        state_sink=_sink,
    )
    purge.register("cache", A.LOCAL_CACHE)
    purge.record_trigger(Trigger.ROUND_ABORTED)
    broken.append(True)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.run(lambda _artifact: None)
    assert caught.value.code == "state_sink_failed"
    assert caught.value.__cause__ is None
    _assert_safe(str(caught.value))


def _purgeable(payload: dict[str, Any]) -> dict[str, Any]:
    return next(
        artifact
        for artifact in payload["artifacts"]
        if artifact["artifact_class"] != "evidence"
    )


def _mutated_state(tmp_path: Path, mutate: Any) -> str:
    purge, _, _ = _client(tmp_path)
    purge.record_trigger(Trigger.UPDATE_SUBMITTED)
    payload = json.loads(purge.to_state_json())
    mutate(payload)
    return json.dumps(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(policy_digest="sha256:" + "0" * 64),
        lambda p: p.update(side="coordinator"),
        lambda p: p.update(schema_version="v0"),
        lambda p: p.update(extra="value"),
        lambda p: p.update(trigger="commitment_recorded"),
        lambda p: p.update(round_digest=_SITE),
        lambda p: _purgeable(p).update(state="exploded"),
        lambda p: _purgeable(p).update(artifact_class="commitment"),
        lambda p: _purgeable(p).update(artifact_hash="sha256:bad"),
        lambda p: _purgeable(p).update(dependencies=["sha256:" + "9" * 64]),
        lambda p: p["artifacts"].append(dict(_purgeable(p))),
        lambda p: _purgeable(p).update(path="/secret"),
        lambda p: p.update(artifacts=[]),
    ],
)
def test_tampered_state_is_rejected(tmp_path: Path, mutate: Any) -> None:
    state = _mutated_state(tmp_path, mutate)

    with pytest.raises(FederatedPurgeError) as caught:
        FederatedArtifactPurge.from_state_json(
            state, policy=FederatedRetentionPolicy(Side.CLIENT)
        )
    assert caught.value.code == "purge_state_invalid"


def test_preserved_artifacts_cannot_be_marked_deleted_in_state(
    tmp_path: Path,
) -> None:
    def _mutate(payload: dict[str, Any]) -> None:
        for artifact in payload["artifacts"]:
            if artifact["artifact_class"] == "evidence":
                artifact["state"] = "deleted"

    state = _mutated_state(tmp_path, _mutate)
    with pytest.raises(FederatedPurgeError) as caught:
        FederatedArtifactPurge.from_state_json(
            state, policy=FederatedRetentionPolicy(Side.CLIENT)
        )
    assert caught.value.code == "purge_state_invalid"


@pytest.mark.parametrize(
    "state", ["not json", '{"a": 1, "a": 2}', "[]", "NaN", 7, b"{}"]
)
def test_malformed_state_is_rejected(state: Any) -> None:
    with pytest.raises(FederatedPurgeError) as caught:
        FederatedArtifactPurge.from_state_json(
            state, policy=FederatedRetentionPolicy(Side.CLIENT)
        )
    assert caught.value.code == "purge_state_invalid"


# Preservation of commitments and evidence -------------------------------------


@pytest.mark.parametrize("key", ["commitment", "evidence"])
def test_requests_to_delete_preserved_artifacts_are_refused(
    tmp_path: Path, key: str
) -> None:
    purge, deleter, names = _coordinator(tmp_path)
    path = deleter.paths[names[key]]

    for reference in (str(path), names[key]):
        with pytest.raises(FederatedPurgeError) as caught:
            purge.purge_artifact(reference, deleter)
        assert caught.value.code == "preserved_artifact_deletion_refused"
        _assert_safe(str(caught.value))
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    purge.run(deleter)
    with pytest.raises(FederatedPurgeError) as caught:
        purge.purge_artifact(names[key], deleter)
    assert caught.value.code == "preserved_artifact_deletion_refused"

    assert path.exists()
    assert names[key] not in deleter.called_digests


def test_purge_artifact_deletes_one_purge_class_artifact_early(
    tmp_path: Path,
) -> None:
    purge, deleter, names = _client(tmp_path)

    purge.purge_artifact(names["cache"], deleter)
    purge.purge_artifact(names["cache"], deleter)
    assert deleter.called_digests == [names["cache"]]
    assert not deleter.paths[names["cache"]].exists()

    purge.record_trigger(Trigger.UPDATE_SUBMITTED)
    receipt = purge.run(deleter)
    assert receipt.purged_count == 4
    assert deleter.called_digests.count(names["cache"]) == 1


def test_purge_artifact_rejects_unknown_references(tmp_path: Path) -> None:
    purge, deleter, _ = _client(tmp_path)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.purge_artifact("never-registered", deleter)
    assert caught.value.code == "artifact_unknown"
    assert deleter.calls == []


def test_run_never_passes_preserved_artifacts_to_the_deleter(tmp_path: Path) -> None:
    purge, deleter, names = _coordinator(tmp_path)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    purge.run(deleter)

    assert names["commitment"] not in deleter.called_digests
    assert names["evidence"] not in deleter.called_digests
    assert all(
        artifact.kind not in {"commitment", "evidence"} for artifact in deleter.calls
    )


@pytest.mark.parametrize("deleter", [None, "rm", 7])
def test_a_callable_deleter_is_required(tmp_path: Path, deleter: Any) -> None:
    purge, _, _ = _client(tmp_path)
    purge.record_trigger(Trigger.ROUND_ABORTED)

    with pytest.raises(FederatedPurgeError) as caught:
        purge.run(deleter)
    assert caught.value.code == "deleter_required"
    assert purge.status is Status.PENDING


def test_best_effort_failures_do_not_block_evidence(tmp_path: Path) -> None:
    dispositions = dict(FederatedRetentionPolicy(Side.CLIENT).dispositions)
    dispositions[A.LOCAL_CACHE] = D.BEST_EFFORT_PURGE
    policy = FederatedRetentionPolicy(Side.CLIENT, dispositions)
    purge, deleter, names = _client(tmp_path, policy=policy)
    deleter.fail.add(names["cache"])
    purge.record_trigger(Trigger.UPDATE_SUBMITTED)
    receipt = purge.run(deleter)

    assert receipt.status is Status.COMPLETE
    assert receipt.best_effort_failed_count == 1
    assert receipt.failed_count == 0
    assert receipt.mandatory_target_count == 3
    assert receipt.purge_target_count == 4
    assert emit_federated_round_evidence(purge, lambda _payload: None)


def test_mandatory_failure_on_the_client_blocks_evidence(tmp_path: Path) -> None:
    purge, deleter, names = _client(tmp_path)
    deleter.fail.add(names["checkpoint"])
    purge.record_trigger(Trigger.ROUND_ABORTED)

    assert purge.run(deleter).status is Status.FAILED
    with pytest.raises(FederatedPurgeBlockedError) as caught:
        emit_federated_round_evidence(purge, lambda _payload: None)
    assert caught.value.code == "purge_failed"


def test_round_with_only_preserved_artifacts_completes_without_a_plan() -> None:
    purge = FederatedArtifactPurge(
        FederatedRetentionPolicy(Side.COORDINATOR), round_reference=_ROUND
    )
    purge.register("commitment-file", A.COMMITMENT)
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    calls: list[Any] = []
    receipt = purge.run(calls.append)

    assert calls == []
    assert receipt.status is Status.COMPLETE
    assert receipt.plan_digest is None
    assert receipt.purge_target_count == 0
    assert receipt.preserved_count == 1


# Privacy sentinels ------------------------------------------------------------


def test_receipts_state_errors_and_reprs_exclude_paths_sites_and_tensors(
    tmp_path: Path,
) -> None:
    sink: list[str] = []
    purge, deleter, names = _coordinator(tmp_path, sink=sink)
    deleter.fail.add(names["payload0"])
    purge.record_trigger(Trigger.COMMITMENT_RECORDED, commitment_digest=_COMMITMENT)
    failed = purge.run(deleter)
    deleter.fail.clear()
    receipt = purge.run(deleter)
    evidence = emit_federated_round_evidence(purge, lambda _payload: None)

    outputs = [
        failed.to_json(),
        receipt.to_json(),
        repr(receipt),
        repr(failed),
        repr(purge),
        str(purge),
        repr(purge.policy),
        purge.to_state_json(),
        json.dumps(evidence),
        *sink,
    ]
    errors: list[BaseException] = []
    for action in (
        lambda: purge.purge_artifact(str(deleter.paths[names["evidence"]]), deleter),
        lambda: purge.register(str(tmp_path / _SITE / "late.pt"), A.LOCAL_CACHE),
        lambda: purge.record_trigger(Trigger.ROUND_ABORTED),
        lambda: FederatedArtifactPurge.from_state_json(
            f'{{"path": "{tmp_path}/{_SITE}"}}',
            policy=FederatedRetentionPolicy(Side.COORDINATOR),
        ),
    ):
        with pytest.raises(FederatedPurgeError) as caught:
            action()
        errors.append(caught.value)
    outputs.extend(str(error) for error in errors)
    outputs.extend(repr(error.args) for error in errors)

    for text in (repr(receipt), repr(failed), repr(purge)):
        assert not any(char.isdigit() for char in text.split("receipt_digest")[0])
    for output in outputs:
        _assert_safe(output)
        assert str(tmp_path) not in output
    assert all(error.__cause__ is None for error in errors)


def test_deleter_exception_text_never_reaches_the_receipt(tmp_path: Path) -> None:
    purge, deleter, names = _client(tmp_path)
    deleter.fail.update(names.values())
    purge.record_trigger(Trigger.ROUND_ABORTED)
    receipt = purge.run(deleter)

    assert receipt.failed_count == 4
    _assert_safe(receipt.to_json())
    assert "cannot remove" not in receipt.to_json()
    assert "cannot remove" not in purge.to_state_json()
