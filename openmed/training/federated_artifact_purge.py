"""Retention classes and purge orchestration for private-learning artifacts.

A federated round leaves sensitive material behind: per-client update payloads,
masked inputs and share material at the coordinator, and tokenized caches,
resume checkpoints and local update copies at each site.  This module lets a
round policy declare a retention class for every private-learning artifact,
then purges every purge-class artifact once the coordinator records the
aggregate commitment, or once a client submits its update or aborts.

Deletion planning and execution are delegated to
:mod:`openmed.risk.deletion_plan`; the filesystem deleter is always injected by
the caller.  Commitments and verification evidence are preserved and requests
to delete them are refused.  Round finalization and evidence emission are
blocked while a mandatory purge is pending or has failed.

Receipts, resume state, errors and reprs contain only controlled codes, counts
and SHA-256 digests.  Artifact references are hashed at the input boundary, so
paths, site identities and tensor values are never retained.  The module
performs no network, logging or filesystem operation of its own.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from openmed.core.audit import hash_text, stable_hash
from openmed.risk.deletion_plan import (
    DeletionArtifact,
    DeletionExecutionError,
    DeletionImpactPlan,
    DeletionPlanError,
    execute_deletion_plan,
    plan_deletion_impact,
)

from .federated_round import FederatedRoundLifecycle, FederatedRoundState

FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION: Final = (
    "openmed.training.federated_artifact_purge.v1"
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ARTIFACT_DIGEST_RE = re.compile(r"(?:sha256|hmac-sha256):[0-9a-f]{64}\Z")
_MAX_ARTIFACTS: Final = 100_000
_MAX_STATE_CHARS: Final = 64 * 1024 * 1024

DeletionCallback = Callable[[DeletionArtifact], None]
StateSink = Callable[[str], None]
EvidenceEmitter = Callable[[Mapping[str, Any]], None]


class PrivateLearningArtifactClass(str, Enum):
    """Kinds of artifacts a private-learning round produces."""

    CLIENT_UPDATE_PAYLOAD = "client_update_payload"
    MASKED_INPUT = "masked_input"
    SHARE_MATERIAL = "share_material"
    LOCAL_CACHE = "local_cache"
    RESUME_CHECKPOINT = "resume_checkpoint"
    TRANSCRIPT = "transcript"
    COMMITMENT = "commitment"
    EVIDENCE = "evidence"

    def __str__(self) -> str:
        return self.value


class RetentionDisposition(str, Enum):
    """What happens to an artifact class once the purge trigger is recorded."""

    MANDATORY_PURGE = "mandatory_purge"
    BEST_EFFORT_PURGE = "best_effort_purge"
    PRESERVE = "preserve"

    def __str__(self) -> str:
        return self.value


class FederatedPurgeSide(str, Enum):
    """The party that holds the artifacts being purged."""

    COORDINATOR = "coordinator"
    CLIENT = "client"

    def __str__(self) -> str:
        return self.value


class FederatedPurgeTrigger(str, Enum):
    """Round events that start a purge."""

    COMMITMENT_RECORDED = "commitment_recorded"
    UPDATE_SUBMITTED = "update_submitted"
    ROUND_ABORTED = "round_aborted"

    def __str__(self) -> str:
        return self.value


class FederatedPurgeStatus(str, Enum):
    """Aggregate purge state used by the finalization and evidence gate."""

    AWAITING_TRIGGER = "awaiting_trigger"
    PENDING = "pending"
    FAILED = "failed"
    COMPLETE = "complete"

    def __str__(self) -> str:
        return self.value


_A = PrivateLearningArtifactClass
_D = RetentionDisposition
_ALWAYS_PRESERVED: Final = frozenset({_A.COMMITMENT, _A.EVIDENCE})
_ALWAYS_MANDATORY: Final = frozenset(
    {_A.CLIENT_UPDATE_PAYLOAD, _A.MASKED_INPUT, _A.SHARE_MATERIAL}
)
_PURGE_DISPOSITIONS: Final = frozenset({_D.MANDATORY_PURGE, _D.BEST_EFFORT_PURGE})
_SIDE_TRIGGERS: Final = MappingProxyType(
    {
        FederatedPurgeSide.COORDINATOR: frozenset(
            {
                FederatedPurgeTrigger.COMMITMENT_RECORDED,
                FederatedPurgeTrigger.ROUND_ABORTED,
            }
        ),
        FederatedPurgeSide.CLIENT: frozenset(
            {
                FederatedPurgeTrigger.UPDATE_SUBMITTED,
                FederatedPurgeTrigger.ROUND_ABORTED,
            }
        ),
    }
)

_RETAINED: Final = "retained"
_PENDING: Final = "pending"
_DELETED: Final = "deleted"
_FAILED: Final = "failed"
_PURGE_STATES: Final = frozenset({_PENDING, _DELETED, _FAILED})

_ERROR_CODES: Final = frozenset(
    {
        "retention_policy_invalid",
        "artifact_invalid",
        "artifact_duplicate",
        "artifact_unknown",
        "registration_closed",
        "trigger_invalid",
        "trigger_already_recorded",
        "trigger_not_recorded",
        "preserved_artifact_deletion_refused",
        "deleter_required",
        "purge_plan_unsafe",
        "purge_state_invalid",
        "state_sink_failed",
        "evidence_emitter_failed",
        "round_finalization_invalid",
        "purge_awaiting_trigger",
        "purge_pending",
        "purge_failed",
        "purge_invalid",
    }
)


class FederatedPurgeError(ValueError):
    """Fixed-code refusal; references, paths and values are never attached.

    Attributes:
        code: A stable, controlled reason code.
    """

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERROR_CODES else "purge_invalid"
        super().__init__(self.code)


class FederatedPurgeBlockedError(FederatedPurgeError):
    """Raised when finalization or evidence emission waits on a purge."""


_BLOCKED_CODES: Final[Mapping[FederatedPurgeStatus, str]] = MappingProxyType(
    {
        FederatedPurgeStatus.AWAITING_TRIGGER: "purge_awaiting_trigger",
        FederatedPurgeStatus.PENDING: "purge_pending",
        FederatedPurgeStatus.FAILED: "purge_failed",
    }
)


def _default_dispositions() -> dict[PrivateLearningArtifactClass, RetentionDisposition]:
    return {
        cls: _D.PRESERVE if cls in _ALWAYS_PRESERVED else _D.MANDATORY_PURGE
        for cls in _A
    }


@dataclass(frozen=True)
class FederatedRetentionPolicy:
    """Round policy that declares a retention class for every artifact class.

    Commitments and evidence must be preserved.  Client update payloads,
    masked inputs and share material must be purged as a mandatory step.
    Local caches, resume checkpoints and transcripts must be purged, either as
    a mandatory step (the default) or on a best-effort basis.

    Args:
        side: The party that holds the artifacts.
        dispositions: A complete mapping from every
            :class:`PrivateLearningArtifactClass` to a
            :class:`RetentionDisposition`.  Defaults to the strictest policy.

    Raises:
        FederatedPurgeError: ``retention_policy_invalid`` when the side or any
            declaration is missing, unknown or weaker than allowed.
    """

    side: FederatedPurgeSide
    dispositions: Mapping[PrivateLearningArtifactClass, RetentionDisposition] = field(
        default_factory=_default_dispositions
    )

    def __post_init__(self) -> None:
        if not isinstance(self.side, FederatedPurgeSide):
            raise FederatedPurgeError("retention_policy_invalid")
        try:
            declared = dict(self.dispositions)
        except Exception:
            raise FederatedPurgeError("retention_policy_invalid") from None
        if set(declared) != set(_A) or len(declared) != len(_A):
            raise FederatedPurgeError("retention_policy_invalid")
        for cls, disposition in declared.items():
            if not isinstance(cls, _A) or not isinstance(disposition, _D):
                raise FederatedPurgeError("retention_policy_invalid")
            if cls in _ALWAYS_PRESERVED and disposition is not _D.PRESERVE:
                raise FederatedPurgeError("retention_policy_invalid")
            if cls in _ALWAYS_MANDATORY and disposition is not _D.MANDATORY_PURGE:
                raise FederatedPurgeError("retention_policy_invalid")
            if cls not in _ALWAYS_PRESERVED and disposition not in _PURGE_DISPOSITIONS:
                raise FederatedPurgeError("retention_policy_invalid")
        ordered = {cls: declared[cls] for cls in _A}
        object.__setattr__(self, "dispositions", MappingProxyType(ordered))

    def disposition_for(
        self, artifact_class: PrivateLearningArtifactClass
    ) -> RetentionDisposition:
        """Return the declared disposition for one artifact class.

        Args:
            artifact_class: The artifact class to look up.

        Returns:
            The declared :class:`RetentionDisposition`.
        """

        return self.dispositions[artifact_class]

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, label-only policy representation."""

        return {
            "schema_version": FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION,
            "side": self.side.value,
            "dispositions": {
                cls.value: disposition.value
                for cls, disposition in self.dispositions.items()
            },
        }

    @property
    def policy_digest(self) -> str:
        """Return the SHA-256 digest that binds receipts to this policy."""

        return stable_hash(self.to_dict())


@dataclass(frozen=True, repr=False)
class FederatedPurgeReceipt:
    """Digest-and-count receipt for one purge.

    The receipt never lists individual artifacts, and its ``repr`` hides the
    counts, which can reveal how many clients took part.  ``purged_set_digest`` and
    ``preserved_set_digest`` bind the exact sets without exposing members.
    ``plan_digest`` and ``manifest_digest`` come from the deletion planner and
    stay stable across restarts; ``plan_digest`` is ``None`` when the round has
    no purge-class artifacts.
    """

    side: FederatedPurgeSide
    trigger: FederatedPurgeTrigger
    status: FederatedPurgeStatus
    policy_digest: str
    round_digest: str
    commitment_digest: str | None
    manifest_digest: str | None
    plan_digest: str | None
    purged_set_digest: str
    preserved_set_digest: str
    registered_count: int
    purge_target_count: int
    mandatory_target_count: int
    purged_count: int
    pending_count: int
    failed_count: int
    best_effort_failed_count: int
    preserved_count: int
    preserved_dependent_count: int
    _purged_by_class: tuple[tuple[str, int], ...] = ()
    _preserved_by_class: tuple[tuple[str, int], ...] = ()
    schema_version: str = FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION

    def __repr__(self) -> str:
        return (
            f"FederatedPurgeReceipt(side={self.side.value}, "
            f"trigger={self.trigger.value}, status={self.status.value}, "
            f"receipt_digest={self.receipt_digest})"
        )

    @property
    def purged_by_class(self) -> dict[str, int]:
        """Return purged-artifact counts grouped by artifact class."""

        return dict(self._purged_by_class)

    @property
    def preserved_by_class(self) -> dict[str, int]:
        """Return preserved-artifact counts grouped by artifact class."""

        return dict(self._preserved_by_class)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic digest-and-count representation."""

        return {
            "schema_version": self.schema_version,
            "side": self.side.value,
            "trigger": self.trigger.value,
            "status": self.status.value,
            "policy_digest": self.policy_digest,
            "round_digest": self.round_digest,
            "commitment_digest": self.commitment_digest,
            "manifest_digest": self.manifest_digest,
            "plan_digest": self.plan_digest,
            "purged_set_digest": self.purged_set_digest,
            "preserved_set_digest": self.preserved_set_digest,
            "registered_count": self.registered_count,
            "purge_target_count": self.purge_target_count,
            "mandatory_target_count": self.mandatory_target_count,
            "purged_count": self.purged_count,
            "pending_count": self.pending_count,
            "failed_count": self.failed_count,
            "best_effort_failed_count": self.best_effort_failed_count,
            "preserved_count": self.preserved_count,
            "preserved_dependent_count": self.preserved_dependent_count,
            "purged_by_class": self.purged_by_class,
            "preserved_by_class": self.preserved_by_class,
            "raw_values_included": False,
        }

    def to_json(self) -> str:
        """Serialize the receipt with stable JSON formatting."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def receipt_digest(self) -> str:
        """Return the SHA-256 digest of the serialized receipt."""

        return stable_hash(self.to_dict())


@dataclass
class _Entry:
    artifact: DeletionArtifact
    artifact_class: PrivateLearningArtifactClass
    disposition: RetentionDisposition
    state: str


def _reference_digest(value: Any) -> str:
    try:
        return DeletionArtifact(artifact_hash=value).artifact_hash
    except DeletionPlanError:
        raise FederatedPurgeError("artifact_invalid") from None


def _round_digest(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FederatedPurgeError("purge_invalid")
    if _DIGEST_RE.fullmatch(value):
        return value
    return hash_text(value)


def _count(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(counts.items()))


class FederatedArtifactPurge:
    """Purge orchestrator for one party's artifacts in one federated round.

    Register every private-learning artifact with :meth:`register`, record the
    purge trigger with :meth:`record_trigger`, then call :meth:`run` with an
    injected deleter.  The deleter receives a normalized
    :class:`~openmed.risk.deletion_plan.DeletionArtifact` whose ``kind`` is the
    artifact class and whose ``artifact_hash`` is the digest returned by
    :meth:`register`; it resolves that digest through the caller's private
    registry and must treat an already-missing artifact as deleted.

    Args:
        policy: The round retention policy.
        round_reference: An opaque round reference or SHA-256 digest.  Opaque
            references are hashed immediately.
        state_sink: Optional callback that receives the digest-only resume
            state after every registration, trigger and deletion outcome, so a
            restart can resume through :meth:`from_state_json`.
    """

    def __init__(
        self,
        policy: FederatedRetentionPolicy,
        *,
        round_reference: str,
        state_sink: StateSink | None = None,
    ) -> None:
        if not isinstance(policy, FederatedRetentionPolicy):
            raise FederatedPurgeError("retention_policy_invalid")
        if state_sink is not None and not callable(state_sink):
            raise FederatedPurgeError("purge_invalid")
        self._policy = policy
        self._round_digest = _round_digest(round_reference)
        self._state_sink = state_sink
        self._entries: dict[str, _Entry] = {}
        self._trigger: FederatedPurgeTrigger | None = None
        self._commitment_digest: str | None = None

    def __repr__(self) -> str:
        return (
            f"FederatedArtifactPurge(side={self.side.value}, "
            f"status={self.status.value})"
        )

    @property
    def policy(self) -> FederatedRetentionPolicy:
        """Return the round retention policy."""

        return self._policy

    @property
    def side(self) -> FederatedPurgeSide:
        """Return the party that holds the artifacts."""

        return self._policy.side

    @property
    def trigger(self) -> FederatedPurgeTrigger | None:
        """Return the recorded purge trigger, if any."""

        return self._trigger

    @property
    def manifest(self) -> tuple[DeletionArtifact, ...]:
        """Return the digest-only deletion manifest in canonical order."""

        return tuple(self._entries[digest].artifact for digest in sorted(self._entries))

    @property
    def status(self) -> FederatedPurgeStatus:
        """Return the aggregate purge status used by the gate."""

        if self._trigger is None:
            return FederatedPurgeStatus.AWAITING_TRIGGER
        mandatory = [
            entry.state
            for entry in self._entries.values()
            if entry.disposition is _D.MANDATORY_PURGE
        ]
        if _FAILED in mandatory:
            return FederatedPurgeStatus.FAILED
        if _PENDING in mandatory:
            return FederatedPurgeStatus.PENDING
        return FederatedPurgeStatus.COMPLETE

    def register(
        self,
        reference: str,
        artifact_class: PrivateLearningArtifactClass,
        *,
        dependencies: Iterable[str] = (),
    ) -> str:
        """Register one artifact before the purge trigger is recorded.

        Args:
            reference: An opaque local reference or SHA-256 digest.  Opaque
                references, including paths, are hashed and never retained.
            artifact_class: The artifact's private-learning class.
            dependencies: References of already registered artifacts this
                artifact needs, such as the payloads a commitment covers.

        Returns:
            The canonical SHA-256 digest the injected deleter will receive.

        Raises:
            FederatedPurgeError: When registration is closed, the artifact is
                invalid or duplicated, or a dependency is unknown.
        """

        if self._trigger is not None:
            raise FederatedPurgeError("registration_closed")
        if not isinstance(artifact_class, _A):
            raise FederatedPurgeError("artifact_invalid")
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies, Iterable
        ):
            raise FederatedPurgeError("artifact_invalid")
        if len(self._entries) >= _MAX_ARTIFACTS:
            raise FederatedPurgeError("artifact_invalid")
        disposition = self._policy.disposition_for(artifact_class)
        try:
            artifact = DeletionArtifact(
                artifact_hash=reference,
                kind=artifact_class.value,
                retention_class=disposition.value,
                dependencies=tuple(dependencies),
            )
        except (DeletionPlanError, TypeError):
            raise FederatedPurgeError("artifact_invalid") from None
        if artifact.artifact_hash in self._entries:
            raise FederatedPurgeError("artifact_duplicate")
        if any(dependency not in self._entries for dependency in artifact.dependencies):
            raise FederatedPurgeError("artifact_unknown")
        self._entries[artifact.artifact_hash] = _Entry(
            artifact=artifact,
            artifact_class=artifact_class,
            disposition=disposition,
            state=_RETAINED if disposition is _D.PRESERVE else _PENDING,
        )
        self._checkpoint()
        return artifact.artifact_hash

    def record_trigger(
        self,
        trigger: FederatedPurgeTrigger,
        *,
        commitment_digest: str | None = None,
    ) -> None:
        """Record the round event that starts the purge and close registration.

        Args:
            trigger: ``commitment_recorded`` or ``round_aborted`` on the
                coordinator; ``update_submitted`` or ``round_aborted`` on a
                client.
            commitment_digest: The aggregate commitment digest.  Required for
                ``commitment_recorded``; optional otherwise.

        Raises:
            FederatedPurgeError: When the trigger does not fit the side, the
                commitment digest is missing or malformed, or a trigger was
                already recorded.
        """

        if self._trigger is not None:
            raise FederatedPurgeError("trigger_already_recorded")
        _validate_trigger(self.side, trigger, commitment_digest)
        self._trigger = trigger
        self._commitment_digest = commitment_digest
        self._checkpoint()

    def run(self, deleter: DeletionCallback) -> FederatedPurgeReceipt:
        """Purge every pending or failed purge-class artifact.

        Each target executes through its own confirmed single-target plan
        built over the full round manifest, so one failing deleter call does
        not stop the remaining targets.  Already-deleted artifacts are never
        passed to the deleter again, which makes a resumed run idempotent.

        Args:
            deleter: Injected callback that removes one artifact.  Returning
                normally marks the artifact deleted; raising marks it failed.

        Returns:
            The current :class:`FederatedPurgeReceipt`.

        Raises:
            FederatedPurgeError: When no trigger is recorded, the deleter is
                not callable, the plan fails safety checks, or the state sink
                fails.
        """

        if self._trigger is None:
            raise FederatedPurgeError("trigger_not_recorded")
        if not callable(deleter):
            raise FederatedPurgeError("deleter_required")
        manifest = self.manifest
        targets = sorted(
            digest
            for digest, entry in self._entries.items()
            if entry.state in {_PENDING, _FAILED}
        )
        for digest in targets:
            self._purge_one(manifest, digest, deleter)
        return self.receipt()

    def purge_artifact(self, reference: str, deleter: DeletionCallback) -> None:
        """Purge one registered purge-class artifact immediately.

        Requests that target a preserved commitment or evidence artifact are
        refused before the deleter is invoked.  Purging an already-deleted
        artifact is a no-op.

        Args:
            reference: The artifact's reference or digest.
            deleter: Injected callback that removes one artifact.

        Raises:
            FederatedPurgeError: ``preserved_artifact_deletion_refused`` for
                commitments and evidence, ``artifact_unknown`` for
                unregistered references, or ``deleter_required``.
        """

        digest = _reference_digest(reference)
        entry = self._entries.get(digest)
        if entry is None:
            raise FederatedPurgeError("artifact_unknown")
        if entry.disposition is _D.PRESERVE:
            raise FederatedPurgeError("preserved_artifact_deletion_refused")
        if not callable(deleter):
            raise FederatedPurgeError("deleter_required")
        if entry.state == _DELETED:
            return
        self._purge_one(self.manifest, digest, deleter)

    def require_purge_complete(self) -> None:
        """Raise unless every mandatory purge target has been deleted.

        Raises:
            FederatedPurgeBlockedError: ``purge_awaiting_trigger``,
                ``purge_pending`` or ``purge_failed``.
        """

        status = self.status
        if status is not FederatedPurgeStatus.COMPLETE:
            raise FederatedPurgeBlockedError(_BLOCKED_CODES[status])

    def receipt(self) -> FederatedPurgeReceipt:
        """Return the digest-and-count receipt for the current purge state.

        Raises:
            FederatedPurgeError: ``trigger_not_recorded`` before the trigger.
        """

        if self._trigger is None:
            raise FederatedPurgeError("trigger_not_recorded")
        entries = [self._entries[digest] for digest in sorted(self._entries)]
        purge_entries = [e for e in entries if e.disposition in _PURGE_DISPOSITIONS]
        preserved = [e for e in entries if e.disposition is _D.PRESERVE]
        purged = [e for e in purge_entries if e.state == _DELETED]
        mandatory = [e for e in purge_entries if e.disposition is _D.MANDATORY_PURGE]
        plan = self._overall_plan()
        preserved_dependents = 0
        if plan is not None:
            affected = set(plan.affected_hashes)
            preserved_dependents = sum(
                e.artifact.artifact_hash in affected for e in preserved
            )
        return FederatedPurgeReceipt(
            side=self.side,
            trigger=self._trigger,
            status=self.status,
            policy_digest=self._policy.policy_digest,
            round_digest=self._round_digest,
            commitment_digest=self._commitment_digest,
            manifest_digest=None if plan is None else plan.manifest_digest,
            plan_digest=None if plan is None else plan.plan_digest,
            purged_set_digest=stable_hash(
                sorted(e.artifact.artifact_hash for e in purged)
            ),
            preserved_set_digest=stable_hash(
                sorted(e.artifact.artifact_hash for e in preserved)
            ),
            registered_count=len(entries),
            purge_target_count=len(purge_entries),
            mandatory_target_count=len(mandatory),
            purged_count=len(purged),
            pending_count=sum(e.state == _PENDING for e in purge_entries),
            failed_count=sum(e.state == _FAILED for e in mandatory),
            best_effort_failed_count=sum(
                e.state == _FAILED
                for e in purge_entries
                if e.disposition is _D.BEST_EFFORT_PURGE
            ),
            preserved_count=len(preserved),
            preserved_dependent_count=preserved_dependents,
            _purged_by_class=_count(e.artifact_class.value for e in purged),
            _preserved_by_class=_count(e.artifact_class.value for e in preserved),
        )

    def to_state_json(self) -> str:
        """Return the digest-only resume state as deterministic JSON."""

        return json.dumps(
            {
                "schema_version": FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION,
                "side": self.side.value,
                "policy_digest": self._policy.policy_digest,
                "round_digest": self._round_digest,
                "trigger": None if self._trigger is None else self._trigger.value,
                "commitment_digest": self._commitment_digest,
                "artifacts": [
                    {
                        "artifact_hash": digest,
                        "artifact_class": entry.artifact_class.value,
                        "dependencies": list(entry.artifact.dependencies),
                        "state": entry.state,
                    }
                    for digest, entry in sorted(self._entries.items())
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_state_json(
        cls,
        payload: str,
        *,
        policy: FederatedRetentionPolicy,
        state_sink: StateSink | None = None,
    ) -> FederatedArtifactPurge:
        """Restore an orchestrator from resume state after a restart.

        Args:
            payload: JSON produced by :meth:`to_state_json`.
            policy: The same round retention policy; its digest must match.
            state_sink: Optional callback for further checkpoints.

        Returns:
            An orchestrator that resumes the purge on the next :meth:`run`.

        Raises:
            FederatedPurgeError: ``purge_state_invalid`` for malformed,
                tampered or mismatched state.
        """

        if not isinstance(policy, FederatedRetentionPolicy):
            raise FederatedPurgeError("retention_policy_invalid")
        try:
            return cls._restore(payload, policy, state_sink)
        except Exception:
            pass
        raise FederatedPurgeError("purge_state_invalid")

    @classmethod
    def _restore(
        cls,
        payload: Any,
        policy: FederatedRetentionPolicy,
        state_sink: StateSink | None,
    ) -> FederatedArtifactPurge:
        invalid = FederatedPurgeError("purge_state_invalid")
        if not isinstance(payload, str) or len(payload) > _MAX_STATE_CHARS:
            raise invalid
        data = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
        expected = {
            "schema_version",
            "side",
            "policy_digest",
            "round_digest",
            "trigger",
            "commitment_digest",
            "artifacts",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise invalid
        rows = data["artifacts"]
        if (
            data["schema_version"] != FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION
            or data["side"] != policy.side.value
            or data["policy_digest"] != policy.policy_digest
            or not isinstance(data["round_digest"], str)
            or not _DIGEST_RE.fullmatch(data["round_digest"])
            or not isinstance(rows, list)
            or not 0 < len(rows) <= _MAX_ARTIFACTS
        ):
            raise invalid
        restored = cls(policy, round_reference=data["round_digest"])
        entries = [_restore_entry(row, policy) for row in rows]
        digests = [entry.artifact.artifact_hash for entry in entries]
        if digests != sorted(set(digests)):
            raise invalid
        known = set(digests)
        if any(
            dependency not in known
            for entry in entries
            for dependency in entry.artifact.dependencies
        ):
            raise invalid
        restored._entries = {entry.artifact.artifact_hash: entry for entry in entries}
        trigger_value = data["trigger"]
        if trigger_value is None:
            if data["commitment_digest"] is not None:
                raise invalid
        else:
            restored.record_trigger(
                FederatedPurgeTrigger(trigger_value),
                commitment_digest=data["commitment_digest"],
            )
        restored._state_sink = state_sink
        return restored

    def _overall_plan(self) -> DeletionImpactPlan | None:
        targets = [
            digest
            for digest, entry in self._entries.items()
            if entry.disposition in _PURGE_DISPOSITIONS
        ]
        if not targets:
            return None
        try:
            return plan_deletion_impact(self.manifest, targets)
        except DeletionPlanError:
            raise FederatedPurgeError("purge_plan_unsafe") from None

    def _purge_one(
        self,
        manifest: tuple[DeletionArtifact, ...],
        digest: str,
        deleter: DeletionCallback,
    ) -> None:
        entry = self._entries[digest]
        if entry.disposition is _D.PRESERVE:
            raise FederatedPurgeError("preserved_artifact_deletion_refused")
        try:
            plan = plan_deletion_impact(manifest, [digest])
        except DeletionPlanError:
            raise FederatedPurgeError("purge_plan_unsafe") from None

        def _guarded(artifact: DeletionArtifact) -> None:
            guarded = self._entries.get(artifact.artifact_hash)
            if guarded is None or guarded.disposition is _D.PRESERVE:
                raise FederatedPurgeError("preserved_artifact_deletion_refused")
            deleter(artifact)

        try:
            execute_deletion_plan(
                plan,
                confirmation=plan.confirmation_token,
                executor=_guarded,
            )
        except DeletionExecutionError:
            entry.state = _FAILED
        except DeletionPlanError:
            raise FederatedPurgeError("purge_plan_unsafe") from None
        else:
            entry.state = _DELETED
        self._checkpoint()

    def _checkpoint(self) -> None:
        if self._state_sink is None:
            return
        state = self.to_state_json()
        try:
            self._state_sink(state)
        except Exception:
            raise FederatedPurgeError("state_sink_failed") from None


def _restore_entry(row: Any, policy: FederatedRetentionPolicy) -> _Entry:
    invalid = FederatedPurgeError("purge_state_invalid")
    if not isinstance(row, dict) or set(row) != {
        "artifact_hash",
        "artifact_class",
        "dependencies",
        "state",
    }:
        raise invalid
    digest = row["artifact_hash"]
    dependencies = row["dependencies"]
    if (
        not isinstance(digest, str)
        or not _ARTIFACT_DIGEST_RE.fullmatch(digest)
        or not isinstance(dependencies, list)
        or any(
            not isinstance(item, str) or not _ARTIFACT_DIGEST_RE.fullmatch(item)
            for item in dependencies
        )
        or dependencies != sorted(set(dependencies))
    ):
        raise invalid
    artifact_class = _A(row["artifact_class"])
    disposition = policy.disposition_for(artifact_class)
    state = row["state"]
    if disposition is _D.PRESERVE:
        if state != _RETAINED:
            raise invalid
    elif state not in _PURGE_STATES:
        raise invalid
    artifact = DeletionArtifact(
        artifact_hash=digest,
        kind=artifact_class.value,
        retention_class=disposition.value,
        dependencies=tuple(dependencies),
    )
    return _Entry(
        artifact=artifact,
        artifact_class=artifact_class,
        disposition=disposition,
        state=state,
    )


def _validate_trigger(
    side: FederatedPurgeSide,
    trigger: Any,
    commitment_digest: Any,
) -> None:
    if not isinstance(trigger, FederatedPurgeTrigger):
        raise FederatedPurgeError("trigger_invalid")
    if trigger not in _SIDE_TRIGGERS[side]:
        raise FederatedPurgeError("trigger_invalid")
    if commitment_digest is None:
        if trigger is FederatedPurgeTrigger.COMMITMENT_RECORDED:
            raise FederatedPurgeError("trigger_invalid")
        return
    if not isinstance(commitment_digest, str) or not _DIGEST_RE.fullmatch(
        commitment_digest
    ):
        raise FederatedPurgeError("trigger_invalid")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FederatedPurgeError("purge_state_invalid")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise FederatedPurgeError("purge_state_invalid")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FederatedPurgeError("purge_state_invalid")
    return parsed


def emit_federated_round_evidence(
    purge: FederatedArtifactPurge,
    emitter: EvidenceEmitter,
) -> dict[str, Any]:
    """Emit digest-only purge evidence once every mandatory purge completed.

    Args:
        purge: The round's purge orchestrator.
        emitter: Injected callback that publishes or stores the evidence.

    Returns:
        The evidence mapping passed to ``emitter``.

    Raises:
        FederatedPurgeBlockedError: While the purge awaits its trigger, is
            pending, or has failed.  ``emitter`` is not called.
        FederatedPurgeError: ``evidence_emitter_failed`` when the emitter
            raises; its exception text is discarded.
    """

    if not isinstance(purge, FederatedArtifactPurge):
        raise FederatedPurgeError("purge_invalid")
    if not callable(emitter):
        raise FederatedPurgeError("purge_invalid")
    purge.require_purge_complete()
    receipt = purge.receipt()
    evidence = {
        "schema_version": FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION,
        "commitment_digest": receipt.commitment_digest,
        "policy_digest": receipt.policy_digest,
        "round_digest": receipt.round_digest,
        "receipt": receipt.to_dict(),
        "receipt_digest": receipt.receipt_digest,
        "raw_values_included": False,
    }
    try:
        emitter(MappingProxyType(evidence))
    except Exception:
        raise FederatedPurgeError("evidence_emitter_failed") from None
    return evidence


def finalize_federated_round(
    lifecycle: FederatedRoundLifecycle,
    purge: FederatedArtifactPurge,
) -> FederatedRoundLifecycle:
    """Promote a coordinator round only after its mandatory purge completed.

    Args:
        lifecycle: The round lifecycle, normally in the ``evaluating`` state.
        purge: The coordinator purge for the same round.

    Returns:
        The lifecycle in the ``promoted`` state.

    Raises:
        FederatedPurgeBlockedError: While the purge awaits its trigger, is
            pending, or has failed.
        FederatedPurgeError: ``round_finalization_invalid`` for a client purge,
            a purge not triggered by the aggregate commitment, or an invalid
            lifecycle.
        FederatedRoundTransitionError: When promotion is not a legal
            lifecycle transition.
    """

    if not isinstance(lifecycle, FederatedRoundLifecycle) or not isinstance(
        purge, FederatedArtifactPurge
    ):
        raise FederatedPurgeError("round_finalization_invalid")
    purge.require_purge_complete()
    if (
        purge.side is not FederatedPurgeSide.COORDINATOR
        or purge.trigger is not FederatedPurgeTrigger.COMMITMENT_RECORDED
    ):
        raise FederatedPurgeError("round_finalization_invalid")
    return lifecycle.transition_to(FederatedRoundState.PROMOTED)


__all__ = [
    "FEDERATED_ARTIFACT_PURGE_SCHEMA_VERSION",
    "FederatedArtifactPurge",
    "FederatedPurgeBlockedError",
    "FederatedPurgeError",
    "FederatedPurgeReceipt",
    "FederatedPurgeSide",
    "FederatedPurgeStatus",
    "FederatedPurgeTrigger",
    "FederatedRetentionPolicy",
    "PrivateLearningArtifactClass",
    "RetentionDisposition",
    "emit_federated_round_evidence",
    "finalize_federated_round",
]
