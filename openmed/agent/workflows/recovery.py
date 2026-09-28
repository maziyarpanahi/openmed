"""Content-free durable recovery for idempotent workflow effects.

This module records only opaque identifiers, categorical state, and digests. It
does not execute tools, contact FHIR servers, mutate OMOP tables, consume
approval tokens, or automatically compensate a committed clinical effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, cast

from openmed.agent.correlation import ActionId, CorrelationIdError, RunId
from openmed.agent.identifiers import GovernanceIdError, ToolId, WorkflowId

RECOVERY_CHECKPOINT_SCHEMA_VERSION: Final = "openmed.agent.recovery_checkpoint.v1"
RECOVERY_EVIDENCE_SCHEMA_VERSION: Final = "openmed.agent.recovery_evidence.v1"
MAX_CHECKPOINT_BYTES: Final = 1 << 20
_WINDOWS: Final = os.name == "nt"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_IDEMPOTENCY_KEY_RE = re.compile(r"idem_[0-9a-f]{64}")
_CHECKPOINT_FILE_RE = re.compile(r"checkpoint-([0-9]{20})\.json")
_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "workflow_id",
        "run_id",
        "sequence",
        "phase",
        "plan_digest",
        "approval_action_digest",
        "approval_receipt_digest",
        "approval_expires_at",
        "previous_checkpoint_digest",
        "recovery_evidence_digest",
        "effects",
        "checkpoint_digest",
    }
)
_EFFECT_FIELDS = frozenset(
    {
        "ordinal",
        "action_id",
        "tool_id",
        "kind",
        "operation_digest",
        "idempotency_key",
        "approval_required",
        "compensation_limit",
        "state",
        "commit_evidence_digest",
    }
)


class RecoveryError(ValueError):
    """Raised when recovery metadata cannot be trusted.

    Error text contains only a stable code and optional field name. Callers
    should route every failure to human review without logging submitted state.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


class RecoveryPhase(str, Enum):
    """Recovery-specific checkpoint boundaries, not action lifecycle phases."""

    PLANNED = "planned"
    APPROVAL_RECORDED = "approval_recorded"
    DISPATCHING = "dispatching"
    RECONCILING = "reconciling"
    COMPLETED = "completed"
    ABORTED = "aborted"


class EffectKind(str, Enum):
    """Closed set of effect adapters coordinated by recovery."""

    LOCAL_TOOL = "local_tool"
    FHIR_WRITE = "fhir_write"
    OMOP_BATCH = "omop_batch"


class EffectState(str, Enum):
    """Durable knowledge recorded for one intended effect."""

    PENDING = "pending"
    COMMITTED = "committed"


class ObservationState(str, Enum):
    """Result of querying an effect sink by idempotency key."""

    ABSENT = "absent"
    COMMITTED = "committed"
    AMBIGUOUS = "ambiguous"


class CompensationLimit(str, Enum):
    """Maximum recovery response allowed for an already committed effect."""

    NONE = "none"
    PROPOSE_ONLY = "propose_only"


class RecoveryDisposition(str, Enum):
    """A deterministic recovery outcome."""

    RESUME = "resume"
    COMPLETE = "complete"
    REVIEW_REQUIRED = "review_required"


class RecoveryReason(str, Enum):
    """Closed reason codes for recovery decisions."""

    SAFE_TO_RESUME = "safe_to_resume"
    EFFECTS_RECONCILED = "effects_reconciled"
    ALREADY_COMPLETE = "already_complete"
    WORKFLOW_ABORTED = "workflow_aborted"
    AMBIGUOUS_EFFECT = "ambiguous_effect"
    EFFECT_MISMATCH = "effect_mismatch"
    APPROVAL_MISSING = "approval_missing"
    APPROVAL_EXPIRED = "approval_expired"


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """Content-free intent and durable state for one external side effect."""

    ordinal: int
    action_id: ActionId
    tool_id: ToolId
    kind: EffectKind
    operation_digest: str
    idempotency_key: str
    approval_required: bool
    compensation_limit: CompensationLimit
    state: EffectState = EffectState.PENDING
    commit_evidence_digest: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.ordinal, "ordinal")
        if type(self.action_id) is not ActionId:
            raise RecoveryError("invalid_action_id", "action_id")
        if type(self.tool_id) is not ToolId:
            raise RecoveryError("invalid_tool_id", "tool_id")
        if not isinstance(self.kind, EffectKind):
            raise RecoveryError("unknown_effect_kind", "kind")
        _require_digest(self.operation_digest, "operation_digest")
        _require_idempotency_key(self.idempotency_key)
        if type(self.approval_required) is not bool:
            raise RecoveryError("invalid_boolean", "approval_required")
        if not isinstance(self.compensation_limit, CompensationLimit):
            raise RecoveryError("unknown_compensation_limit", "compensation_limit")
        if not isinstance(self.state, EffectState):
            raise RecoveryError("unknown_effect_state", "state")
        if self.state is EffectState.PENDING:
            if self.commit_evidence_digest is not None:
                raise RecoveryError("unexpected_commit_evidence", "effect")
        elif self.commit_evidence_digest is None:
            raise RecoveryError("missing_commit_evidence", "effect")
        else:
            _require_digest(self.commit_evidence_digest, "commit_evidence_digest")

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        run_id: RunId,
        action_id: ActionId,
        tool_id: ToolId,
        kind: EffectKind,
        operation_digest: str,
        approval_required: bool,
        compensation_limit: CompensationLimit,
    ) -> "EffectRecord":
        """Create a pending effect with its deterministic cross-tool key."""

        return cls(
            ordinal=ordinal,
            action_id=action_id,
            tool_id=tool_id,
            kind=kind,
            operation_digest=operation_digest,
            idempotency_key=derive_idempotency_key(
                run_id=run_id,
                action_id=action_id,
                tool_id=tool_id,
                kind=kind,
                operation_digest=operation_digest,
            ),
            approval_required=approval_required,
            compensation_limit=compensation_limit,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical metadata-only effect fields."""

        return {
            "action_id": self.action_id.serialize(),
            "approval_required": self.approval_required,
            "commit_evidence_digest": self.commit_evidence_digest,
            "compensation_limit": self.compensation_limit.value,
            "idempotency_key": self.idempotency_key,
            "kind": self.kind.value,
            "operation_digest": self.operation_digest,
            "ordinal": self.ordinal,
            "state": self.state.value,
            "tool_id": self.tool_id.serialize(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectRecord":
        """Restore one effect from an exact mapping."""

        values = _read_exact_mapping(payload, _EFFECT_FIELDS, "effect")
        try:
            return cls(
                ordinal=cast(int, values["ordinal"]),
                action_id=ActionId.parse(values["action_id"]),
                tool_id=ToolId.parse(values["tool_id"]),
                kind=EffectKind(values["kind"]),
                operation_digest=cast(str, values["operation_digest"]),
                idempotency_key=cast(str, values["idempotency_key"]),
                approval_required=cast(bool, values["approval_required"]),
                compensation_limit=CompensationLimit(values["compensation_limit"]),
                state=EffectState(values["state"]),
                commit_evidence_digest=cast(
                    str | None, values["commit_evidence_digest"]
                ),
            )
        except RecoveryError:
            raise
        except (CorrelationIdError, GovernanceIdError, ValueError, TypeError):
            raise RecoveryError("invalid_effect", "effect") from None


@dataclass(frozen=True, slots=True)
class EffectObservation:
    """Content-free sink observation used to reconcile one effect."""

    action_id: ActionId
    operation_digest: str
    idempotency_key: str
    state: ObservationState
    commit_evidence_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.action_id) is not ActionId:
            raise RecoveryError("invalid_action_id", "action_id")
        _require_digest(self.operation_digest, "operation_digest")
        _require_idempotency_key(self.idempotency_key)
        if not isinstance(self.state, ObservationState):
            raise RecoveryError("unknown_observation_state", "state")
        if self.state is ObservationState.COMMITTED:
            if self.commit_evidence_digest is None:
                raise RecoveryError("missing_commit_evidence", "observation")
            _require_digest(self.commit_evidence_digest, "commit_evidence_digest")
        elif self.commit_evidence_digest is not None:
            raise RecoveryError("unexpected_commit_evidence", "observation")


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryCheckpoint:
    """Hash-bound, content-free snapshot in an append-only lineage."""

    workflow_id: WorkflowId
    run_id: RunId
    sequence: int
    phase: RecoveryPhase
    plan_digest: str
    effects: tuple[EffectRecord, ...]
    checkpoint_digest: str
    approval_action_digest: str | None = None
    approval_receipt_digest: str | None = None
    approval_expires_at: int | None = None
    previous_checkpoint_digest: str | None = None
    recovery_evidence_digest: str | None = None
    schema_version: str = RECOVERY_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_CHECKPOINT_SCHEMA_VERSION:
            raise RecoveryError("unsupported_schema_version", "schema_version")
        if type(self.workflow_id) is not WorkflowId:
            raise RecoveryError("invalid_workflow_id", "workflow_id")
        if type(self.run_id) is not RunId:
            raise RecoveryError("invalid_run_id", "run_id")
        _require_non_negative_int(self.sequence, "sequence")
        if not isinstance(self.phase, RecoveryPhase):
            raise RecoveryError("unknown_phase", "phase")
        _require_digest(self.plan_digest, "plan_digest")
        _validate_effects(self.effects, self.run_id)
        _validate_approval_fields(
            self.approval_action_digest,
            self.approval_receipt_digest,
            self.approval_expires_at,
            self.plan_digest,
        )
        if (
            self.phase is RecoveryPhase.APPROVAL_RECORDED
            and self.approval_receipt_digest is None
        ):
            raise RecoveryError("missing_approval_receipt", "approval")
        if (
            self.phase
            in {
                RecoveryPhase.DISPATCHING,
                RecoveryPhase.COMPLETED,
            }
            and any(effect.approval_required for effect in self.effects)
            and self.approval_receipt_digest is None
        ):
            raise RecoveryError("missing_approval_receipt", "approval")
        if self.phase is RecoveryPhase.COMPLETED and any(
            effect.state is not EffectState.COMMITTED for effect in self.effects
        ):
            raise RecoveryError("incomplete_effects", "effects")
        if self.sequence == 0:
            if self.previous_checkpoint_digest is not None:
                raise RecoveryError("unexpected_previous_digest", "checkpoint")
        elif self.previous_checkpoint_digest is None:
            raise RecoveryError("missing_previous_digest", "checkpoint")
        else:
            _require_digest(
                self.previous_checkpoint_digest, "previous_checkpoint_digest"
            )
        if self.recovery_evidence_digest is not None:
            _require_digest(self.recovery_evidence_digest, "recovery_evidence_digest")
        _require_digest(self.checkpoint_digest, "checkpoint_digest")
        if self.checkpoint_digest != _digest(self._unsigned_dict()):
            raise RecoveryError("checkpoint_digest_mismatch", "checkpoint_digest")

    @classmethod
    def create(
        cls,
        *,
        workflow_id: WorkflowId,
        run_id: RunId,
        sequence: int,
        phase: RecoveryPhase,
        plan_digest: str,
        effects: Iterable[EffectRecord],
        approval_action_digest: str | None = None,
        approval_receipt_digest: str | None = None,
        approval_expires_at: int | None = None,
        previous_checkpoint_digest: str | None = None,
        recovery_evidence_digest: str | None = None,
    ) -> "RecoveryCheckpoint":
        """Create a checkpoint and bind every field into its digest."""

        normalized_effects = _normalize_effects(effects)
        values: dict[str, Any] = {
            "schema_version": RECOVERY_CHECKPOINT_SCHEMA_VERSION,
            "workflow_id": workflow_id,
            "run_id": run_id,
            "sequence": sequence,
            "phase": phase,
            "plan_digest": plan_digest,
            "approval_action_digest": approval_action_digest,
            "approval_receipt_digest": approval_receipt_digest,
            "approval_expires_at": approval_expires_at,
            "previous_checkpoint_digest": previous_checkpoint_digest,
            "recovery_evidence_digest": recovery_evidence_digest,
            "effects": normalized_effects,
        }
        unsigned = _checkpoint_dict(values, include_digest=False)
        return cls(checkpoint_digest=_digest(unsigned), **values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecoveryCheckpoint":
        """Restore a checkpoint from an exact mapping without normalizing it."""

        values = _read_exact_mapping(payload, _CHECKPOINT_FIELDS, "checkpoint")
        raw_effects = values["effects"]
        if not isinstance(raw_effects, list):
            raise RecoveryError("invalid_effects", "effects")
        try:
            return cls(
                workflow_id=WorkflowId.parse(values["workflow_id"]),
                run_id=RunId.parse(values["run_id"]),
                sequence=cast(int, values["sequence"]),
                phase=RecoveryPhase(values["phase"]),
                plan_digest=cast(str, values["plan_digest"]),
                effects=tuple(EffectRecord.from_dict(item) for item in raw_effects),
                checkpoint_digest=cast(str, values["checkpoint_digest"]),
                approval_action_digest=cast(
                    str | None, values["approval_action_digest"]
                ),
                approval_receipt_digest=cast(
                    str | None, values["approval_receipt_digest"]
                ),
                approval_expires_at=cast(int | None, values["approval_expires_at"]),
                previous_checkpoint_digest=cast(
                    str | None, values["previous_checkpoint_digest"]
                ),
                recovery_evidence_digest=cast(
                    str | None, values["recovery_evidence_digest"]
                ),
                schema_version=cast(str, values["schema_version"]),
            )
        except RecoveryError:
            raise
        except (CorrelationIdError, GovernanceIdError, ValueError, TypeError):
            raise RecoveryError("invalid_checkpoint", "checkpoint") from None

    @classmethod
    def from_json(cls, serialized: str | bytes | bytearray) -> "RecoveryCheckpoint":
        """Restore a checkpoint from strict JSON."""

        return cls.from_dict(_parse_json(serialized, "checkpoint"))

    def _unsigned_dict(self) -> dict[str, Any]:
        return _checkpoint_dict(self.__dict_like(), include_digest=False)

    def __dict_like(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "phase": self.phase,
            "plan_digest": self.plan_digest,
            "approval_action_digest": self.approval_action_digest,
            "approval_receipt_digest": self.approval_receipt_digest,
            "approval_expires_at": self.approval_expires_at,
            "previous_checkpoint_digest": self.previous_checkpoint_digest,
            "recovery_evidence_digest": self.recovery_evidence_digest,
            "effects": self.effects,
            "checkpoint_digest": self.checkpoint_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible checkpoint fields."""

        return _checkpoint_dict(self.__dict_like(), include_digest=True)

    def to_json(self) -> str:
        """Serialize the checkpoint deterministically."""

        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        return (
            "RecoveryCheckpoint("
            f"sequence={self.sequence}, phase={self.phase.value!r}, "
            f"effect_count={len(self.effects)})"
        )


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Deterministic evidence describing whether and how recovery may proceed."""

    disposition: RecoveryDisposition
    reason: RecoveryReason
    source_checkpoint_digest: str
    retry_effect_ids: tuple[str, ...]
    retry_idempotency_keys: tuple[str, ...]
    committed_effects: tuple[tuple[str, str], ...]
    compensation_effect_ids: tuple[str, ...]
    evidence_digest: str
    schema_version: str = RECOVERY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_EVIDENCE_SCHEMA_VERSION:
            raise RecoveryError("unsupported_evidence_version", "schema_version")
        if not isinstance(self.disposition, RecoveryDisposition):
            raise RecoveryError("unknown_disposition", "disposition")
        if not isinstance(self.reason, RecoveryReason):
            raise RecoveryError("unknown_reason", "reason")
        _require_digest(self.source_checkpoint_digest, "source_checkpoint_digest")
        _validate_decision_sequences(self)
        _require_digest(self.evidence_digest, "evidence_digest")
        if self.evidence_digest != _digest(self._unsigned_dict()):
            raise RecoveryError("evidence_digest_mismatch", "evidence_digest")

    @classmethod
    def create(
        cls,
        *,
        disposition: RecoveryDisposition,
        reason: RecoveryReason,
        source_checkpoint_digest: str,
        retry_effect_ids: tuple[str, ...] = (),
        retry_idempotency_keys: tuple[str, ...] = (),
        committed_effects: tuple[tuple[str, str], ...] = (),
        compensation_effect_ids: tuple[str, ...] = (),
    ) -> "RecoveryDecision":
        """Create digest-bound recovery evidence."""

        values: dict[str, Any] = {
            "schema_version": RECOVERY_EVIDENCE_SCHEMA_VERSION,
            "disposition": disposition,
            "reason": reason,
            "source_checkpoint_digest": source_checkpoint_digest,
            "retry_effect_ids": retry_effect_ids,
            "retry_idempotency_keys": retry_idempotency_keys,
            "committed_effects": committed_effects,
            "compensation_effect_ids": compensation_effect_ids,
        }
        unsigned = _decision_dict(values, include_digest=False)
        return cls(evidence_digest=_digest(unsigned), **values)

    def _unsigned_dict(self) -> dict[str, Any]:
        return _decision_dict(
            {
                "schema_version": self.schema_version,
                "disposition": self.disposition,
                "reason": self.reason,
                "source_checkpoint_digest": self.source_checkpoint_digest,
                "retry_effect_ids": self.retry_effect_ids,
                "retry_idempotency_keys": self.retry_idempotency_keys,
                "committed_effects": self.committed_effects,
                "compensation_effect_ids": self.compensation_effect_ids,
            },
            include_digest=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic content-free recovery evidence."""

        values = self._unsigned_dict()
        values["evidence_digest"] = self.evidence_digest
        return values

    def to_json(self) -> str:
        """Serialize recovery evidence deterministically."""

        return _canonical_json(self.to_dict())


class CheckpointJournal:
    """Durable append-only checkpoint storage in a caller-owned directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        try:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            mode = self._directory.lstat().st_mode
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise RecoveryError("journal_unavailable", "journal") from None
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise RecoveryError("unsafe_journal", "journal")

    def load(self) -> tuple[RecoveryCheckpoint, ...]:
        """Load and validate the complete checkpoint lineage."""

        try:
            entries = tuple(self._directory.iterdir())
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise RecoveryError("journal_unreadable", "journal") from None

        numbered: list[tuple[int, Path]] = []
        for entry in entries:
            match = _CHECKPOINT_FILE_RE.fullmatch(entry.name)
            if match is None:
                if entry.name.startswith(".checkpoint-"):
                    continue
                raise RecoveryError("unexpected_journal_entry", "journal")
            numbered.append((int(match.group(1)), entry))
        numbered.sort(key=lambda item: item[0])

        checkpoints: list[RecoveryCheckpoint] = []
        for expected_sequence, (sequence, path) in enumerate(numbered):
            if sequence != expected_sequence:
                raise RecoveryError("checkpoint_sequence_gap", "journal")
            checkpoints.append(self._read_checkpoint(path))
        return validate_checkpoint_lineage(checkpoints, allow_empty=True)

    def append(self, checkpoint: RecoveryCheckpoint) -> RecoveryCheckpoint:
        """Atomically append one checkpoint, accepting an identical retry."""

        if type(checkpoint) is not RecoveryCheckpoint:
            raise RecoveryError("invalid_checkpoint", "checkpoint")
        current = self.load()
        if checkpoint.sequence < len(current):
            if current[checkpoint.sequence] == checkpoint:
                return checkpoint
            raise RecoveryError("checkpoint_conflict", "checkpoint")
        if checkpoint.sequence != len(current):
            raise RecoveryError("checkpoint_sequence_gap", "checkpoint")
        validate_checkpoint_lineage((*current, checkpoint))

        target = self._directory / _checkpoint_filename(checkpoint.sequence)
        serialized = checkpoint.to_json().encode("ascii")
        if len(serialized) > MAX_CHECKPOINT_BYTES:
            raise RecoveryError("checkpoint_too_large", "checkpoint")
        descriptor: int | None = None
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".checkpoint-", dir=self._directory
            )
            # mkstemp creates a private file; Windows has no os.fchmod.
            if not _WINDOWS:
                os.fchmod(descriptor, 0o600)
            _write_all(descriptor, serialized)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.link(temporary, target)
            _fsync_directory(self._directory)
        except FileExistsError:
            existing = self._read_checkpoint(target)
            if existing != checkpoint:
                raise RecoveryError("checkpoint_conflict", "checkpoint") from None
        except RecoveryError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise RecoveryError("checkpoint_write_failed", "checkpoint") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
        return checkpoint

    def _read_checkpoint(self, path: Path) -> RecoveryCheckpoint:
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RecoveryError("unsafe_checkpoint_file", "journal")
            if metadata.st_size > MAX_CHECKPOINT_BYTES:
                raise RecoveryError("checkpoint_too_large", "journal")
            payload = path.read_bytes()
        except RecoveryError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise RecoveryError("checkpoint_read_failed", "journal") from None
        checkpoint = RecoveryCheckpoint.from_json(payload)
        expected_name = _checkpoint_filename(checkpoint.sequence)
        if path.name != expected_name:
            raise RecoveryError("checkpoint_filename_mismatch", "journal")
        return checkpoint


def derive_idempotency_key(
    *,
    run_id: RunId,
    action_id: ActionId,
    tool_id: ToolId,
    kind: EffectKind,
    operation_digest: str,
) -> str:
    """Derive one stable key for a logical effect across adapter retries."""

    if type(run_id) is not RunId:
        raise RecoveryError("invalid_run_id", "run_id")
    if type(action_id) is not ActionId:
        raise RecoveryError("invalid_action_id", "action_id")
    if type(tool_id) is not ToolId:
        raise RecoveryError("invalid_tool_id", "tool_id")
    if not isinstance(kind, EffectKind):
        raise RecoveryError("unknown_effect_kind", "kind")
    _require_digest(operation_digest, "operation_digest")
    payload = {
        "action_id": action_id.serialize(),
        "kind": kind.value,
        "operation_digest": operation_digest,
        "run_id": run_id.serialize(),
        "tool_id": tool_id.serialize(),
    }
    return (
        f"idem_{hashlib.sha256(_canonical_json(payload).encode('ascii')).hexdigest()}"
    )


def validate_checkpoint_lineage(
    checkpoints: Iterable[RecoveryCheckpoint], *, allow_empty: bool = False
) -> tuple[RecoveryCheckpoint, ...]:
    """Validate append-only identity, state, approval, and digest continuity."""

    if isinstance(checkpoints, (str, bytes, bytearray, Mapping)):
        raise RecoveryError("invalid_lineage", "lineage")
    try:
        lineage = tuple(checkpoints)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise RecoveryError("invalid_lineage", "lineage") from None
    if not lineage:
        if allow_empty:
            return ()
        raise RecoveryError("empty_lineage", "lineage")
    if any(type(item) is not RecoveryCheckpoint for item in lineage):
        raise RecoveryError("invalid_checkpoint", "lineage")

    for index, checkpoint in enumerate(lineage):
        if checkpoint.sequence != index:
            raise RecoveryError("checkpoint_sequence_gap", "lineage")
        if index == 0:
            continue
        previous = lineage[index - 1]
        if checkpoint.previous_checkpoint_digest != previous.checkpoint_digest:
            raise RecoveryError("lineage_digest_mismatch", "lineage")
        _validate_checkpoint_successor(previous, checkpoint)
    return lineage


def recover_workflow(
    checkpoints: Iterable[RecoveryCheckpoint],
    observations: Iterable[EffectObservation],
    *,
    now: int,
) -> RecoveryDecision:
    """Reconcile all intended effects and return a fail-closed recovery plan.

    The caller must query each adapter by the recorded idempotency key before
    calling this function. No callback or network access occurs here.
    """

    lineage = validate_checkpoint_lineage(checkpoints)
    current = lineage[-1]
    _require_non_negative_int(now, "now")
    if current.phase is RecoveryPhase.COMPLETED:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.COMPLETE,
            reason=RecoveryReason.ALREADY_COMPLETE,
            source_checkpoint_digest=current.checkpoint_digest,
        )
    if current.phase is RecoveryPhase.ABORTED:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.REVIEW_REQUIRED,
            reason=RecoveryReason.WORKFLOW_ABORTED,
            source_checkpoint_digest=current.checkpoint_digest,
        )

    normalized = _normalize_observations(observations, current.effects)
    committed: list[tuple[str, str]] = []
    retry_ids: list[str] = []
    retry_keys: list[str] = []
    ambiguous = False
    mismatch = False
    for effect, observation in zip(current.effects, normalized, strict=True):
        if (
            observation.operation_digest != effect.operation_digest
            or observation.idempotency_key != effect.idempotency_key
        ):
            mismatch = True
            continue
        if observation.state is ObservationState.AMBIGUOUS:
            ambiguous = True
            continue
        if effect.state is EffectState.COMMITTED:
            if (
                observation.state is not ObservationState.COMMITTED
                or observation.commit_evidence_digest != effect.commit_evidence_digest
            ):
                ambiguous = True
            else:
                committed.append(
                    (effect.action_id.serialize(), effect.commit_evidence_digest)
                )
            continue
        if observation.state is ObservationState.COMMITTED:
            committed.append(
                (
                    effect.action_id.serialize(),
                    cast(str, observation.commit_evidence_digest),
                )
            )
        else:
            retry_ids.append(effect.action_id.serialize())
            retry_keys.append(effect.idempotency_key)

    compensation_ids = tuple(
        effect.action_id.serialize()
        for effect in current.effects
        if effect.compensation_limit is CompensationLimit.PROPOSE_ONLY
        and any(effect.action_id.serialize() == item[0] for item in committed)
    )
    if mismatch:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.REVIEW_REQUIRED,
            reason=RecoveryReason.EFFECT_MISMATCH,
            source_checkpoint_digest=current.checkpoint_digest,
            committed_effects=tuple(committed),
            compensation_effect_ids=compensation_ids,
        )
    if ambiguous:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.REVIEW_REQUIRED,
            reason=RecoveryReason.AMBIGUOUS_EFFECT,
            source_checkpoint_digest=current.checkpoint_digest,
            committed_effects=tuple(committed),
            compensation_effect_ids=compensation_ids,
        )

    approval_needed = any(
        effect.approval_required
        for effect in current.effects
        if effect.action_id.serialize() in retry_ids
    )
    if approval_needed and current.approval_receipt_digest is None:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.REVIEW_REQUIRED,
            reason=RecoveryReason.APPROVAL_MISSING,
            source_checkpoint_digest=current.checkpoint_digest,
            committed_effects=tuple(committed),
            compensation_effect_ids=compensation_ids,
        )
    if approval_needed and cast(int, current.approval_expires_at) <= now:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.REVIEW_REQUIRED,
            reason=RecoveryReason.APPROVAL_EXPIRED,
            source_checkpoint_digest=current.checkpoint_digest,
            committed_effects=tuple(committed),
            compensation_effect_ids=compensation_ids,
        )
    if retry_ids:
        return RecoveryDecision.create(
            disposition=RecoveryDisposition.RESUME,
            reason=RecoveryReason.SAFE_TO_RESUME,
            source_checkpoint_digest=current.checkpoint_digest,
            retry_effect_ids=tuple(retry_ids),
            retry_idempotency_keys=tuple(retry_keys),
            committed_effects=tuple(committed),
        )
    return RecoveryDecision.create(
        disposition=RecoveryDisposition.COMPLETE,
        reason=RecoveryReason.EFFECTS_RECONCILED,
        source_checkpoint_digest=current.checkpoint_digest,
        committed_effects=tuple(committed),
    )


def advance_checkpoint(
    checkpoint: RecoveryCheckpoint, decision: RecoveryDecision
) -> RecoveryCheckpoint:
    """Append deterministic recovery evidence and reconciled commit state."""

    if type(checkpoint) is not RecoveryCheckpoint:
        raise RecoveryError("invalid_checkpoint", "checkpoint")
    if type(decision) is not RecoveryDecision:
        raise RecoveryError("invalid_decision", "decision")
    if decision.source_checkpoint_digest != checkpoint.checkpoint_digest:
        raise RecoveryError("decision_checkpoint_mismatch", "decision")
    if checkpoint.phase is RecoveryPhase.COMPLETED:
        if (
            decision.disposition is not RecoveryDisposition.COMPLETE
            or decision.reason is not RecoveryReason.ALREADY_COMPLETE
        ):
            raise RecoveryError("terminal_decision_mismatch", "decision")
        return checkpoint
    if checkpoint.phase is RecoveryPhase.ABORTED:
        if (
            decision.disposition is not RecoveryDisposition.REVIEW_REQUIRED
            or decision.reason is not RecoveryReason.WORKFLOW_ABORTED
        ):
            raise RecoveryError("terminal_decision_mismatch", "decision")
        return checkpoint

    effects_by_id = {
        effect.action_id.serialize(): effect for effect in checkpoint.effects
    }
    committed = dict(decision.committed_effects)
    if not set(committed).issubset(effects_by_id):
        raise RecoveryError("unknown_committed_effect", "decision")
    if not set(decision.retry_effect_ids).issubset(effects_by_id):
        raise RecoveryError("unknown_retry_effect", "decision")
    if not set(decision.compensation_effect_ids).issubset(committed):
        raise RecoveryError("invalid_compensation_effect", "decision")
    expected_retry_keys = tuple(
        effects_by_id[action_id].idempotency_key
        for action_id in decision.retry_effect_ids
    )
    if decision.retry_idempotency_keys != expected_retry_keys:
        raise RecoveryError("retry_key_mismatch", "decision")
    for action_id, evidence_digest in committed.items():
        effect = effects_by_id[action_id]
        if (
            effect.state is EffectState.COMMITTED
            and effect.commit_evidence_digest != evidence_digest
        ):
            raise RecoveryError("commit_evidence_changed", "decision")

    effects = tuple(
        replace(
            effect,
            state=EffectState.COMMITTED,
            commit_evidence_digest=committed[effect.action_id.serialize()],
        )
        if effect.action_id.serialize() in committed
        else effect
        for effect in checkpoint.effects
    )
    if decision.disposition is RecoveryDisposition.COMPLETE:
        phase = RecoveryPhase.COMPLETED
    elif decision.disposition is RecoveryDisposition.RESUME:
        phase = RecoveryPhase.DISPATCHING
    else:
        phase = RecoveryPhase.RECONCILING
    return RecoveryCheckpoint.create(
        workflow_id=checkpoint.workflow_id,
        run_id=checkpoint.run_id,
        sequence=checkpoint.sequence + 1,
        phase=phase,
        plan_digest=checkpoint.plan_digest,
        effects=effects,
        approval_action_digest=checkpoint.approval_action_digest,
        approval_receipt_digest=checkpoint.approval_receipt_digest,
        approval_expires_at=checkpoint.approval_expires_at,
        previous_checkpoint_digest=checkpoint.checkpoint_digest,
        recovery_evidence_digest=decision.evidence_digest,
    )


def _validate_effects(effects: Any, run_id: RunId) -> None:
    if not isinstance(effects, tuple) or not effects:
        raise RecoveryError("invalid_effects", "effects")
    if any(type(effect) is not EffectRecord for effect in effects):
        raise RecoveryError("invalid_effect", "effects")
    if tuple(effect.ordinal for effect in effects) != tuple(range(len(effects))):
        raise RecoveryError("invalid_effect_order", "effects")
    action_ids = tuple(effect.action_id for effect in effects)
    keys = tuple(effect.idempotency_key for effect in effects)
    if len(set(action_ids)) != len(action_ids):
        raise RecoveryError("duplicate_action_id", "effects")
    if len(set(keys)) != len(keys):
        raise RecoveryError("duplicate_idempotency_key", "effects")
    for effect in effects:
        expected = derive_idempotency_key(
            run_id=run_id,
            action_id=effect.action_id,
            tool_id=effect.tool_id,
            kind=effect.kind,
            operation_digest=effect.operation_digest,
        )
        if effect.idempotency_key != expected:
            raise RecoveryError("idempotency_key_mismatch", "effects")


def _normalize_effects(effects: Iterable[EffectRecord]) -> tuple[EffectRecord, ...]:
    if isinstance(effects, (str, bytes, bytearray, Mapping)):
        raise RecoveryError("invalid_effects", "effects")
    try:
        values = tuple(effects)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise RecoveryError("invalid_effects", "effects") from None
    return values


def _validate_approval_fields(
    action_digest: Any,
    receipt_digest: Any,
    expires_at: Any,
    plan_digest: str,
) -> None:
    values = (action_digest, receipt_digest, expires_at)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise RecoveryError("incomplete_approval_receipt", "approval")
    _require_digest(action_digest, "approval_action_digest")
    _require_digest(receipt_digest, "approval_receipt_digest")
    _require_non_negative_int(expires_at, "approval_expires_at")
    if action_digest != plan_digest:
        raise RecoveryError("approval_action_mismatch", "approval_action_digest")


def _validate_checkpoint_successor(
    previous: RecoveryCheckpoint, checkpoint: RecoveryCheckpoint
) -> None:
    if (
        checkpoint.workflow_id != previous.workflow_id
        or checkpoint.run_id != previous.run_id
        or checkpoint.plan_digest != previous.plan_digest
    ):
        raise RecoveryError("workflow_identity_changed", "lineage")
    if (
        checkpoint.approval_action_digest != previous.approval_action_digest
        or checkpoint.approval_receipt_digest != previous.approval_receipt_digest
        or checkpoint.approval_expires_at != previous.approval_expires_at
    ):
        if previous.approval_receipt_digest is not None:
            raise RecoveryError("approval_receipt_changed", "lineage")
        if checkpoint.phase is not RecoveryPhase.APPROVAL_RECORDED:
            raise RecoveryError("approval_added_out_of_phase", "lineage")
    _validate_phase_transition(previous.phase, checkpoint.phase)
    if len(checkpoint.effects) != len(previous.effects):
        raise RecoveryError("effect_set_changed", "lineage")
    for old, new in zip(previous.effects, checkpoint.effects, strict=True):
        if _effect_identity(old) != _effect_identity(new):
            raise RecoveryError("effect_identity_changed", "lineage")
        if old.state is EffectState.COMMITTED and new != old:
            raise RecoveryError("committed_effect_changed", "lineage")
        if old.state is EffectState.PENDING and new.state is EffectState.COMMITTED:
            continue
        if old != new:
            raise RecoveryError("invalid_effect_transition", "lineage")


def _validate_phase_transition(old: RecoveryPhase, new: RecoveryPhase) -> None:
    allowed = {
        RecoveryPhase.PLANNED: frozenset(
            {
                RecoveryPhase.PLANNED,
                RecoveryPhase.APPROVAL_RECORDED,
                RecoveryPhase.DISPATCHING,
                RecoveryPhase.RECONCILING,
                RecoveryPhase.ABORTED,
            }
        ),
        RecoveryPhase.APPROVAL_RECORDED: frozenset(
            {
                RecoveryPhase.APPROVAL_RECORDED,
                RecoveryPhase.DISPATCHING,
                RecoveryPhase.RECONCILING,
                RecoveryPhase.ABORTED,
            }
        ),
        RecoveryPhase.DISPATCHING: frozenset(
            {
                RecoveryPhase.DISPATCHING,
                RecoveryPhase.RECONCILING,
                RecoveryPhase.COMPLETED,
                RecoveryPhase.ABORTED,
            }
        ),
        RecoveryPhase.RECONCILING: frozenset(
            {
                RecoveryPhase.DISPATCHING,
                RecoveryPhase.RECONCILING,
                RecoveryPhase.COMPLETED,
                RecoveryPhase.ABORTED,
            }
        ),
        RecoveryPhase.COMPLETED: frozenset(),
        RecoveryPhase.ABORTED: frozenset(),
    }
    if new not in allowed[old]:
        raise RecoveryError("invalid_phase_transition", "lineage")


def _effect_identity(effect: EffectRecord) -> tuple[Any, ...]:
    return (
        effect.ordinal,
        effect.action_id,
        effect.tool_id,
        effect.kind,
        effect.operation_digest,
        effect.idempotency_key,
        effect.approval_required,
        effect.compensation_limit,
    )


def _normalize_observations(
    observations: Iterable[EffectObservation], effects: tuple[EffectRecord, ...]
) -> tuple[EffectObservation, ...]:
    if isinstance(observations, (str, bytes, bytearray, Mapping)):
        raise RecoveryError("invalid_observations", "observations")
    try:
        values = tuple(observations)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise RecoveryError("invalid_observations", "observations") from None
    if any(type(value) is not EffectObservation for value in values):
        raise RecoveryError("invalid_observation", "observations")
    by_action = {value.action_id: value for value in values}
    if len(by_action) != len(values):
        raise RecoveryError("duplicate_observation", "observations")
    expected = {effect.action_id for effect in effects}
    if set(by_action) != expected:
        raise RecoveryError("observation_coverage_mismatch", "observations")
    return tuple(by_action[effect.action_id] for effect in effects)


def _validate_decision_sequences(decision: RecoveryDecision) -> None:
    fields: Sequence[tuple[str, tuple[str, ...], re.Pattern[str]]] = (
        (
            "retry_effect_ids",
            decision.retry_effect_ids,
            re.compile(r"act_[0-9a-f]{32}"),
        ),
        (
            "retry_idempotency_keys",
            decision.retry_idempotency_keys,
            _IDEMPOTENCY_KEY_RE,
        ),
        (
            "compensation_effect_ids",
            decision.compensation_effect_ids,
            re.compile(r"act_[0-9a-f]{32}"),
        ),
    )
    for field_name, values, pattern in fields:
        if not isinstance(values, tuple) or any(
            type(value) is not str or pattern.fullmatch(value) is None
            for value in values
        ):
            raise RecoveryError("invalid_sequence", field_name)
        if len(set(values)) != len(values):
            raise RecoveryError("duplicate_item", field_name)
    if len(decision.retry_effect_ids) != len(decision.retry_idempotency_keys):
        raise RecoveryError("retry_count_mismatch", "retry_effect_ids")
    if decision.disposition is RecoveryDisposition.RESUME:
        if (
            decision.reason is not RecoveryReason.SAFE_TO_RESUME
            or not decision.retry_effect_ids
            or decision.compensation_effect_ids
        ):
            raise RecoveryError("invalid_resume_decision", "decision")
    elif decision.disposition is RecoveryDisposition.COMPLETE:
        if (
            decision.reason
            not in {
                RecoveryReason.EFFECTS_RECONCILED,
                RecoveryReason.ALREADY_COMPLETE,
            }
            or decision.retry_effect_ids
            or decision.compensation_effect_ids
        ):
            raise RecoveryError("invalid_complete_decision", "decision")
    elif (
        decision.reason
        not in {
            RecoveryReason.WORKFLOW_ABORTED,
            RecoveryReason.AMBIGUOUS_EFFECT,
            RecoveryReason.EFFECT_MISMATCH,
            RecoveryReason.APPROVAL_MISSING,
            RecoveryReason.APPROVAL_EXPIRED,
        }
        or decision.retry_effect_ids
    ):
        raise RecoveryError("invalid_review_decision", "decision")
    if not isinstance(decision.committed_effects, tuple):
        raise RecoveryError("invalid_sequence", "committed_effects")
    seen: set[str] = set()
    for item in decision.committed_effects:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not str
            or re.fullmatch(r"act_[0-9a-f]{32}", item[0]) is None
        ):
            raise RecoveryError("invalid_committed_effect", "committed_effects")
        _require_digest(item[1], "committed_effects")
        if item[0] in seen:
            raise RecoveryError("duplicate_item", "committed_effects")
        seen.add(item[0])


def _checkpoint_dict(
    values: Mapping[str, Any], *, include_digest: bool
) -> dict[str, Any]:
    result = {
        "approval_action_digest": values["approval_action_digest"],
        "approval_expires_at": values["approval_expires_at"],
        "approval_receipt_digest": values["approval_receipt_digest"],
        "effects": [effect.to_dict() for effect in values["effects"]],
        "phase": values["phase"].value,
        "plan_digest": values["plan_digest"],
        "previous_checkpoint_digest": values["previous_checkpoint_digest"],
        "recovery_evidence_digest": values["recovery_evidence_digest"],
        "run_id": values["run_id"].serialize(),
        "schema_version": values["schema_version"],
        "sequence": values["sequence"],
        "workflow_id": values["workflow_id"].serialize(),
    }
    if include_digest:
        result["checkpoint_digest"] = values["checkpoint_digest"]
    return result


def _decision_dict(
    values: Mapping[str, Any], *, include_digest: bool
) -> dict[str, Any]:
    result = {
        "committed_effects": [
            {"action_id": action_id, "commit_evidence_digest": digest}
            for action_id, digest in values["committed_effects"]
        ],
        "compensation_effect_ids": list(values["compensation_effect_ids"]),
        "disposition": values["disposition"].value,
        "reason": values["reason"].value,
        "retry_effect_ids": list(values["retry_effect_ids"]),
        "retry_idempotency_keys": list(values["retry_idempotency_keys"]),
        "schema_version": values["schema_version"],
        "source_checkpoint_digest": values["source_checkpoint_digest"],
    }
    if include_digest:
        result["evidence_digest"] = values["evidence_digest"]
    return result


def _read_exact_mapping(
    payload: Mapping[str, Any], expected: frozenset[str], field_name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes, bytearray)):
        raise RecoveryError("not_a_mapping", field_name)
    try:
        values = dict(payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise RecoveryError("unreadable_mapping", field_name) from None
    if set(values) - expected:
        raise RecoveryError("unknown_field", field_name)
    if expected - set(values):
        raise RecoveryError("missing_field", field_name)
    return values


def _parse_json(serialized: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(serialized, (str, bytes, bytearray)):
        raise RecoveryError("invalid_json", field_name)
    try:
        decoded = json.loads(serialized, object_pairs_hook=_strict_json_object)
    except (KeyboardInterrupt, SystemExit):
        raise
    except RecoveryError:
        raise
    except BaseException:
        raise RecoveryError("invalid_json", field_name) from None
    if not isinstance(decoded, Mapping):
        raise RecoveryError("not_a_mapping", field_name)
    return decoded


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError("duplicate_field", "json")
        result[key] = value
    return result


def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError):
        raise RecoveryError("invalid_payload", "payload") from None


def _digest(payload: Any) -> str:
    return (
        f"sha256:{hashlib.sha256(_canonical_json(payload).encode('ascii')).hexdigest()}"
    )


def _require_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise RecoveryError("invalid_digest", field_name)
    return value


def _require_idempotency_key(value: Any) -> str:
    if type(value) is not str or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
        raise RecoveryError("invalid_idempotency_key", "idempotency_key")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise RecoveryError("invalid_non_negative_integer", field_name)
    return value


def _checkpoint_filename(sequence: int) -> str:
    return f"checkpoint-{sequence:020d}.json"


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _fsync_directory(directory: Path) -> None:
    # Python cannot open a Windows directory for fsync. The checkpoint file
    # itself is flushed before the no-clobber link on every platform.
    if _WINDOWS:
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MAX_CHECKPOINT_BYTES",
    "RECOVERY_CHECKPOINT_SCHEMA_VERSION",
    "RECOVERY_EVIDENCE_SCHEMA_VERSION",
    "CheckpointJournal",
    "CompensationLimit",
    "EffectKind",
    "EffectObservation",
    "EffectRecord",
    "EffectState",
    "ObservationState",
    "RecoveryCheckpoint",
    "RecoveryDecision",
    "RecoveryDisposition",
    "RecoveryError",
    "RecoveryPhase",
    "RecoveryReason",
    "advance_checkpoint",
    "derive_idempotency_key",
    "recover_workflow",
    "validate_checkpoint_lineage",
]
