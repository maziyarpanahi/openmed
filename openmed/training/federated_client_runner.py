"""Fail-closed site-side runner for one federated training round.

The runner owns client orchestration only: it verifies a round offer, decides
whether the site participates, drives the injected training, clipping, privacy
mechanism and encoding ports, seals exactly one update per round and submits
it. Training, clipping, noise, transport, identity and coordinator behavior are
injected ports. Durable progress is written to an injected checkpoint store so
a restarted process resends the byte-identical sealed update instead of
training again.

Status records and errors contain only fixed reason codes, lifecycle states and
SHA-256 digests. They never include round references, site identities, paths,
example text, tensor values or local metrics.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Protocol

from .federated_round import FederatedRoundState
from .federated_schedule import FederatedRoundSchedule
from .federated_status import FederatedRoundReasonCode
from .federated_update_metadata import (
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

FEDERATED_CLIENT_RUNNER_SCHEMA_VERSION = "openmed.training.federated_client_runner.v1"
FEDERATED_CLIENT_CHECKPOINT_SCHEMA_VERSION = (
    "openmed.training.federated_client_checkpoint.v1"
)
MAX_FEDERATED_MANIFEST_BYTES = 1024 * 1024
DEFAULT_FEDERATED_MAX_UPDATE_BYTES = 64 * 1024 * 1024

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ROUND_REF: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CHECKPOINT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "round_digest",
        "state",
        "reason_code",
        "manifest_digest",
        "update_digest",
        "payload",
    }
)


class FederatedClientRoundState(str, Enum):
    """A site-side lifecycle state for one federated round."""

    RECEIVED = "received"
    VERIFIED = "verified"
    DECLINED = "declined"
    TRAINING = "training"
    SEALED = "sealed"
    SUBMITTED = "submitted"
    ABORTED = "aborted"
    EXPIRED = "expired"

    def __str__(self) -> str:
        return self.value


_S = FederatedClientRoundState

FEDERATED_CLIENT_TERMINAL_STATES: Final[frozenset[FederatedClientRoundState]] = (
    frozenset({_S.DECLINED, _S.SUBMITTED, _S.ABORTED, _S.EXPIRED})
)

# A sealed round never returns to training, so its update digest is final.
FEDERATED_CLIENT_TRANSITIONS: Final[
    Mapping[FederatedClientRoundState, frozenset[FederatedClientRoundState]]
] = MappingProxyType(
    {
        _S.RECEIVED: frozenset({_S.VERIFIED, _S.DECLINED, _S.ABORTED, _S.EXPIRED}),
        _S.VERIFIED: frozenset({_S.TRAINING, _S.DECLINED, _S.ABORTED, _S.EXPIRED}),
        _S.TRAINING: frozenset({_S.SEALED, _S.VERIFIED, _S.ABORTED, _S.EXPIRED}),
        _S.SEALED: frozenset({_S.SUBMITTED, _S.ABORTED, _S.EXPIRED}),
        _S.DECLINED: frozenset(),
        _S.SUBMITTED: frozenset(),
        _S.ABORTED: frozenset(),
        _S.EXPIRED: frozenset(),
    }
)


class FederatedClientReasonCode(str, Enum):
    """Stable, value-free reasons for client outcomes and refusals."""

    MANIFEST_DIGEST_DRIFT = "manifest_digest_drift"
    MANIFEST_UNVERIFIED = "manifest_unverified"
    CAPABILITY_INCOMPATIBLE = "capability_incompatible"
    DATA_USE_DENIED = "data_use_denied"
    MEMORY_CLASS_EXCEEDED = "memory_class_exceeded"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    SITE_BUDGET_EXHAUSTED = "site_budget_exhausted"
    ROUND_HELD = "round_held"
    ROUND_ABORTED = "round_aborted"
    DEADLINE_PASSED = "deadline_passed"
    COLLECTION_CLOSED = "collection_closed"
    SUBMISSION_WINDOW_NOT_OPEN = "submission_window_not_open"
    COORDINATOR_STATE_UNAVAILABLE = "coordinator_state_unavailable"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    WALL_CLOCK_LIMIT_EXCEEDED = "wall_clock_limit_exceeded"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    TRAINING_FAILED = "training_failed"
    CLIPPING_FAILED = "clipping_failed"
    PRIVACY_MECHANISM_FAILED = "privacy_mechanism_failed"
    ENCODING_FAILED = "encoding_failed"
    UPDATE_METADATA_INVALID = "update_metadata_invalid"
    SUBMISSION_FAILED = "submission_failed"
    UPDATE_DIGEST_CONFLICT = "update_digest_conflict"
    UPDATE_NOT_SEALED = "update_not_sealed"
    INVALID_OFFER = "invalid_offer"
    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_CLOCK = "invalid_clock"
    CHECKPOINT_INVALID = "checkpoint_invalid"
    CHECKPOINT_STORE_FAILED = "checkpoint_store_failed"

    def __str__(self) -> str:
        return self.value


_R = FederatedClientReasonCode

_DECLINE_REASONS: Final = frozenset(
    {
        _R.MANIFEST_DIGEST_DRIFT,
        _R.MANIFEST_UNVERIFIED,
        _R.CAPABILITY_INCOMPATIBLE,
        _R.DATA_USE_DENIED,
        _R.MEMORY_CLASS_EXCEEDED,
        _R.STEP_BUDGET_EXCEEDED,
        _R.SITE_BUDGET_EXHAUSTED,
    }
)
_PENDING_REASONS: Final = frozenset(
    {None, _R.COORDINATOR_STATE_UNAVAILABLE, _R.SUSPENDED}
)
_STATE_REASONS: Final[Mapping[FederatedClientRoundState, frozenset[Any]]] = (
    MappingProxyType(
        {
            _S.RECEIVED: _PENDING_REASONS,
            _S.VERIFIED: _PENDING_REASONS,
            _S.TRAINING: frozenset({None}),
            _S.SEALED: _PENDING_REASONS
            | {
                _R.SUBMISSION_FAILED,
                _R.SUBMISSION_WINDOW_NOT_OPEN,
                _R.WALL_CLOCK_LIMIT_EXCEEDED,
            },
            _S.SUBMITTED: frozenset({None}),
            _S.DECLINED: _DECLINE_REASONS,
            _S.ABORTED: frozenset(
                {
                    _R.MANIFEST_DIGEST_DRIFT,
                    _R.MANIFEST_UNVERIFIED,
                    _R.ROUND_HELD,
                    _R.ROUND_ABORTED,
                    _R.CANCELLED,
                    _R.STEP_BUDGET_EXCEEDED,
                    _R.WALL_CLOCK_LIMIT_EXCEEDED,
                    _R.MEMORY_LIMIT_EXCEEDED,
                    _R.TRAINING_FAILED,
                    _R.CLIPPING_FAILED,
                    _R.PRIVACY_MECHANISM_FAILED,
                    _R.ENCODING_FAILED,
                    _R.UPDATE_METADATA_INVALID,
                }
            ),
            _S.EXPIRED: frozenset({_R.DEADLINE_PASSED, _R.COLLECTION_CLOSED}),
        }
    )
)
_POLICY_REASONS: Final = frozenset(
    {
        _R.MANIFEST_DIGEST_DRIFT,
        _R.MANIFEST_UNVERIFIED,
        _R.CAPABILITY_INCOMPATIBLE,
        _R.DATA_USE_DENIED,
        _R.MEMORY_CLASS_EXCEEDED,
        _R.STEP_BUDGET_EXCEEDED,
    }
)
_PRIVACY_REASONS: Final = frozenset(
    {_R.SITE_BUDGET_EXHAUSTED, _R.CLIPPING_FAILED, _R.PRIVACY_MECHANISM_FAILED}
)


def round_reason_for(
    reason_code: FederatedClientReasonCode,
) -> FederatedRoundReasonCode:
    """Map a client reason to the coordinator abort reason vocabulary.

    Args:
        reason_code: A client reason code.

    Returns:
        The matching abort reason from ``federated_status``: policy gates for
        manifest, capability, data-use and resource declines, privacy gates for
        budget, clipping and privacy-mechanism failures, operator cancellation
        for cancellation, and ``round_error`` otherwise.

    Raises:
        FederatedClientRunnerError: If ``reason_code`` is not a client reason.
    """

    if not isinstance(reason_code, FederatedClientReasonCode):
        raise FederatedClientRunnerError(_R.INVALID_CONFIGURATION)
    if reason_code is _R.CANCELLED:
        return FederatedRoundReasonCode.OPERATOR_CANCELLED
    if reason_code in _POLICY_REASONS:
        return FederatedRoundReasonCode.POLICY_GATE_FAILED
    if reason_code in _PRIVACY_REASONS:
        return FederatedRoundReasonCode.PRIVACY_GATE_FAILED
    return FederatedRoundReasonCode.ROUND_ERROR


class FederatedClientRunnerError(ValueError):
    """A fail-closed refusal carrying only a stable reason code.

    Attributes:
        reason_code: The fixed reason for the refusal.
    """

    def __init__(self, reason_code: FederatedClientReasonCode):
        if not isinstance(reason_code, FederatedClientReasonCode):
            reason_code = _R.INVALID_CONFIGURATION
        super().__init__(f"federated client runner refused: {reason_code.value}")
        self.reason_code = reason_code


class FederatedMemoryClass(str, Enum):
    """A coarse, declared memory class ordered from smallest to largest."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    def __str__(self) -> str:
        return self.value


_MEMORY_ORDER: Final = MappingProxyType(
    {
        FederatedMemoryClass.SMALL: 0,
        FederatedMemoryClass.MEDIUM: 1,
        FederatedMemoryClass.LARGE: 2,
    }
)


class FederatedCancellationMode(str, Enum):
    """A cooperative cancellation request polled at runner checkpoints.

    ``abort`` ends the round in the aborted state. ``suspend`` stops work and
    leaves a resumable state; a later run continues from the last checkpoint.
    """

    ABORT = "abort"
    SUSPEND = "suspend"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, kw_only=True)
class FederatedClientResourceLimits:
    """Declared site resource limits enforced by the runner.

    Args:
        max_wall_clock_seconds: Positive wall-clock budget for one runner call.
        max_steps: Positive local step budget for one round.
        memory_class: The largest memory class this site can provide.
        max_update_bytes: Positive bound on one encoded update payload.
    """

    max_wall_clock_seconds: int
    max_steps: int
    memory_class: FederatedMemoryClass
    max_update_bytes: int = DEFAULT_FEDERATED_MAX_UPDATE_BYTES

    def __post_init__(self) -> None:
        for value in (
            self.max_wall_clock_seconds,
            self.max_steps,
            self.max_update_bytes,
        ):
            if type(value) is not int or value < 1:
                raise FederatedClientRunnerError(_R.INVALID_CONFIGURATION)
        if not isinstance(self.memory_class, FederatedMemoryClass):
            raise FederatedClientRunnerError(_R.INVALID_CONFIGURATION)


@dataclass(frozen=True, kw_only=True)
class FederatedClientRoundOffer:
    """A round received from the coordinator; plaintext fields stay private.

    The manifest verification port is responsible for proving that the
    structured terms below match the signed manifest bytes.

    Args:
        round_ref: Opaque coordinator round reference. It is used only to
            derive a digest and is never rendered in records or errors.
        manifest: Raw manifest bytes, at most one MiB.
        manifest_digest: Declared lowercase ``sha256:`` digest of
            ``manifest``. A mismatch is treated as manifest drift.
        schedule: Round phase boundaries. Submission is allowed only in the
            update-submission phase.
        update_policy: Trusted update metadata policy for this round.
        local_steps: Positive number of local steps the round requests.
        required_memory_class: Memory class the round declares it needs.
    """

    round_ref: str = field(repr=False)
    manifest: bytes = field(repr=False)
    manifest_digest: str
    schedule: FederatedRoundSchedule
    update_policy: FederatedUpdatePolicy = field(repr=False)
    local_steps: int
    required_memory_class: FederatedMemoryClass

    def __post_init__(self) -> None:
        if (
            type(self.round_ref) is not str
            or _ROUND_REF.fullmatch(self.round_ref) is None
            or type(self.manifest) is not bytes
            or not 1 <= len(self.manifest) <= MAX_FEDERATED_MANIFEST_BYTES
            or not _is_digest(self.manifest_digest)
            or type(self.schedule) is not FederatedRoundSchedule
            or type(self.update_policy) is not FederatedUpdatePolicy
            or type(self.local_steps) is not int
            or self.local_steps < 1
            or not isinstance(self.required_memory_class, FederatedMemoryClass)
        ):
            raise FederatedClientRunnerError(_R.INVALID_OFFER)

    @property
    def round_digest(self) -> str:
        """Return the ``sha256:`` digest of the opaque round reference."""

        return _digest(self.round_ref.encode("ascii"))


@dataclass(frozen=True)
class FederatedEncodedUpdate:
    """An encoded update returned by the encoding port.

    Args:
        payload: The exact bytes to seal and submit. Never rendered.
        metadata: Declared update metadata; its ``update_digest`` must equal
            the SHA-256 digest of ``payload`` and it must declare clipping.
    """

    payload: bytes = field(repr=False)
    metadata: FederatedUpdateMetadata

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or not isinstance(
            self.metadata, FederatedUpdateMetadata
        ):
            raise FederatedClientRunnerError(_R.ENCODING_FAILED)


@dataclass(frozen=True)
class FederatedClientRoundStatus:
    """A value-free status record for one round at one site.

    Args:
        round_digest: ``sha256:`` digest of the opaque round reference.
        state: Current client lifecycle state.
        manifest_digest: ``sha256:`` digest of the manifest first received.
        update_digest: Digest of the sealed update, once one is bound.
        reason_code: Stable reason for declined, aborted, expired or
            interrupted rounds; ``None`` on the happy path.
        schema_version: Exact status schema identifier.
    """

    round_digest: str
    state: FederatedClientRoundState
    manifest_digest: str
    update_digest: str | None = None
    reason_code: FederatedClientReasonCode | None = None
    schema_version: str = FEDERATED_CLIENT_RUNNER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEDERATED_CLIENT_RUNNER_SCHEMA_VERSION:
            raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)
        _check_record(
            self.round_digest,
            self.state,
            self.reason_code,
            self.manifest_digest,
            self.update_digest,
        )

    @property
    def resumable(self) -> bool:
        """Return whether a later run can continue this round."""

        return self.state not in FEDERATED_CLIENT_TERMINAL_STATES

    @property
    def round_reason_code(self) -> FederatedRoundReasonCode | None:
        """Return the coordinator abort reason for an unsuccessful end state."""

        if self.state in {_S.DECLINED, _S.ABORTED, _S.EXPIRED}:
            return round_reason_for(self.reason_code)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a privacy-safe deterministic mapping."""

        round_reason = self.round_reason_code
        return {
            "manifest_digest": self.manifest_digest,
            "reason_code": (
                self.reason_code.value if self.reason_code is not None else None
            ),
            "resumable": self.resumable,
            "round_digest": self.round_digest,
            "round_reason_code": (
                round_reason.value if round_reason is not None else None
            ),
            "schema_version": self.schema_version,
            "state": self.state.value,
            "update_digest": self.update_digest,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class FederatedTrainingStop(Exception):
    """Raised by :class:`FederatedTrainingControl` when training must stop.

    The training port should let it propagate. Swallowing it does not help:
    the runner discards any result once a stop has been requested.
    """

    def __init__(self) -> None:
        super().__init__("federated training stopped")


class FederatedTrainingControl:
    """Cooperative control handle passed to the training port.

    The training port must call :meth:`step` before every local step. Each
    call enforces the step budget, cancellation, the round deadline and the
    wall-clock limit. Counts are never rendered.
    """

    __slots__ = ("_check", "_max_steps", "_memory_class", "_steps", "_stop_reason")

    def __init__(
        self,
        *,
        max_steps: int,
        memory_class: FederatedMemoryClass,
        check: Callable[[], FederatedClientReasonCode | None],
    ) -> None:
        self._check = check
        self._max_steps = max_steps
        self._memory_class = memory_class
        self._steps = 0
        self._stop_reason: FederatedClientReasonCode | None = None

    @property
    def max_steps(self) -> int:
        """Return the number of local steps the trainer may take."""

        return self._max_steps

    @property
    def memory_class(self) -> FederatedMemoryClass:
        """Return the declared memory class the trainer must stay within."""

        return self._memory_class

    @property
    def stopped(self) -> bool:
        """Return whether the runner has required training to stop."""

        return self._stop_reason is not None

    def step(self) -> None:
        """Account for one local step or raise :class:`FederatedTrainingStop`."""

        if self._stop_reason is not None:
            raise FederatedTrainingStop()
        self._steps += 1
        if self._steps > self._max_steps:
            self._stop(_R.STEP_BUDGET_EXCEEDED)
        reason = self._check()
        if reason is not None:
            self._stop(reason)

    def report_memory_breach(self) -> None:
        """Report that the declared memory class was exceeded and stop."""

        self._stop(_R.MEMORY_LIMIT_EXCEEDED)

    def _stop(self, reason: FederatedClientReasonCode) -> None:
        if self._stop_reason is None:
            self._stop_reason = reason
        raise FederatedTrainingStop()

    def __repr__(self) -> str:
        return f"FederatedTrainingControl(stopped={self.stopped})"


class FederatedManifestVerifier(Protocol):
    """Verifies the manifest signature and its binding to the offer terms."""

    def verify(self, offer: FederatedClientRoundOffer) -> bool:
        """Return ``True`` only when the offer is authentic."""


class FederatedCapabilityChecker(Protocol):
    """Checks that local software and hardware satisfy the round."""

    def is_compatible(self, offer: FederatedClientRoundOffer) -> bool:
        """Return ``True`` only when the site can run the round."""


class FederatedDataUsePolicy(Protocol):
    """Decides whether local data may be used for the round."""

    def permits(self, offer: FederatedClientRoundOffer) -> bool:
        """Return ``True`` only for an explicit allow decision."""


class FederatedLocalTrainer(Protocol):
    """Runs local training and returns an opaque update object."""

    def train(
        self, offer: FederatedClientRoundOffer, control: FederatedTrainingControl
    ) -> object:
        """Train locally, calling ``control.step()`` before every step."""


class FederatedUpdateClipper(Protocol):
    """Bounds the contribution of the local update."""

    def clip(self, update: object, offer: FederatedClientRoundOffer) -> object:
        """Return the clipped update."""


class FederatedPrivacyMechanism(Protocol):
    """Owns the site privacy budget and the local privacy mechanism."""

    def has_budget(self, offer: FederatedClientRoundOffer) -> bool:
        """Return ``True`` only when the site budget covers the round."""

    def privatize(self, update: object, offer: FederatedClientRoundOffer) -> object:
        """Return the privatized update."""


class FederatedUpdateEncoder(Protocol):
    """Encodes a privatized update into bytes and declared metadata."""

    def encode(
        self, update: object, offer: FederatedClientRoundOffer
    ) -> FederatedEncodedUpdate:
        """Return the encoded update."""


class FederatedUpdateSubmitter(Protocol):
    """Reports coordinator round state and transmits sealed updates."""

    def round_state(self, offer: FederatedClientRoundOffer) -> FederatedRoundState:
        """Return the coordinator lifecycle state of the round."""

    def submit(
        self, offer: FederatedClientRoundOffer, payload: bytes, update_digest: str
    ) -> None:
        """Transmit the sealed payload; raise on any failure."""


class FederatedClock(Protocol):
    """Supplies timezone-aware UTC time."""

    def now(self) -> datetime:
        """Return the current UTC time."""


class FederatedCheckpointStore(Protocol):
    """Durable local key-value storage with atomic replacement."""

    def load(self, key: str) -> bytes | None:
        """Return stored bytes for ``key`` or ``None``."""

    def save(self, key: str, data: bytes) -> None:
        """Atomically replace the bytes stored for ``key``."""


class FederatedCancellationSource(Protocol):
    """Reports cooperative cancellation requests."""

    def poll(self) -> FederatedCancellationMode | None:
        """Return the pending cancellation request, if any."""


class _NeverCancelled:
    def poll(self) -> None:
        return None


@dataclass(frozen=True)
class _Checkpoint:
    round_digest: str
    state: FederatedClientRoundState
    manifest_digest: str
    update_digest: str | None = None
    reason_code: FederatedClientReasonCode | None = None
    payload: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _check_record(
            self.round_digest,
            self.state,
            self.reason_code,
            self.manifest_digest,
            self.update_digest,
        )
        if (self.payload is not None) != (self.state is _S.SEALED):
            raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)
        if self.payload is not None and (
            type(self.payload) is not bytes
            or not self.payload
            or _digest(self.payload) != self.update_digest
        ):
            raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)

    def status(self) -> FederatedClientRoundStatus:
        return FederatedClientRoundStatus(
            round_digest=self.round_digest,
            state=self.state,
            manifest_digest=self.manifest_digest,
            update_digest=self.update_digest,
            reason_code=self.reason_code,
        )

    def to_bytes(self) -> bytes:
        payload = (
            base64.b64encode(self.payload).decode("ascii")
            if self.payload is not None
            else None
        )
        record = {
            "manifest_digest": self.manifest_digest,
            "payload": payload,
            "reason_code": (
                self.reason_code.value if self.reason_code is not None else None
            ),
            "round_digest": self.round_digest,
            "schema_version": FEDERATED_CLIENT_CHECKPOINT_SCHEMA_VERSION,
            "state": self.state.value,
            "update_digest": self.update_digest,
        }
        return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")

    @classmethod
    def from_bytes(cls, data: object, round_digest: str) -> _Checkpoint:
        failed = False
        try:
            if type(data) is not bytes:
                raise ValueError
            record = json.loads(data, object_pairs_hook=_strict_object)
            if type(record) is not dict or record.keys() != _CHECKPOINT_FIELDS:
                raise ValueError
            if record["schema_version"] != FEDERATED_CLIENT_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError
            if record["round_digest"] != round_digest:
                raise ValueError
            reason = record["reason_code"]
            payload = record["payload"]
            if type(record["state"]) is not str:
                raise ValueError
            if reason is not None and type(reason) is not str:
                raise ValueError
            if payload is not None and type(payload) is not str:
                raise ValueError
            checkpoint = cls(
                round_digest=record["round_digest"],
                state=FederatedClientRoundState(record["state"]),
                manifest_digest=record["manifest_digest"],
                update_digest=record["update_digest"],
                reason_code=(
                    FederatedClientReasonCode(reason) if reason is not None else None
                ),
                payload=(
                    base64.b64decode(payload.encode("ascii"), validate=True)
                    if payload is not None
                    else None
                ),
            )
        except (
            ValueError,
            TypeError,
            RecursionError,
            UnicodeError,
            binascii.Error,
            FederatedClientRunnerError,
        ):
            failed = True
        if failed:
            raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID) from None
        return checkpoint


class FederatedClientRunner:
    """Drive one site through federated rounds with fail-closed semantics.

    Every round is bound to at most one sealed update digest. The sealed
    payload is persisted before any submission attempt, so a retry after a
    crash or a failed submission resends byte-identical bytes and never trains
    again. No submission happens after a hold, abort or deadline.

    Args:
        manifest_verifier: Manifest verification port.
        capability: Capability compatibility port.
        data_use_policy: Data-use policy port.
        trainer: Local training port.
        clipper: Update clipping port.
        privacy_mechanism: Site privacy budget and mechanism port.
        encoder: Update encoding port.
        submitter: Coordinator state and submission port.
        clock: UTC clock port.
        store: Durable checkpoint store port.
        limits: Declared site resource limits.
        cancellation: Optional cooperative cancellation port.

    Raises:
        FederatedClientRunnerError: ``invalid_configuration`` when a port lacks
            a required method or the limits are not a
            :class:`FederatedClientResourceLimits`.
    """

    def __init__(
        self,
        *,
        manifest_verifier: FederatedManifestVerifier,
        capability: FederatedCapabilityChecker,
        data_use_policy: FederatedDataUsePolicy,
        trainer: FederatedLocalTrainer,
        clipper: FederatedUpdateClipper,
        privacy_mechanism: FederatedPrivacyMechanism,
        encoder: FederatedUpdateEncoder,
        submitter: FederatedUpdateSubmitter,
        clock: FederatedClock,
        store: FederatedCheckpointStore,
        limits: FederatedClientResourceLimits,
        cancellation: FederatedCancellationSource | None = None,
    ) -> None:
        if cancellation is None:
            cancellation = _NeverCancelled()
        required = (
            (manifest_verifier, ("verify",)),
            (capability, ("is_compatible",)),
            (data_use_policy, ("permits",)),
            (trainer, ("train",)),
            (clipper, ("clip",)),
            (privacy_mechanism, ("has_budget", "privatize")),
            (encoder, ("encode",)),
            (submitter, ("round_state", "submit")),
            (clock, ("now",)),
            (store, ("load", "save")),
            (cancellation, ("poll",)),
        )
        for port, methods in required:
            if not all(callable(getattr(port, name, None)) for name in methods):
                raise FederatedClientRunnerError(_R.INVALID_CONFIGURATION)
        if type(limits) is not FederatedClientResourceLimits:
            raise FederatedClientRunnerError(_R.INVALID_CONFIGURATION)
        self._verifier = manifest_verifier
        self._capability = capability
        self._data_use = data_use_policy
        self._trainer = trainer
        self._clipper = clipper
        self._privacy = privacy_mechanism
        self._encoder = encoder
        self._submitter = submitter
        self._clock = clock
        self._store = store
        self._limits = limits
        self._cancellation = cancellation

    def __repr__(self) -> str:
        return "FederatedClientRunner()"

    @staticmethod
    def checkpoint_key(offer: FederatedClientRoundOffer) -> str:
        """Return the value-free store key for a round.

        Args:
            offer: The received round offer.

        Returns:
            ``federated-client-`` followed by the hex round digest.
        """

        offer = _require_offer(offer)
        return "federated-client-" + offer.round_digest.removeprefix("sha256:")

    def status(
        self, offer: FederatedClientRoundOffer
    ) -> FederatedClientRoundStatus | None:
        """Return the stored status for a round without calling other ports.

        Args:
            offer: The received round offer.

        Returns:
            The persisted status, or ``None`` when the round was never received.

        Raises:
            FederatedClientRunnerError: ``checkpoint_invalid`` or
                ``checkpoint_store_failed`` for unreadable checkpoints.
        """

        offer = _require_offer(offer)
        record = self._load(offer)
        return record.status() if record is not None else None

    def run_round(self, offer: FederatedClientRoundOffer) -> FederatedClientRoundStatus:
        """Advance a round as far as possible and return its status.

        A terminal round is returned unchanged without calling any port. A
        sealed round resends its stored bytes. Any other round is verified and
        its participation decided before the training port is called.

        Args:
            offer: The received round offer.

        Returns:
            The persisted status after this run. Non-terminal states are
            resumable by calling this method again.

        Raises:
            FederatedClientRunnerError: For invalid offers, clocks or
                checkpoints, or when the checkpoint store fails. Nothing is
                submitted after such a refusal.
        """

        offer = _require_offer(offer)
        started = self._now()
        record = self._load(offer)
        if record is None:
            record = _Checkpoint(
                round_digest=offer.round_digest,
                state=_S.RECEIVED,
                manifest_digest=_digest(offer.manifest),
            )
            self._save(offer, record)
        return self._drive(offer, record, started)

    def resend_update(
        self, offer: FederatedClientRoundOffer, payload: bytes
    ) -> FederatedClientRoundStatus:
        """Resend the sealed update for a round, refusing any other bytes.

        Args:
            offer: The received round offer.
            payload: Bytes the caller wants to submit. They must be identical
                to the sealed update bound to the round.

        Returns:
            The status after the attempt. A submitted round is returned
            without a second transmission.

        Raises:
            FederatedClientRunnerError: ``update_not_sealed`` when no update is
                bound, ``update_digest_conflict`` for different bytes, or the
                round's terminal reason (for example ``round_held``,
                ``round_aborted`` or ``deadline_passed``) when submission is
                closed.
        """

        offer = _require_offer(offer)
        started = self._now()
        record = self._load(offer)
        if record is None or record.update_digest is None:
            raise FederatedClientRunnerError(_R.UPDATE_NOT_SEALED)
        if type(payload) is not bytes or _digest(payload) != record.update_digest:
            raise FederatedClientRunnerError(_R.UPDATE_DIGEST_CONFLICT)
        if record.state is _S.SUBMITTED:
            return record.status()
        if record.state in FEDERATED_CLIENT_TERMINAL_STATES:
            raise FederatedClientRunnerError(record.reason_code)
        status = self._drive(offer, record, started)
        if status.state in FEDERATED_CLIENT_TERMINAL_STATES - {_S.SUBMITTED}:
            raise FederatedClientRunnerError(status.reason_code)
        return status

    # Lifecycle driving.

    def _drive(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        started: datetime,
    ) -> FederatedClientRoundStatus:
        if record.state in FEDERATED_CLIENT_TERMINAL_STATES:
            return record.status()
        if record.state is _S.TRAINING:
            # An interrupted run left no update; restart from verification.
            record = self._move(offer, record, _S.VERIFIED, None)
        computed = _digest(offer.manifest)
        if (
            computed != record.manifest_digest
            or offer.manifest_digest != record.manifest_digest
        ):
            target = _S.ABORTED if record.state is _S.SEALED else _S.DECLINED
            return self._move(offer, record, target, _R.MANIFEST_DIGEST_DRIFT).status()
        if record.state is _S.SEALED:
            if not _accepted(self._verifier.verify, offer):
                return self._abort(offer, record, _R.MANIFEST_UNVERIFIED)
            return self._submit(offer, record, started)
        return self._participate(offer, record, started)

    def _participate(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        started: datetime,
    ) -> FederatedClientRoundStatus:
        reason, _ = self._conditions(offer, started)
        if reason is not None:
            return self._stop(offer, record, reason)
        target, reason = self._coordinator(offer, submitting=False)
        if reason is not None:
            return self._move(offer, record, target or record.state, reason).status()

        if not _accepted(self._verifier.verify, offer):
            return self._decline(offer, record, _R.MANIFEST_UNVERIFIED)
        if record.state is _S.RECEIVED:
            record = self._move(offer, record, _S.VERIFIED, None)
        if not _accepted(self._capability.is_compatible, offer):
            return self._decline(offer, record, _R.CAPABILITY_INCOMPATIBLE)
        if not _accepted(self._data_use.permits, offer):
            return self._decline(offer, record, _R.DATA_USE_DENIED)
        limits = self._limits
        if (
            _MEMORY_ORDER[offer.required_memory_class]
            > _MEMORY_ORDER[limits.memory_class]
        ):
            return self._decline(offer, record, _R.MEMORY_CLASS_EXCEEDED)
        if offer.local_steps > limits.max_steps:
            return self._decline(offer, record, _R.STEP_BUDGET_EXCEEDED)
        if not _accepted(self._privacy.has_budget, offer):
            return self._decline(offer, record, _R.SITE_BUDGET_EXHAUSTED)
        reason, _ = self._conditions(offer, started)
        if reason is not None:
            return self._stop(offer, record, reason)

        record = self._move(offer, record, _S.TRAINING, None)
        control = FederatedTrainingControl(
            max_steps=offer.local_steps,
            memory_class=limits.memory_class,
            check=lambda: self._control_check(offer, started),
        )
        failed = False
        try:
            update = self._trainer.train(offer, control)
        except Exception:
            failed = True
        stop_reason = control._stop_reason
        if stop_reason is _R.INVALID_CLOCK:
            raise FederatedClientRunnerError(_R.INVALID_CLOCK)
        if stop_reason is not None:
            return self._stop(offer, record, stop_reason)
        if failed:
            return self._abort(offer, record, _R.TRAINING_FAILED)

        for port, failure in (
            (self._clipper.clip, _R.CLIPPING_FAILED),
            (self._privacy.privatize, _R.PRIVACY_MECHANISM_FAILED),
        ):
            reason, _ = self._conditions(offer, started)
            if reason is not None:
                return self._stop(offer, record, reason)
            failed = False
            try:
                update = port(update, offer)
            except Exception:
                failed = True
            if failed:
                return self._abort(offer, record, failure)

        reason, _ = self._conditions(offer, started)
        if reason is not None:
            return self._stop(offer, record, reason)
        failed = False
        try:
            encoded = self._encoder.encode(update, offer)
        except Exception:
            failed = True
        if failed or type(encoded) is not FederatedEncodedUpdate:
            return self._abort(offer, record, _R.ENCODING_FAILED)
        payload = encoded.payload
        if not 1 <= len(payload) <= limits.max_update_bytes:
            return self._abort(offer, record, _R.ENCODING_FAILED)
        update_digest = _digest(payload)
        if not _metadata_matches(encoded.metadata, offer, update_digest):
            return self._abort(offer, record, _R.UPDATE_METADATA_INVALID)
        reason, _ = self._conditions(offer, started)
        if reason is not None:
            return self._stop(offer, record, reason)

        # Persist the binding before any transmission is attempted.
        record = self._move(
            offer,
            record,
            _S.SEALED,
            None,
            update_digest=update_digest,
            payload=payload,
        )
        return self._submit(offer, record, started)

    def _submit(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        started: datetime,
    ) -> FederatedClientRoundStatus:
        reason, now = self._conditions(offer, started)
        if reason is not None:
            return self._stop(offer, record, reason)
        if now < offer.schedule.update_submission_starts_at:
            return self._move(
                offer, record, _S.SEALED, _R.SUBMISSION_WINDOW_NOT_OPEN
            ).status()
        target, reason = self._coordinator(offer, submitting=True)
        if reason is not None:
            return self._move(offer, record, target or record.state, reason).status()
        # The coordinator port may block while the deadline or cancellation
        # changes. Recheck local conditions immediately before transmission.
        reason, _ = self._conditions(offer, started)
        if reason is not None:
            return self._stop(offer, record, reason)
        failed = False
        try:
            self._submitter.submit(offer, record.payload, record.update_digest)
        except Exception:
            failed = True
        if failed:
            return self._move(offer, record, _S.SEALED, _R.SUBMISSION_FAILED).status()
        return self._move(offer, record, _S.SUBMITTED, None).status()

    def _stop(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        reason: FederatedClientReasonCode,
    ) -> FederatedClientRoundStatus:
        if reason is _R.DEADLINE_PASSED:
            target = _S.EXPIRED
        elif reason is _R.SUSPENDED:
            target = _S.VERIFIED if record.state is _S.TRAINING else record.state
        elif reason is _R.WALL_CLOCK_LIMIT_EXCEEDED and record.state is _S.SEALED:
            target = _S.SEALED
        else:
            target = _S.ABORTED
        return self._move(offer, record, target, reason).status()

    def _decline(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        reason: FederatedClientReasonCode,
    ) -> FederatedClientRoundStatus:
        return self._move(offer, record, _S.DECLINED, reason).status()

    def _abort(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        reason: FederatedClientReasonCode,
    ) -> FederatedClientRoundStatus:
        return self._move(offer, record, _S.ABORTED, reason).status()

    def _move(
        self,
        offer: FederatedClientRoundOffer,
        record: _Checkpoint,
        target: FederatedClientRoundState,
        reason: FederatedClientReasonCode | None,
        *,
        update_digest: str | None = None,
        payload: bytes | None = None,
    ) -> _Checkpoint:
        if record.state in FEDERATED_CLIENT_TERMINAL_STATES or (
            target is not record.state
            and target not in FEDERATED_CLIENT_TRANSITIONS[record.state]
        ):
            raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)
        bound = record.update_digest
        if update_digest is not None:
            if bound is not None and bound != update_digest:
                raise FederatedClientRunnerError(_R.UPDATE_DIGEST_CONFLICT)
            bound = update_digest
        if target is _S.SEALED:
            kept = payload if payload is not None else record.payload
        else:
            kept = None  # Never retain update bytes outside the sealed state.
        updated = _Checkpoint(
            round_digest=record.round_digest,
            state=target,
            manifest_digest=record.manifest_digest,
            update_digest=bound,
            reason_code=reason,
            payload=kept,
        )
        self._save(offer, updated)
        return updated

    # Port adapters.

    def _now(self) -> datetime:
        failed = False
        try:
            moment = self._clock.now()
            if (
                not isinstance(moment, datetime)
                or moment.tzinfo is None
                or moment.utcoffset() != timedelta(0)
            ):
                raise ValueError
        except Exception:
            failed = True
        if failed:
            raise FederatedClientRunnerError(_R.INVALID_CLOCK) from None
        return moment

    def _conditions(
        self, offer: FederatedClientRoundOffer, started: datetime
    ) -> tuple[FederatedClientReasonCode | None, datetime]:
        now = self._now()
        if now >= offer.schedule.aggregation_starts_at:
            return _R.DEADLINE_PASSED, now
        try:
            mode = self._cancellation.poll()
        except Exception:
            mode = FederatedCancellationMode.ABORT
        if mode is FederatedCancellationMode.SUSPEND:
            return _R.SUSPENDED, now
        if mode is not None:
            # Anything other than an explicit suspend or no request aborts.
            return _R.CANCELLED, now
        if now - started >= timedelta(seconds=self._limits.max_wall_clock_seconds):
            return _R.WALL_CLOCK_LIMIT_EXCEEDED, now
        return None, now

    def _control_check(
        self, offer: FederatedClientRoundOffer, started: datetime
    ) -> FederatedClientReasonCode | None:
        try:
            reason, _ = self._conditions(offer, started)
        except FederatedClientRunnerError:
            return _R.INVALID_CLOCK
        return reason

    def _coordinator(
        self, offer: FederatedClientRoundOffer, *, submitting: bool
    ) -> tuple[FederatedClientRoundState | None, FederatedClientReasonCode | None]:
        try:
            state = self._submitter.round_state(offer)
        except Exception:
            state = None
        if not isinstance(state, FederatedRoundState):
            return None, _R.COORDINATOR_STATE_UNAVAILABLE
        if state is FederatedRoundState.HELD:
            return _S.ABORTED, _R.ROUND_HELD
        if state is FederatedRoundState.ABORTED:
            return _S.ABORTED, _R.ROUND_ABORTED
        if state in {
            FederatedRoundState.AGGREGATING,
            FederatedRoundState.EVALUATING,
            FederatedRoundState.PROMOTED,
        }:
            return _S.EXPIRED, _R.COLLECTION_CLOSED
        if submitting and state is not FederatedRoundState.COLLECTING:
            return None, _R.SUBMISSION_WINDOW_NOT_OPEN
        return None, None

    def _load(self, offer: FederatedClientRoundOffer) -> _Checkpoint | None:
        failed = False
        try:
            data = self._store.load(self.checkpoint_key(offer))
        except Exception:
            failed = True
        if failed:
            raise FederatedClientRunnerError(_R.CHECKPOINT_STORE_FAILED) from None
        if data is None:
            return None
        return _Checkpoint.from_bytes(data, offer.round_digest)

    def _save(self, offer: FederatedClientRoundOffer, record: _Checkpoint) -> None:
        failed = False
        try:
            self._store.save(self.checkpoint_key(offer), record.to_bytes())
        except Exception:
            failed = True
        if failed:
            raise FederatedClientRunnerError(_R.CHECKPOINT_STORE_FAILED) from None


def _accepted(port: Callable[[FederatedClientRoundOffer], object], offer: Any) -> bool:
    try:
        return port(offer) is True
    except Exception:
        return False


def _metadata_matches(
    metadata: FederatedUpdateMetadata,
    offer: FederatedClientRoundOffer,
    update_digest: str,
) -> bool:
    try:
        checked = FederatedUpdateMetadata.from_dict(
            metadata.to_dict(), policy=offer.update_policy
        )
    except FederatedUpdateMetadataError:
        return False
    return checked.update_digest == update_digest and checked.clipped is True


def _check_record(
    round_digest: object,
    state: object,
    reason_code: object,
    manifest_digest: object,
    update_digest: object,
) -> None:
    if (
        not _is_digest(round_digest)
        or not _is_digest(manifest_digest)
        or not isinstance(state, FederatedClientRoundState)
        or (
            reason_code is not None
            and not isinstance(reason_code, FederatedClientReasonCode)
        )
        or reason_code not in _STATE_REASONS[state]
        or (update_digest is not None and not _is_digest(update_digest))
    ):
        raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)
    if state in {_S.SEALED, _S.SUBMITTED} and update_digest is None:
        raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)
    if (
        state in {_S.RECEIVED, _S.VERIFIED, _S.TRAINING, _S.DECLINED}
        and update_digest is not None
    ):
        raise FederatedClientRunnerError(_R.CHECKPOINT_INVALID)


def _require_offer(offer: object) -> FederatedClientRoundOffer:
    if type(offer) is not FederatedClientRoundOffer:
        raise FederatedClientRunnerError(_R.INVALID_OFFER)
    return offer


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


__all__ = [
    "DEFAULT_FEDERATED_MAX_UPDATE_BYTES",
    "FEDERATED_CLIENT_CHECKPOINT_SCHEMA_VERSION",
    "FEDERATED_CLIENT_RUNNER_SCHEMA_VERSION",
    "FEDERATED_CLIENT_TERMINAL_STATES",
    "FEDERATED_CLIENT_TRANSITIONS",
    "MAX_FEDERATED_MANIFEST_BYTES",
    "FederatedCancellationMode",
    "FederatedCancellationSource",
    "FederatedCapabilityChecker",
    "FederatedCheckpointStore",
    "FederatedClientReasonCode",
    "FederatedClientResourceLimits",
    "FederatedClientRoundOffer",
    "FederatedClientRoundState",
    "FederatedClientRoundStatus",
    "FederatedClientRunner",
    "FederatedClientRunnerError",
    "FederatedClock",
    "FederatedDataUsePolicy",
    "FederatedEncodedUpdate",
    "FederatedLocalTrainer",
    "FederatedManifestVerifier",
    "FederatedMemoryClass",
    "FederatedPrivacyMechanism",
    "FederatedTrainingControl",
    "FederatedTrainingStop",
    "FederatedUpdateClipper",
    "FederatedUpdateEncoder",
    "FederatedUpdateSubmitter",
    "round_reason_for",
]
