"""Default-off, local effect admission with signed durable transition receipts.

The ledger and independent high-water anchor must be provisioned explicitly.
The anchor belongs on storage excluded from ledger restores. Admission never
replaces grants, approval tokens, idempotency or clinical review.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Protocol, TypeVar

from openmed.agent.identifiers import GovernanceIdError, WorkflowId

SCHEMA = "openmed.agent.effect_admission.v1"
_ZERO_DIGEST = "sha256:" + "0" * 64
_T = TypeVar("_T")


class AdmissionState(str, Enum):
    """Closed vocabulary for effect admission."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    STOPPED = "stopped"


class AdmissionRole(str, Enum):
    """Operator roles asserted by the authorized local control plane."""

    OPERATOR = "operator"
    INCIDENT_COMMANDER = "incident_commander"


class AdmissionReason(str, Enum):
    """Content-free transition reasons, never free-text incident details."""

    INITIALIZED = "initialized"
    EXPLICIT_ENABLE = "explicit_enable"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"


class AdmissionError(ValueError):
    """Report only a controlled admission failure code."""


@dataclass(frozen=True, slots=True)
class AdmissionStatus:
    """Current scope status without payloads, credentials or private paths."""

    scope: str
    state: AdmissionState
    generation: int
    receipt_digest: str
    reason_code: str

    def to_dict(self) -> dict[str, str | int]:
        """Render only scope identifiers, codes, generations and digests."""
        return {
            "scope": self.scope,
            "reason_code": self.reason_code,
            "generation": self.generation,
            "receipt_digest": self.receipt_digest,
        }


class EffectAdmissionCheck(Protocol):
    """Injected check immediately before each effect and run resume.

    A generation captured at preview is mandatory for resumed or approved work.
    A changed generation requires a fresh preview and authority evaluation.
    """

    def require_admitted(
        self, workflow_id: WorkflowId, *, generation: int | None = None
    ) -> AdmissionStatus:
        """Raise a content-free error unless current admission permits work."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode()).hexdigest()


def _scope(value: WorkflowId | None) -> str:
    if value is None:
        return "global"
    if type(value) is not WorkflowId:
        raise AdmissionError("invalid_scope")
    return value.serialize()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdmissionError("untrusted_state")
        result[key] = value
    return result


class SQLiteAdmissionStore:
    """Signed append-only ledger paired with a separate monotonic anchor.

    SQLite rollback journals atomically commit both attached databases. Missing,
    malformed or restored older files are never automatically initialized. Keep
    the anchor outside ledger backup/restore operations; restoring both files
    and the key defeats local rollback detection and requires an external
    trusted storage boundary. Filesystem/key access is operator authority.

    Args:
        ledger: Local signed receipt database path.
        anchor: Independent high-water database excluded from ledger restores.
        key: Operator-supplied secret key of at least 32 bytes.
    """

    def __init__(self, ledger: Path, anchor: Path, key: bytes) -> None:
        if type(key) is not bytes or len(key) < 32:
            raise AdmissionError("invalid_key")
        self._ledger = Path(ledger)
        self._anchor = Path(anchor)
        try:
            same_path = self._ledger.resolve() == self._anchor.resolve()
            same_file = (
                self._ledger.exists()
                and self._anchor.exists()
                and self._ledger.samefile(self._anchor)
            )
        except (OSError, ValueError, RuntimeError):
            raise AdmissionError("untrusted_state") from None
        if same_path or same_file:
            raise AdmissionError("invalid_anchor")
        self._key = key
        self._seen = 0

    @contextmanager
    def _connection(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            paths = (self._ledger, self._anchor)
            if initialize:
                # Exclusive creation prevents accidental resets of stopped state.
                for path in paths:
                    if path.exists() or path.is_symlink():
                        raise AdmissionError("already_initialized")
                for path in paths:
                    descriptor = os.open(
                        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                    )
                    os.close(descriptor)
            elif any(not path.is_file() or path.is_symlink() for path in paths):
                raise AdmissionError("untrusted_state")
            connection = sqlite3.connect(
                self._ledger.resolve().as_uri() + "?mode=rw", uri=True, timeout=10
            )
            connection.execute(
                "ATTACH DATABASE ? AS anchor",
                (self._anchor.resolve().as_uri() + "?mode=rw",),
            )
            for database in ("main", "anchor"):
                # Multi-database atomic commits require rollback journals, not WAL.
                mode = connection.execute(f"PRAGMA {database}.journal_mode").fetchone()[
                    0
                ]
                if mode not in {"delete", "truncate", "persist"}:
                    raise AdmissionError("untrusted_state")
                connection.execute(f"PRAGMA {database}.synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except AdmissionError:
            if connection is not None:
                connection.rollback()
            raise
        except (
            OSError,
            sqlite3.Error,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            RecursionError,
            RuntimeError,
        ):
            if connection is not None:
                connection.rollback()
            raise AdmissionError("untrusted_state") from None
        finally:
            if connection is not None:
                connection.close()

    def initialize(
        self, *, role: AdmissionRole = AdmissionRole.OPERATOR, now: int | None = None
    ) -> dict[str, Any]:
        """Explicitly provision new disabled state; refuse any existing file.

        Args:
            role: Authorized control-plane role code.
            now: Injected Unix time, or the local system clock.

        Returns:
            The signed content-free initialization receipt.

        Raises:
            AdmissionError: If provisioning or metadata validation fails.
        """
        self._validate_transition(role, AdmissionReason.INITIALIZED, now)
        with self._connection(initialize=True) as connection:
            connection.execute(
                "CREATE TABLE receipts (generation INTEGER PRIMARY KEY, document TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE anchor.head (id INTEGER PRIMARY KEY CHECK(id=1), "
                "generation INTEGER NOT NULL, digest TEXT NOT NULL)"
            )
            receipt = self._receipt(
                "global",
                AdmissionState.DISABLED,
                role,
                AdmissionReason.INITIALIZED,
                now,
                1,
                _ZERO_DIGEST,
            )
            self._append(connection, receipt)
            self._seen = 1
            return receipt

    def _receipt(
        self,
        scope: str,
        state: AdmissionState,
        role: AdmissionRole,
        reason: AdmissionReason,
        now: int | None,
        generation: int,
        previous: str,
    ) -> dict[str, Any]:
        unsigned = {
            "schema_version": SCHEMA,
            "scope": scope,
            "state": state.value,
            "role": role.value,
            "reason_code": reason.value,
            "time": int(time.time()) if now is None else now,
            "generation": generation,
            "previous_digest": previous,
        }
        signature = hmac.new(
            self._key, _json(unsigned).encode(), hashlib.sha256
        ).hexdigest()
        return {**unsigned, "signature": "hmac-sha256:" + signature}

    def _load(
        self, connection: sqlite3.Connection
    ) -> tuple[dict[str, AdmissionState], int, str]:
        states: dict[str, AdmissionState] = {}
        generation = 0
        previous = _ZERO_DIGEST
        for number, serialized in connection.execute(
            "SELECT generation, document FROM receipts ORDER BY generation"
        ):
            if type(serialized) is not str or len(serialized) > 16384:
                raise AdmissionError("untrusted_state")
            receipt = json.loads(serialized, object_pairs_hook=_pairs)
            fields = {
                "schema_version",
                "scope",
                "state",
                "role",
                "reason_code",
                "time",
                "generation",
                "previous_digest",
                "signature",
            }
            if type(receipt) is not dict or set(receipt) != fields:
                raise AdmissionError("untrusted_state")
            unsigned = {
                key: value for key, value in receipt.items() if key != "signature"
            }
            signature = (
                "hmac-sha256:"
                + hmac.new(
                    self._key, _json(unsigned).encode(), hashlib.sha256
                ).hexdigest()
            )
            if type(receipt["signature"]) is not str or not hmac.compare_digest(
                signature, receipt["signature"]
            ):
                raise AdmissionError("untrusted_state")
            if (
                receipt["schema_version"] != SCHEMA
                or type(receipt["generation"]) is not int
                or number != generation + 1
                or receipt["generation"] != number
                or receipt["previous_digest"] != previous
            ):
                raise AdmissionError("untrusted_state")
            if type(receipt["time"]) is not int:
                raise AdmissionError("untrusted_state")
            self._validate_transition(
                AdmissionRole(receipt["role"]),
                AdmissionReason(receipt["reason_code"]),
                receipt["time"],
            )
            scope = receipt["scope"]
            if scope != "global":
                try:
                    WorkflowId.parse(scope)
                except GovernanceIdError:
                    raise AdmissionError("untrusted_state") from None
            state = AdmissionState(receipt["state"])
            reason = AdmissionReason(receipt["reason_code"])
            if generation == 0:
                if (
                    scope != "global"
                    or state is not AdmissionState.DISABLED
                    or reason is not AdmissionReason.INITIALIZED
                ):
                    raise AdmissionError("untrusted_state")
            elif not (
                (
                    state is AdmissionState.ENABLED
                    and reason is AdmissionReason.EXPLICIT_ENABLE
                )
                or (
                    state is AdmissionState.STOPPED
                    and reason
                    in {AdmissionReason.EMERGENCY_STOP, AdmissionReason.MAINTENANCE}
                )
            ):
                raise AdmissionError("untrusted_state")
            elif (
                scope != "global"
                and state is AdmissionState.ENABLED
                and states["global"] is AdmissionState.STOPPED
            ):
                raise AdmissionError("untrusted_state")
            if scope == "global" and state is AdmissionState.STOPPED:
                states.clear()
            states[scope] = state
            generation = number
            previous = _digest(receipt)
        anchor = connection.execute(
            "SELECT generation, digest FROM anchor.head WHERE id=1"
        ).fetchone()
        if (
            generation == 0
            or anchor != (generation, previous)
            or generation < self._seen
        ):
            raise AdmissionError("untrusted_state")
        self._seen = generation
        return states, generation, previous

    def _append(self, connection: sqlite3.Connection, receipt: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO receipts VALUES (?, ?)",
            (receipt["generation"], _json(receipt)),
        )
        connection.execute(
            "INSERT OR REPLACE INTO anchor.head VALUES (1, ?, ?)",
            (receipt["generation"], _digest(receipt)),
        )

    @staticmethod
    def _validate_transition(
        role: AdmissionRole, reason: AdmissionReason, now: int | None
    ) -> None:
        if type(role) is not AdmissionRole or type(reason) is not AdmissionReason:
            raise AdmissionError("invalid_transition")
        if now is not None and (type(now) is not int or now < 0 or now > 2**63 - 1):
            raise AdmissionError("invalid_time")

    def transition(
        self,
        state: AdmissionState,
        *,
        workflow_id: WorkflowId | None = None,
        role: AdmissionRole,
        reason: AdmissionReason,
        now: int | None = None,
    ) -> dict[str, Any]:
        """Commit a fresh signed enable/stop after validating ledger and anchor.

        Args:
            state: Enabled or stopped target state.
            workflow_id: Workflow scope, or global when omitted.
            role: Authorized control-plane role code.
            reason: Controlled reason matching the requested transition.
            now: Injected Unix time, or the local system clock.

        Returns:
            Signed receipt committed with the independent anchor.

        Raises:
            AdmissionError: If authority metadata or durable state is invalid.
        """
        scope = _scope(workflow_id)
        self._validate_transition(role, reason, now)
        if (
            (
                state is AdmissionState.ENABLED
                and reason is not AdmissionReason.EXPLICIT_ENABLE
            )
            or (
                state is AdmissionState.STOPPED
                and reason
                not in {AdmissionReason.EMERGENCY_STOP, AdmissionReason.MAINTENANCE}
            )
            or state not in {AdmissionState.ENABLED, AdmissionState.STOPPED}
        ):
            raise AdmissionError("invalid_transition")
        with self._connection() as connection:
            states, generation, previous = self._load(connection)
            if (
                scope != "global"
                and state is AdmissionState.ENABLED
                and states["global"] is AdmissionState.STOPPED
            ):
                raise AdmissionError("admission_stopped")
            receipt = self._receipt(
                scope, state, role, reason, now, generation + 1, previous
            )
            self._append(connection, receipt)
            self._seen = generation + 1
            return receipt

    def status(self, workflow_id: WorkflowId | None = None) -> AdmissionStatus:
        """Reload durable state at each boundary; errors produce stopped status.

        Args:
            workflow_id: Workflow scope, or global when omitted.

        Returns:
            Content-free status, stopped when durable evidence is untrusted.
        """
        scope = _scope(workflow_id)
        try:
            with self._connection() as connection:
                states, generation, digest = self._load(connection)
                state = states.get(scope, states["global"])
                if states["global"] is AdmissionState.STOPPED:
                    state = AdmissionState.STOPPED
                return AdmissionStatus(
                    scope, state, generation, digest, "admission_" + state.value
                )
        except AdmissionError:
            # Preserve known high-water generation even when the ledger is lost.
            try:
                with sqlite3.connect(
                    self._anchor.resolve().as_uri() + "?mode=ro", uri=True
                ) as anchor:
                    row = anchor.execute(
                        "SELECT generation FROM head WHERE id=1"
                    ).fetchone()
                    if row and type(row[0]) is int and row[0] > self._seen:
                        self._seen = row[0]
            except (OSError, sqlite3.Error, ValueError, RuntimeError):
                pass
            return AdmissionStatus(
                scope,
                AdmissionState.STOPPED,
                self._seen,
                _ZERO_DIGEST,
                "untrusted_state",
            )


class EffectAdmissionController:
    """Default-off check and explicit operator transitions over local storage.

    Args:
        store: Explicitly provisioned local store; omitted means disabled.
    """

    def __init__(self, store: SQLiteAdmissionStore | None = None) -> None:
        self._store = store

    def status(self, workflow_id: WorkflowId | None = None) -> AdmissionStatus:
        """Return disabled without configuration, or current durable status.

        Args:
            workflow_id: Workflow scope, or global when omitted.

        Returns:
            Current content-free admission status.
        """
        if self._store is None:
            return AdmissionStatus(
                _scope(workflow_id),
                AdmissionState.DISABLED,
                0,
                _ZERO_DIGEST,
                "admission_disabled",
            )
        return self._store.status(workflow_id)

    def require_admitted(
        self, workflow_id: WorkflowId, *, generation: int | None = None
    ) -> AdmissionStatus:
        """Check current state before every effect/resume, rejecting stale work.

        Args:
            workflow_id: Canonical workflow identifier.
            generation: Preview/run generation requiring unchanged admission.

        Returns:
            The current enabled admission status.

        Raises:
            AdmissionError: If admission is disabled, stopped or stale.
        """
        if type(workflow_id) is not WorkflowId:
            raise AdmissionError("invalid_scope")
        status = self.status(workflow_id)
        if status.state is not AdmissionState.ENABLED:
            raise AdmissionError(status.reason_code)
        if generation is not None and (
            type(generation) is not int or generation != status.generation
        ):
            raise AdmissionError("stale_generation")
        return status

    def enable(
        self,
        *,
        workflow_id: WorkflowId | None = None,
        role: AdmissionRole = AdmissionRole.OPERATOR,
        now: int | None = None,
    ) -> dict[str, Any]:
        """Explicitly enable a scope; never implicitly resume previous work.

        Args:
            workflow_id: Workflow scope, or global when omitted.
            role: Authorized control-plane role code.
            now: Injected Unix time, or the local system clock.

        Returns:
            Signed receipt for the fresh enable.

        Raises:
            AdmissionError: If state or transition metadata cannot be trusted.
        """
        if self._store is None:
            raise AdmissionError("not_configured")
        return self._store.transition(
            AdmissionState.ENABLED,
            workflow_id=workflow_id,
            role=role,
            reason=AdmissionReason.EXPLICIT_ENABLE,
            now=now,
        )

    def stop(
        self,
        *,
        workflow_id: WorkflowId | None = None,
        role: AdmissionRole = AdmissionRole.INCIDENT_COMMANDER,
        reason: AdmissionReason = AdmissionReason.EMERGENCY_STOP,
        now: int | None = None,
    ) -> dict[str, Any]:
        """Stop a scope at its next boundary; global stop clears workflow enables.

        Args:
            workflow_id: Workflow scope, or global when omitted.
            role: Authorized control-plane role code.
            reason: Controlled incident or maintenance reason.
            now: Injected Unix time, or the local system clock.

        Returns:
            Signed receipt for the stop.

        Raises:
            AdmissionError: If state or transition metadata cannot be trusted.
        """
        if self._store is None:
            raise AdmissionError("not_configured")
        return self._store.transition(
            AdmissionState.STOPPED,
            workflow_id=workflow_id,
            role=role,
            reason=reason,
            now=now,
        )


def dispatch_with_admission(
    dispatch: Callable[[], _T],
    *,
    workflow_id: WorkflowId,
    admission: EffectAdmissionCheck | None = None,
    generation: int | None = None,
) -> _T:
    """Check immediately before invoking one effect; missing check is default-off.

    Compose this inside approval dispatch callbacks so token verification cannot
    bypass a stop issued since preview. Adapters must call it for every effect,
    including retries; run recovery must use the same check before resuming.
    An effect already in progress cannot be recalled by a subsequent stop.

    Args:
        dispatch: Zero-argument callback for exactly one effect.
        workflow_id: Canonical workflow identifier.
        admission: Injected boundary check; omitted means disabled.
        generation: Admission generation saved with the preview/run.

    Returns:
        The effect callback result after current admission succeeds.

    Raises:
        AdmissionError: If current admission refuses the effect.
    """
    controller = EffectAdmissionController() if admission is None else admission
    controller.require_admitted(workflow_id, generation=generation)
    return dispatch()
