"""Append-only, hash-chained evidence of local agent action transitions.

Only opaque correlation IDs, governed role codes, and digests cross this boundary.
No tool payload, clinical resource identifier, or approval credential is accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

from openmed.agent.correlation import ActionId, RunId

ACTION_LEDGER_SCHEMA_VERSION: Final = "openmed.agent.action_ledger.v1"
MAX_ENTRY_BYTES: Final = 16_384
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ROLE_RE = re.compile(r"role:[a-z][a-z0-9.-]{0,62}/[a-z][a-z0-9-]{0,63}")
_FILE_RE = re.compile(r"entry-([0-9]{20})\.json")
_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "action_id",
        "sequence",
        "state",
        "actor_role",
        "grant_digest",
        "tool_digest",
        "resource_refs",
        "previous_digest",
        "entry_digest",
    }
)


class ActionLedgerError(ValueError):
    """Value-free ledger failure with a stable code and no submitted values."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ActionState(str, Enum):
    """Recorded knowledge of a proposed side effect."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    ATTEMPTED = "attempted"
    COMMITTED = "committed"
    REJECTED = "rejected"


_SUCCESSORS: Final = {
    ActionState.PROPOSED: {
        ActionState.APPROVED,
        ActionState.ATTEMPTED,
        ActionState.REJECTED,
    },
    ActionState.APPROVED: {ActionState.ATTEMPTED, ActionState.REJECTED},
    ActionState.ATTEMPTED: {ActionState.COMMITTED, ActionState.REJECTED},
    ActionState.COMMITTED: set(),
    ActionState.REJECTED: set(),
}


def _digest(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


@dataclass(frozen=True, slots=True, repr=False)
class ActionEntry:
    """Immutable metadata-only transition bound to the preceding ledger entry."""

    run_id: RunId
    action_id: ActionId
    sequence: int
    state: ActionState
    actor_role: str
    grant_digest: str
    tool_digest: str
    resource_refs: tuple[str, ...]
    previous_digest: str | None
    entry_digest: str
    schema_version: str = ACTION_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_LEDGER_SCHEMA_VERSION:
            raise ActionLedgerError("unsupported_schema_version")
        if type(self.run_id) is not RunId or type(self.action_id) is not ActionId:
            raise ActionLedgerError("invalid_identifier")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ActionLedgerError("invalid_sequence")
        if type(self.state) is not ActionState:
            raise ActionLedgerError("invalid_state")
        if (
            type(self.actor_role) is not str
            or _ROLE_RE.fullmatch(self.actor_role) is None
        ):
            raise ActionLedgerError("invalid_actor_role")
        if not _valid_digest(self.grant_digest) or not _valid_digest(self.tool_digest):
            raise ActionLedgerError("invalid_digest")
        if (
            type(self.resource_refs) is not tuple
            or len(self.resource_refs) > 32
            or any(not _valid_digest(ref) for ref in self.resource_refs)
            or tuple(sorted(set(self.resource_refs))) != self.resource_refs
        ):
            raise ActionLedgerError("invalid_resource_refs")
        if self.sequence == 0:
            if self.previous_digest is not None:
                raise ActionLedgerError("unexpected_previous_digest")
        elif not _valid_digest(self.previous_digest):
            raise ActionLedgerError("invalid_previous_digest")
        if not _valid_digest(self.entry_digest) or self.entry_digest != _digest(
            self._unsigned_dict()
        ):
            raise ActionLedgerError("entry_digest_mismatch")

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id.serialize(),
            "action_id": self.action_id.serialize(),
            "sequence": self.sequence,
            "state": self.state.value,
            "actor_role": self.actor_role,
            "grant_digest": self.grant_digest,
            "tool_digest": self.tool_digest,
            "resource_refs": list(self.resource_refs),
            "previous_digest": self.previous_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return only canonical value-free evidence fields."""
        return {**self._unsigned_dict(), "entry_digest": self.entry_digest}

    def to_json(self) -> str:
        """Serialize an entry deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def create(
        cls,
        *,
        run_id: RunId,
        action_id: ActionId,
        sequence: int,
        state: ActionState,
        actor_role: str,
        grant_digest: str,
        tool_digest: str,
        resource_refs: tuple[str, ...],
        previous_digest: str | None,
    ) -> ActionEntry:
        """Build an entry and bind all public metadata to its digest."""
        if type(run_id) is not RunId or type(action_id) is not ActionId:
            raise ActionLedgerError("invalid_identifier")
        if type(state) is not ActionState:
            raise ActionLedgerError("invalid_state")
        values = {
            "run_id": run_id,
            "action_id": action_id,
            "sequence": sequence,
            "state": state,
            "actor_role": actor_role,
            "grant_digest": grant_digest,
            "tool_digest": tool_digest,
            "resource_refs": resource_refs,
            "previous_digest": previous_digest,
        }
        unsigned = {
            "schema_version": ACTION_LEDGER_SCHEMA_VERSION,
            "run_id": run_id.serialize(),
            "action_id": action_id.serialize(),
            "sequence": sequence,
            "state": state.value,
            "actor_role": actor_role,
            "grant_digest": grant_digest,
            "tool_digest": tool_digest,
            "resource_refs": list(resource_refs),
            "previous_digest": previous_digest,
        }
        return cls(**values, entry_digest=_digest(unsigned))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionEntry:
        """Restore exact entry fields without echoing malformed input."""
        if type(payload) is not dict or payload.keys() != _FIELDS:
            raise ActionLedgerError("invalid_entry_fields")
        try:
            refs = payload["resource_refs"]
            if type(refs) is not list:
                raise ActionLedgerError("invalid_resource_refs")
            return cls(
                run_id=RunId.parse(payload["run_id"]),
                action_id=ActionId.parse(payload["action_id"]),
                sequence=payload["sequence"],
                state=ActionState(payload["state"]),
                actor_role=payload["actor_role"],
                grant_digest=payload["grant_digest"],
                tool_digest=payload["tool_digest"],
                resource_refs=tuple(refs),
                previous_digest=payload["previous_digest"],
                entry_digest=payload["entry_digest"],
                schema_version=payload["schema_version"],
            )
        except ActionLedgerError:
            raise
        except (ValueError, TypeError, KeyError):
            raise ActionLedgerError("invalid_entry") from None


def verify_action_ledger(entries: tuple[ActionEntry, ...]) -> tuple[ActionEntry, ...]:
    """Check digest continuity, action identity, and valid state transitions."""
    if type(entries) is not tuple or any(type(e) is not ActionEntry for e in entries):
        raise ActionLedgerError("invalid_entries")
    latest: dict[ActionId, ActionEntry] = {}
    for index, entry in enumerate(entries):
        entry.__post_init__()
        if entry.sequence != index:
            raise ActionLedgerError("sequence_gap")
        if entry.previous_digest != (
            entries[index - 1].entry_digest if index else None
        ):
            raise ActionLedgerError("chain_mismatch")
        if index and entry.run_id != entries[0].run_id:
            raise ActionLedgerError("run_mismatch")
        prior = latest.get(entry.action_id)
        if prior is None:
            if entry.state is not ActionState.PROPOSED:
                raise ActionLedgerError("missing_proposal")
        else:
            if entry.state not in _SUCCESSORS[prior.state]:
                raise ActionLedgerError("invalid_transition")
            if (
                entry.grant_digest != prior.grant_digest
                or entry.tool_digest != prior.tool_digest
                or entry.resource_refs != prior.resource_refs
            ):
                raise ActionLedgerError("action_metadata_changed")
        latest[entry.action_id] = entry
    return entries


class ActionLedger:
    """Durable append-only journal in a caller-owned private directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            mode = self.directory.lstat().st_mode
        except OSError:
            raise ActionLedgerError("ledger_unavailable") from None
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ActionLedgerError("unsafe_ledger")

    def load(self) -> tuple[ActionEntry, ...]:
        """Read and verify the complete ledger before accepting another entry."""
        try:
            paths = list(self.directory.iterdir())
        except OSError:
            raise ActionLedgerError("ledger_unreadable") from None
        numbered: list[tuple[int, Path]] = []
        for path in paths:
            match = _FILE_RE.fullmatch(path.name)
            if match is None:
                if path.name.startswith(".entry-"):
                    continue
                raise ActionLedgerError("unexpected_ledger_file")
            numbered.append((int(match.group(1)), path))
        numbered.sort()
        entries = []
        for sequence, path in numbered:
            if sequence != len(entries):
                raise ActionLedgerError("sequence_gap")
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ENTRY_BYTES:
                    raise ActionLedgerError("unsafe_entry_file")
                data = path.read_bytes()

                def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                    result: dict[str, Any] = {}
                    for key, value in pairs:
                        if key in result:
                            raise ActionLedgerError("duplicate_entry_field")
                        result[key] = value
                    return result

                payload = json.loads(
                    data,
                    object_pairs_hook=unique_pairs,
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        ActionLedgerError("invalid_entry_json")
                    ),
                )
                entries.append(ActionEntry.from_dict(payload))
            except ActionLedgerError:
                raise
            except (OSError, ValueError, UnicodeError, TypeError):
                raise ActionLedgerError("invalid_entry_json") from None
        return verify_action_ledger(tuple(entries))

    def append(self, entry: ActionEntry) -> ActionEntry:
        """Atomically add one validated entry; reject conflicting writers."""
        if type(entry) is not ActionEntry:
            raise ActionLedgerError("invalid_entry")
        current = self.load()
        verify_action_ledger((*current, entry))
        target = self.directory / f"entry-{entry.sequence:020d}.json"
        encoded = entry.to_json().encode("ascii")
        descriptor: int | None = None
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".entry-", dir=self.directory
            )
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise ActionLedgerError("append_failed")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.link(temporary, target)
            if os.name != "nt":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except FileExistsError:
            raise ActionLedgerError("append_conflict") from None
        except ActionLedgerError:
            raise
        except OSError:
            raise ActionLedgerError("append_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        return entry

    def record(
        self,
        *,
        run_id: RunId,
        action_id: ActionId,
        state: ActionState,
        actor_role: str,
        grant_digest: str,
        tool_digest: str,
        resource_refs: tuple[str, ...] = (),
    ) -> ActionEntry:
        """Create and persist the next transition from value-free metadata."""
        current = self.load()
        entry = ActionEntry.create(
            run_id=run_id,
            action_id=action_id,
            sequence=len(current),
            state=state,
            actor_role=actor_role,
            grant_digest=grant_digest,
            tool_digest=tool_digest,
            resource_refs=resource_refs,
            previous_digest=current[-1].entry_digest if current else None,
        )
        return self.append(entry)

    def export_evidence(self) -> dict[str, Any]:
        """Export the verified value-free chain and its terminal digest."""
        entries = self.load()
        return {
            "schema_version": ACTION_LEDGER_SCHEMA_VERSION,
            "head_digest": entries[-1].entry_digest if entries else None,
            "entries": [entry.to_dict() for entry in entries],
        }
