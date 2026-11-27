"""Value-free corrective and preventive action records with auditable transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from importlib.resources import files
from threading import RLock

from jsonschema import Draft202012Validator

from openmed.core.audit import stable_hash
from openmed.core.audit_chain import AuditChain, AuditChainEntry

_LIFECYCLE = (
    "open",
    "investigating",
    "action_planned",
    "implemented",
    "verified",
    "closed",
)
_UPDATE_FIELDS = {
    "root_cause",
    "corrections",
    "preventive_actions",
    "artifact_digests",
    "before_evidence_digest",
    "after_evidence_digest",
    "release_tags",
    "local_note",
}
_LIST_FIELDS = {
    "capabilities",
    "versions",
    "artifact_digests",
    "corrections",
    "preventive_actions",
    "release_tags",
}


@lru_cache(maxsize=1)
def _validator():
    schema = files("openmed.core.schemas.json").joinpath(
        "corrective_action.schema.json"
    )
    return Draft202012Validator(json.loads(schema.read_text(encoding="utf-8")))


@dataclass(frozen=True, kw_only=True)
class CorrectiveAction:
    """Schema-validated release finding. Notes stay local to the creating process."""

    record_id: str
    source: str
    severity: str
    capabilities: tuple[str, ...]
    versions: tuple[str, ...]
    artifact_digests: tuple[str, ...] = ()
    root_cause: str = "unknown"
    corrections: tuple[str, ...] = ()
    preventive_actions: tuple[str, ...] = ()
    before_evidence_digest: str | None = None
    after_evidence_digest: str | None = None
    release_tags: tuple[str, ...] = ()
    state: str = "open"
    schema_version: str = "1.0.0"
    local_note: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name in _LIST_FIELDS:
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(
                type(value) is not str for value in values
            ):
                raise ValueError("invalid_corrective_action")
            object.__setattr__(self, name, tuple(sorted(values)))
        if not _validator().is_valid(self.to_dict()):
            raise ValueError("invalid_corrective_action")

    def to_dict(self) -> dict:
        """Export the schema-defined record without local notes."""
        return {
            f.name: list(getattr(self, f.name))
            if f.name in _LIST_FIELDS
            else getattr(self, f.name)
            for f in fields(self)
            if f.name != "local_note"
        }

    @classmethod
    def from_dict(cls, value: dict) -> CorrectiveAction:
        """Read a complete exported record, refusing unknown fields."""
        if not _validator().is_valid(value):
            raise ValueError("invalid_corrective_action")
        return cls(**value)


class CorrectiveActionLedger:
    """Keep immutable successive records and a core audit chain in one snapshot."""

    def __init__(self):
        self._lock = RLock()
        self._events: list[CorrectiveAction] = []
        self._latest: dict[str, CorrectiveAction] = {}
        self._chain = AuditChain()

    def records(self) -> tuple[CorrectiveAction, ...]:
        """Return the latest state of each record in identifier order."""
        with self._lock:
            return tuple(self._latest[key] for key in sorted(self._latest))

    def open(self, record: CorrectiveAction) -> CorrectiveAction:
        """Create an open record and append its first audit entry."""
        with self._lock:
            if record.record_id in self._latest or record.state != "open":
                raise ValueError("invalid_corrective_action_transition")
            return self._append(record)

    def transition(self, record_id: str, state: str, **changes) -> CorrectiveAction:
        """Advance exactly one step and attach controlled action or evidence updates."""
        with self._lock:
            current = self._latest.get(record_id)
            if (
                current is None
                or state not in _LIFECYCLE
                or _LIFECYCLE.index(state) != _LIFECYCLE.index(current.state) + 1
            ):
                raise ValueError("invalid_corrective_action_transition")
            if changes.keys() - _UPDATE_FIELDS:
                raise ValueError("invalid_corrective_action_update")
            return self._append(replace(current, state=state, **changes))

    def _append(self, record: CorrectiveAction) -> CorrectiveAction:
        payload = record.to_dict()
        digest = stable_hash(payload)
        entry = AuditChainEntry(
            sequence=self._chain.next_sequence,
            previous_hash=self._chain.head_hash,
            report_hash=digest,
            input_hash=stable_hash(["corrective_action", record.record_id]),
            deidentified_text_hash=digest,
            spans=(),
            entry_hash=AuditChain.GENESIS_HASH,
        )
        entry = replace(entry, entry_hash=entry.compute_hash())
        self._chain = AuditChain((*self._chain.entries, entry))
        self._events.append(record)
        self._latest[record.record_id] = record
        return record

    def to_dict(self) -> dict:
        """Export a consistent snapshot of all events and their retained chain head."""
        with self._lock:
            return {
                "records": [record.to_dict() for record in self._events],
                "audit_chain": self._chain.to_dict(),
            }

    @classmethod
    def from_dict(cls, payload: dict) -> CorrectiveActionLedger:
        """Verify chain integrity and replay the state machine before accepting events."""
        try:
            if (
                set(payload) != {"records", "audit_chain"}
                or type(payload["records"]) is not list
            ):
                raise ValueError
            chain = AuditChain.from_dict(payload["audit_chain"])
            if not chain.verify().valid:
                raise ValueError
            ledger = cls()
            for value in payload["records"]:
                record = CorrectiveAction.from_dict(value)
                current = ledger._latest.get(record.record_id)
                if current is None:
                    ledger.open(record)
                else:
                    updates = {
                        name: getattr(record, name)
                        for name in _UPDATE_FIELDS
                        if name != "local_note"
                    }
                    if (
                        ledger.transition(record.record_id, record.state, **updates)
                        != record
                    ):
                        raise ValueError
            if ledger._chain.to_dict() != chain.to_dict():
                raise ValueError
            return ledger
        except (TypeError, ValueError, KeyError, AttributeError):
            raise ValueError("invalid_corrective_action_ledger") from None
