"""Minimal hash-chained append-only audit log for compliance workflows.

GDPR access and erasure workflows must record *that* an export or preview was
generated without persisting the underlying personal data. This module provides
a narrow :class:`AuditSink` protocol and a self-contained
:class:`HashChainAuditLog` default implementation: an append-only ledger where
each record commits to its predecessor's hash, so any later mutation is
detectable via :meth:`HashChainAuditLog.verify`.

The interface is deliberately small so it can be swapped for the shared,
tamper-evident audit chain (OM-305) when that lands, without changing callers.
Records hold event metadata, provenance, and content hashes only -- never raw
PHI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openmed.core.audit import stable_hash


@dataclass
class AuditRecord:
    """One entry in a hash-chained audit log.

    ``payload`` carries export metadata / provenance / content hashes only; it
    must not contain raw PHI. ``record_hash`` commits to the sequence, event
    type, payload, and ``previous_hash`` so the chain is tamper-evident.
    """

    sequence: int
    event_type: str
    payload: Mapping[str, Any]
    previous_hash: str
    record_hash: str = field(default="")

    def compute_hash(self) -> str:
        """Return the deterministic hash this record should carry."""

        return stable_hash(
            {
                "sequence": self.sequence,
                "event_type": self.event_type,
                "payload": self.payload,
                "previous_hash": self.previous_hash,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to JSON-compatible fields."""

        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
        }


@runtime_checkable
class AuditSink(Protocol):
    """A narrow append-only audit sink.

    Callers depend on this protocol, not the concrete log, so the eventual
    shared audit chain can be injected in place of the default implementation.
    """

    def append(self, event_type: str, payload: Mapping[str, Any]) -> AuditRecord:
        """Record an event and return the resulting audit record."""
        ...


class HashChainAuditLog:
    """A self-contained, tamper-evident append-only audit log."""

    #: Anchor that the first record's ``previous_hash`` links to.
    GENESIS_HASH: str = stable_hash(
        {"chain": "openmed.compliance.audit", "sequence": -1}
    )

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    @property
    def records(self) -> list[AuditRecord]:
        """The append-only records in insertion order."""

        return self._records

    def append(self, event_type: str, payload: Mapping[str, Any]) -> AuditRecord:
        """Append an event, linking it to the current chain head."""

        if not isinstance(payload, Mapping):
            raise TypeError("audit payload must be a mapping")

        previous_hash = (
            self._records[-1].record_hash if self._records else self.GENESIS_HASH
        )
        record = AuditRecord(
            sequence=len(self._records),
            event_type=str(event_type),
            payload=dict(payload),
            previous_hash=previous_hash,
        )
        record.record_hash = record.compute_hash()
        self._records.append(record)
        return record

    def verify(self) -> bool:
        """Return ``True`` when the chain is intact and untampered."""

        previous_hash = self.GENESIS_HASH
        for index, record in enumerate(self._records):
            if record.sequence != index:
                return False
            if record.previous_hash != previous_hash:
                return False
            if record.record_hash != record.compute_hash():
                return False
            previous_hash = record.record_hash
        return True

    def to_payload(self) -> dict[str, Any]:
        """Serialize the whole chain, including the genesis anchor."""

        return {
            "genesis_hash": self.GENESIS_HASH,
            "records": [record.to_dict() for record in self._records],
        }


__all__ = [
    "AuditRecord",
    "AuditSink",
    "HashChainAuditLog",
]
