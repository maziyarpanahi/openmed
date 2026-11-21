"""Compliance event receipts and a core-backed audit sink.

DSAR and Part 11 workflows append event metadata to an AuditSink. A
CoreAuditSink binds each export-compatible record to the core AuditChain.
The legacy HashChainAuditLog remains available during the 4.x transition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from openmed.__about__ import __version__
from openmed.core.audit import AuditReport, hash_text, stable_hash
from openmed.core.audit_chain import AuditChain
from openmed.utils.deprecation import deprecated


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


class _ComplianceRecordLog:
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

        record = self._next_record(event_type, payload)
        self._records.append(record)
        return record

    def _next_record(self, event_type: str, payload: Mapping[str, Any]) -> AuditRecord:
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


@deprecated(
    since="4.3",
    remove_in="5.0",
    replacement="CoreAuditSink(openmed.core.audit_chain.AuditChain())",
)
class HashChainAuditLog(_ComplianceRecordLog):
    """Legacy self-contained compliance record log."""


class CoreAuditSink(_ComplianceRecordLog):
    """Commit compliance record digests to a core chain and keep export compatibility.

    The Part 11 export retains its existing record format. Persist the supplied
    AuditChain separately to retain the additional core-chain receipts.
    """

    def __init__(self, chain: AuditChain | None = None) -> None:
        super().__init__()
        self.chain = chain if chain is not None else AuditChain()
        self._core_sequences: list[int] = []
        self._lock = RLock()

    def append(self, event_type: str, payload: Mapping[str, Any]) -> AuditRecord:
        """Append a content commitment before making the compatibility record visible."""
        with self._lock:
            record = self._next_record(event_type, payload)
            receipt = AuditReport(
                policy="compliance_event_v1",
                resolved_profile={},
                detectors=[],
                safety_sweep={},
                spans=[],
                thresholds={},
                residual_risk={},
                openmed_version=__version__,
                manifest_hash=stable_hash({"format": "compliance_event_v1"}),
                document_length=0,
                input_hash=record.record_hash,
                deidentified_text_hash=hash_text(""),
            )
            entry = self.chain.append(receipt)
            self._core_sequences.append(entry.sequence)
            self._records.append(record)
            return record

    def verify(self) -> bool:
        """Verify both chains and each retained record's core commitment."""
        with self._lock:
            if (
                not super().verify()
                or not self.chain.verify()
                or len(self._records) != len(self._core_sequences)
            ):
                return False
            for record, sequence in zip(
                self._records, self._core_sequences, strict=True
            ):
                entry = self.chain.entry_at(sequence)
                if entry is None or entry.input_hash != record.record_hash:
                    return False
            return True


__all__ = [
    "AuditRecord",
    "AuditSink",
    "CoreAuditSink",
    "HashChainAuditLog",
]
