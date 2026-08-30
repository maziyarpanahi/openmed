"""PHI-safe 21 CFR Part 11 audit-trail records.

The emitter records who performed an action, what changed, when it changed,
and why it changed.  State values are accepted only as bounded labels and
``sha256:`` references; the trail never persists source values, replacements,
or other raw PHI.  Each record is committed to the existing local hash-chain
sink and also carries its own record hash so the exported artifact can be
verified as a unit without network access.

This module provides technical evidence for a deployment review.  It does not
certify compliance, provide electronic signatures, or replace validation,
identity provisioning, access control, retention, or operating procedures.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from openmed.core.audit import stable_hash

from .audit_chain import AuditRecord, HashChainAuditLog

PART11_FORMAT: Final = "openmed.part11-audit-trail"
PART11_SCHEMA_VERSION: Final = 1
PART11_EVENT_TYPE: Final = "part11.audit"
PART11_NOTICE: Final = (
    "This artifact is technical evidence from an OpenMed run, not a legal "
    "certification or a determination of 21 CFR Part 11 compliance."
)

_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_LABEL_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")
_CORE_PAYLOAD_FIELDS: Final = frozenset(
    {
        "record_id",
        "actor_id",
        "action",
        "timestamp_utc",
        "before_state",
        "after_state",
        "reason_code",
    }
)
_RECORD_FIELDS: Final = frozenset(
    {
        "record_id",
        "actor_id",
        "action",
        "timestamp_utc",
        "before_state",
        "after_state",
        "reason_code",
        "chain_sequence",
        "chain_previous_hash",
        "chain_entry_hash",
        "record_hash",
    }
)
_CHAIN_RECORD_FIELDS: Final = frozenset(
    {"sequence", "event_type", "payload", "previous_hash", "record_hash"}
)
_CHAIN_PAYLOAD_FIELDS: Final = frozenset({"genesis_hash", "records"})
_TRAIL_FIELDS: Final = frozenset(
    {
        "format",
        "version",
        "record_count",
        "head_hash",
        "chain",
        "records",
        "readiness_checklist",
        "notice",
        "trail_hash",
    }
)
_EVENT_FIELDS: Final = frozenset(
    {
        "record_id",
        "actor_id",
        "action",
        "timestamp",
        "timestamp_utc",
        "before",
        "before_state",
        "after",
        "after_state",
        "reason",
        "reason_code",
    }
)


def _reject_unknown_fields(
    data: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    object_name: str,
) -> None:
    if any(key not in allowed for key in data):
        raise ValueError(f"{object_name} contains unsupported fields")


def _metadata_text(value: Any, field_name: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be metadata text")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > max_length
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise ValueError(f"{field_name} must be bounded metadata text")
    return normalized


def _metadata_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a bounded metadata identifier")
    return value


def _state_label(value: Any) -> str:
    if not isinstance(value, str) or not _LABEL_PATTERN.fullmatch(value):
        raise ValueError("state label must be a bounded metadata label")
    return value


def _require_hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a sha256:<hex> hash")
    return value


def _normalize_timestamp(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        rendered = value.strip()
        if rendered.endswith("Z"):
            rendered = rendered[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(rendered)
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 UTC timestamp") from exc
    else:
        raise TypeError("timestamp must be a datetime, ISO-8601 string, or None")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if parsed.microsecond else "seconds"
    return parsed.isoformat(timespec=timespec).replace("+00:00", "Z")


@dataclass(frozen=True)
class Part11State:
    """A non-PHI before/after state reference.

    ``hash`` is a SHA-256 digest of caller-controlled state material.  The
    material itself is intentionally not retained by this object or emitted in
    the audit trail.
    """

    label: str
    hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _state_label(self.label))
        object.__setattr__(self, "hash", _require_hash(self.hash, "state hash"))

    @property
    def state_hash(self) -> str:
        """Compatibility name for callers that prefer ``state_hash``."""

        return self.hash

    def to_dict(self) -> dict[str, str]:
        """Return the PHI-safe JSON representation."""

        return {"label": self.label, "hash": self.hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Part11State":
        """Load a state reference while rejecting raw-value fields."""

        _reject_unknown_fields(data, {"label", "hash", "state_hash"}, "state")
        state_hash = data.get("hash")
        alternate_hash = data.get("state_hash")
        if state_hash is None:
            state_hash = alternate_hash
        elif alternate_hash is not None and alternate_hash != state_hash:
            raise ValueError("state hash fields must agree")
        return cls(label=data.get("label"), hash=state_hash)  # type: ignore[arg-type]


def hash_state(value: Any, *, label: str) -> Part11State:
    """Hash caller-held state without retaining it in the audit artifact.

    This helper is useful when an application has a state value in memory but
    wants the emitted record to contain only a digest and a safe label.
    """

    safe_label = _state_label(label)
    try:
        digest = stable_hash({"part11_state": value})
    except (TypeError, ValueError) as exc:
        raise ValueError("state value could not be converted to a stable hash") from exc
    return Part11State(label=safe_label, hash=digest)


def _normalize_state(value: Any) -> Part11State:
    if isinstance(value, Part11State):
        return value
    if isinstance(value, Mapping):
        return Part11State.from_dict(value)
    if isinstance(value, str):
        label = _state_label(value)
        return Part11State(label=label, hash=stable_hash({"part11_state": label}))
    raise TypeError("state must be a Part11State or a label/hash mapping")


@dataclass(frozen=True)
class Part11ReadinessItem:
    """One technical 21 CFR Part 11 readiness crosswalk item."""

    clause: str
    requirement: str
    emitter_fields: tuple[str, ...]
    status: str
    notes: str

    def __post_init__(self) -> None:
        if not self.clause.strip() or not self.requirement.strip():
            raise ValueError(
                "readiness checklist clauses and requirements are required"
            )
        if not self.emitter_fields:
            raise ValueError("readiness checklist items require field mappings")
        if self.status not in {"partial", "external"}:
            raise ValueError("readiness checklist status must be partial or external")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checklist item."""

        return {
            "clause": self.clause,
            "requirement": self.requirement,
            "emitter_fields": list(self.emitter_fields),
            "status": self.status,
            "notes": self.notes,
        }


PART11_READINESS_CHECKLIST: Final[tuple[Part11ReadinessItem, ...]] = (
    Part11ReadinessItem(
        "11.10(a)",
        "Validate the system for accuracy, reliability, and consistent intended performance.",
        ("record_hash", "trail_hash"),
        "external",
        "OpenMed supplies deterministic integrity evidence; deployment validation remains organizational work.",
    ),
    Part11ReadinessItem(
        "11.10(b)",
        "Generate accurate and complete copies of records in human-readable and electronic form.",
        ("chain", "records", "record_hash"),
        "partial",
        "The JSON export is deterministic and contains the complete PHI-safe trail.",
    ),
    Part11ReadinessItem(
        "11.10(c)",
        "Protect records so they can be readily retrieved throughout the retention period.",
        ("chain_entry_hash", "chain_previous_hash", "trail_hash"),
        "partial",
        "Local storage, backup, retention, and access controls remain deployment responsibilities.",
    ),
    Part11ReadinessItem(
        "11.10(d)",
        "Limit system access to authorized individuals.",
        ("actor_id",),
        "external",
        "The emitter attributes an event; authorization and provisioning are outside this module.",
    ),
    Part11ReadinessItem(
        "11.10(e)",
        "Use secure, computer-generated, time-stamped audit trails for operator actions.",
        (
            "actor_id",
            "action",
            "timestamp_utc",
            "before_state",
            "after_state",
            "reason_code",
            "chain_entry_hash",
        ),
        "partial",
        "This is the primary emitter coverage; qualified regulatory review is still required.",
    ),
    Part11ReadinessItem(
        "11.10(f)",
        "Use operational checks to enforce permitted sequencing and events.",
        ("action", "reason_code", "chain_sequence"),
        "external",
        "The caller must enforce workflow-specific operational checks before emission.",
    ),
    Part11ReadinessItem(
        "11.10(g)",
        "Use authority checks to ensure only authorized people can use the system.",
        ("actor_id", "record_id"),
        "external",
        "Identity and authority verification are not implemented by the emitter.",
    ),
    Part11ReadinessItem(
        "11.10(h)",
        "Use device checks to determine the validity of data sources or operation.",
        ("actor_id",),
        "external",
        "Device trust and endpoint controls must be supplied by the deployment.",
    ),
    Part11ReadinessItem(
        "11.10(i)",
        "Determine that people who develop, maintain, or use systems have education and training.",
        ("actor_id",),
        "external",
        "Training evidence is organization-owned and is not asserted by an audit record.",
    ),
    Part11ReadinessItem(
        "11.10(j)",
        "Establish and adhere to written policies that hold individuals accountable for actions.",
        ("actor_id", "action", "reason_code"),
        "external",
        "Written policies and accountability procedures must be maintained by the deployment.",
    ),
    Part11ReadinessItem(
        "11.10(k)",
        "Control system documentation, including distribution and revision controls.",
        ("record_hash", "trail_hash"),
        "external",
        "Documentation change control is outside the audit-trail emitter.",
    ),
    Part11ReadinessItem(
        "11.30",
        "Apply additional controls for open systems to ensure authenticity, integrity, and confidentiality.",
        ("record_hash", "chain_entry_hash", "trail_hash"),
        "partial",
        "The artifact provides integrity and authenticity evidence; confidentiality controls remain deployment-owned.",
    ),
    Part11ReadinessItem(
        "11.50",
        "Show the signer's printed name, date/time, and meaning for electronic signatures.",
        ("actor_id", "timestamp_utc", "action"),
        "external",
        "Electronic-signature workflow is explicitly out of scope for this emitter.",
    ),
    Part11ReadinessItem(
        "11.70",
        "Link electronic signatures to their respective records.",
        ("record_id", "chain_entry_hash"),
        "external",
        "The emitter links audit records to the chain; electronic signatures are out of scope.",
    ),
    Part11ReadinessItem(
        "11.100",
        "Ensure electronic signatures are unique to one individual and identity is verified.",
        ("actor_id",),
        "external",
        "Identity provisioning and signature uniqueness are explicitly out of scope.",
    ),
    Part11ReadinessItem(
        "11.200",
        "Apply controls for electronic-signature components and linking.",
        ("actor_id", "record_id"),
        "external",
        "Electronic-signature components and controls are explicitly out of scope.",
    ),
    Part11ReadinessItem(
        "11.300",
        "Control identification codes and passwords used with electronic signatures.",
        ("actor_id",),
        "external",
        "Credential lifecycle and password controls are deployment-owned and out of scope.",
    ),
)


def readiness_checklist() -> tuple[Part11ReadinessItem, ...]:
    """Return the immutable Part 11 readiness crosswalk."""

    return PART11_READINESS_CHECKLIST


@dataclass(frozen=True)
class Part11AuditRecord:
    """One PHI-safe, hash-bound Part 11 audit-trail record."""

    record_id: str
    actor_id: str
    action: str
    timestamp_utc: str
    before_state: Part11State
    after_state: Part11State
    reason_code: str
    chain_sequence: int
    chain_previous_hash: str
    chain_entry_hash: str
    record_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", _metadata_identifier(self.record_id, "record_id")
        )
        object.__setattr__(self, "actor_id", _metadata_text(self.actor_id, "actor_id"))
        object.__setattr__(self, "action", _metadata_text(self.action, "action"))
        object.__setattr__(
            self, "timestamp_utc", _normalize_timestamp(self.timestamp_utc)
        )
        if not isinstance(self.before_state, Part11State):
            raise TypeError("before_state must be a Part11State")
        if not isinstance(self.after_state, Part11State):
            raise TypeError("after_state must be a Part11State")
        object.__setattr__(
            self, "reason_code", _metadata_text(self.reason_code, "reason_code")
        )
        if type(self.chain_sequence) is not int or self.chain_sequence < 0:
            raise ValueError("chain_sequence must be a non-negative integer")
        object.__setattr__(
            self,
            "chain_previous_hash",
            _require_hash(self.chain_previous_hash, "chain_previous_hash"),
        )
        object.__setattr__(
            self,
            "chain_entry_hash",
            _require_hash(self.chain_entry_hash, "chain_entry_hash"),
        )
        object.__setattr__(
            self, "record_hash", _require_hash(self.record_hash, "record_hash")
        )

    @property
    def timestamp(self) -> str:
        """Compatibility alias for the canonical UTC timestamp field."""

        return self.timestamp_utc

    @property
    def before(self) -> Part11State:
        """Compatibility alias for the before-state reference."""

        return self.before_state

    @property
    def after(self) -> Part11State:
        """Compatibility alias for the after-state reference."""

        return self.after_state

    @property
    def reason(self) -> str:
        """Compatibility alias for the reason code."""

        return self.reason_code

    def chain_payload(self) -> dict[str, Any]:
        """Return the payload committed by the tamper-evident chain."""

        return {
            "record_id": self.record_id,
            "actor_id": self.actor_id,
            "action": self.action,
            "timestamp_utc": self.timestamp_utc,
            "before_state": self.before_state.to_dict(),
            "after_state": self.after_state.to_dict(),
            "reason_code": self.reason_code,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            **self.chain_payload(),
            "chain_sequence": self.chain_sequence,
            "chain_previous_hash": self.chain_previous_hash,
            "chain_entry_hash": self.chain_entry_hash,
        }

    def compute_hash(self) -> str:
        """Compute the record hash without trusting its stored value."""

        return stable_hash(self._hash_payload())

    def verify(self) -> bool:
        """Return whether this record's own integrity hash is valid."""

        return self.record_hash == self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record without raw state values."""

        return {
            **self.chain_payload(),
            "chain_sequence": self.chain_sequence,
            "chain_previous_hash": self.chain_previous_hash,
            "chain_entry_hash": self.chain_entry_hash,
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Part11AuditRecord":
        """Load a record while rejecting uncommitted fields and raw states."""

        _reject_unknown_fields(data, _RECORD_FIELDS, "Part 11 audit record")
        before_state = data.get("before_state")
        after_state = data.get("after_state")
        if not isinstance(before_state, Mapping) or not isinstance(
            after_state, Mapping
        ):
            raise TypeError("Part 11 record states must be objects")
        return cls(
            record_id=data.get("record_id"),  # type: ignore[arg-type]
            actor_id=data.get("actor_id"),  # type: ignore[arg-type]
            action=data.get("action"),  # type: ignore[arg-type]
            timestamp_utc=data.get("timestamp_utc"),  # type: ignore[arg-type]
            before_state=Part11State.from_dict(before_state),
            after_state=Part11State.from_dict(after_state),
            reason_code=data.get("reason_code"),  # type: ignore[arg-type]
            chain_sequence=data.get("chain_sequence"),  # type: ignore[arg-type]
            chain_previous_hash=data.get("chain_previous_hash"),  # type: ignore[arg-type]
            chain_entry_hash=data.get("chain_entry_hash"),  # type: ignore[arg-type]
            record_hash=data.get("record_hash"),  # type: ignore[arg-type]
        )


class Part11AuditEmitter:
    """Emit and verify a local-first 21 CFR Part 11 audit trail."""

    def __init__(
        self,
        chain: HashChainAuditLog | None = None,
        *,
        audit_sink: HashChainAuditLog | None = None,
    ) -> None:
        if chain is not None and audit_sink is not None:
            raise ValueError("chain and audit_sink cannot both be supplied")
        self._chain = chain or audit_sink or HashChainAuditLog()
        self._records: list[Part11AuditRecord] = []
        self._declared_record_count: int | None = None
        self._declared_head_hash: str | None = None
        self._declared_trail_hash: str | None = None
        self._declared_checklist: list[dict[str, Any]] | None = None

    @property
    def chain(self) -> HashChainAuditLog:
        """Return the append-only chain carrying the emitted records."""

        return self._chain

    @property
    def audit_chain(self) -> HashChainAuditLog:
        """Compatibility alias for :attr:`chain`."""

        return self._chain

    @property
    def records(self) -> tuple[Part11AuditRecord, ...]:
        """Return emitted records in chain order."""

        return tuple(self._records)

    @property
    def head_hash(self) -> str:
        """Return the current hash-chain head."""

        if self._chain.records:
            return self._chain.records[-1].record_hash
        return self._chain.GENESIS_HASH

    @property
    def trail_hash(self) -> str:
        """Return the current deterministic hash of the exported trail."""

        return stable_hash(self._payload())

    @staticmethod
    def readiness_checklist() -> tuple[Part11ReadinessItem, ...]:
        """Return the Part 11 readiness crosswalk."""

        return readiness_checklist()

    def emit(
        self,
        actor_id: str,
        action: str,
        before_state: Any = None,
        after_state: Any = None,
        reason_code: str | None = None,
        *,
        timestamp: datetime | str | None = None,
        timestamp_utc: datetime | str | None = None,
        record_id: str | None = None,
        before: Any = None,
        after: Any = None,
        reason: str | None = None,
    ) -> Part11AuditRecord:
        """Append one Part 11 record to the tamper-evident chain.

        ``before``/``after`` and ``reason`` are accepted as concise aliases for
        callers migrating from event-shaped inputs.  State mappings must
        contain only ``label`` and ``hash`` (or ``state_hash``).
        """

        if before_state is None:
            before_state = before
        if after_state is None:
            after_state = after
        if reason_code is None:
            reason_code = reason
        if timestamp is not None and timestamp_utc is not None:
            raise ValueError("timestamp and timestamp_utc cannot both be supplied")
        event_timestamp = timestamp if timestamp is not None else timestamp_utc

        chain_status = self._chain.verify()
        if not chain_status:
            raise ValueError("cannot append to an invalid audit chain")

        normalized_before = _normalize_state(before_state)
        normalized_after = _normalize_state(after_state)
        safe_actor = _metadata_text(actor_id, "actor_id")
        safe_action = _metadata_text(action, "action")
        safe_reason = _metadata_text(reason_code, "reason_code")
        safe_timestamp = _normalize_timestamp(event_timestamp)
        sequence = len(self._chain.records)
        core_payload = {
            "record_id": record_id
            or stable_hash(
                {
                    "format": PART11_FORMAT,
                    "sequence": sequence,
                    "actor_id": safe_actor,
                    "action": safe_action,
                    "timestamp_utc": safe_timestamp,
                    "before_state": normalized_before.to_dict(),
                    "after_state": normalized_after.to_dict(),
                    "reason_code": safe_reason,
                }
            ),
            "actor_id": safe_actor,
            "action": safe_action,
            "timestamp_utc": safe_timestamp,
            "before_state": normalized_before.to_dict(),
            "after_state": normalized_after.to_dict(),
            "reason_code": safe_reason,
        }
        core_payload["record_id"] = _metadata_identifier(
            core_payload["record_id"], "record_id"
        )
        if any(
            record.record_id == core_payload["record_id"] for record in self._records
        ):
            raise ValueError("record_id must be unique within the audit trail")

        chain_record = self._chain.append(PART11_EVENT_TYPE, core_payload)
        record = Part11AuditRecord(
            record_id=core_payload["record_id"],
            actor_id=core_payload["actor_id"],
            action=core_payload["action"],
            timestamp_utc=core_payload["timestamp_utc"],
            before_state=normalized_before,
            after_state=normalized_after,
            reason_code=core_payload["reason_code"],
            chain_sequence=chain_record.sequence,
            chain_previous_hash=chain_record.previous_hash,
            chain_entry_hash=chain_record.record_hash,
            record_hash=stable_hash(
                {
                    **core_payload,
                    "chain_sequence": chain_record.sequence,
                    "chain_previous_hash": chain_record.previous_hash,
                    "chain_entry_hash": chain_record.record_hash,
                }
            ),
        )
        self._records.append(record)
        self._clear_declared_integrity()
        return record

    append = emit
    emit_record = emit

    def verify(self) -> bool:
        """Verify the trail, its chain references, and its export integrity."""

        if self._declared_record_count is not None and len(self._records) != (
            self._declared_record_count
        ):
            return False
        if self._declared_head_hash is not None and self.head_hash != (
            self._declared_head_hash
        ):
            return False
        if self._declared_checklist is not None and self._declared_checklist != (
            _checklist_payload()
        ):
            return False
        if not self._chain.verify():
            return False
        chain_records = self._chain.records
        seen_ids: set[str] = set()
        for record in self._records:
            if record.record_id in seen_ids or not record.verify():
                return False
            seen_ids.add(record.record_id)
            if record.chain_sequence >= len(chain_records):
                return False
            chain_record = chain_records[record.chain_sequence]
            if chain_record.event_type != PART11_EVENT_TYPE:
                return False
            if chain_record.payload != record.chain_payload():
                return False
            if chain_record.previous_hash != record.chain_previous_hash:
                return False
            if chain_record.record_hash != record.chain_entry_hash:
                return False
        if self._declared_trail_hash is not None:
            return self._declared_trail_hash == stable_hash(
                self._payload(use_declared_fields=True)
            )
        return True

    def _clear_declared_integrity(self) -> None:
        self._declared_record_count = None
        self._declared_head_hash = None
        self._declared_trail_hash = None
        self._declared_checklist = None

    def _payload(self, *, use_declared_fields: bool = False) -> dict[str, Any]:
        record_count = (
            self._declared_record_count
            if use_declared_fields and self._declared_record_count is not None
            else len(self._records)
        )
        head_hash = (
            self._declared_head_hash
            if use_declared_fields and self._declared_head_hash is not None
            else self.head_hash
        )
        checklist = (
            self._declared_checklist
            if use_declared_fields and self._declared_checklist is not None
            else _checklist_payload()
        )
        return {
            "format": PART11_FORMAT,
            "version": PART11_SCHEMA_VERSION,
            "record_count": record_count,
            "head_hash": head_hash,
            "chain": self._chain.to_payload(),
            "records": [record.to_dict() for record in self._records],
            "readiness_checklist": checklist,
            "notice": PART11_NOTICE,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible audit-trail artifact."""

        payload = self._payload(use_declared_fields=True)
        payload["trail_hash"] = self._declared_trail_hash or stable_hash(payload)
        return payload

    def to_payload(self) -> dict[str, Any]:
        """Compatibility alias for the complete exported trail payload."""

        return self.to_dict()

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the trail deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write(self, path: str | Path) -> Path:
        """Write the trail atomically to a local UTF-8 JSON file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                delete=False,
                dir=destination.parent,
                encoding="utf-8",
                prefix=f".{destination.name}.",
            ) as handle:
                temporary = Path(handle.name)
                handle.write(self.to_json())
                handle.write("\n")
            os.replace(temporary, destination)
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Part11AuditEmitter":
        """Load a trail and retain integrity metadata for offline verification."""

        _reject_unknown_fields(data, _TRAIL_FIELDS, "Part 11 audit trail")
        if data.get("format") != PART11_FORMAT:
            raise ValueError(f"audit trail format must be {PART11_FORMAT!r}")
        if data.get("version") != PART11_SCHEMA_VERSION:
            raise ValueError("unsupported Part 11 audit trail version")
        record_count = data.get("record_count")
        if type(record_count) is not int or record_count < 0:
            raise ValueError("record_count must be a non-negative integer")
        head_hash = _require_hash(data.get("head_hash"), "head_hash")
        trail_hash = _require_hash(data.get("trail_hash"), "trail_hash")
        if data.get("notice") != PART11_NOTICE:
            raise ValueError("audit trail notice does not match this format")

        chain_data = data.get("chain")
        if not isinstance(chain_data, Mapping):
            raise TypeError("audit trail chain must be an object")
        chain = _chain_from_payload(chain_data)

        raw_records = data.get("records")
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes)
        ):
            raise TypeError("audit trail records must be a list")
        records: list[Part11AuditRecord] = []
        for item in raw_records:
            if not isinstance(item, Mapping):
                raise TypeError("audit trail records must contain objects")
            records.append(Part11AuditRecord.from_dict(item))

        raw_checklist = data.get("readiness_checklist")
        if not isinstance(raw_checklist, Sequence) or isinstance(
            raw_checklist, (str, bytes)
        ):
            raise TypeError("readiness_checklist must be a list")
        checklist: list[dict[str, Any]] = []
        for item in raw_checklist:
            if not isinstance(item, Mapping):
                raise TypeError("readiness_checklist must contain objects")
            checklist.append(dict(item))
        if checklist != _checklist_payload():
            raise ValueError("readiness checklist does not match this format")

        emitter = cls(chain)
        emitter._records = records
        emitter._declared_record_count = record_count
        emitter._declared_head_hash = head_hash
        emitter._declared_trail_hash = trail_hash
        emitter._declared_checklist = checklist
        return emitter

    @classmethod
    def from_json(cls, data: str | bytes) -> "Part11AuditEmitter":
        """Load a trail from JSON."""

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON for Part 11 audit trail: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("Part 11 audit trail JSON must contain an object")
        return cls.from_dict(parsed)

    @classmethod
    def load(cls, path: str | Path) -> "Part11AuditEmitter":
        """Load a UTF-8 audit-trail file."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))


Part11AuditTrail = Part11AuditEmitter


def _checklist_payload() -> list[dict[str, Any]]:
    return [item.to_dict() for item in PART11_READINESS_CHECKLIST]


def _validate_chain_payload(payload: Mapping[str, Any]) -> None:
    _reject_unknown_fields(payload, _CORE_PAYLOAD_FIELDS, "Part 11 chain payload")
    _metadata_identifier(payload.get("record_id"), "record_id")
    _metadata_text(payload.get("actor_id"), "actor_id")
    _metadata_text(payload.get("action"), "action")
    _normalize_timestamp(payload.get("timestamp_utc"))
    before_state = payload.get("before_state")
    after_state = payload.get("after_state")
    if not isinstance(before_state, Mapping) or not isinstance(after_state, Mapping):
        raise TypeError("Part 11 chain states must be objects")
    Part11State.from_dict(before_state)
    Part11State.from_dict(after_state)
    _metadata_text(payload.get("reason_code"), "reason_code")


def _chain_from_payload(data: Mapping[str, Any]) -> HashChainAuditLog:
    _reject_unknown_fields(data, _CHAIN_PAYLOAD_FIELDS, "audit trail chain")
    if data.get("genesis_hash") != HashChainAuditLog.GENESIS_HASH:
        raise ValueError("audit trail chain genesis_hash does not match this format")
    raw_records = data.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise TypeError("audit trail chain records must be a list")

    chain = HashChainAuditLog()
    loaded: list[AuditRecord] = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise TypeError("audit trail chain records must contain objects")
        _reject_unknown_fields(item, _CHAIN_RECORD_FIELDS, "audit trail chain record")
        payload = item.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("audit trail chain record payload must be an object")
        _validate_chain_payload(payload)
        loaded.append(
            AuditRecord(
                sequence=item.get("sequence"),  # type: ignore[arg-type]
                event_type=item.get("event_type"),  # type: ignore[arg-type]
                payload=dict(payload),
                previous_hash=item.get("previous_hash"),  # type: ignore[arg-type]
                record_hash=item.get("record_hash"),  # type: ignore[arg-type]
            )
        )
    chain._records = loaded
    return chain


def build_part11_audit_trail(
    events: Iterable[Mapping[str, Any]],
    *,
    chain: HashChainAuditLog | None = None,
) -> Part11AuditEmitter:
    """Build a trail from safe event mappings, preserving only safe fields."""

    emitter = Part11AuditEmitter(chain)
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("Part 11 events must be objects")
        _reject_unknown_fields(event, _EVENT_FIELDS, "Part 11 event")
        emitter.emit(
            actor_id=event.get("actor_id"),  # type: ignore[arg-type]
            action=event.get("action"),  # type: ignore[arg-type]
            before_state=event.get("before_state", event.get("before")),
            after_state=event.get("after_state", event.get("after")),
            reason_code=event.get("reason_code", event.get("reason")),
            timestamp_utc=event.get("timestamp_utc", event.get("timestamp")),
            record_id=event.get("record_id"),  # type: ignore[arg-type]
        )
    return emitter


export_part11_audit_trail = build_part11_audit_trail


def verify_part11_audit_trail(
    trail: Part11AuditEmitter | Mapping[str, Any],
) -> bool:
    """Verify a Part 11 audit trail object offline."""

    if not isinstance(trail, Part11AuditEmitter):
        trail = Part11AuditEmitter.from_dict(trail)
    return trail.verify()


__all__ = [
    "PART11_EVENT_TYPE",
    "PART11_FORMAT",
    "PART11_NOTICE",
    "PART11_READINESS_CHECKLIST",
    "PART11_SCHEMA_VERSION",
    "Part11AuditEmitter",
    "Part11AuditRecord",
    "Part11AuditTrail",
    "Part11ReadinessItem",
    "Part11State",
    "build_part11_audit_trail",
    "export_part11_audit_trail",
    "hash_state",
    "readiness_checklist",
    "verify_part11_audit_trail",
]
