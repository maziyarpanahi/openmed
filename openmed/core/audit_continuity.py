"""Tamper-evident audit continuity across key rotation, retention, and restore.

An :class:`~openmed.core.audit_chain.AuditChain` proves that its retained
entries still link to each other. It cannot, on its own, prove that nothing was
silently removed when entries aged out, that the chain restored from a backup
is the chain that was backed up, or which signing key vouched for which part of
the history. This module adds a versioned *continuity manifest* beside the
chain that records those facts as a hash-linked, signed event log:

* ``genesis`` opens signer epoch 0 for a chain identifier.
* ``anchor`` is a periodic signed digest of the chain head.
* ``rotation`` closes the current signer epoch and opens the next one. It is
  signed by both the outgoing and the incoming key, so there is no unsigned gap
  between epochs.
* ``retention`` is a non-reversible pruning tombstone: first and last sequence,
  record count, boundary entry hashes, a digest of the pruned segment, and the
  approving retention-policy fingerprint. It never holds entry content.
* ``restore`` records that the chain was restored from a backup and opens a
  restore epoch.

Every event commits to the chain state at the time it was written (the next
sequence number and the head hash), to the previous event digest, and to the
opaque chain identifier. :func:`verify_continuity` replays the log offline and
returns a bounded, value-free :class:`ContinuityVerificationReport` with a
stable reason code from :data:`CONTINUITY_REASON_CODES`.

:class:`AuditContinuityStore` keeps the chain file and the manifest in step.
Every mutation runs under an exclusive lock (a process-local lock plus an OS
file lock) and writes each document with an fsynced atomic rename. Pruning is
write-ahead: the tombstone is made durable before the chain is rewritten, and
any interrupted pruning is completed deterministically on the next operation
or by :meth:`AuditContinuityStore.recover`.

Signatures reuse the caller-owned key providers from
:mod:`openmed.core.audit_key_rotation`. OpenMed never stores, logs, or exports
key material; manifests and reports carry only key identifiers. Nothing in
this module makes network calls.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import stat
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from .audit import AuditReport, stable_hash
from .audit_chain import (
    CHAIN_FORMAT,
    GENESIS_HASH,
    AuditChain,
    AuditChainEntry,
    _atomic_write_text,
)
from .audit_key_rotation import (
    AUDIT_SIGNATURE_ALGORITHM,
    AuditKeyRotationError,
    KeyProvider,
    _digest_signature_status,
    _sign_digest,
    _validate_provider,
)

if TYPE_CHECKING:
    from openmed.risk.audit_retention import (
        AuditRetentionPolicy,
        AuditRetentionReport,
    )

CONTINUITY_FORMAT: Final = "openmed.audit-continuity"
CONTINUITY_VERSION: Final = 1
CONTINUITY_REPORT_FORMAT: Final = "openmed.audit-continuity.verification"
CONTINUITY_REPORT_VERSION: Final = 1
CONTINUITY_DIGEST_ALGORITHM: Final = "sha256"
CONTINUITY_SIGNATURE_ALGORITHM: Final = AUDIT_SIGNATURE_ALGORITHM
CONTINUITY_SIGNATURE_PURPOSE: Final = "audit-continuity"
MAX_CONTINUITY_EVENTS: Final = 65_536
MAX_CONTINUITY_JSON_BYTES: Final = 64 * 1024 * 1024
CONTINUITY_EVENT_KINDS: Final = (
    "genesis",
    "anchor",
    "rotation",
    "retention",
    "restore",
)

CONTINUITY_REASON_CODES: Final = frozenset(
    {
        "valid",
        "malformed_manifest",
        "unsupported_version",
        "chain_mismatch",
        "chain_invalid",
        "event_gap",
        "event_duplicate",
        "event_reordered",
        "fork",
        "epoch_mismatch",
        "unsigned_epoch_transition",
        "signature_invalid",
        "unknown_signer_key",
        "retired_signer_key",
        "signer_key_unavailable",
        "anchor_mismatch",
        "anchor_missing",
        "sequence_gap",
        "tombstone_mismatch",
        "incomplete_transition",
        "policy_mismatch",
        "unanchored_tail",
    }
)
"""Stable, value-free outcomes of :func:`verify_continuity`."""

CONTINUITY_ERROR_CODES: Final = CONTINUITY_REASON_CODES | frozenset(
    {
        "already_initialized",
        "not_initialized",
        "invalid_argument",
        "lock_failed",
        "io_failed",
    }
)
"""Codes carried by :class:`AuditContinuityError`."""

_SAFE_ID_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNATURE_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_EPOCH_OPENERS: Final = frozenset({"genesis", "rotation", "restore"})
_TRANSITIONS: Final = frozenset({"rotation", "restore"})
_SIGNATURE_ROLES: Final = frozenset({"signer", "outgoing", "incoming"})
_BODY_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "genesis": frozenset({"key_id"}),
        "anchor": frozenset(),
        "rotation": frozenset({"from_key_id", "to_key_id"}),
        "retention": frozenset(
            {
                "first_sequence",
                "last_sequence",
                "record_count",
                "start_hash",
                "end_hash",
                "segment_digest",
                "policy_fingerprint",
                "retention_report_digest",
            }
        ),
        "restore": frozenset({"key_id", "restore_id", "backup_digest"}),
    }
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "format",
        "version",
        "chain_id",
        "chain_format",
        "chain_genesis_hash",
        "digest_algorithm",
        "signature_algorithm",
        "event_count",
        "head_digest",
        "epochs",
        "events",
    }
)
_EVENT_FIELDS: Final = frozenset(
    {
        "index",
        "kind",
        "epoch",
        "previous_digest",
        "next_sequence",
        "head_hash",
        "body",
        "digest",
        "signatures",
    }
)
_SIGNATURE_FIELDS: Final = frozenset({"role", "key_id", "algorithm", "value"})
_EPOCH_FIELDS: Final = frozenset(
    {
        "epoch",
        "key_id",
        "opened_by",
        "opened_at_event",
        "start_sequence",
        "end_sequence",
        "closed_at_event",
        "state",
    }
)
_ANCHOR_REF_FIELDS: Final = frozenset({"index", "digest"})
_COUNT_FIELDS: Final = (
    "event_count",
    "epoch_count",
    "anchor_count",
    "rotation_count",
    "retention_count",
    "restore_count",
    "next_sequence",
    "retained_entry_count",
    "pruned_entry_count",
    "unanchored_entry_count",
    "superseded_event_count",
    "pruned_anchor_count",
    "trusted_anchor_count",
    "superseded_trusted_anchor_count",
    "unverified_signature_count",
)


class AuditContinuityError(ValueError):
    """Raised with a stable, value-free code when continuity cannot hold.

    Attributes:
        code: One of :data:`CONTINUITY_ERROR_CODES`.
    """

    def __init__(self, code: str) -> None:
        self.code = code if code in CONTINUITY_ERROR_CODES else "malformed_manifest"
        super().__init__(f"audit continuity failed: {self.code}")


def _fail(code: str) -> AuditContinuityError:
    return AuditContinuityError(code)


def _require_int(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail("malformed_manifest")
    return value


def _require_digest(value: Any) -> str:
    if type(value) is not str or not _DIGEST_RE.fullmatch(value):
        raise _fail("malformed_manifest")
    return value


def _require_safe_id(value: Any) -> str:
    if type(value) is not str or not _SAFE_ID_RE.fullmatch(value):
        raise _fail("malformed_manifest")
    return value


def _require_fields(data: Any, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise _fail("malformed_manifest")
    try:
        snapshot = dict(data.items())
    except Exception:  # noqa: BLE001 - mappings may execute caller code.
        raise _fail("malformed_manifest") from None
    if set(snapshot) != fields or not all(type(key) is str for key in snapshot):
        raise _fail("malformed_manifest")
    return snapshot


def _require_list(value: Any, *, limit: int) -> list[Any]:
    if type(value) is not list or len(value) > limit:
        raise _fail("malformed_manifest")
    return value


def _new_opaque_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(16)}"


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def continuity_origin_digest(chain_id: str) -> str:
    """Return the domain-separated digest the first event links to.

    Args:
        chain_id: Opaque chain identifier recorded in the manifest.
    """

    return stable_hash(
        {
            "format": f"{CONTINUITY_FORMAT}.origin",
            "version": CONTINUITY_VERSION,
            "chain_id": chain_id,
            "chain_format": CHAIN_FORMAT,
            "chain_genesis_hash": GENESIS_HASH,
        }
    )


def tombstone_segment_digest(
    chain_id: str,
    entries: Sequence[AuditChainEntry],
) -> str:
    """Return the non-reversible digest of a pruned chain segment.

    The digest commits to the chain identifier, the segment bounds, and the
    entry hashes only. Entry hashes are themselves digests, so the result
    reveals nothing about audit payloads.

    Args:
        chain_id: Opaque chain identifier recorded in the manifest.
        entries: The contiguous entries being pruned, in sequence order.

    Raises:
        AuditContinuityError: If ``entries`` is empty.
    """

    if not entries:
        raise _fail("invalid_argument")
    return stable_hash(
        {
            "format": f"{CONTINUITY_FORMAT}.segment",
            "version": CONTINUITY_VERSION,
            "chain_id": chain_id,
            "first_sequence": entries[0].sequence,
            "last_sequence": entries[-1].sequence,
            "entry_hashes": [entry.entry_hash for entry in entries],
        }
    )


@dataclass(frozen=True)
class ContinuitySignature:
    """One HMAC signature over an event digest.

    Attributes:
        role: ``"signer"`` for ordinary events, or ``"outgoing"`` and
            ``"incoming"`` for epoch transitions.
        key_id: Non-secret identifier of the signing key.
        algorithm: Signature algorithm, currently ``"HMAC-SHA256"``.
        value: Hex signature value. It never contains key material.
    """

    role: str
    key_id: str
    algorithm: str
    value: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in _SIGNATURE_ROLES:
            raise _fail("malformed_manifest")
        _require_safe_id(self.key_id)
        _require_safe_id(self.algorithm)
        if type(self.value) is not str or not _SIGNATURE_RE.fullmatch(self.value):
            raise _fail("malformed_manifest")

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-compatible signature."""

        return {
            "role": self.role,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuitySignature:
        """Load a signature, rejecting unknown fields."""

        values = _require_fields(data, _SIGNATURE_FIELDS)
        return cls(
            role=values["role"],
            key_id=values["key_id"],
            algorithm=values["algorithm"],
            value=values["value"],
        )


@dataclass(frozen=True)
class RetentionTombstone:
    """Value-free record of one policy-approved pruned chain segment.

    Attributes:
        first_sequence: First pruned sequence number.
        last_sequence: Last pruned sequence number.
        record_count: Number of pruned entries.
        start_hash: ``previous_hash`` of the first pruned entry.
        end_hash: ``entry_hash`` of the last pruned entry; the retained chain
            links to it.
        segment_digest: :func:`tombstone_segment_digest` of the segment.
        policy_fingerprint: Fingerprint of the approving retention policy.
        retention_report_digest: Integrity digest of the approving retention
            report, when one was supplied.
    """

    first_sequence: int
    last_sequence: int
    record_count: int
    start_hash: str
    end_hash: str
    segment_digest: str
    policy_fingerprint: str
    retention_report_digest: str | None = None

    def __post_init__(self) -> None:
        _require_int(self.first_sequence)
        _require_int(self.last_sequence)
        _require_int(self.record_count, minimum=1)
        _require_digest(self.start_hash)
        _require_digest(self.end_hash)
        _require_digest(self.segment_digest)
        _require_digest(self.policy_fingerprint)
        if self.retention_report_digest is not None:
            _require_digest(self.retention_report_digest)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible tombstone body."""

        return {
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "record_count": self.record_count,
            "start_hash": self.start_hash,
            "end_hash": self.end_hash,
            "segment_digest": self.segment_digest,
            "policy_fingerprint": self.policy_fingerprint,
            "retention_report_digest": self.retention_report_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetentionTombstone:
        """Load a tombstone body, rejecting unknown fields."""

        values = _require_fields(data, _BODY_FIELDS["retention"])
        return cls(**values)


def verify_tombstone_segment(
    tombstone: RetentionTombstone,
    entries: Sequence[AuditChainEntry],
    *,
    chain_id: str,
) -> bool:
    """Return whether ``entries`` are exactly the segment a tombstone describes.

    Use this to check a retained backup copy of a pruned segment before that
    copy is itself disposed of.

    Args:
        tombstone: The tombstone recorded when the segment was pruned.
        entries: Candidate entries, in sequence order.
        chain_id: Opaque chain identifier recorded in the manifest.
    """

    try:
        entries = tuple(entries)
        if len(entries) != tombstone.record_count:
            return False
        sequences = [entry.sequence for entry in entries]
        expected = list(range(tombstone.first_sequence, tombstone.last_sequence + 1))
        if sequences != expected:
            return False
        previous = tombstone.start_hash
        for entry in entries:
            if entry.previous_hash != previous or entry.compute_hash() != (
                entry.entry_hash
            ):
                return False
            previous = entry.entry_hash
        return (
            previous == tombstone.end_hash
            and tombstone_segment_digest(chain_id, entries) == tombstone.segment_digest
        )
    except Exception:  # noqa: BLE001 - verification must fail closed.
        return False


def _validate_body(kind: str, body: Any) -> MappingProxyType[str, Any]:
    fields = _BODY_FIELDS.get(kind)
    if fields is None:
        raise _fail("malformed_manifest")
    values = _require_fields(body, fields)
    if kind == "genesis":
        _require_safe_id(values["key_id"])
    elif kind == "rotation":
        _require_safe_id(values["from_key_id"])
        _require_safe_id(values["to_key_id"])
    elif kind == "retention":
        values = RetentionTombstone.from_dict(values).to_dict()
    elif kind == "restore":
        _require_safe_id(values["key_id"])
        _require_safe_id(values["restore_id"])
        _require_digest(values["backup_digest"])
    return MappingProxyType(dict(sorted(values.items())))


def _event_digest(
    *,
    chain_id: str,
    index: int,
    kind: str,
    epoch: int,
    previous_digest: str,
    next_sequence: int,
    head_hash: str,
    body: Mapping[str, Any],
) -> str:
    return stable_hash(
        {
            "format": f"{CONTINUITY_FORMAT}.event",
            "version": CONTINUITY_VERSION,
            "chain_id": chain_id,
            "index": index,
            "kind": kind,
            "epoch": epoch,
            "previous_digest": previous_digest,
            "next_sequence": next_sequence,
            "head_hash": head_hash,
            "body": dict(body),
        }
    )


@dataclass(frozen=True)
class ContinuityEvent:
    """One hash-linked, signed event in a continuity manifest.

    Attributes:
        index: Position of the event in the manifest, starting at 0.
        kind: One of :data:`CONTINUITY_EVENT_KINDS`.
        epoch: Signer epoch the event belongs to. Transitions carry the epoch
            they open.
        previous_digest: Digest of the preceding event, or the chain origin
            digest for the first event.
        next_sequence: The chain's next sequence number when the event was
            written, i.e. the number of entries ever committed.
        head_hash: The chain head hash when the event was written.
        body: Kind-specific, value-free fields.
        digest: Digest over every field above and the chain identifier.
        signatures: Signatures over ``digest``.
    """

    index: int
    kind: str
    epoch: int
    previous_digest: str
    next_sequence: int
    head_hash: str
    body: Mapping[str, Any]
    digest: str
    signatures: tuple[ContinuitySignature, ...]

    def __post_init__(self) -> None:
        _require_int(self.index)
        if type(self.kind) is not str or self.kind not in CONTINUITY_EVENT_KINDS:
            raise _fail("malformed_manifest")
        _require_int(self.epoch)
        _require_digest(self.previous_digest)
        _require_int(self.next_sequence)
        _require_digest(self.head_hash)
        _require_digest(self.digest)
        object.__setattr__(self, "body", _validate_body(self.kind, self.body))
        signatures = tuple(self.signatures)
        if len(signatures) > len(_SIGNATURE_ROLES) or not all(
            type(item) is ContinuitySignature for item in signatures
        ):
            raise _fail("malformed_manifest")
        object.__setattr__(self, "signatures", signatures)

    def compute_digest(self, chain_id: str) -> str:
        """Return the digest this event should carry for ``chain_id``."""

        return _event_digest(
            chain_id=chain_id,
            index=self.index,
            kind=self.kind,
            epoch=self.epoch,
            previous_digest=self.previous_digest,
            next_sequence=self.next_sequence,
            head_hash=self.head_hash,
            body=self.body,
        )

    @property
    def tombstone(self) -> RetentionTombstone | None:
        """Return the pruning tombstone of a ``retention`` event."""

        if self.kind != "retention":
            return None
        return RetentionTombstone.from_dict(self.body)

    @property
    def opened_key_id(self) -> str | None:
        """Return the key ID an epoch-opening event introduces."""

        if self.kind in {"genesis", "restore"}:
            return str(self.body["key_id"])
        if self.kind == "rotation":
            return str(self.body["to_key_id"])
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible event."""

        return {
            "index": self.index,
            "kind": self.kind,
            "epoch": self.epoch,
            "previous_digest": self.previous_digest,
            "next_sequence": self.next_sequence,
            "head_hash": self.head_hash,
            "body": dict(self.body),
            "digest": self.digest,
            "signatures": [item.to_dict() for item in self.signatures],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuityEvent:
        """Load an event, rejecting unknown fields and malformed values."""

        values = _require_fields(data, _EVENT_FIELDS)
        signatures = _require_list(values["signatures"], limit=len(_SIGNATURE_ROLES))
        return cls(
            index=values["index"],
            kind=values["kind"],
            epoch=values["epoch"],
            previous_digest=values["previous_digest"],
            next_sequence=values["next_sequence"],
            head_hash=values["head_hash"],
            body=values["body"],
            digest=values["digest"],
            signatures=tuple(
                ContinuitySignature.from_dict(item) for item in signatures
            ),
        )


@dataclass(frozen=True)
class SignerEpoch:
    """Summary of one signer epoch and its sequence validity window.

    Attributes:
        epoch: Epoch number, starting at 0.
        key_id: Non-secret identifier of the epoch signing key.
        opened_by: Event kind that opened the epoch.
        opened_at_event: Index of the opening event.
        start_sequence: First chain sequence the epoch vouches for.
        end_sequence: Exclusive end of the window, or ``None`` while active.
        closed_at_event: Index of the closing transition, or ``None``.
    """

    epoch: int
    key_id: str
    opened_by: str
    opened_at_event: int
    start_sequence: int
    end_sequence: int | None = None
    closed_at_event: int | None = None

    def __post_init__(self) -> None:
        _require_int(self.epoch)
        _require_safe_id(self.key_id)
        if self.opened_by not in _EPOCH_OPENERS:
            raise _fail("malformed_manifest")
        _require_int(self.opened_at_event)
        _require_int(self.start_sequence)
        if (self.end_sequence is None) != (self.closed_at_event is None):
            raise _fail("malformed_manifest")
        if self.end_sequence is not None:
            _require_int(self.end_sequence)
            _require_int(self.closed_at_event)

    @property
    def state(self) -> str:
        """Return ``"active"`` for the open epoch and ``"retired"`` otherwise."""

        return "active" if self.end_sequence is None else "retired"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible epoch summary."""

        return {
            "epoch": self.epoch,
            "key_id": self.key_id,
            "opened_by": self.opened_by,
            "opened_at_event": self.opened_at_event,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "closed_at_event": self.closed_at_event,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignerEpoch:
        """Load an epoch summary, rejecting unknown fields."""

        values = _require_fields(data, _EPOCH_FIELDS)
        epoch = cls(
            epoch=values["epoch"],
            key_id=values["key_id"],
            opened_by=values["opened_by"],
            opened_at_event=values["opened_at_event"],
            start_sequence=values["start_sequence"],
            end_sequence=values["end_sequence"],
            closed_at_event=values["closed_at_event"],
        )
        if values["state"] != epoch.state:
            raise _fail("malformed_manifest")
        return epoch


def derive_epochs(events: Sequence[ContinuityEvent]) -> tuple[SignerEpoch, ...]:
    """Derive the ordered signer epochs that a sequence of events describes.

    The derivation is structural only; :func:`verify_continuity` checks the
    signatures and ordering that make the epochs trustworthy.

    Args:
        events: Manifest events in index order.
    """

    epochs: list[SignerEpoch] = []
    for event in events:
        key_id = event.opened_key_id
        if key_id is None:
            continue
        if epochs and event.kind in _TRANSITIONS:
            current = epochs[-1]
            epochs[-1] = SignerEpoch(
                epoch=current.epoch,
                key_id=current.key_id,
                opened_by=current.opened_by,
                opened_at_event=current.opened_at_event,
                start_sequence=current.start_sequence,
                end_sequence=event.next_sequence,
                closed_at_event=event.index,
            )
        epochs.append(
            SignerEpoch(
                epoch=event.epoch,
                key_id=key_id,
                opened_by=event.kind,
                opened_at_event=event.index,
                start_sequence=0 if event.kind == "genesis" else event.next_sequence,
            )
        )
    return tuple(epochs)


@dataclass(frozen=True)
class ContinuityAnchorRef:
    """Externally retained reference to one manifest event.

    Export :meth:`ContinuityManifest.head_ref` into release or compliance
    evidence; passing it back to :func:`verify_continuity` later detects a
    truncated or rewritten manifest and lets retired keys be dropped for the
    history it covers.

    Attributes:
        index: Event index.
        digest: Event digest.
    """

    index: int
    digest: str

    def __post_init__(self) -> None:
        _require_int(self.index)
        _require_digest(self.digest)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible reference."""

        return {"index": self.index, "digest": self.digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuityAnchorRef:
        """Load a reference, rejecting unknown fields."""

        values = _require_fields(data, _ANCHOR_REF_FIELDS)
        return cls(index=values["index"], digest=values["digest"])


@dataclass(frozen=True)
class ContinuityManifest:
    """Versioned, transport-neutral continuity manifest for one audit chain.

    The header fields ``event_count``, ``head_digest``, and ``epochs`` are
    derived from ``events`` when omitted and verified against them by
    :func:`verify_continuity` when loaded from a document.

    Attributes:
        chain_id: Opaque chain identifier. It must not encode patient,
            document, or deployment identifiers.
        events: Events in index order.
        declared_event_count: Event count recorded in the document header.
        declared_head_digest: Last event digest recorded in the header.
        declared_epochs: Epoch summary recorded in the header.
    """

    chain_id: str
    events: tuple[ContinuityEvent, ...]
    declared_event_count: int | None = None
    declared_head_digest: str | None = None
    declared_epochs: tuple[SignerEpoch, ...] | None = None

    def __post_init__(self) -> None:
        _require_safe_id(self.chain_id)
        events = tuple(self.events)
        if not events or len(events) > MAX_CONTINUITY_EVENTS:
            raise _fail("malformed_manifest")
        if not all(type(item) is ContinuityEvent for item in events):
            raise _fail("malformed_manifest")
        object.__setattr__(self, "events", events)
        if self.declared_event_count is None:
            object.__setattr__(self, "declared_event_count", len(events))
        else:
            _require_int(self.declared_event_count)
        if self.declared_head_digest is None:
            object.__setattr__(self, "declared_head_digest", events[-1].digest)
        else:
            _require_digest(self.declared_head_digest)
        if self.declared_epochs is None:
            object.__setattr__(self, "declared_epochs", derive_epochs(events))
        else:
            object.__setattr__(self, "declared_epochs", tuple(self.declared_epochs))

    @property
    def event_count(self) -> int:
        """Return the number of events."""

        return len(self.events)

    @property
    def head_event(self) -> ContinuityEvent:
        """Return the most recent event."""

        return self.events[-1]

    @property
    def head_digest(self) -> str:
        """Return the most recent event digest."""

        return self.events[-1].digest

    @property
    def epochs(self) -> tuple[SignerEpoch, ...]:
        """Return the signer epochs derived from the events."""

        return derive_epochs(self.events)

    @property
    def current_epoch(self) -> SignerEpoch:
        """Return the active signer epoch."""

        epochs = self.epochs
        if not epochs:
            raise _fail("epoch_mismatch")
        return epochs[-1]

    @property
    def used_key_ids(self) -> frozenset[str]:
        """Return every key ID that has opened an epoch."""

        return frozenset(epoch.key_id for epoch in self.epochs)

    @property
    def tombstones(self) -> tuple[RetentionTombstone, ...]:
        """Return the pruning tombstones in event order."""

        return tuple(
            tombstone
            for event in self.events
            if (tombstone := event.tombstone) is not None
        )

    @property
    def pruned_through(self) -> int:
        """Return the number of chain entries the tombstones cover."""

        tombstones = self.tombstones
        return tombstones[-1].last_sequence + 1 if tombstones else 0

    def head_ref(self) -> ContinuityAnchorRef:
        """Return a reference to the head event for external retention."""

        head = self.head_event
        return ContinuityAnchorRef(index=head.index, digest=head.digest)

    def with_event(self, event: ContinuityEvent) -> ContinuityManifest:
        """Return a new manifest with ``event`` appended."""

        return ContinuityManifest(chain_id=self.chain_id, events=(*self.events, event))

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, value-free manifest document."""

        epochs = self.declared_epochs or ()
        return {
            "format": CONTINUITY_FORMAT,
            "version": CONTINUITY_VERSION,
            "chain_id": self.chain_id,
            "chain_format": CHAIN_FORMAT,
            "chain_genesis_hash": GENESIS_HASH,
            "digest_algorithm": CONTINUITY_DIGEST_ALGORITHM,
            "signature_algorithm": CONTINUITY_SIGNATURE_ALGORITHM,
            "event_count": self.declared_event_count,
            "head_digest": self.declared_head_digest,
            "epochs": [epoch.to_dict() for epoch in epochs],
            "events": [event.to_dict() for event in self.events],
        }

    def to_json(self) -> str:
        """Serialize the manifest deterministically."""

        return _canonical_json(self.to_dict())

    def write(self, path: str | Path) -> Path:
        """Atomically and durably write the manifest document."""

        return _atomic_write_text(Path(path), self.to_json() + "\n")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuityManifest:
        """Load a manifest document.

        Raises:
            AuditContinuityError: With ``unsupported_version`` for an unknown
                format, version, or algorithm, ``chain_mismatch`` for a manifest
                bound to another chain format, and ``malformed_manifest`` for
                any structural problem.
        """

        if not isinstance(data, Mapping):
            raise _fail("malformed_manifest")
        if data.get("format") != CONTINUITY_FORMAT:
            raise _fail("unsupported_version")
        version = data.get("version")
        if type(version) is not int or version != CONTINUITY_VERSION:
            raise _fail("unsupported_version")
        values = _require_fields(data, _MANIFEST_FIELDS)
        if (
            values["digest_algorithm"] != CONTINUITY_DIGEST_ALGORITHM
            or values["signature_algorithm"] != CONTINUITY_SIGNATURE_ALGORITHM
        ):
            raise _fail("unsupported_version")
        if (
            values["chain_format"] != CHAIN_FORMAT
            or values["chain_genesis_hash"] != GENESIS_HASH
        ):
            raise _fail("chain_mismatch")
        events = _require_list(values["events"], limit=MAX_CONTINUITY_EVENTS)
        epochs = _require_list(values["epochs"], limit=MAX_CONTINUITY_EVENTS)
        return cls(
            chain_id=_require_safe_id(values["chain_id"]),
            events=tuple(ContinuityEvent.from_dict(item) for item in events),
            declared_event_count=_require_int(values["event_count"]),
            declared_head_digest=_require_digest(values["head_digest"]),
            declared_epochs=tuple(SignerEpoch.from_dict(item) for item in epochs),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> ContinuityManifest:
        """Load a manifest from strict, size-bounded JSON."""

        size = len(data) if isinstance(data, bytes) else len(data.encode("utf-8"))
        if size > MAX_CONTINUITY_JSON_BYTES:
            raise _fail("malformed_manifest")
        try:
            parsed = json.loads(
                data,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (RecursionError, TypeError, ValueError, UnicodeDecodeError):
            raise _fail("malformed_manifest") from None
        return cls.from_dict(parsed)

    @classmethod
    def load(cls, path: str | Path) -> ContinuityManifest:
        """Load a manifest document from disk."""

        source = Path(path)
        try:
            if source.stat().st_size > MAX_CONTINUITY_JSON_BYTES:
                raise _fail("malformed_manifest")
            payload = source.read_bytes()
        except OSError:
            raise _fail("io_failed") from None
        return cls.from_json(payload)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite constant")


@dataclass(frozen=True)
class ContinuityVerificationReport:
    """Bounded, value-free outcome of :func:`verify_continuity`.

    The report holds only versions, counts, digests, opaque identifiers, key
    identifiers, epoch windows, and a controlled reason code. It is suitable
    for offline release or compliance evidence.

    Attributes:
        valid: Whether continuity verified.
        code: One of :data:`CONTINUITY_REASON_CODES`.
        event_index: Index of the event that failed, when one is identifiable.
        chain_id: Opaque chain identifier, when the manifest could be read.
        manifest_digest: Head event digest, when the manifest could be read.
        chain_head_hash: Audit chain head hash, when the chain could be read.
        counts: Fixed set of non-negative counters.
        epochs: Derived signer epochs.
        policy_fingerprints: Retention-policy fingerprints that approved
            pruning, in first-use order.
    """

    valid: bool
    code: str
    event_index: int | None
    chain_id: str | None
    manifest_digest: str | None
    chain_head_hash: str | None
    counts: Mapping[str, int]
    epochs: tuple[SignerEpoch, ...] = ()
    policy_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.code not in CONTINUITY_REASON_CODES or self.valid != (
            self.code == "valid"
        ):
            raise ValueError("continuity report code is not supported")
        counts = {name: int(self.counts.get(name, 0)) for name in _COUNT_FIELDS}
        if any(value < 0 for value in counts.values()):
            raise ValueError("continuity report counts must be non-negative")
        object.__setattr__(self, "counts", MappingProxyType(counts))
        object.__setattr__(self, "epochs", tuple(self.epochs))
        object.__setattr__(self, "policy_fingerprints", tuple(self.policy_fingerprints))

    def __bool__(self) -> bool:
        return self.valid

    def _payload(self) -> dict[str, Any]:
        return {
            "format": CONTINUITY_REPORT_FORMAT,
            "version": CONTINUITY_REPORT_VERSION,
            "manifest_version": CONTINUITY_VERSION,
            "valid": self.valid,
            "code": self.code,
            "event_index": self.event_index,
            "chain_id": self.chain_id,
            "manifest_digest": self.manifest_digest,
            "chain_head_hash": self.chain_head_hash,
            "counts": dict(self.counts),
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "policy_fingerprints": list(self.policy_fingerprints),
        }

    @property
    def report_digest(self) -> str:
        """Return the digest binding the complete report payload."""

        return stable_hash(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Return the report with its integrity digest."""

        return {**self._payload(), "report_digest": self.report_digest}

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )


class _Failure(Exception):
    def __init__(self, code: str, index: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.index = index


def _coerce_chain(chain: AuditChain | Mapping[str, Any]) -> AuditChain:
    if isinstance(chain, AuditChain):
        return chain
    if not isinstance(chain, Mapping):
        raise _Failure("chain_invalid")
    try:
        return AuditChain.from_dict(chain)
    except Exception:  # noqa: BLE001 - malformed input must fail closed.
        raise _Failure("chain_invalid") from None


def _coerce_manifest(
    manifest: ContinuityManifest | Mapping[str, Any],
) -> ContinuityManifest:
    if isinstance(manifest, ContinuityManifest):
        return manifest
    try:
        return ContinuityManifest.from_dict(manifest)
    except AuditContinuityError as error:
        raise _Failure(error.code) from None
    except Exception:  # noqa: BLE001 - malformed input must fail closed.
        raise _Failure("malformed_manifest") from None


def _coerce_refs(
    refs: Iterable[ContinuityAnchorRef | Mapping[str, Any]],
) -> tuple[ContinuityAnchorRef, ...]:
    try:
        items = list(refs)
        if len(items) > MAX_CONTINUITY_EVENTS:
            raise _fail("invalid_argument")
        return tuple(
            item
            if isinstance(item, ContinuityAnchorRef)
            else ContinuityAnchorRef.from_dict(item)
            for item in items
        )
    except Exception:  # noqa: BLE001 - malformed input must fail closed.
        raise _Failure("malformed_manifest") from None


def _coerce_fingerprints(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    try:
        items = frozenset(values)
    except TypeError:
        raise _Failure("policy_mismatch") from None
    if not all(type(item) is str and _DIGEST_RE.fullmatch(item) for item in items):
        raise _Failure("policy_mismatch")
    return items


_DIGEST_MISMATCH_CODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anchor": "anchor_mismatch",
        "retention": "tombstone_mismatch",
        "genesis": "epoch_mismatch",
        "rotation": "epoch_mismatch",
        "restore": "epoch_mismatch",
    }
)

_CHAIN_CODE_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "entry_count_deficit": "sequence_gap",
        "sequence_gap": "sequence_gap",
        "head_mismatch": "sequence_gap",
    }
)


class _Verification:
    """One offline replay of a continuity manifest against an audit chain."""

    def __init__(
        self,
        *,
        key_provider: KeyProvider,
        trusted_anchors: tuple[ContinuityAnchorRef, ...],
        allowed_policy_fingerprints: frozenset[str] | None,
        expected_chain_id: str | None,
        allow_unanchored_tail: bool,
        tolerate_unavailable_keys: bool,
    ) -> None:
        self.provider = key_provider
        self.trusted = trusted_anchors
        self.allowed = allowed_policy_fingerprints
        self.expected_chain_id = expected_chain_id
        self.allow_tail = allow_unanchored_tail
        self.tolerate = tolerate_unavailable_keys
        self.counts: dict[str, int] = dict.fromkeys(_COUNT_FIELDS, 0)
        self.manifest: ContinuityManifest | None = None
        self.chain: AuditChain | None = None
        self.policies: list[str] = []
        self.checkpoint = -1

    def report(
        self, code: str, index: int | None = None
    ) -> ContinuityVerificationReport:
        manifest = self.manifest
        chain = self.chain
        epochs: tuple[SignerEpoch, ...] = ()
        if manifest is not None:
            with contextlib.suppress(Exception):
                epochs = derive_epochs(manifest.events)
            self.counts["event_count"] = manifest.event_count
            self.counts["epoch_count"] = len(epochs)
            for kind in ("anchor", "rotation", "retention", "restore"):
                self.counts[f"{kind}_count"] = sum(
                    1 for event in manifest.events if event.kind == kind
                )
        if chain is not None:
            self.counts["next_sequence"] = chain.next_sequence
            self.counts["retained_entry_count"] = len(chain.entries)
            self.counts["pruned_entry_count"] = chain.base_sequence
        return ContinuityVerificationReport(
            valid=code == "valid",
            code=code,
            event_index=index,
            chain_id=manifest.chain_id if manifest is not None else None,
            manifest_digest=manifest.head_digest if manifest is not None else None,
            chain_head_hash=chain.head_hash if chain is not None else None,
            counts=self.counts,
            epochs=epochs,
            policy_fingerprints=tuple(self.policies),
        )

    def run(
        self,
        chain: AuditChain | Mapping[str, Any],
        manifest: ContinuityManifest | Mapping[str, Any],
    ) -> ContinuityVerificationReport:
        try:
            self.manifest = _coerce_manifest(manifest)
            self.chain = _coerce_chain(chain)
            self._verify_header()
            self._verify_indices()
            self._verify_trusted_anchors()
            self._replay_events()
            self._verify_declared_summary()
            self._verify_chain()
            self._verify_coverage()
            self._verify_chain_states()
            self._verify_tail()
        except _Failure as failure:
            return self.report(failure.code, failure.index)
        return self.report("valid")

    # Manifest-level checks -------------------------------------------------

    def _verify_header(self) -> None:
        manifest = self._manifest()
        if (
            self.expected_chain_id is not None
            and manifest.chain_id != self.expected_chain_id
        ):
            raise _Failure("chain_mismatch")

    def _verify_indices(self) -> None:
        manifest = self._manifest()
        indices = [event.index for event in manifest.events]
        seen: set[int] = set()
        for index in indices:
            if index in seen:
                raise _Failure("event_duplicate", index)
            seen.add(index)
        expected = list(range(len(indices)))
        if indices != sorted(indices):
            position = next(
                position
                for position, (actual, wanted) in enumerate(
                    zip(indices, sorted(indices), strict=True)
                )
                if actual != wanted
            )
            raise _Failure("event_reordered", position)
        if indices != expected:
            missing = next(
                wanted
                for actual, wanted in zip(indices, expected, strict=True)
                if actual != wanted
            )
            raise _Failure("event_gap", missing)
        if manifest.declared_event_count != len(indices):
            raise _Failure("event_gap", len(indices))

    def _verify_trusted_anchors(self) -> None:
        manifest = self._manifest()
        events = manifest.events
        restore_indices = [event.index for event in events if event.kind == "restore"]
        for ref in self.trusted:
            if ref.index < len(events) and events[ref.index].digest == ref.digest:
                self.counts["trusted_anchor_count"] += 1
                self.checkpoint = max(self.checkpoint, ref.index)
            elif any(index <= ref.index for index in restore_indices):
                self.counts["superseded_trusted_anchor_count"] += 1
            elif ref.index >= len(events):
                raise _Failure("anchor_missing", ref.index)
            else:
                raise _Failure("fork", ref.index)

    def _replay_events(self) -> None:
        manifest = self._manifest()
        chain_id = manifest.chain_id
        previous = continuity_origin_digest(chain_id)
        epoch_keys: list[str] = []
        current_key: str | None = None
        last_next = 0
        coverage_next = 0
        coverage_hash = GENESIS_HASH
        for event in manifest.events:
            index = event.index
            if index == 0 and event.previous_digest != previous:
                # The origin digest binds the chain identifier; a mismatch
                # means the manifest was issued for another chain.
                raise _Failure("chain_mismatch", index)
            if event.compute_digest(chain_id) != event.digest:
                raise _Failure(_DIGEST_MISMATCH_CODES[event.kind], index)
            if event.previous_digest != previous:
                raise _Failure("fork", index)
            if (index == 0) != (event.kind == "genesis"):
                raise _Failure("epoch_mismatch", index)
            current_epoch = len(epoch_keys) - 1
            expected_epoch = (
                current_epoch + 1 if event.kind in _EPOCH_OPENERS else current_epoch
            )
            if event.epoch != expected_epoch:
                raise _Failure("epoch_mismatch", index)

            retired = frozenset(epoch_keys) - {current_key}
            if event.kind == "genesis":
                expected = {"signer": str(event.body["key_id"])}
            elif event.kind in {"anchor", "retention"}:
                expected = {"signer": str(current_key)}
            elif event.kind == "rotation":
                if event.body["from_key_id"] != current_key:
                    raise _Failure("epoch_mismatch", index)
                to_key = str(event.body["to_key_id"])
                if to_key in epoch_keys:
                    raise _Failure("retired_signer_key", index)
                expected = {"outgoing": str(current_key), "incoming": to_key}
            else:
                restore_key = str(event.body["key_id"])
                if restore_key in retired:
                    raise _Failure("retired_signer_key", index)
                expected = {"incoming": restore_key}
            all_keys = frozenset(epoch_keys)
            self._verify_signatures(event, expected, all_keys)

            if event.kind != "restore" and event.next_sequence < last_next:
                raise _Failure("fork", index)
            tombstone = event.tombstone
            if tombstone is not None:
                if tombstone.last_sequence < tombstone.first_sequence or (
                    tombstone.record_count
                    != tombstone.last_sequence - tombstone.first_sequence + 1
                ):
                    raise _Failure("tombstone_mismatch", index)
                if tombstone.first_sequence > coverage_next:
                    raise _Failure("sequence_gap", index)
                if (
                    tombstone.first_sequence < coverage_next
                    or tombstone.start_hash != coverage_hash
                    or tombstone.last_sequence >= event.next_sequence
                ):
                    raise _Failure("tombstone_mismatch", index)
                if (
                    self.allowed is not None
                    and tombstone.policy_fingerprint not in self.allowed
                ):
                    raise _Failure("policy_mismatch", index)
                if tombstone.policy_fingerprint not in self.policies:
                    self.policies.append(tombstone.policy_fingerprint)
                coverage_next = tombstone.last_sequence + 1
                coverage_hash = tombstone.end_hash
            if event.kind == "restore" and event.next_sequence < coverage_next:
                raise _Failure("tombstone_mismatch", index)

            opened = event.opened_key_id
            if opened is not None:
                epoch_keys.append(opened)
                current_key = opened
            last_next = event.next_sequence
            previous = event.digest

    def _verify_signatures(
        self,
        event: ContinuityEvent,
        expected: Mapping[str, str],
        known_keys: frozenset[str],
    ) -> None:
        index = event.index
        roles = [signature.role for signature in event.signatures]
        if len(roles) != len(set(roles)):
            raise _Failure("signature_invalid", index)
        if set(roles) != set(expected):
            if event.kind in _TRANSITIONS and set(expected) - set(roles):
                raise _Failure("unsigned_epoch_transition", index)
            raise _Failure("signature_invalid", index)
        for signature in event.signatures:
            wanted = expected[signature.role]
            if signature.key_id != wanted:
                if signature.key_id in known_keys:
                    raise _Failure("retired_signer_key", index)
                raise _Failure("unknown_signer_key", index)
            if signature.algorithm != CONTINUITY_SIGNATURE_ALGORITHM:
                raise _Failure("signature_invalid", index)
            try:
                status = _digest_signature_status(
                    self.provider,
                    event.digest,
                    signature.value,
                    key_id=signature.key_id,
                    purpose=CONTINUITY_SIGNATURE_PURPOSE,
                    role=signature.role,
                )
            except Exception:  # noqa: BLE001 - provider failures fail closed.
                status = "unavailable"
            if status == "unavailable":
                if self.tolerate or index <= self.checkpoint:
                    self.counts["unverified_signature_count"] += 1
                    continue
                raise _Failure("signer_key_unavailable", index)
            if status != "valid":
                raise _Failure("signature_invalid", index)

    def _verify_declared_summary(self) -> None:
        manifest = self._manifest()
        if manifest.declared_head_digest != manifest.head_digest:
            raise _Failure("fork", manifest.event_count - 1)
        derived = [epoch.to_dict() for epoch in derive_epochs(manifest.events)]
        declared = [epoch.to_dict() for epoch in manifest.declared_epochs or ()]
        if derived != declared:
            raise _Failure("epoch_mismatch")

    # Chain-level checks ----------------------------------------------------

    def _verify_chain(self) -> None:
        result = self._chain().verify()
        if not result.valid:
            raise _Failure(_CHAIN_CODE_MAP.get(str(result.code), "chain_invalid"))

    def _verify_coverage(self) -> None:
        manifest = self._manifest()
        chain = self._chain()
        tombstones = manifest.tombstones
        coverage_next = manifest.pruned_through
        coverage_hash = tombstones[-1].end_hash if tombstones else GENESIS_HASH
        if chain.base_sequence > coverage_next:
            raise _Failure("sequence_gap")
        if chain.base_sequence < coverage_next:
            for tombstone in tombstones:
                if tombstone.last_sequence < chain.base_sequence:
                    continue
                if tombstone.first_sequence < chain.base_sequence:
                    raise _Failure("tombstone_mismatch")
                entries = [
                    chain.entry_at(sequence)
                    for sequence in range(
                        tombstone.first_sequence, tombstone.last_sequence + 1
                    )
                ]
                present = [entry for entry in entries if entry is not None]
                if len(present) != len(entries):
                    raise _Failure("sequence_gap")
                if not verify_tombstone_segment(
                    tombstone, present, chain_id=manifest.chain_id
                ):
                    raise _Failure("tombstone_mismatch")
            raise _Failure("incomplete_transition")
        if chain.base_hash != coverage_hash:
            raise _Failure("tombstone_mismatch")

    def _verify_chain_states(self) -> None:
        manifest = self._manifest()
        chain = self._chain()
        restores = [
            (event.index, event.next_sequence)
            for event in manifest.events
            if event.kind == "restore"
        ]
        by_last = {item.last_sequence: item for item in manifest.tombstones}
        by_first = {item.first_sequence: item for item in manifest.tombstones}
        for event in manifest.events:
            if any(
                index > event.index and restored < event.next_sequence
                for index, restored in restores
            ):
                self.counts["superseded_event_count"] += 1
                continue
            committed = event.next_sequence
            if committed == 0:
                if event.head_hash != GENESIS_HASH:
                    raise _Failure("anchor_mismatch", event.index)
                continue
            sequence = committed - 1
            if sequence >= chain.next_sequence:
                raise _Failure("sequence_gap", event.index)
            entry = chain.entry_at(sequence)
            if entry is not None:
                if entry.entry_hash != event.head_hash:
                    raise _Failure("anchor_mismatch", event.index)
                continue
            boundary = by_last.get(sequence)
            if boundary is not None:
                if boundary.end_hash != event.head_hash:
                    raise _Failure("tombstone_mismatch", event.index)
                continue
            following = by_first.get(committed)
            if following is not None and following.start_hash != event.head_hash:
                raise _Failure("tombstone_mismatch", event.index)
            self.counts["pruned_anchor_count"] += 1

    def _verify_tail(self) -> None:
        manifest = self._manifest()
        chain = self._chain()
        unanchored = chain.next_sequence - manifest.head_event.next_sequence
        self.counts["unanchored_entry_count"] = max(unanchored, 0)
        if unanchored > 0 and not self.allow_tail:
            raise _Failure("unanchored_tail")

    def _manifest(self) -> ContinuityManifest:
        if self.manifest is None:  # pragma: no cover - guarded by run().
            raise _Failure("malformed_manifest")
        return self.manifest

    def _chain(self) -> AuditChain:
        if self.chain is None:  # pragma: no cover - guarded by run().
            raise _Failure("chain_invalid")
        return self.chain


def _run_verification(
    chain: AuditChain | Mapping[str, Any],
    manifest: ContinuityManifest | Mapping[str, Any],
    *,
    key_provider: KeyProvider,
    trusted_anchors: Iterable[ContinuityAnchorRef | Mapping[str, Any]] = (),
    allowed_policy_fingerprints: Iterable[str] | None = None,
    expected_chain_id: str | None = None,
    allow_unanchored_tail: bool = False,
    tolerate_unavailable_keys: bool = False,
) -> ContinuityVerificationReport:
    _validate_provider(key_provider)
    empty = _Verification(
        key_provider=key_provider,
        trusted_anchors=(),
        allowed_policy_fingerprints=None,
        expected_chain_id=None,
        allow_unanchored_tail=False,
        tolerate_unavailable_keys=False,
    )
    try:
        refs = _coerce_refs(trusted_anchors)
        allowed = _coerce_fingerprints(allowed_policy_fingerprints)
        if expected_chain_id is not None and (
            type(expected_chain_id) is not str
            or not _SAFE_ID_RE.fullmatch(expected_chain_id)
        ):
            raise _Failure("chain_mismatch")
    except _Failure as failure:
        return empty.report(failure.code, failure.index)
    return _Verification(
        key_provider=key_provider,
        trusted_anchors=refs,
        allowed_policy_fingerprints=allowed,
        expected_chain_id=expected_chain_id,
        allow_unanchored_tail=bool(allow_unanchored_tail),
        tolerate_unavailable_keys=tolerate_unavailable_keys,
    ).run(chain, manifest)


def verify_continuity(
    chain: AuditChain | Mapping[str, Any],
    manifest: ContinuityManifest | Mapping[str, Any],
    *,
    key_provider: KeyProvider,
    trusted_anchors: Iterable[ContinuityAnchorRef | Mapping[str, Any]] = (),
    allowed_policy_fingerprints: Iterable[str] | None = None,
    expected_chain_id: str | None = None,
    allow_unanchored_tail: bool = False,
) -> ContinuityVerificationReport:
    """Verify an audit chain's continuity offline and fail closed.

    The verifier replays every manifest event: it recomputes digests and
    links, checks epoch numbering and that every event is signed by the key of
    the epoch it belongs to (both keys for a rotation), checks that pruning
    tombstones are contiguous and agree with the retained chain and with
    earlier signed anchors, and checks that every signed chain-head commitment
    still matches the chain unless a later signed restore rolled it back.

    Args:
        chain: The audit chain as an object or decoded JSON document.
        manifest: The continuity manifest as an object or decoded JSON.
        key_provider: Caller-owned provider resolving every epoch key ID that
            must still be verified. It is only called with key IDs.
        trusted_anchors: Externally retained :class:`ContinuityAnchorRef`
            values, for example from release evidence. Each must still be in
            the manifest unless a later restore superseded it. Signatures at or
            before the newest matching reference may use keys the provider no
            longer holds; they are counted as ``unverified_signature_count``.
        allowed_policy_fingerprints: When given, every tombstone must name one
            of these retention-policy fingerprints.
        expected_chain_id: When given, the manifest must belong to this chain.
        allow_unanchored_tail: Accept chain entries appended after the last
            signed event. They are reported in ``unanchored_entry_count``.

    Returns:
        A value-free report. ``report.valid`` is ``True`` only when every check
        passed; otherwise ``report.code`` names the first failure.

    Raises:
        TypeError: If ``key_provider`` is neither callable nor a mapping.
    """

    return _run_verification(
        chain,
        manifest,
        key_provider=key_provider,
        trusted_anchors=trusted_anchors,
        allowed_policy_fingerprints=allowed_policy_fingerprints,
        expected_chain_id=expected_chain_id,
        allow_unanchored_tail=allow_unanchored_tail,
    )


# Durable store ----------------------------------------------------------------

_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _process_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Serialize writers across threads and processes with a permanent lock."""

    with _process_lock(path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
        except OSError:
            raise _fail("lock_failed") from None
        try:
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise _fail("lock_failed")
                if sys.platform == "win32":
                    import msvcrt

                    if os.fstat(descriptor).st_size == 0:
                        os.write(descriptor, b"0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError:
                raise _fail("lock_failed") from None
            yield
        finally:
            # Closing the descriptor releases the OS lock. The lock file is
            # never unlinked, which would split writers into two lock domains.
            os.close(descriptor)


@dataclass(frozen=True)
class ContinuityRecoveryResult:
    """Value-free outcome of :meth:`AuditContinuityStore.recover`.

    Attributes:
        retention_completed: Whether an interrupted pruning was completed.
        pruned_entry_count: Entries removed while completing it.
        temporary_files_removed: Stale temporary files removed.
    """

    retention_completed: bool
    pruned_entry_count: int
    temporary_files_removed: int


def _resolve_policy_fingerprint(
    *,
    policy_fingerprint: str | None,
    policy: AuditRetentionPolicy | None,
    retention_report: AuditRetentionReport | None,
) -> tuple[str, str | None]:
    candidates: list[Any] = []
    report_digest: str | None = None
    if policy_fingerprint is not None:
        candidates.append(policy_fingerprint)
    if policy is not None:
        candidates.append(getattr(policy, "fingerprint", None))
    if retention_report is not None:
        candidates.append(getattr(retention_report, "policy_fingerprint", None))
        report_digest = getattr(retention_report, "integrity_digest", None)
        if type(report_digest) is not str or not _DIGEST_RE.fullmatch(report_digest):
            raise _fail("invalid_argument")
    if not candidates:
        raise _fail("invalid_argument")
    if not all(type(item) is str and _DIGEST_RE.fullmatch(item) for item in candidates):
        raise _fail("invalid_argument")
    if len(set(candidates)) != 1:
        raise _fail("policy_mismatch")
    return str(candidates[0]), report_digest


class AuditContinuityStore:
    """Keep an audit chain file and its continuity manifest in step.

    All operations hold an exclusive lock for their whole read-modify-write
    cycle, so appends, anchors, rotations, pruning, and restore transitions
    from any number of threads or processes are applied one at a time and
    each transition commits to the exact chain head at that moment.

    Args:
        chain_path: Audit chain JSON document.
        key_provider: Caller-owned provider that resolves signing keys by
            non-secret key ID. OpenMed never persists the material.
        manifest_path: Continuity manifest path. Defaults to
            ``<chain stem>.continuity.json`` beside the chain.
        anchor_interval: When set, :meth:`append` writes a signed anchor
            whenever this many entries have been appended since the last event.
    """

    def __init__(
        self,
        chain_path: str | Path,
        *,
        key_provider: KeyProvider,
        manifest_path: str | Path | None = None,
        anchor_interval: int | None = None,
    ) -> None:
        _validate_provider(key_provider)
        if anchor_interval is not None and (
            type(anchor_interval) is not int or anchor_interval < 1
        ):
            raise _fail("invalid_argument")
        self._chain_path = Path(chain_path)
        self._manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self._chain_path.with_name(f"{self._chain_path.stem}.continuity.json")
        )
        if self._manifest_path == self._chain_path:
            raise _fail("invalid_argument")
        self._lock_path = self._manifest_path.with_name(
            f".{self._manifest_path.name}.lock"
        )
        self._key_provider = key_provider
        self._anchor_interval = anchor_interval

    def __repr__(self) -> str:
        return f"{type(self).__name__}(anchor_interval={self._anchor_interval!r})"

    @property
    def chain_path(self) -> Path:
        """Return the audit chain document path."""

        return self._chain_path

    @property
    def manifest_path(self) -> Path:
        """Return the continuity manifest path."""

        return self._manifest_path

    def initialize(
        self, *, key_id: str, chain_id: str | None = None
    ) -> ContinuityEvent:
        """Create the manifest and open signer epoch 0.

        An existing, valid, unpruned chain is adopted: the genesis event
        commits to its current head.

        Args:
            key_id: Key ID of the first signing epoch.
            chain_id: Optional opaque chain identifier. A random one is
                generated when omitted.

        Raises:
            AuditContinuityError: ``already_initialized`` if a manifest exists,
                ``chain_invalid`` for an invalid or pruned chain, or
                ``signer_key_unavailable`` if the key cannot be resolved.
        """

        resolved_chain_id = (
            _new_opaque_id("chain") if chain_id is None else self._safe_id(chain_id)
        )
        resolved_key = self._safe_id(key_id)
        with self._locked():
            if self._manifest_path.exists():
                raise _fail("already_initialized")
            chain = self._read_chain(None)
            if not chain.verify().valid or chain.is_pruned:
                raise _fail("chain_invalid")
            if not self._chain_path.exists():
                self._write_chain(chain)
            event = self._build_event(
                None,
                chain,
                chain_id=resolved_chain_id,
                kind="genesis",
                epoch=0,
                body={"key_id": resolved_key},
                signers=(("signer", resolved_key),),
            )
            self._write_manifest(
                ContinuityManifest(chain_id=resolved_chain_id, events=(event,))
            )
            return event

    def append(self, report: AuditReport) -> AuditChainEntry:
        """Append one audit report to the chain under the store lock.

        Args:
            report: The audit report to commit.

        Returns:
            The appended chain entry.

        Raises:
            AuditContinuityError: If the store is not initialized or the chain
                cannot be recovered or written.
            ValueError: If the chain or report is invalid.
        """

        with self._locked():
            manifest = self._read_manifest()
            chain = self._read_chain(manifest)
            chain = self._recover_locked(chain, manifest)[0]
            self._require_head_commitment(chain, manifest)
            entry = chain.append(report)
            self._write_chain(chain)
            interval = self._anchor_interval
            if (
                interval is not None
                and chain.next_sequence - manifest.head_event.next_sequence >= interval
            ):
                self._write_manifest(
                    manifest.with_event(self._signed_event(manifest, chain, "anchor"))
                )
            return entry

    def anchor(self) -> ContinuityEvent:
        """Write a signed digest of the current chain head.

        Raises:
            AuditContinuityError: With a verification code if the existing
                chain and manifest are inconsistent.
        """

        with self._locked():
            chain, manifest = self._read_consistent()
            event = self._signed_event(manifest, chain, "anchor")
            self._write_manifest(manifest.with_event(event))
            return event

    def rotate(self, new_key_id: str) -> ContinuityEvent:
        """Atomically close the active signer epoch and open a new one.

        The rotation event commits to the exact chain head at the moment of
        rollover and is signed by both the outgoing and the incoming key, so
        every entry belongs to exactly one epoch and no window is unsigned.
        Rotating to the key that is already active is idempotent and returns
        the event that opened the active epoch.

        Args:
            new_key_id: Key ID of the incoming epoch. It must not have been
                used by any earlier epoch.

        Raises:
            AuditContinuityError: ``retired_signer_key`` when the key ID was
                used before, ``signer_key_unavailable`` when either key cannot
                be resolved, or a verification code for inconsistent state.
        """

        incoming = self._safe_id(new_key_id)
        with self._locked():
            chain, manifest = self._read_consistent()
            current = manifest.current_epoch
            if incoming == current.key_id:
                return manifest.events[current.opened_at_event]
            if incoming in manifest.used_key_ids:
                raise _fail("retired_signer_key")
            event = self._build_event(
                manifest,
                chain,
                chain_id=manifest.chain_id,
                kind="rotation",
                epoch=current.epoch + 1,
                body={"from_key_id": current.key_id, "to_key_id": incoming},
                signers=(("outgoing", current.key_id), ("incoming", incoming)),
            )
            self._write_manifest(manifest.with_event(event))
            return event

    def prune_through(
        self,
        last_sequence: int,
        *,
        policy_fingerprint: str | None = None,
        policy: AuditRetentionPolicy | None = None,
        retention_report: AuditRetentionReport | None = None,
    ) -> ContinuityEvent:
        """Prune chain entries through ``last_sequence`` behind a tombstone.

        The signed tombstone is made durable before the chain is rewritten.
        If the process stops in between, the next store operation (or
        :meth:`recover`) checks the still-present entries against the
        tombstone and completes the pruning, so retention is never skipped and
        verification never breaks.

        Args:
            last_sequence: Highest retained sequence number to remove.
            policy_fingerprint: Fingerprint of the approving retention policy.
            policy: Approving :class:`~openmed.risk.audit_retention.
                AuditRetentionPolicy`; its fingerprint is recorded.
            retention_report: Approving
                :class:`~openmed.risk.audit_retention.AuditRetentionReport`;
                its policy fingerprint and integrity digest are recorded.

        Raises:
            AuditContinuityError: ``invalid_argument`` without an approving
                policy or for an out-of-range sequence, ``policy_mismatch``
                when the supplied approvals disagree, or a verification code
                for inconsistent state.
        """

        fingerprint, report_digest = _resolve_policy_fingerprint(
            policy_fingerprint=policy_fingerprint,
            policy=policy,
            retention_report=retention_report,
        )
        with self._locked():
            chain, manifest = self._read_consistent()
            if (
                type(last_sequence) is not int
                or not chain.base_sequence <= last_sequence < chain.next_sequence
            ):
                raise _fail("invalid_argument")
            segment = tuple(
                entry for entry in chain.entries if entry.sequence <= last_sequence
            )
            tombstone = RetentionTombstone(
                first_sequence=segment[0].sequence,
                last_sequence=segment[-1].sequence,
                record_count=len(segment),
                start_hash=segment[0].previous_hash,
                end_hash=segment[-1].entry_hash,
                segment_digest=tombstone_segment_digest(manifest.chain_id, segment),
                policy_fingerprint=fingerprint,
                retention_report_digest=report_digest,
            )
            event = self._signed_event(
                manifest, chain, "retention", body=tombstone.to_dict()
            )
            self._write_manifest(manifest.with_event(event))
            chain.prune_through(last_sequence)
            self._write_chain(chain)
            return event

    def record_restore_transition(
        self,
        *,
        backup_digest: str,
        restore_id: str | None = None,
        key_id: str | None = None,
    ) -> ContinuityEvent:
        """Record that the chain was restored from a backup.

        Call this after the chain document (and, when it was backed up, the
        manifest) has been restored, before new appends. Pruning recorded in
        the manifest is re-applied first, so a restore never brings pruned
        entries back. The restore event opens a new signer epoch, commits to
        the restored chain head, and marks any later chain-head commitments
        from the lost timeline as superseded.

        Args:
            backup_digest: ``sha256:<hex>`` digest identifying the backup, for
                example the recovery-point manifest digest.
            restore_id: Optional opaque restore identifier. A random one is
                generated when omitted.
            key_id: Key ID for the restore epoch. Defaults to the active key;
                a key from an earlier, retired epoch is refused.

        Raises:
            AuditContinuityError: A verification code when the restored state
                cannot be reconciled with the manifest.
        """

        if type(backup_digest) is not str or not _DIGEST_RE.fullmatch(backup_digest):
            raise _fail("invalid_argument")
        resolved_restore = (
            _new_opaque_id("restore")
            if restore_id is None
            else self._safe_id(restore_id)
        )
        with self._locked():
            manifest = self._read_manifest()
            chain = self._read_chain(manifest)
            chain = self._recover_locked(chain, manifest)[0]
            current = manifest.current_epoch
            restore_key = current.key_id if key_id is None else self._safe_id(key_id)
            if restore_key != current.key_id and restore_key in manifest.used_key_ids:
                raise _fail("retired_signer_key")
            event = self._build_event(
                manifest,
                chain,
                chain_id=manifest.chain_id,
                kind="restore",
                epoch=current.epoch + 1,
                body={
                    "key_id": restore_key,
                    "restore_id": resolved_restore,
                    "backup_digest": backup_digest,
                },
                signers=(("incoming", restore_key),),
            )
            candidate = manifest.with_event(event)
            self._require_consistent(chain, candidate, allow_unanchored_tail=False)
            self._write_manifest(candidate)
            return event

    def recover(self) -> ContinuityRecoveryResult:
        """Complete any interrupted transition and remove stale temp files.

        Recovery is deterministic: a pending tombstone is completed only when
        the still-present entries match it exactly; otherwise the store
        refuses with ``tombstone_mismatch``.
        """

        with self._locked():
            manifest = self._read_manifest()
            chain = self._read_chain(manifest)
            _, pruned = self._recover_locked(chain, manifest)
            removed = self._remove_stale_temporaries()
            return ContinuityRecoveryResult(
                retention_completed=pruned > 0,
                pruned_entry_count=pruned,
                temporary_files_removed=removed,
            )

    def load(self) -> tuple[AuditChain, ContinuityManifest]:
        """Return the current chain and manifest under the store lock."""

        with self._locked():
            manifest = self._read_manifest()
            return self._read_chain(manifest), manifest

    def verify(
        self,
        *,
        key_provider: KeyProvider | None = None,
        trusted_anchors: Iterable[ContinuityAnchorRef | Mapping[str, Any]] = (),
        allowed_policy_fingerprints: Iterable[str] | None = None,
        allow_unanchored_tail: bool = False,
    ) -> ContinuityVerificationReport:
        """Verify the stored chain and manifest with :func:`verify_continuity`.

        Args:
            key_provider: Verification provider; defaults to the store's.
            trusted_anchors: Externally retained event references.
            allowed_policy_fingerprints: Retention policies allowed to prune.
            allow_unanchored_tail: Accept entries after the last signed event.
        """

        with self._locked():
            try:
                manifest = self._read_manifest()
                chain = self._read_chain(manifest)
            except AuditContinuityError as error:
                code = error.code if error.code in CONTINUITY_REASON_CODES else None
                return _Verification(
                    key_provider=self._key_provider,
                    trusted_anchors=(),
                    allowed_policy_fingerprints=None,
                    expected_chain_id=None,
                    allow_unanchored_tail=False,
                    tolerate_unavailable_keys=False,
                ).report(code or "malformed_manifest")
        return verify_continuity(
            chain,
            manifest,
            key_provider=self._key_provider if key_provider is None else key_provider,
            trusted_anchors=trusted_anchors,
            allowed_policy_fingerprints=allowed_policy_fingerprints,
            expected_chain_id=manifest.chain_id,
            allow_unanchored_tail=allow_unanchored_tail,
        )

    # Internals -------------------------------------------------------------

    @staticmethod
    def _safe_id(value: Any) -> str:
        if type(value) is not str or not _SAFE_ID_RE.fullmatch(value):
            raise _fail("invalid_argument")
        return value

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with _exclusive_lock(self._lock_path):
            yield

    def _read_manifest(self) -> ContinuityManifest:
        if not self._manifest_path.exists():
            raise _fail("not_initialized")
        return ContinuityManifest.load(self._manifest_path)

    def _read_chain(self, manifest: ContinuityManifest | None) -> AuditChain:
        if not self._chain_path.exists():
            if manifest is not None and (
                manifest.head_event.next_sequence > 0 or manifest.tombstones
            ):
                raise _fail("sequence_gap")
            return AuditChain()
        try:
            return AuditChain.load(self._chain_path)
        except OSError:
            raise _fail("io_failed") from None
        except (TypeError, ValueError):
            raise _fail("chain_invalid") from None

    def _read_consistent(self) -> tuple[AuditChain, ContinuityManifest]:
        manifest = self._read_manifest()
        chain = self._read_chain(manifest)
        chain = self._recover_locked(chain, manifest)[0]
        self._require_consistent(chain, manifest, allow_unanchored_tail=True)
        return chain, manifest

    def _require_consistent(
        self,
        chain: AuditChain,
        manifest: ContinuityManifest,
        *,
        allow_unanchored_tail: bool,
    ) -> None:
        # Retired keys may legitimately be gone from the signing provider, so
        # unavailable keys do not block new transitions; every other check,
        # including signatures whose keys are present, must pass.
        report = _run_verification(
            chain,
            manifest,
            key_provider=self._key_provider,
            expected_chain_id=manifest.chain_id,
            allow_unanchored_tail=allow_unanchored_tail,
            tolerate_unavailable_keys=True,
        )
        if not report.valid:
            raise _fail(report.code)

    @staticmethod
    def _require_head_commitment(
        chain: AuditChain,
        manifest: ContinuityManifest,
    ) -> None:
        # A constant-time guard for the hot append path: the newest signed
        # event must still describe a prefix of the chain being extended.
        head = manifest.head_event
        if head.next_sequence > chain.next_sequence:
            raise _fail("sequence_gap")
        if head.next_sequence == 0:
            committed = GENESIS_HASH
        elif head.next_sequence == chain.base_sequence:
            committed = chain.base_hash
        else:
            entry = chain.entry_at(head.next_sequence - 1)
            committed = entry.entry_hash if entry is not None else ""
        if committed != head.head_hash and head.next_sequence >= chain.base_sequence:
            raise _fail("anchor_mismatch")

    def _recover_locked(
        self,
        chain: AuditChain,
        manifest: ContinuityManifest,
    ) -> tuple[AuditChain, int]:
        covered = manifest.pruned_through
        if chain.base_sequence == covered:
            return chain, 0
        if chain.base_sequence > covered:
            raise _fail("sequence_gap")
        if chain.next_sequence < covered:
            raise _fail("sequence_gap")
        for tombstone in manifest.tombstones:
            if tombstone.last_sequence < chain.base_sequence:
                continue
            if tombstone.first_sequence < chain.base_sequence:
                raise _fail("tombstone_mismatch")
            segment = [
                entry
                for entry in chain.entries
                if tombstone.first_sequence <= entry.sequence <= tombstone.last_sequence
            ]
            if not verify_tombstone_segment(
                tombstone, segment, chain_id=manifest.chain_id
            ):
                raise _fail("tombstone_mismatch")
        pruned = covered - chain.base_sequence
        try:
            chain.prune_through(covered - 1)
        except ValueError:
            raise _fail("chain_invalid") from None
        self._write_chain(chain)
        return chain, pruned

    def _remove_stale_temporaries(self) -> int:
        removed = 0
        for target in (self._chain_path, self._manifest_path):
            parent = target.parent
            if not parent.is_dir():
                continue
            for candidate in parent.glob(f".{target.name}.*.tmp"):
                try:
                    info = candidate.lstat()
                    if stat.S_ISREG(info.st_mode):
                        candidate.unlink()
                        removed += 1
                except OSError:
                    continue
        return removed

    def _write_chain(self, chain: AuditChain) -> None:
        try:
            chain.write(self._chain_path)
        except OSError:
            raise _fail("io_failed") from None

    def _write_manifest(self, manifest: ContinuityManifest) -> None:
        try:
            manifest.write(self._manifest_path)
        except OSError:
            raise _fail("io_failed") from None

    def _signed_event(
        self,
        manifest: ContinuityManifest,
        chain: AuditChain,
        kind: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> ContinuityEvent:
        current = manifest.current_epoch
        return self._build_event(
            manifest,
            chain,
            chain_id=manifest.chain_id,
            kind=kind,
            epoch=current.epoch,
            body=body or {},
            signers=(("signer", current.key_id),),
        )

    def _build_event(
        self,
        manifest: ContinuityManifest | None,
        chain: AuditChain,
        *,
        chain_id: str,
        kind: str,
        epoch: int,
        body: Mapping[str, Any],
        signers: Sequence[tuple[str, str]],
    ) -> ContinuityEvent:
        index = 0 if manifest is None else manifest.event_count
        if index >= MAX_CONTINUITY_EVENTS:
            raise _fail("invalid_argument")
        previous = (
            continuity_origin_digest(chain_id)
            if manifest is None
            else manifest.head_digest
        )
        normalized = _validate_body(kind, body)
        digest = _event_digest(
            chain_id=chain_id,
            index=index,
            kind=kind,
            epoch=epoch,
            previous_digest=previous,
            next_sequence=chain.next_sequence,
            head_hash=chain.head_hash,
            body=normalized,
        )
        signatures: list[ContinuitySignature] = []
        for role, signer_key in signers:
            try:
                value = _sign_digest(
                    self._key_provider,
                    digest,
                    key_id=signer_key,
                    purpose=CONTINUITY_SIGNATURE_PURPOSE,
                    role=role,
                )
            except AuditKeyRotationError:
                raise _fail("signer_key_unavailable") from None
            signatures.append(
                ContinuitySignature(
                    role=role,
                    key_id=signer_key,
                    algorithm=CONTINUITY_SIGNATURE_ALGORITHM,
                    value=value,
                )
            )
        return ContinuityEvent(
            index=index,
            kind=kind,
            epoch=epoch,
            previous_digest=previous,
            next_sequence=chain.next_sequence,
            head_hash=chain.head_hash,
            body=normalized,
            digest=digest,
            signatures=tuple(signatures),
        )


def record_restore_transition(
    chain_path: str | Path,
    *,
    key_provider: KeyProvider,
    backup_digest: str,
    manifest_path: str | Path | None = None,
    restore_id: str | None = None,
    key_id: str | None = None,
) -> ContinuityEvent:
    """Record a restore transition for a chain restored from a backup.

    This is the hook a backup and restore workflow calls once the audit chain
    (and its manifest, when backed up) has been restored and before writers
    resume. See :meth:`AuditContinuityStore.record_restore_transition`.

    Args:
        chain_path: Restored audit chain document.
        key_provider: Caller-owned signing-key provider.
        backup_digest: ``sha256:<hex>`` digest identifying the backup.
        manifest_path: Continuity manifest path, if not the default.
        restore_id: Optional opaque restore identifier.
        key_id: Optional key ID for the restore epoch.

    Returns:
        The signed restore event.
    """

    return AuditContinuityStore(
        chain_path,
        key_provider=key_provider,
        manifest_path=manifest_path,
    ).record_restore_transition(
        backup_digest=backup_digest,
        restore_id=restore_id,
        key_id=key_id,
    )


class AuditContinuityRecoveryHook:
    """Recovery-epoch hook that records a restore transition.

    The hook duck-types a service recovery context exposing
    ``manifest_digest`` and ``recovery_point_id``; it does not import the
    service layer. Point it at the staged (not yet live) chain and manifest so
    the restore event is part of the state that becomes live. Raising refuses
    the activation.

    Args:
        chain_path: Restored audit chain document.
        key_provider: Caller-owned signing-key provider.
        manifest_path: Continuity manifest path, if not the default.
        key_id: Optional key ID for the restore epoch.
    """

    def __init__(
        self,
        chain_path: str | Path,
        *,
        key_provider: KeyProvider,
        manifest_path: str | Path | None = None,
        key_id: str | None = None,
    ) -> None:
        self._store = AuditContinuityStore(
            chain_path, key_provider=key_provider, manifest_path=manifest_path
        )
        self._key_id = key_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def open_recovery_epoch(self, context: Any) -> None:
        """Record the restore described by ``context``.

        Args:
            context: Object with a ``manifest_digest`` and an optional
                ``recovery_point_id`` attribute.

        Raises:
            AuditContinuityError: If the restored state cannot be reconciled.
        """

        restore_id = getattr(context, "recovery_point_id", None)
        if restore_id is not None and (
            type(restore_id) is not str or not _SAFE_ID_RE.fullmatch(restore_id)
        ):
            restore_id = None
        self._store.record_restore_transition(
            backup_digest=getattr(context, "manifest_digest", None),  # type: ignore[arg-type]
            restore_id=restore_id,
            key_id=self._key_id,
        )


__all__ = [
    "CONTINUITY_DIGEST_ALGORITHM",
    "CONTINUITY_ERROR_CODES",
    "CONTINUITY_EVENT_KINDS",
    "CONTINUITY_FORMAT",
    "CONTINUITY_REASON_CODES",
    "CONTINUITY_REPORT_FORMAT",
    "CONTINUITY_REPORT_VERSION",
    "CONTINUITY_SIGNATURE_ALGORITHM",
    "CONTINUITY_SIGNATURE_PURPOSE",
    "CONTINUITY_VERSION",
    "MAX_CONTINUITY_EVENTS",
    "MAX_CONTINUITY_JSON_BYTES",
    "AuditContinuityError",
    "AuditContinuityRecoveryHook",
    "AuditContinuityStore",
    "ContinuityAnchorRef",
    "ContinuityEvent",
    "ContinuityManifest",
    "ContinuityRecoveryResult",
    "ContinuitySignature",
    "ContinuityVerificationReport",
    "RetentionTombstone",
    "SignerEpoch",
    "continuity_origin_digest",
    "derive_epochs",
    "record_restore_transition",
    "tombstone_segment_digest",
    "verify_continuity",
    "verify_tombstone_segment",
]
