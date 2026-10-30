"""Privacy-safe, hash-chained ledger of federated participation receipts.

Audit and attribution need a participation history, but a shared ledger must
not reveal institutions, patient counts, or local performance slices.  Each
receipt records exactly five fields:

* ``participant_handle``: a keyed HMAC-SHA256 pseudonym derived under an
  injected per-federation key, never an institution name;
* ``round_ref``: a SHA-256 digest that references the federated round;
* ``round_role``: one controlled role from :class:`ContributionRole`;
* ``policy_version``: the positive integer governing policy version; and
* ``update_commitment``: the declared update SHA-256 digest, required for
  ``participant`` receipts and absent for every other role.

The public ledger stores one salted commitment per field, so no plaintext
receipt value is published and repeat participants are not linkable across
entries.  Entries are chained with :class:`~openmed.compliance.audit_chain.
AuditRecord` hashes from a ledger-specific genesis digest.  A receipt holder can
disclose chosen fields with their openings, and any verifier can check them
against the public entry without learning the remaining fields.

Everything is deterministic and local: there is no logging, clock read, file
access, or network call.  Errors carry fixed codes only and are raised
``from None`` so submitted values never surface in messages or tracebacks.
This module is evidence tooling, not a compliance certification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from openmed.compliance.audit_chain import AuditRecord, HashChainAuditLog
from openmed.core.audit import stable_hash
from openmed.training.federated_update_metadata import FederatedUpdateMetadata

CONTRIBUTION_LEDGER_SCHEMA: Final = (
    "openmed.compliance.governance.contribution_ledger.v1"
)
"""Schema identifier carried by public ledgers, receipts, and proofs."""

RECEIPT_FIELDS: Final[tuple[str, ...]] = (
    "participant_handle",
    "round_ref",
    "round_role",
    "policy_version",
    "update_commitment",
)
"""The closed, canonical set of committed receipt fields."""

MIN_FEDERATION_KEY_BYTES: Final = 32
MAX_FEDERATION_KEY_BYTES: Final = 4096
MAX_LEDGER_ENTRIES: Final = 100_000
MAX_POLICY_VERSION: Final = 1_000_000

_EVENT_TYPE: Final = "contribution_receipt"
_HANDLE_PREFIX: Final = "pseud:"
_PSEUDONYM_CONTEXT: Final = (
    b"openmed.compliance.governance.contribution_ledger.participant.v1"
)
_SALT_CONTEXT: Final = b"openmed.compliance.governance.contribution_ledger.salt.v1"
_FIELD_DOMAIN: Final = "openmed.compliance.governance.contribution_ledger.field.v1"
_MAX_LABEL_CHARS: Final = 512
_MAX_JSON_CHARS: Final = 64 * 1024 * 1024

_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HANDLE_RE: Final = re.compile(r"pseud:[0-9a-f]{64}\Z")
_SALT_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")

_LEDGER_KEYS: Final = frozenset({"schema", "ledger_id", "genesis_digest", "entries"})
_ENTRY_KEYS: Final = frozenset(
    {"sequence", "previous_digest", "commitments", "entry_digest"}
)
_RECEIPT_KEYS: Final = frozenset(
    {"schema", "ledger_id", "sequence", "entry_digest", "values", "salts"}
)
_PROOF_KEYS: Final = frozenset(
    {"schema", "ledger_id", "sequence", "entry_digest", "openings"}
)
_OPENING_KEYS: Final = frozenset({"salt", "value"})


class ContributionRole(str, Enum):
    """Controlled round roles a receipt may record.

    The vocabulary matches the anonymous federation-charter roles so the two
    governance surfaces describe rounds with the same terms.
    """

    PARTICIPANT = "participant"
    COORDINATOR = "coordinator"
    REVIEWER = "reviewer"
    INCIDENT_RESPONDER = "incident_responder"

    def __str__(self) -> str:
        return self.value


class LedgerReasonCode(str, Enum):
    """Stable outcome codes for public ledger continuity verification."""

    OK = "ok"
    MALFORMED = "ledger_malformed"
    GENESIS_MISMATCH = "genesis_mismatch"
    FORK_DETECTED = "fork_detected"
    SEQUENCE_GAP = "sequence_gap"
    SEQUENCE_REORDER = "sequence_reorder"
    ENTRY_TAMPERED = "entry_tampered"
    LINK_MISMATCH = "link_mismatch"
    TRUNCATED = "truncated"

    def __str__(self) -> str:
        return self.value


class DisclosureReasonCode(str, Enum):
    """Stable outcome codes for selective-disclosure verification."""

    OK = "ok"
    PROOF_MALFORMED = "proof_malformed"
    LEDGER_INVALID = "ledger_invalid"
    LEDGER_MISMATCH = "ledger_mismatch"
    ENTRY_NOT_FOUND = "entry_not_found"
    ENTRY_MISMATCH = "entry_mismatch"
    OPENING_INVALID = "opening_invalid"

    def __str__(self) -> str:
        return self.value


class ContributionLedgerError(ValueError):
    """Raised with a fixed code for invalid ledger inputs or documents.

    The message equals :attr:`code`; submitted values are never included.

    Args:
        code: Stable ``contribution_ledger_*`` error code.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> ContributionLedgerError:
    return ContributionLedgerError(f"contribution_ledger_{code}")


# ---------------------------------------------------------------------------
# Field validation and cryptographic helpers
# ---------------------------------------------------------------------------


def _require_key(key: object) -> bytes:
    if type(key) is not bytes:
        raise _fail("invalid_key")
    if not MIN_FEDERATION_KEY_BYTES <= len(key) <= MAX_FEDERATION_KEY_BYTES:
        raise _fail("invalid_key")
    return key


def _require_ledger_id(value: object) -> str:
    if type(value) is not str or _LEDGER_ID_RE.fullmatch(value) is None:
        raise _fail("invalid_ledger_id")
    return value


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _require_digest(value: object, code: str) -> str:
    if not _is_digest(value):
        raise _fail(code)
    return value  # type: ignore[return-value]


def _require_sequence(value: object, code: str) -> int:
    if type(value) is not int or not 0 <= value < MAX_LEDGER_ENTRIES:
        raise _fail(code)
    return value


def _derive(key: bytes, context: bytes) -> bytes:
    return hmac.new(key, context, hashlib.sha256).digest()


def _handle_from_pseudonym_key(pseudonym_key: bytes, label: object) -> str:
    if type(label) is not str or not 1 <= len(label) <= _MAX_LABEL_CHARS:
        raise _fail("invalid_label")
    try:
        encoded = label.encode("utf-8")
    except UnicodeError:
        raise _fail("invalid_label") from None
    digest = hmac.new(pseudonym_key, encoded, hashlib.sha256).hexdigest()
    return _HANDLE_PREFIX + digest


def participant_handle(key: bytes, label: str) -> str:
    """Return the keyed pseudonymous handle for a local participant label.

    The handle is ``pseud:`` followed by an HMAC-SHA256 hex digest under a
    context-separated derivation of ``key``.  It is stable for one key and
    unlinkable across keys.  The label is used transiently and is never
    stored, logged, or echoed.

    Args:
        key: Per-federation secret of 32 to 4096 bytes, supplied by the caller.
        label: Local participant label known only to the key holder.

    Returns:
        The pseudonymous participant handle.

    Raises:
        ContributionLedgerError: If the key or label is unsupported.
    """

    pseudonym_key = _derive(_require_key(key), _PSEUDONYM_CONTEXT)
    return _handle_from_pseudonym_key(pseudonym_key, label)


def _validate_value(name: str, value: object, *, role: str | None = None) -> Any:
    """Validate one receipt field value and return its canonical form."""

    if name == "participant_handle":
        if type(value) is not str or _HANDLE_RE.fullmatch(value) is None:
            raise _fail("invalid_handle")
        return value
    if name == "round_ref":
        return _require_digest(value, "invalid_round_ref")
    if name == "round_role":
        if isinstance(value, ContributionRole):
            return value.value
        if type(value) is not str:
            raise _fail("invalid_role")
        try:
            return ContributionRole(value).value
        except ValueError:
            raise _fail("invalid_role") from None
    if name == "policy_version":
        if type(value) is not int or not 1 <= value <= MAX_POLICY_VERSION:
            raise _fail("invalid_policy_version")
        return value
    if name == "update_commitment":
        if role == ContributionRole.PARTICIPANT.value:
            return _require_digest(value, "invalid_update_commitment")
        if role is not None and value is not None:
            raise _fail("invalid_update_commitment")
        if value is not None and not _is_digest(value):
            raise _fail("invalid_update_commitment")
        return value
    raise _fail("unknown_field")


def _commit(name: str, salt: str, value: Any) -> str:
    return stable_hash(
        {"domain": _FIELD_DOMAIN, "field": name, "salt": salt, "value": value}
    )


def _genesis_digest(ledger_id: str) -> str:
    return stable_hash(
        {"chain": CONTRIBUTION_LEDGER_SCHEMA, "ledger_id": ledger_id, "sequence": -1}
    )


def _chain_payload(ledger_id: str, commitments: Mapping[str, str]) -> dict[str, Any]:
    return {
        "commitments": dict(commitments),
        "ledger_id": ledger_id,
        "schema": CONTRIBUTION_LEDGER_SCHEMA,
    }


def _entry_digest(
    ledger_id: str,
    sequence: int,
    previous_digest: str,
    commitments: Mapping[str, str],
) -> str:
    return AuditRecord(
        sequence=sequence,
        event_type=_EVENT_TYPE,
        payload=_chain_payload(ledger_id, commitments),
        previous_hash=previous_digest,
    ).compute_hash()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_float(_value: str) -> None:
    raise ValueError("non-integer number")


def _load_json(payload: str, code: str) -> Any:
    if len(payload) > _MAX_JSON_CHARS:
        raise _fail(code)
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (ValueError, RecursionError):
        raise _fail(code) from None


def _exact_mapping(payload: object, keys: frozenset[str], code: str) -> Mapping:
    if not isinstance(payload, Mapping):
        raise _fail(code)
    try:
        present = set(payload)
    except Exception:
        raise _fail(code) from None
    if present != keys:
        raise _fail(code)
    return payload


# ---------------------------------------------------------------------------
# Public ledger documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerCheckpoint:
    """A published anchor used to detect truncation and rewritten history.

    Args:
        length: Number of entries the anchor covers.
        head_digest: Digest of entry ``length - 1``, or the genesis digest
            when ``length`` is zero.
    """

    length: int
    head_digest: str

    def __post_init__(self) -> None:
        if type(self.length) is not int or not 0 <= self.length <= MAX_LEDGER_ENTRIES:
            raise _fail("invalid_checkpoint")
        if not _is_digest(self.head_digest):
            raise _fail("invalid_checkpoint")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible checkpoint fields."""

        return {"head_digest": self.head_digest, "length": self.length}


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One public entry: per-field commitments linked to its predecessor.

    Args:
        sequence: Zero-based position claimed by the entry.
        previous_digest: Claimed digest of the preceding entry or genesis.
        commitments: Salted SHA-256 commitment for every receipt field.
        entry_digest: Claimed chain digest of this entry.
    """

    sequence: int
    previous_digest: str
    commitments: Mapping[str, str]
    entry_digest: str

    def __post_init__(self) -> None:
        _require_sequence(self.sequence, "invalid_document")
        _require_digest(self.previous_digest, "invalid_document")
        _require_digest(self.entry_digest, "invalid_document")
        commitments = _exact_mapping(
            self.commitments, frozenset(RECEIPT_FIELDS), "invalid_document"
        )
        ordered = {name: commitments[name] for name in RECEIPT_FIELDS}
        for value in ordered.values():
            _require_digest(value, "invalid_document")
        object.__setattr__(self, "commitments", MappingProxyType(ordered))

    @classmethod
    def from_dict(cls, payload: object) -> LedgerEntry:
        """Parse one strict public entry mapping.

        Args:
            payload: Mapping with exactly the public entry fields.

        Returns:
            The parsed entry; continuity is checked by :func:`verify_ledger`.

        Raises:
            ContributionLedgerError: If the shape or a digest is invalid.
        """

        entry = _exact_mapping(payload, _ENTRY_KEYS, "invalid_document")
        return cls(
            sequence=entry["sequence"],
            previous_digest=entry["previous_digest"],
            commitments=entry["commitments"],
            entry_digest=entry["entry_digest"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible public entry."""

        return {
            "commitments": dict(self.commitments),
            "entry_digest": self.entry_digest,
            "previous_digest": self.previous_digest,
            "sequence": self.sequence,
        }


@dataclass(frozen=True, slots=True)
class PublicContributionLedger:
    """A parsed public ledger document that holds commitments only.

    Parsing checks shape and digest formats; continuity, genesis, and links
    are checked by :func:`verify_ledger` so failures get stable reason codes.

    Args:
        ledger_id: Opaque lowercase federation ledger identifier.
        genesis_digest: Claimed genesis digest for ``ledger_id``.
        entries: Public entries in published order.
        schema: Exact supported schema identifier.
    """

    ledger_id: str
    genesis_digest: str
    entries: tuple[LedgerEntry, ...]
    schema: str = CONTRIBUTION_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONTRIBUTION_LEDGER_SCHEMA:
            raise _fail("invalid_document")
        try:
            _require_ledger_id(self.ledger_id)
        except ContributionLedgerError:
            raise _fail("invalid_document") from None
        _require_digest(self.genesis_digest, "invalid_document")
        if type(self.entries) is not tuple or len(self.entries) > MAX_LEDGER_ENTRIES:
            raise _fail("invalid_document")
        if any(type(entry) is not LedgerEntry for entry in self.entries):
            raise _fail("invalid_document")

    def __repr__(self) -> str:
        return (
            f"PublicContributionLedger(ledger_id={self.ledger_id!r}, "
            f"entries={len(self.entries)})"
        )

    @classmethod
    def from_dict(cls, payload: object) -> PublicContributionLedger:
        """Parse a strict public ledger mapping.

        Args:
            payload: Mapping with exactly ``schema``, ``ledger_id``,
                ``genesis_digest``, and ``entries``.

        Returns:
            The parsed public ledger.

        Raises:
            ContributionLedgerError: ``contribution_ledger_invalid_document``
                for any unsupported shape, field, or value.
        """

        document = _exact_mapping(payload, _LEDGER_KEYS, "invalid_document")
        entries = document["entries"]
        if type(entries) is not list or len(entries) > MAX_LEDGER_ENTRIES:
            raise _fail("invalid_document")
        if type(document["schema"]) is not str:
            raise _fail("invalid_document")
        return cls(
            ledger_id=document["ledger_id"],
            genesis_digest=document["genesis_digest"],
            entries=tuple(LedgerEntry.from_dict(entry) for entry in entries),
            schema=document["schema"],
        )

    @classmethod
    def from_json(cls, payload: str) -> PublicContributionLedger:
        """Parse public ledger JSON, rejecting duplicate keys and floats.

        Args:
            payload: JSON text produced by :meth:`to_json`.

        Returns:
            The parsed public ledger.

        Raises:
            ContributionLedgerError: If the JSON or its shape is invalid.
        """

        if type(payload) is not str:
            raise _fail("invalid_document")
        return cls.from_dict(_load_json(payload, "invalid_document"))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible public ledger document."""

        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "genesis_digest": self.genesis_digest,
            "ledger_id": self.ledger_id,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON for publication."""

        return _canonical_json(self.to_dict())


def _coerce_public(ledger: object) -> PublicContributionLedger:
    if isinstance(ledger, PublicContributionLedger):
        return ledger
    if isinstance(ledger, ContributionLedger):
        return ledger.public_ledger()
    if isinstance(ledger, str):
        return PublicContributionLedger.from_json(ledger)
    return PublicContributionLedger.from_dict(ledger)


# ---------------------------------------------------------------------------
# Continuity verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerVerification:
    """Result of an offline continuity check over a public ledger.

    Args:
        valid: Whether every continuity check passed.
        reason: Stable :class:`LedgerReasonCode` for the first failure.
        entry_count: Number of parsed entries, or zero when malformed.
        head_digest: Verified head digest, present only when valid.
        failed_sequence: Entry position of a per-entry failure, if any.
    """

    valid: bool
    reason: LedgerReasonCode
    entry_count: int
    head_digest: str | None = None
    failed_sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible report with fixed codes only."""

        return {
            "entry_count": self.entry_count,
            "failed_sequence": self.failed_sequence,
            "head_digest": self.head_digest,
            "reason": self.reason.value,
            "valid": self.valid,
        }


def _failure(
    reason: LedgerReasonCode, count: int, position: int | None = None
) -> LedgerVerification:
    return LedgerVerification(False, reason, count, None, position)


def _verify_public(
    public: PublicContributionLedger, checkpoint: LedgerCheckpoint | None
) -> LedgerVerification:
    entries = public.entries
    count = len(entries)
    genesis = _genesis_digest(public.ledger_id)
    if public.genesis_digest != genesis:
        return _failure(LedgerReasonCode.GENESIS_MISMATCH, count)

    sequences = [entry.sequence for entry in entries]
    parents = [entry.previous_digest for entry in entries]
    if len(set(sequences)) != count or len(set(parents)) != count:
        return _failure(LedgerReasonCode.FORK_DETECTED, count)
    if set(sequences) != set(range(count)):
        return _failure(LedgerReasonCode.SEQUENCE_GAP, count)
    if sequences != list(range(count)):
        return _failure(LedgerReasonCode.SEQUENCE_REORDER, count)

    previous = genesis
    for position, entry in enumerate(entries):
        expected = _entry_digest(
            public.ledger_id, entry.sequence, entry.previous_digest, entry.commitments
        )
        if not hmac.compare_digest(entry.entry_digest, expected):
            return _failure(LedgerReasonCode.ENTRY_TAMPERED, count, position)
        if not hmac.compare_digest(entry.previous_digest, previous):
            return _failure(LedgerReasonCode.LINK_MISMATCH, count, position)
        previous = entry.entry_digest

    if checkpoint is not None:
        if count < checkpoint.length:
            return _failure(LedgerReasonCode.TRUNCATED, count)
        anchored = (
            entries[checkpoint.length - 1].entry_digest
            if checkpoint.length
            else genesis
        )
        if not hmac.compare_digest(anchored, checkpoint.head_digest):
            return _failure(LedgerReasonCode.FORK_DETECTED, count)
    return LedgerVerification(True, LedgerReasonCode.OK, count, previous, None)


def verify_ledger(
    ledger: PublicContributionLedger | ContributionLedger | Mapping[str, Any] | str,
    *,
    checkpoint: LedgerCheckpoint | None = None,
) -> LedgerVerification:
    """Verify public ledger continuity offline from the public ledger alone.

    Checks the schema, the ledger-specific genesis digest, unique and
    contiguous sequences in order, each recomputed entry digest, and each
    predecessor link.  Duplicate sequences or shared predecessors report
    ``fork_detected``.  A tail truncation is only detectable against a
    previously published ``checkpoint``; a rewritten anchored prefix reports
    ``fork_detected``.  No key is needed and no field value is revealed.

    Args:
        ledger: Public ledger object, mapping, or JSON text.
        checkpoint: Optional published anchor from an earlier observation.

    Returns:
        A :class:`LedgerVerification` with a stable reason code.

    Raises:
        ContributionLedgerError: If ``checkpoint`` is not a checkpoint.
    """

    if checkpoint is not None and type(checkpoint) is not LedgerCheckpoint:
        raise _fail("invalid_checkpoint")
    try:
        public = _coerce_public(ledger)
    except ContributionLedgerError:
        return _failure(LedgerReasonCode.MALFORMED, 0)
    return _verify_public(public, checkpoint)


# ---------------------------------------------------------------------------
# Receipts and selective disclosure
# ---------------------------------------------------------------------------


def _validate_field_mapping(payload: object, code: str) -> dict[str, Any]:
    values = _exact_mapping(payload, frozenset(RECEIPT_FIELDS), code)
    role = values["round_role"]
    try:
        role = _validate_value("round_role", role)
        return {
            name: _validate_value(name, values[name], role=role)
            for name in RECEIPT_FIELDS
        }
    except ContributionLedgerError:
        raise _fail(code) from None


def _validate_salt(value: object, code: str) -> str:
    if type(value) is not str or _SALT_RE.fullmatch(value) is None:
        raise _fail(code)
    return value


@dataclass(frozen=True, slots=True)
class DisclosureProof:
    """Openings for holder-chosen receipt fields, bound to one public entry.

    Only the chosen fields' values and salts are present; undisclosed fields
    contribute nothing to the proof.

    Args:
        ledger_id: Ledger the receipt was appended to.
        sequence: Entry sequence the proof refers to.
        entry_digest: Public digest of that entry.
        openings: Mapping of field name to ``(value, salt)``.
    """

    ledger_id: str
    sequence: int
    entry_digest: str
    openings: Mapping[str, tuple[Any, str]] = field(repr=False)

    def __post_init__(self) -> None:
        try:
            _require_ledger_id(self.ledger_id)
        except ContributionLedgerError:
            raise _fail("invalid_proof") from None
        _require_sequence(self.sequence, "invalid_proof")
        _require_digest(self.entry_digest, "invalid_proof")
        if not isinstance(self.openings, Mapping) or not self.openings:
            raise _fail("invalid_proof")
        ordered: dict[str, tuple[Any, str]] = {}
        try:
            names = set(self.openings)
        except Exception:
            raise _fail("invalid_proof") from None
        if not names <= set(RECEIPT_FIELDS):
            raise _fail("invalid_proof")
        for name in RECEIPT_FIELDS:
            if name not in names:
                continue
            opening = self.openings[name]
            if type(opening) is not tuple or len(opening) != 2:
                raise _fail("invalid_proof")
            value, salt = opening
            try:
                value = _validate_value(name, value)
            except ContributionLedgerError:
                raise _fail("invalid_proof") from None
            ordered[name] = (value, _validate_salt(salt, "invalid_proof"))
        object.__setattr__(self, "openings", MappingProxyType(ordered))

    def __repr__(self) -> str:
        return (
            f"DisclosureProof(ledger_id={self.ledger_id!r}, "
            f"sequence={self.sequence}, fields={tuple(self.openings)!r})"
        )

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the disclosed field names in canonical order."""

        return tuple(self.openings)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible proof with disclosed openings only."""

        return {
            "entry_digest": self.entry_digest,
            "ledger_id": self.ledger_id,
            "openings": {
                name: {"salt": salt, "value": value}
                for name, (value, salt) in self.openings.items()
            },
            "schema": CONTRIBUTION_LEDGER_SCHEMA,
            "sequence": self.sequence,
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON for transfer to a verifier."""

        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> DisclosureProof:
        """Parse a strict proof mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The parsed proof.

        Raises:
            ContributionLedgerError: ``contribution_ledger_invalid_proof`` for
                unknown fields, metric-like extras, or malformed openings.
        """

        proof = _exact_mapping(payload, _PROOF_KEYS, "invalid_proof")
        if proof["schema"] != CONTRIBUTION_LEDGER_SCHEMA:
            raise _fail("invalid_proof")
        openings = proof["openings"]
        if not isinstance(openings, Mapping):
            raise _fail("invalid_proof")
        parsed: dict[str, tuple[Any, str]] = {}
        for name, opening in openings.items():
            opening = _exact_mapping(opening, _OPENING_KEYS, "invalid_proof")
            parsed[name] = (opening["value"], opening["salt"])
        return cls(
            ledger_id=proof["ledger_id"],
            sequence=proof["sequence"],
            entry_digest=proof["entry_digest"],
            openings=parsed,
        )

    @classmethod
    def from_json(cls, payload: str) -> DisclosureProof:
        """Parse proof JSON, rejecting duplicate keys and floats.

        Args:
            payload: JSON text produced by :meth:`to_json`.

        Returns:
            The parsed proof.

        Raises:
            ContributionLedgerError: If the JSON or its shape is invalid.
        """

        if type(payload) is not str:
            raise _fail("invalid_proof")
        return cls.from_dict(_load_json(payload, "invalid_proof"))


@dataclass(frozen=True, slots=True)
class ContributionReceipt:
    """Holder-side receipt with every field value and its commitment salt.

    The receipt is private to its holder.  Its ``repr`` shows only the ledger
    position; values and salts are reachable through :meth:`disclose` and
    :meth:`to_dict`.

    Args:
        ledger_id: Ledger the receipt was appended to.
        sequence: Entry sequence of the receipt.
        entry_digest: Public digest of the entry.
        values: Validated value for every receipt field.
        salts: Commitment salt for every receipt field.
    """

    ledger_id: str
    sequence: int
    entry_digest: str
    values: Mapping[str, Any] = field(repr=False)
    salts: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        try:
            _require_ledger_id(self.ledger_id)
        except ContributionLedgerError:
            raise _fail("invalid_receipt") from None
        _require_sequence(self.sequence, "invalid_receipt")
        _require_digest(self.entry_digest, "invalid_receipt")
        values = _validate_field_mapping(self.values, "invalid_receipt")
        salts = _exact_mapping(self.salts, frozenset(RECEIPT_FIELDS), "invalid_receipt")
        ordered_salts = {
            name: _validate_salt(salts[name], "invalid_receipt")
            for name in RECEIPT_FIELDS
        }
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "salts", MappingProxyType(ordered_salts))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContributionReceipt):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self.ledger_id, self.sequence, self.entry_digest))

    def disclose(self, fields: Iterable[str]) -> DisclosureProof:
        """Return a proof opening only the chosen fields.

        Args:
            fields: Non-empty iterable of names from :data:`RECEIPT_FIELDS`.

        Returns:
            A :class:`DisclosureProof` containing no undisclosed value or salt.

        Raises:
            ContributionLedgerError: For an empty request or unknown field.
        """

        if isinstance(fields, (str, bytes)) or not isinstance(fields, Iterable):
            raise _fail("unknown_field")
        chosen = list(fields)
        if not chosen:
            raise _fail("empty_disclosure")
        if any(type(name) is not str or name not in RECEIPT_FIELDS for name in chosen):
            raise _fail("unknown_field")
        return DisclosureProof(
            ledger_id=self.ledger_id,
            sequence=self.sequence,
            entry_digest=self.entry_digest,
            openings={name: (self.values[name], self.salts[name]) for name in chosen},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible holder record, including all openings."""

        return {
            "entry_digest": self.entry_digest,
            "ledger_id": self.ledger_id,
            "salts": dict(self.salts),
            "schema": CONTRIBUTION_LEDGER_SCHEMA,
            "sequence": self.sequence,
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, payload: object) -> ContributionReceipt:
        """Parse a strict holder receipt mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The parsed receipt.

        Raises:
            ContributionLedgerError: ``contribution_ledger_invalid_receipt``
                for any unsupported shape, extra field, or value.
        """

        receipt = _exact_mapping(payload, _RECEIPT_KEYS, "invalid_receipt")
        if receipt["schema"] != CONTRIBUTION_LEDGER_SCHEMA:
            raise _fail("invalid_receipt")
        return cls(
            ledger_id=receipt["ledger_id"],
            sequence=receipt["sequence"],
            entry_digest=receipt["entry_digest"],
            values=receipt["values"],
            salts=receipt["salts"],
        )


@dataclass(frozen=True, slots=True)
class DisclosureVerification:
    """Result of checking a disclosure proof against a public ledger.

    Args:
        valid: Whether the ledger and every opening verified.
        reason: Stable :class:`DisclosureReasonCode`.
        sequence: Verified entry sequence, present only when valid.
        disclosed: Verified disclosed values; empty unless valid.
    """

    valid: bool
    reason: DisclosureReasonCode
    sequence: int | None = None
    disclosed: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )

    def __repr__(self) -> str:
        return (
            f"DisclosureVerification(valid={self.valid}, reason={self.reason.value!r}, "
            f"sequence={self.sequence}, fields={tuple(self.disclosed)!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible result with verified field names only."""

        return {
            "fields": list(self.disclosed),
            "reason": self.reason.value,
            "sequence": self.sequence,
            "valid": self.valid,
        }


def _disclosure_failure(reason: DisclosureReasonCode) -> DisclosureVerification:
    return DisclosureVerification(False, reason)


def verify_disclosure(
    ledger: PublicContributionLedger | ContributionLedger | Mapping[str, Any] | str,
    proof: DisclosureProof | Mapping[str, Any] | str,
    *,
    checkpoint: LedgerCheckpoint | None = None,
) -> DisclosureVerification:
    """Verify selectively disclosed receipt fields against a public ledger.

    The ledger must first pass :func:`verify_ledger` (with ``checkpoint`` when
    given).  The proof must name the same ledger and an existing entry with
    the same digest, and each disclosed value and salt must reproduce that
    entry's public commitment.  Undisclosed fields are never needed.

    Args:
        ledger: Public ledger object, mapping, or JSON text.
        proof: Disclosure proof object, mapping, or JSON text.
        checkpoint: Optional published anchor for the ledger.

    Returns:
        A :class:`DisclosureVerification`; ``disclosed`` holds values only
        when every check passed.
    """

    try:
        if isinstance(proof, DisclosureProof):
            parsed = proof
        elif isinstance(proof, str):
            parsed = DisclosureProof.from_json(proof)
        else:
            parsed = DisclosureProof.from_dict(proof)
    except ContributionLedgerError:
        return _disclosure_failure(DisclosureReasonCode.PROOF_MALFORMED)

    try:
        public = _coerce_public(ledger)
    except ContributionLedgerError:
        return _disclosure_failure(DisclosureReasonCode.LEDGER_INVALID)
    if not verify_ledger(public, checkpoint=checkpoint).valid:
        return _disclosure_failure(DisclosureReasonCode.LEDGER_INVALID)
    if parsed.ledger_id != public.ledger_id:
        return _disclosure_failure(DisclosureReasonCode.LEDGER_MISMATCH)
    if parsed.sequence >= len(public.entries):
        return _disclosure_failure(DisclosureReasonCode.ENTRY_NOT_FOUND)
    entry = public.entries[parsed.sequence]
    if not hmac.compare_digest(entry.entry_digest, parsed.entry_digest):
        return _disclosure_failure(DisclosureReasonCode.ENTRY_MISMATCH)

    for name, (value, salt) in parsed.openings.items():
        expected = entry.commitments[name]
        if not hmac.compare_digest(_commit(name, salt, value), expected):
            return _disclosure_failure(DisclosureReasonCode.OPENING_INVALID)
    disclosed = {name: value for name, (value, _salt) in parsed.openings.items()}
    return DisclosureVerification(
        True, DisclosureReasonCode.OK, parsed.sequence, MappingProxyType(disclosed)
    )


# ---------------------------------------------------------------------------
# Coordinator-side append-only ledger
# ---------------------------------------------------------------------------


class _ContributionChain(HashChainAuditLog):
    """Hash-chained audit log anchored at a ledger-specific genesis digest."""

    def __init__(self, genesis_digest: str) -> None:
        super().__init__()
        self.GENESIS_HASH = genesis_digest


class ContributionLedger:
    """Append-only ledger of pseudonymous federated participation receipts.

    The ledger keeps only public commitments.  Two keys are derived from the
    injected per-federation key with HMAC context separation: one for
    participant pseudonyms and one for deterministic per-field commitment
    salts.  The raw key, labels, and receipt values are never stored.

    Args:
        ledger_id: Opaque lowercase federation ledger identifier, such as
            ``fed-synthetic-01``.  Do not use an institution name.
        key: Per-federation secret of 32 to 4096 bytes.

    Raises:
        ContributionLedgerError: If the identifier or key is unsupported.
    """

    __slots__ = ("_chain", "_ledger_id", "_pseudonym_key", "_salt_key")

    def __init__(self, ledger_id: str, key: bytes) -> None:
        self._ledger_id = _require_ledger_id(ledger_id)
        secret = _require_key(key)
        self._pseudonym_key = _derive(secret, _PSEUDONYM_CONTEXT)
        self._salt_key = _derive(secret, _SALT_CONTEXT)
        self._chain = _ContributionChain(_genesis_digest(self._ledger_id))

    def __repr__(self) -> str:
        return (
            f"ContributionLedger(ledger_id={self._ledger_id!r}, "
            f"entries={len(self._chain.records)})"
        )

    def __len__(self) -> int:
        return len(self._chain.records)

    def __reduce__(self) -> Any:
        raise TypeError("ContributionLedger holds derived key material")

    @property
    def ledger_id(self) -> str:
        """Return the opaque ledger identifier."""

        return self._ledger_id

    def participant_handle(self, label: str) -> str:
        """Return the pseudonymous handle for a local label under this key.

        Args:
            label: Local participant label; used transiently only.

        Returns:
            The same handle :func:`participant_handle` returns for this key.

        Raises:
            ContributionLedgerError: If the label is unsupported.
        """

        return _handle_from_pseudonym_key(self._pseudonym_key, label)

    def _salt(self, sequence: int, name: str) -> str:
        message = _canonical_json(
            {"field": name, "ledger_id": self._ledger_id, "sequence": sequence}
        ).encode("ascii")
        return hmac.new(self._salt_key, message, hashlib.sha256).hexdigest()

    def append(
        self,
        *,
        participant_handle: str,
        round_ref: str,
        round_role: ContributionRole | str,
        policy_version: int,
        update_commitment: str | FederatedUpdateMetadata | None = None,
        **refused: Any,
    ) -> ContributionReceipt:
        """Append one receipt and return the holder's private openings.

        Args:
            participant_handle: Handle from :meth:`participant_handle`.
            round_ref: SHA-256 digest referencing the federated round.
            round_role: One :class:`ContributionRole` value.
            policy_version: Positive governing policy version.
            update_commitment: Declared update digest, or validated
                :class:`FederatedUpdateMetadata` whose ``update_digest`` is
                used.  Required for ``participant`` and forbidden otherwise.
            **refused: Any additional field, such as counts or metrics, is
                refused; nothing is appended.

        Returns:
            The :class:`ContributionReceipt` for the new entry.

        Raises:
            ContributionLedgerError: For refused or invalid fields.
        """

        if refused:
            raise _fail("field_refused")
        if len(self._chain.records) >= MAX_LEDGER_ENTRIES:
            raise _fail("ledger_full")
        if isinstance(update_commitment, FederatedUpdateMetadata):
            update_commitment = update_commitment.update_digest
        role = _validate_value("round_role", round_role)
        values = {
            "participant_handle": _validate_value(
                "participant_handle", participant_handle
            ),
            "round_ref": _validate_value("round_ref", round_ref),
            "round_role": role,
            "policy_version": _validate_value("policy_version", policy_version),
            "update_commitment": _validate_value(
                "update_commitment", update_commitment, role=role
            ),
        }
        sequence = len(self._chain.records)
        salts = {name: self._salt(sequence, name) for name in RECEIPT_FIELDS}
        commitments = {
            name: _commit(name, salts[name], values[name]) for name in RECEIPT_FIELDS
        }
        record = self._chain.append(
            _EVENT_TYPE, _chain_payload(self._ledger_id, commitments)
        )
        return ContributionReceipt(
            ledger_id=self._ledger_id,
            sequence=record.sequence,
            entry_digest=record.record_hash,
            values=values,
            salts=salts,
        )

    def public_ledger(self) -> PublicContributionLedger:
        """Return the publishable ledger containing commitments only."""

        entries = tuple(
            LedgerEntry(
                sequence=record.sequence,
                previous_digest=record.previous_hash,
                commitments=record.payload["commitments"],
                entry_digest=record.record_hash,
            )
            for record in self._chain.records
        )
        return PublicContributionLedger(
            ledger_id=self._ledger_id,
            genesis_digest=self._chain.GENESIS_HASH,
            entries=entries,
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible public ledger document."""

        return self.public_ledger().to_dict()

    def to_json(self) -> str:
        """Return deterministic compact public ledger JSON."""

        return self.public_ledger().to_json()

    def checkpoint(self) -> LedgerCheckpoint:
        """Return the current anchor for later truncation and fork checks."""

        records = self._chain.records
        head = records[-1].record_hash if records else self._chain.GENESIS_HASH
        return LedgerCheckpoint(len(records), head)

    def verify(self) -> LedgerVerification:
        """Verify this ledger's public form exactly as an outside verifier."""

        return verify_ledger(self.public_ledger())

    @classmethod
    def resume(
        cls,
        public: PublicContributionLedger | Mapping[str, Any] | str,
        key: bytes,
        *,
        checkpoint: LedgerCheckpoint | None = None,
    ) -> ContributionLedger:
        """Rebuild a coordinator ledger from a verified public ledger.

        Args:
            public: Previously published ledger object, mapping, or JSON.
            key: The same per-federation key used to build it.
            checkpoint: Optional anchor the ledger must satisfy.

        Returns:
            A ledger ready to append after the existing entries.

        Raises:
            ContributionLedgerError: ``contribution_ledger_unverified`` if the
                public ledger fails continuity verification.
        """

        if checkpoint is not None and type(checkpoint) is not LedgerCheckpoint:
            raise _fail("invalid_checkpoint")
        try:
            parsed = _coerce_public(public)
        except ContributionLedgerError:
            raise _fail("unverified") from None
        if not _verify_public(parsed, checkpoint).valid:
            raise _fail("unverified")
        ledger = cls(parsed.ledger_id, key)
        for entry in parsed.entries:
            record = ledger._chain.append(
                _EVENT_TYPE, _chain_payload(parsed.ledger_id, entry.commitments)
            )
            if record.record_hash != entry.entry_digest:
                raise _fail("unverified")
        return ledger


__all__ = [
    "CONTRIBUTION_LEDGER_SCHEMA",
    "MAX_FEDERATION_KEY_BYTES",
    "MAX_LEDGER_ENTRIES",
    "MAX_POLICY_VERSION",
    "MIN_FEDERATION_KEY_BYTES",
    "RECEIPT_FIELDS",
    "ContributionLedger",
    "ContributionLedgerError",
    "ContributionReceipt",
    "ContributionRole",
    "DisclosureProof",
    "DisclosureReasonCode",
    "DisclosureVerification",
    "LedgerCheckpoint",
    "LedgerEntry",
    "LedgerReasonCode",
    "LedgerVerification",
    "PublicContributionLedger",
    "participant_handle",
    "verify_disclosure",
    "verify_ledger",
]
