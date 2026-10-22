"""Hold federated checkpoints derived from updates signed by a compromised key.

A site key compromise discovered after the fact means updates accepted during
the exposure window may be attacker-controlled. This module accepts a signed
incident record, resolves the accepted-update commitments of the compromised
key through injected ports, holds the rounds that accepted them together with
every descendant checkpoint, and classifies remediation.

Lineage, commitments, identity revocation, incident signatures and operator
authority are all application-owned ports. The module performs no network or
file effects. Reports, audit payloads and persisted state carry only fixed
codes, counts, opaque lineage references and digests; they never contain the
key reference, site identity, signatures or update commitments.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Final, Protocol

from openmed.compliance.audit_chain import AuditSink
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundState,
)

FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION = "openmed.training.federated_key_compromise.v1"
FEDERATED_KEY_COMPROMISE_STATE_SCHEMA_VERSION = (
    "openmed.training.federated_key_compromise_state.v1"
)
KEY_COMPROMISE_RECORDED_EVENT: Final = "training.federated.key_compromise.recorded"
KEY_COMPROMISE_LIFTED_EVENT: Final = "training.federated.key_compromise.lifted"

_INCIDENT_DOMAIN: Final = b"openmed.training.federated_key_compromise.incident.v1\n"
_GRANT_DOMAIN: Final = b"openmed.training.federated_key_compromise.lift_grant.v1\n"
_HELD_SET_DOMAIN: Final = "openmed.training.federated_key_compromise.held_set.v1"
_AFFECTED_DOMAIN: Final = "openmed.training.federated_key_compromise.affected.v1"
_MAX_TIME: Final = (1 << 63) - 1
_MAX_SIGNATURE_CHARS: Final = 8192
_MAX_PORT_ITEMS: Final = 100_000
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INCIDENT_ID: Final = re.compile(r"incident:[0-9a-f]{32,64}\Z")
_REFERENCE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_SIGNATURE: Final = re.compile(r"[\x21-\x7e]+\Z")
_INCIDENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "incident_id",
        "key_reference",
        "exposure_start",
        "discovered_at",
    }
)


class KeyCompromiseErrorCode(str, Enum):
    """Controlled refusal codes; messages never echo submitted values."""

    INVALID_INCIDENT = "invalid_incident"
    INVALID_REFERENCE = "invalid_reference"
    INVALID_STATE = "invalid_state"
    SIGNATURE_INVALID = "incident_signature_invalid"
    INCIDENT_CONFLICT = "incident_conflict"
    UNKNOWN_INCIDENT = "unknown_incident"
    IDENTITY_UNAVAILABLE = "identity_port_unavailable"
    COMMITMENT_UNAVAILABLE = "commitment_port_unavailable"
    COMMITMENT_INVALID = "commitment_port_invalid"
    LINEAGE_UNAVAILABLE = "lineage_port_unavailable"
    LINEAGE_INVALID = "lineage_port_invalid"
    LINEAGE_UNRESOLVED = "lineage_unresolved"
    CHECKPOINT_HELD = "checkpoint_held"
    ROUND_HELD = "round_held"
    AUTHORITY_REQUIRED = "operator_authority_required"
    AUTHORITY_INVALID = "operator_authority_invalid"
    AUDIT_UNAVAILABLE = "audit_unavailable"

    def __str__(self) -> str:
        return self.value


class KeyCompromiseHoldError(ValueError):
    """Raised with a fixed message and a controlled :class:`KeyCompromiseErrorCode`.

    Args:
        code: The controlled refusal code.
    """

    def __init__(self, code: KeyCompromiseErrorCode) -> None:
        self.code = code
        super().__init__(f"federated key compromise refused: {code.value}")


class IncidentStatus(str, Enum):
    """Hold status of one recorded incident."""

    HELD = "held"
    NO_AFFECTED_UPDATES = "no_affected_updates"
    LIFTED = "lifted"

    def __str__(self) -> str:
        return self.value


class RemediationClass(str, Enum):
    """Remediation available for the rounds held by one incident.

    Classes are ordered by severity; an incident reports the most severe class
    across its earliest affected rounds.
    """

    NONE_REQUIRED = "none_required"
    REAGGREGATE_WITHOUT_AFFECTED = "reaggregate_without_affected"
    RETRAIN_FROM_CLEAN_ANCESTOR = "retrain_from_clean_ancestor"
    UNSUPPORTED = "unsupported"

    def __str__(self) -> str:
        return self.value


_SEVERITY: Final = {
    RemediationClass.NONE_REQUIRED: 0,
    RemediationClass.REAGGREGATE_WITHOUT_AFFECTED: 1,
    RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR: 2,
    RemediationClass.UNSUPPORTED: 3,
}


def _error(code: KeyCompromiseErrorCode) -> KeyCompromiseHoldError:
    return KeyCompromiseHoldError(code)


def _is_time(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_TIME


def _is_reference(value: object) -> bool:
    return type(value) is str and _REFERENCE.fullmatch(value) is not None


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(domain: bytes, payload: Mapping[str, Any]) -> bytes:
    body = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return domain + body.encode("ascii")


@dataclass(frozen=True)
class KeyCompromiseIncident:
    """An incident payload naming a compromised key and its exposure window.

    Args:
        incident_id: Random opaque identifier, ``incident:`` plus 32 to 64
            lowercase hex characters. The entropy keeps incident digests from
            being linked back to a key reference by enumeration.
        key_reference: Opaque reference to the compromised site signing key.
            It is used only to query the injected ports and never appears in
            reports, audit payloads, persisted state or ``repr``.
        exposure_start: First second, in epoch seconds, at which the key may
            have been controlled by an attacker.
        discovered_at: Second at which the compromise was discovered. Must not
            precede ``exposure_start``.
        schema_version: Must equal ``FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION``.

    Raises:
        KeyCompromiseHoldError: ``invalid_incident`` for any malformed field.
    """

    incident_id: str
    key_reference: str = field(repr=False)
    exposure_start: int
    discovered_at: int
    schema_version: str = FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.incident_id) is not str
            or _INCIDENT_ID.fullmatch(self.incident_id) is None
            or not _is_reference(self.key_reference)
            or not _is_time(self.exposure_start)
            or not _is_time(self.discovered_at)
            or self.discovered_at < self.exposure_start
            or self.schema_version != FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)

    def to_dict(self) -> dict[str, Any]:
        """Return the signed payload fields, including the key reference."""

        return {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "key_reference": self.key_reference,
            "exposure_start": self.exposure_start,
            "discovered_at": self.discovered_at,
        }

    def signing_bytes(self) -> bytes:
        """Return the domain-separated canonical bytes an issuer signs."""

        return _canonical_bytes(_INCIDENT_DOMAIN, self.to_dict())

    @property
    def digest(self) -> str:
        """Return the stable incident digest used in reports and lift grants."""

        return "sha256:" + hashlib.sha256(self.signing_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KeyCompromiseIncident:
        """Parse a strict incident payload.

        Args:
            payload: Mapping with exactly the incident fields.

        Returns:
            The validated incident.

        Raises:
            KeyCompromiseHoldError: ``invalid_incident`` for unknown, missing
                or malformed fields.
        """

        if not isinstance(payload, Mapping):
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)
        try:
            keys = set(payload)
        except Exception:
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT) from None
        if keys != _INCIDENT_FIELDS:
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)
        return cls(
            incident_id=payload["incident_id"],
            key_reference=payload["key_reference"],
            exposure_start=payload["exposure_start"],
            discovered_at=payload["discovered_at"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class SignedKeyCompromiseIncident:
    """An incident payload plus the issuer signature over its signing bytes.

    Args:
        incident: The incident payload.
        signature: Printable ASCII signature checked by the injected verifier.
            Hidden from ``repr``.

    Raises:
        KeyCompromiseHoldError: ``invalid_incident`` for malformed values.
    """

    incident: KeyCompromiseIncident
    signature: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.incident, KeyCompromiseIncident)
            or type(self.signature) is not str
            or len(self.signature) > _MAX_SIGNATURE_CHARS
            or _SIGNATURE.fullmatch(self.signature) is None
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)

    def to_dict(self) -> dict[str, Any]:
        """Return the transport form ``{"incident": ..., "signature": ...}``."""

        return {"incident": self.incident.to_dict(), "signature": self.signature}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SignedKeyCompromiseIncident:
        """Parse a strict signed incident record.

        Args:
            payload: Mapping with exactly ``incident`` and ``signature``.

        Returns:
            The parsed record. Its signature is not verified here.

        Raises:
            KeyCompromiseHoldError: ``invalid_incident`` for malformed records.
        """

        if not isinstance(payload, Mapping):
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)
        try:
            keys = set(payload)
        except Exception:
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT) from None
        if keys != {"incident", "signature"}:
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)
        return cls(
            incident=KeyCompromiseIncident.from_dict(payload["incident"]),
            signature=payload["signature"],
        )


@dataclass(frozen=True)
class AcceptedUpdateCommitment:
    """One accepted update commitment returned by the commitment port.

    Args:
        commitment: ``sha256:`` commitment to the accepted update. Hidden from
            ``repr``; never written to reports or state.
        round_ref: Opaque reference of the round that accepted the update.
        accepted_at: Coordinator acceptance time in epoch seconds. The window
            is compared against this trusted time, not a signer-claimed time,
            because a compromised key can backdate its own signatures.

    Raises:
        KeyCompromiseHoldError: ``invalid_reference`` for malformed values.
    """

    commitment: str = field(repr=False)
    round_ref: str
    accepted_at: int

    def __post_init__(self) -> None:
        if (
            not _is_digest(self.commitment)
            or not _is_reference(self.round_ref)
            or not _is_time(self.accepted_at)
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_REFERENCE)


@dataclass(frozen=True)
class RoundCommitmentSet:
    """Retained update commitments and the quorum of one aggregation round.

    Args:
        retained: Commitments whose updates are still retained and could be
            re-aggregated. Hidden from ``repr``.
        quorum: Minimum number of updates the round requires; at least one.

    Raises:
        KeyCompromiseHoldError: ``invalid_reference`` for malformed values.
    """

    retained: frozenset[str] = field(repr=False)
    quorum: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.retained, frozenset)
            or len(self.retained) > _MAX_PORT_ITEMS
            or not all(_is_digest(item) for item in self.retained)
            or type(self.quorum) is not int
            or not 1 <= self.quorum <= _MAX_PORT_ITEMS
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_REFERENCE)


@dataclass(frozen=True)
class LineageNode:
    """One checkpoint in the training lineage returned by the lineage port.

    Args:
        checkpoint_ref: Opaque checkpoint reference.
        round_ref: Opaque reference of the round that produced the checkpoint,
            or ``None`` for an externally supplied base checkpoint.
        parent_refs: Checkpoints the producing round started from.

    Raises:
        KeyCompromiseHoldError: ``invalid_reference`` for malformed values or
            a checkpoint that names itself as a parent.
    """

    checkpoint_ref: str
    round_ref: str | None
    parent_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not _is_reference(self.checkpoint_ref)
            or not (self.round_ref is None or _is_reference(self.round_ref))
            or type(self.parent_refs) is not tuple
            or len(self.parent_refs) > _MAX_PORT_ITEMS
            or not all(_is_reference(parent) for parent in self.parent_refs)
            or len(set(self.parent_refs)) != len(self.parent_refs)
            or self.checkpoint_ref in self.parent_refs
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_REFERENCE)


@dataclass(frozen=True)
class HoldLiftGrant:
    """Signed operator authority to lift one incident hold.

    Args:
        incident_digest: Digest of the incident whose hold is lifted. A grant
            for one incident cannot lift another.
        operator_reference: Opaque operator reference, checked by the injected
            authority verifier. Hidden from ``repr`` and never persisted.
        issued_at: Grant issue time in epoch seconds.
        signature: Printable ASCII signature over :meth:`signing_bytes`.

    Raises:
        KeyCompromiseHoldError: ``operator_authority_invalid`` for malformed
            values.
    """

    incident_digest: str
    operator_reference: str = field(repr=False)
    issued_at: int
    signature: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not _is_digest(self.incident_digest)
            or not _is_reference(self.operator_reference)
            or not _is_time(self.issued_at)
            or type(self.signature) is not str
            or len(self.signature) > _MAX_SIGNATURE_CHARS
            or _SIGNATURE.fullmatch(self.signature) is None
        ):
            raise _error(KeyCompromiseErrorCode.AUTHORITY_INVALID)

    def signing_bytes(self) -> bytes:
        """Return the domain-separated canonical bytes an operator signs."""

        return _canonical_bytes(
            _GRANT_DOMAIN,
            {
                "action": "lift_hold",
                "incident_digest": self.incident_digest,
                "operator_reference": self.operator_reference,
                "issued_at": self.issued_at,
            },
        )


class SignatureVerifier(Protocol):
    """Application-owned verifier for incident records or operator grants."""

    def verify(self, message: bytes, signature: str) -> bool:
        """Return exactly ``True`` only when ``signature`` signs ``message``."""
        ...


class KeyRevocationPort(Protocol):
    """Application-owned site identity registry."""

    def revocation_time(self, key_reference: str) -> int | None:
        """Return when revocation took effect, or ``None`` if still active."""
        ...


class UpdateCommitmentPort(Protocol):
    """Application-owned store of accepted update commitments."""

    def accepted_updates(
        self, key_reference: str
    ) -> Iterable[AcceptedUpdateCommitment]:
        """Return every accepted update commitment signed by the key."""
        ...

    def round_commitments(self, round_ref: str) -> RoundCommitmentSet | None:
        """Return retained commitments and quorum of a round, or ``None``."""
        ...


class CheckpointLineagePort(Protocol):
    """Application-owned checkpoint lineage store."""

    def nodes_snapshot(self) -> Iterable[LineageNode]:
        """Return a consistent snapshot of every known lineage node."""
        ...


@dataclass(frozen=True)
class KeyCompromiseIncidentReport:
    """Digest-only incident report.

    Args:
        incident_digest: Digest of the signed incident payload.
        status: Current hold status.
        window_closed: Whether revocation bounded the exposure window. An open
            window holds every acceptance from the exposure start onward.
        held_round_count: Rounds held by this incident, descendants included.
        held_checkpoint_count: Checkpoints held by this incident.
        held_set_digest: Digest of the sorted held rounds and checkpoints.
        affected_commitment_digest: Digest of the sorted affected commitments.
        remediation: Most severe remediation class across affected rounds.
        schema_version: Report schema version.
    """

    incident_digest: str
    status: IncidentStatus
    window_closed: bool
    held_round_count: int
    held_checkpoint_count: int
    held_set_digest: str
    affected_commitment_digest: str
    remediation: RemediationClass
    schema_version: str = FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the closed, JSON-compatible report fields."""

        return {
            "schema_version": self.schema_version,
            "incident_digest": self.incident_digest,
            "status": self.status.value,
            "window_closed": self.window_closed,
            "held_round_count": self.held_round_count,
            "held_checkpoint_count": self.held_checkpoint_count,
            "held_set_digest": self.held_set_digest,
            "affected_commitment_digest": self.affected_commitment_digest,
            "remediation": self.remediation.value,
        }

    def to_json(self) -> str:
        """Return deterministic JSON with sorted keys and a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def _from_dict(cls, payload: Any) -> KeyCompromiseIncidentReport:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "incident_digest",
            "status",
            "window_closed",
            "held_round_count",
            "held_checkpoint_count",
            "held_set_digest",
            "affected_commitment_digest",
            "remediation",
        }:
            raise _error(KeyCompromiseErrorCode.INVALID_STATE)
        try:
            report = cls(
                incident_digest=payload["incident_digest"],
                status=IncidentStatus(payload["status"]),
                window_closed=payload["window_closed"],
                held_round_count=payload["held_round_count"],
                held_checkpoint_count=payload["held_checkpoint_count"],
                held_set_digest=payload["held_set_digest"],
                affected_commitment_digest=payload["affected_commitment_digest"],
                remediation=RemediationClass(payload["remediation"]),
                schema_version=payload["schema_version"],
            )
        except (TypeError, ValueError):
            raise _error(KeyCompromiseErrorCode.INVALID_STATE) from None
        if (
            report.schema_version != FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION
            or not _is_digest(report.incident_digest)
            or not _is_digest(report.held_set_digest)
            or not _is_digest(report.affected_commitment_digest)
            or type(report.window_closed) is not bool
            or not _is_time(report.held_round_count)
            or not _is_time(report.held_checkpoint_count)
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_STATE)
        return report


@dataclass
class _HoldRecord:
    incident_id: str
    report: KeyCompromiseIncidentReport
    held_rounds: frozenset[str]
    held_checkpoints: frozenset[str]
    lifted: bool = False
    audited: bool = False

    @property
    def active(self) -> bool:
        return not self.lifted and bool(self.held_rounds or self.held_checkpoints)

    def current_report(self) -> KeyCompromiseIncidentReport:
        if self.lifted:
            return replace(self.report, status=IncidentStatus.LIFTED)
        return self.report

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "report": self.report.to_dict(),
            "held_rounds": sorted(self.held_rounds),
            "held_checkpoints": sorted(self.held_checkpoints),
            "lifted": self.lifted,
            "audited": self.audited,
        }


class KeyCompromiseHoldRegistry:
    """Record compromised-key incidents, hold affected lineage, gate promotion.

    The registry owns only hold state. Persist :meth:`to_dict` after each call
    and pass it back as ``state`` after a restart; re-applying a recorded
    incident is idempotent and never re-holds a lifted incident.

    Args:
        incident_verifier: Verifies issuer signatures on incident records.
        authority_verifier: Verifies operator signatures on lift grants. It
            should also check that the operator currently holds hold-lifting
            authority.
        revocation: Site identity registry resolving revocation times.
        commitments: Accepted update commitment store.
        lineage: Checkpoint lineage store.
        audit_sink: Optional append-only audit sink, for example
            :class:`openmed.compliance.audit_chain.HashChainAuditLog`.
        state: Previously persisted :meth:`to_dict` output.

    Raises:
        KeyCompromiseHoldError: ``invalid_state`` for malformed ``state``.
    """

    def __init__(
        self,
        *,
        incident_verifier: SignatureVerifier,
        authority_verifier: SignatureVerifier,
        revocation: KeyRevocationPort,
        commitments: UpdateCommitmentPort,
        lineage: CheckpointLineagePort,
        audit_sink: AuditSink | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self._incident_verifier = incident_verifier
        self._authority_verifier = authority_verifier
        self._revocation = revocation
        self._commitments = commitments
        self._lineage = lineage
        self._audit_sink = audit_sink
        self._records: dict[str, _HoldRecord] = {}
        if state is not None:
            self._records = _parse_state(state)

    def __repr__(self) -> str:
        active = sum(1 for record in self._records.values() if record.active)
        return (
            f"{type(self).__name__}(incidents={len(self._records)}, "
            f"active_holds={active})"
        )

    # Incident intake ---------------------------------------------------------

    def apply_incident(
        self, signed: SignedKeyCompromiseIncident
    ) -> KeyCompromiseIncidentReport:
        """Verify a signed incident, hold affected lineage and record it.

        Args:
            signed: The signed incident record.

        Returns:
            The digest-only incident report. Re-applying a recorded incident
            returns its current report without a second audit record.

        Raises:
            KeyCompromiseHoldError: ``incident_signature_invalid``,
                ``incident_conflict``, a port code, or ``audit_unavailable``.
                Holds are committed before the audit append, so an audit
                failure leaves the lineage held and the append is retried on
                the next application.
        """

        if not isinstance(signed, SignedKeyCompromiseIncident):
            raise _error(KeyCompromiseErrorCode.INVALID_INCIDENT)
        incident = signed.incident
        if not _verified(self._incident_verifier, incident.signing_bytes(), signed):
            raise _error(KeyCompromiseErrorCode.SIGNATURE_INVALID)
        digest = incident.digest
        for existing_digest, existing in self._records.items():
            if existing.incident_id == incident.incident_id and existing_digest != (
                digest
            ):
                raise _error(KeyCompromiseErrorCode.INCIDENT_CONFLICT)
        record = self._records.get(digest)
        if record is None:
            record = self._resolve(incident, digest)
            self._records[digest] = record
        if not record.audited:
            self._append(KEY_COMPROMISE_RECORDED_EVENT, record.report.to_dict())
            record.audited = True
        return record.current_report()

    def _resolve(self, incident: KeyCompromiseIncident, digest: str) -> _HoldRecord:
        window_end = self._window_end(incident)
        affected = self._affected_updates(incident, window_end)
        nodes = self._lineage_nodes()
        by_round = {node.round_ref: node for node in nodes.values() if node.round_ref}
        affected_rounds = frozenset(update.round_ref for update in affected)
        roots = frozenset(
            by_round[round_ref].checkpoint_ref
            for round_ref in affected_rounds
            if round_ref in by_round
        )
        held_checkpoints = _descendants(nodes, roots)
        held_rounds = affected_rounds | frozenset(
            nodes[checkpoint].round_ref
            for checkpoint in held_checkpoints
            if nodes[checkpoint].round_ref is not None
        )
        remediation = self._remediation(
            nodes=nodes,
            by_round=by_round,
            affected_rounds=affected_rounds,
            tainted=held_checkpoints,
            excluded=frozenset(update.commitment for update in affected),
        )
        report = KeyCompromiseIncidentReport(
            incident_digest=digest,
            status=(
                IncidentStatus.HELD if affected else IncidentStatus.NO_AFFECTED_UPDATES
            ),
            window_closed=window_end is not None,
            held_round_count=len(held_rounds),
            held_checkpoint_count=len(held_checkpoints),
            held_set_digest=_sha256(
                {
                    "domain": _HELD_SET_DOMAIN,
                    "rounds": sorted(held_rounds),
                    "checkpoints": sorted(held_checkpoints),
                }
            ),
            affected_commitment_digest=_sha256(
                {
                    "domain": _AFFECTED_DOMAIN,
                    "commitments": sorted(update.commitment for update in affected),
                }
            ),
            remediation=remediation,
        )
        return _HoldRecord(
            incident_id=incident.incident_id,
            report=report,
            held_rounds=held_rounds,
            held_checkpoints=held_checkpoints,
        )

    def _window_end(self, incident: KeyCompromiseIncident) -> int | None:
        try:
            revoked_at = self._revocation.revocation_time(incident.key_reference)
        except Exception:
            raise _error(KeyCompromiseErrorCode.IDENTITY_UNAVAILABLE) from None
        if revoked_at is None:
            return None
        if not _is_time(revoked_at):
            raise _error(KeyCompromiseErrorCode.IDENTITY_UNAVAILABLE)
        return max(revoked_at, incident.discovered_at)

    def _affected_updates(
        self, incident: KeyCompromiseIncident, window_end: int | None
    ) -> tuple[AcceptedUpdateCommitment, ...]:
        try:
            raw = list(self._commitments.accepted_updates(incident.key_reference))
        except Exception:
            raise _error(KeyCompromiseErrorCode.COMMITMENT_UNAVAILABLE) from None
        if len(raw) > _MAX_PORT_ITEMS:
            raise _error(KeyCompromiseErrorCode.COMMITMENT_INVALID)
        seen: set[str] = set()
        affected = []
        for item in raw:
            if type(item) is not AcceptedUpdateCommitment:
                raise _error(KeyCompromiseErrorCode.COMMITMENT_INVALID)
            try:
                checked = AcceptedUpdateCommitment(
                    item.commitment, item.round_ref, item.accepted_at
                )
            except (AttributeError, KeyCompromiseHoldError):
                raise _error(KeyCompromiseErrorCode.COMMITMENT_INVALID) from None
            if checked.commitment in seen:
                raise _error(KeyCompromiseErrorCode.COMMITMENT_INVALID)
            seen.add(checked.commitment)
            if checked.accepted_at < incident.exposure_start:
                continue
            if window_end is not None and checked.accepted_at > window_end:
                continue
            affected.append(checked)
        return tuple(sorted(affected, key=lambda update: update.commitment))

    def _lineage_nodes(self) -> dict[str, LineageNode]:
        try:
            raw = list(self._lineage.nodes_snapshot())
        except Exception:
            raise _error(KeyCompromiseErrorCode.LINEAGE_UNAVAILABLE) from None
        if len(raw) > _MAX_PORT_ITEMS:
            raise _error(KeyCompromiseErrorCode.LINEAGE_INVALID)
        nodes: dict[str, LineageNode] = {}
        rounds: set[str] = set()
        for item in raw:
            if type(item) is not LineageNode:
                raise _error(KeyCompromiseErrorCode.LINEAGE_INVALID)
            try:
                node = LineageNode(
                    item.checkpoint_ref, item.round_ref, item.parent_refs
                )
            except (AttributeError, KeyCompromiseHoldError):
                raise _error(KeyCompromiseErrorCode.LINEAGE_INVALID) from None
            if node.checkpoint_ref in nodes or node.round_ref in rounds:
                raise _error(KeyCompromiseErrorCode.LINEAGE_INVALID)
            nodes[node.checkpoint_ref] = node
            if node.round_ref is not None:
                rounds.add(node.round_ref)
        for node in nodes.values():
            if any(parent not in nodes for parent in node.parent_refs):
                raise _error(KeyCompromiseErrorCode.LINEAGE_INVALID)
        _require_acyclic(nodes)
        return nodes

    def _remediation(
        self,
        *,
        nodes: Mapping[str, LineageNode],
        by_round: Mapping[str, LineageNode],
        affected_rounds: frozenset[str],
        tainted: frozenset[str],
        excluded: frozenset[str],
    ) -> RemediationClass:
        if not affected_rounds:
            return RemediationClass.NONE_REQUIRED
        unclean = tainted | self._active_checkpoints()
        affected_checkpoints = frozenset(
            by_round[round_ref].checkpoint_ref
            for round_ref in affected_rounds
            if round_ref in by_round
        )
        worst = RemediationClass.NONE_REQUIRED
        for round_ref in sorted(affected_rounds):
            node = by_round.get(round_ref)
            if node is None:
                return RemediationClass.UNSUPPORTED
            if _ancestors(nodes, node.parent_refs) & affected_checkpoints:
                # A descendant of another affected round is replaced by that
                # earlier round's remediation and does not set the class.
                continue
            current = self._classify_root(node, nodes, unclean, excluded)
            if _SEVERITY[current] > _SEVERITY[worst]:
                worst = current
        return worst

    def _classify_root(
        self,
        node: LineageNode,
        nodes: Mapping[str, LineageNode],
        unclean: frozenset[str],
        excluded: frozenset[str],
    ) -> RemediationClass:
        if not node.parent_refs:
            return RemediationClass.UNSUPPORTED
        if all(parent not in unclean for parent in node.parent_refs):
            if node.round_ref is not None and self._reaggregable(
                node.round_ref, excluded
            ):
                return RemediationClass.REAGGREGATE_WITHOUT_AFFECTED
            return RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR
        if _ancestors(nodes, node.parent_refs) - unclean:
            return RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR
        return RemediationClass.UNSUPPORTED

    def _reaggregable(self, round_ref: str, excluded: frozenset[str]) -> bool:
        # Remediation is advisory; an unavailable or malformed answer can only
        # downgrade to retraining, never weaken the hold.
        try:
            summary = self._commitments.round_commitments(round_ref)
        except Exception:
            return False
        if type(summary) is not RoundCommitmentSet:
            return False
        try:
            checked = RoundCommitmentSet(summary.retained, summary.quorum)
        except (AttributeError, KeyCompromiseHoldError):
            return False
        return len(checked.retained - excluded) >= checked.quorum

    # Promotion gate ----------------------------------------------------------

    def _active_checkpoints(self) -> frozenset[str]:
        return frozenset(
            checkpoint
            for record in self._records.values()
            if not record.lifted
            for checkpoint in record.held_checkpoints
        )

    def held_checkpoints(self) -> tuple[str, ...]:
        """Return sorted checkpoint references held by any active incident."""

        return tuple(sorted(self._active_checkpoints()))

    def held_rounds(self) -> tuple[str, ...]:
        """Return sorted round references held by any active incident."""

        return tuple(
            sorted(
                {
                    round_ref
                    for record in self._records.values()
                    if not record.lifted
                    for round_ref in record.held_rounds
                }
            )
        )

    def is_checkpoint_held(self, checkpoint_ref: str) -> bool:
        """Return whether an active incident directly holds the checkpoint."""

        return checkpoint_ref in self._active_checkpoints()

    def is_round_held(self, round_ref: str) -> bool:
        """Return whether an active incident holds the round."""

        return round_ref in self.held_rounds()

    def require_promotable(
        self, *, checkpoint_ref: str, round_ref: str | None = None
    ) -> None:
        """Refuse promotion of held checkpoints and their later descendants.

        Descendants trained from a held checkpoint after the incident was
        recorded are found through the lineage port at promotion time.

        Args:
            checkpoint_ref: Checkpoint about to be promoted.
            round_ref: Optional round that produced the checkpoint.

        Raises:
            KeyCompromiseHoldError: ``checkpoint_held``, ``round_held``,
                ``lineage_unresolved`` when holds are active and the
                checkpoint is unknown, or ``lineage_port_unavailable``.
        """

        if not _is_reference(checkpoint_ref) or not (
            round_ref is None or _is_reference(round_ref)
        ):
            raise _error(KeyCompromiseErrorCode.INVALID_REFERENCE)
        held = self._active_checkpoints()
        if checkpoint_ref in held:
            raise _error(KeyCompromiseErrorCode.CHECKPOINT_HELD)
        if round_ref is not None and self.is_round_held(round_ref):
            raise _error(KeyCompromiseErrorCode.ROUND_HELD)
        if not held and not self.held_rounds():
            return
        nodes = self._lineage_nodes()
        node = nodes.get(checkpoint_ref)
        if node is None:
            raise _error(KeyCompromiseErrorCode.LINEAGE_UNRESOLVED)
        if node.round_ref is not None and self.is_round_held(node.round_ref):
            raise _error(KeyCompromiseErrorCode.ROUND_HELD)
        if _ancestors(nodes, node.parent_refs) & held:
            raise _error(KeyCompromiseErrorCode.CHECKPOINT_HELD)

    # Lifting -----------------------------------------------------------------

    def lift_hold(
        self, incident_digest: str, grant: HoldLiftGrant | None
    ) -> KeyCompromiseIncidentReport:
        """Lift one incident hold under explicit operator authority.

        Other incidents that hold the same lineage keep it held.

        Args:
            incident_digest: Digest from the incident report.
            grant: Signed operator grant bound to ``incident_digest``.

        Returns:
            The incident report with status ``lifted``.

        Raises:
            KeyCompromiseHoldError: ``operator_authority_required``,
                ``operator_authority_invalid``, ``unknown_incident`` or
                ``audit_unavailable``. The hold stays in place on any error.
        """

        if grant is None:
            raise _error(KeyCompromiseErrorCode.AUTHORITY_REQUIRED)
        if (
            not isinstance(grant, HoldLiftGrant)
            or grant.incident_digest != incident_digest
            or not _verified(self._authority_verifier, grant.signing_bytes(), grant)
        ):
            raise _error(KeyCompromiseErrorCode.AUTHORITY_INVALID)
        record = self._records.get(incident_digest)
        if record is None:
            raise _error(KeyCompromiseErrorCode.UNKNOWN_INCIDENT)
        if not record.lifted:
            self._append(
                KEY_COMPROMISE_LIFTED_EVENT,
                {
                    "schema_version": FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION,
                    "incident_digest": incident_digest,
                    "status": IncidentStatus.LIFTED.value,
                    "grant_digest": "sha256:"
                    + hashlib.sha256(grant.signing_bytes()).hexdigest(),
                },
            )
            record.lifted = True
        return record.current_report()

    def report(self, incident_digest: str) -> KeyCompromiseIncidentReport:
        """Return the current report of a recorded incident.

        Args:
            incident_digest: Digest from the incident report.

        Raises:
            KeyCompromiseHoldError: ``unknown_incident``.
        """

        record = self._records.get(incident_digest)
        if record is None:
            raise _error(KeyCompromiseErrorCode.UNKNOWN_INCIDENT)
        return record.current_report()

    # Persistence -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return persisted hold state without key references or commitments."""

        return {
            "schema_version": FEDERATED_KEY_COMPROMISE_STATE_SCHEMA_VERSION,
            "incidents": [
                self._records[digest].to_dict() for digest in sorted(self._records)
            ],
        }

    def to_json(self) -> str:
        """Return deterministic JSON for :meth:`to_dict`."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._audit_sink is None:
            return
        try:
            self._audit_sink.append(event_type, dict(payload))
        except Exception:
            raise _error(KeyCompromiseErrorCode.AUDIT_UNAVAILABLE) from None


def promote_round(
    lifecycle: FederatedRoundLifecycle,
    *,
    checkpoint_ref: str,
    round_ref: str | None,
    registry: KeyCompromiseHoldRegistry,
) -> FederatedRoundLifecycle:
    """Promote a round only when no compromised-key hold covers its checkpoint.

    Args:
        lifecycle: Current round lifecycle, normally ``evaluating``.
        checkpoint_ref: Checkpoint the round produced.
        round_ref: Round reference, when known.
        registry: Hold registry consulted before the transition.

    Returns:
        The lifecycle in ``promoted`` state.

    Raises:
        KeyCompromiseHoldError: When the checkpoint or round is held.
        FederatedRoundTransitionError: When promotion is not a legal
            transition from the current state.
    """

    registry.require_promotable(checkpoint_ref=checkpoint_ref, round_ref=round_ref)
    return lifecycle.transition_to(FederatedRoundState.PROMOTED)


def _verified(verifier: SignatureVerifier, message: bytes, signed: Any) -> bool:
    try:
        result = verifier.verify(message, signed.signature)
    except Exception:
        return False
    return result is True


def _descendants(
    nodes: Mapping[str, LineageNode], roots: frozenset[str]
) -> frozenset[str]:
    children: dict[str, list[str]] = {}
    for node in nodes.values():
        for parent in node.parent_refs:
            children.setdefault(parent, []).append(node.checkpoint_ref)
    seen = set(roots)
    stack = list(roots)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return frozenset(seen)


def _ancestors(
    nodes: Mapping[str, LineageNode], parents: Iterable[str]
) -> frozenset[str]:
    seen: set[str] = set()
    stack = list(parents)
    while stack:
        checkpoint = stack.pop()
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        node = nodes.get(checkpoint)
        if node is not None:
            stack.extend(node.parent_refs)
    return frozenset(seen)


def _require_acyclic(nodes: Mapping[str, LineageNode]) -> None:
    pending = {ref: len(node.parent_refs) for ref, node in nodes.items()}
    children: dict[str, list[str]] = {}
    for node in nodes.values():
        for parent in node.parent_refs:
            children.setdefault(parent, []).append(node.checkpoint_ref)
    ready = [ref for ref, count in pending.items() if count == 0]
    visited = 0
    while ready:
        ref = ready.pop()
        visited += 1
        for child in children.get(ref, ()):
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise _error(KeyCompromiseErrorCode.LINEAGE_INVALID)


def _parse_state(state: Any) -> dict[str, _HoldRecord]:
    invalid = KeyCompromiseErrorCode.INVALID_STATE
    if (
        not isinstance(state, Mapping)
        or set(state) != {"schema_version", "incidents"}
        or state["schema_version"] != FEDERATED_KEY_COMPROMISE_STATE_SCHEMA_VERSION
        or not isinstance(state["incidents"], list)
    ):
        raise _error(invalid)
    records: dict[str, _HoldRecord] = {}
    incident_ids: set[str] = set()
    for item in state["incidents"]:
        if not isinstance(item, Mapping) or set(item) != {
            "incident_id",
            "report",
            "held_rounds",
            "held_checkpoints",
            "lifted",
            "audited",
        }:
            raise _error(invalid)
        report = KeyCompromiseIncidentReport._from_dict(item["report"])
        incident_id = item["incident_id"]
        rounds = item["held_rounds"]
        checkpoints = item["held_checkpoints"]
        if (
            type(incident_id) is not str
            or _INCIDENT_ID.fullmatch(incident_id) is None
            or incident_id in incident_ids
            or report.incident_digest in records
            or report.status is IncidentStatus.LIFTED
            or type(item["lifted"]) is not bool
            or type(item["audited"]) is not bool
            or not isinstance(rounds, list)
            or not isinstance(checkpoints, list)
            or not all(_is_reference(value) for value in rounds + checkpoints)
            or len(set(rounds)) != report.held_round_count
            or len(set(checkpoints)) != report.held_checkpoint_count
        ):
            raise _error(invalid)
        incident_ids.add(incident_id)
        records[report.incident_digest] = _HoldRecord(
            incident_id=incident_id,
            report=report,
            held_rounds=frozenset(rounds),
            held_checkpoints=frozenset(checkpoints),
            lifted=item["lifted"],
            audited=item["audited"],
        )
    return records


__all__ = [
    "FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION",
    "FEDERATED_KEY_COMPROMISE_STATE_SCHEMA_VERSION",
    "KEY_COMPROMISE_LIFTED_EVENT",
    "KEY_COMPROMISE_RECORDED_EVENT",
    "AcceptedUpdateCommitment",
    "CheckpointLineagePort",
    "HoldLiftGrant",
    "IncidentStatus",
    "KeyCompromiseErrorCode",
    "KeyCompromiseHoldError",
    "KeyCompromiseHoldRegistry",
    "KeyCompromiseIncident",
    "KeyCompromiseIncidentReport",
    "KeyRevocationPort",
    "LineageNode",
    "RemediationClass",
    "RoundCommitmentSet",
    "SignatureVerifier",
    "SignedKeyCompromiseIncident",
    "UpdateCommitmentPort",
    "promote_round",
]
