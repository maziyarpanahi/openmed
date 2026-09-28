"""Guarded, value-free state transitions for clinical review packets."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.clinical.review_state_machine import (
    DEFAULT_REVIEW_TRANSITIONS,
    REVIEW_STATES,
    REVIEW_TRANSITION_ADVISORY,
    REVIEW_TRANSITIONS_SCHEMA_VERSION,
    ReviewPolicyRule,
    ReviewState,
    ReviewStateMachine,
    ReviewTransition,
    ReviewTransitionError,
    ReviewTransitionEvent,
    ReviewTransitionPolicy,
    ReviewTransitionReport,
    ReviewTransitionRequest,
    ReviewTransitionValidationError,
    TransitionValidationError,
    compute_provenance_fingerprint,
    make_event_id,
    make_opaque_event_id,
    opaque_event_id,
    provenance_fingerprint,
    validate_review_history,
    validate_review_transition,
    validate_transition,
)
from openmed.structured.store import (
    JobMetadata,
    JobMetadataStore,
    StoreResult,
    StoreState,
)

CLINICAL_REVIEW_PACKET_SCHEMA_VERSION = "1.1.0"
CLINICAL_REVIEW_COMPATIBILITY_POLICY = "same_major"
CLINICAL_REVIEW_STATES = frozenset(
    {"queued", "in_review", "approved", "rejected", "expired", "reopened"}
)
CLINICAL_REVIEW_PRIORITIES = frozenset({"low", "normal", "high", "critical"})

_ALLOWED_TRANSITIONS = {
    "queued": frozenset({"in_review", "expired"}),
    "in_review": frozenset({"approved", "rejected", "expired"}),
    "approved": frozenset({"reopened"}),
    "rejected": frozenset({"reopened"}),
    "expired": frozenset({"reopened"}),
    "reopened": frozenset({"in_review", "expired"}),
}
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)


class ClinicalReviewError(ValueError):
    """Raised when a review packet violates its public contract."""


@dataclass(frozen=True, slots=True)
class ClinicalReviewTransition:
    """One immutable, identity-free review state transition."""

    event_id: str
    packet_id: str
    from_state: str
    to_state: str
    occurred_at: str
    policy_id: str
    policy_version: str
    provenance_fingerprint: str
    reason_code: str
    schema_version: str = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION
    compatibility_policy: str = CLINICAL_REVIEW_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        for name in ("event_id", "packet_id"):
            if _OPAQUE_ID_RE.fullmatch(getattr(self, name)) is None:
                raise ClinicalReviewError(f"{name} must be an opaque identifier")
        if self.from_state not in CLINICAL_REVIEW_STATES:
            raise ClinicalReviewError("unsupported prior review state")
        if self.to_state not in _ALLOWED_TRANSITIONS[self.from_state]:
            raise ClinicalReviewError("review transition is not permitted")
        if _TIMESTAMP_RE.fullmatch(self.occurred_at) is None:
            raise ClinicalReviewError("occurred_at must be an ISO-8601 timestamp")
        if _CONTROLLED_RE.fullmatch(self.policy_id) is None:
            raise ClinicalReviewError("policy_id must be controlled")
        if _VERSION_RE.fullmatch(self.policy_version) is None:
            raise ClinicalReviewError("policy_version must be semantic")
        if _DIGEST_RE.fullmatch(self.provenance_fingerprint) is None:
            raise ClinicalReviewError("provenance_fingerprint must be a digest")
        if _CONTROLLED_RE.fullmatch(self.reason_code) is None:
            raise ClinicalReviewError("reason_code must be controlled")
        if self.schema_version != CLINICAL_REVIEW_PACKET_SCHEMA_VERSION:
            raise ClinicalReviewError("unsupported transition schema version")
        if self.compatibility_policy != CLINICAL_REVIEW_COMPATIBILITY_POLICY:
            raise ClinicalReviewError("unsupported compatibility policy")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic transition metadata without reviewer identity."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "event_id": self.event_id,
            "from_state": self.from_state,
            "occurred_at": self.occurred_at,
            "packet_id": self.packet_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "provenance_fingerprint": self.provenance_fingerprint,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
            "to_state": self.to_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClinicalReviewTransition":
        """Parse one serialized transition."""

        try:
            return cls(
                event_id=str(payload["event_id"]),
                packet_id=str(payload["packet_id"]),
                from_state=str(payload["from_state"]),
                to_state=str(payload["to_state"]),
                occurred_at=str(payload["occurred_at"]),
                policy_id=str(payload["policy_id"]),
                policy_version=str(payload["policy_version"]),
                provenance_fingerprint=str(payload["provenance_fingerprint"]),
                reason_code=str(payload["reason_code"]),
                schema_version=str(payload["schema_version"]),
                compatibility_policy=str(payload["compatibility_policy"]),
            )
        except (KeyError, TypeError, ValueError):
            raise ClinicalReviewError("invalid clinical review transition") from None


@dataclass(frozen=True, slots=True)
class ClinicalReviewPacket:
    """Versioned review packet containing identifiers and provenance only."""

    packet_id: str
    conflict_id: str
    fact_ids: tuple[str, ...]
    state: str
    priority: str
    created_at: str
    expires_at: str | None
    policy_id: str
    policy_version: str
    provenance_fingerprint: str
    transitions: tuple[ClinicalReviewTransition, ...] = ()
    schema_version: str = CLINICAL_REVIEW_PACKET_SCHEMA_VERSION
    compatibility_policy: str = CLINICAL_REVIEW_COMPATIBILITY_POLICY
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in ("packet_id", "conflict_id"):
            if _OPAQUE_ID_RE.fullmatch(getattr(self, name)) is None:
                raise ClinicalReviewError(f"{name} must be an opaque identifier")
        fact_ids = tuple(sorted(set(self.fact_ids)))
        if len(fact_ids) < 2 or any(
            _OPAQUE_ID_RE.fullmatch(item) is None for item in fact_ids
        ):
            raise ClinicalReviewError("review packet requires opaque fact identifiers")
        if self.state not in CLINICAL_REVIEW_STATES:
            raise ClinicalReviewError("unsupported review state")
        if self.priority not in CLINICAL_REVIEW_PRIORITIES:
            raise ClinicalReviewError("unsupported review priority")
        if _TIMESTAMP_RE.fullmatch(self.created_at) is None:
            raise ClinicalReviewError("created_at must be an ISO-8601 timestamp")
        if self.expires_at is not None:
            if _TIMESTAMP_RE.fullmatch(self.expires_at) is None:
                raise ClinicalReviewError("expires_at must be an ISO-8601 timestamp")
            if _parse_timestamp(self.expires_at) <= _parse_timestamp(self.created_at):
                raise ClinicalReviewError("expires_at must follow created_at")
        if _CONTROLLED_RE.fullmatch(self.policy_id) is None:
            raise ClinicalReviewError("policy_id must be controlled")
        if _VERSION_RE.fullmatch(self.policy_version) is None:
            raise ClinicalReviewError("policy_version must be semantic")
        if _DIGEST_RE.fullmatch(self.provenance_fingerprint) is None:
            raise ClinicalReviewError("provenance_fingerprint must be a digest")
        transitions = tuple(self.transitions)
        expected = "queued"
        previous_time = _parse_timestamp(self.created_at)
        seen: set[str] = set()
        for transition in transitions:
            if transition.packet_id != self.packet_id:
                raise ClinicalReviewError("transition packet does not match")
            if transition.event_id in seen:
                raise ClinicalReviewError(
                    "review transition identifiers must be unique"
                )
            if transition.from_state != expected:
                raise ClinicalReviewError("review transition chain is discontinuous")
            occurred = _parse_timestamp(transition.occurred_at)
            if occurred < previous_time:
                raise ClinicalReviewError("review transitions must be chronological")
            expected = transition.to_state
            previous_time = occurred
            seen.add(transition.event_id)
        if self.state != expected:
            raise ClinicalReviewError("packet state does not match transition history")
        if self.schema_version != CLINICAL_REVIEW_PACKET_SCHEMA_VERSION:
            raise ClinicalReviewError("unsupported packet schema version")
        if self.compatibility_policy != CLINICAL_REVIEW_COMPATIBILITY_POLICY:
            raise ClinicalReviewError("unsupported compatibility policy")
        if not isinstance(self.extensions, Mapping):
            raise ClinicalReviewError("extensions must be a mapping")
        serialized_extensions = json_safe_mapping(self.extensions)
        forbidden = {
            "case_content",
            "credential",
            "note",
            "patient",
            "phi",
            "raw",
            "reviewer",
            "source_text",
            "text",
        }
        if forbidden.intersection(key.casefold() for key in serialized_extensions):
            raise ClinicalReviewError("review packet extensions contain unsafe keys")
        object.__setattr__(self, "fact_ids", fact_ids)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "extensions", serialized_extensions)

    @property
    def transition_ids(self) -> tuple[str, ...]:
        """Return append-only transition identifiers in event order."""

        return tuple(item.event_id for item in self.transitions)

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free review packet."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "conflict_id": self.conflict_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "extensions": dict(self.extensions),
            "fact_ids": list(self.fact_ids),
            "packet_id": self.packet_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "priority": self.priority,
            "provenance_fingerprint": self.provenance_fingerprint,
            "schema_version": self.schema_version,
            "state": self.state,
            "transition_ids": list(self.transition_ids),
            "transitions": [item.to_dict() for item in self.transitions],
        }

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClinicalReviewPacket":
        """Parse a current-version packet."""

        try:
            packet = cls(
                packet_id=str(payload["packet_id"]),
                conflict_id=str(payload["conflict_id"]),
                fact_ids=tuple(payload["fact_ids"]),
                state=str(payload["state"]),
                priority=str(payload["priority"]),
                created_at=str(payload["created_at"]),
                expires_at=(
                    None
                    if payload.get("expires_at") is None
                    else str(payload["expires_at"])
                ),
                policy_id=str(payload["policy_id"]),
                policy_version=str(payload["policy_version"]),
                provenance_fingerprint=str(payload["provenance_fingerprint"]),
                transitions=tuple(
                    ClinicalReviewTransition.from_dict(item)
                    for item in payload.get("transitions", ())
                ),
                schema_version=str(payload["schema_version"]),
                compatibility_policy=str(payload["compatibility_policy"]),
                extensions=payload.get("extensions", {}),
            )
            if tuple(payload.get("transition_ids", ())) != packet.transition_ids:
                raise ClinicalReviewError("transition_ids do not match transitions")
            return packet
        except ClinicalReviewError:
            raise
        except (KeyError, TypeError, ValueError):
            raise ClinicalReviewError("invalid clinical review packet") from None


def transition_review_packet(
    packet: ClinicalReviewPacket,
    *,
    to_state: str,
    occurred_at: str,
    policy_id: str,
    policy_version: str,
    provenance_fingerprint: str,
    reason_code: str,
) -> StoreResult[ClinicalReviewPacket]:
    """Append a valid state transition or return a typed conflict."""

    if to_state not in CLINICAL_REVIEW_STATES:
        return StoreResult.outcome(StoreState.UNSUPPORTED, "review_state_unsupported")
    if to_state not in _ALLOWED_TRANSITIONS[packet.state]:
        return StoreResult.outcome(StoreState.CONFLICT, "review_transition_invalid")
    identity = canonical_digest(
        {
            "from_state": packet.state,
            "occurred_at": occurred_at,
            "packet_id": packet.packet_id,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "provenance_fingerprint": provenance_fingerprint,
            "reason_code": reason_code,
            "to_state": to_state,
        }
    ).removeprefix("sha256:")
    try:
        event = ClinicalReviewTransition(
            event_id=f"reviewevent_{identity[:32]}",
            packet_id=packet.packet_id,
            from_state=packet.state,
            to_state=to_state,
            occurred_at=occurred_at,
            policy_id=policy_id,
            policy_version=policy_version,
            provenance_fingerprint=provenance_fingerprint,
            reason_code=reason_code,
        )
        updated = replace(
            packet,
            state=to_state,
            transitions=(*packet.transitions, event),
        )
    except ClinicalReviewError:
        return StoreResult.outcome(StoreState.FAILURE, "review_transition_invalid")
    return StoreResult.success(updated, created=True)


def persist_review_packet(
    packet: ClinicalReviewPacket,
    store: JobMetadataStore,
    *,
    recorded_at: str,
) -> StoreResult[ClinicalReviewPacket]:
    """Append a packet version through the common Journey metadata store."""

    try:
        payload = packet.to_dict()
        written = store.put_job(
            JobMetadata(
                job_id=packet.packet_id,
                state=packet.state,
                recorded_at=recorded_at,
                metadata={
                    "packet": payload,
                    "packet_digest": canonical_digest(payload),
                },
            ),
            committed_at=recorded_at,
        )
    except (TypeError, ValueError, RuntimeError):
        return StoreResult.outcome(StoreState.FAILURE, "review_packet_write_failed")
    if not written.ok:
        return StoreResult.outcome(
            written.state, written.code or "review_packet_write_failed"
        )
    return StoreResult.success(
        packet,
        created=written.created,
        revision=written.revision,
    )


def load_clinical_review_schema(name: str) -> dict[str, Any]:
    """Load a bundled review packet, transition, or queue summary schema."""

    supported = {
        "clinical_review_packet",
        "clinical_review_queue_summary",
        "clinical_review_transition",
    }
    if name not in supported:
        raise KeyError(f"unknown clinical review schema: {name!r}")
    path = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "schemas"
        / "json"
        / f"{name}.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip a mapping through canonical JSON for immutability and safety."""

    import json

    try:
        result = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError):
        raise ClinicalReviewError("extensions must contain JSON-safe values") from None
    if not isinstance(result, dict):
        raise ClinicalReviewError("extensions must be a mapping")
    return result


__all__ = [
    "CLINICAL_REVIEW_COMPATIBILITY_POLICY",
    "CLINICAL_REVIEW_PACKET_SCHEMA_VERSION",
    "CLINICAL_REVIEW_PRIORITIES",
    "CLINICAL_REVIEW_STATES",
    "ClinicalReviewError",
    "ClinicalReviewPacket",
    "ClinicalReviewTransition",
    "load_clinical_review_schema",
    "persist_review_packet",
    "transition_review_packet",
    "DEFAULT_REVIEW_TRANSITIONS",
    "REVIEW_STATES",
    "REVIEW_TRANSITION_ADVISORY",
    "REVIEW_TRANSITIONS_SCHEMA_VERSION",
    "ReviewPolicyRule",
    "ReviewState",
    "ReviewStateMachine",
    "ReviewTransition",
    "ReviewTransitionError",
    "ReviewTransitionEvent",
    "ReviewTransitionPolicy",
    "ReviewTransitionReport",
    "ReviewTransitionRequest",
    "ReviewTransitionValidationError",
    "TransitionValidationError",
    "compute_provenance_fingerprint",
    "make_event_id",
    "make_opaque_event_id",
    "opaque_event_id",
    "provenance_fingerprint",
    "validate_review_history",
    "validate_review_transition",
    "validate_transition",
]
