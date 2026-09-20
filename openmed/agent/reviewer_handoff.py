"""Strict, metadata-only reviewer handoff packets for agent workflows.

A valid packet requests human review. It does not approve, authorize, execute,
or finalize a clinical action. Submitted values are never included in errors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Final

from .artifact_reference import (
    ArtifactReference,
    ArtifactReferenceError,
    validate_artifact_references,
)
from .correlation import CorrelationIdError, RunId
from .identifiers import GovernanceIdError, WorkflowId
from .outcomes import OutcomeClass, allowed_reason_codes

REVIEWER_HANDOFF_SCHEMA_VERSION: Final = "openmed.agent.reviewer_handoff.v1"
MAX_HANDOFF_EVIDENCE_REFERENCES: Final = 64

_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "workflow_id",
        "reason_code",
        "requested_decision",
        "evidence_references",
        "issued_at",
        "expires_at",
    }
)
_REQUIRED_FIELDS = _ALLOWED_FIELDS - {"schema_version"}
_ORDERED_FIELDS = (
    "schema_version",
    "run_id",
    "workflow_id",
    "reason_code",
    "requested_decision",
    "evidence_references",
    "issued_at",
    "expires_at",
)
_HANDOFF_REASON_CODES = allowed_reason_codes(
    OutcomeClass.ABSTAINED
) | allowed_reason_codes(OutcomeClass.REVIEW_REQUIRED)


class RequestedDecision(str, Enum):
    """Closed set of decisions that a handoff may ask a human to make."""

    CONFIRM_ABSTENTION = "confirm_abstention"
    REVIEW_EVIDENCE = "review_evidence"
    RESOLVE_EVIDENCE_CONFLICT = "resolve_evidence_conflict"
    ASSESS_SAFETY = "assess_safety"
    DECIDE_NEXT_STEP = "decide_next_step"


class ReviewerHandoffError(ValueError):
    """Raised when a reviewer handoff packet fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ReviewerHandoffPacket:
    """Immutable request for a human reviewer decision.

    Args:
        run_id: Opaque identifier for the agent run.
        workflow_id: Canonical identifier for the governed workflow.
        reason_code: Stable abstention or review-required outcome reason.
        requested_decision: Bounded decision requested from a human reviewer.
        evidence_references: Ordered, content-free evidence references.
        issued_at: Whole-second UTC time at which the packet was issued.
        expires_at: Whole-second UTC time after which the packet is invalid.
        schema_version: Stable serialization schema version.
        validation_time: Optional UTC clock value for deterministic validation.
            Runtime callers should omit it to use the current time.
    """

    run_id: RunId
    workflow_id: WorkflowId
    reason_code: str
    requested_decision: RequestedDecision
    evidence_references: tuple[ArtifactReference, ...]
    issued_at: datetime
    expires_at: datetime
    schema_version: str = REVIEWER_HANDOFF_SCHEMA_VERSION
    validation_time: InitVar[datetime | None] = None

    def __post_init__(self, validation_time: datetime | None) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != REVIEWER_HANDOFF_SCHEMA_VERSION
        ):
            raise ReviewerHandoffError("invalid_schema_version", "schema_version")
        if type(self.run_id) is not RunId:
            raise ReviewerHandoffError("wrong_identifier_kind", "run_id")
        if type(self.workflow_id) is not WorkflowId:
            raise ReviewerHandoffError("wrong_identifier_kind", "workflow_id")
        if (
            type(self.reason_code) is not str
            or self.reason_code not in _HANDOFF_REASON_CODES
        ):
            raise ReviewerHandoffError("unknown_reason", "reason_code")
        if not isinstance(self.requested_decision, RequestedDecision):
            raise ReviewerHandoffError("unknown_decision", "requested_decision")
        if type(self.evidence_references) is not tuple:
            raise ReviewerHandoffError(
                "invalid_evidence_collection", "evidence_references"
            )
        if len(self.evidence_references) > MAX_HANDOFF_EVIDENCE_REFERENCES:
            raise ReviewerHandoffError("too_many_items", "evidence_references")
        try:
            validated_references = validate_artifact_references(
                self.evidence_references
            )
        except ArtifactReferenceError as error:
            code = (
                "duplicate_evidence_reference"
                if error.code == "duplicate_artifact_id"
                else "invalid_evidence_reference"
            )
            raise ReviewerHandoffError(code, "evidence_references") from None
        if validated_references != self.evidence_references:
            raise ReviewerHandoffError(
                "invalid_evidence_collection", "evidence_references"
            )

        _validate_utc_timestamp(self.issued_at, "issued_at")
        _validate_utc_timestamp(self.expires_at, "expires_at")
        if self.expires_at <= self.issued_at:
            raise ReviewerHandoffError("invalid_expiry", "expires_at")

        effective_time = (
            datetime.now(UTC) if validation_time is None else validation_time
        )
        _validate_utc_timestamp(effective_time, "validation_time", allow_subsecond=True)
        if self.expires_at <= effective_time:
            raise ReviewerHandoffError("expired", "expires_at")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> "ReviewerHandoffPacket":
        """Build a packet from a strict metadata-only mapping.

        Args:
            data: Mapping containing only packet fields.
            now: Optional UTC validation clock for deterministic tests.

        Returns:
            A validated reviewer handoff packet.

        Raises:
            ReviewerHandoffError: If the packet is malformed, expired, or unsafe.
        """

        if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
            raise ReviewerHandoffError("not_a_mapping")
        try:
            fields = set(data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ReviewerHandoffError("not_a_mapping") from None
        if fields - _ALLOWED_FIELDS:
            raise ReviewerHandoffError("unknown_field")
        if _REQUIRED_FIELDS - fields:
            raise ReviewerHandoffError("missing_field")

        try:
            values = {field_name: data[field_name] for field_name in fields}
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ReviewerHandoffError("unreadable_mapping") from None

        return cls(
            schema_version=values.get(
                "schema_version", REVIEWER_HANDOFF_SCHEMA_VERSION
            ),
            run_id=_parse_run_id(values["run_id"]),
            workflow_id=_parse_workflow_id(values["workflow_id"]),
            reason_code=values["reason_code"],
            requested_decision=_parse_requested_decision(values["requested_decision"]),
            evidence_references=_parse_evidence_references(
                values["evidence_references"]
            ),
            issued_at=_parse_timestamp(values["issued_at"], "issued_at"),
            expires_at=_parse_timestamp(values["expires_at"], "expires_at"),
            validation_time=now,
        )

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
        *,
        now: datetime | None = None,
    ) -> "ReviewerHandoffPacket":
        """Build a packet from a strict JSON object."""

        try:
            data = json.loads(payload, object_pairs_hook=_strict_json_object)
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
            pass
        else:
            try:
                return cls.from_dict(data, now=now)
            except ReviewerHandoffError:
                raise
        raise ReviewerHandoffError("malformed_json") from None

    @property
    def requires_human_review(self) -> bool:
        """Return ``True`` because every valid packet is a review request."""

        return True

    @property
    def authorizes_clinical_action(self) -> bool:
        """Return ``False`` because a handoff packet grants no authority."""

        return False

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only packet fields."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id.serialize(),
            "workflow_id": self.workflow_id.serialize(),
            "reason_code": self.reason_code,
            "requested_decision": self.requested_decision.value,
            "evidence_references": [
                reference.to_dict() for reference in self.evidence_references
            ],
            "issued_at": _format_timestamp(self.issued_at),
            "expires_at": _format_timestamp(self.expires_at),
        }
        return {field_name: values[field_name] for field_name in _ORDERED_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with deterministic key ordering."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def allowed_handoff_reason_codes() -> frozenset[str]:
    """Return the closed reason-code set accepted by handoff packets."""

    return _HANDOFF_REASON_CODES


def _parse_run_id(value: Any) -> RunId:
    try:
        return RunId.parse(value)
    except CorrelationIdError:
        raise ReviewerHandoffError("invalid_identifier", "run_id") from None


def _parse_workflow_id(value: Any) -> WorkflowId:
    try:
        return WorkflowId.parse(value)
    except GovernanceIdError:
        raise ReviewerHandoffError("invalid_identifier", "workflow_id") from None


def _parse_requested_decision(value: Any) -> RequestedDecision:
    if type(value) is str:
        try:
            return RequestedDecision(value)
        except ValueError:
            pass
    raise ReviewerHandoffError("unknown_decision", "requested_decision")


def _parse_evidence_references(value: Any) -> tuple[ArtifactReference, ...]:
    if type(value) not in (list, tuple):
        raise ReviewerHandoffError("invalid_evidence_collection", "evidence_references")
    if len(value) > MAX_HANDOFF_EVIDENCE_REFERENCES:
        raise ReviewerHandoffError("too_many_items", "evidence_references")

    references: list[ArtifactReference] = []
    for item in value:
        try:
            references.append(ArtifactReference.from_dict(item))
        except ArtifactReferenceError:
            raise ReviewerHandoffError(
                "invalid_evidence_reference", "evidence_references"
            ) from None
    return tuple(references)


def _parse_timestamp(value: Any, field_name: str) -> datetime:
    if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
        raise ReviewerHandoffError("invalid_timestamp", field_name)
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise ReviewerHandoffError("invalid_timestamp", field_name) from None


def _validate_utc_timestamp(
    value: Any,
    field_name: str,
    *,
    allow_subsecond: bool = False,
) -> None:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
        or (not allow_subsecond and value.microsecond != 0)
    ):
        raise ReviewerHandoffError("invalid_timestamp", field_name)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewerHandoffError("duplicate_field")
        result[key] = value
    return result


__all__ = [
    "MAX_HANDOFF_EVIDENCE_REFERENCES",
    "REVIEWER_HANDOFF_SCHEMA_VERSION",
    "RequestedDecision",
    "ReviewerHandoffError",
    "ReviewerHandoffPacket",
    "allowed_handoff_reason_codes",
]
