"""Emit FHIR R4 provenance and audit resources from local technical evidence.

The signed audit report is OpenMed's deterministic record of a
de-identification run. This module projects that record into FHIR-native
``Provenance`` and ``AuditEvent`` resources so downstream Bundles can carry
auditable metadata without embedding PHI or raw span text.
Governed writes use a separate strict metadata projection with an injected
clock, controlled outcomes and opaque references, without submission or storage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from openmed.core.audit import AuditReport

__all__ = [
    "GovernedWriteAction",
    "GovernedWriteAuditAttempt",
    "GovernedWriteAuditError",
    "GovernedWriteOutcome",
    "GovernedWriteReviewerRole",
    "to_audit_event",
    "to_governed_write_audit_event",
    "to_provenance",
]

_OPENMED_SOFTWARE_SYSTEM = "https://openmed.dev/fhir/sid/software"
_OPENMED_REPRO_HASH_SYSTEM = "https://openmed.dev/fhir/sid/reproducibility-hash"
_OPENMED_ACTIVITY_SYSTEM = "https://openmed.dev/fhir/CodeSystem/audit-activity"
_PROVENANCE_PARTICIPANT_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/provenance-participant-type"
)
_AUDIT_EVENT_TYPE_SYSTEM = "https://openmed.dev/fhir/CodeSystem/audit-event-type"
_AUDIT_SOURCE_TYPE_SYSTEM = "http://terminology.hl7.org/CodeSystem/security-source-type"
_WRITE_OUTCOME_SYSTEM = "https://openmed.dev/fhir/CodeSystem/governed-write-outcome"
_WRITE_ROLE_SYSTEM = "https://openmed.dev/fhir/CodeSystem/governed-reviewer-role"
_OPAQUE_REFERENCE_RE = re.compile(
    r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_RECEIPT_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class GovernedWriteAction(str, Enum):
    """Intended FHIR write action represented by an audit attempt."""

    CREATE = "create"
    UPDATE = "update"


class GovernedWriteOutcome(str, Enum):
    """Closed outcomes asserted by the trusted application controller."""

    SUCCESS = "success"
    SERVER_REJECTED = "server_rejected"
    COMMIT_UNKNOWN = "commit_unknown"
    POLICY_DENIED = "policy_denied"
    ADMISSION_REFUSED = "admission_refused"


class GovernedWriteReviewerRole(str, Enum):
    """Controlled reviewer role categories, without a human identity.

    The application maps a verified policy role to an explicit category.
    This category does not replace that policy role or its verification;
    the approval receipt digest links back to protected local review evidence.
    """

    REVIEWER = "reviewer"
    CLINICAL = "clinical_reviewer"
    PRIVACY = "privacy_reviewer"
    OPERATIONS = "operations_reviewer"
    SECURITY = "security_reviewer"


class GovernedWriteAuditError(ValueError):
    """Controlled projection error whose message contains no input values.

    Args:
        code: One of the fixed projection error codes.
    """

    def __init__(self, code: str) -> None:
        allowed = {
            "invalid_action",
            "invalid_outcome",
            "invalid_software_agent_ref",
            "invalid_target_refs",
            "invalid_reviewer_role",
            "invalid_receipt_digest",
            "incomplete_review_evidence",
            "incomplete_dispatch_evidence",
            "invalid_attempt",
            "invalid_clock",
            "clock_failed",
        }
        self.code = code if type(code) is str and code in allowed else "invalid_attempt"
        super().__init__(self.code)


def _write_enum(value: Any, enum: type[Enum], error: str) -> Any:
    if isinstance(value, enum):
        return value
    if type(value) is str and value in {member.value for member in enum}:
        return enum(value)
    raise GovernedWriteAuditError(error)


def _opaque_write_reference(value: Any) -> bool:
    return (
        type(value) is str
        and _OPAQUE_REFERENCE_RE.fullmatch(value) is not None
        and UUID(value.removeprefix("urn:uuid:")).version == 4
    )


@dataclass(frozen=True, slots=True)
class GovernedWriteAuditAttempt:
    """Value-free write metadata supplied after controller outcome classification.

    Args:
        action: Intended create/update action, including for pre-dispatch refusals.
        outcome: Controlled controller outcome; unknown commit never means absent.
        software_agent_ref: Canonical UUIDv4 URN for the initiating software Device.
        target_refs: At most 128 opaque UUIDv4 URNs for protected targets/proposals.
            They are sorted and deduplicated. Direct clinical IDs/URLs are rejected.
        reviewer_role: Optional controlled category mapped from a verified policy role.
        approval_receipt_digest: Optional verified receipt's SHA-256 digest. Role and
            digest must occur together; this projection does not verify either.

    Dispatch outcomes require review evidence and a target/proposal reference.
    Policy/admission refusals may precede review and data materialization.
    UUID syntax does not prove provenance, unlinkability or reference resolution;
    the application owns opaque mappings and their disclosure/access policy.
    """

    action: GovernedWriteAction
    outcome: GovernedWriteOutcome
    software_agent_ref: str
    target_refs: tuple[str, ...] = ()
    reviewer_role: GovernedWriteReviewerRole | None = None
    approval_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        action = _write_enum(self.action, GovernedWriteAction, "invalid_action")
        outcome = _write_enum(self.outcome, GovernedWriteOutcome, "invalid_outcome")
        if not _opaque_write_reference(self.software_agent_ref):
            raise GovernedWriteAuditError("invalid_software_agent_ref")
        if type(self.target_refs) not in {tuple, list} or len(self.target_refs) > 128:
            raise GovernedWriteAuditError("invalid_target_refs")
        if any(not _opaque_write_reference(ref) for ref in self.target_refs):
            raise GovernedWriteAuditError("invalid_target_refs")
        refs = tuple(sorted(set(self.target_refs)))
        role = self.reviewer_role
        if role is not None:
            role = _write_enum(role, GovernedWriteReviewerRole, "invalid_reviewer_role")
        digest = self.approval_receipt_digest
        if digest is not None and (
            type(digest) is not str or _RECEIPT_DIGEST_RE.fullmatch(digest) is None
        ):
            raise GovernedWriteAuditError("invalid_receipt_digest")
        if (role is None) != (digest is None):
            raise GovernedWriteAuditError("incomplete_review_evidence")
        if outcome in {
            GovernedWriteOutcome.SUCCESS,
            GovernedWriteOutcome.SERVER_REJECTED,
            GovernedWriteOutcome.COMMIT_UNKNOWN,
        } and (not refs or digest is None):
            raise GovernedWriteAuditError("incomplete_dispatch_evidence")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "target_refs", refs)
        object.__setattr__(self, "reviewer_role", role)


def _write_recorded(clock: Callable[[], datetime]) -> str:
    if not callable(clock):
        raise GovernedWriteAuditError("invalid_clock")
    value: Any = None
    failed = False
    try:
        value = clock()
    except Exception:
        failed = True
    if failed:
        raise GovernedWriteAuditError("clock_failed")
    if type(value) is not datetime or value.tzinfo is None:
        raise GovernedWriteAuditError("invalid_clock")
    normalized: str | None = None
    try:
        if value.utcoffset() is not None:
            normalized = (
                value.astimezone(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
    except Exception:
        pass
    if normalized is None:
        raise GovernedWriteAuditError("invalid_clock")
    return normalized


def to_governed_write_audit_event(
    attempt: GovernedWriteAuditAttempt,
    *,
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    """Project a classified governed write attempt into a local FHIR R4 AuditEvent.

    Args:
        attempt: Strict value-free metadata from trusted application classification.
            It grants no authority and does not authenticate review or commit evidence.
        clock: Injected clock returning an aware datetime; called once. Retain the
            recorded instant when reproducing an already-observed event.

    Returns:
        A fresh deterministic resource containing codes, an instant, opaque
        references and an optional receipt digest. No server contact or storage
        occurs. Create/update map to C/U; success maps to 0; rejection and
        refusals to 4; unknown commit to 8 for failure to confirm the attempt.
        The explicit commit_unknown subtype preserves target-effect uncertainty:
        outcome 8 is not proof that the clinical write failed or is absent.

    Raises:
        GovernedWriteAuditError: With a fixed code for invalid metadata or clock.
    """
    if type(attempt) is not GovernedWriteAuditAttempt:
        raise GovernedWriteAuditError("invalid_attempt")
    recorded = _write_recorded(clock)
    software = {"reference": attempt.software_agent_ref, "type": "Device"}
    outcome_codes = {
        GovernedWriteOutcome.SUCCESS: "0",
        GovernedWriteOutcome.SERVER_REJECTED: "4",
        GovernedWriteOutcome.COMMIT_UNKNOWN: "8",
        GovernedWriteOutcome.POLICY_DENIED: "4",
        GovernedWriteOutcome.ADMISSION_REFUSED: "4",
    }
    resource: dict[str, Any] = {
        "resourceType": "AuditEvent",
        "type": {"system": _AUDIT_EVENT_TYPE_SYSTEM, "code": "governed-write-attempt"},
        "subtype": [{"system": _WRITE_OUTCOME_SYSTEM, "code": attempt.outcome.value}],
        "action": "C" if attempt.action is GovernedWriteAction.CREATE else "U",
        "recorded": recorded,
        "outcome": outcome_codes[attempt.outcome],
        "agent": [
            {
                "who": dict(software),
                "requestor": True,
            }
        ],
        "source": {
            "observer": dict(software),
        },
    }
    if attempt.reviewer_role is not None:
        resource["agent"].append(
            {
                "role": [
                    {
                        "coding": [
                            {
                                "system": _WRITE_ROLE_SYSTEM,
                                "code": attempt.reviewer_role.value,
                            }
                        ]
                    }
                ],
                "requestor": False,
            }
        )
    entities: list[dict[str, Any]] = [
        {"what": {"reference": ref}} for ref in attempt.target_refs
    ]
    if attempt.approval_receipt_digest is not None:
        entities.append(
            {
                "detail": [
                    {
                        "type": "openmed.approval_receipt_digest",
                        "valueString": attempt.approval_receipt_digest,
                    }
                ]
            }
        )
    if entities:
        resource["entity"] = entities
    canonical = json.dumps(
        resource, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    resource["id"] = (
        "openmed-write-audit-"
        + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:40]
    )
    return resource


def to_provenance(
    audit_report: AuditReport | Mapping[str, Any],
    target_refs: Sequence[str | Mapping[str, Any]],
) -> dict[str, Any]:
    """Translate an OpenMed audit report into an R4 ``Provenance`` resource.

    Args:
        audit_report: Signed OpenMed audit report, or its dictionary form.
        target_refs: References to Bundle resources produced from the
            de-identified content. Each item may be a reference string such as
            ``"Observation/obs1"`` or a mapping containing ``reference``.

    Returns:
        A FHIR R4 ``Provenance`` mapping carrying the OpenMed software agent,
        target references, activity, recorded instant, and reproducibility hash.

    Raises:
        ValueError: If no target references are supplied.
    """

    report = _coerce_report(audit_report)
    targets = [_target_reference(ref) for ref in target_refs]
    if not targets:
        raise ValueError("at least one target reference is required")

    return {
        "resourceType": "Provenance",
        "id": f"openmed-provenance-{_id_suffix(report.repro_hash)}",
        "target": targets,
        "recorded": _recorded_instant(),
        "activity": {
            "coding": [
                {
                    "system": _OPENMED_ACTIVITY_SYSTEM,
                    "code": "de-identify",
                    "display": "De-identify",
                },
                {
                    "system": _OPENMED_ACTIVITY_SYSTEM,
                    "code": "transform",
                    "display": "Transform",
                },
            ],
            "text": "de-identify/transform",
        },
        "agent": [
            {
                "type": {
                    "coding": [
                        {
                            "system": _PROVENANCE_PARTICIPANT_SYSTEM,
                            "code": "author",
                            "display": "Author",
                        }
                    ]
                },
                "role": [_openmed_role()],
                "who": _software_reference(report),
            }
        ],
        "entity": [_audit_report_entity(report)],
    }


def to_audit_event(
    audit_report: AuditReport | Mapping[str, Any],
) -> dict[str, Any]:
    """Translate an OpenMed audit report into an R4 ``AuditEvent`` resource.

    The emitted resource describes the de-identification activity and outcome.
    Span and risk details are summarized as hashes, labels, counts, and numeric
    values only; raw span text, context windows, and surrogates are not copied.

    Args:
        audit_report: Signed OpenMed audit report, or its dictionary form.

    Returns:
        A FHIR R4 ``AuditEvent`` mapping.
    """

    report = _coerce_report(audit_report)
    outcome, outcome_desc = _outcome(report)

    return {
        "resourceType": "AuditEvent",
        "id": f"openmed-auditevent-{_id_suffix(report.repro_hash)}",
        "type": {
            "system": _AUDIT_EVENT_TYPE_SYSTEM,
            "code": "de-identification",
            "display": "De-identification",
        },
        "subtype": [
            {
                "system": _OPENMED_ACTIVITY_SYSTEM,
                "code": "de-identify",
                "display": "De-identify",
            },
            {
                "system": _OPENMED_ACTIVITY_SYSTEM,
                "code": "transform",
                "display": "Transform",
            },
        ],
        "action": "E",
        "recorded": _recorded_instant(),
        "outcome": outcome,
        "outcomeDesc": outcome_desc,
        "agent": [
            {
                "type": _openmed_role(),
                "who": _software_reference(report),
                "requestor": False,
            }
        ],
        "source": {
            "observer": _software_reference(report),
            "type": [
                {
                    "system": _AUDIT_SOURCE_TYPE_SYSTEM,
                    "code": "9",
                    "display": "Other",
                }
            ],
        },
        "entity": [_audit_event_entity(report)],
    }


def _coerce_report(audit_report: AuditReport | Mapping[str, Any]) -> AuditReport:
    if isinstance(audit_report, AuditReport):
        return audit_report
    if isinstance(audit_report, Mapping):
        return AuditReport.from_dict(audit_report)
    raise TypeError("audit_report must be an AuditReport or mapping")


def _recorded_instant() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _target_reference(ref: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(ref, str):
        if not ref:
            raise ValueError("target reference strings must be non-empty")
        return {"reference": ref}
    if isinstance(ref, Mapping):
        reference = ref.get("reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError("target reference mappings must contain a reference")
        result = {"reference": reference}
        resource_type = ref.get("type")
        if isinstance(resource_type, str) and resource_type:
            result["type"] = resource_type
        return result
    raise TypeError("target references must be strings or mappings")


def _software_reference(report: AuditReport) -> dict[str, Any]:
    display = "openmed"
    if report.openmed_version:
        display = f"{display} {report.openmed_version}"
    return {
        "type": "Device",
        "identifier": {"system": _OPENMED_SOFTWARE_SYSTEM, "value": "openmed"},
        "display": display,
    }


def _openmed_role() -> dict[str, Any]:
    return {
        "coding": [
            {
                "system": _OPENMED_ACTIVITY_SYSTEM,
                "code": "de-identification-software",
                "display": "De-identification software",
            }
        ],
        "text": "de-identification software",
    }


def _repro_identifier(report: AuditReport) -> dict[str, str]:
    return {
        "system": _OPENMED_REPRO_HASH_SYSTEM,
        "value": report.repro_hash,
    }


def _audit_report_entity(report: AuditReport) -> dict[str, Any]:
    return {
        "role": "source",
        "what": {
            "identifier": _repro_identifier(report),
            "display": "OpenMed signed audit report",
        },
    }


def _audit_event_entity(report: AuditReport) -> dict[str, Any]:
    details = [
        _detail("openmed.repro_hash", report.repro_hash),
        _detail("openmed.input_hash", report.input_hash),
        _detail("openmed.deidentified_text_hash", report.deidentified_text_hash),
        _detail("openmed.manifest_hash", report.manifest_hash),
        _detail("openmed.policy", report.policy),
        _detail("openmed.document_length", report.document_length),
        _detail("openmed.span_count", len(report.spans)),
    ]
    details.extend(_span_summary_details(report))
    details.extend(_residual_risk_details(report))

    signature = report.signature
    if signature is not None:
        details.extend(
            [
                _detail("openmed.signature.algorithm", signature.algorithm),
                _detail("openmed.signature.key_id", signature.key_id),
            ]
        )

    return {
        "what": {
            "identifier": _repro_identifier(report),
            "display": "OpenMed signed audit report",
        },
        "detail": details,
    }


def _span_summary_details(report: AuditReport) -> list[dict[str, str]]:
    labels = sorted(
        {
            span.canonical_label or span.label
            for span in report.spans
            if span.canonical_label or span.label
        }
    )
    text_hashes = sorted({span.text_hash for span in report.spans if span.text_hash})
    details: list[dict[str, str]] = []
    if labels:
        details.append(_detail("openmed.span_labels", ",".join(labels)))
    if text_hashes:
        details.append(_detail("openmed.span_text_hashes", ",".join(text_hashes)))
    return details


def _residual_risk_details(report: AuditReport) -> list[dict[str, str]]:
    risk = report.residual_risk
    details: list[dict[str, str]] = []
    for key in ("projected_leakage", "risk_report_record_score"):
        value = risk.get(key)
        if _is_scalar_metric(value):
            details.append(_detail(f"openmed.residual_risk.{key}", value))

    risk_report = risk.get("risk_report")
    if isinstance(risk_report, Mapping):
        for key in ("leakage_rate", "reid_rate", "k_min"):
            value = risk_report.get(key)
            if _is_scalar_metric(value):
                details.append(
                    _detail(f"openmed.residual_risk.risk_report.{key}", value)
                )
        for key in ("singleton_records", "quasi_identifiers"):
            value = risk_report.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                details.append(
                    _detail(
                        f"openmed.residual_risk.risk_report.{key}_count", len(value)
                    )
                )
    return details


def _outcome(report: AuditReport) -> tuple[str, str]:
    if report.repro_hash_matches():
        return "0", "De-identification completed; audit reproducibility hash matches."
    return "8", "De-identification audit report failed reproducibility hash check."


def _detail(detail_type: str, value: Any) -> dict[str, str]:
    return {"type": detail_type, "valueString": str(value)}


def _is_scalar_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and _looks_numeric(value)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _id_suffix(repro_hash: str) -> str:
    tail = repro_hash.split(":", 1)[-1]
    suffix = re.sub(r"[^A-Za-z0-9.-]", "-", tail).strip("-.")
    return (suffix or "unknown")[:16]
