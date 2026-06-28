"""Emit FHIR R4 provenance resources from OpenMed audit reports.

The signed audit report is OpenMed's deterministic record of a
de-identification run. This module projects that record into FHIR-native
``Provenance`` and ``AuditEvent`` resources so downstream Bundles can carry
auditable metadata without embedding PHI or raw span text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from openmed.core.audit import AuditReport

__all__ = ["to_audit_event", "to_provenance"]

_OPENMED_SOFTWARE_SYSTEM = "https://openmed.dev/fhir/sid/software"
_OPENMED_REPRO_HASH_SYSTEM = "https://openmed.dev/fhir/sid/reproducibility-hash"
_OPENMED_ACTIVITY_SYSTEM = "https://openmed.dev/fhir/CodeSystem/audit-activity"
_PROVENANCE_PARTICIPANT_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/provenance-participant-type"
)
_AUDIT_EVENT_TYPE_SYSTEM = "https://openmed.dev/fhir/CodeSystem/audit-event-type"
_AUDIT_SOURCE_TYPE_SYSTEM = "http://terminology.hl7.org/CodeSystem/security-source-type"


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
