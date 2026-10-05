"""Offline origin and provisional-status policy for proposed FHIR R4 writes.

This is a labeling gate, not approval verification or write execution. Receipts
must come from the caller's trusted, action-bound approval verifier.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from openmed.agent.approvals.tokens import ApprovalReceipt

__all__ = [
    "FHIRWriteLabelPolicy",
    "WriteLabelFinding",
    "WriteLabelError",
    "NormalizedFHIRWrite",
    "normalize_proposed_resource",
    "validate_proposed_resource",
]

# Only the three assertion-aware R4 exporters are in this write subset.
_STATUS_SYSTEMS = {
    "Condition": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
    "AllergyIntolerance": (
        "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification"
    ),
}
_PROVISIONAL = {
    "Observation": "preliminary",
    "Condition": "provisional",
    "AllergyIntolerance": "unconfirmed",
}
_STATUS_CODES = {
    "Observation": frozenset(
        {
            "registered",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "cancelled",
            "entered-in-error",
        }
    ),
    "Condition": frozenset(
        {
            "unconfirmed",
            "provisional",
            "differential",
            "confirmed",
            "refuted",
            "entered-in-error",
        }
    ),
    "AllergyIntolerance": frozenset(
        {"unconfirmed", "confirmed", "refuted", "entered-in-error"}
    ),
}
_FINAL_CODES = frozenset({"final", "amended", "corrected", "confirmed"})


@dataclass(frozen=True, repr=False)
class FHIRWriteLabelPolicy:
    """Configured origin labels and roles allowed to retain final assertions.

    No role attests by default. An attesting role can retain a supplied final
    status only during validation with a consumed approval receipt. Normalizing
    for preview always downgrades final assertions, even for clinicians.
    """

    security_system: str = "http://terminology.hl7.org/CodeSystem/v3-ObservationValue"
    security_code: str = "AIAST"
    tag_system: str = "https://openmed.ai/fhir/CodeSystem/write-origin"
    tag_code: str = "machine-generated"
    attesting_roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for value in (
            self.security_system,
            self.security_code,
            self.tag_system,
            self.tag_code,
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("invalid_write_label_policy")
        if type(self.attesting_roles) is not frozenset or any(
            type(role) is not str or not role or role != role.strip()
            for role in self.attesting_roles
        ):
            raise ValueError("invalid_write_label_policy")

    def __repr__(self) -> str:
        """Hide deployment configuration in incidental diagnostics."""
        return "FHIRWriteLabelPolicy(<configured>)"


@dataclass(frozen=True)
class WriteLabelFinding:
    """Controlled resource type, element path and reason code only."""

    resource_type: str
    path: str
    code: str

    def to_dict(self) -> dict[str, str]:
        """Return a value-free finding for review or audit."""
        return {
            "resource_type": self.resource_type,
            "path": self.path,
            "code": self.code,
        }


class WriteLabelError(ValueError):
    """Fail-closed rejection containing only controlled findings."""

    def __init__(self, findings: tuple[WriteLabelFinding, ...]) -> None:
        self.findings = findings
        super().__init__("fhir_write_label_rejected")


@dataclass(frozen=True)
class NormalizedFHIRWrite:
    """Private normalized payload and separately serializable safe findings.

    The resource still contains clinical data: keep it in the protected data
    path. It is excluded from the result representation, never an audit record.
    """

    resource: dict[str, Any] = field(repr=False)
    findings: tuple[WriteLabelFinding, ...]


def validate_proposed_resource(
    resource: dict[str, Any],
    *,
    policy: FHIRWriteLabelPolicy = FHIRWriteLabelPolicy(),
    approval_receipt: ApprovalReceipt | None = None,
) -> tuple[WriteLabelFinding, ...]:
    """Check labels and status without changing or exposing resource values.

    Args:
        resource: Single R4 Observation, Condition or AllergyIntolerance.
        policy: Required label pairs and explicitly configured attesting roles.
        approval_receipt: Already verified, consumed, action-bound receipt. This
            function checks its role only; it does not authorize execution.

    Returns:
        Controlled findings; an empty tuple passes this labeling gate only.
        Any finding must stop preview or execution.
    """
    resource_type, findings = _structure(resource)
    if findings:
        return tuple(findings)
    if approval_receipt is not None and type(approval_receipt) is not ApprovalReceipt:
        findings.append(_finding(resource_type, "resourceType", "invalid_receipt"))
    meta = resource.get("meta", {})
    findings.extend(_labels(meta, resource_type, policy, require=True))
    status, status_findings = _status(resource, resource_type)
    findings.extend(status_findings)
    if status is None and not status_findings:
        findings.append(
            _finding(resource_type, _status_path(resource_type), "missing_status")
        )
    if status in _FINAL_CODES and not (
        type(approval_receipt) is ApprovalReceipt
        and approval_receipt.reviewer_role in policy.attesting_roles
    ):
        findings.append(
            _finding(resource_type, _status_path(resource_type), "attestation_required")
        )
    return tuple(findings)


def normalize_proposed_resource(
    resource: dict[str, Any],
    *,
    policy: FHIRWriteLabelPolicy = FHIRWriteLabelPolicy(),
) -> NormalizedFHIRWrite:
    """Copy exporter output, label it and downgrade final statuses for preview.

    Args:
        resource: Single resource from the declared R4 exporter subset.
        policy: Exact machine-origin security label and OpenMed tag to apply.

    Returns:
        Protected normalized payload plus value-free change findings. Codes,
        values, subjects, clinicalStatus and references remain unchanged.

    Raises:
        WriteLabelError: For unsupported resources, nested resources, malformed
            statuses/meta, or existing label lists missing the configured pair.
            Unlabeled exporter output may receive new labels; mislabeled output
            cannot be silently repaired. Input is never modified on rejection.
    """
    resource_type, findings = _structure(resource)
    if findings:
        raise WriteLabelError(tuple(findings))
    findings.extend(
        _labels(resource.get("meta", {}), resource_type, policy, require=False)
    )
    status, status_findings = _status(resource, resource_type)
    findings.extend(status_findings)
    if findings:
        raise WriteLabelError(tuple(findings))

    normalized = deepcopy(resource)
    meta = normalized.setdefault("meta", {})
    changes = []
    for name, system, code in _label_pairs(policy):
        if name not in meta:
            meta[name] = [{"system": system, "code": code}]
            changes.append(
                _finding(resource_type, f"meta.{name}", "origin_label_added")
            )
    if status is None or status in _FINAL_CODES:
        replacement = _PROVISIONAL[resource_type]
        path = _status_path(resource_type)
        normalized[path] = (
            replacement
            if resource_type == "Observation"
            else {
                "coding": [
                    {"system": _STATUS_SYSTEMS[resource_type], "code": replacement}
                ]
            }
        )
        changes.append(_finding(resource_type, path, "provisional_status_applied"))
    # A copy is returned only after the exact labeling gate passes.
    remaining = validate_proposed_resource(normalized, policy=policy)
    if remaining:
        raise WriteLabelError(remaining)
    return NormalizedFHIRWrite(normalized, tuple(changes))


def _finding(resource_type: str, path: str, code: str) -> WriteLabelFinding:
    return WriteLabelFinding(resource_type, path, code)


def _structure(resource: Any) -> tuple[str, list[WriteLabelFinding]]:
    if type(resource) is not dict:
        return "Resource", [_finding("Resource", "resourceType", "invalid_resource")]
    resource_type = resource.get("resourceType")
    if type(resource_type) is not str or resource_type not in _PROVISIONAL:
        return "Resource", [
            _finding("Resource", "resourceType", "unsupported_resource_type")
        ]
    # Never pass an embedded, unlabelled proposal through a single-resource gate.
    if "contained" in resource and resource["contained"] != []:
        return resource_type, [
            _finding(resource_type, "contained", "nested_resources_unsupported")
        ]
    return resource_type, []


def _label_pairs(policy: FHIRWriteLabelPolicy) -> tuple[tuple[str, str, str], ...]:
    return (
        ("security", policy.security_system, policy.security_code),
        ("tag", policy.tag_system, policy.tag_code),
    )


def _labels(
    meta: Any, resource_type: str, policy: FHIRWriteLabelPolicy, *, require: bool
) -> list[WriteLabelFinding]:
    if type(meta) is not dict:
        return [_finding(resource_type, "meta", "invalid_meta")]
    findings = []
    for name, system, code in _label_pairs(policy):
        if name not in meta:
            if require:
                findings.append(
                    _finding(resource_type, f"meta.{name}", "missing_origin_label")
                )
            continue
        labels = meta[name]
        if (
            type(labels) is not list
            or not labels
            or any(type(item) is not dict for item in labels)
        ):
            findings.append(
                _finding(resource_type, f"meta.{name}", "invalid_origin_label")
            )
        elif not any(
            item.get("system") == system and item.get("code") == code for item in labels
        ):
            findings.append(
                _finding(resource_type, f"meta.{name}", "origin_label_mismatch")
            )
    return findings


def _status_path(resource_type: str) -> str:
    return "status" if resource_type == "Observation" else "verificationStatus"


def _status(
    resource: dict[str, Any], resource_type: str
) -> tuple[str | None, list[WriteLabelFinding]]:
    path = _status_path(resource_type)
    if path not in resource:
        return None, []
    status = resource[path]
    if resource_type != "Observation":
        coding = status.get("coding") if type(status) is dict else None
        if (
            type(coding) is not list
            or len(coding) != 1
            or type(coding[0]) is not dict
            or coding[0].get("system") != _STATUS_SYSTEMS[resource_type]
        ):
            return None, [_finding(resource_type, path, "invalid_status")]
        status = coding[0].get("code")
    if type(status) is not str or status not in _STATUS_CODES[resource_type]:
        return None, [_finding(resource_type, path, "invalid_status")]
    return status, []
