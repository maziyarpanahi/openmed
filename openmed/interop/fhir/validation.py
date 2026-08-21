"""Deterministic, offline structural checks for the FHIR exchange subset."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .profiles import get_profile
from .versions import SUPPORTED_RESOURCE_TYPES, FHIRVersion, parse_fhir_version

__all__ = [
    "FHIRValidationResult",
    "validate",
    "validate_bundle",
    "validate_document",
    "validate_resource",
    "validation_result",
]


@dataclass(frozen=True)
class _ValidationIssue:
    severity: str
    code: str
    path: str | None
    diagnostics: str


@dataclass(frozen=True)
class FHIRValidationResult:
    """PHI-safe local validation result with a FHIR ``OperationOutcome`` view."""

    issues: tuple[_ValidationIssue, ...]
    version: FHIRVersion
    profile: str | None = None

    @property
    def valid(self) -> bool:
        """Whether no fatal or error issue was found."""

        return not any(issue.severity in {"fatal", "error"} for issue in self.issues)

    @property
    def ok(self) -> bool:
        """Alias for :attr:`valid` used by command-line callers."""

        return self.valid

    def to_operation_outcome(self) -> dict[str, Any]:
        """Render the result as a FHIR R4-compatible ``OperationOutcome``."""

        from openmed.clinical.exporters.fhir.operation_outcome import (
            OperationOutcomeIssue,
            to_operation_outcome,
        )

        return to_operation_outcome(
            OperationOutcomeIssue(
                severity=issue.severity,
                code=issue.code,
                diagnostics=issue.diagnostics,
                expression=issue.path,
            )
            for issue in self.issues
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a small machine-readable summary without resource values."""

        return {
            "valid": self.valid,
            "version": self.version.value,
            "profile": self.profile,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "path": issue.path,
                    "diagnostics": issue.diagnostics,
                }
                for issue in self.issues
            ],
        }


def validate(
    payload: Mapping[str, Any],
    version: FHIRVersion | str = FHIRVersion.R4,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Validate one resource or Bundle using only local structural rules."""

    return validation_result(payload, version, profile=profile).to_operation_outcome()


def validation_result(
    payload: Mapping[str, Any],
    version: FHIRVersion | str = FHIRVersion.R4,
    *,
    profile: str | None = None,
) -> FHIRValidationResult:
    """Return the structured result behind :func:`validate`."""

    resolved_version = parse_fhir_version(version)
    issues: list[_ValidationIssue] = []
    if not isinstance(payload, Mapping):
        issues.append(
            _issue("fatal", "structure", None, "FHIR payload must be an object")
        )
        return FHIRValidationResult(tuple(issues), resolved_version, profile)

    resource_type = payload.get("resourceType")
    if resource_type == "Bundle":
        _validate_bundle(payload, resolved_version, profile, issues)
    elif isinstance(resource_type, str):
        _validate_resource(payload, "", resolved_version, issues)
        _validate_profile_release(profile, resolved_version, issues)
        if profile in {"ips", "clinical-document"}:
            issues.append(
                _issue(
                    "error",
                    "value",
                    "resourceType",
                    "selected profile requires a document Bundle",
                )
            )
    else:
        issues.append(
            _issue("fatal", "required", "resourceType", "resourceType is required")
        )

    return FHIRValidationResult(tuple(issues), resolved_version, profile)


def validate_resource(
    resource: Mapping[str, Any],
    version: FHIRVersion | str = FHIRVersion.R4,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Validate a standalone supported FHIR resource."""

    return validate(resource, version, profile=profile)


def validate_bundle(
    bundle: Mapping[str, Any],
    version: FHIRVersion | str = FHIRVersion.R4,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Validate a FHIR Bundle and its supported entry resources."""

    return validate(bundle, version, profile=profile)


def validate_document(
    bundle: Mapping[str, Any],
    version: FHIRVersion | str = FHIRVersion.R4,
    *,
    profile: str = "clinical-document",
) -> dict[str, Any]:
    """Validate a FHIR clinical document Bundle locally."""

    return validate(bundle, version, profile=profile)


def _validate_bundle(
    bundle: Mapping[str, Any],
    version: FHIRVersion,
    profile: str | None,
    issues: list[_ValidationIssue],
) -> None:
    _validate_resource(bundle, "", version, issues)
    _validate_profile_release(profile, version, issues)

    bundle_type = bundle.get("type")
    entries = bundle.get("entry")
    if not isinstance(bundle_type, str):
        issues.append(
            _issue("error", "required", "Bundle.type", "Bundle.type is required")
        )
    if entries is None:
        entries = []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        issues.append(
            _issue(
                "error", "structure", "Bundle.entry", "Bundle.entry must be an array"
            )
        )
        entries = []

    full_urls: set[str] = set()
    logical_ids: set[str] = set()
    resources: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries):
        entry_path = f"Bundle.entry[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(
                _issue(
                    "error", "structure", entry_path, "Bundle entry must be an object"
                )
            )
            continue
        full_url = entry.get("fullUrl")
        if isinstance(full_url, str):
            if full_url in full_urls:
                issues.append(
                    _issue(
                        "error",
                        "duplicate",
                        f"{entry_path}.fullUrl",
                        "Bundle fullUrl must be unique",
                    )
                )
            full_urls.add(full_url)
        resource = entry.get("resource")
        if not isinstance(resource, Mapping):
            issues.append(
                _issue(
                    "error",
                    "required",
                    f"{entry_path}.resource",
                    "Bundle entry resource is required",
                )
            )
            continue
        resources.append(resource)
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if isinstance(resource_type, str) and isinstance(resource_id, str):
            logical_ids.add(f"{resource_type}/{resource_id}")
        _validate_resource(resource, f"{entry_path}.resource", version, issues)

    if bundle_type == "document":
        if not resources:
            issues.append(
                _issue(
                    "error",
                    "required",
                    "Bundle.entry",
                    "document Bundle must contain a Composition",
                )
            )
        elif resources[0].get("resourceType") != "Composition":
            issues.append(
                _issue(
                    "error",
                    "value",
                    "Bundle.entry[0].resource.resourceType",
                    "document Bundle must begin with Composition",
                )
            )

    _validate_references(bundle, full_urls, logical_ids, issues)
    if profile == "ips":
        _validate_ips(resources, bundle_type, issues)
    elif profile == "ipa":
        _validate_ipa(resources, bundle_type, issues)
    elif profile == "clinical-document":
        _validate_clinical_document(resources, bundle_type, issues)


def _validate_resource(
    resource: Mapping[str, Any],
    path: str,
    version: FHIRVersion,
    issues: list[_ValidationIssue],
) -> None:
    resource_type = resource.get("resourceType")
    resource_path = path or "resource"
    if not isinstance(resource_type, str) or not resource_type:
        issues.append(
            _issue(
                "error",
                "required",
                f"{resource_path}.resourceType",
                "resourceType is required",
            )
        )
        return
    if resource_type not in SUPPORTED_RESOURCE_TYPES:
        issues.append(
            _issue(
                "error",
                "not-supported",
                f"{resource_path}.resourceType",
                "resource type is outside the supported exchange subset",
            )
        )
        return

    # Running the version adapter in place mode gives us one authoritative
    # unknown-field boundary while keeping the validator independent of FHIR
    # package downloads.
    try:
        from .versions import convert_resource

        convert_resource(resource, version, version)
    except (TypeError, ValueError) as exc:
        path_value = getattr(exc, "path", None) or resource_path
        reason = "unsupported resource shape at the selected FHIR boundary"
        if type(exc).__name__ == "UnsupportedFHIRFieldError":
            reason = "resource contains a field outside the supported FHIR subset"
        issues.append(_issue("error", "not-supported", path_value, reason))

    required_by_resource: dict[str, tuple[str, ...]] = {
        "Composition": ("status", "type", "date", "author", "title"),
        "Condition": ("code", "subject"),
        "Observation": ("status", "code"),
        "MedicationStatement": ("status", "subject"),
        "AllergyIntolerance": ("code", "patient"),
        "Procedure": ("status", "subject"),
        "DocumentReference": ("status", "content"),
    }
    for field in required_by_resource.get(resource_type, ()):
        if field not in resource or resource[field] in (None, [], ""):
            issues.append(
                _issue(
                    "error",
                    "required",
                    f"{resource_path}.{field}",
                    "required element is missing",
                )
            )

    if resource_type == "MedicationStatement":
        if version == FHIRVersion.R4:
            medication_fields = {"medicationCodeableConcept", "medicationReference"}
        else:
            medication_fields = {"medication"}
        if not medication_fields & set(resource):
            issues.append(
                _issue(
                    "error",
                    "required",
                    f"{resource_path}.medication",
                    "medication[x] is required",
                )
            )
    if resource_type == "Patient" and version == FHIRVersion.R4:
        if not resource.get("name") and not resource.get("birthDate"):
            # This is an IPS-level expectation, not a core FHIR requirement;
            # keep it informational at the core layer.
            issues.append(
                _issue(
                    "information",
                    "informational",
                    f"{resource_path}",
                    "Patient has no name or birthDate for summary use",
                )
            )


def _validate_profile_release(
    profile: str | None,
    version: FHIRVersion,
    issues: list[_ValidationIssue],
) -> None:
    if profile is None:
        return
    try:
        entry = get_profile(profile)
    except KeyError:
        issues.append(
            _issue(
                "error",
                "not-found",
                "$profile",
                "profile is not in the supported local matrix",
            )
        )
        return
    if entry["fhir_release"] != version.value:
        issues.append(
            _issue(
                "error",
                "conflict",
                "$profile",
                "profile FHIR release does not match the selected release",
            )
        )


def _validate_ips(
    resources: Sequence[Mapping[str, Any]],
    bundle_type: Any,
    issues: list[_ValidationIssue],
) -> None:
    if bundle_type != "document":
        issues.append(
            _issue(
                "error",
                "value",
                "Bundle.type",
                "IPS patient summary must be a document Bundle",
            )
        )
    composition = (
        resources[0]
        if resources and resources[0].get("resourceType") == "Composition"
        else None
    )
    if composition is None:
        return
    if not composition.get("section"):
        issues.append(
            _issue(
                "warning",
                "incomplete",
                "Composition.section",
                "IPS sections are absent from the local summary",
            )
        )
    if not any(resource.get("resourceType") == "Patient" for resource in resources):
        issues.append(
            _issue(
                "error",
                "required",
                "Bundle.entry",
                "IPS patient summary must contain Patient",
            )
        )


def _validate_ipa(
    resources: Sequence[Mapping[str, Any]],
    bundle_type: Any,
    issues: list[_ValidationIssue],
) -> None:
    if bundle_type not in {"searchset", "collection", "document"}:
        issues.append(
            _issue(
                "error",
                "value",
                "Bundle.type",
                "IPA example must be a patient-access Bundle",
            )
        )
    patients = [
        resource for resource in resources if resource.get("resourceType") == "Patient"
    ]
    if not patients:
        issues.append(
            _issue(
                "error", "required", "Bundle.entry", "IPA example must contain Patient"
            )
        )
    elif not patients[0].get("identifier"):
        issues.append(
            _issue(
                "error",
                "required",
                "Bundle.entry[0].resource.identifier",
                "IPA Patient requires an identifier",
            )
        )


def _validate_clinical_document(
    resources: Sequence[Mapping[str, Any]],
    bundle_type: Any,
    issues: list[_ValidationIssue],
) -> None:
    if bundle_type != "document":
        issues.append(
            _issue(
                "error",
                "value",
                "Bundle.type",
                "clinical document must be a document Bundle",
            )
        )
    if not resources or resources[0].get("resourceType") != "Composition":
        return
    narrative = resources[0].get("text")
    if not isinstance(narrative, Mapping) or not isinstance(narrative.get("div"), str):
        issues.append(
            _issue(
                "error",
                "required",
                "Bundle.entry[0].resource.text",
                "clinical document Composition requires narrative",
            )
        )


def _validate_references(
    bundle: Mapping[str, Any],
    full_urls: set[str],
    logical_ids: set[str],
    issues: list[_ValidationIssue],
) -> None:
    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            reference = node.get("reference")
            if (
                isinstance(reference, str)
                and reference.startswith("urn:uuid:")
                and reference not in full_urls
            ):
                issues.append(
                    _issue(
                        "error",
                        "not-found",
                        f"{path}.reference",
                        "internal Bundle reference has no matching fullUrl",
                    )
                )
            if isinstance(reference, str) and reference in logical_ids:
                # Literal references are valid before Bundle assembly; they are
                # intentionally not rewritten by the validator.
                pass
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(bundle, "Bundle")


def _issue(
    severity: str,
    code: str,
    path: str | None,
    diagnostics: str,
) -> _ValidationIssue:
    return _ValidationIssue(
        severity=severity,
        code=code,
        path=path,
        diagnostics=diagnostics,
    )
