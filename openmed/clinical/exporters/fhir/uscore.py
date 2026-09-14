"""Offline US Core STU9 checks for selected OpenMed FHIR R4 exports.

The checker intentionally implements a compact, OpenMed-authored subset of US
Core 9.0.0. It validates the four resource profiles used by OpenMed exports,
runs the bundled base-R4 validator first, and never downloads implementation
guide or terminology content. Findings contain only fixed messages and element
paths so clinical values are not copied into logs or audit artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

from ._validation_primitives import _extract_codes, _occurrence_groups
from .validate import ValidationFinding, validate_resource

__all__ = ["ConformanceResult", "US_CORE_VERSION", "check_us_core"]

US_CORE_VERSION = "9.0.0"


@dataclass(frozen=True)
class ConformanceResult:
    """Immutable findings from base R4 and US Core profile checks.

    Attributes:
        profile: Resolved US Core canonical URL, when a profile was selected.
        resource_type: FHIR resource type reported by the input, when valid.
        errors: Findings that prevent a conformance claim.
        warnings: Advisory findings, including absent must-support elements.
    """

    profile: str | None
    resource_type: str | None
    errors: tuple[ValidationFinding, ...] = ()
    warnings: tuple[ValidationFinding, ...] = ()

    @property
    def findings(self) -> tuple[ValidationFinding, ...]:
        """Return every finding, with errors before warnings."""

        return self.errors + self.warnings

    @property
    def issues(self) -> tuple[ValidationFinding, ...]:
        """Return findings for the shared OperationOutcome adapter."""

        return self.findings

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when neither validation layer found an error."""

        return not self.errors

    @property
    def valid(self) -> bool:
        """Alias for :attr:`is_valid`."""

        return self.is_valid


def check_us_core(
    resource: Mapping[str, Any],
    profile: str | None = None,
) -> ConformanceResult:
    """Check one resource against base R4 and a bundled US Core profile subset.

    Supported profiles cover Condition problems/health concerns, Condition
    encounter diagnoses, laboratory Observation, MedicationRequest, and
    AllergyIntolerance. A profile can be supplied as its canonical URL, a
    versioned canonical (``url|9.0.0``), or its StructureDefinition id. When it
    is omitted, a supported canonical in ``meta.profile`` takes precedence,
    followed by the resource type's default profile.

    Args:
        resource: FHIR R4 resource mapping. Malformed resources are returned as
            structured findings rather than raising.
        profile: Optional US Core profile canonical URL or id.

    Returns:
        A :class:`ConformanceResult` containing sanitized base and profile
        findings. No network access is performed.
    """

    base_result = validate_resource(resource)
    resource_type = _resource_type(resource)
    selected, selection_finding = _select_profile(resource, resource_type, profile)

    findings = list(base_result.findings)
    if selection_finding is not None:
        findings.append(selection_finding)
    elif selected is not None:
        findings.extend(_check_profile(resource, selected))

    unique = _deduplicate(findings)
    return ConformanceResult(
        profile=selected["url"] if selected is not None else None,
        resource_type=resource_type,
        errors=tuple(item for item in unique if item.severity == "error"),
        warnings=tuple(item for item in unique if item.severity == "warning"),
    )


def _resource_type(resource: Any) -> str | None:
    if not isinstance(resource, Mapping):
        return None
    resource_type = resource.get("resourceType")
    return resource_type if isinstance(resource_type, str) and resource_type else None


def _select_profile(
    resource: Any,
    resource_type: str | None,
    requested: str | None,
) -> tuple[Mapping[str, Any] | None, ValidationFinding | None]:
    if resource_type is None:
        return None, None

    candidate = (
        requested
        or _declared_profile(resource)
        or _definitions()["defaults"].get(resource_type)
    )
    root = resource_type
    if candidate is None:
        return None, _warning(
            f"{root}.resourceType",
            "Resource type has no bundled US Core profile subset.",
            "not-supported",
        )

    canonical, version = _split_version(candidate)
    if version is not None and version != US_CORE_VERSION:
        return None, _error(
            f"{root}.meta.profile",
            "US Core profile version is not supported by the bundled subset.",
            "not-supported",
        )

    selected = _profile_aliases().get(_normalise_canonical(canonical))
    if selected is None:
        return None, _error(
            f"{root}.meta.profile",
            "US Core profile is not supported by the bundled subset.",
            "not-supported",
        )
    if selected["resourceType"] != resource_type:
        return None, _error(
            f"{root}.meta.profile",
            "US Core profile does not match the resource type.",
            "value",
        )
    return selected, None


def _declared_profile(resource: Any) -> str | None:
    if not isinstance(resource, Mapping):
        return None
    meta = resource.get("meta")
    if not isinstance(meta, Mapping):
        return None
    declared = meta.get("profile")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        return None
    for item in declared:
        if not isinstance(item, str):
            continue
        canonical, _ = _split_version(item)
        if _normalise_canonical(canonical) in _profile_aliases():
            return item
    return None


def _split_version(candidate: str) -> tuple[str, str | None]:
    canonical, separator, version = candidate.strip().partition("|")
    return canonical.rstrip("/"), version if separator else None


def _normalise_canonical(candidate: str) -> str:
    normalised = candidate.strip().rstrip("/")
    for prefix in ("https://hl7.org/", "https://www.hl7.org/", "http://www.hl7.org/"):
        if normalised.startswith(prefix):
            return "http://hl7.org/" + normalised[len(prefix) :]
    return normalised


@lru_cache(maxsize=1)
def _profile_aliases() -> Mapping[str, Mapping[str, Any]]:
    aliases: dict[str, Mapping[str, Any]] = {}
    for profile in _definitions()["profiles"]:
        aliases[_normalise_canonical(profile["url"])] = profile
        aliases[profile["id"]] = profile
        aliases[f"StructureDefinition/{profile['id']}"] = profile
    return aliases


def _check_profile(
    resource: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[ValidationFinding]:
    root = profile["resourceType"]
    findings: list[ValidationFinding] = []
    for element in profile["elements"]:
        findings.extend(_check_element(resource, root, element))
    for group in profile.get("mustSupportAnyOf", ()):
        findings.extend(_check_must_support_group(resource, root, group))
    return findings


def _check_element(
    resource: Mapping[str, Any],
    root: str,
    element: Mapping[str, Any],
) -> list[ValidationFinding]:
    path = element["path"]
    groups = _occurrence_groups(resource, path.split("."), root)
    location = f"{root}.{element.get('location', path.replace('[x]', ''))}"
    findings: list[ValidationFinding] = []

    # A nested must-support child is only actionable when its parent exists.
    if not groups:
        return findings

    for group in groups:
        occurrences = tuple(group.occurrences)
        selector = element.get("selector")
        if selector is not None:
            matching = tuple(
                occurrence
                for occurrence in occurrences
                if _matches_selector(occurrence.value, selector)
            )
        else:
            matching = occurrences

        minimum = int(element.get("min", 0))
        maximum_value = element.get("max", "*")
        maximum = None if maximum_value == "*" else int(maximum_value)

        if len(matching) < minimum:
            code = (
                "code-invalid" if occurrences and selector is not None else "required"
            )
            message = (
                "Element does not satisfy the required US Core binding."
                if code == "code-invalid"
                else "Required US Core profile element is missing or empty."
            )
            findings.append(_error(location, message, code))
        if maximum is not None and len(matching) > maximum:
            findings.append(
                _error(
                    location,
                    "Maximum US Core profile element cardinality is exceeded.",
                    "structure",
                )
            )

        binding_name = element.get("binding")
        if binding_name is not None and selector is None:
            binding = _definitions()["bindings"][binding_name]
            for occurrence in occurrences:
                if not _matches_binding(occurrence.value, binding):
                    findings.append(
                        _error(
                            occurrence.expression,
                            "Element does not satisfy the required US Core binding.",
                            "code-invalid",
                        )
                    )

        if (
            element.get("mustSupport")
            and not matching
            and not any(
                item.severity == "error" and item.location == location
                for item in findings
            )
        ):
            findings.append(
                _warning(
                    location,
                    "US Core must-support element is absent.",
                    "incomplete",
                )
            )
    return findings


def _check_must_support_group(
    resource: Mapping[str, Any],
    root: str,
    group: Mapping[str, Any],
) -> list[ValidationFinding]:
    for path in group["paths"]:
        groups = _occurrence_groups(resource, path.split("."), root)
        if any(item.occurrences for item in groups):
            return []
    return [
        _warning(
            f"{root}.{group['location']}",
            "US Core must-support choice is absent.",
            "incomplete",
        )
    ]


def _matches_selector(value: Any, selector: Mapping[str, Any]) -> bool:
    expected_url = selector.get("url")
    if expected_url is not None:
        return isinstance(value, Mapping) and value.get("url") == expected_url
    binding_name = selector.get("binding")
    if binding_name is not None:
        return _matches_binding(value, _definitions()["bindings"][binding_name])
    return False


def _matches_binding(value: Any, binding: Mapping[str, Any]) -> bool:
    systems = binding["systems"]
    for system, code in _extract_codes(value):
        if not system and isinstance(value, str):
            return any(code in allowed for allowed in systems.values())
        allowed = systems.get(system)
        if allowed is not None and code in allowed:
            return True
    return False


def _deduplicate(findings: Sequence[ValidationFinding]) -> list[ValidationFinding]:
    unique: list[ValidationFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        key = (finding.severity, finding.code, finding.location)
        if key not in seen:
            unique.append(finding)
            seen.add(key)
    return unique


def _error(location: str, message: str, code: str) -> ValidationFinding:
    return ValidationFinding("error", location, message, code)


def _warning(location: str, message: str, code: str) -> ValidationFinding:
    return ValidationFinding("warning", location, message, code)


@lru_cache(maxsize=1)
def _definitions() -> Mapping[str, Any]:
    definition_path = (
        resources.files(__package__)
        .joinpath("definitions")
        .joinpath("us_core_constraints.json")
    )
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    if (
        payload.get("schemaVersion") != 1
        or payload.get("fhirVersion") != "4.0.1"
        or payload.get("usCoreVersion") != US_CORE_VERSION
    ):
        raise RuntimeError("bundled US Core constraint table is incompatible")
    return payload
