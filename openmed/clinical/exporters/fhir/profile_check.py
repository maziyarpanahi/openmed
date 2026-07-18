"""Offline profile checks for FHIR Bundles produced by OpenMed.

The checker intentionally implements a small, predictable subset of FHIR R4
profile validation.  It reads ``StructureDefinition`` and ``ValueSet``
resources from a local npm-package-style IG snapshot and evaluates profiles
explicitly declared in ``Resource.meta.profile``.  It does not fetch packages,
contact terminology servers, execute FHIRPath invariants, or attempt to replace
a complete FHIR validator.

Supported constraints are minimum/maximum cardinality, ``fixed[x]`` values,
locally enumerable bindings, and required ``identifier``/``category`` slices
whose discriminators are expressed with ``fixed[x]`` or ``pattern[x]`` values.
Unsupported constraints produce informational ``OperationOutcome`` issues.
Diagnostics never include values read from checked resources.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from .operation_outcome import OperationOutcomeIssue, to_operation_outcome

__all__ = ["check_bundle"]


@dataclass(frozen=True)
class _Finding:
    severity: str
    code: str
    diagnostics: str
    expression: str | None = None
    identity: tuple[str, ...] | None = None

    def to_issue(self) -> OperationOutcomeIssue:
        return OperationOutcomeIssue(
            severity=self.severity,
            code=self.code,
            diagnostics=self.diagnostics,
            expression=self.expression,
        )


@dataclass(frozen=True)
class _ValueSet:
    codes: frozenset[tuple[str, str]]
    complete: bool


@dataclass(frozen=True)
class _Package:
    profiles: Mapping[str, Mapping[str, Any]]
    value_sets: Mapping[str, _ValueSet]
    findings: tuple[_Finding, ...]


@dataclass(frozen=True)
class _Occurrence:
    value: Any
    expression: str


@dataclass(frozen=True)
class _OccurrenceGroup:
    expression: str
    occurrences: tuple[_Occurrence, ...]


def check_bundle(
    bundle: Mapping[str, Any],
    ig_dir: str | PathLike[str],
    *,
    original_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check declared profiles in a FHIR Bundle against a local IG snapshot.

    Args:
        bundle: The Bundle to check. The input is never mutated.
        ig_dir: A local IG directory. It may be the npm package root or a
            directory containing the conventional ``package/`` directory.
        original_bundle: Optional pre-de-identification Bundle. When supplied,
            violations in ``bundle`` are classified as introduced by
            de-identification or pre-existing.

    Returns:
        A FHIR R4 ``OperationOutcome``. Errors describe supported conformance
        failures, while unsupported constraints are informational. Diagnostics
        include structural details only and never quote checked values.

    Raises:
        TypeError: If either Bundle is not mapping-shaped.
        ValueError: If either input is not a FHIR Bundle.
        FileNotFoundError: If ``ig_dir`` is not a directory.
    """

    _require_bundle(bundle, "bundle")
    if original_bundle is not None:
        _require_bundle(original_bundle, "original_bundle")

    package = _load_package(ig_dir)
    findings = list(package.findings)
    current = _check_bundle(bundle, package)

    if original_bundle is None:
        findings.extend(current)
    else:
        original = _check_bundle(original_bundle, package)
        original_identities = {
            finding.identity for finding in original if finding.identity is not None
        }
        findings.extend(
            _classify_deidentification(finding, original_identities)
            for finding in current
        )

    return to_operation_outcome(finding.to_issue() for finding in findings)


def _require_bundle(bundle: Any, label: str) -> None:
    if not isinstance(bundle, Mapping):
        raise TypeError(f"{label} must be a FHIR Bundle mapping")
    if bundle.get("resourceType") != "Bundle":
        raise ValueError(f"{label} resourceType must be 'Bundle'")


def _classify_deidentification(
    finding: _Finding,
    original_identities: set[tuple[str, ...]],
) -> _Finding:
    if finding.identity is None:
        return finding
    if finding.identity in original_identities:
        prefix = "Pre-existing profile violation"
    else:
        prefix = "De-identification introduced profile violation"
    return _Finding(
        severity=finding.severity,
        code=finding.code,
        diagnostics=f"{prefix}: {finding.diagnostics}",
        expression=finding.expression,
        identity=finding.identity,
    )


def _load_package(ig_dir: str | PathLike[str]) -> _Package:
    root = Path(ig_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"IG snapshot directory does not exist: {root}")
    package_dir = root / "package" if (root / "package").is_dir() else root

    resources: list[Mapping[str, Any]] = []
    findings: list[_Finding] = []
    for path in sorted(package_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            findings.append(
                _Finding(
                    severity="information",
                    code="processing",
                    diagnostics="Skipped an unreadable IG package JSON resource.",
                )
            )
            continue
        resources.extend(_iter_package_resources(payload))

    profiles: dict[str, Mapping[str, Any]] = {}
    raw_value_sets: dict[str, Mapping[str, Any]] = {}
    for resource in resources:
        canonical = _canonical(resource.get("url"))
        if not canonical:
            continue
        if resource.get("resourceType") == "StructureDefinition":
            profiles[canonical] = resource
        elif resource.get("resourceType") == "ValueSet":
            raw_value_sets[canonical] = resource

    value_sets = {
        canonical: _read_value_set(resource)
        for canonical, resource in raw_value_sets.items()
    }
    if not profiles:
        findings.append(
            _Finding(
                severity="information",
                code="not-found",
                diagnostics=(
                    "No StructureDefinition resources were found in the local "
                    "IG snapshot."
                ),
            )
        )
    return _Package(
        profiles=profiles,
        value_sets=value_sets,
        findings=tuple(findings),
    )


def _iter_package_resources(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    if payload.get("resourceType") == "Bundle":
        resources: list[Mapping[str, Any]] = []
        entries = payload.get("entry")
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            for entry in entries:
                if isinstance(entry, Mapping) and isinstance(
                    entry.get("resource"), Mapping
                ):
                    resources.append(entry["resource"])
        return resources
    if isinstance(payload.get("resourceType"), str):
        return [payload]
    return []


def _read_value_set(resource: Mapping[str, Any]) -> _ValueSet:
    expansion = resource.get("expansion")
    if isinstance(expansion, Mapping) and isinstance(expansion.get("contains"), list):
        codes: set[tuple[str, str]] = set()
        concept_count, readable = _collect_expansion_codes(
            expansion["contains"],
            codes,
            inherited_system="",
        )
        return _ValueSet(
            codes=frozenset(codes),
            complete=readable and _expansion_is_complete(expansion, concept_count),
        )

    compose = resource.get("compose")
    if not isinstance(compose, Mapping):
        return _ValueSet(codes=frozenset(), complete=False)
    if compose.get("exclude"):
        return _ValueSet(codes=frozenset(), complete=False)

    includes = compose.get("include")
    if not isinstance(includes, list):
        return _ValueSet(codes=frozenset(), complete=False)

    codes = set()
    complete = True
    for include in includes:
        if not isinstance(include, Mapping):
            complete = False
            continue
        concepts = include.get("concept")
        if (
            include.get("filter")
            or include.get("valueSet")
            or not isinstance(concepts, list)
            or not concepts
        ):
            complete = False
            continue
        system = include.get("system")
        if not isinstance(system, str) or not system:
            complete = False
            continue
        for concept in concepts:
            if isinstance(concept, Mapping) and isinstance(concept.get("code"), str):
                codes.add((system, concept["code"]))
            else:
                complete = False
    return _ValueSet(codes=frozenset(codes), complete=complete)


def _collect_expansion_codes(
    contains: Sequence[Any],
    codes: set[tuple[str, str]],
    *,
    inherited_system: str,
) -> tuple[int, bool]:
    concept_count = 0
    readable = True
    for item in contains:
        if not isinstance(item, Mapping):
            readable = False
            continue
        system = item.get("system")
        if system is not None and not isinstance(system, str):
            readable = False
        system_value = system if isinstance(system, str) else inherited_system
        code = item.get("code")
        if isinstance(code, str):
            concept_count += 1
            if not system_value:
                readable = False
            elif item.get("abstract") is not True:
                codes.add((system_value, code))
        elif code is not None:
            readable = False
        children = item.get("contains")
        if isinstance(children, list):
            child_count, children_readable = _collect_expansion_codes(
                children,
                codes,
                inherited_system=system_value,
            )
            concept_count += child_count
            readable = readable and children_readable
        elif children is not None:
            readable = False
    return concept_count, readable


def _expansion_is_complete(
    expansion: Mapping[str, Any],
    concept_count: int,
) -> bool:
    offset = expansion.get("offset")
    if offset is not None and (type(offset) is not int or offset != 0):
        return False

    total = expansion.get("total")
    if total is not None and (
        type(total) is not int or total < 0 or total != concept_count
    ):
        return False

    parameters = expansion.get("parameter")
    if not isinstance(parameters, list):
        return True
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        name = parameter.get("name")
        if name == "offset" and parameter.get("valueInteger") not in (None, 0):
            return False
        if name == "count" and total is None:
            # Without ``total`` a paged expansion cannot prove that this is the
            # final page, so local membership checks must remain non-blocking.
            return False
    return True


def _canonical(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().split("|", 1)[0]


def _check_bundle(bundle: Mapping[str, Any], package: _Package) -> list[_Finding]:
    entries = bundle.get("entry")
    if entries is None:
        return []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return [
            _Finding(
                severity="error",
                code="structure",
                diagnostics="Bundle.entry must be an array.",
                expression="Bundle.entry",
                identity=("bundle", "entry", "structure"),
            )
        ]

    findings: list[_Finding] = []
    for index, entry in enumerate(entries):
        entry_expression = f"Bundle.entry[{index}].resource"
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("resource"), Mapping
        ):
            findings.append(
                _Finding(
                    severity="error",
                    code="structure",
                    diagnostics="Bundle entry does not contain a FHIR resource.",
                    expression=entry_expression,
                    identity=("bundle", str(index), "resource", "structure"),
                )
            )
            continue
        findings.extend(
            _check_resource(entry["resource"], entry_expression, index, package)
        )
    return findings


def _check_resource(
    resource: Mapping[str, Any],
    root_expression: str,
    entry_index: int,
    package: _Package,
) -> list[_Finding]:
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type:
        return [
            _Finding(
                severity="error",
                code="structure",
                diagnostics="Bundle entry resource is missing resourceType.",
                expression=f"{root_expression}.resourceType",
                identity=("resource", str(entry_index), "resourceType"),
            )
        ]

    profiles, profile_findings = _declared_profiles(resource, root_expression)
    findings = list(profile_findings)
    for profile_index, declared in enumerate(profiles):
        canonical = _canonical(declared)
        profile_expression = f"{root_expression}.meta.profile[{profile_index}]"
        if canonical is None:
            findings.append(
                _Finding(
                    severity="information",
                    code="not-supported",
                    diagnostics="Ignored a malformed meta.profile declaration.",
                    expression=profile_expression,
                )
            )
            continue
        profile = package.profiles.get(canonical)
        if profile is None:
            findings.append(
                _Finding(
                    severity="information",
                    code="not-found",
                    diagnostics=(
                        "Declared profile is not available in the local IG snapshot."
                    ),
                    expression=profile_expression,
                )
            )
            continue
        findings.extend(
            _check_profile(
                resource,
                resource_type,
                root_expression,
                entry_index,
                canonical,
                profile,
                package,
            )
        )
    return findings


def _declared_profiles(
    resource: Mapping[str, Any], root_expression: str
) -> tuple[list[Any], list[_Finding]]:
    meta = resource.get("meta")
    if meta is None:
        return [], []
    if not isinstance(meta, Mapping):
        return [], [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics="Ignored malformed profile metadata.",
                expression=f"{root_expression}.meta",
            )
        ]
    declared = meta.get("profile")
    if declared is None:
        return [], []
    if isinstance(declared, str):
        return [declared], []
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        return list(declared), []
    return [], [
        _Finding(
            severity="information",
            code="not-supported",
            diagnostics="Ignored malformed meta.profile declarations.",
            expression=f"{root_expression}.meta.profile",
        )
    ]


def _check_profile(
    resource: Mapping[str, Any],
    resource_type: str,
    root_expression: str,
    entry_index: int,
    profile_url: str,
    profile: Mapping[str, Any],
    package: _Package,
) -> list[_Finding]:
    profile_type = profile.get("type")
    if profile_type != resource_type:
        return [
            _Finding(
                severity="error",
                code="structure",
                diagnostics="Declared profile targets a different resource type.",
                expression=f"{root_expression}.resourceType",
                identity=(profile_url, str(entry_index), "resourceType", "type"),
            )
        ]

    elements = _profile_elements(profile)
    if not elements:
        return [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics=(
                    "Profile has no snapshot or differential elements to evaluate."
                ),
                expression=f"{root_expression}.meta.profile",
            )
        ]

    findings = _unsupported_constraint_findings(
        elements,
        resource_type,
        root_expression,
    )
    for element in elements:
        if _is_slice_scoped(element):
            continue
        path = element.get("path")
        segments = _relative_segments(path, resource_type)
        if segments is None or not segments:
            continue
        findings.extend(
            _check_element_at_path(
                resource,
                segments,
                root_expression,
                element,
                profile_url,
                entry_index,
                package,
                identity_scope="element",
            )
        )

    for slice_element in elements:
        if not isinstance(slice_element.get("sliceName"), str):
            continue
        findings.extend(
            _check_slice(
                resource,
                resource_type,
                root_expression,
                entry_index,
                profile_url,
                slice_element,
                elements,
                package,
            )
        )
    return findings


def _profile_elements(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for section_name in ("snapshot", "differential"):
        section = profile.get(section_name)
        if not isinstance(section, Mapping):
            continue
        elements = section.get("element")
        if isinstance(elements, list):
            return [item for item in elements if isinstance(item, Mapping)]
    return []


def _is_slice_scoped(element: Mapping[str, Any]) -> bool:
    element_id = element.get("id")
    return isinstance(element_id, str) and ":" in element_id


def _relative_segments(path: Any, resource_type: str) -> tuple[str, ...] | None:
    if not isinstance(path, str):
        return None
    parts = tuple(part for part in path.split(".") if part)
    if not parts or parts[0] != resource_type:
        return None
    return parts[1:]


def _check_element_at_path(
    node: Any,
    segments: Sequence[str],
    root_expression: str,
    element: Mapping[str, Any],
    profile_url: str,
    entry_index: int,
    package: _Package,
    *,
    identity_scope: str,
) -> list[_Finding]:
    groups = _occurrence_groups(node, segments, root_expression)
    element_key = str(element.get("id") or element.get("path") or ".".join(segments))
    findings: list[_Finding] = []

    minimum, maximum, cardinality_finding = _cardinality(element, root_expression)
    if cardinality_finding is not None:
        findings.append(cardinality_finding)
    else:
        for group in groups:
            count = len(group.occurrences)
            if minimum is not None and count < minimum:
                findings.append(
                    _Finding(
                        severity="error",
                        code="required",
                        diagnostics="Required element cardinality is not met.",
                        expression=group.expression,
                        identity=(
                            profile_url,
                            str(entry_index),
                            identity_scope,
                            element_key,
                            group.expression,
                            "minimum",
                        ),
                    )
                )
            if maximum is not None and count > maximum:
                findings.append(
                    _Finding(
                        severity="error",
                        code="structure",
                        diagnostics="Maximum element cardinality is exceeded.",
                        expression=group.expression,
                        identity=(
                            profile_url,
                            str(entry_index),
                            identity_scope,
                            element_key,
                            group.expression,
                            "maximum",
                        ),
                    )
                )

    for fixed_kind, expected in _fixed_constraints(element):
        for group in groups:
            for occurrence in group.occurrences:
                if occurrence.value != expected:
                    findings.append(
                        _Finding(
                            severity="error",
                            code="value",
                            diagnostics=(
                                "Element does not match the profile's "
                                f"{fixed_kind} constraint."
                            ),
                            expression=occurrence.expression,
                            identity=(
                                profile_url,
                                str(entry_index),
                                identity_scope,
                                element_key,
                                occurrence.expression,
                                fixed_kind,
                            ),
                        )
                    )

    findings.extend(
        _check_binding(
            groups,
            element,
            package,
            profile_url,
            entry_index,
            identity_scope,
            element_key,
        )
    )
    return findings


def _cardinality(
    element: Mapping[str, Any], expression: str
) -> tuple[int | None, int | None, _Finding | None]:
    minimum: int | None = None
    maximum: int | None = None
    raw_minimum = element.get("min")
    raw_maximum = element.get("max")
    try:
        if raw_minimum is not None:
            minimum = int(raw_minimum)
        if raw_maximum not in (None, "*"):
            maximum = int(raw_maximum)
    except (TypeError, ValueError):
        return (
            None,
            None,
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics="Ignored malformed profile cardinality metadata.",
                expression=expression,
            ),
        )
    return minimum, maximum, None


def _fixed_constraints(element: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [
        (key, value)
        for key, value in sorted(element.items())
        if key.startswith("fixed") and len(key) > len("fixed") and value is not None
    ]


def _pattern_constraints(element: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [
        (key, value)
        for key, value in sorted(element.items())
        if key.startswith("pattern") and len(key) > len("pattern") and value is not None
    ]


def _check_binding(
    groups: Sequence[_OccurrenceGroup],
    element: Mapping[str, Any],
    package: _Package,
    profile_url: str,
    entry_index: int,
    identity_scope: str,
    element_key: str,
) -> list[_Finding]:
    binding = element.get("binding")
    if not isinstance(binding, Mapping):
        return []
    value_set_url = _canonical(binding.get("valueSet"))
    strength = binding.get("strength")
    expression = groups[0].expression if groups else None
    if value_set_url is None or not isinstance(strength, str):
        return [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics="Ignored malformed profile binding metadata.",
                expression=expression,
            )
        ]
    value_set = package.value_sets.get(value_set_url)
    if value_set is None or not value_set.complete:
        return [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics=(
                    "Binding cannot be evaluated from the local IG snapshot "
                    "without terminology expansion."
                ),
                expression=expression,
            )
        ]

    canonical_strength = strength.strip().lower()
    severity_by_strength = {
        "required": "error",
        "extensible": "warning",
        "preferred": "information",
    }
    severity = severity_by_strength.get(canonical_strength)
    if severity is None:
        if canonical_strength == "example":
            return []
        return [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics="Unsupported binding strength was not evaluated.",
                expression=expression,
            )
        ]

    findings: list[_Finding] = []
    for group in groups:
        for occurrence in group.occurrences:
            if _value_in_value_set(occurrence.value, value_set.codes):
                continue
            findings.append(
                _Finding(
                    severity=severity,
                    code="code-invalid",
                    diagnostics=(
                        "Coded element is outside the locally available "
                        f"{canonical_strength} binding."
                    ),
                    expression=occurrence.expression,
                    identity=(
                        profile_url,
                        str(entry_index),
                        identity_scope,
                        element_key,
                        occurrence.expression,
                        f"binding-{canonical_strength}",
                    ),
                )
            )
    return findings


def _value_in_value_set(value: Any, codes: frozenset[tuple[str, str]]) -> bool:
    candidates = _extract_codes(value)
    primitive_code = isinstance(value, str)
    for system, code in candidates:
        if system:
            if (system, code) in codes:
                return True
        elif primitive_code and any(
            candidate_code == code for _, candidate_code in codes
        ):
            return True
        elif ("", code) in codes:
            return True
    return False


def _extract_codes(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [("", value)]
    if not isinstance(value, Mapping):
        return []
    code = value.get("code")
    if isinstance(code, str):
        system = value.get("system")
        return [((system if isinstance(system, str) else ""), code)]
    coding = value.get("coding")
    if isinstance(coding, list):
        codes: list[tuple[str, str]] = []
        for item in coding:
            codes.extend(_extract_codes(item))
        return codes
    return []


def _check_slice(
    resource: Mapping[str, Any],
    resource_type: str,
    root_expression: str,
    entry_index: int,
    profile_url: str,
    slice_element: Mapping[str, Any],
    elements: Sequence[Mapping[str, Any]],
    package: _Package,
) -> list[_Finding]:
    base_path = slice_element.get("path")
    segments = _relative_segments(base_path, resource_type)
    if segments is None or not segments:
        return []
    if segments[-1] not in {"identifier", "category"}:
        return [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics=(
                    "Required slice is outside the supported identifier/category "
                    "subset."
                ),
                expression=_path_expression(root_expression, segments),
            )
        ]

    slice_id = slice_element.get("id")
    if not isinstance(slice_id, str):
        return []
    descendants = [
        element
        for element in elements
        if isinstance(element.get("id"), str)
        and element["id"].startswith(f"{slice_id}.")
    ]
    slicing_element = next(
        (
            element
            for element in elements
            if element.get("path") == base_path
            and not _is_slice_scoped(element)
            and isinstance(element.get("slicing"), Mapping)
        ),
        None,
    )
    discriminators = _slice_discriminators(
        slice_element,
        descendants,
        slicing_element,
        str(base_path),
    )
    groups = _occurrence_groups(resource, segments, root_expression)
    if not discriminators:
        return [
            _Finding(
                severity="information",
                code="not-supported",
                diagnostics=(
                    "Required slice has no supported fixed or pattern discriminator."
                ),
                expression=groups[0].expression if groups else None,
            )
        ]

    minimum, maximum, cardinality_finding = _cardinality(
        slice_element,
        groups[0].expression if groups else root_expression,
    )
    if cardinality_finding is not None:
        return [cardinality_finding]

    slice_name = str(slice_element.get("sliceName"))
    findings: list[_Finding] = []
    for group in groups:
        matches = [
            occurrence
            for occurrence in group.occurrences
            if _matches_slice(occurrence.value, discriminators)
        ]
        if minimum is not None and len(matches) < minimum:
            findings.append(
                _Finding(
                    severity="error",
                    code="required",
                    diagnostics=(
                        "Required identifier/category slice is missing from the "
                        "profiled resource."
                    ),
                    expression=group.expression,
                    identity=(
                        profile_url,
                        str(entry_index),
                        "slice",
                        slice_id,
                        group.expression,
                        "minimum",
                    ),
                )
            )
        if maximum is not None and len(matches) > maximum:
            findings.append(
                _Finding(
                    severity="error",
                    code="structure",
                    diagnostics="Maximum identifier/category slice count is exceeded.",
                    expression=group.expression,
                    identity=(
                        profile_url,
                        str(entry_index),
                        "slice",
                        slice_id,
                        group.expression,
                        "maximum",
                    ),
                )
            )
        for occurrence in matches:
            for descendant in descendants:
                child_path = descendant.get("path")
                relative = _descendant_segments(child_path, str(base_path))
                if relative is None or not relative:
                    continue
                findings.extend(
                    _check_element_at_path(
                        occurrence.value,
                        relative,
                        occurrence.expression,
                        descendant,
                        profile_url,
                        entry_index,
                        package,
                        identity_scope=f"slice-{slice_name}",
                    )
                )
    return findings


def _slice_discriminators(
    slice_element: Mapping[str, Any],
    descendants: Sequence[Mapping[str, Any]],
    slicing_element: Mapping[str, Any] | None,
    base_path: str,
) -> list[tuple[tuple[str, ...], str, Any]]:
    if slicing_element is None:
        return []
    slicing = slicing_element.get("slicing")
    if not isinstance(slicing, Mapping):
        return []
    declared = slicing.get("discriminator")
    if not isinstance(declared, list) or not declared:
        return []

    constraints: list[tuple[tuple[str, ...], str, Any]] = []
    for discriminator in declared:
        if not isinstance(discriminator, Mapping):
            return []
        discriminator_type = discriminator.get("type")
        raw_path = discriminator.get("path")
        if discriminator_type not in {"value", "pattern"} or not isinstance(
            raw_path, str
        ):
            return []

        relative = () if raw_path == "$this" else tuple(raw_path.split("."))
        if any(not segment for segment in relative):
            return []
        constrained_element = slice_element if not relative else None
        if relative:
            constrained_element = next(
                (
                    descendant
                    for descendant in descendants
                    if _descendant_segments(descendant.get("path"), base_path)
                    == relative
                ),
                None,
            )
        if constrained_element is None:
            return []

        candidates = [
            ("fixed", expected)
            for _, expected in _fixed_constraints(constrained_element)
        ]
        candidates.extend(
            ("pattern", expected)
            for _, expected in _pattern_constraints(constrained_element)
        )
        if not candidates:
            return []
        for mode, expected in candidates:
            constraints.append((relative, mode, expected))

    return constraints


def _descendant_segments(path: Any, base_path: str) -> tuple[str, ...] | None:
    if not isinstance(path, str) or not path.startswith(f"{base_path}."):
        return None
    return tuple(part for part in path[len(base_path) + 1 :].split(".") if part)


def _matches_slice(
    value: Any,
    constraints: Sequence[tuple[tuple[str, ...], str, Any]],
) -> bool:
    for segments, mode, expected in constraints:
        if segments:
            groups = _occurrence_groups(value, segments, "slice")
            candidates = [
                occurrence.value for group in groups for occurrence in group.occurrences
            ]
        else:
            candidates = [value]
        if mode == "fixed":
            if not any(candidate == expected for candidate in candidates):
                return False
        elif not any(_pattern_matches(candidate, expected) for candidate in candidates):
            return False
    return True


def _pattern_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _pattern_matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(_pattern_matches(candidate, item) for candidate in actual)
            for item in expected
        )
    return actual == expected


def _unsupported_constraint_findings(
    elements: Sequence[Mapping[str, Any]],
    resource_type: str,
    root_expression: str,
) -> list[_Finding]:
    findings: list[_Finding] = []
    for element in elements:
        segments = _relative_segments(element.get("path"), resource_type)
        expression = (
            _path_expression(root_expression, segments)
            if segments is not None
            else root_expression
        )
        unsupported = []
        for key in element:
            if key in {"constraint", "condition", "maxLength"} or key.startswith(
                ("minValue", "maxValue")
            ):
                unsupported.append(key)
            if key.startswith("pattern") and not _is_slice_scoped(element):
                unsupported.append(key)
        for _ in unsupported:
            findings.append(
                _Finding(
                    severity="information",
                    code="not-supported",
                    diagnostics=(
                        "Profile constraint kind is outside the lightweight "
                        "checker's supported subset."
                    ),
                    expression=expression,
                )
            )
        slicing = element.get("slicing")
        if isinstance(slicing, Mapping) and isinstance(
            slicing.get("discriminator"), list
        ):
            for discriminator in slicing["discriminator"]:
                discriminator_type = (
                    discriminator.get("type")
                    if isinstance(discriminator, Mapping)
                    else None
                )
                if discriminator_type not in {"value", "pattern"}:
                    findings.append(
                        _Finding(
                            severity="information",
                            code="not-supported",
                            diagnostics=(
                                "Slice discriminator type is outside the "
                                "lightweight checker's supported subset."
                            ),
                            expression=expression,
                        )
                    )
    return findings


def _occurrence_groups(
    node: Any,
    segments: Sequence[str],
    root_expression: str,
) -> list[_OccurrenceGroup]:
    if not segments:
        occurrence = (
            (_Occurrence(node, root_expression),) if _is_present(node) else tuple()
        )
        return [_OccurrenceGroup(root_expression, occurrence)]

    parents = [_Occurrence(node, root_expression)]
    for segment in segments[:-1]:
        next_parents: list[_Occurrence] = []
        for parent in parents:
            next_parents.extend(_children(parent, segment))
        parents = next_parents
        if not parents:
            return []

    final_segment = segments[-1]
    groups: list[_OccurrenceGroup] = []
    for parent in parents:
        children = tuple(_children(parent, final_segment))
        groups.append(
            _OccurrenceGroup(
                expression=_child_expression(parent.expression, final_segment),
                occurrences=children,
            )
        )
    return groups


def _children(parent: _Occurrence, segment: str) -> list[_Occurrence]:
    if not isinstance(parent.value, Mapping):
        return []
    children: list[_Occurrence] = []
    for key in _matching_keys(parent.value, segment):
        value = parent.value[key]
        expression = f"{parent.expression}.{key}"
        if isinstance(value, list):
            children.extend(
                _Occurrence(item, f"{expression}[{index}]")
                for index, item in enumerate(value)
                if _is_present(item)
            )
        elif _is_present(value):
            children.append(_Occurrence(value, expression))
    return children


def _matching_keys(node: Mapping[str, Any], segment: str) -> list[str]:
    if segment.endswith("[x]"):
        prefix = segment[:-3]
        return sorted(
            key
            for key in node
            if key.startswith(prefix)
            and len(key) > len(prefix)
            and key[len(prefix)].isupper()
        )
    return [segment] if segment in node else []


def _child_expression(parent_expression: str, segment: str) -> str:
    if segment.endswith("[x]"):
        segment = segment[:-3]
    return f"{parent_expression}.{segment}"


def _path_expression(root_expression: str, segments: Sequence[str]) -> str:
    expression = root_expression
    for segment in segments:
        expression = _child_expression(expression, segment)
    return expression


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return bool(value)
    return True
