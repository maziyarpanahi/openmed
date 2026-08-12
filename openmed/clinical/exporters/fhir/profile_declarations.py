"""Deterministic, offline checks for FHIR ``meta.profile`` declarations.

FHIR canonical profile declarations are useful export metadata, but a
declaration is not self-validating.  This module checks declarations against a
caller-supplied local catalog.  The catalog supplies the canonical URL,
resource type, and supported FHIR release for each profile; no profile is
resolved, downloaded, or dereferenced.

The primary entry point returns value-free findings.  The companion
``validate_profile_declarations`` function adapts those findings to the
shared FHIR ``OperationOutcome`` shape.  Both functions accept a resource, a
sequence of resources, or a FHIR Bundle containing resources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

from .operation_outcome import to_operation_outcome

__all__ = [
    "FHIR_R4",
    "FHIR_R5",
    "MISSING_PROFILE_DECLARATION",
    "DUPLICATE_PROFILE_DECLARATION",
    "UNKNOWN_PROFILE_DECLARATION",
    "PROFILE_RESOURCE_TYPE_MISMATCH",
    "PROFILE_FHIR_VERSION_MISMATCH",
    "PROFILE_VALIDATION_MODE_MISMATCH",
    "ProfileDeclarationFinding",
    "ProfileDeclarationSpec",
    "check_profile_declarations",
    "validate_profile_declarations",
]

FHIR_R4 = "R4"
FHIR_R5 = "R5"
_FHIR_VERSIONS = frozenset({FHIR_R4, FHIR_R5})

MISSING_PROFILE_DECLARATION = "missing-profile-declaration"
DUPLICATE_PROFILE_DECLARATION = "duplicate-profile-declaration"
UNKNOWN_PROFILE_DECLARATION = "unknown-profile-declaration"
PROFILE_RESOURCE_TYPE_MISMATCH = "profile-resource-type-mismatch"
PROFILE_FHIR_VERSION_MISMATCH = "profile-fhir-version-mismatch"
PROFILE_VALIDATION_MODE_MISMATCH = "profile-validation-mode-mismatch"

_INVALID_PROFILE_DECLARATION = "invalid-profile-declaration"
_INVALID_PROFILE_METADATA = "invalid-profile-metadata"
_INVALID_RESOURCE = "invalid-resource"
_INVALID_RESOURCE_TYPE = "invalid-resource-type"


class ProfileDeclarationFinding(TypedDict):
    """Value-free finding returned by :func:`check_profile_declarations`."""

    finding_code: str
    severity: Literal["error", "warning"]
    code: str
    diagnostics: str
    expression: list[str]


@dataclass(frozen=True)
class ProfileDeclarationSpec:
    """Offline contract for one canonical FHIR profile URL.

    Args:
        canonical_url: Absolute canonical URL used in ``meta.profile``.
        resource_type: FHIR resource type targeted by the profile.  ``"*"``
            is accepted for a URL-only allowlist and skips type matching.
        fhir_versions: FHIR releases in which the profile may be declared.
        validation_modes: Optional caller-defined validation modes.  When
            non-empty, the selected validation mode must be present.
        profile_versions: Optional versions after the ``|`` in a canonical
            declaration.  An empty collection accepts any profile version.
        required: Whether this profile is required for resources of its type.
    """

    canonical_url: str = ""
    resource_type: str = "*"
    fhir_versions: frozenset[str] = _FHIR_VERSIONS
    validation_modes: frozenset[str] = frozenset()
    profile_versions: frozenset[str] = frozenset()
    required: bool = False

    def __post_init__(self) -> None:
        canonical_version: str | None = None
        if self.canonical_url:
            canonical_url, canonical_version = _normalise_canonical(
                self.canonical_url,
                allow_version=True,
            )
            object.__setattr__(self, "canonical_url", canonical_url)
        elif not isinstance(self.canonical_url, str):
            raise TypeError("canonical_url must be a string")
        object.__setattr__(
            self, "resource_type", _normalise_resource_type(self.resource_type)
        )
        object.__setattr__(
            self, "fhir_versions", _normalise_versions(self.fhir_versions)
        )
        object.__setattr__(
            self,
            "validation_modes",
            _normalise_modes(self.validation_modes),
        )
        profile_versions = self.profile_versions
        if not profile_versions and canonical_version is not None:
            profile_versions = (canonical_version,)
        object.__setattr__(
            self,
            "profile_versions",
            _normalise_profile_versions(profile_versions),
        )
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")

    @property
    def url(self) -> str:
        """Return the canonical URL under the common short attribute name."""

        return self.canonical_url


@dataclass(frozen=True)
class _Catalog:
    profiles: Mapping[str, tuple[ProfileDeclarationSpec, ...]]
    required_by_type: Mapping[str, tuple[str, ...]]


def check_profile_declarations(
    resources: Any,
    profiles: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    profile_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
    profile_registry: Mapping[str, Any] | Sequence[Any] | None = None,
    allowed_profiles: Mapping[str, Any] | Sequence[Any] | None = None,
    declarations: Mapping[str, Any] | Sequence[Any] | None = None,
    canonical_urls: Mapping[str, Any] | Sequence[Any] | None = None,
    expected_profiles: Mapping[str, Any] | Sequence[Any] | None = None,
    required_profiles: Mapping[str, Any] | Sequence[Any] | None = None,
    resource_types: Mapping[str, str] | None = None,
    fhir_version: str = FHIR_R4,
    version: str | None = None,
    fhir_mode: str | None = None,
    mode: str | None = None,
    fhir_release: str | None = None,
    validation_mode: str | None = None,
    validation: str | None = None,
    require_profile: bool = True,
    expression: str | None = None,
) -> list[ProfileDeclarationFinding]:
    """Return deterministic findings for FHIR profile declarations.

    Args:
        resources: A FHIR resource, a sequence of resources, or a FHIR Bundle.
        profiles: URL-to-spec catalog.  A spec may be a
            :class:`ProfileDeclarationSpec` or a mapping with
            ``resource_type`` and ``fhir_versions`` fields.
        profile_catalog, profile_registry, allowed_profiles, declarations,
            canonical_urls: Compatibility aliases for ``profiles``.  At most
            one catalog argument may be supplied.
        expected_profiles: Optional required catalog.  It can be keyed by
            resource type or contain ``ProfileDeclarationSpec`` values.
            Required specs declared with ``required=True`` are also used.
        required_profiles: Alias for ``expected_profiles``.
        resource_types: Optional URL-to-resource-type mapping used with a
            URL sequence or mapping that does not include resource types.
        fhir_version: Requested FHIR release, ``R4`` or ``R5``.  Dotted
            release aliases such as ``4.0.1`` and ``5.0.0`` are accepted.
        version, fhir_mode, mode, fhir_release: Compatibility aliases for the
            requested FHIR release.  ``mode`` may instead be a caller-defined
            validation mode when it is not an R4/R5 alias.
        validation_mode, validation: Optional caller-defined validation mode.
        require_profile: Require every inspected resource to declare at least
            one profile unless ``False``.  Explicit required profiles are
            still checked when this is false.
        expression: Optional safe root used in returned FHIRPath expressions.

    Returns:
        A list of value-free finding dictionaries.  An empty list means that
        every inspected declaration is known and consistent with the catalog.

    Raises:
        TypeError or ValueError: If the local catalog or FHIR mode is malformed.
            Malformed resource content is represented as a finding instead.
    """

    catalog_input = _one_catalog(
        profiles,
        profile_catalog=profile_catalog,
        profile_registry=profile_registry,
        allowed_profiles=allowed_profiles,
        declarations=declarations,
        canonical_urls=canonical_urls,
    )
    selected_version = _resolve_fhir_version(
        fhir_version,
        version=version,
        fhir_mode=fhir_mode,
        mode=mode,
        fhir_release=fhir_release,
    )
    selected_validation_mode = _resolve_validation_mode(
        mode=mode,
        validation_mode=validation_mode,
        validation=validation,
    )
    if not isinstance(require_profile, bool):
        raise TypeError("require_profile must be a boolean")

    catalog = _build_catalog(catalog_input, resource_types=resource_types)
    required_input = _one_expected_catalog(expected_profiles, required_profiles)
    required_catalog = _build_catalog(required_input, resource_types=resource_types)
    required_by_type = _merge_required_profiles(
        catalog.required_by_type,
        required_catalog.profiles,
    )
    selected_expression = _safe_expression(expression) if expression else None

    findings: list[ProfileDeclarationFinding] = []
    for index, resource, root in _iter_resources(resources, selected_expression):
        findings.extend(
            _check_resource(
                resource,
                root,
                index=index,
                catalog=catalog,
                required_by_type=required_by_type,
                fhir_version=selected_version,
                validation_mode=selected_validation_mode,
                require_profile=require_profile,
            )
        )
    return findings


def validate_profile_declarations(
    resources: Any,
    profiles: Mapping[str, Any] | Sequence[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return an ``OperationOutcome`` for profile declaration findings."""

    return to_operation_outcome(
        check_profile_declarations(resources, profiles, **kwargs)
    )


def _one_catalog(
    profiles: Mapping[str, Any] | Sequence[Any] | None,
    **aliases: Mapping[str, Any] | Sequence[Any] | None,
) -> Mapping[str, Any] | Sequence[Any] | None:
    supplied = [
        candidate
        for candidate in (profiles, *aliases.values())
        if candidate is not None
    ]
    if len(supplied) > 1:
        raise ValueError("provide only one profile catalog")
    return supplied[0] if supplied else None


def _one_expected_catalog(
    expected_profiles: Mapping[str, Any] | Sequence[Any] | None,
    required_profiles: Mapping[str, Any] | Sequence[Any] | None,
) -> Mapping[str, Any] | Sequence[Any] | None:
    supplied = tuple(
        candidate
        for candidate in (expected_profiles, required_profiles)
        if candidate is not None
    )
    if len(supplied) > 1:
        raise ValueError("provide only one required profile catalog")
    return supplied[0] if supplied else None


def _build_catalog(
    raw_catalog: Mapping[str, Any] | Sequence[Any] | None,
    *,
    resource_types: Mapping[str, str] | None,
) -> _Catalog:
    if raw_catalog is None:
        return _Catalog(profiles=MappingProxyType({}), required_by_type={})

    specs: list[ProfileDeclarationSpec] = []
    if isinstance(raw_catalog, Mapping):
        if _looks_like_single_spec(raw_catalog):
            specs.append(_coerce_spec(raw_catalog))
        else:
            for key, value in raw_catalog.items():
                specs.extend(
                    _coerce_catalog_entry(
                        key,
                        value,
                        resource_types=resource_types,
                    )
                )
    elif isinstance(raw_catalog, Sequence) and not isinstance(
        raw_catalog, (str, bytes, bytearray)
    ):
        for value in raw_catalog:
            if isinstance(value, str) and _is_canonical_url(value):
                specs.append(
                    ProfileDeclarationSpec(
                        canonical_url=value,
                        resource_type=_resource_type_for(
                            value,
                            resource_types,
                            fallback="*",
                        ),
                    )
                )
            else:
                specs.append(_coerce_spec(value))
    else:
        raise TypeError("profile catalog must be a mapping or sequence")

    grouped: dict[str, list[ProfileDeclarationSpec]] = {}
    required: dict[str, list[str]] = {}
    for spec in specs:
        if not spec.canonical_url:
            raise ValueError("profile specifications need a canonical URL")
        grouped.setdefault(spec.canonical_url, []).append(spec)
        if spec.required and spec.resource_type != "*":
            required.setdefault(spec.resource_type, []).append(spec.canonical_url)

    stable_profiles = {
        url: tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.resource_type,
                    tuple(sorted(item.fhir_versions)),
                    tuple(sorted(item.validation_modes)),
                    tuple(sorted(item.profile_versions)),
                ),
            )
        )
        for url, candidates in sorted(grouped.items())
    }
    stable_required = {
        resource_type: tuple(sorted(set(urls)))
        for resource_type, urls in sorted(required.items())
    }
    return _Catalog(
        profiles=MappingProxyType(stable_profiles),
        required_by_type=MappingProxyType(stable_required),
    )


def _coerce_catalog_entry(
    key: Any,
    value: Any,
    *,
    resource_types: Mapping[str, str] | None,
) -> list[ProfileDeclarationSpec]:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("profile catalog keys must be non-empty strings")

    key_is_url = _is_canonical_url(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        items = [value]

    result: list[ProfileDeclarationSpec] = []
    for item in items:
        if key_is_url:
            inferred_type = _resource_type_for(key, resource_types, fallback="*")
            if isinstance(item, Mapping) and not any(
                field in item for field in ("resource_type", "resourceType", "type")
            ):
                item = {
                    **item,
                    "resource_type": inferred_type,
                }
            elif item is None and inferred_type != "*":
                item = {"resource_type": inferred_type}
            elif (
                isinstance(item, ProfileDeclarationSpec)
                and item.resource_type == "*"
                and inferred_type != "*"
            ):
                item = ProfileDeclarationSpec(
                    canonical_url=key,
                    resource_type=inferred_type,
                    fhir_versions=item.fhir_versions,
                    validation_modes=item.validation_modes,
                    profile_versions=item.profile_versions,
                    required=item.required,
                )
            result.append(_coerce_spec(item, canonical_url=key))
            continue
        if isinstance(item, str) and _is_canonical_url(item):
            resource_type = _resource_type_for(item, resource_types, fallback=key)
            result.append(
                ProfileDeclarationSpec(
                    canonical_url=item,
                    resource_type=resource_type,
                )
            )
            continue
        if isinstance(item, Mapping) or isinstance(item, ProfileDeclarationSpec):
            result.append(
                _coerce_spec(
                    item,
                    canonical_url=(
                        item.get("canonical_url")
                        if isinstance(item, Mapping)
                        else item.canonical_url
                    ),
                    resource_type_hint=key,
                )
            )
            continue
        raise ValueError("profile catalog entries must contain canonical profiles")
    return result


def _coerce_spec(
    raw_spec: Any,
    *,
    canonical_url: str | None = None,
    resource_type_hint: str | None = None,
) -> ProfileDeclarationSpec:
    if isinstance(raw_spec, ProfileDeclarationSpec):
        if (
            canonical_url is None
            or raw_spec.canonical_url == _normalise_canonical(canonical_url)[0]
        ):
            return raw_spec
        return ProfileDeclarationSpec(
            canonical_url=canonical_url,
            resource_type=raw_spec.resource_type,
            fhir_versions=raw_spec.fhir_versions,
            validation_modes=raw_spec.validation_modes,
            profile_versions=raw_spec.profile_versions,
            required=raw_spec.required,
        )

    if isinstance(raw_spec, str):
        if canonical_url is None and _is_canonical_url(raw_spec):
            return ProfileDeclarationSpec(
                canonical_url=raw_spec,
                resource_type=resource_type_hint or "*",
            )
        resource_type = raw_spec if canonical_url is not None else resource_type_hint
        if not resource_type:
            raise ValueError("profile specifications need a resource type")
        if canonical_url is None:
            raise ValueError("profile specifications need a canonical URL")
        return ProfileDeclarationSpec(
            canonical_url=canonical_url,
            resource_type=resource_type,
        )

    if raw_spec is None:
        raw_spec = {}
    if not isinstance(raw_spec, Mapping):
        raise TypeError(
            "profile specification must be a mapping or ProfileDeclarationSpec"
        )

    raw_url = canonical_url or _first_value(
        raw_spec,
        "canonical_url",
        "canonicalUrl",
        "canonical",
        "url",
        "profile_url",
        "profileUrl",
        "canonical_urls",
    )
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("profile specifications need a canonical URL")

    resource_type = _first_value(
        raw_spec,
        "resource_type",
        "resourceType",
        "type",
    )
    if resource_type is None:
        resource_type = resource_type_hint or "*"

    raw_fhir_versions = _first_value(
        raw_spec,
        "fhir_versions",
        "fhirVersions",
        "fhir_version",
        "fhirVersion",
        "fhir_release",
        "fhirRelease",
    )
    raw_mode = _first_value(raw_spec, "mode")
    if raw_fhir_versions is None and _looks_like_fhir_version(raw_mode):
        raw_fhir_versions = raw_mode
    raw_generic_version = _first_value(raw_spec, "version")
    if raw_fhir_versions is None:
        raw_fhir_versions = _first_value(raw_spec, "versions")
    if raw_fhir_versions is None and _looks_like_fhir_version(raw_generic_version):
        raw_fhir_versions = raw_generic_version
    if raw_fhir_versions is None:
        raw_fhir_versions = _FHIR_VERSIONS

    raw_modes = _first_value(
        raw_spec,
        "validation_modes",
        "validationModes",
        "validation_mode",
        "validationMode",
        "modes",
    )
    if (
        raw_modes is None
        and raw_mode is not None
        and not _looks_like_fhir_version(raw_mode)
    ):
        raw_modes = raw_mode
    if raw_modes is None:
        raw_modes = ()

    raw_profile_versions = _first_value(
        raw_spec,
        "profile_versions",
        "profileVersions",
        "canonical_versions",
        "canonicalVersions",
        "declared_versions",
        "declaredVersions",
    )
    if raw_profile_versions is None:
        raw_profile_versions = (
            raw_generic_version
            if raw_generic_version is not None
            and not _looks_like_fhir_version(raw_generic_version)
            else ()
        )

    canonical_url, canonical_version = _normalise_canonical(
        raw_url,
        allow_version=True,
    )
    if not raw_profile_versions and canonical_version is not None:
        raw_profile_versions = (canonical_version,)

    required = raw_spec.get("required", False)
    return ProfileDeclarationSpec(
        canonical_url=canonical_url,
        resource_type=resource_type,
        fhir_versions=raw_fhir_versions,
        validation_modes=raw_modes,
        profile_versions=raw_profile_versions,
        required=required,
    )


def _merge_required_profiles(
    required_by_type: Mapping[str, tuple[str, ...]],
    required_catalog: Mapping[str, tuple[ProfileDeclarationSpec, ...]],
) -> Mapping[str, tuple[str, ...]]:
    merged: dict[str, set[str]] = {
        resource_type: set(urls) for resource_type, urls in required_by_type.items()
    }
    for url, specs in required_catalog.items():
        for spec in specs:
            if spec.resource_type != "*":
                merged.setdefault(spec.resource_type, set()).add(url)
    return MappingProxyType(
        {
            resource_type: tuple(sorted(urls))
            for resource_type, urls in sorted(merged.items())
        }
    )


def _iter_resources(
    resources: Any,
    expression: str | None,
) -> Sequence[tuple[int, Mapping[str, Any], str]]:
    if isinstance(resources, Mapping):
        if resources.get("resourceType") == "Bundle" and "entry" in resources:
            entries = resources.get("entry")
            if not isinstance(entries, list):
                return ((0, resources, expression or "Bundle"),)
            result: list[tuple[int, Mapping[str, Any], str]] = []
            for index, entry in enumerate(entries):
                root = f"{expression or 'Bundle'}.entry[{index}].resource"
                if isinstance(entry, Mapping) and isinstance(
                    entry.get("resource"), Mapping
                ):
                    result.append((index, entry["resource"], root))
                else:
                    result.append((index, {"resourceType": "__invalid__"}, root))
            return result
        return ((0, resources, expression or _default_root(resources)),)
    if isinstance(resources, Sequence) and not isinstance(
        resources, (str, bytes, bytearray)
    ):
        result = []
        for index, resource in enumerate(resources):
            root = f"{expression or 'resources'}[{index}]"
            if isinstance(resource, Mapping):
                result.append((index, resource, root))
            else:
                result.append((index, {"resourceType": "__invalid__"}, root))
        return result
    return ((0, {"resourceType": "__invalid__"}, expression or "Resource"),)


def _check_resource(
    resource: Mapping[str, Any],
    root: str,
    *,
    index: int,
    catalog: _Catalog,
    required_by_type: Mapping[str, tuple[str, ...]],
    fhir_version: str,
    validation_mode: str | None,
    require_profile: bool,
) -> list[ProfileDeclarationFinding]:
    del index  # The safe expression already carries the structural location.
    if resource.get("resourceType") == "__invalid__":
        return [
            _finding(
                _INVALID_RESOURCE, "structure", "Resource must be an object.", root
            )
        ]

    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type.strip():
        return [
            _finding(
                _INVALID_RESOURCE_TYPE,
                "invalid",
                "resourceType must be a non-empty string.",
                f"{root}.resourceType",
            )
        ]
    resource_type = resource_type.strip()

    raw_profiles, metadata_findings = _read_profiles(resource, root)
    findings = list(metadata_findings)
    declared_urls: list[str] = []
    declared_bases: set[str] = set()
    duplicate_bases: set[str] = set()

    for profile_index, raw_profile in enumerate(raw_profiles):
        profile_path = f"{root}.meta.profile[{profile_index}]"
        if not isinstance(raw_profile, str):
            findings.append(
                _finding(
                    _INVALID_PROFILE_DECLARATION,
                    "value",
                    "Profile declarations must be canonical strings.",
                    profile_path,
                )
            )
            continue
        try:
            canonical, declared_version = _normalise_canonical(raw_profile)
        except (TypeError, ValueError):
            findings.append(
                _finding(
                    _INVALID_PROFILE_DECLARATION,
                    "value",
                    "Profile declaration is not a valid canonical URI.",
                    profile_path,
                )
            )
            continue

        declared_urls.append(canonical)
        if canonical in declared_bases:
            duplicate_bases.add(canonical)
        declared_bases.add(canonical)
        candidates = catalog.profiles.get(canonical, ())
        spec = _select_spec(candidates, declared_version)
        if spec is None:
            findings.append(
                _finding(
                    UNKNOWN_PROFILE_DECLARATION,
                    "not-found",
                    "Profile declaration is not present in the injected local catalog.",
                    profile_path,
                )
            )
            continue

        if spec.resource_type != "*" and spec.resource_type != resource_type:
            findings.append(
                _finding(
                    PROFILE_RESOURCE_TYPE_MISMATCH,
                    "structure",
                    "Declared profile targets a different resource type.",
                    f"{root}.resourceType",
                )
            )
        if fhir_version not in spec.fhir_versions:
            findings.append(
                _finding(
                    PROFILE_FHIR_VERSION_MISMATCH,
                    "not-supported",
                    "Declared profile is not supported in the requested FHIR release.",
                    profile_path,
                )
            )
        declared_fhir_version = (
            _normalise_fhir_version(declared_version)
            if _looks_like_fhir_version(declared_version)
            else None
        )
        if declared_fhir_version is not None and declared_fhir_version != fhir_version:
            findings.append(
                _finding(
                    PROFILE_FHIR_VERSION_MISMATCH,
                    "value",
                    "Profile declaration version conflicts with the requested FHIR release.",
                    profile_path,
                )
            )
        if (
            validation_mode is not None
            and spec.validation_modes
            and validation_mode not in spec.validation_modes
        ):
            findings.append(
                _finding(
                    PROFILE_VALIDATION_MODE_MISMATCH,
                    "not-supported",
                    "Declared profile is not supported in the requested validation mode.",
                    profile_path,
                )
            )
        if (
            declared_version is not None
            and spec.profile_versions
            and declared_version not in spec.profile_versions
        ):
            findings.append(
                _finding(
                    PROFILE_FHIR_VERSION_MISMATCH,
                    "value",
                    "Declared profile version is not present in the local catalog.",
                    profile_path,
                )
            )

    if duplicate_bases:
        findings.append(
            _finding(
                DUPLICATE_PROFILE_DECLARATION,
                "duplicate",
                "A canonical profile is declared more than once.",
                f"{root}.meta.profile",
            )
        )

    required_urls = set(required_by_type.get(resource_type, ()))
    missing_required = required_urls - set(declared_urls)
    if (
        not raw_profiles
        and (require_profile or missing_required)
        and not metadata_findings
    ):
        findings.append(
            _finding(
                MISSING_PROFILE_DECLARATION,
                "required",
                "Resource does not declare a profile.",
                f"{root}.meta.profile",
            )
        )
    elif missing_required:
        findings.append(
            _finding(
                MISSING_PROFILE_DECLARATION,
                "required",
                "A required profile is not declared for this resource type.",
                f"{root}.meta.profile",
            )
        )
    return findings


def _read_profiles(
    resource: Mapping[str, Any],
    root: str,
) -> tuple[list[Any], list[ProfileDeclarationFinding]]:
    meta = resource.get("meta")
    if meta is None:
        return [], []
    if not isinstance(meta, Mapping):
        return [], [
            _finding(
                _INVALID_PROFILE_METADATA,
                "structure",
                "Resource meta must be an object.",
                f"{root}.meta",
            )
        ]
    raw_profiles = meta.get("profile")
    if raw_profiles is None:
        return [], []
    if not isinstance(raw_profiles, list):
        return [], [
            _finding(
                _INVALID_PROFILE_METADATA,
                "structure",
                "Resource meta.profile must be an array.",
                f"{root}.meta.profile",
            )
        ]
    return list(raw_profiles), []


def _select_spec(
    candidates: Sequence[ProfileDeclarationSpec],
    declared_version: str | None,
) -> ProfileDeclarationSpec | None:
    if not candidates:
        return None
    if declared_version is None:
        return candidates[0]
    matching = [
        candidate
        for candidate in candidates
        if not candidate.profile_versions
        or declared_version in candidate.profile_versions
    ]
    return matching[0] if matching else candidates[0]


def _finding(
    finding_code: str,
    code: str,
    diagnostics: str,
    expression: str,
    *,
    severity: Literal["error", "warning"] = "error",
) -> ProfileDeclarationFinding:
    return {
        "finding_code": finding_code,
        "severity": severity,
        "code": code,
        "diagnostics": diagnostics,
        "expression": [expression],
    }


def _resolve_fhir_version(
    fhir_version: str,
    *,
    version: str | None,
    fhir_mode: str | None,
    mode: str | None,
    fhir_release: str | None,
) -> str:
    aliases = [
        candidate for candidate in (version, fhir_mode, fhir_release) if candidate
    ]
    if mode and _looks_like_fhir_version(mode):
        aliases.append(mode)
    normalised = {_normalise_fhir_version(candidate) for candidate in aliases}
    if len(normalised) > 1:
        raise ValueError("FHIR version aliases must agree")
    if normalised:
        selected = next(iter(normalised))
        if _normalise_fhir_version(fhir_version) not in {FHIR_R4, selected}:
            raise ValueError("FHIR version aliases must agree")
        return selected
    return _normalise_fhir_version(fhir_version)


def _resolve_validation_mode(
    *,
    mode: str | None,
    validation_mode: str | None,
    validation: str | None,
) -> str | None:
    aliases = [
        candidate
        for candidate in (validation_mode, validation)
        if candidate is not None
    ]
    if mode is not None and not _looks_like_fhir_version(mode):
        aliases.append(mode)
    normalised = {_normalise_mode(candidate) for candidate in aliases}
    if len(normalised) > 1:
        raise ValueError("validation mode aliases must agree")
    return next(iter(normalised)) if normalised else None


def _normalise_fhir_version(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("FHIR version must be a string")
    canonical = value.strip().casefold()
    if canonical in {"r4", "4", "4.0", "4.0.1"}:
        return FHIR_R4
    if canonical in {"r5", "5", "5.0", "5.0.0"}:
        return FHIR_R5
    raise ValueError("FHIR version must be R4 or R5")


def _looks_like_fhir_version(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().casefold() in {
        "r4",
        "4",
        "4.0",
        "4.0.1",
        "r5",
        "5",
        "5.0",
        "5.0.0",
    }


def _normalise_versions(values: Any) -> frozenset[str]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, (set, frozenset)):
        values = tuple(sorted(values))
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise TypeError("fhir_versions must be a sequence")
    result = frozenset(_normalise_fhir_version(value) for value in values)
    if not result:
        raise ValueError("fhir_versions must not be empty")
    return result


def _normalise_modes(values: Any) -> frozenset[str]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, (set, frozenset)):
        values = tuple(sorted(values))
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise TypeError("validation_modes must be a sequence")
    return frozenset(_normalise_mode(value) for value in values)


def _normalise_profile_versions(values: Any) -> frozenset[str]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, (set, frozenset)):
        values = tuple(sorted(values))
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise TypeError("profile_versions must be a sequence")
    result = frozenset(
        value.strip() for value in values if isinstance(value, str) and value.strip()
    )
    if len(result) != len(values):
        raise ValueError("profile_versions must contain non-empty strings")
    return result


def _normalise_mode(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("validation mode must be a non-empty string")
    return value.strip().casefold()


def _normalise_resource_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("resource_type must be a non-empty string")
    return value.strip()


def _normalise_canonical(
    value: Any, *, allow_version: bool = False
) -> tuple[str, str | None]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("canonical profile URL must be a non-empty URI")
    if any(character.isspace() for character in value):
        raise ValueError("canonical profile URL must not contain whitespace")
    canonical, separator, version = value.partition("|")
    if not _is_canonical_url(canonical):
        raise ValueError("canonical profile URL must be an absolute URI")
    if separator and (not allow_version or not version):
        raise ValueError("canonical profile version is malformed")
    return canonical, version if separator else None


def _is_canonical_url(value: Any) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(character.isspace() for character in value):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "urn"}:
        return False
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    return bool(parsed.path)


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _looks_like_single_spec(mapping: Mapping[str, Any]) -> bool:
    return any(
        key in mapping
        for key in (
            "canonical_url",
            "canonicalUrl",
            "url",
            "profile_url",
            "profileUrl",
            "canonical",
            "canonical_urls",
            "resource_type",
            "resourceType",
            "fhir_versions",
            "fhir_version",
        )
    )


def _resource_type_for(
    canonical_url: str,
    resource_types: Mapping[str, str] | None,
    *,
    fallback: str,
) -> str:
    if resource_types is None:
        return fallback
    resource_type = resource_types.get(canonical_url)
    if resource_type is None:
        return fallback
    return _normalise_resource_type(resource_type)


def _default_root(resource: Mapping[str, Any]) -> str:
    resource_type = resource.get("resourceType")
    return (
        resource_type
        if isinstance(resource_type, str) and resource_type
        else "Resource"
    )


def _safe_expression(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expression must be a non-empty string")
    candidate = value.strip()
    if any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.[]"
        for character in candidate
    ):
        raise ValueError("expression contains unsupported characters")
    return candidate
