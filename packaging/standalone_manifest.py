"""Deterministic, offline manifest for the narrow local redactor.

The file is intentionally self-contained and uses only the Python standard
library.  It describes the small runtime component, its permissive license
metadata, optional integrations, and assets that must remain user-supplied.
Importing or rendering the manifest does not inspect the environment, resolve
entry points, install packages, load models, or make a network request.

The module can also be executed directly to print the canonical JSON manifest::

    python packaging/standalone_manifest.py

Only static, non-sensitive package metadata belongs in a manifest.  Validation
errors identify a field and a structural reason, never a caller-provided
value, so malformed metadata cannot echo source text into an exception.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar

MANIFEST_FORMAT: Final = "openmed-standalone-redactor"
MANIFEST_SCHEMA_VERSION: Final = "1.0"
STANDALONE_PACKAGE_NAME: Final = "openmed-redactor-standalone"
STANDALONE_PACKAGE_VERSION: Final = "2.2.0"
STANDALONE_LICENSE: Final = "Apache-2.0"
PLATFORM_ANY: Final = "any"
_MAX_TEXT_LENGTH: Final = 512
_MAX_PLATFORMS: Final = 16

# These are the only licenses admitted to the default bundle.  The list is
# deliberately explicit: an unknown license must not silently become part of a
# constrained distribution just because it looks familiar.
PERMISSIVE_LICENSES: Final = frozenset(
    {
        "Apache-2.0",
        "BSD-3-Clause",
        "MIT",
        "PSF-2.0",
    }
)
RESTRICTED_LICENSES: Final = frozenset(
    {
        "DUA-restricted",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "Proprietary",
        "source-available",
    }
)
_REQUIRED_REQUIREMENTS: Final = {
    "faker": "faker>=22.0",
    "jieba": "jieba>=0.42.1,<0.43",
    "pysbd": "pysbd>=0.3.4,<0.4",
    "pyyaml": "pyyaml>=6.0",
}


def _text(value: object, field_name: str) -> str:
    """Normalize required metadata without echoing the supplied value."""

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_TEXT_LENGTH
        or not normalized.isprintable()
    ):
        raise ValueError(f"{field_name} must contain safe printable text")
    return normalized


def _platforms(value: Sequence[str]) -> tuple[str, ...]:
    """Return a stable, detached tuple of normalized platform tags."""

    if type(value) is not tuple:
        raise TypeError("platforms must be a tuple of strings")
    if len(value) > _MAX_PLATFORMS:
        raise ValueError("platforms contains too many entries")
    normalized = tuple(_text(item, "platform") for item in value)
    if not normalized:
        raise ValueError("platforms must not be empty")
    return tuple(sorted({item.casefold() for item in normalized}))


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Static metadata for one component in the standalone distribution."""

    name: str
    version: str
    license: str
    purpose: str
    platforms: tuple[str, ...] = (PLATFORM_ANY,)
    network_egress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "component name"))
        object.__setattr__(self, "version", _text(self.version, "component version"))
        object.__setattr__(self, "license", _text(self.license, "component license"))
        object.__setattr__(self, "purpose", _text(self.purpose, "component purpose"))
        object.__setattr__(self, "platforms", _platforms(self.platforms))
        if type(self.network_egress) is not bool:
            raise TypeError("component network_egress must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-compatible component metadata."""

        return {
            "license": self.license,
            "name": self.name,
            "network_egress": self.network_egress,
            "platforms": list(self.platforms),
            "purpose": self.purpose,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class DependencySpec:
    """Static metadata for a runtime, optional, or restricted dependency."""

    name: str
    requirement: str
    license: str
    purpose: str
    platforms: tuple[str, ...] = (PLATFORM_ANY,)
    network_egress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "dependency name"))
        object.__setattr__(
            self,
            "requirement",
            _text(self.requirement, "dependency requirement"),
        )
        object.__setattr__(self, "license", _text(self.license, "dependency license"))
        object.__setattr__(self, "purpose", _text(self.purpose, "dependency purpose"))
        object.__setattr__(self, "platforms", _platforms(self.platforms))
        if type(self.network_egress) is not bool:
            raise TypeError("dependency network_egress must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-compatible dependency metadata."""

        return {
            "license": self.license,
            "name": self.name,
            "network_egress": self.network_egress,
            "platforms": list(self.platforms),
            "purpose": self.purpose,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class RestrictedAssetSpec:
    """Metadata for an asset that is explicitly excluded from the package."""

    name: str
    license: str
    purpose: str
    source: str = "user-supplied local asset"
    platforms: tuple[str, ...] = (PLATFORM_ANY,)
    bundled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "asset name"))
        object.__setattr__(self, "license", _text(self.license, "asset license"))
        object.__setattr__(self, "purpose", _text(self.purpose, "asset purpose"))
        object.__setattr__(self, "source", _text(self.source, "asset source"))
        object.__setattr__(self, "platforms", _platforms(self.platforms))
        if type(self.bundled) is not bool:
            raise TypeError("asset bundled must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-compatible excluded-asset metadata."""

        return {
            "bundled": self.bundled,
            "license": self.license,
            "name": self.name,
            "platforms": list(self.platforms),
            "purpose": self.purpose,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    """A privacy-safe structural issue in a standalone manifest."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


class ManifestValidationError(ValueError):
    """Raised when a standalone manifest fails its structural safety checks."""

    def __init__(self, issues: Iterable[ManifestIssue]) -> None:
        self.issues = tuple(issues)
        if self.issues:
            detail = "; ".join(str(issue) for issue in self.issues)
            message = f"standalone manifest validation failed: {detail}"
        else:
            message = "standalone manifest validation failed"
        # ``ManifestIssue`` contains only fixed field paths and safe reasons;
        # never interpolate package-provided values into this exception.
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class StandalonePackageManifest:
    """Immutable manifest for a platform-neutral local redaction bundle.

    ``required_dependencies`` is the complete default install set.  Optional
    and restricted entries are descriptive only and are never returned by
    :meth:`default_dependencies`; they remain visible in the full manifest so
    the install boundary is auditable.
    """

    name: str
    version: str
    license: str
    components: tuple[ComponentSpec, ...]
    required_dependencies: tuple[DependencySpec, ...]
    optional_dependencies: tuple[DependencySpec, ...]
    restricted_dependencies: tuple[DependencySpec, ...]
    restricted_assets: tuple[RestrictedAssetSpec, ...]
    python_requires: str = ">=3.10"
    platforms: tuple[str, ...] = (PLATFORM_ANY,)
    network_egress: bool = False
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "package name"))
        object.__setattr__(self, "version", _text(self.version, "package version"))
        object.__setattr__(self, "license", _text(self.license, "package license"))
        object.__setattr__(
            self,
            "python_requires",
            _text(self.python_requires, "python requirement"),
        )
        object.__setattr__(self, "platforms", _platforms(self.platforms))
        object.__setattr__(
            self, "schema_version", _text(self.schema_version, "schema version")
        )
        if type(self.network_egress) is not bool:
            raise TypeError("manifest network_egress must be a boolean")

        collection_types = {
            "components": ComponentSpec,
            "required_dependencies": DependencySpec,
            "optional_dependencies": DependencySpec,
            "restricted_dependencies": DependencySpec,
            "restricted_assets": RestrictedAssetSpec,
        }
        for field_name, expected_type in collection_types.items():
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise TypeError(f"{field_name} must be a tuple")
            if any(type(entry) is not expected_type for entry in values):
                raise TypeError(f"{field_name} contains an unsupported entry")
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(values, key=_entry_name)),
            )

    def default_dependencies(self) -> tuple[DependencySpec, ...]:
        """Return exactly the dependencies permitted in the default bundle."""

        return self.required_dependencies

    def default_dependency_names(self) -> tuple[str, ...]:
        """Return stable distribution names in the default bundle."""

        return tuple(dependency.name for dependency in self.default_dependencies())

    def default_requirements(self) -> tuple[str, ...]:
        """Return stable requirement strings for the default bundle."""

        return tuple(
            dependency.requirement for dependency in self.default_dependencies()
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached, deterministic JSON-compatible manifest."""

        return {
            "components": [component.to_dict() for component in self.components],
            "dependencies": {
                "optional": [
                    dependency.to_dict() for dependency in self.optional_dependencies
                ],
                "required": [
                    dependency.to_dict() for dependency in self.required_dependencies
                ],
                "restricted": [
                    dependency.to_dict() for dependency in self.restricted_dependencies
                ],
            },
            "format": MANIFEST_FORMAT,
            "license": self.license,
            "name": self.name,
            "network_egress": self.network_egress,
            "platforms": list(self.platforms),
            "python_requires": self.python_requires,
            "restricted_assets": [asset.to_dict() for asset in self.restricted_assets],
            "schema_version": self.schema_version,
            "version": self.version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this manifest with stable keys and a trailing newline."""

        assert_valid_manifest(self)
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )


def _entry_name(
    entry: ComponentSpec | DependencySpec | RestrictedAssetSpec,
) -> str:
    """Return a stable sort key for one known manifest entry."""

    return entry.name.casefold()


def _issue(path: str, reason: str) -> ManifestIssue:
    """Build a privacy-safe issue with no caller-provided value."""

    return ManifestIssue(path=path, reason=reason)


def _validate_license(
    license_name: str,
    path: str,
    issues: list[ManifestIssue],
    *,
    allow_restricted: bool = False,
) -> None:
    """Validate a license identifier without echoing it in an issue."""

    if license_name in PERMISSIVE_LICENSES:
        return
    if allow_restricted and license_name in RESTRICTED_LICENSES:
        return
    issues.append(_issue(path, "license is not in the approved manifest policy"))


class _NamedPlatformEntry(Protocol):
    """Structural fields shared by manifest entry records."""

    @property
    def name(self) -> str:
        """Return the entry's normalized distribution name."""

        ...

    @property
    def platforms(self) -> tuple[str, ...]:
        """Return the entry's normalized platform tags."""

        ...


_EntryT = TypeVar("_EntryT", bound=_NamedPlatformEntry)


def _validate_entry_collection(
    entries: Sequence[object],
    path: str,
    issues: list[ManifestIssue],
    expected_type: type[_EntryT],
) -> set[str]:
    """Validate entry types and return normalized names for overlap checks."""

    names: set[str] = set()
    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, expected_type):
            issues.append(_issue(entry_path, "entry has an unsupported type"))
            continue
        normalized_name = entry.name.casefold()
        if normalized_name in names:
            issues.append(_issue(entry_path, "duplicate entry"))
        names.add(normalized_name)
        if entry.platforms != (PLATFORM_ANY,):
            issues.append(_issue(entry_path, "entry must be platform-neutral"))
    return names


def validate_manifest(
    manifest: StandalonePackageManifest,
) -> tuple[ManifestIssue, ...]:
    """Return privacy-safe structural issues for *manifest*.

    The function is pure: it only examines the supplied immutable metadata and
    performs no package discovery, filesystem access, or network operation.
    """

    if type(manifest) is not StandalonePackageManifest:
        return (_issue("manifest", "value has an unsupported type"),)

    issues: list[ManifestIssue] = []
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        issues.append(_issue("schema_version", "unsupported schema version"))
    if manifest.name != STANDALONE_PACKAGE_NAME:
        issues.append(_issue("name", "unsupported package name"))
    if manifest.version != STANDALONE_PACKAGE_VERSION:
        issues.append(_issue("version", "package version is out of sync"))
    if manifest.license != STANDALONE_LICENSE:
        issues.append(_issue("license", "package license is out of sync"))
    if manifest.python_requires != ">=3.10":
        issues.append(_issue("python_requires", "Python requirement is out of sync"))
    if manifest.platforms != (PLATFORM_ANY,):
        issues.append(_issue("platforms", "manifest must be platform-neutral"))
    if manifest.network_egress:
        issues.append(_issue("network_egress", "default manifest must be offline"))
    _validate_license(manifest.license, "license", issues)

    component_names = _validate_entry_collection(
        manifest.components,
        "components",
        issues,
        ComponentSpec,
    )
    if not component_names:
        issues.append(_issue("components", "at least one component is required"))
    for index, component in enumerate(manifest.components):
        if isinstance(component, ComponentSpec):
            if component.network_egress:
                issues.append(
                    _issue(
                        f"components[{index}].network_egress",
                        "default component must be offline",
                    )
                )
            _validate_license(
                component.license,
                f"components[{index}].license",
                issues,
            )
            if component.version != manifest.version:
                issues.append(
                    _issue(
                        f"components[{index}].version",
                        "component version differs from package version",
                    )
                )

    required_names = _validate_entry_collection(
        manifest.required_dependencies,
        "dependencies.required",
        issues,
        DependencySpec,
    )
    optional_names = _validate_entry_collection(
        manifest.optional_dependencies,
        "dependencies.optional",
        issues,
        DependencySpec,
    )
    restricted_names = _validate_entry_collection(
        manifest.restricted_dependencies,
        "dependencies.restricted",
        issues,
        DependencySpec,
    )

    if required_names & (optional_names | restricted_names):
        issues.append(_issue("dependencies", "default and opt-in entries overlap"))
    if optional_names & restricted_names:
        issues.append(_issue("dependencies", "optional and restricted entries overlap"))

    required_requirements = {
        dependency.name.casefold(): dependency.requirement
        for dependency in manifest.required_dependencies
        if type(dependency) is DependencySpec
    }
    if required_requirements != _REQUIRED_REQUIREMENTS:
        issues.append(
            _issue(
                "dependencies.required",
                "default requirements differ from the approved project boundary",
            )
        )

    for index, dependency in enumerate(manifest.required_dependencies):
        if not isinstance(dependency, DependencySpec):
            continue
        dependency_path = f"dependencies.required[{index}]"
        _validate_license(dependency.license, f"{dependency_path}.license", issues)
        if dependency.network_egress:
            issues.append(
                _issue(
                    f"{dependency_path}.network_egress",
                    "default dependency must be offline",
                )
            )

    for index, dependency in enumerate(manifest.optional_dependencies):
        if isinstance(dependency, DependencySpec):
            _validate_license(
                dependency.license,
                f"dependencies.optional[{index}].license",
                issues,
            )

    for index, dependency in enumerate(manifest.restricted_dependencies):
        if isinstance(dependency, DependencySpec):
            _validate_license(
                dependency.license,
                f"dependencies.restricted[{index}].license",
                issues,
                allow_restricted=True,
            )

    _validate_entry_collection(
        manifest.restricted_assets,
        "restricted_assets",
        issues,
        RestrictedAssetSpec,
    )
    for index, asset in enumerate(manifest.restricted_assets):
        if isinstance(asset, RestrictedAssetSpec):
            asset_path = f"restricted_assets[{index}]"
            if asset.bundled:
                issues.append(_issue(f"{asset_path}.bundled", "asset must be excluded"))
            _validate_license(
                asset.license,
                f"{asset_path}.license",
                issues,
                allow_restricted=True,
            )

    return tuple(issues)


def assert_valid_manifest(manifest: StandalonePackageManifest) -> None:
    """Raise a privacy-safe error when *manifest* is structurally unsafe."""

    issues = validate_manifest(manifest)
    if issues:
        raise ManifestValidationError(issues)


def get_standalone_manifest() -> StandalonePackageManifest:
    """Return the immutable canonical manifest without touching external state."""

    return STANDALONE_MANIFEST


def render_manifest(manifest: StandalonePackageManifest | None = None) -> str:
    """Return canonical JSON for *manifest*, defaulting to the local manifest."""

    if manifest is None:
        target = STANDALONE_MANIFEST
    elif type(manifest) is StandalonePackageManifest:
        target = manifest
    else:
        raise ManifestValidationError(
            (_issue("manifest", "value has an unsupported type"),)
        )
    return target.to_json()


_REQUIRED_DEPENDENCIES: Final = (
    DependencySpec(
        name="faker",
        requirement=_REQUIRED_REQUIREMENTS["faker"],
        license="MIT",
        purpose="Deterministic local surrogate generation when explicitly selected.",
    ),
    DependencySpec(
        name="jieba",
        requirement=_REQUIRED_REQUIREMENTS["jieba"],
        license="MIT",
        purpose="Local sentence and token handling for supported Chinese text.",
    ),
    DependencySpec(
        name="pysbd",
        requirement=_REQUIRED_REQUIREMENTS["pysbd"],
        license="MIT",
        purpose="Local sentence boundary detection.",
    ),
    DependencySpec(
        name="pyyaml",
        requirement=_REQUIRED_REQUIREMENTS["pyyaml"],
        license="MIT",
        purpose="Reading local policy configuration.",
    ),
)

_OPTIONAL_DEPENDENCIES: Final = (
    DependencySpec(
        name="huggingface-hub",
        requirement="huggingface-hub>=0.30",
        license="Apache-2.0",
        purpose="Explicitly user-managed model acquisition; never used by default.",
        network_egress=True,
    ),
    DependencySpec(
        name="presidio-analyzer",
        requirement="presidio-analyzer>=2.2.354,<3",
        license="MIT",
        purpose="Optional local interoperability adapter.",
    ),
    DependencySpec(
        name="spacy",
        requirement="spacy>=3.8.9",
        license="MIT",
        purpose="Optional local interoperability adapter.",
    ),
    DependencySpec(
        name="torch",
        requirement="torch>=2.0",
        license="BSD-3-Clause",
        purpose="Optional local inference backend; never installed by the default bundle.",
    ),
    DependencySpec(
        name="transformers",
        requirement="transformers>=4.50",
        license="Apache-2.0",
        purpose="Optional local model runtime; never installed by the default bundle.",
        network_egress=True,
    ),
)

_RESTRICTED_DEPENDENCIES: Final = (
    DependencySpec(
        name="extract-msg",
        requirement="extract-msg>=0.56,<0.57",
        license="GPL-3.0-only",
        purpose="Subprocess-only Outlook parser requiring explicit installation.",
    ),
    DependencySpec(
        name="sdcMicro",
        requirement="external R package, not a Python dependency",
        license="GPL-2.0-or-later",
        purpose="Subprocess-only bridge requiring an explicit user installation.",
    ),
)

_RESTRICTED_ASSETS: Final = (
    RestrictedAssetSpec(
        name="CPT",
        license="Proprietary",
        purpose="Restricted clinical terminology; never bundled.",
    ),
    RestrictedAssetSpec(
        name="i2b2",
        license="DUA-restricted",
        purpose="Restricted clinical corpus; never bundled.",
    ),
    RestrictedAssetSpec(
        name="MIMIC",
        license="DUA-restricted",
        purpose="Restricted clinical corpus; never bundled.",
    ),
    RestrictedAssetSpec(
        name="n2c2",
        license="DUA-restricted",
        purpose="Restricted clinical corpus; never bundled.",
    ),
    RestrictedAssetSpec(
        name="SNOMED CT",
        license="Proprietary",
        purpose="Restricted clinical terminology; never bundled.",
    ),
    RestrictedAssetSpec(
        name="UMLS",
        license="DUA-restricted",
        purpose="Restricted terminology distribution; never bundled.",
    ),
)

STANDALONE_MANIFEST: Final = StandalonePackageManifest(
    name=STANDALONE_PACKAGE_NAME,
    version=STANDALONE_PACKAGE_VERSION,
    license=STANDALONE_LICENSE,
    components=(
        ComponentSpec(
            name="local-redactor",
            version=STANDALONE_PACKAGE_VERSION,
            license=STANDALONE_LICENSE,
            purpose="Deterministic, local text redaction with privacy-safe outputs.",
        ),
    ),
    required_dependencies=_REQUIRED_DEPENDENCIES,
    optional_dependencies=_OPTIONAL_DEPENDENCIES,
    restricted_dependencies=_RESTRICTED_DEPENDENCIES,
    restricted_assets=_RESTRICTED_ASSETS,
)

# Descriptive aliases make the intended install boundary easy to discover for
# callers that use "default manifest" terminology.
DEFAULT_MANIFEST: Final = STANDALONE_MANIFEST
DEFAULT_DEPENDENCIES: Final = STANDALONE_MANIFEST.required_dependencies
OPTIONAL_DEPENDENCIES: Final = STANDALONE_MANIFEST.optional_dependencies
RESTRICTED_DEPENDENCIES: Final = STANDALONE_MANIFEST.restricted_dependencies
RESTRICTED_ASSETS: Final = STANDALONE_MANIFEST.restricted_assets

assert_valid_manifest(STANDALONE_MANIFEST)


__all__ = [
    "ComponentSpec",
    "DEFAULT_DEPENDENCIES",
    "DEFAULT_MANIFEST",
    "DependencySpec",
    "MANIFEST_FORMAT",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestIssue",
    "ManifestValidationError",
    "OPTIONAL_DEPENDENCIES",
    "PERMISSIVE_LICENSES",
    "PLATFORM_ANY",
    "RESTRICTED_ASSETS",
    "RESTRICTED_DEPENDENCIES",
    "RESTRICTED_LICENSES",
    "RestrictedAssetSpec",
    "STANDALONE_LICENSE",
    "STANDALONE_MANIFEST",
    "STANDALONE_PACKAGE_NAME",
    "STANDALONE_PACKAGE_VERSION",
    "StandalonePackageManifest",
    "assert_valid_manifest",
    "get_standalone_manifest",
    "render_manifest",
    "validate_manifest",
]


if __name__ == "__main__":
    print(render_manifest(), end="")
