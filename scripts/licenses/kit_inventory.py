#!/usr/bin/env python3
"""Gate OpenMedKit Swift, Android, and JavaScript dependency licenses offline.

The portable kits declare dependencies in SwiftPM manifests and resolved pins,
Gradle settings, version catalogs and module build files, npm lockfiles, a
Tauri ``Cargo.lock``, a Flutter ``pubspec`` and CocoaPods podspecs.  This gate
parses those committed files with the standard library only, maps every
declared package to the reviewed SPDX expression in
``docs/security/kit-license-inventory.md`` and fails closed when a package is
unreviewed, unknown, copyleft, source-available or proprietary without a
scoped, documented exception.

The gate never resolves packages, runs a build tool or contacts a registry.
Reports carry ecosystems, package names, repository-relative manifest paths,
usage classes and fixed decision codes; never license text, versions, URLs or
local paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _load_python_inventory() -> ModuleType:
    """Load the sibling Python inventory so both gates share one policy."""

    module_name = "_openmed_license_inventory"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().with_name("inventory.py")
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging guard
        raise ImportError("the Python license inventory module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_python_inventory = _load_python_inventory()
InventoryError = _python_inventory.InventoryError
LicenseClass = _python_inventory.LicenseClass
PERMISSIVE_LICENSES: frozenset[str] = _python_inventory.PERMISSIVE_LICENSES
classify_license = _python_inventory.classify_license
safe_identifier = _python_inventory.safe_identifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KIT_INVENTORY = ROOT / "docs/security/kit-license-inventory.md"

# Committed kit manifests.  Android modules, the Gradle version catalog and the
# root build file are discovered from ``settings.gradle.kts``; sibling
# ``package.json`` and ``Cargo.toml`` files are read with their lockfiles.  A
# committed manifest outside this registry fails the registry unit test.
DEFAULT_KIT_MANIFESTS: tuple[str, ...] = (
    "Package.swift",
    "Package.resolved",
    "swift/OpenMedKit/Package.swift",
    "swift/OpenMedDemo/OpenMedDemo.xcodeproj/project.xcworkspace/xcshareddata/"
    "swiftpm/Package.resolved",
    "swift/OpenMedScanDemo/OpenMedScanDemo.xcodeproj/project.xcworkspace/"
    "xcshareddata/swiftpm/Package.resolved",
    "android/settings.gradle.kts",
    "js/openmedkit-electron/package-lock.json",
    "js/openmedkit-react-native/package-lock.json",
    "js/openmedkit-tauri/package-lock.json",
    "js/openmedkit-web/package-lock.json",
    "js/openmedkit-tauri/src-tauri/Cargo.lock",
    "js/openmedkit-flutter/pubspec.yaml",
    "js/openmedkit-flutter/ios/openmedkit_flutter.podspec",
    "js/openmedkit-flutter/macos/openmedkit_flutter.podspec",
    "js/openmedkit-flutter/android/build.gradle",
)

ECOSYSTEMS = frozenset(
    {"cargo", "cocoapods", "gradle", "gradle-plugin", "npm", "pub", "swiftpm"}
)

# Development-only tooling never ships in a kit artifact, so these reviewed
# expressions are accepted for it.  Shipped runtime dependencies still need a
# permissive expression or a scoped exception.
DEVELOPMENT_ONLY_LICENSES = frozenset(
    {"epl-1-0", "epl-2-0", "licenseref-public-domain"}
)

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_INVENTORY_BYTES = 4 * 1024 * 1024
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9@_][A-Za-z0-9@/._:+-]{0,213}$")
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


class Usage:
    """How a kit dependency reaches users."""

    RUNTIME = "runtime"
    DEVELOPMENT = "development"


class Decision:
    """Stable, value-free gate decisions emitted in reports."""

    PERMISSIVE = "permissive"
    DEVELOPMENT_ONLY = "development-only"
    SCOPED_EXCEPTION = "scoped-exception"
    UNREVIEWED = "unreviewed"
    DENIED = "denied"


FAILING_DECISIONS = frozenset({Decision.UNREVIEWED, Decision.DENIED})


class KitManifestError(InventoryError):
    """Raised when a registered kit manifest cannot be interpreted safely.

    The message is a fixed reason plus the repository-relative manifest path.
    It never includes manifest content, versions, URLs or local paths.
    """

    def __init__(self, reason: str, manifest: str) -> None:
        self.reason = reason
        self.manifest = manifest
        super().__init__(f"{reason}: {manifest}")


@dataclass(frozen=True)
class KitDependency:
    """One dependency declared, pinned or locked by a kit manifest."""

    ecosystem: str
    name: str
    usage: str
    manifest: str
    direct: bool = True


@dataclass(frozen=True)
class KitReviewEntry:
    """One reviewed license row from the committed kit inventory."""

    ecosystem: str
    name: str
    license_expression: str
    line_number: int | None = None


@dataclass(frozen=True)
class KitLicenseException:
    """A documented exception for one restricted package in named manifests."""

    exception_id: str
    ecosystem: str
    name: str
    license_expression: str
    manifests: frozenset[str]

    def covers(self, dependency: KitDependency, license_expression: str) -> bool:
        """Return whether this exception covers one dependency occurrence."""

        return (
            dependency.ecosystem == self.ecosystem
            and dependency.name == self.name
            and dependency.manifest in self.manifests
            and license_expression.strip().casefold()
            == self.license_expression.casefold()
        )


# Google ML Kit Text Recognition ships under Google's proprietary ML Kit terms.
# It is allowed only as the optional on-device OCR adapter: the dedicated
# ``:openmedkit-mlkit`` module (#4451), the scan demo that exercises it, and,
# until #4451 lands, the current core module path.  Remove the core path from
# this set once the core AAR no longer depends on ML Kit.
SCOPED_KIT_EXCEPTIONS: tuple[KitLicenseException, ...] = (
    KitLicenseException(
        exception_id="mlkit-optional-ocr",
        ecosystem="gradle",
        name="com.google.mlkit:text-recognition",
        license_expression="LicenseRef-Proprietary-Google-ML-Kit-Terms",
        manifests=frozenset(
            {
                "android/openmedkit-mlkit/build.gradle.kts",
                "android/openmedkit/build.gradle.kts",
                "android/OpenMedScanDemo/app/build.gradle.kts",
            }
        ),
    ),
)


@dataclass(frozen=True)
class KitRecord:
    """A dependency occurrence plus its fail-closed license decision."""

    dependency: KitDependency
    classification: str
    decision: str
    exception_id: str = ""

    @property
    def failed(self) -> bool:
        """Return whether this record fails the gate."""

        return self.decision in FAILING_DECISIONS

    def to_safe_dict(self) -> dict[str, Any]:
        """Return report data without license text, versions or local paths."""

        payload: dict[str, Any] = {
            "classification": self.classification,
            "decision": self.decision,
            "direct": self.dependency.direct,
            "ecosystem": self.dependency.ecosystem,
            "manifest": self.dependency.manifest,
            "package": safe_identifier(self.dependency.name),
            "usage": self.dependency.usage,
        }
        if self.exception_id:
            payload["exception"] = self.exception_id
        return payload


@dataclass
class KitDiscovery:
    """Dependencies found in the registered kit manifests."""

    dependencies: list[KitDependency] = field(default_factory=list)
    counted_development: Counter[str] = field(default_factory=Counter)
    manifests_read: set[str] = field(default_factory=set)

    def add(
        self,
        ecosystem: str,
        name: str,
        usage: str,
        manifest: str,
        *,
        direct: bool = True,
    ) -> None:
        """Record one dependency occurrence after validating its identifier."""

        if not _PACKAGE_NAME_RE.fullmatch(name):
            raise KitManifestError(
                "dependency name is not a package identifier", manifest
            )
        self.dependencies.append(
            KitDependency(ecosystem, name.casefold(), usage, manifest, direct)
        )

    def count_development(self, ecosystem: str) -> None:
        """Count a transitive development-only lock entry without reviewing it."""

        self.counted_development[ecosystem] += 1


@dataclass(frozen=True)
class KitAudit:
    """The complete kit audit result."""

    records: tuple[KitRecord, ...]
    counted_development: Mapping[str, int]
    unused_review_rows: int

    @property
    def failures(self) -> tuple[KitRecord, ...]:
        """Return records that fail the gate."""

        return tuple(record for record in self.records if record.failed)

    @property
    def exceptions(self) -> tuple[KitRecord, ...]:
        """Return records allowed only by a scoped exception."""

        return tuple(
            record
            for record in self.records
            if record.decision == Decision.SCOPED_EXCEPTION
        )


# --------------------------------------------------------------------------
# Shared text helpers
# --------------------------------------------------------------------------


def _relative(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise InventoryError("kit manifest path must be repository-relative")
    return pure.as_posix()


def _read_text(root: Path, manifest: str, *, limit: int = MAX_MANIFEST_BYTES) -> str:
    path = root / manifest
    try:
        if path.stat().st_size > limit:
            raise KitManifestError("kit manifest exceeds the size limit", manifest)
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise KitManifestError("could not read a kit manifest", manifest) from None


def _load_json(root: Path, manifest: str) -> Any:
    text = _read_text(root, manifest)
    try:
        return json.loads(text)
    except ValueError:
        raise KitManifestError("kit manifest is not valid JSON", manifest) from None


def _load_toml(root: Path, manifest: str) -> dict[str, Any]:
    text = _read_text(root, manifest)
    try:
        return tomllib.loads(text)
    except ValueError:
        raise KitManifestError("kit manifest is not valid TOML", manifest) from None


def _strip_comments(text: str, *, hash_comments: bool = False) -> str:
    """Blank ``//`` and ``/* */`` (or Ruby ``#``) comments outside strings."""

    output: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "\"'":
            end = _string_end(text, index)
            output.append(text[index:end])
            index = end
            continue
        if hash_comments and char == "#":
            newline = text.find("\n", index)
            index = length if newline == -1 else newline
            continue
        if not hash_comments and text.startswith("//", index):
            newline = text.find("\n", index)
            index = length if newline == -1 else newline
            continue
        if not hash_comments and text.startswith("/*", index):
            close = text.find("*/", index + 2)
            end = length if close == -1 else close + 2
            output.append("\n" * text.count("\n", index, end))
            index = end
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _string_end(text: str, start: int) -> int:
    """Return the index just past the string literal opened at ``start``."""

    quote = text[start]
    triple = quote * 3
    if text.startswith(triple, start):
        close = text.find(triple, start + 3)
        return len(text) if close == -1 else close + 3
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote or char == "\n":
            return index + 1
        index += 1
    return len(text)


def _string_literals(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, value)`` for each quoted literal in comment-free text."""

    index = 0
    while index < len(text):
        if text[index] in "\"'":
            end = _string_end(text, index)
            width = 3 if text.startswith(text[index] * 3, index) else 1
            yield index, end, text[index + width : max(index + width, end - width)]
            index = end
            continue
        index += 1


def _blank_strings(text: str) -> str:
    """Replace string literal contents with spaces, keeping offsets stable."""

    characters = list(text)
    for start, end, _value in _string_literals(text):
        for position in range(start + 1, max(start + 1, end - 1)):
            if characters[position] != "\n":
                characters[position] = " "
    return "".join(characters)


def _balanced_end(text: str, open_index: int, opening: str, closing: str) -> int:
    """Return the index of the bracket closing ``text[open_index]``."""

    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _string_end(text, index)
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return len(text)


def _blocks(text: str, keyword: str) -> Iterator[tuple[int, str]]:
    """Yield ``(start, body)`` for every ``keyword { ... }`` block."""

    blanked = _blank_strings(text)
    for match in re.finditer(rf"\b{re.escape(keyword)}\s*\{{", blanked):
        open_index = match.end() - 1
        close_index = _balanced_end(text, open_index, "{", "}")
        yield match.start(), text[open_index + 1 : close_index]


def _calls(text: str, function: str) -> Iterator[str]:
    """Yield argument text for every ``.function(...)`` call in Swift source."""

    blanked = _blank_strings(text)
    for match in re.finditer(rf"\.{re.escape(function)}\s*\(", blanked):
        open_index = match.end() - 1
        close_index = _balanced_end(text, open_index, "(", ")")
        yield text[open_index + 1 : close_index]


# --------------------------------------------------------------------------
# SwiftPM
# --------------------------------------------------------------------------

_SWIFT_LABEL_STRING_RE = re.compile(
    r'\b(?P<label>url|path|id|name)\s*:\s*"(?P<value>[^"]*)"'
)
_SWIFT_PACKAGE_REF_RE = re.compile(r'\bpackage\s*:\s*"(?P<value>[^"]*)"')
_SWIFT_TARGET_KINDS = (
    "target",
    "executableTarget",
    "testTarget",
    "plugin",
    "macro",
    "systemLibrary",
)


def _swift_identity(location: str) -> str:
    """Return SwiftPM's package identity for a repository URL or path."""

    stripped = location.strip().rstrip("/")
    last = re.split(r"[/:]", stripped)[-1]
    if last.casefold().endswith(".git"):
        last = last[:-4]
    return last.casefold()


def _swift_manifest_dependencies(root: Path, manifest: str) -> dict[str, str]:
    """Return ``identity -> usage`` for remote packages in a ``Package.swift``."""

    text = _strip_comments(_read_text(root, manifest))
    declared: dict[str, str] = {}
    for arguments in _calls(text, "package"):
        labels = {
            match["label"]: match["value"]
            for match in _SWIFT_LABEL_STRING_RE.finditer(arguments)
        }
        if "url" in labels:
            declared[_swift_identity(labels["url"])] = Usage.RUNTIME
        elif "id" in labels:
            declared[labels["id"].casefold()] = Usage.RUNTIME
        elif "path" not in labels:
            raise KitManifestError(
                "SwiftPM package declaration is not static", manifest
            )

    references: dict[str, set[str]] = {}
    for kind in _SWIFT_TARGET_KINDS:
        usage = Usage.DEVELOPMENT if kind == "testTarget" else Usage.RUNTIME
        for arguments in _calls(text, kind):
            for match in _SWIFT_PACKAGE_REF_RE.finditer(arguments):
                references.setdefault(match["value"].casefold(), set()).add(usage)

    for arguments in _calls(text, "binaryTarget"):
        labels = {
            match["label"]: match["value"]
            for match in _SWIFT_LABEL_STRING_RE.finditer(arguments)
        }
        if "url" in labels and "name" in labels:
            declared[f"binary-target:{labels['name'].casefold()}"] = Usage.RUNTIME

    for identity in list(declared):
        usages = references.get(identity)
        # Unreferenced packages stay runtime: SwiftPM still resolves them.
        if usages and usages == {Usage.DEVELOPMENT}:
            declared[identity] = Usage.DEVELOPMENT
    return declared


def _parse_swift_manifest(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    for identity, usage in sorted(_swift_manifest_dependencies(root, manifest).items()):
        discovery.add("swiftpm", identity, usage, manifest)


def _parse_swift_resolved(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    payload = _load_json(root, manifest)
    if not isinstance(payload, Mapping):
        raise KitManifestError("SwiftPM resolved file has no pins", manifest)
    pins = payload.get("pins")
    if pins is None and isinstance(payload.get("object"), Mapping):
        pins = payload["object"].get("pins")
    if not isinstance(pins, list):
        raise KitManifestError("SwiftPM resolved file has no pins", manifest)

    sibling = PurePosixPath(manifest).with_name("Package.swift").as_posix()
    direct: dict[str, str] = {}
    if (root / sibling).is_file():
        direct = _swift_manifest_dependencies(root, sibling)
        discovery.manifests_read.add(sibling)

    for pin in pins:
        if not isinstance(pin, Mapping):
            raise KitManifestError("SwiftPM pin is not an object", manifest)
        kind = pin.get("kind", "remoteSourceControl")
        if kind in {"fileSystem", "localSourceControl"}:
            continue
        identity = pin.get("identity")
        if not isinstance(identity, str) or not identity:
            location = pin.get("location", pin.get("repositoryURL"))
            if not isinstance(location, str) or not location:
                raise KitManifestError("SwiftPM pin has no identity", manifest)
            identity = _swift_identity(location)
        identity = identity.casefold()
        discovery.add(
            "swiftpm",
            identity,
            direct.get(identity, Usage.RUNTIME),
            manifest,
            direct=identity in direct,
        )


# --------------------------------------------------------------------------
# Gradle
# --------------------------------------------------------------------------

_GRADLE_COORDINATE_RE = re.compile(
    r"^(?P<group>[A-Za-z0-9_][A-Za-z0-9_.-]*):(?P<artifact>[A-Za-z0-9_][A-Za-z0-9_.-]*)"
    r"(?::[^:@\s]*)?(?::[^:@\s]+)?(?:@[A-Za-z0-9]+)?$"
)
_GRADLE_COORDINATE_LIKE_RE = re.compile(r"^[^\s:/]+:[^\s:/]+(?::\S*)?$")
_GRADLE_CONFIG_PREFIX_RE = re.compile(
    r"(?P<config>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*)?"
    r"(?:(?:platform|enforcedPlatform|testFixtures)\s*\(\s*)?\Z"
)
_GRADLE_CATALOG_ACCESSOR_RE = re.compile(
    r"(?<![A-Za-z0-9_.])libs\."
    r"(?P<accessor>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)"
)
_GRADLE_KOTLIN_MODULE_RE = re.compile(
    r"\bkotlin\s*\(\s*[\"'](?P<module>[A-Za-z0-9_.-]+)[\"']"
)
_GRADLE_PLUGIN_ID_RE = re.compile(r"\bid\s*\(?\s*[\"'](?P<id>[A-Za-z0-9_.-]+)[\"']")
_GRADLE_APPLY_PLUGIN_RE = re.compile(
    r"\bapply\s*\(?\s*plugin\s*[:=]\s*[\"'](?P<id>[A-Za-z0-9_.-]+)[\"']"
)
_GRADLE_USE_MODULE_RE = re.compile(
    r"\buseModule\s*\(\s*[\"'](?P<coordinate>[^\"'\n]+)[\"']"
)
_GRADLE_INCLUDE_RE = re.compile(
    r"\binclude(?![A-Za-z0-9_])\s*\(?(?P<arguments>[^)\n]*)\)?"
)
_GRADLE_LOCAL_BINARY_RE = re.compile(r"\b(?:files|fileTree)\s*\(")
_GRADLE_DEVELOPMENT_CONFIGS = frozenset(
    {"annotationprocessor", "classpath", "detektplugins", "lintchecks", "lintpublish"}
)


def _gradle_usage(configuration: str) -> str:
    lowered = configuration.casefold()
    if lowered in _GRADLE_DEVELOPMENT_CONFIGS or lowered.startswith(
        ("androidtest", "debug", "kapt", "ksp", "test")
    ):
        return Usage.DEVELOPMENT
    return Usage.RUNTIME


def _gradle_alias_key(alias: str) -> str:
    return re.sub(r"[-_.]", ".", alias).casefold()


@dataclass
class _GradleCatalog:
    path: str
    libraries: dict[str, str] = field(default_factory=dict)
    plugins: dict[str, str] = field(default_factory=dict)
    bundles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    referenced_libraries: set[str] = field(default_factory=set)
    referenced_plugins: set[str] = field(default_factory=set)


def _gradle_coordinate(value: str, manifest: str) -> str:
    match = _GRADLE_COORDINATE_RE.fullmatch(value.strip())
    if match is None:
        raise KitManifestError(
            "Gradle dependency coordinate is not statically resolvable", manifest
        )
    return f"{match['group']}:{match['artifact']}"


def _parse_gradle_catalog(root: Path, manifest: str) -> _GradleCatalog:
    data = _load_toml(root, manifest)
    catalog = _GradleCatalog(manifest)
    libraries = data.get("libraries", {})
    plugins = data.get("plugins", {})
    bundles = data.get("bundles", {})
    if not all(isinstance(item, Mapping) for item in (libraries, plugins, bundles)):
        raise KitManifestError("Gradle version catalog table is malformed", manifest)

    for alias, value in libraries.items():
        if isinstance(value, str):
            coordinate = value
        elif isinstance(value, Mapping) and isinstance(value.get("module"), str):
            coordinate = value["module"]
        elif (
            isinstance(value, Mapping)
            and isinstance(value.get("group"), str)
            and isinstance(value.get("name"), str)
        ):
            coordinate = f"{value['group']}:{value['name']}"
        else:
            raise KitManifestError("Gradle catalog library is malformed", manifest)
        catalog.libraries[_gradle_alias_key(alias)] = _gradle_coordinate(
            coordinate, manifest
        )

    for alias, value in plugins.items():
        if isinstance(value, str):
            plugin_id = value.split(":", 1)[0]
        elif isinstance(value, Mapping) and isinstance(value.get("id"), str):
            plugin_id = value["id"]
        else:
            raise KitManifestError("Gradle catalog plugin is malformed", manifest)
        catalog.plugins[_gradle_alias_key(alias)] = plugin_id

    for alias, members in bundles.items():
        if not isinstance(members, list) or not all(
            isinstance(member, str) for member in members
        ):
            raise KitManifestError("Gradle catalog bundle is malformed", manifest)
        catalog.bundles[_gradle_alias_key(alias)] = tuple(
            _gradle_alias_key(member) for member in members
        )
    return catalog


def _gradle_configuration(text: str, position: int) -> str:
    match = _GRADLE_CONFIG_PREFIX_RE.search(text[max(0, position - 160) : position])
    return match["config"] if match else ""


def _add_plugin(discovery: KitDiscovery, plugin_id: str, manifest: str) -> None:
    # Unqualified ids are Gradle core plugins or legacy aliases whose artifact
    # is reviewed through its ``classpath`` coordinate.
    if "." in plugin_id:
        discovery.add("gradle-plugin", plugin_id, Usage.DEVELOPMENT, manifest)


def _parse_gradle_build(
    root: Path,
    manifest: str,
    discovery: KitDiscovery,
    catalog: _GradleCatalog | None,
) -> None:
    text = _strip_comments(_read_text(root, manifest))

    for _start, body in _blocks(text, "plugins"):
        blanked = _blank_strings(body)
        for match in _GRADLE_CATALOG_ACCESSOR_RE.finditer(blanked):
            parts = match["accessor"].split(".", 1)
            if parts[0] != "plugins" or len(parts) != 2:
                continue
            if catalog is None or parts[1].casefold() not in catalog.plugins:
                raise KitManifestError(
                    "Gradle catalog plugin alias is undeclared", manifest
                )
            catalog.referenced_plugins.add(parts[1].casefold())
            _add_plugin(discovery, catalog.plugins[parts[1].casefold()], manifest)
        for match in _GRADLE_PLUGIN_ID_RE.finditer(body):
            _add_plugin(discovery, match["id"], manifest)
        for match in _GRADLE_KOTLIN_MODULE_RE.finditer(body):
            _add_plugin(discovery, f"org.jetbrains.kotlin.{match['module']}", manifest)
    for match in _GRADLE_APPLY_PLUGIN_RE.finditer(text):
        _add_plugin(discovery, match["id"], manifest)
    for match in _GRADLE_USE_MODULE_RE.finditer(text):
        coordinate = _gradle_coordinate(match["coordinate"], manifest)
        discovery.add("gradle", coordinate, Usage.DEVELOPMENT, manifest)

    for _start, body in _blocks(text, "dependencies"):
        if _GRADLE_LOCAL_BINARY_RE.search(_blank_strings(body)):
            raise KitManifestError(
                "Gradle module declares a local binary dependency", manifest
            )
        for start, _end, value in _string_literals(body):
            if not _GRADLE_COORDINATE_LIKE_RE.fullmatch(value.strip()):
                continue
            coordinate = _gradle_coordinate(value, manifest)
            usage = _gradle_usage(_gradle_configuration(body, start))
            discovery.add("gradle", coordinate, usage, manifest)

        for match in _GRADLE_KOTLIN_MODULE_RE.finditer(body):
            usage = _gradle_usage(_gradle_configuration(body, match.start()))
            coordinate = f"org.jetbrains.kotlin:kotlin-{match['module']}"
            discovery.add("gradle", coordinate, usage, manifest)

        blanked = _blank_strings(body)
        for match in _GRADLE_CATALOG_ACCESSOR_RE.finditer(blanked):
            accessor = match["accessor"].casefold()
            if accessor.startswith(("versions.", "plugins.")):
                continue
            if catalog is None:
                raise KitManifestError(
                    "Gradle module uses an undeclared catalog", manifest
                )
            usage = _gradle_usage(_gradle_configuration(blanked, match.start()))
            if accessor.startswith("bundles."):
                bundle = catalog.bundles.get(accessor.removeprefix("bundles."))
                if bundle is None:
                    raise KitManifestError(
                        "Gradle catalog bundle alias is undeclared", manifest
                    )
                aliases: tuple[str, ...] = bundle
            else:
                aliases = (accessor,)
            for alias in aliases:
                coordinate = catalog.libraries.get(alias)
                if coordinate is None:
                    raise KitManifestError(
                        "Gradle catalog library alias is undeclared", manifest
                    )
                catalog.referenced_libraries.add(alias)
                discovery.add("gradle", coordinate, usage, manifest)


def _gradle_build_file(root: Path, directory: PurePosixPath) -> str | None:
    for name in ("build.gradle.kts", "build.gradle"):
        candidate = (directory / name).as_posix()
        if (root / candidate).is_file():
            return candidate
    return None


def _parse_gradle_settings(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    text = _strip_comments(_read_text(root, manifest))
    base = PurePosixPath(manifest).parent
    catalog_path = (base / "gradle" / "libs.versions.toml").as_posix()
    catalog: _GradleCatalog | None = None
    if (root / catalog_path).is_file():
        catalog = _parse_gradle_catalog(root, catalog_path)
        discovery.manifests_read.add(catalog_path)

    _parse_gradle_build(root, manifest, discovery, catalog)
    modules: list[PurePosixPath] = [base]
    for match in _GRADLE_INCLUDE_RE.finditer(text):
        for _start, _end, value in _string_literals(match["arguments"]):
            parts = [part for part in value.split(":") if part]
            if not parts or not all(
                re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts
            ):
                raise KitManifestError("Gradle module path is not static", manifest)
            modules.append(base.joinpath(*parts))

    for module in modules:
        build_file = _gradle_build_file(root, module)
        if build_file is None:
            continue
        discovery.manifests_read.add(build_file)
        _parse_gradle_build(root, build_file, discovery, catalog)

    if catalog is not None:
        # Unreferenced catalog entries are still declared in a kit manifest and
        # may pin transitive versions, so they are reviewed conservatively.
        for alias, coordinate in sorted(catalog.libraries.items()):
            if alias not in catalog.referenced_libraries:
                discovery.add("gradle", coordinate, Usage.RUNTIME, catalog.path)
        for alias, plugin_id in sorted(catalog.plugins.items()):
            if alias not in catalog.referenced_plugins:
                _add_plugin(discovery, plugin_id, catalog.path)


def _parse_gradle_standalone(
    root: Path, manifest: str, discovery: KitDiscovery
) -> None:
    _parse_gradle_build(root, manifest, discovery, None)


# --------------------------------------------------------------------------
# npm
# --------------------------------------------------------------------------

_NPM_LOCAL_SPEC_PREFIXES = ("file:", "link:", "workspace:", "portal:")
_NPM_RUNTIME_SECTIONS = ("dependencies", "optionalDependencies", "peerDependencies")


def _npm_direct(declaration: object, manifest: str) -> tuple[set[str], set[str]]:
    runtime: set[str] = set()
    development: set[str] = set()
    if not isinstance(declaration, Mapping):
        return runtime, development

    def names(section: str) -> set[str]:
        value = declaration.get(section) or {}
        if not isinstance(value, Mapping):
            raise KitManifestError("npm dependency section is malformed", manifest)
        return {
            name
            for name, spec in value.items()
            if not (isinstance(spec, str) and spec.startswith(_NPM_LOCAL_SPEC_PREFIXES))
        }

    for section in _NPM_RUNTIME_SECTIONS:
        runtime |= names(section)
    for section in ("bundleDependencies", "bundledDependencies"):
        bundled = declaration.get(section)
        if isinstance(bundled, list):
            runtime |= {name for name in bundled if isinstance(name, str)}
    development = names("devDependencies") - runtime
    return runtime, development


def _parse_npm_lock(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    payload = _load_json(root, manifest)
    if not isinstance(payload, Mapping):
        raise KitManifestError("npm lockfile is malformed", manifest)
    packages = payload.get("packages")
    # Lockfile v2/v3 mirror the package declarations in the root entry; v1
    # lockfiles only carry the installed tree, so package.json supplies them.
    lock_runtime, lock_development = _npm_direct(
        packages.get("") if isinstance(packages, Mapping) else None, manifest
    )
    package_json = PurePosixPath(manifest).with_name("package.json").as_posix()
    json_runtime: set[str] = set()
    json_development: set[str] = set()
    if (root / package_json).is_file():
        json_runtime, json_development = _npm_direct(
            _load_json(root, package_json), package_json
        )
        discovery.manifests_read.add(package_json)
    direct_runtime = lock_runtime | json_runtime
    direct_development = (lock_development | json_development) - direct_runtime

    seen: set[str] = set()
    for name, dev, top_level in _npm_lock_entries(payload, manifest):
        is_direct = top_level and name in direct_runtime | direct_development
        if dev and not is_direct:
            discovery.count_development("npm")
            continue
        usage = (
            Usage.RUNTIME if not dev or name in direct_runtime else Usage.DEVELOPMENT
        )
        discovery.add("npm", name, usage, manifest, direct=is_direct)
        if top_level:
            seen.add(name)

    # Peer and not-yet-installed declarations are still kit dependencies.
    for name in sorted(direct_runtime - seen):
        source = manifest if name in lock_runtime else package_json
        discovery.add("npm", name, Usage.RUNTIME, source)
    for name in sorted(direct_development - seen):
        source = manifest if name in lock_development else package_json
        discovery.add("npm", name, Usage.DEVELOPMENT, source)


def _npm_lock_entries(
    payload: Mapping[str, Any], manifest: str
) -> Iterator[tuple[str, bool, bool]]:
    """Yield ``(name, development_only, top_level)`` for third-party entries."""

    packages = payload.get("packages")
    if isinstance(packages, Mapping):
        for key, entry in packages.items():
            if key == "":
                continue
            if not isinstance(key, str) or not isinstance(entry, Mapping):
                raise KitManifestError("npm lockfile entry is malformed", manifest)
            # Workspace folders and links are first-party sources.
            if entry.get("link") is True or "node_modules/" not in key:
                continue
            name = entry.get("name", key.rsplit("node_modules/", 1)[1])
            if not isinstance(name, str):
                raise KitManifestError("npm lockfile entry is malformed", manifest)
            top_level = key.startswith("node_modules/") and (
                key.count("node_modules/") == 1
            )
            yield name, entry.get("dev") is True, top_level
        return

    def walk(tree: object, top_level: bool) -> Iterator[tuple[str, bool, bool]]:
        if tree is None:
            return
        if not isinstance(tree, Mapping):
            raise KitManifestError("npm lockfile entry is malformed", manifest)
        for name, entry in tree.items():
            if not isinstance(entry, Mapping):
                raise KitManifestError("npm lockfile entry is malformed", manifest)
            version = entry.get("version", "")
            if isinstance(version, str) and version.startswith(
                _NPM_LOCAL_SPEC_PREFIXES
            ):
                continue
            yield name, entry.get("dev") is True, top_level
            yield from walk(entry.get("dependencies"), False)

    yield from walk(payload.get("dependencies"), True)


# --------------------------------------------------------------------------
# Cargo
# --------------------------------------------------------------------------

_CARGO_RUNTIME_TABLES = ("dependencies",)
_CARGO_DEVELOPMENT_TABLES = ("dev-dependencies", "build-dependencies")


def _cargo_table_names(table: object) -> set[str]:
    if not isinstance(table, Mapping):
        return set()
    names: set[str] = set()
    for alias, spec in table.items():
        if isinstance(spec, Mapping) and isinstance(spec.get("package"), str):
            names.add(spec["package"])
        else:
            names.add(alias)
    return names


def _cargo_direct(data: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    runtime: set[str] = set()
    development: set[str] = set()
    sections: list[Mapping[str, Any]] = [data]
    targets = data.get("target")
    if isinstance(targets, Mapping):
        sections.extend(
            value for value in targets.values() if isinstance(value, Mapping)
        )
    for section in sections:
        for table in _CARGO_RUNTIME_TABLES:
            runtime |= _cargo_table_names(section.get(table))
        for table in _CARGO_DEVELOPMENT_TABLES:
            development |= _cargo_table_names(section.get(table))
    return runtime, development - runtime


def _parse_cargo_lock(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    data = _load_toml(root, manifest)
    raw_packages = data.get("package", [])
    if not isinstance(raw_packages, list):
        raise KitManifestError("Cargo lockfile has no package list", manifest)

    packages: list[tuple[str, str, bool, tuple[str, ...]]] = []
    for package in raw_packages:
        if not isinstance(package, Mapping):
            raise KitManifestError("Cargo lockfile package is malformed", manifest)
        name, version = package.get("name"), package.get("version")
        dependencies = package.get("dependencies", [])
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or not isinstance(dependencies, list)
            or not all(isinstance(item, str) for item in dependencies)
        ):
            raise KitManifestError("Cargo lockfile package is malformed", manifest)
        packages.append((name, version, "source" in package, tuple(dependencies)))

    by_name: dict[str, list[tuple[str, str]]] = {}
    for name, version, _remote, _deps in packages:
        by_name.setdefault(name, []).append((name, version))
    edges: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    for name, version, _remote, dependencies in packages:
        resolved: list[tuple[str, str]] = []
        for dependency in dependencies:
            parts = dependency.split()
            if not parts:
                raise KitManifestError("Cargo lockfile package is malformed", manifest)
            candidates = by_name.get(parts[0], [])
            if len(parts) > 1:
                candidates = [item for item in candidates if item[1] == parts[1]]
            if len(candidates) != 1:
                raise KitManifestError(
                    "Cargo lockfile dependency is ambiguous", manifest
                )
            resolved.append(candidates[0])
        edges[(name, version)] = tuple(resolved)

    toml_path = PurePosixPath(manifest).with_name("Cargo.toml").as_posix()
    direct_runtime: set[str] = set()
    direct_development: set[str] = set()
    if (root / toml_path).is_file():
        direct_runtime, direct_development = _cargo_direct(_load_toml(root, toml_path))
        discovery.manifests_read.add(toml_path)

    local_roots = [
        (name, version) for name, version, remote, _ in packages if not remote
    ]
    runtime_seeds: list[tuple[str, str]] = []
    development_seeds: list[tuple[str, str]] = []
    for local in local_roots:
        for dependency in edges[local]:
            if dependency[0] in direct_development:
                development_seeds.append(dependency)
            else:
                runtime_seeds.append(dependency)

    def closure(seeds: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
        reached: set[tuple[str, str]] = set()
        pending = list(seeds)
        while pending:
            node = pending.pop()
            if node in reached:
                continue
            reached.add(node)
            pending.extend(edges.get(node, ()))
        return reached

    runtime = closure(runtime_seeds)
    development = closure(development_seeds) - runtime
    direct_names = direct_runtime | direct_development
    reported: set[tuple[str, str]] = set()
    for name, version, remote, _deps in packages:
        if not remote:
            continue
        key = (name, version)
        is_direct = name in direct_names
        if key in development:
            if not is_direct:
                discovery.count_development("cargo")
                continue
            usage = Usage.DEVELOPMENT
        else:
            # Unreachable lock entries stay runtime: the lock still pins them.
            usage = Usage.RUNTIME
        if (name, usage) in reported:
            continue
        reported.add((name, usage))
        discovery.add("cargo", name, usage, manifest, direct=is_direct)


# --------------------------------------------------------------------------
# Flutter pub and CocoaPods
# --------------------------------------------------------------------------


def _yaml_scalar(value: str) -> str:
    stripped = value.split(" #", 1)[0].strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1]
    return stripped


def _yaml_section(text: str, section: str) -> dict[str, dict[str, str]]:
    """Return ``child -> {key: scalar}`` for one top-level YAML mapping.

    This is a deliberately small reader for the block-mapping subset that
    ``pubspec.yaml`` and ``pubspec.lock`` use for dependency sections.
    """

    children: dict[str, dict[str, str]] = {}
    in_section = False
    child_indent: int | None = None
    detail_indent: int | None = None
    current: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0:
            in_section = line.split(":", 1)[0].strip() == section
            child_indent = None
            current = None
            continue
        if not in_section or ":" not in line or line.startswith("- "):
            continue
        key, _, value = line.partition(":")
        key = _yaml_scalar(key)
        if child_indent is None:
            child_indent = indent
        if indent == child_indent:
            current = key
            detail_indent = None
            children[current] = {}
            scalar = _yaml_scalar(value)
            if scalar.startswith("{") and scalar.endswith("}"):
                for item in scalar[1:-1].split(","):
                    inner_key, _, inner_value = item.partition(":")
                    children[current][inner_key.strip()] = _yaml_scalar(inner_value)
            elif scalar:
                children[current]["version"] = scalar
        elif current is not None and indent > child_indent:
            # Only the dependency's own keys matter; deeper keys (for example
            # a git dependency's ``path``) describe a nested source mapping.
            if detail_indent is None:
                detail_indent = indent
            if indent == detail_indent:
                children[current].setdefault(key, _yaml_scalar(value))
    return children


def _parse_pubspec(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    text = _read_text(root, manifest)
    sections = (
        ("dependencies", Usage.RUNTIME),
        ("dependency_overrides", Usage.RUNTIME),
        ("dev_dependencies", Usage.DEVELOPMENT),
    )
    for section, usage in sections:
        for name, details in sorted(_yaml_section(text, section).items()):
            if "path" in details:
                continue
            discovery.add("pub", name, usage, manifest)


def _parse_pubspec_lock(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    text = _read_text(root, manifest)
    for name, details in sorted(_yaml_section(text, "packages").items()):
        if details.get("source") == "path":
            continue
        kind = details.get("dependency", "transitive")
        usage = Usage.DEVELOPMENT if kind == "direct dev" else Usage.RUNTIME
        discovery.add("pub", name, usage, manifest, direct=kind.startswith("direct"))


_PODSPEC_DEPENDENCY_RE = re.compile(r"\.dependency\s*\(?\s*[\"'](?P<name>[^\"'/\s]+)")
_PODSPEC_BLOCK_START_RE = re.compile(r"\b(?:test_spec|app_spec)\b[^\n]*\bdo\b")
_RUBY_BLOCK_OPEN_RE = re.compile(
    r"\bdo\b|^\s*(?:if|unless|case|begin|def|while|until|class|module)\b", re.MULTILINE
)
_RUBY_BLOCK_CLOSE_RE = re.compile(r"\bend\b")


def _parse_podspec(root: Path, manifest: str, discovery: KitDiscovery) -> None:
    text = _strip_comments(_read_text(root, manifest), hash_comments=True)
    development_ranges: list[tuple[int, int]] = []
    for start in _PODSPEC_BLOCK_START_RE.finditer(text):
        depth = 0
        position = start.start()
        tokens = sorted(
            [
                (match.start(), 1)
                for match in _RUBY_BLOCK_OPEN_RE.finditer(text, position)
            ]
            + [
                (match.start(), -1)
                for match in _RUBY_BLOCK_CLOSE_RE.finditer(text, position)
            ]
        )
        end = len(text)
        for offset, delta in tokens:
            depth += delta
            if depth == 0:
                end = offset
                break
        development_ranges.append((start.start(), end))

    for match in _PODSPEC_DEPENDENCY_RE.finditer(text):
        in_development = any(
            start <= match.start() < end for start, end in development_ranges
        )
        usage = Usage.DEVELOPMENT if in_development else Usage.RUNTIME
        discovery.add("cocoapods", match["name"], usage, manifest)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def _parser_for(manifest: str) -> Any:
    name = PurePosixPath(manifest).name
    if name == "Package.swift":
        return _parse_swift_manifest
    if name == "Package.resolved":
        return _parse_swift_resolved
    if name in {"settings.gradle.kts", "settings.gradle"}:
        return _parse_gradle_settings
    if name in {"build.gradle.kts", "build.gradle"}:
        return _parse_gradle_standalone
    if name == "package-lock.json":
        return _parse_npm_lock
    if name == "Cargo.lock":
        return _parse_cargo_lock
    if name == "pubspec.yaml":
        return _parse_pubspec
    if name == "pubspec.lock":
        return _parse_pubspec_lock
    if name.endswith(".podspec"):
        return _parse_podspec
    raise KitManifestError("kit manifest type is not supported", manifest)


def discover_kit_dependencies(
    root: Path = ROOT,
    manifests: Sequence[str] = DEFAULT_KIT_MANIFESTS,
) -> KitDiscovery:
    """Parse registered kit manifests without network access.

    Args:
        root: Repository root that the manifest paths are relative to.
        manifests: Repository-relative manifest paths to parse.

    Returns:
        Dependency occurrences, counts of transitive development-only lock
        entries, and every repository-relative manifest that was read.

    Raises:
        KitManifestError: A registered manifest is missing or malformed.
    """

    discovery = KitDiscovery()
    for manifest in manifests:
        relative = _relative(manifest)
        if not (root / relative).is_file():
            raise KitManifestError("registered kit manifest is missing", relative)
        discovery.manifests_read.add(relative)
        _parser_for(relative)(root, relative, discovery)
    return discovery


# --------------------------------------------------------------------------
# Review table and audit
# --------------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    return _python_inventory._split_markdown_row(line)


def _header_key(value: str) -> str:
    return _python_inventory._header_key(value)


def parse_kit_inventory(
    path: Path = DEFAULT_KIT_INVENTORY,
) -> tuple[KitReviewEntry, ...]:
    """Parse every ecosystem review table in the committed kit inventory.

    Args:
        path: Markdown document holding ``Ecosystem | Package | License
            expression`` tables.

    Returns:
        Review rows sorted by ecosystem and package.

    Raises:
        InventoryError: The document is unreadable, has no review table, or
            has duplicate or malformed rows.
    """

    try:
        if path.stat().st_size > MAX_INVENTORY_BYTES:
            raise InventoryError("kit inventory exceeds the size limit")
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise InventoryError("could not read the checked-in kit inventory") from None

    entries: dict[tuple[str, str], KitReviewEntry] = {}
    index = 0
    while index < len(lines) - 1:
        keys = [_header_key(cell) for cell in _split_row(lines[index])]
        separator = _split_row(lines[index + 1])
        if (
            "|" not in lines[index]
            or "ecosystem" not in keys
            or "package" not in keys
            or not any(key in {"license", "license_expression", "spdx"} for key in keys)
            or not all(_MARKDOWN_SEPARATOR_RE.fullmatch(cell) for cell in separator)
        ):
            index += 1
            continue
        ecosystem_at = keys.index("ecosystem")
        package_at = keys.index("package")
        license_at = next(
            position
            for position, key in enumerate(keys)
            if key in {"license", "license_expression", "spdx"}
        )
        index += 2
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            cells = _split_row(lines[index])
            line_number = index + 1
            index += 1

            def cell(position: int, row: list[str] = cells) -> str:
                return row[position] if position < len(row) else ""

            ecosystem = cell(ecosystem_at).casefold()
            name = cell(package_at).casefold()
            if ecosystem not in ECOSYSTEMS:
                raise InventoryError("kit inventory row has an unknown ecosystem")
            if not _PACKAGE_NAME_RE.fullmatch(name):
                raise InventoryError("kit inventory row has an invalid package name")
            key = (ecosystem, name)
            if key in entries:
                raise InventoryError("kit inventory contains a duplicate review row")
            entries[key] = KitReviewEntry(
                ecosystem, name, cell(license_at).strip(), line_number
            )
    if not entries:
        raise InventoryError("kit inventory document has no review table")
    return tuple(sorted(entries.values(), key=lambda item: (item.ecosystem, item.name)))


def _decide(
    dependency: KitDependency,
    entry: KitReviewEntry | None,
    exceptions: Sequence[KitLicenseException],
) -> KitRecord:
    if entry is None:
        return KitRecord(dependency, LicenseClass.UNKNOWN, Decision.UNREVIEWED)
    classification = classify_license(entry.license_expression)
    if classification == LicenseClass.PERMISSIVE:
        return KitRecord(dependency, classification, Decision.PERMISSIVE)
    if dependency.usage == Usage.DEVELOPMENT and (
        classify_license(
            entry.license_expression,
            permissive_licenses=PERMISSIVE_LICENSES | DEVELOPMENT_ONLY_LICENSES,
        )
        == LicenseClass.PERMISSIVE
    ):
        return KitRecord(dependency, classification, Decision.DEVELOPMENT_ONLY)
    if classification == LicenseClass.RESTRICTED:
        for exception in exceptions:
            if exception.covers(dependency, entry.license_expression):
                return KitRecord(
                    dependency,
                    classification,
                    Decision.SCOPED_EXCEPTION,
                    exception.exception_id,
                )
    return KitRecord(dependency, classification, Decision.DENIED)


def _merge(dependencies: Iterable[KitDependency]) -> tuple[KitDependency, ...]:
    """Collapse repeated occurrences; runtime usage and direct declarations win."""

    merged: dict[tuple[str, str, str], KitDependency] = {}
    for dependency in dependencies:
        key = (dependency.ecosystem, dependency.name, dependency.manifest)
        previous = merged.get(key)
        if previous is None:
            merged[key] = dependency
            continue
        usage = (
            Usage.RUNTIME
            if Usage.RUNTIME in {previous.usage, dependency.usage}
            else Usage.DEVELOPMENT
        )
        merged[key] = KitDependency(
            dependency.ecosystem,
            dependency.name,
            usage,
            dependency.manifest,
            previous.direct or dependency.direct,
        )
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (item.ecosystem, item.name, item.manifest),
        )
    )


def audit_kit_dependencies(
    entries: Iterable[KitReviewEntry],
    discovery: KitDiscovery,
    exceptions: Sequence[KitLicenseException] = SCOPED_KIT_EXCEPTIONS,
) -> KitAudit:
    """Classify every kit dependency occurrence against the review table.

    Args:
        entries: Reviewed license rows.
        discovery: Dependencies parsed from the registered manifests.
        exceptions: Scoped exceptions for restricted packages.

    Returns:
        Records in deterministic order plus counts for the report.
    """

    reviewed = {(entry.ecosystem, entry.name): entry for entry in entries}
    dependencies = _merge(discovery.dependencies)
    records = tuple(
        _decide(
            dependency,
            reviewed.get((dependency.ecosystem, dependency.name)),
            exceptions,
        )
        for dependency in dependencies
    )
    used = {(dependency.ecosystem, dependency.name) for dependency in dependencies}
    return KitAudit(
        records=records,
        counted_development=dict(sorted(discovery.counted_development.items())),
        unused_review_rows=len(set(reviewed) - used),
    )


def audit_kits(
    root: Path = ROOT,
    inventory_path: Path = DEFAULT_KIT_INVENTORY,
    manifests: Sequence[str] = DEFAULT_KIT_MANIFESTS,
    exceptions: Sequence[KitLicenseException] = SCOPED_KIT_EXCEPTIONS,
) -> KitAudit:
    """Audit the committed kit manifests against the committed review table.

    Args:
        root: Repository root.
        inventory_path: Kit review table document.
        manifests: Repository-relative manifest registry.
        exceptions: Scoped exceptions for restricted packages.

    Returns:
        The audit result.

    Raises:
        InventoryError: The review table or a manifest cannot be interpreted.
    """

    entries = parse_kit_inventory(inventory_path)
    return audit_kit_dependencies(
        entries, discover_kit_dependencies(root, manifests), exceptions
    )


def safe_report(audit: KitAudit) -> dict[str, Any]:
    """Build deterministic report data without license text or local paths."""

    usage_counts = Counter(record.dependency.usage for record in audit.records)
    decision_counts = Counter(record.decision for record in audit.records)
    return {
        "counted_development_transitive": dict(audit.counted_development),
        "entries": [record.to_safe_dict() for record in audit.records],
        "schema_version": 1,
        "summary": {
            "decisions": {
                decision: decision_counts.get(decision, 0)
                for decision in (
                    Decision.PERMISSIVE,
                    Decision.DEVELOPMENT_ONLY,
                    Decision.SCOPED_EXCEPTION,
                    Decision.UNREVIEWED,
                    Decision.DENIED,
                )
            },
            "failures": len(audit.failures),
            "unused_review_rows": audit.unused_review_rows,
            "usage": {
                Usage.RUNTIME: usage_counts.get(Usage.RUNTIME, 0),
                Usage.DEVELOPMENT: usage_counts.get(Usage.DEVELOPMENT, 0),
            },
        },
    }


def write_report(audit: KitAudit, path: Path) -> None:
    """Write a stable, privacy-safe JSON report to a local path."""

    payload = json.dumps(safe_report(audit), indent=2, sort_keys=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    except (OSError, UnicodeError):
        raise InventoryError("could not write the kit inventory report") from None


def _describe(record: KitRecord) -> str:
    dependency = record.dependency
    return (
        f"{dependency.manifest}: {dependency.ecosystem} "
        f"{safe_identifier(dependency.name)} ({dependency.usage})"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the kit license gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root that kit manifest paths are relative to",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_KIT_INVENTORY,
        help="checked-in Markdown kit review table",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help=(
            "repository-relative kit manifest to audit instead of the default "
            "registry; repeat for several manifests"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional privacy-safe JSON report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic, offline OpenMedKit license gate."""

    args = build_parser().parse_args(argv)
    manifests = tuple(args.manifest) if args.manifest else DEFAULT_KIT_MANIFESTS
    try:
        audit = audit_kits(args.root, args.inventory, manifests)
        if args.report:
            write_report(audit, args.report)
    except InventoryError as exc:
        # Messages are fixed reasons plus, at most, a repository-relative
        # manifest path; they never echo manifest content or local paths.
        print(f"kit license gate failed: {exc}", file=sys.stderr)
        return 1

    for record in audit.exceptions:
        print(
            f"scoped exception {record.exception_id}: {_describe(record)} "
            f"is {record.classification}"
        )
    if audit.failures:
        print("kit license gate failed:", file=sys.stderr)
        for record in audit.failures:
            print(f"- {_describe(record)}: {record.decision}", file=sys.stderr)
        return 1

    usage = Counter(record.dependency.usage for record in audit.records)
    counted = sum(audit.counted_development.values())
    print(
        "kit license gate passed: "
        f"{len(audit.records)} reviewed occurrences "
        f"({usage.get(Usage.RUNTIME, 0)} runtime, "
        f"{usage.get(Usage.DEVELOPMENT, 0)} development, "
        f"{len(audit.exceptions)} scoped exceptions); "
        f"{counted} transitive development-only lock entries counted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
