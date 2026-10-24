"""Signed offline clinical language bundles for low-connectivity sites.

An offline language bundle is one directory that carries everything a clinical
language needs without a network: a model, public lexicons, policy metadata,
synthetic self-tests, and evaluation evidence. The directory is a portable
bundle in the :mod:`openmed.skills.bundle_verify` format: ``manifest.json``
pins every file to a SHA-256 digest and is signed with HMAC-SHA256 or Ed25519.
The language descriptor ``offline-bundle.json`` is the bundle's only entry
point, so the signature covers it through its digest.

Verification is local and deterministic. It checks the signature and
checksums, the descriptor contract, a permissive license allowlist,
restricted-source and credential markers, the synthetic self-test fixtures,
and the declared RAM and storage requirements against caller-supplied host
facts. Installation copies the bundle into a staging directory, re-checks the
copied digests, runs the self-tests with outbound sockets blocked, and only
then publishes the bundle with an atomic rename.

Reports carry controlled codes, counts, bundle-relative paths, and digests.
They never carry self-test text, key material, or absolute paths. A bundle is
a deployment artifact, not a compliance certification, and a passing self-test
is a smoke check that the packaged model runs offline, not evidence of clinical
accuracy.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn

from openmed.clinical.grounding.systems import RESTRICTED_SYSTEMS
from openmed.core.model_integrity import sha256_file
from openmed.core.offline import OfflineModeError, network_blocked_if_offline
from openmed.plugins.registry import is_permissive_license
from openmed.skills.bundle_verify import (
    REASON_HASH_MISMATCH,
    REASON_MANIFEST_MALFORMED,
    REASON_SIGNATURE_DEPENDENCY_MISSING,
    SkillBundleManifest,
    canonical_manifest_bytes,
    verify_bundle,
)
from openmed.training.corpus import GATED_DAPT_SOURCES

OFFLINE_BUNDLE_SCHEMA_VERSION = "openmed.offline_language_bundle.v1"
OFFLINE_BUNDLE_MANIFEST_VERSION = "1.0"
MANIFEST_FILENAME = "manifest.json"
DESCRIPTOR_FILENAME = "offline-bundle.json"

COMPONENT_MODEL = "model"
COMPONENT_LEXICON = "lexicon"
COMPONENT_POLICY = "policy"
COMPONENT_SELF_TEST = "self_test"
COMPONENT_EVIDENCE = "evidence"
COMPONENT_KINDS: tuple[str, ...] = (
    COMPONENT_MODEL,
    COMPONENT_LEXICON,
    COMPONENT_POLICY,
    COMPONENT_SELF_TEST,
    COMPONENT_EVIDENCE,
)
REQUIRED_COMPONENT_KINDS = frozenset(COMPONENT_KINDS)
DATA_CLASSIFICATIONS = frozenset({"public", "synthetic"})
SIGNATURE_SCHEMES: tuple[str, ...] = ("hmac-sha256", "ed25519")

CHECK_INTEGRITY = "integrity"
CHECK_DESCRIPTOR = "descriptor"
CHECK_COMPONENTS = "components"
CHECK_LICENSES = "licenses"
CHECK_RESTRICTED_SOURCES = "restricted_sources"
CHECK_CREDENTIALS = "credentials"
CHECK_SELF_TESTS = "self_tests"
CHECK_STORAGE = "storage"
CHECK_RAM = "ram"
CHECK_NAMES: tuple[str, ...] = (
    CHECK_INTEGRITY,
    CHECK_DESCRIPTOR,
    CHECK_COMPONENTS,
    CHECK_LICENSES,
    CHECK_RESTRICTED_SOURCES,
    CHECK_CREDENTIALS,
    CHECK_SELF_TESTS,
    CHECK_STORAGE,
    CHECK_RAM,
)
CHECK_PASSED = "passed"
CHECK_FAILED = "failed"
CHECK_NOT_RUN = "not_run"

REASON_MANIFEST_MISSING = "manifest_missing"
REASON_MANIFEST_FIELDS_UNSUPPORTED = "manifest_fields_unsupported"
REASON_SYMLINK_PRESENT = "symlink_present"
REASON_UNSIGNED_BUNDLE = "unsigned_bundle"
REASON_ENTRY_POINT_MISMATCH = "entry_point_mismatch"
REASON_DESCRIPTOR_MALFORMED = "descriptor_malformed"
REASON_BUNDLE_ID_MISMATCH = "bundle_id_mismatch"
REASON_COMPONENT_FILES_MISMATCH = "component_files_mismatch"
REASON_COMPONENT_MISSING = "component_missing"
REASON_LICENSE_NOT_PERMISSIVE = "license_not_permissive"
REASON_RESTRICTED_SOURCE = "restricted_source"
REASON_CREDENTIAL_LIKE_FILE = "credential_like_file"
REASON_SELF_TEST_MALFORMED = "self_test_malformed"
REASON_SELF_TEST_NOT_SYNTHETIC = "self_test_not_synthetic"
REASON_STORAGE_REQUIREMENT_UNDERSTATED = "storage_requirement_understated"
REASON_INSUFFICIENT_STORAGE = "insufficient_storage"
REASON_INSUFFICIENT_RAM = "insufficient_ram"
REASON_HOST_FACTS_UNAVAILABLE = "host_facts_unavailable"
REASON_SIGNING_KEY_REQUIRED = "signing_key_required"
REASON_SIGNING_KEY_INVALID = "signing_key_invalid"
REASON_SOURCE_FILE_MISSING = "source_file_missing"
REASON_OUTPUT_NOT_EMPTY = "output_not_empty"
REASON_BUNDLE_NOT_VERIFIED = "bundle_not_verified"
REASON_SELF_TEST_RUNNER_REQUIRED = "self_test_runner_required"
REASON_SELF_TEST_FAILED = "self_test_failed"
REASON_NETWORK_ATTEMPTED = "network_attempted"
REASON_INSTALL_TARGET_EXISTS = "install_target_exists"
REASON_INSTALL_COPY_MISMATCH = "install_copy_mismatch"
REASON_INSTALL_IO_ERROR = "install_io_error"

SELF_TEST_ERROR_RUNNER = "runner_error"
SELF_TEST_ERROR_INVALID_OUTPUT = "invalid_runner_output"

OfflineSelfTestRunner = Callable[[Path, str], Iterable[Any]]
"""Run the packaged model on one synthetic self-test text.

The runner receives the root of the (staged) bundle copy and one synthetic
text. It returns spans as ``(start, end, label)`` tuples, mappings with those
keys, or objects with those attributes. The runner must not retain the bundle
root: installation renames the staged copy after the self-tests pass.
"""

_LOGGER = logging.getLogger(__name__)

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "bundle_id",
        "entry_points",
        "files",
        "signature_scheme",
        "signature",
    }
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "bundle_version",
        "language",
        "scripts",
        "capabilities",
        "resources",
        "components",
    }
)
_COMPONENT_FIELDS = frozenset(
    {"component_id", "kind", "files", "license", "source", "data_classification"}
)
_RESOURCE_FIELDS = frozenset({"min_ram_bytes", "min_storage_bytes"})
_SELF_TEST_FIELDS = frozenset({"case_id", "synthetic", "text", "expected"})
_SPAN_FIELDS = frozenset({"start", "end", "label"})
_RESERVED_PATHS = frozenset({MANIFEST_FILENAME, DESCRIPTOR_FILENAME})

_BUNDLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(
    r"^(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})"
    r"(?:-[0-9A-Za-z][0-9A-Za-z.-]{0,63})?$"
)
_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")
_SCRIPT_RE = re.compile(r"^[A-Z][A-Za-z]{1,31}(?:/[A-Z][A-Za-z]{1,31})?$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}$")
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

_MAX_PATH_SEGMENTS = 8
_MAX_COLLECTION_ITEMS = 4096
_MAX_DESCRIPTOR_BYTES = 1024 * 1024
_MAX_SELF_TEST_FILE_BYTES = 4 * 1024 * 1024
_MAX_SELF_TEST_CASES = 2048
_MAX_SELF_TEST_TEXT_CHARS = 16384
_MAX_EXPECTED_SPANS = 256
_MAX_RESOURCE_BYTES = 1 << 50

_RESTRICTED_SOURCE_TOKENS = frozenset(GATED_DAPT_SOURCES) | frozenset(
    RESTRICTED_SYSTEMS
)
_CREDENTIAL_SUFFIXES = frozenset({".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"})
_CREDENTIAL_STEMS = frozenset(
    {"credential", "credentials", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
)


class OfflineBundleError(ValueError):
    """Raised when an offline bundle declaration or packaging request is invalid.

    Args:
        reason: Controlled failure code, one of the ``REASON_*`` constants.
        message: Content-free detail naming the failing field or step.
    """

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {message}")


class _CheckFailure(Exception):
    """Internal control flow for a failed verification check."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _InvalidRunnerOutput(Exception):
    """Internal marker for a runner result that is not a span collection."""


@dataclass(frozen=True)
class ResourceRequirements:
    """Minimum host resources a bundle declares for installation and use.

    Args:
        min_ram_bytes: Physical memory the packaged model needs at runtime.
        min_storage_bytes: Free storage needed to install the bundle. It must
            cover at least the bundle's own payload size.
    """

    min_ram_bytes: int
    min_storage_bytes: int

    def __post_init__(self) -> None:
        for name in ("min_ram_bytes", "min_storage_bytes"):
            value = getattr(self, name)
            if not _is_int(value) or not 0 < value <= _MAX_RESOURCE_BYTES:
                raise OfflineBundleError(
                    REASON_DESCRIPTOR_MALFORMED,
                    f"{name} must be a positive integer byte count",
                )

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible mapping."""

        return {
            "min_ram_bytes": self.min_ram_bytes,
            "min_storage_bytes": self.min_storage_bytes,
        }


@dataclass(frozen=True)
class HostResources:
    """Host facts that verification compares with a bundle's requirements.

    Callers inject these facts so verification stays deterministic and
    testable. :meth:`detect` offers a best-effort local probe.

    Args:
        available_ram_bytes: Physical memory available to the runtime.
        available_storage_bytes: Free storage at the installation root.
    """

    available_ram_bytes: int
    available_storage_bytes: int

    def __post_init__(self) -> None:
        for name in ("available_ram_bytes", "available_storage_bytes"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def detect(cls, path: str | Path) -> "HostResources":
        """Probe this host's physical memory and free storage at ``path``.

        Storage is measured on the nearest existing ancestor of ``path``. RAM
        is the host's total physical memory, so declare
        :attr:`ResourceRequirements.min_ram_bytes` with headroom. The probe is
        local and never touches the network.

        Args:
            path: Intended installation root.

        Returns:
            Probed host facts.

        Raises:
            OfflineBundleError: If the platform does not expose its physical
                memory size; pass :class:`HostResources` explicitly instead.
        """

        probe = Path(path)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        storage = int(shutil.disk_usage(probe).free)
        ram = _physical_memory_bytes()
        if ram is None:
            raise OfflineBundleError(
                REASON_HOST_FACTS_UNAVAILABLE,
                "physical memory size is unavailable; pass HostResources explicitly",
            )
        return cls(available_ram_bytes=ram, available_storage_bytes=storage)

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible mapping."""

        return {
            "available_ram_bytes": self.available_ram_bytes,
            "available_storage_bytes": self.available_storage_bytes,
        }


@dataclass(frozen=True)
class OfflineBundleComponent:
    """One declared part of an offline language bundle.

    Args:
        component_id: Unique lowercase identifier within the bundle.
        kind: One of :data:`COMPONENT_KINDS`.
        files: Bundle-relative POSIX paths owned by this component.
        license: SPDX license expression covering every file.
        source: Provenance identifier, such as a repository reference. It is
            recorded only; installation never fetches it.
        data_classification: ``"public"`` or ``"synthetic"``. Self-test
            components must be ``"synthetic"``.
    """

    component_id: str
    kind: str
    files: tuple[str, ...]
    license: str
    source: str
    data_classification: str = "public"

    def __post_init__(self) -> None:
        if not _matches(_IDENTIFIER_RE, self.component_id):
            _malformed("component_id must be a lowercase identifier")
        if self.kind not in COMPONENT_KINDS:
            _malformed("component kind is not supported")
        if (
            not isinstance(self.license, str)
            or not self.license.strip()
            or len(self.license) > 256
        ):
            _malformed("component license must be a non-empty SPDX expression")
        if not _matches(_SOURCE_RE, self.source):
            _malformed("component source must be a provenance identifier")
        if self.data_classification not in DATA_CLASSIFICATIONS:
            _malformed("component data_classification is not supported")
        if self.kind == COMPONENT_SELF_TEST and self.data_classification != "synthetic":
            raise OfflineBundleError(
                REASON_SELF_TEST_NOT_SYNTHETIC,
                "self-test components must be classified as synthetic",
            )
        files = _string_items(self.files, "component files")
        if not files:
            _malformed("each component must declare at least one file")
        for path in files:
            _validate_bundle_path(path)
        if len(set(files)) != len(files):
            _malformed("component files must be unique")
        object.__setattr__(self, "files", tuple(sorted(files)))

    @classmethod
    def from_mapping(cls, data: Any) -> "OfflineBundleComponent":
        """Parse one component object from a descriptor.

        Args:
            data: Component mapping with exactly the public contract fields.

        Returns:
            A validated component.

        Raises:
            OfflineBundleError: If the mapping is malformed.
        """

        value = _exact_mapping(data, _COMPONENT_FIELDS, "component")
        return cls(
            component_id=value["component_id"],
            kind=value["kind"],
            files=_string_items(value["files"], "component files"),
            license=value["license"],
            source=value["source"],
            data_classification=value["data_classification"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "component_id": self.component_id,
            "kind": self.kind,
            "files": list(self.files),
            "license": self.license,
            "source": self.source,
            "data_classification": self.data_classification,
        }


@dataclass(frozen=True)
class OfflineBundleDescriptor:
    """Language descriptor stored in ``offline-bundle.json``.

    Collections are normalized to sorted tuples, so equal declarations always
    serialize to identical bytes.

    Args:
        bundle_id: Lowercase bundle identifier; must equal the manifest's.
        bundle_version: Semantic version of the bundle contents.
        language: Lowercase ISO 639-1 code matching the language-pack registry.
        scripts: Script names the bundle supports, such as ``"Latin"``.
        capabilities: Controlled task identifiers the bundle declares.
        resources: Minimum RAM and storage requirements.
        components: Declared components; every payload file belongs to one.
        schema_version: Descriptor schema identifier.
    """

    bundle_id: str
    bundle_version: str
    language: str
    scripts: tuple[str, ...]
    capabilities: tuple[str, ...]
    resources: ResourceRequirements
    components: tuple[OfflineBundleComponent, ...]
    schema_version: str = OFFLINE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OFFLINE_BUNDLE_SCHEMA_VERSION:
            _malformed("schema_version is not supported")
        if not _matches(_BUNDLE_ID_RE, self.bundle_id):
            _malformed("bundle_id must be a lowercase identifier")
        if not _matches(_VERSION_RE, self.bundle_version):
            _malformed("bundle_version must be a semantic version")
        if not _matches(_LANGUAGE_RE, self.language):
            _malformed("language must be a lowercase ISO 639-1 code")
        scripts = _unique_sorted(self.scripts, _SCRIPT_RE, "scripts")
        capabilities = _unique_sorted(self.capabilities, _IDENTIFIER_RE, "capabilities")
        if not isinstance(self.resources, ResourceRequirements):
            _malformed("resources must be ResourceRequirements")
        components = _component_items(self.components)
        if not components:
            _malformed("components must not be empty")
        component_ids = [component.component_id for component in components]
        if len(set(component_ids)) != len(component_ids):
            _malformed("component_id values must be unique")
        seen: set[str] = set()
        for component in components:
            for path in component.files:
                folded = path.casefold()
                if folded in seen:
                    _malformed("a file is declared more than once")
                seen.add(folded)
        object.__setattr__(self, "scripts", scripts)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "components",
            tuple(sorted(components, key=lambda item: item.component_id)),
        )

    @property
    def payload_files(self) -> tuple[str, ...]:
        """Return every component file path in sorted order."""

        return tuple(sorted(path for item in self.components for path in item.files))

    def component_files(self, kind: str) -> tuple[str, ...]:
        """Return sorted file paths owned by components of ``kind``."""

        return tuple(
            sorted(
                path
                for component in self.components
                if component.kind == kind
                for path in component.files
            )
        )

    @classmethod
    def from_mapping(cls, data: Any) -> "OfflineBundleDescriptor":
        """Parse a descriptor mapping, rejecting unknown or missing fields.

        Args:
            data: Mapping loaded from ``offline-bundle.json``.

        Returns:
            A validated descriptor.

        Raises:
            OfflineBundleError: If the mapping is malformed.
        """

        value = _exact_mapping(data, _DESCRIPTOR_FIELDS, "descriptor")
        resources = _exact_mapping(value["resources"], _RESOURCE_FIELDS, "resources")
        raw_components = value["components"]
        if not isinstance(raw_components, list):
            _malformed("components must be a list")
        if len(raw_components) > _MAX_COLLECTION_ITEMS:
            _malformed("components exceed the supported size")
        return cls(
            schema_version=value["schema_version"],
            bundle_id=value["bundle_id"],
            bundle_version=value["bundle_version"],
            language=value["language"],
            scripts=_string_items(value["scripts"], "scripts"),
            capabilities=_string_items(value["capabilities"], "capabilities"),
            resources=ResourceRequirements(
                min_ram_bytes=resources["min_ram_bytes"],
                min_storage_bytes=resources["min_storage_bytes"],
            ),
            components=tuple(
                OfflineBundleComponent.from_mapping(item) for item in raw_components
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping in canonical order."""

        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "bundle_version": self.bundle_version,
            "language": self.language,
            "scripts": list(self.scripts),
            "capabilities": list(self.capabilities),
            "resources": self.resources.to_dict(),
            "components": [item.to_dict() for item in self.components],
        }

    def to_json_bytes(self) -> bytes:
        """Return the canonical on-disk encoding of the descriptor."""

        text = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        return f"{text}\n".encode("utf-8")


@dataclass(frozen=True)
class OfflineBundleLicense:
    """License record for one component, as checked against the allowlist.

    Args:
        component_id: Component the license covers.
        kind: Component kind.
        license: Declared SPDX license expression.
        source: Declared provenance identifier.
        permissive: Whether every license token is on the permissive allowlist.
    """

    component_id: str
    kind: str
    license: str
    source: str
    permissive: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "component_id": self.component_id,
            "kind": self.kind,
            "license": self.license,
            "source": self.source,
            "permissive": self.permissive,
        }


@dataclass(frozen=True)
class OfflineBundleCheck:
    """Outcome of one verification check.

    Args:
        name: One of :data:`CHECK_NAMES`.
        status: ``"passed"``, ``"failed"``, or ``"not_run"``.
        reason: Controlled failure code; empty unless ``status`` is failed.
    """

    name: str
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible mapping."""

        return {"name": self.name, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class OfflineBundleVerification:
    """Deterministic, content-free result of verifying an offline bundle.

    A valid result is the hand-off for activation: it exposes the verified
    manifest and descriptor, per-file digests, license records, declared
    capabilities, and resource requirements.

    Args:
        valid: Whether every check passed.
        reason: First failure code; empty when ``valid`` is true.
        failures: Every failure code in check order.
        checks: Per-check outcomes in :data:`CHECK_NAMES` order.
        host: Host facts the requirements were compared with.
        manifest: Verified signed manifest, when integrity checks passed.
        descriptor: Verified language descriptor, when it parsed.
        manifest_digest: ``sha256:`` digest of the canonical signed manifest.
        file_digests: Bundle-relative path to verified ``sha256:`` digest.
        payload_bytes: Total size of the manifest and every declared file.
        licenses: License records for every component.
        self_test_case_count: Number of validated synthetic self-test cases.
    """

    valid: bool
    reason: str
    failures: tuple[str, ...]
    checks: tuple[OfflineBundleCheck, ...]
    host: HostResources
    manifest: SkillBundleManifest | None = None
    descriptor: OfflineBundleDescriptor | None = None
    manifest_digest: str = ""
    file_digests: Mapping[str, str] = field(default_factory=dict)
    payload_bytes: int = 0
    licenses: tuple[OfflineBundleLicense, ...] = ()
    self_test_case_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "file_digests",
            MappingProxyType(dict(sorted(self.file_digests.items()))),
        )

    @property
    def bundle_id(self) -> str:
        """Return the verified bundle identifier, or an empty string."""

        return self.descriptor.bundle_id if self.descriptor is not None else ""

    @property
    def bundle_version(self) -> str:
        """Return the declared bundle version, or an empty string."""

        return self.descriptor.bundle_version if self.descriptor is not None else ""

    @property
    def language(self) -> str:
        """Return the declared ISO 639-1 language code, or an empty string."""

        return self.descriptor.language if self.descriptor is not None else ""

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Return the declared capability identifiers."""

        return self.descriptor.capabilities if self.descriptor is not None else ()

    @property
    def resources(self) -> ResourceRequirements | None:
        """Return the declared resource requirements, if the descriptor parsed."""

        return self.descriptor.resources if self.descriptor is not None else None

    @property
    def signature_scheme(self) -> str:
        """Return the manifest signature scheme, or an empty string."""

        return self.manifest.signature_scheme if self.manifest is not None else ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report without raw content or key material."""

        return {
            "schema_version": OFFLINE_BUNDLE_SCHEMA_VERSION,
            "valid": self.valid,
            "reason": self.reason,
            "failures": list(self.failures),
            "checks": [check.to_dict() for check in self.checks],
            "bundle_id": self.bundle_id,
            "bundle_version": self.bundle_version,
            "language": self.language,
            "signature_scheme": self.signature_scheme,
            "manifest_digest": self.manifest_digest,
            "file_digests": dict(self.file_digests),
            "payload_bytes": self.payload_bytes,
            "licenses": [item.to_dict() for item in self.licenses],
            "capabilities": list(self.capabilities),
            "resources": (
                self.resources.to_dict() if self.resources is not None else None
            ),
            "host": self.host.to_dict(),
            "self_test_case_count": self.self_test_case_count,
            "descriptor": (
                self.descriptor.to_dict() if self.descriptor is not None else None
            ),
        }


@dataclass(frozen=True)
class OfflineSelfTestCaseResult:
    """Content-free outcome of one synthetic self-test case.

    Args:
        case_id: Self-test case identifier.
        expected_count: Number of expected spans.
        predicted_count: Number of distinct spans the runner returned.
        matched_count: Number of expected spans found exactly.
        passed: Whether every expected span was found without an error.
        error: Controlled error code; empty when the runner succeeded.
    """

    case_id: str
    expected_count: int
    predicted_count: int
    matched_count: int
    passed: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "case_id": self.case_id,
            "expected_count": self.expected_count,
            "predicted_count": self.predicted_count,
            "matched_count": self.matched_count,
            "passed": self.passed,
            "error": self.error,
        }


@dataclass(frozen=True)
class OfflineSelfTestReport:
    """Content-free report of running a bundle's synthetic self-tests.

    Args:
        executed: Whether the runner was invoked.
        passed: Whether every case passed with outbound sockets blocked.
        reason: Controlled failure code; empty when ``passed`` is true.
        network_blocked: Whether the cases ran inside the socket guard.
        cases: Per-case outcomes in case order.
    """

    executed: bool
    passed: bool
    reason: str
    network_blocked: bool
    cases: tuple[OfflineSelfTestCaseResult, ...] = ()

    @property
    def case_count(self) -> int:
        """Return the number of cases that ran."""

        return len(self.cases)

    @property
    def passed_count(self) -> int:
        """Return the number of passing cases."""

        return sum(1 for case in self.cases if case.passed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping without self-test text."""

        return {
            "executed": self.executed,
            "passed": self.passed,
            "reason": self.reason,
            "network_blocked": self.network_blocked,
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class OfflineBundleInstallation:
    """Result of installing an offline bundle.

    Args:
        installed: Whether the bundle was published to the install root.
        reason: Controlled failure code; empty when ``installed`` is true.
        verification: Verification of the source bundle.
        self_tests: Self-test report, when self-tests were attempted.
        install_path: Absolute install location; omitted from :meth:`to_dict`.
    """

    installed: bool
    reason: str
    verification: OfflineBundleVerification
    self_tests: OfflineSelfTestReport | None = None
    install_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report with a relative install location."""

        relpath = ""
        if self.installed:
            relpath = (
                f"{self.verification.bundle_id}/{self.verification.bundle_version}"
            )
        return {
            "installed": self.installed,
            "reason": self.reason,
            "install_relpath": relpath,
            "verification": self.verification.to_dict(),
            "self_tests": (
                self.self_tests.to_dict() if self.self_tests is not None else None
            ),
        }


@dataclass(frozen=True)
class PackagedOfflineBundle:
    """Summary of a freshly packaged bundle.

    Args:
        bundle_id: Packaged bundle identifier.
        bundle_version: Packaged bundle version.
        signature_scheme: Signature scheme used for the manifest.
        manifest_digest: ``sha256:`` digest of the canonical signed manifest.
        file_count: Number of declared files, including the descriptor.
        payload_bytes: Total size of the manifest and every declared file.
        path: Output directory; omitted from :meth:`to_dict`.
    """

    bundle_id: str
    bundle_version: str
    signature_scheme: str
    manifest_digest: str
    file_count: int
    payload_bytes: int
    path: Path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary without the output path."""

        return {
            "bundle_id": self.bundle_id,
            "bundle_version": self.bundle_version,
            "signature_scheme": self.signature_scheme,
            "manifest_digest": self.manifest_digest,
            "file_count": self.file_count,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class _SelfTestCase:
    """Parsed synthetic self-test case; never serialized."""

    case_id: str
    text: str
    expected: frozenset[tuple[int, int, str]]


def package_offline_bundle(
    descriptor: OfflineBundleDescriptor,
    sources: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    signature_key: bytes | None = None,
    ed25519_private_key: bytes | None = None,
) -> PackagedOfflineBundle:
    """Package local files into a signed offline language bundle.

    The output is deterministic: equal inputs and keys produce byte-identical
    descriptors and manifests. Packaging is local, applies the same component,
    license, restricted-source, credential, and self-test policy as
    verification before writing, and publishes the directory atomically.

    Args:
        descriptor: Language descriptor declaring every component and file.
        sources: Bundle-relative path mapped to the local file to copy. The
            keys must equal the descriptor's component files.
        output_dir: Destination directory; must be absent or empty.
        signature_key: HMAC-SHA256 key. Provide exactly one signing key.
        ed25519_private_key: Raw 32-byte Ed25519 private key. Requires the
            ``openmed[integrity]`` extra.

    Returns:
        Summary of the packaged bundle.

    Raises:
        OfflineBundleError: If the sources, licenses, self-tests, storage
            requirement, or signing key are invalid.
        TypeError: If ``descriptor`` is not an :class:`OfflineBundleDescriptor`.
    """

    if not isinstance(descriptor, OfflineBundleDescriptor):
        raise TypeError("descriptor must be an OfflineBundleDescriptor")
    scheme = _signing_scheme(signature_key, ed25519_private_key)
    _enforce_packaging_policy(descriptor)
    source_files = _source_files(descriptor, sources)
    _validate_self_test_sources(descriptor, source_files)

    destination = Path(output_dir)
    if destination.is_symlink() or (
        destination.exists()
        and (not destination.is_dir() or any(destination.iterdir()))
    ):
        raise OfflineBundleError(
            REASON_OUTPUT_NOT_EMPTY, "output directory must be absent or empty"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".openmed-bundle-", dir=str(destination.parent))
    )
    try:
        digests: dict[str, str] = {}
        payload_bytes = 0
        for rel_path in descriptor.payload_files:
            target = _bundle_file(staging, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_files[rel_path], target)
            digests[rel_path] = _bare_hex(sha256_file(target))
            payload_bytes += target.stat().st_size
        descriptor_bytes = descriptor.to_json_bytes()
        (staging / DESCRIPTOR_FILENAME).write_bytes(descriptor_bytes)
        digests[DESCRIPTOR_FILENAME] = hashlib.sha256(descriptor_bytes).hexdigest()
        payload_bytes += len(descriptor_bytes)

        unsigned = SkillBundleManifest(
            manifest_version=OFFLINE_BUNDLE_MANIFEST_VERSION,
            bundle_id=descriptor.bundle_id,
            entry_points=(DESCRIPTOR_FILENAME,),
            files=dict(sorted(digests.items())),
            signature_scheme=scheme,
        )
        canonical = canonical_manifest_bytes(unsigned)
        signature = _sign(
            canonical,
            scheme=scheme,
            key=signature_key if signature_key is not None else ed25519_private_key,
        )
        manifest_payload = dict(unsigned.to_dict(), signature=signature)
        manifest_bytes = (
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        (staging / MANIFEST_FILENAME).write_bytes(manifest_bytes)
        payload_bytes += len(manifest_bytes)
        if descriptor.resources.min_storage_bytes < payload_bytes:
            raise OfflineBundleError(
                REASON_STORAGE_REQUIREMENT_UNDERSTATED,
                "min_storage_bytes is smaller than the bundle payload",
            )
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    except OSError:
        raise OfflineBundleError(
            REASON_INSTALL_IO_ERROR, "the bundle could not be written"
        ) from None
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    _LOGGER.info(
        "offline bundle packaged: scheme=%s file_count=%d payload_bytes=%d",
        scheme,
        len(digests),
        payload_bytes,
    )
    return PackagedOfflineBundle(
        bundle_id=descriptor.bundle_id,
        bundle_version=descriptor.bundle_version,
        signature_scheme=scheme,
        manifest_digest=_prefixed(hashlib.sha256(canonical).hexdigest()),
        file_count=len(digests),
        payload_bytes=payload_bytes,
        path=destination,
    )


def verify_offline_bundle(
    bundle_dir: str | Path,
    *,
    host: HostResources,
    signature_key: bytes | None = None,
    signature_public_key: bytes | None = None,
) -> OfflineBundleVerification:
    """Verify an offline language bundle before installation.

    Integrity checks (signature, checksums, manifest shape, descriptor) stop
    at the first failure because later checks would read untrusted metadata.
    Policy checks (components, licenses, restricted sources, credentials,
    self-test fixtures, storage, RAM) all run and report every failure.
    Verification performs no network call and never raises for a bad bundle.

    Args:
        bundle_dir: Bundle directory containing ``manifest.json``.
        host: Host facts to compare with the declared requirements.
        signature_key: HMAC-SHA256 key for ``hmac-sha256`` bundles.
        signature_public_key: Raw 32-byte Ed25519 public key for ``ed25519``
            bundles.

    Returns:
        Deterministic verification result.

    Raises:
        TypeError: If ``host`` is not :class:`HostResources`.
    """

    if not isinstance(host, HostResources):
        raise TypeError("host must be HostResources")
    root = Path(bundle_dir)
    statuses: dict[str, OfflineBundleCheck] = {}

    def finish(**values: Any) -> OfflineBundleVerification:
        checks = tuple(
            statuses.get(name, OfflineBundleCheck(name, CHECK_NOT_RUN))
            for name in CHECK_NAMES
        )
        failures = tuple(check.reason for check in checks if check.reason)
        result = OfflineBundleVerification(
            valid=not failures,
            reason=failures[0] if failures else "",
            failures=failures,
            checks=checks,
            host=host,
            **values,
        )
        _LOGGER.info(
            "offline bundle verification: valid=%s reason=%s failure_count=%d",
            result.valid,
            result.reason or "none",
            len(failures),
        )
        return result

    def record(name: str, reason: str = "") -> None:
        status = CHECK_FAILED if reason else CHECK_PASSED
        statuses[name] = OfflineBundleCheck(name, status, reason)

    try:
        manifest = _verify_integrity(
            root,
            signature_key=signature_key,
            signature_public_key=signature_public_key,
        )
    except _CheckFailure as failure:
        record(CHECK_INTEGRITY, failure.reason)
        return finish()
    record(CHECK_INTEGRITY)
    file_digests = {path: _prefixed(digest) for path, digest in manifest.files.items()}
    manifest_digest = _prefixed(
        hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
    )
    integrity = {
        "manifest": manifest,
        "manifest_digest": manifest_digest,
        "file_digests": file_digests,
    }

    try:
        descriptor = _load_descriptor(root, manifest)
    except _CheckFailure as failure:
        record(CHECK_DESCRIPTOR, failure.reason)
        return finish(**integrity)
    record(CHECK_DESCRIPTOR)

    try:
        payload_bytes = _payload_bytes(root, manifest)
    except _CheckFailure as failure:
        record(CHECK_COMPONENTS, failure.reason)
        return finish(**integrity, descriptor=descriptor)
    declared = set(manifest.files) - {DESCRIPTOR_FILENAME}
    if set(descriptor.payload_files) != declared:
        record(CHECK_COMPONENTS, REASON_COMPONENT_FILES_MISMATCH)
    elif {item.kind for item in descriptor.components} != REQUIRED_COMPONENT_KINDS:
        record(CHECK_COMPONENTS, REASON_COMPONENT_MISSING)
    else:
        record(CHECK_COMPONENTS)

    licenses = _license_records(descriptor)
    record(
        CHECK_LICENSES,
        ""
        if all(item.permissive for item in licenses)
        else REASON_LICENSE_NOT_PERMISSIVE,
    )
    record(
        CHECK_RESTRICTED_SOURCES,
        REASON_RESTRICTED_SOURCE if _has_restricted_source(descriptor) else "",
    )
    record(
        CHECK_CREDENTIALS,
        REASON_CREDENTIAL_LIKE_FILE
        if any(_is_credential_like(path) for path in manifest.files)
        else "",
    )

    case_count = 0
    if statuses[CHECK_COMPONENTS].status == CHECK_PASSED:
        try:
            case_count = len(_load_self_test_cases(root, descriptor, file_digests))
        except _CheckFailure as failure:
            record(CHECK_SELF_TESTS, failure.reason)
        else:
            record(CHECK_SELF_TESTS)

    requirements = descriptor.resources
    if requirements.min_storage_bytes < payload_bytes:
        record(CHECK_STORAGE, REASON_STORAGE_REQUIREMENT_UNDERSTATED)
    elif host.available_storage_bytes < requirements.min_storage_bytes:
        record(CHECK_STORAGE, REASON_INSUFFICIENT_STORAGE)
    else:
        record(CHECK_STORAGE)
    record(
        CHECK_RAM,
        REASON_INSUFFICIENT_RAM
        if host.available_ram_bytes < requirements.min_ram_bytes
        else "",
    )
    return finish(
        **integrity,
        descriptor=descriptor,
        payload_bytes=payload_bytes,
        licenses=licenses,
        self_test_case_count=case_count,
    )


def run_offline_self_tests(
    bundle_dir: str | Path,
    runner: OfflineSelfTestRunner,
    *,
    verification: OfflineBundleVerification,
) -> OfflineSelfTestReport:
    """Run a verified bundle's synthetic self-tests with sockets blocked.

    Each self-test file is re-read and bound to the verified digest before it
    is parsed. A case passes when the runner returns every expected span
    exactly; extra spans are allowed. Any outbound socket attempt aborts the
    run with :data:`REASON_NETWORK_ATTEMPTED`. Like the bundled-model loader,
    the guard also sets the Hugging Face offline environment flags for the
    process.

    Args:
        bundle_dir: Bundle directory (or a byte-identical copy) to test.
        runner: Callable that runs the packaged model on one synthetic text.
        verification: Valid verification result for the same bundle bytes.

    Returns:
        Content-free self-test report.

    Raises:
        TypeError: If ``runner`` is not callable or ``verification`` has the
            wrong type.
    """

    if not callable(runner):
        raise TypeError("runner must be callable")
    if not isinstance(verification, OfflineBundleVerification):
        raise TypeError("verification must be OfflineBundleVerification")
    if not verification.valid or verification.descriptor is None:
        return OfflineSelfTestReport(
            executed=False,
            passed=False,
            reason=REASON_BUNDLE_NOT_VERIFIED,
            network_blocked=False,
        )

    root = Path(bundle_dir)
    results: list[OfflineSelfTestCaseResult] = []
    reason = ""
    with network_blocked_if_offline(local_only=True):
        try:
            cases = _load_self_test_cases(
                root, verification.descriptor, verification.file_digests
            )
        except _CheckFailure as failure:
            return OfflineSelfTestReport(
                executed=False,
                passed=False,
                reason=failure.reason,
                network_blocked=True,
            )
        for case in cases:
            error = ""
            predicted: frozenset[tuple[int, int, str]] = frozenset()
            try:
                predicted = _coerce_predictions(runner(root, case.text))
            except _InvalidRunnerOutput:
                error = SELF_TEST_ERROR_INVALID_OUTPUT
            except Exception as exc:
                if _blocked_by_offline_guard(exc):
                    error = REASON_NETWORK_ATTEMPTED
                    reason = REASON_NETWORK_ATTEMPTED
                else:
                    error = SELF_TEST_ERROR_RUNNER
            matched = len(case.expected & predicted)
            results.append(
                OfflineSelfTestCaseResult(
                    case_id=case.case_id,
                    expected_count=len(case.expected),
                    predicted_count=len(predicted),
                    matched_count=matched,
                    passed=not error and matched == len(case.expected),
                    error=error,
                )
            )
            if reason:
                break

    if not reason and not all(item.passed for item in results):
        reason = REASON_SELF_TEST_FAILED
    report = OfflineSelfTestReport(
        executed=True,
        passed=not reason,
        reason=reason,
        network_blocked=True,
        cases=tuple(results),
    )
    _LOGGER.info(
        "offline bundle self-tests: passed=%s case_count=%d passed_count=%d",
        report.passed,
        report.case_count,
        report.passed_count,
    )
    return report


def install_offline_bundle(
    bundle_dir: str | Path,
    install_root: str | Path,
    *,
    host: HostResources,
    signature_key: bytes | None = None,
    signature_public_key: bytes | None = None,
    self_test_runner: OfflineSelfTestRunner | None = None,
    require_self_tests: bool = True,
) -> OfflineBundleInstallation:
    """Verify, self-test, and atomically install an offline language bundle.

    The complete installation runs with outbound sockets blocked (the guard
    also sets the Hugging Face offline environment flags for the process). The
    bundle is verified, copied into a staging directory under
    ``install_root``, the copied digests are re-checked, and the self-tests run
    against the staged copy. Only then is the copy renamed to
    ``<install_root>/<bundle_id>/<bundle_version>``. Any failure removes the
    staging directory and leaves existing installations untouched. Installing
    a bundle does not activate it in the language-pack registry.

    Args:
        bundle_dir: Source bundle directory, for example on removable media.
        install_root: Directory that holds installed bundles.
        host: Host facts to compare with the declared requirements.
        signature_key: HMAC-SHA256 key for ``hmac-sha256`` bundles.
        signature_public_key: Raw 32-byte Ed25519 public key for ``ed25519``
            bundles.
        self_test_runner: Callable that runs the packaged model on one
            synthetic self-test text.
        require_self_tests: Fail when no runner is supplied. Set to false only
            when the self-tests run later through :func:`run_offline_self_tests`.

    Returns:
        Installation result with verification and self-test reports.
    """

    if self_test_runner is not None and not callable(self_test_runner):
        raise TypeError("self_test_runner must be callable")
    source = Path(bundle_dir)
    root = Path(install_root)
    with network_blocked_if_offline(local_only=True):
        verification = verify_offline_bundle(
            source,
            host=host,
            signature_key=signature_key,
            signature_public_key=signature_public_key,
        )

        def failed(
            reason: str, self_tests: OfflineSelfTestReport | None = None
        ) -> OfflineBundleInstallation:
            _LOGGER.info("offline bundle installation failed: reason=%s", reason)
            return OfflineBundleInstallation(
                installed=False,
                reason=reason,
                verification=verification,
                self_tests=self_tests,
            )

        if not verification.valid:
            return failed(verification.reason)
        if self_test_runner is None and require_self_tests:
            return failed(REASON_SELF_TEST_RUNNER_REQUIRED)
        final = root / verification.bundle_id / verification.bundle_version
        if final.exists() or final.is_symlink():
            return failed(REASON_INSTALL_TARGET_EXISTS)
        try:
            root.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=".openmed-bundle-staging-", dir=str(root))
            )
        except OSError:
            return failed(REASON_INSTALL_IO_ERROR)

        try:
            if not _copy_verified_bundle(source, staging, verification):
                return failed(REASON_INSTALL_COPY_MISMATCH)
            report = None
            if self_test_runner is not None:
                report = run_offline_self_tests(
                    staging, self_test_runner, verification=verification
                )
                if not report.passed:
                    return failed(report.reason, report)
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                staging.rename(final)
            except OSError:
                if final.exists():
                    return failed(REASON_INSTALL_TARGET_EXISTS, report)
                raise
        except OSError:
            return failed(REASON_INSTALL_IO_ERROR)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    _LOGGER.info(
        "offline bundle installed: file_count=%d self_tests=%s",
        len(verification.file_digests),
        "passed" if report is not None else "deferred",
    )
    return OfflineBundleInstallation(
        installed=True,
        reason="",
        verification=verification,
        self_tests=report,
        install_path=final,
    )


def _verify_integrity(
    root: Path,
    *,
    signature_key: bytes | None,
    signature_public_key: bytes | None,
) -> SkillBundleManifest:
    """Verify signature, checksums, and manifest shape; return the manifest."""

    try:
        manifest_bytes = (root / MANIFEST_FILENAME).read_bytes()
    except OSError:
        raise _CheckFailure(REASON_MANIFEST_MISSING) from None
    raw = _strict_json(manifest_bytes, REASON_MANIFEST_MALFORMED)
    if not isinstance(raw, dict):
        raise _CheckFailure(REASON_MANIFEST_MALFORMED)
    if set(raw) != _MANIFEST_FIELDS:
        raise _CheckFailure(REASON_MANIFEST_FIELDS_UNSUPPORTED)
    try:
        manifest = SkillBundleManifest.from_mapping(raw)
    except (TypeError, ValueError):
        raise _CheckFailure(REASON_MANIFEST_MALFORMED) from None
    try:
        if any(path.is_symlink() for path in root.rglob("*")):
            raise _CheckFailure(REASON_SYMLINK_PRESENT)
        result = verify_bundle(
            root,
            signature_key=signature_key,
            signature_public_key=signature_public_key,
        )
    except OSError:
        raise _CheckFailure(REASON_MANIFEST_MISSING) from None
    if not result.valid:
        raise _CheckFailure(result.reason)
    verified_files = {item.path: item.declared_hash for item in result.files}
    if result.bundle_id != manifest.bundle_id or verified_files != dict(manifest.files):
        raise _CheckFailure(REASON_HASH_MISMATCH)
    if manifest.signature_scheme not in SIGNATURE_SCHEMES:
        raise _CheckFailure(REASON_UNSIGNED_BUNDLE)
    if manifest.entry_points != (DESCRIPTOR_FILENAME,):
        raise _CheckFailure(REASON_ENTRY_POINT_MISMATCH)
    return manifest


def _load_descriptor(
    root: Path, manifest: SkillBundleManifest
) -> OfflineBundleDescriptor:
    """Read the descriptor, bind it to its signed digest, and parse it."""

    payload = _read_bound_file(
        root,
        DESCRIPTOR_FILENAME,
        manifest.files[DESCRIPTOR_FILENAME],
        _MAX_DESCRIPTOR_BYTES,
        REASON_DESCRIPTOR_MALFORMED,
    )
    raw = _strict_json(payload, REASON_DESCRIPTOR_MALFORMED)
    try:
        descriptor = OfflineBundleDescriptor.from_mapping(raw)
    except OfflineBundleError as error:
        raise _CheckFailure(error.reason) from None
    if descriptor.bundle_id != manifest.bundle_id:
        raise _CheckFailure(REASON_BUNDLE_ID_MISMATCH)
    return descriptor


def _load_self_test_cases(
    root: Path,
    descriptor: OfflineBundleDescriptor,
    file_digests: Mapping[str, str],
) -> tuple[_SelfTestCase, ...]:
    """Read every self-test file bound to its digest and parse the cases."""

    seen: set[str] = set()
    cases: list[_SelfTestCase] = []
    for rel_path in descriptor.component_files(COMPONENT_SELF_TEST):
        digest = file_digests.get(rel_path)
        if digest is None:
            raise _CheckFailure(REASON_COMPONENT_FILES_MISMATCH)
        payload = _read_bound_file(
            root,
            rel_path,
            _bare_hex(digest),
            _MAX_SELF_TEST_FILE_BYTES,
            REASON_SELF_TEST_MALFORMED,
        )
        cases.extend(_parse_self_test_payload(payload, seen))
        if len(cases) > _MAX_SELF_TEST_CASES:
            raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
    if not cases:
        raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
    return tuple(cases)


def _parse_self_test_payload(payload: bytes, seen: set[str]) -> list[_SelfTestCase]:
    """Parse one JSONL self-test file; ``seen`` tracks case IDs across files."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise _CheckFailure(REASON_SELF_TEST_MALFORMED) from None
    cases: list[_SelfTestCase] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        raw = _strict_json(line.encode("utf-8"), REASON_SELF_TEST_MALFORMED)
        if not isinstance(raw, dict) or set(raw) != _SELF_TEST_FIELDS:
            raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
        if raw["synthetic"] is not True:
            raise _CheckFailure(REASON_SELF_TEST_NOT_SYNTHETIC)
        case_id = raw["case_id"]
        if not _matches(_CASE_ID_RE, case_id) or case_id in seen:
            raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
        case_text = raw["text"]
        if (
            not isinstance(case_text, str)
            or not case_text.strip()
            or len(case_text) > _MAX_SELF_TEST_TEXT_CHARS
        ):
            raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
        expected_raw = raw["expected"]
        if (
            not isinstance(expected_raw, list)
            or len(expected_raw) > _MAX_EXPECTED_SPANS
        ):
            raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
        expected: set[tuple[int, int, str]] = set()
        for span in expected_raw:
            if not isinstance(span, dict) or set(span) != _SPAN_FIELDS:
                raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
            start, end, label = span["start"], span["end"], span["label"]
            if (
                not _is_int(start)
                or not _is_int(end)
                or not 0 <= start < end <= len(case_text)
                or not _matches(_LABEL_RE, label)
            ):
                raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
            expected.add((start, end, label))
        if len(expected) != len(expected_raw):
            raise _CheckFailure(REASON_SELF_TEST_MALFORMED)
        seen.add(case_id)
        cases.append(_SelfTestCase(case_id, case_text, frozenset(expected)))
    return cases


def _copy_verified_bundle(
    source: Path, staging: Path, verification: OfflineBundleVerification
) -> bool:
    """Copy declared files and the manifest; return whether digests still match."""

    manifest = verification.manifest
    if manifest is None:
        return False
    for rel_path, digest in verification.file_digests.items():
        target = _bundle_file(staging, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_bundle_file(source, rel_path), target)
        if not hmac.compare_digest(sha256_file(target), digest):
            return False
    copied_manifest = staging / MANIFEST_FILENAME
    shutil.copyfile(source / MANIFEST_FILENAME, copied_manifest)
    try:
        raw = _strict_json(copied_manifest.read_bytes(), REASON_MANIFEST_MALFORMED)
        copied = SkillBundleManifest.from_mapping(raw)
    except (_CheckFailure, TypeError, ValueError):
        return False
    return copied == manifest


def _coerce_predictions(predictions: Any) -> frozenset[tuple[int, int, str]]:
    """Normalize runner output into a set of ``(start, end, label)`` spans."""

    if predictions is None or isinstance(predictions, (str, bytes, Mapping)):
        raise _InvalidRunnerOutput
    try:
        items = list(predictions)
    except TypeError:
        raise _InvalidRunnerOutput from None
    spans: set[tuple[int, int, str]] = set()
    for item in items:
        if isinstance(item, Mapping):
            values = (item.get("start"), item.get("end"), item.get("label"))
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            values = (item[0], item[1], item[2])
        else:
            values = (
                getattr(item, "start", None),
                getattr(item, "end", None),
                getattr(item, "label", None),
            )
        start, end, label = values
        if not _is_int(start) or not _is_int(end) or not isinstance(label, str):
            raise _InvalidRunnerOutput
        spans.add((start, end, label))
    return frozenset(spans)


def _blocked_by_offline_guard(exc: BaseException) -> bool:
    """Return whether ``exc`` or its cause chain came from the socket guard."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, OfflineModeError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _source_files(
    descriptor: OfflineBundleDescriptor, sources: Mapping[str, str | Path]
) -> dict[str, Path]:
    """Map every component file to an existing local source file."""

    if not isinstance(sources, Mapping) or set(sources) != set(
        descriptor.payload_files
    ):
        raise OfflineBundleError(
            REASON_COMPONENT_FILES_MISMATCH,
            "sources must cover exactly the declared component files",
        )
    files = {rel_path: Path(sources[rel_path]) for rel_path in sources}
    if not all(path.is_file() for path in files.values()):
        raise OfflineBundleError(
            REASON_SOURCE_FILE_MISSING, "a component source file does not exist"
        )
    return files


def _validate_self_test_sources(
    descriptor: OfflineBundleDescriptor, sources: Mapping[str, Path]
) -> None:
    """Parse self-test source files with the same rules as verification."""

    seen: set[str] = set()
    case_count = 0
    for rel_path in descriptor.component_files(COMPONENT_SELF_TEST):
        if sources[rel_path].stat().st_size > _MAX_SELF_TEST_FILE_BYTES:
            raise OfflineBundleError(
                REASON_SELF_TEST_MALFORMED, "a self-test source file is too large"
            )
        try:
            case_count += len(
                _parse_self_test_payload(sources[rel_path].read_bytes(), seen)
            )
        except OSError:
            raise OfflineBundleError(
                REASON_SOURCE_FILE_MISSING, "a self-test source file is unreadable"
            ) from None
        except _CheckFailure as failure:
            raise OfflineBundleError(
                failure.reason, "a self-test source file is invalid"
            ) from None
    if not 0 < case_count <= _MAX_SELF_TEST_CASES:
        raise OfflineBundleError(
            REASON_SELF_TEST_MALFORMED,
            "bundles need at least one synthetic self-test case",
        )


def _enforce_packaging_policy(descriptor: OfflineBundleDescriptor) -> None:
    """Apply verification's component and content policy before writing."""

    if {item.kind for item in descriptor.components} != REQUIRED_COMPONENT_KINDS:
        raise OfflineBundleError(
            REASON_COMPONENT_MISSING,
            "bundles need model, lexicon, policy, self_test, and evidence parts",
        )
    if not all(item.permissive for item in _license_records(descriptor)):
        raise OfflineBundleError(
            REASON_LICENSE_NOT_PERMISSIVE,
            "every component needs a permissive license",
        )
    if _has_restricted_source(descriptor):
        raise OfflineBundleError(
            REASON_RESTRICTED_SOURCE,
            "restricted datasets and terminologies cannot be bundled",
        )
    if any(_is_credential_like(path) for path in descriptor.payload_files):
        raise OfflineBundleError(
            REASON_CREDENTIAL_LIKE_FILE, "credential-like files cannot be bundled"
        )


def _license_records(
    descriptor: OfflineBundleDescriptor,
) -> tuple[OfflineBundleLicense, ...]:
    """Return license records checked against the permissive allowlist."""

    return tuple(
        OfflineBundleLicense(
            component_id=item.component_id,
            kind=item.kind,
            license=item.license,
            source=item.source,
            permissive=is_permissive_license(item.license),
        )
        for item in descriptor.components
    )


def _has_restricted_source(descriptor: OfflineBundleDescriptor) -> bool:
    """Return whether any component names a restricted dataset or terminology."""

    for component in descriptor.components:
        for value in (component.component_id, component.source, *component.files):
            tokens = set(_TOKEN_SPLIT_RE.split(value.lower()))
            if tokens & _RESTRICTED_SOURCE_TOKENS:
                return True
    return False


def _is_credential_like(rel_path: str) -> bool:
    """Return whether a bundle path looks like key material or credentials."""

    name = PurePosixPath(rel_path).name.lower()
    pure = PurePosixPath(name)
    return pure.suffix in _CREDENTIAL_SUFFIXES or pure.stem in _CREDENTIAL_STEMS


def _payload_bytes(root: Path, manifest: SkillBundleManifest) -> int:
    """Return the total size of the manifest and every declared file."""

    try:
        total = (root / MANIFEST_FILENAME).stat().st_size
        for rel_path in manifest.files:
            total += _bundle_file(root, rel_path).stat().st_size
    except OSError:
        raise _CheckFailure(REASON_COMPONENT_FILES_MISMATCH) from None
    return total


def _read_bound_file(
    root: Path, rel_path: str, digest_hex: str, limit: int, reason: str
) -> bytes:
    """Read a bundle file once and require it to match its signed digest."""

    path = _bundle_file(root, rel_path)
    try:
        if path.stat().st_size > limit:
            raise _CheckFailure(reason)
        payload = path.read_bytes()
    except OSError:
        raise _CheckFailure(REASON_HASH_MISMATCH) from None
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), digest_hex):
        raise _CheckFailure(REASON_HASH_MISMATCH)
    return payload


def _strict_json(payload: bytes, reason: str) -> Any:
    """Decode JSON, rejecting duplicate keys and non-finite numbers."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _CheckFailure(reason)
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        raise _CheckFailure(reason)

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise _CheckFailure(reason) from None


def _signing_scheme(signature_key: Any, ed25519_private_key: Any) -> str:
    """Return the signing scheme selected by exactly one key argument."""

    if (signature_key is None) == (ed25519_private_key is None):
        raise OfflineBundleError(
            REASON_SIGNING_KEY_REQUIRED,
            "provide exactly one of signature_key or ed25519_private_key",
        )
    key = signature_key if signature_key is not None else ed25519_private_key
    if not isinstance(key, (bytes, bytearray)) or not key:
        raise OfflineBundleError(
            REASON_SIGNING_KEY_INVALID, "signing keys must be non-empty bytes"
        )
    return "hmac-sha256" if signature_key is not None else "ed25519"


def _sign(canonical: bytes, *, scheme: str, key: Any) -> str:
    """Sign canonical manifest bytes with the verifier-compatible scheme."""

    if scheme == "hmac-sha256":
        return hmac.new(bytes(key), canonical, hashlib.sha256).hexdigest()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        raise OfflineBundleError(
            REASON_SIGNATURE_DEPENDENCY_MISSING,
            "ed25519 signing requires the openmed[integrity] extra",
        ) from None
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes(key))
    except ValueError:
        raise OfflineBundleError(
            REASON_SIGNING_KEY_INVALID, "ed25519 private keys must be 32 raw bytes"
        ) from None
    return private_key.sign(canonical).hex()


def _physical_memory_bytes() -> int | None:
    """Return total physical memory from ``sysconf``, when the OS exposes it."""

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if not _is_int(pages) or not _is_int(page_size) or pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def _validate_bundle_path(path: str) -> None:
    """Require a safe, relative POSIX path that is not a reserved file."""

    if path.casefold() in _RESERVED_PATHS or len(path) > 255:
        _malformed("component file paths must not use reserved names")
    segments = path.split("/")
    if len(segments) > _MAX_PATH_SEGMENTS or not all(
        _PATH_SEGMENT_RE.fullmatch(segment) for segment in segments
    ):
        _malformed("component file paths must be safe relative POSIX paths")


def _bundle_file(root: Path, rel_path: str) -> Path:
    """Join a validated bundle-relative POSIX path onto ``root``."""

    return root.joinpath(*PurePosixPath(rel_path).parts)


def _exact_mapping(value: Any, expected: frozenset[str], name: str) -> dict[str, Any]:
    """Require a mapping with exactly the ``expected`` keys."""

    if not isinstance(value, Mapping) or set(value) != expected:
        _malformed(f"{name} fields do not match the public contract")
    return dict(value)


def _string_items(value: Any, name: str) -> tuple[str, ...]:
    """Return a bounded tuple of strings from a list-like value."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        _malformed(f"{name} must be a list of strings")
    if len(value) > _MAX_COLLECTION_ITEMS or not all(
        isinstance(item, str) for item in value
    ):
        _malformed(f"{name} must be a list of strings")
    return tuple(value)


def _unique_sorted(value: Any, pattern: re.Pattern[str], name: str) -> tuple[str, ...]:
    """Validate a non-empty unique string collection and sort it."""

    items = _string_items(value, name)
    if not items or len(set(items)) != len(items):
        _malformed(f"{name} must be a non-empty list of unique values")
    if not all(pattern.fullmatch(item) for item in items):
        _malformed(f"{name} contains an unsupported value")
    return tuple(sorted(items))


def _component_items(value: Any) -> tuple[OfflineBundleComponent, ...]:
    """Require a tuple or list of components."""

    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, OfflineBundleComponent) for item in value
    ):
        _malformed("components must be OfflineBundleComponent values")
    return tuple(value)


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    """Return whether ``value`` is a string fully matching ``pattern``."""

    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_int(value: Any) -> bool:
    """Return whether ``value`` is an integer and not a boolean."""

    return isinstance(value, int) and not isinstance(value, bool)


def _prefixed(hex_digest: str) -> str:
    """Return a ``sha256:`` prefixed digest."""

    return f"sha256:{hex_digest}"


def _bare_hex(digest: str) -> str:
    """Strip an optional ``sha256:`` prefix from a digest."""

    return digest.removeprefix("sha256:")


def _malformed(message: str) -> NoReturn:
    """Raise a descriptor error with a content-free message."""

    raise OfflineBundleError(REASON_DESCRIPTOR_MALFORMED, message)


__all__ = [
    "CHECK_COMPONENTS",
    "CHECK_CREDENTIALS",
    "CHECK_DESCRIPTOR",
    "CHECK_FAILED",
    "CHECK_INTEGRITY",
    "CHECK_LICENSES",
    "CHECK_NAMES",
    "CHECK_NOT_RUN",
    "CHECK_PASSED",
    "CHECK_RAM",
    "CHECK_RESTRICTED_SOURCES",
    "CHECK_SELF_TESTS",
    "CHECK_STORAGE",
    "COMPONENT_EVIDENCE",
    "COMPONENT_KINDS",
    "COMPONENT_LEXICON",
    "COMPONENT_MODEL",
    "COMPONENT_POLICY",
    "COMPONENT_SELF_TEST",
    "DATA_CLASSIFICATIONS",
    "DESCRIPTOR_FILENAME",
    "MANIFEST_FILENAME",
    "OFFLINE_BUNDLE_MANIFEST_VERSION",
    "OFFLINE_BUNDLE_SCHEMA_VERSION",
    "REASON_BUNDLE_ID_MISMATCH",
    "REASON_BUNDLE_NOT_VERIFIED",
    "REASON_COMPONENT_FILES_MISMATCH",
    "REASON_COMPONENT_MISSING",
    "REASON_CREDENTIAL_LIKE_FILE",
    "REASON_DESCRIPTOR_MALFORMED",
    "REASON_ENTRY_POINT_MISMATCH",
    "REASON_HOST_FACTS_UNAVAILABLE",
    "REASON_INSTALL_COPY_MISMATCH",
    "REASON_INSTALL_IO_ERROR",
    "REASON_INSTALL_TARGET_EXISTS",
    "REASON_INSUFFICIENT_RAM",
    "REASON_INSUFFICIENT_STORAGE",
    "REASON_LICENSE_NOT_PERMISSIVE",
    "REASON_MANIFEST_FIELDS_UNSUPPORTED",
    "REASON_MANIFEST_MISSING",
    "REASON_NETWORK_ATTEMPTED",
    "REASON_OUTPUT_NOT_EMPTY",
    "REASON_RESTRICTED_SOURCE",
    "REASON_SELF_TEST_FAILED",
    "REASON_SELF_TEST_MALFORMED",
    "REASON_SELF_TEST_NOT_SYNTHETIC",
    "REASON_SELF_TEST_RUNNER_REQUIRED",
    "REASON_SIGNING_KEY_INVALID",
    "REASON_SIGNING_KEY_REQUIRED",
    "REASON_SOURCE_FILE_MISSING",
    "REASON_STORAGE_REQUIREMENT_UNDERSTATED",
    "REASON_SYMLINK_PRESENT",
    "REASON_UNSIGNED_BUNDLE",
    "REQUIRED_COMPONENT_KINDS",
    "SELF_TEST_ERROR_INVALID_OUTPUT",
    "SELF_TEST_ERROR_RUNNER",
    "SIGNATURE_SCHEMES",
    "HostResources",
    "OfflineBundleCheck",
    "OfflineBundleComponent",
    "OfflineBundleDescriptor",
    "OfflineBundleError",
    "OfflineBundleInstallation",
    "OfflineBundleLicense",
    "OfflineBundleVerification",
    "OfflineSelfTestCaseResult",
    "OfflineSelfTestReport",
    "OfflineSelfTestRunner",
    "PackagedOfflineBundle",
    "ResourceRequirements",
    "install_offline_bundle",
    "package_offline_bundle",
    "run_offline_self_tests",
    "verify_offline_bundle",
]
