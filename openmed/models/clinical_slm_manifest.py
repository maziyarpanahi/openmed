"""Immutable, offline manifests for local clinical SLM artifacts.

The manifest is the hand-off boundary between a locally prepared clinical
small language model package and a model runtime.  It records only bounded
metadata: the pinned model revision, the relative files that make up the
package, their SHA-256 digests and sizes, quantization metadata, component
licenses, and supported task identifiers.  It never contains model bytes,
prompt text, clinical text, credentials, or URLs.

``verify_clinical_slm_package`` is deliberately a local gate.  It reads the
manifest and the declared files, rejects missing or mutable package members,
and verifies every declared digest before a caller constructs a model.  The
module has no network or optional-runtime dependency, so a failed verification
cannot fall through to a remote model or a different revision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

CLINICAL_SLM_MANIFEST_SCHEMA_VERSION: Final = "openmed.clinical-slm-artifact.v1"
CLINICAL_SLM_MANIFEST_VERSION: Final = 1
CLINICAL_SLM_MANIFEST_FILENAME: Final = "clinical-slm-manifest.json"
MANIFEST_SCHEMA_VERSION: Final = CLINICAL_SLM_MANIFEST_SCHEMA_VERSION
SCHEMA_VERSION: Final = CLINICAL_SLM_MANIFEST_SCHEMA_VERSION
MANIFEST_FILENAME: Final = CLINICAL_SLM_MANIFEST_FILENAME

MAX_COMPONENTS: Final = 4_096
MAX_IDENTIFIER_LENGTH: Final = 128
MAX_MODEL_ID_LENGTH: Final = 256
MAX_PATH_LENGTH: Final = 512
MAX_ARTIFACT_BYTES: Final = (1 << 63) - 1

_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+/@-]{0,127}$")
_TASK_RE = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_VERSION_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_IMMUTABLE_REVISION_RE = re.compile(
    r"^(?:[0-9a-fA-F]{40}|(?:sha256:)?[0-9a-fA-F]{64})$"
)
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_REQUIRED_COMPONENTS: Final = frozenset(
    {"weights", "tokenizer", "templates", "quantization"}
)
_COMPONENT_ALIASES: Final = {
    "weight": "weights",
    "model": "weights",
    "prompt-template": "templates",
    "prompt-templates": "templates",
    "template": "templates",
    "quant": "quantization",
    "quantizer": "quantization",
}

# This is intentionally a small, explicit SPDX vocabulary.  A caller can add
# a new license only by extending this module and its tests; an arbitrary
# ``other`` or ``unknown`` value must never pass the loading gate.
_LICENSE_ALIASES: Final = {
    "apache-2.0": "apache-2.0",
    "apache 2.0": "apache-2.0",
    "apache2": "apache-2.0",
    "bsd-2-clause": "bsd-2-clause",
    "bsd 2-clause": "bsd-2-clause",
    "bsd-3-clause": "bsd-3-clause",
    "bsd 3-clause": "bsd-3-clause",
    "cc-by-3.0": "cc-by-3.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc0-1.0": "cc0-1.0",
    "isc": "isc",
    "mit": "mit",
    "mpl-2.0": "mpl-2.0",
    "public-domain": "public-domain",
    "unlicense": "unlicense",
}

_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_version",
        "model_id",
        "model_version",
        "version",
        "revision",
        "model_revision",
        "components",
        "artifacts",
        "weights",
        "tokenizer",
        "templates",
        "quantization",
        "licenses",
        "license",
        "supported_tasks",
        "tasks",
        "offline",
        "local_only",
        "human_review_required",
        "review_required",
        "manifest_digest",
        "manifest_sha256",
        "digest",
    }
)
_COMPONENT_FIELDS: Final = frozenset(
    {
        "component",
        "kind",
        "name",
        "artifact_type",
        "path",
        "sha256",
        "digest",
        "size_bytes",
        "size",
        "bytes",
        "executable",
        "format",
        "format_name",
    }
)
_QUANTIZATION_FIELDS: Final = frozenset(
    {"scheme", "method", "type", "bits", "group_size", "symmetric"}
)
_LICENSE_FIELDS: Final = frozenset(
    {"component", "kind", "name", "artifact_type", "license", "spdx_id"}
)

_ERROR_MESSAGES: Final = {
    "invalid_input": "clinical SLM manifest input is invalid",
    "unknown_field": "clinical SLM manifest contains an unsupported field",
    "missing_field": "clinical SLM manifest is missing a required field",
    "duplicate_field": "clinical SLM manifest contains duplicate fields",
    "unsupported_schema": "clinical SLM manifest schema is unsupported",
    "invalid_identifier": "clinical SLM manifest contains an invalid identifier",
    "invalid_model_id": "clinical SLM manifest model id is invalid",
    "invalid_version": "clinical SLM manifest version is invalid",
    "mutable_revision": "clinical SLM manifest revision must be immutable",
    "invalid_digest": "clinical SLM manifest digest is invalid",
    "manifest_digest_mismatch": "clinical SLM manifest digest does not match",
    "invalid_component": "clinical SLM manifest component is invalid",
    "missing_component": "clinical SLM manifest is missing a required component",
    "duplicate_component": "clinical SLM manifest contains duplicate components",
    "missing_license": "clinical SLM manifest is missing a component license",
    "unknown_license": "clinical SLM manifest contains an unknown license",
    "duplicate_license": "clinical SLM manifest contains duplicate licenses",
    "invalid_quantization": "clinical SLM manifest quantization is invalid",
    "invalid_tasks": "clinical SLM manifest supported tasks are invalid",
    "offline_required": "clinical SLM manifest must be offline-only",
    "human_review_required": "clinical SLM manifest must require human review",
    "manifest_missing": "clinical SLM manifest file is missing",
    "manifest_unreadable": "clinical SLM manifest file cannot be read",
    "manifest_digest_required": "clinical SLM manifest digest is required",
    "package_invalid": "clinical SLM package root is invalid",
    "component_missing_on_disk": "clinical SLM package component is missing",
    "component_unreadable": "clinical SLM package component cannot be read",
    "unsafe_component_path": "clinical SLM package component path is unsafe",
    "component_size_mismatch": "clinical SLM package component size does not match",
    "component_digest_mismatch": "clinical SLM package component digest does not match",
    "component_mutated": "clinical SLM package component changed during verification",
}
_SAFE_ERROR_FIELDS: Final = frozenset(
    {
        "schema_version",
        "model_id",
        "model_version",
        "revision",
        "components",
        "component",
        "quantization",
        "licenses",
        "supported_tasks",
        "manifest_digest",
        "package_root",
    }
)
_MISSING = object()


class ClinicalSLMManifestError(ValueError):
    """Base error for malformed or unsafe clinical SLM metadata.

    The exception stores only a stable reason code and an allow-listed field
    name.  It intentionally never includes a caller-provided path, digest,
    model id, license text, or other value that could contain sensitive data.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code if code in _ERROR_MESSAGES else "invalid_input"
        self.field_name = field_name if field_name in _SAFE_ERROR_FIELDS else None
        message = _ERROR_MESSAGES[self.code]
        if self.field_name is not None:
            message = f"{message} ({self.field_name})"
        super().__init__(message)


class ClinicalSLMValidationError(ClinicalSLMManifestError):
    """Raised when a manifest value fails the clinical SLM contract."""


class ClinicalSLMArtifactError(ClinicalSLMManifestError):
    """Raised when a local package cannot satisfy its manifest."""


class ClinicalSLMArtifactMissingError(ClinicalSLMArtifactError):
    """Raised when a declared local artifact is absent or not a file."""


class ClinicalSLMArtifactDigestMismatchError(ClinicalSLMArtifactError):
    """Raised when a local artifact does not match its declared digest."""


def _fail(
    code: str,
    field_name: str | None = None,
    *,
    error_type: type[ClinicalSLMManifestError] = ClinicalSLMValidationError,
) -> None:
    raise error_type(code, field_name) from None


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("invalid_input")
    raise AssertionError("unreachable")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _normalise_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail("invalid_digest", field_name)
    digest = value.lower()
    return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def _normalise_identifier(value: Any, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_IDENTIFIER_LENGTH
        or _CONTROL_RE.search(value)
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        _fail("invalid_identifier", field_name)
    return value


def _normalise_component_kind(value: Any) -> str:
    kind = _normalise_identifier(value, "component")
    normalized = _COMPONENT_ALIASES.get(kind.lower(), kind.lower())
    if normalized in {"unknown", "unspecified", "other"}:
        _fail("invalid_component", "component")
    return normalized


def _normalise_model_id(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_MODEL_ID_LENGTH
        or _CONTROL_RE.search(value)
        or "://" in value
        or "\\" in value
        or "//" in value
        or ".." in value
        or _MODEL_ID_RE.fullmatch(value) is None
    ):
        _fail("invalid_model_id", "model_id")
    return value


def _normalise_version(value: Any, field_name: str = "model_version") -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        _fail("invalid_version", field_name)
    return value


def _normalise_revision(value: Any) -> str:
    if (
        type(value) is not str
        or _IMMUTABLE_REVISION_RE.fullmatch(value) is None
        or _CONTROL_RE.search(value)
    ):
        _fail("mutable_revision", "revision")
    return value.lower()


def _normalise_path(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_PATH_LENGTH
        or _CONTROL_RE.search(value)
        or "\\" in value
        or value.startswith("/")
        or "//" in value
        or _PATH_RE.fullmatch(value) is None
    ):
        _fail("unsafe_component_path", "component")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {".", ".."} for part in parts):
        _fail("unsafe_component_path", "component")
    return value


def _normalise_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 < value <= MAX_ARTIFACT_BYTES:
        _fail("invalid_component", field_name)
    return value


def _normalise_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        _fail("invalid_input", field_name)
    return value


def _first_value(
    payload: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: Any = _MISSING,
) -> Any:
    present = [name for name in names if name in payload]
    if len(present) > 1:
        _fail("invalid_input")
    if present:
        return payload[present[0]]
    if default is not _MISSING:
        return default
    _fail("missing_field")
    raise AssertionError("unreachable")


def _mapping_copy(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        _fail("invalid_input", field_name)
    try:
        copied = dict(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail("invalid_input", field_name)
    if any(type(key) is not str for key in copied):
        _fail("invalid_input", field_name)
    return copied


@dataclass(frozen=True, slots=True, repr=False, init=False)
class ClinicalSLMArtifact:
    """One content-addressed file in a clinical SLM package.

    ``component`` is a logical role such as ``weights`` or ``tokenizer``;
    multiple weight shards are allowed.  ``path`` is package-relative and is
    never emitted in verification errors or aggregate reports.
    """

    component: str
    path: str
    sha256: str
    size_bytes: int
    executable: bool
    format: str | None

    def __init__(
        self,
        component: str | None = None,
        path: str | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
        executable: bool = True,
        format: str | None = None,
        *,
        kind: str | None = None,
        name: str | None = None,
        artifact_type: str | None = None,
        digest: str | None = None,
        size: int | None = None,
        bytes: int | None = None,
        format_name: str | None = None,
    ) -> None:
        component_value = _first_alias(
            (component, kind, name, artifact_type), default=None
        )
        digest_value = _first_alias((sha256, digest), default=None)
        size_value = _first_alias((size_bytes, size, bytes), default=None)
        if component_value is None or path is None or digest_value is None:
            _fail("missing_field", "component")
        object.__setattr__(
            self,
            "component",
            _normalise_component_kind(component_value),
        )
        object.__setattr__(self, "path", _normalise_path(path))
        object.__setattr__(self, "sha256", _normalise_digest(digest_value, "component"))
        if size_value is None:
            _fail("missing_field", "component")
        object.__setattr__(
            self,
            "size_bytes",
            _normalise_positive_int(size_value, "component"),
        )
        object.__setattr__(
            self,
            "executable",
            _normalise_bool(executable, "component"),
        )
        if format is None:
            format = format_name
        if format is not None:
            format = _normalise_identifier(format, "component")
        object.__setattr__(self, "format", format)

    def __repr__(self) -> str:
        """Return a content-free representation for safe logging."""

        return "ClinicalSLMArtifact(<metadata>)"

    @property
    def kind(self) -> str:
        """Return the logical component role."""

        return self.component

    @property
    def name(self) -> str:
        """Return the logical component role under the name alias."""

        return self.component

    @property
    def digest(self) -> str:
        """Return the declared SHA-256 digest."""

        return self.sha256

    @property
    def size(self) -> int:
        """Return the declared artifact size."""

        return self.size_bytes

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_component: str | None = None,
    ) -> "ClinicalSLMArtifact":
        """Build an artifact from a strict JSON-compatible mapping."""

        fields = _mapping_copy(payload, field_name="component")
        if set(fields) - _COMPONENT_FIELDS:
            _fail("unknown_field", "component")
        component = _first_value(
            fields,
            ("component", "kind", "name", "artifact_type"),
            default=default_component,
        )
        if component is None:
            _fail("missing_field", "component")
        return cls(
            component=component,
            path=_first_value(fields, ("path",)),
            sha256=_first_value(fields, ("sha256", "digest")),
            size_bytes=_first_value(fields, ("size_bytes", "size", "bytes")),
            executable=_first_value(fields, ("executable",), default=True),
            format=_first_value(fields, ("format", "format_name"), default=None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, metadata-only artifact fields."""

        payload: dict[str, Any] = {
            "component": self.component,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "executable": self.executable,
        }
        if self.format is not None:
            payload["format"] = self.format
        return payload


ManifestArtifact = ClinicalSLMArtifact
ArtifactComponent = ClinicalSLMArtifact
ClinicalSLMComponent = ClinicalSLMArtifact


def _first_alias(values: Sequence[Any], *, default: Any = _MISSING) -> Any:
    present = [value for value in values if value is not None]
    if len(present) > 1:
        _fail("invalid_input")
    if present:
        return present[0]
    return default if default is not _MISSING else None


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMQuantization:
    """Bounded quantization metadata for the runtime-selected weights."""

    scheme: str
    bits: int | None = None
    group_size: int | None = None
    symmetric: bool | None = None

    def __post_init__(self) -> None:
        if (
            type(self.scheme) is not str
            or not self.scheme
            or _CONTROL_RE.search(self.scheme)
            or _TASK_RE.fullmatch(self.scheme.lower()) is None
        ):
            _fail("invalid_quantization", "quantization")
        scheme = self.scheme.lower()
        if scheme in {"unknown", "unspecified", "other"}:
            _fail("invalid_quantization", "quantization")
        object.__setattr__(self, "scheme", scheme)
        if self.bits is not None and (
            type(self.bits) is not int or self.bits not in {2, 3, 4, 8, 16, 32}
        ):
            _fail("invalid_quantization", "quantization")
        if self.group_size is not None and (
            type(self.group_size) is not int
            or not 0 < self.group_size <= MAX_IDENTIFIER_LENGTH * 1024
        ):
            _fail("invalid_quantization", "quantization")
        if self.symmetric is not None and type(self.symmetric) is not bool:
            _fail("invalid_quantization", "quantization")
        if self.bits is None:
            inferred = {
                "int2": 2,
                "int3": 3,
                "int4": 4,
                "int8": 8,
            }.get(scheme)
            if inferred is not None:
                object.__setattr__(self, "bits", inferred)

    def __repr__(self) -> str:
        """Return a safe shape-only representation."""

        return "ClinicalSLMQuantization(<metadata>)"

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | str
    ) -> "ClinicalSLMQuantization":
        """Build quantization metadata from a fixed, value-bounded mapping."""

        if type(payload) is str:
            return cls(scheme=payload)
        fields = _mapping_copy(payload, field_name="quantization")
        if set(fields) - _QUANTIZATION_FIELDS:
            _fail("unknown_field", "quantization")
        return cls(
            scheme=_first_value(fields, ("scheme", "method", "type")),
            bits=_first_value(fields, ("bits",), default=None),
            group_size=_first_value(fields, ("group_size",), default=None),
            symmetric=_first_value(fields, ("symmetric",), default=None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic quantization metadata."""

        payload: dict[str, Any] = {"scheme": self.scheme}
        if self.bits is not None:
            payload["bits"] = self.bits
        if self.group_size is not None:
            payload["group_size"] = self.group_size
        if self.symmetric is not None:
            payload["symmetric"] = self.symmetric
        return payload


QuantizationSpec = ClinicalSLMQuantization


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMLicense:
    """One explicit SPDX license declaration for a component role."""

    component: str
    spdx_id: str

    def __post_init__(self) -> None:
        component = self.component.lower() if type(self.component) is str else None
        if component != "all":
            component = _normalise_component_kind(self.component)
        if component is None:
            _fail("invalid_identifier", "licenses")
        object.__setattr__(self, "component", component)
        if type(self.spdx_id) is not str:
            _fail("unknown_license", "licenses")
        canonical = _LICENSE_ALIASES.get(self.spdx_id.strip().lower())
        if canonical is None:
            _fail("unknown_license", "licenses")
        object.__setattr__(self, "spdx_id", canonical)

    def __repr__(self) -> str:
        """Return a value-free representation for safe diagnostics."""

        return "ClinicalSLMLicense(<metadata>)"

    @property
    def kind(self) -> str:
        """Return the licensed component role."""

        return self.component

    @property
    def license(self) -> str:
        """Return the canonical SPDX identifier."""

        return self.spdx_id

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClinicalSLMLicense":
        """Build a license declaration from a strict mapping."""

        fields = _mapping_copy(payload, field_name="licenses")
        if set(fields) - _LICENSE_FIELDS:
            _fail("unknown_field", "licenses")
        return cls(
            component=_first_value(
                fields,
                ("component", "kind", "name", "artifact_type"),
            ),
            spdx_id=_first_value(fields, ("spdx_id", "license")),
        )

    def to_dict(self) -> dict[str, str]:
        """Return deterministic license metadata."""

        return {"component": self.component, "spdx_id": self.spdx_id}


LicenseDeclaration = ClinicalSLMLicense
ComponentLicense = ClinicalSLMLicense


def _normalise_components(value: Any) -> tuple[ClinicalSLMArtifact, ...]:
    if isinstance(value, Mapping):
        entries: list[ClinicalSLMArtifact] = []
        for component, members in value.items():
            if type(component) is not str:
                _fail("invalid_component", "components")
            if isinstance(members, Mapping):
                members_to_read: Iterable[Any] = (members,)
            elif isinstance(members, Sequence) and not isinstance(
                members, (str, bytes, bytearray)
            ):
                members_to_read = members
            else:
                _fail("invalid_component", "components")
            for member in members_to_read:
                if isinstance(member, ClinicalSLMArtifact):
                    if member.component != _normalise_component_kind(component):
                        _fail("invalid_component", "components")
                    entries.append(member)
                else:
                    entries.append(
                        ClinicalSLMArtifact.from_mapping(
                            member,
                            default_component=component,
                        )
                    )
        values = entries
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = [
            item
            if isinstance(item, ClinicalSLMArtifact)
            else ClinicalSLMArtifact.from_mapping(item)
            for item in value
        ]
    else:
        _fail("invalid_component", "components")

    if not values or len(values) > MAX_COMPONENTS:
        _fail("missing_component", "components")
    ordered = tuple(sorted(values, key=lambda item: (item.component, item.path)))
    identities = [(item.component, item.path) for item in ordered]
    if len(set(identities)) != len(identities):
        _fail("duplicate_component", "components")
    present = {item.component for item in ordered}
    if not _REQUIRED_COMPONENTS.issubset(present):
        _fail("missing_component", "components")
    return ordered


def _normalise_licenses(value: Any) -> tuple[ClinicalSLMLicense, ...]:
    if type(value) is str:
        entries: Iterable[Any] = ({"component": "all", "spdx_id": value},)
    elif isinstance(value, Mapping):
        entries = (
            {"component": component, "spdx_id": license_value}
            for component, license_value in value.items()
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        entries = value
    else:
        _fail("invalid_input", "licenses")

    licenses: list[ClinicalSLMLicense] = []
    for entry in entries:
        if isinstance(entry, ClinicalSLMLicense):
            licenses.append(entry)
        elif isinstance(entry, Mapping):
            licenses.append(ClinicalSLMLicense.from_mapping(entry))
        else:
            _fail("invalid_input", "licenses")
    if not licenses:
        _fail("missing_license", "licenses")
    ordered = tuple(sorted(licenses, key=lambda item: item.component))
    if len({item.component for item in ordered}) != len(ordered):
        _fail("duplicate_license", "licenses")
    return ordered


def _normalise_tasks(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail("invalid_tasks", "supported_tasks")
    tasks: list[str] = []
    for task in value:
        if (
            type(task) is not str
            or not task
            or len(task) > MAX_IDENTIFIER_LENGTH
            or _TASK_RE.fullmatch(task) is None
            or task in {"unknown", "unspecified"}
        ):
            _fail("invalid_tasks", "supported_tasks")
        tasks.append(task)
    if not tasks or len(tasks) > MAX_COMPONENTS or len(set(tasks)) != len(tasks):
        _fail("invalid_tasks", "supported_tasks")
    return tuple(sorted(tasks))


def _manifest_material(
    *,
    schema_version: str,
    model_id: str,
    model_version: str,
    revision: str,
    components: Sequence[ClinicalSLMArtifact],
    quantization: ClinicalSLMQuantization,
    licenses: Sequence[ClinicalSLMLicense],
    supported_tasks: Sequence[str],
    offline: bool,
    human_review_required: bool,
) -> dict[str, Any]:
    return {
        "components": [item.to_dict() for item in components],
        "human_review_required": human_review_required,
        "licenses": [item.to_dict() for item in licenses],
        "model_id": model_id,
        "model_version": model_version,
        "offline": offline,
        "quantization": quantization.to_dict(),
        "revision": revision,
        "schema_version": schema_version,
        "supported_tasks": list(supported_tasks),
    }


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMArtifactManifest(Mapping[str, Any]):
    """Immutable metadata required before loading a local clinical SLM.

    The four required component roles are ``weights``, ``tokenizer``,
    ``templates``, and ``quantization``.  Each role is represented by one or
    more content-addressed artifacts.  ``manifest_digest`` is derived from
    every field except itself and is checked when supplied by a serialized
    manifest.
    """

    model_id: str
    revision: str
    components: tuple[ClinicalSLMArtifact, ...]
    quantization: ClinicalSLMQuantization | Mapping[str, Any] | str
    licenses: tuple[ClinicalSLMLicense, ...] | Mapping[str, Any] | Sequence[Any]
    supported_tasks: tuple[str, ...] | Sequence[str]
    schema_version: str = CLINICAL_SLM_MANIFEST_SCHEMA_VERSION
    model_version: str = "1.0.0"
    offline: bool = True
    human_review_required: bool = True
    manifest_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CLINICAL_SLM_MANIFEST_SCHEMA_VERSION:
            _fail("unsupported_schema", "schema_version")
        object.__setattr__(self, "model_id", _normalise_model_id(self.model_id))
        object.__setattr__(self, "revision", _normalise_revision(self.revision))
        object.__setattr__(
            self,
            "model_version",
            _normalise_version(self.model_version),
        )
        components = _normalise_components(self.components)
        object.__setattr__(self, "components", components)
        if isinstance(self.quantization, ClinicalSLMQuantization):
            quantization = self.quantization
        else:
            quantization = ClinicalSLMQuantization.from_mapping(self.quantization)
        object.__setattr__(self, "quantization", quantization)
        licenses = _normalise_licenses(self.licenses)
        component_kinds = {item.component for item in components}
        licensed_kinds = {item.component for item in licenses}
        if "all" not in licensed_kinds and not component_kinds.issubset(licensed_kinds):
            _fail("missing_license", "licenses")
        object.__setattr__(self, "licenses", licenses)
        tasks = _normalise_tasks(self.supported_tasks)
        object.__setattr__(self, "supported_tasks", tasks)
        if type(self.offline) is not bool or not self.offline:
            _fail("offline_required", "offline")
        if (
            type(self.human_review_required) is not bool
            or not self.human_review_required
        ):
            _fail("human_review_required", "human_review_required")

        material = _manifest_material(
            schema_version=self.schema_version,
            model_id=self.model_id,
            model_version=self.model_version,
            revision=self.revision,
            components=components,
            quantization=quantization,
            licenses=licenses,
            supported_tasks=tasks,
            offline=self.offline,
            human_review_required=self.human_review_required,
        )
        expected_digest = _sha256_bytes(_canonical_json(material).encode("utf-8"))
        if self.manifest_digest is not None:
            supplied_digest = _normalise_digest(self.manifest_digest, "manifest_digest")
            if supplied_digest != expected_digest:
                _fail("manifest_digest_mismatch", "manifest_digest")
        else:
            supplied_digest = expected_digest
        object.__setattr__(self, "manifest_digest", supplied_digest)

    def __repr__(self) -> str:
        """Return a content-free representation for safe logging."""

        return "ClinicalSLMArtifactManifest(<validated metadata>)"

    @property
    def model_revision(self) -> str:
        """Return the immutable revision under the descriptive alias."""

        return self.revision

    @property
    def version(self) -> str:
        """Return the model package version."""

        return self.model_version

    @property
    def artifacts(self) -> tuple[ClinicalSLMArtifact, ...]:
        """Return the declared artifacts in deterministic order."""

        return self.components

    @property
    def manifest_sha256(self) -> str:
        """Return the canonical manifest digest."""

        return self.manifest_digest  # type: ignore[return-value]

    @property
    def component_count(self) -> int:
        """Return the number of declared artifacts."""

        return len(self.components)

    @property
    def weights(self) -> tuple[ClinicalSLMArtifact, ...]:
        """Return the declared weight artifacts."""

        return self.components_for("weights")

    @property
    def tokenizer_artifacts(self) -> tuple[ClinicalSLMArtifact, ...]:
        """Return the declared tokenizer artifacts."""

        return self.components_for("tokenizer")

    @property
    def template_artifacts(self) -> tuple[ClinicalSLMArtifact, ...]:
        """Return the declared template artifacts."""

        return self.components_for("templates")

    @property
    def quantization_artifacts(self) -> tuple[ClinicalSLMArtifact, ...]:
        """Return the content-addressed quantization artifacts."""

        return self.components_for("quantization")

    def components_for(self, component: str) -> tuple[ClinicalSLMArtifact, ...]:
        """Return artifacts for one logical component role."""

        kind = _normalise_component_kind(component)
        return tuple(item for item in self.components if item.component == kind)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        require_manifest_digest: bool = False,
    ) -> "ClinicalSLMArtifactManifest":
        """Build a validated manifest from a strict metadata mapping.

        ``manifest_digest`` is derived when omitted for programmatic records.
        Disk loaders enable ``require_manifest_digest`` so a persisted package
        must carry the explicit immutable record digest.
        """

        fields = _mapping_copy(payload, field_name="manifest_digest")
        if set(fields) - _MANIFEST_FIELDS:
            _fail("unknown_field")
        digest = _first_value(
            fields,
            ("manifest_digest", "manifest_sha256", "digest"),
            default=None,
        )
        if require_manifest_digest and digest is None:
            _fail("manifest_digest_required", "manifest_digest")

        component_value = _first_value(
            fields,
            ("components", "artifacts"),
            default=None,
        )
        role_values: list[Any] = []
        if component_value is not None:
            role_values.append(component_value)
        for role in ("weights", "tokenizer", "templates"):
            if role in fields:
                role_members = fields[role]
                if isinstance(role_members, Mapping):
                    role_members = [role_members]
                elif not isinstance(role_members, Sequence) or isinstance(
                    role_members, (str, bytes, bytearray)
                ):
                    _fail("invalid_component", "components")
                role_values.append(
                    [
                        {
                            **_mapping_copy(member, field_name="component"),
                            "component": role,
                        }
                        for member in role_members
                    ]
                )
        if not role_values:
            _fail("missing_component", "components")
        components = _flatten_component_values(role_values)
        return cls(
            schema_version=_first_value(
                fields,
                ("schema_version",),
                default=CLINICAL_SLM_MANIFEST_SCHEMA_VERSION,
            ),
            model_id=_first_value(fields, ("model_id",)),
            model_version=_first_value(
                fields,
                ("model_version", "version"),
                default="1.0.0",
            ),
            revision=_first_value(fields, ("revision", "model_revision")),
            components=components,
            quantization=_first_value(fields, ("quantization",)),
            licenses=_first_value(fields, ("licenses", "license")),
            supported_tasks=_first_value(
                fields,
                ("supported_tasks", "tasks"),
            ),
            offline=_first_value(
                fields,
                ("offline", "local_only"),
                default=True,
            ),
            human_review_required=_first_value(
                fields,
                ("human_review_required", "review_required"),
                default=True,
            ),
            manifest_digest=digest,
        )

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
        *,
        require_manifest_digest: bool = False,
    ) -> "ClinicalSLMArtifactManifest":
        """Build a validated manifest from a duplicate-free JSON object."""

        try:
            decoded = json.loads(payload, object_pairs_hook=_strict_json_object)
        except ClinicalSLMManifestError:
            raise
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError, RecursionError):
            _fail("manifest_unreadable")
        return cls.from_mapping(
            decoded,
            require_manifest_digest=require_manifest_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible manifest mapping."""

        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "revision": self.revision,
            "components": [item.to_dict() for item in self.components],
            "quantization": self.quantization.to_dict(),
            "licenses": [item.to_dict() for item in self.licenses],
            "supported_tasks": list(self.supported_tasks),
            "offline": self.offline,
            "human_review_required": self.human_review_required,
            "manifest_digest": self.manifest_digest,
        }

    def to_json(self) -> str:
        """Return compact, canonical JSON for a package manifest."""

        return _canonical_json(self.to_dict())

    def verify(self, package_root: str | Path) -> "ClinicalSLMVerificationResult":
        """Verify this manifest against a local package root."""

        return verify_clinical_slm_package(package_root, self)

    def __getitem__(self, key: str) -> Any:
        """Allow mapping-style access to the serialized manifest."""

        return self.to_dict()[key]

    def __iter__(self):
        """Iterate over serialized field names in stable order."""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of serialized manifest fields."""

        return len(self.to_dict())


ClinicalSLMManifest = ClinicalSLMArtifactManifest


def _flatten_component_values(values: Iterable[Any]) -> tuple[ClinicalSLMArtifact, ...]:
    flattened: list[Any] = []
    try:
        for value in values:
            if isinstance(value, Mapping):
                flattened.extend(value.items())
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                flattened.extend(value)
            else:
                _fail("invalid_component", "components")
    except (KeyboardInterrupt, SystemExit):
        raise
    except ClinicalSLMManifestError:
        raise
    except BaseException:
        _fail("invalid_component", "components")

    # A mapping-shaped component collection (``{"weights": [...]}``) is
    # flattened into explicit records while preserving the caller's role.
    records: list[ClinicalSLMArtifact] = []
    for value in flattened:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            role, members = value
            if isinstance(members, Mapping):
                members = [members]
            if not isinstance(members, Sequence) or isinstance(
                members, (str, bytes, bytearray)
            ):
                _fail("invalid_component", "components")
            for member in members:
                if isinstance(member, Mapping):
                    records.append(
                        ClinicalSLMArtifact.from_mapping(
                            member,
                            default_component=role,
                        )
                    )
                else:
                    _fail("invalid_component", "components")
        elif isinstance(value, ClinicalSLMArtifact):
            records.append(value)
        elif isinstance(value, Mapping):
            records.append(ClinicalSLMArtifact.from_mapping(value))
        else:
            _fail("invalid_component", "components")
    return tuple(records)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_field")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMVerificationResult:
    """Aggregate-only result of local clinical SLM artifact verification."""

    verified: bool
    component_count: int
    executable_component_count: int
    bytes_checked: int
    manifest_digest: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.verified) is not bool or not self.verified:
            _fail("package_invalid", error_type=ClinicalSLMArtifactError)
        for value in (
            self.component_count,
            self.executable_component_count,
            self.bytes_checked,
        ):
            if type(value) is not int or value < 0:
                _fail("package_invalid", error_type=ClinicalSLMArtifactError)
        if self.executable_component_count > self.component_count:
            _fail("package_invalid", error_type=ClinicalSLMArtifactError)
        object.__setattr__(
            self,
            "manifest_digest",
            _normalise_digest(self.manifest_digest, "manifest_digest"),
        )
        if type(self.reason_codes) is not tuple or any(
            type(code) is not str or _REASON_CODE_RE.fullmatch(code) is None
            for code in self.reason_codes
        ):
            _fail("package_invalid", error_type=ClinicalSLMArtifactError)

    def __repr__(self) -> str:
        """Return a stable, value-free diagnostic representation."""

        return "ClinicalSLMVerificationResult(<aggregate>)"

    @property
    def ok(self) -> bool:
        """Return whether every declared local artifact was verified."""

        return self.verified

    @property
    def files_checked(self) -> int:
        """Return the number of checked files under a compatibility name."""

        return self.component_count

    def to_dict(self) -> dict[str, Any]:
        """Return an aggregate-only, deterministic result."""

        return {
            "verified": self.verified,
            "component_count": self.component_count,
            "executable_component_count": self.executable_component_count,
            "bytes_checked": self.bytes_checked,
            "manifest_digest": self.manifest_digest,
            "reason_codes": list(self.reason_codes),
        }

    def to_json(self) -> str:
        """Return canonical JSON without paths or artifact contents."""

        return _canonical_json(self.to_dict())


ManifestVerificationResult = ClinicalSLMVerificationResult


def validate_clinical_slm_manifest(
    manifest: ClinicalSLMArtifactManifest | Mapping[str, Any],
    *,
    require_manifest_digest: bool = False,
) -> ClinicalSLMArtifactManifest:
    """Validate and return an immutable clinical SLM manifest."""

    if isinstance(manifest, ClinicalSLMArtifactManifest):
        return manifest
    return ClinicalSLMArtifactManifest.from_mapping(
        manifest,
        require_manifest_digest=require_manifest_digest,
    )


def load_clinical_slm_manifest(
    path: str | Path,
    *,
    require_manifest_digest: bool = True,
) -> ClinicalSLMArtifactManifest:
    """Load one local manifest file without performing network access.

    A directory argument resolves only to
    :data:`CLINICAL_SLM_MANIFEST_FILENAME`.  The default requires the
    persisted manifest to carry its explicit self-digest.
    """

    try:
        manifest_path = Path(path)
        if manifest_path.is_symlink():
            _fail("manifest_missing", error_type=ClinicalSLMArtifactMissingError)
        if manifest_path.is_dir():
            manifest_path = manifest_path / CLINICAL_SLM_MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            _fail("manifest_missing", error_type=ClinicalSLMArtifactMissingError)
        payload = manifest_path.read_bytes()
    except ClinicalSLMManifestError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail("manifest_unreadable", error_type=ClinicalSLMArtifactError)
    return ClinicalSLMArtifactManifest.from_json(
        payload,
        require_manifest_digest=require_manifest_digest,
    )


def _validated_package_root(package_root: str | Path) -> Path:
    try:
        root = Path(package_root)
        if root.is_symlink() or not root.is_dir():
            _fail(
                "package_invalid",
                "package_root",
                error_type=ClinicalSLMArtifactError,
            )
        return root.resolve(strict=True)
    except ClinicalSLMManifestError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail(
            "package_invalid",
            "package_root",
            error_type=ClinicalSLMArtifactError,
        )
    raise AssertionError("unreachable")


def _local_artifact_path(root: Path, relative_path: str) -> Path:
    current = root
    parts = PurePosixPath(relative_path).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError:
            _fail(
                "component_missing_on_disk",
                "component",
                error_type=ClinicalSLMArtifactMissingError,
            )
        if stat.S_ISLNK(metadata.st_mode):
            _fail(
                "unsafe_component_path",
                "component",
                error_type=ClinicalSLMArtifactError,
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            _fail(
                "component_unreadable",
                "component",
                error_type=ClinicalSLMArtifactError,
            )
    try:
        metadata = current.lstat()
    except OSError:
        _fail(
            "component_missing_on_disk",
            "component",
            error_type=ClinicalSLMArtifactMissingError,
        )
    if not stat.S_ISREG(metadata.st_mode):
        _fail(
            "component_unreadable",
            "component",
            error_type=ClinicalSLMArtifactError,
        )
    return current


def _hash_local_artifact(path: Path) -> tuple[str, int]:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except (OSError, ValueError):
        _fail(
            "component_unreadable",
            "component",
            error_type=ClinicalSLMArtifactError,
        )
    if before.st_size != after.st_size:
        _fail("component_mutated", "component", error_type=ClinicalSLMArtifactError)
    return f"sha256:{digest.hexdigest()}", after.st_size


def verify_clinical_slm_package(
    package_root: str | Path,
    manifest: ClinicalSLMArtifactManifest | Mapping[str, Any] | None = None,
    *,
    manifest_filename: str = CLINICAL_SLM_MANIFEST_FILENAME,
) -> ClinicalSLMVerificationResult:
    """Verify a complete local package before model construction.

    No download, hub lookup, runtime import, or socket operation is performed.
    The returned result contains only counts and the manifest digest.  Any
    missing, symlinked, resized, changed, or digest-mismatched artifact raises
    a content-free :class:`ClinicalSLMArtifactError` subclass.
    """

    root = _validated_package_root(package_root)
    if manifest is None:
        if type(manifest_filename) is not str or not manifest_filename:
            _fail(
                "package_invalid",
                "package_root",
                error_type=ClinicalSLMArtifactError,
            )
        manifest_path = root / _normalise_path(manifest_filename)
        loaded = load_clinical_slm_manifest(manifest_path)
    else:
        loaded = validate_clinical_slm_manifest(manifest, require_manifest_digest=True)

    total_bytes = 0
    executable_count = 0
    for artifact in loaded.components:
        local_path = _local_artifact_path(root, artifact.path)
        actual_digest, actual_size = _hash_local_artifact(local_path)
        if actual_size != artifact.size_bytes:
            _fail(
                "component_size_mismatch",
                "component",
                error_type=ClinicalSLMArtifactDigestMismatchError,
            )
        if actual_digest != artifact.sha256:
            _fail(
                "component_digest_mismatch",
                "component",
                error_type=ClinicalSLMArtifactDigestMismatchError,
            )
        total_bytes += actual_size
        executable_count += int(artifact.executable)

    return ClinicalSLMVerificationResult(
        verified=True,
        component_count=len(loaded.components),
        executable_component_count=executable_count,
        bytes_checked=total_bytes,
        manifest_digest=loaded.manifest_digest,
    )


def verify_clinical_slm_artifacts(
    package_root: str | Path,
    manifest: ClinicalSLMArtifactManifest | Mapping[str, Any] | None = None,
) -> ClinicalSLMVerificationResult:
    """Compatibility alias for :func:`verify_clinical_slm_package`."""

    return verify_clinical_slm_package(package_root, manifest)


def compute_clinical_slm_manifest_digest(
    manifest: ClinicalSLMArtifactManifest | Mapping[str, Any],
) -> str:
    """Return the canonical digest of a validated clinical SLM manifest."""

    validated = validate_clinical_slm_manifest(manifest)
    return validated.manifest_digest  # type: ignore[return-value]


# Short aliases make the new module convenient without exposing a second
# implementation or weakening the strict validation boundary.
load_manifest = load_clinical_slm_manifest
validate_manifest = validate_clinical_slm_manifest
verify_package = verify_clinical_slm_package


__all__ = [
    "ArtifactComponent",
    "CLINICAL_SLM_MANIFEST_FILENAME",
    "CLINICAL_SLM_MANIFEST_SCHEMA_VERSION",
    "CLINICAL_SLM_MANIFEST_VERSION",
    "ClinicalSLMArtifact",
    "ClinicalSLMArtifactDigestMismatchError",
    "ClinicalSLMArtifactError",
    "ClinicalSLMArtifactManifest",
    "ClinicalSLMArtifactMissingError",
    "ClinicalSLMComponent",
    "ClinicalSLMLicense",
    "ClinicalSLMManifest",
    "ClinicalSLMManifestError",
    "ClinicalSLMQuantization",
    "ClinicalSLMValidationError",
    "ClinicalSLMVerificationResult",
    "ComponentLicense",
    "LicenseDeclaration",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestArtifact",
    "ManifestVerificationResult",
    "QuantizationSpec",
    "SCHEMA_VERSION",
    "compute_clinical_slm_manifest_digest",
    "load_clinical_slm_manifest",
    "load_manifest",
    "validate_clinical_slm_manifest",
    "validate_manifest",
    "verify_clinical_slm_artifacts",
    "verify_clinical_slm_package",
    "verify_package",
]
