"""Strict license and lineage manifests for synthetic training datasets.

Synthetic dataset releases need a reviewable, machine-checkable record of where
every byte came from: which generator produced it, which model was involved,
which upstream corpora contributed, and whether the result may be
redistributed. This module validates one such manifest against committed,
closed policy: a permissive-only SPDX short-identifier allowlist, a committed
restricted-identifier list, bounded dependency counts, and digest patterns.

Incomplete, unknown, malformed, restricted, or non-redistributable lineage
always fails closed: every finding blocks the manifest. The module does not
read license files, does not consult the network, does not infer a license
from input data, does not emit legal advice, and never copies raw manifest
text into its reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

from openmed.core.audit import stable_hash
from openmed.training.synthetic.spdx_identifier import (
    PERMISSIVE_SPDX_IDENTIFIERS,
    SpdxIdentifierStatus,
    normalize_spdx_identifier,
)

__all__ = [
    "DEFAULT_LICENSE_LINEAGE_POLICY",
    "DEFAULT_MAX_DEPENDENCIES",
    "LICENSE_LINEAGE_FIELDS",
    "LICENSE_LINEAGE_SCHEMA_VERSION",
    "LICENSE_LINEAGE_SOURCE",
    "MAX_DEPENDENCIES",
    "MAX_LICENSE_EXPRESSION_LENGTH",
    "RESTRICTED_LICENSE_IDENTIFIERS",
    "LicenseExpressionAssessment",
    "LicenseExpressionStatus",
    "LicenseLineageFinding",
    "LicenseLineageReasonCode",
    "LicenseLineageReport",
    "LicenseLineageSourceClass",
    "LicenseLineageVerdict",
    "RedistributionDecision",
    "SyntheticDatasetDependency",
    "SyntheticDatasetLicenseManifest",
    "SyntheticGeneratorProvenance",
    "SyntheticLicenseLineageError",
    "SyntheticLicenseLineagePolicy",
    "SyntheticModelProvenance",
    "assess_license_expression",
    "assert_redistribution_allowed",
    "load_license_lineage_manifest",
    "manifest_digest",
    "validate_license_lineage",
]

LICENSE_LINEAGE_SCHEMA_VERSION: Final = "openmed.training.synthetic_license_lineage.v1"
LICENSE_LINEAGE_SOURCE: Final = "openmed.synthetic.license_lineage"

DEFAULT_MAX_DEPENDENCIES: Final = 64
MAX_DEPENDENCIES: Final = 256
MAX_LICENSE_EXPRESSION_LENGTH: Final = 128

# Committed, restricted-only list. These identifiers are recognized so that a
# manifest can name them and be blocked deliberately; the list is a reviewed
# policy decision and is never extended by inference from input data.
RESTRICTED_LICENSE_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    {
        "AGPL-3.0-only",
        "AGPL-3.0-or-later",
        "BUSL-1.1",
        "CC-BY-NC-4.0",
        "CC-BY-NC-ND-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-ND-4.0",
        "Elastic-2.0",
        "GPL-2.0-only",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "SSPL-1.0",
    }
)

# Canonical order of the manifest sections that can carry findings. Reports
# sort findings by this order so two runs over the same manifest agree.
LICENSE_LINEAGE_FIELDS: Final[tuple[str, ...]] = (
    "source_class",
    "license_expression",
    "generator",
    "model_provenance",
    "dependencies",
    "redistribution",
)

_FIELD_ORDER: Final[dict[str, int]] = {
    name: index for index, name in enumerate(LICENSE_LINEAGE_FIELDS)
}

_IDENTIFIER_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EXPRESSION_TOKENS: Final = frozenset({"AND", "OR", "WITH"})
_LICENSEREF_PREFIX: Final = "LicenseRef-"

_RESTRICTED_BY_FOLD: Final[dict[str, str]] = {
    identifier.casefold(): identifier
    for identifier in sorted(RESTRICTED_LICENSE_IDENTIFIERS)
}

_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dataset_id",
        "dependencies",
        "generator",
        "license_expression",
        "model_provenance",
        "redistribution",
        "schema_version",
        "source_class",
    }
)
_MANIFEST_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dataset_id",
        "generator",
        "license_expression",
        "redistribution",
        "source_class",
    }
)
_GENERATOR_KEYS: Final[frozenset[str]] = frozenset(
    {"digest", "generator_id", "version"}
)
_MODEL_KEYS: Final[frozenset[str]] = frozenset(
    {"digest", "license_expression", "model_id"}
)
_DEPENDENCY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dependency_id",
        "digest",
        "license_expression",
        "redistribution",
        "source_class",
    }
)


class SyntheticLicenseLineageError(ValueError):
    """Raised when a manifest is structurally invalid or lineage is blocked."""


class LicenseExpressionStatus(str, Enum):
    """Closed set of licensing outcomes for one candidate expression."""

    CANONICAL = "canonical"
    NORMALIZED = "normalized"
    DEPRECATED_ALIAS = "deprecated_alias"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"
    MALFORMED = "malformed"


class LicenseLineageSourceClass(str, Enum):
    """Closed set of provenance classes a synthetic dataset may declare."""

    SYNTHETIC_GENERATED = "synthetic_generated"
    SYNTHETIC_AUGMENTED = "synthetic_augmented"
    MODEL_GENERATED = "model_generated"
    PUBLIC_DOMAIN_CORPUS = "public_domain_corpus"
    LICENSED_CORPUS = "licensed_corpus"
    MANUAL_CURATION = "manual_curation"


class RedistributionDecision(str, Enum):
    """Closed set of redistribution decisions recorded for a dataset."""

    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class LicenseLineageVerdict(str, Enum):
    """Closed set of lineage outcomes; only ``cleared`` permits release."""

    CLEARED = "cleared"
    BLOCKED = "blocked"


class LicenseLineageReasonCode(str, Enum):
    """Closed set of reasons; every reason blocks redistribution."""

    SOURCE_CLASS_UNKNOWN = "source_class_unknown"
    LICENSE_EXPRESSION_MISSING = "license_expression_missing"
    LICENSE_EXPRESSION_MALFORMED = "license_expression_malformed"
    LICENSE_EXPRESSION_RESTRICTED = "license_expression_restricted"
    LICENSE_EXPRESSION_UNKNOWN = "license_expression_unknown"
    GENERATOR_DIGEST_MISSING = "generator_digest_missing"
    GENERATOR_DIGEST_MALFORMED = "generator_digest_malformed"
    MODEL_PROVENANCE_MISSING = "model_provenance_missing"
    MODEL_DIGEST_MALFORMED = "model_digest_malformed"
    DEPENDENCY_LINEAGE_MISSING = "dependency_lineage_missing"
    DEPENDENCY_DIGEST_MISSING = "dependency_digest_missing"
    DEPENDENCY_DIGEST_MALFORMED = "dependency_digest_malformed"
    DEPENDENCY_LIMIT_EXCEEDED = "dependency_limit_exceeded"
    REDISTRIBUTION_UNKNOWN = "redistribution_unknown"
    REDISTRIBUTION_RESTRICTED = "redistribution_restricted"
    REDISTRIBUTION_PROHIBITED = "redistribution_prohibited"


_VERDICT_BY_REASON: Final[dict[LicenseLineageReasonCode, LicenseLineageVerdict]] = {
    reason: LicenseLineageVerdict.BLOCKED for reason in LicenseLineageReasonCode
}

_REDISTRIBUTION_REASON: Final[
    dict[RedistributionDecision, LicenseLineageReasonCode]
] = {
    RedistributionDecision.RESTRICTED: LicenseLineageReasonCode.REDISTRIBUTION_RESTRICTED,
    RedistributionDecision.PROHIBITED: LicenseLineageReasonCode.REDISTRIBUTION_PROHIBITED,
    RedistributionDecision.UNKNOWN: LicenseLineageReasonCode.REDISTRIBUTION_UNKNOWN,
}


def _check_identifier(value: Any, *, context: str) -> str:
    if type(value) is not str or not _IDENTIFIER_RE.match(value):
        raise SyntheticLicenseLineageError(f"{context} must be a lowercase identifier")
    return value


def _check_digest_or_none(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise SyntheticLicenseLineageError(f"{context} must be a string or None")
    return value


def _coerce_enum(
    enum_type: type[Enum],
    value: Any,
    *,
    context: str,
) -> Any:
    if value is None or isinstance(value, enum_type):
        return value
    if type(value) is str:
        try:
            return enum_type(value)
        except ValueError:
            return None
    raise SyntheticLicenseLineageError(
        f"{context} must be a string or {enum_type.__name__}"
    )


def _require_mapping(payload: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SyntheticLicenseLineageError(f"{context} must be a mapping")
    return payload


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SyntheticLicenseLineageError(
            f"{context} has unsupported keys: {', '.join(unknown)}"
        )


def _require_keys(
    payload: Mapping[str, Any],
    required: frozenset[str],
    *,
    context: str,
) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise SyntheticLicenseLineageError(
            f"{context} is missing required keys: {', '.join(missing)}"
        )


@dataclass(frozen=True, slots=True)
class LicenseExpressionAssessment:
    """One stable licensing outcome for a candidate license expression."""

    schema_version: str
    status: LicenseExpressionStatus
    normalized: str | None
    is_permissive: bool
    is_restricted: bool
    category: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "normalized": self.normalized,
            "is_permissive": self.is_permissive,
            "is_restricted": self.is_restricted,
            "category": self.category,
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )


def _assessment(
    status: LicenseExpressionStatus,
    normalized: str | None,
    *,
    is_permissive: bool,
    is_restricted: bool,
    category: str,
) -> LicenseExpressionAssessment:
    return LicenseExpressionAssessment(
        schema_version=LICENSE_LINEAGE_SCHEMA_VERSION,
        status=status,
        normalized=normalized,
        is_permissive=is_permissive,
        is_restricted=is_restricted,
        category=category,
    )


def assess_license_expression(value: Any) -> LicenseExpressionAssessment:
    """Classify one candidate license expression deterministically.

    Only committed allowlist members, their case and whitespace variants, the
    committed deprecated-alias table, and the committed restricted list
    produce a normalized value. Syntactically valid but unlisted identifiers
    stay unknown, ``LicenseRef-`` references stay unknown, values that are not
    single SPDX short identifiers are malformed, and restricted identifiers
    are flagged as restricted. No network lookup or legal judgement is
    performed, and the raw input is never echoed back.
    """
    if type(value) is not str:
        return _assessment(
            LicenseExpressionStatus.MALFORMED,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_malformed_type",
        )
    trimmed = value.strip()
    if not trimmed:
        return _assessment(
            LicenseExpressionStatus.MALFORMED,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_malformed_empty",
        )
    if len(trimmed) > MAX_LICENSE_EXPRESSION_LENGTH:
        return _assessment(
            LicenseExpressionStatus.MALFORMED,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_malformed_length",
        )
    if any(
        token in _EXPRESSION_TOKENS for token in trimmed.split()
    ) or trimmed.endswith("+"):
        return _assessment(
            LicenseExpressionStatus.MALFORMED,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_malformed_expression",
        )

    restricted = _RESTRICTED_BY_FOLD.get(trimmed.casefold())
    if restricted is not None:
        return _assessment(
            LicenseExpressionStatus.RESTRICTED,
            restricted,
            is_permissive=False,
            is_restricted=True,
            category="license_expression_restricted",
        )

    identifier = normalize_spdx_identifier(trimmed)
    if identifier.status is SpdxIdentifierStatus.MALFORMED:
        return _assessment(
            LicenseExpressionStatus.MALFORMED,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_malformed",
        )
    if identifier.status is SpdxIdentifierStatus.UNKNOWN:
        return _assessment(
            LicenseExpressionStatus.UNKNOWN,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_unknown",
        )
    normalized = identifier.normalized
    if normalized is None or normalized.startswith(_LICENSEREF_PREFIX):
        return _assessment(
            LicenseExpressionStatus.UNKNOWN,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_licenseref",
        )
    if not identifier.is_permissive:
        return _assessment(
            LicenseExpressionStatus.UNKNOWN,
            None,
            is_permissive=False,
            is_restricted=False,
            category="license_expression_unknown",
        )
    if identifier.status is SpdxIdentifierStatus.CANONICAL:
        status = LicenseExpressionStatus.CANONICAL
    elif identifier.status is SpdxIdentifierStatus.DEPRECATED_ALIAS:
        status = LicenseExpressionStatus.DEPRECATED_ALIAS
    else:
        status = LicenseExpressionStatus.NORMALIZED
    return _assessment(
        status,
        normalized,
        is_permissive=True,
        is_restricted=False,
        category=f"license_expression_{status.value}",
    )


@dataclass(frozen=True, slots=True)
class SyntheticGeneratorProvenance:
    """Generator identity and content digest recorded for one dataset."""

    generator_id: str
    digest: str | None
    version: str | None = None

    def __post_init__(self) -> None:
        _check_identifier(self.generator_id, context="generator_id")
        _check_digest_or_none(self.digest, context="generator digest")
        if self.version is not None and type(self.version) is not str:
            raise SyntheticLicenseLineageError("generator version must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "generator_id": self.generator_id,
            "digest": self.digest,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticGeneratorProvenance:
        """Build provenance from a strict mapping, rejecting unknown keys."""
        payload = _require_mapping(payload, context="generator")
        _reject_unknown_keys(payload, _GENERATOR_KEYS, context="generator")
        _require_keys(payload, frozenset({"generator_id"}), context="generator")
        return cls(
            generator_id=payload["generator_id"],
            digest=payload.get("digest"),
            version=payload.get("version"),
        )


@dataclass(frozen=True, slots=True)
class SyntheticModelProvenance:
    """Optional model provenance recorded for a generated dataset."""

    model_id: str
    license_expression: str | None
    digest: str | None = None

    def __post_init__(self) -> None:
        _check_identifier(self.model_id, context="model_id")
        if self.license_expression is not None and (
            type(self.license_expression) is not str
        ):
            raise SyntheticLicenseLineageError(
                "model license_expression must be a string or None"
            )
        _check_digest_or_none(self.digest, context="model digest")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "model_id": self.model_id,
            "license_expression": self.license_expression,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticModelProvenance:
        """Build model provenance from a strict mapping."""
        payload = _require_mapping(payload, context="model_provenance")
        _reject_unknown_keys(payload, _MODEL_KEYS, context="model_provenance")
        _require_keys(payload, frozenset({"model_id"}), context="model_provenance")
        return cls(
            model_id=payload["model_id"],
            license_expression=payload.get("license_expression"),
            digest=payload.get("digest"),
        )


@dataclass(frozen=True, slots=True)
class SyntheticDatasetDependency:
    """One upstream dependency; absent lineage is represented explicitly."""

    dependency_id: str
    source_class: LicenseLineageSourceClass | None = None
    license_expression: str | None = None
    redistribution: RedistributionDecision | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        _check_identifier(self.dependency_id, context="dependency_id")
        if self.source_class is not None and not isinstance(
            self.source_class, LicenseLineageSourceClass
        ):
            raise SyntheticLicenseLineageError(
                "dependency source_class must be a LicenseLineageSourceClass or None"
            )
        if self.license_expression is not None and (
            type(self.license_expression) is not str
        ):
            raise SyntheticLicenseLineageError(
                "dependency license_expression must be a string or None"
            )
        if self.redistribution is not None and not isinstance(
            self.redistribution, RedistributionDecision
        ):
            raise SyntheticLicenseLineageError(
                "dependency redistribution must be a RedistributionDecision or None"
            )
        _check_digest_or_none(self.digest, context="dependency digest")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "dependency_id": self.dependency_id,
            "source_class": (
                self.source_class.value if self.source_class is not None else None
            ),
            "license_expression": self.license_expression,
            "redistribution": (
                self.redistribution.value if self.redistribution is not None else None
            ),
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticDatasetDependency:
        """Build one dependency from a mapping; gaps stay ``None`` on purpose."""
        payload = _require_mapping(payload, context="dependency")
        _reject_unknown_keys(payload, _DEPENDENCY_KEYS, context="dependency")
        _require_keys(payload, frozenset({"dependency_id"}), context="dependency")
        return cls(
            dependency_id=payload["dependency_id"],
            source_class=_coerce_enum(
                LicenseLineageSourceClass,
                payload.get("source_class"),
                context="dependency source_class",
            ),
            license_expression=payload.get("license_expression"),
            redistribution=_coerce_enum(
                RedistributionDecision,
                payload.get("redistribution"),
                context="dependency redistribution",
            ),
            digest=payload.get("digest"),
        )


@dataclass(frozen=True, slots=True)
class SyntheticDatasetLicenseManifest:
    """Strict license and lineage manifest for one synthetic dataset release.

    Absent sections are represented as ``None`` rather than omitted so that
    validation can report them and fail closed instead of raising.
    """

    dataset_id: str
    source_class: LicenseLineageSourceClass | None = None
    license_expression: str | None = None
    generator: SyntheticGeneratorProvenance | None = None
    redistribution: RedistributionDecision | None = None
    dependencies: tuple[SyntheticDatasetDependency, ...] = ()
    model_provenance: SyntheticModelProvenance | None = None
    schema_version: str = LICENSE_LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_identifier(self.dataset_id, context="dataset_id")
        if self.schema_version != LICENSE_LINEAGE_SCHEMA_VERSION:
            raise SyntheticLicenseLineageError(
                "manifest schema_version must match LICENSE_LINEAGE_SCHEMA_VERSION"
            )
        if self.source_class is not None and not isinstance(
            self.source_class, LicenseLineageSourceClass
        ):
            raise SyntheticLicenseLineageError(
                "source_class must be a LicenseLineageSourceClass or None"
            )
        if self.license_expression is not None and (
            type(self.license_expression) is not str
        ):
            raise SyntheticLicenseLineageError(
                "license_expression must be a string or None"
            )
        if self.generator is not None and not isinstance(
            self.generator, SyntheticGeneratorProvenance
        ):
            raise SyntheticLicenseLineageError(
                "generator must be a SyntheticGeneratorProvenance or None"
            )
        if self.redistribution is not None and not isinstance(
            self.redistribution, RedistributionDecision
        ):
            raise SyntheticLicenseLineageError(
                "redistribution must be a RedistributionDecision or None"
            )
        if self.model_provenance is not None and not isinstance(
            self.model_provenance, SyntheticModelProvenance
        ):
            raise SyntheticLicenseLineageError(
                "model_provenance must be a SyntheticModelProvenance or None"
            )
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(item, SyntheticDatasetDependency)
            for item in self.dependencies
        ):
            raise SyntheticLicenseLineageError(
                "dependencies must be a tuple of SyntheticDatasetDependency"
            )
        if len(self.dependencies) > MAX_DEPENDENCIES:
            raise SyntheticLicenseLineageError(
                "dependencies must not exceed MAX_DEPENDENCIES"
            )
        ordered = tuple(sorted(self.dependencies, key=lambda item: item.dependency_id))
        if ordered != self.dependencies:
            object.__setattr__(self, "dependencies", ordered)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, dependency-ordered manifest dictionary."""
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "source_class": (
                self.source_class.value if self.source_class is not None else None
            ),
            "license_expression": self.license_expression,
            "generator": self.generator.to_dict()
            if self.generator is not None
            else None,
            "model_provenance": (
                self.model_provenance.to_dict()
                if self.model_provenance is not None
                else None
            ),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "redistribution": (
                self.redistribution.value if self.redistribution is not None else None
            ),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize deterministically; compact unless ``indent`` is given."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    def write_json(self, path: Path, *, indent: int = 2) -> None:
        """Write the manifest as UTF-8 JSON with a trailing newline."""
        path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticDatasetLicenseManifest:
        """Build a manifest from a strict mapping, rejecting unknown keys."""
        payload = _require_mapping(payload, context="manifest")
        _reject_unknown_keys(payload, _MANIFEST_KEYS, context="manifest")
        _require_keys(payload, _MANIFEST_REQUIRED_KEYS, context="manifest")
        schema_version = payload.get("schema_version", LICENSE_LINEAGE_SCHEMA_VERSION)
        if schema_version != LICENSE_LINEAGE_SCHEMA_VERSION:
            raise SyntheticLicenseLineageError(
                "manifest schema_version must match LICENSE_LINEAGE_SCHEMA_VERSION"
            )
        generator_payload = payload["generator"]
        model_payload = payload.get("model_provenance")
        dependencies_payload = payload.get("dependencies", ())
        if not isinstance(dependencies_payload, (list, tuple)):
            raise SyntheticLicenseLineageError("dependencies must be a list")
        return cls(
            dataset_id=payload["dataset_id"],
            source_class=_coerce_enum(
                LicenseLineageSourceClass,
                payload["source_class"],
                context="source_class",
            ),
            license_expression=payload["license_expression"],
            generator=(
                SyntheticGeneratorProvenance.from_dict(generator_payload)
                if generator_payload is not None
                else None
            ),
            redistribution=_coerce_enum(
                RedistributionDecision,
                payload["redistribution"],
                context="redistribution",
            ),
            dependencies=tuple(
                SyntheticDatasetDependency.from_dict(item)
                for item in dependencies_payload
            ),
            model_provenance=(
                SyntheticModelProvenance.from_dict(model_payload)
                if model_payload is not None
                else None
            ),
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> SyntheticDatasetLicenseManifest:
        """Parse a JSON document into a manifest, rejecting unknown keys."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        if type(text) is not str:
            raise SyntheticLicenseLineageError("manifest JSON must be text")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise SyntheticLicenseLineageError("manifest JSON is not valid") from error
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class SyntheticLicenseLineagePolicy:
    """Committed review policy applied while validating one manifest."""

    allow_unknown_licenses: bool = False
    max_dependencies: int = DEFAULT_MAX_DEPENDENCIES
    require_dependency_digests: bool = False
    require_model_provenance: bool = False

    def __post_init__(self) -> None:
        for name in (
            "allow_unknown_licenses",
            "require_dependency_digests",
            "require_model_provenance",
        ):
            if type(getattr(self, name)) is not bool:
                raise SyntheticLicenseLineageError(f"{name} must be a bool")
        if type(self.max_dependencies) is not int:
            raise SyntheticLicenseLineageError("max_dependencies must be an int")
        if not 1 <= self.max_dependencies <= MAX_DEPENDENCIES:
            raise SyntheticLicenseLineageError(
                "max_dependencies must be between 1 and MAX_DEPENDENCIES"
            )


DEFAULT_LICENSE_LINEAGE_POLICY: Final = SyntheticLicenseLineagePolicy()


@dataclass(frozen=True, slots=True)
class LicenseLineageFinding:
    """One reason a manifest cannot be released, located by field and name."""

    field: str
    reason: LicenseLineageReasonCode
    dependency_id: str | None = None

    def __post_init__(self) -> None:
        if self.field not in _FIELD_ORDER:
            raise SyntheticLicenseLineageError("field must be a known manifest field")
        if not isinstance(self.reason, LicenseLineageReasonCode):
            raise SyntheticLicenseLineageError(
                "reason must be a LicenseLineageReasonCode"
            )
        if self.dependency_id is not None:
            _check_identifier(self.dependency_id, context="dependency_id")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "field": self.field,
            "reason": self.reason.value,
            "dependency_id": self.dependency_id,
        }

    @property
    def sort_key(self) -> tuple[int, str, str]:
        """Return the deterministic ordering key used by reports."""
        return (
            _FIELD_ORDER[self.field],
            self.dependency_id or "",
            self.reason.value,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LicenseLineageFinding:
        """Build one finding from a strict mapping."""
        payload = _require_mapping(payload, context="finding")
        _reject_unknown_keys(
            payload,
            frozenset({"field", "reason", "dependency_id"}),
            context="finding",
        )
        _require_keys(
            payload,
            frozenset({"field", "reason"}),
            context="finding",
        )
        reason = payload["reason"]
        if type(reason) is str:
            try:
                reason = LicenseLineageReasonCode(reason)
            except ValueError as error:
                raise SyntheticLicenseLineageError(
                    "finding reason is not a known reason code"
                ) from error
        return cls(
            field=payload["field"],
            reason=reason,
            dependency_id=payload.get("dependency_id"),
        )


@dataclass(frozen=True, slots=True)
class LicenseLineageReport:
    """Deterministic validation outcome for one synthetic dataset manifest."""

    verdict: LicenseLineageVerdict
    findings: tuple[LicenseLineageFinding, ...]
    dataset_id: str
    dependency_count: int
    license_expressions: tuple[str, ...] = ()
    schema_version: str = LICENSE_LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LICENSE_LINEAGE_SCHEMA_VERSION:
            raise SyntheticLicenseLineageError(
                "report schema_version must match LICENSE_LINEAGE_SCHEMA_VERSION"
            )
        _check_identifier(self.dataset_id, context="dataset_id")
        if not isinstance(self.verdict, LicenseLineageVerdict):
            raise SyntheticLicenseLineageError(
                "verdict must be a LicenseLineageVerdict"
            )
        if not isinstance(self.findings, tuple) or any(
            not isinstance(item, LicenseLineageFinding) for item in self.findings
        ):
            raise SyntheticLicenseLineageError(
                "findings must be a tuple of LicenseLineageFinding"
            )
        if (
            tuple(sorted(self.findings, key=lambda item: item.sort_key))
            != self.findings
        ):
            raise SyntheticLicenseLineageError(
                "findings must be ordered by field, dependency, and reason"
            )
        if len(set(self.findings)) != len(self.findings):
            raise SyntheticLicenseLineageError("findings must be unique")
        if type(self.dependency_count) is not int or self.dependency_count < 0:
            raise SyntheticLicenseLineageError(
                "dependency_count must be a non-negative int"
            )
        if not isinstance(self.license_expressions, tuple) or any(
            type(item) is not str for item in self.license_expressions
        ):
            raise SyntheticLicenseLineageError(
                "license_expressions must be a tuple of strings"
            )
        if tuple(sorted(set(self.license_expressions))) != self.license_expressions:
            raise SyntheticLicenseLineageError(
                "license_expressions must be sorted and unique"
            )
        expected = (
            LicenseLineageVerdict.BLOCKED
            if self.findings
            else LicenseLineageVerdict.CLEARED
        )
        if self.verdict is not expected:
            raise SyntheticLicenseLineageError(
                "verdict must be blocked when findings are present"
            )

    @property
    def ok(self) -> bool:
        """Return True only when the manifest cleared every committed check."""
        return self.verdict is LicenseLineageVerdict.CLEARED

    @property
    def blocked(self) -> bool:
        """Return True when redistribution must not proceed."""
        return self.verdict is LicenseLineageVerdict.BLOCKED

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return sorted unique reason codes for the recorded findings."""
        return tuple(sorted({item.reason.value for item in self.findings}))

    @property
    def blocked_fields(self) -> tuple[str, ...]:
        """Return the manifest fields with at least one finding, in order."""
        return tuple(
            sorted(
                {item.field for item in self.findings},
                key=lambda name: _FIELD_ORDER[name],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "verdict": self.verdict.value,
            "blocked": self.blocked,
            "dependency_count": self.dependency_count,
            "license_expressions": list(self.license_expressions),
            "reason_codes": list(self.reason_codes),
            "blocked_fields": list(self.blocked_fields),
            "findings": [item.to_dict() for item in self.findings],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize deterministically; compact unless ``indent`` is given."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    def write_json(self, path: Path, *, indent: int = 2) -> None:
        """Write the report as UTF-8 JSON with a trailing newline."""
        path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LicenseLineageReport:
        """Build a report from a strict mapping, rejecting unknown keys."""
        payload = _require_mapping(payload, context="report")
        _reject_unknown_keys(
            payload,
            frozenset(
                {
                    "schema_version",
                    "dataset_id",
                    "verdict",
                    "blocked",
                    "dependency_count",
                    "license_expressions",
                    "reason_codes",
                    "blocked_fields",
                    "findings",
                }
            ),
            context="report",
        )
        _require_keys(
            payload,
            frozenset({"dataset_id", "verdict", "findings", "dependency_count"}),
            context="report",
        )
        verdict = payload["verdict"]
        if type(verdict) is str:
            try:
                verdict = LicenseLineageVerdict(verdict)
            except ValueError as error:
                raise SyntheticLicenseLineageError(
                    "report verdict is not a known verdict"
                ) from error
        findings_payload = payload["findings"]
        if not isinstance(findings_payload, (list, tuple)):
            raise SyntheticLicenseLineageError("findings must be a list")
        expressions_payload = payload.get("license_expressions", ())
        if not isinstance(expressions_payload, (list, tuple)):
            raise SyntheticLicenseLineageError("license_expressions must be a list")
        return cls(
            verdict=verdict,
            findings=tuple(
                LicenseLineageFinding.from_dict(item) for item in findings_payload
            ),
            dataset_id=payload["dataset_id"],
            dependency_count=payload["dependency_count"],
            license_expressions=tuple(expressions_payload),
            schema_version=payload.get(
                "schema_version", LICENSE_LINEAGE_SCHEMA_VERSION
            ),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> LicenseLineageReport:
        """Parse a JSON document into a report, rejecting unknown keys."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        if type(text) is not str:
            raise SyntheticLicenseLineageError("report JSON must be text")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise SyntheticLicenseLineageError("report JSON is not valid") from error
        return cls.from_dict(payload)


def _license_findings(
    value: Any,
    *,
    field: str,
    dependency_id: str | None,
    allow_unknown: bool,
    missing_reason: LicenseLineageReasonCode,
) -> tuple[list[LicenseLineageFinding], str | None]:
    if value is None or (type(value) is str and not value.strip()):
        return (
            [
                LicenseLineageFinding(
                    field=field,
                    reason=missing_reason,
                    dependency_id=dependency_id,
                )
            ],
            None,
        )
    assessment = assess_license_expression(value)
    if assessment.is_permissive:
        return [], assessment.normalized
    if assessment.status is LicenseExpressionStatus.MALFORMED:
        reason = LicenseLineageReasonCode.LICENSE_EXPRESSION_MALFORMED
    elif assessment.status is LicenseExpressionStatus.RESTRICTED:
        reason = LicenseLineageReasonCode.LICENSE_EXPRESSION_RESTRICTED
    elif allow_unknown:
        return [], None
    else:
        reason = LicenseLineageReasonCode.LICENSE_EXPRESSION_UNKNOWN
    return (
        [
            LicenseLineageFinding(
                field=field,
                reason=reason,
                dependency_id=dependency_id,
            )
        ],
        None,
    )


def _digest_findings(
    digest: Any,
    *,
    field: str,
    dependency_id: str | None,
    missing_reason: LicenseLineageReasonCode,
    malformed_reason: LicenseLineageReasonCode,
    required: bool,
) -> list[LicenseLineageFinding]:
    if digest is None:
        if not required:
            return []
        return [
            LicenseLineageFinding(
                field=field,
                reason=missing_reason,
                dependency_id=dependency_id,
            )
        ]
    if not _DIGEST_RE.match(digest):
        return [
            LicenseLineageFinding(
                field=field,
                reason=malformed_reason,
                dependency_id=dependency_id,
            )
        ]
    return []


def _redistribution_findings(
    decision: RedistributionDecision | None,
    *,
    dependency_id: str | None = None,
) -> list[LicenseLineageFinding]:
    if decision is RedistributionDecision.ALLOWED:
        return []
    reason = _REDISTRIBUTION_REASON.get(
        decision if decision is not None else RedistributionDecision.UNKNOWN,
        LicenseLineageReasonCode.REDISTRIBUTION_UNKNOWN,
    )
    return [
        LicenseLineageFinding(
            field="redistribution",
            reason=reason,
            dependency_id=dependency_id,
        )
    ]


def _coerce_manifest(payload: Any) -> SyntheticDatasetLicenseManifest:
    if isinstance(payload, SyntheticDatasetLicenseManifest):
        return payload
    if isinstance(payload, (str, bytes)):
        return SyntheticDatasetLicenseManifest.from_json(payload)
    if isinstance(payload, Mapping):
        return SyntheticDatasetLicenseManifest.from_dict(payload)
    raise SyntheticLicenseLineageError(
        "manifest must be a SyntheticDatasetLicenseManifest, mapping, or JSON text"
    )


def validate_license_lineage(
    manifest: Any,
    *,
    policy: SyntheticLicenseLineagePolicy | None = None,
) -> LicenseLineageReport:
    """Validate one synthetic dataset manifest against committed policy.

    The manifest may be a :class:`SyntheticDatasetLicenseManifest`, a mapping,
    or JSON text. Every gap, unknown identifier, restricted identifier,
    malformed digest, over-limit dependency list, or non-allowed
    redistribution decision produces a finding that blocks the release. The
    returned report contains only closed reason codes, canonical permissive
    expressions, and identifiers supplied by the caller.
    """
    resolved = _coerce_manifest(manifest)
    active_policy = policy if policy is not None else DEFAULT_LICENSE_LINEAGE_POLICY
    if not isinstance(active_policy, SyntheticLicenseLineagePolicy):
        raise SyntheticLicenseLineageError(
            "policy must be a SyntheticLicenseLineagePolicy"
        )

    findings: list[LicenseLineageFinding] = []
    expressions: set[str] = set()

    if resolved.source_class is None:
        findings.append(
            LicenseLineageFinding(
                field="source_class",
                reason=LicenseLineageReasonCode.SOURCE_CLASS_UNKNOWN,
            )
        )

    license_findings, normalized = _license_findings(
        resolved.license_expression,
        field="license_expression",
        dependency_id=None,
        allow_unknown=active_policy.allow_unknown_licenses,
        missing_reason=LicenseLineageReasonCode.LICENSE_EXPRESSION_MISSING,
    )
    findings.extend(license_findings)
    if normalized is not None:
        expressions.add(normalized)

    if resolved.generator is None:
        findings.append(
            LicenseLineageFinding(
                field="generator",
                reason=LicenseLineageReasonCode.GENERATOR_DIGEST_MISSING,
            )
        )
    else:
        findings.extend(
            _digest_findings(
                resolved.generator.digest,
                field="generator",
                dependency_id=None,
                missing_reason=LicenseLineageReasonCode.GENERATOR_DIGEST_MISSING,
                malformed_reason=LicenseLineageReasonCode.GENERATOR_DIGEST_MALFORMED,
                required=True,
            )
        )

    if resolved.model_provenance is None:
        if active_policy.require_model_provenance:
            findings.append(
                LicenseLineageFinding(
                    field="model_provenance",
                    reason=LicenseLineageReasonCode.MODEL_PROVENANCE_MISSING,
                )
            )
    else:
        model_findings, model_normalized = _license_findings(
            resolved.model_provenance.license_expression,
            field="model_provenance",
            dependency_id=None,
            allow_unknown=active_policy.allow_unknown_licenses,
            missing_reason=LicenseLineageReasonCode.LICENSE_EXPRESSION_MISSING,
        )
        findings.extend(model_findings)
        if model_normalized is not None:
            expressions.add(model_normalized)
        findings.extend(
            _digest_findings(
                resolved.model_provenance.digest,
                field="model_provenance",
                dependency_id=None,
                missing_reason=LicenseLineageReasonCode.GENERATOR_DIGEST_MISSING,
                malformed_reason=LicenseLineageReasonCode.MODEL_DIGEST_MALFORMED,
                required=False,
            )
        )

    if len(resolved.dependencies) > active_policy.max_dependencies:
        findings.append(
            LicenseLineageFinding(
                field="dependencies",
                reason=LicenseLineageReasonCode.DEPENDENCY_LIMIT_EXCEEDED,
            )
        )

    for dependency in resolved.dependencies:
        if (
            dependency.source_class is None
            or dependency.license_expression is None
            or dependency.redistribution is None
        ):
            findings.append(
                LicenseLineageFinding(
                    field="dependencies",
                    reason=LicenseLineageReasonCode.DEPENDENCY_LINEAGE_MISSING,
                    dependency_id=dependency.dependency_id,
                )
            )
        if dependency.source_class is None:
            findings.append(
                LicenseLineageFinding(
                    field="dependencies",
                    reason=LicenseLineageReasonCode.SOURCE_CLASS_UNKNOWN,
                    dependency_id=dependency.dependency_id,
                )
            )
        dependency_findings, dependency_normalized = _license_findings(
            dependency.license_expression,
            field="dependencies",
            dependency_id=dependency.dependency_id,
            allow_unknown=active_policy.allow_unknown_licenses,
            missing_reason=LicenseLineageReasonCode.DEPENDENCY_LINEAGE_MISSING,
        )
        findings.extend(dependency_findings)
        if dependency_normalized is not None:
            expressions.add(dependency_normalized)
        findings.extend(
            _digest_findings(
                dependency.digest,
                field="dependencies",
                dependency_id=dependency.dependency_id,
                missing_reason=LicenseLineageReasonCode.DEPENDENCY_DIGEST_MISSING,
                malformed_reason=LicenseLineageReasonCode.DEPENDENCY_DIGEST_MALFORMED,
                required=active_policy.require_dependency_digests,
            )
        )
        findings.extend(
            _redistribution_findings(
                dependency.redistribution,
                dependency_id=dependency.dependency_id,
            )
        )

    findings.extend(_redistribution_findings(resolved.redistribution))

    ordered = tuple(sorted(set(findings), key=lambda item: item.sort_key))
    verdict = (
        LicenseLineageVerdict.BLOCKED if ordered else LicenseLineageVerdict.CLEARED
    )
    return LicenseLineageReport(
        verdict=verdict,
        findings=ordered,
        dataset_id=resolved.dataset_id,
        dependency_count=len(resolved.dependencies),
        license_expressions=tuple(sorted(expressions)),
    )


def assert_redistribution_allowed(report: LicenseLineageReport) -> None:
    """Raise when a lineage report blocks redistribution.

    The error message contains only counts and closed reason codes, never
    dataset identifiers or raw manifest text.
    """
    if not isinstance(report, LicenseLineageReport):
        raise SyntheticLicenseLineageError("report must be a LicenseLineageReport")
    if not report.blocked:
        return
    raise SyntheticLicenseLineageError(
        "synthetic dataset lineage is blocked: "
        f"{len(report.findings)} findings across "
        f"{report.dependency_count} dependencies"
    )


def manifest_digest(manifest: Any) -> str:
    """Return the stable content digest of a manifest's canonical form."""
    resolved = _coerce_manifest(manifest)
    return stable_hash(resolved.to_dict())


def load_license_lineage_manifest(path: Path) -> SyntheticDatasetLicenseManifest:
    """Read one manifest from a UTF-8 JSON file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SyntheticLicenseLineageError("manifest file could not be read") from error
    return SyntheticDatasetLicenseManifest.from_json(text)
