"""Deterministic compatibility checks between federated clients and round requirements.

A round manifest describes what a round needs (protocol version, training backend,
model and adapter format, quantization, resource class, deterministic kernels,
privacy mechanism and secure-aggregation mode). A client capability envelope
describes what one anonymous client can offer. ``check_federated_compatibility``
compares the two before enrollment and returns a metadata-only report with stable
field-level reason codes: compatible, review-required or incompatible.

The report never carries client identifiers, site names, hardware serials, paths,
endpoints, patient counts, local metrics or training examples, and it never
echoes client-declared values. Unknown mandatory declarations fail closed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping

from .federated_metrics import FederatedPrivacyMechanism

FEDERATED_COMPATIBILITY_SCHEMA_VERSION = "openmed.training.federated_compatibility.v1"
DEFAULT_MAX_DECLARED_CAPABILITIES: Final = 32
MAX_DECLARED_CAPABILITIES: Final = 128
MIN_PROTOCOL_VERSION: Final = 1
MAX_PROTOCOL_VERSION: Final = 10_000

_TOKEN = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")

FEDERATED_COMPATIBILITY_FIELDS: Final[tuple[str, ...]] = (
    "protocol_version",
    "training_backend",
    "model_format",
    "adapter_format",
    "quantization_format",
    "resource_class",
    "deterministic_kernels",
    "privacy_mechanism",
    "secure_aggregation",
)


class FederatedCompatibilityVerdict(str, Enum):
    """Closed enrollment verdicts for one client and one round."""

    COMPATIBLE = "compatible"
    REVIEW_REQUIRED = "review_required"
    INCOMPATIBLE = "incompatible"

    def __str__(self) -> str:
        """Return the serialized verdict value."""
        return self.value


class FederatedTrainingBackend(str, Enum):
    """Closed training backends a round can require."""

    TORCH = "torch"
    MLX = "mlx"
    ONNX = "onnx"
    COREML = "coreml"

    def __str__(self) -> str:
        """Return the serialized backend value."""
        return self.value


class FederatedResourceClass(str, Enum):
    """Ordered resource classes a client can declare."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"

    def __str__(self) -> str:
        """Return the serialized resource-class value."""
        return self.value


class FederatedSecureAggregationMode(str, Enum):
    """Closed secure-aggregation modes a round can require."""

    DISABLED = "disabled"
    PAIRWISE_MASKING = "pairwise_masking"
    SHAMIR = "shamir"

    def __str__(self) -> str:
        """Return the serialized secure-aggregation value."""
        return self.value


class FederatedCompatibilityReasonCode(str, Enum):
    """Stable field-level reason codes emitted by the comparison."""

    PROTOCOL_VERSION_SUPPORTED = "protocol_version_supported"
    PROTOCOL_VERSION_BELOW_MINIMUM = "protocol_version_below_minimum"
    PROTOCOL_VERSION_ABOVE_MAXIMUM = "protocol_version_above_maximum"
    PROTOCOL_VERSION_UNKNOWN = "protocol_version_unknown"
    TRAINING_BACKEND_SUPPORTED = "training_backend_supported"
    TRAINING_BACKEND_UNSUPPORTED = "training_backend_unsupported"
    MODEL_FORMAT_SUPPORTED = "model_format_supported"
    MODEL_FORMAT_UNSUPPORTED = "model_format_unsupported"
    ADAPTER_FORMAT_SUPPORTED = "adapter_format_supported"
    ADAPTER_FORMAT_UNSUPPORTED = "adapter_format_unsupported"
    QUANTIZATION_SUPPORTED = "quantization_supported"
    QUANTIZATION_UNSUPPORTED = "quantization_unsupported"
    RESOURCE_CLASS_SUFFICIENT = "resource_class_sufficient"
    RESOURCE_CLASS_INSUFFICIENT = "resource_class_insufficient"
    DETERMINISTIC_KERNELS_SUPPORTED = "deterministic_kernels_supported"
    DETERMINISTIC_KERNELS_MISSING = "deterministic_kernels_missing"
    PRIVACY_MECHANISM_SUPPORTED = "privacy_mechanism_supported"
    PRIVACY_MECHANISM_UNSUPPORTED = "privacy_mechanism_unsupported"
    SECURE_AGGREGATION_SUPPORTED = "secure_aggregation_supported"
    SECURE_AGGREGATION_UNSUPPORTED = "secure_aggregation_unsupported"
    CAPABILITY_UNKNOWN = "capability_unknown"
    OPTIONAL_CAPABILITY_DIFFERENCE = "optional_capability_difference"

    def __str__(self) -> str:
        """Return the serialized reason-code value."""
        return self.value


FEDERATED_COMPATIBILITY_REASON_CODES: Final[
    tuple[FederatedCompatibilityReasonCode, ...]
] = tuple(sorted(FederatedCompatibilityReasonCode, key=lambda code: code.value))

_VERDICT_BY_REASON: Final[
    dict[FederatedCompatibilityReasonCode, FederatedCompatibilityVerdict]
] = {
    FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.PROTOCOL_VERSION_BELOW_MINIMUM: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.PROTOCOL_VERSION_ABOVE_MAXIMUM: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.PROTOCOL_VERSION_UNKNOWN: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.TRAINING_BACKEND_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.TRAINING_BACKEND_UNSUPPORTED: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.MODEL_FORMAT_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.MODEL_FORMAT_UNSUPPORTED: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.ADAPTER_FORMAT_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.ADAPTER_FORMAT_UNSUPPORTED: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.QUANTIZATION_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.QUANTIZATION_UNSUPPORTED: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.RESOURCE_CLASS_SUFFICIENT: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.RESOURCE_CLASS_INSUFFICIENT: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.DETERMINISTIC_KERNELS_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.DETERMINISTIC_KERNELS_MISSING: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.PRIVACY_MECHANISM_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.PRIVACY_MECHANISM_UNSUPPORTED: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.SECURE_AGGREGATION_SUPPORTED: FederatedCompatibilityVerdict.COMPATIBLE,
    FederatedCompatibilityReasonCode.SECURE_AGGREGATION_UNSUPPORTED: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN: FederatedCompatibilityVerdict.INCOMPATIBLE,
    FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE: FederatedCompatibilityVerdict.REVIEW_REQUIRED,
}

_RESOURCE_CLASS_ORDER: Final[dict[FederatedResourceClass, int]] = {
    FederatedResourceClass.SMALL: 1,
    FederatedResourceClass.MEDIUM: 2,
    FederatedResourceClass.LARGE: 3,
    FederatedResourceClass.XLARGE: 4,
}

_REQUIREMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter_format",
        "deterministic_kernels",
        "model_format",
        "privacy_mechanism",
        "protocol_version",
        "quantization_format",
        "resource_class",
        "schema_version",
        "secure_aggregation",
        "training_backend",
    }
)

_CAPABILITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter_formats",
        "deterministic_kernels",
        "maximum_protocol_version",
        "minimum_protocol_version",
        "model_formats",
        "privacy_mechanisms",
        "quantization_formats",
        "resource_class",
        "schema_version",
        "secure_aggregation_modes",
        "training_backends",
    }
)


class FederatedCompatibilityError(ValueError):
    """Raised when a requirement, capability envelope or report is invalid."""


def _require_bool(value: object, field: str) -> None:
    if type(value) is not bool:
        raise FederatedCompatibilityError(f"{field} must be a boolean")


def _require_optional_bool(value: object, field: str) -> None:
    if value is not None and type(value) is not bool:
        raise FederatedCompatibilityError(f"{field} must be a boolean or null")


def _require_bounded_int(value: object, minimum: int, maximum: int, field: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise FederatedCompatibilityError(
            f"{field} must be an integer between {minimum} and {maximum}"
        )


def _require_token(value: object, field: str) -> None:
    if type(value) is not str or _TOKEN.match(value) is None:
        raise FederatedCompatibilityError(
            f"{field} must be a lowercase token such as 'safetensors' or 'dense'"
        )


def _require_enum(value: object, enum_type: type[Enum], field: str) -> None:
    if not isinstance(value, enum_type):
        raise FederatedCompatibilityError(f"{field} must be a {enum_type.__name__}")


def _require_schema_version(value: object) -> None:
    if value != FEDERATED_COMPATIBILITY_SCHEMA_VERSION:
        raise FederatedCompatibilityError("unsupported federated compatibility schema")


def _canonical_enums(
    value: object,
    enum_type: type[Enum],
    field: str,
    maximum: int,
) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise FederatedCompatibilityError(
            f"{field} must be a tuple of {enum_type.__name__}"
        )
    if len(value) > maximum:
        raise FederatedCompatibilityError(
            f"{field} must not declare more than {maximum} entries"
        )
    for entry in value:
        _require_enum(entry, enum_type, field)
    return tuple(sorted(set(value), key=lambda entry: entry.value))


def _canonical_tokens(value: object, field: str, maximum: int) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise FederatedCompatibilityError(f"{field} must be a tuple of tokens")
    if len(value) > maximum:
        raise FederatedCompatibilityError(
            f"{field} must not declare more than {maximum} entries"
        )
    for entry in value:
        _require_token(entry, field)
    return tuple(sorted(set(value)))


def _mapping(payload: object, field: str) -> Mapping[Any, Any]:
    if not isinstance(payload, Mapping):
        raise FederatedCompatibilityError(f"{field} payload must be a mapping")
    return payload


def _strict_keys(
    payload: Mapping[Any, Any], expected: frozenset[str], field: str
) -> None:
    if set(payload) != expected:
        raise FederatedCompatibilityError(f"{field} payload keys are not supported")


@dataclass(frozen=True, slots=True)
class FederatedRoundRequirement:
    """What one federated round requires from an enrolling client."""

    protocol_version: int
    training_backend: FederatedTrainingBackend
    model_format: str
    adapter_format: str
    resource_class: FederatedResourceClass
    privacy_mechanism: FederatedPrivacyMechanism
    secure_aggregation: FederatedSecureAggregationMode
    quantization_format: str | None = None
    deterministic_kernels: bool = False
    schema_version: str = FEDERATED_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the round requirement in place."""
        _require_bounded_int(
            self.protocol_version,
            MIN_PROTOCOL_VERSION,
            MAX_PROTOCOL_VERSION,
            "protocol_version",
        )
        _require_enum(
            self.training_backend, FederatedTrainingBackend, "training_backend"
        )
        _require_token(self.model_format, "model_format")
        _require_token(self.adapter_format, "adapter_format")
        _require_enum(self.resource_class, FederatedResourceClass, "resource_class")
        _require_enum(
            self.privacy_mechanism, FederatedPrivacyMechanism, "privacy_mechanism"
        )
        _require_enum(
            self.secure_aggregation,
            FederatedSecureAggregationMode,
            "secure_aggregation",
        )
        if self.quantization_format is not None:
            _require_token(self.quantization_format, "quantization_format")
        _require_bool(self.deterministic_kernels, "deterministic_kernels")
        _require_schema_version(self.schema_version)

    def fields(self) -> tuple[str, ...]:
        """Return the required fields compared against a capability envelope."""
        required = [
            "protocol_version",
            "training_backend",
            "model_format",
            "adapter_format",
            "quantization_format",
            "resource_class",
            "deterministic_kernels",
            "privacy_mechanism",
            "secure_aggregation",
        ]
        if self.quantization_format is None:
            required.remove("quantization_format")
        return tuple(required)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload without client metadata."""
        return {
            "protocol_version": self.protocol_version,
            "training_backend": self.training_backend.value,
            "model_format": self.model_format,
            "adapter_format": self.adapter_format,
            "quantization_format": self.quantization_format,
            "resource_class": self.resource_class.value,
            "deterministic_kernels": self.deterministic_kernels,
            "privacy_mechanism": self.privacy_mechanism.value,
            "secure_aggregation": self.secure_aggregation.value,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return the canonical JSON representation of the requirement."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> FederatedRoundRequirement:
        """Validate and load a round requirement from a mapping."""
        mapping = _mapping(payload, "requirement")
        _strict_keys(mapping, _REQUIREMENT_FIELDS, "requirement")
        return cls(
            protocol_version=mapping["protocol_version"],
            training_backend=_coerce_enum(
                mapping["training_backend"],
                FederatedTrainingBackend,
                "training_backend",
            ),
            model_format=mapping["model_format"],
            adapter_format=mapping["adapter_format"],
            resource_class=_coerce_enum(
                mapping["resource_class"],
                FederatedResourceClass,
                "resource_class",
            ),
            privacy_mechanism=_coerce_enum(
                mapping["privacy_mechanism"],
                FederatedPrivacyMechanism,
                "privacy_mechanism",
            ),
            secure_aggregation=_coerce_enum(
                mapping["secure_aggregation"],
                FederatedSecureAggregationMode,
                "secure_aggregation",
            ),
            quantization_format=mapping["quantization_format"],
            deterministic_kernels=mapping["deterministic_kernels"],
            schema_version=mapping["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: object) -> FederatedRoundRequirement:
        """Validate and load a round requirement from JSON text."""
        return cls.from_dict(_json_payload(payload, "requirement"))


@dataclass(frozen=True, slots=True)
class FederatedClientCapabilityEnvelope:
    """Anonymous metadata describing what one client can offer.

    Undeclared values are ``null`` for scalars and empty tuples for collections.
    An undeclared mandatory capability fails closed instead of being assumed
    satisfied.
    """

    training_backends: tuple[FederatedTrainingBackend, ...] = ()
    model_formats: tuple[str, ...] = ()
    adapter_formats: tuple[str, ...] = ()
    privacy_mechanisms: tuple[FederatedPrivacyMechanism, ...] = ()
    secure_aggregation_modes: tuple[FederatedSecureAggregationMode, ...] = ()
    quantization_formats: tuple[str, ...] = ()
    minimum_protocol_version: int | None = None
    maximum_protocol_version: int | None = None
    resource_class: FederatedResourceClass | None = None
    deterministic_kernels: bool | None = None
    schema_version: str = FEDERATED_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and canonicalize the capability envelope in place."""
        object.__setattr__(
            self,
            "training_backends",
            _canonical_enums(
                self.training_backends,
                FederatedTrainingBackend,
                "training_backends",
                DEFAULT_MAX_DECLARED_CAPABILITIES,
            ),
        )
        object.__setattr__(
            self,
            "model_formats",
            _canonical_tokens(
                self.model_formats,
                "model_formats",
                DEFAULT_MAX_DECLARED_CAPABILITIES,
            ),
        )
        object.__setattr__(
            self,
            "adapter_formats",
            _canonical_tokens(
                self.adapter_formats,
                "adapter_formats",
                DEFAULT_MAX_DECLARED_CAPABILITIES,
            ),
        )
        object.__setattr__(
            self,
            "privacy_mechanisms",
            _canonical_enums(
                self.privacy_mechanisms,
                FederatedPrivacyMechanism,
                "privacy_mechanisms",
                DEFAULT_MAX_DECLARED_CAPABILITIES,
            ),
        )
        object.__setattr__(
            self,
            "secure_aggregation_modes",
            _canonical_enums(
                self.secure_aggregation_modes,
                FederatedSecureAggregationMode,
                "secure_aggregation_modes",
                DEFAULT_MAX_DECLARED_CAPABILITIES,
            ),
        )
        object.__setattr__(
            self,
            "quantization_formats",
            _canonical_tokens(
                self.quantization_formats,
                "quantization_formats",
                DEFAULT_MAX_DECLARED_CAPABILITIES,
            ),
        )
        _require_optional_protocol_bound(
            self.minimum_protocol_version,
            "minimum_protocol_version",
        )
        _require_optional_protocol_bound(
            self.maximum_protocol_version,
            "maximum_protocol_version",
        )
        if (self.minimum_protocol_version is None) != (
            self.maximum_protocol_version is None
        ):
            raise FederatedCompatibilityError(
                "protocol version bounds must be declared together"
            )
        if (
            self.minimum_protocol_version is not None
            and self.maximum_protocol_version is not None
            and self.minimum_protocol_version > self.maximum_protocol_version
        ):
            raise FederatedCompatibilityError(
                "minimum_protocol_version must not exceed maximum_protocol_version"
            )
        if self.resource_class is not None:
            _require_enum(self.resource_class, FederatedResourceClass, "resource_class")
        _require_optional_bool(self.deterministic_kernels, "deterministic_kernels")
        _require_schema_version(self.schema_version)

    def declares_protocol_range(self) -> bool:
        """Return whether the client declared a protocol version range."""
        return self.minimum_protocol_version is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload without any client identity."""
        return {
            "training_backends": [entry.value for entry in self.training_backends],
            "model_formats": list(self.model_formats),
            "adapter_formats": list(self.adapter_formats),
            "privacy_mechanisms": [entry.value for entry in self.privacy_mechanisms],
            "secure_aggregation_modes": [
                entry.value for entry in self.secure_aggregation_modes
            ],
            "quantization_formats": list(self.quantization_formats),
            "minimum_protocol_version": self.minimum_protocol_version,
            "maximum_protocol_version": self.maximum_protocol_version,
            "resource_class": (
                self.resource_class.value if self.resource_class is not None else None
            ),
            "deterministic_kernels": self.deterministic_kernels,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return the canonical JSON representation of the capability envelope."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> FederatedClientCapabilityEnvelope:
        """Validate and load a capability envelope from a mapping."""
        mapping = _mapping(payload, "capability")
        _strict_keys(mapping, _CAPABILITY_FIELDS, "capability")
        resource_class = mapping["resource_class"]
        if resource_class is not None:
            resource_class = _coerce_enum(
                resource_class,
                FederatedResourceClass,
                "resource_class",
            )
        return cls(
            training_backends=_coerce_enum_tuple(
                mapping["training_backends"],
                FederatedTrainingBackend,
                "training_backends",
            ),
            model_formats=_coerce_token_tuple(
                mapping["model_formats"], "model_formats"
            ),
            adapter_formats=_coerce_token_tuple(
                mapping["adapter_formats"],
                "adapter_formats",
            ),
            privacy_mechanisms=_coerce_enum_tuple(
                mapping["privacy_mechanisms"],
                FederatedPrivacyMechanism,
                "privacy_mechanisms",
            ),
            secure_aggregation_modes=_coerce_enum_tuple(
                mapping["secure_aggregation_modes"],
                FederatedSecureAggregationMode,
                "secure_aggregation_modes",
            ),
            quantization_formats=_coerce_token_tuple(
                mapping["quantization_formats"],
                "quantization_formats",
            ),
            minimum_protocol_version=mapping["minimum_protocol_version"],
            maximum_protocol_version=mapping["maximum_protocol_version"],
            resource_class=resource_class,
            deterministic_kernels=mapping["deterministic_kernels"],
            schema_version=mapping["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: object) -> FederatedClientCapabilityEnvelope:
        """Validate and load a capability envelope from JSON text."""
        return cls.from_dict(_json_payload(payload, "capability"))


@dataclass(frozen=True, slots=True)
class FederatedCompatibilityFinding:
    """One field-level outcome of the comparison."""

    field: str
    reason: FederatedCompatibilityReasonCode
    required: bool = True

    def __post_init__(self) -> None:
        """Validate the finding against the closed field and reason sets."""
        if (
            type(self.field) is not str
            or self.field not in FEDERATED_COMPATIBILITY_FIELDS
        ):
            raise FederatedCompatibilityError("finding field is not a compared field")
        if not isinstance(self.reason, FederatedCompatibilityReasonCode):
            raise FederatedCompatibilityError(
                "finding reason must be a FederatedCompatibilityReasonCode"
            )
        _require_bool(self.required, "required")
        optional = (
            self.reason
            is FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE
        )
        if self.required is optional:
            raise FederatedCompatibilityError(
                "finding required flag does not match the reason code"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload for the finding."""
        return {
            "field": self.field,
            "reason": self.reason.value,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class FederatedCompatibilityReport:
    """Deterministic, metadata-only enrollment verdict for one client."""

    verdict: FederatedCompatibilityVerdict
    findings: tuple[FederatedCompatibilityFinding, ...]
    incompatible_fields: tuple[str, ...]
    review_fields: tuple[str, ...]
    schema_version: str = FEDERATED_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate that the report is internally consistent."""
        _require_schema_version(self.schema_version)
        if not isinstance(self.verdict, FederatedCompatibilityVerdict):
            raise FederatedCompatibilityError(
                "verdict must be a FederatedCompatibilityVerdict"
            )
        if type(self.findings) is not tuple or any(
            not isinstance(finding, FederatedCompatibilityFinding)
            for finding in self.findings
        ):
            raise FederatedCompatibilityError(
                "findings must be a tuple of FederatedCompatibilityFinding"
            )
        indexes = [
            FEDERATED_COMPATIBILITY_FIELDS.index(finding.field)
            for finding in self.findings
        ]
        if indexes != sorted(indexes):
            raise FederatedCompatibilityError("findings must be ordered by field")
        if _incompatible_fields(self.findings) != self.incompatible_fields:
            raise FederatedCompatibilityError(
                "incompatible fields do not match the findings"
            )
        if _review_fields(self.findings) != self.review_fields:
            raise FederatedCompatibilityError("review fields do not match the findings")
        if _derive_verdict(self.findings) is not self.verdict:
            raise FederatedCompatibilityError("verdict does not match the findings")

    @property
    def ok(self) -> bool:
        """Return whether the client is compatible without review."""
        return self.verdict is FederatedCompatibilityVerdict.COMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload that carries no client metadata."""
        return {
            "verdict": self.verdict.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "incompatible_fields": list(self.incompatible_fields),
            "review_fields": list(self.review_fields),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return the canonical JSON representation of the report."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def check_federated_compatibility(
    requirement: FederatedRoundRequirement,
    capability: FederatedClientCapabilityEnvelope,
) -> FederatedCompatibilityReport:
    """Compare one anonymous capability envelope with one round requirement.

    Unknown mandatory declarations fail closed, optional capability differences
    are reported separately from hard incompatibilities, and the returned report
    contains no client identity, declared values or local data characteristics.
    """
    if type(requirement) is not FederatedRoundRequirement:
        raise FederatedCompatibilityError(
            "requirement must be a FederatedRoundRequirement"
        )
    if type(capability) is not FederatedClientCapabilityEnvelope:
        raise FederatedCompatibilityError(
            "capability must be a FederatedClientCapabilityEnvelope"
        )

    findings: list[FederatedCompatibilityFinding] = []
    findings.extend(_protocol_findings(requirement, capability))
    findings.extend(
        _declared_findings(
            "training_backend",
            requirement.training_backend,
            capability.training_backends,
            FederatedCompatibilityReasonCode.TRAINING_BACKEND_SUPPORTED,
            FederatedCompatibilityReasonCode.TRAINING_BACKEND_UNSUPPORTED,
        )
    )
    findings.extend(
        _declared_findings(
            "model_format",
            requirement.model_format,
            capability.model_formats,
            FederatedCompatibilityReasonCode.MODEL_FORMAT_SUPPORTED,
            FederatedCompatibilityReasonCode.MODEL_FORMAT_UNSUPPORTED,
        )
    )
    findings.extend(
        _declared_findings(
            "adapter_format",
            requirement.adapter_format,
            capability.adapter_formats,
            FederatedCompatibilityReasonCode.ADAPTER_FORMAT_SUPPORTED,
            FederatedCompatibilityReasonCode.ADAPTER_FORMAT_UNSUPPORTED,
        )
    )
    findings.extend(_quantization_findings(requirement, capability))
    findings.extend(_resource_class_findings(requirement, capability))
    findings.extend(_kernel_findings(requirement, capability))
    findings.extend(
        _declared_findings(
            "privacy_mechanism",
            requirement.privacy_mechanism,
            capability.privacy_mechanisms,
            FederatedCompatibilityReasonCode.PRIVACY_MECHANISM_SUPPORTED,
            FederatedCompatibilityReasonCode.PRIVACY_MECHANISM_UNSUPPORTED,
        )
    )
    findings.extend(
        _declared_findings(
            "secure_aggregation",
            requirement.secure_aggregation,
            capability.secure_aggregation_modes,
            FederatedCompatibilityReasonCode.SECURE_AGGREGATION_SUPPORTED,
            FederatedCompatibilityReasonCode.SECURE_AGGREGATION_UNSUPPORTED,
        )
    )

    ordered = tuple(findings)
    return FederatedCompatibilityReport(
        verdict=_derive_verdict(ordered),
        findings=ordered,
        incompatible_fields=_incompatible_fields(ordered),
        review_fields=_review_fields(ordered),
    )


def _protocol_findings(
    requirement: FederatedRoundRequirement,
    capability: FederatedClientCapabilityEnvelope,
) -> list[FederatedCompatibilityFinding]:
    if not capability.declares_protocol_range():
        return [
            _finding(
                "protocol_version",
                FederatedCompatibilityReasonCode.PROTOCOL_VERSION_UNKNOWN,
            )
        ]
    minimum = capability.minimum_protocol_version
    maximum = capability.maximum_protocol_version
    if minimum is not None and requirement.protocol_version < minimum:
        return [
            _finding(
                "protocol_version",
                FederatedCompatibilityReasonCode.PROTOCOL_VERSION_BELOW_MINIMUM,
            )
        ]
    if maximum is not None and requirement.protocol_version > maximum:
        return [
            _finding(
                "protocol_version",
                FederatedCompatibilityReasonCode.PROTOCOL_VERSION_ABOVE_MAXIMUM,
            )
        ]
    return [
        _finding(
            "protocol_version",
            FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED,
        )
    ]


def _declared_findings(
    field: str,
    required: Any,
    declared: tuple[Any, ...],
    supported: FederatedCompatibilityReasonCode,
    unsupported: FederatedCompatibilityReasonCode,
) -> list[FederatedCompatibilityFinding]:
    if not declared:
        return [_finding(field, FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN)]
    if required not in declared:
        return [_finding(field, unsupported)]
    findings = [_finding(field, supported)]
    if len(declared) > 1:
        findings.append(_optional(field))
    return findings


def _quantization_findings(
    requirement: FederatedRoundRequirement,
    capability: FederatedClientCapabilityEnvelope,
) -> list[FederatedCompatibilityFinding]:
    if requirement.quantization_format is None:
        if capability.quantization_formats:
            return [_optional("quantization_format")]
        return []
    if not capability.quantization_formats:
        return [
            _finding(
                "quantization_format",
                FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN,
            )
        ]
    if requirement.quantization_format not in capability.quantization_formats:
        return [
            _finding(
                "quantization_format",
                FederatedCompatibilityReasonCode.QUANTIZATION_UNSUPPORTED,
            )
        ]
    findings = [
        _finding(
            "quantization_format",
            FederatedCompatibilityReasonCode.QUANTIZATION_SUPPORTED,
        )
    ]
    if len(capability.quantization_formats) > 1:
        findings.append(_optional("quantization_format"))
    return findings


def _resource_class_findings(
    requirement: FederatedRoundRequirement,
    capability: FederatedClientCapabilityEnvelope,
) -> list[FederatedCompatibilityFinding]:
    if capability.resource_class is None:
        return [
            _finding(
                "resource_class", FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN
            )
        ]
    client_rank = _RESOURCE_CLASS_ORDER[capability.resource_class]
    required_rank = _RESOURCE_CLASS_ORDER[requirement.resource_class]
    if client_rank < required_rank:
        return [
            _finding(
                "resource_class",
                FederatedCompatibilityReasonCode.RESOURCE_CLASS_INSUFFICIENT,
            )
        ]
    findings = [
        _finding(
            "resource_class", FederatedCompatibilityReasonCode.RESOURCE_CLASS_SUFFICIENT
        )
    ]
    if client_rank > required_rank:
        findings.append(_optional("resource_class"))
    return findings


def _kernel_findings(
    requirement: FederatedRoundRequirement,
    capability: FederatedClientCapabilityEnvelope,
) -> list[FederatedCompatibilityFinding]:
    declared = capability.deterministic_kernels
    if declared is None:
        return [
            _finding(
                "deterministic_kernels",
                FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN,
            )
        ]
    if requirement.deterministic_kernels and not declared:
        return [
            _finding(
                "deterministic_kernels",
                FederatedCompatibilityReasonCode.DETERMINISTIC_KERNELS_MISSING,
            )
        ]
    findings = [
        _finding(
            "deterministic_kernels",
            FederatedCompatibilityReasonCode.DETERMINISTIC_KERNELS_SUPPORTED,
        )
    ]
    if declared and not requirement.deterministic_kernels:
        findings.append(_optional("deterministic_kernels"))
    return findings


def _finding(
    field: str,
    reason: FederatedCompatibilityReasonCode,
) -> FederatedCompatibilityFinding:
    return FederatedCompatibilityFinding(field=field, reason=reason, required=True)


def _optional(field: str) -> FederatedCompatibilityFinding:
    return FederatedCompatibilityFinding(
        field=field,
        reason=FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE,
        required=False,
    )


def _derive_verdict(
    findings: tuple[FederatedCompatibilityFinding, ...],
) -> FederatedCompatibilityVerdict:
    verdicts = {_VERDICT_BY_REASON[finding.reason] for finding in findings}
    if FederatedCompatibilityVerdict.INCOMPATIBLE in verdicts:
        return FederatedCompatibilityVerdict.INCOMPATIBLE
    if FederatedCompatibilityVerdict.REVIEW_REQUIRED in verdicts:
        return FederatedCompatibilityVerdict.REVIEW_REQUIRED
    return FederatedCompatibilityVerdict.COMPATIBLE


def _ordered_fields(fields: set[str]) -> tuple[str, ...]:
    return tuple(field for field in FEDERATED_COMPATIBILITY_FIELDS if field in fields)


def _incompatible_fields(
    findings: tuple[FederatedCompatibilityFinding, ...],
) -> tuple[str, ...]:
    return _ordered_fields(
        {
            finding.field
            for finding in findings
            if _VERDICT_BY_REASON[finding.reason]
            is FederatedCompatibilityVerdict.INCOMPATIBLE
        }
    )


def _review_fields(
    findings: tuple[FederatedCompatibilityFinding, ...],
) -> tuple[str, ...]:
    return _ordered_fields(
        {
            finding.field
            for finding in findings
            if _VERDICT_BY_REASON[finding.reason]
            is FederatedCompatibilityVerdict.REVIEW_REQUIRED
        }
    )


def _coerce_enum(value: object, enum_type: type[Enum], field: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if type(value) is str:
        try:
            return enum_type(value)
        except ValueError:
            raise FederatedCompatibilityError(
                f"{field} must be a documented {enum_type.__name__} value"
            ) from None
    raise FederatedCompatibilityError(f"{field} must be a {enum_type.__name__}")


def _coerce_enum_tuple(
    value: object,
    enum_type: type[Enum],
    field: str,
) -> tuple[Any, ...]:
    if type(value) is not list:
        raise FederatedCompatibilityError(
            f"{field} must be a list of {enum_type.__name__}"
        )
    return tuple(_coerce_enum(entry, enum_type, field) for entry in value)


def _coerce_token_tuple(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise FederatedCompatibilityError(f"{field} must be a list of tokens")
    return tuple(value)


def _json_payload(payload: object, field: str) -> object:
    if type(payload) is not str:
        raise FederatedCompatibilityError(f"{field} payload must be JSON text")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        raise FederatedCompatibilityError(
            f"{field} payload is not valid JSON"
        ) from None


def _require_optional_protocol_bound(value: object, field: str) -> None:
    if value is None:
        return
    _require_bounded_int(value, MIN_PROTOCOL_VERSION, MAX_PROTOCOL_VERSION, field)


__all__ = [
    "DEFAULT_MAX_DECLARED_CAPABILITIES",
    "FEDERATED_COMPATIBILITY_FIELDS",
    "FEDERATED_COMPATIBILITY_REASON_CODES",
    "FEDERATED_COMPATIBILITY_SCHEMA_VERSION",
    "MAX_PROTOCOL_VERSION",
    "MIN_PROTOCOL_VERSION",
    "FederatedClientCapabilityEnvelope",
    "FederatedCompatibilityError",
    "FederatedCompatibilityFinding",
    "FederatedCompatibilityReasonCode",
    "FederatedCompatibilityReport",
    "FederatedCompatibilityVerdict",
    "FederatedResourceClass",
    "FederatedRoundRequirement",
    "FederatedSecureAggregationMode",
    "FederatedTrainingBackend",
    "check_federated_compatibility",
]
