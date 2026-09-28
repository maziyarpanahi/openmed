"""Offline FHIR R4 capability preflight for content-free write plans.

The parser inspects only bounded, caller-supplied ``CapabilityStatement``
metadata. It does not discover a server, read credentials, accept clinical
resource payloads, or execute a write. A result other than ``compatible`` must
therefore remain a stop or human-review decision for downstream callers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "MAX_CAPABILITY_INTERACTIONS",
    "MAX_CAPABILITY_RESOURCES",
    "MAX_CAPABILITY_REST_BLOCKS",
    "CapabilityStatementError",
    "FHIRCapabilityMetadata",
    "FHIRPreflightReason",
    "FHIRPreflightResult",
    "FHIRPreflightStatus",
    "FHIRResourceCapability",
    "FHIRWriteInteraction",
    "FHIRWritePlan",
    "evaluate_fhir_write_plan",
    "parse_capability_statement",
    "parse_fhir_capability_statement",
    "preflight_fhir_write",
    "preflight_write_plan",
]

MAX_CAPABILITY_REST_BLOCKS: Final = 8
MAX_CAPABILITY_RESOURCES: Final = 512
MAX_CAPABILITY_INTERACTIONS: Final = 64

_FHIR_R4_VERSIONS: Final = frozenset({"4.0", "4.0.0", "4.0.1"})
_RESOURCE_TYPE_RE: Final = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_RESOURCE_INTERACTIONS: Final = frozenset(
    {
        "read",
        "vread",
        "update",
        "patch",
        "delete",
        "history-instance",
        "history-type",
        "create",
        "search-type",
    }
)
_SYSTEM_INTERACTIONS: Final = frozenset(
    {"transaction", "batch", "history-system", "search-system"}
)


class FHIRWriteInteraction(str, Enum):
    """Write shapes that can be checked without a clinical resource."""

    CREATE = "create"
    UPDATE = "update"
    TRANSACTION = "transaction"


class FHIRPreflightStatus(str, Enum):
    """Terminal classification for a planned FHIR write."""

    COMPATIBLE = "compatible"
    REVIEW = "review"
    INCOMPATIBLE = "incompatible"


class FHIRPreflightReason(str, Enum):
    """Stable, content-free reason codes emitted by the preflight."""

    SUPPORTED = "supported"
    CAPABILITY_STATEMENT_MALFORMED = "capability_statement_malformed"
    FHIR_VERSION_NOT_SUPPORTED = "fhir_version_not_supported"
    RESOURCE_NOT_SUPPORTED = "resource_not_supported"
    INTERACTION_NOT_SUPPORTED = "interaction_not_supported"
    CONDITIONAL_CREATE_NOT_SUPPORTED = "conditional_create_not_supported"
    CONDITIONAL_CREATE_UNDECLARED = "conditional_create_undeclared"
    CONDITIONAL_UPDATE_NOT_SUPPORTED = "conditional_update_not_supported"
    CONDITIONAL_UPDATE_UNDECLARED = "conditional_update_undeclared"
    TRANSACTION_NOT_SUPPORTED = "transaction_not_supported"


class CapabilityStatementError(ValueError):
    """Raised when capability metadata cannot be safely parsed."""

    def __init__(self, reason_code: FHIRPreflightReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class FHIRResourceCapability:
    """Normalized write metadata for one FHIR resource type."""

    resource_type: str
    interactions: frozenset[str]
    conditional_create: bool | None
    conditional_update: bool | None

    def __post_init__(self) -> None:
        if not _is_resource_type(self.resource_type):
            raise ValueError("resource_type is invalid")
        interactions = frozenset(self.interactions)
        if not interactions <= _RESOURCE_INTERACTIONS:
            raise ValueError("resource interactions are invalid")
        if (
            self.conditional_create is not None
            and type(self.conditional_create) is not bool
        ):
            raise ValueError("conditional_create must be boolean or None")
        if (
            self.conditional_update is not None
            and type(self.conditional_update) is not bool
        ):
            raise ValueError("conditional_update must be boolean or None")
        object.__setattr__(self, "interactions", interactions)


@dataclass(frozen=True, slots=True)
class FHIRCapabilityMetadata:
    """Bounded normalized server capabilities from a FHIR R4 statement."""

    fhir_version: str
    resources: tuple[FHIRResourceCapability, ...]
    system_interactions: frozenset[str]

    def __post_init__(self) -> None:
        if self.fhir_version not in _FHIR_R4_VERSIONS:
            raise ValueError("fhir_version is unsupported")
        resources = tuple(self.resources)
        if len(resources) > MAX_CAPABILITY_RESOURCES:
            raise ValueError("resources exceed the supported limit")
        if any(not isinstance(item, FHIRResourceCapability) for item in resources):
            raise TypeError("resources must contain FHIRResourceCapability values")
        resource_types = tuple(item.resource_type for item in resources)
        if resource_types != tuple(sorted(resource_types)):
            raise ValueError("resources must be in deterministic order")
        if len(resource_types) != len(set(resource_types)):
            raise ValueError("resources must contain unique resource types")
        system_interactions = frozenset(self.system_interactions)
        if not system_interactions <= _SYSTEM_INTERACTIONS:
            raise ValueError("system interactions are invalid")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "system_interactions", system_interactions)

    def for_resource(self, resource_type: str) -> FHIRResourceCapability | None:
        """Return normalized metadata for ``resource_type``, if declared."""

        return next(
            (
                capability
                for capability in self.resources
                if capability.resource_type == resource_type
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class FHIRWritePlan:
    """Content-free description of one planned FHIR write.

    The plan intentionally has no payload, identifier, endpoint, or credential
    field. Create and update plans name only a resource type. Transaction plans
    are system-level and therefore omit ``resource_type``.
    """

    interaction: FHIRWriteInteraction
    resource_type: str | None = None
    conditional: bool = False

    def __post_init__(self) -> None:
        try:
            interaction = FHIRWriteInteraction(self.interaction)
        except (TypeError, ValueError):
            raise ValueError("interaction is unsupported") from None
        if type(self.conditional) is not bool:
            raise TypeError("conditional must be a boolean")
        if interaction is FHIRWriteInteraction.TRANSACTION:
            if self.resource_type is not None:
                raise ValueError("transaction plans must omit resource_type")
            if self.conditional:
                raise ValueError("transaction plans cannot be conditional")
        elif not _is_resource_type(self.resource_type):
            raise ValueError("create and update plans require a resource_type")
        object.__setattr__(self, "interaction", interaction)


@dataclass(frozen=True, slots=True)
class FHIRPreflightResult:
    """Deterministic decision containing no resource or server content."""

    status: FHIRPreflightStatus
    reason_code: FHIRPreflightReason

    def __post_init__(self) -> None:
        try:
            status = FHIRPreflightStatus(self.status)
            reason_code = FHIRPreflightReason(self.reason_code)
        except (TypeError, ValueError):
            raise ValueError("preflight result is invalid") from None
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason_code)

    @property
    def is_compatible(self) -> bool:
        """Return whether the declared capability permits the plan."""

        return self.status is FHIRPreflightStatus.COMPATIBLE

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible result with stable string values."""

        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
        }


def parse_capability_statement(
    statement: Mapping[str, Any],
) -> FHIRCapabilityMetadata:
    """Parse bounded FHIR R4 server write capabilities from local metadata.

    Args:
        statement: An already-local CapabilityStatement mapping. Only the
            ``resourceType``, ``fhirVersion``, and bounded ``rest`` capability
            structures are inspected.

    Returns:
        Immutable, normalized server capability metadata.

    Raises:
        CapabilityStatementError: If the statement is malformed, declares an
            unsupported FHIR version, or exceeds a parser bound.
    """

    if not isinstance(statement, Mapping):
        raise _malformed("CapabilityStatement must be a mapping")
    if statement.get("resourceType") != "CapabilityStatement":
        raise _malformed("resourceType must be CapabilityStatement")

    fhir_version = statement.get("fhirVersion")
    if type(fhir_version) is not str:
        raise _malformed("fhirVersion must be present")
    if fhir_version not in _FHIR_R4_VERSIONS:
        raise CapabilityStatementError(
            FHIRPreflightReason.FHIR_VERSION_NOT_SUPPORTED,
            "FHIR version is not supported",
        )

    rest_blocks = _bounded_list(
        statement.get("rest"),
        field_name="rest",
        limit=MAX_CAPABILITY_REST_BLOCKS,
    )
    merged: dict[str, _MutableResourceCapability] = {}
    system_interactions: set[str] = set()
    resource_count = 0

    for rest in rest_blocks:
        if not isinstance(rest, Mapping):
            raise _malformed("rest entries must be mappings")
        mode = rest.get("mode")
        if type(mode) is not str or mode not in {"client", "server"}:
            raise _malformed("rest mode is invalid")
        if mode != "server":
            continue

        for code in _parse_interactions(
            rest.get("interaction", []),
            allowed=_SYSTEM_INTERACTIONS,
            field_name="rest.interaction",
        ):
            system_interactions.add(code)

        resources = _bounded_list(
            rest.get("resource", []),
            field_name="rest.resource",
            limit=MAX_CAPABILITY_RESOURCES,
        )
        resource_count += len(resources)
        if resource_count > MAX_CAPABILITY_RESOURCES:
            raise _malformed("resource declarations exceed the supported limit")
        for resource in resources:
            _merge_resource_capability(merged, resource)

    resources = tuple(
        FHIRResourceCapability(
            resource_type=resource_type,
            interactions=frozenset(capability.interactions),
            conditional_create=_merge_conditional(capability.conditional_create),
            conditional_update=_merge_conditional(capability.conditional_update),
        )
        for resource_type, capability in sorted(merged.items())
    )
    return FHIRCapabilityMetadata(
        fhir_version=fhir_version,
        resources=resources,
        system_interactions=frozenset(system_interactions),
    )


def evaluate_fhir_write_plan(
    capabilities: FHIRCapabilityMetadata,
    plan: FHIRWritePlan,
) -> FHIRPreflightResult:
    """Compare normalized server capabilities with a content-free write plan."""

    if not isinstance(capabilities, FHIRCapabilityMetadata):
        raise TypeError("capabilities must be FHIRCapabilityMetadata")
    if not isinstance(plan, FHIRWritePlan):
        raise TypeError("plan must be FHIRWritePlan")

    if plan.interaction is FHIRWriteInteraction.TRANSACTION:
        if "transaction" in capabilities.system_interactions:
            return _result(
                FHIRPreflightStatus.COMPATIBLE,
                FHIRPreflightReason.SUPPORTED,
            )
        return _result(
            FHIRPreflightStatus.INCOMPATIBLE,
            FHIRPreflightReason.TRANSACTION_NOT_SUPPORTED,
        )

    assert plan.resource_type is not None
    resource = capabilities.for_resource(plan.resource_type)
    if resource is None:
        return _result(
            FHIRPreflightStatus.INCOMPATIBLE,
            FHIRPreflightReason.RESOURCE_NOT_SUPPORTED,
        )
    if plan.interaction.value not in resource.interactions:
        return _result(
            FHIRPreflightStatus.INCOMPATIBLE,
            FHIRPreflightReason.INTERACTION_NOT_SUPPORTED,
        )
    if not plan.conditional:
        return _result(
            FHIRPreflightStatus.COMPATIBLE,
            FHIRPreflightReason.SUPPORTED,
        )

    if plan.interaction is FHIRWriteInteraction.CREATE:
        return _conditional_result(
            resource.conditional_create,
            unsupported=FHIRPreflightReason.CONDITIONAL_CREATE_NOT_SUPPORTED,
            undeclared=FHIRPreflightReason.CONDITIONAL_CREATE_UNDECLARED,
        )
    return _conditional_result(
        resource.conditional_update,
        unsupported=FHIRPreflightReason.CONDITIONAL_UPDATE_NOT_SUPPORTED,
        undeclared=FHIRPreflightReason.CONDITIONAL_UPDATE_UNDECLARED,
    )


def preflight_write_plan(
    statement: Mapping[str, Any],
    plan: FHIRWritePlan,
) -> FHIRPreflightResult:
    """Parse a local statement and classify a write plan without side effects.

    Malformed metadata yields ``review`` rather than raising or granting write
    compatibility. A well-formed but non-R4 statement is deterministically
    incompatible with this R4-only contract.
    """

    if not isinstance(plan, FHIRWritePlan):
        raise TypeError("plan must be FHIRWritePlan")
    try:
        capabilities = parse_capability_statement(statement)
    except CapabilityStatementError as exc:
        if exc.reason_code is FHIRPreflightReason.FHIR_VERSION_NOT_SUPPORTED:
            return _result(FHIRPreflightStatus.INCOMPATIBLE, exc.reason_code)
        return _result(
            FHIRPreflightStatus.REVIEW,
            FHIRPreflightReason.CAPABILITY_STATEMENT_MALFORMED,
        )
    return evaluate_fhir_write_plan(capabilities, plan)


# Explicit FHIR-prefixed aliases make direct imports self-describing while the
# shorter names keep documentation and call sites readable.
parse_fhir_capability_statement = parse_capability_statement
preflight_fhir_write = preflight_write_plan


@dataclass(slots=True)
class _MutableResourceCapability:
    interactions: set[str]
    conditional_create: list[bool | None]
    conditional_update: list[bool | None]


def _merge_resource_capability(
    merged: dict[str, _MutableResourceCapability],
    resource: Any,
) -> None:
    if not isinstance(resource, Mapping):
        raise _malformed("resource entries must be mappings")
    resource_type = resource.get("type")
    if not _is_resource_type(resource_type):
        raise _malformed("resource type is invalid")
    interactions = _parse_interactions(
        resource.get("interaction", []),
        allowed=_RESOURCE_INTERACTIONS,
        field_name="resource.interaction",
    )
    conditional_create = _optional_boolean(resource, "conditionalCreate")
    conditional_update = _optional_boolean(resource, "conditionalUpdate")
    capability = merged.setdefault(
        resource_type,
        _MutableResourceCapability(set(), [], []),
    )
    capability.interactions.update(interactions)
    capability.conditional_create.append(conditional_create)
    capability.conditional_update.append(conditional_update)


def _parse_interactions(
    value: Any,
    *,
    allowed: frozenset[str],
    field_name: str,
) -> tuple[str, ...]:
    interactions = _bounded_list(
        value,
        field_name=field_name,
        limit=MAX_CAPABILITY_INTERACTIONS,
    )
    codes: list[str] = []
    for interaction in interactions:
        if not isinstance(interaction, Mapping):
            raise _malformed(f"{field_name} entries must be mappings")
        code = interaction.get("code")
        if type(code) is not str or code not in allowed:
            raise _malformed(f"{field_name} code is invalid")
        codes.append(code)
    return tuple(codes)


def _bounded_list(value: Any, *, field_name: str, limit: int) -> list[Any]:
    if type(value) is not list:
        raise _malformed(f"{field_name} must be a list")
    if len(value) > limit:
        raise _malformed(f"{field_name} exceeds the supported limit")
    return value


def _optional_boolean(mapping: Mapping[str, Any], field_name: str) -> bool | None:
    if field_name not in mapping:
        return None
    value = mapping[field_name]
    if type(value) is not bool:
        raise _malformed(f"{field_name} must be a boolean")
    return value


def _merge_conditional(values: list[bool | None]) -> bool | None:
    if True in values:
        return True
    if values and all(value is False for value in values):
        return False
    return None


def _conditional_result(
    declared: bool | None,
    *,
    unsupported: FHIRPreflightReason,
    undeclared: FHIRPreflightReason,
) -> FHIRPreflightResult:
    if declared is True:
        return _result(
            FHIRPreflightStatus.COMPATIBLE,
            FHIRPreflightReason.SUPPORTED,
        )
    if declared is False:
        return _result(FHIRPreflightStatus.INCOMPATIBLE, unsupported)
    return _result(FHIRPreflightStatus.REVIEW, undeclared)


def _result(
    status: FHIRPreflightStatus,
    reason_code: FHIRPreflightReason,
) -> FHIRPreflightResult:
    return FHIRPreflightResult(status=status, reason_code=reason_code)


def _is_resource_type(value: Any) -> bool:
    return type(value) is str and _RESOURCE_TYPE_RE.fullmatch(value) is not None


def _malformed(message: str) -> CapabilityStatementError:
    return CapabilityStatementError(
        FHIRPreflightReason.CAPABILITY_STATEMENT_MALFORMED,
        message,
    )
