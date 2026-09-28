"""Versioned registry definitions, cases, governance, and safe export."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .contracts import (
    REGISTRY_ADVISORY,
    REGISTRY_COMPATIBILITY_POLICY,
    REGISTRY_SCHEMA_VERSION,
    RegistryAssignment,
    RegistryAssignmentAuthorization,
    RegistryCase,
    RegistryCaseEvent,
    RegistryCaseState,
    RegistryConflictError,
    RegistryContractError,
    RegistryDefinition,
    RegistryDefinitionVersion,
    RegistryExportAuthorization,
    RegistryExportEnvelope,
    RegistryFieldEvidence,
    RegistryFieldResult,
    RegistryFieldRule,
    RegistryFieldState,
    RegistryUnsupportedError,
    RegistryWorkflowPolicy,
)
from .materialize import (
    RegistryFactBinding,
    RegistryMaterialization,
    materialize_registry_cases,
    version_registry_definition,
)
from .workflow import (
    adjudicate_registry_case,
    assign_registry_case,
    begin_registry_review,
    build_registry_export,
    complete_registry_review,
    correct_registry_field,
    mark_registry_case_exported,
)


def load_registry_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for registry artifacts."""

    text = (
        resources.files("openmed.core.schemas.json")
        .joinpath("clinical_registry.schema.json")
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if not isinstance(value, dict):  # pragma: no cover - packaged invariant
        raise RuntimeError("clinical registry schema must be an object")
    return value


__all__ = [
    "REGISTRY_ADVISORY",
    "REGISTRY_COMPATIBILITY_POLICY",
    "REGISTRY_SCHEMA_VERSION",
    "RegistryAssignment",
    "RegistryAssignmentAuthorization",
    "RegistryCase",
    "RegistryCaseEvent",
    "RegistryCaseState",
    "RegistryConflictError",
    "RegistryContractError",
    "RegistryDefinition",
    "RegistryDefinitionVersion",
    "RegistryExportAuthorization",
    "RegistryExportEnvelope",
    "RegistryFactBinding",
    "RegistryFieldEvidence",
    "RegistryFieldResult",
    "RegistryFieldRule",
    "RegistryFieldState",
    "RegistryMaterialization",
    "RegistryUnsupportedError",
    "RegistryWorkflowPolicy",
    "adjudicate_registry_case",
    "assign_registry_case",
    "begin_registry_review",
    "build_registry_export",
    "complete_registry_review",
    "correct_registry_field",
    "load_registry_schema",
    "mark_registry_case_exported",
    "materialize_registry_cases",
    "version_registry_definition",
]
