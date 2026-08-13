"""FHIR R4 export helpers for clinical resources."""

from __future__ import annotations

from .bundle import to_bundle
from .codeable_concept import (
    GROUNDED_CODE_PROVENANCE_EXTENSION_URL,
    MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
    MEDICAL_DEVICE_ASSIST_ONLY_DISCLAIMER,
    POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL,
    postcoordinated_codeable_concept,
    stamp_postcoordination_provenance,
    to_codeable_concept,
)
from .condition import (
    CONDITION_CLINICAL_SYSTEM,
    CONDITION_VER_STATUS_SYSTEM,
    to_condition,
)
from .grounded import (
    COREFERENCE_EVIDENCE_EXTENSION_URL,
    FHIR_RESOURCE_TYPES,
    to_fhir,
)
from .operation_outcome import (
    OperationOutcomeIssue,
    from_validation_result,
    to_operation_outcome,
)
from .privacy import (
    INDIA_HEALTH_ID_REDACTION,
    is_india_health_identifier,
    sanitize_india_health_identifiers,
)
from .profile_check import check_bundle
from .provenance import to_audit_event, to_provenance
from .reference_types import (
    FHIR_R4_REFERENCE_TARGETS,
    FHIR_R5_REFERENCE_TARGETS,
    REFERENCE_TARGET_ALLOWLISTS,
    ReferenceTargetIssue,
    check_reference_targets,
    find_reference_target_issues,
    validate_fhir_reference_types,
    validate_reference_targets,
    validate_reference_types,
)
from .references import deterministic_fullurl

__all__ = [
    "CONDITION_CLINICAL_SYSTEM",
    "CONDITION_VER_STATUS_SYSTEM",
    "COREFERENCE_EVIDENCE_EXTENSION_URL",
    "FHIR_RESOURCE_TYPES",
    "GROUNDED_CODE_PROVENANCE_EXTENSION_URL",
    "MEDICAL_DEVICE_ASSIST_EXTENSION_URL",
    "MEDICAL_DEVICE_ASSIST_ONLY_DISCLAIMER",
    "POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL",
    "to_condition",
    "to_codeable_concept",
    "to_fhir",
    "postcoordinated_codeable_concept",
    "stamp_postcoordination_provenance",
    "deterministic_fullurl",
    "OperationOutcomeIssue",
    "INDIA_HEALTH_ID_REDACTION",
    "is_india_health_identifier",
    "sanitize_india_health_identifiers",
    "check_bundle",
    "to_audit_event",
    "from_validation_result",
    "to_bundle",
    "to_operation_outcome",
    "to_provenance",
    "FHIR_R4_REFERENCE_TARGETS",
    "FHIR_R5_REFERENCE_TARGETS",
    "REFERENCE_TARGET_ALLOWLISTS",
    "ReferenceTargetIssue",
    "check_reference_targets",
    "find_reference_target_issues",
    "validate_fhir_reference_types",
    "validate_reference_targets",
    "validate_reference_types",
]
