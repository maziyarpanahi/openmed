"""Guarded assignment, review, adjudication, correction, and export workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id
from openmed.structured.store import StoreResult, StoreState

from .contracts import (
    RegistryAssignment,
    RegistryAssignmentAuthorization,
    RegistryCase,
    RegistryCaseEvent,
    RegistryCaseState,
    RegistryContractError,
    RegistryDefinitionVersion,
    RegistryExportAuthorization,
    RegistryExportEnvelope,
    RegistryFieldResult,
    RegistryFieldState,
)


def assign_registry_case(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
    *,
    authorization: RegistryAssignmentAuthorization,
    assigned_at: str,
) -> StoreResult[RegistryCase]:
    """Assign a review-required case under exact owner-scope authorization."""

    mismatch = _contract_mismatch(case, definition_version)
    if mismatch is not None:
        return mismatch
    policy = definition_version.definition.workflow
    if case.state is not RegistryCaseState.REVIEW_REQUIRED:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_assignment_state_conflict"
        )
    if not isinstance(authorization, RegistryAssignmentAuthorization):
        return StoreResult.outcome(
            StoreState.DENIED, "registry_assignment_authorization_required"
        )
    if (
        authorization.definition_version_id != definition_version.version_id
        or authorization.owner_scope_id != policy.owner_scope_id
        or authorization.workflow_policy_digest != policy.digest
    ):
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_assignment_authorization_conflict"
        )
    if not authorization.assignment_approved:
        return StoreResult.outcome(
            StoreState.DENIED, "registry_assignment_not_approved"
        )
    assignment = RegistryAssignment(
        assignment_id=derived_opaque_id(
            "registryassignment", case.case_id, authorization.digest, assigned_at
        ),
        queue_id=authorization.queue_id,
        authorization_id=authorization.authorization_id,
        authorization_digest=authorization.digest,
        assigned_at=assigned_at,
    )
    return _transition(
        case,
        to_state=RegistryCaseState.ASSIGNED,
        action="assign",
        occurred_at=assigned_at,
        reason_code="owner_scope_approved",
        policy_digest=policy.digest,
        artifact_digest=authorization.digest,
        assignment=assignment,
    )


def begin_registry_review(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
    *,
    occurred_at: str,
) -> StoreResult[RegistryCase]:
    """Begin review only after any configured assignment gate is satisfied."""

    mismatch = _contract_mismatch(case, definition_version)
    if mismatch is not None:
        return mismatch
    policy = definition_version.definition.workflow
    expected = (
        RegistryCaseState.ASSIGNED
        if policy.assignment_required
        else RegistryCaseState.REVIEW_REQUIRED
    )
    if case.state is not expected:
        return StoreResult.outcome(StoreState.CONFLICT, "registry_review_gate_required")
    return _transition(
        case,
        to_state=RegistryCaseState.IN_REVIEW,
        action="begin_review",
        occurred_at=occurred_at,
        reason_code="review_started",
        policy_digest=policy.digest,
    )


def complete_registry_review(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
    *,
    approved: bool,
    occurred_at: str,
    reason_code: str,
) -> StoreResult[RegistryCase]:
    """Complete review without silently bypassing configured adjudication."""

    mismatch = _contract_mismatch(case, definition_version)
    if mismatch is not None:
        return mismatch
    if type(approved) is not bool:
        return StoreResult.outcome(
            StoreState.FAILURE, "registry_review_decision_invalid"
        )
    if case.state is not RegistryCaseState.IN_REVIEW:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_review_state_conflict"
        )
    policy = definition_version.definition.workflow
    needs_adjudication = any(
        item.state in policy.adjudication_field_states for item in case.fields
    )
    to_state = (
        RegistryCaseState.ADJUDICATION_REQUIRED
        if approved and needs_adjudication
        else RegistryCaseState.EXPORT_READY
        if approved
        else RegistryCaseState.REVIEW_REJECTED
    )
    return _transition(
        case,
        to_state=to_state,
        action="complete_review",
        occurred_at=occurred_at,
        reason_code=reason_code,
        policy_digest=policy.digest,
    )


def adjudicate_registry_case(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
    *,
    approved: bool,
    occurred_at: str,
    reason_code: str,
    decision_digest: str,
) -> StoreResult[RegistryCase]:
    """Adjudicate a reviewed case and retain digest-only decision custody."""

    mismatch = _contract_mismatch(case, definition_version)
    if mismatch is not None:
        return mismatch
    if type(approved) is not bool:
        return StoreResult.outcome(
            StoreState.FAILURE, "registry_adjudication_decision_invalid"
        )
    if case.state is not RegistryCaseState.ADJUDICATION_REQUIRED:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_adjudication_state_conflict"
        )
    to_state = (
        RegistryCaseState.EXPORT_READY
        if approved
        else RegistryCaseState.ADJUDICATION_REJECTED
    )
    return _transition(
        case,
        to_state=to_state,
        action="adjudicate",
        occurred_at=occurred_at,
        reason_code=reason_code,
        policy_digest=definition_version.definition.workflow.digest,
        artifact_digest=decision_digest,
    )


def correct_registry_field(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
    *,
    replacement: RegistryFieldResult,
    expected_prior_digest: str,
    occurred_at: str,
    reason_code: str,
) -> StoreResult[RegistryCase]:
    """Append a field correction and restart configured governance gates."""

    mismatch = _contract_mismatch(case, definition_version)
    if mismatch is not None:
        return mismatch
    policy = definition_version.definition.workflow
    if not policy.correction_allowed:
        return StoreResult.outcome(StoreState.DENIED, "registry_correction_denied")
    if not isinstance(replacement, RegistryFieldResult):
        return StoreResult.outcome(StoreState.FAILURE, "registry_correction_invalid")
    if replacement.state is not RegistryFieldState.CORRECTED:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_correction_state_required"
        )
    prior = next(
        (item for item in case.fields if item.field_id == replacement.field_id), None
    )
    if prior is None:
        return StoreResult.outcome(StoreState.UNKNOWN, "registry_field_not_found")
    if prior.digest != expected_prior_digest:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_field_digest_conflict"
        )
    fields = tuple(
        replacement if item.field_id == replacement.field_id else item
        for item in case.fields
    )
    to_state = (
        RegistryCaseState.REVIEW_REQUIRED
        if any(item.state in policy.review_field_states for item in fields)
        else RegistryCaseState.EXPORT_READY
    )
    return _transition(
        case,
        to_state=to_state,
        action="correct_field",
        occurred_at=occurred_at,
        reason_code=reason_code,
        policy_digest=policy.digest,
        artifact_digest=replacement.digest,
        assignment=None,
        fields=fields,
    )


def build_registry_export(
    definition_version: RegistryDefinitionVersion,
    cases: tuple[RegistryCase, ...],
    *,
    authorization: RegistryExportAuthorization,
    created_at: str,
) -> StoreResult[RegistryExportEnvelope]:
    """Build a privacy-safe export manifest for governance-complete cases."""

    if not isinstance(definition_version, RegistryDefinitionVersion):
        return StoreResult.outcome(StoreState.FAILURE, "registry_definition_invalid")
    if not cases:
        return StoreResult.outcome(StoreState.UNKNOWN, "registry_export_empty")
    if any(not isinstance(item, RegistryCase) for item in cases):
        return StoreResult.outcome(StoreState.FAILURE, "registry_export_case_invalid")
    if any(_contract_mismatch(item, definition_version) is not None for item in cases):
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_export_definition_conflict"
        )
    if len({item.case_id for item in cases}) != len(cases):
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_export_case_duplicate"
        )
    if any(
        item.state not in {RegistryCaseState.EXPORT_READY, RegistryCaseState.EXPORTED}
        for item in cases
    ):
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_export_governance_incomplete"
        )
    if not isinstance(authorization, RegistryExportAuthorization):
        return StoreResult.outcome(
            StoreState.DENIED, "registry_export_authorization_required"
        )
    workflow = definition_version.definition.workflow
    if (
        authorization.definition_version_id != definition_version.version_id
        or authorization.privacy_policy_digest != workflow.privacy_policy_digest
        or authorization.export_policy_digest != workflow.export_policy_digest
    ):
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_export_authorization_conflict"
        )
    if not authorization.export_approved:
        return StoreResult.outcome(StoreState.DENIED, "registry_export_not_approved")
    try:
        case_digests = {item.case_id: item.case_digest for item in cases}
        export_id = derived_opaque_id(
            "registryexport",
            definition_version.version_id,
            case_digests,
            authorization.digest,
            created_at,
        )
        envelope = RegistryExportEnvelope(
            export_id=export_id,
            definition_version_id=definition_version.version_id or "",
            definition_digest=definition_version.definition_digest,
            registry_id=definition_version.definition.registry_id,
            created_at=created_at,
            case_digests=case_digests,
            source_snapshot_ids=tuple(
                sorted({item.source_snapshot_id for item in cases})
            ),
            authorization_id=authorization.authorization_id,
            authorization_digest=authorization.digest,
            privacy_policy_digest=authorization.privacy_policy_digest,
            export_policy_digest=authorization.export_policy_digest,
        )
    except (RegistryContractError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "registry_export_invalid")
    return StoreResult.success(envelope)


def mark_registry_case_exported(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
    *,
    envelope: RegistryExportEnvelope,
    occurred_at: str,
) -> StoreResult[RegistryCase]:
    """Record that the exact case digest belongs to a verified export envelope."""

    mismatch = _contract_mismatch(case, definition_version)
    if mismatch is not None:
        return mismatch
    if case.state is not RegistryCaseState.EXPORT_READY:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_export_state_conflict"
        )
    if not isinstance(envelope, RegistryExportEnvelope):
        return StoreResult.outcome(
            StoreState.FAILURE, "registry_export_manifest_invalid"
        )
    if envelope.case_digests.get(case.case_id) != case.case_digest:
        return StoreResult.outcome(StoreState.CONFLICT, "registry_export_case_conflict")
    workflow = definition_version.definition.workflow
    if (
        envelope.definition_version_id != definition_version.version_id
        or envelope.definition_digest != definition_version.definition_digest
        or envelope.registry_id != definition_version.definition.registry_id
        or envelope.privacy_policy_digest != workflow.privacy_policy_digest
        or envelope.export_policy_digest != workflow.export_policy_digest
        or case.source_snapshot_id not in envelope.source_snapshot_ids
    ):
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_export_context_conflict"
        )
    return _transition(
        case,
        to_state=RegistryCaseState.EXPORTED,
        action="export",
        occurred_at=occurred_at,
        reason_code="export_manifest_recorded",
        policy_digest=definition_version.definition.workflow.digest,
        artifact_digest=envelope.manifest_digest,
    )


def _contract_mismatch(
    case: RegistryCase,
    definition_version: RegistryDefinitionVersion,
) -> StoreResult[RegistryCase] | None:
    if not isinstance(case, RegistryCase) or not isinstance(
        definition_version, RegistryDefinitionVersion
    ):
        return StoreResult.outcome(StoreState.FAILURE, "registry_contract_invalid")
    if (
        case.definition_version_id != definition_version.version_id
        or case.definition_digest != definition_version.definition_digest
        or case.workflow_policy_digest != definition_version.definition.workflow.digest
    ):
        return StoreResult.outcome(StoreState.CONFLICT, "registry_definition_conflict")
    expected_fields = {item.field_id for item in definition_version.definition.fields}
    if {item.field_id for item in case.fields} != expected_fields:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_field_contract_conflict"
        )
    expected_origin = (
        RegistryCaseState.REVIEW_REQUIRED
        if any(
            item.state in definition_version.definition.workflow.review_field_states
            for item in case.fields
        )
        else RegistryCaseState.EXPORT_READY
    )
    if case.origin_state is not expected_origin and not case.events:
        return StoreResult.outcome(
            StoreState.CONFLICT, "registry_origin_state_conflict"
        )
    return None


def _transition(
    case: RegistryCase,
    *,
    to_state: RegistryCaseState,
    action: str,
    occurred_at: str,
    reason_code: str,
    policy_digest: str,
    artifact_digest: str | None = None,
    assignment: RegistryAssignment | None | object = ...,
    fields: tuple[RegistryFieldResult, ...] | None = None,
) -> StoreResult[RegistryCase]:
    try:
        identity: dict[str, Any] = {
            "action": action,
            "artifact_digest": artifact_digest,
            "case_id": case.case_id,
            "from_state": case.state.value,
            "occurred_at": occurred_at,
            "policy_digest": policy_digest,
            "reason_code": reason_code,
            "to_state": to_state.value,
        }
        event = RegistryCaseEvent(
            event_id=derived_opaque_id("registryevent", identity),
            case_id=case.case_id,
            action=action,
            from_state=case.state,
            to_state=to_state,
            occurred_at=occurred_at,
            reason_code=reason_code,
            policy_digest=policy_digest,
            artifact_digest=artifact_digest,
        )
        changes: dict[str, Any] = {
            "events": (*case.events, event),
            "state": to_state,
        }
        if assignment is not ...:
            changes["assignment"] = assignment
        if fields is not None:
            changes["fields"] = fields
        updated = replace(case, **changes)
    except (RegistryContractError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "registry_transition_invalid")
    return StoreResult.success(updated, created=True)


__all__ = [
    "adjudicate_registry_case",
    "assign_registry_case",
    "begin_registry_review",
    "build_registry_export",
    "complete_registry_review",
    "correct_registry_field",
    "mark_registry_case_exported",
]
