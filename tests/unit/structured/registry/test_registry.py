"""Registry definition, materialization, governance, and export tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.clinical.review_transitions import ClinicalReviewPacket
from openmed.structured.cohort import (
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)
from openmed.structured.registry import (
    RegistryAssignmentAuthorization,
    RegistryCase,
    RegistryCaseEvent,
    RegistryCaseState,
    RegistryDefinition,
    RegistryExportAuthorization,
    RegistryFactBinding,
    RegistryFieldEvidence,
    RegistryFieldResult,
    RegistryFieldRule,
    RegistryFieldState,
    RegistryWorkflowPolicy,
    adjudicate_registry_case,
    assign_registry_case,
    begin_registry_review,
    build_registry_export,
    complete_registry_review,
    correct_registry_field,
    load_registry_schema,
    mark_registry_case_exported,
    materialize_registry_cases,
    version_registry_definition,
)
from openmed.structured.store import StoreState

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "cohort"
    / "phenotypes"
    / "diabetes_on_metformin.json"
)
CREATED = "2026-01-02T03:04:05Z"


def _criterion(
    criterion_id: str,
    state: MembershipState,
    suffix: str,
) -> CriterionMembership:
    return CriterionMembership(
        criterion_id=criterion_id,
        state=state,
        evidence=(
            ()
            if state is MembershipState.UNKNOWN
            else (
                MembershipEvidence(
                    evidence_id=f"evidence_{suffix * 16}",
                    fact_id=f"fact_{suffix * 16}",
                    time_window_id=f"window_{suffix * 16}",
                ),
            )
        ),
        reason_codes=("evidence_missing",) if state is MembershipState.UNKNOWN else (),
    )


def _execution():
    definition = save_cohort_definition(PhenotypeDefinition.load(FIXTURE))
    eligible = CohortMembership(
        patient_key="patient_eligible00000001",
        state=MembershipState.MET,
        criteria=(
            _criterion("has-diabetes", MembershipState.MET, "d"),
            _criterion("has-metformin", MembershipState.MET, "m"),
        ),
    )
    unresolved = CohortMembership(
        patient_key="patient_unresolved000001",
        state=MembershipState.UNKNOWN,
        criteria=(
            _criterion("has-diabetes", MembershipState.MET, "e"),
            _criterion("has-metformin", MembershipState.UNKNOWN, "u"),
        ),
    )
    result = build_cohort_execution(
        definition,
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_registrysource01",
            digest="sha256:" + "1" * 64,
            schema_version="journey-snapshot-v1",
            license_tags=("synthetic",),
        ),
        vocabulary_digest="sha256:" + "2" * 64,
        policy_digest="sha256:" + "3" * 64,
        evaluator_version="registry-test-1.0",
        memberships=(eligible, unresolved),
    )
    assert result.value is not None
    return definition, result.value, eligible


def _definition():
    cohort_definition, _, _ = _execution()
    definition = RegistryDefinition(
        registry_id="registry_syntheticregistry1",
        cohort_definition_version_id=cohort_definition.version_id or "",
        cohort_definition_digest=cohort_definition.definition_digest,
        fields=(
            RegistryFieldRule(
                field_id="condition",
                fact_type="condition",
                required=True,
                allowed_statuses=("active", "corrected"),
            ),
            RegistryFieldRule(
                field_id="medication",
                fact_type="medication",
                required=True,
                allowed_statuses=("active", "corrected"),
            ),
            RegistryFieldRule(
                field_id="laboratory",
                fact_type="laboratory",
                required=False,
                allowed_statuses=("final",),
            ),
        ),
        workflow=RegistryWorkflowPolicy(
            policy_id="registry_governance",
            version="1.0.0",
            owner_scope_id="ownerscope_syntheticowner01",
            privacy_policy_digest="sha256:" + "4" * 64,
            export_policy_digest="sha256:" + "5" * 64,
        ),
        definition_version="1.0.0",
    )
    return version_registry_definition(definition)


def _fact(
    suffix: str,
    *,
    fact_type: str,
    status: str = "active",
    value: str | None = None,
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=f"fact_{suffix * 16}",
        subject_id="patient_eligible00000001",
        fact_type=fact_type,
        value=value or f"sensitive-{suffix}-value",
        status=status,
        evidence_ids=(f"evidence_{suffix * 16}",),
        derivation_hash=canonical_digest({"synthetic_suffix": suffix}),
    )


def _materialize(*facts: RegistryFactBinding | ClinicalFact, packets=()):
    definition = _definition()
    _, execution, _ = _execution()
    result = materialize_registry_cases(
        definition,
        execution,
        facts,
        created_at=CREATED,
        review_packets=packets,
    )
    assert result.value is not None
    return definition, result


def _assignment_authorization(definition) -> RegistryAssignmentAuthorization:
    return RegistryAssignmentAuthorization(
        authorization_id="authorization_syntheticassign01",
        owner_scope_id=definition.definition.workflow.owner_scope_id,
        definition_version_id=definition.version_id or "",
        workflow_policy_digest=definition.definition.workflow.digest,
        queue_id="registry_review_queue",
        assignment_approved=True,
    )


def _export_authorization(definition) -> RegistryExportAuthorization:
    workflow = definition.definition.workflow
    return RegistryExportAuthorization(
        authorization_id="authorization_syntheticexport01",
        definition_version_id=definition.version_id or "",
        privacy_policy_digest=workflow.privacy_policy_digest,
        export_policy_digest=workflow.export_policy_digest,
        export_approved=True,
    )


def test_definition_and_case_are_deterministic_value_free_and_schema_valid() -> None:
    condition = _fact("a", fact_type="condition")
    medication = _fact("b", fact_type="medication")
    first_definition, first = _materialize(condition, medication)
    second_definition, second = _materialize(medication, condition)

    assert first.state is StoreState.SUCCESS
    assert second.state is StoreState.SUCCESS
    assert first_definition.to_json() == second_definition.to_json()
    assert first.value is not None and second.value is not None
    assert first.value.to_json() == second.value.to_json()
    case = first.value.cases[0]
    assert case.state is RegistryCaseState.EXPORT_READY
    assert case.completion_state == "complete"
    assert first.value.excluded_membership_counts == {"unknown": 1}
    serialized = first.value.to_json()
    assert "sensitive-a-value" not in serialized
    assert "sensitive-b-value" not in serialized

    restored_definition = type(first_definition).from_json(first_definition.to_json())
    restored_case = RegistryCase.from_json(case.to_json())
    assert restored_definition == first_definition
    assert restored_case == case

    schema = load_registry_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    for payload in (
        first_definition.to_dict(),
        case.to_dict(),
        first.value.to_dict(),
    ):
        assert not tuple(validator(schema).iter_errors(payload))


@given(st.permutations(("a", "b", "c")))
def test_fact_order_cannot_change_case_identity(order: list[str]) -> None:
    facts = tuple(
        _fact(item, fact_type="condition", value="same-synthetic-value")
        for item in order
    )
    _, first = _materialize(*facts, _fact("m", fact_type="medication"))
    _, second = _materialize(*reversed(facts), _fact("m", fact_type="medication"))
    assert first.value is not None and second.value is not None
    assert first.value.cases[0].to_json() == second.value.cases[0].to_json()


def test_required_unknown_conflict_corrected_and_optional_states_stay_distinct() -> (
    None
):
    corrected = RegistryFactBinding(
        fact=_fact("c", fact_type="condition", status="corrected"),
        corrected_from_fact_ids=("fact_aaaaaaaaaaaaaaaa",),
    )
    unknown = _fact("u", fact_type="medication", status="uncertain")
    _, corrected_result = _materialize(corrected, unknown)
    assert corrected_result.value is not None
    states = {
        item.field_id: item.state for item in corrected_result.value.cases[0].fields
    }
    assert states == {
        "condition": RegistryFieldState.CORRECTED,
        "laboratory": RegistryFieldState.NOT_APPLICABLE,
        "medication": RegistryFieldState.UNKNOWN,
    }

    conflict_a = _fact("a", fact_type="condition", value="first-sensitive-value")
    conflict_b = _fact("b", fact_type="condition", value="second-sensitive-value")
    _, conflict_result = _materialize(conflict_a, conflict_b)
    assert conflict_result.value is not None
    states = {
        item.field_id: item.state for item in conflict_result.value.cases[0].fields
    }
    assert states["condition"] is RegistryFieldState.CONFLICT
    assert states["medication"] is RegistryFieldState.MISSING_REQUIRED


def test_existing_review_packets_and_counts_only_sla_are_reused() -> None:
    first = _fact("a", fact_type="condition", value="first-sensitive-value")
    second = _fact("b", fact_type="condition", value="second-sensitive-value")
    packet = ClinicalReviewPacket(
        packet_id="reviewpacket_registrypacket01",
        conflict_id="conflict_registryconflict1",
        fact_ids=(first.fact_id, second.fact_id),
        state="queued",
        priority="high",
        created_at="2026-01-02T01:00:00Z",
        expires_at="2026-01-02T05:00:00Z",
        policy_id="registry_governance",
        policy_version="1.0.0",
        provenance_fingerprint="sha256:" + "8" * 64,
    )
    _, result = _materialize(first, second, packets=(packet,))

    assert result.value is not None
    case = result.value.cases[0]
    assert case.review_packets == (packet,)
    assert result.value.review_summary.total == 1
    assert result.value.review_summary.priority_counts["high"] == 1


def test_assignment_review_and_adjudication_gates_cannot_be_skipped() -> None:
    first = _fact("a", fact_type="condition", value="first-sensitive-value")
    second = _fact("b", fact_type="condition", value="second-sensitive-value")
    medication = _fact("m", fact_type="medication")
    definition, materialized = _materialize(first, second, medication)
    assert materialized.value is not None
    case = materialized.value.cases[0]
    assert case.state is RegistryCaseState.REVIEW_REQUIRED

    skipped_review = begin_registry_review(case, definition, occurred_at=CREATED)
    skipped_adjudication = adjudicate_registry_case(
        case,
        definition,
        approved=True,
        occurred_at=CREATED,
        reason_code="approved",
        decision_digest="sha256:" + "9" * 64,
    )
    blocked_export = build_registry_export(
        definition,
        (case,),
        authorization=_export_authorization(definition),
        created_at=CREATED,
    )
    assert skipped_review.state is StoreState.CONFLICT
    assert skipped_adjudication.state is StoreState.CONFLICT
    assert blocked_export.state is StoreState.CONFLICT

    assigned = assign_registry_case(
        case,
        definition,
        authorization=_assignment_authorization(definition),
        assigned_at="2026-01-02T03:05:00Z",
    )
    assert assigned.value is not None
    reviewing = begin_registry_review(
        assigned.value,
        definition,
        occurred_at="2026-01-02T03:06:00Z",
    )
    assert reviewing.value is not None
    reviewed = complete_registry_review(
        reviewing.value,
        definition,
        approved=True,
        occurred_at="2026-01-02T03:07:00Z",
        reason_code="review_complete",
    )
    assert reviewed.value is not None
    assert reviewed.value.state is RegistryCaseState.ADJUDICATION_REQUIRED
    assert (
        build_registry_export(
            definition,
            (reviewed.value,),
            authorization=_export_authorization(definition),
            created_at="2026-01-02T03:08:00Z",
        ).state
        is StoreState.CONFLICT
    )

    adjudicated = adjudicate_registry_case(
        reviewed.value,
        definition,
        approved=True,
        occurred_at="2026-01-02T03:08:00Z",
        reason_code="adjudication_complete",
        decision_digest="sha256:" + "9" * 64,
    )
    assert adjudicated.value is not None
    assert adjudicated.value.state is RegistryCaseState.EXPORT_READY


def test_serialized_event_history_rejects_a_direct_export_skip() -> None:
    definition, result = _materialize(_fact("a", fact_type="condition"))
    assert result.value is not None
    case = result.value.cases[0]
    invalid_event = RegistryCaseEvent(
        event_id="registryevent_illegaltransition1",
        case_id=case.case_id,
        action="export",
        from_state=case.state,
        to_state=RegistryCaseState.EXPORTED,
        occurred_at="2026-01-02T03:05:00Z",
        reason_code="illegal_skip",
        policy_digest=definition.definition.workflow.digest,
    )
    with pytest.raises(Exception, match="case transition is invalid"):
        replace(
            case,
            state=RegistryCaseState.EXPORTED,
            events=(invalid_event,),
        )


def test_owner_scope_controls_assignment_without_reviewer_identity() -> None:
    definition, result = _materialize(_fact("a", fact_type="condition"))
    assert result.value is not None
    case = result.value.cases[0]
    denied = assign_registry_case(
        case,
        definition,
        authorization=RegistryAssignmentAuthorization(
            authorization_id="authorization_syntheticassign02",
            owner_scope_id="ownerscope_differentowner01",
            definition_version_id=definition.version_id or "",
            workflow_policy_digest=definition.definition.workflow.digest,
            queue_id="registry_review_queue",
            assignment_approved=True,
        ),
        assigned_at="2026-01-02T03:05:00Z",
    )
    assert denied.state is StoreState.CONFLICT
    assert denied.code == "registry_assignment_authorization_conflict"
    assert "reviewer" not in json.dumps(case.to_dict()).casefold()


def test_correction_preserves_prior_digest_and_restarts_review() -> None:
    definition, result = _materialize(_fact("a", fact_type="condition"))
    assert result.value is not None
    case = result.value.cases[0]
    prior = next(item for item in case.fields if item.field_id == "medication")
    replacement = RegistryFieldResult(
        field_id="medication",
        state=RegistryFieldState.CORRECTED,
        evidence=RegistryFieldEvidence(
            fact_ids=("fact_cccccccccccccccc",),
            evidence_ids=("evidence_cccccccccccccccc",),
            value_digests=(canonical_digest("corrected-sensitive-value"),),
            derivation_digests=("sha256:" + "c" * 64,),
            corrected_from_fact_ids=("fact_bbbbbbbbbbbbbbbb",),
        ),
        reason_code="human_correction",
    )
    stale = correct_registry_field(
        case,
        definition,
        replacement=replacement,
        expected_prior_digest="sha256:" + "0" * 64,
        occurred_at="2026-01-02T03:05:00Z",
        reason_code="stale_attempt",
    )
    corrected = correct_registry_field(
        case,
        definition,
        replacement=replacement,
        expected_prior_digest=prior.digest,
        occurred_at="2026-01-02T03:05:00Z",
        reason_code="correction_recorded",
    )
    assert stale.state is StoreState.CONFLICT
    assert corrected.value is not None
    assert corrected.value.state is RegistryCaseState.REVIEW_REQUIRED
    assert corrected.value.fields[2].state is RegistryFieldState.CORRECTED
    assert case.case_digest != corrected.value.case_digest
    assert "corrected-sensitive-value" not in corrected.value.to_json()


def test_export_is_policy_bound_value_free_and_records_exact_case_digest() -> None:
    definition, materialized = _materialize(
        _fact("a", fact_type="condition"),
        _fact("b", fact_type="medication"),
    )
    assert materialized.value is not None
    case = materialized.value.cases[0]
    authorization = _export_authorization(definition)
    conflict = build_registry_export(
        definition,
        (case,),
        authorization=RegistryExportAuthorization(
            authorization_id="authorization_syntheticexport02",
            definition_version_id=definition.version_id or "",
            privacy_policy_digest="sha256:" + "9" * 64,
            export_policy_digest=definition.definition.workflow.export_policy_digest,
            export_approved=True,
        ),
        created_at="2026-01-02T03:05:00Z",
    )
    exported = build_registry_export(
        definition,
        (case,),
        authorization=authorization,
        created_at="2026-01-02T03:05:00Z",
    )
    assert conflict.state is StoreState.CONFLICT
    assert exported.value is not None
    assert exported.value.case_digests == {case.case_id: case.case_digest}
    assert "sensitive" not in exported.value.to_json()
    assert type(exported.value).from_json(exported.value.to_json()) == exported.value

    schema = load_registry_schema()
    assert not tuple(
        validator_for(schema)(schema).iter_errors(exported.value.to_dict())
    )
    marked = mark_registry_case_exported(
        case,
        definition,
        envelope=exported.value,
        occurred_at="2026-01-02T03:06:00Z",
    )
    assert marked.value is not None
    assert marked.value.state is RegistryCaseState.EXPORTED
    assert marked.value.events[-1].artifact_digest == exported.value.manifest_digest


def test_definition_or_cohort_drift_fails_closed() -> None:
    definition = _definition()
    _, execution, _ = _execution()
    drifted_definition = version_registry_definition(
        RegistryDefinition(
            registry_id=definition.definition.registry_id,
            cohort_definition_version_id="cohortdefinition_different0000001",
            cohort_definition_digest=definition.definition.cohort_definition_digest,
            fields=definition.definition.fields,
            workflow=definition.definition.workflow,
            definition_version="2.0.0",
        )
    )
    result = materialize_registry_cases(
        drifted_definition,
        execution,
        (),
        created_at=CREATED,
    )
    assert result.state is StoreState.CONFLICT
    assert result.code == "registry_cohort_definition_conflict"
