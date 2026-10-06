"""Deterministic synthetic evidence for the v3.1 agent release gates."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from openmed.eval.suites.agent_release import (
    APPROVAL_BYPASS,
    CLINICIAN_REVIEW_AGREEMENT,
    EVIDENCE_COMPLETENESS,
    RECOVERY_CORRECTNESS,
    REFERENCE_SERVER_COVERAGE,
    REPLAY_EQUIVALENCE,
    UNAUTHORIZED_ACTION_ESCAPE,
    UNSAFE_SIDE_EFFECT,
    WORKFLOW_LATENCY,
    MetricEvidence,
    SliceEvidence,
    evaluate_agent_release_gates,
)


def digest(label: str) -> str:
    """Return a stable SHA-256 reference for synthetic fixture evidence."""
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _count_evidence(
    metric: str,
    *,
    event_count: int,
    sample_size: int,
    slice_counts: tuple[tuple[str, int, int], ...],
    limitation: str,
) -> MetricEvidence:
    return MetricEvidence(
        metric=metric,
        value=event_count / sample_size,
        event_count=event_count,
        sample_size=sample_size,
        evidence_digest=digest(metric),
        slices=tuple(
            SliceEvidence(
                slice_ref=slice_ref,
                value=count / size,
                event_count=count,
                sample_size=size,
            )
            for slice_ref, count, size in slice_counts
        ),
        limitations=(limitation,),
    )


def passing_evidence() -> tuple[MetricEvidence, ...]:
    """Return a fully synthetic baseline that clears every published gate."""
    return (
        _count_evidence(
            UNAUTHORIZED_ACTION_ESCAPE,
            event_count=0,
            sample_size=1_000,
            slice_counts=(("read_actions", 0, 500), ("write_actions", 0, 500)),
            limitation="synthetic_policy_matrix",
        ),
        _count_evidence(
            APPROVAL_BYPASS,
            event_count=0,
            sample_size=500,
            slice_counts=(("approval_required", 0, 500),),
            limitation="synthetic_approval_challenges",
        ),
        _count_evidence(
            UNSAFE_SIDE_EFFECT,
            event_count=0,
            sample_size=500,
            slice_counts=(("mutating_tools", 0, 500),),
            limitation="synthetic_side_effects",
        ),
        _count_evidence(
            REPLAY_EQUIVALENCE,
            event_count=1_000,
            sample_size=1_000,
            slice_counts=(("clean_replay", 500, 500), ("resumed_replay", 500, 500)),
            limitation="deterministic_backends_only",
        ),
        _count_evidence(
            RECOVERY_CORRECTNESS,
            event_count=996,
            sample_size=1_000,
            slice_counts=(("process_restart", 498, 500), ("tool_retry", 498, 500)),
            limitation="synthetic_fault_injection",
        ),
        _count_evidence(
            EVIDENCE_COMPLETENESS,
            event_count=990,
            sample_size=1_000,
            slice_counts=(("successful_runs", 495, 500), ("abstained_runs", 495, 500)),
            limitation="synthetic_workflow_records",
        ),
        _count_evidence(
            REFERENCE_SERVER_COVERAGE,
            event_count=98,
            sample_size=100,
            slice_counts=(
                ("read_interactions", 49, 50),
                ("write_interactions", 49, 50),
            ),
            limitation="declared_reference_profiles",
        ),
        MetricEvidence(
            metric=WORKFLOW_LATENCY,
            value=850.0,
            sample_size=400,
            ci_lower=810.0,
            ci_upper=910.0,
            evidence_digest=digest(WORKFLOW_LATENCY),
            slices=(
                SliceEvidence(
                    "read_workflows", 700.0, 200, ci_lower=660.0, ci_upper=760.0
                ),
                SliceEvidence(
                    "write_workflows", 990.0, 200, ci_lower=920.0, ci_upper=1_080.0
                ),
            ),
            limitations=("single_local_runtime_profile",),
        ),
        MetricEvidence(
            metric=CLINICIAN_REVIEW_AGREEMENT,
            value=0.90,
            sample_size=120,
            ci_lower=0.84,
            ci_upper=0.94,
            evidence_digest=digest(CLINICIAN_REVIEW_AGREEMENT),
            slices=(
                SliceEvidence(
                    "evidence_grounding", 0.91, 60, ci_lower=0.82, ci_upper=0.96
                ),
                SliceEvidence(
                    "workflow_safety", 0.89, 60, ci_lower=0.81, ci_upper=0.95
                ),
            ),
            limitations=("synthetic_case_review",),
        ),
    )


def critical_failure_evidence() -> tuple[MetricEvidence, ...]:
    """Return the baseline with one unauthorized-action escape injected."""
    evidence = list(passing_evidence())
    original = evidence[0]
    evidence[0] = replace(
        original,
        value=0.001,
        event_count=1,
        slices=(
            original.slices[0],
            replace(original.slices[1], value=0.002, event_count=1),
        ),
    )
    return tuple(evidence)


CANDIDATE_DIGEST = digest("synthetic-v3.1-release-candidate")


def passing_report():
    """Evaluate and return the deterministic passing baseline report."""
    return evaluate_agent_release_gates(
        passing_evidence(), candidate_digest=CANDIDATE_DIGEST
    )


def critical_failure_report():
    """Evaluate and return the deterministic critical-failure report."""
    return evaluate_agent_release_gates(
        critical_failure_evidence(), candidate_digest=CANDIDATE_DIGEST
    )


def executed_adversarial_report():
    """Execute the existing synthetic corpus through a conforming boundary."""
    from openmed.agent.security.adversarial import (
        DEFAULT_ADVERSARIAL_FIXTURES,
        AttackClass,
        BoundaryVerdict,
        run_adversarial_suite,
    )

    reasons = {
        fixture.attack_class: fixture.expected_reason_code
        for fixture in DEFAULT_ADVERSARIAL_FIXTURES
    }

    def boundary(attempt, dispatch):
        if attempt.attack_class is AttackClass.BENIGN_CONTROL:
            dispatch()
            return BoundaryVerdict.allow()
        return BoundaryVerdict.deny(reasons[attempt.attack_class])

    return run_adversarial_suite(boundary)


def executed_recovery_cases():
    """Execute synthetic absent-effect recovery in each checkpoint phase.

    Golden expectations are declared independently of recover_workflow outputs.
    """
    from openmed.agent.correlation import ActionId, RunId
    from openmed.agent.identifiers import ToolId, WorkflowId
    from openmed.agent.workflows.recovery import (
        CompensationLimit,
        EffectKind,
        EffectObservation,
        EffectRecord,
        EffectState,
        ObservationState,
        RecoveryCheckpoint,
        RecoveryDecision,
        RecoveryDisposition,
        RecoveryPhase,
        RecoveryReason,
        recover_workflow,
    )
    from openmed.eval.suites.agent_release_adapters import RecoveryEvidenceCase

    cases = []
    for index, phase in enumerate(RecoveryPhase):
        run_id = RunId(f"run_{index + 1:032x}")
        effect = EffectRecord.create(
            ordinal=0,
            run_id=run_id,
            action_id=ActionId(f"act_{index + 1:032x}"),
            tool_id=ToolId("tool:org.openmed/recovery-fixture@1.0.0"),
            kind=EffectKind.LOCAL_TOOL,
            operation_digest=digest("operation"),
            approval_required=True,
            compensation_limit=CompensationLimit.NONE,
        )
        terminal = phase is RecoveryPhase.COMPLETED
        if terminal:
            effect = replace(
                effect,
                state=EffectState.COMMITTED,
                commit_evidence_digest=digest("commit"),
            )
        approved = phase is not RecoveryPhase.PLANNED
        checkpoint = RecoveryCheckpoint.create(
            workflow_id=WorkflowId("workflow:org.openmed/recovery-fixture@1.0.0"),
            run_id=run_id,
            sequence=0,
            phase=phase,
            plan_digest=digest("plan"),
            effects=(effect,),
            approval_action_digest=digest("plan") if approved else None,
            approval_receipt_digest=digest("receipt") if approved else None,
            approval_expires_at=100 if approved else None,
        )
        observation = EffectObservation(
            action_id=effect.action_id,
            operation_digest=effect.operation_digest,
            idempotency_key=effect.idempotency_key,
            state=ObservationState.COMMITTED if terminal else ObservationState.ABSENT,
            commit_evidence_digest=digest("commit") if terminal else None,
        )
        if phase is RecoveryPhase.PLANNED:
            disposition, reason = (
                RecoveryDisposition.REVIEW_REQUIRED,
                RecoveryReason.APPROVAL_MISSING,
            )
        elif phase is RecoveryPhase.ABORTED:
            disposition, reason = (
                RecoveryDisposition.REVIEW_REQUIRED,
                RecoveryReason.WORKFLOW_ABORTED,
            )
        elif terminal:
            disposition, reason = (
                RecoveryDisposition.COMPLETE,
                RecoveryReason.ALREADY_COMPLETE,
            )
        else:
            disposition, reason = (
                RecoveryDisposition.RESUME,
                RecoveryReason.SAFE_TO_RESUME,
            )
        retry = disposition is RecoveryDisposition.RESUME
        expected = RecoveryDecision.create(
            disposition=disposition,
            reason=reason,
            source_checkpoint_digest=checkpoint.checkpoint_digest,
            retry_effect_ids=(effect.action_id.serialize(),) if retry else (),
            retry_idempotency_keys=(effect.idempotency_key,) if retry else (),
        )
        cases.append(
            RecoveryEvidenceCase(
                digest("recovery_" + phase.value),
                checkpoint,
                recover_workflow((checkpoint,), (observation,), now=10),
                expected,
            )
        )
    return tuple(cases)
