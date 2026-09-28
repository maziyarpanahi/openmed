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
