"""End-of-run checks on synthetic, content-free agent metadata."""

from __future__ import annotations

import traceback
from dataclasses import replace

import pytest

from openmed.agent.action_graph import ActionNode
from openmed.agent.artifact_reference import ArtifactKind, ArtifactReference
from openmed.agent.correlation import ActionCorrelation, ActionId, RunId
from openmed.agent.errors import ErrorEnvelope, ErrorStage
from openmed.agent.event_sequence import EventReference
from openmed.agent.outcomes import OutcomeClass, WorkflowOutcome
from openmed.agent.run_invariants import (
    CompletedRun,
    RunAction,
    RunInvariantError,
    RunInvariantReport,
    check_run_invariants,
)
from openmed.agent.timing import ActionTiming, RunTiming

RUN = RunId("run_" + "1" * 32)
OTHER_RUN = RunId("run_" + "2" * 32)
ACTION = ActionId("act_" + "3" * 32)
OTHER_ACTION = ActionId("act_" + "4" * 32)
ARTIFACT = "art_" + "5" * 32
TOOL = "tool:openmed.agent/summarize"


def outcome(kind: OutcomeClass) -> WorkflowOutcome:
    """Choose a documented reason for each synthetic outcome class."""

    reason = {
        OutcomeClass.SUCCESS: "completed",
        OutcomeClass.ABSTAINED: "insufficient_evidence",
        OutcomeClass.REVIEW_REQUIRED: "human_gate",
        OutcomeClass.POLICY_DENIED: "consent_required",
        OutcomeClass.FAILED: "tool_error",
    }
    return WorkflowOutcome(kind, reason[kind])


def error(*, action: bool = False) -> ErrorEnvelope:
    """Create a categorical failure tied to the run or its action."""

    return ErrorEnvelope.for_code(
        "provider_failed",
        stage=ErrorStage.TOOL_CALL,
        run_id=RUN.serialize(),
        action_id=ACTION.serialize() if action else None,
    )


def sample(
    kind: OutcomeClass = OutcomeClass.SUCCESS,
    *,
    phase: str = "completed",
    artifact: bool = False,
    reviewed: bool = False,
) -> CompletedRun:
    """Build one consistent, entirely synthetic completed run."""

    action = RunAction(
        ActionCorrelation(RUN, ACTION),
        phase,
        outcome(kind),
        error(action=True) if kind is OutcomeClass.FAILED else None,
        (ARTIFACT,) if artifact else (),
        reviewed,
    )
    reference = ArtifactReference(
        ARTIFACT,
        ArtifactKind.EVIDENCE,
        "openmed.agent.evidence.v1",
        "a" * 64,
        12,
    )
    return CompletedRun(
        run_id=RUN,
        events=(EventReference(RUN.serialize(), "event0", 0),),
        terminal_sequence_number=0,
        graph=(ActionNode(ACTION.serialize(), TOOL),),
        actions=(action,),
        timing=RunTiming(0, 3, correlation_id=RUN.serialize()),
        action_timings=(ActionTiming(ACTION.serialize(), 1, 2),),
        outcome=outcome(kind),
        error=error() if kind is OutcomeClass.FAILED else None,
        artifacts=(reference,) if artifact else (),
    )


@pytest.mark.parametrize(
    ("kind", "phase"),
    [
        (OutcomeClass.SUCCESS, "completed"),
        (OutcomeClass.ABSTAINED, "aborted"),
        (OutcomeClass.REVIEW_REQUIRED, "aborted"),
        (OutcomeClass.POLICY_DENIED, "aborted"),
        (OutcomeClass.FAILED, "aborted"),
    ],
)
def test_clean_end_states_pass(kind: OutcomeClass, phase: str) -> None:
    run = sample(kind, phase=phase)
    if kind is OutcomeClass.POLICY_DENIED:
        denial = ErrorEnvelope.for_code(
            "policy_denied", stage=ErrorStage.TOOL_CALL, run_id=RUN.serialize()
        )
        action_denial = ErrorEnvelope.for_code(
            "policy_denied",
            stage=ErrorStage.TOOL_CALL,
            run_id=RUN.serialize(),
            action_id=ACTION.serialize(),
        )
        run = replace(
            run,
            error=denial,
            actions=(replace(run.actions[0], error=action_denial),),
        )
    report = check_run_invariants(run)
    assert report.is_valid
    assert report.to_json() == (
        '{"findings":[],"schema_version":"openmed.agent.run_invariants.v1"}'
    )


def test_reviewed_completion_uses_final_metadata_without_claiming_review_proof() -> (
    None
):
    assert check_run_invariants(sample(reviewed=True)).is_valid


def test_graph_orphan_cycle_and_event_failures_are_independent() -> None:
    run = sample()
    corrupt = replace(
        run,
        graph=(
            ActionNode(ACTION.serialize(), TOOL, (OTHER_ACTION.serialize(),)),
            ActionNode(OTHER_ACTION.serialize(), TOOL, (ACTION.serialize(),)),
        ),
        events=(
            EventReference(RUN.serialize(), "event0", 0),
            EventReference(RUN.serialize(), "event2", 2),
            EventReference(RUN.serialize(), "event3", 3),
        ),
        terminal_sequence_number=2,
    )
    codes = set(check_run_invariants(corrupt).findings)
    assert {
        "graph_dependency_cycle",
        "missing_action",
        "event_sequence_gap",
        "event_post_terminal_event",
    } <= codes
    assert "orphan_action" in check_run_invariants(replace(run, graph=())).findings


def test_unfinished_missing_outcomes_and_error_conflicts_are_independent() -> None:
    run = sample()
    action = replace(
        run.actions[0],
        phase="running",
        outcome=None,
        error=error(action=True),
    )
    codes = set(
        check_run_invariants(replace(run, actions=(action,), outcome=None)).findings
    )
    assert {
        "unfinished_action",
        "missing_action_outcome",
        "missing_run_outcome",
    } <= codes
    assert (
        "error_outcome_conflict"
        in check_run_invariants(replace(run, error=error())).findings
    )
    assert (
        "missing_error"
        in check_run_invariants(
            replace(run, outcome=outcome(OutcomeClass.FAILED))
        ).findings
    )


def test_timing_and_correlation_fail_independently() -> None:
    run = sample()
    wrong = replace(
        run,
        actions=(
            replace(run.actions[0], correlation=ActionCorrelation(OTHER_RUN, ACTION)),
        ),
        action_timings=(ActionTiming(ACTION.serialize(), 0, 9),),
    )
    assert {"cross_run_action", "invalid_timing"} <= set(
        check_run_invariants(wrong).findings
    )
    assert (
        "timing_run_mismatch"
        in check_run_invariants(replace(run, timing=RunTiming(0, 3))).findings
    )


def test_orphan_parent_and_duplicate_actions_are_reported() -> None:
    run = sample()
    orphan = replace(
        run.actions[0],
        correlation=ActionCorrelation(RUN, ACTION, OTHER_ACTION),
    )
    duplicate = replace(run, actions=(orphan, orphan))
    assert {"orphan_parent", "duplicate_action"} <= set(
        check_run_invariants(duplicate).findings
    )


def test_artifacts_need_declared_successful_unique_producers() -> None:
    run = sample(artifact=True)
    assert check_run_invariants(run).is_valid
    unsupported = replace(run, actions=(replace(run.actions[0], artifact_ids=()),))
    assert "unsupported_artifact" in check_run_invariants(unsupported).findings
    unsuccessful = replace(
        run,
        actions=(replace(run.actions[0], phase="aborted"),),
    )
    assert (
        "artifact_from_unsuccessful_action"
        in check_run_invariants(unsuccessful).findings
    )
    duplicate = replace(
        run, actions=(replace(run.actions[0], artifact_ids=(ARTIFACT, ARTIFACT)),)
    )
    assert "duplicate_artifact_production" in check_run_invariants(duplicate).findings


def test_sentinel_never_reaches_reports_or_exceptions() -> None:
    sentinel = (
        "prompt arguments tool-output clinical-text credential "
        "/private/chart traceback PATIENT-SENTINEL"
    )
    run = sample()
    with pytest.raises(RunInvariantError) as caught:
        RunAction(run.actions[0].correlation, sentinel)
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(vars(caught.value))
    assert sentinel not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    with pytest.raises(RunInvariantError) as report_error:
        RunInvariantReport((sentinel,))
    assert sentinel not in str(report_error.value)
    report = check_run_invariants(replace(run, outcome=None))
    assert sentinel not in repr(report)
    assert sentinel not in report.to_json()


def test_report_is_byte_stable_for_identical_evidence() -> None:
    run = replace(sample(), outcome=None)
    assert check_run_invariants(run).to_json() == check_run_invariants(run).to_json()


def test_checker_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.check_run_invariants is check_run_invariants
    assert agent.CompletedRun is CompletedRun
    assert agent.RunInvariantReport is RunInvariantReport
