from __future__ import annotations

import itertools
import json

import pytest

import openmed.agent as agent
from openmed.agent import OutcomeClass, WorkflowOutcome
from openmed.agent.run_summary import RunEvent, RunSummary
from openmed.agent.workflow_rollup import (
    WORKFLOW_ROLLUP_SCHEMA_VERSION,
    WorkflowRollup,
    WorkflowRollupError,
    WorkflowRollupRow,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_REASONS = {
    OutcomeClass.SUCCESS: "completed",
    OutcomeClass.ABSTAINED: "insufficient_evidence",
    OutcomeClass.REVIEW_REQUIRED: "human_gate",
    OutcomeClass.POLICY_DENIED: "phi_policy",
    OutcomeClass.FAILED: "tool_error",
}


def _outcome(outcome_class: OutcomeClass = OutcomeClass.SUCCESS) -> WorkflowOutcome:
    return WorkflowOutcome(outcome_class, _REASONS[outcome_class])


def _counts(**updates: int) -> dict[str, int]:
    counts = {
        "abstained": 0,
        "failed": 0,
        "policy_denied": 0,
        "review_required": 0,
        "success": 0,
    }
    counts.update(updates)
    return counts


def test_empty_events_produce_hand_checked_zero_totals() -> None:
    rollup = WorkflowRollup.from_events([])

    assert rollup.workflows == ()
    assert rollup.to_dict() == {
        "schema_version": WORKFLOW_ROLLUP_SCHEMA_VERSION,
        "workflows": [],
        "overall": {
            "outcome_counts": _counts(),
            "tool_call_count": 0,
            "duration_seconds": 0.0,
            "run_count": 0,
            "artifact_digest_count": 0,
        },
    }
    assert "| **Overall** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in (
        rollup.to_markdown()
    )


def test_single_event_produces_hand_checked_workflow_row() -> None:
    rollup = WorkflowRollup.from_events(
        [
            RunEvent(
                "review-workflow",
                _outcome(OutcomeClass.REVIEW_REQUIRED),
                tool_call_count=3,
                duration_seconds=1.25,
                artifact_digests=(_DIGEST_A,),
            )
        ]
    )

    assert rollup.workflows[0].to_dict() == {
        "workflow_id": "review-workflow",
        "outcome_counts": _counts(review_required=1),
        "tool_call_count": 3,
        "duration_seconds": 1.25,
        "run_count": 1,
        "artifact_digest_count": 1,
    }


def test_repeated_workflow_deduplicates_its_artifact_digests() -> None:
    rollup = WorkflowRollup.from_events(
        [
            RunEvent(
                "workflow-a",
                _outcome(),
                tool_call_count=2,
                duration_seconds=0.5,
                artifact_digests=(_DIGEST_A,),
            ),
            RunEvent(
                "workflow-a",
                _outcome(OutcomeClass.ABSTAINED),
                tool_call_count=1,
                duration_seconds=1.5,
                artifact_digests=(_DIGEST_A, _DIGEST_B),
            ),
        ]
    )

    row = rollup.workflows[0]
    assert row.outcome_counts == _counts(abstained=1, success=1)
    assert row.run_count == 2
    assert row.tool_call_count == 3
    assert row.duration_seconds == 2.0
    assert row.artifact_digest_count == 2


def test_mixed_workflows_are_sorted_with_hand_checked_totals() -> None:
    events = [
        RunEvent(
            "workflow-z",
            _outcome(OutcomeClass.FAILED),
            tool_call_count=4,
            duration_seconds=2.0,
            artifact_digests=(_DIGEST_A,),
        ),
        RunEvent(
            "workflow-a",
            _outcome(),
            tool_call_count=2,
            duration_seconds=1.0,
            artifact_digests=(_DIGEST_A, _DIGEST_B),
        ),
        RunEvent(
            "workflow-z",
            _outcome(OutcomeClass.POLICY_DENIED),
            tool_call_count=1,
            duration_seconds=0.25,
        ),
    ]

    rollup = WorkflowRollup.from_events(events)

    assert tuple(row.workflow_id for row in rollup.workflows) == (
        "workflow-a",
        "workflow-z",
    )
    assert rollup.outcome_counts == _counts(failed=1, policy_denied=1, success=1)
    assert rollup.run_count == 3
    assert rollup.tool_call_count == 7
    assert rollup.duration_seconds == 3.25
    # Digests are unique within each workflow, then those row counts reconcile.
    assert rollup.artifact_digest_count == 3


def test_each_overall_total_reconciles_exactly_with_rows() -> None:
    rollup = WorkflowRollup.from_events(
        [
            RunEvent("workflow-b", _outcome(), tool_call_count=2),
            RunEvent(
                "workflow-a",
                _outcome(OutcomeClass.ABSTAINED),
                tool_call_count=3,
                duration_seconds=1.5,
                artifact_digests=(_DIGEST_A,),
            ),
        ]
    )

    assert rollup.run_count == sum(row.run_count for row in rollup.workflows)
    assert rollup.tool_call_count == sum(
        row.tool_call_count for row in rollup.workflows
    )
    assert rollup.duration_seconds == sum(
        row.duration_seconds for row in rollup.workflows
    )
    assert rollup.artifact_digest_count == sum(
        row.artifact_digest_count for row in rollup.workflows
    )
    for outcome in _counts():
        assert rollup.outcome_counts[outcome] == sum(
            row.outcome_counts[outcome] for row in rollup.workflows
        )


def test_json_and_markdown_are_deterministic_in_workflow_id_order() -> None:
    events = [
        RunEvent("workflow-z", _outcome(), duration_seconds=0.1),
        RunEvent("workflow-a", _outcome(), duration_seconds=0.2),
        RunEvent("workflow-z", _outcome(), duration_seconds=0.3),
    ]

    forward = WorkflowRollup.from_events(events)
    reverse = WorkflowRollup.from_events(reversed(events))

    assert forward.to_json() == reverse.to_json()
    assert forward.to_markdown() == reverse.to_markdown()
    assert forward.to_markdown().index("`workflow-a`") < forward.to_markdown().index(
        "`workflow-z`"
    )
    json.loads(forward.to_json())


def test_rollup_accepts_empty_and_single_workflow_summaries() -> None:
    empty = RunSummary.from_events([])
    first = RunSummary.from_events(
        [
            RunEvent(
                "workflow-a",
                _outcome(),
                tool_call_count=2,
                duration_seconds=0.5,
                artifact_digests=(_DIGEST_A,),
            )
        ]
    )
    second = RunSummary.from_events(
        [
            RunEvent(
                "workflow-a",
                _outcome(OutcomeClass.ABSTAINED),
                tool_call_count=1,
                duration_seconds=1.5,
                artifact_digests=(_DIGEST_A, _DIGEST_B),
            )
        ]
    )

    rollup = WorkflowRollup.from_summaries([empty, first, second])

    assert rollup.workflows[0].outcome_counts == _counts(abstained=1, success=1)
    assert rollup.workflows[0].run_count == 2
    assert rollup.workflows[0].tool_call_count == 3
    assert rollup.workflows[0].duration_seconds == 2.0
    assert rollup.workflows[0].artifact_digest_count == 2


def test_multi_workflow_summary_is_rejected_as_ambiguous() -> None:
    summary = RunSummary.from_events(
        [RunEvent("workflow-a", _outcome()), RunEvent("workflow-b", _outcome())]
    )

    with pytest.raises(WorkflowRollupError, match="ambiguous_workflow_ids"):
        WorkflowRollup.from_summaries([summary])


def test_raw_inputs_are_rejected_without_echoing_payload_content() -> None:
    phi = "Jane Synthetic has diagnosis Z99.999"
    bearer = "Bearer synthetic-secret-token"
    raw_event = {
        "workflow_id": "workflow-a",
        "prompt": phi,
        "tool_output": bearer,
    }

    with pytest.raises(WorkflowRollupError) as caught:
        WorkflowRollup.from_events([raw_event])  # type: ignore[list-item]

    assert phi not in str(caught.value)
    assert bearer not in str(caught.value)


def test_rendered_output_omits_artifact_digests() -> None:
    rollup = WorkflowRollup.from_events(
        [RunEvent("workflow-a", _outcome(), artifact_digests=(_DIGEST_A,))]
    )

    assert _DIGEST_A not in rollup.to_json()
    assert _DIGEST_A not in rollup.to_markdown()
    assert rollup.artifact_digest_count == 1


def test_direct_rows_require_reconciled_counts_and_safe_identifiers() -> None:
    with pytest.raises(WorkflowRollupError, match="run_count_mismatch"):
        WorkflowRollupRow(
            workflow_id="workflow-a",
            outcome_counts=_counts(success=1),
            tool_call_count=0,
            duration_seconds=0,
            run_count=2,
            artifact_digest_count=0,
        )
    with pytest.raises(WorkflowRollupError, match="invalid_identifier"):
        WorkflowRollupRow(
            workflow_id="private workflow text",
            outcome_counts=_counts(),
            tool_call_count=0,
            duration_seconds=0,
            run_count=0,
            artifact_digest_count=0,
        )


def test_direct_row_rejects_unbounded_duration_with_stable_error() -> None:
    with pytest.raises(WorkflowRollupError, match="^duration_seconds: out_of_range$"):
        WorkflowRollupRow(
            workflow_id="workflow-a",
            outcome_counts=_counts(),
            tool_call_count=0,
            duration_seconds=10**1000,
            run_count=0,
            artifact_digest_count=0,
        )


def test_inputs_are_bounded() -> None:
    event = RunEvent("workflow-a", _outcome())

    with pytest.raises(WorkflowRollupError, match="events: too_many_items"):
        WorkflowRollup.from_events(itertools.repeat(event, 10_001))


def test_workflow_rollup_is_available_from_public_agent_api() -> None:
    assert agent.WORKFLOW_ROLLUP_SCHEMA_VERSION == WORKFLOW_ROLLUP_SCHEMA_VERSION
    assert agent.WorkflowRollup is WorkflowRollup
    assert agent.WorkflowRollupError is WorkflowRollupError
    assert agent.WorkflowRollupRow is WorkflowRollupRow
