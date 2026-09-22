from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent import OutcomeClass, WorkflowOutcome
from openmed.agent.run_summary import RunEvent, RunSummary
from openmed.agent.workflow_rollup import (
    WORKFLOW_ROLLUP_SCHEMA_VERSION,
    WorkflowRollup,
    WorkflowRollupEntry,
    WorkflowRollupError,
    rollup_workflows,
)

_REASONS = {
    OutcomeClass.SUCCESS: "completed",
    OutcomeClass.ABSTAINED: "insufficient_evidence",
    OutcomeClass.REVIEW_REQUIRED: "human_gate",
    OutcomeClass.POLICY_DENIED: "phi_policy",
    OutcomeClass.FAILED: "tool_error",
}
_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64

_OUTCOME_NAMES = ("abstained", "failed", "policy_denied", "review_required", "success")


def _zero_counts(**updates: int) -> dict[str, int]:
    counts = {name: 0 for name in _OUTCOME_NAMES}
    counts.update(updates)
    return counts


def _event(
    workflow_id: str,
    outcome_class: OutcomeClass = OutcomeClass.SUCCESS,
    *,
    tool_calls: int = 0,
    duration: float = 0.0,
    digests: tuple[str, ...] = (),
) -> RunEvent:
    return RunEvent(
        workflow_id=workflow_id,
        outcome=WorkflowOutcome(outcome_class, _REASONS[outcome_class]),
        tool_call_count=tool_calls,
        duration_seconds=duration,
        artifact_digests=digests,
    )


def _mixed_events() -> list[RunEvent]:
    return [
        _event("intake", tool_calls=3, duration=1.5, digests=(_DIGEST_A,)),
        _event("intake", tool_calls=1, duration=0.5, digests=(_DIGEST_A, _DIGEST_B)),
        _event(
            "review",
            OutcomeClass.REVIEW_REQUIRED,
            tool_calls=2,
            duration=0.25,
            digests=(_DIGEST_B,),
        ),
    ]


def test_empty_events_produce_empty_rollup() -> None:
    rollup = rollup_workflows([])

    assert rollup.entries == ()
    assert rollup.to_dict()["run_count"] == 0
    assert rollup.to_dict()["workflow_count"] == 0
    assert rollup.to_dict()["outcome_counts"] == _zero_counts()


def test_single_workflow_totals_match_hand_calculation() -> None:
    rollup = rollup_workflows(
        [_event("intake", tool_calls=3, duration=1.5, digests=(_DIGEST_A,))]
    )

    (entry,) = rollup.entries
    assert entry.workflow_id == "intake"
    assert entry.run_count == 1
    assert dict(entry.outcome_counts) == _zero_counts(success=1)
    assert entry.tool_call_count == 3
    assert entry.duration_seconds == 1.5
    assert entry.artifact_digests == (_DIGEST_A,)


def test_mixed_workflows_produce_hand_checked_totals() -> None:
    rollup = rollup_workflows(_mixed_events())

    assert [entry.workflow_id for entry in rollup.entries] == ["intake", "review"]

    intake, review = rollup.entries
    assert intake.run_count == 2
    assert dict(intake.outcome_counts) == _zero_counts(success=2)
    assert intake.tool_call_count == 4
    assert intake.duration_seconds == 2.0
    assert intake.artifact_digests == (_DIGEST_A, _DIGEST_B)

    assert review.run_count == 1
    assert dict(review.outcome_counts) == _zero_counts(review_required=1)
    assert review.tool_call_count == 2
    assert review.duration_seconds == 0.25
    assert review.artifact_digests == (_DIGEST_B,)


def test_totals_reconcile_exactly_with_run_summary() -> None:
    events = _mixed_events()
    rollup = rollup_workflows(events)
    summary = RunSummary.from_events(events)
    payload = rollup.to_dict()

    assert payload["outcome_counts"] == dict(summary.outcome_counts)
    assert payload["tool_call_count"] == summary.tool_call_count
    assert payload["duration_seconds"] == summary.duration_seconds
    assert payload["artifact_count"] == len(summary.artifact_digests)
    assert payload["run_count"] == len(events)


def test_artifact_digests_are_unique_per_workflow() -> None:
    rollup = rollup_workflows(
        [
            _event("intake", digests=(_DIGEST_A, _DIGEST_B)),
            _event("intake", digests=(_DIGEST_A, _DIGEST_C)),
        ]
    )

    (entry,) = rollup.entries
    assert entry.artifact_digests == (_DIGEST_A, _DIGEST_B, _DIGEST_C)


def test_entries_are_sorted_by_workflow_id_not_input_order() -> None:
    rollup = rollup_workflows([_event("zebra"), _event("apple"), _event("mango")])

    assert [entry.workflow_id for entry in rollup.entries] == [
        "apple",
        "mango",
        "zebra",
    ]


def test_json_is_deterministic_and_ordered() -> None:
    rollup = rollup_workflows(_mixed_events())

    assert rollup.to_json() == rollup_workflows(_mixed_events()).to_json()
    assert list(rollup.to_dict()) == [
        "schema_version",
        "workflow_count",
        "run_count",
        "outcome_counts",
        "tool_call_count",
        "duration_seconds",
        "artifact_count",
        "entries",
    ]
    payload = json.loads(rollup.to_json())
    assert payload["schema_version"] == WORKFLOW_ROLLUP_SCHEMA_VERSION
    assert payload["entries"][0]["workflow_id"] == "intake"


def test_markdown_matches_golden_output() -> None:
    rollup = rollup_workflows(_mixed_events())

    assert rollup.to_markdown() == (
        "# Agent Workflow Rollup\n"
        "\n"
        "| Workflow | Runs | Success | Abstained | Review | Denied | Failed "
        "| Tool calls | Duration (s) | Artifacts |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        "| `intake` | 2 | 2 | 0 | 0 | 0 | 0 | 4 | 2 | 2 |\n"
        "| `review` | 1 | 0 | 0 | 1 | 0 | 0 | 2 | 0.25 | 1 |\n"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "events",
        b"events",
        {"intake": 1},
        [_event("intake"), "not-an-event"],
    ],
)
def test_invalid_iterables_and_items_are_rejected(value: Any) -> None:
    with pytest.raises(WorkflowRollupError):
        rollup_workflows(value)


def _entry_fields(**updates: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "workflow_id": "intake",
        "run_count": 1,
        "outcome_counts": _zero_counts(success=1),
        "tool_call_count": 0,
        "duration_seconds": 0.0,
        "artifact_digests": (),
    }
    fields.update(updates)
    return fields


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"workflow_id": "has spaces"}, "workflow_id: invalid_identifier"),
        ({"workflow_id": 42}, "workflow_id: invalid_identifier"),
        ({"run_count": -1}, "run_count: invalid_count"),
        ({"run_count": "one"}, "run_count: invalid_count"),
        (
            {"outcome_counts": _zero_counts(success=2)},
            "outcome_counts: does_not_match_run_count",
        ),
        (
            {"outcome_counts": {**_zero_counts(success=1), "extra": 0}},
            "outcome_counts: invalid_keys",
        ),
        ({"tool_call_count": -1}, "tool_call_count: invalid_count"),
        ({"duration_seconds": float("nan")}, "duration_seconds: out_of_range"),
        ({"duration_seconds": "fast"}, "duration_seconds: invalid_number"),
        (
            {"artifact_digests": (_DIGEST_A, _DIGEST_A)},
            "artifact_digests: not_sorted_unique",
        ),
        (
            {"artifact_digests": ("not-a-digest",)},
            "artifact_digests: invalid_digest",
        ),
    ],
)
def test_entry_construction_rejects_unsafe_values_without_echo(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(WorkflowRollupError) as exc_info:
        WorkflowRollupEntry(**_entry_fields(**updates))

    assert str(exc_info.value) == message


def test_rollup_construction_rejects_unsorted_or_duplicate_entries() -> None:
    intake = WorkflowRollupEntry(**_entry_fields(workflow_id="intake"))
    review = WorkflowRollupEntry(
        **_entry_fields(
            workflow_id="review", outcome_counts=_zero_counts(review_required=1)
        )
    )

    with pytest.raises(WorkflowRollupError, match="entries: not_sorted"):
        WorkflowRollup(entries=(review, intake))
    with pytest.raises(WorkflowRollupError, match="entries: duplicate_workflow_id"):
        WorkflowRollup(entries=(intake, intake))
    with pytest.raises(
        WorkflowRollupError, match="schema_version: unsupported_version"
    ):
        WorkflowRollup(entries=(intake,), schema_version="other")


def test_sentinel_payloads_cannot_enter_output() -> None:
    sentinel = "Synthetic_Patient_Secret_987"

    for unsafe in (
        f"{sentinel} prompt",
        "/home/clinic/notes.txt",
        "Bearer secret-token",
    ):
        with pytest.raises(ValueError):
            rollup_workflows([_event(unsafe)])

    rollup = rollup_workflows(_mixed_events())
    rendered = rollup.to_json() + rollup.to_markdown()
    assert sentinel not in rendered


def test_rollup_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.rollup_workflows is rollup_workflows
    assert agent.WorkflowRollup is WorkflowRollup
    assert agent.WorkflowRollupEntry is WorkflowRollupEntry
    assert agent.WorkflowRollupError is WorkflowRollupError
    assert agent.WORKFLOW_ROLLUP_SCHEMA_VERSION == WORKFLOW_ROLLUP_SCHEMA_VERSION
