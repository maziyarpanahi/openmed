from __future__ import annotations

import json

import pytest

from openmed.agent.run_summary import (
    RunEvent,
    RunSummary,
    RunSummaryError,
    RunSummaryPrivacyError,
)


def test_run_event_accepts_safe_metadata() -> None:
    event = RunEvent(
        workflow_id="clinical-review",
        outcome="success",
        tool_call_count=3,
        duration_seconds=1.25,
        artifact_digests=("sha256:" + "a" * 64,),
    )

    assert event.workflow_id == "clinical-review"
    assert event.outcome == "success"
    assert event.tool_call_count == 3
    assert event.duration_seconds == 1.25
    assert event.artifact_digests == ("sha256:" + "a" * 64,)


@pytest.mark.parametrize("outcome", ["success", "failure", "abstained"])
def test_run_event_accepts_allowed_outcomes(outcome: str) -> None:
    event = RunEvent(workflow_id="workflow-1", outcome=outcome)

    assert event.outcome == outcome


def test_run_event_rejects_unsafe_workflow_id() -> None:
    with pytest.raises(RunSummaryError):
        RunEvent(workflow_id="workflow with spaces", outcome="success")


def test_run_event_rejects_unknown_outcome() -> None:
    with pytest.raises(RunSummaryError):
        RunEvent(workflow_id="workflow-1", outcome="unknown")


@pytest.mark.parametrize("tool_call_count", [-1, True, 1.5])
def test_run_event_rejects_invalid_tool_call_count(
    tool_call_count: object,
) -> None:
    with pytest.raises(RunSummaryError):
        RunEvent(
            workflow_id="workflow-1",
            outcome="success",
            tool_call_count=tool_call_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("duration", [-1, float("inf"), float("nan"), True])
def test_run_event_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises(RunSummaryError):
        RunEvent(
            workflow_id="workflow-1",
            outcome="success",
            duration_seconds=duration,  # type: ignore[arg-type]
        )


def test_run_event_rejects_invalid_artifact_digest() -> None:
    with pytest.raises(RunSummaryError):
        RunEvent(
            workflow_id="workflow-1",
            outcome="success",
            artifact_digests=("not-a-digest",),
        )


def test_summary_aggregates_events_deterministically() -> None:
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64

    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="workflow-b",
                outcome="failure",
                tool_call_count=2,
                duration_seconds=2.5,
                artifact_digests=(digest_b,),
            ),
            RunEvent(
                workflow_id="workflow-a",
                outcome="success",
                tool_call_count=3,
                duration_seconds=1.5,
                artifact_digests=(digest_a, digest_b),
            ),
            RunEvent(
                workflow_id="workflow-a",
                outcome="abstained",
                tool_call_count=1,
                duration_seconds=0.5,
                artifact_digests=(digest_a,),
            ),
        ]
    )

    assert summary.workflow_ids == ("workflow-a", "workflow-b")
    assert summary.outcome_counts == {
        "abstained": 1,
        "failure": 1,
        "success": 1,
    }
    assert summary.tool_call_count == 6
    assert summary.duration_seconds == 4.5
    assert summary.artifact_digests == (digest_a, digest_b)


def test_empty_events_produce_zero_summary() -> None:
    summary = RunSummary.from_events([])

    assert summary.workflow_ids == ()
    assert summary.outcome_counts == {
        "abstained": 0,
        "failure": 0,
        "success": 0,
    }
    assert summary.tool_call_count == 0
    assert summary.duration_seconds == 0.0
    assert summary.artifact_digests == ()


def test_from_events_rejects_non_run_event() -> None:
    with pytest.raises(RunSummaryError):
        RunSummary.from_events([object()])  # type: ignore[list-item]


def test_to_dict_contains_only_safe_metadata() -> None:
    digest = "sha256:" + "c" * 64

    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="workflow-1",
                outcome="success",
                tool_call_count=2,
                duration_seconds=1.0,
                artifact_digests=(digest,),
            )
        ]
    )

    payload = summary.to_dict()

    assert payload == {
        "workflow_ids": ["workflow-1"],
        "outcome_counts": {
            "abstained": 0,
            "failure": 0,
            "success": 1,
        },
        "tool_call_count": 2,
        "duration_seconds": 1.0,
        "artifact_digests": [digest],
    }

    json.dumps(payload)


def test_to_json_is_deterministic() -> None:
    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="workflow-2",
                outcome="success",
                tool_call_count=1,
                duration_seconds=2.0,
            ),
            RunEvent(
                workflow_id="workflow-1",
                outcome="failure",
                tool_call_count=2,
                duration_seconds=1.0,
            ),
        ]
    )

    expected = (
        '{"artifact_digests":[],"duration_seconds":3.0,'
        '"outcome_counts":{"abstained":0,"failure":1,"success":1},'
        '"tool_call_count":3,"workflow_ids":["workflow-1","workflow-2"]}'
    )

    assert summary.to_json() == expected


def test_markdown_contains_metadata_only() -> None:
    digest = "sha256:" + "d" * 64

    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="clinical-review",
                outcome="abstained",
                tool_call_count=4,
                duration_seconds=3.5,
                artifact_digests=(digest,),
            )
        ]
    )

    markdown = summary.to_markdown()

    assert "# Agent Run Summary" in markdown
    assert "| `clinical-review` |" in markdown
    assert "| `abstained` | 1 |" in markdown
    assert "| Tool calls | 4 |" in markdown
    assert "| Duration (seconds) | 3.5 |" in markdown
    assert digest in markdown


def test_summary_deduplicates_and_sorts_metadata() -> None:
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64

    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="z-workflow",
                outcome="success",
                artifact_digests=(digest_b, digest_a),
            ),
            RunEvent(
                workflow_id="a-workflow",
                outcome="success",
                artifact_digests=(digest_a, digest_b),
            ),
        ]
    )

    assert summary.workflow_ids == ("a-workflow", "z-workflow")
    assert summary.artifact_digests == (digest_a, digest_b)


def test_summary_does_not_echo_sensitive_event_payloads() -> None:
    sensitive_prompt = "patient diagnosis: secret synthetic PHI"
    sensitive_tool_output = "hidden clinical evidence"

    event = RunEvent(
        workflow_id="clinical-review",
        outcome="success",
    )

    summary = RunSummary.from_events([event])

    serialized = json.dumps(summary.to_dict(), sort_keys=True)
    markdown = summary.to_markdown()

    assert sensitive_prompt not in serialized
    assert sensitive_prompt not in markdown
    assert sensitive_tool_output not in serialized
    assert sensitive_tool_output not in markdown


def test_privacy_guard_rejects_arbitrary_text_values() -> None:
    with pytest.raises(RunSummaryPrivacyError):
        from openmed.agent.run_summary import _assert_safe_payload

        _assert_safe_payload({"unsafe": "arbitrary text"})


def test_privacy_guard_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(RunSummaryPrivacyError):
        from openmed.agent.run_summary import _assert_safe_payload

        _assert_safe_payload({1: "value"})


def test_privacy_guard_rejects_non_finite_numbers() -> None:
    with pytest.raises(RunSummaryPrivacyError):
        from openmed.agent.run_summary import _assert_safe_payload

        _assert_safe_payload({"value": float("inf")})