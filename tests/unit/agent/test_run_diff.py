from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent import OutcomeClass, WorkflowOutcome
from openmed.agent.run_diff import (
    RUN_DIFF_SCHEMA_VERSION,
    RunDiffError,
    RunSummaryDiff,
    diff_run_summaries,
)
from openmed.agent.run_summary import RunEvent, RunSummary

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


def _zero_deltas(**updates: int) -> dict[str, int]:
    deltas = {
        "abstained": 0,
        "failed": 0,
        "policy_denied": 0,
        "review_required": 0,
        "success": 0,
    }
    deltas.update(updates)
    return deltas


@pytest.fixture
def before() -> RunSummary:
    return RunSummary.from_events(
        [
            _event("intake", tool_calls=3, duration=1.5, digests=(_DIGEST_A,)),
            _event(
                "review",
                OutcomeClass.REVIEW_REQUIRED,
                tool_calls=2,
                duration=0.5,
                digests=(_DIGEST_B,),
            ),
        ]
    )


@pytest.fixture
def after() -> RunSummary:
    return RunSummary.from_events(
        [
            _event("intake", tool_calls=4, duration=1.25, digests=(_DIGEST_B,)),
            _event(
                "coding",
                OutcomeClass.FAILED,
                tool_calls=6,
                duration=2.0,
                digests=(_DIGEST_C,),
            ),
            _event("coding", tool_calls=1, duration=0.25),
        ]
    )


def test_equal_summaries_have_no_changes(before: RunSummary) -> None:
    diff = diff_run_summaries(before, before)

    assert diff.changed is False
    assert diff.workflow_ids_added == ()
    assert diff.workflow_ids_removed == ()
    assert dict(diff.outcome_count_deltas) == _zero_deltas()
    assert diff.tool_call_count_delta == 0
    assert diff.duration_seconds_delta == 0.0
    assert diff.artifact_digests_added == ()
    assert diff.artifact_digests_removed == ()


def test_changed_summaries_match_hand_calculations(
    before: RunSummary, after: RunSummary
) -> None:
    diff = diff_run_summaries(before, after)

    # before: intake(success), review(review_required); 5 calls; 2.0 s; {A, B}
    # after: intake(success), coding(failed), coding(success); 11 calls; 3.5 s;
    # {B, C}
    assert diff.changed is True
    assert diff.workflow_ids_added == ("coding",)
    assert diff.workflow_ids_removed == ("review",)
    assert dict(diff.outcome_count_deltas) == _zero_deltas(
        failed=1, review_required=-1, success=1
    )
    assert diff.tool_call_count_delta == 6
    assert diff.duration_seconds_delta == 1.5
    assert diff.artifact_digests_added == (_DIGEST_C,)
    assert diff.artifact_digests_removed == (_DIGEST_A,)


def test_reversing_inputs_negates_deltas_and_swaps_sets(
    before: RunSummary, after: RunSummary
) -> None:
    forward = diff_run_summaries(before, after)
    backward = diff_run_summaries(after, before)

    assert backward.workflow_ids_added == forward.workflow_ids_removed
    assert backward.workflow_ids_removed == forward.workflow_ids_added
    assert backward.artifact_digests_added == forward.artifact_digests_removed
    assert backward.artifact_digests_removed == forward.artifact_digests_added
    assert {
        name: -delta for name, delta in forward.outcome_count_deltas.items()
    } == dict(backward.outcome_count_deltas)
    assert backward.tool_call_count_delta == -forward.tool_call_count_delta
    assert backward.duration_seconds_delta == -forward.duration_seconds_delta


@pytest.mark.parametrize(
    ("before_event", "after_event", "field_name", "expected"),
    [
        (
            _event("wf"),
            _event("wf-next"),
            "workflow_ids_added",
            ("wf-next",),
        ),
        (
            _event("wf"),
            _event("wf", OutcomeClass.ABSTAINED),
            "outcome_count_deltas",
            _zero_deltas(abstained=1, success=-1),
        ),
        (
            _event("wf", duration=2.0),
            _event("wf", duration=0.75),
            "duration_seconds_delta",
            -1.25,
        ),
        (
            _event("wf", tool_calls=7),
            _event("wf", tool_calls=2),
            "tool_call_count_delta",
            -5,
        ),
        (
            _event("wf", digests=(_DIGEST_A,)),
            _event("wf", digests=(_DIGEST_A, _DIGEST_C)),
            "artifact_digests_added",
            (_DIGEST_C,),
        ),
    ],
)
def test_single_field_changes_are_isolated(
    before_event: RunEvent,
    after_event: RunEvent,
    field_name: str,
    expected: Any,
) -> None:
    diff = diff_run_summaries(
        RunSummary.from_events([before_event]),
        RunSummary.from_events([after_event]),
    )

    value = getattr(diff, field_name)
    if field_name == "outcome_count_deltas":
        value = dict(value)
    assert value == expected
    assert diff.changed is True


def test_equal_durations_serialize_without_negative_zero() -> None:
    summary = RunSummary.from_events([_event("wf", duration=0.0)])
    diff = RunSummaryDiff(
        workflow_ids_added=(),
        workflow_ids_removed=(),
        outcome_count_deltas=_zero_deltas(),
        tool_call_count_delta=0,
        duration_seconds_delta=-0.0,
        artifact_digests_added=(),
        artifact_digests_removed=(),
    )

    assert diff == diff_run_summaries(summary, summary)
    assert '"duration_seconds_delta":0.0' in diff.to_json()


def test_json_is_deterministic_and_complete(
    before: RunSummary, after: RunSummary
) -> None:
    diff = diff_run_summaries(before, after)
    payload = diff.to_json()

    assert payload == diff_run_summaries(before, after).to_json()
    assert json.loads(payload) == {
        "artifact_digests": {"added": [_DIGEST_C], "removed": [_DIGEST_A]},
        "changed": True,
        "duration_seconds_delta": 1.5,
        "outcome_count_deltas": _zero_deltas(failed=1, review_required=-1, success=1),
        "schema_version": RUN_DIFF_SCHEMA_VERSION,
        "tool_call_count_delta": 6,
        "workflow_ids": {"added": ["coding"], "removed": ["review"]},
    }
    assert list(diff.to_dict()) == [
        "schema_version",
        "changed",
        "workflow_ids",
        "outcome_count_deltas",
        "tool_call_count_delta",
        "duration_seconds_delta",
        "artifact_digests",
    ]


def test_markdown_matches_golden_output(before: RunSummary, after: RunSummary) -> None:
    assert diff_run_summaries(before, after).to_markdown() == (
        "# Agent Run Summary Diff\n"
        "\n"
        "Changed: yes\n"
        "\n"
        "## Workflows\n"
        "\n"
        "| Change | Workflow |\n"
        "| --- | --- |\n"
        "| added | `coding` |\n"
        "| removed | `review` |\n"
        "\n"
        "## Outcomes\n"
        "\n"
        "| Outcome | Delta |\n"
        "| --- | ---: |\n"
        "| `abstained` | 0 |\n"
        "| `failed` | +1 |\n"
        "| `policy_denied` | 0 |\n"
        "| `review_required` | -1 |\n"
        "| `success` | +1 |\n"
        "\n"
        "## Execution\n"
        "\n"
        "| Metric | Delta |\n"
        "| --- | ---: |\n"
        "| Tool calls | +6 |\n"
        "| Duration (seconds) | +1.5 |\n"
        "\n"
        "## Artifacts\n"
        "\n"
        "| Change | SHA-256 |\n"
        "| --- | --- |\n"
        f"| added | `{_DIGEST_C}` |\n"
        f"| removed | `{_DIGEST_A}` |\n"
    )


def test_unchanged_markdown_reports_no_change(before: RunSummary) -> None:
    markdown = diff_run_summaries(before, before).to_markdown()

    assert "Changed: no\n" in markdown
    assert "| Tool calls | 0 |" in markdown
    assert "| added |" not in markdown
    assert "| removed |" not in markdown


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"workflow_ids": ["wf"]},
        '{"schema_version": "openmed.agent.run_summary.v1"}',
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_non_summary_inputs_are_rejected(
    before: RunSummary, value: Any, position: str
) -> None:
    arguments = {"before": before, "after": before, position: value}

    with pytest.raises(RunDiffError, match=rf"^{position}: invalid_summary$"):
        diff_run_summaries(**arguments)


_SENTINEL = "Synthetic_Patient_Secret_987"


def _diff_fields(**updates: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "workflow_ids_added": (),
        "workflow_ids_removed": (),
        "outcome_count_deltas": _zero_deltas(),
        "tool_call_count_delta": 0,
        "duration_seconds_delta": 0.0,
        "artifact_digests_added": (),
        "artifact_digests_removed": (),
    }
    fields.update(updates)
    return fields


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"workflow_ids_added": (f"{_SENTINEL} prompt text",)},
            "workflow_ids_added: invalid_item",
        ),
        (
            {"workflow_ids_removed": ("/home/clinic/notes.txt",)},
            "workflow_ids_removed: invalid_item",
        ),
        ({"workflow_ids_added": _SENTINEL}, "workflow_ids_added: invalid_sequence"),
        (
            {"workflow_ids_added": ("b", "a")},
            "workflow_ids_added: not_sorted_unique",
        ),
        (
            {"workflow_ids_added": ("wf",), "workflow_ids_removed": ("wf",)},
            "workflow_ids: added_and_removed",
        ),
        (
            {"artifact_digests_added": (f"sha256:{_SENTINEL}",)},
            "artifact_digests_added: invalid_item",
        ),
        (
            {
                "artifact_digests_added": (_DIGEST_A,),
                "artifact_digests_removed": (_DIGEST_A,),
            },
            "artifact_digests: added_and_removed",
        ),
        (
            {"outcome_count_deltas": {**_zero_deltas(), _SENTINEL: 1}},
            "outcome_count_deltas: invalid_keys",
        ),
        (
            {"outcome_count_deltas": _zero_deltas(success=True)},
            "outcome_count_deltas.success: invalid_delta",
        ),
        ({"tool_call_count_delta": 1.0}, "tool_call_count_delta: invalid_delta"),
        (
            {"tool_call_count_delta": 10_000_001},
            "tool_call_count_delta: invalid_delta",
        ),
        (
            {"duration_seconds_delta": float("nan")},
            "duration_seconds_delta: out_of_range",
        ),
        (
            {"duration_seconds_delta": _SENTINEL},
            "duration_seconds_delta: invalid_number",
        ),
        ({"schema_version": _SENTINEL}, "schema_version: unsupported_version"),
    ],
)
def test_direct_construction_rejects_unsafe_values_without_echoing_them(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(RunDiffError) as exc_info:
        RunSummaryDiff(**_diff_fields(**updates))

    assert str(exc_info.value) == message
    assert _SENTINEL not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_sentinel_payloads_cannot_enter_output(before: RunSummary) -> None:
    for unsafe in (
        f"{_SENTINEL} prompt",
        "Bearer secret-token",
        "C:\\clinic\\notes.txt",
        "patient has pneumonia",
    ):
        with pytest.raises(ValueError) as exc_info:
            RunSummary.from_events([_event(unsafe)])
        assert unsafe not in str(exc_info.value)

    diff = diff_run_summaries(before, before)
    rendered = diff.to_json() + diff.to_markdown()
    assert _SENTINEL not in rendered


def test_diff_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.diff_run_summaries is diff_run_summaries
    assert agent.RunSummaryDiff is RunSummaryDiff
    assert agent.RunDiffError is RunDiffError
    assert agent.RUN_DIFF_SCHEMA_VERSION == RUN_DIFF_SCHEMA_VERSION
