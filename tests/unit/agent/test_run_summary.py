from __future__ import annotations

import itertools
import json
from collections.abc import Callable

import pytest

from openmed.agent import OutcomeClass, WorkflowOutcome
from openmed.agent.run_summary import (
    MAX_RUN_SUMMARY_JSON_BYTES,
    RUN_SUMMARY_SCHEMA_VERSION,
    RunEvent,
    RunSummary,
    RunSummaryError,
    RunSummaryPrivacyError,
)

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


def test_run_event_accepts_safe_metadata() -> None:
    event = RunEvent(
        workflow_id="clinical-review",
        outcome=_outcome(),
        tool_call_count=3,
        duration_seconds=1.25,
        artifact_digests=("sha256:" + "a" * 64,),
    )

    assert event.workflow_id == "clinical-review"
    assert event.outcome.outcome_class is OutcomeClass.SUCCESS
    assert event.tool_call_count == 3
    assert event.duration_seconds == 1.25
    assert event.artifact_digests == ("sha256:" + "a" * 64,)


@pytest.mark.parametrize("outcome_class", list(OutcomeClass))
def test_run_event_accepts_closed_outcome_vocabulary(
    outcome_class: OutcomeClass,
) -> None:
    event = RunEvent(workflow_id="workflow-1", outcome=_outcome(outcome_class))

    assert event.outcome.outcome_class is outcome_class


@pytest.mark.parametrize(
    "workflow_id",
    ["workflow with spaces", "/tmp/workflow", "https://example.test", "../secret"],
)
def test_run_event_rejects_paths_urls_and_free_text(workflow_id: str) -> None:
    with pytest.raises(RunSummaryError, match="invalid_identifier"):
        RunEvent(workflow_id=workflow_id, outcome=_outcome())


@pytest.mark.parametrize("outcome", ["success", {"outcome_class": "success"}, None])
def test_run_event_requires_typed_outcome(outcome: object) -> None:
    with pytest.raises(RunSummaryError, match="outcome: invalid_type"):
        RunEvent(
            workflow_id="workflow-1",
            outcome=outcome,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("tool_call_count", [-1, True, 1.5, 10_000_001])
def test_run_event_rejects_invalid_tool_call_count(tool_call_count: object) -> None:
    with pytest.raises(RunSummaryError, match="tool_call_count"):
        RunEvent(
            workflow_id="workflow-1",
            outcome=_outcome(),
            tool_call_count=tool_call_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "duration",
    [-1, float("inf"), float("nan"), True, 31_536_001.0],
)
def test_run_event_rejects_invalid_duration(duration: object) -> None:
    with pytest.raises(RunSummaryError, match="duration_seconds"):
        RunEvent(
            workflow_id="workflow-1",
            outcome=_outcome(),
            duration_seconds=duration,  # type: ignore[arg-type]
        )


def test_run_event_rejects_invalid_or_duplicate_artifact_digests() -> None:
    digest = "sha256:" + "a" * 64
    with pytest.raises(RunSummaryError, match="invalid_digest"):
        RunEvent(
            workflow_id="workflow-1",
            outcome=_outcome(),
            artifact_digests=("not-a-digest",),
        )
    with pytest.raises(RunSummaryError, match="duplicate_item"):
        RunEvent(
            workflow_id="workflow-1",
            outcome=_outcome(),
            artifact_digests=(digest, digest),
        )


def test_summary_aggregates_all_outcomes_deterministically() -> None:
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    classes = list(OutcomeClass)
    events = [
        RunEvent(
            workflow_id=f"workflow-{index % 2}",
            outcome=_outcome(outcome_class),
            tool_call_count=index,
            duration_seconds=index / 10,
            artifact_digests=((digest_a if index % 2 else digest_b),),
        )
        for index, outcome_class in enumerate(classes, start=1)
    ]

    summary = RunSummary.from_events(events)

    assert summary.workflow_ids == ("workflow-0", "workflow-1")
    assert summary.outcome_counts == _counts(
        abstained=1,
        failed=1,
        policy_denied=1,
        review_required=1,
        success=1,
    )
    assert summary.tool_call_count == 15
    assert summary.duration_seconds == 1.5
    assert summary.artifact_digests == (digest_a, digest_b)


def test_empty_events_produce_zero_summary() -> None:
    summary = RunSummary.from_events([])

    assert summary.workflow_ids == ()
    assert summary.outcome_counts == _counts()
    assert summary.tool_call_count == 0
    assert summary.duration_seconds == 0.0
    assert summary.artifact_digests == ()


def test_raw_source_event_is_rejected_without_echoing_phi_or_credentials() -> None:
    phi = "Jane Synthetic has diagnosis Z99.999"
    bearer = "Bearer synthetic-secret-token"
    source_event = {
        "workflow_id": "clinical-review",
        "outcome": "success",
        "prompt": phi,
        "tool_output": bearer,
    }

    with pytest.raises(RunSummaryError) as exc_info:
        RunSummary.from_events([source_event])  # type: ignore[list-item]

    assert phi not in str(exc_info.value)
    assert bearer not in str(exc_info.value)


def test_to_dict_contains_only_safe_metadata() -> None:
    digest = "sha256:" + "c" * 64
    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="workflow-1",
                outcome=_outcome(),
                tool_call_count=2,
                duration_seconds=1.0,
                artifact_digests=(digest,),
            )
        ]
    )

    payload = summary.to_dict()

    assert payload == {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "workflow_ids": ["workflow-1"],
        "outcome_counts": _counts(success=1),
        "tool_call_count": 2,
        "duration_seconds": 1.0,
        "artifact_digests": [digest],
    }
    json.dumps(payload)


def test_to_json_is_byte_deterministic() -> None:
    summary = RunSummary.from_events(
        [
            RunEvent("workflow-2", _outcome(), tool_call_count=1),
            RunEvent("workflow-1", _outcome(OutcomeClass.FAILED), tool_call_count=2),
        ]
    )
    expected = (
        '{"artifact_digests":[],"duration_seconds":0.0,'
        '"outcome_counts":{"abstained":0,"failed":1,"policy_denied":0,'
        '"review_required":0,"success":1},'
        f'"schema_version":"{RUN_SUMMARY_SCHEMA_VERSION}","tool_call_count":3,'
        '"workflow_ids":["workflow-1","workflow-2"]}'
    )

    assert summary.to_json() == expected
    assert summary.to_json() == expected


def test_summary_round_trips_byte_stably_through_dict_and_json() -> None:
    summary = RunSummary.from_events(
        [RunEvent("workflow-1", _outcome(), tool_call_count=2)]
    )

    from_dict = RunSummary.from_dict(summary.to_dict())
    from_json = RunSummary.from_json(summary.to_json())

    assert from_dict == summary
    assert from_json == summary
    assert from_dict.to_json() == summary.to_json()
    assert from_json.to_json() == summary.to_json()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload, sentinel: payload.update({sentinel: sentinel}),
            "unknown_field",
        ),
        (lambda payload, sentinel: payload.pop("workflow_ids"), "missing_field"),
        (
            lambda payload, sentinel: payload.update({"schema_version": sentinel}),
            "unsupported_version",
        ),
        (
            lambda payload, sentinel: payload.update({"tool_call_count": -1}),
            "invalid_count",
        ),
        (
            lambda payload, sentinel: payload.update(
                {"workflow_ids": [f"{sentinel} private note"]}
            ),
            "invalid_identifier",
        ),
    ],
)
def test_from_dict_rejects_invalid_shapes_without_echoing_values(
    mutation: Callable[[dict[str, object], str], object], error: str
) -> None:
    summary = RunSummary.from_events([RunEvent("workflow-1", _outcome())])
    payload = summary.to_dict()
    sentinel = "Synthetic_Patient_Secret_987"
    mutation(payload, sentinel)

    with pytest.raises(RunSummaryError, match=error) as exc_info:
        RunSummary.from_dict(payload)

    assert sentinel not in str(exc_info.value)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("{", "invalid_json"),
        (
            '{"schema_version":"openmed.agent.run_summary.v1",'
            '"schema_version":"openmed.agent.run_summary.v1"}',
            "duplicate_field",
        ),
        (
            '{"schema_version":"openmed.agent.run_summary.v1",'
            '"workflow_ids":[],"outcome_counts":{},"tool_call_count":0,'
            '"duration_seconds":NaN,"artifact_digests":[]}',
            "non_finite_number",
        ),
    ],
)
def test_from_json_rejects_malformed_duplicate_and_non_finite_values(
    payload: str, error: str
) -> None:
    with pytest.raises(RunSummaryError, match=error):
        RunSummary.from_json(payload)


def test_from_json_rejects_oversized_input_without_parsing() -> None:
    payload = " " * (MAX_RUN_SUMMARY_JSON_BYTES + 1)

    with pytest.raises(RunSummaryError, match="json_too_large"):
        RunSummary.from_json(payload)


def test_markdown_contains_metadata_only() -> None:
    digest = "sha256:" + "d" * 64
    summary = RunSummary.from_events(
        [
            RunEvent(
                workflow_id="clinical-review",
                outcome=_outcome(OutcomeClass.ABSTAINED),
                tool_call_count=4,
                duration_seconds=3.5,
                artifact_digests=(digest,),
            )
        ]
    )

    markdown = summary.to_markdown()

    assert "| `clinical-review` |" in markdown
    assert "| `abstained` | 1 |" in markdown
    assert "| Tool calls | 4 |" in markdown
    assert "| Duration (seconds) | 3.5 |" in markdown
    assert digest in markdown


def test_summary_deduplicates_metadata_across_events() -> None:
    digest = "sha256:" + "a" * 64
    summary = RunSummary.from_events(
        [
            RunEvent("workflow-1", _outcome(), artifact_digests=(digest,)),
            RunEvent("workflow-1", _outcome(), artifact_digests=(digest,)),
        ]
    )

    assert summary.workflow_ids == ("workflow-1",)
    assert summary.artifact_digests == (digest,)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"workflow_ids": ("z", "a")}, "not_sorted_unique"),
        ({"outcome_counts": {"success": 1}}, "invalid_keys"),
        ({"outcome_counts": _counts(success=-1)}, "invalid_count"),
        ({"tool_call_count": True}, "invalid_count"),
        ({"duration_seconds": float("inf")}, "duration_seconds"),
        (
            {
                "artifact_digests": (
                    "sha256:" + "b" * 64,
                    "sha256:" + "a" * 64,
                )
            },
            "not_sorted_unique",
        ),
    ],
)
def test_direct_summary_construction_cannot_bypass_invariants(
    overrides: dict[str, object], error: str
) -> None:
    values: dict[str, object] = {
        "workflow_ids": ("workflow-1",),
        "outcome_counts": _counts(success=1),
        "tool_call_count": 0,
        "duration_seconds": 0.0,
        "artifact_digests": (),
    }
    values.update(overrides)

    with pytest.raises(RunSummaryError, match=error):
        RunSummary(**values)  # type: ignore[arg-type]


def test_aggregate_totals_and_event_count_are_bounded() -> None:
    event = RunEvent("workflow-1", _outcome(), tool_call_count=6_000_000)
    with pytest.raises(RunSummaryError, match="total_out_of_range"):
        RunSummary.from_events([event, event])

    empty_event = RunEvent("workflow-1", _outcome())
    with pytest.raises(RunSummaryError, match="too_many_items"):
        RunSummary.from_events(itertools.repeat(empty_event, 10_001))


def test_privacy_guard_never_echoes_dynamic_mapping_keys_or_values() -> None:
    from openmed.agent.run_summary import _assert_safe_payload

    phi_key = "Jane Synthetic diagnosis Z99.999"
    bearer_value = "Bearer synthetic-secret-token"
    with pytest.raises(RunSummaryPrivacyError) as exc_info:
        _assert_safe_payload({phi_key: bearer_value})

    assert phi_key not in str(exc_info.value)
    assert bearer_value not in str(exc_info.value)


def test_privacy_guard_rejects_non_string_keys_and_non_finite_numbers() -> None:
    from openmed.agent.run_summary import _assert_safe_payload

    with pytest.raises(RunSummaryPrivacyError, match="invalid_mapping_key"):
        _assert_safe_payload({1: "value"})
    with pytest.raises(RunSummaryPrivacyError, match="non_finite_number"):
        _assert_safe_payload({"value": float("inf")})
