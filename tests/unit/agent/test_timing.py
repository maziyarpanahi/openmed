"""Offline tests for monotonic agent timing metadata."""

from __future__ import annotations

import json

import pytest

from openmed.agent.timing import (
    ActionTiming,
    AgentRunTiming,
    RunTiming,
    TimingValidationError,
)


def test_run_timing_accepts_zero_duration_and_exact_maximum() -> None:
    zero = RunTiming(start_ns=8, end_ns=8, max_duration_ns=0)
    bounded = RunTiming(start_ns=10, end_ns=25, max_duration_ns=15)

    assert zero.duration_ns == 0
    assert bounded.duration_ns == 15
    assert bounded.to_dict() == {
        "start_ns": 10,
        "end_ns": 25,
        "duration_ns": 15,
    }


def test_agent_timing_accepts_nested_and_sequential_actions() -> None:
    timing = AgentRunTiming(
        run=RunTiming(start_ns=10, end_ns=40, correlation_id="run-opaque"),
        actions=(
            ActionTiming(action_id="extract", start_ns=12, end_ns=30),
            ActionTiming(
                action_id="tool",
                parent_action_id="extract",
                start_ns=15,
                end_ns=20,
                correlation_id="action-opaque",
            ),
            ActionTiming(action_id="classify", start_ns=30, end_ns=35),
        ),
    )

    assert timing.actions[0].duration_ns == 18
    assert timing.actions[1].duration_ns == 5
    assert timing.actions[2].duration_ns == 5
    assert timing.to_dict() == {
        "run": {
            "start_ns": 10,
            "end_ns": 40,
            "duration_ns": 30,
            "correlation_id": "run-opaque",
        },
        "actions": [
            {
                "action_id": "extract",
                "start_ns": 12,
                "end_ns": 30,
                "duration_ns": 18,
            },
            {
                "action_id": "tool",
                "parent_action_id": "extract",
                "start_ns": 15,
                "end_ns": 20,
                "duration_ns": 5,
                "correlation_id": "action-opaque",
            },
            {
                "action_id": "classify",
                "start_ns": 30,
                "end_ns": 35,
                "duration_ns": 5,
            },
        ],
    }


@pytest.mark.parametrize(
    "kwargs, field, hidden_value",
    [
        ({"start_ns": -1, "end_ns": 2}, "run.start_ns", "-1"),
        ({"start_ns": 5, "end_ns": 4}, "run.end_ns", "5"),
        ({"start_ns": True, "end_ns": 2}, "run.start_ns", "True"),
        ({"start_ns": 1.25, "end_ns": 2}, "run.start_ns", "1.25"),
        ({"start_ns": 0, "end_ns": 3, "max_duration_ns": 2}, "run.duration_ns", "3"),
    ],
)
def test_invalid_run_boundaries_fail_with_field_only_errors(
    kwargs: dict[str, object],
    field: str,
    hidden_value: str,
) -> None:
    with pytest.raises(TimingValidationError) as caught:
        RunTiming(**kwargs)  # type: ignore[arg-type]

    message = str(caught.value)
    assert field in message
    assert hidden_value not in message


def test_overlapping_actions_fail_when_disallowed_without_echoing_values() -> None:
    with pytest.raises(TimingValidationError) as caught:
        AgentRunTiming(
            run=RunTiming(start_ns=0, end_ns=30),
            actions=(
                ActionTiming(action_id="first", start_ns=5, end_ns=15),
                ActionTiming(action_id="second", start_ns=14, end_ns=20),
            ),
            allow_action_overlaps=False,
        )

    message = str(caught.value)
    assert "actions" in message
    assert "first" not in message
    assert "second" not in message
    assert "14" not in message


def test_parent_and_run_bounds_fail_without_echoing_identifiers() -> None:
    with pytest.raises(TimingValidationError) as caught:
        AgentRunTiming(
            run=RunTiming(start_ns=10, end_ns=40),
            actions=(
                ActionTiming(action_id="outer", start_ns=12, end_ns=20),
                ActionTiming(
                    action_id="inner",
                    parent_action_id="outer",
                    start_ns=15,
                    end_ns=21,
                ),
            ),
        )

    message = str(caught.value)
    assert "parent_action_id" in message
    assert "outer" not in message
    assert "inner" not in message
    assert "21" not in message


def test_serialization_stays_relative_and_payload_free() -> None:
    timing = AgentRunTiming(
        run=RunTiming(start_ns=0, end_ns=1, correlation_id="opaque-run"),
        actions=(
            ActionTiming(
                action_id="scan",
                start_ns=0,
                end_ns=1,
                correlation_id="opaque-action",
            ),
        ),
    )
    serialized = timing.to_json()

    assert serialized == json.dumps(
        timing.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert "opaque-run" in serialized
    assert "opaque-action" in serialized
    assert "wall" not in serialized
    assert "timestamp" not in serialized
    assert "payload" not in serialized
    assert "event" not in serialized


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    (
        (RunTiming, {"start_ns": 0, "end_ns": 1, "correlation_id": "/phi/run"}),
        (
            ActionTiming,
            {"action_id": "https://phi.example/action", "start_ns": 0, "end_ns": 1},
        ),
        (
            ActionTiming,
            {
                "action_id": "safe",
                "start_ns": 0,
                "end_ns": 1,
                "correlation_id": "x" * 129,
            },
        ),
    ),
)
def test_identifiers_reject_paths_urls_and_unbounded_text_without_echo(
    constructor, kwargs
) -> None:
    with pytest.raises(TimingValidationError) as caught:
        constructor(**kwargs)

    message = str(caught.value)
    assert "/phi/run" not in message
    assert "phi.example" not in message
    assert "x" * 129 not in message


def test_parent_action_graph_rejects_cycles() -> None:
    with pytest.raises(TimingValidationError, match="acyclic"):
        AgentRunTiming(
            run=RunTiming(start_ns=0, end_ns=10),
            actions=(
                ActionTiming("first", 0, 10, parent_action_id="second"),
                ActionTiming("second", 0, 10, parent_action_id="first"),
            ),
        )


def test_timing_contract_is_available_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.RunTiming is RunTiming
    assert agent.ActionTiming is ActionTiming
    assert agent.AgentRunTiming is AgentRunTiming
