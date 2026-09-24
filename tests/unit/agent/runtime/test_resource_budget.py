"""Offline checks for per-run resource reservations and safe stop reports."""

import json
import traceback
from concurrent.futures import ThreadPoolExecutor

import pytest

from openmed.agent.runtime.resource_budget import (
    BudgetStopReason,
    ResourceBudgetError,
    ResourceBudgetLimits,
    ResourceBudgetReport,
    RunResourceBudget,
)


def _limits() -> ResourceBudgetLimits:
    return ResourceBudgetLimits(
        max_steps=2,
        max_tool_calls=1,
        max_wall_time_ns=100,
        max_memory_estimate_bytes=50,
        max_artifact_bytes=10,
    )


def _meter() -> RunResourceBudget:
    return RunResourceBudget(_limits(), started_ns=1_000)


@pytest.mark.parametrize(
    ("reservation", "reason"),
    [
        ({"now_ns": 1_101}, BudgetStopReason.WALL_TIME),
        ({"now_ns": 1_000, "steps": 3}, BudgetStopReason.STEPS),
        ({"now_ns": 1_000, "tool_calls": 2}, BudgetStopReason.TOOL_CALLS),
        (
            {"now_ns": 1_000, "memory_estimate_bytes": 51},
            BudgetStopReason.MEMORY_ESTIMATE,
        ),
        ({"now_ns": 1_000, "artifact_bytes": 11}, BudgetStopReason.ARTIFACT_STORAGE),
    ],
)
def test_each_limit_stops_before_scheduling_work(
    reservation: dict[str, int], reason: BudgetStopReason
) -> None:
    meter = _meter()
    report = meter.reserve(**reservation)

    assert report.stopped is True
    assert report.stop_reason is reason
    assert report.steps == report.tool_calls == report.artifact_bytes == 0
    assert report.peak_memory_estimate_bytes == 0
    assert meter.report() == report
    with pytest.raises(ResourceBudgetError, match="run_stopped"):
        meter.reserve(now_ns=1_000)


def test_exact_limits_and_cumulative_usage_are_deterministic() -> None:
    first = _meter()
    second = _meter()
    for meter in (first, second):
        meter.reserve(
            now_ns=1_040,
            steps=1,
            tool_calls=1,
            memory_estimate_bytes=50,
            artifact_bytes=4,
        )
        report = meter.reserve(
            now_ns=1_100,
            steps=1,
            memory_estimate_bytes=20,
            artifact_bytes=6,
        )
        assert report.stopped is False
        assert report.steps == 2
        assert report.tool_calls == 1
        assert report.elapsed_ns == 100
        assert report.peak_memory_estimate_bytes == 50
        assert report.artifact_bytes == 10
        assert json.loads(report.to_json()) == report.to_dict()
    assert first.report().to_json() == second.report().to_json()
    assert first.reserve(now_ns=1_100, steps=1).stop_reason is BudgetStopReason.STEPS


def test_clock_rollback_fails_closed_without_charging_work() -> None:
    meter = _meter()
    meter.reserve(now_ns=1_020, steps=1)
    report = meter.reserve(now_ns=1_019, tool_calls=1)
    assert report.stop_reason is BudgetStopReason.INVALID_CLOCK
    assert report.elapsed_ns == 20
    assert report.steps == 1
    assert report.tool_calls == 0


def test_parallel_reservations_cannot_oversubscribe_one_run() -> None:
    meter = RunResourceBudget(ResourceBudgetLimits(20, 20, 100, 0, 0), started_ns=1_000)

    def reserve_one(_: int) -> bool:
        try:
            return not meter.reserve(now_ns=1_000, steps=1).stopped
        except ResourceBudgetError as exc:
            assert exc.code == "run_stopped"
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve_one, range(40)))

    assert sum(results) == 20
    assert meter.report().steps == 20
    assert meter.report().stop_reason is BudgetStopReason.STEPS


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "synthetic patient identifier"])
def test_invalid_limits_and_reservations_do_not_echo_values(invalid: object) -> None:
    sentinel = "synthetic patient identifier"
    with pytest.raises(ResourceBudgetError) as caught:
        ResourceBudgetLimits(invalid, 1, 1, 1, 1)  # type: ignore[arg-type]
    assert sentinel not in "".join(traceback.format_exception(caught.value))

    meter = _meter()
    with pytest.raises(ResourceBudgetError) as caught:
        meter.reserve(now_ns=1_000, artifact_bytes=invalid)  # type: ignore[arg-type]
    assert sentinel not in "".join(traceback.format_exception(caught.value))
    assert meter.report().steps == meter.report().artifact_bytes == 0
    assert meter.stopped is False


def test_report_has_only_numeric_usage_and_closed_reason() -> None:
    meter = _meter()
    report = meter.reserve(now_ns=1_101)
    assert set(report.to_dict()) == {"limits", "usage", "stopped", "stop_reason"}
    assert report.to_dict()["stop_reason"] == "wall_time"
    assert all(type(value) is int for value in report.to_dict()["usage"].values())
    assert "synthetic patient identifier" not in report.to_json()

    with pytest.raises(ResourceBudgetError, match="invalid_stop_reason"):
        ResourceBudgetReport(
            limits=_limits(),
            steps=0,
            tool_calls=0,
            elapsed_ns=0,
            peak_memory_estimate_bytes=0,
            artifact_bytes=0,
            stop_reason="synthetic patient identifier",  # type: ignore[arg-type]
        )
