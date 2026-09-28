"""Deterministic, metadata-only rollups of agent runs by workflow.

The rollup boundary accepts only validated :class:`RunEvent` or
:class:`RunSummary` instances. Rendered reports contain bounded identifiers,
counts, and durations; artifact digests and event payloads are not retained.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .outcomes import OutcomeClass
from .run_summary import RunEvent, RunSummary

WORKFLOW_ROLLUP_SCHEMA_VERSION = "openmed.agent.workflow_rollup.v1"

_OUTCOME_NAMES = tuple(sorted(outcome.value for outcome in OutcomeClass))
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_MAX_INPUTS = 10_000
_MAX_RUNS = 10_000
_MAX_WORKFLOWS = 1_024
_MAX_TOOL_CALLS = 10_000_000
_MAX_DURATION_SECONDS = 31_536_000.0
_MAX_ARTIFACT_REFERENCES = 4_096


class WorkflowRollupError(ValueError):
    """Raised when workflow-rollup input fails closed validation."""


def _validate_count(value: Any, field_name: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise WorkflowRollupError(f"{field_name}: invalid_count")
    return value


def _validate_duration(value: Any, field_name: str) -> float:
    if type(value) not in (int, float):
        raise WorkflowRollupError(f"{field_name}: invalid_number")
    if not 0 <= value <= _MAX_DURATION_SECONDS:
        raise WorkflowRollupError(f"{field_name}: out_of_range")
    normalized = float(value)
    if (
        not math.isfinite(normalized)
        or normalized < 0.0
        or normalized > _MAX_DURATION_SECONDS
    ):
        raise WorkflowRollupError(f"{field_name}: out_of_range")
    return normalized


def _validate_outcome_counts(
    value: Mapping[str, int], *, field_name: str
) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_OUTCOME_NAMES):
        raise WorkflowRollupError(f"{field_name}: invalid_keys")
    normalized = {
        outcome: _validate_count(value[outcome], f"{field_name}.{outcome}", _MAX_RUNS)
        for outcome in _OUTCOME_NAMES
    }
    if sum(normalized.values()) > _MAX_RUNS:
        raise WorkflowRollupError(f"{field_name}: total_out_of_range")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class WorkflowRollupRow:
    """Aggregate metadata for one bounded workflow identifier.

    Args:
        workflow_id: Validated workflow identifier inherited from source runs.
        outcome_counts: Counts for every closed workflow outcome class.
        tool_call_count: Total tool calls across the workflow's runs.
        duration_seconds: Total finite duration across the workflow's runs.
        run_count: Number of runs represented by the row.
        artifact_digest_count: Digests unique within this workflow.
    """

    workflow_id: str
    outcome_counts: Mapping[str, int]
    tool_call_count: int
    duration_seconds: float
    run_count: int
    artifact_digest_count: int

    def __post_init__(self) -> None:
        if (
            type(self.workflow_id) is not str
            or _IDENTIFIER_RE.fullmatch(self.workflow_id) is None
        ):
            raise WorkflowRollupError("workflow_id: invalid_identifier")
        normalized_counts = _validate_outcome_counts(
            self.outcome_counts, field_name="outcome_counts"
        )
        run_count = _validate_count(self.run_count, "run_count", _MAX_RUNS)
        if sum(normalized_counts.values()) != run_count:
            raise WorkflowRollupError("outcome_counts: run_count_mismatch")
        _validate_count(self.tool_call_count, "tool_call_count", _MAX_TOOL_CALLS)
        duration = _validate_duration(self.duration_seconds, "duration_seconds")
        _validate_count(
            self.artifact_digest_count,
            "artifact_digest_count",
            _MAX_ARTIFACT_REFERENCES,
        )
        object.__setattr__(self, "outcome_counts", normalized_counts)
        object.__setattr__(self, "duration_seconds", duration)

    def to_dict(self) -> dict[str, Any]:
        """Return metadata-only JSON-compatible row data."""
        return {
            "workflow_id": self.workflow_id,
            "outcome_counts": dict(self.outcome_counts),
            "tool_call_count": self.tool_call_count,
            "duration_seconds": self.duration_seconds,
            "run_count": self.run_count,
            "artifact_digest_count": self.artifact_digest_count,
        }


@dataclass(frozen=True, slots=True)
class _Contribution:
    workflow_id: str
    outcome_counts: Mapping[str, int]
    tool_call_count: int
    duration_seconds: float
    artifact_digests: tuple[str, ...]


@dataclass(slots=True)
class _Accumulator:
    outcome_counts: dict[str, int]
    tool_call_count: int
    durations: list[float]
    artifact_digests: set[str]


@dataclass(frozen=True, slots=True)
class WorkflowRollup:
    """Stable per-workflow rows and their exactly reconciled overall totals."""

    workflows: tuple[WorkflowRollupRow, ...]
    schema_version: str = WORKFLOW_ROLLUP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != WORKFLOW_ROLLUP_SCHEMA_VERSION
        ):
            raise WorkflowRollupError("schema_version: unsupported_version")
        if type(self.workflows) not in (list, tuple):
            raise WorkflowRollupError("workflows: invalid_sequence")
        rows = tuple(self.workflows)
        if len(rows) > _MAX_WORKFLOWS:
            raise WorkflowRollupError("workflows: too_many_items")
        if any(not isinstance(row, WorkflowRollupRow) for row in rows):
            raise WorkflowRollupError("workflows: invalid_item")
        workflow_ids = tuple(row.workflow_id for row in rows)
        if workflow_ids != tuple(sorted(set(workflow_ids))):
            raise WorkflowRollupError("workflows: not_sorted_unique")

        if self.run_count > _MAX_RUNS:
            raise WorkflowRollupError("run_count: total_out_of_range")
        if self.tool_call_count > _MAX_TOOL_CALLS:
            raise WorkflowRollupError("tool_call_count: total_out_of_range")
        if self.duration_seconds > _MAX_DURATION_SECONDS:
            raise WorkflowRollupError("duration_seconds: total_out_of_range")
        if self.artifact_digest_count > _MAX_ARTIFACT_REFERENCES:
            raise WorkflowRollupError("artifact_digest_count: total_out_of_range")
        object.__setattr__(self, "workflows", rows)

    @classmethod
    def from_events(cls, events: Iterable[RunEvent]) -> "WorkflowRollup":
        """Roll up a bounded iterable of validated run events."""
        iterator = _bounded_iterator(events, field_name="events")

        def contributions() -> Iterable[_Contribution]:
            for event in iterator:
                if not isinstance(event, RunEvent):
                    raise WorkflowRollupError("events: invalid_item")
                counts = {outcome: 0 for outcome in _OUTCOME_NAMES}
                counts[event.outcome.outcome_class.value] = 1
                yield _Contribution(
                    workflow_id=event.workflow_id,
                    outcome_counts=counts,
                    tool_call_count=event.tool_call_count,
                    duration_seconds=event.duration_seconds,
                    artifact_digests=event.artifact_digests,
                )

        return cls._from_contributions(contributions())

    @classmethod
    def from_summaries(cls, summaries: Iterable[RunSummary]) -> "WorkflowRollup":
        """Roll up bounded summaries that each represent at most one workflow.

        Empty summaries are ignored. A non-empty summary containing multiple
        workflow identifiers is rejected because its metrics cannot be
        attributed without reopening its source events.
        """
        iterator = _bounded_iterator(summaries, field_name="summaries")

        def contributions() -> Iterable[_Contribution]:
            for summary in iterator:
                if not isinstance(summary, RunSummary):
                    raise WorkflowRollupError("summaries: invalid_item")
                run_count = sum(summary.outcome_counts.values())
                has_metrics = any(
                    (
                        run_count,
                        summary.tool_call_count,
                        summary.duration_seconds,
                        len(summary.artifact_digests),
                    )
                )
                if not summary.workflow_ids:
                    if has_metrics:
                        raise WorkflowRollupError("summaries: missing_workflow_id")
                    continue
                if len(summary.workflow_ids) != 1:
                    raise WorkflowRollupError("summaries: ambiguous_workflow_ids")
                if run_count == 0 and has_metrics:
                    raise WorkflowRollupError("summaries: inconsistent_counts")
                yield _Contribution(
                    workflow_id=summary.workflow_ids[0],
                    outcome_counts=summary.outcome_counts,
                    tool_call_count=summary.tool_call_count,
                    duration_seconds=summary.duration_seconds,
                    artifact_digests=summary.artifact_digests,
                )

        return cls._from_contributions(contributions())

    @classmethod
    def _from_contributions(
        cls, contributions: Iterable[_Contribution]
    ) -> "WorkflowRollup":
        accumulators: dict[str, _Accumulator] = {}
        for contribution in contributions:
            accumulator = accumulators.get(contribution.workflow_id)
            if accumulator is None:
                if len(accumulators) >= _MAX_WORKFLOWS:
                    raise WorkflowRollupError("workflows: too_many_items")
                accumulator = _Accumulator(
                    outcome_counts={outcome: 0 for outcome in _OUTCOME_NAMES},
                    tool_call_count=0,
                    durations=[],
                    artifact_digests=set(),
                )
                accumulators[contribution.workflow_id] = accumulator

            for outcome in _OUTCOME_NAMES:
                accumulator.outcome_counts[outcome] += contribution.outcome_counts[
                    outcome
                ]
            if sum(accumulator.outcome_counts.values()) > _MAX_RUNS:
                raise WorkflowRollupError("run_count: total_out_of_range")
            accumulator.tool_call_count += contribution.tool_call_count
            if accumulator.tool_call_count > _MAX_TOOL_CALLS:
                raise WorkflowRollupError("tool_call_count: total_out_of_range")
            accumulator.durations.append(contribution.duration_seconds)
            accumulator.artifact_digests.update(contribution.artifact_digests)
            if len(accumulator.artifact_digests) > _MAX_ARTIFACT_REFERENCES:
                raise WorkflowRollupError("artifact_digest_count: total_out_of_range")

        rows = []
        for workflow_id in sorted(accumulators):
            accumulator = accumulators[workflow_id]
            duration = math.fsum(sorted(accumulator.durations))
            if duration > _MAX_DURATION_SECONDS:
                raise WorkflowRollupError("duration_seconds: total_out_of_range")
            rows.append(
                WorkflowRollupRow(
                    workflow_id=workflow_id,
                    outcome_counts=accumulator.outcome_counts,
                    tool_call_count=accumulator.tool_call_count,
                    duration_seconds=duration,
                    run_count=sum(accumulator.outcome_counts.values()),
                    artifact_digest_count=len(accumulator.artifact_digests),
                )
            )
        return cls(workflows=tuple(rows))

    @property
    def outcome_counts(self) -> Mapping[str, int]:
        """Return overall outcome counts reconciled from workflow rows."""
        return MappingProxyType(
            {
                outcome: sum(row.outcome_counts[outcome] for row in self.workflows)
                for outcome in _OUTCOME_NAMES
            }
        )

    @property
    def tool_call_count(self) -> int:
        """Return the overall tool-call count."""
        return sum(row.tool_call_count for row in self.workflows)

    @property
    def duration_seconds(self) -> float:
        """Return the overall duration in seconds."""
        return math.fsum(row.duration_seconds for row in self.workflows)

    @property
    def run_count(self) -> int:
        """Return the overall run count."""
        return sum(row.run_count for row in self.workflows)

    @property
    def artifact_digest_count(self) -> int:
        """Return the sum of within-workflow unique digest counts."""
        return sum(row.artifact_digest_count for row in self.workflows)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "workflows": [row.to_dict() for row in self.workflows],
            "overall": {
                "outcome_counts": dict(self.outcome_counts),
                "tool_call_count": self.tool_call_count,
                "duration_seconds": self.duration_seconds,
                "run_count": self.run_count,
                "artifact_digest_count": self.artifact_digest_count,
            },
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON in workflow-ID order."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Return deterministic metadata-only Markdown in workflow-ID order."""
        headers = [
            "Workflow ID",
            "Runs",
            *(_markdown_label(outcome) for outcome in _OUTCOME_NAMES),
            "Tool calls",
            "Duration (seconds)",
            "Unique artifact digests",
        ]
        lines = [
            "# Agent Workflow Rollup",
            "",
            "| " + " | ".join(headers) + " |",
            "| :--- | " + " | ".join("---:" for _ in headers[1:]) + " |",
        ]
        lines.extend(_markdown_row(row) for row in self.workflows)
        lines.append(
            _markdown_values(
                "**Overall**",
                self.run_count,
                self.outcome_counts,
                self.tool_call_count,
                self.duration_seconds,
                self.artifact_digest_count,
            )
        )
        return "\n".join(lines) + "\n"


def _bounded_iterator(value: Any, *, field_name: str) -> Iterable[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise WorkflowRollupError(f"{field_name}: invalid_iterable")
    try:
        iterator = iter(value)
    except TypeError:
        raise WorkflowRollupError(f"{field_name}: invalid_iterable") from None

    def bounded() -> Iterable[Any]:
        for index, item in enumerate(iterator):
            if index >= _MAX_INPUTS:
                raise WorkflowRollupError(f"{field_name}: too_many_items")
            yield item

    return bounded()


def _markdown_label(outcome: str) -> str:
    return outcome.replace("_", " ").capitalize()


def _markdown_row(row: WorkflowRollupRow) -> str:
    return _markdown_values(
        f"`{row.workflow_id}`",
        row.run_count,
        row.outcome_counts,
        row.tool_call_count,
        row.duration_seconds,
        row.artifact_digest_count,
    )


def _markdown_values(
    workflow: str,
    run_count: int,
    outcome_counts: Mapping[str, int],
    tool_call_count: int,
    duration_seconds: float,
    artifact_digest_count: int,
) -> str:
    values = [
        workflow,
        str(run_count),
        *(str(outcome_counts[outcome]) for outcome in _OUTCOME_NAMES),
        str(tool_call_count),
        f"{duration_seconds:g}",
        str(artifact_digest_count),
    ]
    return "| " + " | ".join(values) + " |"


__all__ = [
    "WORKFLOW_ROLLUP_SCHEMA_VERSION",
    "WorkflowRollup",
    "WorkflowRollupError",
    "WorkflowRollupRow",
]
