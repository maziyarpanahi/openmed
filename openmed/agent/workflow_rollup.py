"""Deterministic, privacy-safe per-workflow rollups of agent run events.

Benchmark and operator reports need one stable row per workflow without
reopening raw events. This module aggregates validated run events into bounded
per-workflow outcome counts, tool-call counts, durations, run counts, and
unique artifact digests, then renders deterministic JSON and Markdown in
workflow-identifier order. Prompts, tool payloads, evidence text, paths, and
credentials never reach this layer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .outcomes import OutcomeClass
from .run_summary import RunEvent, _assert_safe_payload

WORKFLOW_ROLLUP_SCHEMA_VERSION = "openmed.agent.workflow_rollup.v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_OUTCOME_NAMES = tuple(sorted(outcome.value for outcome in OutcomeClass))

_MAX_EVENTS = 10_000
_MAX_WORKFLOWS = 1_024
_MAX_TOOL_CALLS = 10_000_000
_MAX_DURATION_SECONDS = 31_536_000.0
_MAX_ARTIFACT_DIGESTS = 4_096


class WorkflowRollupError(ValueError):
    """Value-free failure for invalid workflow-rollup input."""


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise WorkflowRollupError(f"{field_name}: invalid_identifier")
    return value


def _validate_digest(value: Any) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise WorkflowRollupError("artifact_digests: invalid_digest")
    return value


def _validate_count(value: Any, field_name: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise WorkflowRollupError(f"{field_name}: invalid_count")
    return value


def _validate_duration(value: Any) -> float:
    if type(value) not in (int, float):
        raise WorkflowRollupError("duration_seconds: invalid_number")
    if not 0 <= value <= _MAX_DURATION_SECONDS:
        raise WorkflowRollupError("duration_seconds: out_of_range")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise WorkflowRollupError("duration_seconds: out_of_range")
    return normalized


def _as_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise WorkflowRollupError(f"{field_name}: invalid_sequence")
    try:
        return tuple(value)
    except (TypeError, ValueError):
        pass
    raise WorkflowRollupError(f"{field_name}: invalid_sequence")


@dataclass(frozen=True, slots=True)
class WorkflowRollupEntry:
    """One workflow's metadata-only totals."""

    workflow_id: str
    run_count: int
    outcome_counts: Mapping[str, int]
    tool_call_count: int
    duration_seconds: float
    artifact_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.workflow_id, "workflow_id")
        _validate_count(self.run_count, "run_count", _MAX_EVENTS)

        if not isinstance(self.outcome_counts, Mapping):
            raise WorkflowRollupError("outcome_counts: invalid_mapping")
        if set(self.outcome_counts) != set(_OUTCOME_NAMES):
            raise WorkflowRollupError("outcome_counts: invalid_keys")
        normalized_counts = {
            name: _validate_count(
                self.outcome_counts[name], f"outcome_counts.{name}", _MAX_EVENTS
            )
            for name in _OUTCOME_NAMES
        }
        if sum(normalized_counts.values()) != self.run_count:
            raise WorkflowRollupError("outcome_counts: does_not_match_run_count")

        _validate_count(self.tool_call_count, "tool_call_count", _MAX_TOOL_CALLS)
        normalized_duration = _validate_duration(self.duration_seconds)

        digests = _as_tuple(self.artifact_digests, "artifact_digests")
        if len(digests) > _MAX_ARTIFACT_DIGESTS:
            raise WorkflowRollupError("artifact_digests: too_many_items")
        normalized_digests = tuple(_validate_digest(digest) for digest in digests)
        if normalized_digests != tuple(sorted(set(normalized_digests))):
            raise WorkflowRollupError("artifact_digests: not_sorted_unique")

        object.__setattr__(self, "outcome_counts", MappingProxyType(normalized_counts))
        object.__setattr__(self, "duration_seconds", normalized_duration)
        object.__setattr__(self, "artifact_digests", normalized_digests)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""
        return {
            "workflow_id": self.workflow_id,
            "run_count": self.run_count,
            "outcome_counts": dict(self.outcome_counts),
            "tool_call_count": self.tool_call_count,
            "duration_seconds": self.duration_seconds,
            "artifact_count": len(self.artifact_digests),
            "artifact_digests": list(self.artifact_digests),
        }


@dataclass(frozen=True, slots=True)
class WorkflowRollup:
    """Deterministic per-workflow rollup plus reconciled overall totals."""

    entries: tuple[WorkflowRollupEntry, ...]
    schema_version: str = WORKFLOW_ROLLUP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != WORKFLOW_ROLLUP_SCHEMA_VERSION
        ):
            raise WorkflowRollupError("schema_version: unsupported_version")
        entries = _as_tuple(self.entries, "entries")
        if len(entries) > _MAX_WORKFLOWS:
            raise WorkflowRollupError("entries: too_many_items")
        for entry in entries:
            if not isinstance(entry, WorkflowRollupEntry):
                raise WorkflowRollupError("entries: invalid_item")
        if entries != tuple(sorted(entries, key=lambda entry: entry.workflow_id)):
            raise WorkflowRollupError("entries: not_sorted")
        workflow_ids = tuple(entry.workflow_id for entry in entries)
        if len(set(workflow_ids)) != len(workflow_ids):
            raise WorkflowRollupError("entries: duplicate_workflow_id")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""
        outcome_totals = {
            name: sum(entry.outcome_counts[name] for entry in self.entries)
            for name in _OUTCOME_NAMES
        }
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "workflow_count": len(self.entries),
            "run_count": sum(entry.run_count for entry in self.entries),
            "outcome_counts": outcome_totals,
            "tool_call_count": sum(entry.tool_call_count for entry in self.entries),
            "duration_seconds": math.fsum(
                entry.duration_seconds for entry in self.entries
            ),
            "artifact_count": len(
                {digest for entry in self.entries for digest in entry.artifact_digests}
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        _assert_safe_payload(payload)
        return payload

    def to_json(self) -> str:
        """Return compact, deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Return deterministic metadata-only Markdown."""
        self.to_dict()
        lines = [
            "# Agent Workflow Rollup",
            "",
            "| Workflow | Runs | Success | Abstained | Review | Denied | Failed "
            "| Tool calls | Duration (s) | Artifacts |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for entry in self.entries:
            counts = entry.outcome_counts
            lines.append(
                f"| `{entry.workflow_id}` | {entry.run_count} | "
                f"{counts['success']} | {counts['abstained']} | "
                f"{counts['review_required']} | {counts['policy_denied']} | "
                f"{counts['failed']} | {entry.tool_call_count} | "
                f"{entry.duration_seconds:g} | {len(entry.artifact_digests)} |"
            )
        return "\n".join(lines) + "\n"


def rollup_workflows(events: Iterable[RunEvent]) -> WorkflowRollup:
    """Aggregate validated run events into per-workflow totals.

    Args:
        events: A bounded iterable of :class:`RunEvent` values.

    Returns:
        A :class:`WorkflowRollup` whose entries are ordered by workflow
        identifier and whose overall totals reconcile exactly with the
        aggregated events.

    Raises:
        WorkflowRollupError: If any item is not a :class:`RunEvent`, or an
            aggregated bound is exceeded.
    """
    if isinstance(events, (str, bytes, bytearray, Mapping)):
        raise WorkflowRollupError("events: invalid_iterable")
    try:
        iterator = iter(events)
    except TypeError:
        raise WorkflowRollupError("events: invalid_iterable") from None

    buckets: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(iterator):
        if index >= _MAX_EVENTS:
            raise WorkflowRollupError("events: too_many_items")
        if not isinstance(event, RunEvent):
            raise WorkflowRollupError("events: invalid_item")

        bucket = buckets.get(event.workflow_id)
        if bucket is None:
            if len(buckets) >= _MAX_WORKFLOWS:
                raise WorkflowRollupError("workflow_ids: too_many_items")
            bucket = {
                "run_count": 0,
                "outcome_counts": {name: 0 for name in _OUTCOME_NAMES},
                "tool_call_count": 0,
                "durations": [],
                "artifact_digests": set(),
            }
            buckets[event.workflow_id] = bucket

        bucket["run_count"] += 1
        if bucket["run_count"] > _MAX_EVENTS:
            raise WorkflowRollupError("run_count: total_out_of_range")
        bucket["outcome_counts"][event.outcome.outcome_class.value] += 1
        bucket["tool_call_count"] += event.tool_call_count
        if bucket["tool_call_count"] > _MAX_TOOL_CALLS:
            raise WorkflowRollupError("tool_call_count: total_out_of_range")
        bucket["durations"].append(event.duration_seconds)
        bucket["artifact_digests"].update(event.artifact_digests)
        if len(bucket["artifact_digests"]) > _MAX_ARTIFACT_DIGESTS:
            raise WorkflowRollupError("artifact_digests: too_many_items")

    entries = tuple(
        WorkflowRollupEntry(
            workflow_id=workflow_id,
            run_count=bucket["run_count"],
            outcome_counts=bucket["outcome_counts"],
            tool_call_count=bucket["tool_call_count"],
            duration_seconds=math.fsum(bucket["durations"]),
            artifact_digests=tuple(sorted(bucket["artifact_digests"])),
        )
        for workflow_id, bucket in sorted(buckets.items())
    )
    return WorkflowRollup(entries=entries)


__all__ = [
    "WORKFLOW_ROLLUP_SCHEMA_VERSION",
    "WorkflowRollup",
    "WorkflowRollupEntry",
    "WorkflowRollupError",
    "rollup_workflows",
]
