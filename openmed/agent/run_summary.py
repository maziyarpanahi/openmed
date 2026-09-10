"""Privacy-safe, deterministic summaries of agent runs.

Only bounded identifiers, closed workflow outcomes, counts, durations, and
artifact digests cross this boundary. Prompts, tool payloads, evidence text,
paths, credentials, and exception text are never accepted or rendered.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .outcomes import OutcomeClass, WorkflowOutcome

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_OUTCOME_NAMES = tuple(sorted(outcome.value for outcome in OutcomeClass))

_MAX_EVENTS = 10_000
_MAX_WORKFLOWS = 1_024
_MAX_TOOL_CALLS = 10_000_000
_MAX_DURATION_SECONDS = 31_536_000.0
_MAX_EVENT_DIGESTS = 128
_MAX_SUMMARY_DIGESTS = 4_096


class RunSummaryError(ValueError):
    """Base error for invalid run-summary input."""


class RunSummaryPrivacyError(RunSummaryError):
    """Raised when unsafe data would cross the summary boundary."""


def _validate_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RunSummaryError(f"{field_name}: invalid_identifier")
    return value


def _validate_digest(value: Any) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RunSummaryError("artifact_digests: invalid_digest")
    return value


def _validate_count(value: Any, field_name: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise RunSummaryError(f"{field_name}: invalid_count")
    return value


def _validate_duration(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunSummaryError("duration_seconds: invalid_number")
    normalized = float(value)
    if (
        not math.isfinite(normalized)
        or normalized < 0.0
        or normalized > _MAX_DURATION_SECONDS
    ):
        raise RunSummaryError("duration_seconds: out_of_range")
    return normalized


def _as_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RunSummaryError(f"{field_name}: invalid_sequence")
    try:
        return tuple(value)
    except (TypeError, ValueError):
        pass
    raise RunSummaryError(f"{field_name}: invalid_sequence")


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Safe metadata describing one agent workflow execution."""

    workflow_id: str
    outcome: WorkflowOutcome
    tool_call_count: int = 0
    duration_seconds: float = 0.0
    artifact_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.workflow_id, "workflow_id")
        if not isinstance(self.outcome, WorkflowOutcome):
            raise RunSummaryError("outcome: invalid_type")
        _validate_count(self.tool_call_count, "tool_call_count", _MAX_TOOL_CALLS)
        object.__setattr__(
            self, "duration_seconds", _validate_duration(self.duration_seconds)
        )

        digests = _as_tuple(self.artifact_digests, "artifact_digests")
        if len(digests) > _MAX_EVENT_DIGESTS:
            raise RunSummaryError("artifact_digests: too_many_items")
        normalized = tuple(_validate_digest(digest) for digest in digests)
        if len(set(normalized)) != len(normalized):
            raise RunSummaryError("artifact_digests: duplicate_item")
        object.__setattr__(self, "artifact_digests", normalized)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Deterministic, metadata-only summary of bounded agent events."""

    workflow_ids: tuple[str, ...]
    outcome_counts: Mapping[str, int]
    tool_call_count: int
    duration_seconds: float
    artifact_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        workflow_ids = _as_tuple(self.workflow_ids, "workflow_ids")
        if len(workflow_ids) > _MAX_WORKFLOWS:
            raise RunSummaryError("workflow_ids: too_many_items")
        normalized_ids = tuple(
            _validate_identifier(value, "workflow_ids") for value in workflow_ids
        )
        if normalized_ids != tuple(sorted(set(normalized_ids))):
            raise RunSummaryError("workflow_ids: not_sorted_unique")

        if not isinstance(self.outcome_counts, Mapping):
            raise RunSummaryError("outcome_counts: invalid_mapping")
        if set(self.outcome_counts) != set(_OUTCOME_NAMES):
            raise RunSummaryError("outcome_counts: invalid_keys")
        normalized_counts = {
            name: _validate_count(
                self.outcome_counts[name], f"outcome_counts.{name}", _MAX_EVENTS
            )
            for name in _OUTCOME_NAMES
        }
        if sum(normalized_counts.values()) > _MAX_EVENTS:
            raise RunSummaryError("outcome_counts: total_out_of_range")

        _validate_count(self.tool_call_count, "tool_call_count", _MAX_TOOL_CALLS)
        normalized_duration = _validate_duration(self.duration_seconds)

        digests = _as_tuple(self.artifact_digests, "artifact_digests")
        if len(digests) > _MAX_SUMMARY_DIGESTS:
            raise RunSummaryError("artifact_digests: too_many_items")
        normalized_digests = tuple(_validate_digest(digest) for digest in digests)
        if normalized_digests != tuple(sorted(set(normalized_digests))):
            raise RunSummaryError("artifact_digests: not_sorted_unique")

        object.__setattr__(self, "workflow_ids", normalized_ids)
        object.__setattr__(self, "outcome_counts", MappingProxyType(normalized_counts))
        object.__setattr__(self, "duration_seconds", normalized_duration)
        object.__setattr__(self, "artifact_digests", normalized_digests)

    @classmethod
    def from_events(cls, events: Iterable[RunEvent]) -> "RunSummary":
        """Aggregate a bounded iterable of validated run events."""
        if isinstance(events, (str, bytes, bytearray, Mapping)):
            raise RunSummaryError("events: invalid_iterable")
        try:
            iterator = iter(events)
        except TypeError:
            raise RunSummaryError("events: invalid_iterable") from None

        workflow_ids: set[str] = set()
        outcome_counts = {outcome: 0 for outcome in _OUTCOME_NAMES}
        tool_call_count = 0
        durations: list[float] = []
        artifact_digests: set[str] = set()

        for index, event in enumerate(iterator):
            if index >= _MAX_EVENTS:
                raise RunSummaryError("events: too_many_items")
            if not isinstance(event, RunEvent):
                raise RunSummaryError("events: invalid_item")

            workflow_ids.add(event.workflow_id)
            if len(workflow_ids) > _MAX_WORKFLOWS:
                raise RunSummaryError("workflow_ids: too_many_items")
            outcome_counts[event.outcome.outcome_class.value] += 1
            tool_call_count += event.tool_call_count
            if tool_call_count > _MAX_TOOL_CALLS:
                raise RunSummaryError("tool_call_count: total_out_of_range")
            durations.append(event.duration_seconds)
            artifact_digests.update(event.artifact_digests)
            if len(artifact_digests) > _MAX_SUMMARY_DIGESTS:
                raise RunSummaryError("artifact_digests: too_many_items")

        duration_seconds = math.fsum(durations)
        if duration_seconds > _MAX_DURATION_SECONDS:
            raise RunSummaryError("duration_seconds: total_out_of_range")

        return cls(
            workflow_ids=tuple(sorted(workflow_ids)),
            outcome_counts=outcome_counts,
            tool_call_count=tool_call_count,
            duration_seconds=duration_seconds,
            artifact_digests=tuple(sorted(artifact_digests)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""
        payload = {
            "workflow_ids": list(self.workflow_ids),
            "outcome_counts": dict(self.outcome_counts),
            "tool_call_count": self.tool_call_count,
            "duration_seconds": self.duration_seconds,
            "artifact_digests": list(self.artifact_digests),
        }
        _assert_safe_payload(payload)
        return payload

    def to_json(self) -> str:
        """Return compact, deterministic JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_markdown(self) -> str:
        """Return deterministic metadata-only Markdown."""
        lines = [
            "# Agent Run Summary",
            "",
            "## Workflows",
            "",
            "| Workflow |",
            "| --- |",
            *(f"| `{workflow_id}` |" for workflow_id in self.workflow_ids),
            "",
            "## Outcomes",
            "",
            "| Outcome | Count |",
            "| --- | ---: |",
            *(
                f"| `{outcome}` | {count} |"
                for outcome, count in self.outcome_counts.items()
            ),
            "",
            "## Execution",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Tool calls | {self.tool_call_count} |",
            f"| Duration (seconds) | {self.duration_seconds:g} |",
            "",
            "## Artifacts",
            "",
            "| SHA-256 |",
            "| --- |",
            *(f"| `{digest}` |" for digest in self.artifact_digests),
        ]
        return "\n".join(lines) + "\n"


def _assert_safe_payload(payload: Any, *, location: str = "root") -> None:
    """Reject values outside the closed summary serialization vocabulary."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if type(key) is not str:
                raise RunSummaryPrivacyError(f"{location}: invalid_mapping_key")
            _assert_safe_payload(value, location="mapping_value")
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            _assert_safe_payload(value, location="sequence_item")
        return
    if payload is None or type(payload) in {bool, int}:
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise RunSummaryPrivacyError(f"{location}: non_finite_number")
        return
    if type(payload) is str:
        if (
            _IDENTIFIER_RE.fullmatch(payload) is None
            and _SHA256_RE.fullmatch(payload) is None
        ):
            raise RunSummaryPrivacyError(f"{location}: unsafe_string")
        return
    raise RunSummaryPrivacyError(f"{location}: forbidden_type")


__all__ = [
    "RunEvent",
    "RunSummary",
    "RunSummaryError",
    "RunSummaryPrivacyError",
]
