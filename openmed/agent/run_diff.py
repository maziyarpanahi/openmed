"""Deterministic, privacy-safe comparison of two agent run summaries.

A diff is computed only from validated :class:`RunSummary` values, so it can
contain nothing beyond bounded workflow identifiers, closed outcome names,
signed numeric deltas, and SHA-256 artifact digests. Prompts, tool arguments,
clinical text, paths, and credentials never reach the comparison layer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .outcomes import OutcomeClass
from .run_summary import RunSummary, _assert_safe_payload

RUN_DIFF_SCHEMA_VERSION = "openmed.agent.run_diff.v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OUTCOME_NAMES = tuple(sorted(outcome.value for outcome in OutcomeClass))

_MAX_WORKFLOWS = 1_024
_MAX_DIGESTS = 4_096
_MAX_EVENTS = 10_000
_MAX_TOOL_CALLS = 10_000_000
_MAX_DURATION_SECONDS = 31_536_000.0


class RunDiffError(ValueError):
    """Raised when run-diff input or direct construction is invalid.

    Messages name only a stable field and error code, never submitted values.
    """


def _sorted_unique(
    value: Any,
    field_name: str,
    pattern: re.Pattern[str],
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or type(value) not in (
        list,
        tuple,
    ):
        raise RunDiffError(f"{field_name}: invalid_sequence")
    items = tuple(value)
    if len(items) > maximum:
        raise RunDiffError(f"{field_name}: too_many_items")
    for item in items:
        if type(item) is not str or pattern.fullmatch(item) is None:
            raise RunDiffError(f"{field_name}: invalid_item")
    if items != tuple(sorted(set(items))):
        raise RunDiffError(f"{field_name}: not_sorted_unique")
    return items


def _signed_count(value: Any, field_name: str, maximum: int) -> int:
    if type(value) is not int or not -maximum <= value <= maximum:
        raise RunDiffError(f"{field_name}: invalid_delta")
    return value


def _format_signed(value: int | float) -> str:
    if value == 0:
        return "0"
    if isinstance(value, float):
        return f"{value:+g}"
    return f"{value:+d}"


@dataclass(frozen=True, slots=True)
class RunSummaryDiff:
    """Signed, metadata-only differences from one run summary to another.

    Deltas are ``after - before``. Added identifiers and digests appear only in
    the ``after`` summary; removed ones appear only in the ``before`` summary.
    Swapping the inputs negates every delta and swaps added with removed.
    """

    workflow_ids_added: tuple[str, ...]
    workflow_ids_removed: tuple[str, ...]
    outcome_count_deltas: Mapping[str, int]
    tool_call_count_delta: int
    duration_seconds_delta: float
    artifact_digests_added: tuple[str, ...]
    artifact_digests_removed: tuple[str, ...]
    schema_version: str = RUN_DIFF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != RUN_DIFF_SCHEMA_VERSION
        ):
            raise RunDiffError("schema_version: unsupported_version")

        added_ids = _sorted_unique(
            self.workflow_ids_added,
            "workflow_ids_added",
            _IDENTIFIER_RE,
            _MAX_WORKFLOWS,
        )
        removed_ids = _sorted_unique(
            self.workflow_ids_removed,
            "workflow_ids_removed",
            _IDENTIFIER_RE,
            _MAX_WORKFLOWS,
        )
        if set(added_ids) & set(removed_ids):
            raise RunDiffError("workflow_ids: added_and_removed")

        if not isinstance(self.outcome_count_deltas, Mapping):
            raise RunDiffError("outcome_count_deltas: invalid_mapping")
        if set(self.outcome_count_deltas) != set(_OUTCOME_NAMES):
            raise RunDiffError("outcome_count_deltas: invalid_keys")
        deltas = {
            name: _signed_count(
                self.outcome_count_deltas[name],
                f"outcome_count_deltas.{name}",
                _MAX_EVENTS,
            )
            for name in _OUTCOME_NAMES
        }

        _signed_count(
            self.tool_call_count_delta, "tool_call_count_delta", _MAX_TOOL_CALLS
        )

        duration = self.duration_seconds_delta
        if type(duration) not in (int, float):
            raise RunDiffError("duration_seconds_delta: invalid_number")
        if not -_MAX_DURATION_SECONDS <= duration <= _MAX_DURATION_SECONDS:
            raise RunDiffError("duration_seconds_delta: out_of_range")
        normalized_duration = float(duration)
        if not math.isfinite(normalized_duration):
            raise RunDiffError("duration_seconds_delta: out_of_range")
        if normalized_duration == 0.0:
            normalized_duration = 0.0

        added_digests = _sorted_unique(
            self.artifact_digests_added,
            "artifact_digests_added",
            _SHA256_RE,
            _MAX_DIGESTS,
        )
        removed_digests = _sorted_unique(
            self.artifact_digests_removed,
            "artifact_digests_removed",
            _SHA256_RE,
            _MAX_DIGESTS,
        )
        if set(added_digests) & set(removed_digests):
            raise RunDiffError("artifact_digests: added_and_removed")

        object.__setattr__(self, "workflow_ids_added", added_ids)
        object.__setattr__(self, "workflow_ids_removed", removed_ids)
        object.__setattr__(self, "outcome_count_deltas", MappingProxyType(deltas))
        object.__setattr__(self, "duration_seconds_delta", normalized_duration)
        object.__setattr__(self, "artifact_digests_added", added_digests)
        object.__setattr__(self, "artifact_digests_removed", removed_digests)

    @property
    def changed(self) -> bool:
        """Return whether any compared field differs between the summaries."""
        return bool(
            self.workflow_ids_added
            or self.workflow_ids_removed
            or any(self.outcome_count_deltas.values())
            or self.tool_call_count_delta
            or self.duration_seconds_delta
            or self.artifact_digests_added
            or self.artifact_digests_removed
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""
        payload = {
            "schema_version": self.schema_version,
            "changed": self.changed,
            "workflow_ids": {
                "added": list(self.workflow_ids_added),
                "removed": list(self.workflow_ids_removed),
            },
            "outcome_count_deltas": dict(self.outcome_count_deltas),
            "tool_call_count_delta": self.tool_call_count_delta,
            "duration_seconds_delta": self.duration_seconds_delta,
            "artifact_digests": {
                "added": list(self.artifact_digests_added),
                "removed": list(self.artifact_digests_removed),
            },
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
            "# Agent Run Summary Diff",
            "",
            f"Changed: {'yes' if self.changed else 'no'}",
            "",
            "## Workflows",
            "",
            "| Change | Workflow |",
            "| --- | --- |",
            *(f"| added | `{value}` |" for value in self.workflow_ids_added),
            *(f"| removed | `{value}` |" for value in self.workflow_ids_removed),
            "",
            "## Outcomes",
            "",
            "| Outcome | Delta |",
            "| --- | ---: |",
            *(
                f"| `{outcome}` | {_format_signed(delta)} |"
                for outcome, delta in self.outcome_count_deltas.items()
            ),
            "",
            "## Execution",
            "",
            "| Metric | Delta |",
            "| --- | ---: |",
            f"| Tool calls | {_format_signed(self.tool_call_count_delta)} |",
            f"| Duration (seconds) | {_format_signed(self.duration_seconds_delta)} |",
            "",
            "## Artifacts",
            "",
            "| Change | SHA-256 |",
            "| --- | --- |",
            *(f"| added | `{value}` |" for value in self.artifact_digests_added),
            *(f"| removed | `{value}` |" for value in self.artifact_digests_removed),
        ]
        return "\n".join(lines) + "\n"


def diff_run_summaries(before: RunSummary, after: RunSummary) -> RunSummaryDiff:
    """Compare two validated run summaries.

    Args:
        before: The baseline summary.
        after: The summary compared against the baseline.

    Returns:
        A :class:`RunSummaryDiff` whose deltas are ``after - before``.

    Raises:
        RunDiffError: If either input is not a :class:`RunSummary`.
    """
    if not isinstance(before, RunSummary):
        raise RunDiffError("before: invalid_summary")
    if not isinstance(after, RunSummary):
        raise RunDiffError("after: invalid_summary")

    before_ids = set(before.workflow_ids)
    after_ids = set(after.workflow_ids)
    before_digests = set(before.artifact_digests)
    after_digests = set(after.artifact_digests)

    return RunSummaryDiff(
        workflow_ids_added=tuple(sorted(after_ids - before_ids)),
        workflow_ids_removed=tuple(sorted(before_ids - after_ids)),
        outcome_count_deltas={
            name: after.outcome_counts[name] - before.outcome_counts[name]
            for name in _OUTCOME_NAMES
        },
        tool_call_count_delta=after.tool_call_count - before.tool_call_count,
        duration_seconds_delta=after.duration_seconds - before.duration_seconds,
        artifact_digests_added=tuple(sorted(after_digests - before_digests)),
        artifact_digests_removed=tuple(sorted(before_digests - after_digests)),
    )


__all__ = [
    "RUN_DIFF_SCHEMA_VERSION",
    "RunDiffError",
    "RunSummaryDiff",
    "diff_run_summaries",
]
