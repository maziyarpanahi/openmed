"""Privacy-safe summaries of agent runs.

The summary intentionally accepts only structured metadata and never publishes
prompts, tool arguments, tool outputs, evidence text, filesystem paths, or
credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")

_ALLOWED_OUTCOMES = frozenset(
    {
        "success",
        "failure",
        "abstained",
    }
)


class RunSummaryError(ValueError):
    """Base error for invalid run-summary input."""


class RunSummaryPrivacyError(RunSummaryError):
    """Raised when unsafe data would cross the summary boundary."""


def _validate_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RunSummaryError(f"{field_name} must be a safe identifier")
    return value


def _validate_digest(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RunSummaryError("artifact digest must be a sha256 digest")
    return value


def _validate_duration(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunSummaryError("duration_seconds must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise RunSummaryError(
            "duration_seconds must be a finite non-negative number"
        )
    return value


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Safe metadata describing one agent-run event.

    Event payloads intentionally contain no arbitrary text fields.
    """

    workflow_id: str
    outcome: str
    tool_call_count: int = 0
    duration_seconds: float = 0.0
    artifact_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.workflow_id, "workflow_id")

        if self.outcome not in _ALLOWED_OUTCOMES:
            raise RunSummaryError(
                f"outcome must be one of {sorted(_ALLOWED_OUTCOMES)!r}"
            )

        if isinstance(self.tool_call_count, bool) or not isinstance(
            self.tool_call_count, int
        ):
            raise RunSummaryError("tool_call_count must be an integer")
        if self.tool_call_count < 0:
            raise RunSummaryError("tool_call_count must be non-negative")

        _validate_duration(self.duration_seconds)

        digests = tuple(self.artifact_digests)
        for digest in digests:
            _validate_digest(digest)

        object.__setattr__(self, "artifact_digests", digests)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Deterministic, metadata-only summary of an agent run."""

    workflow_ids: tuple[str, ...]
    outcome_counts: Mapping[str, int]
    tool_call_count: int
    duration_seconds: float
    artifact_digests: tuple[str, ...]

    @classmethod
    def from_events(cls, events: Iterable[RunEvent]) -> "RunSummary":
        workflow_ids: set[str] = set()
        outcome_counts = {outcome: 0 for outcome in sorted(_ALLOWED_OUTCOMES)}
        tool_call_count = 0
        duration_seconds = 0.0
        artifact_digests: set[str] = set()

        for event in events:
            if not isinstance(event, RunEvent):
                raise RunSummaryError("events must contain RunEvent instances")

            workflow_ids.add(event.workflow_id)
            outcome_counts[event.outcome] += 1
            tool_call_count += event.tool_call_count
            duration_seconds += float(event.duration_seconds)
            artifact_digests.update(event.artifact_digests)

        return cls(
            workflow_ids=tuple(sorted(workflow_ids)),
            outcome_counts=dict(sorted(outcome_counts.items())),
            tool_call_count=tool_call_count,
            duration_seconds=duration_seconds,
            artifact_digests=tuple(sorted(artifact_digests)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""

        payload = {
            "workflow_ids": list(self.workflow_ids),
            "outcome_counts": dict(sorted(self.outcome_counts.items())),
            "tool_call_count": self.tool_call_count,
            "duration_seconds": self.duration_seconds,
            "artifact_digests": list(self.artifact_digests),
        }

        _assert_safe_payload(payload)
        return payload

    def to_json(self) -> str:
        """Return deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_markdown(self) -> str:
        """Return deterministic metadata-only Markdown."""

        lines = [
            "# Agent Run Summary",
            "",
            "## Workflows",
            "",
            "| Workflow |",
            "| --- |",
        ]

        for workflow_id in self.workflow_ids:
            lines.append(f"| `{workflow_id}` |")

        lines.extend(
            [
                "",
                "## Outcomes",
                "",
                "| Outcome | Count |",
                "| --- | ---: |",
            ]
        )

        for outcome, count in sorted(self.outcome_counts.items()):
            lines.append(f"| `{outcome}` | {count} |")

        lines.extend(
            [
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
            ]
        )

        for digest in self.artifact_digests:
            lines.append(f"| `{digest}` |")

        return "\n".join(lines) + "\n"


def _assert_safe_payload(
    payload: Any,
    *,
    path: str = "$",
) -> None:
    """Reject unsafe values before publication."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise RunSummaryPrivacyError(
                    f"unsafe mapping key at {path}"
                )
            _assert_safe_payload(value, path=f"{path}.{key}")
        return

    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_safe_payload(value, path=f"{path}[{index}]")
        return

    if payload is None or isinstance(payload, (bool, int)):
        return

    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise RunSummaryPrivacyError(
                f"non-finite number at {path}"
            )
        return

    if isinstance(payload, str):
        if len(payload) > 128:
            raise RunSummaryPrivacyError(
                f"string value is too long at {path}"
            )
        if not _IDENTIFIER_RE.fullmatch(payload):
            raise RunSummaryPrivacyError(
                f"unsafe string value at {path}"
            )
        return

    raise RunSummaryPrivacyError(
        f"forbidden value type at {path}"
    )


__all__ = [
    "RunEvent",
    "RunSummary",
    "RunSummaryError",
    "RunSummaryPrivacyError",
]