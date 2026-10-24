"""Strict, offline golden vectors for metadata-only agent run summaries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from openmed.agent.outcomes import OUTCOME_SCHEMA_VERSION, WorkflowOutcome
from openmed.agent.run_commitment import (
    RUN_COMMITMENT_VERSION,
    compute_run_summary_commitment,
)
from openmed.agent.run_summary import RUN_SUMMARY_SCHEMA_VERSION, RunEvent, RunSummary

AGENT_RUN_SUMMARY_FIXTURE_VERSION: Final = "openmed.eval.agent_run_summary_fixture.v1"
MAX_RUN_SUMMARY_FIXTURE_BYTES: Final = 1_048_576
MAX_RUN_SUMMARY_FIXTURE_EVENTS: Final = 128
DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH: Final = (
    Path(__file__).with_name("fixtures") / "agent_run_summaries.json"
)

_CASE_IDS = frozenset(
    {"empty", "success", "abstention", "denial", "review", "failure", "mixed"}
)
_WORKFLOW_ID = re.compile(r"wf_[0-9a-f]{32}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_VERSIONS = {
    "schema_version": AGENT_RUN_SUMMARY_FIXTURE_VERSION,
    "run_summary_schema_version": RUN_SUMMARY_SCHEMA_VERSION,
    "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
    "commitment_version": RUN_COMMITMENT_VERSION,
}
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "events",
        "expected_json",
        "expected_markdown_digest",
        "expected_commitment",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "workflow_id",
        "outcome",
        "tool_call_count",
        "duration_seconds",
        "artifact_digests",
    }
)
_OUTCOME_FIELDS = frozenset({"schema_version", "outcome_class", "reason_code"})


class AgentRunSummaryFixtureError(ValueError):
    """Report a fixed validation code without retaining input content.

    Args:
        code: Controlled validation reason.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class AgentRunSummaryFixture:
    """One validated synthetic vector with committed serialization expectations.

    Args:
        case_id: One of the seven declared metadata-only scenarios.
        events: Bounded, typed events with opaque synthetic workflow IDs.
        expected_json: Exact canonical summary JSON.
        expected_markdown_digest: SHA-256 of the canonical Markdown bytes.
        expected_commitment: Domain-separated run-summary commitment.
    """

    case_id: str
    events: tuple[RunEvent, ...]
    expected_json: str
    expected_markdown_digest: str
    expected_commitment: str

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or self.case_id not in _CASE_IDS:
            raise AgentRunSummaryFixtureError("invalid_case_id")
        if (
            type(self.events) is not tuple
            or len(self.events) > MAX_RUN_SUMMARY_FIXTURE_EVENTS
        ):
            raise AgentRunSummaryFixtureError("invalid_events")
        for event in self.events:
            if (
                type(event) is not RunEvent
                or _WORKFLOW_ID.fullmatch(event.workflow_id) is None
            ):
                raise AgentRunSummaryFixtureError("invalid_event")
        for value in (self.expected_markdown_digest, self.expected_commitment):
            if type(value) is not str or _DIGEST.fullmatch(value) is None:
                raise AgentRunSummaryFixtureError("invalid_digest")
        if (
            type(self.expected_json) is not str
            or len(self.expected_json) > MAX_RUN_SUMMARY_FIXTURE_BYTES
        ):
            raise AgentRunSummaryFixtureError("invalid_expected_json")
        summary = self.build_summary()
        if self.expected_json != summary.to_json():
            raise AgentRunSummaryFixtureError("stale_json")
        markdown_digest = (
            "sha256:"
            + hashlib.sha256(summary.to_markdown().encode("utf-8")).hexdigest()
        )
        if self.expected_markdown_digest != markdown_digest:
            raise AgentRunSummaryFixtureError("stale_markdown")
        if self.expected_commitment != compute_run_summary_commitment(summary):
            raise AgentRunSummaryFixtureError("stale_commitment")

    def build_summary(self) -> RunSummary:
        """Rebuild the summary solely from validated metadata events.

        Returns:
            A deterministic, metadata-only summary.

        Raises:
            AgentRunSummaryFixtureError: If aggregation cannot satisfy bounds.
        """
        failed = False
        try:
            summary = RunSummary.from_events(self.events)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            failed = True
        if failed:
            raise AgentRunSummaryFixtureError("invalid_summary")
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return the exact synthetic fixture fields for deterministic hashing."""
        return {
            "case_id": self.case_id,
            "events": [
                {
                    "workflow_id": event.workflow_id,
                    "outcome": event.outcome.to_dict(),
                    "tool_call_count": event.tool_call_count,
                    "duration_seconds": event.duration_seconds,
                    "artifact_digests": list(event.artifact_digests),
                }
                for event in self.events
            ],
            "expected_json": self.expected_json,
            "expected_markdown_digest": self.expected_markdown_digest,
            "expected_commitment": self.expected_commitment,
        }


def load_agent_run_summary_fixtures(
    path: str | Path | None = None,
) -> tuple[AgentRunSummaryFixture, ...]:
    """Load bounded, committed golden vectors without models or network access.

    Args:
        path: Optional caller-owned fixture file; otherwise use bundled vectors.

    Returns:
        Validated vectors in their committed order.

    Raises:
        AgentRunSummaryFixtureError: For malformed, unsafe or stale vectors.
            Errors expose controlled codes only, never input values or paths.
    """
    if path is not None and type(path) is not str and not isinstance(path, Path):
        raise AgentRunSummaryFixtureError("invalid_path")
    read_failed = False
    try:
        fixture_path = (
            DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH if path is None else Path(path)
        )
        with fixture_path.open("rb") as stream:
            payload = stream.read(MAX_RUN_SUMMARY_FIXTURE_BYTES + 1)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        read_failed = True
    if read_failed:
        raise AgentRunSummaryFixtureError("fixture_read_failed")
    if len(payload) > MAX_RUN_SUMMARY_FIXTURE_BYTES:
        raise AgentRunSummaryFixtureError("fixture_too_large")
    error_code = None
    try:
        document = json.loads(
            payload, object_pairs_hook=_strict_object, parse_constant=_reject_constant
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except AgentRunSummaryFixtureError as exc:
        error_code = exc.code
    except Exception:
        error_code = "invalid_json"
    if error_code is not None:
        raise AgentRunSummaryFixtureError(error_code)
    _exact_fields(document, frozenset(_VERSIONS) | {"cases"})
    if any(
        type(document[key]) is not str or document[key] != version
        for key, version in _VERSIONS.items()
    ):
        raise AgentRunSummaryFixtureError("unsupported_version")
    cases = document["cases"]
    if type(cases) is not list or not 1 <= len(cases) <= len(_CASE_IDS):
        raise AgentRunSummaryFixtureError("invalid_cases")
    vectors = tuple(_case_from_dict(case) for case in cases)
    if len({vector.case_id for vector in vectors}) != len(vectors):
        raise AgentRunSummaryFixtureError("duplicate_case_id")
    return vectors


def _case_from_dict(case: Any) -> AgentRunSummaryFixture:
    _exact_fields(case, _CASE_FIELDS)
    events = case["events"]
    if type(events) is not list or len(events) > MAX_RUN_SUMMARY_FIXTURE_EVENTS:
        raise AgentRunSummaryFixtureError("invalid_events")
    return AgentRunSummaryFixture(
        case_id=case["case_id"],
        events=tuple(_event_from_dict(event) for event in events),
        expected_json=case["expected_json"],
        expected_markdown_digest=case["expected_markdown_digest"],
        expected_commitment=case["expected_commitment"],
    )


def _event_from_dict(event: Any) -> RunEvent:
    _exact_fields(event, _EVENT_FIELDS)
    _exact_fields(event["outcome"], _OUTCOME_FIELDS)
    if (
        type(event["workflow_id"]) is not str
        or _WORKFLOW_ID.fullmatch(event["workflow_id"]) is None
    ):
        raise AgentRunSummaryFixtureError("invalid_workflow_id")
    if type(event["artifact_digests"]) is not list:
        raise AgentRunSummaryFixtureError("invalid_artifact_digests")
    failed = False
    try:
        result = RunEvent(
            workflow_id=event["workflow_id"],
            outcome=WorkflowOutcome.from_dict(event["outcome"]),
            tool_call_count=event["tool_call_count"],
            duration_seconds=event["duration_seconds"],
            artifact_digests=tuple(event["artifact_digests"]),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        failed = True
    if failed:
        raise AgentRunSummaryFixtureError("invalid_event")
    return result


def _exact_fields(value: Any, fields: frozenset[str]) -> None:
    if type(value) is not dict:
        raise AgentRunSummaryFixtureError("invalid_object")
    if set(value) - fields:
        raise AgentRunSummaryFixtureError("unknown_field")
    if fields - set(value):
        raise AgentRunSummaryFixtureError("missing_field")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AgentRunSummaryFixtureError("duplicate_field")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise AgentRunSummaryFixtureError("non_finite_number")


__all__ = [
    "AGENT_RUN_SUMMARY_FIXTURE_VERSION",
    "DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH",
    "MAX_RUN_SUMMARY_FIXTURE_BYTES",
    "MAX_RUN_SUMMARY_FIXTURE_EVENTS",
    "AgentRunSummaryFixture",
    "AgentRunSummaryFixtureError",
    "load_agent_run_summary_fixtures",
]
