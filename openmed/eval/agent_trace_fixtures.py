"""Strict loader for synthetic governed-agent trace fixtures.

The fixture contract is intentionally metadata-only. It accepts opaque run and
action identifiers, a digest binding the omitted trace content, and a closed
expected workflow outcome. It never accepts prompts, clinical text, tool
arguments, credentials, paths, or arbitrary annotations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from openmed.agent.correlation import (
    ActionId,
    CorrelationIdError,
    RunId,
)
from openmed.agent.outcomes import OutcomeClass, OutcomeError, WorkflowOutcome

AGENT_TRACE_FIXTURE_SCHEMA_VERSION: Final = "openmed.eval.agent_trace_fixture.v1"
MAX_AGENT_TRACE_FIXTURE_BYTES: Final = 1_048_576
MAX_AGENT_TRACE_FIXTURE_CASES: Final = 10_000

DEFAULT_AGENT_TRACE_FIXTURE_PATH: Final = (
    Path(__file__).with_name("fixtures") / "governed_agent_traces.jsonl"
)

_CASE_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "scenario",
        "run_id",
        "action_ids",
        "trace_digest",
        "expected_outcome",
        "expected_reason_code",
    }
)
_ORDERED_FIELDS = (
    "schema_version",
    "case_id",
    "scenario",
    "run_id",
    "action_ids",
    "trace_digest",
    "expected_outcome",
    "expected_reason_code",
)


class GovernedAgentScenario(str, Enum):
    """Closed set of scenarios represented by the synthetic fixture pack."""

    READ_ONLY_ALLOW = "read_only_allow"
    MINIMUM_DATA_PROJECTION = "minimum_data_projection"
    MISSING_CAPABILITY = "missing_capability"
    PURPOSE_MISMATCH = "purpose_mismatch"
    EXPIRED_CONSENT = "expired_consent"
    HUMAN_REVIEW = "human_review"
    BOUNDED_FAILURE = "bounded_failure"


class AgentTraceFixtureError(ValueError):
    """Raised when a governed-agent fixture fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public schema field associated with the failure.
        line_number: Optional one-based JSONL line containing the failure.
    """

    def __init__(
        self,
        code: str,
        field_name: str | None = None,
        line_number: int | None = None,
    ) -> None:
        self.code = code
        self.field_name = field_name
        self.line_number = line_number
        parts = [f"line {line_number}" if line_number is not None else None]
        parts.extend((field_name, code))
        super().__init__(": ".join(part for part in parts if part is not None))


@dataclass(frozen=True, slots=True)
class GovernedAgentTraceFixture:
    """One synthetic, metadata-only governed-agent trace expectation."""

    case_id: str
    scenario: GovernedAgentScenario
    run_id: RunId
    action_ids: tuple[ActionId, ...]
    trace_digest: str
    expected_outcome: OutcomeClass
    expected_reason_code: str
    schema_version: str = AGENT_TRACE_FIXTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != AGENT_TRACE_FIXTURE_SCHEMA_VERSION
        ):
            raise AgentTraceFixtureError("invalid_schema_version", "schema_version")
        if type(self.case_id) is not str or _CASE_ID_RE.fullmatch(self.case_id) is None:
            raise AgentTraceFixtureError("invalid_case_id", "case_id")
        if not isinstance(self.scenario, GovernedAgentScenario):
            raise AgentTraceFixtureError("unknown_scenario", "scenario")
        if type(self.run_id) is not RunId:
            raise AgentTraceFixtureError("invalid_identifier", "run_id")
        if type(self.action_ids) is not tuple or not self.action_ids:
            raise AgentTraceFixtureError("invalid_action_ids", "action_ids")
        if any(type(action_id) is not ActionId for action_id in self.action_ids):
            raise AgentTraceFixtureError("invalid_identifier", "action_ids")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise AgentTraceFixtureError("duplicate_action_id", "action_ids")
        if (
            type(self.trace_digest) is not str
            or _DIGEST_RE.fullmatch(self.trace_digest) is None
        ):
            raise AgentTraceFixtureError("invalid_digest", "trace_digest")
        try:
            WorkflowOutcome(
                outcome_class=self.expected_outcome,
                reason_code=self.expected_reason_code,
            )
        except OutcomeError:
            raise AgentTraceFixtureError(
                "invalid_expected_outcome", "expected_outcome"
            ) from None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernedAgentTraceFixture":
        """Build a fixture from an exact metadata-only mapping."""

        values = _read_exact_mapping(data)
        try:
            scenario = GovernedAgentScenario(values["scenario"])
        except (TypeError, ValueError):
            raise AgentTraceFixtureError("unknown_scenario", "scenario") from None
        if type(values["action_ids"]) not in (list, tuple):
            raise AgentTraceFixtureError("invalid_action_ids", "action_ids")
        try:
            run_id = RunId.parse(values["run_id"])
            action_ids = tuple(ActionId.parse(value) for value in values["action_ids"])
        except CorrelationIdError:
            raise AgentTraceFixtureError("invalid_identifier") from None
        try:
            outcome = WorkflowOutcome.from_dict(
                {
                    "outcome_class": values["expected_outcome"],
                    "reason_code": values["expected_reason_code"],
                }
            )
        except OutcomeError:
            raise AgentTraceFixtureError("invalid_expected_outcome") from None
        return cls(
            schema_version=values["schema_version"],
            case_id=values["case_id"],
            scenario=scenario,
            run_id=run_id,
            action_ids=action_ids,
            trace_digest=values["trace_digest"],
            expected_outcome=outcome.outcome_class,
            expected_reason_code=outcome.reason_code,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata-only JSON-compatible data."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "scenario": self.scenario.value,
            "run_id": self.run_id.serialize(),
            "action_ids": [action_id.serialize() for action_id in self.action_ids],
            "trace_digest": self.trace_digest,
            "expected_outcome": self.expected_outcome.value,
            "expected_reason_code": self.expected_reason_code,
        }
        return {field_name: values[field_name] for field_name in _ORDERED_FIELDS}


def load_governed_agent_trace_fixtures(
    path: str | Path | None = None,
) -> tuple[GovernedAgentTraceFixture, ...]:
    """Load and validate a deterministic governed-agent JSONL fixture pack.

    Args:
        path: Optional JSONL path. The bundled synthetic fixture is used when
            omitted.

    Returns:
        Fixtures in their declared JSONL order.

    Raises:
        AgentTraceFixtureError: If reading or validation fails. Errors contain
            only stable codes, public field names, and line numbers.
    """

    fixture_path = (
        DEFAULT_AGENT_TRACE_FIXTURE_PATH if path is None else _coerce_path(path)
    )
    try:
        payload = fixture_path.read_bytes()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise AgentTraceFixtureError("fixture_read_failed") from None
    if len(payload) > MAX_AGENT_TRACE_FIXTURE_BYTES:
        raise AgentTraceFixtureError("fixture_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise AgentTraceFixtureError("invalid_encoding") from None

    fixtures: list[GovernedAgentTraceFixture] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise AgentTraceFixtureError("empty_line", line_number=line_number)
        if len(fixtures) >= MAX_AGENT_TRACE_FIXTURE_CASES:
            raise AgentTraceFixtureError("too_many_cases", line_number=line_number)
        fixture = _parse_fixture_line(line, line_number)
        if fixture.case_id in case_ids:
            raise AgentTraceFixtureError("duplicate_case_id", "case_id", line_number)
        case_ids.add(fixture.case_id)
        fixtures.append(fixture)
    if not fixtures:
        raise AgentTraceFixtureError("empty_fixture")
    return tuple(fixtures)


def _coerce_path(path: str | Path) -> Path:
    if type(path) is str or isinstance(path, Path):
        return Path(path)
    raise AgentTraceFixtureError("invalid_path")


def _parse_fixture_line(line: str, line_number: int) -> GovernedAgentTraceFixture:
    error_details: tuple[str, str | None] | None = None
    fixture: GovernedAgentTraceFixture | None = None
    try:
        decoded = json.loads(line, object_pairs_hook=_strict_json_object)
        fixture = GovernedAgentTraceFixture.from_dict(decoded)
    except (KeyboardInterrupt, SystemExit):
        raise
    except AgentTraceFixtureError as error:
        error_details = (error.code, error.field_name)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        error_details = ("invalid_json", None)
    if error_details is not None:
        raise AgentTraceFixtureError(*error_details, line_number)
    assert fixture is not None
    return fixture


def _read_exact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise AgentTraceFixtureError("not_a_mapping")
    try:
        fields = set(data)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise AgentTraceFixtureError("not_a_mapping") from None
    if fields - _CASE_FIELDS:
        raise AgentTraceFixtureError("unknown_field")
    if _CASE_FIELDS - fields:
        raise AgentTraceFixtureError("missing_field")
    try:
        return {field_name: data[field_name] for field_name in _CASE_FIELDS}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise AgentTraceFixtureError("unreadable_mapping") from None


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AgentTraceFixtureError("duplicate_field")
        result[key] = value
    return result


__all__ = [
    "AGENT_TRACE_FIXTURE_SCHEMA_VERSION",
    "DEFAULT_AGENT_TRACE_FIXTURE_PATH",
    "MAX_AGENT_TRACE_FIXTURE_BYTES",
    "MAX_AGENT_TRACE_FIXTURE_CASES",
    "AgentTraceFixtureError",
    "GovernedAgentScenario",
    "GovernedAgentTraceFixture",
    "load_governed_agent_trace_fixtures",
]
