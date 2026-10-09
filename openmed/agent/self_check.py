"""Offline agreement checks for the bundled agent governance contracts.

Only closed report fields leave this module. Inputs are bundled synthetic
metadata; no user path, environment, clock, model, credential or tool is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from . import correlation, outcomes, run_commitment, run_summary, timing

AGENT_SELF_CHECK_VERSION: Final = "openmed.agent.self_check.v1"
_VERSIONS: Final = {
    "correlation": "openmed.agent.correlation.v1",
    "outcome": "openmed.agent.outcome.v1",
    "run_summary": "openmed.agent.run_summary.v1",
    "commitment": "openmed.agent.run_commitment.v1",
    "golden": "openmed.eval.agent_run_summary_fixture.v1",
    "timing": "urn:openmed:agent:timing:v1",
    "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
}
_SCHEMA_DIGESTS: Final = {
    "correlation": "sha256:c36133f0bd6d7a39e35b4486ba81b0b847ab8bfedd83fcedce085148c70454a0",
    "outcome": "sha256:779d96e55c22eb2b5591d6607d667a54a33858afa688e60d3d2d88bf6a0c22da",
    "run_summary": "sha256:5e0fb6c541b163a898fef5575b47c1aab6ac926c0898d2cf4391615bb1fed95a",
    "timing": "sha256:772aea820547fce3cdc8afa571f2f096d01c08bfefd36c004a6672d9768e87ca",
}
_RESULT_DIGESTS: Final = {
    "schema": "sha256:73412ddadbad7250b3f8174a82722a5ed7a7b9b2b00fd08ae2a45074156bb29c",
    "golden": "sha256:3f174d47df94694145d38eb9e0e95be1a497673198a093fce70d90c96e9ee040",
    "unsafe_fields": "sha256:c9eb73d7fdeb0b63cfe8c0b946c5fbca3011406c0e5525e06346dd7c1212aa85",
    "outcomes": "sha256:93a3f7adaa783a22f3c9879f08b0fe3f520a7e971733003420c90aa8485bb48a",
    "correlations": "sha256:17ed71b3df3679d4245b0494e1b849fd550ce4d046ad02ca074da7caadda524b",
    "timings": "sha256:795f781ce315d157071c4810c394bf47d1fd7ba78bac253aceb223d06f815f49",
    "commitments": "sha256:0410f901ac2ff4968c832d40c3c3a946f9263dbf7cb6531ffcbd99f533150af8",
}
_CASE_IDS: Final = (
    "empty",
    "success",
    "abstention",
    "denial",
    "review",
    "failure",
    "mixed",
)
_FIXTURE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "run_summary_schema_version",
        "outcome_schema_version",
        "commitment_version",
        "cases",
    }
)
_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "events",
        "expected_json",
        "expected_markdown_digest",
        "expected_commitment",
    }
)
_EVENT_FIELDS: Final = frozenset(
    {
        "workflow_id",
        "outcome",
        "tool_call_count",
        "duration_seconds",
        "artifact_digests",
    }
)
_OUTCOME_FIELDS: Final = frozenset(
    {
        "schema_version",
        "outcome_class",
        "reason_code",
    }
)
_SUMMARY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "workflow_ids",
        "outcome_counts",
        "tool_call_count",
        "duration_seconds",
        "artifact_digests",
    }
)
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}")
_WORKFLOW: Final = re.compile(r"wf_[0-9a-f]{32}")
_MAX_BYTES: Final = 1_048_576
_MAX_NODES: Final = 20_000
_MAX_DEPTH: Final = 24
_RUN: Final = "run_" + "1" * 32
_PARENT: Final = "act_" + "2" * 32
_CHILD: Final = "act_" + "3" * 32

# Fixed opaque vectors, with no randomness or measured elapsed time.
_CORRELATION_CASES: Final = (
    {
        "schema_version": _VERSIONS["correlation"],
        "run_id": _RUN,
        "action_id": _PARENT,
        "parent_action_id": None,
    },
    {
        "schema_version": _VERSIONS["correlation"],
        "run_id": _RUN,
        "action_id": _CHILD,
        "parent_action_id": _PARENT,
    },
)
_TIMING_CASES: Final = (
    {"run": {"start_ns": 0, "end_ns": 0, "duration_ns": 0}, "actions": []},
    {
        "run": {
            "start_ns": 10,
            "end_ns": 110,
            "duration_ns": 100,
            "correlation_id": _RUN,
        },
        "actions": [
            {
                "action_id": _PARENT,
                "start_ns": 20,
                "end_ns": 100,
                "duration_ns": 80,
                "correlation_id": _RUN,
            },
            {
                "action_id": _CHILD,
                "start_ns": 30,
                "end_ns": 50,
                "duration_ns": 20,
                "parent_action_id": _PARENT,
                "correlation_id": _RUN,
            },
        ],
    },
)


class SelfCheckName(str, Enum):
    """Closed names of independent governance agreement checks."""

    SCHEMA = "schema"
    GOLDEN = "golden"
    UNSAFE_FIELDS = "unsafe_fields"
    OUTCOMES = "outcomes"
    CORRELATIONS = "correlations"
    TIMINGS = "timings"
    COMMITMENTS = "commitments"


class SelfCheckStatus(str, Enum):
    """Closed pass and fail states, with no submitted values."""

    PASSED = "passed"
    FAILED = "failed"


class SelfCheckReason(str, Enum):
    """Content-free reasons usable in terminal and JSON reports."""

    OK = "ok"
    CONTRACT_MISMATCH = "contract_mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class AgentSelfCheckResult:
    """One immutable check result containing validated report metadata.

    Args:
        name: A declared check name.
        status: Passed or failed.
        case_count: Number of positive cases attempted.
        control_count: Number of negative controls attempted.
        reason: A closed diagnostic category.
        digest: SHA-256 over validated metadata, present only on success.
    """

    name: SelfCheckName
    status: SelfCheckStatus
    case_count: int
    control_count: int
    reason: SelfCheckReason
    digest: str | None

    def __post_init__(self) -> None:
        for value, kind in (
            (self.name, SelfCheckName),
            (self.status, SelfCheckStatus),
            (self.reason, SelfCheckReason),
        ):
            if type(value) is not kind:
                raise ValueError("self_check: invalid_report_category")
        if any(
            type(n) is not int or not 0 <= n <= 2048
            for n in (self.case_count, self.control_count)
        ):
            raise ValueError("self_check: invalid_report_count")
        if self.status is SelfCheckStatus.PASSED:
            if (
                self.reason is not SelfCheckReason.OK
                or type(self.digest) is not str
                or _DIGEST.fullmatch(self.digest) is None
            ):
                raise ValueError("self_check: invalid_passed_result")
        elif self.digest is not None or self.reason is SelfCheckReason.OK:
            raise ValueError("self_check: invalid_failed_result")

    def to_dict(self) -> dict[str, Any]:
        """Return controlled categories, bounded counts and a safe digest."""
        return {
            "name": self.name.value,
            "status": self.status.value,
            "case_count": self.case_count,
            "control_count": self.control_count,
            "reason": self.reason.value,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class AgentSelfCheckReport:
    """Complete deterministic report of all seven independent checks.

    Args:
        checks: Exactly one validated result per check, in declared order.
    """

    checks: tuple[AgentSelfCheckResult, ...]

    def __post_init__(self) -> None:
        if (
            type(self.checks) is not tuple
            or any(type(c) is not AgentSelfCheckResult for c in self.checks)
            or tuple(c.name for c in self.checks) != tuple(SelfCheckName)
        ):
            raise ValueError("self_check: incomplete_report")

    @property
    def passed(self) -> bool:
        """Return true only when every agreement check passed."""
        return all(c.status is SelfCheckStatus.PASSED for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Return normative versions and checks, never source payloads."""
        return {
            "schema_version": AGENT_SELF_CHECK_VERSION,
            "passed": self.passed,
            "versions": dict(_VERSIONS),
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self) -> str:
        """Return byte-stable compact ASCII JSON without timestamps."""
        return _canonical(self.to_dict())

    def to_text(self) -> str:
        """Return a concise content-free terminal report with full digests."""
        lines = [
            "Agent governance self-check: " + ("passed" if self.passed else "failed"),
            "report: " + AGENT_SELF_CHECK_VERSION,
        ]
        lines.extend(
            f"version {key}: {value}" for key, value in sorted(_VERSIONS.items())
        )
        lines.extend(
            f"{c.name.value}: {c.status.value}; cases={c.case_count}; "
            f"controls={c.control_count}; reason={c.reason.value}; "
            f"digest={c.digest or 'unavailable'}"
            for c in self.checks
        )
        return "\n".join(lines)


class _Probes:
    def __init__(self) -> None:
        self.cases = 0
        self.controls = 0
        self.good = True

    def positive(self, operation: Callable[[], Any]) -> Any:
        self.cases += 1
        try:
            return operation()
        except Exception:
            self.good = False
            return None

    def reject(self, operation: Callable[[], Any]) -> None:
        self.controls += 1
        try:
            operation()
        except ValueError:
            return
        except Exception:
            self.good = False
            return
        self.good = False

    def require(self, value: bool) -> None:
        self.good = self.good and bool(value)

    def result(self, name: SelfCheckName, metadata: Any) -> AgentSelfCheckResult:
        digest = _digest(metadata) if self.good else None
        self.require(digest == _RESULT_DIGESTS[name.value])
        return AgentSelfCheckResult(
            name,
            SelfCheckStatus.PASSED if self.good else SelfCheckStatus.FAILED,
            self.cases,
            self.controls,
            SelfCheckReason.OK if self.good else SelfCheckReason.CONTRACT_MISMATCH,
            digest if self.good else None,
        )


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("self_check: duplicate_field")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("self_check: nonfinite_number")


def _bounded_json(payload: bytes | str) -> Any:
    if type(payload) not in (bytes, str) or len(payload) > _MAX_BYTES:
        raise ValueError("self_check: invalid_json_size")
    document = json.loads(
        payload, object_pairs_hook=_strict_object, parse_constant=_reject_constant
    )
    stack = [(document, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("self_check: invalid_json_bounds")
        if type(item) is dict:
            stack.extend((value, depth + 1) for value in item.values())
        elif type(item) is list:
            stack.extend((value, depth + 1) for value in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("self_check: nonfinite_number")
    return document


def _load_fixture_document() -> Any:
    from openmed.eval.agent_run_summary_fixtures import (
        DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH,
    )

    with DEFAULT_AGENT_RUN_SUMMARY_FIXTURE_PATH.open("rb") as stream:
        return _bounded_json(stream.read(_MAX_BYTES + 1))


def _project(value: Any, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or not fields <= value.keys():
        raise ValueError("self_check: missing_fields")
    return {key: value[key] for key in sorted(fields)}


def _exact(value: Any, fields: frozenset[str]) -> None:
    if type(value) is not dict or value.keys() != fields:
        raise ValueError("self_check: unsafe_fields")


def _cases(document: Any) -> tuple[dict[str, Any], ...]:
    root = _project(document, _FIXTURE_FIELDS)
    versions = {
        "schema_version": _VERSIONS["golden"],
        "outcome_schema_version": _VERSIONS["outcome"],
        "run_summary_schema_version": _VERSIONS["run_summary"],
        "commitment_version": _VERSIONS["commitment"],
    }
    if any(type(root[k]) is not str or root[k] != v for k, v in versions.items()):
        raise ValueError("self_check: unsupported_version")
    values = root["cases"]
    if (
        type(values) is not list
        or len(values) != len(_CASE_IDS)
        or any(type(c) is not dict or type(c.get("case_id")) is not str for c in values)
        or {c["case_id"] for c in values} != set(_CASE_IDS)
    ):
        raise ValueError("self_check: incomplete_cases")
    return tuple(next(c for c in values if c["case_id"] == name) for name in _CASE_IDS)


def _event(value: Any) -> run_summary.RunEvent:
    data = _project(value, _EVENT_FIELDS)
    if (
        type(data["workflow_id"]) is not str
        or _WORKFLOW.fullmatch(data["workflow_id"]) is None
        or type(data["artifact_digests"]) is not list
    ):
        raise ValueError("self_check: unsafe_event")
    return run_summary.RunEvent(
        workflow_id=data["workflow_id"],
        outcome=outcomes.WorkflowOutcome.from_dict(
            _project(data["outcome"], _OUTCOME_FIELDS)
        ),
        tool_call_count=data["tool_call_count"],
        duration_seconds=data["duration_seconds"],
        artifact_digests=tuple(data["artifact_digests"]),
    )


def _events(case: Any) -> tuple[run_summary.RunEvent, ...]:
    data = _project(case, frozenset({"events"}))
    if type(data["events"]) is not list or len(data["events"]) > 128:
        raise ValueError("self_check: invalid_events")
    return tuple(_event(e) for e in data["events"])


def _fixture_cases(document: Any, probes: _Probes) -> tuple[dict[str, Any], ...]:
    try:
        return _cases(document)
    except ValueError:
        probes.good = False
        return ()


def _safe_summary(payload: Any) -> run_summary.RunSummary:
    data = _project(payload, _SUMMARY_FIELDS)
    if type(data["workflow_ids"]) is not list or any(
        type(v) is not str or _WORKFLOW.fullmatch(v) is None
        for v in data["workflow_ids"]
    ):
        raise ValueError("self_check: unsafe_workflow_id")
    return run_summary.RunSummary.from_dict(data)


def _golden_check(document: Any) -> AgentSelfCheckResult:
    from openmed.eval.agent_run_summary_fixtures import (
        AGENT_RUN_SUMMARY_FIXTURE_VERSION,
        MAX_RUN_SUMMARY_FIXTURE_BYTES,
        MAX_RUN_SUMMARY_FIXTURE_EVENTS,
        AgentRunSummaryFixture,
    )

    probes = _Probes()
    probes.require(AGENT_RUN_SUMMARY_FIXTURE_VERSION == _VERSIONS["golden"])
    probes.require(MAX_RUN_SUMMARY_FIXTURE_BYTES == _MAX_BYTES)
    probes.require(MAX_RUN_SUMMARY_FIXTURE_EVENTS == 128)
    probes.require(run_summary.RUN_SUMMARY_SCHEMA_VERSION == _VERSIONS["run_summary"])
    metadata = []
    for case in _fixture_cases(document, probes):

        def validate(case: Any = case) -> dict[str, Any]:
            events = _events(case)
            summary = run_summary.RunSummary.from_events(events)
            expected = _bounded_json(case["expected_json"])
            _safe_summary(expected)
            # Unknown fields are reported independently by unsafe_fields. For
            # closed inputs the golden comparison still checks the exact bytes.
            expected_json = (
                case["expected_json"]
                if expected.keys() == _SUMMARY_FIELDS
                else _canonical(_project(expected, _SUMMARY_FIELDS))
            )
            fixture = AgentRunSummaryFixture(
                case["case_id"],
                events,
                expected_json,
                case["expected_markdown_digest"],
                run_commitment.compute_run_summary_commitment(summary),
            )
            if fixture.build_summary().to_json() != expected_json:
                raise ValueError("self_check: serialization_mismatch")
            return {
                "case_id": case["case_id"],
                "json": summary.to_json(),
                "markdown_digest": case["expected_markdown_digest"],
            }

        value = probes.positive(validate)
        if value is not None:
            metadata.append(value)
    baseline = run_summary.RunSummary.from_events(())
    for field, value in (
        ("tool_call_count", True),
        ("duration_seconds", -1),
        ("schema_version", "unsupported"),
    ):
        bad = baseline.to_dict()
        bad[field] = value
        probes.reject(lambda: run_summary.RunSummary.from_dict(bad))
    return probes.result(SelfCheckName.GOLDEN, metadata)


def _unsafe_case(case: Any) -> None:
    _exact(case, _CASE_FIELDS)
    _events(case)
    for event in case["events"]:
        _exact(event, _EVENT_FIELDS)
        _exact(event["outcome"], _OUTCOME_FIELDS)
        _event(event)
    expected = _bounded_json(case["expected_json"])
    _exact(expected, _SUMMARY_FIELDS)
    _safe_summary(expected)


def _unsafe_check(document: Any) -> AgentSelfCheckResult:
    probes = _Probes()
    try:
        _exact(document, _FIXTURE_FIELDS)
    except ValueError:
        probes.good = False
    for case in _fixture_cases(document, probes):
        probes.positive(lambda: _unsafe_case(case))
    summary = run_summary.RunSummary.from_events(()).to_dict()
    outcome = outcomes.WorkflowOutcome(
        outcomes.OutcomeClass.SUCCESS, "completed"
    ).to_dict()
    for field in ("prompt", "clinical_text", "path", "environment", "patient_id"):
        probes.reject(
            lambda: run_summary.RunSummary.from_dict(
                {**summary, field: "synthetic_control"}
            )
        )
        probes.reject(
            lambda: outcomes.WorkflowOutcome.from_dict(
                {**outcome, field: "synthetic_control"}
            )
        )
    for payload in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
        probes.reject(lambda: _bounded_json(payload))
    return probes.result(
        SelfCheckName.UNSAFE_FIELDS,
        {"closed_case_count": 7, "controlled_field_checks": probes.controls},
    )


def _outcomes_check(document: Any) -> AgentSelfCheckResult:
    probes = _Probes()
    probes.require(outcomes.OUTCOME_SCHEMA_VERSION == _VERSIONS["outcome"])
    metadata = []
    for kind in outcomes.OutcomeClass:
        for reason in sorted(outcomes.allowed_reason_codes(kind)):
            value = probes.positive(lambda: outcomes.WorkflowOutcome(kind, reason))
            if value is not None:
                probes.require(
                    outcomes.WorkflowOutcome.from_json(value.to_json()) == value
                )
                metadata.append(value.to_dict())
    for case in _fixture_cases(document, probes):
        for event in case["events"]:
            probes.positive(
                lambda: outcomes.WorkflowOutcome.from_dict(
                    _project(event["outcome"], _OUTCOME_FIELDS)
                )
            )
    for kind in outcomes.OutcomeClass:
        for other in outcomes.OutcomeClass:
            if other is kind:
                continue
            for reason in sorted(outcomes.allowed_reason_codes(other)):
                probes.reject(lambda: outcomes.WorkflowOutcome(kind, reason))
    for data in (
        {"outcome_class": "unknown", "reason_code": "completed"},
        {"outcome_class": "success", "reason_code": "unknown"},
        {
            "schema_version": "unsupported",
            "outcome_class": "success",
            "reason_code": "completed",
        },
    ):
        probes.reject(lambda: outcomes.WorkflowOutcome.from_dict(data))
    return probes.result(SelfCheckName.OUTCOMES, metadata)


def _correlation_case(data: Any) -> correlation.ActionCorrelation:
    _exact(
        data, frozenset({"schema_version", "run_id", "action_id", "parent_action_id"})
    )
    result = correlation.ActionCorrelation.from_dict(data)
    if result.to_dict() != data:
        raise ValueError("self_check: correlation_roundtrip")
    if correlation.ActionCorrelation.from_json(result.to_json()) != result:
        raise ValueError("self_check: correlation_roundtrip")
    return result


def _correlations_check() -> AgentSelfCheckResult:
    probes = _Probes()
    probes.require(correlation.CORRELATION_SCHEMA_VERSION == _VERSIONS["correlation"])
    if type(_CORRELATION_CASES) is not tuple or len(_CORRELATION_CASES) != 2:
        raise ValueError("self_check: incomplete_correlations")
    metadata = []
    for case in _CORRELATION_CASES:
        value = probes.positive(lambda: _correlation_case(case))
        if value is not None:
            metadata.append(value.to_dict())
    good = {
        "schema_version": _VERSIONS["correlation"],
        "run_id": _RUN,
        "action_id": _CHILD,
        "parent_action_id": _PARENT,
    }
    for field, value in (
        ("run_id", "run_invalid"),
        ("action_id", _CHILD + "\n"),
        ("parent_action_id", _CHILD),
        ("schema_version", "unsupported"),
        ("prompt", "control"),
    ):
        probes.reject(
            lambda: correlation.ActionCorrelation.from_dict({**good, field: value})
        )
    return probes.result(SelfCheckName.CORRELATIONS, metadata)


def _timing_case(data: Any) -> timing.AgentRunTiming:
    _exact(data, frozenset({"run", "actions"}))
    run = data["run"]
    _project(run, frozenset({"start_ns", "end_ns", "duration_ns"}))
    if run.keys() - {"start_ns", "end_ns", "duration_ns", "correlation_id"}:
        raise ValueError("self_check: unsafe_timing")
    if "correlation_id" in run:
        correlation.RunId.parse(run["correlation_id"])
    record = timing.RunTiming(
        run["start_ns"], run["end_ns"], correlation_id=run.get("correlation_id")
    )
    if type(run["duration_ns"]) is not int or record.duration_ns != run["duration_ns"]:
        raise ValueError("self_check: duration_mismatch")
    if type(data["actions"]) is not list or len(data["actions"]) > 128:
        raise ValueError("self_check: invalid_timing_actions")
    actions = []
    for action in data["actions"]:
        _project(action, frozenset({"action_id", "start_ns", "end_ns", "duration_ns"}))
        if action.keys() - {
            "action_id",
            "start_ns",
            "end_ns",
            "duration_ns",
            "parent_action_id",
            "correlation_id",
        }:
            raise ValueError("self_check: unsafe_timing")
        correlation.ActionId.parse(action["action_id"])
        if "parent_action_id" in action:
            correlation.ActionId.parse(action["parent_action_id"])
        if "correlation_id" in action:
            correlation.RunId.parse(action["correlation_id"])
        item = timing.ActionTiming(
            action["action_id"],
            action["start_ns"],
            action["end_ns"],
            parent_action_id=action.get("parent_action_id"),
            correlation_id=action.get("correlation_id"),
        )
        if (
            type(action["duration_ns"]) is not int
            or item.duration_ns != action["duration_ns"]
        ):
            raise ValueError("self_check: duration_mismatch")
        actions.append(item)
    result = timing.AgentRunTiming(record, tuple(actions))
    if result.to_dict() != data:
        raise ValueError("self_check: timing_roundtrip")
    return result


def _timings_check() -> AgentSelfCheckResult:
    probes = _Probes()
    if type(_TIMING_CASES) is not tuple or len(_TIMING_CASES) != 2:
        raise ValueError("self_check: incomplete_timings")
    metadata = []
    for case in _TIMING_CASES:
        value = probes.positive(lambda: _timing_case(case))
        if value is not None:
            metadata.append(value.to_dict())
    for start, end in ((True, 1), (-1, 1), (2, 1)):
        probes.reject(lambda: timing.RunTiming(start, end))
    run = timing.RunTiming(0, 100)
    for actions in (
        (timing.ActionTiming(_PARENT, 0, 101),),
        (timing.ActionTiming(_PARENT, 0, 10), timing.ActionTiming(_PARENT, 20, 30)),
        (timing.ActionTiming(_CHILD, 10, 20, parent_action_id=_PARENT),),
        (
            timing.ActionTiming(_PARENT, 0, 30, parent_action_id=_CHILD),
            timing.ActionTiming(_CHILD, 0, 30, parent_action_id=_PARENT),
        ),
        (
            timing.ActionTiming(_PARENT, 10, 20),
            timing.ActionTiming(_CHILD, 0, 30, parent_action_id=_PARENT),
        ),
    ):
        probes.reject(lambda: timing.AgentRunTiming(run, actions))
    probes.reject(lambda: timing.RunTiming(0, 10, max_duration_ns=9))
    probes.reject(
        lambda: timing.AgentRunTiming(
            run,
            (timing.ActionTiming(_PARENT, 0, 20), timing.ActionTiming(_CHILD, 10, 30)),
            allow_action_overlaps=False,
        )
    )
    return probes.result(SelfCheckName.TIMINGS, metadata)


def _commitments_check(document: Any) -> AgentSelfCheckResult:
    probes = _Probes()
    probes.require(run_commitment.RUN_COMMITMENT_VERSION == _VERSIONS["commitment"])
    metadata = []
    for case in _fixture_cases(document, probes):

        def validate(case: Any = case) -> str:
            summary = run_summary.RunSummary.from_events(_events(case))
            independent = (
                "sha256:"
                + hashlib.sha256(
                    _VERSIONS["commitment"].encode("ascii")
                    + b"\x00"
                    + summary.to_json().encode("utf-8")
                ).hexdigest()
            )
            computed = run_commitment.compute_run_summary_commitment(summary)
            if computed != independent or case["expected_commitment"] != independent:
                raise ValueError("self_check: commitment_mismatch")
            if run_commitment.verify_run_summary_commitment(
                summary, independent
            ) is not (run_commitment.RunCommitmentVerificationResult.VERIFIED):
                raise ValueError("self_check: commitment_mismatch")
            return independent

        value = probes.positive(validate)
        if value is not None:
            metadata.append(value)
    summary = run_summary.RunSummary.from_events(())
    for candidate, result in (
        ("sha256:" + "0" * 64, run_commitment.RunCommitmentVerificationResult.MISMATCH),
        ("invalid", run_commitment.RunCommitmentVerificationResult.MALFORMED),
        (
            "sha256:" + hashlib.sha256(summary.to_json().encode()).hexdigest(),
            run_commitment.RunCommitmentVerificationResult.MISMATCH,
        ),
    ):
        probes.controls += 1
        probes.require(
            run_commitment.verify_run_summary_commitment(summary, candidate) is result
        )
    original = run_commitment.compute_run_summary_commitment(summary)
    for changed in (
        run_summary.RunSummary.from_events(
            (
                run_summary.RunEvent(
                    "wf_" + "4" * 32,
                    outcomes.WorkflowOutcome(
                        outcomes.OutcomeClass.SUCCESS, "completed"
                    ),
                ),
            )
        ),
        run_summary.RunSummary.from_dict({**summary.to_dict(), "tool_call_count": 1}),
        run_summary.RunSummary.from_dict(
            {**summary.to_dict(), "duration_seconds": 1.0}
        ),
        run_summary.RunSummary.from_dict(
            {**summary.to_dict(), "artifact_digests": ["sha256:" + "4" * 64]}
        ),
    ):
        probes.controls += 1
        probes.require(
            run_commitment.verify_run_summary_commitment(changed, original)
            is run_commitment.RunCommitmentVerificationResult.MISMATCH
        )
    return probes.result(SelfCheckName.COMMITMENTS, metadata)


def _schema_check() -> AgentSelfCheckResult:
    from jsonschema import Draft202012Validator

    from .schemas import build_agent_schema_catalog, render_agent_schema

    probes = _Probes()
    catalog = build_agent_schema_catalog()
    _exact(catalog, frozenset(_SCHEMA_DIGESTS))
    metadata = {}
    outcome_examples = [
        outcomes.WorkflowOutcome(kind, reason).to_dict()
        for kind in outcomes.OutcomeClass
        for reason in sorted(outcomes.allowed_reason_codes(kind))
    ]
    examples: dict[str, list[dict[str, Any]]] = {
        "outcome": outcome_examples,
        "correlation": [
            correlation.ActionCorrelation(
                correlation.RunId.parse(_RUN), correlation.ActionId.parse(_PARENT)
            ).to_dict(),
            correlation.ActionCorrelation(
                correlation.RunId.parse(_RUN),
                correlation.ActionId.parse(_CHILD),
                parent_action_id=correlation.ActionId.parse(_PARENT),
            ).to_dict(),
        ],
        "timing": [
            timing.AgentRunTiming(timing.RunTiming(0, 0)).to_dict(),
            timing.AgentRunTiming(
                timing.RunTiming(0, 100),
                (
                    timing.ActionTiming(_PARENT, 10, 90),
                    timing.ActionTiming(_CHILD, 20, 30, parent_action_id=_PARENT),
                ),
            ).to_dict(),
        ],
        "run_summary": [
            run_summary.RunSummary.from_events(()).to_dict(),
            run_summary.RunSummary.from_events(
                tuple(
                    run_summary.RunEvent(
                        "wf_" + "4" * 32,
                        outcomes.WorkflowOutcome.from_dict(value),
                        tool_call_count=1,
                        duration_seconds=0.125,
                        artifact_digests=("sha256:" + "4" * 64,),
                    )
                    for value in outcome_examples
                )
            ).to_dict(),
        ],
    }
    for name in sorted(_SCHEMA_DIGESTS):

        def validate(name: str = name) -> str:
            schema = catalog[name]
            # Fingerprints bind the complete actual schema catalog, including
            # every bound, enum, closed object and fragment-local reference.
            # A changed schema is rejected before validator construction.
            digest = _digest(schema)
            if (
                digest != _SCHEMA_DIGESTS[name]
                or schema.get("$schema") != _VERSIONS["schema_dialect"]
            ):
                raise ValueError("self_check: schema_drift")
            if render_agent_schema(name) != _canonical(schema):
                raise ValueError("self_check: schema_render_drift")
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema)
            for example in examples[name]:
                validator.validate(example)
            bad_examples = [{**examples[name][0], "prompt": "control"}, {}]
            if name == "outcome":
                bad_examples.extend(
                    [
                        {**examples[name][0], "outcome_class": "unknown"},
                        {
                            "schema_version": _VERSIONS["outcome"],
                            "outcome_class": "success",
                            "reason_code": "tool_error",
                        },
                        {**examples[name][0], "schema_version": "unsupported"},
                    ]
                )
            elif name == "correlation":
                bad_examples.extend(
                    {**examples[name][0], key: value}
                    for key, value in (
                        ("run_id", "run_invalid"),
                        ("action_id", _CHILD + "\n"),
                        ("parent_action_id", "invalid"),
                        ("schema_version", "unsupported"),
                    )
                )
            elif name == "timing":
                bad_examples.extend(
                    {"run": {**examples[name][0]["run"], key: value}, "actions": []}
                    for key, value in (
                        ("start_ns", True),
                        ("end_ns", -1),
                        ("duration_ns", 0.5),
                        ("prompt", "control"),
                    )
                )
                bad_examples.append(
                    {
                        "run": examples[name][0]["run"],
                        "actions": [
                            {
                                "action_id": _CHILD,
                                "start_ns": 0,
                                "end_ns": 0,
                                "duration_ns": 0,
                                "prompt": "control",
                            }
                        ],
                    }
                )
            elif name == "run_summary":
                bad_examples.extend(
                    {**examples[name][0], key: value}
                    for key, value in (
                        ("tool_call_count", True),
                        ("duration_seconds", -1),
                        ("artifact_digests", ["invalid"]),
                        ("workflow_ids", ["control\n"]),
                        ("schema_version", "unsupported"),
                        ("outcome_counts", {"success": 0}),
                    )
                )
            for bad in bad_examples:
                probes.controls += 1
                probes.require(not validator.is_valid(bad))
            return digest

        value = probes.positive(validate)
        if value is not None:
            metadata[name] = value
    return probes.result(SelfCheckName.SCHEMA, metadata)


def run_agent_self_check() -> AgentSelfCheckReport:
    """Check all bundled governance contracts offline and independently.

    Every check runs even when another fails. Failed checks have a fixed reason
    and no digest: no raw fixture, submitted value, exception message, path,
    environment value, clock reading or model output is returned.

    Returns:
        An immutable deterministic report, passed only when all checks pass.
    """
    try:
        document = _load_fixture_document()
    except Exception:
        document = None
    operations: tuple[tuple[SelfCheckName, Callable[[], AgentSelfCheckResult]], ...] = (
        (SelfCheckName.SCHEMA, _schema_check),
        (SelfCheckName.GOLDEN, lambda: _golden_check(document)),
        (SelfCheckName.UNSAFE_FIELDS, lambda: _unsafe_check(document)),
        (SelfCheckName.OUTCOMES, lambda: _outcomes_check(document)),
        (SelfCheckName.CORRELATIONS, _correlations_check),
        (SelfCheckName.TIMINGS, _timings_check),
        (SelfCheckName.COMMITMENTS, lambda: _commitments_check(document)),
    )
    results = []
    for name, operation in operations:
        try:
            result = operation()
        except Exception:
            result = AgentSelfCheckResult(
                name,
                SelfCheckStatus.FAILED,
                0,
                0,
                SelfCheckReason.UNAVAILABLE
                if document is None
                and name
                in {
                    SelfCheckName.GOLDEN,
                    SelfCheckName.UNSAFE_FIELDS,
                    SelfCheckName.OUTCOMES,
                    SelfCheckName.COMMITMENTS,
                }
                else SelfCheckReason.CONTRACT_MISMATCH,
                None,
            )
        results.append(result)
    return AgentSelfCheckReport(tuple(results))


__all__ = [
    "AGENT_SELF_CHECK_VERSION",
    "AgentSelfCheckReport",
    "AgentSelfCheckResult",
    "SelfCheckName",
    "SelfCheckReason",
    "SelfCheckStatus",
    "run_agent_self_check",
]
