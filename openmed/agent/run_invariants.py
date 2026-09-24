"""Fail-closed, metadata-only checks for a completed agent run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .action_graph import GRAPH_REASON_CODES, ActionNode, validate_action_graph
from .artifact_reference import ArtifactReference, validate_artifact_references
from .correlation import ActionCorrelation, RunId
from .errors import ErrorEnvelope
from .event_sequence import (
    SEQUENCE_REASON_CODES,
    EventReference,
    validate_event_sequence,
)
from .outcomes import OutcomeClass, WorkflowOutcome
from .timing import ActionTiming, AgentRunTiming, RunTiming, TimingValidationError

RUN_INVARIANT_SCHEMA_VERSION = "openmed.agent.run_invariants.v1"
_FINAL_PHASES = frozenset({"completed", "aborted"})
_KNOWN_PHASES = (
    frozenset({"queued", "preflight", "ready", "running", "waiting-review"})
    | _FINAL_PHASES
)
_ERROR_OUTCOMES = frozenset({OutcomeClass.FAILED, OutcomeClass.POLICY_DENIED})
_ARTIFACT_ID_RE = re.compile(r"art_[0-9a-f]{32}")
RUN_INVARIANT_REASON_CODES = frozenset(
    {
        "duplicate_action",
        "missing_action",
        "orphan_action",
        "cross_run_action",
        "orphan_parent",
        "unfinished_action",
        "missing_action_outcome",
        "phase_outcome_conflict",
        "error_correlation_mismatch",
        "invalid_timing",
        "timing_action_mismatch",
        "timing_run_mismatch",
        "missing_run_outcome",
        "run_outcome_conflict",
        "missing_error",
        "error_outcome_conflict",
        "invalid_artifacts",
        "duplicate_artifact_production",
        "artifact_from_unsuccessful_action",
        "unsupported_artifact",
        "undeclared_artifact",
        "invalid_sequence",
    }
    | {f"event_{code}" for code in SEQUENCE_REASON_CODES}
    | {f"graph_{code}" for code in GRAPH_REASON_CODES}
)


class RunInvariantError(ValueError):
    """Structural error with a fixed code and no submitted values."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class RunAction:
    """Content-free final evidence for one action.

    Args:
        correlation: Run and parent identifiers for the action.
        phase: Final action phase from the v3.1 phase vocabulary.
        outcome: Action outcome, when recorded.
        error: Categorical error envelope, when recorded.
        artifact_ids: Opaque identifiers of artifacts this action produced.
        reviewed: Whether external review was declared before completion.
    """

    correlation: ActionCorrelation
    phase: str
    outcome: WorkflowOutcome | None = None
    error: ErrorEnvelope | None = None
    artifact_ids: tuple[str, ...] = ()
    reviewed: bool = False

    def __post_init__(self) -> None:
        if type(self.correlation) is not ActionCorrelation:
            raise RunInvariantError("invalid_correlation")
        if type(self.phase) is not str or self.phase not in _KNOWN_PHASES:
            raise RunInvariantError("invalid_phase")
        if self.outcome is not None and type(self.outcome) is not WorkflowOutcome:
            raise RunInvariantError("invalid_outcome")
        if self.error is not None and type(self.error) is not ErrorEnvelope:
            raise RunInvariantError("invalid_error")
        if type(self.reviewed) is not bool:
            raise RunInvariantError("invalid_reviewed")
        if type(self.artifact_ids) is not tuple or any(
            type(value) is not str or _ARTIFACT_ID_RE.fullmatch(value) is None
            for value in self.artifact_ids
        ):
            raise RunInvariantError("invalid_artifact_ids")


@dataclass(frozen=True, slots=True, repr=False)
class CompletedRun:
    """Typed metadata collected at the end of one run.

    Event references are in append order. ``terminal_sequence_number`` names
    the terminal event. Artifact references are declarations; action artifact
    IDs identify their producers without reading artifact content.
    """

    run_id: RunId
    events: tuple[EventReference, ...]
    terminal_sequence_number: int
    graph: tuple[ActionNode, ...]
    actions: tuple[RunAction, ...]
    timing: RunTiming
    action_timings: tuple[ActionTiming, ...]
    outcome: WorkflowOutcome | None
    error: ErrorEnvelope | None = None
    artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId:
            raise RunInvariantError("invalid_run_id")
        typed_collections = (
            (self.events, EventReference),
            (self.graph, ActionNode),
            (self.actions, RunAction),
            (self.action_timings, ActionTiming),
            (self.artifacts, ArtifactReference),
        )
        for collection, item_type in typed_collections:
            if type(collection) is not tuple or any(
                type(item) is not item_type for item in collection
            ):
                raise RunInvariantError("invalid_collection")
        if type(self.timing) is not RunTiming:
            raise RunInvariantError("invalid_timing")
        if self.outcome is not None and type(self.outcome) is not WorkflowOutcome:
            raise RunInvariantError("invalid_outcome")
        if self.error is not None and type(self.error) is not ErrorEnvelope:
            raise RunInvariantError("invalid_error")


@dataclass(frozen=True, slots=True)
class RunInvariantReport:
    """Deterministic verdict containing only a closed set of finding codes."""

    findings: tuple[str, ...]
    schema_version: str = RUN_INVARIANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or (self.schema_version != RUN_INVARIANT_SCHEMA_VERSION)
            or (
                type(self.findings) is not tuple
                or any(
                    type(code) is not str or code not in RUN_INVARIANT_REASON_CODES
                    for code in self.findings
                )
                or self.findings != tuple(sorted(set(self.findings)))
            )
        ):
            raise RunInvariantError("invalid_report")

    @property
    def is_valid(self) -> bool:
        """Whether every independent end-of-run check passed."""

        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        """Return a content-free, JSON-safe report."""

        return {"schema_version": self.schema_version, "findings": list(self.findings)}

    def to_json(self) -> str:
        """Return byte-stable compact JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def check_run_invariants(run: CompletedRun) -> RunInvariantReport:
    """Check independent end-of-run safety invariants without executing work.

    Args:
        run: Typed, content-free final run evidence.

    Returns:
        A deterministic report whose findings cannot compensate for each other.

    Raises:
        RunInvariantError: If the input is not a typed completed run.
    """

    if type(run) is not CompletedRun:
        raise RunInvariantError("invalid_run")
    findings: set[str] = set()
    run_id = run.run_id.serialize()

    try:
        sequence = validate_event_sequence(
            run_id,
            run.events,
            terminal_sequence_number=run.terminal_sequence_number,
        )
    except ValueError:
        findings.add("invalid_sequence")
    else:
        findings.update(f"event_{code}" for code in sequence.reason_codes)
    graph = validate_action_graph(run.graph)
    findings.update(f"graph_{code}" for code in graph.reason_codes)

    graph_ids = {node.action_id for node in run.graph}
    action_ids = [action.correlation.action_id.serialize() for action in run.actions]
    recorded_ids = set(action_ids)
    if len(action_ids) != len(recorded_ids):
        findings.add("duplicate_action")
    if graph_ids - recorded_ids:
        findings.add("missing_action")
    if recorded_ids - graph_ids:
        findings.add("orphan_action")
    for action in run.actions:
        correlation = action.correlation
        if correlation.run_id != run.run_id:
            findings.add("cross_run_action")
        parent = correlation.parent_action_id
        if parent is not None and parent.serialize() not in recorded_ids:
            findings.add("orphan_parent")
        if action.phase not in _FINAL_PHASES:
            findings.add("unfinished_action")
        if action.outcome is None:
            findings.add("missing_action_outcome")
        elif action.phase == "completed" and action.outcome.outcome_class in (
            OutcomeClass.FAILED,
            OutcomeClass.POLICY_DENIED,
        ):
            findings.add("phase_outcome_conflict")
        elif (
            action.phase == "aborted"
            and action.outcome.outcome_class is OutcomeClass.SUCCESS
        ):
            findings.add("phase_outcome_conflict")
        _check_error(action.outcome, action.error, findings)
        if action.error is not None and (
            action.error.run_id != run_id
            or action.error.action_id != correlation.action_id.serialize()
        ):
            findings.add("error_correlation_mismatch")

    try:
        AgentRunTiming(run.timing, run.action_timings)
    except TimingValidationError:
        findings.add("invalid_timing")
    timing_ids = {timing.action_id for timing in run.action_timings}
    if timing_ids != recorded_ids:
        findings.add("timing_action_mismatch")
    if run.timing.correlation_id != run_id:
        findings.add("timing_run_mismatch")

    if run.outcome is None:
        findings.add("missing_run_outcome")
    else:
        _check_error(run.outcome, run.error, findings)
        if run.outcome.outcome_class is OutcomeClass.SUCCESS and any(
            action.phase != "completed" for action in run.actions
        ):
            findings.add("run_outcome_conflict")
    if run.error is not None and (
        run.error.run_id != run_id or run.error.action_id is not None
    ):
        findings.add("error_correlation_mismatch")

    try:
        validate_artifact_references(run.artifacts)
    except ValueError:
        findings.add("invalid_artifacts")
    declared = {artifact.artifact_id for artifact in run.artifacts}
    produced: set[str] = set()
    for action in run.actions:
        if len(action.artifact_ids) != len(set(action.artifact_ids)):
            findings.add("duplicate_artifact_production")
        for artifact_id in action.artifact_ids:
            if artifact_id in produced:
                findings.add("duplicate_artifact_production")
            produced.add(artifact_id)
            if (
                action.phase != "completed"
                or action.outcome is None
                or (action.outcome.outcome_class is not OutcomeClass.SUCCESS)
            ):
                findings.add("artifact_from_unsuccessful_action")
    if declared - produced:
        findings.add("unsupported_artifact")
    if produced - declared:
        findings.add("undeclared_artifact")
    return RunInvariantReport(tuple(sorted(findings)))


def _check_error(
    outcome: WorkflowOutcome | None,
    error: ErrorEnvelope | None,
    findings: set[str],
) -> None:
    if outcome is None:
        return
    if outcome.outcome_class in _ERROR_OUTCOMES and error is None:
        findings.add("missing_error")
    if outcome.outcome_class not in _ERROR_OUTCOMES and error is not None:
        findings.add("error_outcome_conflict")


__all__ = [
    "RUN_INVARIANT_REASON_CODES",
    "RUN_INVARIANT_SCHEMA_VERSION",
    "CompletedRun",
    "RunAction",
    "RunInvariantError",
    "RunInvariantReport",
    "check_run_invariants",
]
