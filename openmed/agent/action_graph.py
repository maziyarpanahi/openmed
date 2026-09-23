"""Deterministic dependency validation for multi-step agent workflows.

A workflow graph is modelled with opaque action identifiers, canonical tool
identifiers, and dependency identifiers. Tool arguments, tool outputs,
prompts, filesystem paths, and clinical text are never accepted, retained, or
echoed, so a graph or a finding cannot carry them.
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .identifiers import GovernanceIdError, ToolId

ACTION_GRAPH_SCHEMA_VERSION = "openmed.agent.action_graph.v1"
MAX_ACTION_GRAPH_NODES = 10_000
MAX_ACTION_DEPENDENCIES = 64
MAX_ACTION_GRAPH_FINDINGS = 1_000

GRAPH_REASON_CODES = frozenset(
    {
        "dependency_cycle",
        "duplicate_action_id",
        "duplicate_dependency",
        "findings_truncated",
        "missing_dependency",
        "self_dependency",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$")
_NODE_FIELDS = ("action_id", "tool_id", "depends_on")
_FINDING_FIELDS = ("reason_code", "action_id", "dependency_id")
_REPORT_FIELDS = (
    "schema_version",
    "node_count",
    "edge_count",
    "order",
    "findings",
)


class ActionGraphError(ValueError):
    """Raised when action-graph input fails closed structural validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional fixed public field associated with the failure.

    Messages and attributes carry controlled diagnostic metadata only; the
    rejected value is never retained or echoed.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ActionNode:
    """One planned agent action expressed as identifiers only.

    Attributes:
        action_id: Bounded opaque identifier, unique within the graph.
        tool_id: Canonical governance tool identifier, as accepted by
            :class:`~openmed.agent.identifiers.ToolId`.
        depends_on: Action identifiers that must complete first.
    """

    action_id: str
    tool_id: str
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.action_id, "action_id")
        _validate_tool_id(self.tool_id)
        depends_on = _as_tuple(self.depends_on, "depends_on")
        if len(depends_on) > MAX_ACTION_DEPENDENCIES:
            raise ActionGraphError("too_many_dependencies", "depends_on")
        for dependency in depends_on:
            _validate_identifier(dependency, "depends_on")
        object.__setattr__(self, "depends_on", depends_on)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "action_id": self.action_id,
            "tool_id": self.tool_id,
            "depends_on": list(self.depends_on),
        }
        return {field: values[field] for field in _NODE_FIELDS}


@dataclass(frozen=True, slots=True)
class GraphFinding:
    """One stable, categorical reason why a graph cannot be executed.

    Attributes:
        reason_code: Member of :data:`GRAPH_REASON_CODES`.
        action_id: Opaque action the finding refers to, when one applies.
        dependency_id: Opaque dependency the finding refers to, when one
            applies.
    """

    reason_code: str
    action_id: str | None = None
    dependency_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.reason_code) is not str
            or self.reason_code not in GRAPH_REASON_CODES
        ):
            raise ActionGraphError("unknown_reason_code", "reason_code")
        if self.action_id is not None:
            _validate_identifier(self.action_id, "action_id")
        if self.dependency_id is not None:
            _validate_identifier(self.dependency_id, "dependency_id")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "reason_code": self.reason_code,
            "action_id": self.action_id,
            "dependency_id": self.dependency_id,
        }
        return {field: values[field] for field in _FINDING_FIELDS}


@dataclass(frozen=True, slots=True)
class ActionGraphReport:
    """Deterministic, identifier-only verdict for one workflow graph."""

    node_count: int
    edge_count: int
    order: tuple[str, ...]
    findings: tuple[GraphFinding, ...]
    schema_version: str = ACTION_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ACTION_GRAPH_SCHEMA_VERSION
        ):
            raise ActionGraphError("invalid_schema_version", "schema_version")

    @property
    def is_valid(self) -> bool:
        """Return whether the graph produced no findings."""

        return not self.findings

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return the finding reason codes in report order."""

        return tuple(finding.reason_code for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "order": list(self.order),
            "findings": [finding.to_dict() for finding in self.findings],
        }
        return {field: values[field] for field in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_action_graph(nodes: Iterable[ActionNode]) -> ActionGraphReport:
    """Validate a workflow graph and order it for execution.

    Structural problems (wrong types, malformed identifiers, oversized input)
    fail closed with :class:`ActionGraphError`. Graph problems are reported as
    findings so a caller sees every defect in one pass.

    Duplicate action identifiers keep the first occurrence, self-edges and
    dangling edges are dropped, and the remaining graph is then checked for
    cycles. A topological order is returned only when there are no findings.
    Ready actions are emitted in ascending action-identifier order, which makes
    the order stable for graphs with independent branches.

    Args:
        nodes: Iterable of :class:`ActionNode` values in any order.

    Returns:
        An :class:`ActionGraphReport` whose findings are ordered by action
        identifier, then reason code, then dependency identifier.

    Raises:
        ActionGraphError: If the input is structurally invalid.
    """

    collected = _materialize(nodes)
    findings: list[GraphFinding] = []

    unique: dict[str, ActionNode] = {}
    for node in collected:
        if node.action_id in unique:
            findings.append(GraphFinding("duplicate_action_id", node.action_id))
            continue
        unique[node.action_id] = node

    edges: dict[str, tuple[str, ...]] = {}
    for action_id, node in unique.items():
        accepted: list[str] = []
        seen: set[str] = set()
        for dependency in node.depends_on:
            if dependency in seen:
                findings.append(
                    GraphFinding("duplicate_dependency", action_id, dependency)
                )
                continue
            seen.add(dependency)
            if dependency == action_id:
                findings.append(GraphFinding("self_dependency", action_id, dependency))
                continue
            if dependency not in unique:
                findings.append(
                    GraphFinding("missing_dependency", action_id, dependency)
                )
                continue
            accepted.append(dependency)
        edges[action_id] = tuple(accepted)

    order, cyclic = _topological_order(edges)
    findings.extend(GraphFinding("dependency_cycle", action_id) for action_id in cyclic)
    ordered_findings = _order_findings(findings)
    return ActionGraphReport(
        node_count=len(unique),
        edge_count=sum(len(dependencies) for dependencies in edges.values()),
        order=order if not ordered_findings else (),
        findings=ordered_findings,
    )


def _topological_order(
    edges: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    indegree = {
        action_id: len(dependencies) for action_id, dependencies in edges.items()
    }
    dependents: dict[str, list[str]] = {action_id: [] for action_id in edges}
    for action_id, dependencies in edges.items():
        for dependency in dependencies:
            dependents[dependency].append(action_id)

    ready = [action_id for action_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        action_id = heapq.heappop(ready)
        order.append(action_id)
        for dependent in dependents[action_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(order) == len(edges):
        return tuple(order), ()
    placed = set(order)
    return (), tuple(
        sorted(action_id for action_id in edges if action_id not in placed)
    )


def _order_findings(findings: list[GraphFinding]) -> tuple[GraphFinding, ...]:
    ordered = sorted(
        findings,
        key=lambda finding: (
            finding.action_id or "",
            finding.reason_code,
            finding.dependency_id or "",
        ),
    )
    if len(ordered) > MAX_ACTION_GRAPH_FINDINGS:
        truncated = ordered[: MAX_ACTION_GRAPH_FINDINGS - 1]
        truncated.append(GraphFinding("findings_truncated"))
        return tuple(truncated)
    return tuple(ordered)


def _materialize(nodes: Any) -> tuple[ActionNode, ...]:
    if isinstance(nodes, (str, bytes, bytearray, ActionNode)):
        raise ActionGraphError("invalid_sequence", "nodes")
    try:
        iterator = iter(nodes)
    except TypeError:
        pass
    else:
        collected: list[ActionNode] = []
        for item in iterator:
            if not isinstance(item, ActionNode):
                raise ActionGraphError("invalid_node_type", "nodes")
            if len(collected) == MAX_ACTION_GRAPH_NODES:
                raise ActionGraphError("too_many_nodes", "nodes")
            collected.append(item)
        return tuple(collected)
    raise ActionGraphError("invalid_sequence", "nodes")


def _as_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ActionGraphError("invalid_sequence", field_name)
    try:
        return tuple(value)
    except TypeError:
        pass
    raise ActionGraphError("invalid_sequence", field_name)


def _validate_identifier(value: Any, field_name: str) -> None:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ActionGraphError("invalid_identifier", field_name)


def _validate_tool_id(value: Any) -> None:
    try:
        ToolId.parse(value)
    except GovernanceIdError:
        pass
    else:
        return
    # Raised outside the handler so the rejected identifier cannot be chained.
    raise ActionGraphError("invalid_tool_id", "tool_id")


__all__ = [
    "ACTION_GRAPH_SCHEMA_VERSION",
    "GRAPH_REASON_CODES",
    "MAX_ACTION_DEPENDENCIES",
    "MAX_ACTION_GRAPH_FINDINGS",
    "MAX_ACTION_GRAPH_NODES",
    "ActionGraphError",
    "ActionGraphReport",
    "ActionNode",
    "GraphFinding",
    "validate_action_graph",
]
