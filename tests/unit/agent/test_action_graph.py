"""Synthetic unit tests for agent action dependency graphs."""

from __future__ import annotations

import json

import pytest

from openmed.agent.action_graph import (
    ACTION_GRAPH_SCHEMA_VERSION,
    GRAPH_REASON_CODES,
    MAX_ACTION_DEPENDENCIES,
    MAX_ACTION_GRAPH_FINDINGS,
    MAX_ACTION_GRAPH_NODES,
    ActionGraphError,
    ActionGraphReport,
    ActionNode,
    GraphFinding,
    validate_action_graph,
)

TOOL = "tool:openmed.agent/summarize"
OTHER_TOOL = "tool:openmed.agent/extract@1.2.0"

SENTINEL_PAYLOAD = "PATIENT JANE DOE MRN 8675309 Bearer abc.def /var/phi/note.txt"


def node(action_id: str, *dependencies: str, tool_id: str = TOOL) -> ActionNode:
    return ActionNode(action_id=action_id, tool_id=tool_id, depends_on=dependencies)


def codes(report: ActionGraphReport) -> list[str]:
    return list(report.reason_codes)


def test_linear_graph_orders_by_dependency() -> None:
    report = validate_action_graph([node("c", "b"), node("b", "a"), node("a")])
    assert report.is_valid
    assert report.order == ("a", "b", "c")
    assert (report.node_count, report.edge_count) == (3, 2)
    assert report.schema_version == ACTION_GRAPH_SCHEMA_VERSION


def test_branching_graph_breaks_ties_by_action_id() -> None:
    report = validate_action_graph(
        [node("root"), node("beta", "root"), node("alpha", "root")]
    )
    assert report.order == ("root", "alpha", "beta")


def test_joining_graph_places_the_join_last() -> None:
    report = validate_action_graph(
        [node("join", "left", "right"), node("left"), node("right")]
    )
    assert report.order == ("left", "right", "join")
    assert report.edge_count == 2


def test_independent_actions_order_lexicographically() -> None:
    report = validate_action_graph([node("z"), node("m"), node("a")])
    assert report.order == ("a", "m", "z")
    assert report.edge_count == 0


def test_input_order_does_not_change_the_result() -> None:
    nodes = [node("join", "left", "right"), node("left", "root"), node("right", "root")]
    nodes.append(node("root"))
    first = validate_action_graph(nodes)
    second = validate_action_graph(list(reversed(nodes)))
    assert first.to_json() == second.to_json()
    assert first.order == ("root", "left", "right", "join")


def test_empty_graph_is_valid_and_empty() -> None:
    report = validate_action_graph([])
    assert report.is_valid
    assert (report.order, report.node_count, report.edge_count) == ((), 0, 0)


def test_single_node_graph() -> None:
    report = validate_action_graph((item for item in [node("only")]))
    assert report.order == ("only",)


def test_duplicate_action_id_keeps_the_first_node() -> None:
    report = validate_action_graph(
        [node("a"), node("a", tool_id=OTHER_TOOL), node("b", "a")]
    )
    assert codes(report) == ["duplicate_action_id"]
    assert report.findings[0].action_id == "a"
    assert report.node_count == 2
    assert report.order == ()


def test_missing_dependency_is_reported_and_dropped() -> None:
    report = validate_action_graph([node("a", "ghost")])
    assert codes(report) == ["missing_dependency"]
    assert report.findings[0].dependency_id == "ghost"
    assert report.edge_count == 0


def test_self_dependency_is_reported_and_dropped() -> None:
    report = validate_action_graph([node("a", "a")])
    assert codes(report) == ["self_dependency"]
    assert (report.findings[0].action_id, report.findings[0].dependency_id) == (
        "a",
        "a",
    )
    assert report.edge_count == 0


def test_duplicate_dependency_is_reported_once() -> None:
    report = validate_action_graph([node("a"), node("b", "a", "a")])
    assert codes(report) == ["duplicate_dependency"]
    assert report.edge_count == 1


def test_direct_cycle_is_reported_for_every_member() -> None:
    report = validate_action_graph([node("a", "b"), node("b", "a")])
    assert codes(report) == ["dependency_cycle", "dependency_cycle"]
    assert [finding.action_id for finding in report.findings] == ["a", "b"]
    assert report.order == ()


def test_indirect_cycle_is_reported_without_acyclic_nodes() -> None:
    report = validate_action_graph(
        [node("a", "c"), node("b", "a"), node("c", "b"), node("free")]
    )
    assert [finding.action_id for finding in report.findings] == ["a", "b", "c"]
    assert set(report.reason_codes) == {"dependency_cycle"}


def test_findings_are_ordered_deterministically() -> None:
    report = validate_action_graph(
        [node("b", "b", "ghost"), node("a", "missing"), node("a")]
    )
    assert [
        (finding.action_id, finding.reason_code, finding.dependency_id)
        for finding in report.findings
    ] == [
        ("a", "duplicate_action_id", None),
        ("a", "missing_dependency", "missing"),
        ("b", "missing_dependency", "ghost"),
        ("b", "self_dependency", "b"),
    ]


def test_findings_are_truncated_with_a_stable_marker() -> None:
    nodes = [
        node(f"a{index:05d}", "ghost")
        for index in range(MAX_ACTION_GRAPH_FINDINGS + 10)
    ]
    report = validate_action_graph(nodes)
    assert len(report.findings) == MAX_ACTION_GRAPH_FINDINGS
    assert report.findings[-1].reason_code == "findings_truncated"
    assert report.findings[-1].action_id is None


def test_report_serialization_is_byte_stable_and_field_ordered() -> None:
    report = validate_action_graph([node("a"), node("b", "a")])
    assert list(report.to_dict()) == [
        "schema_version",
        "node_count",
        "edge_count",
        "order",
        "findings",
    ]
    assert json.loads(report.to_json()) == report.to_dict()
    assert report.to_json() == report.to_json()


def test_node_serialization_is_identifier_only() -> None:
    assert node("a", "b").to_dict() == {
        "action_id": "a",
        "tool_id": TOOL,
        "depends_on": ["b"],
    }
    assert list(GraphFinding("self_dependency", "a", "a").to_dict()) == [
        "reason_code",
        "action_id",
        "dependency_id",
    ]


@pytest.mark.parametrize("value", ["", "-lead", "trail-", "a" * 129, 7, None, b"x"])
def test_invalid_action_identifiers_fail_closed(value) -> None:
    with pytest.raises(ActionGraphError) as excinfo:
        ActionNode(action_id=value, tool_id=TOOL)
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "invalid_identifier",
        "action_id",
    )


@pytest.mark.parametrize("value", ["", "-lead", 7, None])
def test_invalid_dependency_identifiers_fail_closed(value) -> None:
    with pytest.raises(ActionGraphError, match="^depends_on: invalid_identifier$"):
        ActionNode(action_id="a", tool_id=TOOL, depends_on=(value,))


@pytest.mark.parametrize(
    "value",
    ["", "summarize", "policy:openmed.agent/summarize", "tool:openmed/Summarize", 7],
)
def test_invalid_tool_identifiers_fail_closed(value) -> None:
    with pytest.raises(ActionGraphError, match="^tool_id: invalid_tool_id$"):
        ActionNode(action_id="a", tool_id=value)


@pytest.mark.parametrize("value", ["abc", b"abc", 7, None])
def test_invalid_dependency_containers_fail_closed(value) -> None:
    with pytest.raises(ActionGraphError, match="^depends_on: invalid_sequence$"):
        ActionNode(action_id="a", tool_id=TOOL, depends_on=value)


def test_dependency_fan_in_is_bounded() -> None:
    dependencies = tuple(f"d{index:03d}" for index in range(MAX_ACTION_DEPENDENCIES))
    assert ActionNode("a", TOOL, dependencies).depends_on == dependencies
    with pytest.raises(ActionGraphError, match="^depends_on: too_many_dependencies$"):
        ActionNode("a", TOOL, dependencies + ("extra",))


@pytest.mark.parametrize("nodes", [node("a"), "abc", b"abc", 5, None, [node("a"), 1]])
def test_invalid_node_containers_fail_closed(nodes) -> None:
    with pytest.raises(ActionGraphError) as excinfo:
        validate_action_graph(nodes)
    assert excinfo.value.code in {"invalid_sequence", "invalid_node_type"}
    assert excinfo.value.field_name == "nodes"


def test_oversized_graphs_fail_closed() -> None:
    class _Endless:
        def __iter__(self):
            index = 0
            while True:
                yield node(f"a{index:06d}")
                index += 1

    with pytest.raises(ActionGraphError, match="^nodes: too_many_nodes$"):
        validate_action_graph(_Endless())
    assert MAX_ACTION_GRAPH_NODES > 0


def test_unknown_reason_codes_fail_closed() -> None:
    with pytest.raises(ActionGraphError, match="^reason_code: unknown_reason_code$"):
        GraphFinding("not_a_reason")
    assert "findings_truncated" in GRAPH_REASON_CODES


def test_invalid_schema_version_fails_closed() -> None:
    with pytest.raises(ActionGraphError, match="^schema_version: "):
        ActionGraphReport(
            node_count=0,
            edge_count=0,
            order=(),
            findings=(),
            schema_version="openmed.agent.action_graph.v0",
        )


def test_no_payload_text_can_reach_reports_or_errors() -> None:
    report = validate_action_graph([node("a", "ghost")])
    assert SENTINEL_PAYLOAD not in report.to_json()

    with pytest.raises(ActionGraphError) as excinfo:
        ActionNode(action_id=SENTINEL_PAYLOAD, tool_id=TOOL)
    assert SENTINEL_PAYLOAD not in str(excinfo.value)
    assert SENTINEL_PAYLOAD not in repr(excinfo.value)

    with pytest.raises(ActionGraphError) as tool_error:
        ActionNode(action_id="a", tool_id=SENTINEL_PAYLOAD)
    assert SENTINEL_PAYLOAD not in str(tool_error.value)
    assert tool_error.value.__cause__ is None


def test_action_graph_contract_is_available_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.ActionNode is ActionNode
    assert agent.ActionGraphError is ActionGraphError
    assert agent.ActionGraphReport is ActionGraphReport
    assert agent.GraphFinding is GraphFinding
    assert agent.validate_action_graph is validate_action_graph
    exported = {
        "ACTION_GRAPH_SCHEMA_VERSION",
        "ActionGraphError",
        "ActionGraphReport",
        "ActionNode",
        "GRAPH_REASON_CODES",
        "GraphFinding",
        "MAX_ACTION_DEPENDENCIES",
        "MAX_ACTION_GRAPH_FINDINGS",
        "MAX_ACTION_GRAPH_NODES",
        "validate_action_graph",
    }
    assert exported.issubset(set(agent.__all__))
