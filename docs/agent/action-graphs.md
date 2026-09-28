# Agent Action Graphs

`validate_action_graph` checks a planned multi-step workflow and returns a
deterministic execution order. It is an explicit helper, not a scheduler, an
executor, or a judgement about whether a tool is clinically appropriate.

```python
from openmed.agent.action_graph import ActionNode, validate_action_graph

nodes = [
    ActionNode(action_id="fetch", tool_id="tool:openmed.agent/fetch-notes"),
    ActionNode(
        action_id="summarize",
        tool_id="tool:openmed.agent/summarize@1.0.0",
        depends_on=("fetch",),
    ),
]
report = validate_action_graph(nodes)
print(report.is_valid, report.order)
```

## What a node carries

A node is identifiers only. `action_id` and every entry in `depends_on` are
bounded opaque identifiers matching
`^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$`, so the canonical
`ActionId` strings from [Agent Event Correlation](event-correlation.md) are
accepted unchanged. `tool_id` must be a canonical
[governance identifier](governance-identifiers.md) of kind `tool`.

There is no field for tool arguments, tool outputs, prompts, filesystem paths,
or clinical text, so none of those can enter a graph or a finding. Fan-in is
bounded by `MAX_ACTION_DEPENDENCIES` and the graph by `MAX_ACTION_GRAPH_NODES`.

## Validation and ordering

Structural problems fail closed with `ActionGraphError`: malformed or
non-string identifiers, a non-`ToolId` tool, a string or non-iterable passed
where a sequence is expected, a non-node item, and oversized input.

Graph problems are reported as findings so one pass shows every defect:

| Reason code | Meaning |
| --- | --- |
| `duplicate_action_id` | An action identifier repeats; the first node is kept. |
| `duplicate_dependency` | A node lists the same dependency twice. |
| `self_dependency` | A node depends on itself; the edge is dropped. |
| `missing_dependency` | A dependency names no node in the graph; the edge is dropped. |
| `dependency_cycle` | The node is part of a direct or indirect cycle. |
| `findings_truncated` | More than `MAX_ACTION_GRAPH_FINDINGS` findings were produced. |

Duplicates, self-edges and dangling edges are removed first, and the cleaned
graph is then checked for cycles, so a single call reports both classes of
problem. `order` is populated only when there are no findings; otherwise it is
empty, because a partial order would invite executing a broken plan.

Ordering is Kahn's algorithm with one documented tie-break: whenever several
actions are ready, the smallest action identifier is emitted first. Independent
branches therefore order lexicographically, and reordering the input never
changes `order`, `findings` or `to_json()`. Findings are ordered by action
identifier, then reason code, then dependency identifier.

## Privacy and scope

Findings, reports and exceptions contain opaque identifiers, counts and stable
codes only. `ActionGraphError` exposes a stable `.code` and an optional
`.field_name`, and a rejected tool identifier is not chained onto the raised
error. Executing actions, scheduling parallel work, persisting workflow state,
and clinical appropriateness are out of scope. A valid graph is a structural
statement about ordering, not a clinical or security approval.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/agent/test_action_graph.py -q
```

Fixtures are synthetic. Tests cover linear, branching, joining and independent
graphs, input-order independence, every reason code, finding order,
serialization stability, and bounded fan-in and graph size.
