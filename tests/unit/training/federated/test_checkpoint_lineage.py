"""Offline complete DAG ancestry, stage, cycle and privacy-boundary controls."""

import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.training.federated.checkpoint_lineage import (
    CheckpointLineageGraph,
    CheckpointLineageNode,
)


def digest(index):
    return "sha256:" + f"{index:064x}"


def chain():
    return [
        CheckpointLineageNode(digest(1), "base_model"),
        CheckpointLineageNode(digest(2), "round_manifest", (digest(1),)),
        CheckpointLineageNode(digest(3), "aggregate", (digest(2),)),
        CheckpointLineageNode(digest(4), "released_adapter", (digest(3),)),
    ]


def test_complete_base_round_aggregate_adapter_ancestry():
    graph = CheckpointLineageGraph(chain())
    assert [node.kind for node in graph.ancestry(digest(4))] == [
        "base_model",
        "round_manifest",
        "aggregate",
        "released_adapter",
    ]
    result = json.loads(graph.to_json(digest(4)))
    assert set(result) == {"schema_version", "checkpoint_digest", "nodes"}
    assert all(
        set(node) == {"digest", "kind", "parent_digests"} for node in result["nodes"]
    )


def test_reordered_graph_and_parent_order_render_identically():
    nodes = chain() + [
        CheckpointLineageNode(digest(5), "round_manifest", (digest(4),)),
        CheckpointLineageNode(digest(6), "aggregate", (digest(2), digest(5))),
        CheckpointLineageNode(digest(7), "released_adapter", (digest(6),)),
    ]
    a = CheckpointLineageGraph(nodes)
    b = CheckpointLineageGraph(
        CheckpointLineageNode(
            node.digest, node.kind, tuple(reversed(node.parent_digests))
        )
        for node in reversed(nodes)
    )
    assert a.to_json(digest(7)) == b.to_json(digest(7))
    assert len(a.ancestry(digest(7))) == 7


def test_unrelated_branches_are_excluded():
    graph = CheckpointLineageGraph(
        chain() + [CheckpointLineageNode(digest(8), "base_model")]
    )
    assert digest(8) not in graph.to_json(digest(4))
    assert graph.ancestry(digest(1)) == (chain()[0],)


@pytest.mark.parametrize(
    "nodes",
    [
        chain() + [chain()[0]],
        [CheckpointLineageNode(digest(2), "round_manifest", (digest(1),))],
        [
            CheckpointLineageNode(digest(1), "base_model"),
            CheckpointLineageNode(digest(3), "aggregate", (digest(1),)),
        ],
        [
            CheckpointLineageNode(digest(2), "round_manifest", (digest(4),)),
            CheckpointLineageNode(digest(3), "aggregate", (digest(2),)),
            CheckpointLineageNode(digest(4), "released_adapter", (digest(3),)),
        ],
    ],
)
def test_duplicate_unresolved_wrong_stage_and_cycle_reject(nodes):
    with pytest.raises(ValueError):
        CheckpointLineageGraph(nodes)


@pytest.mark.parametrize(
    "args",
    [
        ("synthetic/private", "base_model", ()),
        (digest(1), "unknown", ()),
        (digest(1), "base_model", (digest(2),)),
        (digest(2), "aggregate", ()),
        (digest(2), "round_manifest", (digest(2),)),
        (digest(2), "round_manifest", (digest(1), digest(1))),
    ],
)
def test_invalid_nodes_reject_without_echo(args):
    with pytest.raises(ValueError) as exc:
        CheckpointLineageNode(*args)
    assert "synthetic/private" not in str(exc.value)


def test_missing_and_noncheckpoint_targets_reject():
    graph = CheckpointLineageGraph(chain())
    for target in [digest(99), digest(2), digest(3), "synthetic/private"]:
        with pytest.raises(ValueError):
            graph.to_json(target)


def test_snapshot_parent_inputs_and_frozen_nodes():
    parents = [digest(1)]
    node = CheckpointLineageNode(digest(2), "round_manifest", parents)
    parents.append(digest(99))
    assert node.parent_digests == (digest(1),)
    with pytest.raises(FrozenInstanceError):
        node.kind = "base_model"


def test_infinite_nodes_and_parents_consume_only_bounded_prefixes():
    counts = []

    def nodes():
        while True:
            counts.append(1)
            yield chain()[0]

    with pytest.raises(ValueError, match="limit"):
        CheckpointLineageGraph(nodes())
    assert len(counts) == 10001
    counts.clear()

    def parents():
        while True:
            counts.append(1)
            yield digest(1)

    with pytest.raises(ValueError, match="limit"):
        CheckpointLineageNode(digest(2), "round_manifest", parents())
    assert len(counts) == 33


def test_long_lineage_uses_iterative_traversal():
    nodes = [CheckpointLineageNode(digest(1), "base_model")]
    for i in range(2, 1502):
        kind = ("round_manifest", "aggregate", "released_adapter")[(i - 2) % 3]
        nodes.append(CheckpointLineageNode(digest(i), kind, (digest(i - 1),)))
    graph = CheckpointLineageGraph(nodes)
    assert len(graph.ancestry(digest(1501))) == 1501
