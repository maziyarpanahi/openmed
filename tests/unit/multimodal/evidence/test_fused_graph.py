"""Coordinate, lineage, hold-propagation and value-free failure controls."""

from dataclasses import replace
from itertools import permutations

import pytest

from openmed.multimodal.evidence import (
    ClaimState,
    DerivationEdge,
    EdgeKind,
    EvidenceNode,
    Finding,
    FindingCode,
    FusedEvidenceError,
    FusedEvidenceGraph,
    NodeKind,
    PageGeometry,
    SpatialUnit,
)
from openmed.multimodal.evidence.fused_graph import MAX_EDGES, MAX_NODES


def ref(index):
    return f"{index:032x}"


def edge(source, target, kind):
    return DerivationEdge(source=ref(source), target=ref(target), kind=kind)


@pytest.fixture
def graph():
    geometry = PageGeometry(page=0, width=100, height=200)
    source = dict(source_digest="a" * 64, geometry=geometry)
    return FusedEvidenceGraph(
        nodes=(
            EvidenceNode(node_id=ref(1), kind=NodeKind.PAGE, **source),
            EvidenceNode(
                node_id=ref(2), kind=NodeKind.PIXELS, region=(0, 0, 60, 30), **source
            ),
            EvidenceNode(
                node_id=ref(3),
                kind=NodeKind.OCR,
                region=(5, 5, 50, 20),
                span=(10, 18),
                **source,
            ),
            EvidenceNode(
                node_id=ref(4),
                kind=NodeKind.METADATA,
                source_digest="b" * 64,
                field_index=7,
            ),
            EvidenceNode(node_id=ref(5), kind=NodeKind.FACT),
            EvidenceNode(node_id=ref(6), kind=NodeKind.FACT),
            EvidenceNode(node_id=ref(7), kind=NodeKind.CLAIM),
        ),
        edges=(
            edge(1, 2, EdgeKind.LOCATES),
            edge(2, 3, EdgeKind.OCR),
            edge(3, 5, EdgeKind.EXTRACTS),
            edge(4, 6, EdgeKind.EXTRACTS),
            edge(5, 7, EdgeKind.SUPPORTS),
            edge(6, 7, EdgeKind.SUPPORTS),
        ),
    )


def test_fused_claim_retains_separate_ocr_and_metadata_sources(graph):
    (claim,) = graph.assess_claims()
    assert claim.state is ClaimState.CITED
    assert claim.source_node_ids == tuple(ref(i) for i in range(1, 5))
    assert graph.findings() == ()
    nodes = graph.to_dict()["nodes"]
    assert nodes[2]["span"] == (10, 18)
    assert nodes[2]["region"] == (5, 5, 50, 20)
    assert nodes[2]["geometry"]["page"] == 0
    assert nodes[2]["source_digest"] != nodes[3]["source_digest"]
    assert nodes[3]["field_index"] == 7


def test_order_does_not_change_bytes_or_findings(graph):
    expected = graph.to_json()
    for order in permutations(graph.nodes[:3]):
        reordered = replace(
            graph, nodes=order + graph.nodes[3:], edges=tuple(reversed(graph.edges))
        )
        assert reordered.to_json() == expected


def test_conflicts_taint_descendants_and_have_no_direction(graph):
    forward = replace(graph, edges=graph.edges + (edge(5, 6, EdgeKind.CONFLICTS),))
    reverse = replace(graph, edges=graph.edges + (edge(6, 5, EdgeKind.CONFLICTS),))
    assert forward.to_json() == reverse.to_json()
    assert forward.assess_claims()[0].state is ClaimState.REVIEW
    assert forward.assess_claims()[0].finding_codes == (FindingCode.CONFLICT,)
    assert (
        forward.assess_claims()[0].source_node_ids
        == graph.assess_claims()[0].source_node_ids
    )


def test_missing_citation_is_not_rescued_by_conflict_edge(graph):
    orphan = EvidenceNode(node_id=ref(8), kind=NodeKind.CLAIM)
    graph = replace(
        graph,
        nodes=graph.nodes + (orphan,),
        edges=graph.edges + (edge(7, 8, EdgeKind.CONFLICTS),),
    )
    claim = graph.assess_claims()[1]
    assert claim.state is ClaimState.UNSUPPORTED
    assert claim.source_node_ids == ()
    assert set(claim.finding_codes) == {FindingCode.CONFLICT, FindingCode.UNSOURCED}


def test_unsupported_modality_jump_never_becomes_a_citation():
    metadata = EvidenceNode(
        node_id=ref(1), kind=NodeKind.METADATA, source_digest="a" * 64, field_index=0
    )
    fact = EvidenceNode(node_id=ref(2), kind=NodeKind.FACT)
    claim = EvidenceNode(node_id=ref(3), kind=NodeKind.CLAIM)
    graph = FusedEvidenceGraph(
        nodes=(metadata, fact, claim),
        edges=(edge(1, 2, EdgeKind.INTERPRETS), edge(2, 3, EdgeKind.SUPPORTS)),
    )
    assert graph.assess_claims()[0].state is ClaimState.UNSUPPORTED
    assert graph.assess_claims()[0].source_node_ids == ()
    assert FindingCode.MODALITY_JUMP in graph.assess_claims()[0].finding_codes


@pytest.mark.parametrize(
    "change",
    [
        dict(source_digest="c" * 64),
        dict(region=(50, 5, 70, 20)),
        dict(geometry=PageGeometry(page=1, width=100, height=200)),
    ],
)
def test_wrong_asset_page_or_outside_parent_region_requires_review(graph, change):
    nodes = list(graph.nodes)
    nodes[2] = replace(nodes[2], **change)
    graph = replace(graph, nodes=tuple(nodes))
    claim = graph.assess_claims()[0]
    assert claim.state is ClaimState.REVIEW
    assert FindingCode.SOURCE_MISMATCH in claim.finding_codes
    # The malformed OCR connection cannot borrow the original pixel citation.
    assert ref(2) not in claim.source_node_ids
    assert ref(3) in claim.source_node_ids


def test_valid_branch_does_not_hide_unsupported_branch(graph):
    orphan = EvidenceNode(node_id=ref(8), kind=NodeKind.FACT)
    graph = replace(
        graph,
        nodes=graph.nodes + (orphan,),
        edges=graph.edges + (edge(8, 7, EdgeKind.SUPPORTS),),
    )
    assert graph.assess_claims()[0].state is ClaimState.REVIEW
    assert FindingCode.UNSOURCED in graph.assess_claims()[0].finding_codes


def test_normalization_retains_transitive_ancestry(graph):
    fact = EvidenceNode(node_id=ref(8), kind=NodeKind.FACT)
    edges = tuple(e for e in graph.edges if e != edge(5, 7, EdgeKind.SUPPORTS))
    graph = replace(
        graph,
        nodes=graph.nodes + (fact,),
        edges=edges + (edge(5, 8, EdgeKind.NORMALIZES), edge(8, 7, EdgeKind.SUPPORTS)),
    )
    assert graph.assess_claims()[0].state is ClaimState.CITED
    assert ref(3) in graph.assess_claims()[0].source_node_ids


def test_visual_inference_is_distinct_from_ocr_extraction(graph):
    edges = tuple(e for e in graph.edges if e != edge(3, 5, EdgeKind.EXTRACTS))
    valid = replace(graph, edges=edges + (edge(2, 5, EdgeKind.INTERPRETS),))
    invalid = replace(graph, edges=edges + (edge(2, 5, EdgeKind.EXTRACTS),))
    assert valid.assess_claims()[0].state is ClaimState.CITED
    assert invalid.assess_claims()[0].state is ClaimState.REVIEW
    assert FindingCode.MODALITY_JUMP in invalid.assess_claims()[0].finding_codes


@pytest.mark.parametrize(
    "extra",
    [
        (edge(5, 6, EdgeKind.NORMALIZES), edge(6, 5, EdgeKind.NORMALIZES)),
        (edge(1, 999, EdgeKind.LOCATES),),
    ],
)
def test_cycles_and_dangling_references_fail_closed(graph, extra):
    with pytest.raises(FusedEvidenceError, match="^invalid fused evidence$"):
        replace(graph, edges=graph.edges + extra)


def test_duplicate_nodes_and_edges_are_rejected(graph):
    for changes in (
        dict(nodes=graph.nodes + graph.nodes[:1]),
        dict(edges=graph.edges + graph.edges[:1]),
    ):
        with pytest.raises(FusedEvidenceError):
            replace(graph, **changes)


@pytest.mark.parametrize(
    "geometry",
    [
        PageGeometry(page=0, width=101, height=200),
        PageGeometry(page=0, width=100, height=200, unit=SpatialUnit.POINT),
    ],
)
def test_one_source_page_cannot_have_inconsistent_geometry(graph, geometry):
    with pytest.raises(FusedEvidenceError):
        replace(
            graph, nodes=(replace(graph.nodes[0], geometry=geometry),) + graph.nodes[1:]
        )


@pytest.mark.parametrize(
    "value", [True, -1, float("nan"), float("inf"), 10**1000, "synthetic-secret-value"]
)
def test_invalid_geometry_is_value_free(value):
    with pytest.raises(FusedEvidenceError) as caught:
        PageGeometry(page=0, width=value, height=10)
    assert str(caught.value) == "invalid fused evidence"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "changes",
    [
        dict(node_id="synthetic-secret-value"),
        dict(source_digest="synthetic-secret-value"),
        dict(region=(0, 0, 101, 20)),
        dict(region=(0, 0, 0, 20)),
        dict(region=[0, 0, 5, 5]),
        dict(span=(0, 1)),
        dict(field_index=0),
        dict(kind="pixels"),
    ],
)
def test_source_shapes_reject_invalid_or_free_text_fields(graph, changes):
    with pytest.raises(FusedEvidenceError, match="^invalid fused evidence$"):
        replace(graph.nodes[1], **changes)


@pytest.mark.parametrize("span", [(1, 1), (-1, 5), (True, 5), (0, 2**40), [0, 2]])
def test_ocr_span_integrity(graph, span):
    with pytest.raises(FusedEvidenceError):
        replace(graph.nodes[2], span=span)


def test_derived_nodes_cannot_smuggle_source_fields(graph):
    with pytest.raises(FusedEvidenceError):
        replace(graph.nodes[-1], source_digest="a" * 64)
    with pytest.raises(FusedEvidenceError):
        Finding("synthetic-secret-value", (ref(1),))


def test_limits_fail_before_traversal(graph):
    with pytest.raises(FusedEvidenceError):
        replace(graph, nodes=graph.nodes[:1] * (MAX_NODES + 1))
    with pytest.raises(FusedEvidenceError):
        replace(graph, edges=graph.edges[:1] * (MAX_EDGES + 1))


def test_iterative_traversal_handles_deep_lineage_without_recursion():
    source = EvidenceNode(
        node_id=ref(0), kind=NodeKind.METADATA, source_digest="a" * 64, field_index=0
    )
    facts = tuple(
        EvidenceNode(node_id=ref(i), kind=NodeKind.FACT)
        for i in range(1, MAX_NODES - 1)
    )
    claim = EvidenceNode(node_id=ref(MAX_NODES - 1), kind=NodeKind.CLAIM)
    edges = (
        (edge(0, 1, EdgeKind.EXTRACTS),)
        + tuple(edge(i, i + 1, EdgeKind.NORMALIZES) for i in range(1, MAX_NODES - 2))
        + (edge(MAX_NODES - 2, MAX_NODES - 1, EdgeKind.SUPPORTS),)
    )
    graph = FusedEvidenceGraph(nodes=(source,) + facts + (claim,), edges=edges)
    assert graph.assess_claims()[0].state is ClaimState.CITED
    assert graph.assess_claims()[0].source_node_ids == (ref(0),)


def test_reports_and_representations_have_no_payload_fields(graph, caplog):
    report = graph.to_json()
    assert "openmed.multimodal.fused-evidence.v1" in report
    assert all(
        key not in report for key in ('"text":', '"value":', '"path":', '"url":')
    )
    assert "synthetic-secret-value" not in repr(graph)
    assert caplog.records == []
