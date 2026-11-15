"""Bounded image/text provenance graphs, without clinical payload values.

Edges point from evidence to its derivative. Structural citation is not clinical
verification: callers retain protected values separately and declare conflicts.
Reports preserve source coordinates, opaque node references and controlled codes.
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = "openmed.multimodal.fused-evidence.v1"
MAX_NODES = 1024
MAX_EDGES = 4096
_MAX_COORDINATE = 2**31 - 1
_OPAQUE_ID = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class FusedEvidenceError(ValueError):
    """Invalid graph input, with no rejected values in the exception."""


class NodeKind(str, Enum):
    """Distinct evidence origins and derived conclusions."""

    PAGE = "page_geometry"
    PIXELS = "pixels"
    OCR = "ocr_span"
    METADATA = "metadata_field"
    FACT = "normalized_fact"
    CLAIM = "fused_claim"


class SpatialUnit(str, Enum):
    """Coordinate units in an untransformed source page, top-left origin."""

    PIXEL = "pixel"
    POINT = "point"


class EdgeKind(str, Enum):
    """Declared derivation operations; conflicts are undirected."""

    LOCATES = "locates"
    OCR = "ocr"
    EXTRACTS = "extracts"
    INTERPRETS = "interprets"
    NORMALIZES = "normalizes"
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"


class FindingCode(str, Enum):
    """Closed review vocabulary; no user-supplied diagnostic messages."""

    MODALITY_JUMP = "unsupported_modality_jump"
    SOURCE_MISMATCH = "source_coordinate_mismatch"
    CONFLICT = "declared_conflict"
    UNSOURCED = "missing_source_citation"


class ClaimState(str, Enum):
    """Citation state, not a clinical correctness or publication decision."""

    CITED = "cited"
    REVIEW = "review_required"
    UNSUPPORTED = "unsupported"


def _require(condition: bool) -> None:
    if not condition:
        raise FusedEvidenceError("invalid fused evidence")


def _opaque(value: object) -> None:
    _require(type(value) is str and _OPAQUE_ID.fullmatch(value) is not None)


def _integer(value: object, maximum: int = _MAX_COORDINATE) -> None:
    _require(type(value) is int and 0 <= value <= maximum)


def _coordinate(value: object) -> None:
    # Range checks also reject NaN/infinity and huge ints without float coercion.
    _require(
        (type(value) is int or type(value) is float) and 0 <= value <= _MAX_COORDINATE
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PageGeometry:
    """Declared source-page extent; indices are zero-based.

    Args:
        page: Index within the source asset.
        width: Positive width in the declared unit.
        height: Positive height in the declared unit.
        unit: Pixels or PDF points, always with a top-left origin.
    """

    page: int
    width: float
    height: float
    unit: SpatialUnit = SpatialUnit.PIXEL

    def __post_init__(self) -> None:
        _integer(self.page)
        _coordinate(self.width)
        _coordinate(self.height)
        _require(self.width > 0 and self.height > 0)
        _require(type(self.unit) is SpatialUnit)
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceNode:
    """Typed evidence reference, never the referenced clinical content.

    Args:
        node_id: Caller-assigned opaque 128-bit lowercase hexadecimal reference.
        kind: Evidence origin or derived fact/claim.
        source_digest: SHA-256 of the source asset for source nodes only.
        geometry: Source-page geometry for page, pixel and OCR nodes.
        region: Half-open (left, top, right, bottom) source-page rectangle.
        span: Half-open Unicode code-point offsets in the caller's OCR text.
        field_index: Metadata field ordinal in the caller's protected inventory.
    """

    node_id: str
    kind: NodeKind
    source_digest: str | None = None
    geometry: PageGeometry | None = None
    region: tuple[float, float, float, float] | None = None
    span: tuple[int, int] | None = None
    field_index: int | None = None

    def __post_init__(self) -> None:
        _opaque(self.node_id)
        _require(type(self.kind) is NodeKind)
        source = self.kind not in (NodeKind.FACT, NodeKind.CLAIM)
        if source:
            _require(
                type(self.source_digest) is str
                and _DIGEST.fullmatch(self.source_digest) is not None
            )
        else:
            _require(self.source_digest is None)
        spatial = self.kind in (NodeKind.PAGE, NodeKind.PIXELS, NodeKind.OCR)
        _require(
            type(self.geometry) is PageGeometry if spatial else self.geometry is None
        )
        if self.kind in (NodeKind.PIXELS, NodeKind.OCR):
            _require(type(self.region) is tuple and len(self.region) == 4)
            assert self.region is not None and self.geometry is not None
            for coordinate in self.region:
                _coordinate(coordinate)
            x0, y0, x1, y1 = self.region
            _require(0 <= x0 < x1 <= self.geometry.width)
            _require(0 <= y0 < y1 <= self.geometry.height)
            object.__setattr__(self, "region", tuple(float(x) for x in self.region))
        else:
            _require(self.region is None)
        if self.kind is NodeKind.OCR:
            _require(type(self.span) is tuple and len(self.span) == 2)
            assert self.span is not None
            for offset in self.span:
                _integer(offset)
            _require(self.span[0] < self.span[1])
        else:
            _require(self.span is None)
        if self.kind is NodeKind.METADATA:
            _integer(self.field_index)
        else:
            _require(self.field_index is None)


@dataclass(frozen=True, slots=True, kw_only=True)
class DerivationEdge:
    """An explicit evidence-to-derivative relation between opaque node IDs.

    Args:
        source: Evidence node ID.
        target: Derived node ID; conflict endpoints are canonicalized instead.
        kind: Operation connecting the nodes.
    """

    source: str
    target: str
    kind: EdgeKind

    def __post_init__(self) -> None:
        _opaque(self.source)
        _opaque(self.target)
        _require(self.source != self.target and type(self.kind) is EdgeKind)
        if self.kind is EdgeKind.CONFLICTS and self.source > self.target:
            source, target = self.target, self.source
            object.__setattr__(self, "source", source)
            object.__setattr__(self, "target", target)


@dataclass(frozen=True, slots=True)
class Finding:
    """A controlled structural finding attached to opaque references."""

    code: FindingCode
    node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(type(self.code) is FindingCode)
        _require(type(self.node_ids) is tuple and 1 <= len(self.node_ids) <= 2)
        for node_id in self.node_ids:
            _opaque(node_id)


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """Source ancestry and inherited review findings for one fused claim."""

    claim_id: str
    state: ClaimState
    source_node_ids: tuple[str, ...]
    finding_codes: tuple[FindingCode, ...]

    def __post_init__(self) -> None:
        _opaque(self.claim_id)
        _require(type(self.state) is ClaimState)
        _require(
            type(self.source_node_ids) is tuple
            and len(self.source_node_ids) <= MAX_NODES
        )
        for node_id in self.source_node_ids:
            _opaque(node_id)
        _require(
            type(self.finding_codes) is tuple
            and len(self.finding_codes) <= len(FindingCode)
        )
        _require(all(type(code) is FindingCode for code in self.finding_codes))


_ALLOWED: dict[EdgeKind, set[tuple[NodeKind, NodeKind]]] = {
    EdgeKind.LOCATES: {(NodeKind.PAGE, NodeKind.PIXELS)},
    EdgeKind.OCR: {(NodeKind.PIXELS, NodeKind.OCR)},
    EdgeKind.EXTRACTS: {
        (NodeKind.OCR, NodeKind.FACT),
        (NodeKind.METADATA, NodeKind.FACT),
    },
    EdgeKind.INTERPRETS: {(NodeKind.PIXELS, NodeKind.FACT)},
    EdgeKind.NORMALIZES: {(NodeKind.FACT, NodeKind.FACT)},
    EdgeKind.SUPPORTS: {
        (kind, NodeKind.CLAIM)
        for kind in (NodeKind.PIXELS, NodeKind.OCR, NodeKind.METADATA, NodeKind.FACT)
    },
}


def _edge_problem(
    edge: DerivationEdge, nodes: dict[str, EvidenceNode]
) -> FindingCode | None:
    source, target = nodes[edge.source], nodes[edge.target]
    if (source.kind, target.kind) not in _ALLOWED[edge.kind]:
        return FindingCode.MODALITY_JUMP
    if edge.kind in (EdgeKind.LOCATES, EdgeKind.OCR):
        if (
            source.source_digest != target.source_digest
            or source.geometry != target.geometry
        ):
            return FindingCode.SOURCE_MISMATCH
        if source.region is not None:
            assert target.region is not None
            a, b = source.region, target.region
            if not (a[0] <= b[0] < b[2] <= a[2] and a[1] <= b[1] < b[3] <= a[3]):
                return FindingCode.SOURCE_MISMATCH
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class FusedEvidenceGraph:
    """Immutable, bounded DAG with separate declared conflict edges.

    Args:
        nodes: At most 1,024 typed nodes, with unique opaque references.
        edges: At most 4,096 unique typed edges. Derivation cycles are rejected.

    Missing endpoints and malformed shapes fail at construction. Unsupported
    derivations and conflicts remain visible as findings and taint downstream
    claims. Input order never changes canonical serialization or assessments.
    """

    nodes: tuple[EvidenceNode, ...]
    edges: tuple[DerivationEdge, ...] = ()

    def __post_init__(self) -> None:
        _require(type(self.nodes) is tuple and 0 < len(self.nodes) <= MAX_NODES)
        _require(type(self.edges) is tuple and len(self.edges) <= MAX_EDGES)
        _require(all(type(node) is EvidenceNode for node in self.nodes))
        _require(all(type(edge) is DerivationEdge for edge in self.edges))
        ids = {node.node_id for node in self.nodes}
        _require(
            len(ids) == len(self.nodes) and len(set(self.edges)) == len(self.edges)
        )
        _require(all(edge.source in ids and edge.target in ids for edge in self.edges))
        frames: dict[tuple[str | None, int], PageGeometry] = {}
        for node in self.nodes:
            if node.geometry is not None:
                key = (node.source_digest, node.geometry.page)
                _require(key not in frames or frames[key] == node.geometry)
                frames[key] = node.geometry
        object.__setattr__(
            self, "nodes", tuple(sorted(self.nodes, key=lambda n: n.node_id))
        )
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(self.edges, key=lambda e: (e.source, e.target, e.kind))),
        )
        self._ordered()

    def _ordered(self) -> tuple[str, ...]:
        degree = dict.fromkeys((node.node_id for node in self.nodes), 0)
        children: dict[str, list[str]] = {key: [] for key in degree}
        for edge in self.edges:
            if edge.kind is not EdgeKind.CONFLICTS:
                degree[edge.target] += 1
                children[edge.source].append(edge.target)
        ready = [key for key in degree if degree[key] == 0]
        heapq.heapify(ready)
        ordered = []
        while ready:
            key = heapq.heappop(ready)
            ordered.append(key)
            for child in children[key]:
                degree[child] -= 1
                if degree[child] == 0:
                    heapq.heappush(ready, child)
        _require(len(ordered) == len(self.nodes))
        return tuple(ordered)

    def _analyze(self) -> tuple[tuple[Finding, ...], tuple[ClaimAssessment, ...]]:
        nodes = {node.node_id: node for node in self.nodes}
        sources: dict[str, set[str]] = {key: set() for key in nodes}
        problems: dict[str, set[FindingCode]] = {key: set() for key in nodes}
        parents: dict[str, list[tuple[str, bool]]] = {key: [] for key in nodes}
        findings: set[Finding] = set()
        for edge in self.edges:
            if edge.kind is EdgeKind.CONFLICTS:
                code = FindingCode.CONFLICT
                problems[edge.source].add(code)
                problems[edge.target].add(code)
                findings.add(Finding(code, (edge.source, edge.target)))
            else:
                problem = _edge_problem(edge, nodes)
                parents[edge.target].append((edge.source, problem is None))
                if problem is not None:
                    problems[edge.target].add(problem)
                    findings.add(Finding(problem, (edge.source, edge.target)))
        for key in self._ordered():
            node = nodes[key]
            if node.kind not in (NodeKind.FACT, NodeKind.CLAIM):
                sources[key].add(key)
            for parent, valid in parents[key]:
                problems[key].update(problems[parent])
                if valid:
                    sources[key].update(sources[parent])
            if node.kind in (NodeKind.FACT, NodeKind.CLAIM) and not sources[key]:
                problems[key].add(FindingCode.UNSOURCED)
                findings.add(Finding(FindingCode.UNSOURCED, (key,)))
        claims = []
        for node in self.nodes:
            if node.kind is not NodeKind.CLAIM:
                continue
            key = node.node_id
            state = (
                ClaimState.UNSUPPORTED
                if not sources[key]
                else ClaimState.REVIEW
                if problems[key]
                else ClaimState.CITED
            )
            claims.append(
                ClaimAssessment(
                    key,
                    state,
                    tuple(sorted(sources[key])),
                    tuple(sorted(problems[key])),
                )
            )
        return (
            tuple(sorted(findings, key=lambda f: (f.code, f.node_ids))),
            tuple(claims),
        )

    def findings(self) -> tuple[Finding, ...]:
        """Return deterministic structural defects and declared conflicts."""
        return self._analyze()[0]

    def assess_claims(self) -> tuple[ClaimAssessment, ...]:
        """Return source citations and inherited holds for every fused claim."""
        return self._analyze()[1]

    def to_dict(self) -> dict[str, Any]:
        """Export source coordinates, derivations and assessments without values."""
        findings, claims = self._analyze()
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "findings": [asdict(finding) for finding in findings],
            "claims": [asdict(claim) for claim in claims],
        }

    def to_json(self) -> str:
        """Return canonical ASCII JSON, with no source payload or network IO."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
