"""Evidence-linked, value-free graph view for longitudinal journey events."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import canonical_json

TIMELINE_GRAPH_SCHEMA_VERSION = "1.0.0"
TIMELINE_GRAPH_COMPATIBILITY_POLICY = "same_major"
TIMELINE_GRAPH_STATES = frozenset({"current", "historical", "conflicted"})
TIMELINE_CORRECTION_STATES = frozenset(
    {"none", "amends", "superseded", "amends_and_superseded"}
)

_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class TimelineGraphNode:
    """One ordered fact-backed timeline node without a clinical value."""

    event_id: str
    fact_id: str
    position: int
    event_type: str
    journey_state: str
    correction_state: str
    effective_time: Mapping[str, Any] = field(default_factory=dict, repr=False)
    evidence_ids: tuple[str, ...] = ()
    derivation_hash: str = ""
    assertion_status: str | None = None
    certainty: str | None = None
    experiencer: str | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "fact_id"):
            if _OPAQUE_ID_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be an opaque identifier")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("timeline node position must be non-negative")
        if _CONTROLLED_RE.fullmatch(self.event_type) is None:
            raise ValueError("timeline node event_type must be controlled")
        if self.journey_state not in TIMELINE_GRAPH_STATES:
            raise ValueError("unsupported timeline journey state")
        if self.correction_state not in TIMELINE_CORRECTION_STATES:
            raise ValueError("unsupported timeline correction state")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids or any(
            _OPAQUE_ID_RE.fullmatch(item) is None for item in evidence_ids
        ):
            raise ValueError("timeline nodes require opaque evidence identifiers")
        if _DIGEST_RE.fullmatch(self.derivation_hash) is None:
            raise ValueError("timeline node derivation_hash must be a digest")
        for name in ("assertion_status", "certainty", "experiencer"):
            value = getattr(self, name)
            if value is not None and _CONTROLLED_RE.fullmatch(value) is None:
                raise ValueError(f"timeline node {name} must be controlled")
        object.__setattr__(self, "effective_time", _freeze_mapping(self.effective_time))
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free graph node."""

        return {
            "assertion_status": self.assertion_status,
            "certainty": self.certainty,
            "correction_state": self.correction_state,
            "derivation_hash": self.derivation_hash,
            "effective_time": _plain(self.effective_time),
            "event_id": self.event_id,
            "event_type": self.event_type,
            "evidence_ids": list(self.evidence_ids),
            "experiencer": self.experiencer,
            "fact_id": self.fact_id,
            "journey_state": self.journey_state,
            "position": self.position,
        }


@dataclass(frozen=True, slots=True)
class TimelineGraphEdge:
    """One provenance, correction, relation, or temporal graph edge."""

    edge_id: str
    source_event_id: str
    target_event_id: str
    relation_type: str
    ordering_basis: str
    evidence_ids: tuple[str, ...]
    derivation_hash: str

    def __post_init__(self) -> None:
        for name in ("edge_id", "source_event_id", "target_event_id"):
            if _OPAQUE_ID_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be an opaque identifier")
        if self.source_event_id == self.target_event_id:
            raise ValueError("timeline edges cannot be self-referential")
        for name in ("relation_type", "ordering_basis"):
            if _CONTROLLED_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"timeline edge {name} must be controlled")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids or any(
            _OPAQUE_ID_RE.fullmatch(item) is None for item in evidence_ids
        ):
            raise ValueError("timeline edges require opaque evidence identifiers")
        if _DIGEST_RE.fullmatch(self.derivation_hash) is None:
            raise ValueError("timeline edge derivation_hash must be a digest")
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free graph edge."""

        return {
            "derivation_hash": self.derivation_hash,
            "edge_id": self.edge_id,
            "evidence_ids": list(self.evidence_ids),
            "ordering_basis": self.ordering_basis,
            "relation_type": self.relation_type,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLinkedTimelineGraph:
    """Deterministic graph projection over a page of journey events."""

    nodes: tuple[TimelineGraphNode, ...]
    edges: tuple[TimelineGraphEdge, ...]
    schema_version: str = TIMELINE_GRAPH_SCHEMA_VERSION
    compatibility_policy: str = TIMELINE_GRAPH_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        raw_nodes = tuple(self.nodes)
        raw_edges = tuple(self.edges)
        if any(not isinstance(item, TimelineGraphNode) for item in raw_nodes):
            raise TypeError("timeline graph nodes must be TimelineGraphNode values")
        if any(not isinstance(item, TimelineGraphEdge) for item in raw_edges):
            raise TypeError("timeline graph edges must be TimelineGraphEdge values")
        nodes = tuple(
            sorted(raw_nodes, key=lambda item: (item.position, item.event_id))
        )
        edges = tuple(
            sorted(
                raw_edges,
                key=lambda item: (
                    item.source_event_id,
                    item.target_event_id,
                    item.relation_type,
                    item.edge_id,
                ),
            )
        )
        node_ids = [item.event_id for item in nodes]
        edge_ids = [item.edge_id for item in edges]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("timeline graph node identifiers must be unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("timeline graph edge identifiers must be unique")
        positions = [item.position for item in nodes]
        if len(positions) != len(set(positions)):
            raise ValueError("timeline graph positions must be unique")
        known_nodes = set(node_ids)
        if any(
            item.source_event_id not in known_nodes
            or item.target_event_id not in known_nodes
            for item in edges
        ):
            raise ValueError("timeline graph edge references an unknown node")
        _reject_ordering_cycles(nodes, edges)
        if self.schema_version != TIMELINE_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported timeline graph schema version")
        if self.compatibility_policy != TIMELINE_GRAPH_COMPATIBILITY_POLICY:
            raise ValueError("unsupported timeline graph compatibility policy")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic value-free graph payload."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "edges": [item.to_dict() for item in self.edges],
            "nodes": [item.to_dict() for item in self.nodes],
            "schema_version": self.schema_version,
        }


def _reject_ordering_cycles(
    nodes: tuple[TimelineGraphNode, ...],
    edges: tuple[TimelineGraphEdge, ...],
) -> None:
    ordering_types = {"chronological_precedes", "corrects"}
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree = {item.event_id: 0 for item in nodes}
    for edge in edges:
        if edge.relation_type not in ordering_types:
            continue
        if edge.target_event_id not in adjacency[edge.source_event_id]:
            adjacency[edge.source_event_id].add(edge.target_event_id)
            indegree[edge.target_event_id] += 1
    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if visited != len(nodes):
        raise ValueError("timeline graph contains an ordering cycle")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("effective_time must be a mapping")
    try:
        payload = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError):
        raise ValueError("effective_time must contain JSON values") from None
    return _freeze(payload)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "TIMELINE_CORRECTION_STATES",
    "TIMELINE_GRAPH_COMPATIBILITY_POLICY",
    "TIMELINE_GRAPH_SCHEMA_VERSION",
    "TIMELINE_GRAPH_STATES",
    "EvidenceLinkedTimelineGraph",
    "TimelineGraphEdge",
    "TimelineGraphNode",
]
