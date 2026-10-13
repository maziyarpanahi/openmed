"""Digest-addressed federated checkpoint ancestry with no client-level metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from types import MappingProxyType

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PARENTS = {
    "base_model": frozenset(),
    "round_manifest": frozenset({"base_model", "released_adapter"}),
    "aggregate": frozenset({"round_manifest"}),
    "released_adapter": frozenset({"aggregate"}),
}
MAX_NODES = 10000
MAX_PARENTS = 32


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("checkpoint lineage requires SHA-256 references")
    return value


def _bounded(values: Iterable, maximum: int) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError("checkpoint lineage requires a bounded collection")
    try:
        result = tuple(islice(values, maximum + 1))
    except Exception:
        raise ValueError("invalid checkpoint lineage collection") from None
    if len(result) > maximum:
        raise ValueError("checkpoint lineage collection limit exceeded")
    return result


@dataclass(frozen=True, slots=True)
class CheckpointLineageNode:
    """One declared artifact digest and its typed direct ancestry.

    Kind is base_model, round_manifest, aggregate or released_adapter. Digests
    describe artifacts declared by the caller; this node does not verify files,
    signatures, training provenance or whether a checkpoint was actually released.
    """

    digest: str
    kind: str
    parent_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.digest)
        if not isinstance(self.kind, str) or self.kind not in _PARENTS:
            raise ValueError("invalid checkpoint lineage artifact kind")
        parents = tuple(
            _digest(value) for value in _bounded(self.parent_digests, MAX_PARENTS)
        )
        if len(set(parents)) != len(parents):
            raise ValueError("duplicate checkpoint lineage parent")
        if self.digest in parents:
            raise ValueError("self-referential checkpoint lineage")
        if (self.kind == "base_model") != (not parents):
            raise ValueError("checkpoint lineage kind has invalid parent count")
        object.__setattr__(self, "parent_digests", tuple(sorted(parents)))

    def to_dict(self) -> dict[str, object]:
        """Export only artifact digest, kind and direct parent digests."""
        return {
            "digest": self.digest,
            "kind": self.kind,
            "parent_digests": list(self.parent_digests),
        }


class CheckpointLineageGraph:
    """An immutable bounded DAG enforcing base-to-round-to-aggregate ancestry."""

    def __init__(self, nodes: Iterable[CheckpointLineageNode]) -> None:
        nodes = _bounded(nodes, MAX_NODES)
        if any(not isinstance(node, CheckpointLineageNode) for node in nodes):
            raise ValueError("invalid checkpoint lineage node")
        by_digest = {node.digest: node for node in nodes}
        if len(by_digest) != len(nodes):
            raise ValueError("duplicate checkpoint lineage artifact")
        children = {digest: [] for digest in by_digest}
        indegrees = {}
        for node in nodes:
            indegrees[node.digest] = len(node.parent_digests)
            for parent in node.parent_digests:
                if parent not in by_digest:
                    raise ValueError("unresolved checkpoint lineage parent")
                if by_digest[parent].kind not in _PARENTS[node.kind]:
                    raise ValueError("invalid checkpoint lineage stage transition")
                children[parent].append(node.digest)
        pending = [digest for digest, count in indegrees.items() if count == 0]
        visited = 0
        while pending:
            digest = pending.pop()
            visited += 1
            for child in children[digest]:
                indegrees[child] -= 1
                if indegrees[child] == 0:
                    pending.append(child)
        if visited != len(nodes):
            raise ValueError("cyclic checkpoint lineage")
        self._nodes = MappingProxyType(by_digest)

    def ancestry(self, checkpoint_digest: str) -> tuple[CheckpointLineageNode, ...]:
        """Return the selected checkpoint and every transitive ancestor, sorted.

        Args:
            checkpoint_digest: Declared base-model or released-adapter SHA-256.

        Returns:
            Stable immutable ancestry including shared parents only once. Unrelated
            graph branches are excluded. No file or network lookup is performed.
        """
        _digest(checkpoint_digest)
        if checkpoint_digest not in self._nodes:
            raise ValueError("checkpoint lineage target is missing")
        if self._nodes[checkpoint_digest].kind not in {
            "base_model",
            "released_adapter",
        }:
            raise ValueError("checkpoint lineage target is not a checkpoint")
        reached = set()
        pending = [checkpoint_digest]
        while pending:
            digest = pending.pop()
            if digest in reached:
                continue
            reached.add(digest)
            pending.extend(self._nodes[digest].parent_digests)
        return tuple(self._nodes[digest] for digest in sorted(reached))

    def to_manifest(self, checkpoint_digest: str) -> dict[str, object]:
        """Export content-free ancestry without site identities or local metrics."""
        return {
            "schema_version": 1,
            "checkpoint_digest": checkpoint_digest,
            "nodes": [node.to_dict() for node in self.ancestry(checkpoint_digest)],
        }

    def to_json(self, checkpoint_digest: str) -> str:
        """Serialize deterministic ancestry JSON with no caller-supplied labels."""
        return (
            json.dumps(
                self.to_manifest(checkpoint_digest),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
