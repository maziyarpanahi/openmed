"""Longitudinal document near-duplicate hash and copy-forward linker.

Implements shingle/MinHash near-duplicate detection and amendment-edge
discovery for a caller-provided set of clinical documents.  All processing
is fully offline; no network calls or external identity services are used.

Assistive disclaimer
--------------------
Document-linking outputs are assistive software outputs, not a medical
device, diagnosis, or substitute for qualified clinical judgment.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import combinations
from typing import Any

DOCUMENT_LINKING_ADVISORY = (
    "Document-linking outputs are assistive software outputs, not a medical "
    "device, diagnosis, or substitute for qualified clinical judgment."
)

_DEFAULT_SHINGLE_SIZE: int = 5
_DEFAULT_NUM_HASHES: int = 128
_DEFAULT_DUP_THRESHOLD: float = 0.92
_DEFAULT_AMEND_THRESHOLD: float = 0.60

_MERSENNE_PRIME: int = (1 << 61) - 1
_MAX_HASH: int = (1 << 32) - 1

import random as _random
_rng = _random.Random(42)
_HASH_PARAMS: list[tuple[int, int]] = [
    (_rng.randint(1, _MERSENNE_PRIME), _rng.randint(0, _MERSENNE_PRIME))
    for _ in range(_DEFAULT_NUM_HASHES)
]


class EdgeKind(str, Enum):
    """Type of relationship between two documents in a cluster."""
    NEAR_DUPLICATE = "near_duplicate"
    AMENDMENT      = "amendment"


@dataclass(frozen=True)
class DocumentEdge:
    """A directed edge from *source_id* to *target_id*."""
    source_id: str
    target_id: str
    kind: EdgeKind
    similarity: float
    superseded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind.value,
            "similarity": round(self.similarity, 4),
            "superseded": self.superseded,
        }


@dataclass
class DocumentCluster:
    """An ordered group of related clinical documents."""
    cluster_id: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    edges: list[DocumentEdge] = field(default_factory=list)

    @property
    def superseded_ids(self) -> frozenset[str]:
        return frozenset(
            e.target_id for e in self.edges
            if e.kind == EdgeKind.AMENDMENT and e.superseded
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "documents": [
                {
                    **doc,
                    "superseded": doc.get("doc_id", "") in self.superseded_ids,
                }
                for doc in self.documents
            ],
            "edges": [e.to_dict() for e in self.edges],
        }


def _normalise_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _shingles(text: str, k: int = _DEFAULT_SHINGLE_SIZE) -> frozenset[int]:
    norm = _normalise_text(text)
    if len(norm) < k:
        h = int(hashlib.md5(norm.encode()).hexdigest(), 16) & _MAX_HASH
        return frozenset({h})
    return frozenset(
        int(hashlib.md5(norm[i: i + k].encode()).hexdigest(), 16) & _MAX_HASH
        for i in range(len(norm) - k + 1)
    )


def _minhash_signature(shingle_set: frozenset[int],
                        params: list[tuple[int, int]] = _HASH_PARAMS,
                        prime: int = _MERSENNE_PRIME) -> list[int]:
    sig: list[int] = []
    for a, b in params:
        min_val = _MAX_HASH
        for s in shingle_set:
            h = ((a * s + b) % prime) & _MAX_HASH
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig


def _jaccard_from_signatures(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def _containment_ratio(text_a: str, text_b: str,
                        k: int = _DEFAULT_SHINGLE_SIZE) -> float:
    """Return |shingles(a) ∩ shingles(b)| / |shingles(b)|."""
    sh_a = _shingles(text_a, k)
    sh_b = _shingles(text_b, k)
    if not sh_b:
        return 0.0
    return len(sh_a & sh_b) / len(sh_b)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _sort_key(doc: dict[str, Any]) -> tuple[datetime, str]:
    dt = _parse_datetime(doc.get("note_datetime")) or datetime.min
    return (dt, doc.get("doc_id", ""))


def link_documents(
    docs: list[dict[str, Any]],
    *,
    dup_threshold: float = _DEFAULT_DUP_THRESHOLD,
    amend_threshold: float = _DEFAULT_AMEND_THRESHOLD,
    shingle_size: int = _DEFAULT_SHINGLE_SIZE,
    num_hashes: int = _DEFAULT_NUM_HASHES,
) -> list[DocumentCluster]:
    """Group, de-duplicate, and order a set of clinical documents.

    Parameters
    ----------
    docs:
        List of document dicts. Each must contain ``doc_id`` and ``text``,
        and should contain ``note_datetime``.
    dup_threshold:
        Minimum estimated Jaccard similarity for near-duplicate grouping.
    amend_threshold:
        Minimum containment ratio to detect an amendment edge.
    shingle_size:
        Character n-gram length for shingling.
    num_hashes:
        MinHash signature width.

    Returns
    -------
    list[DocumentCluster]
        One cluster per group, ordered by ``note_datetime``.
        Superseded documents are retained and flagged, never dropped.
        No network calls are made.
    """
    if not docs:
        return []

    if num_hashes != _DEFAULT_NUM_HASHES:
        rng = _random.Random(42)
        params: list[tuple[int, int]] = [
            (rng.randint(1, _MERSENNE_PRIME), rng.randint(0, _MERSENNE_PRIME))
            for _ in range(num_hashes)
        ]
    else:
        params = _HASH_PARAMS

    # ------------------------------------------------------------------
    # 1. Compute signatures and shingles
    # ------------------------------------------------------------------
    signatures: dict[str, list[int]] = {}
    doc_map: dict[str, dict[str, Any]] = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        doc_map[doc_id] = doc
        sh = _shingles(doc.get("text", ""), shingle_size)
        signatures[doc_id] = _minhash_signature(sh, params)

    doc_ids = [doc["doc_id"] for doc in docs]

    # ------------------------------------------------------------------
    # 2. Union-Find — merge near-duplicates AND amendments together
    # ------------------------------------------------------------------
    parent: dict[str, str] = {did: did for did in doc_ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        parent[_find(x)] = _find(y)

    # Collect all edges first, then union
    all_edges: list[DocumentEdge] = []

    for id_a, id_b in combinations(doc_ids, 2):
        text_a = doc_map[id_a].get("text", "")
        text_b = doc_map[id_b].get("text", "")
        len_a, len_b = len(text_a), len(text_b)
        max_len = max(len_a, len_b)
        len_ratio = min(len_a, len_b) / max_len if max_len > 0 else 1.0

        sim = _jaccard_from_signatures(signatures[id_a], signatures[id_b])

        # Determine which is longer (potential amendment)
        longer  = id_a if len_a >= len_b else id_b
        shorter = id_b if len_a >= len_b else id_a
        containment = _containment_ratio(
            doc_map[longer].get("text", ""),
            doc_map[shorter].get("text", ""),
            shingle_size,
        )
        
        if containment >= amend_threshold and len_ratio < 0.99:
            # Amendment relationship — sort by datetime to find direction
            dt_a = _parse_datetime(doc_map[id_a].get("note_datetime")) or datetime.min
            dt_b = _parse_datetime(doc_map[id_b].get("note_datetime")) or datetime.min
            later_id   = id_a if dt_a >= dt_b else id_b
            earlier_id = id_b if dt_a >= dt_b else id_a
            _union(later_id, earlier_id)
            all_edges.append(
                DocumentEdge(
                    source_id=later_id,
                    target_id=earlier_id,
                    kind=EdgeKind.AMENDMENT,
                    similarity=containment,
                    superseded=True,
                )
            )
        elif sim >= dup_threshold:
            # Near-duplicate relationship
            _union(id_a, id_b)
            all_edges.append(
                DocumentEdge(
                    source_id=id_a,
                    target_id=id_b,
                    kind=EdgeKind.NEAR_DUPLICATE,
                    similarity=sim,
                    superseded=True,
                )
            )

    # ------------------------------------------------------------------
    # 3. Build clusters
    # ------------------------------------------------------------------
    groups: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        root = _find(doc_id)
        groups.setdefault(root, []).append(doc_id)

    clusters: list[DocumentCluster] = []
    for root, members in groups.items():
        sorted_docs = sorted([doc_map[m] for m in members], key=_sort_key)
        cluster = DocumentCluster(
            cluster_id=sorted_docs[0]["doc_id"],
            documents=sorted_docs,
        )
        member_set = set(members)
        for e in all_edges:
            if e.source_id in member_set and e.target_id in member_set:
                cluster.edges.append(e)

        # Also detect amendments among docs within cluster
        for i in range(len(sorted_docs) - 1, -1, -1):
            later = sorted_docs[i]
            for j in range(i - 1, -1, -1):
                earlier = sorted_docs[j]
                # Skip if already have an amendment edge between these two
                existing = any(
                    e.source_id == later["doc_id"] and
                    e.target_id == earlier["doc_id"] and
                    e.kind == EdgeKind.AMENDMENT
                    for e in cluster.edges
                )
                if existing:
                    continue
                ratio = _containment_ratio(
                    later.get("text", ""),
                    earlier.get("text", ""),
                    shingle_size,
                )
                if ratio >= amend_threshold:
                    cluster.edges.append(
                        DocumentEdge(
                            source_id=later["doc_id"],
                            target_id=earlier["doc_id"],
                            kind=EdgeKind.AMENDMENT,
                            similarity=ratio,
                            superseded=True,
                        )
                    )

        clusters.append(cluster)

    clusters.sort(key=lambda c: _sort_key(c.documents[0]))
    return clusters


__all__ = [
    "DOCUMENT_LINKING_ADVISORY",
    "DocumentCluster",
    "DocumentEdge",
    "EdgeKind",
    "link_documents",
]