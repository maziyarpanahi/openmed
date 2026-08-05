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
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from typing import Any, Mapping, Sequence

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

_rng = random.Random(42)
_HASH_PARAMS: tuple[tuple[int, int], ...] = tuple(
    (_rng.randint(1, _MERSENNE_PRIME), _rng.randint(0, _MERSENNE_PRIME))
    for _ in range(_DEFAULT_NUM_HASHES)
)


class EdgeKind(str, Enum):
    """Type of relationship between two documents in a cluster."""

    NEAR_DUPLICATE = "near_duplicate"
    AMENDMENT = "amendment"


@dataclass(frozen=True)
class DocumentProvenance:
    """Non-text provenance for one document represented by an edge.

    Attributes:
        doc_id: Caller-provided document identifier.
        note_datetime: Normalized ISO 8601 note timestamp, when present.
        metadata: Caller-provided provenance metadata. Document text is never
            copied into this mapping by the linker.
    """

    doc_id: str
    note_datetime: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance dictionary."""

        return {
            "doc_id": self.doc_id,
            "note_datetime": self.note_datetime,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DocumentEdge:
    """A directed, provenance-carrying relationship between two documents."""

    source_id: str
    target_id: str
    kind: EdgeKind
    similarity: float
    superseded: bool
    source_provenance: DocumentProvenance
    target_provenance: DocumentProvenance

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable edge dictionary."""

        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind.value,
            "similarity": round(self.similarity, 4),
            "superseded": self.superseded,
            "source_provenance": self.source_provenance.to_dict(),
            "target_provenance": self.target_provenance.to_dict(),
        }


@dataclass
class DocumentCluster:
    """An ordered group of related clinical documents."""

    cluster_id: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    edges: list[DocumentEdge] = field(default_factory=list)

    @property
    def superseded_ids(self) -> frozenset[str]:
        """Return document identifiers superseded by any retained edge."""

        return frozenset(edge.target_id for edge in self.edges if edge.superseded)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable cluster dictionary."""

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
    if not norm:
        return frozenset()
    if len(norm) < k:
        return frozenset({_stable_shingle_hash(norm)})
    return frozenset(
        _stable_shingle_hash(norm[i : i + k]) for i in range(len(norm) - k + 1)
    )


def _stable_shingle_hash(value: str) -> int:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


def _minhash_signature(
    shingle_set: frozenset[int],
    params: Sequence[tuple[int, int]] = _HASH_PARAMS,
    prime: int = _MERSENNE_PRIME,
) -> list[int]:
    if not shingle_set:
        return []
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


def _containment_ratio(
    text_a: str,
    text_b: str,
    k: int = _DEFAULT_SHINGLE_SIZE,
) -> float:
    """Return |shingles(a) ∩ shingles(b)| / |shingles(b)|."""
    sh_a = _shingles(text_a, k)
    sh_b = _shingles(text_b, k)
    if not sh_b:
        return 0.0
    return len(sh_a & sh_b) / len(sh_b)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _normalise_datetime(value: Any) -> str | None:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed is not None else None


def _document_provenance(doc: Mapping[str, Any]) -> DocumentProvenance:
    metadata = doc.get("provenance", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("document provenance must be a mapping")
    return DocumentProvenance(
        doc_id=str(doc["doc_id"]),
        note_datetime=_normalise_datetime(doc.get("note_datetime")),
        metadata=dict(metadata),
    )


def _validate_options(
    *,
    dup_threshold: float,
    amend_threshold: float,
    shingle_size: int,
    num_hashes: int,
) -> None:
    for name, value in (
        ("dup_threshold", dup_threshold),
        ("amend_threshold", amend_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{name} must be a finite number between 0.0 and 1.0")
    for name, value in (("shingle_size", shingle_size), ("num_hashes", num_hashes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _validate_documents(
    docs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_doc in enumerate(docs):
        if not isinstance(raw_doc, Mapping):
            raise TypeError(f"document at index {index} must be a mapping")
        doc = dict(raw_doc)
        doc_id = doc.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"document at index {index} must have a non-empty doc_id")
        if doc_id in seen_ids:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        seen_ids.add(doc_id)
        text = doc.get("text")
        if not isinstance(text, str):
            raise TypeError(f"document {doc_id!r} text must be a string")
        note_datetime = doc.get("note_datetime")
        if note_datetime is not None and _parse_datetime(note_datetime) is None:
            raise ValueError(
                f"document {doc_id!r} note_datetime must be an ISO 8601 timestamp"
            )
        _document_provenance(doc)
        validated.append(doc)
    return validated


def _chronological_pair(
    id_a: str,
    id_b: str,
    doc_map: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    if _sort_key(doc_map[id_a]) >= _sort_key(doc_map[id_b]):
        return id_a, id_b
    return id_b, id_a


def _sort_key(doc: Mapping[str, Any]) -> tuple[datetime, str]:
    dt = _parse_datetime(doc.get("note_datetime")) or datetime.min
    return (dt, str(doc.get("doc_id", "")))


def link_documents(
    docs: Sequence[Mapping[str, Any]],
    *,
    dup_threshold: float = _DEFAULT_DUP_THRESHOLD,
    amend_threshold: float = _DEFAULT_AMEND_THRESHOLD,
    shingle_size: int = _DEFAULT_SHINGLE_SIZE,
    num_hashes: int = _DEFAULT_NUM_HASHES,
) -> list[DocumentCluster]:
    """Group, de-duplicate, and order a set of clinical documents.

    Args:
        docs: Document mappings with unique ``doc_id`` and string ``text``
            fields. ``note_datetime`` may be a datetime or ISO 8601 string.
            Optional ``provenance`` mappings are copied onto every related
            edge without copying document text.
        dup_threshold: Minimum estimated Jaccard similarity for grouping
            near-duplicates.
        amend_threshold: Minimum directional containment for a later,
            longer document to supersede an earlier document.
        shingle_size: Character n-gram length for shingling.
        num_hashes: MinHash signature width.

    Returns:
        Clusters ordered by their first document timestamp. Documents inside
        each cluster are ordered by timestamp, and superseded documents remain
        present with a flag in :meth:`DocumentCluster.to_dict`.

    Raises:
        TypeError: If a document, text, or provenance value has the wrong type.
        ValueError: If options, identifiers, or timestamps are invalid.
    """
    _validate_options(
        dup_threshold=dup_threshold,
        amend_threshold=amend_threshold,
        shingle_size=shingle_size,
        num_hashes=num_hashes,
    )
    if not docs:
        return []
    validated_docs = _validate_documents(docs)

    if num_hashes != _DEFAULT_NUM_HASHES:
        rng = random.Random(42)
        params: Sequence[tuple[int, int]] = tuple(
            (rng.randint(1, _MERSENNE_PRIME), rng.randint(0, _MERSENNE_PRIME))
            for _ in range(num_hashes)
        )
    else:
        params = _HASH_PARAMS

    signatures: dict[str, list[int]] = {}
    doc_map: dict[str, dict[str, Any]] = {}
    provenance: dict[str, DocumentProvenance] = {}
    for doc in validated_docs:
        doc_id = doc["doc_id"]
        doc_map[doc_id] = doc
        sh = _shingles(doc["text"], shingle_size)
        signatures[doc_id] = _minhash_signature(sh, params)
        provenance[doc_id] = _document_provenance(doc)

    doc_ids = [doc["doc_id"] for doc in validated_docs]

    parent: dict[str, str] = {did: did for did in doc_ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        parent[_find(x)] = _find(y)

    all_edges: list[DocumentEdge] = []

    for id_a, id_b in combinations(doc_ids, 2):
        sim = _jaccard_from_signatures(signatures[id_a], signatures[id_b])
        later_id, earlier_id = _chronological_pair(id_a, id_b, doc_map)
        later_text = doc_map[later_id]["text"]
        earlier_text = doc_map[earlier_id]["text"]
        containment = _containment_ratio(
            later_text,
            earlier_text,
            shingle_size,
        )
        later_datetime = _parse_datetime(doc_map[later_id].get("note_datetime"))
        earlier_datetime = _parse_datetime(doc_map[earlier_id].get("note_datetime"))
        is_later = (
            later_datetime is not None
            and earlier_datetime is not None
            and later_datetime > earlier_datetime
        )
        is_amendment = (
            is_later
            and len(_normalise_text(later_text)) > len(_normalise_text(earlier_text))
            and containment >= amend_threshold
        )

        if is_amendment:
            _union(later_id, earlier_id)
            all_edges.append(
                DocumentEdge(
                    source_id=later_id,
                    target_id=earlier_id,
                    kind=EdgeKind.AMENDMENT,
                    similarity=containment,
                    superseded=True,
                    source_provenance=provenance[later_id],
                    target_provenance=provenance[earlier_id],
                )
            )
        elif sim >= dup_threshold:
            _union(later_id, earlier_id)
            all_edges.append(
                DocumentEdge(
                    source_id=later_id,
                    target_id=earlier_id,
                    kind=EdgeKind.NEAR_DUPLICATE,
                    similarity=sim,
                    superseded=True,
                    source_provenance=provenance[later_id],
                    target_provenance=provenance[earlier_id],
                )
            )

    groups: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        root = _find(doc_id)
        groups.setdefault(root, []).append(doc_id)

    clusters: list[DocumentCluster] = []
    for members in groups.values():
        sorted_docs = sorted([doc_map[m] for m in members], key=_sort_key)
        cluster = DocumentCluster(
            cluster_id=sorted_docs[0]["doc_id"],
            documents=sorted_docs,
        )
        member_set = set(members)
        for e in all_edges:
            if e.source_id in member_set and e.target_id in member_set:
                cluster.edges.append(e)

        clusters.append(cluster)

    clusters.sort(key=lambda c: _sort_key(c.documents[0]))
    return clusters


__all__ = [
    "DOCUMENT_LINKING_ADVISORY",
    "DocumentCluster",
    "DocumentEdge",
    "DocumentProvenance",
    "EdgeKind",
    "link_documents",
]
