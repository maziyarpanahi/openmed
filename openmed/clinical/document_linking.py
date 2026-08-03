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

_ENTITY_CONTAINER_NAMES = ("entities", "clinical_entities", "spans")
_CATEGORY_FIELD_NAMES = (
    "category",
    "entity_category",
    "clinical_category",
    "semantic_type",
    "kind",
    "type",
    "entity_type",
    "entity_group",
    "label",
    "canonical_label",
)
_CODING_CONTAINER_NAMES = (
    "coding",
    "codings",
    "codes",
    "codeable_concept",
    "codeableConcept",
    "concept",
)
_SURFACE_FIELD_NAMES = (
    "canonical_text",
    "normalized_text",
    "text",
    "surface",
    "word",
    "name",
    "value",
)
_CONTEXT_FIELD_NAMES = (
    "experiencer",
    "negation",
    "temporality",
    "certainty",
    "status",
    "clinical_status",
)
_CATEGORY_ALIASES = {
    "condition": "problems",
    "diagnosis": "problems",
    "disease": "problems",
    "disorder": "problems",
    "finding": "problems",
    "problem": "problems",
    "problems": "problems",
    "symptom": "problems",
    "drug": "medications",
    "medication": "medications",
    "medications": "medications",
    "lab": "labs",
    "lab_result": "labs",
    "laboratory": "labs",
    "procedure": "procedures",
    "surgery": "procedures",
}

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
class EntityOccurrence:
    """Privacy-conscious provenance for one source entity occurrence.

    The occurrence retains document identity, source offsets, and a stable hash,
    but never copies the entity surface into provenance.
    """

    doc_id: str
    entity_index: int
    note_datetime: str | None
    start: int | None
    end: int | None
    surface_hash: str
    document_provenance: DocumentProvenance
    source_entity_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable occurrence provenance dictionary."""

        return {
            "doc_id": self.doc_id,
            "entity_index": self.entity_index,
            "note_datetime": self.note_datetime,
            "start": self.start,
            "end": self.end,
            "surface_hash": self.surface_hash,
            "source_entity_hash": self.source_entity_hash,
            "document_provenance": self.document_provenance.to_dict(),
        }


@dataclass(frozen=True)
class DeduplicatedEntity:
    """One conservative cross-document entity with complete provenance."""

    entity_id: str
    category: str
    codings: tuple[Mapping[str, str], ...]
    provenance: tuple[EntityOccurrence, ...]
    identity_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "codings",
            tuple(dict(coding) for coding in self.codings),
        )

    @property
    def system(self) -> str | None:
        """Return the first coding system for summary-card compatibility."""

        if not self.codings:
            return None
        return self.codings[0].get("system") or None

    @property
    def code(self) -> str | None:
        """Return the first code for summary-card compatibility."""

        if not self.codings:
            return None
        return self.codings[0].get("code") or None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready entity without copying source surface text."""

        return {
            "entity_id": self.entity_id,
            "category": self.category,
            "codings": [dict(coding) for coding in self.codings],
            "identity_hash": self.identity_hash,
            "occurrence_count": len(self.provenance),
            "provenance": [occurrence.to_dict() for occurrence in self.provenance],
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
    patient_id: str | None = None
    entities: list[DeduplicatedEntity] = field(default_factory=list)

    @property
    def superseded_ids(self) -> frozenset[str]:
        """Return document identifiers superseded by any retained edge."""

        return frozenset(edge.target_id for edge in self.edges if edge.superseded)

    @property
    def document_provenance(self) -> tuple[DocumentProvenance, ...]:
        """Return provenance for every retained document in timeline order."""

        return tuple(_document_provenance(document) for document in self.documents)

    @property
    def deduplicated_entities(self) -> tuple[DeduplicatedEntity, ...]:
        """Return the cluster's summary-ready unique clinical entities."""

        return tuple(self.entities)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable cluster dictionary."""

        return {
            "cluster_id": self.cluster_id,
            "patient_id": self.patient_id,
            "documents": [
                {
                    **doc,
                    "superseded": doc.get("doc_id", "") in self.superseded_ids,
                }
                for doc in self.documents
            ],
            "document_provenance": [
                item.to_dict() for item in self.document_provenance
            ],
            "edges": [e.to_dict() for e in self.edges],
            "entities": [entity.to_dict() for entity in self.entities],
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
        patient_id = doc.get("patient_id")
        if "patient_id" in doc and (
            not isinstance(patient_id, str) or not patient_id.strip()
        ):
            raise ValueError(
                f"document {doc_id!r} patient_id must be a non-empty string"
            )
        if isinstance(patient_id, str):
            doc["patient_id"] = patient_id.strip()
        validated.append(doc)
    patient_id_presence = ["patient_id" in doc for doc in validated]
    if any(patient_id_presence) and not all(patient_id_presence):
        raise ValueError("patient_id must be provided for every document or none")
    return validated


def _field_value(source: object, field_name: str) -> object | None:
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _entity_items(document: Mapping[str, Any]) -> tuple[object, ...]:
    entity_source: object | None = None
    for field_name in _ENTITY_CONTAINER_NAMES:
        if field_name in document and document[field_name] is not None:
            entity_source = document[field_name]
            break
    if entity_source is None:
        return ()
    if isinstance(entity_source, Mapping):
        return (entity_source,)
    if isinstance(entity_source, Sequence) and not isinstance(
        entity_source, (str, bytes)
    ):
        return tuple(entity_source)
    raise TypeError(
        f"document {document['doc_id']!r} entities must be a mapping or sequence"
    )


def _normalise_entity_value(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _entity_category(entity: object) -> str:
    for field_name in _CATEGORY_FIELD_NAMES:
        value = _field_value(entity, field_name)
        normalised = _normalise_entity_value(value)
        if normalised:
            return _CATEGORY_ALIASES.get(normalised, normalised)
    return "other"


def _iter_code_pairs(value: object) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if value is None or isinstance(value, (str, bytes)):
        return pairs
    if isinstance(value, Mapping):
        code = value.get("code")
        if code is not None and not isinstance(code, bool) and str(code).strip():
            system = value.get("system")
            pairs.append(
                (
                    str(system).strip().casefold() if system is not None else "",
                    str(code).strip(),
                )
            )
        for field_name in _CODING_CONTAINER_NAMES:
            if field_name in value:
                pairs.extend(_iter_code_pairs(value[field_name]))
        return pairs
    if isinstance(value, Sequence):
        for item in value:
            if isinstance(item, str) and item.strip():
                pairs.append(("", item.strip()))
            else:
                pairs.extend(_iter_code_pairs(item))
        return pairs

    code = getattr(value, "code", None)
    if code is not None and not isinstance(code, bool) and str(code).strip():
        system = getattr(value, "system", None)
        pairs.append(
            (
                str(system).strip().casefold() if system is not None else "",
                str(code).strip(),
            )
        )
    for field_name in _CODING_CONTAINER_NAMES:
        nested = getattr(value, field_name, None)
        if nested is not None:
            pairs.extend(_iter_code_pairs(nested))
    return pairs


def _entity_codings(entity: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(set(_iter_code_pairs(entity))))


def _entity_surface(entity: object) -> str:
    for field_name in _SURFACE_FIELD_NAMES:
        value = _field_value(entity, field_name)
        if isinstance(value, str) and value.strip():
            return _normalise_text(value)
    return ""


def _entity_context(entity: object) -> tuple[tuple[str, str], ...]:
    context: list[tuple[str, str]] = []
    for field_name in _CONTEXT_FIELD_NAMES:
        value = _normalise_entity_value(_field_value(entity, field_name))
        if value:
            context.append((field_name, value))
    return tuple(context)


def _entity_offsets(
    entity: object,
    *,
    doc_id: str,
    text_length: int,
) -> tuple[int | None, int | None]:
    start = _field_value(entity, "start")
    end = _field_value(entity, "end")
    offsets = _field_value(entity, "offsets")
    if start is None and end is None and isinstance(offsets, Sequence):
        if len(offsets) != 2:
            raise ValueError(
                f"entity offsets in document {doc_id!r} require two values"
            )
        start, end = offsets
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise ValueError(
            f"entity offsets in document {doc_id!r} require both start and end"
        )
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"entity offsets in document {doc_id!r} must be integers")
    if start < 0 or end <= start or end > text_length:
        raise ValueError(
            f"entity offsets in document {doc_id!r} must satisfy "
            "0 <= start < end <= len(text)"
        )
    return start, end


def _entity_identity(
    entity: object,
    *,
    doc_id: str,
    entity_index: int,
    lineage_id: str,
) -> tuple[str, str, tuple[tuple[str, str], ...], str]:
    category = _entity_category(entity)
    codings = _entity_codings(entity)
    surface = _entity_surface(entity)
    context = _entity_context(entity)
    if codings:
        core = ("codes", category, repr(codings))
    elif surface:
        core = ("surface", category, surface)
    else:
        core = ("occurrence", category, doc_id, str(entity_index))
    if category not in {"problems", "medications"}:
        core = (*core, "document_lineage", lineage_id)
    identity = repr((*core, context))
    surface_hash = hashlib.blake2s(
        (surface or identity).encode("utf-8"), digest_size=16
    ).hexdigest()
    return category, identity, codings, surface_hash


def _deduplicate_entities(
    documents: Sequence[Mapping[str, Any]],
    *,
    cluster_id: str,
    edges: Sequence[DocumentEdge],
) -> list[DeduplicatedEntity]:
    grouped: dict[
        str,
        tuple[str, tuple[tuple[str, str], ...], list[EntityOccurrence]],
    ] = {}
    lineage_ids = _document_lineage_ids(documents, edges)
    for document in documents:
        doc_id = str(document["doc_id"])
        doc_provenance = _document_provenance(document)
        for entity_index, entity in enumerate(_entity_items(document)):
            category, identity, codings, surface_hash = _entity_identity(
                entity,
                doc_id=doc_id,
                entity_index=entity_index,
                lineage_id=lineage_ids[doc_id],
            )
            start, end = _entity_offsets(
                entity,
                doc_id=doc_id,
                text_length=len(str(document["text"])),
            )
            raw_source_id = _field_value(entity, "entity_id")
            source_entity_hash = None
            if raw_source_id is not None and str(raw_source_id).strip():
                source_entity_hash = hashlib.blake2s(
                    str(raw_source_id).strip().encode("utf-8"),
                    digest_size=16,
                ).hexdigest()
            occurrence = EntityOccurrence(
                doc_id=doc_id,
                entity_index=entity_index,
                note_datetime=doc_provenance.note_datetime,
                start=start,
                end=end,
                surface_hash=surface_hash,
                source_entity_hash=source_entity_hash,
                document_provenance=doc_provenance,
            )
            if identity not in grouped:
                grouped[identity] = (category, codings, [])
            grouped[identity][2].append(occurrence)

    deduplicated: list[DeduplicatedEntity] = []
    for identity, (category, codings, occurrences) in grouped.items():
        identity_hash = hashlib.blake2s(
            identity.encode("utf-8"), digest_size=16
        ).hexdigest()
        entity_id = hashlib.blake2s(
            f"{cluster_id}\0{identity}".encode("utf-8"), digest_size=12
        ).hexdigest()
        deduplicated.append(
            DeduplicatedEntity(
                entity_id=f"entity-{entity_id}",
                category=category,
                codings=tuple(
                    {"system": system, "code": code} for system, code in codings
                ),
                provenance=tuple(occurrences),
                identity_hash=identity_hash,
            )
        )
    return deduplicated


def _document_lineage_ids(
    documents: Sequence[Mapping[str, Any]],
    edges: Sequence[DocumentEdge],
) -> dict[str, str]:
    doc_ids = [str(document["doc_id"]) for document in documents]
    parent = {doc_id: doc_id for doc_id in doc_ids}

    def _find(doc_id: str) -> str:
        while parent[doc_id] != doc_id:
            parent[doc_id] = parent[parent[doc_id]]
            doc_id = parent[doc_id]
        return doc_id

    def _union(left: str, right: str) -> None:
        left_root = _find(left)
        right_root = _find(right)
        canonical_root = min(left_root, right_root)
        parent[left_root] = canonical_root
        parent[right_root] = canonical_root

    for edge in edges:
        _union(edge.source_id, edge.target_id)
    return {doc_id: _find(doc_id) for doc_id in doc_ids}


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
            edge without copying document text. When every document provides
            ``patient_id``, exact caller-supplied identifiers define patient
            clusters and comparisons never cross those boundaries. When none
            provide it, the legacy similarity-connected grouping is retained.
            Optional ``entities``, ``clinical_entities``, or ``spans`` are
            conservatively de-duplicated inside each resulting cluster.
            Problems and medications may link across a patient cluster;
            event-like entities link only inside a detected duplicate or
            amendment lineage so repeated labs or procedures remain distinct.
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
    has_patient_ids = all("patient_id" in doc for doc in validated_docs)

    parent: dict[str, str] = {did: did for did in doc_ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        parent[_find(x)] = _find(y)

    if has_patient_ids:
        patient_representatives: dict[str, str] = {}
        for doc_id in doc_ids:
            patient_id = str(doc_map[doc_id]["patient_id"])
            representative = patient_representatives.setdefault(patient_id, doc_id)
            _union(doc_id, representative)

    all_edges: list[DocumentEdge] = []

    for id_a, id_b in combinations(doc_ids, 2):
        if has_patient_ids and (
            doc_map[id_a]["patient_id"] != doc_map[id_b]["patient_id"]
        ):
            continue
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
        cluster_id = sorted_docs[0]["doc_id"]
        member_set = set(members)
        member_edges = [
            edge
            for edge in all_edges
            if edge.source_id in member_set and edge.target_id in member_set
        ]
        cluster = DocumentCluster(
            cluster_id=cluster_id,
            documents=sorted_docs,
            patient_id=(str(sorted_docs[0]["patient_id"]) if has_patient_ids else None),
            edges=member_edges,
            entities=_deduplicate_entities(
                sorted_docs,
                cluster_id=cluster_id,
                edges=member_edges,
            ),
        )
        clusters.append(cluster)

    clusters.sort(key=lambda c: _sort_key(c.documents[0]))
    return clusters


__all__ = [
    "DOCUMENT_LINKING_ADVISORY",
    "DeduplicatedEntity",
    "DocumentCluster",
    "DocumentEdge",
    "DocumentProvenance",
    "EdgeKind",
    "EntityOccurrence",
    "link_documents",
]
