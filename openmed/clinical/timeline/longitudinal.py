"""Privacy-conscious timeline view over linked longitudinal documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from openmed.clinical.document_linking import DocumentCluster, DocumentEdge


@dataclass(frozen=True)
class LinkedTimelineDocument:
    """One retained document positioned on a longitudinal timeline."""

    doc_id: str
    note_datetime: str | None
    superseded: bool
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready timeline entry without source note text."""

        return {
            "doc_id": self.doc_id,
            "note_datetime": self.note_datetime,
            "superseded": self.superseded,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class LinkedDocumentTimeline:
    """Ordered documents and their near-duplicate/amendment relationships."""

    cluster_id: str
    patient_id: str | None
    documents: tuple[LinkedTimelineDocument, ...]
    relationships: tuple[DocumentEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable longitudinal timeline."""

        return {
            "cluster_id": self.cluster_id,
            "patient_id": self.patient_id,
            "documents": [document.to_dict() for document in self.documents],
            "relationships": [edge.to_dict() for edge in self.relationships],
        }


def build_linked_document_timeline(
    cluster: DocumentCluster,
) -> LinkedDocumentTimeline:
    """Adapt a document cluster for timeline consumers without copying text.

    Args:
        cluster: A cluster returned by
            :func:`openmed.clinical.document_linking.link_documents`.

    Returns:
        Chronologically ordered document references plus their directed
        near-duplicate and amendment relationships.

    Raises:
        TypeError: If ``cluster`` is not a :class:`DocumentCluster`.
    """

    if not isinstance(cluster, DocumentCluster):
        raise TypeError("cluster must be a DocumentCluster")

    superseded_ids = cluster.superseded_ids
    documents = tuple(
        LinkedTimelineDocument(
            doc_id=provenance.doc_id,
            note_datetime=provenance.note_datetime,
            superseded=provenance.doc_id in superseded_ids,
            provenance=provenance.metadata,
        )
        for provenance in cluster.document_provenance
    )
    return LinkedDocumentTimeline(
        cluster_id=cluster.cluster_id,
        patient_id=cluster.patient_id,
        documents=documents,
        relationships=tuple(cluster.edges),
    )


__all__ = [
    "LinkedDocumentTimeline",
    "LinkedTimelineDocument",
    "build_linked_document_timeline",
]
