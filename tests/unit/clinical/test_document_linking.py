"""Unit tests for openmed.clinical.document_linking."""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from openmed.clinical import (
    DOCUMENT_LINKING_ADVISORY,
    DeduplicatedEntity,
    DocumentCluster,
    DocumentEdge,
    DocumentProvenance,
    EdgeKind,
    EntityOccurrence,
    LinkedDocumentTimeline,
    LinkedTimelineDocument,
    build_linked_document_timeline,
    build_summary_card,
    link_documents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str,
    text: str,
    note_datetime: str | None = None,
    *,
    provenance: dict | None = None,
    patient_id: str | None = None,
    entities: list[dict] | None = None,
) -> dict:
    d: dict = {"doc_id": doc_id, "text": text}
    if note_datetime is not None:
        d["note_datetime"] = note_datetime
    if provenance is not None:
        d["provenance"] = provenance
    if patient_id is not None:
        d["patient_id"] = patient_id
    if entities is not None:
        d["entities"] = entities
    return d


LONG_TEXT = (
    "Patient presents with chest pain radiating to the left arm. "
    "History of hypertension and type 2 diabetes mellitus. "
    "Current medications include metformin 500 mg twice daily and lisinopril 10 mg daily. "
    "Vital signs stable. EKG shows normal sinus rhythm. "
    "Plan: continue current medications, follow up in two weeks."
) * 4  # repeat to give enough shingles


AMENDMENT_TEXT = LONG_TEXT + (
    " Addendum: troponin level returned within normal limits. "
    "No acute coronary syndrome. Discharge planned for tomorrow."
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


class TestLinkDocumentsEmpty:
    def test_empty_list_returns_empty(self):
        assert link_documents([]) == []


class TestLinkDocumentsSingleDoc:
    def test_single_doc_forms_one_cluster(self):
        docs = [_doc("d1", LONG_TEXT, "2026-01-01")]
        clusters = link_documents(docs)
        assert len(clusters) == 1
        assert clusters[0].documents[0]["doc_id"] == "d1"
        assert clusters[0].edges == []

    def test_single_doc_not_superseded(self):
        docs = [_doc("d1", LONG_TEXT, "2026-01-01")]
        clusters = link_documents(docs)
        assert "d1" not in clusters[0].superseded_ids


class TestInputValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"dup_threshold": -0.1}, "dup_threshold"),
            ({"amend_threshold": 1.1}, "amend_threshold"),
            ({"shingle_size": 0}, "shingle_size"),
            ({"num_hashes": 0}, "num_hashes"),
        ],
    )
    def test_invalid_options_fail_closed(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            link_documents([_doc("d1", LONG_TEXT)], **kwargs)

    def test_duplicate_document_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate doc_id"):
            link_documents([_doc("d1", "alpha"), _doc("d1", "beta")])

    def test_invalid_timestamp_is_rejected(self):
        with pytest.raises(ValueError, match="ISO 8601"):
            link_documents([_doc("d1", LONG_TEXT, "not-a-date")])

    def test_empty_texts_do_not_form_false_duplicate_cluster(self):
        clusters = link_documents([_doc("d1", ""), _doc("d2", "!!!")])
        assert len(clusters) == 2

    def test_patient_id_is_all_or_none(self):
        with pytest.raises(ValueError, match="every document or none"):
            link_documents(
                [
                    _doc("d1", LONG_TEXT, patient_id="patient-a"),
                    _doc("d2", LONG_TEXT),
                ]
            )

    def test_entity_offsets_must_reference_source_document(self):
        with pytest.raises(ValueError, match=r"len\(text\)"):
            link_documents(
                [
                    _doc(
                        "d1",
                        "Synthetic note",
                        entities=[
                            {
                                "category": "problem",
                                "text": "outside",
                                "start": 20,
                                "end": 27,
                            }
                        ],
                    )
                ]
            )


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


class TestNearDuplicates:
    def test_identical_docs_form_one_cluster(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", LONG_TEXT, "2026-01-03"),  # re-fax of d1
        ]
        clusters = link_documents(docs)
        assert len(clusters) == 1, "Identical docs must cluster together"

    def test_near_duplicate_edge_present(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", LONG_TEXT, "2026-01-03"),
        ]
        clusters = link_documents(docs)
        dup_edges = [e for e in clusters[0].edges if e.kind == EdgeKind.NEAR_DUPLICATE]
        assert len(dup_edges) >= 1

    def test_dissimilar_docs_form_separate_clusters(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc(
                "d2",
                "Completely unrelated ophthalmology note about glaucoma screening.",
                "2026-01-02",
            ),
        ]
        clusters = link_documents(docs)
        assert len(clusters) == 2, "Dissimilar docs must NOT cluster together"

    def test_timeline_ordered_ascending(self):
        docs = [
            _doc("d2", LONG_TEXT, "2026-03-01"),
            _doc("d1", LONG_TEXT, "2026-01-01"),
        ]
        clusters = link_documents(docs)
        assert len(clusters) == 1
        dates = [c["note_datetime"] for c in clusters[0].documents]
        assert dates == sorted(dates)

    def test_earlier_near_duplicate_is_retained_and_flagged(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", LONG_TEXT, "2026-01-03"),
        ]
        cluster = link_documents(docs)[0]

        assert cluster.superseded_ids == frozenset({"d1"})
        serialized = {doc["doc_id"]: doc for doc in cluster.to_dict()["documents"]}
        assert serialized["d1"]["superseded"] is True
        assert serialized["d2"]["superseded"] is False

    def test_timezone_aware_timestamps_sort_in_utc(self):
        docs = [
            _doc("later", LONG_TEXT, "2026-01-01T10:00:00+00:00"),
            _doc("earlier", LONG_TEXT, "2026-01-01T12:00:00+03:00"),
        ]
        cluster = link_documents(docs)[0]
        assert [doc["doc_id"] for doc in cluster.documents] == ["earlier", "later"]


# ---------------------------------------------------------------------------
# Amendment detection
# ---------------------------------------------------------------------------


class TestAmendmentDetection:
    def test_amendment_edge_detected(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", AMENDMENT_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        amend_edges = [e for e in clusters[0].edges if e.kind == EdgeKind.AMENDMENT]
        assert len(amend_edges) >= 1, "Amendment edge must be detected"

    def test_superseded_doc_retained_not_dropped(self):
        """Superseded documents must be retained and flagged, never dropped."""
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", AMENDMENT_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        all_ids = {d["doc_id"] for c in clusters for d in c.documents}
        assert "d1" in all_ids, "Superseded doc must be retained"
        assert "d2" in all_ids

    def test_amendment_direction_later_supersedes_earlier(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", AMENDMENT_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        amend_edges = [e for e in clusters[0].edges if e.kind == EdgeKind.AMENDMENT]
        assert any(e.source_id == "d2" and e.target_id == "d1" for e in amend_edges), (
            "Amendment edge must point from later to earlier doc"
        )

    def test_older_longer_document_does_not_supersede_later_shorter_document(self):
        docs = [
            _doc("older", AMENDMENT_TEXT, "2026-01-01"),
            _doc("later", LONG_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        amendment_edges = [
            edge
            for cluster in clusters
            for edge in cluster.edges
            if edge.kind == EdgeKind.AMENDMENT
        ]
        assert amendment_edges == []


# ---------------------------------------------------------------------------
# Clustering F1 >= 0.85
# ---------------------------------------------------------------------------


class TestClusteringF1:
    def test_f1_on_synthetic_bundle(self):
        """F1 on a synthetic multi-encounter bundle must be >= 0.85."""
        base = LONG_TEXT
        docs = [
            _doc("orig_1", base, "2026-01-01"),
            _doc("copy_1", base, "2026-01-03"),  # near-dup
            _doc("amend_1", base + " Addendum: BP rechecked 130/85.", "2026-01-05"),
            _doc(
                "uniq_1",
                "Dermatology note: psoriasis plaque on left elbow.",
                "2026-02-01",
            ),
            _doc(
                "uniq_2",
                "Ophthalmology note: intraocular pressure within normal limits.",
                "2026-03-01",
            ),
        ]
        # Ground truth: {orig_1, copy_1, amend_1} form one cluster; uniq_1 and uniq_2 are singletons
        gold_clusters = [
            {"orig_1", "copy_1", "amend_1"},
            {"uniq_1"},
            {"uniq_2"},
        ]

        pred_clusters = link_documents(docs)
        pred_sets = [{d["doc_id"] for d in c.documents} for c in pred_clusters]

        # Compute pairwise F1
        all_ids = [d["doc_id"] for d in docs]

        def _pairs(sets):
            return {(a, b) for s in sets for a, b in combinations(sorted(s), 2)}

        gold_pairs = _pairs(gold_clusters)
        pred_pairs = _pairs(pred_sets)

        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        assert f1 >= 0.85, f"Clustering F1 {f1:.3f} is below the required 0.85"


# ---------------------------------------------------------------------------
# Patient grouping and cross-document entity de-duplication
# ---------------------------------------------------------------------------


class TestPatientGrouping:
    def test_exact_caller_patient_ids_define_offline_cluster_boundaries(self):
        docs = [
            _doc(
                "p2-copy",
                LONG_TEXT,
                "2026-01-04",
                patient_id="patient-b",
            ),
            _doc(
                "p1-first",
                LONG_TEXT,
                "2026-01-01",
                patient_id="patient-a",
            ),
            _doc(
                "p1-unrelated",
                "Synthetic dermatology follow-up for a stable plaque.",
                "2026-02-01",
                patient_id="patient-a",
            ),
        ]

        clusters = link_documents(docs)

        assert [cluster.patient_id for cluster in clusters] == [
            "patient-a",
            "patient-b",
        ]
        assert [document["doc_id"] for document in clusters[0].documents] == [
            "p1-first",
            "p1-unrelated",
        ]
        assert clusters[1].edges == []


class TestEntityDeduplication:
    def test_repeated_event_entities_in_one_document_remain_distinct(self):
        cluster = link_documents(
            [
                _doc(
                    "encounter-1",
                    "Synthetic repeated creatinine results.",
                    patient_id="patient-a",
                    entities=[
                        {
                            "category": "lab",
                            "system": "loinc",
                            "code": "2160-0",
                            "text": "creatinine",
                        },
                        {
                            "category": "lab",
                            "system": "loinc",
                            "code": "2160-0",
                            "text": "creatinine",
                        },
                    ],
                )
            ]
        )[0]

        assert len(cluster.entities) == 2
        assert [
            occurrence.entity_index
            for entity in cluster.entities
            for occurrence in entity.provenance
        ] == [0, 1]

    def test_coded_event_surfaces_remain_distinct_in_linked_documents(self):
        cluster = link_documents(
            [
                _doc(
                    "encounter-1",
                    LONG_TEXT,
                    "2026-01-01",
                    patient_id="patient-a",
                    entities=[
                        {
                            "category": "lab",
                            "system": "loinc",
                            "code": "2160-0",
                            "value": "5.1",
                        }
                    ],
                ),
                _doc(
                    "encounter-2",
                    LONG_TEXT,
                    "2026-01-02",
                    patient_id="patient-a",
                    entities=[
                        {
                            "category": "lab",
                            "system": "loinc",
                            "code": "2160-0",
                            "value": "7.2",
                        }
                    ],
                ),
            ]
        )[0]

        assert len(cluster.edges) == 1
        assert len(cluster.entities) == 2

    def test_summary_card_uses_canonical_category_aliases(self):
        cluster = link_documents(
            [
                _doc(
                    "encounter-1",
                    "Synthetic longitudinal category aliases.",
                    patient_id="patient-a",
                    entities=[
                        {"category": "conditions", "text": "diabetes"},
                        {"category": "diagnoses", "text": "hypertension"},
                        {"category": "medicines", "text": "aspirin"},
                        {
                            "category": "laboratory_tests",
                            "system": "loinc",
                            "code": "2160-0",
                        },
                        {"category": "procedures", "text": "appendectomy"},
                    ],
                )
            ]
        )[0]

        card = build_summary_card(cluster)

        assert card.problems == 2
        assert card.medications == 1
        assert card.labs == 1
        assert card.procedures == 1
        assert card.other == 0

    def test_carried_forward_entities_count_once_with_every_occurrence(self):
        first_text = "Diabetes mellitus. Continue metformin."
        second_text = "DM is stable. Continue metformin. New asthma."
        docs = [
            _doc(
                "encounter-1",
                first_text,
                "2026-01-01",
                patient_id="patient-a",
                provenance={"source": "synthetic", "encounter": 1},
                entities=[
                    {
                        "entity_id": "raw-diabetes-1",
                        "category": "problem",
                        "text": "Diabetes mellitus",
                        "system": "snomed",
                        "code": "44054006",
                        "start": 0,
                        "end": 17,
                    },
                    {
                        "category": "medication",
                        "text": "metformin",
                        "start": 28,
                        "end": 37,
                    },
                ],
            ),
            _doc(
                "encounter-2",
                second_text,
                "2026-02-01",
                patient_id="patient-a",
                provenance={"source": "synthetic", "encounter": 2},
                entities=[
                    {
                        "entity_id": "raw-diabetes-2",
                        "category": "condition",
                        "text": "DM",
                        "coding": [{"system": "snomed", "code": "44054006"}],
                        "start": 0,
                        "end": 2,
                    },
                    {
                        "category": "drug",
                        "text": "metformin",
                        "start": 23,
                        "end": 32,
                    },
                    {
                        "category": "problem",
                        "text": "asthma",
                        "start": 38,
                        "end": 44,
                    },
                ],
            ),
        ]

        cluster = link_documents(docs)[0]

        assert len(cluster.deduplicated_entities) == 3
        diabetes = next(
            entity for entity in cluster.entities if entity.code == "44054006"
        )
        medication = next(
            entity for entity in cluster.entities if entity.category == "medications"
        )
        assert [item.doc_id for item in diabetes.provenance] == [
            "encounter-1",
            "encounter-2",
        ]
        assert [item.doc_id for item in medication.provenance] == [
            "encounter-1",
            "encounter-2",
        ]
        assert all(item.surface_hash for item in diabetes.provenance)
        assert all(item.document_provenance.metadata for item in diabetes.provenance)

        card = build_summary_card(cluster)
        assert card.problems == 2
        assert card.medications == 1
        assert card.coded_entities == 1
        assert build_summary_card(cluster.to_dict()) == card

        serialized = json.dumps(
            [entity.to_dict() for entity in cluster.entities], sort_keys=True
        )
        for source_surface in ("Diabetes mellitus", "DM is stable", "metformin"):
            assert source_surface not in serialized
        assert "raw-diabetes-1" not in serialized
        assert "raw-diabetes-2" not in serialized

    def test_synthetic_cross_document_deduplication_precision_is_at_least_090(self):
        docs = [
            _doc(
                "a1",
                "Synthetic diabetes and aspirin note.",
                patient_id="patient-a",
                entities=[
                    {"category": "problem", "system": "icd10", "code": "E11.9"},
                    {"category": "medication", "text": "aspirin"},
                    {"category": "lab", "system": "loinc", "code": "2160-0"},
                ],
            ),
            _doc(
                "a2",
                "Synthetic DM and aspirin follow-up.",
                patient_id="patient-a",
                entities=[
                    {
                        "category": "condition",
                        "coding": [{"system": "icd10", "code": "E11.9"}],
                    },
                    {"category": "drug", "text": "aspirin"},
                    {"category": "lab", "system": "loinc", "code": "2160-0"},
                    {"category": "problem", "text": "asthma"},
                ],
            ),
            _doc(
                "b1",
                "Separate synthetic diabetes note.",
                patient_id="patient-b",
                entities=[{"category": "problem", "system": "icd10", "code": "E11.9"}],
            ),
        ]
        gold_pairs = {
            frozenset({("a1", 0), ("a2", 0)}),
            frozenset({("a1", 1), ("a2", 1)}),
        }

        clusters = link_documents(docs)
        predicted_pairs = {
            frozenset(
                {
                    (left.doc_id, left.entity_index),
                    (right.doc_id, right.entity_index),
                }
            )
            for cluster in clusters
            for entity in cluster.entities
            for left, right in combinations(entity.provenance, 2)
        }
        true_positives = len(predicted_pairs & gold_pairs)
        precision = true_positives / len(predicted_pairs)

        assert precision >= 0.90
        assert predicted_pairs == gold_pairs
        patient_a = next(
            cluster for cluster in clusters if cluster.patient_id == "patient-a"
        )
        assert sum(entity.category == "labs" for entity in patient_a.entities) == 2

    def test_context_disagreement_prevents_false_entity_merge(self):
        docs = [
            _doc(
                "d1",
                "Synthetic diabetes mention.",
                patient_id="patient-a",
                entities=[
                    {
                        "category": "problem",
                        "system": "icd10",
                        "code": "E11.9",
                        "experiencer": "patient",
                    }
                ],
            ),
            _doc(
                "d2",
                "Synthetic family diabetes mention.",
                patient_id="patient-a",
                entities=[
                    {
                        "category": "problem",
                        "system": "icd10",
                        "code": "E11.9",
                        "experiencer": "family",
                    }
                ],
            ),
        ]

        assert len(link_documents(docs)[0].entities) == 2


class TestTimelineConsumption:
    def test_timeline_adapter_exposes_retained_documents_and_relationships(self):
        cluster = link_documents(
            [
                _doc(
                    "earlier",
                    LONG_TEXT,
                    "2026-01-01",
                    patient_id="patient-a",
                    provenance={"source": "synthetic"},
                ),
                _doc(
                    "later",
                    LONG_TEXT,
                    "2026-01-02",
                    patient_id="patient-a",
                    provenance={"source": "synthetic"},
                ),
            ]
        )[0]

        timeline = build_linked_document_timeline(cluster)

        assert isinstance(timeline, LinkedDocumentTimeline)
        assert all(
            isinstance(item, LinkedTimelineDocument) for item in timeline.documents
        )
        assert [item.doc_id for item in timeline.documents] == ["earlier", "later"]
        assert timeline.documents[0].superseded is True
        assert timeline.relationships[0].kind == EdgeKind.NEAR_DUPLICATE
        assert "text" not in json.dumps(timeline.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Offline / no-network assertion
# ---------------------------------------------------------------------------


class TestOfflineBehavior:
    def test_no_network_calls(self, monkeypatch):
        """link_documents must not open any sockets."""
        import socket

        def _blocked(*args, **kwargs):
            raise AssertionError("link_documents must not make network calls")

        monkeypatch.setattr(socket, "socket", _blocked)

        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", LONG_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# to_dict / serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_cluster_to_dict_keys(self):
        docs = [_doc("d1", LONG_TEXT, "2026-01-01")]
        cluster = link_documents(docs)[0]
        d = cluster.to_dict()
        assert "cluster_id" in d
        assert "documents" in d
        assert "edges" in d

    def test_edge_to_dict_keys(self):
        docs = [
            _doc(
                "d1",
                LONG_TEXT,
                "2026-01-01",
                provenance={"source": "synthetic", "record_index": 1},
            ),
            _doc(
                "d2",
                LONG_TEXT,
                "2026-01-02",
                provenance={"source": "synthetic", "record_index": 2},
            ),
        ]
        clusters = link_documents(docs)
        for edge in clusters[0].edges:
            d = edge.to_dict()
            assert {
                "source_id",
                "target_id",
                "kind",
                "similarity",
                "superseded",
                "source_provenance",
                "target_provenance",
            } <= d.keys()
            assert d["source_provenance"]["doc_id"] == "d2"
            assert d["target_provenance"]["doc_id"] == "d1"
            assert d["source_provenance"]["metadata"]["source"] == "synthetic"
            assert "text" not in d["source_provenance"]
            assert "text" not in d["target_provenance"]

    def test_public_types_and_disclaimer_are_exported(self):
        assert DeduplicatedEntity
        assert DocumentCluster
        assert DocumentEdge
        assert DocumentProvenance
        assert EdgeKind
        assert EntityOccurrence
        assert LinkedDocumentTimeline
        assert LinkedTimelineDocument
        assert "assistive software outputs" in DOCUMENT_LINKING_ADVISORY
