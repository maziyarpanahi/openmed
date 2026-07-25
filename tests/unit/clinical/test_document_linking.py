"""Unit tests for openmed.clinical.document_linking."""
from __future__ import annotations

import pytest
from datetime import datetime

from openmed.clinical.document_linking import (
    DocumentCluster,
    DocumentEdge,
    EdgeKind,
    link_documents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(doc_id: str, text: str, note_datetime: str | None = None) -> dict:
    d: dict = {"doc_id": doc_id, "text": text}
    if note_datetime is not None:
        d["note_datetime"] = note_datetime
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


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

class TestNearDuplicates:
    def test_identical_docs_form_one_cluster(self):
        docs = [
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", LONG_TEXT, "2026-01-03"),   # re-fax of d1
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
            _doc("d2", "Completely unrelated ophthalmology note about glaucoma screening.", "2026-01-02"),
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


# ---------------------------------------------------------------------------
# Amendment detection
# ---------------------------------------------------------------------------

class TestAmendmentDetection:
    def test_amendment_edge_detected(self):
        docs = [
            _doc("d1", LONG_TEXT,      "2026-01-01"),
            _doc("d2", AMENDMENT_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        amend_edges = [e for e in clusters[0].edges if e.kind == EdgeKind.AMENDMENT]
        assert len(amend_edges) >= 1, "Amendment edge must be detected"

    def test_superseded_doc_retained_not_dropped(self):
        """Superseded documents must be retained and flagged, never dropped."""
        docs = [
            _doc("d1", LONG_TEXT,      "2026-01-01"),
            _doc("d2", AMENDMENT_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        all_ids = {d["doc_id"] for c in clusters for d in c.documents}
        assert "d1" in all_ids, "Superseded doc must be retained"
        assert "d2" in all_ids

    def test_amendment_direction_later_supersedes_earlier(self):
        docs = [
            _doc("d1", LONG_TEXT,      "2026-01-01"),
            _doc("d2", AMENDMENT_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        amend_edges = [e for e in clusters[0].edges if e.kind == EdgeKind.AMENDMENT]
        assert any(e.source_id == "d2" and e.target_id == "d1" for e in amend_edges), \
            "Amendment edge must point from later to earlier doc"


# ---------------------------------------------------------------------------
# Clustering F1 >= 0.85
# ---------------------------------------------------------------------------

class TestClusteringF1:
    def test_f1_on_synthetic_bundle(self):
        """F1 on a synthetic multi-encounter bundle must be >= 0.85."""
        base = LONG_TEXT
        docs = [
            _doc("orig_1",  base,                    "2026-01-01"),
            _doc("copy_1",  base,                    "2026-01-03"),   # near-dup
            _doc("amend_1", base + " Addendum: BP rechecked 130/85.", "2026-01-05"),
            _doc("uniq_1",  "Dermatology note: psoriasis plaque on left elbow.",      "2026-02-01"),
            _doc("uniq_2",  "Ophthalmology note: intraocular pressure within normal limits.", "2026-03-01"),
        ]
        # Ground truth: {orig_1, copy_1, amend_1} form one cluster; uniq_1 and uniq_2 are singletons
        gold_clusters = [
            {"orig_1", "copy_1", "amend_1"},
            {"uniq_1"},
            {"uniq_2"},
        ]

        pred_clusters = link_documents(docs)
        pred_sets = [
            {d["doc_id"] for d in c.documents} for c in pred_clusters
        ]

        # Compute pairwise F1
        all_ids = [d["doc_id"] for d in docs]
        from itertools import combinations

        def _pairs(sets):
            return {
                (a, b)
                for s in sets
                for a, b in combinations(sorted(s), 2)
            }

        gold_pairs = _pairs(gold_clusters)
        pred_pairs = _pairs(pred_sets)

        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        assert f1 >= 0.85, f"Clustering F1 {f1:.3f} is below the required 0.85"


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
            _doc("d1", LONG_TEXT, "2026-01-01"),
            _doc("d2", LONG_TEXT, "2026-01-02"),
        ]
        clusters = link_documents(docs)
        for edge in clusters[0].edges:
            d = edge.to_dict()
            assert "source_id" in d
            assert "target_id" in d
            assert "kind" in d
            assert "similarity" in d
            assert "superseded" in d