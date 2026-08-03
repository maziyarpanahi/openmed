"""Tests for the synthetic multi-annotator gold corpus and consensus loader."""

from __future__ import annotations

import json

import pytest

from openmed.eval.golden.loader import (
    ConsensusDocument,
    ConsensusRelation,
    load_consensus_corpus,
    load_golden_fixtures,
)
from openmed.eval.metrics import EvalSpan

_DOC = {
    "id": "consensus-doc-001",
    "synthetic": True,
    "text": "Patient Jane Roe started metformin on 2025-03-02.",
    "annotators": {
        "ann_a": {
            "synthetic": True,
            "spans": [
                {"start": 8, "end": 16, "label": "PERSON"},
                {"start": 25, "end": 34, "label": "MEDICATION"},
            ],
        },
        "ann_b": {
            "synthetic": True,
            "spans": [
                {"start": 8, "end": 16, "label": "PERSON"},
                {"start": 38, "end": 48, "label": "DATE"},
            ],
        },
    },
    "consensus": {
        "spans": [
            {"start": 8, "end": 16, "label": "PERSON"},
            {"start": 25, "end": 34, "label": "MEDICATION"},
            {"start": 38, "end": 48, "label": "DATE"},
        ],
        "relations": [
            {
                "relation_type": "drug_to_date",
                "label": "confirmed",
                "head": {"start": 25, "end": 34, "label": "MEDICATION"},
                "tail": {"start": 38, "end": 48, "label": "DATE"},
            }
        ],
    },
}


def _write(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# Committed corpus
# --------------------------------------------------------------------------


def test_committed_corpus_has_multiple_annotators_and_consensus():
    documents = load_consensus_corpus()

    assert documents, "committed consensus corpus must not be empty"
    for document in documents:
        assert isinstance(document, ConsensusDocument)
        assert len(document.annotators) >= 2
        assert document.consensus_spans


def test_consensus_loads_eval_compatible_spans_and_relations():
    document = load_consensus_corpus()[0]

    assert all(isinstance(span, EvalSpan) for span in document.consensus_spans)
    assert set(document.annotator_relations) == set(document.annotators)
    assert any(document.annotator_relations.values())
    for relations in document.annotator_relations.values():
        assert all(isinstance(relation, ConsensusRelation) for relation in relations)
    for relation in document.consensus_relations:
        assert isinstance(relation, ConsensusRelation)
        assert isinstance(relation.head, EvalSpan)
        assert isinstance(relation.tail, EvalSpan)


def test_source_spans_match_document_text_offsets():
    for document in load_consensus_corpus():
        for spans in document.annotators.values():
            for span in spans:
                assert document.text[span.start : span.end]
        for span in document.consensus_spans:
            assert 0 <= span.start < span.end <= len(document.text)


def test_committed_corpus_does_not_break_golden_fixture_loader():
    # The consensus corpus must be excluded from the strict de-id validator.
    assert load_golden_fixtures() is not None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rejects_document_missing_synthetic_marker(tmp_path):
    bad = {**_DOC, "synthetic": False}
    path = _write(tmp_path / "c.jsonl", [bad])
    with pytest.raises(ValueError, match="synthetic"):
        load_consensus_corpus(path)


def test_rejects_annotation_missing_synthetic_marker(tmp_path):
    bad = json.loads(json.dumps(_DOC))
    bad["annotators"]["ann_a"]["synthetic"] = False
    path = _write(tmp_path / "c.jsonl", [bad])
    with pytest.raises(ValueError, match="synthetic"):
        load_consensus_corpus(path)


def test_rejects_offset_drift(tmp_path):
    drifted = json.loads(json.dumps(_DOC))
    drifted["consensus"]["spans"][0]["end"] = 999  # past end of text
    path = _write(tmp_path / "c.jsonl", [drifted])
    with pytest.raises(ValueError):
        load_consensus_corpus(path)


def test_rejects_single_annotator(tmp_path):
    single = json.loads(json.dumps(_DOC))
    del single["annotators"]["ann_b"]
    path = _write(tmp_path / "c.jsonl", [single])
    with pytest.raises(ValueError, match="two annotators"):
        load_consensus_corpus(path)


def test_rejects_relation_endpoint_outside_annotator_spans(tmp_path):
    bad = json.loads(json.dumps(_DOC))
    bad["annotators"]["ann_a"]["relations"] = [
        {
            "relation_type": "drug_to_date",
            "label": "confirmed",
            "head": {"start": 25, "end": 34, "label": "MEDICATION"},
            "tail": {"start": 38, "end": 48, "label": "DATE"},
        }
    ]
    path = _write(tmp_path / "c.jsonl", [bad])

    with pytest.raises(ValueError, match="tail must reference"):
        load_consensus_corpus(path)


def test_synthetic_fixture_round_trips(tmp_path):
    path = _write(tmp_path / "c.jsonl", [_DOC])
    document = load_consensus_corpus(path)[0]

    assert document.doc_id == "consensus-doc-001"
    assert set(document.annotators) == {"ann_a", "ann_b"}
    assert len(document.consensus_relations) == 1
