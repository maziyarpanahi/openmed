"""Tests for multi-annotator gold annotation import adapters."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.eval.golden import (
    load_brat_multi_annotator_document,
    load_label_studio_multi_annotator_export,
    parse_brat_multi_annotator,
    parse_label_studio_multi_annotator_export,
)

FIXTURE_DIR = Path("tests/fixtures/eval/golden_annotations")


def test_brat_multi_annotator_imports_spans_and_relations_with_offsets() -> None:
    document = load_brat_multi_annotator_document(
        FIXTURE_DIR / "brat" / "case_001.txt",
        {
            "annotator-a": FIXTURE_DIR / "brat" / "case_001_annotator_a.ann",
            "annotator-b": FIXTURE_DIR / "brat" / "case_001_annotator_b.ann",
        },
    )

    assert document.document_id == "case_001"
    assert document.source_format == "brat"
    assert document.annotators == ("annotator-a", "annotator-b")
    assert len(document.spans_for("annotator-a")) == 3
    assert len(document.spans_for("annotator-b")) == 4

    spans = {
        (
            span.metadata["annotator_id"],
            span.metadata["source_annotation_id"],
        ): span
        for span in document.spans
    }
    alex = spans[("annotator-a", "T1")]
    assert (alex.start, alex.end, alex.label, alex.text) == (
        8,
        19,
        "PERSON",
        "Alex Rivera",
    )
    assert document.text[alex.start : alex.end] == alex.text

    medication = spans[("annotator-b", "T2")]
    symptom = spans[("annotator-b", "T4")]
    assert medication.text == "Metformin"
    assert symptom.text == "nausea"

    assert [
        relation.to_tuple() for relation in document.relations_for("annotator-b")
    ] == [
        ("TREATS", "T2", "T3"),
        ("HAS_ADVERSE_EVENT", "T2", "T4"),
    ]
    adverse = document.relations_for("annotator-b")[1]
    assert adverse.source_span == medication
    assert adverse.target_span == symptom


def test_label_studio_imports_spans_and_relation_triples() -> None:
    documents = load_label_studio_multi_annotator_export(
        FIXTURE_DIR / "label_studio" / "case_001.json"
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.document_id == "case_001"
    assert document.source_format == "label_studio"
    assert document.annotators == ("annotator-a", "annotator-b")

    spans = {
        (
            span.metadata["annotator_id"],
            span.metadata["source_annotation_id"],
        ): span
        for span in document.spans
    }
    assert (spans[("annotator-a", "s2")].start, spans[("annotator-a", "s2")].end) == (
        26,
        35,
    )
    assert spans[("annotator-a", "s2")].text == "Metformin"
    assert spans[("annotator-b", "s4")].text == "nausea"
    assert spans[("annotator-b", "s4")].metadata["source_format"] == "label_studio"

    assert [
        relation.to_tuple() for relation in document.relations_for("annotator-a")
    ] == [("TREATS", "s2", "s3")]
    assert [
        relation.to_tuple() for relation in document.relations_for("annotator-b")
    ] == [
        ("TREATS", "s2", "s3"),
        ("HAS_ADVERSE_EVENT", "s2", "s4"),
    ]


def test_brat_import_rejects_span_text_mismatch() -> None:
    text = "Patient Alex Rivera takes Metformin."

    with pytest.raises(ValueError, match="span text mismatch"):
        parse_brat_multi_annotator(
            text,
            {"annotator-a": "T1\tPERSON 8 19\tWrong Name\n"},
            document_id="bad-offset",
        )


def test_label_studio_import_rejects_missing_relation_endpoint() -> None:
    payload = json.loads(
        (FIXTURE_DIR / "label_studio" / "case_001.json").read_text(encoding="utf-8")
    )
    malformed = copy.deepcopy(payload)
    malformed[0]["annotations"][0]["result"][-1]["to_id"] = "missing-span"

    with pytest.raises(ValueError, match="missing label_studio relation endpoint"):
        parse_label_studio_multi_annotator_export(malformed)


def test_committed_multi_annotator_samples_are_marked_synthetic() -> None:
    readme = (FIXTURE_DIR / "README.md").read_text(encoding="utf-8").lower()
    payload = json.loads(
        (FIXTURE_DIR / "label_studio" / "case_001.json").read_text(encoding="utf-8")
    )

    assert "synthetic-only" in readme
    assert payload[0]["data"]["synthetic"] is True
