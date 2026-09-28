"""Tests for multi-annotator BRAT and Label Studio import adapters."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/eval/golden_annotations"


def test_brat_fixture_imports_two_annotators_and_relations() -> None:
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
    assert alex.metadata["document_id"] == "case_001"
    assert alex.metadata["source_format"] == "brat"

    assert [
        relation.to_tuple() for relation in document.relations_for("annotator-b")
    ] == [
        ("TREATS", "T2", "T3"),
        ("HAS_ADVERSE_EVENT", "T2", "T4"),
    ]
    adverse = document.relations_for("annotator-b")[1]
    assert adverse.source_span == spans[("annotator-b", "T2")]
    assert adverse.target_span == spans[("annotator-b", "T4")]
    assert adverse.metadata["source_annotation_id"] == "R2"
    assert adverse.metadata["source_id"] == "T2"
    assert adverse.metadata["target_id"] == "T4"
    assert adverse.to_eval_relation().head == adverse.source_span


def test_label_studio_fixture_imports_two_annotators_and_relations() -> None:
    documents = load_label_studio_multi_annotator_export(
        FIXTURE_DIR / "label_studio" / "case_001.json"
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.document_id == "case_001"
    assert document.source_format == "label_studio"
    assert document.annotators == ("annotator-a", "annotator-b")
    assert document.metadata["synthetic"] is True

    spans = {
        (
            span.metadata["annotator_id"],
            span.metadata["source_annotation_id"],
        ): span
        for span in document.spans
    }
    medication = spans[("annotator-a", "s2")]
    assert (medication.start, medication.end, medication.text) == (
        26,
        35,
        "Metformin",
    )
    assert medication.label == "MEDICATION"
    assert medication.metadata["source_format"] == "label_studio"
    assert spans[("annotator-b", "s4")].text == "nausea"

    assert [
        relation.to_tuple() for relation in document.relations_for("annotator-a")
    ] == [("TREATS", "s2", "s3")]
    assert [
        relation.to_tuple() for relation in document.relations_for("annotator-b")
    ] == [
        ("TREATS", "s2", "s3"),
        ("HAS_ADVERSE_EVENT", "s2", "s4"),
    ]
    assert document.relations[0].metadata["source_annotation_id"] == "r1"


def test_imports_accept_raw_brat_content_and_validate_relation_order() -> None:
    text = "Synthetic Alex takes aspirin."
    document = parse_brat_multi_annotator(
        text,
        {
            "reviewer": (
                "R1\tTAKES Arg1:T1 Arg2:T2\n"
                "T1\tPERSON 10 14\tAlex\n"
                "T2\tMEDICATION 21 28\taspirin\n"
            )
        },
        document_id="synthetic-raw",
    )

    assert document.relation_triples == (("TAKES", "T1", "T2"),)
    assert document.spans[0].metadata["document_id"] == "synthetic-raw"


@pytest.mark.parametrize(
    ("annotation_text", "message"),
    [
        (
            "T1\tPERSON 8 999\tAlex Rivera\n",
            "invalid offsets",
        ),
        (
            "T1\tPERSON 8 19\tWrong Name\n",
            "span text mismatch",
        ),
        (
            "T1\tPERSON 8 19\tAlex Rivera\nR1\tKNOWS Arg1:T1 Arg2:T9\n",
            "missing brat relation endpoint",
        ),
    ],
)
def test_brat_import_rejects_invalid_spans_and_relations(
    annotation_text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_brat_multi_annotator(
            "Patient Alex Rivera takes Metformin.",
            {"annotator-a": annotation_text},
            document_id="synthetic-invalid",
        )


def _label_studio_fixture() -> list[dict]:
    return json.loads(
        (FIXTURE_DIR / "label_studio" / "case_001.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload[0]["annotations"][0]["result"][0]["value"].update(
                {"end": 999}
            ),
            "invalid offsets",
        ),
        (
            lambda payload: payload[0]["annotations"][0]["result"][-1].update(
                {"to_id": "missing-span"}
            ),
            "missing label_studio relation endpoint",
        ),
    ],
)
def test_label_studio_import_rejects_invalid_offsets_and_endpoints(
    mutate,
    message: str,
) -> None:
    payload = _label_studio_fixture()
    malformed = copy.deepcopy(payload)
    mutate(malformed)

    with pytest.raises(ValueError, match=message):
        parse_label_studio_multi_annotator_export(malformed)


def test_committed_annotation_samples_are_synthetic_only() -> None:
    readme = (FIXTURE_DIR / "README.md").read_text(encoding="utf-8").lower()
    label_studio = _label_studio_fixture()
    brat_text = (FIXTURE_DIR / "brat" / "case_001.txt").read_text(encoding="utf-8")

    assert "synthetic-only" in readme
    assert "real clinical notes" in readme
    assert all(task["data"]["synthetic"] is True for task in label_studio)
    assert "synthetic" not in brat_text.lower()
