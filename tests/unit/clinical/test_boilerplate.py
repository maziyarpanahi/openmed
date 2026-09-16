"""Synthetic offline tests for boilerplate and copy-forward span annotations."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from openmed.clinical import (
    DEFAULT_BOILERPLATE_TEMPLATE_RESOURCE,
    BoilerplateSpan,
    CopyForwardSpan,
    build_summary_card,
    detect_boilerplate,
    detect_copy_forward,
    load_boilerplate_template_corpus,
)
from openmed.processing import apply_boilerplate_suppression

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "boilerplate_copy_forward.json"
)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _gold_offsets(text: str, gold_surfaces: list[dict]) -> set[tuple[int, int]]:
    offsets: set[tuple[int, int]] = set()
    for item in gold_surfaces:
        surface = item["surface"]
        start = -1
        for _ in range(item["occurrence"] + 1):
            start = text.find(surface, start + 1)
        assert start >= 0
        offsets.add((start, start + len(surface)))
    return offsets


def _exact_span_metrics(
    gold: set[tuple[str, int, int]],
    predicted: set[tuple[str, int, int]],
) -> dict[str, float]:
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else float(not gold)
    recall = true_positive / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def test_detect_boilerplate_returns_offset_aligned_non_destructive_spans() -> None:
    text = (
        "Assessment: Synthetic cough is improving today.\n"
        "  Constitutional: well developed, well nourished, and in no acute distress.  \n"
    )
    original = text[:]

    spans = detect_boilerplate(text)

    assert text == original
    assert spans == detect_boilerplate(text)
    assert len(spans) == 1
    span = spans[0]
    assert isinstance(span, BoilerplateSpan)
    assert span.type == "boilerplate"
    assert text[span.start : span.end] == (
        "Constitutional: well developed, well nourished, and in no acute distress."
    )
    assert span.provenance["detector"] == "openmed-boilerplate-v1"
    assert span.to_dict()["provenance"]["rule_ids"] == (
        "template:constitutional-normal-exam",
    )


def test_copy_forward_spans_include_cross_and_intra_source_provenance() -> None:
    repeated = "Synthetic abdominal discomfort improved with hydration and gentle activity today."
    text = f"Assessment:\n{repeated}\nPlan:\n{repeated}\n"
    source = (
        "Synthetic respiratory symptoms improved after hydration and quiet rest today."
    )
    current = f"Assessment:\n{source}\n"

    intra = detect_copy_forward(text, document_id="current-note")
    cross = detect_copy_forward(
        current,
        source_documents={"prior-note": source},
        document_id="current-note",
    )

    assert len(intra) == 1
    assert isinstance(intra[0], CopyForwardSpan)
    assert intra[0].type == "copy_forward"
    assert intra[0].copied_from == "current-note"
    assert intra[0].provenance["match_kind"] == "intra_document"
    assert text[intra[0].start : intra[0].end] == repeated
    assert len(cross) == 1
    assert cross[0].copied_from == "prior-note"
    assert cross[0].provenance["match_kind"] == "cross_document"
    assert current[cross[0].start : cross[0].end] == source


def test_synthetic_detection_fixture_meets_span_f1_gates() -> None:
    fixture = _load_fixture()
    boilerplate_gold: set[tuple[str, int, int]] = set()
    boilerplate_predicted: set[tuple[str, int, int]] = set()
    for case in fixture["boilerplate_cases"]:
        boilerplate_gold.update(
            (case["case_id"], start, end)
            for start, end in _gold_offsets(case["text"], case["gold_surfaces"])
        )
        boilerplate_predicted.update(
            (case["case_id"], span.start, span.end)
            for span in detect_boilerplate(case["text"])
        )

    copy_gold: set[tuple[str, int, int]] = set()
    copy_predicted: set[tuple[str, int, int]] = set()
    for case in fixture["copy_forward_cases"]:
        copy_gold.update(
            (case["case_id"], start, end)
            for start, end in _gold_offsets(case["text"], case["gold_surfaces"])
        )
        copy_predicted.update(
            (case["case_id"], span.start, span.end)
            for span in detect_copy_forward(
                case["text"],
                source_documents=case["source_documents"],
                document_id=case["document_id"],
            )
        )

    boilerplate_metrics = _exact_span_metrics(boilerplate_gold, boilerplate_predicted)
    copy_metrics = _exact_span_metrics(copy_gold, copy_predicted)
    assert boilerplate_metrics["precision"] >= 0.85
    assert boilerplate_metrics["recall"] >= 0.85
    assert boilerplate_metrics["f1"] >= 0.85
    assert copy_metrics["precision"] >= 0.80
    assert copy_metrics["recall"] >= 0.80
    assert copy_metrics["f1"] >= 0.80


def test_suppression_drops_only_fully_flagged_entities_and_records_provenance() -> None:
    text = (
        "Assessment: new synthetic cough.\n"
        "Constitutional: well developed, well nourished, and in no acute distress.\n"
    )
    annotations = detect_boilerplate(text)
    boilerplate_start = text.index("no acute distress")
    current_start = text.index("new synthetic cough")
    entities = [
        {
            "category": "problem",
            "start": boilerplate_start,
            "end": boilerplate_start + len("no acute distress"),
        },
        {
            "category": "problem",
            "start": current_start,
            "end": current_start + len("new synthetic cough"),
        },
        {
            "category": "medication",
            "start": annotations[0].end - 4,
            "end": annotations[0].end + 1,
        },
    ]
    original_entities = [dict(entity) for entity in entities]

    disabled = apply_boilerplate_suppression(entities, annotations)
    enabled = apply_boilerplate_suppression(
        entities, annotations, suppress_boilerplate=True
    )
    card = build_summary_card(
        entities,
        boilerplate_spans=annotations,
        suppress_boilerplate=True,
    )
    document_card = build_summary_card(
        {"clinical_entities": entities, "boilerplate_spans": annotations},
        suppress_boilerplate=True,
    )
    copy_suppression = apply_boilerplate_suppression(
        [{"category": "lab", "offsets": (4, 9)}],
        [{"type": "copy_forward", "start": 3, "end": 10}],
        suppress_boilerplate=True,
    )

    assert len(disabled.records) == 3
    assert len(enabled.records) == 2
    assert entities == original_entities
    assert copy_suppression.records == ()
    assert copy_suppression.provenance["suppressed_entity_count"] == 1
    assert enabled.provenance == {
        "enabled": True,
        "policy": "drop_fully_contained",
        "input_entity_count": 3,
        "retained_entity_count": 2,
        "suppressed_entity_count": 1,
        "annotation_count": 1,
    }
    assert card.to_dict() == {
        "entity_counts": {
            "problems": 1,
            "medications": 1,
            "labs": 0,
            "procedures": 0,
            "other": 0,
        },
        "coding_counts": {
            "coded_entities": 0,
            "uncoded_entities": 2,
            "distinct_codes": 0,
        },
        "section_count": 0,
        "provenance": {"boilerplate_suppression": enabled.provenance},
    }
    assert document_card == card


def test_template_and_evaluation_data_are_synthetic_and_unrestricted() -> None:
    resource = resources.files("openmed.clinical").joinpath(
        DEFAULT_BOILERPLATE_TEMPLATE_RESOURCE
    )
    corpus = json.loads(resource.read_text(encoding="utf-8"))
    fixture = _load_fixture()

    expected_provenance = {
        "source": "OpenMed synthetic clinical template phrases",
        "license": "Apache-2.0",
        "restricted_data": False,
        "synthetic": True,
    }
    assert corpus["provenance"] == expected_provenance
    assert fixture["provenance"]["restricted_data"] is False
    assert fixture["provenance"]["synthetic"] is True
    assert {entry["source_type"] for entry in corpus["templates"]} <= {
        "synthetic",
        "public_domain",
    }
    assert len(load_boilerplate_template_corpus()) == len(corpus["templates"])
    corpus_text = json.dumps(corpus, sort_keys=True).casefold()
    for restricted_source in ("mimic", "i2b2", "n2c2", '"source_type": "dua"'):
        assert restricted_source not in corpus_text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"shingle_size": 1}, "shingle_size"),
        ({"shingle_size": 9, "min_tokens": 8}, "min_tokens"),
        ({"document_id": ""}, "document_id"),
    ),
)
def test_copy_forward_rejects_invalid_options(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        detect_copy_forward(
            "Synthetic text with enough words for validation.", **kwargs
        )
