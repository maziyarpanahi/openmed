"""Synthetic offline tests for section and document-type label assembly."""

from __future__ import annotations

from collections import defaultdict

from openmed.clinical.sections import classify_document, detect_sections
from openmed.eval.datasets.public import PublicDatasetRecord
from openmed.training.data.section_doctype import (
    DOCUMENT_TYPES,
    RULE_LABEL_SOURCE,
    SECTION_LABELS,
    SYNTHETIC_LABEL_SOURCE,
    SectionDoctypeNote,
    build_document_type_examples,
    build_section_doctype_examples,
    build_section_examples,
    generate_synthetic_notes,
)

_EXPECTED_SECTIONS = {
    "history_of_present_illness",
    "past_medical_history",
    "medications",
    "allergies",
    "social_history",
    "assessment_and_plan",
    "findings",
    "impression",
}
_EXPECTED_DOCUMENT_TYPES = {
    "discharge_summary",
    "progress_note",
    "radiology_report",
    "pathology_report",
    "operative_note",
    "consult_note",
}


def test_synthetic_builder_covers_all_section_and_document_type_labels() -> None:
    notes = generate_synthetic_notes(seed=17)
    training_set = build_section_doctype_examples(notes)

    assert set(SECTION_LABELS) == _EXPECTED_SECTIONS
    assert {
        example.section_label for example in training_set.section_examples
    } == _EXPECTED_SECTIONS
    assert set(DOCUMENT_TYPES) == _EXPECTED_DOCUMENT_TYPES
    assert {example.doc_type for example in training_set.document_type_examples} == set(
        _EXPECTED_DOCUMENT_TYPES
    )
    assert {classify_document(note.text)["type"] for note in notes} == (
        _EXPECTED_DOCUMENT_TYPES
    )


def test_section_examples_preserve_contiguous_non_overlapping_runtime_spans() -> None:
    notes = generate_synthetic_notes(seed=23)
    examples = build_section_examples(notes)
    by_record = defaultdict(list)
    for example in examples:
        by_record[example.record_id].append(example)

    for note in notes:
        note_examples = by_record[note.record_id]
        detected = [
            span
            for span in detect_sections(note.text, language=note.language)
            if span.label in SECTION_LABELS
        ]

        assert [example.to_section_span() for example in note_examples] == detected
        assert all(
            left.end == right.start
            for left, right in zip(note_examples, note_examples[1:])
        )
        assert all(example.start < example.end for example in note_examples)
        assert all(
            note.text[example.start : example.end] == example.text
            for example in note_examples
        )


def test_rule_and_synthetic_examples_record_label_source() -> None:
    rule_note = PublicDatasetRecord(
        record_id="public-progress-note",
        dataset="shield",
        text="PROGRESS NOTE\nHPI: Synthetic cough.\nPMH: None reported.",
        spans=(),
        split="train",
    )
    synthetic_note = generate_synthetic_notes(seed=29)[0]

    section_examples = build_section_examples((rule_note, synthetic_note))
    document_examples = build_document_type_examples((rule_note, synthetic_note))

    assert {example.label_source for example in section_examples} == {
        RULE_LABEL_SOURCE,
        SYNTHETIC_LABEL_SOURCE,
    }
    assert {example.label_source for example in document_examples} == {
        RULE_LABEL_SOURCE,
        SYNTHETIC_LABEL_SOURCE,
    }
    rule_document = next(
        example
        for example in document_examples
        if example.record_id == "public-progress-note"
    )
    assert rule_document.doc_type == "progress_note"
    assert rule_document.metadata["rule_confidence"] > 0.5
    assert rule_document.metadata["dataset"] == "shield"
    assert rule_document.metadata["split"] == "train"
    assert synthetic_note.metadata["synthetic_sources"] == (
        "locale_phi",
        "social_history",
    )


def test_document_type_examples_use_only_the_first_n_tokens() -> None:
    note = SectionDoctypeNote(
        text="PROGRESS NOTE alpha, beta gamma delta epsilon.",
        record_id="window-test",
        doc_type="progress_note",
    )

    (example,) = build_document_type_examples((note,), max_tokens=4)

    assert example.text == "PROGRESS NOTE alpha, beta"
    assert example.token_count == 4
    assert example.max_tokens == 4
    assert "gamma" not in example.text
