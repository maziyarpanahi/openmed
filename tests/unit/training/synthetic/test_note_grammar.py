"""All five offline grammars and independently corrupted synthetic truth."""

from dataclasses import replace

import pytest

from openmed.training.synthetic.note_grammar import (
    GRAMMARS,
    SyntheticNoteFact,
    render_synthetic_note,
    synthetic_note_facts,
    validate_synthetic_note,
)


@pytest.mark.parametrize("kind", sorted(GRAMMARS))
def test_all_grammars_render_exact_gold_and_shared_facts(kind):
    facts = synthetic_note_facts(seed=42)
    allowed = {key for _section, keys in GRAMMARS[kind] for key in keys}
    note = render_synthetic_note(kind, tuple(f for f in facts if f.key in allowed))
    assert validate_synthetic_note(note) == ()
    assert all(note.text[s.start : s.end] == s.text for s in note.gold_spans)
    assert all(
        s.metadata["role"] in {"identifier", "clinical_fact", "distractor"}
        for s in note.gold_spans
    )
    assert note.synthetic


def test_surrogates_reproduce_and_notes_do_not_mutate_facts():
    assert synthetic_note_facts(seed=42) == synthetic_note_facts(seed=42)
    facts = synthetic_note_facts(seed=42)
    a = render_synthetic_note("discharge", facts)
    assert a == render_synthetic_note("discharge", facts)
    diagnoses = [
        s
        for s in a.gold_spans
        if s.metadata["fact_id"]
        == next(f.fact_id for f in facts if f.key == "diagnosis")
    ]
    assert len(diagnoses) == 2
    assert diagnoses[0].text == diagnoses[1].text


def test_omissions_and_missing_facts_remain_missing():
    facts = synthetic_note_facts(seed=42)
    note = render_synthetic_note("encounter", facts, omitted_sections=("medications",))
    assert "MEDICATIONS:" not in note.text
    assert not any(s.label == "DRUG" for s in note.gold_spans)
    partial = render_synthetic_note(
        "encounter", tuple(f for f in facts if f.key in {"person", "identifier"})
    )
    assert len(partial.sections) == 1
    assert "synthetic condition" not in partial.text


def test_independent_validator_catches_offset_label_and_provenance_changes():
    note = render_synthetic_note("encounter", synthetic_note_facts(seed=42))
    span = note.gold_spans[0]
    assert "span_mismatch" in validate_synthetic_note(
        replace(
            note,
            gold_spans=(replace(span, start=span.start + 1),) + note.gold_spans[1:],
        )
    )
    assert "fact_truth_mismatch" in validate_synthetic_note(
        replace(note, gold_spans=(replace(span, label="DRUG"),) + note.gold_spans[1:])
    )
    assert "synthetic_provenance_missing" in validate_synthetic_note(
        replace(note, synthetic=False)
    )


def test_invalid_grammar_cross_field_dependency_and_duplicate_facts_reject():
    with pytest.raises(ValueError, match="unknown"):
        render_synthetic_note("unknown", ())
    dose = SyntheticNoteFact("dose", "DOSAGE", "synthetic dose")
    with pytest.raises(ValueError, match="requires medication"):
        render_synthetic_note("encounter", (dose,))
    with pytest.raises(ValueError, match="duplicate"):
        render_synthetic_note("encounter", (dose, dose))


def test_missing_or_duplicate_gold_annotations_do_not_pass_validator():
    note = render_synthetic_note("encounter", synthetic_note_facts(seed=42))
    assert "incomplete_or_duplicate_fact_truth" in validate_synthetic_note(
        replace(note, gold_spans=())
    )
    assert "incomplete_or_duplicate_fact_truth" in validate_synthetic_note(
        replace(note, gold_spans=note.gold_spans + (note.gold_spans[0],))
    )
