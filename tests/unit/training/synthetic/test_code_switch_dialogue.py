"""Multilingual semantic dialogue, offsets and adjudication controls."""

from dataclasses import replace

import pytest

from openmed.core.decoding.spans import iter_grapheme_clusters
from openmed.training.synthetic.code_switch_dialogue import (
    DialogueTurnSpec,
    generate_code_switch_dialogue,
)


def specifications():
    return [
        DialogueTurnSpec("patient", ("en", "hi")),
        DialogueTurnSpec("caregiver", ("es",), status="negated"),
        DialogueTurnSpec("interpreter", ("hi", "es"), status="negated"),
        DialogueTurnSpec("clinician", ("en",), status="uncertain", correction_of=1),
    ]


def test_actual_multiturn_roles_switches_and_clinical_truth_are_preserved():
    result = generate_code_switch_dialogue(specifications(), seed=7)
    assert [turn.speaker_role for turn in result.turns] == [
        "patient",
        "caregiver",
        "interpreter",
        "clinician",
    ]
    assert result.report()["switch_count"] == 2
    assert set(result.report()["language_segment_counts"]) == {"en", "hi", "es"}
    for turn, spec in zip(result.turns, specifications()):
        assert all(fact.status == spec.status for fact in turn.facts)
        assert all(fact.value["synthetic_code"] == spec.concept for fact in turn.facts)
        assert len(turn.facts) == len(spec.languages)


def test_every_phi_span_and_language_boundary_is_grapheme_safe():
    for turn in generate_code_switch_dialogue(specifications(), seed=7).turns:
        boundaries = {0, len(turn.text)} | {
            point for pair in iter_grapheme_clusters(turn.text) for point in pair
        }
        for span in turn.phi_spans:
            assert span.start in boundaries and span.end in boundaries
            assert turn.text[span.start : span.end] == span.text
            assert span.metadata["language"] in {
                tag
                for tag, start, end in turn.language_segments
                if start <= span.start < span.end <= end
            }
        assert all(
            start in boundaries and end in boundaries
            for _language, start, end in turn.language_segments
        )


def test_gold_clinical_evidence_points_to_actual_language_tagged_statement():
    result = generate_code_switch_dialogue(specifications(), seed=9)
    for turn in result.turns:
        for fact, locator in zip(turn.facts, turn.evidence):
            assert fact.evidence_ids == (locator.locator_id,)
            assert locator.artifact_id == turn.artifact.artifact_id
            excerpt = turn.text[locator.location["start"] : locator.location["end"]]
            assert excerpt
            assert fact.attributes["language"] in dict(
                (language, None) for language, _start, _end in turn.language_segments
            )
            assert fact.attributes["experiencer_role"] == "patient"


def test_contradictions_and_corrections_keep_prior_evidence_and_need_review():
    result = generate_code_switch_dialogue(specifications(), seed=7)
    assert all(turn.needs_adjudication for turn in result.turns)
    assert result.turns[3].correction_of == 1
    assert set(result.turns[3].facts[0].parent_fact_ids) == {
        fact.fact_id for fact in result.turns[1].facts
    }
    assert result.turns[0].facts[0].status == "affirmed"
    assert result.turns[1].facts[0].status == "negated"


def test_distinct_experiencer_does_not_create_false_contradiction():
    result = generate_code_switch_dialogue(
        [
            DialogueTurnSpec("patient", ("en",)),
            DialogueTurnSpec(
                "caregiver", ("es",), status="negated", experiencer_role="caregiver"
            ),
        ]
    )
    assert not any(turn.needs_adjudication for turn in result.turns)


def test_repeat_seed_reproduces_training_records_and_same_role_locale_identity():
    first = generate_code_switch_dialogue(specifications(), seed=7)
    second = generate_code_switch_dialogue(specifications(), seed=7)
    assert [turn.to_training_item() for turn in first.turns] == [
        turn.to_training_item() for turn in second.turns
    ]
    assert first.turns[0].phi_spans[0].text == first.turns[3].phi_spans[0].text


@pytest.mark.parametrize(
    "plan",
    [
        [DialogueTurnSpec("patient", ("en",), correction_of=0)],
        [
            DialogueTurnSpec("patient", ("en",)),
            DialogueTurnSpec(
                "clinician", ("es",), concept="medication_example", correction_of=0
            ),
        ],
    ],
)
def test_invalid_correction_identity_or_future_reference_fails(plan):
    with pytest.raises(ValueError):
        generate_code_switch_dialogue(plan)


def test_language_switches_are_explicit_not_random_string_concatenation():
    with pytest.raises(ValueError):
        DialogueTurnSpec("patient", ("en", "en"))
    with pytest.raises(ValueError):
        DialogueTurnSpec("patient", ("unreviewed",))


def test_summary_contains_no_generated_name_or_turn_text():
    result = generate_code_switch_dialogue(specifications(), seed=7)
    summary = str(result.report())
    assert all(
        span.text not in summary for turn in result.turns for span in turn.phi_spans
    )


def test_public_turn_constructor_cannot_export_stale_source_bound_gold():
    turn = generate_code_switch_dialogue([DialogueTurnSpec("patient", ("en",))]).turns[
        0
    ]
    with pytest.raises(ValueError, match="artifact"):
        replace(turn, text="different text")
    with pytest.raises(ValueError, match="PHI"):
        replace(turn, phi_spans=(replace(turn.phi_spans[0], start=0),))


@pytest.mark.parametrize(
    "field", ["speaker_role", "experiencer_role", "concept", "status"]
)
def test_nested_semantic_codes_raise_controlled_validation_error(field):
    args = dict(speaker_role="patient", languages=("en",))
    args[field] = []
    with pytest.raises(ValueError):
        DialogueTurnSpec(**args)


def test_constructor_rejects_evidence_beyond_source_and_duplicate_facts():
    turn = generate_code_switch_dialogue(specifications(), seed=7).turns[0]
    locator = replace(
        turn.evidence[0], location={"start": 0, "end": len(turn.text) + 10}
    )
    with pytest.raises(ValueError):
        replace(turn, evidence=(locator,) + turn.evidence[1:])
    with pytest.raises(ValueError):
        replace(turn, facts=(turn.facts[0], turn.facts[0]))


def test_constructor_refuses_to_consume_unbounded_sequences():
    turn = generate_code_switch_dialogue(specifications(), seed=7).turns[0]

    def forbidden():
        raise AssertionError("iterator must not be consumed")
        yield 0

    with pytest.raises(ValueError):
        replace(turn, evidence=forbidden())


def test_generated_turn_role_and_language_have_controlled_type_guards():
    turn = generate_code_switch_dialogue(specifications(), seed=7).turns[0]
    with pytest.raises(ValueError):
        replace(turn, speaker_role=[])
    with pytest.raises(ValueError):
        replace(turn, language_segments=(([], 0, len(turn.text)),))
