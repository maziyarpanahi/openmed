"""Tests for assertion-gated drug-to-ADE and drug-to-Reason extraction."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical.relations import (
    ADE_RELATION_DISCLAIMER,
    DRUG_TO_ADE,
    DRUG_TO_REASON,
    extract_ade_relations,
    reconstruct_medication_ade_records,
)
from openmed.eval.golden.loader import list_fixture_paths

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "ade_relations.jsonl"
)


def _cases() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]


def _signature(relation: object) -> tuple[int, int, int, int, str, str]:
    return (
        relation.head.start,
        relation.head.end,
        relation.tail.start,
        relation.tail.end,
        relation.relation_type,
        relation.tail_assertion_status,
    )


def test_synthetic_gold_relation_f1_meets_threshold() -> None:
    predicted: set[tuple[str, int, int, int, int, str, str]] = set()
    gold: set[tuple[str, int, int, int, int, str, str]] = set()
    for case in _cases():
        predicted.update(
            (case["id"], *_signature(relation))
            for relation in extract_ade_relations(case["text"], case["spans"])
        )
        gold.update((case["id"], *relation) for relation in case["gold_relations"])

    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted)
    recall = true_positives / len(gold)
    relation_f1 = 2 * precision * recall / (precision + recall)

    assert relation_f1 >= 0.70


def test_negated_ade_is_refuted_and_never_asserted_positive() -> None:
    case = next(case for case in _cases() if case["id"] == "negated-ade-trap")

    [relation] = extract_ade_relations(case["text"], case["spans"])

    assert relation.relation_type == DRUG_TO_ADE
    assert relation.tail_assertion_status == "refuted"
    assert relation.tail_negation == "negated"
    assert not relation.asserted_positive


def test_family_experiencer_ade_is_excluded_from_patient_relations() -> None:
    case = next(case for case in _cases() if case["id"] == "family-experiencer-trap")

    assert extract_ade_relations(case["text"], case["spans"]) == ()


def test_family_history_section_hint_is_excluded_without_surface_cue() -> None:
    text = "Penicillin caused rash."
    spans = [
        {"start": 0, "end": 10, "label": "MEDICATION", "section": "family_history"},
        {"start": 18, "end": 22, "label": "PROBLEM", "section": "family_history"},
    ]

    assert extract_ade_relations(text, spans) == ()


def test_indication_label_is_typed_as_reason_without_lexical_cue() -> None:
    text = "Metformin diabetes"
    spans = [
        {"start": 0, "end": 9, "label": "MEDICATION"},
        {"start": 10, "end": 18, "label": "INDICATION"},
    ]

    [relation] = extract_ade_relations(text, spans)

    assert relation.relation_type == DRUG_TO_REASON


def test_ade_and_reason_are_not_conflated_for_one_pair() -> None:
    for case in _cases():
        relations = extract_ade_relations(case["text"], case["spans"])
        types_by_pair: dict[tuple[int, int, int, int], set[str]] = {}
        for relation in relations:
            pair = (*relation.head.offset_key(), *relation.tail.offset_key())
            types_by_pair.setdefault(pair, set()).add(relation.relation_type)

        assert all(
            len(relation_types) == 1 for relation_types in types_by_pair.values()
        )


def test_every_relation_carries_offsets_provenance_and_disclaimer() -> None:
    relations = tuple(
        relation
        for case in _cases()
        for relation in extract_ade_relations(case["text"], case["spans"])
    )

    assert relations
    assert all(len(relation.evidence_offsets) >= 2 for relation in relations)
    assert all(relation.provenance["candidate_hash"] for relation in relations)
    assert all(
        relation.provenance["source"] == "cue_fallback" for relation in relations
    )
    assert all(
        relation.head.text not in json.dumps(dict(relation.provenance))
        for relation in relations
    )
    assert all(
        relation.tail.text not in json.dumps(dict(relation.provenance))
        for relation in relations
    )
    assert all(relation.disclaimer_flag for relation in relations)
    assert all(relation.disclaimer == ADE_RELATION_DISCLAIMER for relation in relations)


def test_optional_relation_head_selects_one_highest_scoring_type() -> None:
    class LocalHead:
        def predict_relation_scores(self, text, *, head_offsets, tail_offsets):
            del text, head_offsets, tail_offsets
            return {DRUG_TO_ADE: 0.61, DRUG_TO_REASON: 0.91}

    case = next(case for case in _cases() if case["id"] == "asserted-ade-after")

    [relation] = extract_ade_relations(
        case["text"],
        case["spans"],
        relation_head=LocalHead(),
    )

    assert relation.relation_type == DRUG_TO_REASON
    assert relation.score == 0.91
    assert relation.provenance["source"] == "model"


def test_regimen_and_ade_set_reconstruct_into_one_medication_record() -> None:
    case = next(case for case in _cases() if case["id"] == "regimen-reason-and-ade")

    [record] = reconstruct_medication_ade_records(case["text"], case["spans"])
    payload = record.to_dict()

    assert record.regimen is not None
    assert set(record.regimen.dosage) == {"dose", "frequency"}
    assert [relation.tail.text for relation in record.adverse_events] == ["cough"]
    assert [relation.tail.text for relation in record.reasons] == ["hypertension"]
    assert payload["regimen"]["record_type"] == "MedicationStatement"
    assert payload["disclaimer_flag"] is True


def test_output_is_deterministic_and_fixture_is_specialized() -> None:
    case = _cases()[-1]

    first = extract_ade_relations(case["text"], case["spans"])
    second = extract_ade_relations(case["text"], reversed(case["spans"]))

    assert first == second
    assert FIXTURE not in list_fixture_paths()
