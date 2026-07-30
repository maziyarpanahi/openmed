"""Tests for deterministic medication attribute relation linking."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import (
    MEDICATION_LINK_ADVISORY,
    MedicationRelationScorer,
    link_medication_attributes,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "medication_relations_gold.json"
)
MICRO_F1_THRESHOLD = 0.85


def test_link_medication_attributes_is_byte_deterministic_for_gold_corpus() -> None:
    corpus = _load_corpus()
    baseline = _corpus_bytes(corpus)

    for _ in range(100):
        assert _corpus_bytes(corpus) == baseline


def test_relation_eval_harness_meets_micro_f1_threshold() -> None:
    corpus = _load_corpus()
    predicted: set[tuple[str, int, int, int, int, str]] = set()
    gold: set[tuple[str, int, int, int, int, str]] = set()
    per_case_scores = []

    for case in corpus:
        case_predicted = _predicted_relations(case)
        case_gold = _gold_relations(case)
        predicted.update((case["id"], *relation) for relation in case_predicted)
        gold.update((case["id"], *relation) for relation in case_gold)
        per_case_scores.append(_f1(case_predicted, case_gold))

    micro_f1 = _f1(predicted, gold)
    macro_f1 = sum(per_case_scores) / len(per_case_scores)

    assert micro_f1 >= MICRO_F1_THRESHOLD
    assert macro_f1 >= MICRO_F1_THRESHOLD


def test_cardinality_constraints_hold_on_gold_corpus() -> None:
    for case in _load_corpus():
        groups = link_medication_attributes(case["text"], case["spans"])
        attribute_to_heads: dict[tuple[str, int, int], set[tuple[int, int]]] = {}
        drug_frequency_counts: dict[tuple[int, int], int] = {}

        for group in groups:
            for relation in group.relations:
                attribute_key = (
                    relation.relation_type,
                    relation.attribute.start,
                    relation.attribute.end,
                )
                attribute_to_heads.setdefault(attribute_key, set()).add(
                    relation.head.offset_key()
                )
                if relation.relation_type == "drug_to_frequency":
                    drug_frequency_counts[relation.head.offset_key()] = (
                        drug_frequency_counts.get(relation.head.offset_key(), 0) + 1
                    )

        assert all(len(heads) == 1 for heads in attribute_to_heads.values())
        assert all(count <= 1 for count in drug_frequency_counts.values())


def test_emitted_relation_offsets_round_trip_to_source_text() -> None:
    for case in _load_corpus():
        for group in link_medication_attributes(case["text"], case["spans"]):
            for relation in group.relations:
                assert (
                    case["text"][relation.head.start : relation.head.end]
                    == relation.head.text
                )
                assert (
                    case["text"][relation.attribute.start : relation.attribute.end]
                    == relation.attribute.text
                )


def test_frequency_and_duration_relations_carry_normalized_outputs() -> None:
    case = _load_corpus()[0]
    groups = link_medication_attributes(case["text"], case["spans"])
    metformin = next(group for group in groups if group.medication.text == "metformin")
    attributes = metformin.attributes

    assert attributes["frequency"].normalized["frequency_per_day"] == 2.0
    assert attributes["duration"].normalized["days"] == 30


def test_public_api_docstring_and_advisory_include_clinical_disclaimer() -> None:
    docstring = link_medication_attributes.__doc__ or ""

    assert "not a prescription decision" in docstring
    assert "not a substitute for clinician review" in docstring
    assert "MEDICATION_SIG_ADVISORY" in docstring
    assert "not a prescription decision" in MEDICATION_LINK_ADVISORY


def test_default_scorer_loads_versioned_config_resource() -> None:
    scorer = MedicationRelationScorer.from_default_config()

    assert scorer.config["version"] == 1
    assert "drug_to_frequency" in scorer.config["relations"]


def _load_corpus() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _corpus_bytes(corpus: list[dict]) -> str:
    payload = [
        [
            group.to_dict()
            for group in link_medication_attributes(case["text"], case["spans"])
        ]
        for case in corpus
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _predicted_relations(case: dict) -> set[tuple[int, int, int, int, str]]:
    relations = set()
    for group in link_medication_attributes(case["text"], case["spans"]):
        for relation in group.relations:
            relations.add(
                (
                    relation.head.start,
                    relation.head.end,
                    relation.attribute.start,
                    relation.attribute.end,
                    relation.relation_type,
                )
            )
    return relations


def _gold_relations(case: dict) -> set[tuple[int, int, int, int, str]]:
    span_by_id = {span["id"]: span for span in case["spans"]}
    relations = set()
    for relation in case["relations"]:
        head = span_by_id[relation["head"]]
        attribute = span_by_id[relation["attribute"]]
        relations.add(
            (
                head["start"],
                head["end"],
                attribute["start"],
                attribute["end"],
                relation["type"],
            )
        )
    return relations


def _f1(predicted: set, gold: set) -> float:
    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2 * precision * recall / (precision + recall)
