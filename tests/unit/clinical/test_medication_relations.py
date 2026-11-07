"""Tests for deterministic medication attribute relation linking."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import (
    MEDICATION_LINK_ADVISORY,
    CoreferenceChain,
    MedicationRelationScorer,
    extract_medication_relations,
    extract_relations,
    link_medication_attributes,
    reconstruct_medication_statements,
)
from openmed.clinical.sections import detect_sections
from openmed.core.schemas import OpenMedSpan, hmac_text_hash

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "medication_relations_gold.json"
)
MICRO_F1_THRESHOLD = 0.85
COREFERENCE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "medication_coreference_collapse.json"
)
_SYNTHETIC_HASH_SECRET = "synthetic-medication-coreference-secret"


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

    assert scorer.config["version"] == 2
    assert "drug_to_frequency" in scorer.config["relations"]


def test_extract_medication_relations_binds_canonical_regimen_attributes() -> None:
    text = "lisinopril 10 mg PO daily"
    spans = [
        _span(text, "lisinopril", "MEDICATION"),
        _span(text, "10 mg", "DOSAGE"),
        _span(text, "PO", "ROUTE"),
        _span(text, "daily", "FREQUENCY"),
    ]

    relations = extract_medication_relations(text, spans)

    assert [(relation.type, relation.tail.text) for relation in relations] == [
        ("dose", "10 mg"),
        ("route", "PO"),
        ("frequency", "daily"),
    ]
    assert all(relation.head.text == "lisinopril" for relation in relations)
    assert all(0 < relation.score <= 1 for relation in relations)


def test_extract_relations_defaults_to_medication_regimen_relations() -> None:
    text = "lisinopril 10 mg PO daily"
    spans = [
        _span(text, "lisinopril", "MEDICATION"),
        _span(text, "10 mg", "DOSAGE"),
        _span(text, "PO", "ROUTE"),
        _span(text, "daily", "FREQUENCY"),
    ]

    relations = extract_relations(text, spans)
    statements = reconstruct_medication_statements(relations)

    assert [(relation.type, relation.tail.text) for relation in relations] == [
        ("dose", "10 mg"),
        ("route", "PO"),
        ("frequency", "daily"),
    ]
    assert len(statements) == 1
    assert statements[0].record_type == "MedicationStatement"
    assert statements[0].medication.text == "lisinopril"
    assert set(statements[0].dosage) == {"dose", "route", "frequency"}


def test_extract_medication_relations_supports_every_attribute_type() -> None:
    cases = (
        ("1 tablet", "DOSAGE", "dose"),
        ("PO", "ROUTE", "route"),
        ("daily", "FREQUENCY", "frequency"),
        ("for 5 days", "DURATION", "duration"),
        ("tablet", "FORM", "form"),
        ("10 mg", "STRENGTH", "strength"),
        ("hypertension", "INDICATION", "indication"),
    )

    for tail, label, relation_type in cases:
        text = f"Lisinopril {tail}"
        relations = extract_medication_relations(
            text,
            [
                _span(text, "Lisinopril", "MEDICATION"),
                _span(text, tail, label),
            ],
        )

        assert [(relation.type, relation.tail.text) for relation in relations] == [
            (relation_type, tail)
        ]


def test_multi_medication_line_keeps_each_regimen_separate() -> None:
    text = "aspirin 81 mg daily; metformin 500 mg BID"
    spans = [
        _span(text, "aspirin", "MEDICATION"),
        _span(text, "81 mg", "DOSAGE"),
        _span(text, "daily", "FREQUENCY"),
        _span(text, "metformin", "MEDICATION"),
        _span(text, "500 mg", "DOSAGE"),
        _span(text, "BID", "FREQUENCY"),
    ]

    relations = extract_medication_relations(text, spans)

    assert {
        (relation.head.text, relation.type, relation.tail.text)
        for relation in relations
    } == {
        ("aspirin", "dose", "81 mg"),
        ("aspirin", "frequency", "daily"),
        ("metformin", "dose", "500 mg"),
        ("metformin", "frequency", "BID"),
    }


def test_attribute_outside_medication_clause_and_section_is_dropped() -> None:
    text = "MEDICATIONS:\nLisinopril\nA/P:\nDaily"
    spans = [
        _span(text, "Lisinopril", "MEDICATION"),
        _span(text, "Daily", "FREQUENCY"),
    ]

    relations = extract_medication_relations(
        text,
        spans,
        sections=detect_sections(text),
    )

    assert relations == ()


def test_relations_reconstruct_one_statement_record_per_regimen() -> None:
    text = "aspirin 81 mg daily for headache; metformin 500 mg BID"
    spans = [
        _span(text, "aspirin", "MEDICATION"),
        _span(text, "81 mg", "DOSAGE"),
        _span(text, "daily", "FREQUENCY"),
        _span(text, "headache", "INDICATION"),
        _span(text, "metformin", "MEDICATION"),
        _span(text, "500 mg", "DOSAGE"),
        _span(text, "BID", "FREQUENCY"),
    ]

    statements = reconstruct_medication_statements(
        extract_medication_relations(text, spans)
    )

    assert [statement.medication.text for statement in statements] == [
        "aspirin",
        "metformin",
    ]
    assert statements[0].record_type == "MedicationStatement"
    assert statements[0].dosage["dose"].text == "81 mg"
    assert statements[0].dosage["frequency"].text == "daily"
    assert statements[0].indication is not None
    assert statements[0].indication.text == "headache"
    assert statements[1].dosage["dose"].text == "500 mg"
    assert statements[1].dosage["frequency"].text == "BID"
    assert statements[0].to_dict()["record_type"] == "MedicationStatement"


def test_coreferent_medication_heads_collapse_with_safe_supporting_evidence() -> None:
    case = json.loads(COREFERENCE_FIXTURE.read_text(encoding="utf-8"))
    chain = _coreference_chain(case)

    groups = link_medication_attributes(
        case["text"],
        case["relation_spans"],
        coreference_chains=(chain,),
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.medication.offset_key() == (
        chain.representative.start,
        chain.representative.end,
    )
    assert [relation.attribute_type for relation in group.relations] == ["dose"]
    assert all(relation.head == group.medication for relation in group.relations)
    assert group.coreference is not None
    assert group.coreference.cluster_id == chain.chain_id
    assert [
        (mention.start, mention.end, mention.text_hash)
        for mention in group.coreference.supporting_mentions
    ] == [(member.start, member.end, member.text_hash) for member in chain.members]
    assert all(
        relation.coreference == group.coreference for relation in group.relations
    )

    provenance_values = {
        value.casefold() for value in _string_values(group.coreference.to_dict())
    }
    for mention in case["mentions"]:
        assert mention["surface"].casefold() not in provenance_values


def _coreference_chain(case: dict) -> CoreferenceChain:
    members = tuple(
        OpenMedSpan(
            doc_id=case["document_id"],
            start=mention["start"],
            end=mention["end"],
            text_hash=hmac_text_hash(
                case["text"][mention["start"] : mention["end"]],
                _SYNTHETIC_HASH_SECRET,
            ),
            entity_type="DRUG",
            canonical_label="MEDICATION",
        )
        for mention in case["mentions"]
    )
    return CoreferenceChain(
        chain_id="coref-synthetic-medication",
        members=members,
        representative=members[case["representative_index"]],
        confidence=0.98,
    )


def _span(text: str, surface: str, label: str) -> dict[str, object]:
    start = text.index(surface)
    return {
        "text": surface,
        "label": label,
        "start": start,
        "end": start + len(surface),
        "score": 1.0,
    }


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _string_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _string_values(child)]
    return []


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
