"""Tests for bounded document-level relation extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openmed.clinical.relations import (
    RelationCandidateRule,
    build_relation_candidates,
    extract_document_relations,
)
from openmed.core.decoding import EdgeCardinality
from openmed.eval.golden import list_fixture_paths
from openmed.eval.metrics import compute_document_level_relation_metrics

FIXTURE_PATH = (
    Path(__file__).parents[3] / "openmed/eval/golden/fixtures/doclevel_relations.jsonl"
)
_SYNTHETIC_HASH_SECRET = "synthetic-document-relation-secret"


def test_candidate_generation_uses_a_bounded_document_sentence_window() -> None:
    case = _cases()[0]
    rule = _rules(case)

    local = build_relation_candidates(
        case["text"],
        case["spans"],
        rule,
        language="en",
    )
    document = build_relation_candidates(
        case["text"],
        case["spans"],
        rule,
        language="en",
        max_sentence_distance=case["max_sentence_distance"],
    )

    assert local.candidates == ()
    assert len(document.candidates) == 3
    assert {edge.metadata["sentence_distance"] for edge in document.candidates} == {
        1,
        2,
        3,
    }
    assert all(edge.metadata["cross_sentence"] for edge in document.candidates)
    assert all(
        edge.metadata["evidence_sentence_offsets"] for edge in document.candidates
    )


def test_document_relations_aggregate_mentions_and_minimal_evidence() -> None:
    case = _cases()[0]

    relations = _predicted(case)

    assert len(relations) == 1
    relation = relations[0]
    assert len(relation.mention_pairs) == 3
    assert relation.is_cross_sentence
    assert relation.evidence_sentence_offsets == (
        (0, 23),
        (52, 81),
        (82, 102),
        (103, 123),
    )
    assert (24, 51) not in relation.evidence_sentence_offsets
    payload = json.dumps(relation.to_dict()).casefold()
    assert "lisinopril" not in payload
    assert "cough" not in payload
    assert relation.head.text_hash.startswith("hmac-sha256:")
    assert relation.head_entity_id.startswith("hmac-sha256:")


def test_document_wide_cardinality_keeps_one_dose_edge() -> None:
    case = _case("docrel-document-cardinality")

    relations = _predicted(case)

    assert len(relations) == 1
    assert relations[0].relation_type == "drug_to_dose"
    assert relations[0].tail.offset == (34, 40)
    assert all(pair.tail.offset != (64, 71) for pair in relations[0].mention_pairs)


def test_same_type_distractor_without_relation_cue_is_not_emitted() -> None:
    case = _case("docrel-distractor-only")

    assert _predicted(case) == ()


def test_committed_document_relation_gold_meets_f1_and_recall_slices() -> None:
    gold: list[dict[str, Any]] = []
    predicted: list[dict[str, Any]] = []
    for case in _cases():
        assert case["synthetic"] is True
        assert "not copied from a clinical corpus" in case["provenance"]
        gold.extend(
            {**relation, "fixture_id": case["case_id"]}
            for relation in case["gold_relations"]
        )
        predicted.extend(
            {**relation.to_dict(), "fixture_id": case["case_id"]}
            for relation in _predicted(case)
        )

    metrics = compute_document_level_relation_metrics(gold, predicted)

    assert metrics.f1 >= 0.65
    assert metrics.intra_sentence_recall == 1.0
    assert metrics.cross_sentence_recall == 1.0
    assert metrics.intra_sentence_gold == 1
    assert metrics.cross_sentence_gold == 2


def test_document_relation_gold_is_not_loaded_as_deidentification_gold() -> None:
    assert FIXTURE_PATH.resolve() not in {
        path.resolve() for path in list_fixture_paths()
    }


def _cases() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in _cases() if case["case_id"] == case_id)


def _rules(case: dict[str, Any]) -> tuple[RelationCandidateRule, ...]:
    return tuple(
        RelationCandidateRule(
            relation_type=rule["relation_type"],
            source_relation=rule["source_relation"],
            head_labels=frozenset(rule["head_labels"]),
            tail_labels=frozenset(rule["tail_labels"]),
            cues=tuple(rule["cues"]),
            max_character_distance=rule["max_character_distance"],
        )
        for rule in case["rules"]
    )


def _cardinality(case: dict[str, Any]) -> dict[str, EdgeCardinality]:
    return {
        relation_type: EdgeCardinality(**values)
        for relation_type, values in case["cardinality"].items()
    }


def _predicted(case: dict[str, Any]):
    return extract_document_relations(
        case["text"],
        case["spans"],
        _rules(case),
        document_id=case["document_id"],
        max_sentence_distance=case["max_sentence_distance"],
        cardinality=_cardinality(case),
        hash_secret=_SYNTHETIC_HASH_SECRET,
    )
