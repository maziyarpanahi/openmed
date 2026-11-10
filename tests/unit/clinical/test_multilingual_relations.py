"""Tests for Chinese and Indic clinical relation extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from openmed.clinical import (
    CMEIE_RELATION_MAPPING,
    INDIC_RELATION_MAPPING,
    MULTILINGUAL_RELATION_REGISTRY_VERSION,
    build_relation_candidates,
    extract_relations,
    multilingual_relation_rules,
)
from openmed.clinical.relations.assertion_filter import (
    RELATION_CONFIRMED,
    RELATION_REFUTED,
)
from openmed.core.decoding import decode_span_graph
from openmed.eval.suites.relations import (
    DEFAULT_MULTILINGUAL_RELATION_GOLD_PATHS,
    load_multilingual_relation_fixtures,
    score_relation_fixtures,
)

STRICT_F1_FLOOR = 0.60
FORBIDDEN_FIXTURE_MARKERS = ("mimic", "i2b2", "n2c2", "snomed", "umls")


def test_versioned_registry_maps_full_cmeie_and_indic_subset() -> None:
    assert MULTILINGUAL_RELATION_REGISTRY_VERSION == 1
    assert len(CMEIE_RELATION_MAPPING) == 44
    assert CMEIE_RELATION_MAPPING["药物治疗"] == "drug_treatment"
    assert CMEIE_RELATION_MAPPING["手术治疗"] == "surgical_treatment"
    assert CMEIE_RELATION_MAPPING["预防"] == "prevention"
    assert INDIC_RELATION_MAPPING["दवा उपचार"] == "drug_treatment"
    assert INDIC_RELATION_MAPPING["शल्य उपचार"] == "surgical_treatment"


def test_chinese_candidate_generation_preserves_character_offsets() -> None:
    text = "肺炎使用阿莫西林治疗。"
    spans = [
        {"text": "肺炎", "label": "CONDITION", "start": 0, "end": 2},
        {"text": "阿莫西林", "label": "MEDICATION", "start": 4, "end": 8},
    ]

    batch = build_relation_candidates(
        text,
        spans,
        multilingual_relation_rules("zh"),
        language="zh",
    )

    assert [(node.start, node.end) for node in batch.nodes] == [(0, 2), (4, 8)]
    candidate = next(
        edge for edge in batch.candidates if edge.label == "drug_treatment"
    )
    assert batch.spans_by_node_id[candidate.head].text == "肺炎"
    assert batch.spans_by_node_id[candidate.tail].text == "阿莫西林"
    assert candidate.metadata["character_distance"] == 2


def test_extract_relations_reuses_shared_graph_decoder() -> None:
    text = "阑尾炎需要手术切除。"
    spans = [
        {"label": "CONDITION", "start": 0, "end": 3},
        {"label": "PROCEDURE", "start": 5, "end": 9},
    ]

    with patch(
        "openmed.clinical.relations.multilingual.decode_span_graph",
        wraps=decode_span_graph,
    ) as decoder:
        relations = extract_relations(text, spans, language="zh")

    decoder.assert_called_once()
    assert [relation.relation_type for relation in relations] == ["surgical_treatment"]


def test_non_english_context_does_not_apply_missing_english_cue() -> None:
    text = "编号no1：肺炎使用阿莫西林治疗。"
    spans = [
        {"label": "CONDITION", "start": 6, "end": 8},
        {"label": "MEDICATION", "start": 10, "end": 14},
    ]

    relations = extract_relations(text, spans, language="zh")

    assert len(relations) == 1
    assert relations[0].assertion_status == RELATION_CONFIRMED


def test_chinese_negation_is_propagated_without_false_fact() -> None:
    text = "未见肺炎，使用阿莫西林治疗。"
    spans = [
        {"label": "CONDITION", "start": 2, "end": 4},
        {"label": "MEDICATION", "start": 7, "end": 11},
    ]

    assert extract_relations(text, spans, language="zh") == ()
    retained = extract_relations(
        text,
        spans,
        language="zh",
        asserted_only=False,
    )
    assert retained[0].assertion_status == RELATION_REFUTED


def test_committed_multilingual_gold_exceeds_per_language_strict_f1_floor() -> None:
    fixtures = load_multilingual_relation_fixtures()
    predictions = {
        fixture.fixture_id: list(
            extract_relations(
                fixture.text,
                fixture.entities.values(),
                language=fixture.language,
            )
        )
        for fixture in fixtures
    }

    scorecard = score_relation_fixtures(fixtures, predictions)

    assert scorecard["metadata"]["languages"] == ["hi", "zh"]
    for language in ("hi", "zh"):
        assert (
            scorecard["metrics"]["by_language"][language]["strict"]["f1"]
            >= STRICT_F1_FLOOR
        )


def test_multilingual_relation_fixtures_are_synthetic_only() -> None:
    fixtures = load_multilingual_relation_fixtures()

    assert {fixture.language for fixture in fixtures} == {"hi", "zh"}
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    fixture_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in DEFAULT_MULTILINGUAL_RELATION_GOLD_PATHS
    ).casefold()
    assert all(marker not in fixture_text for marker in FORBIDDEN_FIXTURE_MARKERS)
