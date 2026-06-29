"""Tests for document-level clinical mention coreference."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from openmed.clinical import (
    COMPATIBILITY_SCORER_VERSION,
    COMPATIBILITY_WEIGHTS,
    COREFERENCE_ADVISORY,
    FAMILY_EXPERIENCER,
    PATIENT_EXPERIENCER,
    bcubed_precision_recall_f1,
    canonicalize_mentions,
    deduplicate_problem_list,
    link_mentions,
)

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/clinical/coref_gold.json"


def _load_gold_corpus() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text())["documents"]


def _all_mentions() -> list[dict[str, Any]]:
    mentions = []
    for document in _load_gold_corpus():
        for mention in document["mentions"]:
            mentions.append({**mention, "document_id": document["doc_id"]})
    return mentions


def _predicted_labels(
    mentions: list[dict[str, Any]],
) -> dict[tuple[str, int, int], str]:
    result = link_mentions(mentions)
    labels: dict[tuple[str, int, int], str] = {}
    for cluster in result.clusters:
        for member in cluster.members:
            labels[(member.document_id, member.start, member.end)] = cluster.entity_id
    return labels


def _gold_labels(mentions: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    return {
        (mention["document_id"], mention["start"], mention["end"]): mention[
            "gold_entity"
        ]
        for mention in mentions
    }


def test_canonicalize_mentions_expands_abbreviations_and_preserves_offsets() -> None:
    mentions = canonicalize_mentions(
        [
            {
                "text": "the patient's diabetes",
                "start": 0,
                "end": 22,
                "semantic_type": "condition",
            },
            {
                "text": "DM",
                "start": 40,
                "end": 42,
                "semantic_type": "condition",
            },
        ]
    )

    assert [mention.canonical_text for mention in mentions] == [
        "diabetes mellitus",
        "diabetes mellitus",
    ]
    assert [mention.offset for mention in mentions] == [(0, 22), (40, 42)]


def test_public_link_mentions_exposes_disclaimer_and_member_provenance() -> None:
    mentions = _all_mentions()[:4]

    result = link_mentions(mentions)
    patient_diabetes = next(
        cluster for cluster in result.clusters if len(cluster.members) == 3
    )

    assert result.advisory == COREFERENCE_ADVISORY
    assert "not a clinical decision" in result.advisory
    assert patient_diabetes.advisory == COREFERENCE_ADVISORY
    assert patient_diabetes.representative == "diabetes mellitus"
    assert patient_diabetes.member_offsets == ((28, 45), (51, 57), (84, 86))
    assert patient_diabetes.members[1].text == "sugars"
    assert patient_diabetes.entity_id.startswith("doc-coref-1:entity:")


def test_bcubed_gold_corpus_meets_ci_gate_and_never_violates_cannot_link() -> None:
    mentions = _all_mentions()

    predicted = _predicted_labels(mentions)
    gold = _gold_labels(mentions)
    metric = bcubed_precision_recall_f1(predicted, gold)

    assert metric.f1 >= 0.80
    result = link_mentions(mentions)
    for cluster in result.clusters:
        experiencers = {member.experiencer for member in cluster.members}
        assert {PATIENT_EXPERIENCER, FAMILY_EXPERIENCER} != experiencers


def test_clustering_is_order_invariant_after_canonicalization() -> None:
    mentions = _all_mentions()
    expected = _predicted_labels(mentions)
    variants = [
        list(reversed(mentions)),
        mentions[3:] + mentions[:3],
        sorted(mentions, key=lambda mention: (mention["text"], mention["start"])),
    ]

    for variant in variants:
        assert _predicted_labels(variant) == expected


def test_feature_weights_are_versioned_and_visible() -> None:
    assert COMPATIBILITY_SCORER_VERSION == "clinical-coref-compat-v1"
    assert set(COMPATIBILITY_WEIGHTS) == {
        "string_similarity",
        "semantic_type",
        "section_temporality",
        "distance",
        "code",
    }
    assert round(sum(COMPATIBILITY_WEIGHTS.values()), 6) == 1.0


def test_problem_list_coref_dedup_reduces_duplicates_without_false_merges() -> None:
    mentions = _all_mentions()
    result = link_mentions(mentions)
    entity_ids_by_offset = result.entity_ids_by_offset()
    with_coref = [
        {
            **mention,
            "coref_entity_id": entity_ids_by_offset[
                (mention["document_id"], (mention["start"], mention["end"]))
            ],
            "offset": (mention["start"], mention["end"]),
        }
        for mention in mentions
    ]
    without_coref = [
        {
            **mention,
            "offset": (mention["start"], mention["end"]),
        }
        for mention in mentions
    ]

    baseline = deduplicate_problem_list(without_coref)
    deduplicated = deduplicate_problem_list(with_coref)

    assert len(deduplicated) < len(baseline)
    assert _problem_list_pairwise_precision(deduplicated, mentions) >= 0.90


def _problem_list_pairwise_precision(
    deduplicated: list[Any],
    mentions: list[dict[str, Any]],
) -> float:
    gold_by_offset = {
        (mention["start"], mention["end"]): mention["gold_entity"]
        for mention in mentions
    }
    true_positive = 0
    false_positive = 0
    for problem in deduplicated:
        for left, right in combinations(problem.source_offsets, 2):
            if gold_by_offset[left] == gold_by_offset[right]:
                true_positive += 1
            else:
                false_positive += 1
    if true_positive + false_positive == 0:
        return 1.0
    return true_positive / (true_positive + false_positive)
