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
    event_coreference_mentions,
    link_mentions,
    resolve_coreference,
)
from openmed.eval.metrics import compute_coreference_clustering_score

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/clinical/coref_gold.json"
EVENT_FIXTURE_PATH = (
    Path(__file__).parents[3] / "openmed/eval/golden/fixtures/event_coref.jsonl"
)


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


def _load_event_coref_gold() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in EVENT_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_seed_mentions(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**mention, "document_id": document["doc_id"]}
        for mention in document["seed_mentions"]
    ]


def _event_gold_labels(document: dict[str, Any]) -> dict[tuple[str, int, int], str]:
    return {
        (document["doc_id"], mention["start"], mention["end"]): mention["gold_entity"]
        for mention in document["gold_mentions"]
    }


def _event_predicted_labels(
    document: dict[str, Any],
) -> dict[tuple[str, int, int], str]:
    result = resolve_coreference(
        _event_seed_mentions(document),
        document_text=document["text"],
        document_id=document["doc_id"],
        hash_secret="synthetic-fixture-secret",
    )
    return {
        (cluster.document_id, start, end): cluster.cluster_id
        for cluster in result.clusters
        for start, end in cluster.member_offsets
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


def test_event_candidates_cover_all_event_types_and_cross_section_anaphora() -> None:
    documents = {item["doc_id"]: item for item in _load_event_coref_gold()}
    cross_section = documents["event-coref-cross-section"]

    candidates = event_coreference_mentions(
        cross_section["text"],
        _event_seed_mentions(cross_section),
        document_id=cross_section["doc_id"],
    )
    canonical = canonicalize_mentions(candidates, document_text=cross_section["text"])
    by_offset = {(mention.start, mention.end): mention for mention in canonical}

    assert set(by_offset) == {(11, 27), (41, 43), (57, 67), (80, 82)}
    assert by_offset[(11, 27)].semantic_type == "problem"
    assert by_offset[(41, 43)].semantic_type == "test"
    assert by_offset[(57, 67)].canonical_text == "left lung lesion"
    assert by_offset[(57, 67)].canonical_section == "assessment"
    assert by_offset[(80, 82)].canonical_text == "left lung lesion"

    treatment = documents["event-coref-treatment"]
    treatment_candidates = event_coreference_mentions(
        treatment["text"],
        _event_seed_mentions(treatment),
        document_id=treatment["doc_id"],
    )
    assert {candidate["semantic_type"] for candidate in treatment_candidates} == {
        "treatment"
    }


def test_resolve_coreference_meets_event_gold_gate_with_sanitized_output() -> None:
    predicted: dict[tuple[str, int, int], str] = {}
    gold: dict[tuple[str, int, int], str] = {}
    documents = _load_event_coref_gold()
    for document in documents:
        document_predicted = _event_predicted_labels(document)
        document_gold = _event_gold_labels(document)
        assert set(document_gold).issubset(document_predicted)
        predicted.update({key: document_predicted[key] for key in document_gold})
        gold.update(document_gold)

    score = compute_coreference_clustering_score(predicted, gold)

    assert score.metric == "bcubed"
    assert score.f1 >= 0.60

    document = next(
        item for item in documents if item["doc_id"] == "event-coref-treatment"
    )
    result = resolve_coreference(
        _event_seed_mentions(document),
        document_text=document["text"],
        document_id=document["doc_id"],
        hash_secret="synthetic-fixture-secret",
    )
    payload = json.dumps(result.to_dict()).casefold()
    assert "aspirin" not in payload
    assert "the medication" not in payload
    assert all(
        member_hash.startswith("hmac-sha256:")
        for cluster in result.clusters
        for member_hash in cluster.member_hashes
    )
    assert all(
        cluster.canonical_hash.startswith("hmac-sha256:") for cluster in result.clusters
    )


def test_event_coreference_keeps_opposite_assertion_polarities_apart() -> None:
    document = next(
        item
        for item in _load_event_coref_gold()
        if item["doc_id"] == "event-coref-polarity"
    )
    predicted = _event_predicted_labels(document)

    assert (
        predicted[(document["doc_id"], 15, 24)]
        != predicted[(document["doc_id"], 41, 50)]
    )


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
