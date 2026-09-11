"""Tests for the public DrugProt joint relation-example builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.datasets import DRUGPROT, DRUGPROT_RELATION_TYPES, license_for
from openmed.training.data import (
    DRUGPROT_RELATION_LABELS,
    DRUGPROT_TO_RELATION_LABEL,
    build_drugprot_relation_examples,
    load_drugprot_relation_examples,
    map_drugprot_relation_type,
)

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "drugprot_synthetic" / "training"


def test_drugprot_relation_types_map_to_canonical_label_space() -> None:
    assert tuple(DRUGPROT_TO_RELATION_LABEL) == DRUGPROT_RELATION_TYPES
    assert DRUGPROT_RELATION_LABELS == (
        "activator",
        "agonist",
        "agonist_activator",
        "agonist_inhibitor",
        "antagonist",
        "direct_regulator",
        "indirect_downregulator",
        "indirect_upregulator",
        "inhibitor",
        "part_of",
        "product_of",
        "substrate",
        "substrate_product_of",
    )
    assert map_drugprot_relation_type(" agonist-activator ") == "agonist_activator"

    with pytest.raises(ValueError, match="unknown DrugProt relation type"):
        map_drugprot_relation_type("not-a-relation")


def test_positive_examples_match_joint_and_clinical_relation_shapes() -> None:
    examples = load_drugprot_relation_examples(
        FIXTURE_DIR,
        negatives_per_positive=0,
    )

    assert [example.relation_type for example in examples] == [
        "inhibitor",
        "activator",
    ]
    first = examples[0]
    assert first.text == "Aspirin inhibits TP53 Metformin activates EGFR"
    assert first.head is first.head_span
    assert first.tail is first.tail_span
    assert first.type == "inhibitor"
    assert first.head_span.to_dict() == {
        "text": "Aspirin",
        "label": "chemical",
        "start": 0,
        "end": 7,
        "score": 1.0,
    }
    assert first.tail_span.to_dict() == {
        "text": "TP53",
        "label": "gene",
        "start": 17,
        "end": 21,
        "score": 1.0,
    }
    assert first.to_relation_dict() == {
        "head": first.head_span.to_dict(),
        "type": "inhibitor",
        "tail": first.tail_span.to_dict(),
    }
    assert set(first.to_dict()) == {
        "text",
        "head_span",
        "tail_span",
        "relation_type",
        "metadata",
    }
    assert json.loads(json.dumps(first.to_dict()))["relation_type"] == "inhibitor"


def test_every_example_carries_public_zenodo_license_metadata() -> None:
    examples = load_drugprot_relation_examples(FIXTURE_DIR)
    expected_license = license_for(DRUGPROT).to_dict()

    assert examples
    for example in examples:
        assert example.metadata["dataset"] == DRUGPROT
        assert example.metadata["source"] == "Zenodo"
        assert example.metadata["public"] is True
        assert example.metadata["license"] == expected_license
        assert example.metadata["license"]["license_id"] == "CC-BY-4.0"
        assert example.metadata["license"]["source_url"].startswith(
            "https://zenodo.org/"
        )


def test_hard_negatives_are_deterministic_cooccurring_unrelated_pairs() -> None:
    examples = load_drugprot_relation_examples(
        FIXTURE_DIR,
        negatives_per_positive=1,
        seed=202,
    )
    repeated = load_drugprot_relation_examples(
        FIXTURE_DIR,
        negatives_per_positive=1,
        seed=202,
    )
    positives = [example for example in examples if not example.is_negative]
    negatives = [example for example in examples if example.is_negative]
    gold_pairs = {
        (example.metadata["head_entity_id"], example.metadata["tail_entity_id"])
        for example in positives
    }

    assert [example.to_dict() for example in examples] == [
        example.to_dict() for example in repeated
    ]
    assert len(negatives) == 2
    assert {
        (example.metadata["head_entity_id"], example.metadata["tail_entity_id"])
        for example in negatives
    } == {("T1", "T4"), ("T3", "T2")}
    assert all(example.relation_type is None for example in negatives)
    assert all(example.metadata["is_negative"] is True for example in negatives)
    assert all(example.metadata["source_pmid"] == "DP1" for example in negatives)
    assert all(example.head_span.label == "chemical" for example in negatives)
    assert all(example.tail_span.label == "gene" for example in negatives)
    assert all(
        (
            example.metadata["head_entity_id"],
            example.metadata["tail_entity_id"],
        )
        not in gold_pairs
        for example in negatives
    )
    assert all(
        example.text[example.head_span.start : example.head_span.end]
        == example.head_span.text
        and example.text[example.tail_span.start : example.tail_span.end]
        == example.tail_span.text
        for example in negatives
    )


def test_builder_can_consume_the_loaded_adapter_corpus() -> None:
    from openmed.eval.datasets import load_drugprot_corpus

    corpus = load_drugprot_corpus(FIXTURE_DIR)

    assert build_drugprot_relation_examples(corpus) == (
        load_drugprot_relation_examples(FIXTURE_DIR)
    )
