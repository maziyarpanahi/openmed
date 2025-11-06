from __future__ import annotations

from typing import List

from openmed.ner.adapter import to_token_classification
from openmed.ner.infer import Entity


def make_entity(text: str, start: int, end: int, label: str, score: float = 0.9, group: str | None = None) -> Entity:
    return Entity(text=text, start=start, end=end, label=label, score=score, group=group)


def test_to_token_classification_bio_scheme() -> None:
    entities = [make_entity("Aspirin", 0, 7, "Drug"), make_entity("fever", 15, 20, "Disease")]
    text = "Aspirin treats fever."
    result = to_token_classification(entities, text, scheme="BIO")
    labels = result.labels()
    assert labels[0] == "B-Drug"
    assert "I-Drug" not in labels
    assert any(label.startswith("B-Disease") or label.startswith("I-Disease") for label in labels)


def test_to_token_classification_bilou_scheme() -> None:
    entities = [make_entity("New York", 0, 8, "Location", group="1")]
    text = "New York City"
    result = to_token_classification(entities, text, scheme="BILOU")
    labels = result.labels()
    assert labels[0] == "B-Location"
    assert labels[1] == "L-Location" or labels[1] == "I-Location"
    assert result.metadata["groups"]["1"]


def test_to_token_classification_handles_overlaps_preferring_high_score() -> None:
    entities = [
        make_entity("OpenMed", 0, 7, "Org", score=0.7),
        make_entity("OpenMed", 0, 7, "Product", score=0.5),
    ]
    text = "OpenMed launched."
    result = to_token_classification(entities, text)
    assert result.labels()[0].startswith("B-Org")
