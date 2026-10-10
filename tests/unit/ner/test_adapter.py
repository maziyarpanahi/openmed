from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import List

import pytest

from openmed.ner.adapter import to_token_classification
from openmed.ner.infer import Entity


def make_entity(
    text: str,
    start: int,
    end: int,
    label: str,
    score: float = 0.9,
    group: str | None = None,
) -> Entity:
    return Entity(
        text=text, start=start, end=end, label=label, score=score, group=group
    )


def test_to_token_classification_bio_scheme() -> None:
    entities = [
        make_entity("Aspirin", 0, 7, "Drug"),
        make_entity("fever", 15, 20, "Disease"),
    ]
    text = "Aspirin treats fever."
    result = to_token_classification(entities, text, scheme="BIO")
    labels = result.labels()
    assert labels[0] == "B-Drug"
    assert "I-Drug" not in labels
    assert any(
        label.startswith("B-Disease") or label.startswith("I-Disease")
        for label in labels
    )


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


class _OffsetRows(Mapping):
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


@pytest.mark.parametrize("container", [dict, UserDict, MappingProxyType, _OffsetRows])
@pytest.mark.parametrize(
    "scheme,labels", [("BIO", ["B-TEST", "I-TEST"]), ("BILOU", ["B-TEST", "L-TEST"])]
)
def test_mapping_outputs_preserve_subwords_labels_and_exact_offsets(
    container, scheme, labels
):
    text = "synthetic"
    encoded = container({"offset_mapping": [(0, 3), (3, 9)]})
    result = to_token_classification(
        [make_entity(text, 0, len(text), "TEST")],
        text,
        tokenizer=lambda *_args, **_kwargs: encoded,
        scheme=scheme,
    )
    assert [(token.token, token.start, token.end) for token in result.tokens] == [
        ("syn", 0, 3),
        ("thetic", 3, 9),
    ]
    assert result.labels() == labels


@pytest.mark.parametrize("wrapped", [False, True])
def test_mapping_offsets_work_with_positional_tokenizers_and_wrappers(wrapped):
    def tokenizer(text):
        return UserDict(
            {"offset_mapping": [(0, 0), (0, 1), (1, len(text)), (None, None)]}
        )

    supplied = (
        SimpleNamespace(get_tokenizer=lambda: tokenizer) if wrapped else tokenizer
    )
    result = to_token_classification([], "abc", tokenizer=supplied)
    assert [(token.token, token.start, token.end) for token in result.tokens] == [
        ("a", 0, 1),
        ("bc", 1, 3),
    ]
    assert result.labels() == ["O", "O"]


@pytest.mark.parametrize(
    "encoded",
    [
        None,
        {},
        UserDict(),
        UserDict({"offset_mapping": []}),
        MappingProxyType({"offset_mapping": None}),
    ],
)
def test_missing_mapping_offsets_retain_whitespace_fallback(encoded):
    result = to_token_classification([], "ab cd", tokenizer=lambda *_a, **_k: encoded)
    assert [(token.token, token.start, token.end) for token in result.tokens] == [
        ("ab", 0, 2),
        ("cd", 3, 5),
    ]


def test_non_mapping_output_and_no_tokenizer_retain_fallback():
    for tokenizer in (None, lambda *_a, **_k: SimpleNamespace(offset_mapping=[(0, 1)])):
        result = to_token_classification([], "ab cd", tokenizer=tokenizer)
        assert [token.token for token in result.tokens] == ["ab", "cd"]
