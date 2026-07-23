"""Tests for the optional offset-preserving Indic NER adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openmed.core.labels import LOCATION, ORGANIZATION, PERSON, normalize_label
from openmed.ner.families.indic import (
    IndicNerAdapter,
    IndicNerWeightsUnavailable,
    load_indic_ner_adapter,
)


class _Tokenizer:
    is_fast = True

    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, text, **kwargs):
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["return_tensors"] == "pt"
        return {
            "attention_mask": [[1] * len(self.offsets)],
            "input_ids": [[0] * len(self.offsets)],
            "offset_mapping": [self.offsets],
        }


class _Model:
    config = SimpleNamespace(
        id2label={
            0: "O",
            1: "B-PER",
            2: "B-LOC",
            3: "B-ORG",
            4: "I-ORG",
        }
    )

    def __init__(self, label_ids):
        self.label_ids = label_ids

    def __call__(self, **kwargs):
        assert "offset_mapping" not in kwargs
        logits = []
        for label_id in self.label_ids:
            row = [-5.0] * 5
            row[label_id] = 5.0
            logits.append(row)
        return SimpleNamespace(logits=[logits])


def test_conll_aliases_map_to_canonical_labels():
    assert normalize_label("B-PER") == PERSON
    assert normalize_label("I-LOC") == LOCATION
    assert normalize_label("S-ORG") == ORGANIZATION


def test_adapter_preserves_offsets_and_never_returns_surface_text():
    text = "आरव दिल्ली अपोलो अस्पताल"
    surfaces = ["आरव", "दिल्ली", "अपोलो", "अस्पताल"]
    offsets = (
        [(0, 0)]
        + [
            (text.index(surface), text.index(surface) + len(surface))
            for surface in surfaces
        ]
        + [(0, 0)]
    )
    adapter = IndicNerAdapter(
        model_id="/models/indic",
        tokenizer=_Tokenizer(offsets),
        model=_Model([0, 1, 2, 3, 4, 0]),
    )

    predictions = adapter.predict(text)

    assert [(row.start, row.end, row.label) for row in predictions] == [
        (0, 3, PERSON),
        (4, 10, LOCATION),
        (11, len(text), ORGANIZATION),
    ]
    serialized = [row.to_dict() for row in predictions]
    assert all("text" not in row for row in serialized)
    assert all(
        text[row.start : row.end] not in str(row.to_dict()) for row in predictions
    )


def test_loader_requires_explicit_weights_before_importing_dependencies(monkeypatch):
    monkeypatch.delenv("OPENMED_INDIC_NER_MODEL", raising=False)

    with pytest.raises(IndicNerWeightsUnavailable, match="is not configured"):
        load_indic_ner_adapter()


def test_loader_accepts_explicit_user_path(monkeypatch):
    tokenizer = _Tokenizer([(0, 0)])
    model = _Model([0])
    model.eval = lambda: None
    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer
        ),
        AutoModelForTokenClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    monkeypatch.setattr(
        "openmed.ner.families.indic.importlib.import_module",
        lambda name: module,
    )

    adapter = load_indic_ner_adapter("/models/indic")

    assert adapter.model_id == "/models/indic"
    assert adapter.tokenizer is tokenizer
    assert adapter.model is model
