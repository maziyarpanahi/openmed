"""Tests for the optional offset-preserving Indic NER adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openmed.core.labels import LOCATION, ORGANIZATION, PERSON, normalize_label
from openmed.ner.families.indic import (
    IndicNerAdapter,
    IndicNerCheckpointUnavailable,
    IndicNerCompatibilityError,
    IndicNerWeightsUnavailable,
    load_indic_ner_adapter,
)


class _Tokenizer:
    is_fast = True

    def __init__(self, offsets, *, error_text=None):
        self.offsets = offsets
        self.error_text = error_text

    def __call__(self, text, **kwargs):
        if self.error_text is not None:
            raise RuntimeError(self.error_text)
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["return_tensors"] == "pt"
        return {
            "attention_mask": [[1] * len(self.offsets)],
            "input_ids": [[0] * len(self.offsets)],
            "offset_mapping": [self.offsets],
        }


class _Model:
    def __init__(self, label_ids, *, config=None, label_count=5, error_text=None):
        self.config = config or SimpleNamespace(
            id2label={
                0: "O",
                1: "B-PER",
                2: "B-LOC",
                3: "B-ORG",
                4: "I-ORG",
            }
        )
        self.label_ids = label_ids
        self.label_count = label_count
        self.error_text = error_text

    def __call__(self, **kwargs):
        assert "offset_mapping" not in kwargs
        if self.error_text is not None:
            raise RuntimeError(self.error_text)
        logits = []
        for label_id in self.label_ids:
            row = [-5.0] * self.label_count
            row[label_id] = 5.0
            logits.append(row)
        return SimpleNamespace(logits=[logits])


def _bio_config():
    return SimpleNamespace(
        id2label={
            0: "O",
            1: "B-PER",
            2: "B-LOC",
            3: "B-ORG",
            4: "I-ORG",
        }
    )


def _bilou_label2id_config():
    return SimpleNamespace(
        label2id={
            "O": 0,
            "U-PERSON": 1,
            "U-LOCATION": 2,
            "B-ORGANIZATION": 3,
            "L-ORGANIZATION": 4,
        }
    )


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


def test_adapter_accepts_string_keyed_id2label_layout():
    config = SimpleNamespace(
        id2label={
            "0": "O",
            "1": "B-PER",
            "2": "B-LOC",
            "3": "B-ORG",
            "4": "I-ORG",
        }
    )
    adapter = IndicNerAdapter(
        model_id="/models/indic",
        tokenizer=_Tokenizer([(0, 0), (0, 2), (0, 0)]),
        model=_Model([0, 1, 0], config=config),
    )

    assert [(row.start, row.end, row.label) for row in adapter.predict("आर")] == [
        (0, 2, PERSON)
    ]


def test_adapter_accepts_label2id_bilou_layout_and_subword_offsets():
    text = "संस्था"
    adapter = IndicNerAdapter(
        model_id="org/indic",
        tokenizer=_Tokenizer([(0, 0), (0, 2), (2, len(text)), (0, 0)]),
        model=_Model(
            [0, 3, 4, 0],
            config=_bilou_label2id_config(),
        ),
    )

    predictions = adapter.predict(text)

    assert [(row.start, row.end, row.label) for row in predictions] == [
        (0, len(text), ORGANIZATION)
    ]


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(id2label={0: "O", 1: "B-PER"}),
        SimpleNamespace(label2id={"O": 0, "B-PER": 1, "B-LOC": 3, "B-ORG": 4}),
        SimpleNamespace(
            id2label={0: "O", 1: "B-PER", 2: "B-LOC", 3: "B-ORG"},
            num_labels=5,
        ),
    ],
)
def test_adapter_rejects_malformed_label_maps(config):
    with pytest.raises(IndicNerCompatibilityError):
        IndicNerAdapter(
            model_id="/models/indic",
            tokenizer=_Tokenizer([(0, 0)]),
            model=_Model([0], config=config, label_count=4),
        )


def test_adapter_rejects_conflicting_compatible_label_maps():
    config = _bio_config()
    config.label2id = {
        "O": 0,
        "B-LOC": 1,
        "B-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
    }

    with pytest.raises(IndicNerCompatibilityError, match="conflicting label maps"):
        IndicNerAdapter(
            model_id="/models/indic",
            tokenizer=_Tokenizer([(0, 0)]),
            model=_Model([0], config=config),
        )


@pytest.mark.parametrize(
    "offsets",
    [
        [(0, 0), (-1, 1), (0, 0)],
        [(0, 0), (1, 1), (0, 0)],
        [(0, 0), (0, 100), (0, 0)],
        [(0, 0), (0, 2), (1, 3), (0, 0)],
        [(0, 0), ("0", 2), (0, 0)],
        [(0, 0), (0, 1, 2), (0, 0)],
    ],
)
def test_adapter_rejects_unsafe_offsets_without_exposing_input(offsets):
    text = "गोपनीय"
    label_ids = [0] * len(offsets)
    adapter = IndicNerAdapter(
        model_id="/models/indic",
        tokenizer=_Tokenizer(offsets),
        model=_Model(label_ids),
    )

    with pytest.raises(IndicNerCompatibilityError) as exc_info:
        adapter.predict(text)

    assert text not in str(exc_info.value)


def test_adapter_rejects_logit_width_mismatch():
    adapter = IndicNerAdapter(
        model_id="/models/indic",
        tokenizer=_Tokenizer([(0, 2)]),
        model=_Model([1], label_count=4),
    )

    with pytest.raises(IndicNerCompatibilityError, match="logit width"):
        adapter.predict("आर")


@pytest.mark.parametrize("component", ["tokenizer", "model"])
def test_checkpoint_failures_do_not_expose_input_text(component):
    text = "गोपनीय इनपुट"
    tokenizer = _Tokenizer(
        [(0, len(text))],
        error_text=text if component == "tokenizer" else None,
    )
    model = _Model(
        [1],
        error_text=text if component == "model" else None,
    )
    adapter = IndicNerAdapter(
        model_id="/models/indic",
        tokenizer=tokenizer,
        model=model,
    )

    with pytest.raises(IndicNerCompatibilityError) as exc_info:
        adapter.predict(text)

    assert text not in str(exc_info.value)
    assert exc_info.value.__context__ is None


def test_loader_requires_explicit_weights_before_importing_dependencies(monkeypatch):
    monkeypatch.delenv("OPENMED_INDIC_NER_MODEL", raising=False)
    monkeypatch.setattr(
        "openmed.ner.families.indic.importlib.import_module",
        lambda name: pytest.fail("default loading must not import optional runtimes"),
    )

    with pytest.raises(IndicNerWeightsUnavailable, match="is not configured"):
        load_indic_ner_adapter()


def test_loader_accepts_explicit_user_path(monkeypatch, tmp_path):
    tokenizer = _Tokenizer([(0, 0)])
    model = _Model([0])
    model.eval = lambda: None
    calls = []

    def load_tokenizer(*args, **kwargs):
        calls.append(("tokenizer", args, kwargs))
        return tokenizer

    def load_model(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return model

    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setattr(
        "openmed.ner.families.indic.importlib.import_module",
        lambda name: module,
    )
    model_dir = tmp_path / "indic-checkpoint"
    model_dir.mkdir()

    adapter = load_indic_ner_adapter(str(model_dir))

    assert adapter.model_id == str(model_dir)
    assert adapter.tokenizer is tokenizer
    assert adapter.model is model
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)


def test_loader_allows_only_an_explicit_remote_checkpoint(monkeypatch):
    tokenizer = _Tokenizer([(0, 0)])
    model = _Model([0])
    calls = []

    def load(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return tokenizer if kwargs.get("use_fast", False) else model

    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=load),
    )
    monkeypatch.setattr(
        "openmed.ner.families.indic.importlib.import_module",
        lambda name: module,
    )

    adapter = load_indic_ner_adapter(
        "owner/indic-checkpoint",
        revision="reviewed-revision",
        token="test-token",
    )

    assert adapter.model_id == "owner/indic-checkpoint"
    assert all(call[1]["local_files_only"] is False for call in calls)
    assert all(call[1]["revision"] == "reviewed-revision" for call in calls)
    assert all(call[1]["token"] == "test-token" for call in calls)


def test_loader_sanitizes_checkpoint_load_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("private checkpoint path and credentials")

    module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=fail),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=fail),
    )
    monkeypatch.setattr(
        "openmed.ner.families.indic.importlib.import_module",
        lambda name: module,
    )

    with pytest.raises(IndicNerCheckpointUnavailable) as exc_info:
        load_indic_ner_adapter("owner/private-checkpoint")

    assert "private checkpoint path" not in str(exc_info.value)
    assert "credentials" not in str(exc_info.value)
    assert exc_info.value.__context__ is None
