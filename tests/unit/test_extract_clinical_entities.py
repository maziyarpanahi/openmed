"""Focused tests for the high-level clinical NER helper."""

from types import SimpleNamespace

import pytest

from openmed.core.offline import OfflineModeError
from openmed.processing import advanced_ner


class _FakeLoader:
    instances = []
    predictions = [
        {
            "entity_group": "B-disease",
            "score": 0.93,
            "start": 16,
            "end": 24,
            "word": "diabetes",
        }
    ]

    def __init__(self):
        self.config = SimpleNamespace(local_only=False)
        self.calls = []
        self.instances.append(self)

    def create_pipeline(self, model_name, **kwargs):
        self.calls.append((model_name, kwargs))

        def classify(_text):
            return list(self.predictions)

        return classify


def test_top_level_import_is_stable():
    from openmed import extract_clinical_entities

    assert extract_clinical_entities is advanced_ner.extract_clinical_entities


def test_domain_resolution_selects_registry_model_and_canonicalizes_spans(monkeypatch):
    model = SimpleNamespace(
        model_id="OpenMed/synthetic-disease-model",
        category="Disease",
        task="token-classification",
        size_category="Medium",
        param_count=125_000_000,
    )
    category_calls = []
    recommendation_calls = []

    def fake_category(category):
        category_calls.append(category)
        return [model]

    def fake_recommendations(tier):
        recommendation_calls.append(tier)
        return [model]

    _FakeLoader.instances.clear()
    monkeypatch.setattr(advanced_ner, "ModelLoader", _FakeLoader)
    monkeypatch.setattr(advanced_ner, "get_models_by_category", fake_category)
    monkeypatch.setattr(advanced_ner, "get_recommended_models", fake_recommendations)

    spans = advanced_ner.extract_clinical_entities(
        "Synthetic note: diabetes",
        domain="disease",
    )

    assert category_calls == ["Disease"]
    assert recommendation_calls == ["balanced"]
    assert _FakeLoader.instances[0].calls == [
        (
            "OpenMed/synthetic-disease-model",
            {"task": "token-classification", "aggregation_strategy": "simple"},
        )
    ]
    assert [span.to_dict() for span in spans] == [
        {
            "text": "diabetes",
            "label": "DISEASE",
            "start": 16,
            "end": 24,
            "score": 0.93,
        }
    ]


def test_model_id_override_does_not_require_registry_resolution(monkeypatch):
    _FakeLoader.instances.clear()
    monkeypatch.setattr(advanced_ner, "ModelLoader", _FakeLoader)
    monkeypatch.setattr(
        advanced_ner,
        "get_models_by_category",
        lambda _category: pytest.fail("registry should not be used for an override"),
    )
    monkeypatch.setattr(
        advanced_ner,
        "get_recommended_models",
        lambda _tier: pytest.fail("registry should not be used for an override"),
    )

    spans = advanced_ner.extract_clinical_entities(
        "Synthetic note: diabetes",
        model_id="/tmp/synthetic-clinical-model",
    )

    assert _FakeLoader.instances[0].calls[0][0] == "/tmp/synthetic-clinical-model"
    assert spans[0].label == "DISEASE"


def test_offline_uncached_model_has_actionable_error(monkeypatch):
    class UncachedLoader:
        def __init__(self):
            self.config = SimpleNamespace(local_only=False)

        def create_pipeline(self, _model_name, **_kwargs):
            raise FileNotFoundError("synthetic model is not cached")

    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    monkeypatch.setattr(advanced_ner, "ModelLoader", UncachedLoader)

    with pytest.raises(OfflineModeError, match="local cache"):
        advanced_ner.extract_clinical_entities(
            "Synthetic note: diabetes",
            model_id="OpenMed/synthetic-uncached-model",
        )
