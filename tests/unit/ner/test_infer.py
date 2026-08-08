from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

infer_module = importlib.import_module("openmed.ner.infer")
from openmed.ner.indexing import (
    GLINER_BIOMED_MODEL_ID,
    ModelIndex,
    ModelRecord,
    load_index,
)
from openmed.ner.infer import Entity, NerRequest, infer, infer_biomedical


class DummyPipeline:
    def __call__(self, text: str):
        return [
            {
                "entity_group": "Drug",
                "score": 0.9,
                "word": "Aspirin",
                "start": 0,
                "end": 7,
            },
            {
                "entity_group": "Disease",
                "score": 0.4,
                "word": "fever",
                "start": 11,
                "end": 16,
            },
        ]


class DummyLoader:
    def create_pipeline(self, model_id: str, **kwargs: Any):
        return DummyPipeline()


class DummyGLiNERHandle:
    def __init__(self, entities: List[Dict[str, Any]]):
        self._entities = entities

    def predict_entities(self, text: str, labels=None, threshold=0.5, flat_ner=True):
        if labels:
            return [
                entity for entity in self._entities if entity.get("label") in labels
            ]
        return self._entities


@pytest.fixture
def sample_index(tmp_path: Path) -> ModelIndex:
    records = (
        ModelRecord(
            id="gliner-biomed-tiny",
            family="gliner",
            domains=("biomedical",),
            languages=("en",),
            path=str(tmp_path / "gliner-biomed-tiny"),
        ),
        ModelRecord(
            id="hf-generic",
            family="other",
            domains=("generic",),
            languages=("en",),
            path=str(tmp_path / "hf-generic"),
        ),
    )
    return ModelIndex(
        models=records, generated_at=datetime.now(timezone.utc), source_dir=tmp_path
    )


def test_infer_non_gliner_threshold(sample_index: ModelIndex) -> None:
    request = NerRequest(
        model_id="hf-generic", text="Aspirin treats fever", threshold=0.5
    )
    response = infer(request, index=sample_index, loader=DummyLoader())
    assert len(response.entities) == 1
    assert response.entities[0].label == "Drug"


def test_infer_label_precedence(sample_index: ModelIndex, monkeypatch) -> None:
    mock_entities = [
        {"text": "Imatinib", "start": 0, "end": 8, "label": "Drug", "score": 0.8},
        {"text": "CML", "start": 20, "end": 23, "label": "Disease", "score": 0.6},
    ]

    def mock_load_gliner_handle(model_id: str, **kwargs: Any):
        return DummyGLiNERHandle(mock_entities)

    monkeypatch.setattr(infer_module, "load_gliner_handle", mock_load_gliner_handle)
    monkeypatch.setattr(infer_module, "ensure_gliner_available", lambda: None)

    request = NerRequest(
        model_id="gliner-biomed-tiny",
        text="Imatinib treats CML",
        threshold=0.5,
        labels=["Drug"],
    )
    response = infer(request, index=sample_index)
    assert {entity.label for entity in response.entities} == {"DRUG"}


def test_infer_fallback_to_default_labels(
    sample_index: ModelIndex, monkeypatch
) -> None:
    mock_entities = [
        {"text": "Imatinib", "start": 0, "end": 8, "label": "Drug", "score": 0.8},
    ]

    def mock_load_gliner_handle(model_id: str, **kwargs: Any):
        return DummyGLiNERHandle(mock_entities)

    monkeypatch.setattr(infer_module, "load_gliner_handle", mock_load_gliner_handle)
    monkeypatch.setattr(infer_module, "ensure_gliner_available", lambda: None)

    request = NerRequest(
        model_id="gliner-biomed-tiny",
        text="Imatinib treats patients",
        threshold=0.5,
        labels=None,
        domain=None,
    )
    response = infer(request, index=sample_index)
    assert response.meta["labels_used"]


def test_builtin_gliner_biomed_record_is_discoverable() -> None:
    record = next(
        record for record in load_index().models if record.id == GLINER_BIOMED_MODEL_ID
    )

    assert record.family == "gliner"
    assert record.domains == ("biomedical", "clinical")
    assert record.license == "Apache-2.0"


def test_infer_biomedical_uses_defaults_threshold_and_canonical_labels(
    monkeypatch,
) -> None:
    calls: Dict[str, Any] = {}

    class CapturingHandle:
        def predict_entities(self, text, labels=None, threshold=0.5, flat_ner=True):
            calls.update(
                text=text,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
            )
            return [
                {
                    "text": "Aspirin",
                    "start": 0,
                    "end": 7,
                    "label": "Drug",
                    "score": 0.91,
                },
                {
                    "text": "fever",
                    "start": 21,
                    "end": 26,
                    "label": "Disease",
                    "score": 0.49,
                },
                {
                    "text": "synthetic marker",
                    "start": 28,
                    "end": 44,
                    "label": "CustomThing",
                    "score": 0.82,
                },
            ]

    monkeypatch.setattr(
        infer_module, "load_gliner_handle", lambda *args, **kwargs: CapturingHandle()
    )
    monkeypatch.setattr(infer_module, "ensure_gliner_available", lambda: None)

    response = infer(
        NerRequest(
            model_id=GLINER_BIOMED_MODEL_ID,
            text="Aspirin was used for fever; synthetic marker.",
            threshold=0.5,
            domain="biomedical",
        )
    )
    convenience_response = infer_biomedical(
        "Aspirin was used for fever; synthetic marker.", threshold=0.5
    )

    assert calls == {
        "text": "Aspirin was used for fever; synthetic marker.",
        "labels": ["Disease", "Drug", "Gene", "Organism"],
        "threshold": 0.5,
        "flat_ner": True,
    }
    assert [(entity.text, entity.label) for entity in response.entities] == [
        ("Aspirin", "DRUG"),
        ("synthetic marker", "CustomThing"),
    ]
    assert response.meta["model_id"] == GLINER_BIOMED_MODEL_ID
    assert response.meta["domain_used"] == "biomedical"
    assert convenience_response.meta["model_id"] == GLINER_BIOMED_MODEL_ID
