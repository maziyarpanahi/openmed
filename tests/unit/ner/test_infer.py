from __future__ import annotations

import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

infer_module = importlib.import_module("openmed.ner.infer")
from openmed.core.config import OpenMedConfig
from openmed.ner.indexing import (
    GLIFORMER_DISEASE_MODEL_ID,
    GLINER2_DISEASE_MODEL_ID,
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


def test_malformed_gliner_offset_error_does_not_echo_entity_text() -> None:
    import traceback

    marker = "synthetic-private-5550199"
    entity = {"text": marker, "label": "Drug", "score": 0.9}
    with pytest.raises(KeyError) as caught:
        infer_module._convert_gliner_entity(entity)
    # Keep literals out of the raising call's source excerpt on Python 3.13+.
    # Assert the message AND the full exception chain; neither may leak values.
    rendered = "".join(
        traceback.format_exception_only(type(caught.value), caught.value)
    )
    assert marker not in rendered
    assert marker not in "".join(traceback.format_exception(caught.value))


def test_gliner_offsets_are_not_silently_coerced(subtests) -> None:
    for offset in [True, 1.5, "private-offset", -1]:
        with subtests.test(offset=offset):
            with pytest.raises(ValueError, match="offsets must"):
                infer_module._convert_gliner_entity(
                    {"text": "synthetic", "start": offset, "end": 4, "label": "Drug"}
                )


@pytest.fixture
def native_inference(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("openmed.core.models"), "HF_AVAILABLE", True
    )
    calls = []

    class Handle:
        model = SimpleNamespace(config=SimpleNamespace(max_len=128))

        def predict_entities(self, text, *, labels, threshold, flat_ner):
            calls.append(
                {
                    "text": text,
                    "labels": labels,
                    "threshold": threshold,
                    "flat_ner": flat_ner,
                }
            )
            return [
                {
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "label": "disease",
                    "score": 0.9 if match.group().lower() == "asthma" else 0.6,
                }
                for match in re.finditer(r"asthma|diabetes", text, re.IGNORECASE)
            ]

    loads = []

    def load(model_id, **options):
        loads.append(
            (model_id, {key: value for key, value in options.items() if key != "token"})
        )
        return Handle()

    for family in ("gliformer", "gliner2"):
        monkeypatch.setattr(infer_module, f"ensure_{family}_available", lambda: None)
        monkeypatch.setattr(infer_module, f"load_{family}_handle", load)
    return calls, loads


@pytest.mark.parametrize(
    "model_id,family",
    [(GLIFORMER_DISEASE_MODEL_ID, "gliformer"), (GLINER2_DISEASE_MODEL_ID, "gliner2")],
)
def test_disease_infer_defaults_offsets_and_confidence(
    native_inference, model_id, family
):
    calls, loads = native_inference
    text = "asthma; diabetes; asthma"
    response = infer(NerRequest(model_id=model_id, text=text, threshold=0.7))
    assert [(e.text, e.start, e.end, e.label) for e in response.entities] == [
        ("asthma", 0, 6, "DISEASE"),
        ("asthma", 18, 24, "DISEASE"),
    ]
    assert calls[0]["labels"] == ["disease"]
    assert response.meta["family"] == family
    assert loads[0][0] == model_id


@pytest.mark.parametrize(
    "model_id,family",
    [
        ("knowledgator/gliformer-base-v1", "gliformer"),
        ("fastino/gliner2.5-base-v1", "gliner2"),
        ("fastino/gliner2-base-v1", "gliner2"),
    ],
)
def test_unindexed_native_models_use_explicit_labels(
    native_inference, model_id, family
):
    calls, _ = native_inference
    response = infer(
        NerRequest(model_id=model_id, text="asthma", labels=["disease"], flat_ner=False)
    )
    assert response.meta["family"] == family
    assert calls[0]["flat_ner"] is False
    assert calls[0]["labels"] == ["disease"]


def test_unindexed_ambiguous_native_model_accepts_explicit_family(native_inference):
    response = infer(
        NerRequest(
            model_id="synthetic/ner",
            family="gliformer",
            text="asthma",
            labels=["disease"],
        )
    )
    assert response.entities[0].label == "DISEASE"


def test_unrecognized_model_still_requires_registration_or_family():
    with pytest.raises(ValueError, match="not found in index"):
        infer(NerRequest(model_id="synthetic/unknown", text="asthma"))


@pytest.mark.parametrize("labels", [[], [""], [" "], [1], "disease"])
def test_invalid_native_labels_fail_before_loading(native_inference, labels):
    _, loads = native_inference
    with pytest.raises(ValueError, match="labels must"):
        infer(
            NerRequest(
                model_id=GLIFORMER_DISEASE_MODEL_ID, text="asthma", labels=labels
            )
        )
    assert not loads


def test_native_local_index_uses_checkpoint_path(native_inference, tmp_path):
    _, loads = native_inference
    record = ModelRecord(
        id="disease-local",
        family="gliformer",
        path=str(tmp_path),
        default_labels=("disease",),
    )
    index = ModelIndex(
        models=(record,), generated_at=datetime.now(timezone.utc), source_dir=tmp_path
    )
    response = infer(NerRequest(model_id=record.id, text="asthma"), index=index)
    assert response.entities
    assert loads[0][0] == str(tmp_path)
    assert loads[0][1]["local_files_only"] is True


def test_analyze_text_native_keeps_offsets_context_and_result_type(native_inference):
    from openmed import AnalyzeResult, analyze_text

    calls, _ = native_inference
    text = (
        "No asthma. "
        + "Diabetes was documented in this synthetic note. " * 14
        + "Prior asthma."
    )
    result = analyze_text(
        text, model_id=GLIFORMER_DISEASE_MODEL_ID, threshold=0.7, assert_context=True
    )
    assert isinstance(result, AnalyzeResult)
    assert [(e.text, e.start) for e in result.entities] == [
        ("asthma", 3),
        ("asthma", text.rindex("asthma")),
    ]
    assert all(text[e.start : e.end] == e.text for e in result.entities)
    assert result.entities[0].metadata["clinical_context"]["negation"] == "negated"
    assert len(calls) > 1


@pytest.mark.parametrize("output_format", ["dict", "json", "html", "csv"])
@pytest.mark.parametrize(
    "model_id", [GLIFORMER_DISEASE_MODEL_ID, "fastino/gliner2.5-base-v1"]
)
def test_analyze_text_native_preserves_renderers(
    native_inference, model_id, output_format
):
    from openmed import analyze_text

    result = analyze_text(
        "asthma; diabetes",
        model_id=model_id,
        labels=["disease"],
        threshold=0.5,
        confidence_threshold=0.7,
        output_format=output_format,
        sentence_detection=False,
    )
    if output_format == "dict":
        assert [(e.text, e.label) for e in result.entities] == [("asthma", "DISEASE")]
    elif output_format == "json":
        assert json.loads(result)["entities"][0]["text"] == "asthma"
    elif output_format == "csv":
        assert result[0]["text"] == "asthma"
        assert result[0]["label"] == "DISEASE"
    else:
        assert "asthma" in result
        assert "DISEASE" in result


def test_native_pipeline_handles_empty_batch(native_inference):
    record = infer_module.resolve_model_record(GLIFORMER_DISEASE_MODEL_ID)
    pipeline = infer_module.SpanPipeline(record, OpenMedConfig())
    assert pipeline([]) == []
    assert pipeline(["synthetic", "asthma"])[0] == []


def test_native_malformed_predictions_fail_closed(subtests):
    for entity in [
        {"text": "asthma", "start": 0, "end": 99, "score": 0.9},
        {"text": "asthma", "start": 4, "end": 2, "score": 0.9},
        {"text": "wrong", "start": 0, "end": 6, "score": 0.9},
        {"text": "asthma", "start": 0, "end": 6, "score": float("nan")},
        {"text": "asthma", "start": 0, "end": 6, "score": 1.1},
    ]:
        with subtests.test(entity=entity):
            handle = SimpleNamespace(
                predict_entities=lambda *_a, **_k: [{**entity, "label": "disease"}]
            )
            with pytest.raises(ValueError):
                infer_module._native_predictions(
                    handle, "asthma", ["disease"], 0.5, True, OpenMedConfig()
                )


def test_native_offline_loading_and_prediction_block_network(
    native_inference, monkeypatch
):
    import socket

    from openmed.core.offline import OfflineModeError

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    config = OpenMedConfig(local_only=True)
    calls, loads = native_inference
    infer(NerRequest(model_id=GLIFORMER_DISEASE_MODEL_ID, text="asthma"), config=config)
    assert loads[0][1]["local_files_only"] is True
    handle = SimpleNamespace(
        predict_entities=lambda *_a, **_k: socket.create_connection(
            ("example.com", 443)
        )
    )
    with pytest.raises(OfflineModeError):
        infer_module._native_predictions(
            handle, "asthma", ["disease"], 0.5, True, config
        )
    assert calls


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_native_invalid_threshold_does_not_load_model(native_inference, threshold):
    _, loads = native_inference
    with pytest.raises(ValueError):
        infer(
            NerRequest(
                model_id=GLIFORMER_DISEASE_MODEL_ID, text="asthma", threshold=threshold
            )
        )
    assert not loads
