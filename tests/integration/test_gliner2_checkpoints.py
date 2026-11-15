"""Opt-in GLiNER2.5 checkpoint tests through ``gliner2.AutoExtractor``.

The tests download public Apache-2.0 checkpoints once (296 MB, 774 MB and 1.15 GB of
safetensors), reload them from the local cache with every socket blocked, and
run inference with every socket blocked. Set
``OPENMED_RUN_GLINER2_CHECKPOINT_TESTS=1`` with ``openmed[gliner2]`` installed
to run them. All inputs are synthetic.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from openmed.ner.families.gliner2 import (
    clear_gliner2_cache,
    load_gliner2_handle,
)
from openmed.structured.decision import DecisionMode, DecisionRequest, decide

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_RUN_ENV = "OPENMED_RUN_GLINER2_CHECKPOINT_TESTS"
_ENGLISH_NOTE = (
    "Synthetic patient Jane Roe was started on metformin 500 mg for type 2 "
    "diabetes by Dr. Alan Poe."
)
_SPANISH_REQUEST = (
    "La erupción volvió después de terminar los antibióticos. ¿Puedo pedir una "
    "cita con dermatología esta semana?"
)


def _require_runtime() -> None:
    if os.environ.get(_RUN_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"set {_RUN_ENV}=1 to download and run GLiNER2.5 checkpoints")
    pytest.importorskip("gliner2", reason="install openmed[gliner2]")
    pytest.importorskip("torch", reason="install openmed[gliner2]")
    pytest.importorskip("transformers", reason="install openmed[gliner2]")


@contextmanager
def _no_network() -> Iterator[list[str]]:
    attempts: list[str] = []
    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
        socket.getaddrinfo,
    )

    def blocked(*args: Any, **kwargs: Any) -> Any:
        attempts.append("socket")
        raise AssertionError("GLiNER2 attempted network access after download")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]
    try:
        yield attempts
    finally:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.create_connection,
            socket.getaddrinfo,
        ) = originals
    assert attempts == []


def _load_from_cache(model_id: str):
    # The first load may download the snapshot; the second must be cache-only.
    load_gliner2_handle(model_id)
    clear_gliner2_cache()
    with _no_network():
        handle = load_gliner2_handle(model_id, local_files_only=True)
    return handle


def test_small_checkpoint_extracts_entities_and_classifies_offline(caplog) -> None:
    _require_runtime()
    caplog.set_level(logging.DEBUG)
    handle = _load_from_cache("fastino/gliner2.5-small-v1")

    assert handle.architecture == "boundary"
    assert handle.revision and len(handle.revision) == 40
    assert handle.license_id == "Apache-2.0"

    with _no_network():
        entities = handle.extract_entities(
            _ENGLISH_NOTE, ["person", "medication", "condition"], threshold=0.5
        )
        classification = handle.classify_text(
            _ENGLISH_NOTE,
            {"note_type": ["prescription", "billing", "scheduling"]},
        )["note_type"]

    assert entities
    for entity in entities:
        assert _ENGLISH_NOTE[entity.start : entity.end] == entity.text
        assert 0.5 <= entity.score <= 1.0
    assert any(
        entity.label == "medication" and "metformin" in entity.text
        for entity in entities
    )
    assert [item.label for item in classification.scores] == [
        "prescription",
        "billing",
        "scheduling",
    ]
    assert sum(item.score for item in classification.scores) == pytest.approx(
        1.0, abs=1e-3
    )
    assert classification.label in {"prescription", "billing", "scheduling"}
    assert "Jane Roe" not in caplog.text
    assert "metformin" not in caplog.text


def test_multilingual_decide_checkpoint_scores_a_spanish_decision_offline() -> None:
    _require_runtime()
    from openmed.ner.families.gliner2_decision import GLiNER2DecisionBackend

    handle = _load_from_cache("fastino/GLiNER2.5-multi-Decide")
    backend = GLiNER2DecisionBackend(handle, task="request")
    options = (
        "book_appointment",
        "refill_prescription",
        "test_results",
        "billing_question",
    )

    with _no_network():
        result = decide(
            DecisionRequest(
                mode=DecisionMode.FIXED_CHOICE,
                input_text=_SPANISH_REQUEST,
                options=options,
            ),
            backend=backend,
        )

    assert result.backend["model_id"] == "fastino/GLiNER2.5-multi-Decide"
    assert result.backend["license_id"] == "Apache-2.0"
    assert result.backend["runtime"] == "gliner2"
    assert [item["option"] for item in result.option_scores] == list(options)
    assert sum(item["score"] for item in result.option_scores) == pytest.approx(
        1.0, abs=1e-3
    )
    assert result.ranking[0] == "book_appointment"
    assert result.autonomous_action is False
    assert result.review["required"] is True


def test_openmed_disease_checkpoint_runs_through_public_ner_apis_offline() -> None:
    _require_runtime()
    from huggingface_hub import snapshot_download

    from openmed import ModelLoader, OpenMedConfig, analyze_text
    from openmed.ner import NerRequest, infer
    from openmed.ner.indexing import GLINER2_DISEASE_MODEL_ID

    revision = "fccb29669b847cd6ca6384660d06c750c55366e2"
    snapshot = Path(
        snapshot_download(
            GLINER2_DISEASE_MODEL_ID,
            revision=revision,
            token=False,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    )
    config = OpenMedConfig(
        device="cpu", cache_dir=str(snapshot.parents[2]), local_only=True
    )
    text = (
        "65-year-old man with type 2 diabetes mellitus, stage 3 "
        "chronic kidney disease and atrial fibrillation, "
        "admitted with community-acquired pneumonia."
    )
    expected = {
        "type 2 diabetes mellitus",
        "stage 3 chronic kidney disease",
        "atrial fibrillation",
        "community-acquired pneumonia",
    }
    clear_gliner2_cache()
    with _no_network():
        result = analyze_text(
            text,
            model_id=GLINER2_DISEASE_MODEL_ID,
            revision=revision,
            threshold=0.7,
            config=config,
        )
        assert {entity.text for entity in result.entities} == expected
        assert all(entity.label == "DISEASE" for entity in result.entities)
        assert all(text[e.start : e.end] == e.text for e in result.entities)
        assert all(0.7 <= e.confidence <= 1 for e in result.entities)

        local = infer(
            NerRequest(model_id=str(snapshot), text=text, threshold=0.7),
            config=config,
        )
        assert {entity.text for entity in local.entities} == expected
        assert local.meta["labels_used"] == ["disease"]
        assert all(text[e.start : e.end] == e.text for e in local.entities)

        pipeline = ModelLoader(config).create_pipeline(str(snapshot), threshold=0.7)
        assert {item["word"] for item in pipeline(text)} == expected
        negative = "Started metformin and lisinopril; follow up in four weeks."
        assert pipeline(negative) == []
