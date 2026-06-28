"""Tests for typed analyze_text results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from openmed.core.results import AnalyzeResult
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service.app import _result_to_dict


def _prediction_result() -> PredictionResult:
    return PredictionResult(
        text="Patient has diabetes.",
        entities=[
            EntityPrediction(
                text="diabetes",
                label="CONDITION",
                confidence=0.91,
                start=12,
                end=20,
            )
        ],
        model_name="disease_detection_superclinical",
        timestamp=datetime.now().isoformat(),
        processing_time=0.02,
        metadata={"sentence_detection": False},
    )


def test_analyze_result_preserves_legacy_dict_shape() -> None:
    result = AnalyzeResult.from_prediction_result(_prediction_result())

    payload = result.to_dict()

    assert set(payload) == {
        "text",
        "entities",
        "model_name",
        "timestamp",
        "processing_time",
        "metadata",
    }
    assert payload == _result_to_dict(result)
    assert payload["model_name"] == result.model
    assert payload["entities"][0]["label"] == "CONDITION"


def test_analyze_result_is_frozen_and_mapping_compatible() -> None:
    result = AnalyzeResult.from_prediction_result(_prediction_result())

    assert result["model_name"] == "disease_detection_superclinical"
    assert dict(result) == result.to_dict()
    assert result.model_name == result.model
    assert result.text_length == len(result.text)
    with pytest.raises(FrozenInstanceError):
        result.model = "other"


def test_analyze_result_reexported_from_openmed() -> None:
    from openmed import AnalyzeResult as PublicAnalyzeResult

    assert PublicAnalyzeResult is AnalyzeResult


@patch("openmed.processing.sentences.segment_text")
@patch("openmed.ModelLoader")
def test_analyze_text_returns_analyze_result(
    mock_loader_cls,
    mock_segment_text,
) -> None:
    loader = Mock()
    pipeline = Mock(
        return_value=[
            {
                "entity": "CONDITION",
                "score": 0.91,
                "word": "diabetes",
                "start": 12,
                "end": 20,
            }
        ]
    )
    pipeline.tokenizer = Mock()
    loader.create_pipeline.return_value = pipeline
    loader.get_max_sequence_length.return_value = 256
    mock_loader_cls.return_value = loader
    mock_segment_text.return_value = []

    from openmed import AnalyzeResult as PublicAnalyzeResult
    from openmed import analyze_text

    result = analyze_text(
        "Patient has diabetes.",
        model_name="disease_detection_superclinical",
        sentence_detection=False,
    )

    assert isinstance(result, PublicAnalyzeResult)
    assert result.model == "disease_detection_superclinical"
    assert result.entities[0].text == "diabetes"
    assert result.to_dict()["entities"][0]["text"] == "diabetes"
