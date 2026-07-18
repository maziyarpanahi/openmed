"""Pytest configuration and fixtures for OpenMed tests."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

import openmed
from openmed.core.config import OpenMedConfig
from openmed.processing.tokenizer_cache import clear_tokenizer_cache

_PAGED_KV_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_MLX_PAGED_KV_CACHE_BUDGET",
    "OPENMED_SERVICE_MLX_PAGED_KV_CACHE_PAGE_TOKENS",
    "OPENMED_SERVICE_MLX_PAGED_KV_CACHE_CHUNK_TOKENS",
    "OPENMED_SERVICE_MLX_PAGED_KV_CACHE_WINDOW_TOKENS",
    "OPENMED_SERVICE_MLX_PAGED_KV_CACHE_BYTES_PER_TOKEN",
)


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return OpenMedConfig(
        default_org="TestOrg",
        cache_dir="/tmp/test_cache",
        device="cpu",
        log_level="DEBUG",
        timeout=60,
    )


@pytest.fixture
def sample_text():
    """Provide sample medical text for testing."""
    return "Patient John Doe has diabetes and hypertension. Prescribed metformin 500mg daily."


@pytest.fixture
def sample_long_text():
    """Provide longer sample medical text for testing."""
    return """
    The patient is a 65-year-old male with a history of type 2 diabetes mellitus,
    hypertension, and coronary artery disease. He presents today with complaints of
    chest pain and shortness of breath. Current medications include metformin 1000mg
    twice daily, lisinopril 10mg daily, and atorvastatin 40mg daily.

    Physical examination reveals blood pressure of 140/90 mmHg, heart rate of 85 bpm,
    and temperature of 98.6°F. Laboratory results show glucose levels of 180 mg/dL
    and HbA1c of 8.2%.

    Assessment and plan: Continue current diabetes management with metformin.
    Consider adding insulin if glucose control does not improve.
    Follow up in 3 months for routine diabetes monitoring.
    """


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.tokenize.return_value = ["patient", "has", "diabetes"]
    tokenizer.convert_ids_to_tokens.return_value = [
        "[CLS]",
        "patient",
        "has",
        "diabetes",
        "[SEP]",
    ]
    tokenizer.convert_tokens_to_string.return_value = "patient has diabetes"

    class MockEncoding(dict):
        def word_ids(self):
            return [None, 0, 1, 2, None]

    tokenizer.return_value = MockEncoding(
        {
            "input_ids": [[101, 1234, 1235, 1236, 102]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "offset_mapping": [[(0, 0), (0, 7), (8, 11), (12, 20), (0, 0)]],
            "special_tokens_mask": [[1, 0, 0, 0, 1]],
        }
    )

    return tokenizer


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.config.num_labels = 3
    model.config.problem_type = "token_classification"
    model.config.architectures = ["BertForTokenClassification"]
    return model


@pytest.fixture
def mock_pipeline():
    """Provide a mock HuggingFace pipeline for testing."""
    pipeline = Mock()
    pipeline.return_value = [
        {
            "entity": "B-CONDITION",
            "score": 0.95,
            "word": "diabetes",
            "start": 21,
            "end": 29,
        },
        {
            "entity": "B-MEDICATION",
            "score": 0.89,
            "word": "metformin",
            "start": 59,
            "end": 68,
        },
    ]
    return pipeline


@pytest.fixture
def sample_predictions():
    """Provide sample model predictions for testing."""
    return [
        {
            "entity": "B-CONDITION",
            "score": 0.95,
            "word": "diabetes",
            "start": 21,
            "end": 29,
        },
        {
            "entity": "B-MEDICATION",
            "score": 0.89,
            "word": "metformin",
            "start": 59,
            "end": 68,
        },
        {"entity": "B-DOSAGE", "score": 0.87, "word": "500mg", "start": 69, "end": 74},
    ]


@pytest.fixture
def mock_model_info():
    """Provide mock model info for testing."""
    info = Mock()
    info.modelId = "TestOrg/test-model"
    info.author = "TestOrg"
    info.downloads = 1000
    info.likes = 50
    info.library_name = "transformers"
    info.tags = ["medical", "ner", "token-classification"]
    info.pipeline_tag = "token-classification"
    return info


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global configuration after each test."""
    yield
    # Reset to default configuration
    openmed.core.config.set_config(OpenMedConfig())


@pytest.fixture(autouse=True)
def reset_tokenizer_cache():
    """Reset process-level tokenizer cache around each test."""
    clear_tokenizer_cache()
    yield
    clear_tokenizer_cache()


@pytest.fixture(autouse=True)
def clear_paged_kv_service_env(monkeypatch):
    """Keep optional MLX paged KV-cache service env from leaking into tests."""
    for env_var in _PAGED_KV_SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


class TestHelpers:
    """Helper class with utility methods for tests."""

    @staticmethod
    def create_entity_prediction(
        text: str, label: str, confidence: float, start: int = None, end: int = None
    ):
        """Create an EntityPrediction for testing."""
        from openmed.processing.outputs import EntityPrediction

        return EntityPrediction(
            text=text, label=label, confidence=confidence, start=start, end=end
        )

    @staticmethod
    def create_prediction_result(
        text: str, entities: List[Dict], model_name: str = "test-model"
    ):
        """Create a PredictionResult for testing."""
        from datetime import datetime

        from openmed.processing.outputs import EntityPrediction, PredictionResult

        entity_objects = [
            EntityPrediction(
                text=e["word"],
                label=e["entity"],
                confidence=e["score"],
                start=e.get("start"),
                end=e.get("end"),
            )
            for e in entities
        ]

        return PredictionResult(
            text=text,
            entities=entity_objects,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            processing_time=0.123,
        )


# Make TestHelpers available as a fixture
@pytest.fixture
def test_helpers():
    """Provide access to test helper methods."""
    return TestHelpers
