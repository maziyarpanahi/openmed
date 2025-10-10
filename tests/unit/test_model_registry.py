"""Tests for the static OpenMed model registry helpers."""

import pytest

from openmed.core import model_registry


def test_get_model_info_known_key():
    info = model_registry.get_model_info("disease_detection_tiny")
    assert info is not None
    assert "Disease" in info.category
    assert info.size_mb is not None


def test_get_model_info_unknown_key():
    assert model_registry.get_model_info("does_not_exist") is None


def test_get_models_by_category():
    disease_models = model_registry.get_models_by_category("Disease")
    assert disease_models
    assert all(m.category == "Disease" for m in disease_models)

    assert model_registry.get_models_by_category("Unknown") == []


def test_get_all_models_structure():
    models = model_registry.get_all_models()
    assert isinstance(models, dict)
    key, info = next(iter(models.items()))
    assert isinstance(info.model_id, str)


def test_get_model_suggestions():
    suggestions = model_registry.get_model_suggestions("Patient diagnosed with cancer")
    assert suggestions
    top_key, info, reason = suggestions[0]
    assert isinstance(top_key, str)
    assert isinstance(reason, str)


def test_model_suggestions_various_texts():
    """Different inputs should still yield meaningful suggestions."""
    oncology_suggestions = model_registry.get_model_suggestions(
        "Patient diagnosed with metastatic cancer"
    )
    pharma_suggestions = model_registry.get_model_suggestions(
        "Chemotherapy regimen includes cisplatin"
    )

    assert oncology_suggestions
    assert pharma_suggestions
    assert all(reason for _key, _info, reason in oncology_suggestions[:3])
    assert all(reason for _key, _info, reason in pharma_suggestions[:3])
