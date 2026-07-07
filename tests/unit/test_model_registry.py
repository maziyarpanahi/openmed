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


def test_resolve_draft_model_for_laneformer_uses_separate_permissive_artifact():
    draft = model_registry.resolve_draft_model_for("laneformer-2b-it")

    assert draft is not None
    assert draft.target_model_id == "OpenMed/laneformer-2b-it-q4-mlx"
    assert draft.draft_model_id == "OpenMed/laneformer-pii-draft-350m-q4-mlx"
    assert draft.tokenizer == "shared"
    assert draft.permissive_license is True


def test_biomedical_ner_category_entity_types_are_reconciled():
    expected = {
        "Oncology": {
            "AMINO_ACID",
            "CANCER",
            "CELL",
            "CELLULAR_COMPONENT",
            "CHEM",
            "GENE_OR_GENE_PRODUCT",
            "ORGAN",
            "ORGANISM",
            "PATHOLOGICAL_FORMATION",
            "SIMPLE_CHEMICAL",
            "SPECIES",
            "TISSUE",
        },
        "Anatomy": {"ANATOMY", "ORGAN", "TISSUE"},
        "Genomics": {"DNA", "GENE", "GENE_OR_GENE_PRODUCT", "PROTEIN", "RNA"},
        "Pathology": {"CONDITION", "DISEASE", "PATHOLOGY"},
        "Hematology": {"CANCER", "DISEASE"},
        "Protein": {"GENE_OR_GENE_PRODUCT", "PROTEIN"},
        "Chemical": {"CHEM", "DRUG", "MEDICATION", "SIMPLE_CHEMICAL"},
    }

    for category, entity_types in expected.items():
        registered = model_registry._CATEGORY_ENTITY_TYPES[category]
        registered_set = set(registered)
        assert registered_set == entity_types
        assert len(registered) == len(registered_set)


def test_get_entity_types_by_category_surfaces_biomedical_family_labels():
    oncology_types = model_registry.get_entity_types_by_category("Oncology")

    assert {
        "AMINO_ACID",
        "CELL",
        "CELLULAR_COMPONENT",
        "GENE_OR_GENE_PRODUCT",
        "ORGAN",
        "PATHOLOGICAL_FORMATION",
        "TISSUE",
    }.issubset(oncology_types)
    assert {"DRUG", "MEDICATION"}.issubset(
        model_registry.get_entity_types_by_category("Chemical")
    )
    assert "CONDITION" in model_registry.get_entity_types_by_category("Pathology")


def test_find_models_by_entity_type_returns_models_with_reconciled_family_labels():
    organ_models = {
        model.model_id for model in model_registry.find_models_by_entity_type("ORGAN")
    }
    oncology = model_registry.get_model_info("oncology_detection_superclinical")

    assert "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M" in organ_models
    assert "OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M" in organ_models
    assert oncology is not None
    assert "ORGAN" in oncology.entity_types
