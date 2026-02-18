"""Tests for multilingual PII model registry entries."""

import pytest

from openmed.core.model_registry import (
    OPENMED_MODELS,
    CATEGORIES,
    ModelInfo,
    get_pii_models_by_language,
    get_default_pii_model,
)
from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# Registry Completeness Tests
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Verify all 176+ PII models are registered."""

    def test_total_pii_model_count(self):
        """All PII models including legacy alias should be >= 176."""
        pii_keys = [k for k in OPENMED_MODELS if k.startswith("pii_")]
        # 36 English (35 + pii_detection alias) + 35x4 langs = 176
        assert len(pii_keys) >= 176

    def test_english_pii_model_count(self):
        """English has 35 PII models + 1 legacy alias = 36."""
        en_models = get_pii_models_by_language("en")
        assert len(en_models) >= 35

    def test_french_pii_model_count(self):
        """French has 35 PII models."""
        fr_models = get_pii_models_by_language("fr")
        assert len(fr_models) == 35

    def test_german_pii_model_count(self):
        """German has 35 PII models."""
        de_models = get_pii_models_by_language("de")
        assert len(de_models) == 35

    def test_italian_pii_model_count(self):
        """Italian has 35 PII models."""
        it_models = get_pii_models_by_language("it")
        assert len(it_models) == 35

    def test_spanish_pii_model_count(self):
        """Spanish has 35 PII models."""
        es_models = get_pii_models_by_language("es")
        assert len(es_models) == 35

    def test_privacy_category_includes_all(self):
        """Privacy category should list all PII model keys."""
        pii_keys = sorted(k for k in OPENMED_MODELS if k.startswith("pii_"))
        privacy_keys = sorted(CATEGORIES["Privacy"])
        assert pii_keys == privacy_keys


# ---------------------------------------------------------------------------
# Model Naming Convention Tests
# ---------------------------------------------------------------------------


class TestModelNaming:
    """Verify generated model IDs follow HuggingFace naming convention."""

    def test_french_model_ids_contain_french(self):
        fr_models = get_pii_models_by_language("fr")
        for key, info in fr_models.items():
            assert "French-" in info.model_id, (
                f"French model {key} missing 'French-' in model_id: {info.model_id}"
            )

    def test_german_model_ids_contain_german(self):
        de_models = get_pii_models_by_language("de")
        for key, info in de_models.items():
            assert "German-" in info.model_id, (
                f"German model {key} missing 'German-' in model_id: {info.model_id}"
            )

    def test_italian_model_ids_contain_italian(self):
        it_models = get_pii_models_by_language("it")
        for key, info in it_models.items():
            assert "Italian-" in info.model_id, (
                f"Italian model {key} missing 'Italian-' in model_id: {info.model_id}"
            )

    def test_spanish_model_ids_contain_spanish(self):
        es_models = get_pii_models_by_language("es")
        for key, info in es_models.items():
            assert "Spanish-" in info.model_id, (
                f"Spanish model {key} missing 'Spanish-' in model_id: {info.model_id}"
            )

    def test_english_model_ids_no_language_prefix(self):
        en_models = get_pii_models_by_language("en")
        for key, info in en_models.items():
            for prefix in ("French-", "German-", "Italian-", "Spanish-"):
                assert prefix not in info.model_id, (
                    f"English model {key} has unexpected prefix in: {info.model_id}"
                )

    def test_all_pii_model_ids_start_with_openmed(self):
        pii_models = {k: v for k, v in OPENMED_MODELS.items() if k.startswith("pii_")}
        for key, info in pii_models.items():
            assert info.model_id.startswith("OpenMed/OpenMed-PII-"), (
                f"Model {key} has unexpected model_id prefix: {info.model_id}"
            )

    def test_all_pii_model_ids_end_with_v1(self):
        pii_models = {k: v for k, v in OPENMED_MODELS.items() if k.startswith("pii_")}
        for key, info in pii_models.items():
            assert info.model_id.endswith("-v1"), (
                f"Model {key} model_id doesn't end with -v1: {info.model_id}"
            )

    def test_generated_keys_follow_pattern(self):
        """Generated keys should be pii_{lang}_{architecture}."""
        for lang in ("fr", "de", "it", "es"):
            lang_models = get_pii_models_by_language(lang)
            for key in lang_models:
                assert key.startswith(f"pii_{lang}_"), (
                    f"Key {key} doesn't start with pii_{lang}_"
                )


# ---------------------------------------------------------------------------
# Mirror Structure Tests
# ---------------------------------------------------------------------------


class TestMirrorStructure:
    """Verify multilingual models mirror English model architectures."""

    def test_each_language_mirrors_english(self):
        """Each non-English language should have the same architectures."""
        en_models = get_pii_models_by_language("en")
        # Filter out the legacy pii_detection alias
        en_archs = sorted(k[4:] for k in en_models if k != "pii_detection")

        for lang in ("fr", "de", "it", "es"):
            lang_models = get_pii_models_by_language(lang)
            lang_archs = sorted(k[len(f"pii_{lang}_"):] for k in lang_models)
            assert lang_archs == en_archs, (
                f"{lang} architectures don't match English. "
                f"Missing: {set(en_archs) - set(lang_archs)}, "
                f"Extra: {set(lang_archs) - set(en_archs)}"
            )

    def test_size_categories_preserved(self):
        """Size categories should be preserved across languages."""
        en_models = get_pii_models_by_language("en")
        for en_key, en_info in en_models.items():
            if en_key == "pii_detection":
                continue
            arch = en_key[4:]  # strip "pii_"
            for lang in ("fr", "de", "it", "es"):
                lang_key = f"pii_{lang}_{arch}"
                assert lang_key in OPENMED_MODELS, f"Missing: {lang_key}"
                assert OPENMED_MODELS[lang_key].size_category == en_info.size_category


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for get_pii_models_by_language and get_default_pii_model."""

    def test_get_default_pii_model_en(self):
        model_id = get_default_pii_model("en")
        assert model_id == DEFAULT_PII_MODELS["en"]
        assert "SuperClinical-Small-44M" in model_id

    def test_get_default_pii_model_fr(self):
        model_id = get_default_pii_model("fr")
        assert model_id == DEFAULT_PII_MODELS["fr"]
        assert "French-" in model_id

    def test_get_default_pii_model_de(self):
        model_id = get_default_pii_model("de")
        assert model_id == DEFAULT_PII_MODELS["de"]
        assert "German-" in model_id

    def test_get_default_pii_model_it(self):
        model_id = get_default_pii_model("it")
        assert model_id == DEFAULT_PII_MODELS["it"]
        assert "Italian-" in model_id

    def test_get_default_pii_model_es(self):
        model_id = get_default_pii_model("es")
        assert model_id == DEFAULT_PII_MODELS["es"]
        assert "Spanish-" in model_id

    def test_get_default_pii_model_unsupported(self):
        result = get_default_pii_model("ja")
        assert result is None

    def test_get_pii_models_returns_model_info(self):
        for lang in SUPPORTED_LANGUAGES:
            models = get_pii_models_by_language(lang)
            for key, info in models.items():
                assert isinstance(info, ModelInfo)
                assert info.category == "Privacy"

    def test_all_default_models_in_registry(self):
        """Each default model ID should correspond to a registered model."""
        all_model_ids = {v.model_id for v in OPENMED_MODELS.values()}
        for lang, model_id in DEFAULT_PII_MODELS.items():
            assert model_id in all_model_ids, (
                f"Default model for {lang} ({model_id}) not found in registry"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
