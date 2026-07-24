"""Tests for manifest-backed multilingual PII model registry entries."""

import pytest

from openmed.core import model_registry
from openmed.core.model_registry import (
    CATEGORIES,
    OPENMED_MODELS,
    ModelInfo,
    get_default_pii_model,
    get_pii_models_by_language,
    load_manifest_rows,
)
from openmed.core.pii_i18n import (
    DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
    DEFAULT_PII_MODELS,
    INDIC_NER_LANGUAGES,
    LANGUAGE_NAMES,
    OPTIONAL_PII_MODEL,
    SUPPORTED_LANGUAGES,
)

MULTILINGUAL_DEFAULT_LANGUAGES = {"he", "id", "th", "ro"}
V2_REGISTRY_LANGUAGES = {"bn", "ta", "zh"}
OPTIONAL_ONLY_LANGUAGES = INDIC_NER_LANGUAGES - {"bn", "hi", "ta", "te"}


class TestRegistryCompleteness:
    """Verify PII model registry entries are derived from the manifest."""

    def test_manifest_pii_ids_are_registered(self):
        manifest_pii_ids = {
            row["repo_id"] for row in load_manifest_rows() if row["family"] == "PII"
        }
        registry_ids = {
            info.model_id
            for key, info in OPENMED_MODELS.items()
            if key.startswith("pii_")
        }
        assert manifest_pii_ids <= registry_ids

    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES))
    def test_supported_language_has_pii_models(self, lang):
        models = get_pii_models_by_language(lang)
        if lang in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES:
            assert not models
            assert get_default_pii_model(lang) == DEFAULT_PII_MODELS[lang]
            return
        assert models, f"No PII models found for language {lang!r}"
        assert all(lang in info.languages for info in models.values())

    def test_privacy_category_includes_all_pii_keys(self):
        pii_keys = sorted(k for k in OPENMED_MODELS if k.startswith("pii_"))
        privacy_keys = sorted(CATEGORIES["Privacy"])
        assert set(pii_keys) <= set(privacy_keys)

    def test_default_models_are_registered(self):
        registry_model_ids = {info.model_id for info in OPENMED_MODELS.values()}
        for lang, model_id in DEFAULT_PII_MODELS.items():
            if model_id == OPTIONAL_PII_MODEL:
                continue
            assert model_id in registry_model_ids, (
                f"Default model for {lang} ({model_id}) not found in registry"
            )

    @pytest.mark.parametrize("lang", sorted(V2_REGISTRY_LANGUAGES))
    def test_v2_language_has_manifest_backed_model(self, lang):
        models = get_pii_models_by_language(lang)

        assert models
        assert all(isinstance(info, ModelInfo) for info in models.values())
        assert DEFAULT_PII_MODELS[lang] in {info.model_id for info in models.values()}

    @pytest.mark.parametrize("lang", sorted(V2_REGISTRY_LANGUAGES))
    def test_v2_manifest_rows_include_release_metadata(self, lang):
        row = next(
            row
            for row in load_manifest_rows()
            if row["repo_id"] == DEFAULT_PII_MODELS[lang]
        )

        assert row["languages"] == [lang]
        assert row["license"] == "apache-2.0"
        assert row["param_count"] > 0
        assert row["script_coverage"]
        assert set(row["download_sizes"]) == {
            "safetensors",
            "mlx",
            "coreml",
            "onnx",
        }
        if row["released"] is None:
            assert row.get("download_mb") is None
            assert row.get("disk_mb") is None
            assert all(size is None for size in row["download_sizes"].values())
        else:
            assert row["download_mb"] > 0
            assert row["disk_mb"] > 0


class TestModelNaming:
    """Verify language-specific PII model IDs retain language identity."""

    def test_english_default_has_no_language_prefix(self):
        model_id = get_default_pii_model("en")
        assert model_id == DEFAULT_PII_MODELS["en"]
        for lang, name in LANGUAGE_NAMES.items():
            if lang != "en":
                assert f"{name}-" not in model_id

    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES - {"en"}))
    def test_language_specific_models_contain_language_name(self, lang):
        if lang in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES:
            pytest.skip("language intentionally uses a documented model placeholder")
        if lang in MULTILINGUAL_DEFAULT_LANGUAGES:
            pytest.skip("language intentionally defaults to multilingual family")
        if lang in OPTIONAL_ONLY_LANGUAGES:
            assert get_pii_models_by_language(lang) == {}
            return
        language_name = LANGUAGE_NAMES[lang]
        models = get_pii_models_by_language(lang)
        assert models
        if DEFAULT_PII_MODELS[lang].startswith("OpenMed/privacy-filter"):
            assert any(
                info.model_id == DEFAULT_PII_MODELS[lang] for info in models.values()
            )
            return

        assert any(f"{language_name}-" in info.model_id for info in models.values())

    def test_all_pii_model_ids_are_openmed_repos(self):
        pii_models = {k: v for k, v in OPENMED_MODELS.items() if k.startswith("pii_")}
        for key, info in pii_models.items():
            assert info.model_id.startswith("OpenMed/"), (
                f"Model {key} has unexpected model_id prefix: {info.model_id}"
            )

    @pytest.mark.parametrize(
        ("lang", "expected_key"),
        (
            ("bn", "pii_bn_msuperclinical_large"),
            ("ta", "pii_ta_msuperclinical_large"),
            ("zh", "pii_zh_bigmed_large"),
        ),
    )
    def test_v2_language_compatibility_aliases_are_stable(self, lang, expected_key):
        assert expected_key in get_pii_models_by_language(lang)

    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES - {"en"}))
    def test_language_bucket_keys_use_language_prefix_when_specific(self, lang):
        models = get_pii_models_by_language(lang)
        if lang in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES:
            assert not models
            return
        if lang in MULTILINGUAL_DEFAULT_LANGUAGES:
            assert any(
                info.model_id == DEFAULT_PII_MODELS[lang] for info in models.values()
            )
            return
        prefixed_keys = [key for key in models if key.startswith(f"pii_{lang}_")]
        if DEFAULT_PII_MODELS[lang].startswith("OpenMed/privacy-filter"):
            assert any("privacy_filter_multilingual" in key for key in models)
        else:
            assert prefixed_keys, f"No pii_{lang}_ keys found"


class TestHelperFunctions:
    """Tests for get_pii_models_by_language and get_default_pii_model."""

    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES))
    def test_get_default_pii_model(self, lang):
        model_id = get_default_pii_model(lang)
        if lang in OPTIONAL_ONLY_LANGUAGES:
            assert model_id is None
            return
        assert model_id == DEFAULT_PII_MODELS[lang]

    @pytest.mark.parametrize("lang", sorted(INDIC_NER_LANGUAGES))
    def test_configured_indic_model_is_registered(self, monkeypatch, lang):
        monkeypatch.setenv("OPENMED_INDIC_NER_MODEL", "/models/indic-ner")

        models = get_pii_models_by_language(lang)

        optional = models[f"pii_{lang}_indic_ner"]
        assert optional.model_id == "/models/indic-ner"
        assert optional.languages == [lang]
        assert optional.entity_types == ["PERSON", "LOCATION", "ORGANIZATION"]

    def test_get_default_pii_model_unsupported(self):
        result = get_default_pii_model("xx")
        assert result is None

    def test_get_pii_models_returns_model_info(self):
        for lang in SUPPORTED_LANGUAGES:
            models = get_pii_models_by_language(lang)
            for key, info in models.items():
                assert isinstance(info, ModelInfo)
                assert key.startswith("pii_")
                assert info.category == "Privacy"

    def test_get_pii_models_excludes_unsupported_claimed_script(self, monkeypatch):
        common = {
            "display_name": "Hindi PII",
            "category": "Privacy",
            "specialization": "HI PII detection",
            "description": "Privacy token-classification model",
            "entity_types": ["PERSON"],
            "size_category": "Small",
            "languages": ["hi"],
        }
        supported = ModelInfo(
            model_id="OpenMed/pii-hi-supported",
            script_coverage={"devanagari": {"verdict": "supported"}},
            **common,
        )
        unsupported = ModelInfo(
            model_id="OpenMed/pii-hi-unsupported",
            script_coverage={"devanagari": {"verdict": "unsupported"}},
            **common,
        )
        monkeypatch.setattr(
            model_registry,
            "OPENMED_MODELS",
            {
                "pii_hi_supported": supported,
                "pii_hi_unsupported": unsupported,
            },
        )

        assert model_registry.get_pii_models_by_language("hi") == {
            "pii_hi_supported": supported
        }
