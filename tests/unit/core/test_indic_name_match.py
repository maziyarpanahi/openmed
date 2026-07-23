"""Tests for transliteration-aware Indic personal-name matching."""

from __future__ import annotations

import json
import unicodedata

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.config import OpenMedConfig
from openmed.core.indic_name_match import (
    MAX_INDIC_NAME_SURFACE_CHARS,
    IndicNameNormalizer,
    canonical_indic_name_key,
    detect_name_script,
    indic_names_match,
)
from openmed.core.pii import deidentify
from openmed.core.surrogate_vault import SurrogateVault
from openmed.processing.outputs import EntityPrediction, PredictionResult


@pytest.mark.parametrize(
    ("variants", "expected_suffix"),
    [
        (("संजय", "Sanjay", "Sanjai"), "sanjay"),
        (("कृष्णा", "Krishna", "Krishnaa"), "krishna"),
        (("लक्ष्मी", "Lakshmi", "Laxmi"), "lakshmi"),
    ],
)
def test_stdlib_normalizer_folds_script_and_romanization_variants(
    variants,
    expected_suffix,
):
    keys = {canonical_indic_name_key(surface) for surface in variants}

    assert keys == {f"indic-name-v1:{expected_suffix}"}


def test_collision_gate_keeps_distinct_names_separate():
    assert indic_names_match("संजय", "Sanjai") is True
    assert indic_names_match("Sanjay", "Sanjana") is False


def test_similarity_threshold_is_validated_and_tunable():
    assert indic_names_match("Sanjay", "Sanjai", similarity_threshold=0.80)
    assert not indic_names_match("Sanjay", "Sanjai", similarity_threshold=0.90)
    with pytest.raises(ValueError, match="between 0.5 and 1.0"):
        IndicNameNormalizer(similarity_threshold=0.49)
    with pytest.raises(TypeError, match="real number"):
        IndicNameNormalizer(similarity_threshold=True)
    with pytest.raises(ValueError, match="between 0.5 and 1.0"):
        IndicNameNormalizer(similarity_threshold=float("nan"))


def test_oversized_surfaces_use_distinct_bounded_fallback_keys():
    normalizer = IndicNameNormalizer()
    prefix = "S" * MAX_INDIC_NAME_SURFACE_CHARS

    first = normalizer.canonical_key(f"{prefix}a")
    second = normalizer.canonical_key(f"{prefix}b")

    assert first.startswith("indic-name-v1:overflow:")
    assert second.startswith("indic-name-v1:overflow:")
    assert first != second


def test_user_supplied_transliterator_is_used_without_bundled_weights():
    class LocalAdapter:
        def to_latin(self, text: str) -> str:
            assert text == "सं जय"
            return "Sanjai"

        def from_latin(self, text: str, target_script: str) -> str:
            assert target_script == "devanagari"
            return "नव नाम"

    normalizer = IndicNameNormalizer(transliterator=LocalAdapter())

    assert normalizer.canonical_key("सं जय") == "indic-name-v1:sanjay"
    assert normalizer.render_surrogate("New Name", source_surface="सं जय") == "नव नाम"


def test_stdlib_rendering_preserves_source_script_class():
    normalizer = IndicNameNormalizer()
    rendered = normalizer.render_surrogate(
        "Ketan Sharma",
        source_surface="संजय",
    )

    assert detect_name_script(rendered) == "devanagari"
    assert (
        normalizer.render_surrogate(
            "Ketan Sharma",
            source_surface="Sanjay",
        )
        == "Ketan Sharma"
    )


def test_stdlib_rendering_never_leaves_latin_letters_in_indic_output():
    rendered = IndicNameNormalizer().render_surrogate(
        "Victor Xavier C. Fox",
        source_surface="संजय",
    )

    letters = [char for char in rendered if char.isalpha()]
    assert letters
    assert all(unicodedata.name(char, "").startswith("DEVANAGARI ") for char in letters)


@pytest.mark.parametrize(
    ("surface", "lang", "script", "unicode_prefix"),
    [
        ("रवि", "hi", "devanagari", "DEVANAGARI "),
        ("রবি", "bn", "bengali", "BENGALI "),
        ("ਰਵੀ", "pa", "gurmukhi", "GURMUKHI "),
        ("રવિ", "gu", "gujarati", "GUJARATI "),
        ("ରବି", "or", "odia", "ORIYA "),
        ("ரவி", "ta", "tamil", "TAMIL "),
        ("రవి", "te", "telugu", "TELUGU "),
        ("ರವಿ", "kn", "kannada", "KANNADA "),
        ("രവി", "ml", "malayalam", "MALAYALAM "),
    ],
)
def test_vault_renders_supported_brahmic_scripts_without_mixed_letters(
    surface,
    lang,
    script,
    unicode_prefix,
):
    vault = SurrogateVault.in_memory(
        "synthetic-test-secret",
        transliteration_aware_name_matching=True,
    )

    rendered = vault.get_or_create(
        surface,
        label="PERSON",
        lang=lang,
        create_surrogate=lambda attempt: "Victor Xavier C. Fox",
    )

    assert vault.key_for(surface, label="PERSON", lang=lang).lang == "indic"
    assert detect_name_script(rendered) == script
    letters = [char for char in rendered if char.isalpha()]
    assert letters
    assert all(
        unicodedata.name(char, "").startswith(unicode_prefix) for char in letters
    )


def test_vault_rejects_mixed_script_renderer_output():
    vault = SurrogateVault.in_memory(
        "synthetic-test-secret",
        transliteration_aware_name_matching=True,
    )

    rendered = vault.get_or_create(
        "संजय",
        label="PERSON",
        lang="hi",
        create_surrogate=lambda attempt: "Victor Cross",
        render_surrogate=lambda identity: "विक्टरMixed",
    )

    letters = [char for char in rendered if char.isalpha()]
    assert letters
    assert all(unicodedata.name(char, "").startswith("DEVANAGARI ") for char in letters)


def test_anonymizer_reuses_identity_while_rendering_each_script():
    anonymizer = Anonymizer(
        lang="hi",
        consistent=True,
        seed=668,
        transliteration_aware_name_matching=True,
    )
    variants = ("संजय", "Sanjay", "Sanjai")

    identities = {
        anonymizer.surrogate_identity(surface, "PERSON", lang="hi")
        for surface in variants
    }
    assert len(identities) == 1
    assert anonymizer.surrogate("Sanjay", "PERSON", lang="hi") == anonymizer.surrogate(
        "Sanjai", "PERSON", lang="hi"
    )
    assert (
        detect_name_script(anonymizer.surrogate("संजय", "PERSON", lang="hi"))
        == "devanagari"
    )


def test_config_round_trip_and_environment(monkeypatch):
    config = OpenMedConfig.from_dict(
        {
            "transliteration_aware_name_matching": True,
            "indic_name_similarity_threshold": 0.85,
        }
    )
    assert config.to_dict()["transliteration_aware_name_matching"] is True
    assert config.to_dict()["indic_name_similarity_threshold"] == 0.85

    monkeypatch.setenv("OPENMED_TRANSLITERATION_AWARE_NAME_MATCHING", "1")
    monkeypatch.setenv("OPENMED_INDIC_NAME_SIMILARITY_THRESHOLD", "0.90")
    environment_config = OpenMedConfig()
    assert environment_config.transliteration_aware_name_matching is True
    assert environment_config.indic_name_similarity_threshold == 0.90

    monkeypatch.setenv("OPENMED_TRANSLITERATION_AWARE_NAME_MATCHING", " false ")
    assert OpenMedConfig().transliteration_aware_name_matching is False

    with pytest.raises(TypeError, match="must be a boolean"):
        OpenMedConfig(transliteration_aware_name_matching="false")
    with pytest.raises(TypeError, match="real number"):
        OpenMedConfig(indic_name_similarity_threshold=True)


def test_vault_links_variants_and_renders_one_identity_per_script(tmp_path):
    path = tmp_path / "indic-vault.json"
    vault = SurrogateVault.from_file(
        path,
        hmac_secret="synthetic-test-secret",
        transliteration_aware_name_matching=True,
    )

    def create(attempt: int) -> str:
        return "Ketan Sharma" if attempt == 0 else f"Ketan Sharma {attempt}"

    normalizer = vault.indic_name_normalizer
    surfaces = {}
    for source in ("संजय", "Sanjay", "Sanjai"):
        surfaces[source] = vault.get_or_create(
            source,
            label="PERSON",
            lang="hi",
            create_surrogate=create,
            render_surrogate=lambda identity, source=source: (
                normalizer.render_surrogate(identity, source_surface=source)
            ),
        )

    assert len(vault.entries()) == 1
    assert {entry.surrogate for entry in vault.entries()} == {"Ketan Sharma"}
    assert surfaces["Sanjay"] == surfaces["Sanjai"] == "Ketan Sharma"
    assert detect_name_script(surfaces["संजय"]) == "devanagari"
    assert (
        detect_name_script(vault.get("संजय", label="PERSON", lang="hi") or "")
        == "devanagari"
    )
    assert vault.get("Sanjai", label="PERSON", lang="hi") == "Ketan Sharma"
    assert vault.key_for("Sanjay", label="PERSON", lang="hi").lang == "indic"

    negative_key = vault.key_for("Sanjana", label="PERSON", lang="hi")
    assert negative_key != vault.key_for("Sanjay", label="PERSON", lang="hi")

    persisted = path.read_text(encoding="utf-8")
    assert all(source not in persisted for source in ("संजय", "Sanjay", "Sanjai"))
    payload = json.loads(persisted)
    assert payload["name_matching"] == {
        "backend": "stdlib",
        "enabled": True,
        "normalizer_version": "indic-name-v1",
        "similarity_threshold": 0.8,
    }
    assert payload["name_matching_tag"].startswith("hmac-sha256:")

    reloaded = SurrogateVault.from_file(
        path,
        hmac_secret="synthetic-test-secret",
    )
    assert reloaded.transliteration_aware_name_matching is True
    assert reloaded.get("Sanjai", label="PERSON", lang="hi") == "Ketan Sharma"


def test_file_vault_rejects_tampered_name_matching_metadata(tmp_path):
    path = tmp_path / "tampered-indic-vault.json"
    vault = SurrogateVault.from_file(
        path,
        hmac_secret="synthetic-test-secret",
        transliteration_aware_name_matching=True,
    )
    vault.get_or_create(
        "Sanjay",
        label="PERSON",
        lang="hi",
        create_surrogate=lambda attempt: "Ketan Sharma",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["name_matching"]["similarity_threshold"] = 0.9
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata is invalid"):
        SurrogateVault.from_file(path, hmac_secret="synthetic-test-secret")


def test_vault_rejects_ambiguous_matching_flag():
    with pytest.raises(TypeError, match="must be a boolean"):
        SurrogateVault.in_memory(
            "synthetic-test-secret",
            transliteration_aware_name_matching="false",
        )


def test_disabled_vault_keeps_existing_exact_cross_script_behavior():
    vault = SurrogateVault.in_memory(
        "synthetic-test-secret",
        transliteration_aware_name_matching=False,
    )
    for index, source in enumerate(("संजय", "Sanjay", "Sanjai")):
        vault.get_or_create(
            source,
            label="PERSON",
            lang="hi",
            create_surrogate=lambda attempt, index=index: f"Synthetic {index}",
        )

    assert vault.key_for("संजय", label="PERSON", lang="hi") == vault.key_for(
        "Sanjay", label="PERSON", lang="hi"
    )
    assert vault.key_for("Sanjai", label="PERSON", lang="hi") != vault.key_for(
        "Sanjay", label="PERSON", lang="hi"
    )
    assert len(vault.entries()) == 2


def test_deidentify_config_enables_code_mixed_vault_matching(monkeypatch):
    text = "Synthetic patient संजय is also Sanjay and Sanjai."
    variants = ("संजय", "Sanjay", "Sanjai")

    def fake_extract(value: str, *args, **kwargs) -> PredictionResult:
        entities = [
            EntityPrediction(
                text=surface,
                label="PERSON",
                start=value.index(surface),
                end=value.index(surface) + len(surface),
                confidence=0.99,
            )
            for surface in variants
        ]
        return PredictionResult(
            text=value,
            entities=entities,
            model_name="synthetic-test",
            timestamp="now",
        )

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    config = OpenMedConfig(transliteration_aware_name_matching=True)
    vault = SurrogateVault.in_memory("synthetic-test-secret")

    result = deidentify(
        text,
        method="replace",
        lang="hi",
        config=config,
        consistent=True,
        seed=668,
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    assert len(vault.entries()) == 1
    assert (
        len({vault.key_for(surface, label="PERSON", lang="hi") for surface in variants})
        == 1
    )
    assert len({entry.surrogate for entry in vault.entries()}) == 1
    assert all(surface not in result.deidentified_text for surface in variants)
    latin_surrogates = {
        entity.surrogate
        for entity in result.pii_entities
        if detect_name_script(entity.original_text or "") == "latin"
    }
    assert len(latin_surrogates) == 1
    indic_surrogate = next(
        entity.surrogate
        for entity in result.pii_entities
        if detect_name_script(entity.original_text or "") == "devanagari"
    )
    assert indic_surrogate is not None
    assert detect_name_script(indic_surrogate) == "devanagari"
