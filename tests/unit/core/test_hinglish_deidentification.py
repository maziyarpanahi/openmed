from __future__ import annotations

import json
from datetime import datetime

import pytest

from openmed.core.custom_recognizer import build_transliterated_name_recognizer
from openmed.core.pii import deidentify, extract_pii
from openmed.core.pii_i18n import (
    get_hinglish_patterns_for_token_tags,
    normalize_code_mixed_token_tags,
)
from openmed.core.pipeline import Pipeline
from openmed.core.surrogate_vault import SurrogateVault
from openmed.eval.suites.code_mixed_routing import load_code_mixed_fixtures
from openmed.processing.outputs import PredictionResult


def _empty_prediction(text: str, **kwargs) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="fixture-pii-model",
        timestamp=datetime(2026, 1, 1).isoformat(),
    )


def test_hinglish_pattern_bank_requires_a_hindi_token_tag() -> None:
    fixtures = load_code_mixed_fixtures()
    hinglish = fixtures[0]
    english = fixtures[-1]

    assert get_hinglish_patterns_for_token_tags(
        hinglish.text,
        hinglish.token_language_tags,
    )
    assert not get_hinglish_patterns_for_token_tags(
        english.text,
        english.token_language_tags,
    )


def test_code_mixed_extract_merges_name_age_and_phone(monkeypatch) -> None:
    fixture = load_code_mixed_fixtures()[0]
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)

    result = extract_pii(
        fixture.text,
        model_name="fixture-pii-model",
        code_mixed=True,
        token_language_tags=fixture.token_language_tags,
    )

    detected = {
        (entity.start, entity.end, entity.label.casefold())
        for entity in result.entities
    }
    assert any(start == 16 and end == 21 for start, end, _ in detected)
    assert any(start == 28 and end == 30 for start, end, _ in detected)
    assert any(start == 44 and end == 54 for start, end, _ in detected)


def test_deidentify_masks_hinglish_and_final_sweep_sees_merged_spans(
    monkeypatch,
) -> None:
    fixture = load_code_mixed_fixtures()[0]
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)

    result = deidentify(
        fixture.text,
        model_name="fixture-pii-model",
        code_mixed=True,
        token_language_tags=fixture.token_language_tags,
    )

    assert "Rahul" not in result.deidentified_text
    assert "45" not in result.deidentified_text
    assert "9876543210" not in result.deidentified_text
    assert result.metadata["safety_sweep"]["enabled"] is True


def test_pure_english_tags_do_not_activate_hinglish_patterns(monkeypatch) -> None:
    fixture = load_code_mixed_fixtures()[-1]
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)

    result = deidentify(
        fixture.text,
        model_name="fixture-pii-model",
        code_mixed=True,
        token_language_tags=fixture.token_language_tags,
    )

    assert result.deidentified_text == fixture.text
    assert result.pii_entities == []


def test_transliterated_name_bridge_is_configurable_with_allow_and_deny() -> None:
    recognizer = build_transliterated_name_recognizer(
        {
            "include_defaults": False,
            "given_names": ["Anaya"],
            "deny": ["Dev"],
            "allow": ["Taylor"],
        }
    )

    detected = recognizer.detect_entities("Anaya met Dev and Taylor.")

    assert [entity.text for entity in detected] == ["Anaya", "Dev"]
    assert all(entity.label == "NAME" for entity in detected)


def test_repeated_roman_name_uses_one_non_echoing_latin_vault_surrogate(
    monkeypatch,
) -> None:
    fixture = load_code_mixed_fixtures()[2]
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)
    vault = SurrogateVault.in_memory("hinglish-test-secret")

    result = deidentify(
        fixture.text,
        method="replace",
        model_name="fixture-pii-model",
        code_mixed=True,
        token_language_tags=fixture.token_language_tags,
        surrogate_vault=vault,
        seed=17,
    )

    names = [
        entity.redacted_text
        for entity in result.pii_entities
        if (entity.canonical_label or "").upper() == "PERSON"
    ]
    assert len(names) == 2
    assert names[0] == names[1]
    assert names[0].casefold() != "rahul"
    assert all(
        not char.isalpha() or "LATIN" in __import__("unicodedata").name(char, "")
        for char in names[0]
    )


def test_pipeline_audit_contains_offsets_and_no_raw_note_surfaces() -> None:
    fixture = load_code_mixed_fixtures()[0]
    result = Pipeline(
        model_name="fixture-pii-model",
        model_detector=_empty_prediction,
        code_mixed=True,
        token_language_tags=fixture.token_language_tags,
    ).run(fixture.text)

    serialized = json.dumps(result.audit_record, sort_keys=True)
    assert "Rahul" not in serialized
    assert "9876543210" not in serialized
    assert '"start": 8' in serialized
    assert "text_hash" in serialized or "input_text_hash" in serialized


def test_code_mixed_mode_validates_token_tag_contract() -> None:
    with pytest.raises(ValueError, match="requires token_language_tags"):
        Pipeline(code_mixed=True)

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        normalize_code_mixed_token_tags(
            "naam Rahul",
            [
                {"start": 5, "end": 10, "label": "ne"},
                {"start": 0, "end": 4, "label": "hi"},
            ],
        )


def test_surrogate_vault_script_constraint_rejects_non_latin_candidates() -> None:
    vault = SurrogateVault.in_memory("script-constraint-secret")

    surrogate = vault.get_or_create(
        "Rahul",
        label="PERSON",
        lang="en",
        create_surrogate=lambda attempt: "राहुल" if attempt == 0 else "Aarav",
        required_script="Latin",
    )

    assert surrogate == "Aarav"
    assert vault.get("Rahul", label="PERSON", lang="en") == "Aarav"
