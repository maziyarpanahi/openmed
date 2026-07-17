from __future__ import annotations

from datetime import datetime

import pytest

from openmed.core.lang_id_codemix import (
    TokenLanguageIdentifier,
    identify_token_languages,
    token_language_runs,
)
from openmed.core.pii import extract_pii
from openmed.core.pii_i18n import (
    get_code_mixed_pattern_runs,
    get_patterns_for_code_mixed_text,
    get_patterns_for_language,
)
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _empty_prediction(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="fixture-pii-model",
        timestamp=datetime.now().isoformat(),
    )


def test_heuristic_lid_is_deterministic_and_preserves_exact_offsets():
    text = "Patient Ravi ka aadhaar 246778325484 hai."

    first = identify_token_languages(text)
    second = identify_token_languages(text)

    assert first == second
    assert [token.label for token in first] == [
        "en",
        "ne",
        "hi",
        "hi",
        "univ",
        "hi",
        "univ",
    ]
    assert [text[token.start : token.end] for token in first] == [
        "Patient",
        "Ravi",
        "ka",
        "aadhaar",
        "246778325484",
        "hai",
        ".",
    ]
    assert token_language_runs(first)


def test_token_metadata_contains_offsets_and_hash_but_no_raw_surface():
    text = "Ravi ka phone"
    token = identify_token_languages(text)[0]

    metadata = token.to_metadata(text)

    assert metadata["start"] == 0
    assert metadata["end"] == 4
    assert str(metadata["token_hash"]).startswith("sha256:")
    assert not ({"text", "token", "surface", "value"} & set(metadata))
    assert "Ravi" not in repr(token)


def test_script_and_universal_guards_override_model_hook():
    class EnglishOnlyHook:
        def predict(self, text, spans):
            return ["en"] * len(spans)

    text = "रोगी 123 Ravi"
    tokens = TokenLanguageIdentifier(EnglishOnlyHook()).identify(text)

    assert [token.label for token in tokens] == ["hi", "univ", "en"]
    assert tokens[0].source == "script"
    assert tokens[1].source == "universal"
    assert tokens[2].source == "model_hook"


def test_model_hook_must_return_one_supported_label_per_token():
    with pytest.raises(ValueError, match="one label per token"):
        identify_token_languages("Ravi ka", model=lambda text, spans: ["ne"])

    with pytest.raises(ValueError, match="unsupported label"):
        identify_token_languages(
            "Ravi",
            model=lambda text, spans: ["unsupported"],
        )


def test_named_entity_runs_route_to_both_language_packs():
    routes = get_code_mixed_pattern_runs("Patient Ravi ko fever hai.")
    named_entity_routes = [route for route in routes if route.token_label == "ne"]

    assert named_entity_routes
    assert all(set(route.languages) == {"hi", "en"} for route in named_entity_routes)
    assert all(route.patterns for route in named_entity_routes)


def test_code_mixed_patterns_add_hindi_pack_to_english_baseline():
    text = "Patient Ravi ka aadhaar 246778325484 hai."

    baseline = get_patterns_for_language("en")
    routed = get_patterns_for_code_mixed_text(text)

    assert len(routed) > len(baseline)
    assert any(
        pattern.entity_type == "national_id" and "aadhaar" in pattern.context_words
        for pattern in routed
    )


def test_extract_pii_uses_hinglish_routing_with_empty_model(monkeypatch):
    text = "Patient Ravi ka aadhaar 246778325484 hai."
    monkeypatch.setattr(
        "openmed.analyze_text", lambda *args, **kwargs: _empty_prediction(text)
    )

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        use_smart_merging=True,
        code_mixed=True,
    )

    assert any(
        entity.label == "national_id" and entity.start == 24 and entity.end == 36
        for entity in result.entities
    )


def test_pipeline_deidentifies_hinglish_identifier_with_english_route():
    text = "Patient Ravi ka aadhaar 246778325484 hai."

    result = Pipeline(
        model_detector=lambda model_text, **kwargs: _empty_prediction(model_text),
        lang="en",
        code_mixed=True,
    ).run(text, method="mask")

    assert "246778325484" not in result.redacted_text
    assert "[national_id]" in result.redacted_text


def test_named_entity_model_output_is_not_dropped_by_lid_routing(monkeypatch):
    text = "patient arjun ko fever hai"
    start = text.index("arjun")
    model_result = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text="arjun",
                label="NAME",
                start=start,
                end=start + len("arjun"),
                confidence=0.99,
            )
        ],
        model_name="fixture-pii-model",
        timestamp=datetime.now().isoformat(),
    )
    monkeypatch.setattr("openmed.analyze_text", lambda *args, **kwargs: model_result)

    def named_entity_hook(model_text, spans):
        return ["en", "ne", "hi", "en", "hi"]

    result = extract_pii(
        text,
        model_name="fixture-pii-model",
        lang="en",
        code_mixed=True,
        lid_model=named_entity_hook,
    )

    assert any(
        entity.text == "arjun" and entity.label == "NAME" for entity in result.entities
    )
