from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from openmed.clinical.context import INDIA_CLINICAL_NER_DISCLAIMER
from openmed.core.pii import deidentify, extract_pii
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    INDIA_CLINICAL_MULTILINGUAL_FALLBACK,
    LANGUAGE_MODEL_PREFIX,
    get_india_clinical_model_route,
    india_clinical_route_active,
)
from openmed.core.script_detect import (
    india_clinical_script_windows,
    normalize_for_pii_detection,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

_FIXTURE_DIR = Path("openmed/eval/golden/fixtures/i18n")
_INDIC_WORD_RE = re.compile(r"[A-Za-z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+")


def _india_fixtures() -> list[dict]:
    rows: list[dict] = []
    for language in ("hi", "te"):
        path = _FIXTURE_DIR / f"{language}.jsonl"
        rows.extend(
            row
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in (json.loads(line),)
            if row["metadata"].get("india_clinical_codemixed") is True
        )
    return rows


def _fixture_analyzer(fixture: dict, called_models: list[str]):
    def analyze(text: str, *, model_name: str, **_kwargs) -> PredictionResult:
        called_models.append(model_name)
        entities: list[EntityPrediction] = []
        for span in fixture["gold_spans"]:
            surfaces = [span["text"]]
            if span["label"] in {"PERSON", "LOCATION"}:
                surfaces = _INDIC_WORD_RE.findall(span["text"])
            for surface in surfaces:
                start = text.find(surface)
                if start < 0:
                    continue
                entities.append(
                    EntityPrediction(
                        text=surface,
                        label=span["label"],
                        confidence=0.98,
                        start=start,
                        end=start + len(surface),
                    )
                )
        return PredictionResult(
            text=text,
            entities=entities,
            model_name=model_name,
            timestamp=datetime(2026, 1, 1).isoformat(),
        )

    return analyze


@pytest.mark.parametrize(
    ("lang", "text", "native_script"),
    [
        ("hi", "Patient राहुल Sharma", "Devanagari"),
        ("te", "Patient కావ్య Rao", "Telugu"),
        ("te", "Patient राहुल Rao", "Devanagari"),
    ],
)
def test_india_script_windows_activate_with_context(
    lang: str,
    text: str,
    native_script: str,
) -> None:
    windows = india_clinical_script_windows(text, lang)

    assert {window.script for window in windows} == {"Latin", native_script}
    assert all(window.extract(text) for window in windows)
    assert all(
        window.start <= window.core_start < window.core_end <= window.end
        for window in windows
    )
    assert india_clinical_route_active(text, lang) is True


def test_india_route_does_not_activate_for_single_script_or_other_languages() -> None:
    assert india_clinical_script_windows("Patient Asha Sharma", "hi") == ()
    assert india_clinical_script_windows("रोगी राहुल", "hi") == ()
    assert india_clinical_script_windows("Patient राहुल", "en") == ()


def test_detection_normalization_preserves_indic_combining_marks() -> None:
    text = "राहुल మరియు కావ్య"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == text
    assert normalized.stripped_combining_marks == 0


@pytest.mark.parametrize("lang", ["hi", "te"])
def test_india_model_route_uses_registered_first_party_models(lang: str) -> None:
    route = get_india_clinical_model_route(lang)

    assert route.latin_model == DEFAULT_PII_MODELS["en"]
    assert route.native_model == DEFAULT_PII_MODELS[lang]
    assert route.fallback_model == INDIA_CLINICAL_MULTILINGUAL_FALLBACK
    assert route.latin_prefix == LANGUAGE_MODEL_PREFIX["en"]
    assert route.native_prefix == LANGUAGE_MODEL_PREFIX[lang]
    assert route.model_for_script("Latin") == DEFAULT_PII_MODELS["en"]
    assert route.model_for_script("Devanagari") == DEFAULT_PII_MODELS[lang]
    assert (
        route.model_for_script("Latin", user_model="/models/india-clinical")
        == "/models/india-clinical"
    )


def test_user_supplied_model_is_run_for_both_script_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _india_fixtures()[0]
    called_models: list[str] = []
    monkeypatch.setattr(
        "openmed.analyze_text",
        _fixture_analyzer(fixture, called_models),
    )

    extract_pii(
        fixture["text"],
        lang=fixture["language"],
        model_name="/models/india-clinical",
    )

    assert called_models
    assert set(called_models) == {"/models/india-clinical"}
    assert len(called_models) >= 2


@pytest.mark.parametrize("fixture", _india_fixtures(), ids=lambda row: row["id"])
def test_code_mixed_fixture_spans_route_and_keep_exact_offsets(
    fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_models: list[str] = []
    monkeypatch.setattr(
        "openmed.analyze_text",
        _fixture_analyzer(fixture, called_models),
    )

    with warnings.catch_warnings(record=True) as caught:
        result = extract_pii(fixture["text"], lang=fixture["language"])

    assert not caught
    expected = {
        (span["label"], span["start"], span["end"])
        for span in fixture["gold_spans"]
        if span["label"] in {"PERSON", "LOCATION", "DATE", "MEDICATION"}
    }
    observed = {
        (entity.label.upper(), entity.start, entity.end) for entity in result.entities
    }
    assert expected <= observed

    route = get_india_clinical_model_route(fixture["language"])
    assert route.latin_model in called_models
    assert route.native_model in called_models
    assert result.metadata["india_clinical"]["active"] is True
    assert result.metadata["india_clinical"]["fallback_model"] == route.fallback_model
    assert (
        result.metadata["india_clinical"]["disclaimer"] == INDIA_CLINICAL_NER_DISCLAIMER
    )

    medication = next(
        entity for entity in result.entities if entity.label.upper() == "MEDICATION"
    )
    assert medication.metadata["clinical_normalization"]["expansions"] in (
        ["tab"],
        ["cap"],
    )


@pytest.mark.parametrize("fixture", _india_fixtures(), ids=lambda row: row["id"])
def test_code_mixed_fixture_deidentification_has_zero_name_or_id_leakage(
    fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openmed.analyze_text",
        _fixture_analyzer(fixture, []),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = deidentify(fixture["text"], lang=fixture["language"])

    for substring in fixture["metadata"]["leakage_substrings"]:
        assert substring.casefold() not in result.deidentified_text.casefold()
    unexpected = [
        warning
        for warning in caught
        if "has no native Faker locale" not in str(warning.message)
    ]
    assert not unexpected


def test_india_clinical_disclaimer_is_explicitly_assistive() -> None:
    normalized = INDIA_CLINICAL_NER_DISCLAIMER.casefold()

    assert "assists review" in normalized
    assert "does not make clinical or disclosure decisions" in normalized


def test_india_clinical_fallback_and_disclaimer_are_documented() -> None:
    documentation = Path("docs/languages.md").read_text(encoding="utf-8")

    assert INDIA_CLINICAL_MULTILINGUAL_FALLBACK in documentation
    assert INDIA_CLINICAL_NER_DISCLAIMER in " ".join(documentation.split())
