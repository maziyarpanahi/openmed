"""Regression tests for non-Latin PERSON surrogate round trips."""

from __future__ import annotations

from datetime import datetime

import pytest

import openmed
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import LANG_TO_LOCALE
from openmed.core.labels import normalize_label
from openmed.core.pii import deidentify, extract_pii
from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES
from openmed.core.script_detect import detect_script
from openmed.core.translit import romanize_name
from openmed.processing.outputs import EntityPrediction, PredictionResult

TARGET_LANGUAGES = ("ja", "zh", "ko", "ru", "el")
SOURCE_NAMES = {
    "ja": "山田 太郎",
    "zh": "王伟",
    "ko": "김민준",
    "ru": "Иван Петров",
    "el": "Γιώργος Παπαδόπουλος",
}
ROMANIZATION_SAMPLES = {
    "ja": "渡辺 春香",
    "zh": "刘兰英",
    "ko": "박지영",
    "ru": "Людмила Гурьева",
    "el": "Μαρία Νικολάου",
}


def _require_language_pack(lang: str) -> None:
    missing: list[str] = []
    if lang not in SUPPORTED_LANGUAGES:
        missing.append("SUPPORTED_LANGUAGES")
    if lang not in DEFAULT_PII_MODELS:
        missing.append("DEFAULT_PII_MODELS")
    if lang not in LANG_TO_LOCALE:
        missing.append("LANG_TO_LOCALE")
    if missing:
        pytest.skip(
            f"{lang} PERSON language pack is not wired; missing " + ", ".join(missing)
        )


def _person_detector(*names: str, model_calls: list[str] | None = None):
    expected = tuple(sorted(set(names), key=len, reverse=True))

    def analyze(text: str, model_name: str, **_kwargs) -> PredictionResult:
        if model_calls is not None:
            model_calls.append(model_name)
        entities: list[EntityPrediction] = []
        for name in expected:
            cursor = 0
            while True:
                start = text.find(name, cursor)
                if start < 0:
                    break
                end = start + len(name)
                entities.append(
                    EntityPrediction(
                        text=text[start:end],
                        label="PERSON",
                        start=start,
                        end=end,
                        confidence=0.99,
                    )
                )
                cursor = end
        return PredictionResult(
            text=text,
            entities=sorted(entities, key=lambda entity: entity.start or 0),
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )

    return analyze


def _assert_person_at(result: PredictionResult, text: str, name: str) -> None:
    start = text.index(name)
    end = start + len(name)
    matches = [
        entity
        for entity in result.entities
        if normalize_label(entity.label) == "PERSON"
    ]
    assert [(entity.start, entity.end, entity.text) for entity in matches] == [
        (start, end, name)
    ]
    assert result.text[start:end] == name


@pytest.mark.parametrize("lang", TARGET_LANGUAGES)
def test_person_surrogate_redetects_with_exact_mixed_script_offsets(
    lang: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detect, replace, and re-detect a PERSON under the same language pack."""

    _require_language_pack(lang)
    source_name = SOURCE_NAMES[lang]
    surrogate = Anonymizer(lang=lang, consistent=True, seed=270).surrogate(
        source_name,
        "PERSON",
    )
    assert surrogate != source_name
    assert detect_script(surrogate) != "Latin"

    model_calls: list[str] = []
    monkeypatch.setattr(
        openmed,
        "analyze_text",
        _person_detector(source_name, surrogate, model_calls=model_calls),
    )
    source_line = f"Patient A-270: {source_name}; dose 5 mg"
    detected = extract_pii(
        source_line,
        use_smart_merging=False,
        lang=lang,
    )
    _assert_person_at(detected, source_line, source_name)

    replaced = deidentify(
        source_line,
        method="replace",
        use_smart_merging=False,
        use_safety_sweep=False,
        lang=lang,
        consistent=True,
        seed=270,
    )
    replaced_line = replaced.deidentified_text
    assert replaced_line == source_line.replace(source_name, surrogate)

    redetected = extract_pii(
        replaced_line,
        use_smart_merging=False,
        lang=lang,
    )
    _assert_person_at(redetected, replaced_line, surrogate)
    assert set(model_calls) == {DEFAULT_PII_MODELS[lang]}


@pytest.mark.parametrize("lang", TARGET_LANGUAGES)
def test_romanized_surrogate_redetects_under_same_pack(
    lang: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_language_pack(lang)
    surrogate = Anonymizer(lang=lang, consistent=True, seed=270).surrogate(
        SOURCE_NAMES[lang],
        "PERSON",
    )
    romanized = romanize_name(surrogate, lang=lang)
    assert romanized != surrogate
    assert detect_script(romanized) == "Latin"
    assert romanized.isascii()

    model_calls: list[str] = []
    monkeypatch.setattr(
        openmed,
        "analyze_text",
        _person_detector(romanized, model_calls=model_calls),
    )
    text = f"Patient A-270: {romanized}; follow-up complete"
    result = extract_pii(
        text,
        use_smart_merging=False,
        lang=lang,
    )
    _assert_person_at(result, text, romanized)
    assert model_calls == [DEFAULT_PII_MODELS[lang]]


@pytest.mark.parametrize("lang", TARGET_LANGUAGES)
def test_target_script_has_dependency_free_latin_rendering(lang: str) -> None:
    romanized = romanize_name(ROMANIZATION_SAMPLES[lang], lang=lang)

    assert romanized
    assert romanized.isascii()
    assert detect_script(romanized) == "Latin"
