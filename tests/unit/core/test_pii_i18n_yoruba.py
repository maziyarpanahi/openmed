"""Yoruba pattern, grapheme-offset, surrogate, and leakage regressions."""

from __future__ import annotations

import json
import re
import unicodedata
import warnings
from pathlib import Path

import pytest
from faker.config import AVAILABLE_LOCALES

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import LANG_TO_LOCALE, resolve_locale
from openmed.core.pii import (
    _apply_safety_sweep_to_result,
    _build_deidentification_result,
)
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    LANGUAGE_FAKE_DATA,
    LANGUAGE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    NIGERIA_NIN_PATTERN,
    NIGERIA_PHONE_PATTERN,
    get_patterns_for_language,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.core.script_detect import normalize_for_pii_detection
from openmed.processing.outputs import EntityPrediction, PredictionResult

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "i18n"
    / "yo.jsonl"
)

LABEL_TO_ENTITY_TYPE = {
    "DATE": "date",
    "PHONE": "phone_number",
    "ID_NUM": "national_id",
    "STREET_ADDRESS": "street_address",
    "ZIPCODE": "postcode",
}


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _without_marks(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if not unicodedata.category(char).startswith("M")
    )


def _recovered_spans(text: str) -> set[tuple[str, int, int, str]]:
    recovered: set[tuple[str, int, int, str]] = set()
    for pattern in get_patterns_for_language("yo"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is None or pattern.validator(value):
                recovered.add((pattern.entity_type, match.start(), match.end(), value))
    return recovered


def test_yoruba_pack_uses_native_faker_locale_without_approximation_warning():
    assert "yo" in NATIONAL_ID_ONLY_LANGUAGES
    assert "yo" not in DEFAULT_PII_MODELS
    assert LANG_TO_LOCALE["yo"] == "yo_NG"
    assert "yo_NG" in AVAILABLE_LOCALES
    assert LANGUAGE_PII_PATTERNS["yo"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_locale("yo") == "yo_NG"
        surrogate = Anonymizer(lang="yo", consistent=True, seed=863).surrogate(
            "Adéọlá Ọládìpọ̀",
            "PERSON",
        )

    assert surrogate
    assert not [warning for warning in caught if warning.category is UserWarning]


def test_yoruba_fake_names_carry_dot_below_and_tone_marks():
    names = LANGUAGE_FAKE_DATA["yo"]["NAME"]

    assert names
    for name in names:
        decomposed = unicodedata.normalize("NFD", name)
        assert "\u0323" in decomposed
        assert {"\u0300", "\u0301"} & set(decomposed)


def test_yoruba_pack_reuses_nigeria_nin_and_phone_constants():
    patterns = LANGUAGE_PII_PATTERNS["yo"]

    assert any(pattern.pattern == NIGERIA_NIN_PATTERN for pattern in patterns)
    assert any(pattern.pattern == NIGERIA_PHONE_PATTERN for pattern in patterns)


@pytest.mark.parametrize(
    ("marked_context", "value", "entity_types", "minimum_confidence"),
    (
        ("nọ́mbà ìdánimọ̀ NIN", "12345678901", {"national_id", "NG_NIN"}, 0.95),
        (
            "nọ́ńbà tẹlifóònù",
            "+234 803 123 4567",
            {"phone_number", "NG_PHONE"},
            0.9,
        ),
    ),
)
@pytest.mark.parametrize("normalization", ("NFC", "NFD", "unmarked"))
def test_tone_marked_decomposed_and_unmarked_contexts_trigger_patterns(
    marked_context,
    value,
    entity_types,
    minimum_confidence,
    normalization,
):
    if normalization == "unmarked":
        context = _without_marks(marked_context)
    else:
        context = unicodedata.normalize(normalization, marked_context)
    text = f"{context}: {value}."

    matches = [
        entity
        for entity in safety_sweep(text, [], lang="yo")
        if entity.text == value and entity.label in entity_types
    ]

    assert len(matches) == 1
    assert matches[0].confidence >= minimum_confidence - 1e-12


def test_yoruba_fixtures_recover_every_gold_span_at_exact_offsets():
    rows = _fixture_rows()
    assert {row["metadata"]["normalization"] for row in rows} == {
        "NFC",
        "NFD",
        "UNMARKED",
    }

    for row in rows:
        text = row["text"]
        normalization = row["metadata"]["normalization"]
        if normalization in {"NFC", "NFD"}:
            assert unicodedata.is_normalized(normalization, text)
        else:
            assert not any(unicodedata.category(char).startswith("M") for char in text)

        observed = _recovered_spans(text)
        for span in row["gold_spans"]:
            assert text[span["start"] : span["end"]] == span["text"]
            assert (
                LABEL_TO_ENTITY_TYPE[span["label"]],
                span["start"],
                span["end"],
                span["text"],
            ) in observed


def test_yoruba_fixture_normalized_spans_remap_to_exact_original_surfaces():
    for row in _fixture_rows():
        text = row["text"]
        normalization = normalize_for_pii_detection(text)
        for span in row["gold_spans"]:
            normalized_surface = normalize_for_pii_detection(span["text"]).text
            normalized_start = normalization.text.index(normalized_surface)
            original_start, original_end = normalization.remap_span(
                normalized_start,
                normalized_start + len(normalized_surface),
            )

            assert text[original_start:original_end] == span["text"]
            assert original_end == len(text) or not unicodedata.category(
                text[original_end]
            ).startswith("M")


def test_yoruba_synthetic_fixture_leakage_is_zero_offline():
    for row in _fixture_rows():
        text = row["text"]
        empty_result = PredictionResult(
            text=text,
            entities=[],
            model_name="offline-safety-sweep",
            timestamp="2026-07-18T00:00:00Z",
            metadata={},
        )

        swept_result, added_count = _apply_safety_sweep_to_result(
            text,
            empty_result,
            lang="yo",
        )
        result = _build_deidentification_result(
            text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="yo",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        leaked = [
            span["text"]
            for span in row["gold_spans"]
            if span["text"] in result.deidentified_text
        ]
        assert added_count == len(row["gold_spans"])
        assert leaked == []


def test_replacement_snaps_model_span_to_full_decomposed_name(monkeypatch):
    original_name = unicodedata.normalize("NFD", "Ọ́ládìpọ̀")
    replacement = "Bọ́láńlé Adébáyọ̀"
    text = unicodedata.normalize("NFD", f"Aláìsàn {original_name} dé.")
    full_start = text.index(original_name)
    full_end = full_start + len(original_name)
    split_start = full_start + 1
    split_end = full_end - 2

    def _marked_surrogate(self, original, label, **kwargs):
        assert original == original_name
        assert label == "PERSON"
        return replacement

    monkeypatch.setattr(Anonymizer, "surrogate", _marked_surrogate)
    model_result = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=text[split_start:split_end],
                label="PERSON",
                confidence=1.0,
                start=split_start,
                end=split_end,
            )
        ],
        model_name="synthetic-name-fixture",
        timestamp="2026-07-18T00:00:00Z",
        metadata={},
    )

    result = _build_deidentification_result(
        text,
        model_result,
        effective_method="replace",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="yo",
        consistent=True,
        seed=863,
        locale=None,
        use_safety_sweep=False,
    )

    assert result.deidentified_text == text[:full_start] + replacement + text[full_end:]
    assert result.pii_entities[0].start == full_start
    assert result.pii_entities[0].end == full_end
    assert result.pii_entities[0].original_text == original_name
    assert result.pii_entities[0].metadata["grapheme_boundary_adjustment"] == {
        "start_codepoints": 1,
        "end_codepoints": 2,
    }
