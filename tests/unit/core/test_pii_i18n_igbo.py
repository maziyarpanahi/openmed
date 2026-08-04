"""Igbo pattern, grapheme-offset, surrogate, and leakage regressions."""

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
from openmed.core.script_detect import (
    candidate_languages_for_script,
    normalize_for_pii_detection,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

FIXTURE_PATH = Path("openmed/eval/golden/fixtures/i18n/ig.jsonl")

LABEL_TO_ENTITY_TYPE = {
    "AGE": "age",
    "DATE": "date",
    "ID_NUM": "national_id",
    "PHONE": "phone_number",
    "STREET_ADDRESS": "street_address",
    "ZIPCODE": "postcode",
}


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recovered_spans(text: str) -> set[tuple[str, int, int, str]]:
    recovered: set[tuple[str, int, int, str]] = set()
    for pattern in get_patterns_for_language("ig"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is None or pattern.validator(value):
                recovered.add((pattern.entity_type, match.start(), match.end(), value))
    return recovered


def test_igbo_pack_uses_native_faker_locale_without_approximation_warning():
    assert "ig" in NATIONAL_ID_ONLY_LANGUAGES
    assert "ig" not in DEFAULT_PII_MODELS
    assert LANG_TO_LOCALE["ig"] == "ig_NG"
    assert "ig_NG" in AVAILABLE_LOCALES
    assert LANGUAGE_PII_PATTERNS["ig"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_locale("ig") == "ig_NG"
        surrogate = Anonymizer(lang="ig", consistent=True, seed=864).surrogate(
            "Chịọma Nwankwọ",
            "PERSON",
        )

    assert surrogate
    assert not [warning for warning in caught if warning.category is UserWarning]


def test_igbo_fake_names_use_dot_below_and_include_required_locations():
    fake_data = LANGUAGE_FAKE_DATA["ig"]

    assert fake_data["NAME"]
    for name in fake_data["NAME"]:
        assert "\u0323" in unicodedata.normalize("NFD", name)
    assert {"Enugu", "Onitsha", "Owerri", "Aba"} <= set(fake_data["LOCATION"])


def test_igbo_pack_reuses_shared_nigerian_patterns_and_required_context():
    patterns = LANGUAGE_PII_PATTERNS["ig"]

    assert any(pattern.pattern == NIGERIA_NIN_PATTERN for pattern in patterns)
    assert any(pattern.pattern == NIGERIA_PHONE_PATTERN for pattern in patterns)
    context_words = {
        word.casefold() for pattern in patterns for word in pattern.context_words
    }
    assert {"nomba", "afo ndu", "ubochi omumu"} <= context_words


def test_igbo_fixtures_cover_native_and_code_mixed_nfc_nfd_text():
    rows = _fixture_rows()

    assert {row["metadata"]["fixture_kind"] for row in rows} == {
        "Igbo-only",
        "English-Igbo",
    }
    assert {row["metadata"]["normalization"] for row in rows} == {"NFC", "NFD"}
    for row in rows:
        normalization = row["metadata"]["normalization"]
        assert row["language"] == "ig"
        assert row["metadata"]["synthetic"] is True
        assert unicodedata.is_normalized(normalization, row["text"])


def test_igbo_fixtures_recover_every_structured_span_at_exact_offsets():
    for row in _fixture_rows():
        text = row["text"]
        observed = _recovered_spans(text)
        for span in row["gold_spans"]:
            assert text[span["start"] : span["end"]] == span["text"]
            if span["label"] == "PERSON":
                continue
            assert (
                LABEL_TO_ENTITY_TYPE[span["label"]],
                span["start"],
                span["end"],
                span["text"],
            ) in observed


def test_igbo_fixture_normalized_spans_remap_to_exact_original_surfaces():
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

            assert (original_start, original_end) == (span["start"], span["end"])
            assert text[original_start:original_end] == span["text"]
            assert original_end == len(text) or not unicodedata.category(
                text[original_end]
            ).startswith("M")


def test_igbo_synthetic_fixture_leakage_is_zero_offline():
    for row in _fixture_rows():
        text = row["text"]
        person = next(span for span in row["gold_spans"] if span["label"] == "PERSON")
        model_result = PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=person["text"],
                    label="PERSON",
                    confidence=1.0,
                    start=person["start"],
                    end=person["end"],
                )
            ],
            model_name="synthetic-name-fixture",
            timestamp="2026-07-18T00:00:00Z",
            metadata={},
        )

        swept_result, added_count = _apply_safety_sweep_to_result(
            text,
            model_result,
            lang="ig",
        )
        result = _build_deidentification_result(
            text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="ig",
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
        leakage_rate = len(leaked) / len(row["gold_spans"])

        assert added_count == len(row["gold_spans"]) - 1
        assert leaked == []
        assert leakage_rate == 0.0


@pytest.mark.parametrize("normalization", ("NFC", "NFD"))
def test_igbo_name_replacement_preserves_grapheme_offsets(monkeypatch, normalization):
    original_name = unicodedata.normalize(normalization, "Ọbịanụju Nwankwọ")
    replacement = "Ngọzi Okafọ"
    text = unicodedata.normalize(
        normalization,
        f"Onye ọrịa {original_name} bịara ụlọ ọgwụ.",
    )
    full_start = text.index(original_name)
    full_end = full_start + len(original_name)
    if normalization == "NFD":
        model_start = full_start + 1
        model_end = full_end - 1
    else:
        model_start = full_start
        model_end = full_end

    def _marked_surrogate(self, original, label, **kwargs):
        assert original == original_name
        assert label == "PERSON"
        return replacement

    monkeypatch.setattr(Anonymizer, "surrogate", _marked_surrogate)
    model_result = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=text[model_start:model_end],
                label="PERSON",
                confidence=1.0,
                start=model_start,
                end=model_end,
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
        lang="ig",
        consistent=True,
        seed=864,
        locale=None,
        use_safety_sweep=False,
    )

    assert result.deidentified_text == text[:full_start] + replacement + text[full_end:]
    assert result.pii_entities[0].start == full_start
    assert result.pii_entities[0].end == full_end
    assert result.pii_entities[0].original_text == original_name
    if normalization == "NFD":
        assert result.pii_entities[0].metadata["grapheme_boundary_adjustment"] == {
            "start_codepoints": 1,
            "end_codepoints": 1,
        }


def test_latin_script_routing_includes_igbo():
    assert "ig" in candidate_languages_for_script("Latin")
