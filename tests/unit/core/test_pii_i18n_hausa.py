"""Hausa Boko/Ajami pattern, surrogate, and leakage regressions (OM-862)."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

from faker.config import AVAILABLE_LOCALES

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import LANG_TO_LOCALE, resolve_locale
from openmed.core.pii import (
    _apply_safety_sweep_to_result,
    _build_deidentification_result,
)
from openmed.core.pii_i18n import (
    LANGUAGE_FAKE_DATA,
    LANGUAGE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    get_patterns_for_language,
)
from openmed.core.script_detect import (
    SCRIPT_LANGUAGE_HINTS,
    candidate_languages_for_script,
    detect_script,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "pii"
    / "hausa_synthetic_notes.jsonl"
)

LABEL_TO_ENTITY_TYPE = {
    "AGE": "age",
    "DATE": "date",
    "ID_NUM": "national_id",
    "PHONE": "phone_number",
}


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recovered_spans(text: str) -> set[tuple[str, int, int, str]]:
    recovered: set[tuple[str, int, int, str]] = set()
    for pattern in get_patterns_for_language("ha"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            recovered.add((pattern.entity_type, match.start(), match.end(), value))
    return recovered


def test_hausa_pack_uses_native_faker_locale_without_approximation_warning():
    assert "ha" in NATIONAL_ID_ONLY_LANGUAGES
    assert LANG_TO_LOCALE["ha"] == "ha_NG"
    assert "ha_NG" in AVAILABLE_LOCALES

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_locale("ha") == "ha_NG"
        surrogate = Anonymizer(lang="ha", consistent=True, seed=862).surrogate(
            "Amina Ɗanladi",
            "PERSON",
        )

    assert surrogate
    assert not [warning for warning in caught if warning.category is UserWarning]


def test_hausa_fake_data_includes_hooked_names_and_required_locations():
    fake_data = LANGUAGE_FAKE_DATA["ha"]
    hooked = set("ƁɓƊɗƘƙ")

    assert any(hooked & set(name) for name in fake_data["NAME"])
    assert {"Kano", "Kaduna", "Sokoto", "Niamey"} <= set(fake_data["LOCATION"])


def test_boko_and_ajami_fixtures_recover_every_gold_span_at_exact_offsets():
    rows = _fixture_rows()
    assert {row["metadata"]["script"] for row in rows} == {"Boko", "Ajami"}

    for row in rows:
        assert row["language"] == "ha"
        assert row["metadata"]["synthetic"] is True
        observed = _recovered_spans(row["text"])
        for span in row["gold_spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
            assert (
                LABEL_TO_ENTITY_TYPE[span["label"]],
                span["start"],
                span["end"],
                span["text"],
            ) in observed


def test_ajami_fixture_contract_is_numeric_pattern_only_for_both_digit_sets():
    ajami_rows = [
        row for row in _fixture_rows() if row["metadata"]["script"] == "Ajami"
    ]

    assert {row["metadata"]["digits"] for row in ajami_rows} == {
        "Western",
        "Arabic-Indic",
    }
    assert all(
        row["metadata"]["coverage"] == "numeric-pattern-only" for row in ajami_rows
    )
    assert all(
        {span["label"] for span in row["gold_spans"]} == {"DATE", "PHONE", "ID_NUM"}
        for row in ajami_rows
    )


def test_hausa_synthetic_fixture_leakage_is_zero_offline():
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
            lang="ha",
        )
        result = _build_deidentification_result(
            text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="ha",
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

        assert added_count == len(row["gold_spans"])
        assert leaked == []
        assert leakage_rate == 0.0


def test_arabic_script_routes_to_arabic_and_hausa_without_changing_ar_patterns():
    assert SCRIPT_LANGUAGE_HINTS["Arabic"] == ("ar", "ha", "ur")
    assert candidate_languages_for_script("Arabic") == ("ar", "ha", "ur")

    text = "تاريخ الميلاد 01/01/2000، هاتف +20 10 1234 5678"
    observed = {
        (entity_type, start, end, value)
        for entity_type, start, end, value in _recovered_arabic_spans(text)
    }
    assert ("date", 14, 24, "01/01/2000") in observed
    assert ("phone_number", 31, 47, "+20 10 1234 5678") in observed


def _recovered_arabic_spans(text: str) -> set[tuple[str, int, int, str]]:
    recovered: set[tuple[str, int, int, str]] = set()
    for pattern in get_patterns_for_language("ar"):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is None or pattern.validator(value):
                recovered.add((pattern.entity_type, match.start(), match.end(), value))
    return recovered


def test_hooked_boko_consonants_are_classified_as_latin():
    uppercase_extended_b = "ƁƊƘ"
    lowercase_ipa_and_extended_b = "ɓɗƙ"

    assert all(0x0180 <= ord(char) <= 0x024F for char in uppercase_extended_b)
    assert 0x0180 <= ord("ƙ") <= 0x024F
    assert all(0x0250 <= ord(char) <= 0x02AF for char in "ɓɗ")
    for char in uppercase_extended_b + lowercase_ipa_and_extended_b:
        assert detect_script(char) == "Latin"


def test_hooked_boko_name_replacement_preserves_source_offsets(monkeypatch):
    text = "Majiyyaci Ɗanladi Ƙasimu ya zo asibiti."
    original_name = "Ɗanladi Ƙasimu"
    replacement = "Bilkisu Ɗanƙande"
    start = text.index(original_name)
    end = start + len(original_name)

    def _hooked_surrogate(self, original, label, **kwargs):
        assert original == original_name
        assert label == "PERSON"
        return replacement

    monkeypatch.setattr(Anonymizer, "surrogate", _hooked_surrogate)
    model_result = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=original_name,
                label="PERSON",
                confidence=1.0,
                start=start,
                end=end,
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
        lang="ha",
        consistent=True,
        seed=862,
        locale=None,
        use_safety_sweep=False,
    )

    assert result.deidentified_text == text[:start] + replacement + text[end:]
    assert result.deidentified_text.encode("utf-8").decode("utf-8") == (
        result.deidentified_text
    )
    assert result.pii_entities[0].start == start
    assert result.pii_entities[0].end == end
    assert text[start:end] == original_name
    assert result.deidentified_text.endswith(text[end:])
    assert LANGUAGE_PII_PATTERNS["ha"]
