"""Swahili and Sheng-style code-mixed PII regression coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.anonymizer.locales import LANG_TO_LOCALE
from openmed.core.pii import (
    _apply_safety_sweep_to_result,
    _build_deidentification_result,
)
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    LANGUAGE_FAKE_DATA,
    LANGUAGE_MODEL_PREFIX,
    LANGUAGE_NAMES,
    LANGUAGE_PII_PATTERNS,
    SUPPORTED_LANGUAGES,
    get_patterns_for_language,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.eval import harness
from openmed.eval.golden import GoldenFixture
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = Path("openmed/eval/golden/fixtures/i18n/sw.jsonl")


def _fixtures() -> list[GoldenFixture]:
    rows = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [GoldenFixture.from_mapping(row) for row in rows]


def test_swahili_language_pack_registries_and_surrogate_data():
    assert "sw" in SUPPORTED_LANGUAGES
    assert LANGUAGE_NAMES["sw"] == "Swahili"
    assert LANGUAGE_MODEL_PREFIX["sw"] == "Swahili-"
    assert DEFAULT_PII_MODELS["sw"] == "OpenMed/privacy-filter-multilingual"
    assert LANGUAGE_PII_PATTERNS["sw"]
    assert LANG_TO_LOCALE["sw"] == "sw"

    fake_data = LANGUAGE_FAKE_DATA["sw"]
    assert {"Nairobi", "Dar es Salaam", "Kampala", "Mombasa"} <= set(
        fake_data["LOCATION"]
    )
    assert {"Amina Hassan", "Daniel Otieno", "Wanjiku Njeri"} <= set(fake_data["NAME"])
    assert all(
        len(postcode) == 5 and postcode.isdigit() for postcode in fake_data["ZIPCODE"]
    )


@pytest.mark.parametrize(
    ("text", "expected_label", "expected_text"),
    (
        ("Nambari ya kitambulisho 12345678", "KE_NATIONAL_ID", "12345678"),
        ("National ID 87654321", "KE_NATIONAL_ID", "87654321"),
        ("Nambari ya NHIF 987654321", "national_id", "987654321"),
        ("NHIF member number 246813579", "national_id", "246813579"),
        (
            "Nambari ya NIDA 19791103-12345-67890-12",
            "national_id",
            "19791103-12345-67890-12",
        ),
        ("Simu +254 712 345 678", "phone_number", "+254 712 345 678"),
        ("Call +255 754 321 098", "phone_number", "+255 754 321 098"),
        ("phone +256 772 456 789", "phone_number", "+256 772 456 789"),
        ("Jina: Amina Hassan.", "name", "Amina Hassan"),
        ("Patient name: Daniel Otieno.", "name", "Daniel Otieno"),
        ("Umri wa miaka 38.", "age", "38"),
        ("Aged 47.", "age", "47"),
        ("Anwani: Kenyatta Avenue 12", "street_address", "Kenyatta Avenue 12"),
        ("msimbo wa posta 00100", "postcode", "00100"),
    ),
)
def test_bilingual_context_detects_structured_pii(
    text: str,
    expected_label: str,
    expected_text: str,
):
    detected = safety_sweep(text, [], lang="sw")
    assert [(entity.label, entity.text) for entity in detected] == [
        (expected_label, expected_text)
    ]


@pytest.mark.parametrize(
    "text",
    (
        "Kipimo cha maabara 12345678 kilirudi kawaida.",
        "Dozi 987654321 ilirekodiwa bila nambari ya bima.",
        "Nambari 19791103123456789012 ilirekodiwa kama thamani ya maabara.",
    ),
)
def test_ambiguous_bare_numbers_do_not_become_swahili_identifiers(text: str):
    identifiers = [
        entity
        for entity in safety_sweep(text, [], lang="sw")
        if entity.label == "national_id"
    ]
    assert identifiers == []


def test_synthetic_registers_have_exact_offsets():
    fixtures = _fixtures()
    assert [fixture.metadata["register"] for fixture in fixtures] == [
        "swahili_only",
        "english_swahili_code_switched",
        "sheng_style_mixed",
    ]

    expected_spans = {
        "golden-i18n-sw-swahili-clinical-pii": {
            ("PERSON", 6, 18, "Amina Hassan"),
            ("DATE", 39, 49, "14/05/1988"),
            ("ID_NUM", 75, 83, "12345678"),
            ("ID_NUM", 101, 110, "987654321"),
            ("PHONE", 117, 133, "+254 712 345 678"),
            ("AGE", 149, 151, "38"),
            ("STREET_ADDRESS", 161, 179, "Kenyatta Avenue 12"),
            ("ZIPCODE", 197, 202, "00100"),
        },
        "golden-i18n-sw-code-switched-clinical-pii": {
            ("PERSON", 14, 27, "Daniel Otieno"),
            ("DATE", 48, 58, "03/11/1979"),
            ("ID_NUM", 72, 80, "87654321"),
            ("ID_NUM", 98, 121, "19791103-12345-67890-12"),
            ("PHONE", 128, 144, "+255 754 321 098"),
        },
        "golden-i18n-sw-sheng-clinical-pii": {
            ("PERSON", 6, 19, "Wanjiku Njeri"),
            ("AGE", 47, 49, "29"),
            ("ID_NUM", 61, 69, "11223344"),
            ("ID_NUM", 84, 93, "246813579"),
            ("PHONE", 104, 120, "+256 772 456 789"),
        },
    }

    for fixture in fixtures:
        observed = {
            (span.label, span.start, span.end, span.text) for span in fixture.gold_spans
        }
        assert observed == expected_spans[fixture.fixture_id]
        assert all(
            fixture.text[span.start : span.end] == span.text
            for span in fixture.gold_spans
        )


def test_code_switched_fixture_detects_swahili_and_embedded_english_spans():
    fixture = _fixtures()[1]
    detected = safety_sweep(fixture.text, [], lang="sw")

    assert {(entity.start, entity.end, entity.text) for entity in detected} == {
        (span.start, span.end, span.text) for span in fixture.gold_spans
    }


def test_swahili_fixtures_score_zero_leakage_and_exact_spans():
    benchmark_fixtures = [fixture.to_benchmark_fixture() for fixture in _fixtures()]

    def offline_runner(fixture, model_name, device):
        assert model_name == "offline-swahili-patterns"
        assert device == "cpu"
        return safety_sweep(fixture.text, [], lang="sw")

    report = harness.run_benchmark(
        benchmark_fixtures,
        suite="swahili-code-mixed",
        model_name="offline-swahili-patterns",
        runner=offline_runner,
        generated_at="2026-07-16T00:00:00Z",
    )

    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0
    assert report.metrics["recall_slices"]["by_language"]["sw"] == 1.0


def test_swahili_fixtures_deidentify_offline_with_no_residuals():
    for fixture in _fixtures():
        empty_result = PredictionResult(
            text=fixture.text,
            entities=[],
            model_name="offline-swahili-patterns",
            timestamp="2026-07-16T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="sw",
        )
        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="sw",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert span.text not in result.deidentified_text


def test_swahili_pattern_lookup_keeps_universal_patterns():
    patterns = get_patterns_for_language("sw")
    assert all(pattern in patterns for pattern in LANGUAGE_PII_PATTERNS["sw"])
    assert any(pattern.entity_type == "email" for pattern in patterns)
