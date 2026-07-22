"""Amharic PII, Ethiopic numeral, and leakage regression coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.core.anonymizer.locales import FAKER_BACKEND_LOCALE, LANG_TO_LOCALE
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
from openmed.core.script_detect import detect_script
from openmed.eval import harness
from openmed.eval.golden import GoldenFixture
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "openmed/eval/golden/fixtures/i18n/am.jsonl"
)


def _fixtures() -> list[GoldenFixture]:
    rows = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [GoldenFixture.from_mapping(row) for row in rows]


def test_amharic_language_pack_registries_and_script_correct_surrogates():
    assert "am" in SUPPORTED_LANGUAGES
    assert LANGUAGE_NAMES["am"] == "Amharic"
    assert LANGUAGE_MODEL_PREFIX["am"] == "Amharic-"
    assert DEFAULT_PII_MODELS["am"] == "OpenMed/privacy-filter-multilingual"
    assert LANGUAGE_PII_PATTERNS["am"]
    assert LANG_TO_LOCALE["am"] == "am_ET"
    assert FAKER_BACKEND_LOCALE["am_ET"] == "en_KE"

    fake_data = LANGUAGE_FAKE_DATA["am"]
    assert {"አዲስ አበባ", "ባሕር ዳር", "ሀዋሳ"} <= set(fake_data["LOCATION"])
    assert all(detect_script(name) == "Ethiopic" for name in fake_data["NAME"])
    assert all(
        detect_script(location) == "Ethiopic" for location in fake_data["LOCATION"]
    )


@pytest.mark.parametrize(
    ("text", "expected_label", "expected_text"),
    (
        ("ስም፡ ሰላም፟ ተስፋዬ።", "name", "ሰላም፟ ተስፋዬ"),
        ("የትውልድ ቀን፡ ፲፬/፭/፲፱፻፹፰።", "date", "፲፬/፭/፲፱፻፹፰"),
        ("የትውልድ ቀን፡ 14/05/1988።", "date", "14/05/1988"),
        ("ዕድሜ፡ ፴፭።", "age", "፴፭"),
        ("ዕድሜ፡ 42።", "age", "42"),
        ("የፋይዳ መለያ ቁጥር፡ 123456789012።", "national_id", "123456789012"),
        ("መለያ ቁጥር፡ ፩፪፫፬፭፮፯፰፱።", "national_id", "፩፪፫፬፭፮፯፰፱"),
        ("ስልክ፡ +251 911 234 567።", "phone_number", "+251 911 234 567"),
        ("ስልክ፡ 0911 765 432።", "phone_number", "0911 765 432"),
        ("አድራሻ፡ አዲስ አበባ ቦሌ 12።", "street_address", "አዲስ አበባ ቦሌ 12"),
    ),
)
def test_amharic_context_detects_structured_pii_between_ethiopic_boundaries(
    text: str,
    expected_label: str,
    expected_text: str,
):
    detected = safety_sweep(text, [], lang="am")

    assert [(entity.label, entity.text) for entity in detected] == [
        (expected_label, expected_text)
    ]


def test_ethiopic_native_patterns_do_not_request_case_insensitive_matching():
    case_insensitive_patterns = [
        pattern
        for pattern in LANGUAGE_PII_PATTERNS["am"]
        if pattern.flags & re.IGNORECASE
    ]

    assert len(case_insensitive_patterns) == 1
    assert "Patient name" in case_insensitive_patterns[0].pattern


def test_fayda_format_requires_identity_context_and_exactly_twelve_digits():
    for text in (
        "የላብራቶሪ ውጤት 123456789012 ተመዝግቧል።",
        "የፋይዳ መለያ ቁጥር፡ 12345678901።",
        "የፋይዳ መለያ ቁጥር፡ 1234567890123።",
    ):
        identifiers = [
            entity
            for entity in safety_sweep(text, [], lang="am")
            if entity.label == "national_id"
        ]
        assert identifiers == []


def test_synthetic_amharic_registers_have_exact_offsets():
    fixtures = _fixtures()
    assert [fixture.metadata["register"] for fixture in fixtures] == [
        "amharic_only",
        "amharic_latin_mixed",
    ]

    for fixture in fixtures:
        detected = safety_sweep(fixture.text, [], lang="am")
        observed = {(entity.start, entity.end, entity.text) for entity in detected}
        expected = {(span.start, span.end, span.text) for span in fixture.gold_spans}

        assert observed == expected
        assert all(
            fixture.text[span.start : span.end] == span.text
            for span in fixture.gold_spans
        )


def test_amharic_fixtures_score_zero_leakage_and_exact_spans():
    benchmark_fixtures = [fixture.to_benchmark_fixture() for fixture in _fixtures()]

    def offline_runner(fixture, model_name, device):
        assert model_name == "offline-amharic-patterns"
        assert device == "cpu"
        return safety_sweep(fixture.text, [], lang="am")

    report = harness.run_benchmark(
        benchmark_fixtures,
        suite="amharic-mixed",
        model_name="offline-amharic-patterns",
        runner=offline_runner,
        generated_at="2026-07-16T00:00:00Z",
    )

    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0
    assert report.metrics["recall_slices"]["by_language"]["am"] == 1.0


def test_amharic_fixtures_deidentify_offline_with_no_residuals():
    for fixture in _fixtures():
        empty_result = PredictionResult(
            text=fixture.text,
            entities=[],
            model_name="offline-amharic-patterns",
            timestamp="2026-07-16T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="am",
        )
        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="am",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert span.text not in result.deidentified_text


def test_amharic_pattern_lookup_keeps_universal_patterns():
    patterns = get_patterns_for_language("am")

    assert all(pattern in patterns for pattern in LANGUAGE_PII_PATTERNS["am"])
    assert any(pattern.entity_type == "email" for pattern in patterns)
