"""Afrikaans PII transfer, South African ID, and leakage regressions.

The pack deliberately transfers only Dutch-compatible date and address
structure. Afrikaans context replaces Dutch cue words, while South African ID,
phone, and postcode rules replace Dutch BSN, +31 phone, and postcode rules.
This is the regression template for future related-language pattern transfers.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import (
    FAKER_BACKEND_LOCALE,
    LANG_TO_LOCALE,
    NATIONAL_ID_PROVIDERS,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.pii import (
    _apply_safety_sweep_to_result,
    _build_deidentification_result,
)
from openmed.core.pii_i18n import (
    LANGUAGE_FAKE_DATA,
    LANGUAGE_MONTH_NAMES,
    LANGUAGE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    get_patterns_for_language,
    validate_dutch_bsn,
    validate_za_id_number,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.eval import harness
from openmed.eval.golden import GoldenFixture
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = Path("openmed/eval/golden/fixtures/i18n/af.jsonl")


def _fixtures() -> list[GoldenFixture]:
    rows = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [GoldenFixture.from_mapping(row) for row in rows]


def test_afrikaans_deterministic_pack_and_surrogate_data_are_registered():
    assert "af" in NATIONAL_ID_ONLY_LANGUAGES
    assert LANGUAGE_PII_PATTERNS["af"]
    assert LANG_TO_LOCALE["af"] == "af_ZA"
    assert FAKER_BACKEND_LOCALE["af_ZA"] == "nl_NL"
    assert LANGUAGE_MONTH_NAMES["af"] == [
        "Januarie",
        "Februarie",
        "Maart",
        "April",
        "Mei",
        "Junie",
        "Julie",
        "Augustus",
        "September",
        "Oktober",
        "November",
        "Desember",
    ]

    fake_data = LANGUAGE_FAKE_DATA["af"]
    assert {
        "Johan van der Merwe",
        "Annelie Botha",
        "Pieter du Plessis",
    } <= set(fake_data["NAME"])
    assert {
        "Kaapstad",
        "Pretoria",
        "Bloemfontein",
        "Stellenbosch",
    } <= set(fake_data["LOCATION"])


def test_afrikaans_generic_surrogates_use_dutch_backend():
    afrikaans = Anonymizer(lang="af", consistent=True, seed=865)
    dutch = Anonymizer(lang="nl", consistent=True, seed=865)

    for label in ("NAME", "LOCATION", "STREET_ADDRESS"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            afrikaans_surrogate = afrikaans.surrogate(f"source-{label}", label)
            dutch_surrogate = dutch.surrogate(f"source-{label}", label)
        assert afrikaans_surrogate == dutch_surrogate


def test_generated_sa_id_surrogates_round_trip_for_afrikaans():
    locale, method = NATIONAL_ID_PROVIDERS["af"]
    assert (locale, method) == ("af_ZA", "south_african_id")

    spec = get_national_id("af", "sa_id_number")
    assert spec is not None
    assert spec.faker_method == method

    for seed in range(40):
        anonymizer = Anonymizer(lang="af", consistent=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate = anonymizer.surrogate(
                "8001015009087",
                "national_id",
                locale=locale,
            )
        assert validate_za_id_number(surrogate)
        assert spec.validate(surrogate)


def test_afrikaans_patterns_exclude_dutch_bsn_and_numbering_plans():
    assert validate_dutch_bsn("111222333")
    validators = {
        pattern.validator
        for pattern in LANGUAGE_PII_PATTERNS["af"]
        if pattern.validator is not None
    }
    assert validate_dutch_bsn not in validators

    text = (
        "Lêernommer 111222333. Nederlandse telefoon +31 6 12345678. "
        "Nederlandse poskode 1234 AB."
    )
    detected = safety_sweep(text, [], lang="af")
    assert [
        entity
        for entity in detected
        if entity.label in {"national_id", "phone_number", "postcode"}
    ] == []


def test_dutch_patterns_still_detect_dutch_bsn_phone_and_postcode():
    text = "BSN 111222333. Telefoon +31 6 12345678. Postcode 1234 AB."
    detected = safety_sweep(text, [], lang="nl")
    observed = {(entity.label, entity.text) for entity in detected}

    assert ("national_id", "111222333") in observed
    assert ("phone_number", "+31 6 12345678") in observed
    assert ("postcode", "1234 AB") in observed


def test_synthetic_afrikaans_fixtures_have_exact_offline_spans():
    fixtures = _fixtures()
    assert [fixture.metadata["register"] for fixture in fixtures] == [
        "afrikaans_clinical",
        "afrikaans_prose",
        "english_afrikaans_code_switched",
    ]

    for fixture in fixtures:
        detected = safety_sweep(fixture.text, [], lang="af")
        observed = {(entity.start, entity.end, entity.text) for entity in detected}
        expected = {(span.start, span.end, span.text) for span in fixture.gold_spans}
        assert observed == expected
        assert all(
            fixture.text[span.start : span.end] == span.text
            for span in fixture.gold_spans
        )


def test_synthetic_afrikaans_fixtures_score_zero_leakage():
    benchmark_fixtures = [fixture.to_benchmark_fixture() for fixture in _fixtures()]

    def offline_runner(fixture, model_name, device):
        assert model_name == "offline-afrikaans-patterns"
        assert device == "cpu"
        return safety_sweep(fixture.text, [], lang=fixture.language)

    report = harness.run_benchmark(
        benchmark_fixtures,
        suite="afrikaans-related-language-transfer",
        model_name="offline-afrikaans-patterns",
        runner=offline_runner,
        generated_at="2026-07-25T00:00:00Z",
    )

    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0
    assert report.metrics["recall_slices"]["by_language"]["af"] == 1.0


def test_synthetic_afrikaans_fixtures_deidentify_without_residuals():
    for fixture in _fixtures():
        empty_result = PredictionResult(
            text=fixture.text,
            entities=[],
            model_name="offline-afrikaans-patterns",
            timestamp="2026-07-25T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="af",
        )
        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="af",
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert span.text not in result.deidentified_text


def test_afrikaans_pattern_lookup_keeps_universal_patterns():
    patterns = get_patterns_for_language("af")
    assert all(pattern in patterns for pattern in LANGUAGE_PII_PATTERNS["af"])
    assert any(pattern.entity_type == "email" for pattern in patterns)
