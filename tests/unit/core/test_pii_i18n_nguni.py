"""isiZulu/isiXhosa PII, South African ID, and leakage regressions."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import (
    FAKER_BACKEND_LOCALE,
    LANG_TO_LOCALE,
    NATIONAL_ID_PROVIDERS,
)
from openmed.core.anonymizer.providers.clinical_ids import generate_za_id_number
from openmed.core.anonymizer.providers.registry_ids import get_national_id
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
    validate_za_id_number,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.eval import harness
from openmed.eval.golden import GoldenFixture
from openmed.processing.outputs import PredictionResult

FIXTURE_PATHS = (
    Path("openmed/eval/golden/fixtures/i18n/zu.jsonl"),
    Path("openmed/eval/golden/fixtures/i18n/xh.jsonl"),
)


def _fixtures() -> list[GoldenFixture]:
    rows = [
        json.loads(line)
        for path in FIXTURE_PATHS
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [GoldenFixture.from_mapping(row) for row in rows]


@pytest.mark.parametrize(
    ("lang", "name", "locale"),
    (("zu", "isiZulu", "zu_ZA"), ("xh", "isiXhosa", "xh_ZA")),
)
def test_nguni_language_pack_registries_and_surrogate_data(
    lang: str,
    name: str,
    locale: str,
):
    assert lang in SUPPORTED_LANGUAGES
    assert LANGUAGE_NAMES[lang] == name
    assert LANGUAGE_MODEL_PREFIX[lang] == f"{name}-"
    assert DEFAULT_PII_MODELS[lang] == "OpenMed/privacy-filter-multilingual"
    assert LANGUAGE_PII_PATTERNS[lang]
    assert LANG_TO_LOCALE[lang] == locale

    fake_data = LANGUAGE_FAKE_DATA[lang]
    assert {"Nomcebo", "Xolani", "Qhawe"} <= set(fake_data["FIRST_NAME"])
    assert {"Durban", "Umlazi", "East London", "Gqeberha"} <= set(fake_data["LOCATION"])
    assert all(
        len(postcode) == 4 and postcode.isdigit() for postcode in fake_data["ZIPCODE"]
    )


@pytest.mark.parametrize(
    "valid_id",
    ("8001015009087", "9003030123082", "7903116001080", "0102034000186"),
)
def test_validate_za_id_number_accepts_valid_shape_date_citizenship_and_luhn(
    valid_id: str,
):
    assert validate_za_id_number(valid_id)


@pytest.mark.parametrize(
    "invalid_id",
    (
        "8001015009086",  # invalid Luhn check digit
        "8013015009085",  # invalid month
        "8001325009083",  # invalid day
        "8001015009285",  # invalid citizenship digit
        "80010150090870",  # wrong length
        "８００１０１５００９０８７",  # non-ASCII digits
    ),
)
def test_validate_za_id_number_rejects_invalid_candidates(invalid_id: str):
    assert not validate_za_id_number(invalid_id)


@pytest.mark.parametrize("lang", ("zu", "xh"))
def test_generated_sa_id_surrogates_round_trip_for_both_languages(lang: str):
    locale, method = NATIONAL_ID_PROVIDERS[lang]
    assert method == "south_african_id"
    if lang == "xh":
        assert FAKER_BACKEND_LOCALE[locale] == "zu_ZA"

    spec = get_national_id(lang, "sa_id_number")
    assert spec is not None
    assert spec.faker_method == method

    for seed in range(40):
        direct = generate_za_id_number()
        assert validate_za_id_number(direct)

        anonymizer = Anonymizer(lang=lang, consistent=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate = anonymizer.surrogate(
                "8001015009087",
                "national_id",
                locale=locale,
            )
        assert spec.validate(surrogate)


@pytest.mark.parametrize("lang", ("zu", "xh"))
def test_validator_gates_sa_id_detection_both_ways(lang: str):
    valid = safety_sweep("Inombolo kamazisi 8001015009087.", [], lang=lang)
    invalid = safety_sweep("Inombolo kamazisi 8001015009086.", [], lang=lang)

    assert [(entity.label, entity.text) for entity in valid] == [
        ("national_id", "8001015009087")
    ]
    assert [entity for entity in invalid if entity.label == "national_id"] == []


@pytest.mark.parametrize(
    ("lang", "text", "expected_label", "expected_text"),
    (
        ("zu", "Igama lesiguli: Nomcebo Dlamini.", "name", "Nomcebo Dlamini"),
        ("xh", "Igama lesigulane: Xolani Qwabe.", "name", "Xolani Qwabe"),
        (
            "zu",
            "Inombolo yosizo lwezempilo GEMS-48201973.",
            "national_id",
            "GEMS-48201973",
        ),
        (
            "xh",
            "Inombolo yoncedo lwezonyango BON-765432109.",
            "national_id",
            "BON-765432109",
        ),
        ("zu", "Ucingo +27 82 123 4567.", "phone_number", "+27 82 123 4567"),
        ("xh", "Ifowuni 071 234 5678.", "phone_number", "071 234 5678"),
        ("zu", "Iminyaka 38.", "age", "38"),
        ("xh", "Umhla wokuzalwa 03/11/1979.", "date", "03/11/1979"),
        ("zu", "Ikheli: 12 Umgeni Road", "street_address", "12 Umgeni Road"),
        ("xh", "Idilesi: 18 Oxford Street", "street_address", "18 Oxford Street"),
        ("zu", "ikhodi yeposi 4001", "postcode", "4001"),
        ("xh", "ikhowudi yeposi 5201", "postcode", "5201"),
    ),
)
def test_bilingual_context_detects_nguni_structured_pii(
    lang: str,
    text: str,
    expected_label: str,
    expected_text: str,
):
    detected = safety_sweep(text, [], lang=lang)
    assert [(entity.label, entity.text) for entity in detected] == [
        (expected_label, expected_text)
    ]


def test_medical_aid_membership_format_requires_explicit_context():
    detected = safety_sweep(
        "Umphumela welabhorethri GEMS-48201973 ubhalwe phansi.",
        [],
        lang="zu",
    )
    assert [entity for entity in detected if entity.label == "national_id"] == []


def test_synthetic_nguni_registers_and_click_names_have_exact_offsets():
    fixtures = _fixtures()
    assert [fixture.metadata["register"] for fixture in fixtures] == [
        "isizulu_only",
        "isixhosa_only",
        "english_nguni_code_switched",
    ]

    for fixture in fixtures:
        detected = safety_sweep(fixture.text, [], lang=fixture.language)
        observed = {(entity.start, entity.end, entity.text) for entity in detected}
        expected = {(span.start, span.end, span.text) for span in fixture.gold_spans}
        assert observed == expected
        assert all(
            fixture.text[span.start : span.end] == span.text
            for span in fixture.gold_spans
        )

    click_names = {fixture.gold_spans[0].text for fixture in fixtures}
    assert click_names == {"Nomcebo Dlamini", "Xolani Qwabe", "Qhawe Ndlovu"}


def test_click_name_replacement_is_consistent_and_removes_original_spans():
    text = "Igama lesigulane: Qhawe Ndlovu. Patient name: Qhawe Ndlovu."
    detected = safety_sweep(text, [], lang="xh")
    name_spans = [entity for entity in detected if entity.label == "name"]
    assert [(entity.start, entity.end, entity.text) for entity in name_spans] == [
        (18, 30, "Qhawe Ndlovu"),
        (46, 58, "Qhawe Ndlovu"),
    ]

    prediction = PredictionResult(
        text=text,
        entities=detected,
        model_name="offline-nguni-patterns",
        timestamp="2026-07-16T00:00:00Z",
        metadata={},
    )
    result = _build_deidentification_result(
        text,
        prediction,
        effective_method="replace",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="xh",
        consistent=True,
        seed=845,
        locale=None,
        use_safety_sweep=True,
    )
    assert "Qhawe Ndlovu" not in result.deidentified_text
    assert len(result.pii_entities) == 2
    assert result.pii_entities[0].surrogate == result.pii_entities[1].surrogate


def test_nguni_fixtures_score_zero_leakage_and_exact_spans():
    benchmark_fixtures = [fixture.to_benchmark_fixture() for fixture in _fixtures()]

    def offline_runner(fixture, model_name, device):
        assert model_name == "offline-nguni-patterns"
        assert device == "cpu"
        return safety_sweep(fixture.text, [], lang=fixture.language)

    report = harness.run_benchmark(
        benchmark_fixtures,
        suite="nguni-code-mixed",
        model_name="offline-nguni-patterns",
        runner=offline_runner,
        generated_at="2026-07-16T00:00:00Z",
    )

    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0
    by_language = report.metrics["recall_slices"]["by_language"]
    assert by_language["xh"] == 1.0
    assert by_language["zu"] == 1.0


def test_nguni_fixtures_deidentify_offline_with_no_residuals():
    for fixture in _fixtures():
        empty_result = PredictionResult(
            text=fixture.text,
            entities=[],
            model_name="offline-nguni-patterns",
            timestamp="2026-07-16T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang=fixture.language,
        )
        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang=fixture.language,
            consistent=False,
            seed=None,
            locale=None,
            use_safety_sweep=True,
        )

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert span.text not in result.deidentified_text


@pytest.mark.parametrize("lang", ("zu", "xh"))
def test_nguni_pattern_lookup_keeps_universal_patterns(lang: str):
    patterns = get_patterns_for_language(lang)
    assert all(pattern in patterns for pattern in LANGUAGE_PII_PATTERNS[lang])
    assert any(pattern.entity_type == "email" for pattern in patterns)
