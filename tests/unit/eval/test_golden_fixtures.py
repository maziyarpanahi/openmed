"""Tests for synthetic golden de-identification fixtures."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from openmed.core.decoding.spans import is_grapheme_boundary
from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_entity_merger import find_semantic_units, validate_luhn
from openmed.core.pii_i18n import (
    INDIC_NER_LANGUAGES,
    LANGUAGE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    normalize_kannada_digits,
    validate_aadhaar,
    validate_czechoslovak_rodne_cislo,
    validate_danish_cpr,
    validate_hungarian_taj,
    validate_israeli_teudat_zehut,
    validate_kannada_aadhaar,
    validate_kannada_indian_phone,
    validate_karnataka_pin,
    validate_latvian_personas_kods,
    validate_malaysian_mykad,
    validate_philhealth_pin,
    validate_philsys_psn,
    validate_portuguese_cpf,
    validate_romanian_cnp,
)
from openmed.eval import harness
from openmed.eval.golden import (
    CRITICAL_FINDINGS_CATEGORY,
    GOLDEN_CATEGORIES,
    HARD_NEGATIVE_CATEGORY,
    GoldenFixture,
    fixture_languages,
    fixtures_by_category,
    fixtures_by_language,
    list_fixture_paths,
    load_benchmark_fixtures,
    load_golden_fixtures,
)
from openmed.eval.metrics import (
    CRITICAL_FINDING_CATEGORIES,
    compute_date_shift_consistency,
)

EXPANDED_MULTILINGUAL_LANGUAGES = ("ar", "ja", "tr")


def test_golden_directory_documents_synthetic_only_no_dua():
    readme = Path("openmed/eval/golden/README.md").read_text(encoding="utf-8").lower()

    assert "synthetic-only" in readme
    assert "no dua" in readme
    assert "no real phi" in readme


def test_golden_fixtures_cover_required_categories_and_languages():
    fixtures = load_golden_fixtures()
    grouped = fixtures_by_category(fixtures)
    multilingual_languages = fixture_languages(fixtures, category="multilingual")

    assert set(grouped) == set(GOLDEN_CATEGORIES)
    assert SUPPORTED_LANGUAGES.issubset(multilingual_languages)
    assert multilingual_languages <= (
        SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | INDIC_NER_LANGUAGES
    )

    multilingual = grouped["multilingual"]
    assert len(multilingual) >= len(SUPPORTED_LANGUAGES)
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)


def test_expanded_multilingual_fixtures_cover_person_date_and_locale_id():
    grouped = fixtures_by_language(
        load_golden_fixtures(),
        category="multilingual",
    )

    assert set(EXPANDED_MULTILINGUAL_LANGUAGES).issubset(grouped)
    for language in EXPANDED_MULTILINGUAL_LANGUAGES:
        # The OM-019 expanded fixtures live in multilingual.json; the per-language
        # OM-100 i18n fixtures (golden-i18n-*) are a separate multilingual set.
        expanded = [
            fixture
            for fixture in grouped[language]
            if fixture.fixture_id.startswith("golden-multilingual-")
        ]
        assert len(expanded) == 1
        fixture = expanded[0]
        spans_by_label = {span.label: span for span in fixture.gold_spans}

        assert list(spans_by_label) == ["PERSON", "DATE", "ID_NUM"]
        assert fixture.metadata["locale"]
        assert "[PERSON]" in fixture.expected_output["text"]
        assert "[DATE]" in fixture.expected_output["text"]
        assert "[ID_NUM]" in fixture.expected_output["text"]
        assert (
            spans_by_label["ID_NUM"].metadata["identifier_type"]
            == fixture.metadata["identifier_type"]
        )


def test_expanded_multilingual_fixtures_run_through_harness_scoring():
    grouped = fixtures_by_language(
        load_golden_fixtures(),
        category="multilingual",
    )
    benchmark_fixtures = [
        next(
            fixture
            for fixture in grouped[language]
            if fixture.fixture_id.startswith("golden-multilingual-")
        ).to_benchmark_fixture()
        for language in EXPANDED_MULTILINGUAL_LANGUAGES
    ]

    def exact_gold_runner(fixture, model_name, device):
        assert model_name == "golden-test-model"
        assert device == "cpu"
        return fixture.gold_spans

    report = harness.run_benchmark(
        benchmark_fixtures,
        suite="golden-multilingual",
        model_name="golden-test-model",
        runner=exact_gold_runner,
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.fixture_count == len(EXPANDED_MULTILINGUAL_LANGUAGES)
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["exact_span_f1"]["f1"] == 1.0
    for language in EXPANDED_MULTILINGUAL_LANGUAGES:
        assert report.metrics["recall_slices"]["by_language"][language] == 1.0


def test_golden_fixtures_parse_offsets_expected_output_and_round_trip():
    seen_ids: set[str] = set()

    for fixture in load_golden_fixtures():
        assert fixture.fixture_id not in seen_ids
        seen_ids.add(fixture.fixture_id)
        assert fixture.expected_output["text"]
        assert fixture.expected_output["method"]
        assert fixture.gold_spans or fixture.category == HARD_NEGATIVE_CATEGORY

        for span in fixture.gold_spans:
            assert span.label in CANONICAL_LABELS
            assert fixture.text[span.start : span.end] == span.text

        mapping = fixture.to_mapping()
        assert GoldenFixture.from_mapping(mapping).to_mapping() == mapping


def test_golden_json_files_are_harness_loadable():
    for fixture_path in list_fixture_paths():
        loaded = harness.load_fixtures(fixture_path)
        assert loaded
        assert all(
            item.gold_spans or item.metadata["category"] == HARD_NEGATIVE_CATEGORY
            for item in loaded
        )
        assert all(item.metadata["expected_output"]["text"] for item in loaded)

    benchmark_fixtures = load_benchmark_fixtures()
    assert len(benchmark_fixtures) == len(load_golden_fixtures())
    assert all(item.metadata["category"] for item in benchmark_fixtures)


def test_critical_finding_fixture_is_synthetic_and_disclaimer_marked():
    fixtures = [
        fixture
        for fixture in load_golden_fixtures()
        if fixture.category == CRITICAL_FINDINGS_CATEGORY
    ]

    assert fixtures
    categories = set()
    for fixture in fixtures:
        disclaimer = fixture.metadata["medical_device_disclaimer"].lower()
        assert fixture.metadata["synthetic"] is True
        assert "assistive safety probe" in disclaimer
        assert "not clinical ground truth" in disclaimer
        for span in fixture.gold_spans:
            assert span.metadata["critical_finding"] is True
            assert span.metadata["fixture_id"] == fixture.fixture_id
            categories.add(span.metadata["critical_finding_category"])

    assert categories == set(CRITICAL_FINDING_CATEGORIES)


def test_hard_negative_fixtures_are_synthetic_zero_span_non_phi():
    fixtures = [
        fixture
        for fixture in load_golden_fixtures()
        if fixture.category == HARD_NEGATIVE_CATEGORY
    ]

    assert fixtures
    for fixture in fixtures:
        assert fixture.gold_spans == ()
        assert fixture.expected_output["method"] == "none"
        assert fixture.expected_output["text"] == fixture.text
        assert fixture.metadata["synthetic"] is True
        assert "dua" not in json.dumps(fixture.to_mapping()).lower()
        for candidate in fixture.metadata["hard_negative_candidates"]:
            assert candidate["synthetic"] is True
            assert candidate["label"] in CANONICAL_LABELS
            assert (
                fixture.text[candidate["start"] : candidate["end"]]
                == (candidate["text"])
            )


def test_kannada_i18n_fixtures_are_grapheme_safe_and_validator_equivalent():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/kn.jsonl")
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(fixtures) == 2
    assert {fixture.metadata["fixture_kind"] for fixture in fixtures} == {
        "Kannada-script native digits",
        "Kannada-English ASCII identifiers",
    }

    aadhaar_values = []
    phone_values = []
    pin_values = []
    for fixture in fixtures:
        person_spans = [span for span in fixture.gold_spans if span.label == "PERSON"]
        assert len(person_spans) == 1
        person = person_spans[0]
        assert len([token for token in person.text.split() if token.endswith(".")]) in {
            1,
            2,
        }
        assert fixture.text[person.end :].lstrip().startswith("ಅವರು")
        assert "ಅವರು" not in person.text

        for span in fixture.gold_spans:
            assert is_grapheme_boundary(span.start, fixture.text)
            assert is_grapheme_boundary(span.end, fixture.text)
            assert fixture.text[span.start : span.end] == span.text
            if span.label == "ID_NUM":
                aadhaar_values.append(span.text)
            elif span.label == "PHONE":
                phone_values.append(span.text)
            elif span.label == "ZIPCODE":
                pin_values.append(span.text)

    assert all(validate_kannada_aadhaar(value) for value in aadhaar_values)
    assert all(validate_kannada_indian_phone(value) for value in phone_values)
    assert all(validate_karnataka_pin(value) for value in pin_values)
    assert len({normalize_kannada_digits(value) for value in aadhaar_values}) == 1
    assert len({normalize_kannada_digits(value) for value in pin_values}) == 1


def test_kannada_name_patterns_use_all_honorifics_and_exclude_avaru_suffix():
    examples = (
        ("ಶ್ರೀ", "ಕೆ. ಎಸ್. ರವಿ"),
        ("ಶ್ರೀಮತಿ", "ಎಂ. ಲತಾ"),
        ("ಕುಮಾರಿ", "ಆರ್. ಅನಿತಾ"),
        ("ಡಾ.", "K. S. Ravi"),
    )

    for honorific, name in examples:
        text = f"ರೋಗಿ {honorific} {name} ಅವರು."
        units = find_semantic_units(text, LANGUAGE_PII_PATTERNS["kn"])
        names = [
            text[start:end]
            for start, end, entity_type, *_rest in units
            if entity_type == "name"
        ]
        assert names == [name]
        name_end = text.index(name) + len(name)
        assert text[name_end:].startswith(" ಅವರು")


def test_kannada_fixtures_pass_zero_leakage_release_gate_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.eval.release_gates import _per_language_residual_leakage_check
    from openmed.processing.outputs import PredictionResult

    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/kn.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    predictions = {}

    for fixture in fixtures:
        empty_result = PredictionResult(
            text=fixture.text,
            entities=[],
            model_name="offline-safety-sweep",
            timestamp="2026-07-24T00:00:00Z",
            metadata={},
        )
        swept_result, _added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="kn",
        )
        predictions[fixture.fixture_id] = swept_result.entities
        observed = {
            (entity.start, entity.end, normalize_label(entity.label, "kn"))
            for entity in swept_result.entities
        }

        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label) in observed

        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="kn",
            consistent=False,
            seed=None,
            locale="kn_IN",
            use_safety_sweep=True,
        )
        assert all(
            span.text not in result.deidentified_text for span in fixture.gold_spans
        )
        assert "ಅವರು" in result.deidentified_text

    report = harness.run_benchmark(
        [fixture.to_benchmark_fixture() for fixture in fixtures],
        suite="golden-kannada",
        model_name="offline-safety-sweep",
        runner=lambda fixture, _model_name, _device: predictions[fixture.fixture_id],
        generated_at="2026-07-24T00:00:00Z",
    )
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["leakage"]["by_language"]["kn"] == 0.0

    gate = _per_language_residual_leakage_check(report.metrics, report.metadata)
    assert gate.passed is True
    assert gate.details["evaluated"] == {"kn": 0.0}


def test_hebrew_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/he.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "he"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "15/03/1985"
    assert gold_by_label["PHONE"] == "+972 54-123-4567"
    assert gold_by_label["ZIPCODE"] == "6423905"
    assert gold_by_label["STREET_ADDRESS"] == "רחוב הרצל 12"
    assert validate_israeli_teudat_zehut(gold_by_label["ID_NUM"])


def test_latvian_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/lv.jsonl")

    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "lv"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "16.11.1975"
    assert gold_by_label["PHONE"] == "+371 2123 4567"
    assert gold_by_label["ZIPCODE"] == "LV-1010"
    assert gold_by_label["STREET_ADDRESS"] == "Brivibas iela 12"
    assert validate_latvian_personas_kods(gold_by_label["ID_NUM"])


def test_slovak_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/sk.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "sk"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "05.05.1985"
    assert gold_by_label["PHONE"] == "+421 903 123 456"
    assert gold_by_label["ZIPCODE"] == "81101"
    assert gold_by_label["STREET_ADDRESS"] == "Hlavna ulica 12"
    assert validate_czechoslovak_rodne_cislo(gold_by_label["ID_NUM"])


def test_hungarian_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/hu.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "hu"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "1985. május 5."
    assert gold_by_label["PHONE"] == "+36 30 123 4567"
    assert gold_by_label["ZIPCODE"] == "1051"
    assert gold_by_label["STREET_ADDRESS"] == "Kossuth Lajos utca 12"
    assert validate_hungarian_taj(gold_by_label["ID_NUM"])


def test_hungarian_i18n_jsonl_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/hu.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    empty_result = PredictionResult(
        text=fixture.text,
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
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


def test_czech_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/cs.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "cs"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "16.11.1975"
    assert gold_by_label["PHONE"] == "+420 601 234 567"
    assert gold_by_label["ZIPCODE"] == "110 00"
    assert gold_by_label["STREET_ADDRESS"] == "Vodickova ulice 12"
    assert validate_czechoslovak_rodne_cislo(gold_by_label["ID_NUM"])


def test_czech_i18n_jsonl_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/cs.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    empty_result = PredictionResult(
        text=fixture.text,
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-14T00:00:00Z",
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


def test_romanian_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ro.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    fixtures = {row["id"]: GoldenFixture.from_mapping(row) for row in rows}
    fixture = fixtures["golden-i18n-ro-clinical-pii"]
    assert fixture.language == "ro"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "12 martie 1985"
    assert gold_by_label["PHONE"] == "+40 721 234 567"
    assert gold_by_label["ZIPCODE"] == "010011"
    assert gold_by_label["STREET_ADDRESS"] == "Str. Mihai Eminescu 12"
    assert validate_romanian_cnp(gold_by_label["ID_NUM"])

    diacritic_fixture = fixtures["golden-i18n-ro-diacritics"]
    diacritic_by_label = {
        span.label: span.text for span in diacritic_fixture.gold_spans
    }
    assert "Pacientă" in diacritic_fixture.text
    assert "București" in diacritic_fixture.text
    assert diacritic_by_label["DATE"] == "22 iulie 2005"
    assert diacritic_by_label["PHONE"] == "0721 234 567"
    assert diacritic_by_label["STREET_ADDRESS"] == "Șoseaua Ștefan cel Mare 15"
    assert diacritic_by_label["ZIPCODE"] == "010101"
    assert validate_romanian_cnp(diacritic_by_label["ID_NUM"])


def test_malay_i18n_jsonl_fixture_offsets_and_checksum():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ms.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "ms"

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert gold_by_label["DATE"] == "17/08/1985"
    assert gold_by_label["PHONE"] == "+60 12-345 6789"
    assert gold_by_label["STREET_ADDRESS"] == "Jalan Merdeka 10"
    assert validate_malaysian_mykad(gold_by_label["ID_NUM"])


def test_malay_i18n_jsonl_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ms.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    empty_result = PredictionResult(
        text=fixture.text,
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-02T00:00:00Z",
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


def test_tagalog_i18n_jsonl_fixture_offsets_and_ids():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/tl.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "tl"

    spans = {
        (span.label, span.start, span.end, span.text) for span in fixture.gold_spans
    }
    assert spans == {
        ("DATE", 34, 44, "17/08/1985"),
        ("PHONE", 55, 71, "+63 917 123 4567"),
        ("ID_NUM", 77, 91, "1234-5678-9012"),
        ("ID_NUM", 104, 118, "98-765432109-8"),
        ("STREET_ADDRESS", 128, 145, "Barangay Maligaya"),
    }

    ids_by_type = {
        span.metadata["identifier_type"]: span.text
        for span in fixture.gold_spans
        if span.label == "ID_NUM"
    }
    assert validate_philsys_psn(ids_by_type["philsys_psn"])
    assert validate_philhealth_pin(ids_by_type["philhealth_pin"])


def test_tagalog_i18n_jsonl_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/tl.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    empty_result = PredictionResult(
        text=fixture.text,
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-03T00:00:00Z",
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


def test_danish_i18n_jsonl_fixture_offsets_and_cpr():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/da.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    assert fixture.language == "da"

    spans = {
        (span.label, span.start, span.end, span.text) for span in fixture.gold_spans
    }
    assert spans == {
        ("DATE", 26, 36, "1985-08-17"),
        ("PHONE", 46, 61, "+45 20 12 34 56"),
        ("ID_NUM", 67, 78, "170885-1234"),
        ("STREET_ADDRESS", 88, 103, "Nørrebrogade 12"),
        ("ZIPCODE", 105, 109, "2200"),
    }

    gold_by_label = {span.label: span.text for span in fixture.gold_spans}
    assert validate_danish_cpr(gold_by_label["ID_NUM"])


def test_danish_i18n_jsonl_fixture_deidentifies_with_no_leakage_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.processing.outputs import PredictionResult

    fixture_path = Path("openmed/eval/golden/fixtures/i18n/da.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    fixture = GoldenFixture.from_mapping(rows[0])
    empty_result = PredictionResult(
        text=fixture.text,
        entities=[],
        model_name="offline-safety-sweep",
        timestamp="2026-07-03T00:00:00Z",
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


def test_nested_overlap_fixture_asserts_resolution_not_just_detection():
    fixture = _one("nested_overlapping")

    assert _has_overlap(fixture.gold_spans)

    expected_spans = fixture.metadata["resolution"]["expected_spans"]
    assert not _has_overlap(expected_spans)
    assert [span["label"] for span in expected_spans] == ["PERSON", "EMAIL"]
    assert fixture.expected_output["text"] == (
        "Synthetic patient [PERSON] uses [EMAIL] in Clinic Alpha."
    )


def test_chunk_boundary_fixture_crosses_max_length_window_and_keeps_global_offsets():
    fixture = _one("chunk_boundary")
    span = fixture.gold_spans[0]
    chunk_window = fixture.metadata["chunk_window"]
    max_length = chunk_window["max_length"]

    assert span.start < max_length < span.end
    assert chunk_window["crosses_boundary"] is True
    assert chunk_window["expected_global_start"] == span.start
    assert chunk_window["expected_global_end"] == span.end


def test_checksum_fixture_has_valid_gold_ids_and_invalid_hard_negatives():
    matches = [
        fixture
        for fixture in load_golden_fixtures()
        if fixture.fixture_id == "golden-checksum-valid-invalid-identifiers"
    ]
    assert len(matches) == 1
    fixture = matches[0]
    gold_by_type = {
        span.metadata["identifier_type"]: span.text for span in fixture.gold_spans
    }
    hard_negatives = {
        item["identifier_type"]: item["text"]
        for item in fixture.metadata["hard_negatives"]
    }

    assert validate_luhn(gold_by_type["credit_card"])
    assert not validate_luhn(hard_negatives["credit_card"])
    assert validate_portuguese_cpf(gold_by_type["cpf"])
    assert not validate_portuguese_cpf(hard_negatives["cpf"])
    assert validate_aadhaar(gold_by_type["aadhaar"])
    assert not validate_aadhaar(hard_negatives["aadhaar"])

    for invalid_text in hard_negatives.values():
        assert invalid_text in fixture.text
        assert invalid_text in fixture.expected_output["text"]
        assert all(span.text != invalid_text for span in fixture.gold_spans)


def test_date_arithmetic_fixture_preserves_intervals_after_shift_dates():
    fixture = _one("date_arithmetic")
    date_chain = fixture.metadata["date_chain"]
    original_dates = date_chain["original_dates"]
    shifted_dates = date_chain["shifted_dates"]

    assert compute_date_shift_consistency(original_dates, shifted_dates).score == 1.0
    assert _interval_days(original_dates) == date_chain["expected_interval_days"]
    assert _interval_days(shifted_dates) == date_chain["expected_interval_days"]
    for original, shifted in zip(original_dates, shifted_dates):
        assert original in fixture.text
        assert shifted in fixture.expected_output["text"]


def _one(category: str) -> GoldenFixture:
    matches = [
        fixture for fixture in load_golden_fixtures() if fixture.category == category
    ]
    assert len(matches) == 1
    return matches[0]


def _has_overlap(spans) -> bool:
    rows = [
        (
            span["start"] if isinstance(span, dict) else span.start,
            span["end"] if isinstance(span, dict) else span.end,
        )
        for span in spans
    ]
    return any(
        first_start < second_end and second_start < first_end
        for index, (first_start, first_end) in enumerate(rows)
        for second_start, second_end in rows[index + 1 :]
    )


def _interval_days(values: list[str]) -> list[int]:
    parsed = [date.fromisoformat(value) for value in values]
    return [
        (parsed[index + 1] - parsed[index]).days for index in range(len(parsed) - 1)
    ]
