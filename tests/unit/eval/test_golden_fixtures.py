"""Tests for synthetic golden de-identification fixtures."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from openmed.core.decoding.spans import (
    is_grapheme_boundary,
    iter_grapheme_clusters,
)
from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.language_pack import LanguagePack, get_language_pack
from openmed.core.language_router import LanguageRouter
from openmed.core.pii_entity_merger import (
    find_semantic_units,
    validate_luhn,
    validate_ssn,
)
from openmed.core.pii_i18n import (
    INDIC_NER_LANGUAGES,
    LANGUAGE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    get_patterns_for_language,
    normalize_bengali_assamese_digits,
    normalize_odia_digits,
    validate_aadhaar,
    validate_assam_pin,
    validate_assamese_aadhaar,
    validate_assamese_indian_phone,
    validate_czechoslovak_rodne_cislo,
    validate_danish_cpr,
    validate_dutch_bsn,
    validate_egyptian_national_id,
    validate_french_nir,
    validate_german_steuer_id,
    validate_hungarian_taj,
    validate_israeli_teudat_zehut,
    validate_italian_codice_fiscale,
    validate_latvian_personas_kods,
    validate_maharashtra_pin,
    validate_malaysian_mykad,
    validate_marathi_aadhaar,
    validate_marathi_indian_phone,
    validate_odia_aadhaar,
    validate_odia_indian_phone,
    validate_odisha_pin,
    validate_philhealth_pin,
    validate_philsys_psn,
    validate_portuguese_cpf,
    validate_romanian_cnp,
    validate_spanish_dni,
    validate_tamil_aadhaar,
    validate_tamil_nadu_puducherry_pin,
    validate_turkish_tckn,
    validate_vietnamese_cccd,
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


def test_golden_loader_rejects_duplicate_fixture_ids(tmp_path):
    fixture = _one("date_arithmetic").to_mapping()
    fixture_pack = {
        "fixtures": [fixture],
        "synthetic": True,
        "version": 1,
    }
    for filename in ("first.json", "second.json"):
        (tmp_path / filename).write_text(
            json.dumps(fixture_pack),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="duplicate golden fixture id"):
        load_golden_fixtures(tmp_path)


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


def test_assamese_i18n_fixtures_are_grapheme_safe_and_validator_equivalent():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/as.jsonl")
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(fixtures) == 2
    assert {fixture.metadata["digit_set"] for fixture in fixtures} == {
        "ascii",
        "bengali-assamese",
    }

    names = []
    aadhaar_values = []
    phone_values = []
    pin_values = []
    for fixture in fixtures:
        person_spans = [span for span in fixture.gold_spans if span.label == "PERSON"]
        assert len(person_spans) == 1
        names.append(person_spans[0].text)

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

    fixture_marks = set("".join(names))
    assert {"া", "ী", "ু", "্"}.issubset(fixture_marks)
    assert {"ৰ", "ৱ"}.issubset(set("".join(fixture.text for fixture in fixtures)))
    assert all(validate_assamese_aadhaar(value) for value in aadhaar_values)
    assert all(validate_assamese_indian_phone(value) for value in phone_values)
    assert all(validate_assam_pin(value) for value in pin_values)
    assert all(
        validate_aadhaar(normalize_bengali_assamese_digits(value))
        for value in aadhaar_values
    )
    assert (
        len({normalize_bengali_assamese_digits(value) for value in aadhaar_values}) == 1
    )
    assert (
        len({normalize_bengali_assamese_digits(value) for value in phone_values}) == 1
    )
    assert len({normalize_bengali_assamese_digits(value) for value in pin_values}) == 1


def test_assamese_name_patterns_require_honorific_and_common_surname():
    examples = (
        ("শ্ৰী", "অৰুণ বৰুৱা"),
        ("শ্ৰীমতী", "মণিকা শইকীয়া"),
        ("ডা.", "দীপালী গগৈ"),
        ("শ্ৰী", "ৰঞ্জিত বৰা"),
    )

    for honorific, name in examples:
        text = f"ৰোগী {honorific} {name}."
        units = find_semantic_units(text, LANGUAGE_PII_PATTERNS["as"])
        detected_names = [
            text[start:end]
            for start, end, entity_type, *_rest in units
            if entity_type == "name"
        ]
        assert detected_names == [name]

    bare_text = "গগৈ আৰু বৰা সাধাৰণ উপাধি।"
    bare_units = find_semantic_units(bare_text, LANGUAGE_PII_PATTERNS["as"])
    assert all(entity_type != "name" for _, _, entity_type, *_rest in bare_units)


def test_assamese_cues_disambiguate_the_shared_bengali_script():
    assamese_pack = get_language_pack("as")
    assert assamese_pack is not None
    bengali_pack = LanguagePack(
        code="bn",
        scripts=("Bengali",),
        default_model="env:OPENMED_INDIC_NER_MODEL",
        segmenter_id="pysbd",
        recognizers=("builtin-patterns", "model"),
        surrogate_locale="bn_BD",
    )
    router = LanguageRouter(
        packs=(assamese_pack, bengali_pack),
        use_optional_lid=False,
    )
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/as.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    for fixture in fixtures:
        decision = router.route(fixture.text)
        assert decision.language == "as"
        assert any(run.source == "stdlib:assamese-cues" for run in decision.runs)

    bengali = "রোগী শ্রী অরুণ দাস। জন্ম ১৪ জানুয়ারি ২০২৬।"
    decision = router.route(bengali)
    assert decision.language == "bn"
    assert all(run.language != "as" for run in decision.runs)


def _i18n_fixtures(code: str) -> list[GoldenFixture]:
    return [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path(f"openmed/eval/golden/fixtures/i18n/{code}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_urdu_cues_disambiguate_the_shared_arabic_script():
    # The catalog ships no ``ur`` pack yet (issue #1520 owns that), so the
    # Urdu pack is injected here exactly as the Bengali pack is above. This
    # proves the routing contract the built-in catalog will satisfy the moment
    # a ``ur`` pack is registered, with no further change to the router.
    arabic_pack = get_language_pack("ar")
    assert arabic_pack is not None
    urdu_pack = LanguagePack(
        code="ur",
        scripts=("Arabic",),
        default_model="env:OPENMED_URDU_NER_MODEL",
        segmenter_id="unicode-sentence",
        recognizers=("builtin-patterns", "model"),
        surrogate_locale="ur_PK",
    )
    router = LanguageRouter(packs=(arabic_pack, urdu_pack), use_optional_lid=False)

    for fixture in _i18n_fixtures("ur"):
        decision = router.route(fixture.text)
        assert decision.language == "ur"
        assert any(run.source == "stdlib:urdu-cues" for run in decision.runs)
        for run in decision.runs:
            if run.script == "Arabic":
                assert run.candidates == ("ur", "ar", "ha")

    for fixture in _i18n_fixtures("ar"):
        decision = router.route(fixture.text)
        assert decision.language == "ar"
        assert all(run.language != "ur" for run in decision.runs)
        assert all(run.source != "stdlib:urdu-cues" for run in decision.runs)
        for run in decision.runs:
            if run.script == "Arabic":
                assert run.candidates == ("ar", "ha", "ur")


def test_urdu_fixtures_fall_back_to_arabic_until_an_urdu_pack_ships():
    router = LanguageRouter(use_optional_lid=False)

    for fixture in _i18n_fixtures("ur"):
        decision = router.route(fixture.text)
        assert decision.language == "ar"
        assert any(run.source == "stdlib:arabic-fallback" for run in decision.runs)
        # The unroutable Urdu evidence still reaches callers through the run
        # metadata, so a consumer can see why the fallback fired.
        assert any(run.candidates[:1] == ("ur",) for run in decision.runs)

    for fixture in _i18n_fixtures("ar"):
        decision = router.route(fixture.text)
        assert decision.language == "ar"
        assert all(run.source != "stdlib:arabic-fallback" for run in decision.runs)
        assert all(run.candidates[:1] != ("ur",) for run in decision.runs)


def test_urdu_disambiguation_preserves_fixture_offsets_and_graphemes():
    router = LanguageRouter(use_optional_lid=False)

    for fixture in _i18n_fixtures("ur"):
        runs = router.route_runs(fixture.text)
        assert "".join(fixture.text[run.start : run.end] for run in runs) == (
            fixture.text
        )
        for run in runs:
            assert is_grapheme_boundary(run.start, fixture.text)
            assert is_grapheme_boundary(run.end, fixture.text)
        for span in fixture.gold_spans:
            assert fixture.text[span.start : span.end] == span.text


def test_assamese_fixtures_pass_zero_leakage_release_gate_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.eval.release_gates import _per_language_residual_leakage_check
    from openmed.processing.outputs import PredictionResult

    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/as.jsonl")
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
            timestamp="2026-07-25T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="as",
        )
        predictions[fixture.fixture_id] = swept_result.entities
        observed = {
            (entity.start, entity.end, normalize_label(entity.label, "as"))
            for entity in swept_result.entities
        }

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label) in observed

        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="as",
            consistent=False,
            seed=None,
            locale="as_IN",
            use_safety_sweep=True,
        )
        assert all(
            span.text not in result.deidentified_text for span in fixture.gold_spans
        )

    report = harness.run_benchmark(
        [fixture.to_benchmark_fixture() for fixture in fixtures],
        suite="golden-assamese",
        model_name="offline-safety-sweep",
        runner=lambda fixture, _model_name, _device: predictions[fixture.fixture_id],
        generated_at="2026-07-25T00:00:00Z",
    )
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["leakage"]["by_language"]["as"] == 0.0

    gate = _per_language_residual_leakage_check(report.metrics, report.metadata)
    assert gate.passed is True
    assert gate.details["evaluated"] == {"as": 0.0}


def test_marathi_i18n_fixtures_are_grapheme_safe_and_validator_equivalent():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/mr.jsonl")
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(fixtures) == 3
    assert {fixture.metadata["fixture_kind"] for fixture in fixtures} == {
        "Marathi-script native digits",
        "Marathi ASCII identifiers",
        "Marathi numeric date",
    }

    names = []
    aadhaar_values = []
    phone_values = []
    pin_values = []
    for fixture in fixtures:
        person_spans = [span for span in fixture.gold_spans if span.label == "PERSON"]
        assert len(person_spans) == 1
        names.append(person_spans[0].text)
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

    assert all(len(name.split()) == 4 for name in names)
    assert all(validate_marathi_aadhaar(value) for value in aadhaar_values)
    assert all(validate_marathi_indian_phone(value) for value in phone_values)
    assert all(validate_maharashtra_pin(value) for value in pin_values)


def test_marathi_fixtures_pass_zero_leakage_release_gate_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.eval.release_gates import _per_language_residual_leakage_check
    from openmed.processing.outputs import PredictionResult

    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/mr.jsonl")
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
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="mr",
        )
        predictions[fixture.fixture_id] = swept_result.entities
        observed = {
            (entity.start, entity.end, normalize_label(entity.label, "mr"))
            for entity in swept_result.entities
        }

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label) in observed

        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="mr",
            consistent=False,
            seed=None,
            locale="mr_IN",
            use_safety_sweep=True,
        )
        assert all(
            span.text not in result.deidentified_text for span in fixture.gold_spans
        )

    report = harness.run_benchmark(
        [fixture.to_benchmark_fixture() for fixture in fixtures],
        suite="golden-marathi",
        model_name="offline-safety-sweep",
        runner=lambda fixture, _model_name, _device: predictions[fixture.fixture_id],
        generated_at="2026-07-24T00:00:00Z",
    )
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["leakage"]["by_language"]["mr"] == 0.0

    gate = _per_language_residual_leakage_check(report.metrics, report.metadata)
    assert gate.passed is True
    assert gate.details["evaluated"] == {"mr": 0.0}


def test_tamil_i18n_fixtures_are_grapheme_safe_and_validator_equivalent():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/ta.jsonl")
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fixtures = [GoldenFixture.from_mapping(row) for row in rows]

    assert len(fixtures) == 3
    assert {fixture.metadata["fixture_kind"] for fixture in fixtures} == {
        "Tamil-script native digits",
        "Tamil-English code-mixed",
        "Tamil Sri grapheme",
    }
    assert list(iter_grapheme_clusters("ஸ்ரீ")) == [(0, len("ஸ்ரீ"))]

    aadhaar_values = []
    pin_values = []
    for fixture in fixtures:
        for span in fixture.gold_spans:
            assert is_grapheme_boundary(span.start, fixture.text)
            assert is_grapheme_boundary(span.end, fixture.text)
            assert fixture.text[span.start : span.end] == span.text
            if span.label == "ID_NUM":
                aadhaar_values.append(span.text)
            if span.label == "ZIPCODE":
                pin_values.append(span.text)

    assert all(validate_tamil_aadhaar(value) for value in aadhaar_values)
    assert all(validate_tamil_nadu_puducherry_pin(value) for value in pin_values)
    assert validate_tamil_aadhaar("௨௪௬௭ ௭௮௩௨ ௫௪௮௪")
    assert validate_tamil_aadhaar("2467 7832 5484")
    assert validate_tamil_nadu_puducherry_pin("௬௦௫௦௦௧")
    assert validate_tamil_nadu_puducherry_pin("605001")


def test_tamil_fixtures_pass_zero_leakage_release_gate_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.eval.release_gates import _per_language_residual_leakage_check
    from openmed.processing.outputs import PredictionResult

    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/ta.jsonl")
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
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="ta",
        )
        predictions[fixture.fixture_id] = swept_result.entities
        observed = {
            (entity.start, entity.end, normalize_label(entity.label, "ta"))
            for entity in swept_result.entities
        }

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label) in observed

        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="ta",
            consistent=False,
            seed=None,
            locale="ta_IN",
            use_safety_sweep=True,
        )
        assert all(
            span.text not in result.deidentified_text for span in fixture.gold_spans
        )

    report = harness.run_benchmark(
        [fixture.to_benchmark_fixture() for fixture in fixtures],
        suite="golden-tamil",
        model_name="offline-safety-sweep",
        runner=lambda fixture, _model_name, _device: predictions[fixture.fixture_id],
        generated_at="2026-07-24T00:00:00Z",
    )
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["leakage"]["by_language"]["ta"] == 0.0

    gate = _per_language_residual_leakage_check(report.metrics, report.metadata)
    assert gate.passed is True
    assert gate.details["evaluated"] == {"ta": 0.0}


def test_odia_i18n_fixtures_are_grapheme_safe_and_validator_equivalent():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/or.jsonl")
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(fixtures) == 2
    assert {fixture.metadata["digit_set"] for fixture in fixtures} == {
        "ascii",
        "odia",
    }

    names = []
    aadhaar_values = []
    phone_values = []
    pin_values = []
    for fixture in fixtures:
        person_spans = [span for span in fixture.gold_spans if span.label == "PERSON"]
        assert len(person_spans) == 1
        names.append(person_spans[0].text)

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

    fixture_marks = set("".join(names))
    assert {"ା", "ୀ", "ୁ", "୍"}.issubset(fixture_marks)
    assert all(validate_odia_aadhaar(value) for value in aadhaar_values)
    assert all(validate_odia_indian_phone(value) for value in phone_values)
    assert all(validate_odisha_pin(value) for value in pin_values)
    assert all(
        validate_aadhaar(normalize_odia_digits(value)) for value in aadhaar_values
    )
    assert len({normalize_odia_digits(value) for value in aadhaar_values}) == 1
    assert len({normalize_odia_digits(value) for value in phone_values}) == 1
    assert len({normalize_odia_digits(value) for value in pin_values}) == 1


def test_odia_name_patterns_require_honorific_and_common_surname():
    examples = (
        ("ଶ୍ରୀ", "ଅରୁଣ ଦାସ"),
        ("ଶ୍ରୀମତୀ", "ସୁନୀତା ମହାନ୍ତି"),
        ("ଡା.", "ବିକାଶ ପଟ୍ଟନାୟକ"),
        ("ଶ୍ରୀ", "ରବି ସାହୁ"),
    )

    for honorific, name in examples:
        text = f"ରୋଗୀ {honorific} {name}."
        units = find_semantic_units(text, LANGUAGE_PII_PATTERNS["or"])
        detected_names = [
            text[start:end]
            for start, end, entity_type, *_rest in units
            if entity_type == "name"
        ]
        assert detected_names == [name]

    bare_text = "ଦାସ ଓ ସାହୁ ସାଧାରଣ ଉପନାମ।"
    bare_units = find_semantic_units(bare_text, LANGUAGE_PII_PATTERNS["or"])
    assert all(entity_type != "name" for _, _, entity_type, *_rest in bare_units)


def test_odia_and_bengali_script_blocks_never_cross_route():
    router = LanguageRouter(use_optional_lid=False)
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/or.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    for fixture in fixtures:
        odia_runs = [
            run for run in router.route_runs(fixture.text) if run.script == "Odia"
        ]
        assert odia_runs
        assert {run.language for run in odia_runs} == {"or"}

    bengali_runs = [
        run for run in router.route_runs("রোগী শ্রী অরুণ দাস।") if run.script == "Bengali"
    ]
    assert bengali_runs
    assert all(run.language != "or" for run in bengali_runs)


def test_odia_fixtures_pass_zero_leakage_release_gate_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.eval.release_gates import _per_language_residual_leakage_check
    from openmed.processing.outputs import PredictionResult

    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/or.jsonl")
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
            timestamp="2026-07-25T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="or",
        )
        predictions[fixture.fixture_id] = swept_result.entities
        observed = {
            (entity.start, entity.end, normalize_label(entity.label, "or"))
            for entity in swept_result.entities
        }

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label) in observed

        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="or",
            consistent=False,
            seed=None,
            locale="or_IN",
            use_safety_sweep=True,
        )
        assert all(
            span.text not in result.deidentified_text for span in fixture.gold_spans
        )

    report = harness.run_benchmark(
        [fixture.to_benchmark_fixture() for fixture in fixtures],
        suite="golden-odia",
        model_name="offline-safety-sweep",
        runner=lambda fixture, _model_name, _device: predictions[fixture.fixture_id],
        generated_at="2026-07-25T00:00:00Z",
    )
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["leakage"]["by_language"]["or"] == 0.0

    gate = _per_language_residual_leakage_check(report.metrics, report.metadata)
    assert gate.passed is True
    assert gate.details["evaluated"] == {"or": 0.0}


def test_vietnamese_i18n_fixtures_are_grapheme_safe_and_validator_equivalent():
    fixture_path = Path("openmed/eval/golden/fixtures/i18n/vi.jsonl")
    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(fixtures) == 2
    assert {fixture.language for fixture in fixtures} == {"vi"}
    assert {fixture.metadata["locale"] for fixture in fixtures} == {"vi_VN"}
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)

    cccd_values = []
    phone_values = []
    for fixture in fixtures:
        # Vietnamese text must stay in NFC so codepoint offsets remain
        # grapheme-aligned; NFD would split every tone mark into its own scalar.
        assert unicodedata.normalize("NFC", fixture.text) == fixture.text

        for span in fixture.gold_spans:
            assert is_grapheme_boundary(span.start, fixture.text)
            assert is_grapheme_boundary(span.end, fixture.text)
            assert fixture.text[span.start : span.end] == span.text
            if span.label == "ID_NUM":
                if span.metadata.get("identifier_type") == "cccd":
                    cccd_values.append(span.text)
            elif span.label == "PHONE":
                phone_values.append(span.text)

    assert cccd_values
    assert all(validate_vietnamese_cccd(value) for value in cccd_values)
    assert all(value.startswith(("+84", "0")) for value in phone_values)
    # Acceptance coverage: a native "ngay D thang M nam YYYY" date, a 0xx
    # mobile, and a diacritic-bearing address all appear across the two rows.
    all_spans = [span for fixture in fixtures for span in fixture.gold_spans]
    assert any(
        span.label == "DATE" and span.text.startswith("ngày ") for span in all_spans
    )
    assert any(
        span.label == "PHONE" and span.text.startswith("0") for span in all_spans
    )
    assert any(
        span.label == "STREET_ADDRESS" and any(char in span.text for char in "ườảãạệ")
        for span in all_spans
    )


def test_vietnamese_fixtures_pass_zero_leakage_release_gate_offline():
    from openmed.core.pii import (
        _apply_safety_sweep_to_result,
        _build_deidentification_result,
    )
    from openmed.eval.release_gates import _per_language_residual_leakage_check
    from openmed.processing.outputs import PredictionResult

    fixtures = [
        GoldenFixture.from_mapping(json.loads(line))
        for line in Path("openmed/eval/golden/fixtures/i18n/vi.jsonl")
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
            timestamp="2026-08-01T00:00:00Z",
            metadata={},
        )
        swept_result, added_count = _apply_safety_sweep_to_result(
            fixture.text,
            empty_result,
            lang="vi",
        )
        predictions[fixture.fixture_id] = swept_result.entities
        observed = {
            (entity.start, entity.end, normalize_label(entity.label, "vi"))
            for entity in swept_result.entities
        }

        assert added_count == len(fixture.gold_spans)
        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label) in observed

        result = _build_deidentification_result(
            fixture.text,
            swept_result,
            effective_method="mask",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang="vi",
            consistent=False,
            seed=None,
            locale="vi_VN",
            use_safety_sweep=True,
        )
        assert all(
            span.text not in result.deidentified_text for span in fixture.gold_spans
        )

    report = harness.run_benchmark(
        [fixture.to_benchmark_fixture() for fixture in fixtures],
        suite="golden-vietnamese",
        model_name="offline-safety-sweep",
        runner=lambda fixture, _model_name, _device: predictions[fixture.fixture_id],
        generated_at="2026-08-01T00:00:00Z",
    )
    assert report.metrics["leakage"]["overall"] == 0.0
    assert report.metrics["leakage"]["by_language"]["vi"] == 0.0

    gate = _per_language_residual_leakage_check(report.metrics, report.metadata)
    assert gate.passed is True
    assert gate.details["evaluated"] == {"vi": 0.0}


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


# OM-120 freezes the 12-language baseline named in issue #285. The live
# language registry now includes later packs, so deriving this historical
# acceptance set from SUPPORTED_LANGUAGES would silently expand the task.
OM_120_WIRED_LANGUAGES = frozenset(
    {"en", "fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "ja", "tr"}
)

_ID_TRAP_VALIDATORS: dict[str, tuple[Callable[[str], bool], ...]] = {
    "en": (validate_ssn,),
    "fr": (validate_french_nir,),
    "de": (validate_german_steuer_id,),
    "it": (validate_italian_codice_fiscale,),
    "es": (validate_spanish_dni,),
    "nl": (validate_dutch_bsn,),
    "hi": (validate_aadhaar,),
    "te": (validate_aadhaar,),
    "pt": (validate_portuguese_cpf,),
    "ar": (validate_egyptian_national_id,),
    "tr": (validate_turkish_tckn,),
}


def _id_trap_fixtures() -> list[GoldenFixture]:
    return [
        fixture
        for fixture in load_golden_fixtures()
        if fixture.fixture_id.startswith("golden-per-language-id-trap-")
    ]


def _date_trap_fixtures() -> list[GoldenFixture]:
    return [
        fixture
        for fixture in load_golden_fixtures()
        if fixture.fixture_id.startswith("golden-per-language-date-trap-")
    ]


def _mask_gold_spans(fixture: GoldenFixture) -> str:
    masked = fixture.text
    for span in sorted(fixture.gold_spans, key=lambda item: item.start, reverse=True):
        masked = masked[: span.start] + f"[{span.label}]" + masked[span.end :]
    return masked


def test_per_language_id_traps_cover_all_wired_languages():
    fixtures = _id_trap_fixtures()
    languages = {fixture.language for fixture in fixtures}

    assert languages == OM_120_WIRED_LANGUAGES
    assert len(fixtures) == len(OM_120_WIRED_LANGUAGES)

    for fixture in fixtures:
        assert fixture.category == "checksum_ids"
        assert fixture.metadata["synthetic"] is True
        assert len(fixture.gold_spans) >= 1
        assert fixture.gold_spans[0].label in CANONICAL_LABELS
        hard_negatives = fixture.metadata.get("hard_negatives", [])
        assert len(hard_negatives) == 1
        hn = hard_negatives[0]
        assert "start" in hn and "end" in hn
        assert "text" in hn and "identifier_type" in hn and "reason" in hn
        assert fixture.text[hn["start"] : hn["end"]] == hn["text"]
        for span in fixture.gold_spans:
            assert not (hn["start"] < span.end and span.start < hn["end"]), (
                f"{fixture.fixture_id}: hard negative [{hn['start']}:{hn['end']}] "
                f"overlaps gold span [{span.start}:{span.end}]"
            )
        assert fixture.expected_output["method"] == "mask"
        assert fixture.expected_output["text"] == _mask_gold_spans(fixture)


def test_per_language_date_traps_cover_all_wired_languages():
    fixtures = _date_trap_fixtures()
    languages = {fixture.language for fixture in fixtures}

    assert languages == OM_120_WIRED_LANGUAGES
    assert len(fixtures) == len(OM_120_WIRED_LANGUAGES)

    for fixture in fixtures:
        assert fixture.category == "multilingual"
        assert fixture.metadata["synthetic"] is True
        date_spans = [span for span in fixture.gold_spans if span.label == "DATE"]
        assert len(date_spans) == 3
        assert fixture.expected_output["method"] == "mask"
        assert fixture.expected_output["text"].count("[DATE]") == 3
        assert fixture.expected_output["text"] == _mask_gold_spans(fixture)


def test_per_language_id_traps_invalid_ids_fail_validators():
    """Valid IDs pass their language's checksum validator; invalid hard
    negatives fail it.  For ``ja`` there is no My Number checksum validator
    in the repository, so the fixture explicitly uses
    ``checksum_status="not_validated"`` with a ``format_mismatch`` hard
    negative instead of a checksum failure.
    """
    for fixture in _id_trap_fixtures():
        lang = fixture.language
        valid_span = fixture.gold_spans[0]
        valid_id = valid_span.text
        hn = fixture.metadata["hard_negatives"][0]
        invalid_id = hn["text"]

        if lang == "ja":
            assert valid_span.metadata["checksum_status"] == "not_validated"
            assert hn["reason"] == "format_mismatch"
            continue

        validators = _ID_TRAP_VALIDATORS[lang]
        assert any(v(valid_id) for v in validators), (
            f"{lang}: valid ID {valid_id!r} should pass at least one validator"
        )
        assert all(not v(invalid_id) for v in validators), (
            f"{lang}: invalid ID {invalid_id!r} should fail all validators"
        )


def test_per_language_traps_recover_through_language_patterns():
    """Every gold span in the per-language trap fixtures is recovered by the
    language's PII patterns at the exact recorded offset.  This catches
    regressions where a language pack's regex stops matching native formats.
    """
    for fixture in [*_id_trap_fixtures(), *_date_trap_fixtures()]:
        units = find_semantic_units(
            fixture.text,
            get_patterns_for_language(fixture.language),
        )
        recovered = {
            (
                start,
                end,
                normalize_label(entity_type, fixture.language),
                fixture.text[start:end],
            )
            for start, end, entity_type, _score, _pattern, validated in units
            if validated
        }
        for span in fixture.gold_spans:
            assert (span.start, span.end, span.label, span.text) in recovered, (
                f"{fixture.fixture_id}: span {span.text!r} ({span.label}) "
                f"at [{span.start}:{span.end}] not recovered by "
                f"{fixture.language} patterns"
            )

        for hard_negative in fixture.metadata.get("hard_negatives", []):
            assert not any(
                start < hard_negative["end"]
                and end > hard_negative["start"]
                and validated
                for start, end, _entity_type, _score, _pattern, validated in units
            ), (
                f"{fixture.fixture_id}: hard negative {hard_negative['text']!r} "
                "was accepted by a production pattern"
            )


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
