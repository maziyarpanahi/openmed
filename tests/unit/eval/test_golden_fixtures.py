"""Tests for synthetic golden de-identification fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_entity_merger import validate_luhn
from openmed.core.pii_i18n import (
    SUPPORTED_LANGUAGES,
    validate_aadhaar,
    validate_portuguese_cpf,
)
from openmed.eval import harness
from openmed.eval.golden import (
    GOLDEN_CATEGORIES,
    GoldenFixture,
    fixture_languages,
    fixtures_by_category,
    fixtures_by_language,
    list_fixture_paths,
    load_benchmark_fixtures,
    load_golden_fixtures,
)
from openmed.eval.metrics import compute_date_shift_consistency

EXPANDED_MULTILINGUAL_LANGUAGES = ("ar", "ja", "tr")


def test_golden_directory_documents_synthetic_only_no_dua():
    readme = Path("openmed/eval/golden/README.md").read_text(encoding="utf-8").lower()

    assert "synthetic-only" in readme
    assert "no dua" in readme
    assert "no real phi" in readme


def test_golden_fixtures_cover_required_categories_and_languages():
    fixtures = load_golden_fixtures()
    grouped = fixtures_by_category(fixtures)

    assert set(grouped) == set(GOLDEN_CATEGORIES)
    assert fixture_languages(fixtures, category="multilingual") == SUPPORTED_LANGUAGES

    multilingual = grouped["multilingual"]
    assert len(multilingual) == len(SUPPORTED_LANGUAGES)
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)


def test_expanded_multilingual_fixtures_cover_person_date_and_locale_id():
    grouped = fixtures_by_language(
        load_golden_fixtures(),
        category="multilingual",
    )

    assert set(EXPANDED_MULTILINGUAL_LANGUAGES).issubset(grouped)
    for language in EXPANDED_MULTILINGUAL_LANGUAGES:
        assert len(grouped[language]) == 1
        fixture = grouped[language][0]
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
        grouped[language][0].to_benchmark_fixture()
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
        assert fixture.gold_spans

        for span in fixture.gold_spans:
            assert span.label in CANONICAL_LABELS
            assert fixture.text[span.start : span.end] == span.text

        mapping = fixture.to_mapping()
        assert GoldenFixture.from_mapping(mapping).to_mapping() == mapping


def test_golden_json_files_are_harness_loadable():
    for fixture_path in list_fixture_paths():
        loaded = harness.load_fixtures(fixture_path)
        assert loaded
        assert all(item.gold_spans for item in loaded)
        assert all(item.metadata["expected_output"]["text"] for item in loaded)

    benchmark_fixtures = load_benchmark_fixtures()
    assert len(benchmark_fixtures) == len(load_golden_fixtures())
    assert all(item.metadata["category"] for item in benchmark_fixtures)


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
    fixture = _one("checksum_ids")
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
