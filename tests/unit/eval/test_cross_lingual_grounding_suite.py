"""Tests for the standalone cross-lingual grounding metrics suite."""

from __future__ import annotations

import json

import pytest

from openmed.eval.suites import (
    CROSS_LINGUAL_GROUNDING,
    DEFAULT_SUITES,
    NON_ENGLISH_LANGUAGES,
    NON_ENGLISH_TOP1_FLOOR,
    CrossLingualGroundingReport,
    build_fixture_grounder,
    cross_lingual_grounding_metadata,
    load_cross_lingual_grounding_fixtures,
    run_cross_lingual_grounding,
    scan_restricted_corpus_markers,
)

_EXPECTED_LANGUAGES = ("en", "es", "fr", "de", "zh")


def test_cross_lingual_grounding_is_standalone_not_a_default_suite():
    # Registered as a standalone suite like chinese_terminology / indic_encoder;
    # the DEFAULT_SUITES exact-tuple gate must stay untouched.
    assert CROSS_LINGUAL_GROUNDING == "cross_lingual_grounding"
    assert CROSS_LINGUAL_GROUNDING not in DEFAULT_SUITES


def test_load_fixtures_returns_synthetic_multilingual_gold():
    cases = load_cross_lingual_grounding_fixtures()

    assert len(cases) == 3
    assert {case.system for case in cases} == {"icd10cm", "rxnorm", "hpo"}
    for case in cases:
        assert case.provenance == "synthetic-permissive"
        assert set(case.gold_mentions) == set(_EXPECTED_LANGUAGES)


def test_run_reports_perfect_per_language_top1_on_synthetic_gold():
    report = run_cross_lingual_grounding()

    assert isinstance(report, CrossLingualGroundingReport)
    assert report.languages == _EXPECTED_LANGUAGES
    assert report.concept_count == 3
    assert report.per_language_top1 == {
        "en": 1.0,
        "es": 1.0,
        "fr": 1.0,
        "de": 1.0,
        "zh": 1.0,
    }
    assert report.per_language_total == {
        "en": 3,
        "es": 3,
        "fr": 3,
        "de": 3,
        "zh": 3,
    }
    assert report.english_top1 == 1.0
    assert report.macro_top1_non_english == 1.0
    assert report.micro_top1_non_english == 1.0
    assert report.cross_lingual_consistency == 1.0
    assert report.fully_consistent_concepts == 3
    assert report.per_language_english_agreement == {
        "es": 1.0,
        "fr": 1.0,
        "de": 1.0,
        "zh": 1.0,
    }
    assert report.english_baseline_intact is True
    assert report.synthetic_provenance is True
    assert report.restricted_markers == ()
    assert report.passed is True


def test_english_baseline_reported_and_non_english_meets_floor():
    report = run_cross_lingual_grounding()

    assert "en" in report.per_language_top1
    for language in NON_ENGLISH_LANGUAGES:
        assert report.per_language_top1[language] >= NON_ENGLISH_TOP1_FLOOR


def test_report_is_byte_identical_across_runs():
    first = run_cross_lingual_grounding().to_dict()
    second = run_cross_lingual_grounding().to_dict()

    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(
        second, sort_keys=True, ensure_ascii=False
    )


def test_floor_gate_fails_when_a_non_english_language_mis_grounds():
    cases = load_cross_lingual_grounding_fixtures()
    reference = build_fixture_grounder()

    def _broken(system: str, mention: str, language: str):
        if language == "es":
            return ["WRONG"]
        return reference(system, mention, language)

    report = run_cross_lingual_grounding(cases=cases, grounder=_broken)

    assert report.per_language_top1["es"] == 0.0
    assert report.per_language_top1["es"] < NON_ENGLISH_TOP1_FLOOR
    assert report.per_language_english_agreement["es"] == 0.0
    assert report.cross_lingual_consistency < 1.0
    assert report.fully_consistent_concepts == 0
    assert report.passed is False
    # The English baseline is unaffected by the non-English regression.
    assert report.english_top1 == 1.0
    assert report.english_baseline_intact is True


def test_leakage_scan_fails_on_restricted_markers_and_passes_on_fixture():
    assert scan_restricted_corpus_markers(
        "Pulled from MIMIC-III discharge summary"
    ) == (
        "mimic",
        "discharge summary",
    )
    assert scan_restricted_corpus_markers("i2b2 and n2c2 and umls and snomed") == (
        "i2b2",
        "n2c2",
        "umls",
        "snomed",
    )

    fixture_text = "\n".join(
        json.dumps(
            {
                "system": case.system,
                "expected_code": case.expected_code,
                "gold_mentions": dict(case.gold_mentions),
                "provenance": case.provenance,
            }
        )
        for case in load_cross_lingual_grounding_fixtures()
    )
    assert scan_restricted_corpus_markers(fixture_text) == ()


def test_run_rejects_fixture_with_restricted_markers(tmp_path):
    tainted = tmp_path / "tainted.jsonl"
    tainted.write_text(
        json.dumps(
            {
                "system": "icd10cm",
                "code": "E119",
                "expected_code": "E11.9",
                "display": "Type 2 diabetes mellitus without complications",
                "synonyms": ["type 2 diabetes mellitus", "type 2 diabetes"],
                "language_aliases": {
                    "es": ["diabetes mellitus tipo 2"],
                    "fr": ["diabete de type 2"],
                    "de": ["typ-2-diabetes"],
                    "zh": ["2型糖尿病"],
                },
                "gold_mentions": {
                    "en": "type 2 diabetes mellitus",
                    "es": "diabetes mellitus tipo 2",
                    "fr": "diabete de type 2",
                    "de": "typ-2-diabetes",
                    "zh": "2型糖尿病",
                },
                "provenance": "synthetic-permissive",
                "note": "sourced from a MIMIC discharge summary",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_cross_lingual_grounding(path=tainted)

    assert "mimic" in report.restricted_markers
    assert "discharge summary" in report.restricted_markers
    assert report.passed is False


def test_run_scans_in_memory_cases_for_restricted_markers():
    # A tainted in-memory case (no on-disk path) must be caught by the leakage
    # scan too, not only the committed fixture file.
    from dataclasses import replace

    base = load_cross_lingual_grounding_fixtures()[0]
    tainted = replace(
        base,
        gold_mentions={
            **base.gold_mentions,
            "en": "diabetes from the MIMIC discharge summary",
        },
    )

    report = run_cross_lingual_grounding(cases=[tainted], grounder=lambda *_: ())

    assert report.restricted_markers  # in-memory tainted data is flagged
    assert report.passed is False


def test_metadata_advertises_languages_and_floor():
    metadata = cross_lingual_grounding_metadata()

    assert metadata["suite"] == CROSS_LINGUAL_GROUNDING
    assert metadata["languages"] == list(_EXPECTED_LANGUAGES)
    assert metadata["non_english_top1_floor"] == pytest.approx(0.80)
    assert metadata["fixture"] == "grounding_crosslingual.jsonl"
