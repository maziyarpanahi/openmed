"""Tests for the synthetic 11-language Naamapadam-style suite."""

from __future__ import annotations

import json

import pytest

from openmed.core.model_registry import get_pii_models_by_language
from openmed.core.pii_i18n import INDIC_NER_LANGUAGES, get_patterns_for_language
from openmed.core.script_detect import candidate_languages_for_script, detect_script
from openmed.eval.coverage import naamapadam_language_coverage
from openmed.eval.dataset_card import build_dataset_card
from openmed.eval.suites.naamapadam import (
    NAAMAPADAM,
    NAAMAPADAM_LANGUAGES,
    load_naamapadam_fixtures,
    run_naamapadam,
)

EXPECTED_SCRIPTS = {
    "as": "Bengali",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Devanagari",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Devanagari",
    "or": "Oriya",
    "pa": "Gurmukhi",
    "ta": "Tamil",
    "te": "Telugu",
}


class _GoldPredictor:
    def __init__(self):
        self.by_text = {
            fixture.text: fixture.gold_spans for fixture in load_naamapadam_fixtures()
        }

    def predict(self, text):
        return [
            {"start": span.start, "end": span.end, "label": span.label}
            for span in self.by_text[text]
        ]


class _EmptyPredictor:
    def predict(self, text):
        return []


def test_fixtures_cover_all_languages_labels_and_native_scripts():
    fixtures = load_naamapadam_fixtures()

    assert len(fixtures) == 11
    assert {fixture.language for fixture in fixtures} == INDIC_NER_LANGUAGES
    for fixture in fixtures:
        language = fixture.language
        script = detect_script(fixture.text)
        assert script == EXPECTED_SCRIPTS[language]
        assert language in candidate_languages_for_script(script)
        assert get_patterns_for_language(language)
        assert isinstance(get_pii_models_by_language(language), dict)
        assert {span.label for span in fixture.gold_spans} == {
            "PERSON",
            "LOCATION",
            "ORGANIZATION",
        }


def test_suite_reports_per_language_micro_f1_and_zero_entity_leakage():
    report = run_naamapadam(_GoldPredictor())

    assert report.status == "completed"
    assert report.gate_passed
    assert report.micro is not None and report.micro.f1 == 1.0
    assert tuple(sorted(report.languages)) == NAAMAPADAM_LANGUAGES
    assert all(row.recall == 1.0 for row in report.languages.values())
    assert all(row.leaked_entities == 0 for row in report.languages.values())


def test_suite_failure_report_is_aggregate_only():
    fixtures = load_naamapadam_fixtures()
    report = run_naamapadam(_EmptyPredictor())

    assert not report.gate_passed
    assert all(row.recall == 0.0 for row in report.languages.values())
    assert all(row.leakage_rate == 1.0 for row in report.languages.values())
    serialized = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    for fixture in fixtures:
        assert fixture.text not in serialized
        for span in fixture.gold_spans:
            assert fixture.text[span.start : span.end] not in serialized


def test_suite_skips_with_reason_when_optional_weights_are_absent(monkeypatch):
    monkeypatch.delenv("OPENMED_INDIC_NER_MODEL", raising=False)

    report = run_naamapadam()

    assert report.status == "skipped"
    assert "OPENMED_INDIC_NER_MODEL is not configured" in report.skip_reason
    assert report.languages == {}


def test_configured_registry_round_trip_for_every_language(monkeypatch):
    monkeypatch.setenv("OPENMED_INDIC_NER_MODEL", "/models/indic")

    for language in NAAMAPADAM_LANGUAGES:
        models = get_pii_models_by_language(language)
        assert models[f"pii_{language}_indic_ner"].model_id == "/models/indic"


def test_coverage_rows_and_dataset_card_register_cc0_provenance():
    rows = naamapadam_language_coverage()
    card = build_dataset_card(NAAMAPADAM)

    assert len(rows) == 11
    assert {row["language"] for row in rows} == INDIC_NER_LANGUAGES
    assert all(row["fixture_count"] == 1 for row in rows)
    assert card.dataset == NAAMAPADAM
    assert card.license_id == "CC0-1.0"
    assert card.record_count == 11
    assert set(card.languages) == INDIC_NER_LANGUAGES
