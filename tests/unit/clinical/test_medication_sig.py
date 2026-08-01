"""Tests for medication sig frequency and duration normalization."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    MEDICATION_CANDIDATES,
    MEDICATION_SIG_ADVISORY,
    MedicationCandidatePreset,
    filter_medication_candidates,
    normalize_duration,
    normalize_frequency,
)
from openmed.clinical.grounding import Candidate
from openmed.processing.outputs import EntityPrediction


@pytest.mark.parametrize(
    ("text", "expected_per_day"),
    [
        ("qd", 1.0),
        ("daily", 1.0),
        ("BID", 2.0),
        ("TID", 3.0),
        ("QID", 4.0),
        ("q.h.s.", 1.0),
    ],
)
def test_normalize_frequency_common_sig_cues(text, expected_per_day):
    normalized = normalize_frequency(text)
    assert normalized["recognized"] is True
    assert normalized["frequency_per_day"] == expected_per_day
    assert normalized["confidence"] > 0


def test_normalize_frequency_interval_q8h():
    normalized = normalize_frequency("q8h")
    assert normalized["recognized"] is True
    assert normalized["period"] == 8
    assert normalized["period_unit"] == "h"
    assert normalized["frequency_per_day"] == 3.0


def test_normalize_frequency_interval_every_twelve_hours():
    normalized = normalize_frequency("every 12 hours")
    assert normalized["recognized"] is True
    assert normalized["period"] == 12
    assert normalized["period_unit"] == "h"
    assert normalized["frequency_per_day"] == 2.0


def test_normalize_frequency_weekly_interval():
    normalized = normalize_frequency("weekly")
    assert normalized["recognized"] is True
    assert normalized["period"] == 1
    assert normalized["period_unit"] == "wk"
    assert normalized["frequency_per_day"] == 1.0 / 7.0


def test_normalize_frequency_prn_is_flagged_without_numeric_frequency():
    normalized = normalize_frequency("PRN")
    assert normalized["recognized"] is True
    assert normalized["as_needed"] is True
    assert normalized["frequency_per_day"] is None
    assert normalized["period"] is None
    assert normalized["period_unit"] is None


def test_normalize_frequency_scheduled_prn_keeps_schedule_and_flag():
    normalized = normalize_frequency("BID PRN")
    assert normalized["recognized"] is True
    assert normalized["as_needed"] is True
    assert normalized["frequency_per_day"] == 2.0


def test_normalize_frequency_unrecognized_preserves_raw_text():
    normalized = normalize_frequency("with meals on alternating clinic days")
    assert normalized["recognized"] is False
    assert normalized["confidence"] == 0.0
    assert normalized["raw"] == "with meals on alternating clinic days"
    assert normalized["frequency_per_day"] is None


@pytest.mark.parametrize(
    ("text", "value", "unit", "days"),
    [
        ("x 7 days", 7, "d", 7),
        ("x7d", 7, "d", 7),
        ("for 2 weeks", 2, "wk", 14),
        ("10/7", 10, "d", 10),
        ("2/52", 2, "wk", 14),
    ],
)
def test_normalize_duration_common_cues(text, value, unit, days):
    normalized = normalize_duration(text)
    assert normalized["recognized"] is True
    assert normalized["value"] == value
    assert normalized["unit"] == unit
    assert normalized["days"] == days
    assert normalized["confidence"] == 1.0


def test_normalize_duration_unrecognized_preserves_raw_text():
    normalized = normalize_duration("until next visit")
    assert normalized["recognized"] is False
    assert normalized["confidence"] == 0.0
    assert normalized["raw"] == "until next visit"
    assert normalized["days"] is None


def test_medication_sig_advisory_documents_prn_and_review_scope():
    assert "PRN" in MEDICATION_SIG_ADVISORY
    assert "clinician review" in MEDICATION_SIG_ADVISORY


def test_medication_candidate_preset_improves_reported_false_positives():
    text = (
        "Diabetic with PP 202mg/dl. Medication controls sugar. "
        "On glimiperide and metformin.\nCyclopalm\nOndam"
    )

    def entity(surface, score):
        start = text.index(surface)
        return {
            "text": surface,
            "label": "CHEM",
            "confidence": score,
            "start": start,
            "end": start + len(surface),
        }

    entities = [
        entity("PP", 0.9274),
        entity("sugar", 0.7221),
        entity("glimiperide", 0.9399),
        entity("metformin", 0.9422),
        entity("Cyclopalm", 0.954),
        entity("Ondam", 0.954),
    ]

    candidates = filter_medication_candidates(text, entities)

    assert [candidate.text for candidate in candidates] == [
        "glimiperide",
        "metformin",
        "Cyclopalm",
        "Ondam",
    ]
    assert all(candidate.source_label == "CHEM" for candidate in candidates)
    assert entities[0]["text"] == "PP"


def test_medication_candidate_filter_accepts_entity_predictions_at_boundary():
    text = "Started metformin."
    entity = EntityPrediction(
        text="metformin",
        label="CHEM",
        confidence=0.75,
        start=8,
        end=17,
    )

    candidates = filter_medication_candidates(text, [entity])

    assert [(item.text, item.start, item.end) for item in candidates] == [
        ("metformin", 8, 17)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "PP 202mg/dL",
        "FBS: 202 mg/dL",
        "HR 72 bpm",
        "BP 120 mmHg",
    ],
)
def test_observation_abbreviations_are_rejected_from_medication_candidates(text):
    surface = text.split(maxsplit=1)[0].rstrip(":")
    entity = {
        "text": surface,
        "entity_group": "CHEM",
        "score": 0.99,
        "start": 0,
        "end": len(surface),
    }

    assert filter_medication_candidates(text, [entity]) == []


def test_simple_mass_dose_does_not_trigger_observation_filter():
    text = "ASA 81 mg"
    entity = {
        "text": "ASA",
        "entity_group": "CHEM",
        "score": 0.99,
        "start": 0,
        "end": 3,
    }

    assert [item.text for item in filter_medication_candidates(text, [entity])] == [
        "ASA"
    ]


def test_observation_filter_does_not_inspect_the_next_line():
    text = "PP\n202mg/dL"
    entity = {
        "text": "PP",
        "entity_group": "CHEM",
        "score": 0.99,
        "start": 0,
        "end": 2,
    }

    assert [item.text for item in filter_medication_candidates(text, [entity])] == [
        "PP"
    ]


def test_grounding_can_rescue_an_abbreviation_candidate():
    text = "HR 72 bpm"
    entity = {
        "text": "HR",
        "entity_group": "CHEM",
        "score": 0.99,
        "start": 0,
        "end": 2,
    }

    candidates = filter_medication_candidates(
        text,
        [entity],
        grounder=lambda _surface: [Candidate("LOCAL", "1", "HR", 1.0)],
    )

    assert candidates[0].text == "HR"
    assert candidates[0].validation_performed is True
    assert candidates[0].grounding_candidates[0].code == "1"


def test_strict_grounding_keeps_only_matched_candidates():
    text = "aspirin and regionalbrand"
    entities = []
    for surface in ("aspirin", "regionalbrand"):
        start = text.index(surface)
        entities.append(
            {
                "text": surface,
                "label": "CHEM",
                "confidence": 0.99,
                "start": start,
                "end": start + len(surface),
            }
        )

    def grounder(surface):
        if surface == "aspirin":
            return [Candidate("RXNORM", "1191", "aspirin", 1.0)]
        return []

    candidates = filter_medication_candidates(
        text,
        entities,
        preset=MedicationCandidatePreset(require_grounding=True),
        grounder=grounder,
    )

    assert [item.text for item in candidates] == ["aspirin"]


def test_strict_grounding_requires_a_grounder():
    with pytest.raises(ValueError, match="requires a grounder"):
        filter_medication_candidates(
            "aspirin",
            [
                {
                    "text": "aspirin",
                    "label": "CHEM",
                    "confidence": 0.99,
                    "start": 0,
                    "end": 7,
                }
            ],
            preset=MedicationCandidatePreset(require_grounding=True),
        )


def test_unknown_medication_candidate_preset_is_rejected():
    assert MEDICATION_CANDIDATES == "medication_candidates"
    with pytest.raises(ValueError, match="unknown medication candidate preset"):
        filter_medication_candidates("aspirin", [], preset="unknown")
