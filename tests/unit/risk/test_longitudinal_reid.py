from __future__ import annotations

import json

import pytest

from openmed.risk import build_longitudinal_corpus, longitudinal_risk_report

HMAC_KEY = "unit-longitudinal-key"


def _longitudinal_records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "alpha-note-1",
            "patient_id": "patient-alpha",
            "text": "Follow-up note: 70-year-old with [RARE_CONDITION].",
            "age": 70,
            "diagnosis": "rare alpha syndrome",
            "audit_spans": [
                {
                    "canonical_label": "PERSON",
                    "surrogate": "Jordan Vale",
                    "text_hash": "sha256:alpha-person",
                }
            ],
        },
        {
            "record_id": "alpha-note-2",
            "patient_id": "patient-alpha",
            "text": "Second note: 71-year-old returned for [RARE_CONDITION].",
            "age": 71,
            "diagnosis": "rare alpha syndrome",
            "audit_spans": [
                {
                    "canonical_label": "PERSON",
                    "surrogate": "Jordan Vale",
                    "text_hash": "sha256:alpha-person",
                }
            ],
        },
        {
            "record_id": "beta-note-1",
            "patient_id": "patient-beta",
            "text": "Routine visit: 71-year-old with hypertension.",
            "age": 71,
            "diagnosis": "hypertension",
            "audit_spans": [
                {
                    "canonical_label": "PERSON",
                    "surrogate": "Casey Rowan",
                    "text_hash": "sha256:beta-person",
                }
            ],
        },
    ]


def test_longitudinal_corpus_hashes_patient_keys_notes_and_evidence() -> None:
    corpus = build_longitudinal_corpus(_longitudinal_records(), hmac_key=HMAC_KEY)

    assert corpus.patient_count == 2
    assert corpus.document_count == 3
    assert all(
        patient.patient_pseudonym.startswith("hmac-sha256:")
        for patient in corpus.patients
    )
    assert all(
        note.note_hash.startswith("hmac-sha256:")
        for patient in corpus.patients
        for note in patient.notes
    )
    assert all(
        item.value_hash.startswith("hmac-sha256:")
        for patient in corpus.patients
        for item in patient.evidence
    )

    payload = json.dumps(corpus.to_dict(), sort_keys=True)
    assert "patient-alpha" not in payload
    assert "alpha-note-1" not in payload
    assert "Jordan Vale" not in payload
    assert "rare alpha syndrome" not in payload


def test_longitudinal_corpus_inherits_patient_key_from_grouped_notes() -> None:
    corpus = build_longitudinal_corpus(
        {
            "patient_id": "patient-alpha",
            "notes": [
                {"record_id": "alpha-note-1", "text": "Age 70.", "age": 70},
                {"record_id": "alpha-note-2", "text": "Age 71.", "age": 71},
            ],
        },
        hmac_key=HMAC_KEY,
    )

    assert corpus.patient_count == 1
    assert corpus.document_count == 2
    assert corpus.patients[0].document_count == 2


def test_longitudinal_risk_report_exposes_only_hashed_breakdown() -> None:
    report = longitudinal_risk_report(_longitudinal_records(), hmac_key=HMAC_KEY)

    assert report["patient_count"] == 2
    assert report["document_count"] == 3
    assert report["residual_direct_identifier_leakage"] == pytest.approx(0.0)
    assert report["residual_direct_identifier_leakage_count"] == 0
    assert report["linkage_success_upper_bound"] == pytest.approx(1.0)
    assert report["linkable_patient_count"] == 1

    high_risk = report["high_risk_patients"][0]
    assert high_risk["document_count"] == 2
    assert high_risk["stable_surrogate_reuse_count"] == 1
    assert high_risk["age_observation_count"] == 2
    assert high_risk["rare_attribute_count"] == 2
    assert high_risk["attack_fingerprint"]

    payload = json.dumps(report, sort_keys=True)
    assert "patient-alpha" not in payload
    assert "patient-beta" not in payload
    assert "Jordan Vale" not in payload
    assert "Casey Rowan" not in payload
    assert "rare alpha syndrome" not in payload
    assert "hypertension" not in payload
    assert '"value"' not in payload


def test_longitudinal_upper_bound_is_monotone_as_documents_are_added() -> None:
    records = _longitudinal_records()
    prefixes = [records[:1], records[:2], records[:3]]

    bounds = [
        longitudinal_risk_report(prefix, hmac_key=HMAC_KEY)[
            "linkage_success_upper_bound"
        ]
        for prefix in prefixes
    ]

    assert bounds == sorted(bounds)
    assert bounds == [0.0, 1.0, 1.0]
