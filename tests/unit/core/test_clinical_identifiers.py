"""Synthetic context and counterexample tests for clinical identity fields."""

from __future__ import annotations

from datetime import datetime

import pytest

from openmed.core.clinical_identifiers import (
    detect_clinical_identifiers,
    personal_name_role,
)
from openmed.core.clinical_protect import filter_protected_spans, protect_spans
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _entity(text, surface, label):
    start = text.index(surface)
    return EntityPrediction(surface, label, 0.99, start, start + len(surface))


@pytest.mark.parametrize(
    "text, expected, role",
    [
        (
            "Patientin: Anna Beispiel, geboren am 14.03.1962.",
            "Anna Beispiel",
            "patient",
        ),
        ("Patient: Jean de Meier. Keine Dyspnoe.", "Jean de Meier", "patient"),
        ("Ärztin: Dr. med. Erika Müller; Kardiologie", "Erika Müller", "clinician"),
        (
            "Patient name: A. B. Example\nDiagnosis: No dyspnea",
            "A. B. Example",
            "patient",
        ),
        ("Dr. Parkinson veranlasste die Untersuchung.", "Parkinson", "clinician"),
        ("Patient: Dr. Parkinson; Morbus Parkinson", "Parkinson", "patient"),
    ],
)
def test_complete_names_have_explicit_role_and_exact_source(text, expected, role):
    entities = [
        e
        for e in detect_clinical_identifiers(text, language="de")
        if e.label == "PERSON"
    ]
    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == expected == text[entity.start : entity.end]
    assert entity.metadata["clinical_role"] == role
    assert personal_name_role(text, entity.end - 2, entity.end) == role


@pytest.mark.parametrize(
    "text",
    [
        "Sozialanamnese: Nichtraucherin. Kardioversion in Kurznarkose.",
        "Morbus Parkinson. Keine Dyspnoe. LVEF 55 %. Metoprolol 47,5 mg.",
        "The patient denies dyspnea. No family history of Parkinson disease.",
    ],
)
def test_unanchored_clinical_words_are_not_names(text):
    assert detect_clinical_identifiers(text, language="de") == []


def test_dob_is_distinct_from_treatment_dates_and_rejects_invalid_calendar_date():
    text = (
        "Geburtsdatum: 14.03.1962. Aufnahme: 07.09.2026. "
        "Geboren am 31.02.1962. Entlassung: 09.09.2026."
    )
    assert [
        (e.text, e.label) for e in detect_clinical_identifiers(text, language="de")
    ] == [
        ("14.03.1962", "DATE_OF_BIRTH"),
    ]


def test_patient_id_is_not_a_lab_value():
    text = "Patienten-ID: DE-00123; LVEF 55 %; Hb 13,2 g/dl."
    assert [
        (e.text, e.label) for e in detect_clinical_identifiers(text, language="de")
    ] == [
        ("DE-00123", "ID_NUM"),
    ]


def test_anchored_phone_includes_country_code_but_not_clinical_numbers():
    text = "Phone: +1 202-555-0142. LVEF 55 %. Metoprolol 47,5 mg."
    assert [
        (e.text, e.label) for e in detect_clinical_identifiers(text, language="en")
    ] == [
        ("+1 202-555-0142", "PHONE"),
    ]


def test_decomposed_umlaut_name_preserves_original_offsets():
    text = "Patient: Anna Müller."
    entity = detect_clinical_identifiers(text, language="de")[0]
    assert entity.text == text[entity.start : entity.end] == "Anna Müller"


@pytest.mark.parametrize("header", ["Patient:", "Arzt: Dr.", "Name:"])
def test_eponym_like_real_name_cannot_be_protected(header):
    text = f"{header} Parkinson. Morbus Parkinson."
    name = _entity(text, "Parkinson", "LAST_NAME")
    clinical_start = text.rindex("Parkinson")
    clinical = EntityPrediction(
        "Parkinson", "LAST_NAME", 0.99, clinical_start, clinical_start + 9
    )
    assert protect_spans([name, clinical], text, lang="de") == [name]


def test_clinical_protection_is_whole_span_and_preserves_negation():
    text = (
        "Keine Dyspnoe. Kardioversion in Kurznarkose; Sozialanamnese: Nichtraucherin."
    )
    candidates = [
        _entity(text, "Keine", "first_name"),
        _entity(text, "Kurznarkose", "city"),
        _entity(text, "Sozialanamnese", "last_name"),
    ]
    assert protect_spans(candidates, text, lang="de") == []
    partial = _entity(text, "Kurz", "first_name")
    assert protect_spans([partial], text, lang="de") == [partial]


@pytest.mark.parametrize(
    "text,surface,label,protected",
    [
        ("Keine Dyspnoe.", "Ke", "first_name", True),
        ("Keine Dyspnoe.", " Dys", "last_name", True),
        ("Patient: Keine Parkinson.", " Ke", "first_name", False),
        ("Dr. Parkinson untersucht.", " Park", "last_name", False),
        ("Keinemann berichtet.", "Ke", "first_name", False),
        ("Keine Dyspnoe.", "Keine", "ID_NUM", False),
    ],
)
def test_fragment_protection_requires_whole_clinical_source_word_outside_name_context(
    text, surface, label, protected
):
    candidate = _entity(text, surface, label)
    result = filter_protected_spans(
        [candidate], text, lang="de", protect_word_fragments=True
    )
    assert (result.spans == []) is protected


def test_pipeline_merges_full_name_and_preserves_source_layout():
    text = (
        "Patientin: Anna Beispiel, geboren am 14.03.1962.\r\n"
        "Sozialanamnese: Nichtraucherin. Keine Dyspnoe.\n"
        "Kardioversion in Kurznarkose. Metoprolol 47,5 mg; LVEF 55 %."
    )

    def prediction(source, entities):
        return PredictionResult(
            source, entities, "synthetic-detector", datetime.now().isoformat()
        )

    def model_detector(source, **kwargs):
        return prediction(
            source,
            [
                _entity(source, "Anna", "first_name"),
                _entity(source, "Kurznarkose", "city"),
                _entity(source, "Keine", "first_name"),
            ],
        )

    def context_detector(source, **kwargs):
        return prediction(source, detect_clinical_identifiers(source, language="de"))

    result = Pipeline(
        lang="de",
        model_detector=model_detector,
        clinical_model_detector=context_detector,
        policy="clinical_preserve",
        preserve_whitespace=True,
    ).run(text, method="mask")
    assert "Anna" not in result.redacted_text
    assert "Beispiel" not in result.redacted_text
    assert "14.03.1962" not in result.redacted_text
    assert result.redacted_text.endswith(text[text.index(".\r\n") :])


def test_unsupported_language_has_no_silent_english_context_fallback():
    with pytest.raises(ValueError, match="support de and en"):
        detect_clinical_identifiers("Patient: Jean Meier", language="fr")
