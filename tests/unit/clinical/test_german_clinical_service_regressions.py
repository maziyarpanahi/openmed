"""Authored German subject, section and medication regressions for service reuse."""

import pytest

from openmed.clinical.context import assert_context
from openmed.clinical.experiencer import refine_experiencer, resolve_experiencer
from openmed.clinical.medication_sig import normalize_frequency
from openmed.clinical.sections import detect_sections, validate_section_spans


def span(text, surface, label="Disease"):
    start = text.index(surface)
    return {"start": start, "end": start + len(surface), "label": label}


def test_german_medication_header_stops_family_history_scope():
    text = "Familienanamnese: Mutter mit Diabetes.\r\nMedikation: Metformin 500 mg zweimal täglich.\r\nAllergien: Keine Penicillinallergie."
    sections = detect_sections(text, language="de")
    validate_section_spans(text, sections)
    assert [s.label for s in sections] == ["family_history", "medications", "allergies"]
    records = assert_context(
        text,
        [span(text, "Diabetes"), span(text, "Metformin", "Drug")],
        language="de",
        sections=sections,
    )
    assert records[0]["experiencer"] == "family"
    assert records[1]["experiencer"] == "patient"
    assert records[1]["context_sources"]["experiencer"] != "local"
    assert (
        normalize_frequency("zweimal täglich", language="de")["frequency_per_day"] == 2
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Mutter mit Diabetes.", "family"),
        ("Der Vater hat Diabetes.", "family"),
        ("Großmutter mit Diabetes.", "family"),
        ("Spender mit Diabetes.", "other"),
        ("Mutter hat Asthma, Patientin hat Diabetes.", "patient"),
        ("Mutter hat Asthma, aber Diabetes liegt nicht vor.", "patient"),
        ("Mutter hat Asthma. Diabetes liegt nicht vor.", "patient"),
        ("Mutter hat Asthma.\nDiabetes liegt nicht vor.", "patient"),
        ("Muttersprachlerin mit Diabetes.", "patient"),
    ],
)
def test_local_german_experiencer_is_scoped_and_does_not_match_subwords(text, expected):
    entity = span(text, "Diabetes")
    assignment = resolve_experiencer(text, entity, language="de-DE")
    assert assignment.experiencer == expected
    record = assert_context(text, [entity], language="de")[0]
    assert record["experiencer"] == expected
    if assignment.cue_offset:
        assert text[slice(*assignment.cue_offset)].lower() == assignment.cue


def test_explicit_patient_cue_overrides_german_family_section_prior():
    text = "Familienanamnese: Patientin hat Diabetes."
    entity = span(text, "Diabetes")
    record = assert_context(
        text, [entity], language="de", sections=detect_sections(text, language="de")
    )[0]
    assert record["experiencer"] == "patient"
    assert record["context_sources"]["experiencer"] == "local"
    refined = refine_experiencer(
        text, entity, language="de", section_experiencer="family"
    )
    assert refined.experiencer == "patient"


def test_german_negation_stays_independent_from_subject_attribution():
    text = "Mutter ohne Diabetes. Patientin ohne Dyspnoe."
    records = assert_context(
        text, [span(text, "Diabetes"), span(text, "Dyspnoe", "Symptom")], language="de"
    )
    assert [(r["experiencer"], r["negation"]) for r in records] == [
        ("family", "negated"),
        ("patient", "negated"),
    ]


@pytest.mark.parametrize(
    "text,frequency",
    [
        ("einmal täglich", 1),
        ("zweimal täglich", 2),
        ("dreimal taeglich", 3),
        ("viermal täglich", 4),
    ],
)
def test_written_german_frequency_preserves_source_and_maps_deterministically(
    text, frequency
):
    normalized = normalize_frequency(text, language="de")
    assert normalized["recognized"] and normalized["frequency_per_day"] == frequency
    assert normalized["raw"] == text
    assert not normalized["as_needed"]


@pytest.mark.parametrize(
    "text",
    [
        "zweimal täglich vergessen",
        "nicht zweimal täglich",
        "zweimal wöchentlich",
        "zweimal täglich?",
        "nur bei Bedarf",
    ],
)
def test_ambiguous_or_different_frequency_is_not_promoted_to_known_schedule(text):
    result = normalize_frequency(text, language="de")
    assert result["frequency_per_day"] is None
