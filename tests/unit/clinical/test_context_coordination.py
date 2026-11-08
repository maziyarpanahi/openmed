"""Coordinated findings inherit ConText cues across "and" / "or" lists.

Every sentence here is synthetic and written for this regression suite.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.clinical import negated_spans
from openmed.clinical.context import (
    AFFIRMED,
    CERTAIN,
    FAMILY_EXPERIENCER,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    RECENT,
    UNCERTAIN,
    assert_context,
)
from openmed.clinical.lexicons import context_cues
from openmed.clinical.lexicons.context_cues import get_clinical_cue_lexicon


def _spans(text: str, terms: tuple[str, ...]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    cursor = 0
    for term in terms:
        start = text.index(term, cursor)
        spans.append(
            {"start": start, "end": start + len(term), "label": "PROBLEM", "text": term}
        )
        cursor = start + len(term)
    return spans


def _axes(
    text: str,
    *terms: str,
    language: str | None = None,
) -> dict[str, dict[str, object]]:
    results = assert_context(text, _spans(text, terms), language=language)
    return {str(result["text"]): result for result in results}


def _negation(text: str, *terms: str, language: str | None = None) -> list[object]:
    axes = _axes(text, *terms, language=language)
    return [axes[term]["negation"] for term in terms]


# ---------------------------------------------------------------------------
# Negation: coordinated findings inherit the trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("Denies fever and cough.", ("fever", "cough")),
        ("No fever or cough.", ("fever", "cough")),
        ("No fever nor cough.", ("fever", "cough")),
        ("No fever and/or cough.", ("fever", "cough")),
        ("No fever, chills, or cough.", ("fever", "chills", "cough")),
        ("No fever, chills and cough.", ("fever", "chills", "cough")),
        ("Presents without fever or cough.", ("fever", "cough")),
        ("Patient denies fever and cough.", ("fever", "cough")),
        ("No fever or cough is reported.", ("fever", "cough")),
        ("No pneumothorax or new consolidation.", ("pneumothorax", "consolidation")),
        (
            "Denies chest pain, shortness of breath with exertion, palpitations, "
            "or syncope.",
            ("chest pain", "shortness of breath", "palpitations", "syncope"),
        ),
    ],
)
def test_forward_negation_covers_every_coordinated_finding(
    text: str, terms: tuple[str, ...]
) -> None:
    assert _negation(text, *terms) == [NEGATED] * len(terms)


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("Fever and cough are absent.", ("Fever", "cough")),
        ("Pneumonia or effusion was ruled out.", ("Pneumonia", "effusion")),
        ("Fever, cough, and chills are absent.", ("Fever", "cough", "chills")),
    ],
)
def test_backward_negation_covers_every_coordinated_finding(
    text: str, terms: tuple[str, ...]
) -> None:
    assert _negation(text, *terms) == [NEGATED] * len(terms)


@pytest.mark.parametrize(
    ("text", "negated", "affirmed"),
    [
        ("No fever but cough.", "fever", "cough"),
        ("No fever however cough.", "fever", "cough"),
        ("No fever although cough.", "fever", "cough"),
        ("Denies fever. Reports cough.", "fever", "cough"),
        ("No fever and reports cough.", "fever", "cough"),
        ("No fever and she reports cough.", "fever", "cough"),
        ("No fever and the patient has cough.", "fever", "cough"),
        ("Denies fever, reports cough.", "fever", "cough"),
        ("Negative for fever and positive for cough.", "fever", "cough"),
        ("Without fever and with cough.", "fever", "cough"),
    ],
)
def test_negation_scope_stops_at_terminators_and_new_clauses(
    text: str, negated: str, affirmed: str
) -> None:
    axes = _axes(text, negated, affirmed)

    assert axes[negated]["negation"] == NEGATED
    assert axes[affirmed]["negation"] == AFFIRMED


@pytest.mark.parametrize(
    ("text", "affirmed", "negated"),
    [
        ("Cough, no fever.", "Cough", "fever"),
        ("Has cough and no fever.", "cough", "fever"),
        ("Has cough and fever absent.", "cough", "fever"),
        ("Cough present and fever absent.", "Cough", "fever"),
    ],
)
def test_cue_scoped_to_a_later_member_does_not_leak_backward(
    text: str, affirmed: str, negated: str
) -> None:
    axes = _axes(text, affirmed, negated)

    assert axes[affirmed]["negation"] == AFFIRMED
    assert axes[negated]["negation"] == NEGATED


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("No fever and no cough.", ("fever", "cough")),
        ("No fever, no chills, no cough.", ("fever", "chills", "cough")),
        ("Denies fever, chills, no cough.", ("fever", "chills", "cough")),
        ("Denies fever or chills, no cough.", ("fever", "chills", "cough")),
    ],
)
def test_repeated_triggers_are_not_double_counted(
    text: str, terms: tuple[str, ...]
) -> None:
    # Negation uses cue parity, so a list member must see exactly one trigger.
    assert _negation(text, *terms) == [NEGATED] * len(terms)


def test_double_negation_stays_consistent_across_a_list() -> None:
    assert _negation("Not without fever or cough.", "fever", "cough") == [
        AFFIRMED,
        AFFIRMED,
    ]


def test_pseudo_negation_in_a_later_conjunct_is_not_a_trigger() -> None:
    axes = _axes("No fever and no increase in effusion.", "fever", "effusion")

    assert axes["fever"]["negation"] == NEGATED
    assert axes["effusion"]["negation"] == AFFIRMED


def test_conjunct_with_its_own_cue_starts_a_new_scope() -> None:
    axes = _axes("No fever and possible pneumonia.", "fever", "pneumonia")

    assert axes["fever"]["negation"] == NEGATED
    assert axes["pneumonia"]["negation"] == AFFIRMED
    assert axes["pneumonia"]["uncertainty"] == UNCERTAIN


# ---------------------------------------------------------------------------
# Uncertainty and temporality share the same scope window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("Possible pneumonia or bronchitis.", ("pneumonia", "bronchitis")),
        (
            "Concern for pneumonia, effusion, or abscess.",
            ("pneumonia", "effusion", "abscess"),
        ),
        ("Pneumonia or effusion cannot be excluded.", ("Pneumonia", "effusion")),
    ],
)
def test_uncertainty_covers_every_coordinated_finding(
    text: str, terms: tuple[str, ...]
) -> None:
    axes = _axes(text, *terms)

    assert [axes[term]["uncertainty"] for term in terms] == [UNCERTAIN] * len(terms)
    assert [axes[term]["negation"] for term in terms] == [AFFIRMED] * len(terms)


@pytest.mark.parametrize(
    ("text", "uncertain", "certain"),
    [
        ("Possible pneumonia but bronchitis is confirmed.", "pneumonia", "bronchitis"),
        ("Possible pneumonia. Bronchitis on exam.", "pneumonia", "Bronchitis"),
        ("Possible pneumonia and reports bronchitis.", "pneumonia", "bronchitis"),
        ("Possible pneumonia and no effusion.", "pneumonia", "effusion"),
    ],
)
def test_uncertainty_scope_stops_at_terminators_and_new_clauses(
    text: str, uncertain: str, certain: str
) -> None:
    axes = _axes(text, uncertain, certain)

    assert axes[uncertain]["uncertainty"] == UNCERTAIN
    assert axes[certain]["uncertainty"] == CERTAIN


def test_temporality_covers_coordinated_findings() -> None:
    history = _axes("History of MI and stroke.", "MI", "stroke")
    negated_history = _axes(
        "No history of diabetes or hypertension.", "diabetes", "hypertension"
    )
    conditional = _axes("If fever or chills develop, return.", "fever", "chills")

    assert [history[term]["temporality"] for term in ("MI", "stroke")] == [
        HISTORICAL,
        HISTORICAL,
    ]
    for term in ("diabetes", "hypertension"):
        assert negated_history[term]["temporality"] == HISTORICAL
        assert negated_history[term]["negation"] == NEGATED
    for term in ("fever", "chills"):
        assert conditional[term]["temporality"] == HYPOTHETICAL


def test_recent_cue_splits_only_the_temporality_scope() -> None:
    axes = _axes("History of MI and new stroke.", "MI", "stroke")

    assert axes["MI"]["temporality"] == HISTORICAL
    assert axes["stroke"]["temporality"] == RECENT


# ---------------------------------------------------------------------------
# Experiencer already spans coordinated findings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Mother has diabetes and hypertension.",
        "Family history of diabetes or hypertension.",
    ],
)
def test_experiencer_covers_coordinated_findings(text: str) -> None:
    axes = _axes(text, "diabetes", "hypertension")

    assert axes["diabetes"]["experiencer"] == FAMILY_EXPERIENCER
    assert axes["hypertension"]["experiencer"] == FAMILY_EXPERIENCER


# ---------------------------------------------------------------------------
# Lexicon contract and agreement with the negation-scope detector
# ---------------------------------------------------------------------------


def test_coordinated_scope_is_an_english_opt_in() -> None:
    assert get_clinical_cue_lexicon("en").coordinators == ("and", "or")
    for language in ("es", "fr", "de", "zh", "hi", "ja", "pt"):
        assert get_clinical_cue_lexicon(language).coordinators == ()


def test_language_pack_opts_in_without_resolver_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = context_cues.ClinicalCueLexicon(
        language="xy",
        negation=("zzno",),
        pseudo_negation=(),
        historical=(),
        hypothetical=(),
        recent=(),
        uncertainty=(),
        backward=(),
        scope_terminators=("zzand", "zzbut"),
        conjunction_terminators=("zzbut",),
    )
    text = "zzno fever zzand cough."
    new_clause = "zzno fever zzand zzhas cough."

    monkeypatch.setitem(context_cues._LEXICONS, "xy", stub)
    assert _negation(text, "fever", "cough", language="xy") == [NEGATED, AFFIRMED]

    coordinated = replace(stub, coordinators=("zzand",), clause_openers=("zzhas",))
    monkeypatch.setitem(context_cues._LEXICONS, "xy", coordinated)
    assert _negation(text, "fever", "cough", language="xy") == [NEGATED, NEGATED]
    assert _negation(new_clause, "fever", "cough", language="xy") == [
        NEGATED,
        AFFIRMED,
    ]


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("Denies fever and cough.", ("fever", "cough")),
        ("No fever, chills, or cough.", ("fever", "chills", "cough")),
        ("No fever but cough.", ("fever", "cough")),
        ("Denies fever. Reports cough.", ("fever", "cough")),
        ("No fever and reports cough.", ("fever", "cough")),
        ("No fever and no cough.", ("fever", "cough")),
        ("Cough, no fever.", ("Cough", "fever")),
        ("Fever and cough are absent.", ("Fever", "cough")),
        ("Pneumonia or effusion ruled out.", ("Pneumonia", "effusion")),
        ("Has cough and fever absent.", ("cough", "fever")),
        ("No history of diabetes or hypertension.", ("diabetes", "hypertension")),
        ("No fever and possible pneumonia.", ("fever", "pneumonia")),
    ],
)
def test_negation_scope_detector_agrees_with_assert_context(
    text: str, terms: tuple[str, ...]
) -> None:
    spans = _spans(text, terms)
    scoped = set(negated_spans(text, spans))
    detector = [
        NEGATED if (span["start"], span["end"]) in scoped else AFFIRMED
        for span in spans
    ]

    assert detector == _negation(text, *terms)
