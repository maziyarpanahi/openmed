"""Policy conflicts, conservative language routing and role-specific decisions."""

from __future__ import annotations

import pytest

from openmed.core.clinical_language import (
    ReliableClinicalLanguageIdentifier,
    normalize_clinical_language,
    resolve_clinical_language,
)
from openmed.core.clinical_policy import resolve_clinical_policy
from openmed.core.language_router import LanguagePrediction, LanguageRouter


@pytest.mark.parametrize(
    "label", ["PERSON", "date_of_birth", "email", "city", "postcode", "MRN", "PHONE"]
)
def test_default_profile_masks_requested_direct_identifiers(label):
    policy = resolve_clinical_policy()
    assert policy.profile.action_for(label) == "mask"
    assert policy.profile.safety_sweep_mandatory
    assert not policy.narrowed


@pytest.mark.parametrize(
    "label",
    ["DATE", "AGE", "GENDER", "DISEASE", "DRUG", "DOSAGE", "LAB_VALUE", "OCCUPATION"],
)
def test_clinical_content_and_treatment_dates_remain_kept(label):
    assert resolve_clinical_policy().profile.action_for(label) == "keep"


def test_explicit_date_category_changes_treatment_date_policy():
    policy = resolve_clinical_policy(redact_categories=["dates", "date_of_birth"])
    assert policy.profile.action_for("DATE") == "mask"
    assert policy.narrowed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"redact_categories": ["invented"]},
        {"redact_categories": []},
        {"redact_categories": "email"},
        {"redact_roles": ["family"]},
        {"redact_roles": []},
        {"keep_labels": ["invented"]},
        {"keep_labels": ["DATE_OF_BIRTH"]},
        {"keep_labels": ["FIRST_NAME"]},
    ],
)
def test_unknown_or_conflicting_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_clinical_policy(**kwargs)


def test_role_selection_requires_evidence_and_marks_unknown_names_for_review():
    text = "Patient: Anna Beispiel. Arzt: Dr. Parkinson. John Doe called."
    policy = resolve_clinical_policy(redact_roles=["patient"])
    assert policy.decide("PERSON", text=text, start=9, end=22).action == "mask"
    start = text.index("Parkinson")
    doctor = policy.decide("PERSON", text=text, start=start, end=start + 9)
    assert doctor.action == "keep"
    assert doctor.role == "clinician"
    start = text.index("John Doe")
    unknown = policy.decide("PERSON", text=text, start=start, end=start + 8)
    assert unknown.action == "mask"
    assert unknown.needs_review
    assert unknown.reason == "person_role_uncertain"
    assert policy.metadata()["policy_narrowed"]


@pytest.mark.parametrize(
    "value, expected",
    [
        ("DE", ("de", None)),
        ("de-DE", ("de", "de_DE")),
        ("de_AT", ("de", "de_AT")),
        (" en-GB ", ("en", "en_GB")),
        ("AUTO", ("auto", None)),
    ],
)
def test_language_aliases_keep_locale(value, expected):
    assert normalize_clinical_language(value) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "unknown"},
        {"language": "de-ZZ"},
        {"language": "de-DE", "locale": "de_AT"},
        {"language": "de", "locale": "en_US"},
        {"locale": "de"},
    ],
)
def test_invalid_language_and_locale_conflicts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_clinical_language("synthetic text", **kwargs)


def test_explicit_german_route_does_not_require_lid_dependency():
    result = resolve_clinical_language("Kurznarkose", language="DE")
    assert (result.language, result.locale, result.source) == (
        "de",
        "de_DE",
        "explicit",
    )
    assert not result.needs_review


def test_automatic_pack_fallback_cannot_claim_certain_language():
    result = resolve_clinical_language(
        "Sample note", router=LanguageRouter(use_optional_lid=False)
    )
    assert result.needs_review


def test_mixed_paragraph_languages_keep_original_coordinate_runs():
    class Identifier:
        name = "fixture"

        def identify(self, text, candidates):
            return LanguagePrediction("de" if "Dyspnoe" in text else "en", 0.99)

    text = "Keine Dyspnoe.\r\nThe patient has no chest pain."
    result = resolve_clinical_language(
        text, router=LanguageRouter(language_identifier=Identifier())
    )
    assert result.mixed and result.needs_review
    assert {run.language for run in result.runs} >= {"de", "en"}
    assert result.runs[0].start == 0
    assert result.runs[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(result.runs, result.runs[1:]))


def test_unreliable_cld2_result_is_not_promoted(monkeypatch):
    from types import SimpleNamespace

    identifier = ReliableClinicalLanguageIdentifier()
    module = SimpleNamespace(
        detect=lambda *args, **kwargs: (False, 20, [("German", "de", 99, 1)])
    )
    monkeypatch.setattr(identifier, "_load", lambda: module)
    assert identifier.identify("Short note", ["de", "en"]) is None


def test_numeric_only_text_has_unknown_language():
    result = resolve_clinical_language("12345\n67890")
    assert result.language == "und" and result.needs_review
