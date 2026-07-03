"""Tests for section-aware clinical assertion priors."""

from __future__ import annotations

from openmed.clinical import (
    CANONICAL_SECTION_LABELS,
    CERTAIN,
    FAMILY_EXPERIENCER,
    HISTORICAL,
    PATIENT_EXPERIENCER,
    RECENT,
    SECTION_CONTEXT_PRIORS,
    SECTION_LABEL_ALIASES,
    ClinicalAssertion,
    apply_section_context,
    assert_context_axes,
    canonical_section_label,
)
from openmed.core.labels import CONDITION
from openmed.core.schemas import OpenMedSpan, hmac_text_hash


def _span(section: str | None, text: str = "asthma") -> OpenMedSpan:
    return OpenMedSpan(
        doc_id="doc-1",
        start=0,
        end=len(text),
        text_hash=hmac_text_hash(text, "test-secret"),
        entity_type="condition",
        canonical_label=CONDITION,
        section=section,
    )


def test_past_medical_history_span_section_applies_historical_prior() -> None:
    assertion = ClinicalAssertion(temporality=RECENT, certainty=CERTAIN)

    adjusted = apply_section_context(_span("Past Medical History"), None, assertion)

    assert adjusted.temporality == HISTORICAL
    assert adjusted.certainty == CERTAIN


def test_unlabeled_section_leaves_temporality_unchanged() -> None:
    assertion = ClinicalAssertion(temporality=RECENT, certainty=CERTAIN)

    adjusted = apply_section_context(_span(None), None, assertion)

    assert adjusted == assertion


def test_explicit_current_temporal_cue_overrides_section_prior() -> None:
    span = {"text": "acute asthma", "section": "Past Medical History"}

    assertion = assert_context_axes(span)

    assert assertion.temporality == RECENT


def test_explicit_hypothetical_temporal_cue_overrides_section_prior() -> None:
    span = {"text": "if wheezing recurs", "section": "Past Medical History"}

    assertion = assert_context_axes(span)

    assert assertion.temporality == "hypothetical"


def test_section_priors_do_not_cross_spans() -> None:
    assertion = ClinicalAssertion(temporality=RECENT, certainty=CERTAIN)

    pmh = apply_section_context(
        {"text": "asthma", "section": "Past Medical History"},
        None,
        assertion,
    )
    assessment = apply_section_context(
        {"text": "asthma", "section": "Assessment"},
        None,
        assertion,
    )

    assert pmh.temporality == HISTORICAL
    assert assessment.temporality == RECENT


def test_explicit_section_argument_can_supply_section_label() -> None:
    assertion = ClinicalAssertion(temporality=RECENT, certainty=CERTAIN)

    adjusted = apply_section_context({"text": "asthma"}, "PMH", assertion)

    assert adjusted.temporality == HISTORICAL


def test_family_and_social_history_experiencer_priors_are_conservative() -> None:
    assertion = ClinicalAssertion(temporality=RECENT, certainty=CERTAIN)

    family = apply_section_context(
        {"text": "diabetes", "section": "Family History"},
        None,
        assertion,
    )
    social_with_explicit_experiencer = apply_section_context(
        {"text": "tobacco use", "section": "Social History"},
        None,
        ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            experiencer=FAMILY_EXPERIENCER,
        ),
    )

    assert family.experiencer == FAMILY_EXPERIENCER
    assert family.temporality == RECENT
    assert social_with_explicit_experiencer.experiencer == FAMILY_EXPERIENCER
    assert (
        SECTION_CONTEXT_PRIORS["social_history"]["experiencer"] == PATIENT_EXPERIENCER
    )


def test_section_label_mapping_documents_om_086_detector_alignment() -> None:
    docstring = apply_section_context.__doc__ or ""

    assert "OM-086" in docstring
    assert "detect_sections" in docstring
    assert "Past Medical History" in CANONICAL_SECTION_LABELS["past_medical_history"]
    assert SECTION_LABEL_ALIASES["past medical history"] == "past_medical_history"
    assert SECTION_LABEL_ALIASES["hpi"] == "history_of_present_illness"
    assert canonical_section_label("HPI") == "history_of_present_illness"
