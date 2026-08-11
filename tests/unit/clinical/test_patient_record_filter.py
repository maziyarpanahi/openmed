"""Tests for the experiencer-aware patient-record span filter (OM-304)."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    AFFIRMED,
    CERTAIN,
    FAMILY_EXPERIENCER,
    HYPOTHETICAL,
    NEGATED,
    OTHER_EXPERIENCER,
    PATIENT_EXPERIENCER,
    PATIENT_RECORD_FILTER_ADVISORY,
    RECENT,
    ClinicalAssertion,
    PatientRecordSpan,
    filter_patient_record,
)


def _span(text: str = "diabetes") -> dict[str, object]:
    return {"text": text, "start": 0, "end": len(text), "label": "CONDITION"}


def _assertion(
    *,
    experiencer: str | None = PATIENT_EXPERIENCER,
    negation: str = AFFIRMED,
    temporality: str = RECENT,
    certainty: str = CERTAIN,
) -> ClinicalAssertion:
    return ClinicalAssertion(
        temporality=temporality,
        certainty=certainty,
        negation=negation,
        experiencer=experiencer,
    )


def test_empty_input_returns_empty_lists():
    included, excluded = filter_patient_record([], [])

    assert included == []
    assert excluded == []


def test_affirmed_patient_span_is_recorded():
    span = _span()
    assertion = _assertion()

    included, excluded = filter_patient_record([span], [assertion])

    assert included == [
        PatientRecordSpan(
            span=span,
            assertion=assertion,
            record_status="recorded",
            exclusion_reason=None,
        )
    ]
    assert excluded == []


def test_negated_patient_span_is_included_and_refuted():
    span = _span()
    assertion = _assertion(negation=NEGATED)

    included, excluded = filter_patient_record([span], [assertion])

    assert excluded == []
    assert len(included) == 1
    assert included[0].record_status == "refuted"
    assert included[0].exclusion_reason is None


def test_hypothetical_patient_span_is_excluded():
    span = _span()
    assertion = _assertion(temporality=HYPOTHETICAL)

    included, excluded = filter_patient_record([span], [assertion])

    assert included == []
    assert excluded == [
        PatientRecordSpan(
            span=span,
            assertion=assertion,
            record_status=None,
            exclusion_reason="hypothetical",
        )
    ]


def test_other_experiencer_is_excluded_as_non_patient():
    span = _span()
    assertion = _assertion(experiencer=OTHER_EXPERIENCER)

    included, excluded = filter_patient_record([span], [assertion])

    assert included == []
    assert excluded == [
        PatientRecordSpan(
            span=span,
            assertion=assertion,
            record_status=None,
            exclusion_reason="non-patient experiencer",
        )
    ]


def test_family_experiencer_is_excluded_as_non_patient():
    span = _span()
    assertion = _assertion(experiencer=FAMILY_EXPERIENCER)

    included, excluded = filter_patient_record([span], [assertion])

    assert included == []
    assert len(excluded) == 1
    assert excluded[0].exclusion_reason == "non-patient experiencer"


def test_non_patient_boundary_outranks_negation():
    span = _span()
    assertion = _assertion(experiencer=FAMILY_EXPERIENCER, negation=NEGATED)

    included, excluded = filter_patient_record([span], [assertion])

    assert included == []
    assert excluded[0].exclusion_reason == "non-patient experiencer"


def test_hypothetical_boundary_outranks_negation():
    span = _span()
    assertion = _assertion(temporality=HYPOTHETICAL, negation=NEGATED)

    included, excluded = filter_patient_record([span], [assertion])

    assert included == []
    assert excluded[0].exclusion_reason == "hypothetical"


def test_unset_experiencer_defaults_to_patient():
    span = _span()
    assertion = ClinicalAssertion(
        temporality=RECENT,
        certainty=CERTAIN,
        negation=AFFIRMED,
    )

    included, excluded = filter_patient_record([span], [assertion])

    assert excluded == []
    assert included[0].record_status == "recorded"


def test_mixed_spans_partition_correctly():
    spans = [_span("diabetes"), _span("asthma"), _span("rash"), _span("donor")]
    assertions = [
        _assertion(),
        _assertion(negation=NEGATED),
        _assertion(temporality=HYPOTHETICAL),
        _assertion(experiencer=OTHER_EXPERIENCER),
    ]

    included, excluded = filter_patient_record(spans, assertions)

    assert len(included) == 2
    assert len(excluded) == 2
    assert {item.record_status for item in included} == {"recorded", "refuted"}
    assert {item.exclusion_reason for item in excluded} == {
        "hypothetical",
        "non-patient experiencer",
    }

    all_results = included + excluded
    assert len(all_results) == len(spans)
    assert len({id(item.span) for item in all_results}) == len(spans)
    for item in all_results:
        assert any(item.span is s for s in spans)


def test_mismatched_span_assertion_lengths_raise():
    with pytest.raises(ValueError, match="spans and assertions must have"):
        filter_patient_record([_span(), _span()], [_assertion()])


def test_advisory_is_non_empty_string():
    assert isinstance(PATIENT_RECORD_FILTER_ADVISORY, str)
    assert PATIENT_RECORD_FILTER_ADVISORY
    assert "not a clinical decision" in PATIENT_RECORD_FILTER_ADVISORY
