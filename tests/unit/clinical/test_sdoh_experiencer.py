"""Synthetic offline tests for SDOH experiencer filtering."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sdoh import SDOHFinding
from openmed.clinical.sdoh_experiencer import (
    FAMILY,
    HOUSEHOLD,
    PATIENT,
    SDOH_EXPERIENCER_SCHEMA_VERSION,
    UNKNOWN,
    SDOHExperiencerEvidence,
    SDOHExperiencerFilterResult,
    classify_sdoh_experiencer,
    classify_sdoh_experiencers,
    filter_sdoh_evidence,
    filter_sdoh_findings,
)


def _candidate(text: str, value: str) -> dict[str, int]:
    start = text.index(value)
    return {"start": start, "end": start + len(value)}


def test_classifies_patient_household_family_and_unknown_without_raw_output():
    text = (
        "The patient reports factor alpha. "
        "The household member reports factor beta. "
        "The patient's mother reports factor gamma. "
        "The source of factor delta is unknown."
    )
    candidates = [
        _candidate(text, "factor alpha"),
        _candidate(text, "factor beta"),
        _candidate(text, "factor gamma"),
        _candidate(text, "factor delta"),
    ]

    records = classify_sdoh_experiencers(text, candidates)

    assert [record.experiencer for record in records] == [
        PATIENT,
        HOUSEHOLD,
        FAMILY,
        UNKNOWN,
    ]
    assert [record.source for record in records] == ["cue"] * 4
    assert records[0].cue_offsets == (
        (text.index("patient"), text.index("patient") + 7),
    )
    assert records[1].cue_offsets == (
        (text.index("household member"), text.index("household member") + 16),
    )
    assert records[2].cue_offsets == ((text.index("mother"), text.index("mother") + 6),)
    assert records[3].review_required is True

    serialized = json.dumps([record.to_dict() for record in records])
    assert "factor alpha" not in serialized
    assert "factor beta" not in serialized
    assert "value" not in serialized
    assert "text" not in serialized


def test_patient_filter_excludes_non_patient_and_preserves_review_metadata():
    text = (
        "The patient reports factor alpha. "
        "The roommate reports factor beta. "
        "A family member reports factor gamma."
    )

    result = filter_sdoh_findings(
        text,
        [
            _candidate(text, "factor alpha"),
            _candidate(text, "factor beta"),
            _candidate(text, "factor gamma"),
        ],
    )

    assert isinstance(result, SDOHExperiencerFilterResult)
    assert [item.experiencer for item in result.patient] == [PATIENT]
    assert [item.experiencer for item in result.excluded] == [HOUSEHOLD, FAMILY]
    assert result.patient[0].patient_record_eligible is True
    assert all(
        item.exclusion_reason == "non-patient experiencer" for item in result.excluded
    )
    assert result.all_evidence == result.patient_evidence + result.excluded_evidence

    included, excluded = result
    assert included == result.patient_evidence
    assert excluded == result.excluded_evidence
    report = result.to_dict()
    assert report["schema_version"] == SDOH_EXPERIENCER_SCHEMA_VERSION
    assert SDOHExperiencerFilterResult.from_dict(report) == result
    assert "factor alpha" not in result.to_json()


def test_section_priors_are_conservative_and_local_cues_override_them():
    text = (
        "Social History:\n"
        "factor alpha\n"
        "The patient's roommate reports factor beta\n"
        "Family History:\n"
        "factor gamma"
    )
    sections = [
        {"start": 0, "end": text.index("Family History:"), "label": "Social History"},
        {
            "start": text.index("Family History:"),
            "end": len(text),
            "label": "Family History",
        },
    ]

    records = classify_sdoh_experiencers(
        text,
        [
            _candidate(text, "factor alpha"),
            _candidate(text, "factor beta"),
            _candidate(text, "factor gamma"),
        ],
        sections=sections,
    )

    assert [(record.experiencer, record.source) for record in records] == [
        (PATIENT, "section"),
        (HOUSEHOLD, "cue"),
        (FAMILY, "section"),
    ]


def test_contrastive_clause_does_not_leak_family_cue_to_patient_finding():
    text = "A family member reports factor alpha, but the patient reports factor beta."
    records = classify_sdoh_experiencers(
        text,
        [_candidate(text, "factor alpha"), _candidate(text, "factor beta")],
    )

    assert [record.experiencer for record in records] == [FAMILY, PATIENT]
    assert records[1].conflicting_experiencers == ()


def test_subject_change_after_a_candidate_does_not_retroactively_reclassify_it():
    text = "The patient reports factor alpha, and the mother reports factor beta."
    records = classify_sdoh_experiencers(
        text,
        [_candidate(text, "factor alpha"), _candidate(text, "factor beta")],
    )

    assert [record.experiencer for record in records] == [PATIENT, FAMILY]


def test_conflicting_subjects_abstain_to_unknown_for_review():
    text = "The patient and roommate report factor alpha."
    record = classify_sdoh_experiencer(text, _candidate(text, "factor alpha"))

    assert record.experiencer == UNKNOWN
    assert record.conflicting_experiencers == (PATIENT, HOUSEHOLD)
    assert record.review_required is True
    assert record.patient_record_eligible is False


def test_accepts_sdoh_finding_and_preserves_input_index_after_offset_sorting():
    text = "The patient reports factor alpha and factor beta."
    later = SDOHFinding(
        category="synthetic",
        value="factor beta",
        status="current",
        extent=None,
        temporality="recent",
        span=(
            text.index("factor beta"),
            text.index("factor beta") + len("factor beta"),
        ),
        score=1.0,
    )
    earlier = {
        "start": text.index("factor alpha"),
        "end": text.index("factor alpha") + len("factor alpha"),
    }

    records = classify_sdoh_experiencers(text, [later, earlier])

    assert [record.source_offsets for record in records] == [
        (earlier["start"], earlier["end"]),
        later.span,
    ]
    assert [record.input_index for record in records] == [1, 0]
    assert all(record.experiencer == PATIENT for record in records)


def test_preclassified_value_free_evidence_can_be_filtered_and_round_tripped():
    records = (
        SDOHExperiencerEvidence(
            source_offsets=(0, 5),
            experiencer=PATIENT,
            source="cue",
            cue_offsets=((0, 5),),
            input_index=0,
        ),
        SDOHExperiencerEvidence(
            source_offsets=(6, 11),
            experiencer=UNKNOWN,
            source="default",
            review_required=True,
            input_index=1,
        ),
    )

    result = filter_sdoh_evidence(records)

    assert result.patient_evidence == (records[0],)
    assert result.excluded_evidence == (records[1],)
    assert SDOHExperiencerEvidence.from_dict(records[0].to_dict()) == records[0]


@pytest.mark.parametrize(
    ("bad_evidence", "message"),
    [
        ({"start": -1, "end": 2}, "offsets"),
        ({"start": 0, "end": 200}, "source text"),
    ],
)
def test_invalid_input_errors_are_content_free(bad_evidence, message):
    text = "factor alpha"

    with pytest.raises((TypeError, ValueError), match=message) as exc_info:
        classify_sdoh_experiencers(text, [bad_evidence])

    assert "factor alpha" not in str(exc_info.value)
