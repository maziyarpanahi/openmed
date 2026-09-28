"""Synthetic offline tests for SDOH temporal qualifiers."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sdoh import SDOHFinding
from openmed.clinical.sdoh_temporal import (
    CURRENT,
    FUTURE,
    HISTORICAL,
    UNKNOWN,
    SDOHTemporalEvidence,
    qualify_sdoh_evidence,
)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    (
        ("Currently has synthetic housing support.", CURRENT),
        ("Current synthetic housing support is recorded.", CURRENT),
        ("Formerly had synthetic housing instability.", HISTORICAL),
        ("Plans to seek synthetic housing next year.", FUTURE),
        ("synthetic housing status is not documented.", UNKNOWN),
    ),
)
def test_temporal_classes_require_explicit_local_cues(
    phrase: str,
    expected: str,
) -> None:
    finding_start = phrase.index("synthetic housing")
    finding_end = finding_start + len("synthetic housing")

    [result] = qualify_sdoh_evidence(
        phrase,
        [{"start": finding_start, "end": finding_end}],
    )

    assert result.temporal_class == expected
    assert result.source_offsets == (finding_start, finding_end)
    assert result.review_required is (expected == UNKNOWN)
    assert result.conflicting_classes == ()


def test_adversative_boundaries_keep_temporal_cues_with_their_finding() -> None:
    phrase = (
        "Formerly had synthetic housing instability, but currently has "
        "synthetic employment."
    )
    housing_start = phrase.index("synthetic housing")
    employment_start = phrase.index("synthetic employment")
    evidence = [
        {"start": housing_start, "end": housing_start + len("synthetic housing")},
        {
            "start": employment_start,
            "end": employment_start + len("synthetic employment"),
        },
    ]

    housing, employment = qualify_sdoh_evidence(phrase, evidence)

    assert housing.temporal_class == HISTORICAL
    assert employment.temporal_class == CURRENT
    assert housing.review_required is False
    assert employment.review_required is False


def test_missing_time_cue_does_not_default_a_finding_to_current() -> None:
    phrase = "Synthetic employment concern is recorded."
    start = phrase.index("Synthetic employment")
    end = start + len("Synthetic employment")

    [result] = qualify_sdoh_evidence(phrase, [(start, end)])

    assert result.temporal_class == UNKNOWN
    assert result.review_required is True
    assert result.cue_offsets == ()


def test_conflicting_cues_are_unknown_and_require_review() -> None:
    phrase = "Previously and currently had synthetic housing instability."
    start = phrase.index("synthetic housing")
    end = start + len("synthetic housing")

    [result] = qualify_sdoh_evidence(
        phrase,
        [SDOHFinding("housing", "instability", None, None, None, (start, end), 1.0)],
    )

    assert result.temporal_class == UNKNOWN
    assert result.conflicting_classes == (CURRENT, HISTORICAL)
    assert result.review_required is True
    assert len(result.cue_offsets) == 2
    assert phrase[result.source_offsets[0] : result.source_offsets[1]] == (
        "synthetic housing"
    )


def test_trailing_historical_cue_preserves_finding_offsets() -> None:
    phrase = "Synthetic employment concern in 2020."
    start = phrase.index("Synthetic employment")
    end = start + len("Synthetic employment")

    [result] = qualify_sdoh_evidence(
        phrase,
        [{"source_offsets": {"start": start, "end": end}}],
    )

    assert result.temporal_class == HISTORICAL
    assert result.source_span == (start, end)
    cue_start = phrase.index("in 2020")
    assert result.cue_offsets == ((cue_start, cue_start + 7),)


def test_results_are_sorted_and_serialized_without_source_values() -> None:
    phrase = "Current synthetic job. Former synthetic housing concern."
    job_start = phrase.index("synthetic job")
    housing_start = phrase.index("synthetic housing")
    evidence = [
        {"start": housing_start, "end": housing_start + len("synthetic housing")},
        {"start": job_start, "end": job_start + len("synthetic job")},
    ]

    results = qualify_sdoh_evidence(phrase, evidence)
    serialized = json.dumps([result.to_dict() for result in results], sort_keys=True)

    assert [result.source_offsets for result in results] == [
        (job_start, job_start + len("synthetic job")),
        (housing_start, housing_start + len("synthetic housing")),
    ]
    assert "synthetic job" not in serialized
    assert "synthetic housing" not in serialized
    assert all("text" not in result.to_dict() for result in results)


def test_record_round_trip_is_value_free() -> None:
    record = SDOHTemporalEvidence(
        source_offsets=(4, 12),
        temporal_class=HISTORICAL,
        cue_offsets=((0, 3),),
        review_required=False,
    )

    assert SDOHTemporalEvidence.from_dict(record.to_dict()) == record
    assert "synthetic" not in record.to_json()


def test_malformed_offsets_fail_without_echoing_input() -> None:
    with pytest.raises(ValueError, match="within the source text"):
        qualify_sdoh_evidence("Synthetic housing.", [{"start": 0, "end": 999}])
