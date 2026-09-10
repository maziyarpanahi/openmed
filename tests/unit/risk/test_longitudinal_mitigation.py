from __future__ import annotations

import copy
import json
from collections.abc import Callable

import pytest

from openmed.risk import (
    LongitudinalMitigationPolicy,
    longitudinal_risk_report,
    mitigate_longitudinal_linkage,
)

HMAC_KEY = "unit-longitudinal-mitigation-key"


def _records() -> list[dict[str, object]]:
    return [
        {
            "patient_id": "patient-high-risk",
            "record_id": "high-note-1",
            "text": "On 2025-01-10, the 70-year-old had [RARE_CONDITION].",
            "age": 70,
            "visit_date": "2025-01-10",
            "diagnosis": "rare alpha syndrome",
            "audit_spans": [
                {
                    "canonical_label": "PERSON",
                    "surrogate": "Jordan Vale",
                    "text_hash": "sha256:source-person",
                },
                {
                    "canonical_label": "RARE_CONDITION",
                    "value": "rare alpha syndrome",
                },
            ],
        },
        {
            "patient_id": "patient-high-risk",
            "record_id": "high-note-2",
            "text": "On 2025-02-10, the 71-year-old had [RARE_CONDITION].",
            "age": 71,
            "visit_date": "2025-02-10",
            "diagnosis": "rare alpha syndrome",
            "audit_spans": [
                {
                    "canonical_label": "PERSON",
                    "surrogate": "Jordan Vale",
                    "text_hash": "sha256:source-person",
                },
                {
                    "canonical_label": "RARE_CONDITION",
                    "value": "rare alpha syndrome",
                },
            ],
        },
        {
            "patient_id": "patient-low-risk",
            "record_id": "low-note-1",
            "text": "Synthetic routine follow-up.",
            "audit_spans": [
                {
                    "canonical_label": "PERSON",
                    "surrogate": "Casey Rowan",
                    "text_hash": "sha256:other-person",
                }
            ],
        },
    ]


def test_mitigation_brings_highest_risk_cohort_below_ceiling() -> None:
    result = mitigate_longitudinal_linkage(_records(), hmac_key=HMAC_KEY)

    assert result.before_report["linkage_success_upper_bound"] == pytest.approx(1.0)
    assert result.after_report["linkage_success_upper_bound"] == pytest.approx(0.0)
    assert result.meets_ceiling is True
    assert result.after_report["high_risk_patients"] == []
    assert result.records[0]["age"] == 90
    assert result.records[1]["age"] == 51
    assert result.records[0]["visit_date"] == "2025-07-09"
    assert result.records[1]["visit_date"] == "2024-08-14"
    assert "diagnosis" not in result.records[0]
    assert "diagnosis" not in result.records[1]

    first_surrogate = result.records[0]["audit_spans"][0]["surrogate"]
    second_surrogate = result.records[1]["audit_spans"][0]["surrogate"]
    assert first_surrogate != second_surrogate


def test_mitigation_changes_only_cohorts_above_the_ceiling() -> None:
    records = _records()
    low_risk_before = copy.deepcopy(records[2])

    result = mitigate_longitudinal_linkage(records, hmac_key=HMAC_KEY)

    assert result.records[2] == low_risk_before
    assert records == _records()


def test_mitigation_is_deterministic_and_report_is_phi_free() -> None:
    first = mitigate_longitudinal_linkage(_records(), hmac_key=HMAC_KEY)
    second = mitigate_longitudinal_linkage(_records(), hmac_key=HMAC_KEY)

    assert first.records == second.records
    assert first.to_report_dict() == second.to_report_dict()
    report = first.to_report_dict()
    payload = json.dumps(report, sort_keys=True)
    for raw_value in (
        "patient-high-risk",
        "high-note-1",
        "Jordan Vale",
        "rare alpha syndrome",
        "2025-01-10",
    ):
        assert raw_value not in payload
    assert report["meets_ceiling"] is True
    assert report["mitigated_patient_count"] == 1
    assert report["action_count"] == len(first.actions)
    assert all(
        action.before_hash.startswith("hmac-sha256:") for action in first.actions
    )


def test_cohort_controls_can_preserve_consistency_deliberately() -> None:
    policy = LongitudinalMitigationPolicy(
        surrogate_cohort_size=2,
        age_cohort_size=2,
        age_perturbation_years=5,
        date_cohort_size=2,
        date_perturbation_days=30,
        suppress_rare_attributes=False,
    )

    result = mitigate_longitudinal_linkage(
        _records(),
        hmac_key=HMAC_KEY,
        policy=policy,
    )

    first_surrogate = result.records[0]["audit_spans"][0]["surrogate"]
    second_surrogate = result.records[1]["audit_spans"][0]["surrogate"]
    assert first_surrogate == second_surrogate
    assert result.records[0]["age"] == 75
    assert result.records[1]["age"] == 76
    assert result.after_report["linkage_success_upper_bound"] == pytest.approx(1.0)
    assert result.meets_ceiling is False


def test_single_document_release_remains_byte_compatible() -> None:
    record = _records()[2]
    before = longitudinal_risk_report([record], hmac_key=HMAC_KEY)

    result = mitigate_longitudinal_linkage([record], hmac_key=HMAC_KEY)

    assert result.records == (record,)
    assert result.actions == ()
    assert result.before_report == before
    assert result.after_report == before


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: LongitudinalMitigationPolicy(linkage_ceiling=1.1), "linkage_ceiling"),
        (
            lambda: LongitudinalMitigationPolicy(surrogate_cohort_size=0),
            "surrogate_cohort_size",
        ),
        (
            lambda: LongitudinalMitigationPolicy(age_perturbation_years=-1),
            "age_perturbation_years",
        ),
    ],
)
def test_policy_rejects_invalid_controls(
    factory: Callable[[], LongitudinalMitigationPolicy],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_date_trajectory_is_scored_without_serializing_dates() -> None:
    records = [
        {
            "patient_id": "date-patient",
            "record_id": "date-note-1",
            "text": "Synthetic note.",
            "visit_date": "2025-01-01",
        },
        {
            "patient_id": "date-patient",
            "record_id": "date-note-2",
            "text": "Synthetic note.",
            "visit_date": "2025-02-01",
        },
    ]

    report = longitudinal_risk_report(records, hmac_key=HMAC_KEY)

    assert report["linkage_success_upper_bound"] == pytest.approx(1.0)
    assert report["patient_risks"][0]["attack_fingerprint"][0]["category"] == (
        "date_trajectory"
    )
    assert "2025-01-01" not in json.dumps(report, sort_keys=True)


def test_trajectory_bound_stays_monotone_after_an_incoherent_note() -> None:
    records = [
        {
            "patient_id": "age-patient",
            "record_id": "age-note-1",
            "text": "Synthetic note.",
            "age": 70,
        },
        {
            "patient_id": "age-patient",
            "record_id": "age-note-2",
            "text": "Synthetic note.",
            "age": 71,
        },
        {
            "patient_id": "age-patient",
            "record_id": "age-note-3",
            "text": "Synthetic note.",
            "age": 30,
        },
    ]

    bounds = [
        longitudinal_risk_report(records[:count], hmac_key=HMAC_KEY)[
            "linkage_success_upper_bound"
        ]
        for count in range(1, len(records) + 1)
    ]

    assert bounds == [0.0, 1.0, 1.0]
