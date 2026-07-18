from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.attacks import (
    LinkageAttackResult,
    LongitudinalLinkageAttackResult,
    linkage_attack,
    longitudinal_linkage_attack,
)
from openmed.eval.attacks.reid import run_reid_benchmark

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "eval"
    / "external_quasi_id_table.json"
)


def _external_qi_table() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_linkage_attack_reports_zero_when_no_record_matches_unique_row() -> None:
    result = linkage_attack(
        [{"record_id": "no-hit", "age": 88, "zip": "99999"}],
        _external_qi_table(),
        quasi_identifiers=["age", "zip"],
    )

    assert result.unique_match_rate == pytest.approx(0.0)
    assert result.no_match_rate == pytest.approx(1.0)
    assert result.unique_matches == 0
    assert result.details[0]["outcome"] == "no_match"


def test_linkage_attack_counts_unique_ambiguous_and_no_match_records() -> None:
    result = linkage_attack(
        [
            {"record_id": "unique-hit", "age": 42, "zip": "02139"},
            {"record_id": "ambiguous-hit", "age": "71 years old", "zip": "60612"},
            {"record_id": "no-hit", "age": 55, "zip": "94110"},
        ],
        _external_qi_table(),
        quasi_identifiers=["age", "zip"],
    )

    assert isinstance(result, LinkageAttackResult)
    assert result.unique_match_rate == pytest.approx(1 / 3)
    assert result.ambiguous_match_rate == pytest.approx(1 / 3)
    assert result.no_match_rate == pytest.approx(1 / 3)
    assert result.unique_matches == 1
    assert result.ambiguous_matches == 1
    assert result.no_matches == 1

    unique_detail = result.details[0]
    assert unique_detail["record_id"] == "unique-hit"
    assert unique_detail["outcome"] == "unique"
    assert unique_detail["external_record_id"] == "external-unique"

    ambiguous_detail = result.details[1]
    assert ambiguous_detail["outcome"] == "ambiguous"
    assert ambiguous_detail["match_count"] == 2


def test_run_reid_benchmark_exposes_linkage_metric_for_linkage_mode() -> None:
    report = run_reid_benchmark(
        suite="golden",
        model_name="unit-model",
        attack_mode="linkage",
        deidentified_records=[
            {"record_id": "unique-hit", "age": 42, "zip": "02139"},
            {"record_id": "ambiguous-hit", "age": 71, "zip": "60612"},
            {"record_id": "no-hit", "age": 55, "zip": "94110"},
        ],
        quasi_id_table=_external_qi_table(),
        quasi_identifiers=["age", "zip"],
        generated_at="2026-06-27T00:00:00+00:00",
    )

    assert report.metadata["attack"] == "linkage"
    assert report.metrics["linkage_unique_match_rate"] == pytest.approx(1 / 3)
    assert report.metrics["linkage_attack"]["unique_matches"] == 1
    assert report.metrics["linkage_attack"]["ambiguous_matches"] == 1
    assert report.metrics["linkage_attack"]["no_matches"] == 1


def test_longitudinal_linkage_attack_stays_below_reported_bound() -> None:
    records = [
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
        },
    ]

    result = longitudinal_linkage_attack(
        records,
        hmac_key="unit-longitudinal-key",
    )

    assert isinstance(result, LongitudinalLinkageAttackResult)
    assert result.patient_count == 2
    assert result.successful_links == 1
    assert result.realized_success_rate <= result.reported_upper_bound
    assert result.bound_violated is False

    payload = json.dumps(result.to_metric(), sort_keys=True)
    assert "patient-alpha" not in payload
    assert "Jordan Vale" not in payload
    assert "rare alpha syndrome" not in payload
