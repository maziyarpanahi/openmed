"""Tests for structured attacker-model re-identification reports."""

from __future__ import annotations

import json
import re

import pytest

from openmed.eval import release_gates
from openmed.structured import reid_report


@pytest.fixture
def analytic_sample() -> list[dict[str, object]]:
    """Return synthetic equivalence classes with sizes one, two, and three."""

    return [
        {"region": "sample-canary-a", "age_band": "20-29", "row_id": "row-a"},
        {"region": "sample-canary-b", "age_band": "30-39", "row_id": "row-b1"},
        {"region": "sample-canary-b", "age_band": "30-39", "row_id": "row-b2"},
        {"region": "sample-canary-c", "age_band": "40-49", "row_id": "row-c1"},
        {"region": "sample-canary-c", "age_band": "40-49", "row_id": "row-c2"},
        {"region": "sample-canary-c", "age_band": "40-49", "row_id": "row-c3"},
    ]


@pytest.fixture
def analytic_population() -> list[dict[str, str]]:
    """Return matching population classes with sizes two, four, and six."""

    rows: list[dict[str, str]] = []
    for region, age_band, count in (
        ("sample-canary-a", "20-29", 2),
        ("sample-canary-b", "30-39", 4),
        ("sample-canary-c", "40-49", 6),
    ):
        rows.extend(
            {"region": region, "age_band": age_band, "population_id": f"unit-{i}"}
            for i in range(count)
        )
    return rows


def test_sample_scenario_risks_match_analytic_distribution_and_hide_values(
    analytic_sample: list[dict[str, object]],
) -> None:
    report = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
    )

    assert report["k_min"] == 1
    assert report["singleton_records"] == 1
    assert report["equivalence_class_count"] == 3
    assert report["prosecutor"]["risk"] == pytest.approx(1.0)
    assert report["prosecutor"]["expected_probability"] == pytest.approx(3 / 6)
    assert report["prosecutor"]["maximum_probability"] == pytest.approx(1.0)
    assert report["journalist"]["risk"] == pytest.approx(1.0)
    assert report["journalist"]["expected_probability"] == pytest.approx(3 / 6)
    assert report["marketer"]["risk"] == pytest.approx(3 / 6)
    assert report["marketer"]["maximum_probability"] == pytest.approx(1.0)

    serialized = json.dumps(report, sort_keys=True)
    for raw_value in (
        "sample-canary-a",
        "sample-canary-b",
        "sample-canary-c",
        "20-29",
        "30-39",
        "40-49",
        "row-a",
    ):
        assert raw_value not in serialized
    assert report["highest_risk_records"]
    for item in report["highest_risk_records"]:
        assert re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            item["equivalence_class_key"],
        )
        assert set(item) == {
            "equivalence_class_key",
            "sample_count",
            "population_count",
            "prosecutor_probability",
            "journalist_probability",
            "marketer_probability",
            "population_model_consistent",
        }


def test_population_changes_journalist_and_marketer_risk_as_expected(
    analytic_sample: list[dict[str, object]],
    analytic_population: list[dict[str, str]],
) -> None:
    sample_only = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
    )
    population_scaled = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
        population=analytic_population,
    )

    assert population_scaled["population_model_consistent"] is True
    assert population_scaled["prosecutor"] == sample_only["prosecutor"]
    assert population_scaled["journalist"]["maximum_probability"] == pytest.approx(
        1 / 2
    )
    assert population_scaled["journalist"]["expected_probability"] == pytest.approx(
        (1 / 2 + 2 / 4 + 3 / 6) / 6
    )
    assert population_scaled["marketer"]["risk"] == pytest.approx(1 / 4)
    assert population_scaled["journalist"]["risk"] < sample_only["journalist"]["risk"]
    assert population_scaled["marketer"]["risk"] < sample_only["marketer"]["risk"]


@pytest.mark.parametrize(
    ("thresholds", "blocked_scenario"),
    [
        ({"journalist": 0.49, "marketer": 1.0}, "journalist"),
        ({"journalist": 1.0, "marketer": 0.24}, "marketer"),
    ],
)
def test_release_gate_blocks_each_population_scenario_above_threshold(
    analytic_sample: list[dict[str, object]],
    analytic_population: list[dict[str, str]],
    thresholds: dict[str, float],
    blocked_scenario: str,
) -> None:
    report = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
        population=analytic_population,
    )

    check = release_gates.evaluate_reidentification_risk_gate(report, thresholds)

    assert check.passed is False
    assert set(check.details["violations"]) == {blocked_scenario}


def test_release_gate_passes_when_population_scenarios_are_below_threshold(
    analytic_sample: list[dict[str, object]],
    analytic_population: list[dict[str, str]],
) -> None:
    report = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
        population=analytic_population,
    )

    check = release_gates.evaluate_reidentification_risk_gate(
        report,
        {"journalist": 0.51, "marketer": 0.26},
    )

    assert check.passed is True
    assert check.details["violations"] == {}


def test_release_gate_integration_uses_report_and_policy_threshold_metrics(
    analytic_sample: list[dict[str, object]],
) -> None:
    report = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
    )

    check = release_gates._reidentification_risk_check(
        {
            "structured_reid_report": report,
            "reid_risk_thresholds": {"marketer": 0.49},
        },
        {},
    )

    assert check.passed is False
    assert check.gate == release_gates.REIDENTIFICATION_RISK_GATE
    assert check.details["violations"]["marketer"] == {
        "observed": pytest.approx(0.5),
        "threshold": pytest.approx(0.49),
    }


def test_release_gate_rejects_tampered_headline_without_echoing_values(
    analytic_sample: list[dict[str, object]],
) -> None:
    report = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
    )
    report["marketer"]["risk"] = 0.0
    report["unexpected_raw_value"] = "gate-canary-value"

    check = release_gates._reidentification_risk_check(
        {
            "structured_reid_report": report,
            "reid_risk_thresholds": {"marketer": 0.1},
        },
        {},
    )

    assert check.passed is False
    assert check.details == {"thresholds_configured": True}
    assert "gate-canary-value" not in json.dumps(check.to_dict(), sort_keys=True)


def test_inconsistent_population_reference_fails_closed(
    analytic_sample: list[dict[str, object]],
) -> None:
    report = reid_report(
        analytic_sample,
        quasi_identifiers=("region", "age_band"),
        population=[
            {"region": "sample-canary-b", "age_band": "30-39"},
            {"region": "sample-canary-c", "age_band": "40-49"},
        ],
    )

    assert report["population_model_consistent"] is False
    assert report["population_inconsistent_class_count"] == 3
    assert report["population_inconsistent_record_count"] == 6
    assert report["journalist"]["risk"] == pytest.approx(1.0)
    assert report["marketer"]["risk"] == pytest.approx(1.0)

    check = release_gates.evaluate_reidentification_risk_gate(
        report,
        {"journalist": 1.0, "marketer": 1.0},
    )
    assert check.passed is False
    assert check.details["violations"]["population_model"] == {"consistent": False}
