"""Tests for corpus-level k-anonymity enforcement."""

from __future__ import annotations

from itertools import product

import pytest

import openmed.risk.kanon as kanon_module
from openmed.risk import enforce_kanon, risk_report

QIS = ["age", "zip", "visit_date"]


def _balanced_records() -> list[dict[str, object]]:
    return [
        {
            "patient_name": "Alice Jones",
            "age": 31,
            "zip": "10001",
            "visit_date": "2024-01-01",
            "disease": "flu",
        },
        {
            "patient_name": "Bob Smith",
            "age": 32,
            "zip": "10002",
            "visit_date": "2024-01-02",
            "disease": "cold",
        },
        {
            "patient_name": "Carol Lee",
            "age": 41,
            "zip": "20001",
            "visit_date": "2024-01-03",
            "disease": "flu",
        },
        {
            "patient_name": "Dan Patel",
            "age": 42,
            "zip": "20002",
            "visit_date": "2024-01-04",
            "disease": "cold",
        },
    ]


def test_enforce_kanon_meets_k_l_t_and_reports_provable_bounds() -> None:
    records = _balanced_records()

    enforced = enforce_kanon(
        records,
        quasi_identifiers=QIS,
        sensitive_attributes=["disease"],
        target_k=2,
        target_l=2,
        target_t=0.0,
    )

    assert enforced["kanon"]["k"] >= 2
    assert enforced["released_count"] == len(records)
    assert enforced["bounds"]["max_reidentification_upper_bound"] <= 0.5
    assert enforced["bounds"]["numeric_self_check"]["passed"] is True
    for item in enforced["bounds"]["per_record"]:
        assert item["reidentification_upper_bound"] <= 0.5


def test_enforcement_removes_direct_identifier_fields_from_released_records() -> None:
    records = _balanced_records()

    enforced = enforce_kanon(
        records,
        quasi_identifiers=QIS,
        sensitive_attributes=["disease"],
        target_k=2,
        target_l=2,
        target_t=0.0,
    )

    assert all("patient_name" not in record for record in enforced["records"])
    leakage = risk_report(enforced["records"], original=records)
    assert leakage["leakage_rate"] == 0.0


def test_suppression_cap_reports_records_by_offset_and_hash_only() -> None:
    records = [
        {"age": 30, "zip": "10001", "visit_date": "2024-01-01", "disease": "flu"},
        {"age": 30, "zip": "10001", "visit_date": "2024-01-01", "disease": "cold"},
        {
            "age": 40,
            "zip": "20001",
            "visit_date": "2024-01-01",
            "disease": "asthma",
        },
        {"age": 40, "zip": "20001", "visit_date": "2024-01-01", "disease": "flu"},
        {"age": 99, "zip": "99999", "visit_date": "1901-01-01", "disease": "rare"},
    ]

    enforced = enforce_kanon(
        records,
        quasi_identifiers=QIS,
        sensitive_attributes=["disease"],
        target_k=2,
        suppression_limit=1,
    )

    assert enforced["suppressed_count"] == 1
    suppressed = enforced["suppressed_records"][0]
    assert suppressed["offset"] == 4
    assert suppressed["record_hash"].startswith("sha256:")
    assert "rare" not in str(suppressed)
    assert enforced["kanon"]["k"] >= 2


def _brute_force_best_loss(records: list[dict[str, object]]) -> float:
    coerced = kanon_module._coerce_records(records, source="deidentified")
    levels = kanon_module._build_hierarchy_levels(coerced, QIS, None)
    candidates = []
    ranges = [range(len(levels[field])) for field in QIS]
    for node in product(*ranges):
        candidate = kanon_module._evaluate_lattice_node(
            coerced,
            QIS,
            ["disease"],
            levels,
            node,
            target_k=2,
            target_l=1,
            target_t=1.0,
            suppression_budget=0,
            remove_direct_identifiers=True,
        )
        if candidate is not None:
            candidates.append(candidate.information_loss)
    assert candidates
    return min(candidates)


@pytest.mark.parametrize(
    "records",
    [
        _balanced_records(),
        [
            {"age": 31, "zip": "10001", "visit_date": "2024-01-01", "disease": "a"},
            {"age": 33, "zip": "10002", "visit_date": "2024-01-02", "disease": "b"},
            {"age": 35, "zip": "10003", "visit_date": "2024-01-03", "disease": "a"},
            {"age": 37, "zip": "10004", "visit_date": "2024-01-04", "disease": "b"},
        ],
        [
            {"age": 51, "zip": "60601", "visit_date": "2024-02-01", "disease": "x"},
            {"age": 52, "zip": "60602", "visit_date": "2024-02-11", "disease": "y"},
            {"age": 61, "zip": "60603", "visit_date": "2025-02-01", "disease": "x"},
            {"age": 62, "zip": "60604", "visit_date": "2025-02-11", "disease": "y"},
        ],
    ],
)
def test_lattice_search_matches_exhaustive_optimum_on_synthetic_corpora(
    records: list[dict[str, object]],
) -> None:
    enforced = enforce_kanon(
        records,
        quasi_identifiers=QIS,
        sensitive_attributes=["disease"],
        target_k=2,
    )

    assert enforced["generalization"]["optimality_tolerance"] == 0.0
    assert enforced["generalization"]["information_loss"] == pytest.approx(
        _brute_force_best_loss(records)
    )


def test_k_anonymity_monotonicity_over_coarser_lattice_nodes() -> None:
    records = _balanced_records()
    coerced = kanon_module._coerce_records(records, source="deidentified")
    levels = kanon_module._build_hierarchy_levels(coerced, QIS, None)
    ranges = [range(len(levels[field])) for field in QIS]
    nodes = list(product(*ranges))
    satisfying = {
        node
        for node in nodes
        if kanon_module.kanon_report(
            [
                kanon_module._transform_record(
                    record,
                    QIS,
                    levels,
                    node,
                    remove_direct_identifiers=True,
                )
                for record in coerced
            ],
            quasi_identifiers=QIS,
            sensitive_attributes=["disease"],
        )["k"]
        >= 2
    }

    assert satisfying
    for node in satisfying:
        for coarser in nodes:
            if all(coarser[index] >= node[index] for index in range(len(QIS))):
                report = kanon_module.kanon_report(
                    [
                        kanon_module._transform_record(
                            record,
                            QIS,
                            levels,
                            coarser,
                            remove_direct_identifiers=True,
                        )
                        for record in coerced
                    ],
                    quasi_identifiers=QIS,
                    sensitive_attributes=["disease"],
                )
                assert report["k"] >= 2


def test_enforcement_is_exported_from_risk_package() -> None:
    import openmed.risk as risk

    assert hasattr(risk, "enforce_kanon")
    assert "enforce_kanon" in risk.__all__
