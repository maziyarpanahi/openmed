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


def test_enforcement_reports_and_honors_lattice_search_budget() -> None:
    records = _balanced_records()

    with pytest.raises(ValueError, match="search budget"):
        enforce_kanon(
            records,
            quasi_identifiers=QIS,
            target_k=2,
            max_lattice_nodes=10,
        )

    enforced = enforce_kanon(
        records,
        quasi_identifiers=QIS,
        target_k=2,
        max_lattice_nodes=1_000,
    )

    search = enforced["generalization"]
    assert search["search_space_size"] == search["nodes_evaluated"]
    assert search["nodes_evaluated"] <= search["max_lattice_nodes"]
    assert search["max_lattice_nodes"] == 1_000


def test_enforcement_supports_entropy_l_diversity_as_an_explicit_variant() -> None:
    records = [
        {"age": 30, "zip": "10001", "disease": "flu"},
        {"age": 30, "zip": "10001", "disease": "cold"},
        {"age": 40, "zip": "20001", "disease": "flu"},
        {"age": 40, "zip": "20001", "disease": "cold"},
    ]

    enforced = enforce_kanon(
        records,
        quasi_identifiers=["age", "zip"],
        sensitive_attributes=["disease"],
        target_k=2,
        target_l=2,
        l_metric="entropy",
        target_t=1.0,
    )

    assert enforced["l_metric"] == "entropy"
    assert enforced["kanon"]["l_metric"] == "entropy"
    assert enforced["bounds"]["l_metric"] == "entropy"
    assert enforced["bounds"]["numeric_self_check"]["l_diversity_satisfied"] is True


def test_enforcement_rejects_unknown_l_diversity_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported l_metric"):
        enforce_kanon(
            _balanced_records(),
            quasi_identifiers=QIS,
            target_k=2,
            l_metric="recursive",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_k": True},
        {"target_l": 1.5},
        {"target_t": float("nan")},
        {"suppression_rate": float("inf")},
    ],
)
def test_enforcement_rejects_ambiguous_or_nonfinite_policy_values(kwargs) -> None:
    with pytest.raises(ValueError):
        enforce_kanon(
            _balanced_records(),
            quasi_identifiers=QIS,
            **kwargs,
        )


@pytest.mark.parametrize(
    "levels",
    [
        [
            {"name": "exact", "loss": 0.0},
            {"name": "bad", "loss": float("nan")},
        ],
        [
            {"name": "exact", "loss": 0.5},
            {"name": "less-general", "loss": 0.25},
        ],
        [
            {"name": "exact", "loss": 0.0},
            {"name": "bad", "loss": 1.5},
        ],
    ],
)
def test_user_hierarchy_losses_must_be_finite_bounded_and_monotonic(
    levels,
) -> None:
    with pytest.raises(ValueError, match="loss"):
        enforce_kanon(
            _balanced_records(),
            quasi_identifiers=["age"],
            hierarchies={"age": levels},
        )


@pytest.mark.parametrize(
    "levels",
    [
        [{"name": "collapsed", "loss": 0.0, "default": "*"}],
        [
            {"name": "exact", "loss": 0.0},
            {"name": "collapsed", "loss": 0.0, "default": "*"},
        ],
    ],
)
def test_user_hierarchy_requires_zero_loss_identity_then_positive_coarsening(
    levels,
) -> None:
    with pytest.raises(ValueError, match="identity|greater than 0"):
        enforce_kanon(
            _balanced_records(),
            quasi_identifiers=["age"],
            hierarchies={"age": levels},
        )


@pytest.mark.parametrize(
    "coarsening",
    [
        {
            "name": "mapped",
            "loss": 0.5,
            "values": {"30": "__OPENMED_INTERNAL_QI__:state:null"},
        },
        {
            "name": "defaulted",
            "loss": 0.5,
            "default": "__OPENMED_INTERNAL_QI__:state:missing",
        },
    ],
)
def test_user_hierarchy_cannot_emit_reserved_internal_values(coarsening) -> None:
    with pytest.raises(ValueError, match="reserved internal namespace"):
        enforce_kanon(
            _balanced_records(),
            quasi_identifiers=["age"],
            hierarchies={
                "age": [
                    {"name": "exact", "loss": 0.0},
                    coarsening,
                ]
            },
        )


def test_user_hierarchy_rejects_keys_that_collide_after_string_coercion() -> None:
    with pytest.raises(ValueError, match="collide after string coercion"):
        enforce_kanon(
            [{"age": 1}],
            quasi_identifiers=["age"],
            target_k=1,
            hierarchies={
                "age": [
                    {"name": "exact", "loss": 0.0},
                    {
                        "name": "mapped",
                        "loss": 0.5,
                        "values": {1: "one", "1": "string-one"},
                    },
                ]
            },
        )


def test_user_hierarchy_rejects_split_after_merge() -> None:
    with pytest.raises(ValueError, match="splits values merged"):
        enforce_kanon(
            [{"facility": "north"}, {"facility": "south"}],
            quasi_identifiers=["facility"],
            target_k=1,
            hierarchies={
                "facility": [
                    {"name": "exact", "loss": 0.0},
                    {
                        "name": "campus",
                        "loss": 0.5,
                        "values": {"north": "all", "south": "all"},
                    },
                    {
                        "name": "invalid-split",
                        "loss": 0.75,
                        "values": {"north": "n", "south": "s"},
                    },
                ]
            },
        )


def test_user_hierarchy_rejects_default_merge_followed_by_exception_split() -> None:
    with pytest.raises(ValueError, match="splits values merged"):
        enforce_kanon(
            [{"facility": "A"}, {"facility": "B"}],
            quasi_identifiers=["facility"],
            target_k=1,
            hierarchies={
                "facility": [
                    {"name": "exact", "loss": 0.0},
                    {"name": "all", "loss": 0.5, "default": "all"},
                    {
                        "name": "invalid-split",
                        "loss": 0.75,
                        "values": {"A": "exception"},
                        "default": "other",
                    },
                ]
            },
        )


def test_hierarchies_for_undeclared_qis_are_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared quasi-identifiers"):
        enforce_kanon(
            _balanced_records(),
            quasi_identifiers=["age"],
            hierarchies={
                "zip": [
                    {"name": "exact", "loss": 0.0},
                    {"name": "suppressed", "loss": 1.0, "default": "*"},
                ]
            },
        )


def test_unknown_categories_require_explicit_semantic_hierarchies() -> None:
    records = [
        {"facility": "North Clinic", "disease": "a"},
        {"facility": "North Campus", "disease": "b"},
    ]

    levels = kanon_module.build_generalization_hierarchies(
        records,
        quasi_identifiers=["facility"],
    )

    assert [level["name"] for level in levels["facility"]] == [
        "exact",
        "suppressed",
    ]


def test_t_closeness_suppression_search_finds_the_global_optimum() -> None:
    records = [
        {"group": "a", "disease": 0},
        {"group": "b", "disease": 1},
        {"group": "c", "disease": 0},
        {"group": "c", "disease": 1},
    ]

    enforced = enforce_kanon(
        records,
        quasi_identifiers=["group"],
        sensitive_attributes=["disease"],
        target_k=1,
        target_l=1,
        target_t=0.35,
        suppression_limit=1,
    )

    assert enforced["generalization"]["node"] == {"group": 0}
    assert enforced["suppressed_count"] == 1
    assert enforced["generalization"]["information_loss"] == pytest.approx(0.25)
    assert [record["group"] for record in enforced["records"]] == ["b", "c", "c"]
    search = enforced["generalization"]
    assert search["suppression_subsets_evaluated"] == 4
    assert search["suppression_subsets_possible"] == 4
    assert search["search_complete"] is True


def test_t_closeness_suppression_search_fails_closed_at_its_bound() -> None:
    records = [
        {"group": "a", "disease": 0},
        {"group": "b", "disease": 1},
        {"group": "c", "disease": 0},
        {"group": "c", "disease": 1},
    ]

    with pytest.raises(ValueError, match="Suppression subset search exceeds"):
        enforce_kanon(
            records,
            quasi_identifiers=["group"],
            sensitive_attributes=["disease"],
            target_k=1,
            target_l=1,
            target_t=0.35,
            suppression_limit=1,
            max_suppression_subsets=3,
        )


def test_zero_loss_exact_release_prunes_optional_suppression_search() -> None:
    records = [{"group": f"group-{index}", "disease": index % 2} for index in range(18)]

    enforced = enforce_kanon(
        records,
        quasi_identifiers=["group"],
        sensitive_attributes=["disease"],
        target_k=1,
        target_l=1,
        target_t=0.5,
        suppression_limit=9,
    )

    generalization = enforced["generalization"]
    assert enforced["released_count"] == len(records)
    assert enforced["suppressed_count"] == 0
    assert generalization["information_loss"] == 0.0
    assert generalization["suppression_subsets_evaluated"] == 1
    assert generalization["suppression_subsets_possible"] is None
    assert generalization["nodes_evaluated"] == 1
    assert generalization["nodes_evaluated"] < generalization["search_space_size"]
    assert generalization["search_complete"] is False
    assert generalization["optimum_proven"] is True
    assert generalization["search"] == "zero-loss lower-bound lattice"


def test_enforcement_preserves_supported_typed_qi_values() -> None:
    from datetime import date
    from decimal import Decimal

    records = [
        {
            "visit_date": date(2026, 7, 26),
            "amount": Decimal("1.20"),
            "payload": b"\x01",
        }
    ]

    enforced = enforce_kanon(
        records,
        quasi_identifiers=["visit_date", "amount", "payload"],
        target_k=1,
        remove_direct_identifiers=False,
    )

    assert enforced["records"] == records
