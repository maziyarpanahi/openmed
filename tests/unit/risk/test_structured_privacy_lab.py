"""Focused regression tests for the structured privacy risk lab."""

from __future__ import annotations

import json

import pytest

from openmed.cli import main_module
from openmed.risk import (
    AggregateDPBudgetLedger,
    DPAggregateBudgetExceeded,
    membership_inference_self_test,
    release_aggregate,
)
from openmed.structured import (
    StructuredPrivacyPolicy,
    make_synthetic_privacy_fixture,
    profile_structured_table,
    run_structured_privacy_lab,
)


def _fixture_policy(
    *,
    target_k: int = 3,
    suppression_limit: int | None = 0,
    membership_max_inference_rate: float | None = None,
) -> StructuredPrivacyPolicy:
    return StructuredPrivacyPolicy(
        quasi_identifiers=("age", "postal_prefix"),
        sensitive_attributes=("sensitive_outcome",),
        direct_identifiers=("synthetic_record_id",),
        target_k=target_k,
        target_l=2,
        target_t=1.0,
        suppression_limit=suppression_limit,
        membership_max_inference_rate=membership_max_inference_rate,
    )


def test_profile_reports_roles_missingness_uniqueness_and_population_assumptions():
    rows = list(make_synthetic_privacy_fixture(group_count=2, rows_per_group=2))
    rows[0]["sensitive_outcome"] = None

    profile = profile_structured_table(
        rows,
        quasi_identifiers=("age", "postal_prefix"),
        sensitive_attributes=("sensitive_outcome",),
        direct_identifiers=("synthetic_record_id",),
        population_assumptions={
            "scope": "synthetic-cohort",
            "population_kind": "fixture",
            "notes": "not emitted",
        },
    )

    payload = profile.to_dict()
    assert payload["row_count"] == 4
    assert payload["missingness"]["missing_cell_count"] == 1
    assert payload["population_assumptions"]["scope"] == "synthetic-cohort"
    assert payload["population_assumptions"]["details_digest"].startswith("sha256:")
    assert {item["role"] for item in payload["columns"]} == {
        "direct_identifier",
        "quasi_identifier",
        "sensitive",
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "synthetic-0-0" not in serialized
    assert "not emitted" not in serialized


def test_stricter_generalization_reduces_identity_risk_and_reports_utility_loss():
    rows = list(make_synthetic_privacy_fixture(group_count=4, rows_per_group=3))
    loose = run_structured_privacy_lab(
        rows,
        _fixture_policy(target_k=1),
        population_assumptions={"scope": "synthetic"},
    )
    strict = run_structured_privacy_lab(
        rows,
        _fixture_policy(target_k=4),
        population_assumptions={"scope": "synthetic"},
    )

    assert loose.meets_policy is True
    assert strict.meets_policy is True
    assert strict.after is not None
    assert strict.after.max_sample_identity_risk <= loose.after.max_sample_identity_risk
    assert strict.anonymization is not None
    assert (
        strict.anonymization.utility.to_dict()["quasi_identifier_cell_change_rate"]
        >= (loose.anonymization.utility.to_dict()["quasi_identifier_cell_change_rate"])
    )
    assert strict.to_dict()["utility"]


def test_suppression_reduces_singletons_and_reports_row_utility_loss():
    rows = [
        {"synthetic_id": "alpha", "qi": "A", "outcome": "x"},
        {"synthetic_id": "beta", "qi": "A", "outcome": "y"},
        {"synthetic_id": "gamma", "qi": "B", "outcome": "x"},
    ]
    result = run_structured_privacy_lab(
        rows,
        StructuredPrivacyPolicy(
            quasi_identifiers=("qi",),
            sensitive_attributes=("outcome",),
            direct_identifiers=("synthetic_id",),
            target_k=2,
            suppression_limit=1,
        ),
        population_assumptions={"scope": "synthetic"},
    )

    assert result.meets_policy is True
    assert result.anonymization is not None
    summary = result.anonymization.generalization.to_dict()
    assert summary["suppressed_privacy_units"] == 1
    assert result.anonymization.utility.row_suppression_rate > 0.0
    assert result.after is not None
    assert result.after.singleton_class_count == 0


def test_unique_membership_fixture_fails_configured_policy_without_raw_values():
    rows = [
        {
            "synthetic_record_id": "unique-source-alpha",
            "age": 47,
            "postal_prefix": "SYN-99",
            "sensitive_outcome": "synthetic-rare",
        },
    ]
    result = run_structured_privacy_lab(
        rows,
        StructuredPrivacyPolicy(
            quasi_identifiers=("age", "postal_prefix"),
            sensitive_attributes=("sensitive_outcome",),
            direct_identifiers=("synthetic_record_id",),
            target_k=1,
            target_l=1,
            target_t=1.0,
            membership_max_inference_rate=0.0,
        ),
        population_assumptions={"scope": "synthetic"},
        membership_candidates=rows,
    )

    assert result.meets_policy is False
    assert result.membership_after is not None
    assert result.membership_after.membership_inference_rate == 1.0
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "unique-source-alpha" not in serialized
    assert "synthetic-rare" not in serialized
    assert "SYN-99" not in serialized


def test_dp_composition_is_deterministic_and_exhaustion_is_transactional():
    first = AggregateDPBudgetLedger(max_epsilon=1.0, max_delta=0.01)
    second = AggregateDPBudgetLedger(max_epsilon=1.0, max_delta=0.01)
    first.spend(0.25, 0.001, label="count")
    first.spend(0.5, 0.002, label="sum")
    second.spend(0.25, 0.001, label="count")
    second.spend(0.5, 0.002, label="sum")
    assert first.to_dict() == second.to_dict()

    with pytest.raises(DPAggregateBudgetExceeded):
        first.spend(0.3, 0.0, label="exhausted")
    assert len(first.spends) == 2
    assert first.spent_epsilon == pytest.approx(0.75)


def test_dp_aggregate_is_reproducible_and_rejects_row_level_input():
    first_ledger = AggregateDPBudgetLedger(max_epsilon=1.0, max_delta=0.0)
    second_ledger = AggregateDPBudgetLedger(max_epsilon=1.0, max_delta=0.0)
    first = release_aggregate(
        {"synthetic_count": 12},
        ledger=first_ledger,
        epsilon=0.25,
        seed="offline-test",
    )
    second = release_aggregate(
        {"synthetic_count": 12},
        ledger=second_ledger,
        epsilon=0.25,
        seed="offline-test",
    )
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["row_level_anonymization"] is False
    with pytest.raises(TypeError, match="row-level"):
        release_aggregate(
            [{"synthetic_count": 12}],
            ledger=AggregateDPBudgetLedger(max_epsilon=1.0, max_delta=0.0),
            epsilon=0.25,
        )


def test_membership_self_test_is_bounded_and_aggregate_only():
    candidates = [
        {"age": 30, "postal_prefix": "SYN-01"},
        {"age": 30, "postal_prefix": "SYN-01"},
        {"age": 40, "postal_prefix": "SYN-02"},
    ]
    result = membership_inference_self_test(
        candidates,
        candidates,
        quasi_identifiers=("age", "postal_prefix"),
        max_candidates=2,
        max_inference_rate=0.5,
    )
    payload = result.to_dict()
    assert payload["candidate_truncated"] is True
    assert payload["candidate_count"] == 2
    assert "SYN-01" not in json.dumps(payload)
    assert '"age"' not in json.dumps(payload)


def test_cli_lab_writes_safe_evidence_and_keeps_failed_release_separate(
    tmp_path,
):
    input_path = tmp_path / "synthetic.jsonl"
    evidence_path = tmp_path / "evidence.json"
    rows = list(make_synthetic_privacy_fixture(group_count=2, rows_per_group=2))
    input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    code = main_module.main(
        [
            "risk",
            "lab",
            str(input_path),
            "--evidence",
            str(evidence_path),
            "--qi",
            "age,postal_prefix",
            "--sensitive",
            "sensitive_outcome",
            "--direct-id",
            "synthetic_record_id",
            "--k",
            "2",
            "--l",
            "2",
            "--population-scope",
            "synthetic",
        ]
    )

    assert code == 0
    evidence = evidence_path.read_text(encoding="utf-8")
    assert "structured_privacy_method_evidence" in evidence
    assert "synthetic-0-0" not in evidence
    assert "synthetic-a" not in evidence
