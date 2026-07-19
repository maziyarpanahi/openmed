"""Tests for distribution-preserving synthetic tabular generation (issue #945)."""

from __future__ import annotations

import json
import math

import pytest

from openmed.risk import (
    DEFAULT_MARGINAL_TOLERANCE,
    TabularProfile,
    fit_tabular_profile,
    sample_synthetic_table,
    tabular_fidelity_report,
)


def _source_rows(count: int = 240) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        age = 20 + index % 61
        rows.append(
            {
                "age": age,
                "cost": 3.5 * age + (index % 7) * 0.25,
                "cohort": "high" if index % 5 < 2 else "standard",
                "flag": index % 4 == 0,
            }
        )
    return rows


def _row_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[name] for name in ("age", "cost", "cohort", "flag"))


def test_fit_profiles_marginals_and_pairwise_numeric_correlations() -> None:
    profile = fit_tabular_profile(_source_rows())

    assert isinstance(profile, TabularProfile)
    assert profile.column_names == ("age", "cost", "cohort", "flag")
    assert [column.kind for column in profile.columns] == [
        "numeric",
        "numeric",
        "categorical",
        "categorical",
    ]
    assert profile.correlation_columns == ("age", "cost")
    assert profile.correlation_matrix[0][1] > 0.95
    assert profile.source_row_count == 240
    assert len(profile.source_row_hashes) == 240


def test_generation_is_reproducible_and_seed_sensitive() -> None:
    profile = fit_tabular_profile(_source_rows())

    first = sample_synthetic_table(profile, rows=300, seed=587)
    second = sample_synthetic_table(profile, rows=300, seed=587)
    different = sample_synthetic_table(profile, rows=300, seed=588)

    assert first == second
    assert first != different


def test_generated_table_meets_documented_fidelity_tolerances() -> None:
    source = _source_rows()
    synthetic = sample_synthetic_table(fit_tabular_profile(source), rows=2_000, seed=42)

    report = tabular_fidelity_report(source, synthetic)

    assert report["passed"] is True
    assert report["marginal_tolerance"] == DEFAULT_MARGINAL_TOLERANCE
    assert all(
        column["distance"] <= DEFAULT_MARGINAL_TOLERANCE
        for column in report["columns"].values()
    )
    assert report["correlations"][0]["difference"] <= 0.15
    assert report["score"] >= 0.95
    assert json.loads(json.dumps(report)) == report


def test_output_contains_no_rows_copied_from_source() -> None:
    source = _source_rows()
    synthetic = sample_synthetic_table(fit_tabular_profile(source), rows=1_000, seed=19)
    source_keys = {_row_key(row) for row in source}

    assert all(_row_key(row) not in source_keys for row in synthetic)
    assert tabular_fidelity_report(source, synthetic)["copied_row_count"] == 0


def test_marginals_cover_missing_constant_and_categorical_values() -> None:
    source = [
        {"constant": 7, "score": 10.0, "status": "a", "optional": None},
        {"constant": 7, "score": 20.0, "status": "a", "optional": 1.0},
        {"constant": 7, "score": 30.0, "status": "b", "optional": 2.0},
        {"constant": 7, "score": 40.0, "status": "b", "optional": None},
    ]
    profile = fit_tabular_profile(source)
    synthetic = sample_synthetic_table(profile, rows=100, seed=4)

    assert {row["constant"] for row in synthetic} == {7}
    assert {row["status"] for row in synthetic} == {"a", "b"}
    assert sum(row["optional"] is None for row in synthetic) == 50
    assert all(
        value is None or math.isfinite(value)
        for value in (row["optional"] for row in synthetic)
    )


def test_fidelity_report_detects_distribution_shift_and_source_copies() -> None:
    source = _source_rows(60)
    shifted = [dict(row, cohort="shifted") for row in source]

    report = tabular_fidelity_report(source, shifted)

    assert report["passed"] is False
    assert report["columns"]["cohort"]["distance"] == pytest.approx(1.0)
    assert report["copied_row_count"] == 0

    copied = tabular_fidelity_report(source, source)
    assert copied["passed"] is False
    assert copied["copied_row_count"] == len(source)


def test_categorical_support_exhaustion_fails_instead_of_copying() -> None:
    profile = fit_tabular_profile([{"only": "value"}])

    with pytest.raises(RuntimeError, match="without copying"):
        sample_synthetic_table(profile, rows=1, seed=1)


def test_categorical_profile_keeps_boolean_and_integer_values_distinct() -> None:
    profile = fit_tabular_profile(
        [{"mixed": True}, {"mixed": 1}, {"mixed": False}, {"mixed": 0}]
    )
    column = profile.columns[0]

    assert column.values == (True, 1, False, 0)
    assert column.probabilities == (0.25, 0.25, 0.25, 0.25)


@pytest.mark.parametrize("records", [[], [{"value": float("inf")}], [1, 2]])
def test_invalid_source_tables_are_rejected(records) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_tabular_profile(records)


def test_invalid_row_count_and_schema_are_rejected() -> None:
    source = _source_rows(10)
    profile = fit_tabular_profile(source)

    with pytest.raises(ValueError, match="non-negative"):
        sample_synthetic_table(profile, rows=-1)
    with pytest.raises(TypeError, match="integer"):
        sample_synthetic_table(profile, rows=1.5)
    with pytest.raises(ValueError, match="exactly the source columns"):
        tabular_fidelity_report(source, [{"age": 1}])


def test_public_api_is_exported() -> None:
    import openmed.risk as risk

    for name in (
        "fit_tabular_profile",
        "sample_synthetic_table",
        "tabular_fidelity_report",
    ):
        assert name in risk.__all__
        assert hasattr(risk, name)
