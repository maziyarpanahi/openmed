"""Tests for anonymize_table, the declarative-hierarchy wrapper over enforce_kanon.

All fixtures are synthetic and generated algorithmically; no real values are
used. The suite proves the wrapper delegates the lattice search to
``openmed.risk.kanon.enforce_kanon`` (driving it with #1800's declarative
generalization family), reaches the requested k with zero classes below it,
respects the suppression bound while preferring generalization, records only
levels and counts in the manifest (never raw values), and is deterministic.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from openmed.core.date_shift import stable_offset_for
from openmed.risk.kanon import kanon_report
from openmed.structured import (
    MANIFEST_SCHEMA_VERSION,
    MODEL_K_ANON,
    REFERENCE_AVERAGE_GENERALIZATION_HEIGHT_CAP,
    REFERENCE_SUPPRESSION_RATE_CAP,
    SUPPORTED_MODELS,
    AnonymizationError,
    AnonymizationResult,
    anonymize_table,
    build_enforcement_hierarchies,
)
from openmed.structured.table_io import write_table


# --------------------------------------------------------------------------- #
# Synthetic, algorithmically generated fixtures                               #
# --------------------------------------------------------------------------- #
def _mixed_table(row_count: int = 72) -> list[dict[str, object]]:
    """A synthetic age/ZIP/diagnosis table reachable by generalization alone.

    Ages span a contiguous range and ZIPs are drawn from a small pool so the
    generalization family reaches k without exhausting the enforcement engine's
    suppression-subset budget.
    """
    rows: list[dict[str, object]] = []
    for i in range(row_count):
        rows.append(
            {
                "pid": i,
                "age": 20 + (i % 60),
                "zip": f"{90000 + (i % 5) * 100:05d}",
                "diagnosis": ("A", "B", "C", "D")[i % 4],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Reaches target k with zero classes below k                                   #
# --------------------------------------------------------------------------- #
def test_reaches_target_k():
    rows = _mixed_table()
    result = anonymize_table(
        rows, {"age": "age", "zip": "zip"}, target_k=4, suppression_rate=0.1
    )
    assert isinstance(result, AnonymizationResult)
    assert result.manifest["achieved_k"] >= 4

    report = kanon_report(list(result.records), quasi_identifiers=["age", "zip"])
    assert report["k"] >= 4
    assert all(cls["size"] >= 4 for cls in report["equivalence_classes"])


def test_default_target_k_uses_documented_default():
    result = anonymize_table(_mixed_table(), {"age": "age"})
    assert result.manifest["target_k"] == 2
    report = kanon_report(list(result.records), quasi_identifiers=["age"])
    assert report["k"] >= 2


# --------------------------------------------------------------------------- #
# Respects the suppression bound, prefers generalization                       #
# --------------------------------------------------------------------------- #
def test_respects_suppression_bound():
    rows = _mixed_table()
    result = anonymize_table(
        rows, {"age": "age", "zip": "zip"}, target_k=4, suppression_limit=5
    )
    manifest = result.manifest
    # The engine never suppresses more than the declared bound.
    assert manifest["suppressed_count"] <= 5
    # On this fixture generalization alone reaches k, so nothing is suppressed:
    # the engine prefers generalization over spending the suppression budget.
    assert manifest["suppressed_count"] == 0
    assert manifest["released_count"] == len(rows)

    report = kanon_report(list(result.records), quasi_identifiers=["age", "zip"])
    assert report["k"] >= 4


def test_manifest_records_nonzero_row_suppression_per_column():
    rows = [{"age": age} for age in (20, 20, 30, 30, 99)]
    result = anonymize_table(rows, {"age": "age"}, k=2, suppression_limit=1)

    assert result.manifest["suppressed_count"] == 1
    assert result.manifest["columns"][0]["suppression_count"] == 1
    assert result.manifest["utility"]["suppression_rate"] == pytest.approx(0.2)


def test_reference_fixture_stays_below_documented_utility_caps():
    result = anonymize_table(
        _mixed_table(),
        {"age": "age", "zip": "zip"},
        k=4,
        suppression_rate=REFERENCE_SUPPRESSION_RATE_CAP,
    )

    utility = result.manifest["utility"]
    assert (
        utility["average_generalization_height"]
        <= REFERENCE_AVERAGE_GENERALIZATION_HEIGHT_CAP
    )
    assert utility["suppression_rate"] <= REFERENCE_SUPPRESSION_RATE_CAP


def test_infeasible_when_k_exceeds_row_count():
    # Even the fully suppressed ceiling is a single class of size ``n``; a target
    # k above the row count cannot be reached at any suppression bound, and the
    # engine's failure surfaces as AnonymizationError.
    rows = [{"zip": f"{10000 + i:05d}"} for i in range(6)]
    with pytest.raises(AnonymizationError):
        anonymize_table(rows, {"zip": "zip"}, target_k=len(rows) + 1)


# --------------------------------------------------------------------------- #
# Uses #1800's declarative hierarchies to drive the engine                     #
# --------------------------------------------------------------------------- #
def test_uses_declarative_hierarchies():
    rows = _mixed_table()
    quasi_identifiers = {"age": "age", "zip": "zip"}
    result = anonymize_table(rows, quasi_identifiers, target_k=4, suppression_rate=0.1)

    # The rung names the engine chose must come from the declarative family, not
    # from enforce_kanon's built-in default hierarchies.
    hierarchies = build_enforcement_hierarchies(quasi_identifiers, rows)
    for column in result.manifest["columns"]:
        family_keys = {level["name"] for level in hierarchies[column["column"]]}
        assert column["level_name"] in family_keys
        assert column["level_name"].startswith(f"{column['column_type']}:")


def test_declarative_rung_names_are_the_family_keys():
    rows = _mixed_table()
    result = anonymize_table(rows, {"age": "age"}, target_k=4, suppression_rate=0.1)
    (age_column,) = result.manifest["columns"]
    assert age_column["level_name"] in {
        "age:exact",
        "age:5y",
        "age:10y",
        "age:20y",
        "age:suppressed",
    }


def test_clinical_codes_roll_up_only_through_caller_supplied_parent_data():
    rows = [
        {"code": "A1", "outcome": "x"},
        {"code": "A2", "outcome": "y"},
        {"code": "B1", "outcome": "x"},
        {"code": "B2", "outcome": "y"},
    ]
    parent_chains = {
        "A1": ("A", "ROOT"),
        "A2": ("A", "ROOT"),
        "B1": ("B", "ROOT"),
        "B2": ("B", "ROOT"),
    }

    result = anonymize_table(
        rows,
        {"code": "clinical_code"},
        k=2,
        clinical_code_hierarchies={"code": parent_chains},
    )

    assert [record["code"] for record in result.records] == ["A", "A", "B", "B"]
    assert result.manifest["columns"][0]["level_name"] == "clinical_code:parent_1"
    assert result.manifest["suppressed_count"] == 0


def test_optional_l_and_t_targets_are_enforced_with_compact_aliases():
    rows = [
        {"age": 20, "diagnosis": "x"},
        {"age": 21, "diagnosis": "y"},
        {"age": 30, "diagnosis": "x"},
        {"age": 31, "diagnosis": "y"},
    ]
    result = anonymize_table(
        rows,
        {"age": "age"},
        sensitive_attributes=["diagnosis"],
        k=2,
        l=2,
        t=0.0,
    )

    report = kanon_report(
        list(result.records),
        quasi_identifiers=["age"],
        sensitive_attributes=["diagnosis"],
    )
    assert report["k"] >= 2
    for equivalence_class in report["equivalence_classes"]:
        assert equivalence_class["l_diversity"]["diagnosis"]["distinct"] >= 2
        assert equivalence_class["t_closeness"]["diagnosis"] <= 0.0
    assert result.manifest["target_l"] == 2
    assert result.manifest["target_t"] == 0.0


def test_date_qis_use_one_core_offset_per_subject_and_remove_subject_id():
    rows = [
        {
            "patient_id": "patient-a",
            "visit_date": "2025-01-10",
            "discharge_date": "2025-01-12",
        },
        {
            "patient_id": "patient-a",
            "visit_date": "2025-02-20",
            "discharge_date": "2025-02-25",
        },
    ]
    secret = "synthetic-date-shift-secret"
    result = anonymize_table(
        rows,
        {"visit_date": "date", "discharge_date": "date"},
        k=1,
        subject_id_column="patient_id",
        date_shift_secret=secret,
        date_shift_max_days=30,
    )

    offset = stable_offset_for("patient-a", max_days=30, secret=secret)
    for column in ("visit_date", "discharge_date"):
        expected = [
            (date.fromisoformat(row[column]) + timedelta(days=offset)).isoformat()
            for row in rows
        ]
        assert [record[column] for record in result.records] == expected
    assert all("patient_id" not in record for record in result.records)
    assert result.manifest["date_shift"] == {
        "applied": True,
        "columns": ["visit_date", "discharge_date"],
        "max_days": 30,
        "subject_identifier_removed": True,
    }


# --------------------------------------------------------------------------- #
# Manifest records levels and counts, never raw values                         #
# --------------------------------------------------------------------------- #
def test_manifest_records_levels_and_suppression_count():
    rows = _mixed_table()
    result = anonymize_table(
        rows, {"age": "age", "zip": "zip"}, target_k=4, suppression_rate=0.1
    )
    manifest = result.manifest
    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["model"] == MODEL_K_ANON
    assert set(manifest["generalization_levels"]) == {"age", "zip"}
    assert manifest["released_count"] + manifest["suppressed_count"] == len(rows)
    assert manifest["search"]["engine"] == "openmed.risk.kanon.enforce_kanon"
    for column in manifest["columns"]:
        assert set(column) == {
            "column",
            "column_type",
            "level",
            "level_name",
            "loss",
            "suppression_count",
        }
        assert isinstance(column["level"], int)
        assert column["suppression_count"] == manifest["suppressed_count"]


def test_manifest_has_no_raw_values():
    rows = _mixed_table()
    result = anonymize_table(
        rows, {"age": "age", "zip": "zip"}, target_k=4, suppression_rate=0.1
    )
    manifest_text = json.dumps(result.manifest)
    # No raw source ZIP (a full 5-digit value) may appear in the manifest.
    for row in rows:
        assert str(row["zip"]) not in manifest_text


# --------------------------------------------------------------------------- #
# Determinism under identical inputs                                           #
# --------------------------------------------------------------------------- #
def test_identical_inputs_yield_identical_output():
    rows = _mixed_table()
    quasi_identifiers = {"age": "age", "zip": "zip"}

    first = anonymize_table(rows, quasi_identifiers, target_k=4, suppression_rate=0.1)
    second = anonymize_table(rows, quasi_identifiers, target_k=4, suppression_rate=0.1)

    assert first.records == second.records
    assert first.manifest == second.manifest
    assert first.manifest["output_hash"] == second.manifest["output_hash"]
    assert json.dumps(first.manifest, sort_keys=True) == json.dumps(
        second.manifest, sort_keys=True
    )


def test_identical_seed_and_date_inputs_yield_identical_output_hash():
    rows = [
        {"patient_id": 7, "visit_date": "2025-03-01"},
        {"patient_id": 7, "visit_date": "2025-03-11"},
    ]
    kwargs = {
        "k": 1,
        "subject_id_column": "patient_id",
        "seed": 744,
    }
    first = anonymize_table(rows, {"visit_date": "date"}, **kwargs)
    second = anonymize_table(rows, {"visit_date": "date"}, **kwargs)

    assert first.records == second.records
    assert first.manifest["output_hash"] == second.manifest["output_hash"]


# --------------------------------------------------------------------------- #
# Default offline path: works from a file and a DataFrame-like object          #
# --------------------------------------------------------------------------- #
def test_anonymize_table_reads_from_a_local_file(tmp_path):
    rows = _mixed_table()
    path = tmp_path / "table.jsonl"
    write_table(path, rows)

    result = anonymize_table(
        str(path), {"age": "age", "zip": "zip"}, target_k=4, suppression_rate=0.1
    )
    report = kanon_report(list(result.records), quasi_identifiers=["age", "zip"])
    assert report["k"] >= 4


def test_anonymize_table_accepts_dataframe_like_object():
    rows = _mixed_table()

    class _FrameLike:
        def __init__(self, records):
            self._records = records

        def to_dicts(self):
            return [dict(record) for record in self._records]

    result = anonymize_table(
        _FrameLike(rows), {"age": "age"}, target_k=3, suppression_rate=0.1
    )
    assert result.manifest["achieved_k"] >= 3


# --------------------------------------------------------------------------- #
# Validation and error handling                                                #
# --------------------------------------------------------------------------- #
def test_unknown_model_is_rejected():
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {"age": "age"}, model="unsupported")


def test_unknown_column_type_is_rejected():
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {"age": "salary"})


def test_clinical_code_qi_requires_caller_supplied_parent_data():
    with pytest.raises(AnonymizationError, match="parent-chain data"):
        anonymize_table([{"code": "A1"}, {"code": "A2"}], {"code": "clinical_code"})


def test_date_configuration_fails_closed_without_echoing_source_value():
    source_value = "2099-12-31"
    with pytest.raises(AnonymizationError) as raised:
        anonymize_table(
            [{"patient_id": "synthetic-subject", "visit_date": source_value}],
            {"visit_date": "date"},
            subject_id_column="patient_id",
        )
    assert source_value not in str(raised.value)


def test_missing_quasi_identifier_column_is_rejected():
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {"absent": "age"})


def test_empty_quasi_identifiers_is_rejected():
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {})


@pytest.mark.parametrize("bad_k", [0, -1])
def test_invalid_target_k_is_rejected(bad_k):
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {"age": "age"}, target_k=bad_k)


def test_invalid_suppression_rate_is_rejected():
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {"age": "age"}, suppression_rate=1.5)


def test_supported_models_cover_the_public_table_orchestrator():
    assert SUPPORTED_MODELS == frozenset(
        {MODEL_K_ANON, "dp", "l-diversity", "t-closeness"}
    )
