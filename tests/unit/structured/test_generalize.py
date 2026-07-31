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

import pytest

from openmed.risk.kanon import kanon_report
from openmed.structured import (
    MANIFEST_SCHEMA_VERSION,
    MODEL_K_ANON,
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
        }
        assert isinstance(column["level"], int)


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
    assert json.dumps(first.manifest, sort_keys=True) == json.dumps(
        second.manifest, sort_keys=True
    )


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
        anonymize_table(_mixed_table(), {"age": "age"}, model="l-diversity")


def test_unknown_column_type_is_rejected():
    with pytest.raises(AnonymizationError):
        anonymize_table(_mixed_table(), {"age": "salary"})


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


def test_supported_models_contains_only_k_anon():
    assert SUPPORTED_MODELS == frozenset({MODEL_K_ANON})
