"""Tests for deterministic tabular column-role detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.structured import (
    ColumnClassification,
    ColumnRole,
    RoleOverrideError,
    TableRoleScan,
    scan_column_roles,
)
from openmed.structured.scan import scan_table
from openmed.structured.table_io import write_table

# Expected role for every column in the labeled fixture. ``name`` is a direct
# identifier; ``zip`` and ``dob`` are quasi-identifiers by canonical label;
# ``admission_date`` is promoted to a quasi-identifier by its date-like value
# shape; ``diagnosis`` and ``note`` are sensitive; ``record_id`` is an
# unrecognized column that defaults to safe.
EXPECTED_ROLES = {
    "name": ColumnRole.DIRECT_ID,
    "zip": ColumnRole.QUASI_ID,
    "dob": ColumnRole.QUASI_ID,
    "age": ColumnRole.QUASI_ID,
    "sex": ColumnRole.QUASI_ID,
    "admission_date": ColumnRole.QUASI_ID,
    "diagnosis": ColumnRole.SENSITIVE,
    "note": ColumnRole.SENSITIVE,
    "record_id": ColumnRole.SAFE,
}


def build_rows(count: int = 24) -> list[dict[str, Any]]:
    """Generate a synthetic labeled schema algorithmically (no real PHI)."""

    zips = ["02139", "30301", "94105", "10001"]
    sexes = ["F", "M"]
    diagnoses = ["E11.9", "I10", "J45.909", "N18.3"]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "name": f"Synthetic Patient {index:03d}",
                "zip": zips[index % len(zips)],
                "dob": f"19{50 + index % 40:02d}-{1 + index % 12:02d}-"
                f"{1 + index % 28:02d}",
                "age": str(30 + index % 50),
                "sex": sexes[index % len(sexes)],
                "admission_date": f"2025-{1 + index % 12:02d}-{1 + index % 28:02d}",
                "diagnosis": diagnoses[index % len(diagnoses)],
                "note": (
                    f"Encounter {index}: patient reports symptoms and a "
                    "follow-up plan documented in full during the visit."
                ),
                "record_id": f"rec-{index:04d}",
            }
        )
    return rows


def test_labeled_fixture_roles_match_exactly() -> None:
    scan = scan_table(build_rows())

    assert scan.to_dict() == {
        column: role.value for column, role in EXPECTED_ROLES.items()
    }


def test_direct_and_quasi_columns_reach_the_accuracy_floor() -> None:
    scan = scan_table(build_rows())

    gated = {
        column: role
        for column, role in EXPECTED_ROLES.items()
        if role in (ColumnRole.DIRECT_ID, ColumnRole.QUASI_ID)
    }
    for column, role in gated.items():
        assert scan[column] is role, column


def test_result_is_a_column_to_role_mapping() -> None:
    scan = scan_table(build_rows())

    assert isinstance(scan, TableRoleScan)
    assert dict(scan) == EXPECTED_ROLES
    assert scan["zip"] == "quasi_id"
    assert scan.quasi_identifiers == (
        "zip",
        "dob",
        "age",
        "sex",
        "admission_date",
    )
    assert scan.direct_identifiers == ("name",)
    assert scan.sensitive == ("diagnosis", "note")
    assert scan.safe == ("record_id",)
    assert all(isinstance(item, ColumnClassification) for item in scan.classifications)


def test_package_alias_matches_module_function() -> None:
    assert scan_column_roles is scan_table


def test_unknown_column_defaults_to_safe_with_low_confidence() -> None:
    scan = scan_table(
        [{"widget_serial_token": "abcd"}, {"widget_serial_token": "efgh"}]
    )

    assert scan["widget_serial_token"] is ColumnRole.SAFE
    assert scan.confidence["widget_serial_token"] < 0.5


def test_date_likeness_promotes_unlabeled_column() -> None:
    rows = [{"seen_on": f"2024-0{month}-15"} for month in range(1, 8)]

    scan = scan_table(rows)

    assert scan["seen_on"] is ColumnRole.QUASI_ID


def test_numeric_range_promotes_unlabeled_column() -> None:
    rows = [{"reading_index": str(value)} for value in range(20, 60)]

    scan = scan_table(rows)

    assert scan["reading_index"] is ColumnRole.QUASI_ID


def test_out_of_range_numeric_column_stays_safe() -> None:
    rows = [{"account_balance": str(value)} for value in range(5000, 5040)]

    scan = scan_table(rows)

    assert scan["account_balance"] is ColumnRole.SAFE


def test_determinism_same_input_same_role_map() -> None:
    rows = build_rows()

    first = scan_table(rows).as_dict()
    second = scan_table(rows).as_dict()

    assert first == second


def test_no_raw_cell_value_appears_in_output() -> None:
    # Distinctive canary values that cannot collide with label names or signal
    # tokens, so any leak of a raw cell into the payload is caught. Roles derive
    # from column names, so opaque values do not change the classification; the
    # date-shaped column keeps date-like values so its promotion still fires.
    rows = [
        {
            "name": f"CANARY-NAME-{index}",
            "zip": f"CANARY-ZIP-{index}",
            "dob": f"CANARY-DOB-{index}",
            "age": "CANARY-AGE",
            "sex": "CANARY-SEX",
            "admission_date": f"2025-0{1 + index}-14",
            "diagnosis": f"CANARY-DX-{index}",
            "note": (
                f"CANARYNOTE-{index} a sufficiently long free-text clinical "
                "narrative to trigger the sensitive shape signal."
            ),
            "record_id": f"CANARY-REC-{index}",
        }
        for index in range(6)
    ]
    scan = scan_table(rows)

    serialized = json.dumps(scan.as_dict(), sort_keys=True)
    raw_values = {str(value) for row in rows for value in row.values()}
    for raw_value in raw_values:
        assert raw_value not in serialized


def test_csv_jsonl_parquet_agree(tmp_path: Path) -> None:
    rows = build_rows()
    csv_path = write_table(tmp_path / "table.csv", rows)
    jsonl_path = write_table(tmp_path / "table.jsonl", rows)

    csv_roles = scan_table(csv_path).to_dict()
    jsonl_roles = scan_table(jsonl_path).to_dict()
    assert (
        csv_roles
        == jsonl_roles
        == {column: role.value for column, role in EXPECTED_ROLES.items()}
    )

    pytest.importorskip("pyarrow")
    parquet_path = write_table(tmp_path / "table.parquet", rows)
    parquet_roles = scan_table(parquet_path).to_dict()
    assert parquet_roles == csv_roles


def test_override_pins_role_and_confidence() -> None:
    scan = scan_table(build_rows(), overrides={"record_id": "direct_id"})

    assert scan["record_id"] is ColumnRole.DIRECT_ID
    classification = next(
        item for item in scan.classifications if item.column == "record_id"
    )
    assert classification.overridden is True
    assert classification.confidence == 1.0


def test_override_rejects_unknown_column() -> None:
    with pytest.raises(RoleOverrideError):
        scan_table(build_rows(), overrides={"missing_column": "safe"})


def test_override_rejects_unknown_role() -> None:
    with pytest.raises(RoleOverrideError):
        scan_table(build_rows(), overrides={"record_id": "identifier"})


def test_accepts_columnar_mapping_and_dataframe_like() -> None:
    columnar = {
        "zip": ["02139", "30301"],
        "note": [
            "A sufficiently long free-text clinical narrative for shape.",
            "Another sufficiently long free-text clinical narrative here.",
        ],
    }
    mapping_scan = scan_table(columnar)
    assert mapping_scan["zip"] is ColumnRole.QUASI_ID
    assert mapping_scan["note"] is ColumnRole.SENSITIVE

    class _FrameLike:
        def __init__(self, data: dict[str, list[Any]]) -> None:
            self._data = data
            self.columns = list(data)

        def __getitem__(self, key: str) -> list[Any]:
            return self._data[key]

    frame_scan = scan_table(_FrameLike(columnar))
    assert frame_scan.to_dict() == mapping_scan.to_dict()


def test_max_rows_bounds_the_profile() -> None:
    rows = build_rows(count=100)

    scan = scan_table(rows, max_rows=10)

    classification = next(
        item for item in scan.classifications if item.column == "name"
    )
    assert "non_null_count=10" in classification.signals


def test_empty_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        scan_table([])
