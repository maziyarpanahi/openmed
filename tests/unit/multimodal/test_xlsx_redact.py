"""Tests for cell-level XLSX redaction and PHI-safe reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl", exc_type=ImportError)

from openmed.multimodal.xlsx import redact_xlsx


def _synthetic_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    patients = workbook.active
    patients.title = "Patient John Doe"
    patients.append(["patient_name", "age", "notes", "calculated"])
    patients.append(["John Doe", 42, "Call John Doe at 555-0101", "=B2*2"])
    patients.append(["Jane Roe", 37, "No PHI in this cell", "=B3*2"])
    patients.freeze_panes = "A2"

    archive = workbook.create_sheet("Archive")
    archive.append(["ssn", "status"])
    archive.append(["123-45-6789", "active"])
    workbook.save(path)
    workbook.close()


def _cell_deidentifier(value: str) -> tuple[str, tuple[str, ...]]:
    if value == "Call John Doe at 555-0101":
        return "Call [PERSON] at [PHONE]", ("PERSON", "PHONE")
    return value, ()


def test_redact_xlsx_preserves_structure_formulas_and_non_string_cells(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic.xlsx"
    output = tmp_path / "synthetic.redacted.xlsx"
    _synthetic_workbook(source)

    result = redact_xlsx(
        source,
        output,
        cell_deidentifier=_cell_deidentifier,
    )

    workbook = openpyxl.load_workbook(output, data_only=False)
    assert workbook.sheetnames == ["Patient John Doe", "Archive"]
    assert workbook["Patient John Doe"]["A2"].value == "[PERSON]"
    assert workbook["Patient John Doe"]["A3"].value == "[PERSON]"
    assert workbook["Patient John Doe"]["B2"].value == 42
    assert workbook["Patient John Doe"]["B3"].value == 37
    assert workbook["Patient John Doe"]["C2"].value == ("Call [PERSON] at [PHONE]")
    assert workbook["Patient John Doe"]["C3"].value == "No PHI in this cell"
    assert workbook["Patient John Doe"]["D2"].value == "=B2*2"
    assert workbook["Patient John Doe"]["D3"].value == "=B3*2"
    assert workbook["Patient John Doe"].freeze_panes == "A2"
    assert workbook["Archive"]["A2"].value == "[SSN]"
    assert workbook["Archive"]["B2"].value == "active"
    workbook.close()

    original = openpyxl.load_workbook(source, data_only=False)
    assert original["Patient John Doe"]["A2"].value == "John Doe"
    assert original["Archive"]["A2"].value == "123-45-6789"
    original.close()

    assert result.output_path == output
    assert result.sheet_count == 2


def test_redaction_report_contains_coordinates_and_labels_without_raw_phi(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic.xlsx"
    output = tmp_path / "synthetic.redacted.xlsx"
    _synthetic_workbook(source)

    result = redact_xlsx(
        source,
        output,
        cell_deidentifier=_cell_deidentifier,
    )
    report = result.redaction_report

    assert report == (
        {"sheet_index": 0, "coordinate": "A2", "labels": ["PERSON"]},
        {
            "sheet_index": 0,
            "coordinate": "C2",
            "labels": ["PERSON", "PHONE"],
        },
        {"sheet_index": 0, "coordinate": "A3", "labels": ["PERSON"]},
        {"sheet_index": 1, "coordinate": "A2", "labels": ["SSN"]},
    )
    serialized = str(report)
    assert "John Doe" not in serialized
    assert "Jane Roe" not in serialized
    assert "123-45-6789" not in serialized
    assert "Patient John Doe" not in serialized
    assert all(
        set(entry) == {"sheet_index", "coordinate", "labels"} for entry in report
    )

    document = result.to_document()
    assert document.text == ""
    assert document.metadata["format"] == "xlsx"
    assert document.metadata["sheet_count"] == 2
    assert document.metadata["redaction_report"] == list(report)


def test_redact_xlsx_rejects_in_place_redaction(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.xlsx"
    _synthetic_workbook(source)

    with pytest.raises(ValueError, match="must differ"):
        redact_xlsx(source, source, cell_deidentifier=_cell_deidentifier)
