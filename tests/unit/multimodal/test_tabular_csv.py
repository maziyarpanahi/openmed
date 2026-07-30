"""Tests for CSV/TSV column-scoped PHI redaction."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from openmed.multimodal import ExtractedDocument, redact_document
from openmed.multimodal.tabular_csv import (
    ACTION_DATE_SHIFT,
    ACTION_FREE_TEXT_REDACT,
    ACTION_HASH,
    ACTION_KEEP,
    ACTION_MASK,
    DIRECT_ID,
    QUASI_ID,
    SAFE,
    redact_table,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(text: str, delimiter: str = ",") -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def test_redact_table_sniffs_csv_and_preserves_safe_columns_and_rows():
    result = redact_table(
        FIXTURES / "synthetic_phi.csv",
        date_shift_days=7,
        text_redactor=lambda text: text.replace("555-0101", "[PHONE]"),
    )

    rows = _rows(result.text)
    assert len(rows) == 2
    assert [row["diagnosis"] for row in rows] == ["hypertension", "diabetes"]
    assert rows[0]["patient_name"] == "[PERSON]"
    assert rows[1]["patient_name"] == "[PERSON]"
    assert rows[0]["mrn"].startswith("ID_NUM_")
    assert rows[0]["ssn"] == "[SSN]"
    assert rows[0]["visit_date"] == "2026-01-22"
    assert rows[0]["notes"] == "Called [PHONE] about labs"


def test_redact_table_sniffs_tsv_without_manual_delimiter():
    result = redact_table(FIXTURES / "synthetic_phi.tsv", date_shift_days=3)

    rows = _rows(result.text, delimiter="\t")
    assert len(rows) == 2
    assert [row["diagnosis"] for row in rows] == ["hypertension", "diabetes"]
    assert rows[0]["patient_name"] == "[PERSON]"
    assert rows[0]["mrn"].startswith("ID_NUM_")
    assert rows[0]["visit_date"] == "2026-03-04"


def test_columns_are_detected_by_value_sampling_without_phi_headers():
    source = (
        "record_key,subject,encounter_date,value\n"
        "MRN-9001,John Doe,2026-04-01,42\n"
        "MRN-9002,Jane Roe,2026-04-02,43\n"
    )

    result = redact_table(source, date_shift_days=5)
    rows = _rows(result.text)
    manifest = {entry["column_name"]: entry for entry in result.manifest}

    assert rows[0]["record_key"].startswith("ID_NUM_")
    assert rows[0]["subject"] == "[PERSON]"
    assert rows[0]["encounter_date"] == "2026-04-06"
    assert manifest["record_key"]["detection_source"] == "value_sample"
    assert manifest["subject"]["detection_source"] == "value_sample"


def test_date_of_birth_columns_default_to_mask_not_keep_year_shift():
    source = "patient_name,dob,visit_date\nJohn Doe,1980-01-01,2026-04-01\n"

    result = redact_table(source, date_shift_days=30, keep_year=True)
    rows = _rows(result.text)
    manifest = {entry["column_name"]: entry for entry in result.manifest}

    assert rows[0]["dob"] == "[DATE_OF_BIRTH]"
    assert rows[0]["visit_date"] == "2026-05-01"
    assert manifest["dob"]["assigned_class"] == DIRECT_ID
    assert manifest["dob"]["action"] == ACTION_MASK
    assert manifest["visit_date"]["assigned_class"] == QUASI_ID
    assert manifest["visit_date"]["action"] == ACTION_DATE_SHIFT
    assert "1980" not in result.text


def test_manifest_lists_each_column_class_and_action_without_raw_phi():
    result = redact_table(
        FIXTURES / "synthetic_phi.csv",
        date_shift_days=7,
        text_redactor=lambda text: text.replace("555-0101", "[PHONE]"),
    )
    manifest = {entry["column_name"]: entry for entry in result.manifest}

    assert manifest["patient_name"]["assigned_class"] == DIRECT_ID
    assert manifest["patient_name"]["action"] == ACTION_MASK
    assert manifest["mrn"]["assigned_class"] == DIRECT_ID
    assert manifest["mrn"]["action"] == ACTION_HASH
    assert manifest["visit_date"]["assigned_class"] == QUASI_ID
    assert manifest["visit_date"]["action"] == ACTION_DATE_SHIFT
    assert manifest["diagnosis"]["assigned_class"] == SAFE
    assert manifest["diagnosis"]["action"] == ACTION_KEEP
    assert manifest["notes"]["assigned_class"] == SAFE
    assert manifest["notes"]["action"] == ACTION_FREE_TEXT_REDACT
    assert manifest["mrn"]["row_count_affected"] == 2
    assert "John Doe" not in str(result.manifest)
    assert "123-45-6789" not in str(result.manifest)


def test_action_overrides_can_drop_columns():
    result = redact_table(
        FIXTURES / "synthetic_phi.csv",
        action_overrides={"ssn": "drop", "notes": "keep"},
    )

    rows = _rows(result.text)
    manifest = {entry["column_name"]: entry for entry in result.manifest}

    assert "ssn" not in rows[0]
    assert manifest["ssn"]["action"] == "drop"
    assert manifest["ssn"]["row_count_affected"] == 2
    assert rows[0]["notes"] == "Called 555-0101 about labs"


def test_redact_document_dispatches_csv_handler_with_manifest_metadata():
    doc = redact_document(
        str(FIXTURES / "synthetic_phi.csv"),
        models={"text_redactor": lambda text: text.replace("555-0101", "[PHONE]")},
    )

    assert isinstance(doc, ExtractedDocument)
    assert doc.metadata["format"] == "csv"
    assert doc.metadata["row_count"] == 2
    assert any(
        entry["column_name"] == "mrn" and entry["action"] == ACTION_HASH
        for entry in doc.metadata["redaction_manifest"]
    )
