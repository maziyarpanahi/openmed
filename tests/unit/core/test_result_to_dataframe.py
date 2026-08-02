"""Tests for DeidentificationResult.to_dataframe()."""

from __future__ import annotations

import builtins
import subprocess
import sys
from datetime import datetime

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity

EXPECTED_COLUMNS = [
    "text",
    "label",
    "entity_type",
    "start",
    "end",
    "confidence",
    "action",
    "result_id",
]


def _result_with_entities() -> DeidentificationResult:
    return DeidentificationResult(
        original_text="Patient John Doe called 555-1234",
        deidentified_text="Patient [NAME] called [PHONE]",
        pii_entities=[
            PIIEntity(
                text="John Doe",
                label="NAME",
                entity_type="PERSON",
                start=8,
                end=16,
                confidence=0.98,
                action="mask",
            ),
            PIIEntity(
                text="555-1234",
                label="PHONE",
                start=24,
                end=32,
                confidence=0.91,
                action="mask",
            ),
        ],
        method="mask",
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )


def _empty_result() -> DeidentificationResult:
    return DeidentificationResult(
        original_text="No identifiers here.",
        deidentified_text="No identifiers here.",
        pii_entities=[],
        method="mask",
        timestamp=datetime(2026, 1, 2, 3, 4, 5),
    )


def test_to_dataframe_returns_one_row_per_entity_with_documented_columns():
    pytest.importorskip("pandas", exc_type=ImportError)

    df = _result_with_entities().to_dataframe()

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2

    records = df.to_dict("records")
    result_id = records[0]["result_id"]
    assert isinstance(result_id, str)
    assert len(result_id) == 64
    assert records[1]["result_id"] == result_id
    assert records == [
        {
            "text": "John Doe",
            "label": "NAME",
            "entity_type": "PERSON",
            "start": 8,
            "end": 16,
            "confidence": 0.98,
            "action": "mask",
            "result_id": result_id,
        },
        {
            "text": "555-1234",
            "label": "PHONE",
            "entity_type": "PHONE",
            "start": 24,
            "end": 32,
            "confidence": 0.91,
            "action": "mask",
            "result_id": result_id,
        },
    ]


def test_to_dataframe_result_id_is_stable_for_same_result():
    pytest.importorskip("pandas", exc_type=ImportError)
    result = _result_with_entities()

    first = result.to_dataframe()["result_id"].tolist()
    second = result.to_dataframe()["result_id"].tolist()

    assert first == second


def test_to_dataframe_empty_entities_returns_empty_documented_frame():
    pytest.importorskip("pandas", exc_type=ImportError)

    df = _empty_result().to_dataframe()

    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.empty


def test_importing_core_pii_does_not_import_pandas():
    code = """
import sys
sys.modules.pop("pandas", None)
import openmed.core.pii  # noqa: F401
if "pandas" in sys.modules:
    raise SystemExit("pandas was imported")
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_to_dataframe_without_pandas_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def block_pandas_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("blocked pandas import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_pandas_import)

    with pytest.raises(ImportError, match="pip install pandas"):
        _empty_result().to_dataframe()
