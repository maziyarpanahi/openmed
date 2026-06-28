"""Tests for the clinical entity flat-table exporter."""

from __future__ import annotations

import builtins
import csv
import importlib
import io
from dataclasses import dataclass

import pytest

from openmed.clinical.exporters import (
    FLAT_TABLE_COLUMNS,
    flatten_entities,
    to_csv,
    to_dataframe,
)


@dataclass
class _Entity:
    text: str
    label: str
    confidence: float
    start: int
    end: int
    metadata: dict


def test_flatten_entities_returns_one_ordered_row_per_entity():
    entities = [
        {
            "label": "condition",
            "normalized_text": "type 2 diabetes",
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "44054006",
                        "display": "Diabetes mellitus type 2",
                    }
                ]
            },
            "context": {
                "negation": "affirmed",
                "temporality": "recent",
                "certainty": "certain",
            },
            "start": 12,
            "end": 27,
            "metadata": {"section": "assessment", "note_text": "not exported"},
        },
        _Entity(
            text="metformin",
            label="medication",
            confidence=0.93,
            start=42,
            end=51,
            metadata={
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": "6809",
                "display": "Metformin",
                "negation": "affirmed",
                "temporality": "recent",
                "certainty": "certain",
                "section": "plan",
            },
        ),
    ]

    rows = flatten_entities(entities)

    assert len(rows) == 2
    assert list(rows[0]) == list(FLAT_TABLE_COLUMNS)
    assert rows[0] == {
        "entity_label": "condition",
        "normalized_text": "type 2 diabetes",
        "system": "http://snomed.info/sct",
        "code": "44054006",
        "display": "Diabetes mellitus type 2",
        "negation": "affirmed",
        "temporality": "recent",
        "certainty": "certain",
        "start": 12,
        "end": 27,
        "section": "assessment",
    }
    assert "note_text" not in rows[0]
    assert rows[1]["normalized_text"] == "metformin"
    assert rows[1]["code"] == "6809"
    assert rows[1]["section"] == "plan"


def test_to_csv_writes_header_for_empty_entities():
    output = to_csv([])

    assert output == ",".join(FLAT_TABLE_COLUMNS) + "\n"
    parsed = list(csv.DictReader(io.StringIO(output)))
    assert parsed == []


def test_to_csv_returns_parseable_csv_and_quotes_values():
    output = to_csv(
        [
            {
                "entity_label": "condition",
                "normalized_text": 'nausea, "severe"',
                "system": "http://snomed.info/sct",
                "code": "422587007",
                "display": 'Nausea, "severe"',
            }
        ]
    )

    rows = list(csv.DictReader(io.StringIO(output)))

    assert rows == [
        {
            "entity_label": "condition",
            "normalized_text": 'nausea, "severe"',
            "system": "http://snomed.info/sct",
            "code": "422587007",
            "display": 'Nausea, "severe"',
            "negation": "",
            "temporality": "",
            "certainty": "",
            "start": "",
            "end": "",
            "section": "",
        }
    ]


def test_to_csv_writes_to_text_stream():
    stream = io.StringIO(newline="")

    result = to_csv([{"label": "procedure", "text": "biopsy"}], stream)

    assert result is None
    assert stream.getvalue().splitlines()[0] == ",".join(FLAT_TABLE_COLUMNS)


def test_to_dataframe_raises_clear_error_when_pandas_missing(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match="to_dataframe requires pandas"):
        to_dataframe([])


def test_to_dataframe_returns_dataframe_with_fixed_schema():
    pd = pytest.importorskip("pandas")

    frame = to_dataframe([{"label": "condition", "text": "asthma"}])

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(FLAT_TABLE_COLUMNS)
    assert frame.to_dict("records")[0]["entity_label"] == "condition"


def test_importing_flat_table_does_not_import_pandas(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pandas":
            raise AssertionError("pandas should not be imported at module import time")
        return real_import(name, *args, **kwargs)

    import openmed.clinical.exporters.flat_table as flat_table

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(flat_table)
