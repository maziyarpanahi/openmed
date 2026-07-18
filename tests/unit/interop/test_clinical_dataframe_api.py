from __future__ import annotations

import json
import logging
import socket
import warnings
from html.parser import HTMLParser
from typing import Any

import duckdb
import pytest

from openmed.clinical.exporters.flat_table import FLAT_TABLE_COLUMNS, flatten_entities
from openmed.interop import duckdb_udf, get_adapter
from openmed.interop.notebook.widget import ClinicalExtractionWidget, render_span_html

NOTE = "Patient [PERSON] takes metformin for type 2 diabetes."
RAW_PHI_NOTE = "Jane Roe takes metformin for type 2 diabetes."


def fixture_extractor(text: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for surface, label, code, display in (
        ("metformin", "medication", "6809", "Metformin"),
        (
            "type 2 diabetes",
            "condition",
            "E11.9",
            "Type 2 diabetes mellitus without complications",
        ),
    ):
        start = text.index(surface)
        entities.append(
            {
                "label": label,
                "normalized_text": surface,
                "start": start,
                "end": start + len(surface),
                "coding": {
                    "system": (
                        "http://www.nlm.nih.gov/research/umls/rxnorm"
                        if label == "medication"
                        else "http://hl7.org/fhir/sid/icd-10-cm"
                    ),
                    "code": code,
                    "display": display,
                },
                "context": {"negation": "affirmed", "certainty": "certain"},
            }
        )
    return entities


def test_pandas_polars_extractors_match_flat_table_schema_and_rows():
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("pandas")
    polars_adapter = get_adapter("polars")

    pandas_frame = pd.DataFrame({"note": [NOTE]})
    polars_frame = pl.DataFrame({"note": [NOTE]})

    pandas_result = pandas_frame.openmed.extract(
        "note",
        extractor=fixture_extractor,
        warn_on_phi=False,
    )
    polars_result = polars_adapter.extract_frame(
        polars_frame,
        "note",
        extractor=fixture_extractor,
        warn_on_phi=False,
    )

    expected_rows = flatten_entities(fixture_extractor(NOTE))
    pandas_rows = pandas_result.to_dict("records")
    polars_rows = polars_result.to_dicts()

    assert list(pandas_result.columns) == list(FLAT_TABLE_COLUMNS)
    assert polars_result.columns == list(FLAT_TABLE_COLUMNS)
    assert pandas_rows == expected_rows
    assert json.dumps(pandas_rows, separators=(",", ":")) == json.dumps(
        polars_rows,
        separators=(",", ":"),
    )


def test_ground_alias_matches_extract_for_pandas_and_polars():
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("pandas")
    polars_adapter = get_adapter("polars")

    pandas_frame = pd.DataFrame({"note": [NOTE]})
    polars_frame = pl.DataFrame({"note": [NOTE]})

    assert pandas_frame.openmed.ground(
        "note",
        extractor=fixture_extractor,
        warn_on_phi=False,
    ).to_dict("records") == pandas_frame.openmed.extract(
        "note",
        extractor=fixture_extractor,
        warn_on_phi=False,
    ).to_dict("records")
    assert (
        polars_adapter.ground_frame(
            polars_frame,
            "note",
            extractor=fixture_extractor,
            warn_on_phi=False,
        ).to_dicts()
        == polars_adapter.extract_frame(
            polars_frame,
            "note",
            extractor=fixture_extractor,
            warn_on_phi=False,
        ).to_dicts()
    )


def test_duckdb_clinical_extract_relation_matches_dataframe_rows():
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    pandas_rows = (
        pd.DataFrame({"note": [NOTE]})
        .openmed.extract("note", extractor=fixture_extractor, warn_on_phi=False)
        .to_dict("records")
    )

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE notes AS SELECT ? AS note", [NOTE])

    rows = duckdb_udf.clinical_extract_relation(
        con,
        "SELECT note FROM notes",
        "note",
        extractor=fixture_extractor,
    ).fetchall()

    duckdb_rows = [dict(zip(FLAT_TABLE_COLUMNS, row)) for row in rows]
    assert duckdb_rows == pandas_rows


class _MarkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.marks: list[dict[str, str]] = []
        self.layers: list[dict[str, str]] = []
        self._active: dict[str, str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "pre" and attributes.get("class") == "openmed-span-layer":
            self.layers.append(attributes)
        if tag == "mark":
            self._active = attributes
            self._active["text"] = ""
            self.marks.append(self._active)

    def handle_endtag(self, tag: str) -> None:
        if tag == "mark":
            self._active = None

    def handle_data(self, data: str) -> None:
        if self._active is not None:
            self._active["text"] += data


def test_widget_renders_span_offsets_labels_and_codes_in_dom():
    rows = flatten_entities(fixture_extractor(NOTE))

    html = ClinicalExtractionWidget(NOTE, tuple(rows)).to_html()
    parser = _MarkParser()
    parser.feed(html)

    assert parser.layers == [
        {
            "class": "openmed-span-layer",
            "data-layer": "0",
            "aria-label": "Annotation layer 1",
        }
    ]
    assert parser.marks == [
        {
            "class": "openmed-span",
            "data-start": str(NOTE.index("metformin")),
            "data-end": str(NOTE.index("metformin") + len("metformin")),
            "data-label": "medication",
            "data-code": "6809",
            "title": "medication | 6809 | Metformin",
            "style": ("background:#dcfce7;padding:0 0.15rem;border-radius:3px;"),
            "text": "metformin",
        },
        {
            "class": "openmed-span",
            "data-start": str(NOTE.index("type 2 diabetes")),
            "data-end": str(NOTE.index("type 2 diabetes") + len("type 2 diabetes")),
            "data-label": "condition",
            "data-code": "E11.9",
            "title": (
                "condition | E11.9 | Type 2 diabetes mellitus without complications"
            ),
            "style": ("background:#d7f0ff;padding:0 0.15rem;border-radius:3px;"),
            "text": "type 2 diabetes",
        },
    ]
    assert render_span_html(NOTE, rows) == html


def test_widget_preserves_nested_and_crossing_spans_in_dom():
    text = "alpha beta gamma"
    rows = [
        {
            "start": 0,
            "end": 10,
            "entity_label": "condition",
            "code": "A",
            "display": "Alpha beta",
        },
        {
            "start": 6,
            "end": 16,
            "entity_label": "procedure",
            "code": "B",
            "display": "Beta gamma",
        },
        {
            "start": 6,
            "end": 10,
            "entity_label": "finding",
            "code": "C",
            "display": "Beta",
        },
    ]

    parser = _MarkParser()
    parser.feed(render_span_html(text, rows))

    assert [layer["data-layer"] for layer in parser.layers] == ["0", "1", "2"]
    assert [
        (
            mark["data-start"],
            mark["data-end"],
            mark["data-label"],
            mark["data-code"],
            mark["text"],
        )
        for mark in parser.marks
    ] == [
        ("0", "10", "condition", "A", "alpha beta"),
        ("6", "16", "procedure", "B", "beta gamma"),
        ("6", "10", "finding", "C", "beta"),
    ]


def test_widget_preserves_note_when_there_are_no_spans():
    parser = _MarkParser()
    html = render_span_html("No <entities> found.", [])
    parser.feed(html)

    assert [layer["data-layer"] for layer in parser.layers] == ["0"]
    assert parser.marks == []
    assert "No &lt;entities&gt; found." in html


def test_accessors_widget_and_duckdb_do_not_open_sockets_or_log_raw_phi(
    monkeypatch,
    caplog,
):
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("pandas")
    polars_adapter = get_adapter("polars")

    def blocked_socket(*args, **kwargs):
        raise AssertionError("network egress is not allowed")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    caplog.set_level(logging.DEBUG)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pandas_rows = (
            pd.DataFrame({"raw_phi_note": [RAW_PHI_NOTE]})
            .openmed.extract("raw_phi_note", extractor=fixture_extractor)
            .to_dict("records")
        )
        polars_adapter.extract_frame(
            pl.DataFrame({"raw_phi_note": [RAW_PHI_NOTE]}),
            "raw_phi_note",
            extractor=fixture_extractor,
        ).to_dicts()

        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE notes AS SELECT ? AS note", [RAW_PHI_NOTE])
        duckdb_udf.clinical_extract_relation(
            con,
            "SELECT note FROM notes",
            "note",
            extractor=fixture_extractor,
        ).fetchall()
        render_span_html(RAW_PHI_NOTE, pandas_rows)

    assert caught
    assert all(RAW_PHI_NOTE not in str(item.message) for item in caught)
    assert all("Jane Roe" not in record.getMessage() for record in caplog.records)
