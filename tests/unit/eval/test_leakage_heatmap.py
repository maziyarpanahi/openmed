"""Unit tests for label-by-language leakage heatmaps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.leakage_heatmap import (
    HeatmapCell,
    LeakageHeatmap,
    compute_leakage_heatmap,
    render_leakage_heatmap_markdown,
)
from openmed.eval.metrics import compute_leakage_rate


def _span(start: int, end: int, label: str, language: str) -> dict[str, Any]:
    return {"start": start, "end": end, "label": label, "language": language}


def test_leakage_heatmap_cells_match_restricted_leakage_rates():
    text = "John      FR-ID-1234     123456      9999"
    gold = [
        {
            "start": 0,
            "end": 4,
            "label": "PERSON",
            "language": "en",
            "text": "John",
        },
        {
            "start": 10,
            "end": 20,
            "label": "ID_NUM",
            "language": "fr",
            "text": "FR-ID-1234",
        },
        {
            "start": 25,
            "end": 31,
            "label": "SSN",
            "language": "fr",
            "text": "123456",
        },
        {
            "start": 37,
            "end": 41,
            "label": "SSN",
            "language": "en",
            "text": "9999",
        },
    ]
    predicted = [
        {"start": 0, "end": 4, "label": "PERSON", "language": "en"},
        {"start": 10, "end": 15, "label": "ID_NUM", "language": "fr"},
        {"start": 25, "end": 31, "label": "SSN", "language": "fr"},
        {"start": 37, "end": 41, "label": "PERSON", "language": "en"},
    ]

    heatmap = compute_leakage_heatmap(gold, predicted, source_text=text)

    id_fr = heatmap.cells["ID_NUM"]["fr"]
    id_fr_restricted = compute_leakage_rate(
        [gold[1]],
        predicted,
        source_text=text,
    )
    assert id_fr.rate == pytest.approx(id_fr_restricted.overall)
    assert id_fr.leaked_chars == 5
    assert id_fr.total_chars == 10
    assert heatmap.cells[("ID_NUM", "fr")] is id_fr
    assert heatmap.cells["PERSON"]["en"].rate == pytest.approx(0.0)
    assert heatmap.cells["SSN"]["fr"].rate == pytest.approx(0.0)
    assert heatmap.cells["SSN"]["en"].rate == pytest.approx(1.0)
    assert heatmap.row_totals["SSN"].leaked_chars == 4
    assert heatmap.row_totals["SSN"].total_chars == 10
    assert heatmap.column_totals["fr"].leaked_chars == 5
    assert heatmap.column_totals["fr"].total_chars == 16
    assert heatmap.column_totals["en"].leaked_chars == 4
    assert heatmap.column_totals["en"].total_chars == 8
    assert heatmap.col_totals["fr"] is heatmap.column_totals["fr"]
    assert heatmap.total.leaked_chars == 9
    assert heatmap.total.total_chars == 24


def test_no_phi_cell_hidden_by_aggregate():
    gold: list[dict[str, Any]] = []
    predicted: list[dict[str, Any]] = []
    offset = 0
    for language in ["en", "de", "es"]:
        for label in ["PERSON", "DATE", "ID_NUM"]:
            span = _span(offset, offset + 10, label, language)
            gold.append(span)
            predicted.append(span)
            offset += 15

    for label in ["PERSON", "DATE"]:
        span = _span(offset, offset + 10, label, "fr")
        gold.append(span)
        predicted.append(span)
        offset += 15

    for _ in range(5):
        gold.append(_span(offset, offset + 10, "ID_NUM", "fr"))
        offset += 15

    heatmap = compute_leakage_heatmap(gold, predicted)

    assert heatmap.cells["ID_NUM"]["fr"].rate == pytest.approx(1.0)
    assert ("ID_NUM", "fr") in {(cell.label, cell.language) for cell in heatmap.worst}


def test_leakage_heatmap_worst_cells_tie_break_by_label_then_language():
    gold = [
        _span(0, 1, "AGE", "en"),
        _span(2, 3, "ACCOUNT_NUMBER", "fr"),
        _span(4, 5, "ACCOUNT_NUMBER", "en"),
    ]

    heatmap = compute_leakage_heatmap(gold, [], worst_n=2)

    assert [(cell.canonical_label, cell.language) for cell in heatmap.worst_cells] == [
        ("ACCOUNT_NUMBER", "en"),
        ("ACCOUNT_NUMBER", "fr"),
    ]
    assert heatmap.worst[0].label == "ACCOUNT_NUMBER"


def test_worst_n_respects_limit_and_zero():
    gold = [
        _span(index * 15, index * 15 + 10, "PERSON", f"L{index}") for index in range(10)
    ]

    limited = compute_leakage_heatmap(gold, [], worst_n=3)
    empty = compute_leakage_heatmap(gold, [], worst_n=0)

    assert len(limited.worst_cells) == 3
    assert limited.worst == list(limited.worst_cells)
    assert empty.worst_cells == ()
    assert empty.worst == []


def test_leakage_heatmap_markdown_is_deterministic_and_explicit_for_empty_cells():
    heatmap = compute_leakage_heatmap(
        [_span(0, 4, "PERSON", "en")],
        [],
    )

    markdown = heatmap.to_markdown()
    header = markdown.splitlines()[0]
    expected_header = (
        "| Canonical label | " + " | ".join(sorted(SUPPORTED_LANGUAGES)) + " | Total |"
    )
    person_row = next(
        line for line in markdown.splitlines() if line.startswith("| `PERSON` |")
    )

    assert heatmap.labels == tuple(sorted(CANONICAL_LABELS))
    assert heatmap.languages == tuple(sorted(SUPPORTED_LANGUAGES))
    assert header == expected_header
    assert "| `ACCOUNT_NUMBER` | 0/0 (0.000) |" in markdown
    assert "4/4 (1.000)" in person_row
    assert person_row.count("0/0 (0.000)") == len(SUPPORTED_LANGUAGES) - 1
    assert markdown == heatmap.to_markdown()
    assert render_leakage_heatmap_markdown(heatmap) == markdown


def test_leakage_heatmap_serialization_excludes_span_text_and_offsets():
    heatmap = compute_leakage_heatmap(
        [
            {
                "start": 0,
                "end": 5,
                "label": "PERSON",
                "language": "en",
                "text": "Alice",
            },
            {
                "start": 10,
                "end": 21,
                "label": "SSN",
                "language": "en",
                "text": "123-45-6789",
            },
        ],
        [],
    )

    payload = heatmap.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    keys = set(_walk_keys(payload))

    assert isinstance(heatmap, LeakageHeatmap)
    assert "Alice" not in encoded
    assert "123-45-6789" not in encoded
    assert {"text", "start", "end", "offset", "offsets"}.isdisjoint(keys)


def test_cell_alias_constructor_remains_supported():
    cell = HeatmapCell(
        label="ID_NUM", language="fr", leaked_chars=3, total_chars=5, rate=0.6
    )

    assert cell.canonical_label == "ID_NUM"
    assert cell.label == "ID_NUM"
    assert cell["language"] == "fr"
    assert cell["rate"] == pytest.approx(0.6)


def test_row_and_column_totals_keep_interim_aliases():
    gold = [
        _span(0, 10, "ID_NUM", "fr"),
        _span(20, 30, "ID_NUM", "en"),
        _span(40, 50, "DATE", "fr"),
    ]
    predicted = [
        _span(20, 30, "ID_NUM", "en"),
        _span(40, 50, "DATE", "fr"),
    ]

    heatmap = compute_leakage_heatmap(gold, predicted)

    assert heatmap.row_totals["ID_NUM"].leaked_chars == 10
    assert heatmap.row_totals["ID_NUM"].total_chars == 20
    assert heatmap.row_totals["ID_NUM"].rate == pytest.approx(0.5)
    assert heatmap.row_totals["ID_NUM"].language == "all"
    assert heatmap.row_totals["DATE"].rate == pytest.approx(0.0)
    assert heatmap.col_totals["fr"].leaked_chars == 10
    assert heatmap.col_totals["fr"].total_chars == 20
    assert heatmap.col_totals["fr"].rate == pytest.approx(0.5)
    assert heatmap.col_totals["fr"].label == "all"
    assert heatmap.col_totals["en"].rate == pytest.approx(0.0)


def test_empty_gold_returns_explicit_empty_heatmap():
    heatmap = compute_leakage_heatmap([], [])

    assert heatmap.labels == tuple(sorted(CANONICAL_LABELS))
    assert heatmap.languages == tuple(sorted(SUPPORTED_LANGUAGES))
    assert heatmap.cells["PERSON"]["en"].total_chars == 0
    assert heatmap.worst_cells == ()
    assert heatmap.worst == []
    assert heatmap.total.leaked_chars == 0
    assert heatmap.total.total_chars == 0


def test_partial_coverage_cell_rate():
    gold = [_span(0, 10, "DATE", "en")]
    predicted = [_span(0, 4, "DATE", "en")]

    heatmap = compute_leakage_heatmap(gold, predicted)

    assert heatmap.cells["DATE"]["en"].leaked_chars == 6
    assert heatmap.cells[("DATE", "en")].rate == pytest.approx(0.6)


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        keys = [str(key) for key in value]
        for nested in value.values():
            keys.extend(_walk_keys(nested))
        return keys
    if isinstance(value, list):
        keys: list[str] = []
        for item in value:
            keys.extend(_walk_keys(item))
        return keys
    return []
