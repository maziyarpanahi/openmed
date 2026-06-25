"""Unit tests for label-by-language leakage heatmaps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.leakage_heatmap import (
    LeakageHeatmap,
    compute_leakage_heatmap,
    render_leakage_heatmap_markdown,
)
from openmed.eval.metrics import compute_leakage_rate


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
    assert heatmap.cells["PERSON"]["en"].rate == pytest.approx(0.0)
    assert heatmap.cells["SSN"]["fr"].rate == pytest.approx(0.0)
    assert heatmap.cells["SSN"]["en"].rate == pytest.approx(1.0)
    assert heatmap.row_totals["SSN"].leaked_chars == 4
    assert heatmap.row_totals["SSN"].total_chars == 10
    assert heatmap.column_totals["fr"].leaked_chars == 5
    assert heatmap.column_totals["fr"].total_chars == 16
    assert heatmap.column_totals["en"].leaked_chars == 4
    assert heatmap.column_totals["en"].total_chars == 8
    assert heatmap.total.leaked_chars == 9
    assert heatmap.total.total_chars == 24


def test_leakage_heatmap_worst_cells_tie_break_by_label_then_language():
    gold = [
        {"start": 0, "end": 1, "label": "AGE", "language": "en"},
        {"start": 2, "end": 3, "label": "ACCOUNT_NUMBER", "language": "fr"},
        {"start": 4, "end": 5, "label": "ACCOUNT_NUMBER", "language": "en"},
    ]

    heatmap = compute_leakage_heatmap(gold, [], worst_n=2)

    assert [(cell.canonical_label, cell.language) for cell in heatmap.worst_cells] == [
        ("ACCOUNT_NUMBER", "en"),
        ("ACCOUNT_NUMBER", "fr"),
    ]


def test_leakage_heatmap_markdown_is_deterministic_and_explicit_for_empty_cells():
    heatmap = compute_leakage_heatmap(
        [{"start": 0, "end": 4, "label": "PERSON", "language": "en"}],
        [],
    )

    markdown = heatmap.to_markdown()
    header = markdown.splitlines()[0]

    assert heatmap.labels == tuple(sorted(CANONICAL_LABELS))
    assert heatmap.languages == tuple(sorted(SUPPORTED_LANGUAGES))
    assert (
        header
        == "| Canonical label | ar | de | en | es | fr | hi | it | ja | nl | pt | te | tr | Total |"
    )
    assert "| `ACCOUNT_NUMBER` | 0/0 (0.000) |" in markdown
    assert "| `PERSON` | 0/0 (0.000) | 0/0 (0.000) | 4/4 (1.000) |" in markdown
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
