"""Unit tests for the (label, language) leakage heatmap."""

from __future__ import annotations

import pytest

from openmed.eval.leakage_heatmap import (
    HeatmapCell,
    LeakageHeatmap,
    compute_leakage_heatmap,
)


def _span(start, end, label, lang):
    return {"start": start, "end": end, "label": label, "language": lang}


def test_cell_rate_equals_leakage_restricted_to_label_and_language():
    # 10-char French ID_NUM fully leaked, 10-char English ID_NUM fully covered
    gold = [_span(0, 10, "ID_NUM", "fr"), _span(20, 30, "ID_NUM", "en")]
    predicted = [_span(20, 30, "ID_NUM", "en")]

    heatmap = compute_leakage_heatmap(gold, predicted)

    assert heatmap.cells[("ID_NUM", "fr")].rate == pytest.approx(1.0)
    assert heatmap.cells[("ID_NUM", "fr")].leaked_chars == 10
    assert heatmap.cells[("ID_NUM", "fr")].total_chars == 10

    assert heatmap.cells[("ID_NUM", "en")].rate == pytest.approx(0.0)
    assert heatmap.cells[("ID_NUM", "en")].leaked_chars == 0


def test_no_phi_cell_hidden_by_aggregate():
    # Reproduce the blind spot: overall and 1D slices look fine,
    # but (ID_NUM, fr) is 100% leaked.
    gold, predicted = [], []
    offset = 0
    for lang in ["en", "de", "es"]:
        for label in ["PERSON", "DATE", "ID_NUM"]:
            s = _span(offset, offset + 10, label, lang)
            gold.append(s)
            predicted.append(s)
            offset += 15

    for label in ["PERSON", "DATE"]:
        s = _span(offset, offset + 10, label, "fr")
        gold.append(s)
        predicted.append(s)
        offset += 15

    for _ in range(5):
        gold.append(_span(offset, offset + 10, "ID_NUM", "fr"))
        offset += 15

    heatmap = compute_leakage_heatmap(gold, predicted)

    assert heatmap.cells[("ID_NUM", "fr")].rate == pytest.approx(1.0)
    assert ("ID_NUM", "fr") in {(c.label, c.language) for c in heatmap.worst}


def test_worst_n_is_deterministic_and_tie_broken_by_label_language():
    # Two cells with identical 100% rate — tie should break by (label, lang)
    gold = [
        _span(0, 10, "ID_NUM", "fr"),
        _span(20, 30, "PERSON", "de"),
    ]
    heatmap = compute_leakage_heatmap(gold, [], worst_n=2)

    assert len(heatmap.worst) == 2
    # Both are 100% — tie-break: "ID_NUM" < "PERSON" alphabetically
    assert heatmap.worst[0].label == "ID_NUM"
    assert heatmap.worst[1].label == "PERSON"


def test_worst_n_respects_limit():
    gold = [_span(i * 15, i * 15 + 10, "PERSON", f"L{i}") for i in range(10)]
    heatmap = compute_leakage_heatmap(gold, [], worst_n=3)
    assert len(heatmap.worst) <= 3


def test_output_contains_no_span_text():
    gold = [_span(0, 10, "ID_NUM", "fr")]
    heatmap = compute_leakage_heatmap(gold, [])
    d = heatmap.to_dict()
    flat = str(d)
    # to_dict must not include start/end offsets or text field
    assert "start" not in flat
    assert "end" not in flat
    assert "text" not in flat


def test_to_markdown_renders_matrix():
    gold = [_span(0, 10, "ID_NUM", "fr"), _span(20, 30, "PERSON", "en")]
    predicted = [_span(20, 30, "PERSON", "en")]
    heatmap = compute_leakage_heatmap(gold, predicted)

    md = heatmap.to_markdown()
    assert "ID_NUM" in md
    assert "PERSON" in md
    assert "fr" in md
    assert "en" in md
    assert "100.0%" in md
    assert "0.0%" in md


def test_row_and_col_totals_aggregate_correctly():
    # ID_NUM/fr: 10 chars leaked; ID_NUM/en: 0 leaked; DATE/fr: 0 leaked
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

    # row total for ID_NUM: 10 leaked / 20 total = 50%
    assert heatmap.row_totals["ID_NUM"].leaked_chars == 10
    assert heatmap.row_totals["ID_NUM"].total_chars == 20
    assert heatmap.row_totals["ID_NUM"].rate == pytest.approx(0.5)
    assert heatmap.row_totals["ID_NUM"].language == "all"

    # row total for DATE: 0 leaked / 10 total = 0%
    assert heatmap.row_totals["DATE"].rate == pytest.approx(0.0)

    # col total for fr: 10 leaked / 20 total = 50%
    assert heatmap.col_totals["fr"].leaked_chars == 10
    assert heatmap.col_totals["fr"].total_chars == 20
    assert heatmap.col_totals["fr"].rate == pytest.approx(0.5)
    assert heatmap.col_totals["fr"].label == "all"

    # col total for en: 0 leaked / 10 total = 0%
    assert heatmap.col_totals["en"].rate == pytest.approx(0.0)


def test_to_markdown_includes_totals_row_and_column():
    gold = [_span(0, 10, "ID_NUM", "fr"), _span(20, 30, "PERSON", "en")]
    predicted = [_span(20, 30, "PERSON", "en")]
    md = compute_leakage_heatmap(gold, predicted).to_markdown()
    assert "Total" in md


def test_empty_gold_returns_empty_heatmap():
    heatmap = compute_leakage_heatmap([], [])
    assert heatmap.cells == {}
    assert heatmap.worst == []


def test_partial_coverage_cell_rate():
    # 10-char span, 4 chars covered → 6 leaked → rate = 0.6
    gold = [_span(0, 10, "DATE", "en")]
    predicted = [_span(0, 4, "DATE", "en")]
    heatmap = compute_leakage_heatmap(gold, predicted)
    assert heatmap.cells[("DATE", "en")].leaked_chars == 6
    assert heatmap.cells[("DATE", "en")].rate == pytest.approx(0.6)
