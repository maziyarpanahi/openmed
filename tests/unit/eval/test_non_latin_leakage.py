"""Grapheme-aware, script-stratified leakage evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.leakage_heatmap import compute_script_leakage_heatmap
from openmed.eval.metrics import (
    compute_character_recall,
    compute_leakage_rate,
    compute_metrics_bundle,
)
from openmed.eval.report import BenchmarkReport
from openmed.eval.scorecard import ModelScorecard

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "eval" / "non_latin_script_phi.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_split_devanagari_aksara_counts_as_one_leaked_grapheme() -> None:
    text = "क्षि"
    gold = [{"start": 0, "end": len(text), "label": "PERSON", "text": text}]
    predicted = [{"start": 0, "end": len(text) - 1, "label": "PERSON"}]

    leakage = compute_leakage_rate(gold, predicted, source_text=text)
    recall = compute_character_recall(gold, predicted, source_text=text)

    assert leakage.total_graphemes == 1
    assert leakage.leaked_graphemes == 1
    assert leakage.overall == 1.0
    assert recall.denominator == 1
    assert recall.numerator == 0
    assert recall.rate == 0.0


def test_metrics_bundle_reports_every_gold_script() -> None:
    fixture = _fixture()
    bundle = compute_metrics_bundle(
        fixture["gold_spans"],
        fixture["predicted_spans_all_covered"],
        source_text=fixture["text"],
    )

    expected_scripts = {"Arabic", "Devanagari", "Han", "Latin", "Telugu"}
    assert set(bundle["recall_slices"]["by_script"]) == expected_scripts
    assert set(bundle["leakage"]["by_script"]) == expected_scripts
    assert set(bundle["recall_slices"]["total_graphemes_by_script"]) == (
        expected_scripts
    )
    assert all(value == 1.0 for value in bundle["recall_slices"]["by_script"].values())
    assert all(value == 0.0 for value in bundle["leakage"]["by_script"].values())


def test_script_heatmap_is_json_serializable_and_complete() -> None:
    fixture = _fixture()
    heatmap = compute_script_leakage_heatmap(
        fixture["gold_spans"],
        fixture["predicted_spans_telugu_drop"],
        source_text=fixture["text"],
    )

    payload = heatmap.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert heatmap.scripts == (
        "Arabic",
        "Devanagari",
        "Han",
        "Latin",
        "Telugu",
    )
    assert heatmap.cells["Telugu"].rate == 1.0
    assert heatmap.cells["Latin"].rate == 0.0
    assert json.loads(encoded)["scripts"] == list(heatmap.scripts)


def test_scorecard_exposes_per_script_recall_and_leakage() -> None:
    fixture = _fixture()
    metrics = compute_metrics_bundle(
        fixture["gold_spans"],
        fixture["predicted_spans_telugu_drop"],
        source_text=fixture["text"],
    )
    report = BenchmarkReport(
        suite="synthetic-non-latin",
        model_name="OpenMed/pii-script-test",
        device="cpu",
        fixture_count=1,
        metrics=metrics,
    )

    scorecard = ModelScorecard.from_reports([report])
    payload = scorecard.to_dict()
    per_script = payload["device_tiers"][0]["per_script"]

    assert payload["covered_scripts"] == [
        "Arabic",
        "Devanagari",
        "Han",
        "Latin",
        "Telugu",
    ]
    assert per_script["Latin"]["recall"] == pytest.approx(1.0)
    assert per_script["Telugu"]["recall"] == pytest.approx(0.0)
    assert per_script["Telugu"]["leakage_rate"] == pytest.approx(1.0)
    assert "| `cpu` | `Telugu` | 0.00% | 100.00% |" in scorecard.to_markdown()
