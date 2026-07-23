"""Offline eval gate for TNM/AJCC cancer-staging decomposition.

Scores the deterministic parser against the committed synthetic gold set and
enforces the acceptance threshold: per-field staging accuracy >= 0.92 across
basis / T / N / M and their subcategories. Runs fully offline with no models or
network.
"""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import TNM_STAGING_ADVISORY, parse_tnm
from openmed.eval.metrics import TNM_FIELDS, tnm_field_accuracy

GOLD = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "cancer_staging.jsonl"
)

FIELD_FLOOR = 0.92


def _load_gold() -> list[dict]:
    with GOLD.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_gold_set_is_present_and_synthetic():
    rows = _load_gold()
    assert len(rows) >= 12
    assert all(row["metadata"]["synthetic"] is True for row in rows)


def test_per_field_accuracy_meets_floor():
    rows = _load_gold()
    predicted = [parse_tnm(row["text"]) for row in rows]
    gold = [row["gold"] for row in rows]

    per_field = tnm_field_accuracy(predicted, gold)
    assert set(per_field) == set(TNM_FIELDS)
    for field, metric in per_field.items():
        assert metric.rate >= FIELD_FLOOR, (
            f"{field} accuracy {metric.rate:.3f} < {FIELD_FLOOR}"
        )


def test_gold_covers_bases_edge_cases_and_post_therapy():
    rows = _load_gold()
    gold = [row["gold"] for row in rows]

    bases = {row["basis"] for row in gold}
    assert {"yc", "yp"} <= bases  # post-therapy clinical and pathologic
    assert None in bases  # a stage with no written prefix

    t_categories = {row["t"] for row in gold}
    assert "Tis" in t_categories
    assert "TX" in t_categories
    assert {"NX"} <= {row["n"] for row in gold}


def test_confidence_matches_gold_across_the_set():
    # Gate the confidence flag too (not only the per-field decomposition): high
    # for clean stages, low for malformed/ambiguous ones. The gold set carries
    # both, including adversarial rows (T10, N2d, mixed prefixes, bare y).
    rows = _load_gold()
    for row in rows:
        assert parse_tnm(row["text"])["confidence"] == row["gold"]["confidence"], row[
            "text"
        ]
    confidences = {row["gold"]["confidence"] for row in rows}
    assert confidences == {"high", "low"}


def test_unrecognized_tokens_are_surfaced_with_reasons():
    # The never-coerce guarantee, enforced directly (not only via the aggregate
    # floor): every gold row carrying unparsed tokens must reproduce them, and
    # no such row may silently assign a category it flagged.
    rows = _load_gold()
    checked = 0
    for row in rows:
        if row["gold"]["unparsed"]:
            parsed = parse_tnm(row["text"])
            assert parsed["unparsed"] == row["gold"]["unparsed"], row["text"]
            assert parsed["confidence"] == "low"
            for token in parsed["unparsed"]:
                assert token["reason"]
            checked += 1
    assert checked >= 2  # out-of-range value + secondary staging expression


def test_advisory_emitted_on_every_parse():
    rows = _load_gold()
    for row in rows:
        assert parse_tnm(row["text"])["advisory"] == TNM_STAGING_ADVISORY
    assert "not a substitute" in TNM_STAGING_ADVISORY.lower()
