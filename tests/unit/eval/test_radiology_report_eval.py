"""Offline eval gate for radiology report segmentation + stated-category capture.

Scores the deterministic parser against the committed synthetic gold set and
enforces the acceptance thresholds: section-boundary accuracy >= 0.90 and
stated-category extraction accuracy >= 0.95. Runs fully offline with no models
or network.
"""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import RADIOLOGY_REPORT_ADVISORY, parse_radiology_report
from openmed.eval.metrics import section_boundary_accuracy, stated_category_accuracy

GOLD = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "radiology_report.jsonl"
)

BOUNDARY_FLOOR = 0.90
CATEGORY_FLOOR = 0.95


def _load_gold() -> list[dict]:
    with GOLD.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_gold_set_is_present_and_synthetic():
    rows = _load_gold()
    assert len(rows) >= 10
    assert all(row["metadata"]["synthetic"] is True for row in rows)


def test_section_boundary_and_category_accuracy_meet_floor():
    rows = _load_gold()
    predicted = [parse_radiology_report(row["text"]) for row in rows]
    gold = [row["gold"] for row in rows]

    boundary = section_boundary_accuracy(predicted, gold)
    category = stated_category_accuracy(predicted, gold)

    assert boundary.rate >= BOUNDARY_FLOOR, (
        f"section-boundary accuracy {boundary.rate:.3f} < {BOUNDARY_FLOOR}"
    )
    assert category.rate >= CATEGORY_FLOOR, (
        f"stated-category accuracy {category.rate:.3f} < {CATEGORY_FLOOR}"
    )


def test_gold_covers_birads_and_lungrads_and_absent_categories():
    rows = _load_gold()
    systems = {row["gold"]["assessment_system"] for row in rows}
    assert "BI-RADS" in systems
    assert "Lung-RADS" in systems
    assert None in systems  # at least one report with no stated category


def test_no_category_inferred_where_gold_has_none():
    # The read-not-computed guarantee: every gold row with no category must be
    # parsed as having no category.
    rows = _load_gold()
    for row in rows:
        if row["gold"]["assessment_system"] is None:
            parsed = parse_radiology_report(row["text"])
            assert parsed["assessment_system"] is None
            assert parsed["assessment_category"] is None


def test_advisory_available_for_eval_reports():
    assert "read" in RADIOLOGY_REPORT_ADVISORY.lower()
