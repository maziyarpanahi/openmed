"""Offline eval gate for OncoTree tumor-type mapping accuracy.

Scores the deterministic mapper against the committed synthetic gold set.
The stub mapper is designed to hit every row, so CI requires perfect top-1
code accuracy. Runs fully offline with the synthetic stub release (exact /
normalized lookup only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import ONCOTREE_ADVISORY, load_oncotree, map_tumor_type
from openmed.eval.metrics import oncotree_top1_accuracy

ROOT = Path(__file__).resolve().parents[3]
GOLD = ROOT / "openmed" / "eval" / "golden" / "fixtures" / "oncotree_map.jsonl"
STUB = ROOT / "tests" / "fixtures" / "clinical" / "oncotree_stub.json"

VERSION = "synthetic-oncotree-1"
MIN_GOLD_ROWS = 40
TOP1_FLOOR = 0.90


def _load_gold() -> list[dict]:
    with GOLD.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_gold_set_is_present_and_synthetic():
    rows = _load_gold()
    assert len(rows) >= MIN_GOLD_ROWS
    assert all(row["metadata"]["synthetic"] is True for row in rows)


def test_top1_accuracy_meets_floor():
    release = load_oncotree(STUB, version=VERSION)
    rows = _load_gold()
    predicted = [map_tumor_type(row["mention"], release) for row in rows]
    gold = [row["gold"] for row in rows]
    assert len(predicted) == len(gold) >= MIN_GOLD_ROWS
    metric = oncotree_top1_accuracy(predicted, gold)
    assert metric.rate >= TOP1_FLOOR, (
        f"top-1 OncoTree accuracy {metric.rate:.3f} < {TOP1_FLOOR}"
    )


def test_top1_metric_penalizes_wrong_codes_and_rejects_length_mismatch():
    predicted = [{"code": "SYN_A"}, {"code": "SYN_WRONG"}]
    gold = [{"code": "SYN_A"}, {"code": "SYN_B"}]
    metric = oncotree_top1_accuracy(predicted, gold)
    assert metric.rate == 0.5
    assert metric.numerator == 1
    assert metric.denominator == 2

    with pytest.raises(ValueError):
        oncotree_top1_accuracy(predicted[:1], gold)


def test_provenance_and_advisory_on_every_mapping():
    release = load_oncotree(STUB, version=VERSION)
    for row in _load_gold():
        mapped = map_tumor_type(row["mention"], release)
        assert mapped["oncotree_version"] == VERSION
        assert mapped["advisory"] == ONCOTREE_ADVISORY


def test_unmapped_rows_carry_reasons():
    release = load_oncotree(STUB, version=VERSION)
    checked = 0
    reasons: set[str] = set()
    for row in _load_gold():
        if row["gold"]["code"] is not None:
            continue
        mapped = map_tumor_type(row["mention"], release)
        assert mapped["code"] is None
        assert mapped["reason"] == row["gold"]["reason"]
        assert mapped["match_confidence"] == 0.0
        assert mapped["oncotree_version"] == VERSION
        reasons.add(mapped["reason"])
        checked += 1
    assert checked >= 8
    assert {"ambiguous", "no_match", "empty_mention"} <= reasons
