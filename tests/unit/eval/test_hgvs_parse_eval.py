"""Offline eval gate for HGVS variant-descriptor parsing.

Scores the deterministic parser against the committed synthetic gold set and
enforces the acceptance thresholds: field-level accuracy >= 0.95 across
reference_sequence / coordinate_type / position / edit / status, and byte-stable
character offsets. Runs fully offline with no models or network.
"""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import GENOMICS_ADVISORY, parse_hgvs
from openmed.eval.metrics import HGVS_FIELDS, hgvs_field_accuracy

GOLD = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "hgvs_parse.jsonl"
)

FIELD_FLOOR = 0.95


def _load_gold() -> list[dict]:
    with GOLD.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _predicted_and_gold_mentions(rows):
    # Align per row first (so a false-positive extra or a missed descriptor in one
    # row cannot silently offset the flattened comparison against another row's
    # gold), then flatten the verified-aligned pairs.
    predicted: list = []
    gold: list = []
    for row in rows:
        row_predicted = parse_hgvs(row["text"])
        assert len(row_predicted) == len(row["gold"]), row["text"]
        predicted.extend(row_predicted)
        gold.extend(row["gold"])
    return predicted, gold


def test_gold_set_is_present_and_synthetic():
    rows = _load_gold()
    assert len(rows) >= 12
    assert all(row["metadata"]["synthetic"] is True for row in rows)


def test_per_field_accuracy_meets_floor():
    rows = _load_gold()
    predicted, gold = _predicted_and_gold_mentions(rows)
    assert len(predicted) == len(gold) >= 12  # non-vacuous, per-row aligned

    per_field = hgvs_field_accuracy(predicted, gold)
    assert set(per_field) == set(HGVS_FIELDS)
    for field, metric in per_field.items():
        assert metric.rate >= FIELD_FLOOR, (
            f"{field} accuracy {metric.rate:.3f} < {FIELD_FLOOR}"
        )


def test_character_offsets_are_byte_stable():
    # Every predicted span must index the source text back to raw_text exactly.
    rows = _load_gold()
    checked = 0
    for row in rows:
        for mention in parse_hgvs(row["text"]):
            start, end = mention["span"]
            assert row["text"][start:end] == mention["raw_text"], row["text"]
            checked += 1
    assert checked >= 12


def test_gold_covers_all_coordinate_types_and_statuses():
    rows = _load_gold()
    mentions = [mention for row in rows for mention in row["gold"]]
    assert {m["coordinate_type"] for m in mentions} == set("cpgmnr")
    assert {m["status"] for m in mentions} == {"valid", "malformed", "unparseable"}


def test_malformed_descriptors_carry_reason_codes_and_are_not_coerced():
    # Never-coerce guarantee, enforced directly: every gold malformed/unparseable
    # mention must be reproduced with the same status and a reason code.
    rows = _load_gold()
    checked = 0
    for row in rows:
        predicted = parse_hgvs(row["text"])
        for mention, gold in zip(predicted, row["gold"]):
            if gold["status"] in ("malformed", "unparseable"):
                assert mention["status"] == gold["status"], row["text"]
                assert mention["reason"], row["text"]
                checked += 1
    assert checked >= 3


def test_advisory_emitted_on_every_mention():
    rows = _load_gold()
    mentions = [mention for row in rows for mention in parse_hgvs(row["text"])]
    assert mentions
    assert all(mention["advisory"] == GENOMICS_ADVISORY for mention in mentions)
    assert "syntactic" in GENOMICS_ADVISORY.lower()
