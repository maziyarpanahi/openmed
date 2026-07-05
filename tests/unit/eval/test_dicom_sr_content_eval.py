"""Offline eval gate for DICOM SR content-tree extraction accuracy.

Runs fully offline over a committed synthetic TID1500-style SR gold set and the
in-memory synthetic SR object. No network and no real study data are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.golden import load_golden_fixtures
from openmed.eval.metrics import SrContentAccuracy, compute_sr_content_accuracy

_GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "dicom_sr_content.jsonl"
)


def _load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in _GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _meta(rows: list[dict]) -> dict:
    return next(row for row in rows if row.get("kind") == "meta")


def _fixtures(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("kind") == "fixture"]


def test_gold_fixture_is_synthetic_and_well_formed():
    rows = _load_rows()
    meta = _meta(rows)
    fixtures = _fixtures(rows)

    assert meta["synthetic"] is True
    assert meta["suite"] == "dicom_sr_content"
    assert meta["min_accuracy"] >= 0.95
    assert fixtures, "gold set must contain at least one SR fixture"

    for fixture in fixtures:
        assert fixture["synthetic"] is True
        items = fixture["content_items"]
        assert items, "fixture must have content items"
        # node_paths are unique and the root is present.
        paths = [item["node_path"] for item in items]
        assert len(set(paths)) == len(paths)
        assert "1" in set(paths)
        for item in items:
            assert set(item) >= {
                "concept_name",
                "value_type",
                "value",
                "unit_code",
                "relationship",
                "template_id",
                "node_path",
            }
            if item["value_type"] == "NUM":
                assert item["value"]
                assert item["unit_code"]


def test_gold_fixture_excluded_from_deid_golden_loader():
    # The SR content fixture is not a de-identification golden fixture; the
    # golden loader must skip it rather than fail validating it.
    fixtures = load_golden_fixtures()
    assert all(
        fixture.metadata.get("suite") != "dicom_sr_content" for fixture in fixtures
    )


def test_content_accuracy_is_perfect_on_gold_when_prediction_matches():
    rows = _load_rows()
    for fixture in _fixtures(rows):
        gold_items = fixture["content_items"]
        accuracy = compute_sr_content_accuracy(gold_items, gold_items)
        assert isinstance(accuracy, SrContentAccuracy)
        assert accuracy.accuracy == 1.0
        assert accuracy.matched == accuracy.total == len(gold_items)
        assert accuracy.missing_nodes == ()
        assert accuracy.mismatched_nodes == ()
        assert accuracy.extra_nodes == ()


def test_content_accuracy_detects_missing_and_mismatched_nodes():
    rows = _load_rows()
    gold_items = _fixtures(rows)[0]["content_items"]

    # Drop one node and corrupt another to prove the metric is discriminative.
    predicted = [dict(item) for item in gold_items[1:]]
    predicted[2] = dict(predicted[2], value="WRONG")

    accuracy = compute_sr_content_accuracy(predicted, gold_items)
    assert accuracy.total == len(gold_items)
    assert accuracy.matched < accuracy.total
    assert accuracy.accuracy < 0.95
    assert gold_items[0]["node_path"] in accuracy.missing_nodes


def test_extractor_meets_offline_accuracy_gate(tmp_path: Path):
    pytest.importorskip("pydicom")
    from openmed.multimodal import extract_dicom_sr
    from tests.fixtures.multimodal.dicom_sr_synthetic import write_synthetic_sr

    rows = _load_rows()
    meta = _meta(rows)
    gold_items = _fixtures(rows)[0]["content_items"]

    source = write_synthetic_sr(tmp_path / "sr.dcm")
    document = extract_dicom_sr(source, policy={"date_shift_days": 7})

    accuracy = compute_sr_content_accuracy(
        document.metadata["content_items"], gold_items
    )
    assert accuracy.accuracy >= meta["min_accuracy"], accuracy.to_dict()
    assert accuracy.missing_nodes == ()
    assert accuracy.mismatched_nodes == ()
