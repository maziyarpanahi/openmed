"""Tests for per-label reliability and calibration curves (issue #935)."""

from __future__ import annotations

import json

import pytest

from openmed.eval.per_label_calibration import per_label_calibration_report


def _sample(label, score, target):
    return {"model_id": "m", "label": label, "score": score, "target": target}


class TestPerLabelEce:
    def test_ece_matches_reference_computation(self):
        # One bin ([0.8,0.9)) with mean confidence 0.8 and accuracy 0.5.
        samples = [
            _sample("PERSON", 0.8, True),
            _sample("PERSON", 0.8, False),
        ]
        report = per_label_calibration_report(samples)
        assert report["labels"]["PERSON"]["ece"] == pytest.approx(0.3)
        assert report["labels"]["PERSON"]["sample_count"] == 2

    def test_perfectly_calibrated_is_near_zero(self):
        samples = [_sample("EMAIL", 1.0, True) for _ in range(5)]
        samples += [_sample("EMAIL", 0.0, False) for _ in range(5)]
        report = per_label_calibration_report(samples)
        assert report["labels"]["EMAIL"]["ece"] == pytest.approx(0.0, abs=1e-9)

    def test_labels_scored_independently(self):
        samples = [
            _sample("PERSON", 0.8, True),
            _sample("PERSON", 0.8, False),
            _sample("EMAIL", 1.0, True),
            _sample("EMAIL", 0.0, False),
        ]
        report = per_label_calibration_report(samples)
        assert set(report["labels"]) == {"PERSON", "EMAIL"}
        assert report["labels"]["EMAIL"]["ece"] == pytest.approx(0.0, abs=1e-9)
        assert report["labels"]["PERSON"]["ece"] == pytest.approx(0.3)


class TestReliabilityCurve:
    def test_bins_carry_confidence_and_accuracy(self):
        samples = [
            _sample("PERSON", 0.85, True),
            _sample("PERSON", 0.85, False),
        ]
        report = per_label_calibration_report(samples, num_bins=10)
        curve = report["labels"]["PERSON"]["reliability"]
        assert len(curve) == 10
        populated = [b for b in curve if b["count"] > 0]
        assert len(populated) == 1
        assert populated[0]["mean_confidence"] == pytest.approx(0.85)
        assert populated[0]["accuracy"] == pytest.approx(0.5)


class TestBudgetFlagging:
    def test_labels_over_budget_are_flagged(self):
        samples = [
            _sample("PERSON", 0.8, True),
            _sample("PERSON", 0.8, False),
            _sample("EMAIL", 1.0, True),
            _sample("EMAIL", 0.0, False),
        ]
        report = per_label_calibration_report(samples, ece_budget=0.1)
        assert report["labels"]["PERSON"]["over_budget"] is True
        assert report["labels"]["EMAIL"]["over_budget"] is False
        assert report["flagged_labels"] == ["PERSON"]

    def test_no_budget_flags_nothing(self):
        report = per_label_calibration_report([_sample("PERSON", 0.8, True)])
        assert report["flagged_labels"] == []
        assert report["labels"]["PERSON"]["over_budget"] is False


class TestContract:
    def test_deterministic_and_json_serializable(self):
        samples = [
            _sample("PERSON", 0.8, True),
            _sample("PERSON", 0.3, False),
            _sample("EMAIL", 0.9, True),
        ]
        first = per_label_calibration_report(samples, ece_budget=0.2)
        second = per_label_calibration_report(samples, ece_budget=0.2)
        assert first == second
        assert json.loads(json.dumps(first)) == first

    def test_output_has_no_raw_text(self):
        report = per_label_calibration_report([_sample("PERSON", 0.8, True)])
        blob = json.dumps(report)
        # Only labels and numbers are emitted; no free-text sample content.
        assert "text" not in report
        assert isinstance(blob, str)
