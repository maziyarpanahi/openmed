"""Tests for k-anonymity / l-diversity / t-closeness measurement (issue #500)."""

from __future__ import annotations

import json
import math

import pytest

from openmed.risk import kanon_report


def _classes_by_size(report):
    return sorted(c["size"] for c in report["equivalence_classes"])


class TestKAnonymity:
    RECORDS = [
        {"age": 30, "zip": "1000", "disease": "flu"},
        {"age": 30, "zip": "1000", "disease": "cold"},
        {"age": 41, "zip": "2000", "disease": "flu"},  # singleton
    ]

    def test_k_min_and_singleton_class(self):
        report = kanon_report(
            self.RECORDS,
            quasi_identifiers=["age", "zip"],
            sensitive_attributes=["disease"],
        )
        assert report["k"] == 1
        assert report["class_count"] == 2
        assert _classes_by_size(report) == [1, 2]
        singletons = [c for c in report["equivalence_classes"] if c["size"] == 1]
        assert len(singletons) == 1

    def test_higher_k_when_all_classes_share_key(self):
        records = [
            {"age": 30, "zip": "1000", "disease": "flu"},
            {"age": 30, "zip": "1000", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["age", "zip"], sensitive_attributes=["disease"]
        )
        assert report["k"] == 2
        assert report["class_count"] == 1


class TestLDiversity:
    def test_distinct_is_one_when_each_class_has_single_value(self):
        records = [
            {"age": 30, "zip": "1000", "disease": "flu"},
            {"age": 30, "zip": "1000", "disease": "flu"},
            {"age": 41, "zip": "2000", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["age", "zip"], sensitive_attributes=["disease"]
        )
        for cls in report["equivalence_classes"]:
            assert cls["l_diversity"]["disease"]["distinct"] == 1
            assert cls["l_diversity"]["disease"]["entropy"] == 0.0
        assert report["l_diversity"]["disease"]["min_distinct"] == 1

    def test_distinct_counts_multiple_sensitive_values(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["g"], sensitive_attributes=["disease"]
        )
        cls = report["equivalence_classes"][0]
        assert cls["l_diversity"]["disease"]["distinct"] == 2
        assert cls["l_diversity"]["disease"]["entropy"] == pytest.approx(1.0)


class TestTCloseness:
    def test_class_matching_global_distribution_is_zero(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "cold"},
            {"g": "B", "disease": "flu"},
            {"g": "B", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["g"], sensitive_attributes=["disease"]
        )
        for cls in report["equivalence_classes"]:
            assert cls["t_closeness"]["disease"] == pytest.approx(0.0, abs=1e-9)
        assert report["t_closeness"]["disease"] == pytest.approx(0.0, abs=1e-9)

    def test_skewed_class_has_positive_distance(self):
        records = [
            {"g": "A", "disease": "flu"},
            {"g": "A", "disease": "flu"},
            {"g": "B", "disease": "cold"},
            {"g": "B", "disease": "cold"},
        ]
        report = kanon_report(
            records, quasi_identifiers=["g"], sensitive_attributes=["disease"]
        )
        # Global is 50/50; each class is 100% one value -> TV distance 0.5.
        assert report["t_closeness"]["disease"] == pytest.approx(0.5)


class TestContract:
    RECORDS = [
        {"age": 30, "zip": "1000", "disease": "flu"},
        {"age": 30, "zip": "1000", "disease": "cold"},
        {"age": 41, "zip": "2000", "disease": "flu"},
    ]

    def test_deterministic_and_json_serializable(self):
        kwargs = dict(
            quasi_identifiers=["age", "zip"], sensitive_attributes=["disease"]
        )
        first = kanon_report(self.RECORDS, **kwargs)
        second = kanon_report(self.RECORDS, **kwargs)
        assert first == second
        assert json.loads(json.dumps(first)) == first

    def test_importable_and_exported(self):
        import openmed.risk as risk

        assert hasattr(risk, "kanon_report")
        assert "kanon_report" in risk.__all__

    def test_no_sensitive_attributes_reports_k_only(self):
        report = kanon_report(self.RECORDS, quasi_identifiers=["age", "zip"])
        assert report["k"] == 1
        assert report["l_diversity"] == {}
        assert report["t_closeness"] == {}

    def test_auto_quasi_identifier_detection_runs(self):
        # Without explicit QIs, fall back to risk_report-consistent profiling.
        records = [
            {"text": "SSN 123-45-6789"},
            {"text": "SSN 987-65-4321"},
        ]
        report = kanon_report(records, sensitive_attributes=None)
        assert report["k"] >= 1
        assert json.loads(json.dumps(report)) == report
