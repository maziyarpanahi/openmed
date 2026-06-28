"""Tests for the membership-inference re-identification probe (issue #511)."""

from __future__ import annotations

from openmed.eval.attacks import membership_inference_attack, run_reid_benchmark


class TestMembershipInference:
    def test_no_residual_signal_yields_zero_advantage(self):
        candidates = [
            {"record_id": "r1", "text": "Alice Smith visited clinic"},
            {"record_id": "r2", "text": "Bob Smith visited clinic"},
        ]
        # Fully redacted: residual tokens are shared/common, not distinguishing.
        deidentified = [
            {"record_id": "r1", "text": "[NAME] visited clinic"},
            {"record_id": "r2", "text": "[NAME] visited clinic"},
        ]
        result = membership_inference_attack(deidentified, candidates)
        assert result.advantage == 0.0
        assert all(row["matched_candidate"] is None for row in result.per_record)

    def test_unique_residual_token_identifies_candidate(self):
        candidates = [
            {"record_id": "r1", "text": "Alice Zsoltay visited clinic"},
            {"record_id": "r2", "text": "Bob Smith visited clinic"},
        ]
        deidentified = [
            {"record_id": "r1", "text": "[NAME] Zsoltay visited clinic"},  # leaks
            {"record_id": "r2", "text": "[NAME] [NAME] visited clinic"},
        ]
        result = membership_inference_attack(deidentified, candidates)
        assert result.advantage > 0.0
        matched = {
            row["record_id"]: row["matched_candidate"] for row in result.per_record
        }
        assert matched["r1"] == "r1"
        assert matched["r2"] is None

    def test_deterministic_for_fixed_input(self):
        candidates = [
            {"record_id": "r1", "text": "Alice Zsoltay visited clinic"},
            {"record_id": "r2", "text": "Bob Smith visited clinic"},
        ]
        deidentified = [
            {"record_id": "r1", "text": "[NAME] Zsoltay visited clinic"},
            {"record_id": "r2", "text": "[NAME] [NAME] visited clinic"},
        ]
        first = membership_inference_attack(deidentified, candidates)
        second = membership_inference_attack(deidentified, candidates)
        assert first.to_metric() == second.to_metric()

    def test_advantage_is_accuracy_above_half(self):
        candidates = [
            {"record_id": "r1", "text": "Alice Zsoltay visited clinic"},
            {"record_id": "r2", "text": "Bob Smith visited clinic"},
        ]
        deidentified = [
            {"record_id": "r1", "text": "[NAME] Zsoltay visited clinic"},
            {"record_id": "r2", "text": "[NAME] [NAME] visited clinic"},
        ]
        result = membership_inference_attack(deidentified, candidates)
        assert result.advantage == result.accuracy - 0.5
        assert result.baseline == 0.5


class TestBenchmarkWiring:
    def test_membership_inference_metric_appears_in_report(self):
        candidates = [
            {"record_id": "r1", "text": "Alice Zsoltay visited clinic"},
            {"record_id": "r2", "text": "Bob Smith visited clinic"},
        ]
        deidentified = [
            {"record_id": "r1", "text": "[NAME] Zsoltay visited clinic"},
            {"record_id": "r2", "text": "[NAME] [NAME] visited clinic"},
        ]
        report = run_reid_benchmark(
            deidentified_records=deidentified,
            candidate_members=candidates,
        )
        assert "membership_inference" in report.metrics
        assert "advantage" in report.metrics["membership_inference"]

    def test_membership_inference_absent_without_candidates(self):
        report = run_reid_benchmark(
            deidentified_records=[{"record_id": "r1", "text": "[NAME] visited"}],
        )
        assert "membership_inference" not in report.metrics
