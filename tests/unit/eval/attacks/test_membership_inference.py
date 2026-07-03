"""Tests for the membership-inference re-identification probe (issue #511)."""

from __future__ import annotations

import json

from openmed.eval.attacks import (
    membership_inference_attack,
    run_reid_attack,
    run_reid_benchmark,
    shadow_membership_inference_attack,
)


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

    def test_metric_serialization_is_aggregate_only(self):
        candidates = [
            {"record_id": "r1", "text": "Alice Zsoltay visited clinic"},
            {"record_id": "r2", "text": "Bob Smith visited clinic"},
        ]
        deidentified = [
            {"record_id": "r1", "text": "[NAME] Zsoltay visited clinic"},
            {"record_id": "r2", "text": "[NAME] [NAME] visited clinic"},
        ]
        metric = membership_inference_attack(deidentified, candidates).to_metric()

        raw = json.dumps(metric, sort_keys=True)
        assert "per_record" not in metric
        assert "Zsoltay" not in raw
        assert "Alice" not in raw
        assert all(item.startswith("sha256:") for item in metric["record_hashes"])


class TestShadowMembershipInference:
    def test_trained_attacker_beats_random_on_leaky_scores(self):
        members, heldout = _shadow_records()

        result = shadow_membership_inference_attack(members, heldout)

        assert result.attacker_auc > 0.9
        assert result.attacker_advantage > 0.5
        assert result.per_label["PERSON"]["attacker_advantage"] > 0.5

    def test_defense_reduces_advantage_below_ceiling(self):
        members, heldout = _shadow_records()

        result = shadow_membership_inference_attack(
            members,
            heldout,
            defense_policy={
                "enabled": True,
                "clip_min": 0.5,
                "clip_max": 0.5,
                "advantage_ceiling": 0.05,
            },
        )

        assert result.attacker_auc == 0.5
        assert result.attacker_advantage <= result.advantage_ceiling

    def test_shadow_metric_contains_no_raw_phi(self):
        members, heldout = _shadow_records()

        metric = shadow_membership_inference_attack(
            members,
            heldout,
        ).to_metric()

        raw = json.dumps(metric, sort_keys=True)
        assert "Alice" not in raw
        assert "555-0101" not in raw
        assert "text" not in raw
        assert metric["feature_hash"].startswith("sha256:")

    def test_defended_deidentified_outputs_have_zero_residual_leakage(self):
        fixture = {
            "id": "synthetic-1",
            "language": "en",
            "text": "Patient Alice Example called 555-0101.",
            "gold_spans": [
                {"start": 8, "end": 21, "label": "PERSON"},
                {"start": 29, "end": 37, "label": "PHONE"},
            ],
            "metadata": {
                "synthetic": True,
                "category": "checksum_ids",
                "expected_output": {
                    "method": "mask",
                    "text": "Patient [NAME] called [PHONE].",
                },
            },
        }
        result = run_reid_attack(
            [fixture],
            deidentified_records=[
                {
                    "record_id": "synthetic-1",
                    "text": "Patient [NAME] called [PHONE].",
                    "metadata": {"membership_defense": {"enabled": True}},
                }
            ],
        )

        assert result.to_metric()["leakage_rate"] == 0.0


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
        assert "per_record" not in report.metrics["membership_inference"]

    def test_shadow_membership_metric_appears_in_report(self):
        members, heldout = _shadow_records()
        report = run_reid_benchmark(
            shadow_member_records=members,
            shadow_heldout_records=heldout,
        )

        assert "membership_leakage" in report.metrics
        assert "attacker_auc" in report.metrics["membership_leakage"]
        assert "per_label" in report.metrics["membership_leakage"]

    def test_membership_inference_absent_without_candidates(self):
        report = run_reid_benchmark(
            deidentified_records=[{"record_id": "r1", "text": "[NAME] visited"}],
        )
        assert "membership_inference" not in report.metrics


def _shadow_records():
    members = [
        {
            "record_id": f"m{i}",
            "text": f"Patient Alice Example {i} called 555-0101.",
            "entities": [
                {"label": "PERSON", "score": 0.97 - (i * 0.005)},
                {"label": "PHONE", "score": 0.96 - (i * 0.005)},
            ],
        }
        for i in range(6)
    ]
    heldout = [
        {
            "record_id": f"h{i}",
            "text": f"Patient Casey Example {i} called 555-0199.",
            "entities": [
                {"label": "PERSON", "score": 0.56 + (i * 0.005)},
                {"label": "PHONE", "score": 0.55 + (i * 0.005)},
            ],
        }
        for i in range(6)
    ]
    return members, heldout
