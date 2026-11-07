"""Tests for deterministic aggregate-only retraining trigger decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.eval.drift_monitor import (
    DRIFT_TRIGGER_SCHEMA_VERSION,
    VERDICT_DRIFT,
    VERDICT_STABLE,
    DriftPrivacyError,
    DriftTriggerSignal,
)
from openmed.eval.error_analysis import (
    RETRAINING_SLICE_SCHEMA_VERSION,
    RetrainingSliceArtifact,
    RetrainingSlicePriority,
)
from openmed.eval.history import REGRESSION, BenchmarkHistoryDiff, MetricDelta
from openmed.eval.release_gates import GateCheck, GateReport
from openmed.eval.retrain_trigger import (
    RETRAIN_QUEUE_SCHEMA_VERSION,
    RETRAIN_TRIGGER_EVIDENCE_SCHEMA_VERSION,
    RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION,
    SIGNAL_DRIFT,
    SIGNAL_GATE_FAILURE,
    SIGNAL_NONE,
    SIGNAL_STALENESS,
    RetrainCandidateSignals,
    RetrainRecipeSlice,
    RetrainTriggerInputError,
    RetrainTriggerPolicy,
    candidate_signals_from_artifacts,
    compute_candidate_fleet_freshness,
    parse_retrain_trigger_inputs,
    score_retrain_candidates,
    write_retrain_trigger_artifacts,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _gate_report(*checks: GateCheck, family: str = "pii") -> GateReport:
    return GateReport(
        repo_id="OpenMed/unit-model",
        family=family,
        tier="tiny",
        param_count=44_000_000,
        format="mlx-fp",
        per_label_recall={"SSN": 0.99},
        per_label_precision={"SSN": 0.98},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=50.0,
        p95_ms=100.0,
        ram_mb=128.0,
        eval_set_hash=HASH_A,
        leakage_fixture_hash=HASH_B,
        decision="hold" if checks else "releasable",
        gate_results=checks,
    )


def _drift_signal(*, divergence: float, threshold: float = 0.25) -> DriftTriggerSignal:
    detected = divergence >= threshold
    return DriftTriggerSignal(
        schema_version=DRIFT_TRIGGER_SCHEMA_VERSION,
        drift_detected=detected,
        verdict=VERDICT_DRIFT if detected else VERDICT_STABLE,
        max_divergence=divergence,
        threshold=threshold,
        warning_threshold=0.1,
        per_family_divergence={"label_rate": divergence},
        dominant_family="label_rate" if detected else None,
        dominant_drifting_label="SSN" if detected else None,
        reference_window_id="reference-v1",
        observation_window_id="observation-v1",
        generated_at="2026-09-05T00:00:00+00:00",
    )


def _slice_artifact() -> RetrainingSliceArtifact:
    return RetrainingSliceArtifact(
        source_queue_hash=HASH_A,
        schema_version=RETRAINING_SLICE_SCHEMA_VERSION,
        slices=(
            RetrainingSlicePriority(
                rank=1,
                label="SSN",
                language="en",
                candidate_count=3,
                priority_score=4.5,
                mean_priority=1.5,
                mean_uncertainty=0.7,
                max_gate_impact=2.0,
                candidate_hashes=(HASH_A,),
                fixture_hashes=(HASH_B,),
            ),
        ),
    )


def test_g3_g7_failures_queue_only_the_affected_family() -> None:
    report = _gate_report(
        GateCheck(
            "G3",
            False,
            reason="critical leakage",
            details={"critical_leakage_count": 2},
        ),
        GateCheck(
            "G7",
            False,
            reason="baseline regression",
            details={
                "violations": {"recall_drop": {"SSN": {"drop": 0.004, "limit": 0.002}}}
            },
        ),
    )
    affected = candidate_signals_from_artifacts(
        "pii",
        "tiny",
        gate_report=report,
    )
    unaffected = RetrainCandidateSignals(family="structured", tier="tiny")

    result = score_retrain_candidates([unaffected, affected])

    assert [(item.candidate.family, item.queued) for item in result.decisions] == [
        ("pii", True),
        ("structured", False),
    ]
    decision = result.decisions[0]
    assert decision.dominant_signal == SIGNAL_GATE_FAILURE
    assert decision.candidate.failed_gates == ("G3", "G7")
    assert decision.candidate.dominant_label == "SSN"
    assert decision.candidate.gate_failure_ratio == 2.0
    assert json.loads(result.queue_jsonl())["schema_version"] == (
        RETRAIN_QUEUE_SCHEMA_VERSION
    )


def test_history_regression_normalizes_against_g7_boundary() -> None:
    delta = MetricDelta(
        metric="per_label_recall.SSN",
        baseline=0.999,
        current=0.995,
        delta=-0.004,
        relative_delta=-0.004004,
        direction="higher_is_better",
        verdict=REGRESSION,
    )
    history = BenchmarkHistoryDiff(
        metrics={delta.metric: delta},
        largest_regressions=(delta,),
    )

    candidate = candidate_signals_from_artifacts(
        "pii",
        "tiny",
        history_diff=history,
    )

    assert candidate.gate_failure_ratio == pytest.approx(2.0)
    assert candidate.failed_gates == ("G7",)
    assert score_retrain_candidates([candidate]).queued


def test_drift_and_configurable_weights_produce_auditable_score() -> None:
    candidate = candidate_signals_from_artifacts(
        "pii",
        "tiny",
        drift_signal=_drift_signal(divergence=0.375),
    )
    policy = RetrainTriggerPolicy(
        threshold=0.75,
        drift_weight=0.5,
        gate_failure_weight=0.0,
        staleness_weight=0.0,
    )

    decision = score_retrain_candidates([candidate], policy=policy).decisions[0]

    assert candidate.drift_ratio == 1.5
    assert decision.score == 0.75
    assert decision.queued
    assert decision.dominant_signal == SIGNAL_DRIFT
    assert decision.weighted_components[SIGNAL_DRIFT] == 0.75


def test_fleet_freshness_is_isolated_by_family_and_tier() -> None:
    metrics = compute_candidate_fleet_freshness(
        [
            {
                "repo_id": "OpenMed/pii-tiny",
                "family": "pii",
                "tier": "tiny",
                "released": "2026-07-23",
                "languages": ["en"],
            },
            {
                "repo_id": "OpenMed/other-old-model",
                "family": "pii",
                "tier": "small",
                "released": "2020-01-01",
                "languages": ["en"],
            },
        ],
        family="pii",
        tier="tiny",
        as_of="2026-09-01",
        median_age_target_days=30,
    )
    candidate = candidate_signals_from_artifacts(
        "pii",
        "tiny",
        fleet_freshness=metrics,
    )
    decision = score_retrain_candidates([candidate]).decisions[0]

    assert metrics.total_model_count == 1
    assert metrics.median_age_days == 40.0
    assert candidate.staleness_ratio == pytest.approx(4 / 3)
    assert decision.queued
    assert decision.dominant_signal == SIGNAL_STALENESS


def test_recipe_slices_are_context_free_and_canonical() -> None:
    candidate = candidate_signals_from_artifacts(
        "pii",
        "tiny",
        retraining_slices=_slice_artifact(),
    )
    payload = candidate.to_dict()

    assert payload["recipe_slices"] == [
        {
            "candidate_count": 3,
            "evidence_hash": payload["recipe_slices"][0]["evidence_hash"],
            "fixture_hashes": [HASH_B],
            "label": "SSN",
            "language": "en",
            "priority_score": 4.5,
            "rank": 1,
        }
    ]
    serialized = json.dumps(payload, sort_keys=True)
    assert "context" not in serialized
    assert "candidate_hashes" not in serialized
    assert set(candidate.source_hashes) == {"active_learning"}


def test_reversed_inputs_produce_byte_identical_artifacts() -> None:
    first_slice = RetrainRecipeSlice(
        rank=1,
        label="SSN",
        language="en",
        candidate_count=2,
        priority_score=3.0,
        evidence_hash=HASH_A,
        fixture_hashes=(HASH_A, HASH_B),
    )
    reversed_slice = RetrainRecipeSlice(
        rank=1,
        label="SSN",
        language="en",
        candidate_count=2,
        priority_score=3.0,
        evidence_hash=HASH_A,
        fixture_hashes=(HASH_B, HASH_A),
    )
    candidate_a = RetrainCandidateSignals(
        family="pii",
        tier="tiny",
        drift_ratio=1.1,
        source_hashes={"drift": HASH_A, "fleet": HASH_B},
        recipe_slices=(first_slice,),
    )
    candidate_a_reversed = RetrainCandidateSignals(
        family="pii",
        tier="tiny",
        drift_ratio=1.1,
        source_hashes={"fleet": HASH_B, "drift": HASH_A},
        recipe_slices=(reversed_slice,),
    )
    candidate_b = RetrainCandidateSignals(family="structured", tier="small")

    first = score_retrain_candidates([candidate_a, candidate_b])
    second = score_retrain_candidates([candidate_b, candidate_a_reversed])

    assert first.queue_jsonl() == second.queue_jsonl()
    assert first.evidence_json() == second.evidence_json()
    assert (
        first.to_evidence_dict()["artifact_hash"]
        == (second.to_evidence_dict()["artifact_hash"])
    )


def test_strict_input_round_trip_and_raw_text_rejection() -> None:
    candidate = RetrainCandidateSignals(
        family="pii",
        tier="tiny",
        gate_failure_ratio=1.0,
        failed_gates=("G3",),
        source_hashes={"gate": HASH_A},
    )
    document: dict[str, Any] = {
        "schema_version": RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION,
        "candidates": [candidate.to_dict()],
    }

    assert parse_retrain_trigger_inputs(document) == (candidate,)

    document["candidates"][0]["raw_text"] = "synthetic patient context"
    with pytest.raises(DriftPrivacyError, match="raw-text"):
        parse_retrain_trigger_inputs(document)


def test_write_artifacts_records_nonqueued_decisions_deterministically(
    tmp_path: Path,
) -> None:
    result = score_retrain_candidates(
        [RetrainCandidateSignals(family="pii", tier="tiny")]
    )
    queue_path = tmp_path / "nested" / "retrain_queue.jsonl"
    evidence_path = tmp_path / "nested" / "decision_evidence.json"

    write_retrain_trigger_artifacts(
        result,
        queue_path=queue_path,
        evidence_path=evidence_path,
    )
    first_queue = queue_path.read_bytes()
    first_evidence = evidence_path.read_bytes()
    write_retrain_trigger_artifacts(
        result,
        queue_path=queue_path,
        evidence_path=evidence_path,
    )

    assert first_queue == b""
    assert queue_path.read_bytes() == first_queue
    assert evidence_path.read_bytes() == first_evidence
    evidence = json.loads(first_evidence)
    assert evidence["schema_version"] == RETRAIN_TRIGGER_EVIDENCE_SCHEMA_VERSION
    assert evidence["queue_count"] == 0
    assert evidence["decisions"][0]["dominant_signal"] == SIGNAL_NONE
    assert evidence["decisions"][0]["queued"] is False


def test_invalid_aggregate_inputs_fail_closed() -> None:
    with pytest.raises(RetrainTriggerInputError, match="family"):
        RetrainCandidateSignals(family="patient name", tier="tiny")
    with pytest.raises(RetrainTriggerInputError, match="finite"):
        RetrainCandidateSignals(family="pii", tier="tiny", drift_ratio=float("nan"))
    with pytest.raises(RetrainTriggerInputError, match="contiguous"):
        RetrainCandidateSignals(
            family="pii",
            tier="tiny",
            recipe_slices=(
                RetrainRecipeSlice(
                    rank=2,
                    label="SSN",
                    language="en",
                    candidate_count=1,
                    priority_score=1.0,
                    evidence_hash=HASH_A,
                ),
            ),
        )
    with pytest.raises(RetrainTriggerInputError, match="identities"):
        score_retrain_candidates(
            [
                RetrainCandidateSignals(family="pii", tier="tiny"),
                RetrainCandidateSignals(family="pii", tier="tiny"),
            ]
        )
    with pytest.raises(RetrainTriggerInputError, match="family"):
        candidate_signals_from_artifacts(
            "pii",
            "tiny",
            gate_report=_gate_report(family="structured"),
        )
