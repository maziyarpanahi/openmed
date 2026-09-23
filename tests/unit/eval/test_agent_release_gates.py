"""Tests for quantitative v3.1 workflow release gates."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.eval.suites.agent_release import (
    APPROVAL_BYPASS,
    CLINICIAN_REVIEW_AGREEMENT,
    DEFAULT_AGENT_RELEASE_GATES,
    NOT_READY,
    READY,
    UNAUTHORIZED_ACTION_ESCAPE,
    AgentReleaseGateError,
    MetricEvidence,
    evaluate_agent_release_gates,
)

_FIXTURE_PATH = Path("tests/fixtures/eval/agent_release.py")


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "agent_release_fixtures", _FIXTURE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(report, metric):
    return next(item for item in report.gate_results if item.metric == metric)


def test_synthetic_baseline_is_ready_and_byte_deterministic() -> None:
    fixtures = _fixtures()

    first = fixtures.passing_report()
    second = evaluate_agent_release_gates(
        tuple(reversed(fixtures.passing_evidence())),
        candidate_digest=fixtures.CANDIDATE_DIGEST,
    )

    assert first.decision == READY
    assert all(result.passed for result in first.gate_results)
    assert first.to_json() == second.to_json()
    assert first.report_digest == second.report_digest
    assert json.loads(first.to_json()) == first.to_dict()


def test_every_published_gate_is_independently_non_compensable() -> None:
    fixtures = _fixtures()
    passing = fixtures.passing_evidence()

    for index, spec in enumerate(DEFAULT_AGENT_RELEASE_GATES):
        evidence = list(passing)
        original = evidence[index]
        if spec.direction == "maximum":
            failing_count = (
                min(original.sample_size, max(1, original.event_count or 0))
                if spec.evidence_basis == "exact_count"
                else original.event_count
            )
            evidence[index] = replace(
                original,
                value=(
                    failing_count / original.sample_size
                    if failing_count is not None
                    else spec.threshold + 1.0
                ),
                event_count=failing_count,
                ci_lower=(
                    spec.threshold + 0.5
                    if spec.evidence_basis == "confidence_interval"
                    else original.ci_lower
                ),
                ci_upper=(
                    spec.threshold + 1.5
                    if spec.evidence_basis == "confidence_interval"
                    else original.ci_upper
                ),
            )
        else:
            failing_count = (
                int((spec.threshold - 0.01) * original.sample_size)
                if spec.evidence_basis == "exact_count"
                else original.event_count
            )
            evidence[index] = replace(
                original,
                value=(
                    failing_count / original.sample_size
                    if spec.evidence_basis == "exact_count"
                    and failing_count is not None
                    else spec.threshold - 0.01
                ),
                event_count=failing_count,
                ci_lower=(
                    spec.threshold - 0.02
                    if spec.evidence_basis == "confidence_interval"
                    else original.ci_lower
                ),
                ci_upper=(
                    spec.threshold
                    if spec.evidence_basis == "confidence_interval"
                    else original.ci_upper
                ),
            )

        report = evaluate_agent_release_gates(
            evidence, candidate_digest=fixtures.CANDIDATE_DIGEST
        )

        assert report.decision == NOT_READY
        assert report.failing_gates() == (_result(report, spec.metric),)
        assert report.failing_gates()[0].reason_code == "threshold_failed"


def test_critical_violation_cannot_be_offset_by_other_metrics() -> None:
    fixtures = _fixtures()

    report = fixtures.critical_failure_report()

    assert report.decision == NOT_READY
    failure = _result(report, UNAUTHORIZED_ACTION_ESCAPE)
    assert failure.critical is True
    assert failure.event_count == 1
    assert failure.reason_code == "threshold_failed"
    assert all(
        result.passed
        for result in report.gate_results
        if result.metric != UNAUTHORIZED_ACTION_ESCAPE
    )


def test_missing_metric_fails_closed_without_shifting_gate_order() -> None:
    fixtures = _fixtures()
    evidence = tuple(
        item for item in fixtures.passing_evidence() if item.metric != APPROVAL_BYPASS
    )

    report = evaluate_agent_release_gates(
        evidence, candidate_digest=fixtures.CANDIDATE_DIGEST
    )

    assert report.decision == NOT_READY
    assert tuple(item.metric for item in report.gate_results) == tuple(
        item.metric for item in DEFAULT_AGENT_RELEASE_GATES
    )
    failure = _result(report, APPROVAL_BYPASS)
    assert failure.reason_code == "missing_evidence"
    assert failure.sample_size == 0


def test_confidence_interval_uses_conservative_bound() -> None:
    fixtures = _fixtures()
    evidence = list(fixtures.passing_evidence())
    index = next(
        index
        for index, item in enumerate(evidence)
        if item.metric == CLINICIAN_REVIEW_AGREEMENT
    )
    evidence[index] = replace(evidence[index], value=0.90, ci_lower=0.79, ci_upper=0.95)

    report = evaluate_agent_release_gates(
        evidence, candidate_digest=fixtures.CANDIDATE_DIGEST
    )

    result = _result(report, CLINICIAN_REVIEW_AGREEMENT)
    assert result.passed is False
    assert result.observed_value == 0.90
    assert result.evaluated_bound == 0.79


def test_reports_publish_slice_sizes_limitations_counts_and_intervals() -> None:
    fixtures = _fixtures()
    report = evaluate_agent_release_gates(
        fixtures.passing_evidence(), candidate_digest=fixtures.CANDIDATE_DIGEST
    )

    count_result = _result(report, UNAUTHORIZED_ACTION_ESCAPE).to_dict()
    interval_result = _result(report, CLINICIAN_REVIEW_AGREEMENT).to_dict()

    assert count_result["event_count"] == 0
    assert count_result["sample_size"] == 1_000
    assert count_result["slice_sizes"] == [
        {"sample_size": 500, "slice_ref": "read_actions"},
        {"sample_size": 500, "slice_ref": "write_actions"},
    ]
    assert count_result["limitations"] == ["synthetic_policy_matrix"]
    assert interval_result["ci_lower"] == 0.84
    assert interval_result["ci_upper"] == 0.94


def test_evidence_schema_rejects_free_text_and_missing_statistical_basis() -> None:
    fixtures = _fixtures()
    original = fixtures.passing_evidence()[0]

    with pytest.raises(AgentReleaseGateError, match="invalid_reference"):
        replace(original, limitations=("Patient Jane Doe was included",))
    with pytest.raises(AgentReleaseGateError, match="invalid_sequence"):
        replace(original, limitations="synthetic_policy_matrix")
    with pytest.raises(AgentReleaseGateError, match="count_or_interval_required"):
        replace(original, event_count=None)
    with pytest.raises(AgentReleaseGateError, match="rate_count_mismatch"):
        replace(original, value=0.5)


def test_duplicate_and_unknown_metrics_are_rejected() -> None:
    fixtures = _fixtures()
    evidence = fixtures.passing_evidence()

    with pytest.raises(AgentReleaseGateError, match="duplicate_metric"):
        evaluate_agent_release_gates(
            (*evidence, evidence[0]), candidate_digest=fixtures.CANDIDATE_DIGEST
        )
    with pytest.raises(AgentReleaseGateError, match="unexpected_metric"):
        evaluate_agent_release_gates(
            (replace(evidence[0], metric="aggregate_score"), *evidence[1:]),
            candidate_digest=fixtures.CANDIDATE_DIGEST,
        )


def test_report_contains_no_raw_workflow_or_reviewer_fields() -> None:
    fixtures = _fixtures()
    report = evaluate_agent_release_gates(
        fixtures.passing_evidence(), candidate_digest=fixtures.CANDIDATE_DIGEST
    )
    rendered = report.to_json()

    for forbidden in (
        "prompt",
        "tool_argument",
        "clinical_output",
        "credential",
        "reviewer_identity",
        "patient",
    ):
        assert forbidden not in rendered.lower()


def test_evidence_round_trip_preserves_closed_schema() -> None:
    fixtures = _fixtures()
    original = fixtures.passing_evidence()[0]

    restored = MetricEvidence.from_dict(original.to_dict())

    assert restored == original


def test_report_rejects_an_incomplete_or_changed_gate_contract() -> None:
    fixtures = _fixtures()
    report = fixtures.passing_report()

    with pytest.raises(AgentReleaseGateError, match="invalid_gate_set"):
        replace(report, gate_results=report.gate_results[:-1])
    with pytest.raises(AgentReleaseGateError, match="gate_contract_mismatch"):
        replace(
            report,
            gate_results=(
                replace(report.gate_results[0], threshold=1.0),
                *report.gate_results[1:],
            ),
        )
