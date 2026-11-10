"""Tests for the India DPDP policy-coverage evaluation and per-label report."""

from __future__ import annotations

import json

import pytest

from openmed.core.labels import DIRECT_IDENTIFIER, PERSON, policy_label_for
from openmed.core.policy import load_policy
from openmed.core.schemas.span import ACTION_KEEP
from openmed.eval import (
    INDIA_DPDP_COVERAGE,
    INDIA_DPDP_POLICY,
    assert_india_dpdp_coverage_gate,
    india_dpdp_coverage_metadata,
    run_india_dpdp_coverage,
)
from openmed.eval.datasets import load_india_clinical_phi_corpus

_EXPECTED_DIRECT_IDENTIFIER_LABELS = ("ID_NUM", "PERSON", "PHONE", "STREET_ADDRESS")


def test_coverage_report_is_deterministic_and_covers_every_direct_label() -> None:
    report = run_india_dpdp_coverage()

    assert report == run_india_dpdp_coverage()
    assert report.policy == INDIA_DPDP_POLICY
    assert report.to_dict()["suite"] == INDIA_DPDP_COVERAGE

    corpus = load_india_clinical_phi_corpus()
    assert report.document_count == len(corpus.records)
    assert report.span_count == sum(len(record.gold_spans) for record in corpus.records)

    assert report.direct_identifier_labels() == _EXPECTED_DIRECT_IDENTIFIER_LABELS
    for coverage in report.per_label:
        assert coverage.policy_label == policy_label_for(coverage.label)
        assert coverage.total > 0
        assert 0.0 <= coverage.coverage_rate <= 1.0
        assert sum(coverage.action_counts.values()) == coverage.total
        if coverage.direct_identifier:
            assert coverage.policy_label == DIRECT_IDENTIFIER


def test_direct_identifiers_reach_full_replace_coverage_with_zero_keep() -> None:
    report = run_india_dpdp_coverage()

    assert report.passed is True
    assert report.failures == ()
    assert report.direct_identifier_span_count > 0
    assert (
        report.covered_direct_identifier_span_count
        == report.direct_identifier_span_count
    )
    assert report.direct_identifier_coverage_rate == 1.0
    assert report.keep_action_count == 0

    for coverage in report.per_label:
        if coverage.direct_identifier:
            assert coverage.action_counts == {"replace": coverage.total}
            assert coverage.covered == coverage.total


def test_assert_gate_returns_passing_report() -> None:
    report = assert_india_dpdp_coverage_gate()

    assert report.passed is True


def test_report_contains_no_raw_fixture_phi_values() -> None:
    report = run_india_dpdp_coverage()
    serialized = json.dumps(report.to_dict(), ensure_ascii=False)

    corpus = load_india_clinical_phi_corpus()
    raw_surfaces = {
        span.text for record in corpus.records for span in record.gold_spans
    }
    assert raw_surfaces
    for surface in raw_surfaces:
        assert surface not in serialized


def test_keep_regression_fails_gate_with_label_and_document_id() -> None:
    base = load_policy(INDIA_DPDP_POLICY)
    leaking = base.derive(actions={**dict(base.actions), PERSON: ACTION_KEEP})

    report = run_india_dpdp_coverage(policy=leaking)

    assert report.passed is False
    assert report.keep_action_count > 0
    person_failures = [
        failure for failure in report.failures if failure.label == PERSON
    ]
    assert person_failures
    assert all(
        failure.reason == "direct_identifier_keep_action" for failure in person_failures
    )
    assert all(failure.action == ACTION_KEEP for failure in person_failures)
    corpus = load_india_clinical_phi_corpus()
    expected_documents = {
        record.document_id
        for record in corpus.records
        for span in record.gold_spans
        if span.label == PERSON
    }
    assert {failure.document_id for failure in person_failures} == expected_documents

    for failure in person_failures:
        evidence = failure.evidence
        assert set(evidence) >= {"start", "end", "length", "text_hash"}
        assert str(evidence["text_hash"]).startswith("sha256:")

    with pytest.raises(AssertionError, match="India DPDP policy-coverage gate failed"):
        assert_india_dpdp_coverage_gate(policy=leaking)


def test_metadata_is_raw_text_free_and_declares_requirements() -> None:
    metadata = india_dpdp_coverage_metadata()

    assert metadata == {
        "suite": INDIA_DPDP_COVERAGE,
        "policy": INDIA_DPDP_POLICY,
        "synthetic": True,
        "required_direct_identifier_coverage": 1.0,
        "required_keep_actions": 0,
    }
