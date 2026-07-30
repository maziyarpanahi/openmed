"""Tests for the India clinical residual-PHI leakage gate."""

from __future__ import annotations

import pytest

from openmed.core.policy import load_policy
from openmed.eval.datasets import load_india_clinical_phi_corpus
from openmed.eval.suites import (
    INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES,
    INDIA_CLINICAL_PHI_LEAKAGE,
    IndiaClinicalLeakageResult,
    assert_india_clinical_leakage_gate,
    deidentify_india_clinical_record,
    evaluate_india_clinical_leakage,
    india_clinical_leakage_metadata,
    run_india_clinical_leakage_gate,
)


def _raw_direct_identifier_values() -> set[str]:
    corpus = load_india_clinical_phi_corpus()
    return {
        span.text
        for record in corpus.records
        for span in record.gold_spans
        if span.metadata.get("direct_identifier") is True
    }


def test_shipped_corpus_produces_zero_surviving_direct_identifiers() -> None:
    result = run_india_clinical_leakage_gate()

    assert isinstance(result, IndiaClinicalLeakageResult)
    assert result.passed is True
    assert result.residual_leak_count == 0
    assert result.findings == ()
    assert result.document_count == 3
    assert result.direct_identifier_count > 0
    assert result.policy == "india_dpdp_act"

    covered_types = {verdict.identifier_type for verdict in result.label_verdicts}
    assert covered_types == set(INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES)
    assert all(verdict.passed for verdict in result.label_verdicts)


def test_assert_helper_returns_passing_result_on_clean_corpus() -> None:
    result = assert_india_clinical_leakage_gate()

    assert result.passed is True
    assert result.to_dict()["suite"] == INDIA_CLINICAL_PHI_LEAKAGE


def test_gate_fails_closed_when_one_identifier_is_preserved() -> None:
    corpus = load_india_clinical_phi_corpus()
    policy = load_policy("india_dpdp_act")

    # Locate one synthetic direct identifier to intentionally preserve.
    target_record = corpus.records[0]
    preserved = next(
        span
        for span in target_record.gold_spans
        if span.metadata.get("direct_identifier") is True
    )
    preserved_id = str(preserved.metadata["span_id"])

    def leaky_deidentify(record: object) -> str:
        preserve = (
            frozenset({preserved_id})
            if record.document_id == target_record.document_id
            else frozenset()
        )
        return deidentify_india_clinical_record(
            record, policy=policy, preserve=preserve
        )

    result = evaluate_india_clinical_leakage(
        corpus.records, deidentify=leaky_deidentify
    )

    assert result.passed is False
    assert result.residual_leak_count == 1
    assert len(result.findings) == 1

    finding = result.findings[0]
    assert finding.document_id == target_record.document_id
    assert finding.identifier_type == preserved.metadata["identifier_type"]
    assert finding.start == preserved.start
    assert finding.end == preserved.end

    leaked_verdicts = [v for v in result.label_verdicts if not v.passed]
    assert [v.identifier_type for v in leaked_verdicts] == [
        preserved.metadata["identifier_type"]
    ]


def test_failure_evidence_never_reproduces_the_raw_value() -> None:
    corpus = load_india_clinical_phi_corpus()
    policy = load_policy("india_dpdp_act")
    raw_values = _raw_direct_identifier_values()

    result = evaluate_india_clinical_leakage(
        corpus.records,
        deidentify=lambda record: deidentify_india_clinical_record(
            record,
            policy=policy,
            preserve=frozenset(
                str(span.metadata["span_id"])
                for span in record.gold_spans
                if span.metadata.get("direct_identifier") is True
            ),
        ),
    )

    assert result.passed is False
    assert result.residual_leak_count == result.direct_identifier_count

    serialized = str(result.to_dict())
    for value in raw_values:
        assert value not in serialized
    for finding in result.findings:
        assert finding.span_hash.startswith("hmac-sha256:")


def test_assert_helper_raises_without_leaking_raw_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.eval.suites import india_clinical_leakage as module

    corpus = load_india_clinical_phi_corpus()
    policy = load_policy("india_dpdp_act")
    raw_values = _raw_direct_identifier_values()

    def _leak_everything(**_kwargs: object) -> IndiaClinicalLeakageResult:
        return evaluate_india_clinical_leakage(
            corpus.records,
            deidentify=lambda record: record.text,
            policy_name="india_dpdp_act",
        )

    monkeypatch.setattr(module, "run_india_clinical_leakage_gate", _leak_everything)

    with pytest.raises(AssertionError) as excinfo:
        module.assert_india_clinical_leakage_gate()

    message = str(excinfo.value)
    assert "leakage gate failed" in message
    for value in raw_values:
        assert value not in message

    # Guard against an accidentally vacuous policy check.
    assert policy.name == "india_dpdp_act"


def test_metadata_is_offline_and_raw_text_free() -> None:
    metadata = india_clinical_leakage_metadata()

    assert metadata["suite"] == INDIA_CLINICAL_PHI_LEAKAGE
    assert metadata["synthetic"] is True
    assert metadata["required_residual_leakage"] == 0
    assert metadata["policy"] == "india_dpdp_act"
    assert set(metadata["direct_identifier_types"]) == set(
        INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES
    )
