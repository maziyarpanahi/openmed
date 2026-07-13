"""Unit tests for policy-profile compliance evals."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.core.labels import (
    DIRECT_IDENTIFIER,
    HIPAA_SAFE_HARBOR_CLASSES,
    ID_NUM,
    PERSON,
    policy_label_for,
)
from openmed.core.policy import list_policies, load_policy
from openmed.core.schemas.span import ACTION_KEEP
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites import (
    POLICY_COMPLIANCE,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.eval.suites.policy_compliance import (
    BUNDLED_DEIDENTIFICATION_POLICIES,
    BUNDLED_POLICY_LABEL_ACTION_REQUIREMENTS,
    DPDP_DIRECT_IDENTIFIER_METADATA_KEY,
    DPDP_DIRECT_IDENTIFIER_TYPES,
    derive_profile_expectations,
    evaluate_profile_compliance,
    load_policy_compliance_fixtures,
    run_policy_compliance,
)

CHINA_PIPL_FIXTURE_ID = "policy-compliance-china-pipl-direct-identifiers"


def test_policy_compliance_fixtures_are_synthetic_and_cover_safe_harbor() -> None:
    fixtures = load_policy_compliance_fixtures()

    assert fixtures
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    safe_harbor_classes = {
        str(span.metadata["hipaa_safe_harbor_class"])
        for fixture in fixtures
        for span in fixture.gold_spans
        if "hipaa_safe_harbor_class" in span.metadata
    }
    assert safe_harbor_classes == HIPAA_SAFE_HARBOR_CLASSES
    dpdp_identifier_types = {
        str(span.metadata[DPDP_DIRECT_IDENTIFIER_METADATA_KEY])
        for fixture in fixtures
        for span in fixture.gold_spans
        if DPDP_DIRECT_IDENTIFIER_METADATA_KEY in span.metadata
    }
    assert dpdp_identifier_types == DPDP_DIRECT_IDENTIFIER_TYPES

    for fixture in fixtures:
        for span in fixture.gold_spans:
            assert fixture.text[span.start : span.end] == span.text


def test_policy_compliance_registry_loads_fixtures_and_metadata() -> None:
    fixtures = load_suite_fixtures(POLICY_COMPLIANCE)
    metadata = suite_metadata(POLICY_COMPLIANCE)

    assert fixtures
    assert metadata["suite"] == POLICY_COMPLIANCE
    assert metadata["profiles"] == list(BUNDLED_DEIDENTIFICATION_POLICIES)
    assert metadata["synthetic"] is True


def test_run_policy_compliance_reports_every_bundled_profile() -> None:
    report = run_policy_compliance(generated_at="2026-06-27T00:00:00Z")
    payload = json.loads(report.to_json())
    profiles = payload["metrics"]["profiles"]

    assert isinstance(report, BenchmarkReport)
    assert payload["suite"] == POLICY_COMPLIANCE
    assert payload["generated_at"] == "2026-06-27T00:00:00Z"
    assert payload["metrics"]["overall_passed"] is True
    assert payload["metrics"]["profile_count"] == len(list_policies())
    assert set(profiles) == set(BUNDLED_DEIDENTIFICATION_POLICIES)

    for profile_name in BUNDLED_DEIDENTIFICATION_POLICIES:
        result = profiles[profile_name]
        assert result["passed"] is True
        assert result["residual_direct_identifier_count"] == 0
        assert result["fixture_count"] == report.fixture_count

    strict = profiles["strict_no_leak"]
    assert strict["residual_direct_identifier_count"] == 0
    assert strict["action_counts"]["keep"] == 0

    safe_harbor = profiles["hipaa_safe_harbor"]
    assert set(safe_harbor["covered_safe_harbor_classes"]) == (
        HIPAA_SAFE_HARBOR_CLASSES
    )
    assert safe_harbor["missing_safe_harbor_classes"] == []

    india_dpdp = profiles["india_dpdp_act"]
    assert set(india_dpdp["covered_dpdp_direct_identifier_types"]) == (
        DPDP_DIRECT_IDENTIFIER_TYPES
    )
    assert india_dpdp["missing_dpdp_direct_identifier_types"] == []
    assert india_dpdp["residual_direct_identifier_count"] == 0


def test_india_dpdp_fixture_replaces_every_required_direct_identifier() -> None:
    profile = load_policy("india_dpdp_act")
    fixtures = load_policy_compliance_fixtures()
    india_fixture = next(
        fixture
        for fixture in fixtures
        if fixture.fixture_id == "policy-compliance-india-dpdp-identifiers"
    )
    expectations = derive_profile_expectations(profile)

    direct_spans = [
        span
        for span in india_fixture.gold_spans
        if DPDP_DIRECT_IDENTIFIER_METADATA_KEY in span.metadata
    ]
    assert {
        str(span.metadata[DPDP_DIRECT_IDENTIFIER_METADATA_KEY]) for span in direct_spans
    } == DPDP_DIRECT_IDENTIFIER_TYPES
    assert all(
        policy_label_for(span.label) == DIRECT_IDENTIFIER for span in direct_spans
    )
    assert all(expectations[span.label].action == "replace" for span in direct_spans)

    result = evaluate_profile_compliance(profile, fixtures)

    assert result.passed is True
    assert result.residual_direct_identifier_count == 0
    assert set(result.covered_dpdp_direct_identifier_types) == (
        DPDP_DIRECT_IDENTIFIER_TYPES
    )
    assert result.missing_dpdp_direct_identifier_types == ()


def test_india_dpdp_gate_rejects_non_replace_national_identifiers() -> None:
    profile = load_policy("india_dpdp_act")
    weakened = replace(profile, actions={**profile.actions, "ID_NUM": "mask"})

    result = evaluate_profile_compliance(
        weakened,
        load_policy_compliance_fixtures(),
    )

    assert result.passed is False
    assert {
        failure.reason
        for failure in result.failures
        if failure.reason.startswith("dpdp_direct_identifier_not_replaced:")
    } == {
        "dpdp_direct_identifier_not_replaced:AADHAAR",
        "dpdp_direct_identifier_not_replaced:ABHA",
        "dpdp_direct_identifier_not_replaced:PAN",
    }


def test_china_pipl_fixture_has_zero_direct_identifier_leakage() -> None:
    profile = load_policy("china_pipl")
    fixture = next(
        fixture
        for fixture in load_policy_compliance_fixtures()
        if fixture.fixture_id == CHINA_PIPL_FIXTURE_ID
    )

    expectations = derive_profile_expectations(profile)
    result = evaluate_profile_compliance(profile, [fixture])

    assert "china_pipl" in BUNDLED_DEIDENTIFICATION_POLICIES
    assert fixture.language == "zh"
    assert fixture.metadata["content_language"] == "zh"
    assert result.passed is True
    assert result.residual_direct_identifier_count == 0
    assert result.action_counts["replace"] == len(fixture.gold_spans)
    assert all(
        expectations[span.label].action == profile.action_for(span.label)
        for span in fixture.gold_spans
    )


def test_china_pipl_compliance_rejects_weakened_direct_identifier_action() -> None:
    profile = load_policy("china_pipl")
    weakened = replace(profile, actions={**profile.actions, ID_NUM: "mask"})

    result = evaluate_profile_compliance(
        weakened,
        load_policy_compliance_fixtures(),
    )

    assert (
        BUNDLED_POLICY_LABEL_ACTION_REQUIREMENTS["china_pipl"][policy_label_for(ID_NUM)]
        == "replace"
    )
    assert result.passed is False
    assert any(
        failure.label == ID_NUM
        and failure.reason == "profile_action_requirement_mismatch"
        for failure in result.failures
    )


def test_weakened_safe_harbor_profile_reports_cleartext_failure() -> None:
    profile = load_policy("hipaa_safe_harbor")
    weakened = replace(profile, actions={**profile.actions, PERSON: ACTION_KEEP})

    result = evaluate_profile_compliance(
        weakened,
        load_policy_compliance_fixtures(),
    )

    assert result.passed is False
    assert result.residual_direct_identifier_count >= 1
    assert any(
        failure.label == PERSON and failure.reason == "direct_identifier_residual"
        for failure in result.failures
    )
    assert any(
        failure.label == PERSON
        and failure.reason == "safe_harbor_action_keeps_identifier"
        for failure in result.failures
    )


def test_expectations_are_derived_from_profile_actions() -> None:
    profile = load_policy("hipaa_safe_harbor")
    changed = replace(profile, actions={**profile.actions, PERSON: "replace"})

    expectations = derive_profile_expectations(changed)

    assert expectations[PERSON].action == "replace"
    assert expectations[PERSON].source == (
        "openmed.core.policy.PolicyProfile.action_for"
    )


def test_invalid_profile_action_name_fails_closed() -> None:
    profile = load_policy("hipaa_safe_harbor")
    renamed = replace(profile, actions={**profile.actions, PERSON: "remove"})

    with pytest.raises(ValueError, match="unsupported policy action"):
        derive_profile_expectations(renamed)
