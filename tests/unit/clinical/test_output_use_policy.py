"""Synthetic offline tests for the clinical-output use-policy gate."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.output_use_policy import (
    DEFAULT_OUTPUT_USE_POLICY,
    DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
    OutputUseDeclaration,
    OutputUsePolicy,
    OutputUsePolicyError,
    OutputUseReasonCode,
    OutputUseRule,
    check_output_use_policy,
    enforce_output_use,
    evaluate_output_use,
)


def _reviewable_summary() -> OutputUseDeclaration:
    return OutputUseDeclaration(
        category="summary",
        purpose="documentation",
        audience="clinician",
        review_state="reviewed",
        decision_triggering=False,
        policy_fingerprint=DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
    )


def test_default_policy_allows_reviewed_synthetic_documentation() -> None:
    decision = evaluate_output_use(
        _reviewable_summary(),
        policy=DEFAULT_OUTPUT_USE_POLICY,
    )

    assert decision.allowed is True
    assert decision.reason_codes == ()
    assert decision.policy_fingerprint == DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT
    assert decision.to_safe_dict() == {
        "schema_version": 1,
        "allowed": True,
        "reason_codes": [],
        "policy_fingerprint": DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
    }


def test_keyword_and_mapping_forms_are_equivalent() -> None:
    declaration = _reviewable_summary()
    mapping = {
        "category": declaration.category,
        "declared_purpose": declaration.purpose,
        "intended_audience": declaration.audience,
        "review": declaration.review_state,
        "decision_trigger": declaration.decision_triggering,
        "policy_digest": declaration.policy_fingerprint,
    }

    assert check_output_use_policy(
        mapping,
        policy=DEFAULT_OUTPUT_USE_POLICY,
    ) == evaluate_output_use(
        category="summary",
        purpose="documentation",
        audience="clinician",
        review_state="reviewed",
        decision_triggering=False,
        policy_fingerprint=DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
        policy=DEFAULT_OUTPUT_USE_POLICY,
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("category", OutputUseReasonCode.CATEGORY_UNDECLARED.value),
        ("purpose", OutputUseReasonCode.PURPOSE_UNDECLARED.value),
        ("audience", OutputUseReasonCode.AUDIENCE_UNDECLARED.value),
        ("review_state", OutputUseReasonCode.REVIEW_STATE_UNDECLARED.value),
        (
            "decision_triggering",
            OutputUseReasonCode.DECISION_TRIGGERING_UNDECLARED.value,
        ),
        (
            "policy_fingerprint",
            OutputUseReasonCode.POLICY_FINGERPRINT_UNDECLARED.value,
        ),
    ],
)
def test_undeclared_metadata_fails_closed_without_echoing_values(
    field: str,
    expected: str,
) -> None:
    declaration = _reviewable_summary()
    object.__setattr__(declaration, field, None)

    decision = evaluate_output_use(declaration, policy=DEFAULT_OUTPUT_USE_POLICY)

    assert decision.allowed is False
    assert expected in decision.reason_codes
    assert "synthetic" not in json.dumps(decision.to_dict())


def test_incompatible_category_purpose_and_audience_fail_closed() -> None:
    decision = evaluate_output_use(
        OutputUseDeclaration(
            category="summary",
            purpose="research",
            audience="clinician",
            review_state="approved",
            decision_triggering=False,
            policy_fingerprint=DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
        ),
        policy=DEFAULT_OUTPUT_USE_POLICY,
    )

    assert decision.allowed is False
    assert decision.reason_codes == (OutputUseReasonCode.INCOMPATIBLE_USE.value,)


def test_decision_triggering_use_is_denied_even_when_other_fields_match() -> None:
    declaration = _reviewable_summary()
    object.__setattr__(declaration, "decision_triggering", True)

    decision = evaluate_output_use(declaration, policy=DEFAULT_OUTPUT_USE_POLICY)

    assert decision.allowed is False
    assert decision.reason_codes == (OutputUseReasonCode.DECISION_TRIGGERING_USE.value,)


@pytest.mark.parametrize(
    ("review_state", "expected"),
    [
        ("draft", OutputUseReasonCode.REVIEW_STATE_INSUFFICIENT.value),
        ("pending_review", OutputUseReasonCode.REVIEW_STATE_INSUFFICIENT.value),
        ("rejected", OutputUseReasonCode.REVIEW_REJECTED.value),
    ],
)
def test_release_requires_the_rule_review_state(
    review_state: str,
    expected: str,
) -> None:
    declaration = _reviewable_summary()
    object.__setattr__(declaration, "review_state", review_state)

    decision = evaluate_output_use(declaration, policy=DEFAULT_OUTPUT_USE_POLICY)

    assert decision.allowed is False
    assert expected in decision.reason_codes


def test_policy_fingerprint_is_order_independent_and_binds_rules() -> None:
    rules = list(DEFAULT_OUTPUT_USE_POLICY.rules)
    reordered = OutputUsePolicy(
        name="synthetic-policy",
        rules=tuple(reversed(rules)),
    )
    same_order = OutputUsePolicy(name="synthetic-policy", rules=tuple(rules))
    changed = OutputUsePolicy(
        name="synthetic-policy",
        rules=tuple(
            rules[:-1]
            + [
                OutputUseRule(
                    "summary",
                    "research",
                    "researcher",
                    "reviewed",
                )
            ]
        ),
    )

    assert reordered.fingerprint == same_order.fingerprint
    assert changed.fingerprint != same_order.fingerprint


def test_mismatched_fingerprint_and_ambiguous_input_fail_closed() -> None:
    mismatched_declaration = _reviewable_summary()
    object.__setattr__(
        mismatched_declaration,
        "policy_fingerprint",
        "sha256:" + "0" * 64,
    )
    mismatch = evaluate_output_use(
        mismatched_declaration,
        policy=DEFAULT_OUTPUT_USE_POLICY,
    )
    ambiguous = evaluate_output_use(
        _reviewable_summary(),
        category="summary",
        policy=DEFAULT_OUTPUT_USE_POLICY,
    )

    assert mismatch.reason_codes == (
        OutputUseReasonCode.POLICY_FINGERPRINT_MISMATCH.value,
    )
    assert ambiguous.reason_codes == (OutputUseReasonCode.DECLARATION_AMBIGUOUS.value,)


def test_safe_serialization_never_echoes_unknown_declaration_values() -> None:
    declaration = OutputUseDeclaration(
        category="synthetic-summary",
        purpose="synthetic-purpose",
        audience="synthetic-audience",
        review_state="synthetic-review",
        decision_triggering=False,
        policy_fingerprint="not-a-fingerprint",
    )

    safe_payload = declaration.to_safe_dict()

    assert safe_payload == {
        "category": None,
        "purpose": None,
        "audience": None,
        "review_state": None,
        "decision_triggering": False,
        "policy_fingerprint": None,
    }
    assert "synthetic" not in repr(declaration)


def test_enforce_raises_with_reason_codes_only() -> None:
    declaration = _reviewable_summary()
    object.__setattr__(declaration, "purpose", "unregistered-purpose")

    with pytest.raises(OutputUsePolicyError) as error:
        enforce_output_use(declaration, policy=DEFAULT_OUTPUT_USE_POLICY)

    assert error.value.reason_codes == (OutputUseReasonCode.PURPOSE_UNSUPPORTED.value,)
    assert "unregistered-purpose" not in str(error.value)
