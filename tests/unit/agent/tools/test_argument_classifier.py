"""Offline tests for grant-aware tool-argument classification."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent.permissions.grants import (
    CapabilityGrantConstraint,
    CapabilityGrantRequest,
    CapabilityGrantRequiredError,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
)
from openmed.agent.tools.argument_classifier import (
    ARGUMENT_CLASSIFICATION_REPORT_SCHEMA_VERSION,
    ArgumentAction,
    ArgumentClassificationCoverageError,
    ArgumentClassificationPolicy,
    ArgumentClassifier,
    ArgumentClassifierValidationError,
    ArgumentDataClassDecision,
    ArgumentDispatchBlockedError,
    ArgumentPathRule,
    dispatch_with_argument_classification,
)

KEY = b"local-classification-test-key-32-bytes"
EXPIRES_AT = 2_000_000_000
POLICY = "policy:org.example/minimum-necessary@1.0.0"
PUBLIC = "data:org.example/non-sensitive@1.0.0"
CLINICAL = "data:org.example/clinical-text@1.0.0"
CREDENTIAL = "data:org.example/credential@1.0.0"


def _policy(
    *,
    public: ArgumentAction = ArgumentAction.ALLOW,
    clinical: ArgumentAction = ArgumentAction.ALLOW,
    credential: ArgumentAction = ArgumentAction.ALLOW,
) -> ArgumentClassificationPolicy:
    return ArgumentClassificationPolicy(
        policy_profile=POLICY,
        decisions=(
            ArgumentDataClassDecision(PUBLIC, public),
            ArgumentDataClassDecision(
                CLINICAL,
                clinical,
                replacement="synthetic-redaction"
                if clinical is ArgumentAction.REDACT
                else None,
            ),
            ArgumentDataClassDecision(CREDENTIAL, credential),
        ),
    )


def _classifier(
    rules: tuple[ArgumentPathRule, ...],
    *,
    policy: ArgumentClassificationPolicy | None = None,
) -> ArgumentClassifier:
    return ArgumentClassifier(rules, policy or _policy(), hash_key=KEY)


def _grant() -> tuple[Any, CapabilityGrantRequest, CapabilityGrantVerifier]:
    constraint = CapabilityGrantConstraint(
        tool="tool:org.example/summarize@1.0.0",
        resource="resource:org.example/clinical-document@1.0.0",
        action="action:org.example/read@1.0.0",
        policy_profile=POLICY,
    )
    manifest = CapabilityGrantSigner(KEY).issue([constraint], expires_at=EXPIRES_AT)
    request = CapabilityGrantRequest(**constraint.to_dict())
    return manifest, request, CapabilityGrantVerifier(KEY)


def test_nested_arguments_are_classified_without_flattening_schema() -> None:
    arguments = {
        "patient": {"summary": "synthetic-clinical-value", "age": 42},
        "observations": [
            {"value": 7.5, "unit": "synthetic-unit"},
            {"value": 8.0, "unit": "synthetic-unit"},
        ],
        "flags": (True, False),
    }
    classifier = _classifier(
        (
            ArgumentPathRule(("patient", "summary"), CLINICAL),
            ArgumentPathRule(("patient", "age"), CLINICAL),
            ArgumentPathRule(("observations", "*", "value"), CLINICAL),
            ArgumentPathRule(("observations", "*", "unit"), PUBLIC),
            ArgumentPathRule(("flags", "*"), PUBLIC),
        )
    )

    first = classifier.classify(arguments)
    second = classifier.classify(arguments)

    assert first.arguments == arguments
    assert type(first.arguments["observations"]) is list
    assert type(first.arguments["flags"]) is tuple
    assert first.report.to_json() == second.report.to_json()
    assert [item.path for item in first.report.allowed] == [
        "/flags/0",
        "/flags/1",
        "/observations/0/unit",
        "/observations/0/value",
        "/observations/1/unit",
        "/observations/1/value",
        "/patient/age",
        "/patient/summary",
    ]
    assert not first.report.redacted
    assert not first.report.blocked
    observation_hashes = {
        item.value_hash for item in first.report.allowed if item.path.endswith("/unit")
    }
    assert len(observation_hashes) == 2


def test_redaction_uses_a_copy_and_dispatches_only_sanitized_arguments() -> None:
    arguments = {
        "summary": "synthetic-clinical-value",
        "mode": "synthetic-mode",
    }
    classifier = _classifier(
        (
            ArgumentPathRule(("summary",), CLINICAL),
            ArgumentPathRule(("mode",), PUBLIC),
        ),
        policy=_policy(clinical=ArgumentAction.REDACT),
    )
    manifest, request, verifier = _grant()
    received: list[dict[str, Any]] = []

    result = dispatch_with_argument_classification(
        manifest,
        request,
        verifier,
        classifier,
        arguments,
        lambda safe: received.append(safe) or "done",
        now=EXPIRES_AT - 1,
    )

    assert result.value == "done"
    assert received == [{"summary": "synthetic-redaction", "mode": "synthetic-mode"}]
    assert arguments["summary"] == "synthetic-clinical-value"
    assert [item.path for item in result.report.redacted] == ["/summary"]
    assert [item.path for item in result.report.allowed] == ["/mode"]


def test_blocked_value_prevents_dispatch_and_exposes_only_safe_evidence() -> None:
    rejected = "synthetic-secret-must-not-escape"
    classifier = _classifier(
        (ArgumentPathRule(("credential",), CREDENTIAL),),
        policy=_policy(credential=ArgumentAction.BLOCK),
    )
    manifest, request, verifier = _grant()
    calls = 0

    def dispatch(_: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(ArgumentDispatchBlockedError) as caught:
        dispatch_with_argument_classification(
            manifest,
            request,
            verifier,
            classifier,
            {"credential": rejected},
            dispatch,
            now=EXPIRES_AT - 1,
        )

    report_json = caught.value.report.to_json()
    assert calls == 0
    assert rejected not in report_json
    assert rejected not in str(caught.value)
    assert rejected not in repr(caught.value)
    assert json.loads(report_json) == {
        "schema_version": ARGUMENT_CLASSIFICATION_REPORT_SCHEMA_VERSION,
        "allowed": [],
        "redacted": [],
        "blocked": [
            {
                "path": "/credential",
                "data_class": CREDENTIAL,
                "value_hash": caught.value.report.blocked[0].value_hash,
            }
        ],
    }
    assert caught.value.report.blocked[0].value_hash.startswith("hmac-sha256:")


def test_grant_and_policy_profile_are_checked_before_dispatch() -> None:
    classifier = _classifier((ArgumentPathRule(("mode",), PUBLIC),))
    manifest, request, verifier = _grant()
    calls = 0

    def dispatch(_: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(CapabilityGrantRequiredError):
        dispatch_with_argument_classification(
            None,
            request,
            verifier,
            classifier,
            {"mode": "synthetic-mode"},
            dispatch,
            now=EXPIRES_AT - 1,
        )

    other_policy = ArgumentClassificationPolicy(
        policy_profile="policy:org.example/other@1.0.0",
        decisions=(ArgumentDataClassDecision(PUBLIC, ArgumentAction.ALLOW),),
    )
    with pytest.raises(ArgumentClassifierValidationError) as mismatch:
        dispatch_with_argument_classification(
            manifest,
            request,
            verifier,
            _classifier((ArgumentPathRule(("mode",), PUBLIC),), policy=other_policy),
            {"mode": "synthetic-mode"},
            dispatch,
            now=EXPIRES_AT - 1,
        )

    assert mismatch.value.code == "policy_profile_mismatch"
    assert calls == 0


def test_unclassified_and_ambiguous_leaves_fail_closed_without_values() -> None:
    rejected = "synthetic-unclassified-value"
    classifier = _classifier((ArgumentPathRule(("items", "*"), PUBLIC),))

    with pytest.raises(ArgumentClassificationCoverageError) as unclassified:
        classifier.classify({"other": rejected})
    assert unclassified.value.code == "unclassified_argument"
    assert rejected not in str(unclassified.value)

    ambiguous = _classifier(
        (
            ArgumentPathRule(("items", "*"), PUBLIC),
            ArgumentPathRule(("items", 0), CLINICAL),
        )
    )
    with pytest.raises(ArgumentClassificationCoverageError) as overlap:
        ambiguous.classify({"items": [rejected]})
    assert overlap.value.code == "ambiguous_rule"
    assert rejected not in str(overlap.value)


@pytest.mark.parametrize(
    "arguments",
    [
        {"unsafe field": "synthetic-value"},
        {"field": {1: "synthetic-value"}},
        {"field": object()},
        {"field": float("nan")},
        {"field": "\ud800"},
    ],
)
def test_non_typed_or_non_json_arguments_fail_closed(arguments: dict[Any, Any]) -> None:
    classifier = _classifier((ArgumentPathRule(("field",), PUBLIC),))

    with pytest.raises(ArgumentClassifierValidationError):
        classifier.classify(arguments)


def test_reprs_never_expose_rules_policy_keys_arguments_or_results() -> None:
    arguments = {"summary": "synthetic-clinical-value"}
    rule = ArgumentPathRule(("summary",), CLINICAL)
    policy = _policy()
    classifier = _classifier((rule,), policy=policy)
    classified = classifier.classify(arguments)
    manifest, request, verifier = _grant()
    dispatched = dispatch_with_argument_classification(
        manifest,
        request,
        verifier,
        classifier,
        arguments,
        lambda safe: safe["summary"],
        now=EXPIRES_AT - 1,
    )

    combined = " ".join(
        repr(value)
        for value in (
            rule,
            policy,
            classifier,
            classified,
            classified.report,
            classified.report.allowed[0],
            dispatched,
        )
    )
    assert arguments["summary"] not in combined
    assert KEY.decode() not in combined
