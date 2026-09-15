"""Tests for deterministic, value-free privacy-policy composition."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

import pytest

from openmed.risk import (
    DEFAULT_SCOPE_PRECEDENCE,
    ConflictCategory,
    PolicyContext,
    PolicyDecision,
    PolicyScope,
    PrivacyPolicy,
    compose_policies,
    composition_policy_fingerprint,
)


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-exception")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-exception")

    def __len__(self) -> int:
        return 1


class _EndlessPolicies:
    def __iter__(self) -> Iterator[Mapping[str, str]]:
        while True:
            yield {"scope": "field", "decision": "deny", "selector": "field"}


def test_deny_overrides_overlapping_scopes_and_records_inheritance() -> None:
    policies = [
        PrivacyPolicy.for_transport("external", "allow", policy_id="transport"),
        PrivacyPolicy.for_field("diagnosis", "allow", policy_id="field"),
        PrivacyPolicy.for_resource(
            "records",
            "deny",
            policy_id="resource",
        ),
    ]

    result = compose_policies(
        policies,
        resource="records/entry",
        field="diagnosis",
        transport="external",
    )

    assert result.decision is PolicyDecision.DENY
    assert result.trace.conflict_category is ConflictCategory.DENY_OVERRIDES
    selected = [entry for entry in result.trace.entries if entry.selected]
    assert len(selected) == 1
    assert selected[0].scope is PolicyScope.RESOURCE
    assert selected[0].inherited is True
    assert all(entry.shadowed for entry in result.trace.entries if not entry.selected)


def test_nested_resource_inheritance_can_be_disabled() -> None:
    parent = PrivacyPolicy.for_resource(
        "records",
        "deny",
        policy_id="parent",
        inherit=False,
    )
    child = PrivacyPolicy.for_resource(
        "records/entry",
        "allow",
        policy_id="child",
    )

    result = compose_policies(
        [parent, child],
        resource="records/entry/detail",
        default_decision="allow",
    )

    assert result.decision is PolicyDecision.ALLOW
    assert result.trace.conflict_category is ConflictCategory.INHERITED_ALLOW
    assert len(result.trace.entries) == 1
    assert result.trace.entries[0].inherited is True


def test_scope_precedence_is_explicit_for_same_decision() -> None:
    field_policy = PrivacyPolicy.for_field("diagnosis", "allow", policy_id="field")
    resource_policy = PrivacyPolicy.for_resource(
        "records",
        "allow",
        policy_id="resource",
    )

    default_result = compose_policies(
        [resource_policy, field_policy],
        resource="records/entry",
        field="diagnosis",
    )
    reversed_result = compose_policies(
        [field_policy, resource_policy],
        resource="records/entry",
        field="diagnosis",
    )
    transport_first = compose_policies(
        [
            field_policy,
            resource_policy,
            PrivacyPolicy.for_transport("local", "allow", policy_id="transport"),
        ],
        resource="records/entry",
        field="diagnosis",
        transport="local",
        precedence=(PolicyScope.TRANSPORT, PolicyScope.RESOURCE, PolicyScope.FIELD),
    )

    assert default_result == reversed_result
    assert default_result.trace.precedence == DEFAULT_SCOPE_PRECEDENCE
    assert default_result.trace.selected_policy_fingerprint == field_policy.fingerprint
    assert (
        transport_first.trace.selected_policy_fingerprint
        == composition_policy_fingerprint(
            PrivacyPolicy.for_transport("local", "allow", policy_id="transport")
        )
    )
    assert transport_first.trace.conflict_category is ConflictCategory.PRECEDENCE


def test_policy_order_does_not_change_decision_trace_or_fingerprints() -> None:
    policies = [
        PrivacyPolicy.for_field(
            "contact_marker",
            "allow",
            policy_id="field-policy",
            metadata={"version": 1},
        ),
        PrivacyPolicy.for_resource("records", "deny", policy_id="resource-policy"),
        PrivacyPolicy.for_transport("local", "allow", policy_id="transport-policy"),
    ]
    context = PolicyContext(
        resource=("records", "entry"),
        field="contact_marker",
        transport="local",
    )

    first = compose_policies(policies, context=context)
    second = compose_policies(list(reversed(policies)), context=context)

    assert first == second
    assert first.trace.to_json() == second.trace.to_json()
    assert first.trace.policy_set_fingerprint == second.trace.policy_set_fingerprint


def test_policy_fingerprint_is_stable_after_input_metadata_mutation() -> None:
    metadata = {"nested": {"revision": 1}}
    policy = PrivacyPolicy.for_field(
        "synthetic-marker",
        "deny",
        policy_id="stable-policy",
        metadata=metadata,
    )
    fingerprint = policy.fingerprint

    metadata["nested"]["revision"] = 2

    assert policy.fingerprint == fingerprint
    with pytest.raises(TypeError):
        policy.metadata["nested"]["revision"] = 3  # type: ignore[index]


def test_value_free_trace_contains_fingerprints_not_policy_values() -> None:
    sensitive_selector = "synthetic-secret-field-0042"
    sensitive_resource = "synthetic-subject-0042/notes"
    sensitive_transport = "synthetic-endpoint-0042"
    policy = PrivacyPolicy.for_field(
        sensitive_selector,
        "deny",
        policy_id="synthetic-policy-name",
        metadata={"note": "synthetic-sensitive-marker"},
    )
    result = compose_policies(
        [policy],
        resource=sensitive_resource,
        field=sensitive_selector,
        transport=sensitive_transport,
    )

    serialized = result.to_json()
    payload = result.to_dict()

    assert sensitive_selector not in serialized
    assert sensitive_resource not in serialized
    assert sensitive_transport not in serialized
    assert "synthetic-policy-name" not in serialized
    assert "synthetic-sensitive-marker" not in serialized
    assert policy.fingerprint in serialized
    assert payload["trace"]["selected_policy_fingerprint"] == policy.fingerprint
    assert repr(policy).find(sensitive_selector) == -1
    assert repr(result).find(sensitive_resource) == -1


def test_default_is_fail_closed_and_can_be_explicitly_changed() -> None:
    denied = compose_policies(resource="records/entry")
    allowed = compose_policies(resource="records/entry", default_decision="allow")

    assert denied.decision is PolicyDecision.DENY
    assert denied.trace.conflict_category is ConflictCategory.DEFAULT
    assert denied.trace.defaulted is True
    assert allowed.decision is PolicyDecision.ALLOW
    assert allowed.trace.defaulted is True


def test_mapping_aliases_and_wildcards_are_supported_without_raw_trace_values() -> None:
    result = compose_policies(
        [
            {
                "scope": "resource",
                "effect": "deny",
                "target": "records/*/notes",
                "name": "synthetic-resource-rule",
            },
            {
                "scope": "transport",
                "decision": "allow",
                "selector": "*",
            },
        ],
        resource="records/entry/notes",
        transport="local",
    )

    assert result.decision is PolicyDecision.DENY
    assert result.trace.conflict_category is ConflictCategory.DENY_OVERRIDES
    assert "records/entry/notes" not in result.to_json()
    json.loads(result.to_json())


@pytest.mark.parametrize(
    ("scope", "selector"),
    [
        ("field", "synthetic-field"),
        ("resource", "synthetic-resource"),
        ("transport", "synthetic-transport"),
    ],
)
def test_invalid_policy_values_raise_without_echoing_input(
    scope: str, selector: str
) -> None:
    with pytest.raises(ValueError) as error:
        PrivacyPolicy(scope=scope, decision="unsupported", selector=selector)

    assert selector not in str(error.value)


def test_policy_and_context_mappings_are_closed_and_unambiguous() -> None:
    invalid_policies: tuple[Mapping[str, Any], ...] = (
        {
            "scope": "field",
            "decision": "deny",
            "selector": "synthetic-field",
            "note": "synthetic-sensitive-value",
        },
        {
            "scope": "field",
            "decision": "deny",
            "effect": "deny",
            "selector": "synthetic-field",
        },
        {
            "scope": "resource",
            "decision": "deny",
            "selector": "records",
            "field": "synthetic-sensitive-value",
        },
        _ExplodingMapping(),
    )

    for policy in invalid_policies:
        with pytest.raises(ValueError) as exc_info:
            compose_policies([policy], field="synthetic-field")
        error = str(exc_info.value)
        assert "synthetic-sensitive-value" not in error
        assert "synthetic-sensitive-exception" not in error

    with pytest.raises(ValueError) as exc_info:
        compose_policies(
            [],
            context={
                "field": "synthetic-field",
                "raw": "synthetic-sensitive-value",
            },
        )
    assert "synthetic-sensitive-value" not in str(exc_info.value)


def test_policy_inputs_are_bounded_and_duplicate_rules_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid"):
        compose_policies(_EndlessPolicies())

    policy = PrivacyPolicy.for_field("field", "deny")
    with pytest.raises(ValueError, match="invalid"):
        compose_policies([policy, policy])

    with pytest.raises(ValueError, match="length limit"):
        PrivacyPolicy.for_field("x" * 513, "deny")
    with pytest.raises(ValueError, match="supported range"):
        PrivacyPolicy.for_field("field", "deny", priority=2**31)


def test_metadata_is_finite_bounded_acyclic_and_normalization_safe() -> None:
    with pytest.raises(ValueError, match="finite"):
        PrivacyPolicy.for_field("field", "deny", metadata={"score": float("inf")})

    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycle"):
        PrivacyPolicy.for_field("field", "deny", metadata=cyclic)

    with pytest.raises(ValueError, match="collide"):
        PrivacyPolicy.for_field(
            "field",
            "deny",
            metadata={"\u00e9": 1, "e\u0301": 2},
        )


def test_public_decision_records_enforce_value_free_invariants() -> None:
    result = compose_policies([PrivacyPolicy.for_field("field", "deny")], field="field")
    entry = result.trace.entries[0]

    with pytest.raises(ValueError, match="fingerprint"):
        replace(result.trace, context_fingerprint="synthetic-sensitive-value")
    with pytest.raises(ValueError, match="selection flags"):
        replace(entry, selected=True, shadowed=True)
    with pytest.raises(ValueError, match="does not match"):
        replace(result, decision=PolicyDecision.ALLOW)
