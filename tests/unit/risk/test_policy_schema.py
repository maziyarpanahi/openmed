"""Focused tests for the versioned privacy policy-as-data schema."""

from __future__ import annotations

import json

import pytest

from openmed.risk import (
    DEFAULT_ACTION,
    DEFAULT_AUDIT_RETENTION_DAYS,
    DEFAULT_CRITICAL_RECALL_FLOOR,
    DEFAULT_DIRECT_IDENTIFIER_RECALL_FLOOR,
    DEFAULT_RECALL_FLOOR,
    AuditRetention,
    PrivacyPolicy,
    SurrogateStrategy,
    default_policy_schema,
    lint_policy_schema,
    load_policy_schema,
)


def test_defaults_are_explicit_and_preserve_legacy_local_behavior() -> None:
    policy = default_policy_schema()

    assert policy.schema_version == 1
    assert policy.default_action == DEFAULT_ACTION == "mask"
    assert policy.recall_floors.default == DEFAULT_RECALL_FLOOR
    assert (
        policy.recall_floors.direct_identifier == DEFAULT_DIRECT_IDENTIFIER_RECALL_FLOOR
    )
    assert policy.recall_floors.critical == DEFAULT_CRITICAL_RECALL_FLOOR
    assert policy.surrogate_strategy.kind == "none"
    assert policy.surrogate_strategy.reversible is False
    assert policy.audit_retention.enabled is False
    assert policy.audit_retention.retention_days == DEFAULT_AUDIT_RETENTION_DAYS


def test_policy_covers_jurisdiction_floors_actions_surrogates_and_retention() -> None:
    policy = PrivacyPolicy.from_mapping(
        {
            "schema_version": 1,
            "name": "synthetic-clinical-review",
            "jurisdiction": {"code": "EU", "name": "European Union", "region": "test"},
            "recall_floors": {
                "default": 0.97,
                "direct_identifier": 0.995,
                "critical": 1.0,
                "by_label": {"PERSON": 0.999},
            },
            "default_action": "mask",
            "actions": {"PERSON": "replace", "EMAIL": "format_preserve"},
            "surrogate_strategy": {
                "kind": "deterministic",
                "consistent": True,
                "key_ref": "env:OPENMED_POLICY_KEY",
            },
            "audit_retention": {
                "enabled": True,
                "retention_days": 30,
            },
        }
    )

    assert policy.jurisdiction.code == "EU"
    assert policy.recall_floor_for("PERSON") == pytest.approx(0.999)
    assert policy.recall_floor_for("EMAIL") == pytest.approx(0.995)
    assert policy.action_for("person") == "replace"
    assert policy.action_for("PHONE") == "mask"
    assert policy.surrogate_strategy.consistent is True
    assert policy.surrogate_strategy.key_ref == "env:OPENMED_POLICY_KEY"
    assert policy.audit_retention.days == 30
    assert policy.audit_retention.include_text is False


def test_unknown_actions_are_rejected_without_echoing_the_configured_value() -> None:
    configured_value = "synthetic-invalid-action"

    with pytest.raises(ValueError, match="one of") as exc_info:
        PrivacyPolicy.from_mapping({"actions": {"PERSON": configured_value}})

    assert configured_value not in str(exc_info.value)


def test_actions_accept_existing_policy_label_defaults_and_normalize_labels() -> None:
    policy = PrivacyPolicy.from_mapping(
        {
            "default_action": "redact",
            "policy_label_actions": {"DIRECT_IDENTIFIER": "mask"},
            "actions": {"first_name": "replace"},
        }
    )

    assert policy.action_for("FIRST_NAME") == "replace"
    assert policy.action_for("EMAIL") == "mask"
    assert policy.action_for("CONDITION") == "redact"


def test_serialization_is_deterministic_and_round_trips() -> None:
    policy = PrivacyPolicy(
        name="deterministic-policy",
        actions={"EMAIL": "hash", "PERSON": "mask"},
        recall_floors={"default": 0.9, "PERSON": 0.99},
    )

    first = policy.canonical_json()
    second = PrivacyPolicy.from_json(policy.to_json()).canonical_json()

    assert first == second
    assert policy.digest == PrivacyPolicy.from_json(first).digest
    assert policy.digest.startswith("sha256:")
    assert json.loads(first)["actions"] == {"EMAIL": "hash", "PERSON": "mask"}


def test_legacy_flat_fields_keep_safe_defaults_explicit() -> None:
    policy = PrivacyPolicy.from_mapping(
        {
            "name": "legacy-policy",
            "default_action": "mask",
            "actions": {"PERSON": "mask"},
            "method": "replace",
            "consistent": True,
        }
    )

    assert policy.recall_floors.default == DEFAULT_RECALL_FLOOR
    assert policy.surrogate_strategy.kind == "random"
    assert policy.surrogate_strategy.consistent is True
    assert policy.audit_retention == AuditRetention()


def test_surrogate_strategy_never_accepts_key_material_or_mappings() -> None:
    with pytest.raises(ValueError, match="key material") as exc_info:
        SurrogateStrategy.from_value(
            {"kind": "deterministic", "secret": "synthetic-secret"}
        )

    assert "synthetic-secret" not in str(exc_info.value)
    with pytest.raises(ValueError, match="key_ref"):
        SurrogateStrategy(kind="deterministic", reversible=True)


def test_audit_retention_rejects_raw_text_and_mapping_storage() -> None:
    with pytest.raises(ValueError, match="privacy-safe"):
        AuditRetention.from_value(
            {"enabled": True, "retention_days": 7, "include_text": True}
        )
    with pytest.raises(ValueError, match="privacy-safe"):
        AuditRetention.from_value(
            {"enabled": True, "retention_days": 7, "store_mappings": True}
        )


def test_local_loader_and_linter_do_not_need_network(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(default_policy_schema().to_json(), encoding="utf-8")

    loaded = load_policy_schema(path)

    assert loaded.digest == default_policy_schema().digest
    assert lint_policy_schema({"actions": {"PERSON": "unknown"}})
    with pytest.raises(ValueError, match="local"):
        load_policy_schema("https://example.invalid/policy.json")


def test_unknown_top_level_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        PrivacyPolicy.from_mapping({"not_a_policy_field": True})
