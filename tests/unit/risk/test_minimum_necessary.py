"""Focused tests for deterministic minimum-necessary field selection."""

from __future__ import annotations

import json

import pytest

from openmed.risk.minimum_necessary import (
    FieldPolicyProfile,
    MinimumNecessarySelector,
    PurposeMapping,
    select_fields,
)

PURPOSE_MAPPINGS = {
    "cohort_review": {
        "fields": ("age_band", "condition_code", "visit_month"),
        "required_fields": ("condition_code",),
    },
    "billing_summary": ("charge_code", "service_month"),
}
POLICY_PROFILES = {
    "research_limited": {
        "allowed_fields": ("age_band", "condition_code", "visit_month"),
    },
    "strict_export": {
        "allowed_fields": ("condition_code",),
        "denied_fields": ("visit_month",),
    },
    "blocked_export": {
        "allowed_fields": ("visit_month",),
    },
}


def test_selection_is_deterministic_and_projects_only_approved_fields() -> None:
    record = {
        "visit_month": "synthetic-month",
        "extra_sensitive_value": "synthetic-sensitive-value",
        "condition_code": "SYNTHETIC-CODE",
        "age_band": "synthetic-age-band",
    }
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    first = selector.select(
        record,
        purpose="cohort_review",
        policy_profile="research-limited",
    )
    second = selector.select(
        tuple(reversed(record)),
        purpose="cohort_review",
        policy_profile="research_limited",
    )

    assert first.allowed is True
    assert first.selected_fields == (
        "age_band",
        "condition_code",
        "visit_month",
    )
    assert first.selected_fields == second.selected_fields
    assert first.to_json() == second.to_json()
    assert first.project(record) == {
        "age_band": "synthetic-age-band",
        "condition_code": "SYNTHETIC-CODE",
        "visit_month": "synthetic-month",
    }
    assert "extra_sensitive_value" not in first.project(record)


def test_policy_allowlist_and_denylist_intersect_with_the_purpose() -> None:
    result = select_fields(
        {"condition_code": "SYNTHETIC-CODE", "visit_month": "synthetic-month"},
        "cohort_review",
        "strict_export",
        purpose_mappings=PURPOSE_MAPPINGS,
        policy_profiles=POLICY_PROFILES,
    )

    assert result.allowed is True
    assert result.fields == ("condition_code",)
    assert result.project(
        {"condition_code": "SYNTHETIC-CODE", "visit_month": "synthetic-month"}
    ) == {"condition_code": "SYNTHETIC-CODE"}
    assert result.reason == "purpose_and_policy_allowlisted"


def test_selection_can_be_computed_from_purpose_schema_without_record_values() -> None:
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    result = selector.select(
        purpose="cohort_review",
        policy_profile="research_limited",
    )

    assert result.allowed is True
    assert result.fields == ("age_band", "condition_code", "visit_month")
    assert result.explanation.available_field_count == 3


def test_selection_requires_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("minimum-necessary selection must stay offline")

    monkeypatch.setattr(socket, "socket", fail_socket)
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    result = selector.select(
        ("condition_code", "age_band"),
        purpose="cohort_review",
        policy_profile="research_limited",
    )

    assert result.allowed is True


@pytest.mark.parametrize(
    ("purpose", "policy_profile", "reason"),
    [
        ("unknown-purpose", "research_limited", "unknown_purpose_mapping"),
        ("cohort_review", "unknown-profile", "unknown_policy_profile"),
    ],
)
def test_unknown_purpose_or_profile_fails_closed_with_safe_explanation(
    purpose: str,
    policy_profile: str,
    reason: str,
) -> None:
    record = {
        "condition_code": "SYNTHETIC-CODE",
        "raw_sensitive_value": "SYNTHETIC-SECRET",
    }
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    result = selector.select(
        record,
        purpose=purpose,
        policy_profile=policy_profile,
    )

    assert result.allowed is False
    assert result.fields == ()
    assert result.project(record) == {}
    assert result.reason == reason
    report = result.to_json()
    assert "SYNTHETIC-SECRET" not in report
    assert "raw_sensitive_value" not in report
    assert json.loads(report)["allowed"] is False


def test_missing_or_policy_blocked_required_field_denies_the_whole_selection() -> None:
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    missing = selector.select(
        {"age_band": "synthetic-age-band"},
        purpose="cohort_review",
        policy_profile="research_limited",
    )
    blocked = selector.select(
        {"condition_code": "SYNTHETIC-CODE"},
        purpose="cohort_review",
        policy_profile="blocked_export",
    )

    assert missing.allowed is False
    assert missing.reason == "required_fields_unavailable"
    assert blocked.allowed is False
    assert blocked.reason == "required_fields_not_permitted"


def test_profile_and_purpose_inputs_are_copied_and_normalized() -> None:
    purpose_fields = ["condition_code", "age_band"]
    allowed_fields = ["age_band", "condition_code"]
    selector = MinimumNecessarySelector(
        {"cohort-review": PurposeMapping(purpose_fields)},
        {"research-limited": FieldPolicyProfile("research-limited", allowed_fields)},
    )
    purpose_fields.clear()
    allowed_fields.clear()

    result = selector.select(
        {"condition_code": "SYNTHETIC-CODE", "age_band": "synthetic-age-band"},
        purpose="COHORT_REVIEW",
        policy_profile="RESEARCH-LIMITED",
    )

    assert result.allowed is True
    assert result.fields == ("age_band", "condition_code")


def test_invalid_configuration_and_record_shape_use_value_free_errors() -> None:
    with pytest.raises(ValueError, match="required fields") as purpose_error:
        PurposeMapping(fields=("condition_code",), required_fields=("SECRET",))
    assert "SECRET" not in str(purpose_error.value)

    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)
    with pytest.raises(TypeError, match="field collections") as record_error:
        selector.select(
            "not-a-record", purpose="cohort_review", policy_profile="research_limited"
        )  # type: ignore[arg-type]
    assert "not-a-record" not in str(record_error.value)


def test_public_report_contains_only_value_free_metadata() -> None:
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)
    result = selector.select(
        {"condition_code": "SYNTHETIC-CODE", "age_band": "synthetic-age-band"},
        purpose="cohort_review",
        policy_profile="research_limited",
    )

    report = result.to_dict()

    assert report["selected_fields"] == ["age_band", "condition_code"]
    assert "SYNTHETIC-CODE" not in json.dumps(report)
    assert "synthetic-age-band" not in json.dumps(report)
    assert "project" not in report


def test_selector_is_exported_from_the_risk_package() -> None:
    import openmed.risk as risk

    for name in (
        "FieldPolicyProfile",
        "FieldSelection",
        "MinimumNecessarySelector",
        "PurposeMapping",
        "SelectionExplanation",
        "select_fields",
        "select_minimum_necessary_fields",
    ):
        assert name in risk.__all__
        assert hasattr(risk, name)
