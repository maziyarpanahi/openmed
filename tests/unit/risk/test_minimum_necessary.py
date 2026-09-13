"""Focused tests for deterministic minimum-necessary field selection."""

from __future__ import annotations

import json

import pytest

from openmed.risk.minimum_necessary import (
    MAX_AVAILABLE_FIELDS,
    MAX_FIELDS_PER_DECLARATION,
    MAX_POLICY_PROFILES,
    MAX_PURPOSE_MAPPINGS,
    FieldPolicyProfile,
    FieldSelection,
    MinimumNecessarySelector,
    PurposeMapping,
    SelectionExplanation,
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


def test_projection_fails_closed_when_schema_only_selection_lacks_required_field() -> (
    None
):
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)
    result = selector.select(
        purpose="cohort_review",
        policy_profile="research_limited",
    )

    assert result.allowed is True
    assert (
        result.project(
            {"age_band": "synthetic-age-band", "raw_sensitive": "SYNTHETIC-SECRET"}
        )
        == {}
    )


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
    with pytest.raises(TypeError, match="available fields") as record_error:
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
        "MAX_AVAILABLE_FIELDS",
        "MAX_FIELDS_PER_DECLARATION",
        "MAX_POLICY_PROFILES",
        "MAX_PURPOSE_MAPPINGS",
        "MinimumNecessarySelector",
        "PurposeMapping",
        "SelectionExplanation",
        "select_fields",
        "select_minimum_necessary_fields",
    ):
        assert name in risk.__all__
        assert hasattr(risk, name)


def test_explanation_never_includes_undeclared_source_keys() -> None:
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    result = selector.select(
        {
            "condition_code": "SYNTHETIC-CODE",
            "visit_month": "synthetic-month",
            "patient_SYNTHETIC-SECRET": "SYNTHETIC-SECRET",
        },
        purpose="cohort_review",
        policy_profile="strict_export",
    )

    report = result.to_json()
    assert result.allowed is True
    assert "patient_SYNTHETIC-SECRET" not in report
    assert "SYNTHETIC-SECRET" not in report
    assert result.explanation.omitted_fields == ("age_band", "visit_month")


def test_unknown_declarations_fail_closed_without_reading_the_record() -> None:
    class HostileRecord(dict[str, str]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("SYNTHETIC-SECRET")

    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)
    record = HostileRecord(condition_code="SYNTHETIC-CODE")

    unknown_purpose = selector.select(
        record,
        purpose="unknown_purpose",
        policy_profile="research_limited",
    )
    unknown_profile = selector.select(
        record,
        purpose="cohort_review",
        policy_profile="unknown_profile",
    )

    assert unknown_purpose.reason == "unknown_purpose_mapping"
    assert unknown_profile.reason == "unknown_policy_profile"
    assert unknown_purpose.explanation.available_field_count == 0
    assert unknown_profile.explanation.available_field_count == 0


def test_hostile_inputs_use_value_free_errors() -> None:
    class HostileRegistry(dict[str, object]):
        def items(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("SYNTHETIC-SECRET")

    with pytest.raises(ValueError, match="could not be read") as registry_error:
        MinimumNecessarySelector(HostileRegistry(), POLICY_PROFILES)
    assert "SYNTHETIC-SECRET" not in str(registry_error.value)

    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    def hostile_fields():  # type: ignore[no-untyped-def]
        yield "condition_code"
        raise RuntimeError("SYNTHETIC-SECRET")

    with pytest.raises(ValueError, match="could not be read") as fields_error:
        selector.select(
            hostile_fields(),
            purpose="cohort_review",
            policy_profile="research_limited",
        )
    assert "SYNTHETIC-SECRET" not in str(fields_error.value)


def test_projection_sanitizes_caller_mapping_failures() -> None:
    class HostileRecord(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            raise RuntimeError("SYNTHETIC-SECRET")

    selection = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES).select(
        ("condition_code",),
        purpose="cohort_review",
        policy_profile="research_limited",
    )

    with pytest.raises(ValueError, match="projected safely") as error:
        selection.project(HostileRecord(condition_code="SYNTHETIC-CODE"))
    assert "SYNTHETIC-SECRET" not in str(error.value)


def test_public_result_cannot_be_constructed_as_a_projection_bypass() -> None:
    explanation = SelectionExplanation(
        allowed=True,
        reason="purpose_and_policy_allowlisted",
        purpose="cohort_review",
        policy_profile="research_limited",
        selected_fields=("raw_secret",),
        omitted_fields=(),
        required_fields=(),
        available_field_count=1,
    )

    with pytest.raises(TypeError, match="created by a selector"):
        FieldSelection(("raw_secret",), explanation)

    with pytest.raises(ValueError, match="outcome is inconsistent"):
        SelectionExplanation(
            allowed=True,
            reason="unknown_purpose_mapping",
            purpose=None,
            policy_profile=None,
            selected_fields=(),
            omitted_fields=(),
            required_fields=(),
            available_field_count=0,
        )


def test_configuration_is_closed_and_bounded() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        MinimumNecessarySelector(
            {"cohort_review": {"fields": ("condition_code",), "extra": ()}},
            POLICY_PROFILES,
        )
    with pytest.raises(ValueError, match="duplicate allowlist aliases"):
        MinimumNecessarySelector(
            PURPOSE_MAPPINGS,
            {
                "research_limited": {
                    "fields": ("condition_code",),
                    "allowed_fields": ("condition_code",),
                }
            },
        )
    with pytest.raises(ValueError, match="item limit"):
        PurposeMapping(
            tuple(f"field_{index}" for index in range(MAX_FIELDS_PER_DECLARATION + 1))
        )
    with pytest.raises(ValueError, match="item limit"):
        MinimumNecessarySelector(
            {
                f"purpose_{index}": ("field",)
                for index in range(MAX_PURPOSE_MAPPINGS + 1)
            },
            {},
        )
    with pytest.raises(ValueError, match="item limit"):
        MinimumNecessarySelector(
            {},
            {
                f"profile_{index}": ("field",)
                for index in range(MAX_POLICY_PROFILES + 1)
            },
        )


def test_available_fields_are_bounded_and_identifiers_are_strict() -> None:
    selector = MinimumNecessarySelector(PURPOSE_MAPPINGS, POLICY_PROFILES)

    with pytest.raises(ValueError, match="item limit"):
        selector.select(
            (f"field_{index}" for index in range(MAX_AVAILABLE_FIELDS + 1)),
            purpose="cohort_review",
            policy_profile="research_limited",
        )
    with pytest.raises(ValueError, match="safe bounded identifier") as error:
        selector.select(
            ("condition_code",),
            purpose="cohort_review\nSYNTHETIC-SECRET",
            policy_profile="research_limited",
        )
    assert "SYNTHETIC-SECRET" not in str(error.value)
