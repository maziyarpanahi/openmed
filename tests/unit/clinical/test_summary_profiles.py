"""Synthetic offline tests for versioned local summary template profiles."""

from __future__ import annotations

import dataclasses
import json
import socket
from pathlib import Path

import pytest

from openmed.clinical import (
    BHC_V1,
    BRIEF_HOSPITAL_COURSE_V1,
    CLINICAL_HANDOFF_V1,
    DISCHARGE_SUMMARY_V1,
    PROBLEM_ORIENTED_V1,
    SUMMARY_PROFILE_DISCLAIMER,
    SUMMARY_PROFILE_VERSION,
    SUMMARY_PROFILES,
    SummaryFieldType,
    SummaryProfileError,
    SummaryTemplateProfile,
    UnknownSummaryProfileError,
    UnknownSummaryProfileVersionError,
    available_summary_profile_versions,
    available_summary_profiles,
    default_summary_profile,
    get_summary_profile,
    load_summary_profile,
    summary_profile_digest,
    validate_summary_output,
)


def _complete_output(profile: SummaryTemplateProfile) -> dict[str, object]:
    output: dict[str, object] = {}
    for field in profile.fields:
        if field.field_type is SummaryFieldType.TEXT:
            value: object = "synthetic typed summary value"
        else:
            value = ["synthetic typed summary item"]
        if field.required:
            output[field.name] = value
    return output


def test_catalog_exposes_bounded_version_one_forms_and_guardrails() -> None:
    assert available_summary_profiles() == (
        "brief_hospital_course",
        "clinical_handoff",
        "discharge_summary",
        "problem_oriented",
    )
    assert set(SUMMARY_PROFILES) == set(available_summary_profiles())
    assert default_summary_profile() is BRIEF_HOSPITAL_COURSE_V1
    assert BHC_V1 is BRIEF_HOSPITAL_COURSE_V1
    assert available_summary_profile_versions("bhc") == (SUMMARY_PROFILE_VERSION,)

    for profile in (
        BRIEF_HOSPITAL_COURSE_V1,
        DISCHARGE_SUMMARY_V1,
        PROBLEM_ORIENTED_V1,
        CLINICAL_HANDOFF_V1,
    ):
        assert profile.version == SUMMARY_PROFILE_VERSION
        assert profile.format == "typed_json"
        assert profile.requires_clinician_review is True
        assert profile.autonomous_decision is False
        assert profile.disclaimer == SUMMARY_PROFILE_DISCLAIMER
        assert profile.fields
        assert any(field.required for field in profile.fields)
        assert len(profile.field_names) == len(set(profile.field_names))
        for field in profile.fields:
            assert field.name
            assert field.value_type in {
                "text",
                "text_list",
                "problem_list",
                "medication_list",
                "procedure_list",
                "follow_up_list",
            }
            if field.repeated:
                assert field.max_items is not None


def test_known_aliases_return_the_canonical_profile() -> None:
    assert get_summary_profile("bhc") is BRIEF_HOSPITAL_COURSE_V1
    assert get_summary_profile("discharge-summary") is DISCHARGE_SUMMARY_V1
    assert get_summary_profile("problem-list") is PROBLEM_ORIENTED_V1
    assert get_summary_profile("clinical-handoff") is CLINICAL_HANDOFF_V1
    assert summary_profile_digest("bhc") == BRIEF_HOSPITAL_COURSE_V1.digest


def test_digest_and_json_are_deterministic_and_round_trip_locally() -> None:
    profile = BRIEF_HOSPITAL_COURSE_V1

    assert profile.canonical_json() == profile.canonical_json()
    assert profile.digest.startswith("sha256:")
    assert len(profile.digest) == 71
    restored = load_summary_profile(profile.to_json())
    assert restored is profile
    assert restored.digest == profile.digest
    assert SummaryTemplateProfile.from_json(profile.to_json()).digest == profile.digest

    reordered = profile.to_dict()
    reordered["fields"] = list(reversed(reordered["fields"]))
    assert load_summary_profile(reordered) is profile


def test_profile_digest_changes_when_a_known_definition_changes() -> None:
    original = BRIEF_HOSPITAL_COURSE_V1.to_dict()
    altered = json.loads(json.dumps(original))
    altered["fields"][0]["max_characters"] += 1

    with pytest.raises(SummaryProfileError, match="definition"):
        load_summary_profile(altered)
    assert BRIEF_HOSPITAL_COURSE_V1.digest == summary_profile_digest("bhc")


def test_unknown_profile_version_is_rejected_without_echoing_configured_value() -> None:
    sensitive_version = "SYNTHETIC_PRIVATE_VERSION_CANARY"

    with pytest.raises(UnknownSummaryProfileVersionError) as caught:
        load_summary_profile("bhc", version=sensitive_version)
    assert sensitive_version not in str(caught.value)

    payload = BRIEF_HOSPITAL_COURSE_V1.to_dict()
    payload["version"] = sensitive_version
    with pytest.raises(UnknownSummaryProfileVersionError) as caught:
        load_summary_profile(payload)
    assert sensitive_version not in str(caught.value)


def test_unknown_names_and_free_form_prompts_fail_closed_without_echoing_values() -> (
    None
):
    sensitive_name = "SYNTHETIC_PRIVATE_PROFILE_NAME"
    with pytest.raises(UnknownSummaryProfileError) as caught:
        load_summary_profile({"name": sensitive_name, "version": "1.0", "fields": []})
    assert sensitive_name not in str(caught.value)

    payload = BRIEF_HOSPITAL_COURSE_V1.to_dict()
    sensitive_prompt = "SYNTHETIC_PRIVATE_SYSTEM_PROMPT"
    payload["system_prompt"] = sensitive_prompt
    with pytest.raises(SummaryProfileError) as prompt_error:
        load_summary_profile(payload)
    assert sensitive_prompt not in str(prompt_error.value)


def test_local_json_loading_rejects_remote_sources_and_needs_no_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "summary-profile.json"
    path.write_text(BRIEF_HOSPITAL_COURSE_V1.to_json(), encoding="utf-8")

    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("summary profile loading must remain local")

    monkeypatch.setattr(socket, "socket", fail_socket)
    assert load_summary_profile(path) is BRIEF_HOSPITAL_COURSE_V1
    assert load_summary_profile(str(path)) is BRIEF_HOSPITAL_COURSE_V1

    with pytest.raises(SummaryProfileError, match="local"):
        load_summary_profile("https://example.invalid/summary-profile.json")


def test_json_loader_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    with pytest.raises(SummaryProfileError, match="invalid"):
        load_summary_profile(
            '{"name":"brief_hospital_course","version":"1.0",'
            '"version":"2.0","fields":[]}'
        )
    with pytest.raises(SummaryProfileError, match="invalid"):
        load_summary_profile(
            '{"name":"brief_hospital_course","version":"1.0","fields":[],"max":NaN}'
        )


def test_typed_output_validation_is_bounded_and_value_free() -> None:
    profile = BRIEF_HOSPITAL_COURSE_V1
    valid = validate_summary_output(profile, _complete_output(profile))
    assert valid.valid is True
    assert valid.findings == ()
    assert valid.to_dict()["profile_digest"] == profile.digest

    sensitive_value = "SYNTHETIC_PRIVATE_SUMMARY_VALUE"
    invalid_output = _complete_output(profile)
    invalid_output["hospital_course"] = [sensitive_value]
    invalid_output["unknown_output"] = sensitive_value
    report = validate_summary_output(profile, invalid_output)

    assert report.valid is False
    assert report.unknown_field_count == 1
    assert report.error_codes == ("invalid_type",)
    serialized = report.to_json()
    assert sensitive_value not in serialized
    assert "unknown_output" not in serialized


def test_typed_output_limits_and_missing_required_fields_are_reported_safely() -> None:
    profile = DISCHARGE_SUMMARY_V1
    output = _complete_output(profile)
    del output["hospital_course"]
    medications = next(
        field for field in profile.fields if field.name == "discharge_medications"
    )
    output[medications.name] = ["synthetic medication"] * (medications.max_items + 1)  # type: ignore[operator]

    report = profile.validate(output)

    assert report.valid is False
    assert {finding.reason_code for finding in report.findings} == {
        "missing_required",
        "too_many_items",
    }
    assert report.profile_name == profile.name
    assert report.profile_version == profile.version


def test_repeated_fields_report_invalid_items_without_serializing_item_values() -> None:
    profile = BRIEF_HOSPITAL_COURSE_V1
    output = _complete_output(profile)
    output["discharge_diagnoses"] = [object()]

    report = profile.validate(output)

    assert report.valid is False
    assert len(report.findings) == 1
    assert report.findings[0].reason_code == "invalid_item_type"
    assert report.findings[0].item_index == 0
    assert "object at" not in report.to_json()


def test_profiles_and_fields_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        BRIEF_HOSPITAL_COURSE_V1.version = "2.0"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        BRIEF_HOSPITAL_COURSE_V1.fields[0].name = "assessment"  # type: ignore[misc]
