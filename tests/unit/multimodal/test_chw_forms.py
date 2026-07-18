"""Tests for ODK, CommCare, and KoBoToolbox export de-identification."""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import openmed.multimodal.base as base
from openmed.interop import assert_redacted
from openmed.multimodal import ExtractedDocument, redact_document
from openmed.multimodal.chw_forms import (
    ACTION_GENERALIZE_GEO,
    classify_chw_field,
    parse_xform_path,
    redact_chw_form,
)
from openmed.multimodal.tabular_csv import (
    ACTION_DROP,
    ACTION_FREE_TEXT_REDACT,
    ACTION_HASH,
    ACTION_MASK,
    DIRECT_ID,
    QUASI_ID,
)

FIXTURES = Path(__file__).parent / "fixtures" / "chw_forms"

PLATFORM_EXPORTS = (
    ("odk", "json"),
    ("odk", "csv"),
    ("commcare", "json"),
    ("commcare", "csv"),
    ("kobo", "json"),
    ("kobo", "csv"),
)

SEEDED_NAMES = (
    "Amina Njeri",
    "Neema Njeri",
    "Juma Njeri",
    "Otieno Kamau",
    "Ayo Kamau",
    "Lindiwe Dlamini",
    "Sipho Dlamini",
    "Zola Dlamini",
    "Thabo Mokoena",
    "Naledi Mokoena",
    "Mariam Diallo",
    "Awa Diallo",
    "Ibrahima Diallo",
    "Fatou Sarr",
    "Moussa Sarr",
)

SEEDED_IDENTIFIERS = SEEDED_NAMES + (
    "+254700111222",
    "+254700111223",
    "+27821234567",
    "+27821234568",
    "+221771112233",
    "+221771112234",
    "KEN-873-001",
    "KEN-873-002",
    "ZA-873-001",
    "ZA-873-002",
    "SN-873-001",
    "SN-873-002",
    "-1.292066 36.821946 1798 4",
    "-1.283330 36.816670 1790 5",
    "-26.204103 28.047305 1753 6",
    "-25.747868 28.229271 1339 7",
    "14.716677 -17.467686 25 3",
    "14.692778 -17.446667 22 4",
    "uuid:odk-instance-873-001",
    "uuid:commcare-instance-873-001",
    "uuid:kobo-873-001",
    "commcare-case-873-001",
)


def _synthetic_text_redactor(text: str) -> str:
    for name in SEEDED_NAMES:
        text = text.replace(name, "[PERSON]")
    return text


def _json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_schema(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_schema(child) for child in value]
    return type(value).__name__


def _list_lengths(value: Any, path: str = "root") -> dict[str, int]:
    lengths: dict[str, int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            lengths.update(_list_lengths(child, f"{path}/{key}"))
    elif isinstance(value, list):
        lengths[path] = len(value)
        for index, child in enumerate(value):
            lengths.update(_list_lengths(child, f"{path}/{index}"))
    return lengths


def _coded_json_values(value: Any, path: str = "") -> dict[str, list[Any]]:
    coded: dict[str, list[Any]] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}" if path else key
            coded.update(_coded_json_values(child, child_path))
    elif isinstance(value, list):
        for child in value:
            for key, entries in _coded_json_values(child, path).items():
                coded.setdefault(key, []).extend(entries)
    elif path.lower().endswith(("symptoms", "status", "immunization_status")):
        coded.setdefault(path, []).append(value)
    return coded


@pytest.mark.parametrize("platform,extension", PLATFORM_EXPORTS)
def test_platform_exports_round_trip_without_structural_loss(platform, extension):
    source_path = FIXTURES / f"{platform}.{extension}"
    result = redact_chw_form(
        source_path,
        text_redactor=_synthetic_text_redactor,
    )

    assert result.platform == platform
    assert result.row_count == 2
    if extension == "json":
        source = json.loads(source_path.read_text(encoding="utf-8"))
        output = json.loads(result.text)
        assert _json_schema(output) == _json_schema(source)
        assert _list_lengths(output) == _list_lengths(source)
        assert _coded_json_values(output) == _coded_json_values(source)
    else:
        source_rows = list(csv.DictReader(source_path.open(encoding="utf-8")))
        output_rows = list(csv.DictReader(io.StringIO(result.text)))
        assert output_rows[0].keys() == source_rows[0].keys()
        assert len(output_rows) == len(source_rows)
        coded_headers = [
            header
            for header in source_rows[0]
            if header.lower().endswith(("symptoms", "status"))
        ]
        for header in coded_headers:
            assert [row[header] for row in output_rows] == [
                row[header] for row in source_rows
            ]


@pytest.mark.parametrize("platform,extension", PLATFORM_EXPORTS)
def test_no_seeded_phi_leaks_to_output_or_manifest(platform, extension):
    result = redact_chw_form(
        FIXTURES / f"{platform}.{extension}",
        text_redactor=_synthetic_text_redactor,
    )
    serialized = result.text + json.dumps(result.manifest, sort_keys=True)
    mapping = {f"seed-{index}": value for index, value in enumerate(SEEDED_IDENTIFIERS)}

    assert_redacted(serialized, mapping)
    assert not any(value in serialized for value in SEEDED_IDENTIFIERS)


def test_nested_xform_paths_map_to_canonical_semantics():
    assert parse_xform_path("form.household/repeats/national_id") == (
        "form",
        "household",
        "repeats",
        "national_id",
    )

    national_id = classify_chw_field("form.household/repeats/national_id")
    person = classify_chw_field("household/members/respondent_name")
    phone = classify_chw_field("form.patient.mobile")
    birth_date = classify_chw_field("household/child/date_of_birth")
    address = classify_chw_field("form.patient/residential_address")
    narrative = classify_chw_field("visit/counselling_notes")
    coded = classify_chw_field("visit/symptoms")

    assert (national_id.canonical_label, national_id.action) == ("ID_NUM", ACTION_HASH)
    assert (person.canonical_label, person.action) == ("PERSON", ACTION_MASK)
    assert (phone.canonical_label, phone.action) == ("PHONE", ACTION_MASK)
    assert (birth_date.canonical_label, birth_date.assigned_class) == (
        "DATE_OF_BIRTH",
        DIRECT_ID,
    )
    assert address.canonical_label == "STREET_ADDRESS"
    assert narrative.action == ACTION_FREE_TEXT_REDACT
    assert coded.action == ACTION_FREE_TEXT_REDACT


@pytest.mark.parametrize(
    "path",
    (
        "_uuid",
        "_submission_time",
        "meta/deviceID",
        "form.meta.instanceID",
        "form/case/@case_id",
        "KEY",
        "PARENT_KEY",
    ),
)
def test_platform_metadata_is_hashed_or_dropped(path):
    hashed = classify_chw_field(path)
    dropped = classify_chw_field(path, policy={"metadata_action": "drop"})

    assert hashed.action == ACTION_HASH
    assert hashed.assigned_class == DIRECT_ID
    assert hashed.detection_source == "platform_metadata"
    assert dropped.action == ACTION_DROP


def test_geopoints_are_quasi_identifiers_and_are_generalized_or_dropped():
    decision = classify_chw_field(
        "household/repeat/geotrace",
        sample_values=("-1.292066 36.821946 1798 4",),
    )
    assert decision.canonical_label == "GPS_COORDINATES"
    assert decision.assigned_class == QUASI_ID
    assert decision.action == ACTION_GENERALIZE_GEO

    generalized = redact_chw_form(
        FIXTURES / "odk.csv",
        text_redactor=_synthetic_text_redactor,
    )
    dropped = redact_chw_form(
        FIXTURES / "odk.csv",
        policy={"geopoint_action": "drop"},
        text_redactor=_synthetic_text_redactor,
    )

    assert "-1.29 36.82" in generalized.text
    assert "-1.292066" not in generalized.text
    assert "household/geopoint" not in next(csv.reader(io.StringIO(dropped.text)))


def test_manifest_contains_field_policy_and_counts_but_no_values():
    result = redact_chw_form(
        FIXTURES / "kobo.json",
        text_redactor=_synthetic_text_redactor,
    )
    manifest = {entry["field_path"]: entry for entry in result.manifest}

    assert manifest["_uuid"]["action"] == ACTION_HASH
    assert manifest["household/geolocation"]["action"] == ACTION_GENERALIZE_GEO
    assert manifest["household/counselling_notes"]["value_count_affected"] == 2
    assert manifest["children/status"]["action"] == ACTION_FREE_TEXT_REDACT
    assert manifest["children/status"]["value_count"] == 3
    assert "Mariam Diallo" not in json.dumps(result.manifest)
    assert "14.716677" not in json.dumps(result.manifest)


@pytest.mark.parametrize("extension", ("json", "csv"))
def test_same_input_and_policy_are_byte_deterministic(extension):
    path = FIXTURES / f"commcare.{extension}"
    first = redact_chw_form(path, text_redactor=_synthetic_text_redactor)
    second = redact_chw_form(path, text_redactor=_synthetic_text_redactor)

    assert first.text.encode() == second.text.encode()
    assert (
        json.dumps(first.manifest, sort_keys=True).encode()
        == json.dumps(
            second.manifest,
            sort_keys=True,
        ).encode()
    )


@pytest.mark.parametrize("platform,extension", PLATFORM_EXPORTS)
def test_registered_handler_is_lazy_and_optional_dependency_free(
    monkeypatch,
    platform,
    extension,
):
    def fail_if_called():
        raise AssertionError("optional multimodal dependency check was called")

    monkeypatch.setattr(base, "ensure_multimodal_available", fail_if_called)
    document = redact_document(
        FIXTURES / f"{platform}.{extension}",
        models={"text_redactor": _synthetic_text_redactor},
    )

    assert isinstance(document, ExtractedDocument)
    assert document.metadata["format"] == f"chw_form_{extension}"
    assert document.metadata["platform"] == platform
    assert document.metadata["row_count"] == 2
    assert document.metadata["redaction_manifest"]


def test_drop_policy_removes_metadata_without_disturbing_repeats():
    source = json.loads((FIXTURES / "commcare.json").read_text(encoding="utf-8"))
    result = redact_chw_form(
        FIXTURES / "commcare.json",
        policy={"metadata_action": "drop"},
        text_redactor=_synthetic_text_redactor,
    )
    output = json.loads(result.text)

    assert "instanceID" not in output[0]["form"]["meta"]
    assert "deviceID" not in output[0]["form"]["meta"]
    assert "@case_id" not in output[0]["form"]["case"]
    assert len(output[0]["form"]["children"]) == len(source[0]["form"]["children"])


def test_policy_can_map_custom_headers_and_override_actions():
    result = redact_chw_form(
        "meta/instanceID,beneficiary/custom_contact,visit/other_answer\n"
        "uuid:custom-873,Amina Njeri,Amina Njeri requested help\n",
        platform="odk",
        policy={
            "header_heuristics": {"custom_contact": "PERSON"},
            "action_overrides": {"visit/other_answer": "free_text_redact"},
        },
        text_redactor=_synthetic_text_redactor,
    )
    row = next(csv.DictReader(io.StringIO(result.text)))

    assert row["beneficiary/custom_contact"] == "[PERSON]"
    assert row["visit/other_answer"] == "[PERSON] requested help"


def test_unknown_text_is_redacted_while_codes_and_scalar_types_survive():
    source = {
        "_uuid": "uuid:custom-873",
        "unexpected_answer": "Amina Njeri requested help",
        "symptoms": "fever cough",
        "visit_count": 3,
        "consent": True,
    }

    result = redact_chw_form(
        json.dumps(source),
        platform="odk",
        text_redactor=_synthetic_text_redactor,
    )
    output = json.loads(result.text)

    assert output["unexpected_answer"] == "[PERSON] requested help"
    assert output["symptoms"] == source["symptoms"]
    assert output["visit_count"] == 3
    assert output["consent"] is True


def test_default_narrative_pipeline_receives_language_hint(monkeypatch):
    observed: list[tuple[str, dict[str, Any]]] = []

    class Result:
        deidentified_text = "[PERSON] requested help"

    def fake_deidentify(text: str, **kwargs: Any) -> Result:
        observed.append((text, dict(kwargs)))
        return Result()

    monkeypatch.setattr("openmed.core.pii.deidentify", fake_deidentify)
    result = redact_chw_form(
        '{"unexpected_answer":"Amina Njeri requested help"}',
        platform="odk",
        lang="fr",
    )

    assert json.loads(result.text)["unexpected_answer"] == "[PERSON] requested help"
    assert observed == [
        (
            "Amina Njeri requested help",
            {"method": "mask", "lang": "fr"},
        )
    ]


def test_default_date_shift_preserves_intervals_within_json_record():
    result = redact_chw_form(
        '{"visit_date":"2026-03-10","followup_date":"2026-03-20"}',
        platform="odk",
    )
    output = json.loads(result.text)

    source_interval = date.fromisoformat("2026-03-20") - date.fromisoformat(
        "2026-03-10"
    )
    output_interval = date.fromisoformat(output["followup_date"]) - date.fromisoformat(
        output["visit_date"]
    )
    assert output_interval == source_interval


def test_bom_prefixed_csv_still_uses_chw_handler_and_protects_metadata(tmp_path):
    path = tmp_path / "submissions.csv"
    path.write_text(
        "\ufeff_uuid,visit/note\nuuid:custom-873,Amina Njeri requested help\n",
        encoding="utf-8",
    )
    document = redact_document(
        path,
        models={"text_redactor": _synthetic_text_redactor},
    )
    row = next(csv.DictReader(io.StringIO(document.text)))

    assert "_uuid" in row
    assert row["_uuid"].startswith("ID_NUM_")
    assert "uuid:custom-873" not in document.text
    assert document.metadata["format"] == "chw_form_csv"


def test_csv_width_mismatch_is_rejected_without_silent_truncation():
    with pytest.raises(ValueError, match="header width"):
        redact_chw_form(
            "_uuid,visit/note\nuuid:custom-873,Amina Njeri,unexpected\n",
            platform="odk",
            text_redactor=_synthetic_text_redactor,
        )
