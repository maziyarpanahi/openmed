"""Tests for streaming FHIR Bulk Data NDJSON de-identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from openmed.interop.fhir_bulk import (
    FHIRNDJSONLineError,
    deidentify_export,
    deidentify_ndjson,
    iter_ndjson,
)


@dataclass
class _FakeResult:
    deidentified_text: str


def fake_deidentify(text, *, method="replace", policy="hipaa_safe_harbor"):
    """Deterministic offline stand-in for the privacy pipeline."""

    assert method == "replace"
    assert policy in {"hipaa_safe_harbor", "unit_test_policy"}

    redacted = text
    for needle, replacement in (
        ("123 Main Street", "[ADDRESS]"),
        ("Jane Roe", "[NAME]"),
        ("John Doe", "[NAME]"),
        ("555-0100", "[PHONE]"),
    ):
        redacted = redacted.replace(needle, replacement)
    return _FakeResult(deidentified_text=redacted)


def _observation(resource_id: str, note_text: str) -> dict:
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "1234-5",
                    "display": "Synthetic clinical note",
                }
            ],
            "text": "Observation note for John Doe",
        },
        "subject": {
            "reference": "Patient/pat-1",
            "display": "John Doe",
        },
        "note": [{"text": note_text}],
    }


def _patient(resource_id: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": resource_id,
        "name": [{"text": "Jane Roe"}],
        "telecom": [{"system": "phone", "value": "555-0100"}],
    }


def _write_ndjson(path: Path, resources: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(resource) + "\n" for resource in resources),
        encoding="utf-8",
    )


def _read_ndjson(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_iter_ndjson_yields_lazily_and_reports_malformed_line_numbers(tmp_path):
    in_path = tmp_path / "Observation.ndjson"
    in_path.write_text(
        json.dumps(_observation("obs-1", "Seen by John Doe.")) + "\n"
        "{not valid json}\n" + json.dumps(_observation("obs-2", "Seen by Jane Roe.")),
        encoding="utf-8",
    )

    resources = iter_ndjson(in_path)

    assert next(resources)["id"] == "obs-1"
    with pytest.raises(FHIRNDJSONLineError) as excinfo:
        next(resources)
    assert excinfo.value.line_number == 2
    assert "malformed JSON" in excinfo.value.message


def test_deidentify_ndjson_streams_resources_and_preserves_order(tmp_path):
    in_path = tmp_path / "Observation.ndjson"
    out_path = tmp_path / "out" / "Observation.ndjson"
    _write_ndjson(
        in_path,
        [
            _observation("obs-1", "John Doe lives at 123 Main Street."),
            _observation("obs-2", "Jane Roe called from 555-0100."),
        ],
    )

    summary = deidentify_ndjson(
        in_path,
        out_path,
        policy="unit_test_policy",
        method="replace",
        deidentifier=fake_deidentify,
    )

    assert summary.lines_processed == 2
    assert summary.resources_deidentified == 2
    assert summary.blank_lines == 0
    assert summary.errors == ()

    output = _read_ndjson(out_path)
    assert [resource["id"] for resource in output] == ["obs-1", "obs-2"]
    assert output[0]["note"][0]["text"] == "[NAME] lives at [ADDRESS]."
    assert output[1]["note"][0]["text"] == "[NAME] called from [PHONE]."
    assert output[0]["subject"]["display"] == "[NAME]"
    assert output[0]["subject"]["reference"] == "Patient/pat-1"
    assert output[0]["code"]["coding"][0]["display"] == "Synthetic clinical note"


def test_deidentify_ndjson_reports_bad_lines_without_stopping(tmp_path):
    in_path = tmp_path / "Observation.ndjson"
    out_path = tmp_path / "out" / "Observation.ndjson"
    in_path.write_text(
        json.dumps(_observation("obs-1", "Seen by John Doe.")) + "\n"
        "{not valid json}\n"
        "\n" + json.dumps(_observation("obs-2", "Seen by Jane Roe.")) + "\n",
        encoding="utf-8",
    )

    summary = deidentify_ndjson(
        in_path,
        out_path,
        deidentifier=fake_deidentify,
    )

    assert summary.lines_processed == 4
    assert summary.resources_deidentified == 2
    assert summary.blank_lines == 1
    assert len(summary.errors) == 1
    assert summary.errors[0].line_number == 2
    assert "malformed JSON" in summary.errors[0].message
    assert "John Doe" not in summary.errors[0].message

    output = _read_ndjson(out_path)
    assert [resource["id"] for resource in output] == ["obs-1", "obs-2"]
    assert output[0]["note"][0]["text"] == "Seen by [NAME]."
    assert output[1]["note"][0]["text"] == "Seen by [NAME]."


def test_deidentify_ndjson_reports_non_resource_lines_without_stopping(tmp_path):
    in_path = tmp_path / "Patient.ndjson"
    out_path = tmp_path / "out" / "Patient.ndjson"
    in_path.write_text(
        json.dumps(_patient("pat-1")) + "\n"
        "[]\n"
        + json.dumps({"id": "missing-resource-type", "name": "John Doe"})
        + "\n"
        + json.dumps(_patient("pat-2"))
        + "\n",
        encoding="utf-8",
    )

    summary = deidentify_ndjson(
        in_path,
        out_path,
        deidentifier=fake_deidentify,
    )

    assert summary.lines_processed == 4
    assert summary.resources_deidentified == 2
    assert [error.line_number for error in summary.errors] == [2, 3]
    assert summary.errors[0].message == "line must contain a JSON object"
    assert summary.errors[1].message == "resource is missing 'resourceType'"

    output = _read_ndjson(out_path)
    assert [resource["id"] for resource in output] == ["pat-1", "pat-2"]
    assert output[0]["name"][0]["text"] == "[NAME]"


def test_deidentify_export_mirrors_ndjson_directory_structure(tmp_path):
    in_dir = tmp_path / "bulk-export"
    out_dir = tmp_path / "deidentified"
    _write_ndjson(in_dir / "Patient.ndjson", [_patient("pat-1")])
    _write_ndjson(
        in_dir / "nested" / "Observation.ndjson",
        [_observation("obs-1", "John Doe lives at 123 Main Street.")],
    )
    (in_dir / "README.txt").write_text("not part of the export\n", encoding="utf-8")

    summary = deidentify_export(
        in_dir,
        out_dir,
        deidentifier=fake_deidentify,
    )

    assert summary.file_count == 2
    assert summary.lines_processed == 2
    assert summary.resources_deidentified == 2
    assert summary.errors == ()
    assert (out_dir / "Patient.ndjson").is_file()
    assert (out_dir / "nested" / "Observation.ndjson").is_file()
    assert not (out_dir / "README.txt").exists()

    patient = _read_ndjson(out_dir / "Patient.ndjson")[0]
    observation = _read_ndjson(out_dir / "nested" / "Observation.ndjson")[0]
    assert patient["name"][0]["text"] == "[NAME]"
    assert observation["note"][0]["text"] == "[NAME] lives at [ADDRESS]."
