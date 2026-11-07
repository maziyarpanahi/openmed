"""Focused offline tests for the FHIR Bulk Data privacy gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from openmed.interop.fhir.bulk import (
    BulkDataGateway,
    BulkGatewayConfig,
    deidentify_ndjson,
)


@dataclass
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text: str, **_: object) -> _FakeResult:
    return _FakeResult(
        deidentified_text=(
            text.replace("Jane Roe", "[NAME]")
            .replace("555-0100", "[PHONE]")
            .replace("Synthetic Street", "[ADDRESS]")
        )
    )


def _patient(resource_id: str, note: str) -> dict[str, object]:
    return {
        "resourceType": "Patient",
        "id": resource_id,
        "name": [{"text": "Jane Roe"}],
        "telecom": [{"system": "phone", "value": "555-0100"}],
        "text": {
            "status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml"><p>{note}</p></div>',
        },
    }


def _write_file(path: Path, resources: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(resource, separators=(",", ":")) + "\n" for resource in resources
        ),
        encoding="utf-8",
    )


def test_interrupted_multi_file_export_resumes_byte_identically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    resumed_output = tmp_path / "resumed"
    clean_output = tmp_path / "clean"
    _write_file(source / "Observation.ndjson", [_patient("obs-1", "Synthetic Street")])
    _write_file(source / "Patient.ndjson", [_patient("pat-1", "interrupt me")])

    def fail_once(text: str, **kwargs: object) -> _FakeResult:
        if "interrupt me" in text:
            raise RuntimeError("synthetic interruption")
        return _fake_deidentify(text, **kwargs)

    gateway = BulkDataGateway(
        BulkGatewayConfig(source, resumed_output, max_buffered_resources=1),
        deidentifier=fail_once,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        gateway.export(job_id="interrupted")

    resumed = BulkDataGateway(
        BulkGatewayConfig(source, resumed_output, max_buffered_resources=1),
        deidentifier=_fake_deidentify,
    ).export(job_id="resumed")
    clean = BulkDataGateway(
        BulkGatewayConfig(source, clean_output, max_buffered_resources=1),
        deidentifier=_fake_deidentify,
    ).export(job_id="clean")

    assert resumed.summary.resumed_files == 1
    assert resumed.summary.peak_buffered_resources == 1
    assert resumed.summary.output_sha256 == clean.summary.output_sha256
    for path in source.rglob("*.ndjson"):
        relative = path.relative_to(source)
        assert (resumed_output / relative).read_bytes() == (
            clean_output / relative
        ).read_bytes()
    assert not list(resumed_output.rglob("*.part"))


def test_binary_and_unsafe_narrative_are_rejected_by_hash_only(tmp_path: Path) -> None:
    source = tmp_path / "input.ndjson"
    destination = tmp_path / "output.ndjson"
    binary = {
        "resourceType": "Binary",
        "id": "binary-1",
        "contentType": "application/octet-stream",
        "data": "synthetic-binary-secret",
    }
    unsafe = {
        "resourceType": "Patient",
        "id": "patient-unsafe",
        "text": {
            "status": "generated",
            "div": '<div xmlns="http://www.w3.org/1999/xhtml">'
            "<script>Jane Roe</script></div>",
        },
    }
    source.write_text(
        "\n".join(
            json.dumps(resource, separators=(",", ":")) for resource in (binary, unsafe)
        )
        + "\n",
        encoding="utf-8",
    )

    summary = deidentify_ndjson(
        source,
        destination,
        deidentifier=_fake_deidentify,
    )

    assert summary.resources_deidentified == 0
    assert [item.reason for item in summary.rejections] == [
        "unsupported_binary",
        "unsafe_narrative",
    ]
    report = json.dumps(summary.to_dict(), sort_keys=True)
    assert "synthetic-binary-secret" not in report
    assert "Jane Roe" not in report
    assert destination.read_text(encoding="utf-8") == ""
    assert all(len(item.resource_sha256) == 64 for item in summary.rejections)


def test_malformed_line_is_reported_without_echoing_resource_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.ndjson"
    destination = tmp_path / "output.ndjson"
    source.write_text(
        '{"resourceType":"Patient","id":"ok"}\n'
        '{"resourceType":"Patient","note":"synthetic-phi-should-not-echo"\n',
        encoding="utf-8",
    )

    summary = deidentify_ndjson(
        source,
        destination,
        deidentifier=_fake_deidentify,
    )

    assert summary.resources_deidentified == 1
    assert summary.error_count == 1
    assert "synthetic-phi-should-not-echo" not in json.dumps(summary.to_dict())


def test_schema_policy_and_date_key_changes_invalidate_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_file(
        source / "Patient.ndjson",
        [
            {
                "resourceType": "Patient",
                "id": "patient-1",
                "birthDate": "1980-01-15",
                "name": [{"text": "Jane Roe"}],
            }
        ],
    )

    def run(schema_policy: str, secret: bytes):
        config = BulkGatewayConfig(
            source,
            output,
            schema_policy=schema_policy,
            date_shift_secret=secret,
        )
        assert secret.decode() not in repr(config)
        return BulkDataGateway(
            config,
            deidentifier=_fake_deidentify,
        ).export()

    first_secret = b"synthetic-first-checkpoint-secret"
    second_secret = b"synthetic-second-checkpoint-secret"
    assert run("fhir_hipaa_safe_harbor", first_secret).summary.resumed_files == 0
    assert run("fhir_hipaa_safe_harbor", first_secret).summary.resumed_files == 1
    assert run("fhir_research_limited_dataset", first_secret).summary.resumed_files == 0
    assert run("fhir_hipaa_safe_harbor", second_secret).summary.resumed_files == 0

    checkpoint = (output / ".openmed-fhir-bulk-checkpoint.json").read_bytes()
    assert first_secret not in checkpoint
    assert second_secret not in checkpoint
    assert b"fhir_hipaa_safe_harbor" not in checkpoint


def test_schema_policy_rejects_unmapped_resource_type_without_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Condition.ndjson"
    destination = tmp_path / "output.ndjson"
    _write_file(
        source,
        [
            {
                "resourceType": "Condition",
                "id": "condition-1",
                "note": [{"text": "Jane Roe"}],
            }
        ],
    )

    summary = deidentify_ndjson(
        source,
        destination,
        schema_policy="fhir_hipaa_safe_harbor",
        date_shift_secret=b"synthetic-schema-key",
        deidentifier=_fake_deidentify,
    )

    assert summary.resources_deidentified == 0
    assert summary.error_count == 1
    assert summary.errors[0].message == "resource transformation failed"
    assert destination.read_text(encoding="utf-8") == ""
    assert "Jane Roe" not in json.dumps(summary.to_dict())
