"""Focused contract tests for the stable CLI result envelope."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from openmed.cli.result_envelope import (
    MAX_REMEDIATION_CODES,
    ArtifactFingerprint,
    RemediationCode,
    ResultCategory,
    ResultEnvelope,
    ResultEnvelopeError,
    ResultStatus,
    create_failure_envelope,
    create_success_envelope,
    serialize_envelope,
)


def test_success_envelope_has_a_complete_canonical_wire_shape() -> None:
    artifact = ArtifactFingerprint(
        name="report",
        sha256="a" * 64,
        size_bytes=17,
    )

    result = create_success_envelope(
        counters={"redacted": 2, "processed": 4},
        artifacts=[artifact],
    )

    assert result.to_dict() == {
        "schema_version": 1,
        "status": "success",
        "category": "success",
        "counters": {"processed": 4, "redacted": 2},
        "artifacts": [artifact.to_dict()],
        "remediation_codes": [],
    }
    assert result.to_json() == (
        '{"artifacts":[{"name":"report","sha256":"'
        + "a" * 64
        + '","size_bytes":17}],"category":"success",'
        '"counters":{"processed":4,"redacted":2},'
        '"remediation_codes":[],"schema_version":1,"status":"success"}'
    )


def test_failure_order_is_normalized_without_free_text() -> None:
    first = create_failure_envelope(
        ResultCategory.VALIDATION,
        counters=[("z_count", 2), ("a_count", 1)],
        remediation_codes=[
            RemediationCode.RETRY_COMMAND,
            RemediationCode.CHECK_INPUT,
        ],
    )
    second = create_failure_envelope(
        "validation",
        counters=[("a_count", 1), ("z_count", 2)],
        remediation_codes=["check_input", "retry_command"],
    )

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == {
        "artifacts": [],
        "category": "validation",
        "counters": {"a_count": 1, "z_count": 2},
        "remediation_codes": ["check_input", "retry_command"],
        "schema_version": 1,
        "status": "failure",
    }


def test_artifact_fingerprint_is_hash_only_and_deterministic() -> None:
    content = b"synthetic offline artifact\n"
    fingerprint = ArtifactFingerprint.from_bytes("bundle", content)

    assert fingerprint.to_dict() == {
        "name": "bundle",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    assert "synthetic offline artifact" not in fingerprint.to_dict()["sha256"]


def test_file_fingerprint_reads_local_bytes_without_serializing_the_path(
    tmp_path,
) -> None:
    path = tmp_path / "synthetic-report.json"
    content = b'{"status":"synthetic"}\n'
    path.write_bytes(content)

    fingerprint = ArtifactFingerprint.from_file("report", path)

    assert fingerprint.sha256 == hashlib.sha256(content).hexdigest()
    assert fingerprint.size_bytes == len(content)
    assert str(path) not in json.dumps(fingerprint.to_dict())


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unknown"),
        ("category", "operator note"),
        ("counters", {"processed": -1}),
        ("counters", {"free text": 1}),
        ("remediation_codes", ["unbounded instruction"]),
    ],
)
def test_invalid_values_are_rejected_without_echoing_them(field, value) -> None:
    kwargs = {
        "status": ResultStatus.FAILURE,
        "category": ResultCategory.RUNTIME,
        "counters": {},
        "artifacts": (),
        "remediation_codes": (),
    }
    kwargs[field] = value

    with pytest.raises(ResultEnvelopeError) as raised:
        ResultEnvelope(**kwargs)

    assert str(value) not in str(raised.value)


def test_free_text_wire_fields_are_rejected() -> None:
    document = {
        "schema_version": 1,
        "status": "failure",
        "category": "runtime",
        "counters": {},
        "artifacts": [],
        "remediation_codes": [],
        "message": "synthetic rejected input",
    }

    with pytest.raises(ResultEnvelopeError) as raised:
        ResultEnvelope.from_dict(document)

    assert "synthetic rejected input" not in str(raised.value)


def test_remediation_codes_are_finite_and_bounded() -> None:
    codes = list(RemediationCode)
    assert len(codes) > MAX_REMEDIATION_CODES

    with pytest.raises(ResultEnvelopeError):
        create_failure_envelope(
            ResultCategory.RUNTIME,
            remediation_codes=codes,
        )


def test_success_and_failure_category_invariants_are_enforced() -> None:
    with pytest.raises(ResultEnvelopeError):
        ResultEnvelope(ResultStatus.SUCCESS, ResultCategory.RUNTIME)
    with pytest.raises(ResultEnvelopeError):
        ResultEnvelope(ResultStatus.FAILURE, ResultCategory.SUCCESS)
    with pytest.raises(ResultEnvelopeError):
        ResultEnvelope(
            ResultStatus.SUCCESS,
            ResultCategory.SUCCESS,
            remediation_codes=[RemediationCode.RETRY_COMMAND],
        )


def test_round_trip_and_newline_writer_are_stable() -> None:
    result = create_failure_envelope(
        ResultCategory.INTEGRITY,
        artifacts=[{"name": "manifest", "sha256": "b" * 64, "size_bytes": 0}],
        remediation_codes=[RemediationCode.VERIFY_ARTIFACT],
    )
    stream = io.StringIO()

    result.write_json(stream)

    assert stream.getvalue() == result.to_json() + "\n"
    assert ResultEnvelope.from_json(result.to_json()) == result
    assert serialize_envelope(result) == result.to_json()


def test_malformed_json_is_not_reflected_in_the_exception() -> None:
    malformed = '{"message":"synthetic rejected input"'

    with pytest.raises(ResultEnvelopeError) as raised:
        ResultEnvelope.from_json(malformed)

    assert "synthetic rejected input" not in str(raised.value)
