"""Focused contract tests for the stable CLI result envelope."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import openmed.cli.result_envelope as result_envelope_module
from openmed.cli.result_envelope import (
    MAX_ARTIFACTS,
    MAX_COUNTERS,
    MAX_JSON_CHARS,
    MAX_REMEDIATION_CODES,
    MAX_SAFE_INTEGER,
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


class _HostileMapping(Mapping[str, Any]):
    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic sensitive mapping value")

    def __len__(self) -> int:
        return 6

    def __getitem__(self, key: str) -> Any:
        del key
        raise RuntimeError("synthetic sensitive mapping value")


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


def test_file_fingerprint_validates_name_before_opening_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_open(*args: Any, **kwargs: Any) -> int:
        del args, kwargs
        raise AssertionError("path must not be opened")

    monkeypatch.setattr(result_envelope_module.os, "open", unexpected_open)

    with pytest.raises(ResultEnvelopeError, match="artifact names"):
        ArtifactFingerprint.from_file("not a logical name", "unused")


def test_file_fingerprint_rejects_symlink_before_opening_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = tmp_path / "synthetic-sensitive-target.json"
    artifact_path = tmp_path / "artifact-link.json"
    target_path.write_bytes(b'{"synthetic":"sensitive"}\n')
    try:
        artifact_path.symlink_to(target_path)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    def unexpected_open(*args: Any, **kwargs: Any) -> int:
        del args, kwargs
        raise AssertionError("symbolic-link target must not be opened")

    monkeypatch.setattr(result_envelope_module.os, "open", unexpected_open)

    with pytest.raises(ResultEnvelopeError) as raised:
        ArtifactFingerprint.from_file("report", artifact_path)

    assert "synthetic-sensitive-target" not in str(raised.value)


def test_numeric_wire_values_stay_in_the_interoperable_json_range() -> None:
    with pytest.raises(ResultEnvelopeError, match="bounded non-negative integer"):
        ArtifactFingerprint("report", "a" * 64, MAX_SAFE_INTEGER + 1)

    with pytest.raises(ResultEnvelopeError, match="non-negative integers"):
        create_success_envelope(counters={"processed": MAX_SAFE_INTEGER + 1})


def test_file_fingerprint_rejects_a_swapped_verification_descriptor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "synthetic-sensitive-artifact.json"
    replacement_path = tmp_path / "replacement.json"
    content = b'{"status":"synthetic"}\n'
    artifact_path.write_bytes(content)
    replacement_path.write_bytes(content)
    original_open = os.open
    artifact_open_count = 0

    def substitute_verification_descriptor(
        path: str | os.PathLike[str], flags: int, mode: int = 0o777
    ) -> int:
        nonlocal artifact_open_count
        if Path(path) == artifact_path:
            artifact_open_count += 1
            if artifact_open_count == 2:
                return original_open(replacement_path, flags, mode)
        return original_open(path, flags, mode)

    monkeypatch.setattr(
        result_envelope_module.os,
        "open",
        substitute_verification_descriptor,
    )
    with pytest.raises(ResultEnvelopeError) as raised:
        ArtifactFingerprint.from_file("report", artifact_path)

    assert "synthetic-sensitive-artifact" not in str(raised.value)
    assert "replacement" not in str(raised.value)


def test_file_fingerprint_rejects_a_swapped_path_stat(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "synthetic-sensitive-artifact.json"
    replacement_path = tmp_path / "replacement.json"
    content = b'{"status":"synthetic"}\n'
    artifact_path.write_bytes(content)
    replacement_path.write_bytes(content)
    artifact_stat = artifact_path.stat()
    os.utime(
        replacement_path,
        ns=(artifact_stat.st_atime_ns, artifact_stat.st_mtime_ns),
    )
    original_stat = os.stat

    def substitute_path_stat(
        path: str | bytes | os.PathLike[str] | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == artifact_path and not follow_symlinks:
            return original_stat(replacement_path, follow_symlinks=False)
        return original_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(result_envelope_module.os, "stat", substitute_path_stat)
    with pytest.raises(ResultEnvelopeError) as raised:
        ArtifactFingerprint.from_file("report", artifact_path)

    assert "synthetic-sensitive-artifact" not in str(raised.value)
    assert "replacement" not in str(raised.value)


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


@pytest.mark.parametrize(
    "document",
    [
        '{"status":"success","status":"failure"}',
        '{"counter":NaN}',
        "x" * (MAX_JSON_CHARS + 1),
    ],
)
def test_json_parser_rejects_ambiguous_or_oversized_documents(document: str) -> None:
    with pytest.raises(ResultEnvelopeError) as raised:
        ResultEnvelope.from_json(document)

    assert document not in str(raised.value)


def test_wire_mapping_failures_are_contained_without_echoing_values() -> None:
    with pytest.raises(ResultEnvelopeError) as raised:
        ResultEnvelope.from_dict(_HostileMapping())

    assert "synthetic sensitive mapping value" not in str(raised.value)


def test_iterator_failures_are_contained_without_echoing_values() -> None:
    def poisoned_counters() -> Iterator[tuple[str, int]]:
        yield ("processed", 1)
        raise RuntimeError("synthetic sensitive counter value")

    with pytest.raises(ResultEnvelopeError) as raised:
        create_success_envelope(counters=poisoned_counters())

    assert "synthetic sensitive counter value" not in str(raised.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"counters": itertools.repeat(("processed", 1), MAX_COUNTERS + 1)},
        {
            "artifacts": itertools.repeat(
                ArtifactFingerprint("report", "a" * 64, 1),
                MAX_ARTIFACTS + 1,
            )
        },
    ],
)
def test_collection_limits_are_enforced_before_normalization(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ResultEnvelopeError):
        create_success_envelope(**kwargs)


def test_serializer_rejects_subclasses_without_dispatching_overrides() -> None:
    marker = "synthetic sensitive serialized value"

    class UnsafeEnvelope(ResultEnvelope):
        def to_json(self) -> str:
            return marker

    envelope = UnsafeEnvelope(ResultStatus.SUCCESS, ResultCategory.SUCCESS)

    with pytest.raises(ResultEnvelopeError) as raised:
        serialize_envelope(envelope)

    assert marker not in str(raised.value)
