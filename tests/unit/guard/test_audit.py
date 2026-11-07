"""Focused tests for counts-only trace privacy audit artifacts."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

import openmed.guard.audit as audit_module
from openmed.guard.audit import (
    TraceAuditArtifact,
    TraceAuditError,
    build_trace_audit,
    count_categories,
    fingerprint_file,
    hash_bytes,
    hash_policy,
)

SYNTHETIC_SOURCE = "SYNTHETIC-TRACE-SOURCE-001"
POLICY_HASH = hash_policy("synthetic policy fixture")
FILE_HASH_A = hash_bytes(b"synthetic file a")
FILE_HASH_B = hash_bytes(b"synthetic file b")


def test_artifact_json_and_markdown_are_deterministic_and_counts_only():
    first = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        file_fingerprints=[FILE_HASH_B, FILE_HASH_A],
        category_counts={"PHONE": 1, "NAME": 2},
        disposition="redacted",
    )
    second = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        file_fingerprints=[FILE_HASH_A, FILE_HASH_B],
        category_counts={"NAME": 2, "PHONE": 1},
        disposition="redacted",
    )

    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert json.loads(first.to_json()) == {
        "artifact": "trace_privacy_audit",
        "category_counts": {"NAME": 2, "PHONE": 1},
        "disposition": "redacted",
        "file_fingerprints": sorted([FILE_HASH_A, FILE_HASH_B]),
        "policy_hash": POLICY_HASH,
        "scanner_version": "trace-scanner/1.0",
        "schema_version": 1,
    }


def test_scan_summary_allowlist_drops_source_values_mappings_prompts_and_tools():
    summary = {
        "scanner_version": "trace-scanner/1.0",
        "policy_hash": POLICY_HASH,
        "file_fingerprints": [FILE_HASH_A],
        "category_counts": {"NAME": 1},
        "disposition": "quarantined",
        "source_values": [SYNTHETIC_SOURCE],
        "replacement_mappings": {SYNTHETIC_SOURCE: "<NAME>"},
        "prompt": SYNTHETIC_SOURCE,
        "tool_outputs": [SYNTHETIC_SOURCE],
    }

    artifact = TraceAuditArtifact.from_scan_summary(summary)
    rendered = artifact.to_json() + artifact.to_markdown()

    assert SYNTHETIC_SOURCE not in rendered
    assert "source_values" not in rendered
    assert "replacement_mappings" not in rendered
    assert "prompt" not in rendered
    assert "tool_outputs" not in rendered


def test_local_file_fingerprint_never_serializes_file_content(tmp_path):
    source = tmp_path / "synthetic-trace.jsonl"
    source.write_text(SYNTHETIC_SOURCE, encoding="utf-8")

    artifact = TraceAuditArtifact.from_files(
        scanner_version="trace-scanner/1.0",
        policy_hash=hash_policy("synthetic policy"),
        files=[source],
        category_counts=count_categories(["NAME", "NAME", "PHONE"]),
        disposition="reviewed",
    )

    assert artifact.file_fingerprints == (
        f"sha256:{hashlib.sha256(SYNTHETIC_SOURCE.encode()).hexdigest()}",
    )
    assert SYNTHETIC_SOURCE not in artifact.to_json()
    assert str(source) not in artifact.to_json()
    assert fingerprint_file(source) == artifact.file_fingerprints[0]


def test_fingerprint_preserves_raw_newlines_and_end_of_file_bytes(tmp_path):
    payload = b"synthetic-first\r\nsynthetic-second\x1asynthetic-tail\r\n"
    source = tmp_path / "trace.bin"
    source.write_bytes(payload)

    assert fingerprint_file(source) == f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_invalid_inputs_fail_without_echoing_values(tmp_path):
    sensitive_name = "SYNTHETIC-SENSITIVE-PATH-VALUE"
    missing = tmp_path / sensitive_name

    with pytest.raises(
        TraceAuditError, match="unable to fingerprint trace file"
    ) as exc:
        fingerprint_file(missing)
    assert sensitive_name not in str(exc.value)

    with pytest.raises(TraceAuditError, match="category counts"):
        build_trace_audit(
            "trace-scanner/1.0",
            POLICY_HASH,
            category_counts={"NAME": -1},
            disposition="blocked",
        )


def test_json_round_trip_keeps_only_allowlisted_fields():
    artifact = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        file_fingerprints=[FILE_HASH_A],
        category_counts={"NAME": 1},
        disposition="clean",
    )
    payload = json.loads(artifact.to_json())
    payload["raw_trace"] = SYNTHETIC_SOURCE
    payload["tool_output"] = SYNTHETIC_SOURCE

    restored = TraceAuditArtifact.from_dict(payload)

    assert restored == artifact
    assert SYNTHETIC_SOURCE not in restored.to_json()


def test_hash_references_must_be_canonical_sha256_values():
    with pytest.raises(TraceAuditError, match="SHA-256"):
        build_trace_audit(
            "trace-scanner/1.0",
            "sha256:not-a-digest",
            category_counts={},
            disposition="blocked",
        )


def test_category_counts_are_immutable_after_validation():
    artifact = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        category_counts={"NAME": 1},
        disposition="reviewed",
    )

    with pytest.raises(TypeError):
        artifact.category_counts["RAW_TEXT"] = 1  # type: ignore[index]

    assert artifact.to_dict()["category_counts"] == {"NAME": 1}


def test_from_dict_requires_the_serialized_schema_discriminators():
    payload = {
        "scanner_version": "trace-scanner/1.0",
        "policy_hash": POLICY_HASH,
        "file_fingerprints": [],
        "category_counts": {},
        "disposition": "clean",
    }

    with pytest.raises(TraceAuditError, match="artifact type"):
        TraceAuditArtifact.from_dict(payload)


@pytest.mark.skipif(os.name == "nt", reason="symlink behavior is platform-specific")
def test_fingerprint_file_rejects_symlinks_without_reading_target(tmp_path):
    target = tmp_path / "sensitive-target"
    target.write_text(SYNTHETIC_SOURCE, encoding="utf-8")
    link = tmp_path / "trace-link"
    link.symlink_to(target)

    with pytest.raises(TraceAuditError, match="unable to fingerprint") as exc:
        fingerprint_file(link)

    assert SYNTHETIC_SOURCE not in str(exc.value)


def test_atomic_write_failure_preserves_existing_artifact_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    artifact = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        category_counts={"NAME": 1},
        disposition="reviewed",
    )
    output = tmp_path / "trace-audit.json"
    output.write_text("existing\n", encoding="utf-8")

    def fail_replace(source, destination):
        del source, destination
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(audit_module.os, "replace", fail_replace)

    with pytest.raises(TraceAuditError, match="unable to write"):
        artifact.write_json(output)

    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not tuple(tmp_path.glob(".openmed-trace-audit-*"))


@pytest.mark.skipif(os.name == "nt", reason="mode behavior is platform-specific")
def test_artifact_write_is_private_and_rejects_symlink_targets(tmp_path):
    artifact = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        category_counts={"NAME": 1},
        disposition="reviewed",
    )
    output = tmp_path / "audit.json"

    artifact.write_json(output)

    assert output.stat().st_mode & 0o777 == 0o600
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    output.unlink()
    output.symlink_to(outside)

    with pytest.raises(TraceAuditError, match="unable to write"):
        artifact.write_json(output)

    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert output.is_symlink()
