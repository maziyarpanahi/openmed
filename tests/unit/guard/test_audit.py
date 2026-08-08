"""Focused tests for counts-only trace privacy audit artifacts."""

from __future__ import annotations

import hashlib
import json

import pytest

from openmed.guard.audit import (
    TraceAuditArtifact,
    TraceAuditError,
    build_trace_audit,
    count_categories,
    fingerprint_file,
    hash_policy,
)

SYNTHETIC_SOURCE = "SYNTHETIC-TRACE-SOURCE-001"
POLICY_HASH = "sha256:policy-fixture"


def test_artifact_json_and_markdown_are_deterministic_and_counts_only():
    first = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        file_fingerprints=["sha256:file-b", "sha256:file-a"],
        category_counts={"PHONE": 1, "NAME": 2},
        disposition="redacted",
    )
    second = build_trace_audit(
        "trace-scanner/1.0",
        POLICY_HASH,
        file_fingerprints=["sha256:file-a", "sha256:file-b"],
        category_counts={"NAME": 2, "PHONE": 1},
        disposition="redacted",
    )

    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert json.loads(first.to_json()) == {
        "artifact": "trace_privacy_audit",
        "category_counts": {"NAME": 2, "PHONE": 1},
        "disposition": "redacted",
        "file_fingerprints": ["sha256:file-a", "sha256:file-b"],
        "policy_hash": POLICY_HASH,
        "scanner_version": "trace-scanner/1.0",
        "schema_version": 1,
    }


def test_scan_summary_allowlist_drops_source_values_mappings_prompts_and_tools():
    summary = {
        "scanner_version": "trace-scanner/1.0",
        "policy_hash": POLICY_HASH,
        "file_fingerprints": ["sha256:file-a"],
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
        file_fingerprints=["sha256:file-a"],
        category_counts={"NAME": 1},
        disposition="clean",
    )
    payload = json.loads(artifact.to_json())
    payload["raw_trace"] = SYNTHETIC_SOURCE
    payload["tool_output"] = SYNTHETIC_SOURCE

    restored = TraceAuditArtifact.from_dict(payload)

    assert restored == artifact
    assert SYNTHETIC_SOURCE not in restored.to_json()
