"""Focused tests for local evidence-bundle integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openmed.risk import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    check_evidence_bundle,
    verify_evidence_bundle,
)

_POLICY_FINGERPRINT = "sha256:" + "1" * 64
_SOURCE_FINGERPRINT = "sha256:" + "2" * 64


def _file_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _write_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "bundle"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    files = {
        "evidence/metrics.json": ("metrics", b'{"count": 4}\n'),
        "evidence/provenance.json": ("provenance", b'{"source": "synthetic"}\n'),
        "evidence/summary.json": ("summary", b'{"passed": true}\n'),
    }
    entries = []
    for relative_path, (section, content) in files.items():
        path = root / relative_path
        path.write_bytes(content)
        entries.append(
            {
                "path": relative_path,
                "section": section,
                "sha256": _file_digest(content),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "policy_fingerprint": _POLICY_FINGERPRINT,
        "required_sections": ["summary", "metrics", "provenance"],
        "provenance": {
            "source_fingerprint": _SOURCE_FINGERPRINT,
            "generator": "openmed-test",
            "created_at": "2026-08-08T12:00:00Z",
            "policy_fingerprint": _POLICY_FINGERPRINT,
        },
        "files": entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, manifest


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_valid_bundle_is_deterministic_and_aggregate_only(tmp_path: Path) -> None:
    root, _ = _write_bundle(tmp_path)

    first = check_evidence_bundle(
        root,
        expected_policy_fingerprint=_POLICY_FINGERPRINT,
    )
    second = verify_evidence_bundle(
        root,
        expected_policy_fingerprint=_POLICY_FINGERPRINT,
    )

    assert first == second
    assert first.passed is True
    assert first.failure_categories == ()
    assert first.checked_file_count == 3
    assert first.to_dict() == {
        "checked_file_count": 3,
        "failure_categories": [],
        "failure_counts": {},
        "passed": True,
    }


def test_missing_file_returns_a_stable_category_without_a_path(tmp_path: Path) -> None:
    root, _ = _write_bundle(tmp_path)
    (root / "evidence/summary.json").unlink()

    result = check_evidence_bundle(root)

    assert result.passed is False
    assert result.failures == ("missing_file",)
    assert result.failure_counts == (("missing_file", 1),)
    assert "evidence/summary.json" not in json.dumps(result.to_dict())


def test_changed_file_hash_is_detected_without_reading_into_the_report(
    tmp_path: Path,
) -> None:
    root, _ = _write_bundle(tmp_path)
    (root / "evidence/metrics.json").write_text(
        '{"count": 5, "note": "synthetic-change"}\n',
        encoding="utf-8",
    )

    result = check_evidence_bundle(root)

    assert result.failures == ("hash_mismatch",)
    assert result.failure_counts == (("hash_mismatch", 1),)
    assert "synthetic-change" not in json.dumps(result.to_dict())


def test_manifest_hash_mutation_uses_the_same_hash_failure_category(
    tmp_path: Path,
) -> None:
    root, manifest = _write_bundle(tmp_path)
    manifest["manifest_hash"] = "sha256:" + "9" * 64
    _write_manifest(root, manifest)

    result = check_evidence_bundle(root)

    assert result.failures == ("hash_mismatch",)


def test_metadata_failures_have_stable_order_and_no_manifest_values(
    tmp_path: Path,
) -> None:
    root, manifest = _write_bundle(tmp_path)
    manifest["schema_version"] = "openmed.evidence_bundle.v0"
    manifest["policy_fingerprint"] = "sha256:" + "3" * 64
    manifest["required_sections"] = ["summary", "metrics", "provenance", "chart"]
    manifest["provenance"] = {
        "source_fingerprint": _SOURCE_FINGERPRINT,
        "created_at": "2026-08-08T12:00:00Z",
    }
    _write_manifest(root, manifest)

    result = check_evidence_bundle(
        root,
        expected_policy_fingerprint=_POLICY_FINGERPRINT,
    )

    assert result.failures == (
        "schema_mismatch",
        "policy_mismatch",
        "missing_section",
        "incomplete_provenance",
    )
    rendered = f"{result}\n{json.dumps(result.to_dict(), sort_keys=True)}"
    assert "openmed.evidence_bundle.v0" not in rendered
    assert "sha256:" + "3" * 64 not in rendered


def test_mapping_input_requires_an_explicit_local_root(tmp_path: Path) -> None:
    root, manifest = _write_bundle(tmp_path)

    result = check_evidence_bundle(manifest)
    rooted_result = check_evidence_bundle(manifest, root=root)

    assert result.failures == ("manifest_unreadable",)
    assert rooted_result.passed is True


def test_relative_path_escape_is_rejected_as_an_unsafe_path(
    tmp_path: Path,
) -> None:
    root, manifest = _write_bundle(tmp_path)
    entries = list(manifest["files"])
    entries[0] = {
        "path": "../outside.json",
        "section": "metrics",
        "sha256": _file_digest(b"outside"),
    }
    manifest["files"] = entries
    _write_manifest(root, manifest)

    result = check_evidence_bundle(root)

    assert result.failures == ("unsafe_path",)
