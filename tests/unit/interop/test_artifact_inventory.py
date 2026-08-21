"""Focused tests for the deterministic offline artifact inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openmed.interop.artifact_inventory import (
    ArtifactInventory,
    ArtifactInventoryEntry,
    ArtifactInventoryError,
    ArtifactPathError,
    DuplicateArtifactError,
    UnreadableArtifactError,
    build_artifact_inventory,
    render_artifact_inventory_json,
    render_artifact_inventory_markdown,
)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_index_is_sorted_and_fingerprinted_without_retaining_contents(tmp_path: Path):
    root = tmp_path / "synthetic-artifacts"
    _write(root / "zeta.json", b'{"kind":"synthetic"}')
    _write(root / "nested" / "alpha.txt", b"synthetic offline fixture")

    inventory = build_artifact_inventory(
        ["zeta.json", "nested/alpha.txt"],
        root=root,
    )

    assert [entry.path for entry in inventory] == [
        "nested/alpha.txt",
        "zeta.json",
    ]
    assert inventory.artifact_count == 2
    assert inventory.total_bytes == len(b'{"kind":"synthetic"}') + len(
        b"synthetic offline fixture"
    )
    assert inventory.media_type_counts == {
        "application/json": 1,
        "text/plain": 1,
    }
    assert inventory.entries[1].fingerprint == (
        "sha256:" + hashlib.sha256(b'{"kind":"synthetic"}').hexdigest()
    )
    assert "synthetic offline fixture" not in json.dumps(
        inventory.to_dict(counts_only=False)
    )


def test_json_and_markdown_reports_are_counts_only_and_deterministic(tmp_path: Path):
    root = tmp_path / "synthetic-artifacts"
    _write(root / "b.json", b"{}")
    _write(root / "a.json", b"[]")

    first = build_artifact_inventory(["b.json", "a.json"], root=root)
    second = build_artifact_inventory(["a.json", "b.json"], root=root)

    json_report = render_artifact_inventory_json(first)
    markdown_report = render_artifact_inventory_markdown(first)
    assert json_report == render_artifact_inventory_json(second)
    assert markdown_report == render_artifact_inventory_markdown(second)
    assert json.loads(json_report) == {
        "artifact_count": 2,
        "media_type_counts": {"application/json": 2},
        "schema_version": "openmed.interop.artifact_inventory.v1",
        "total_bytes": 4,
        "unique_fingerprint_count": 2,
    }
    assert "| Artifact count | 2 |" in markdown_report
    assert "| `application/json` | 2 |" in markdown_report
    assert "a.json" not in json_report
    assert "a.json" not in markdown_report
    assert "artifacts" not in json_report


def test_full_render_contains_metadata_but_not_file_contents(tmp_path: Path):
    root = tmp_path / "synthetic-artifacts"
    secret_fixture_value = b"synthetic-not-for-report"
    _write(root / "artifact.bin", secret_fixture_value)
    inventory = build_artifact_inventory(["artifact.bin"], root=root)

    full_json = render_artifact_inventory_json(inventory, counts_only=False)
    full_markdown = render_artifact_inventory_markdown(inventory, counts_only=False)

    assert "artifact.bin" in full_json
    assert "artifact.bin" in full_markdown
    assert secret_fixture_value.decode() not in full_json
    assert secret_fixture_value.decode() not in full_markdown
    assert inventory.to_json(counts_only=False) == full_json


def test_traversal_is_rejected_without_echoing_the_input_path(tmp_path: Path):
    root = tmp_path / "synthetic-artifacts"
    _write(root / "valid.json", b"{}")
    sensitive_path = "../synthetic-sensitive-value.json"

    with pytest.raises(ArtifactPathError) as excinfo:
        build_artifact_inventory([sensitive_path], root=root)

    assert "traversal" in str(excinfo.value)
    assert sensitive_path not in str(excinfo.value)


def test_symlink_escape_is_rejected(tmp_path: Path):
    root = tmp_path / "synthetic-artifacts"
    outside = tmp_path / "outside.json"
    _write(outside, b"synthetic outside fixture")
    root.mkdir()
    (root / "linked.json").symlink_to(outside)

    with pytest.raises(ArtifactPathError, match="escapes"):
        build_artifact_inventory(["linked.json"], root=root)


def test_duplicate_normalized_paths_are_rejected(tmp_path: Path):
    root = tmp_path / "synthetic-artifacts"
    _write(root / "artifact.json", b"{}")

    with pytest.raises(DuplicateArtifactError, match="duplicate"):
        build_artifact_inventory(["./artifact.json", "artifact.json"], root=root)


def test_unreadable_entries_are_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    root = tmp_path / "synthetic-artifacts"
    unreadable = root / "synthetic-sensitive-value.json"
    _write(unreadable, b"synthetic content")
    original_open = Path.open

    def fail_open(self: Path, *args, **kwargs):
        if self == unreadable:
            raise PermissionError("synthetic-sensitive-value.json contents")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(UnreadableArtifactError) as excinfo:
        build_artifact_inventory([unreadable], root=root)

    assert "synthetic-sensitive-value" not in str(excinfo.value)
    assert "contents" not in str(excinfo.value)


def test_counts_only_metadata_labels_cannot_be_caller_injected():
    digest = "sha256:" + "0" * 64
    sensitive_label = "application/syntheticpatientsecret"

    with pytest.raises(ArtifactInventoryError) as media_excinfo:
        ArtifactInventoryEntry(
            path="artifact.bin",
            byte_count=1,
            media_type=sensitive_label,
            fingerprint=digest,
        )
    assert sensitive_label not in str(media_excinfo.value)

    with pytest.raises(ArtifactInventoryError) as schema_excinfo:
        ArtifactInventory(schema_version="syntheticpatientsecret")
    assert "syntheticpatientsecret" not in str(schema_excinfo.value)


def test_expanded_markdown_escapes_html_looking_paths():
    entry = ArtifactInventoryEntry(
        path="reports/<synthetic>&summary.txt",
        byte_count=1,
        media_type="text/plain",
        fingerprint="sha256:" + "0" * 64,
    )

    report = ArtifactInventory((entry,)).to_markdown(counts_only=False)

    assert "<synthetic>" not in report
    assert "reports/&lt;synthetic&gt;&amp;summary.txt" in report


def test_path_iterable_failures_do_not_leak_caller_values():
    sensitive_value = "synthetic-patient-iterator-value"

    class FailingPaths:
        def __iter__(self):
            raise RuntimeError(sensitive_value)

    with pytest.raises(TypeError) as excinfo:
        build_artifact_inventory(FailingPaths())  # type: ignore[arg-type]

    assert sensitive_value not in str(excinfo.value)


def test_inventory_entry_iterable_failures_do_not_leak_caller_values():
    sensitive_value = "synthetic-patient-entry-value"

    class FailingEntries:
        def __iter__(self):
            raise RuntimeError(sensitive_value)

    with pytest.raises(TypeError) as excinfo:
        ArtifactInventory(entries=FailingEntries())  # type: ignore[arg-type]

    assert sensitive_value not in str(excinfo.value)


def test_report_options_reject_caller_injected_values():
    inventory = ArtifactInventory()
    sensitive_value = "synthetic-patient-report-value"

    with pytest.raises(ArtifactInventoryError) as indent_excinfo:
        inventory.to_json(indent=sensitive_value)  # type: ignore[arg-type]
    assert sensitive_value not in str(indent_excinfo.value)

    class FailingFlag:
        def __bool__(self):
            raise RuntimeError(sensitive_value)

    with pytest.raises(TypeError) as flag_excinfo:
        inventory.to_markdown(  # type: ignore[arg-type]
            counts_only=FailingFlag()
        )
    assert sensitive_value not in str(flag_excinfo.value)


def test_report_write_failures_do_not_leak_output_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    inventory = ArtifactInventory()
    sensitive_value = "synthetic-patient-report-path"
    output_path = tmp_path / sensitive_value / "inventory.json"

    def fail_write(*args, **kwargs):
        raise OSError(str(output_path))

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(ArtifactInventoryError) as excinfo:
        inventory.write_json(output_path)

    assert sensitive_value not in str(excinfo.value)


def test_artifact_count_is_bounded_before_file_access(tmp_path: Path):
    paths = ("artifact.json" for _ in range(10_001))

    with pytest.raises(ArtifactInventoryError, match="entry limit"):
        build_artifact_inventory(paths, root=tmp_path)
