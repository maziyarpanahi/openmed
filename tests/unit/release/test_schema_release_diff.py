"""Tests for the packaged JSON Schema comparison against the last release tag."""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "schema_release_diff.py"
ALLOWLIST = ROOT / "scripts" / "release" / "schema_break_allowlist.json"

spec = importlib.util.spec_from_file_location("schema_release_diff", SCRIPT)
assert spec is not None and spec.loader is not None
schema_release_diff = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schema_release_diff
spec.loader.exec_module(schema_release_diff)

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test Maintainer",
    "GIT_AUTHOR_EMAIL": "maintainer@example.org",
    "GIT_COMMITTER_NAME": "Test Maintainer",
    "GIT_COMMITTER_EMAIL": "maintainer@example.org",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
SCHEMA_DIR = "pkg/core/schemas/json"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **GIT_ENV},
    )
    return result.stdout


def _schema(*, version: str = "1.0.0", properties: tuple[str, ...] = ("a", "b")):
    return {
        "$id": f"https://example.org/schemas/record/{version}",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "a"],
        "properties": {
            "schema_version": {"const": version},
            **{name: {"type": "string"} for name in properties},
        },
    }


def _write(repo: Path, relative: str, payload: Any) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fingerprints(repo: Path) -> None:
    """Regenerate a same-commit snapshot, which must not hide a break."""

    snapshot = {}
    for path in sorted((repo / SCHEMA_DIR).glob("*.schema.json")):
        document = json.loads(path.read_text())
        snapshot[path.name] = sorted(document.get("properties", {}))
    _write(repo, f"{SCHEMA_DIR}/schema-fingerprints.json", snapshot)


def _commit(repo: Path, message: str, *, tag: str | None = None) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    if tag is not None:
        _git(repo, "tag", tag)


@pytest.fixture
def release_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, f"{SCHEMA_DIR}/record.schema.json", _schema())
    _write(repo, f"{SCHEMA_DIR}/stable.schema.json", _schema())
    _write(repo, "pkg/other/legacy.schema.json", {"type": "object"})
    (repo / "pkg" / "notes.json").write_text("{}")
    _fingerprints(repo)
    _commit(repo, "release", tag="v1.0.0")
    return repo


def _allowlist(tmp_path: Path, *entries: dict[str, Any]) -> Path:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": list(entries)}))
    return path


def _main(repo: Path, allowlist: Path, *extra: str) -> int:
    return schema_release_diff.main(
        [
            "--repo-root",
            str(repo),
            "--package",
            "pkg",
            "--allowlist",
            str(allowlist),
            *extra,
        ]
    )


def test_removed_property_with_regenerated_snapshot_still_fails(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        release_repo,
        f"{SCHEMA_DIR}/record.schema.json",
        _schema(properties=("a",)),
    )
    _fingerprints(release_repo)
    _commit(release_repo, "drop b and refresh the snapshot in one commit")

    diff = schema_release_diff.compare_refs(
        release_repo, head_ref="HEAD", package="pkg"
    )
    code = _main(release_repo, _allowlist(tmp_path), "--head-ref", "HEAD")

    assert diff.base_ref == "v1.0.0"
    record = next(change for change in diff.changes if change.name == "record")
    assert record.status == "breaking"
    assert record.failing is True
    assert [finding.to_dict() for finding in record.findings] == [
        {"code": "property_removed", "pointer": "/properties/b"}
    ]
    assert code == 1
    captured = capsys.readouterr()
    assert "breaking: record (1.0.0 -> 1.0.0)" in captured.out
    assert "property_removed at /properties/b" in captured.out
    assert "schema break allowlist" in captured.err


def test_worktree_candidate_is_compared_without_a_commit(
    release_repo: Path, tmp_path: Path
) -> None:
    _write(
        release_repo,
        f"{SCHEMA_DIR}/record.schema.json",
        _schema(properties=("a",)),
    )

    assert _main(release_repo, _allowlist(tmp_path)) == 1


def test_allowlisted_break_passes_and_is_reported_with_its_reason(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        release_repo,
        f"{SCHEMA_DIR}/record.schema.json",
        _schema(properties=("a",)),
    )
    _commit(release_repo, "drop b")
    allowlist = _allowlist(
        tmp_path,
        {
            "schema": "record",
            "reason": "b was never populated by any release",
            "target_version": "1.1.0",
        },
    )
    output = tmp_path / "report.json"

    code = _main(release_repo, allowlist, "--head-ref", "HEAD", "--json", str(output))

    assert code == 0
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["summary"]["allowed"] == 1
    record = next(item for item in report["changes"] if item["name"] == "record")
    assert record["status"] == "breaking"
    assert record["failing"] is False
    assert record["allowed"] == {
        "schema": "record",
        "reason": "b was never populated by any release",
        "target_version": "1.1.0",
    }
    out = capsys.readouterr().out
    assert "allowed for 1.1.0: b was never populated by any release" in out


def test_schema_missing_at_head_is_reported_as_removed(
    release_repo: Path, tmp_path: Path
) -> None:
    (release_repo / SCHEMA_DIR / "stable.schema.json").unlink()
    _commit(release_repo, "remove stable")

    diff = schema_release_diff.compare_refs(
        release_repo, "v1.0.0", "HEAD", package="pkg"
    )

    stable = next(change for change in diff.changes if change.name == "stable")
    assert stable.status == "removed"
    assert stable.base_path == f"{SCHEMA_DIR}/stable.schema.json"
    assert stable.head_path is None
    assert stable.failing is True
    assert _main(release_repo, _allowlist(tmp_path), "--head-ref", "HEAD") == 1
    allowed = _allowlist(
        tmp_path,
        {"schema": "stable", "reason": "folded into record", "target_version": "2.0.0"},
    )
    assert _main(release_repo, allowed, "--head-ref", "HEAD") == 0


def test_version_bump_added_and_compatible_changes_pass(
    release_repo: Path, tmp_path: Path
) -> None:
    _write(
        release_repo,
        f"{SCHEMA_DIR}/record.schema.json",
        _schema(version="2.0.0", properties=("a",)),
    )
    widened = _schema(properties=("a", "b", "c"))
    widened["properties"]["b"]["type"] = ["string", "null"]
    _write(release_repo, f"{SCHEMA_DIR}/stable.schema.json", widened)
    _write(release_repo, f"{SCHEMA_DIR}/fresh.schema.json", _schema())
    _commit(release_repo, "bump, widen and add")

    diff = schema_release_diff.compare_refs(
        release_repo, head_ref="HEAD", package="pkg"
    )

    statuses = {change.name: change.status for change in diff.changes}
    assert statuses == {
        "fresh": "added",
        "legacy": "unchanged",
        "record": "version_bump",
        "stable": "compatible",
    }
    record = next(change for change in diff.changes if change.name == "record")
    assert (record.base_version, record.head_version) == ("1.0.0", "2.0.0")
    assert {finding.code for finding in record.findings} >= {"property_removed"}
    assert not diff.failing
    assert _main(release_repo, _allowlist(tmp_path), "--head-ref", "HEAD") == 0
    assert diff.counts() == {
        "added": 1,
        "removed": 0,
        "unchanged": 1,
        "compatible": 1,
        "version_bump": 1,
        "breaking": 0,
    }


def test_check_is_offline_and_never_imports_openmed(
    release_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        release_repo,
        f"{SCHEMA_DIR}/record.schema.json",
        _schema(properties=("a",)),
    )
    _commit(release_repo, "drop b")
    imported_before = set(sys.modules)
    original_import = builtins.__import__

    def reject_openmed_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openmed" or name.startswith("openmed."):
            raise AssertionError(f"schema diff imported {name}")
        return original_import(name, globals, locals, fromlist, level)

    def reject_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("schema diff opened a network socket")

    monkeypatch.setattr(builtins, "__import__", reject_openmed_import)
    monkeypatch.setattr(socket, "socket", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)

    diff = schema_release_diff.compare_refs(
        release_repo, head_ref="HEAD", package="pkg"
    )

    assert diff.failing
    assert not {name for name in set(sys.modules) - imported_before} & {"openmed"}


def test_missing_release_tag_fails_with_a_shallow_clone_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "untagged"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, f"{SCHEMA_DIR}/record.schema.json", _schema())
    _commit(repo, "untagged")

    assert _main(repo, _allowlist(tmp_path)) == 2
    assert "fetch-depth: 0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "entry",
    [
        {"schema": "record", "target_version": "1.1.0"},
        {"schema": "record", "reason": "  ", "target_version": "1.1.0"},
        {"schema": "record", "reason": "approved"},
        {"schema": "record", "reason": "approved", "target_version": "next"},
        {"reason": "approved", "target_version": "1.1.0"},
    ],
)
def test_allowlist_entries_need_a_schema_reason_and_target_version(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    with pytest.raises(schema_release_diff.SchemaDiffError):
        schema_release_diff.load_allowlist(_allowlist(tmp_path, entry))


def test_allowlist_rejects_duplicates_and_bad_documents(tmp_path: Path) -> None:
    entry = {"schema": "record", "reason": "approved", "target_version": "1.1.0"}
    with pytest.raises(schema_release_diff.SchemaDiffError, match="twice"):
        schema_release_diff.load_allowlist(_allowlist(tmp_path, entry, entry))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"entries": []}))
    with pytest.raises(schema_release_diff.SchemaDiffError):
        schema_release_diff.load_allowlist(bad)


def test_unused_allowlist_entries_are_reported(release_repo: Path) -> None:
    entry = schema_release_diff.AllowlistEntry("stable", "no longer needed", "1.1.0")

    diff = schema_release_diff.compare_refs(
        release_repo, head_ref="HEAD", package="pkg", allowlist=(entry,)
    )

    assert diff.unused_allowlist == (entry,)
    assert diff.to_dict()["unused_allowlist"] == [entry.to_dict()]


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (
            {"required": ["a"]},
            {"required": ["a", "b"]},
            [("required_added", "/required/b")],
        ),
        (
            {"enum": ["x", "y"]},
            {"enum": ["x"]},
            [("enum_value_removed", "/enum")],
        ),
        ({"enum": ["x"]}, {"enum": ["x", "y"]}, []),
        (
            {"type": ["string", "null"]},
            {"type": "string"},
            [("type_narrowed", "/type")],
        ),
        ({"type": "integer"}, {"type": "number"}, []),
        ({"const": 1}, {"const": 2}, [("const_changed", "/const")]),
        (
            {"properties": {"a": {}}},
            {"properties": {"a": {}}, "additionalProperties": False},
            [("additional_properties_closed", "/additionalProperties")],
        ),
        (
            {"properties": {"outer": {"properties": {"inner": {}, "kept": {}}}}},
            {"properties": {"outer": {"properties": {"kept": {}}}}},
            [("property_removed", "/properties/outer/properties/inner")],
        ),
        (
            {"$defs": {"item": {"type": "string"}}},
            {"$defs": {}},
            [("definition_removed", "/$defs/item")],
        ),
        (
            {"items": {"maxLength": 10}},
            {"items": {"maxLength": 5}},
            [("constraint_narrowed", "/items/maxLength")],
        ),
        ({"maxLength": 5}, {"maxLength": 10}, []),
        (
            {"allOf": [{"required": ["a"]}]},
            {"allOf": [{"required": ["a"]}, {"required": ["b"]}]},
            [("all_of_constraint_added", "/allOf")],
        ),
        (
            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            {"oneOf": [{"type": "string"}]},
            [("branch_removed", "/oneOf")],
        ),
        (
            {"properties": {"a~/b": {"type": "string"}}},
            {"properties": {}},
            [("property_removed", "/properties/a~0~1b")],
        ),
        ({"description": "old"}, {"description": "new", "title": "Added"}, []),
    ],
)
def test_breaking_findings_classify_narrowing_changes(
    before: dict[str, Any], after: dict[str, Any], expected: list[tuple[str, str]]
) -> None:
    findings = schema_release_diff.breaking_findings(before, after)

    assert [(finding.code, finding.pointer) for finding in findings] == expected


@pytest.mark.parametrize(
    ("schema", "version"),
    [
        ({"schema_version": 1}, "1"),
        ({"properties": {"schema_version": {"const": "1.1.0"}}}, "1.1.0"),
        (
            {"properties": {"schema_version": {"enum": ["1.10.0", "1.2.0"]}}},
            "1.10.0",
        ),
        ({"$id": "https://example.org/schemas/decision-result/1.2.0"}, "1.2.0"),
        ({"$id": "https://example.org/schemas/v3/record.schema.json"}, "3"),
        ({"title": "unversioned"}, None),
    ],
)
def test_declared_version_reads_every_marker(
    schema: dict[str, Any], version: str | None
) -> None:
    assert schema_release_diff.declared_version(schema) == version


def test_duplicate_schema_names_are_refused(release_repo: Path) -> None:
    _write(release_repo, "pkg/other/record.schema.json", _schema())

    with pytest.raises(schema_release_diff.SchemaDiffError, match="share the name"):
        schema_release_diff.load_schemas(release_repo, "WORKTREE", "pkg")


def test_committed_allowlist_is_valid() -> None:
    entries = schema_release_diff.load_allowlist(ALLOWLIST)

    assert all(entry.reason and entry.target_version for entry in entries)


def test_release_gates_workflow_runs_the_schema_comparison() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )

    assert "Compare packaged JSON Schemas with the last release" in workflow
    assert "python scripts/release/schema_release_diff.py" in workflow
    assert "fetch-depth: 0" in workflow


@pytest.mark.slow
def test_packaged_schemas_have_no_unallowlisted_break_since_the_last_tag() -> None:
    described = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "describe",
            "--tags",
            "--abbrev=0",
            "--match",
            "v[0-9]*",
            "HEAD",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if described.returncode != 0:
        pytest.skip(
            "release tags are unavailable in this shallow checkout; "
            "tag builds fetch history"
        )

    diff = schema_release_diff.compare_refs(
        ROOT,
        described.stdout.strip(),
        allowlist=schema_release_diff.load_allowlist(ALLOWLIST),
    )

    assert [change.name for change in diff.failing] == []
