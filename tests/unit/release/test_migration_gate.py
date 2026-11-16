"""Tests for the gate tying schema version bumps to record migrators."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core.record_migrations import MIGRATION_FIXTURE_ROOT, MigratorRegistry

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "record_migration_gate.py"
EXEMPTIONS = ROOT / "scripts" / "release" / "record_migration_exemptions.json"

spec = importlib.util.spec_from_file_location("record_migration_gate", SCRIPT)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test Maintainer",
    "GIT_AUTHOR_EMAIL": "maintainer@example.org",
    "GIT_COMMITTER_NAME": "Test Maintainer",
    "GIT_COMMITTER_EMAIL": "maintainer@example.org",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
SCHEMA_PATH = "pkg/schemas/synthetic_note.schema.json"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
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
        check=True,
        env={**os.environ, **GIT_ENV},
    )


def _schema(version: str) -> dict[str, Any]:
    return {
        "$id": f"https://example.org/schemas/synthetic-note/{version}",
        "type": "object",
        "properties": {"schema_version": {"const": version}},
    }


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _fixture_pair(repo: Path, name: str) -> None:
    directory = repo / MIGRATION_FIXTURE_ROOT / name
    _write(directory / "before.json", {"schema_version": "1.0.0"})
    _write(directory / "after.json", {"schema_version": "1.1.0"})


@pytest.fixture
def bumped_repo(tmp_path: Path) -> Path:
    """A repository tagged at 1.0.0 whose candidate bumps the schema to 1.1.0."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo / SCHEMA_PATH, _schema("1.0.0"))
    _write(repo / "pkg/schemas/stable.schema.json", _schema("1.0.0"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "release")
    _git(repo, "tag", "v1.0.0")
    _write(repo / SCHEMA_PATH, _schema("1.1.0"))
    _write(repo / "pkg/schemas/added.schema.json", _schema("3.0.0"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bump synthetic_note")
    return repo


def _exemptions(tmp_path: Path, *entries: dict[str, Any]) -> Path:
    return _write(
        tmp_path / "exemptions.json",
        {"schema_version": 1, "no_record_migration": list(entries)},
    )


def _registry(*edges: tuple[str, str]) -> MigratorRegistry:
    registry = MigratorRegistry()
    registry.register_schema("synthetic_note", current_version="1.1.0")
    for source, target in edges:
        registry.register_migrator(
            "synthetic_note",
            source,
            target,
            lambda record, target=target: {**record, "schema_version": target},
            lossless=True,
            fixture=f"synthetic_note/{source}-to-{target}",
        )
    return registry


def _run(repo: Path, exemptions: Path, registry: MigratorRegistry, *extra: str) -> int:
    return gate.main(
        [
            "--repo-root",
            str(repo),
            "--package",
            "pkg",
            "--exemptions",
            str(exemptions),
            *extra,
        ],
        registry=registry,
    )


def test_bump_without_migrator_or_exemption_fails(
    bumped_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = gate.run_gate(
        bumped_repo,
        package="pkg",
        exemptions_path=_exemptions(tmp_path),
        registry=MigratorRegistry(),
    )

    assert report.base_ref == "v1.0.0"
    assert report.bumps == (gate.Bump("synthetic_note", "1.0.0", "1.1.0"),)
    assert [finding.to_dict() for finding in report.failing] == [
        {
            "code": "bump_without_migrator",
            "schema": "synthetic_note",
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "detail": None,
            "failing": True,
        }
    ]
    assert _run(bumped_repo, _exemptions(tmp_path), MigratorRegistry()) == 1
    captured = capsys.readouterr()
    assert "[FAIL] bump_without_migrator: synthetic_note 1.0.0 -> 1.1.0" in (
        captured.out
    )
    assert "no_record_migration exemption" in captured.err


def test_exemption_without_a_reason_fails(bumped_repo: Path, tmp_path: Path) -> None:
    entry = {
        "schema": "synthetic_note",
        "from_version": "1.0.0",
        "to_version": "1.1.0",
    }

    for exemption in (entry, {**entry, "reason": "   "}):
        report = gate.run_gate(
            bumped_repo,
            package="pkg",
            exemptions_path=_exemptions(tmp_path, exemption),
            registry=MigratorRegistry(),
        )
        assert {finding.code for finding in report.failing} == {
            "bump_without_migrator",
            "exemption_missing_reason",
        }
        assert (
            _run(bumped_repo, _exemptions(tmp_path, exemption), MigratorRegistry()) == 1
        )


def test_exemption_with_a_reason_passes_and_is_reported(
    bumped_repo: Path, tmp_path: Path
) -> None:
    exemptions = _exemptions(
        tmp_path,
        {
            "schema": "synthetic_note",
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "reason": "the bump only renames the schema document",
        },
    )
    output = tmp_path / "report.json"

    code = _run(bumped_repo, exemptions, MigratorRegistry(), "--json", str(output))

    assert code == 0
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["findings"] == [
        {
            "code": "exempted",
            "schema": "synthetic_note",
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "detail": "the bump only renames the schema document",
            "failing": False,
        }
    ]


def test_registered_migrator_with_a_fixture_pair_passes(
    bumped_repo: Path, tmp_path: Path
) -> None:
    _fixture_pair(bumped_repo, "synthetic_note/1.0.0-to-1.1.0")

    report = gate.run_gate(
        bumped_repo,
        package="pkg",
        exemptions_path=_exemptions(tmp_path),
        registry=_registry(("1.0.0", "1.1.0")),
    )

    assert report.failing == ()
    assert [(finding.code, finding.detail) for finding in report.findings] == [
        ("migrated", "1.0.0 -> 1.1.0")
    ]


def test_multi_step_chain_satisfies_a_skipped_version_bump(tmp_path: Path) -> None:
    for name in ("synthetic_note/1.0.0-to-1.0.5", "synthetic_note/1.0.5-to-1.1.0"):
        _fixture_pair(tmp_path, name)

    report = gate.evaluate(
        [gate.Bump("synthetic_note", "1.0.0", "1.1.0")],
        _registry(("1.0.0", "1.0.5"), ("1.0.5", "1.1.0")),
        (),
        repo_root=tmp_path,
    )

    assert report.failing == ()
    assert report.findings[0].detail == "1.0.0 -> 1.0.5 -> 1.1.0"


def test_every_registered_edge_must_name_an_existing_fixture_pair(
    tmp_path: Path,
) -> None:
    directory = tmp_path / MIGRATION_FIXTURE_ROOT / "synthetic_note/1.0.0-to-1.1.0"
    _write(directory / "before.json", {"schema_version": "1.0.0"})

    report = gate.evaluate((), _registry(("1.0.0", "1.1.0")), (), repo_root=tmp_path)

    assert [finding.to_dict() for finding in report.failing] == [
        {
            "code": "fixture_missing",
            "schema": "synthetic_note",
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "detail": "synthetic_note/1.0.0-to-1.1.0",
            "failing": True,
        }
    ]


def test_exemption_for_another_bump_does_not_cover_and_is_stale(
    bumped_repo: Path, tmp_path: Path
) -> None:
    exemptions = _exemptions(
        tmp_path,
        {
            "schema": "synthetic_note",
            "from_version": "1.0.0",
            "to_version": "2.0.0",
            "reason": "approved for a different bump",
        },
    )

    report = gate.run_gate(
        bumped_repo,
        package="pkg",
        exemptions_path=exemptions,
        registry=MigratorRegistry(),
    )

    assert [(finding.code, finding.failing) for finding in report.findings] == [
        ("bump_without_migrator", True),
        ("stale_exemption", False),
    ]


def test_migrator_and_exemption_together_pass_and_flag_the_exemption(
    bumped_repo: Path, tmp_path: Path
) -> None:
    _fixture_pair(bumped_repo, "synthetic_note/1.0.0-to-1.1.0")
    exemptions = _exemptions(
        tmp_path,
        {
            "schema": "synthetic_note",
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "reason": "kept until the migrator ships",
        },
    )

    report = gate.run_gate(
        bumped_repo,
        package="pkg",
        exemptions_path=exemptions,
        registry=_registry(("1.0.0", "1.1.0")),
    )

    assert report.failing == ()
    assert {finding.code for finding in report.findings} == {
        "migrated",
        "redundant_exemption",
    }


@pytest.mark.parametrize(
    "entry",
    [
        "not an object",
        {"schema": "synthetic_note", "to_version": "1.1.0", "reason": "missing"},
    ],
)
def test_malformed_exemption_entries_fail(tmp_path: Path, entry: Any) -> None:
    exemptions, findings = gate.load_exemptions(_exemptions(tmp_path, entry))

    assert exemptions == ()
    assert [finding.code for finding in findings] == ["exemption_invalid"]
    report = gate.evaluate(
        (),
        MigratorRegistry(),
        exemptions,
        repo_root=tmp_path,
        exemption_findings=findings,
    )
    assert report.failing


def test_unreadable_exemption_file_and_missing_tag_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1}))
    repo = tmp_path / "untagged"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo / SCHEMA_PATH, _schema("1.0.0"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "untagged")

    assert _run(repo, bad, MigratorRegistry(), "--base-ref", "HEAD") == 2
    assert _run(repo, _exemptions(tmp_path), MigratorRegistry()) == 2
    assert "fetch-depth: 0" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("schema", "version"),
    [
        ({"properties": {"schema_version": {"const": "1.1.0"}}}, "1.1.0"),
        (
            {"properties": {"schema_version": {"enum": ["1.10.0", "1.2.0"]}}},
            "1.10.0",
        ),
        (
            {"schema_version": 1, "properties": {"schema_version": {"const": "1.0.0"}}},
            "1.0.0",
        ),
        ({"schema_version": 2}, "2"),
        ({"$id": "https://example.org/schemas/decision-result/1.2.0"}, "1.2.0"),
        ({"title": "unversioned"}, None),
    ],
)
def test_record_version_prefers_the_record_schema_version_marker(
    schema: dict[str, Any], version: str | None
) -> None:
    assert gate.record_version(schema) == version


def test_committed_exemptions_all_carry_reasons() -> None:
    exemptions, findings = gate.load_exemptions(EXEMPTIONS)

    assert findings == ()
    assert exemptions
    assert all(len(item.reason) >= 20 for item in exemptions)


def test_release_gates_workflow_runs_the_migration_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )

    assert "Require record migrators for schema version bumps" in workflow
    assert "python scripts/release/record_migration_gate.py" in workflow


def test_gate_passes_on_this_checkout() -> None:
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

    report = gate.run_gate(ROOT, base_ref=described.stdout.strip())

    assert [finding.to_dict() for finding in report.failing] == []
