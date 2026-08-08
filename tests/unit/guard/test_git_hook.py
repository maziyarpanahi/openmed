"""Focused tests for the local pre-push privacy scanner."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path

import pytest

from openmed.guard.git_hook import (
    ALLOWLIST_VERSION,
    Finding,
    PrivacyScanError,
    ScanResult,
    changed_paths,
    format_report,
    load_allowlist,
    main,
    parse_pre_push_updates,
    scan_commit_ranges,
    scan_pushed_updates,
    scan_text,
)
from scripts.install_privacy_hook import HOOK_MARKER, install_hook


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Synthetic Test")
    _git(repo, "config", "user.email", "guard" + "@" + "example.test")


def test_scan_text_reports_categories_without_matched_values() -> None:
    mail_value = "person" + "@" + "clinic.local"
    telephone_value = "555" + "-010" + "-0123"
    government_id = "321" + "-45-" + "6789"
    token = "sk-" + ("x" * 24)
    name_key = "patient" + "_name"
    raw_text_key = "raw" + "_text"
    name_value = "Ada " + "Lovelace"
    raw_text_value = "clinical " + "note content"
    text = "\n".join(
        (
            mail_value,
            telephone_value,
            government_id,
            token,
            '{"' + name_key + '": "' + name_value + '"}',
            '{"' + raw_text_key + '": "' + raw_text_value + '"}',
        )
    )

    findings = scan_text(text, path="tests/fixtures/note.json")
    categories = {finding.category for finding in findings}
    report = format_report(
        ScanResult(
            findings=findings,
            scanned_files=("tests/fixtures/note.json",),
            skipped_files=(),
        )
    )

    assert {
        "email",
        "phone",
        "government_id",
        "secret",
        "name",
        "raw_text",
    } <= categories
    assert mail_value not in report
    assert telephone_value not in report
    assert government_id not in report
    assert token not in report
    assert "tests/fixtures/note.json" in report
    assert "email" in report
    assert "secret" in report


def test_reserved_synthetic_values_are_allowlisted() -> None:
    text = "\n".join(
        (
            "fixture" + "@" + "example.test",
            "555" + "-0100",
            "192.0.2.25",
            "000" + "-00-" + "0000",
        )
    )

    assert scan_text(text, path="tests/fixtures/synthetic.txt") == ()


def test_versioned_allowlist_extension_is_narrow_and_deterministic(
    tmp_path: Path,
) -> None:
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(
        json.dumps(
            {
                "version": ALLOWLIST_VERSION,
                "entries": [
                    {
                        "path": "tests/fixtures/custom.txt",
                        "category": "email",
                        "pattern": r"person@clinic\.local",
                        "reason": "synthetic fixture used by the focused test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    allowlist = load_allowlist(allowlist_path)
    mail_value = "person" + "@" + "clinic.local"

    assert (
        scan_text(mail_value, path="tests/fixtures/custom.txt", allowlist=allowlist)
        == ()
    )
    assert scan_text(mail_value, path="docs/custom.txt", allowlist=allowlist)


def test_allowlist_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"version": ALLOWLIST_VERSION + 1, "entries": []}))

    with pytest.raises(PrivacyScanError, match="version"):
        load_allowlist(path)


def test_changed_paths_select_only_added_and_modified_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "unchanged.txt").write_text("base\n", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("removed\n", encoding="utf-8")
    _git(tmp_path, "add", "unchanged.txt", "deleted.txt")
    _git(tmp_path, "commit", "--quiet", "--message", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "unchanged.txt").write_text("changed\n", encoding="utf-8")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "added.txt").write_text("new\n", encoding="utf-8")
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "--quiet", "--message", "candidate files")
    head = _git(tmp_path, "rev-parse", "HEAD")

    assert changed_paths(tmp_path, base, head) == ("added.txt", "unchanged.txt")


def test_pre_push_updates_scan_the_pushed_range(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "note.txt").write_text("safe\n", encoding="utf-8")
    _git(tmp_path, "add", "note.txt")
    _git(tmp_path, "commit", "--quiet", "--message", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    mail_value = "person" + "@" + "clinic.local"
    (tmp_path / "note.txt").write_text(mail_value + "\n", encoding="utf-8")
    _git(tmp_path, "add", "note.txt")
    _git(tmp_path, "commit", "--quiet", "--message", "privacy candidate")
    head = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "note.txt").write_text("working tree only\n", encoding="utf-8")

    updates = parse_pre_push_updates(
        [f"refs/heads/topic {head} refs/heads/topic {base}\n"]
    )
    result = scan_pushed_updates(tmp_path, updates)
    range_result = scan_commit_ranges(tmp_path, [f"{base}..{head}"])

    assert result.passed is False
    assert result.files == {"note.txt": {"email": 1}}
    assert range_result.files == {"note.txt": {"email": 1}}


def test_new_branch_pre_push_update_uses_empty_tree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "note.txt").write_text("safe\n", encoding="utf-8")
    _git(tmp_path, "add", "note.txt")
    _git(tmp_path, "commit", "--quiet", "--message", "first")
    head = _git(tmp_path, "rev-parse", "HEAD")

    updates = parse_pre_push_updates(
        [f"refs/heads/topic {head} refs/heads/topic {'0' * 40}\n"]
    )
    result = scan_pushed_updates(tmp_path, updates)

    assert result.passed
    assert result.scanned_files == ("note.txt",)


def test_pre_push_input_rejects_malformed_records() -> None:
    with pytest.raises(PrivacyScanError, match="malformed"):
        parse_pre_push_updates(["not a valid update\n"])


def test_cli_returns_nonzero_and_keeps_report_value_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mail_value = "person" + "@" + "clinic.local"
    (tmp_path / "candidate.txt").write_text(mail_value, encoding="utf-8")

    exit_code = main(
        ["--repo", str(tmp_path), "--path", "candidate.txt"],
        stdin=StringIO(),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert mail_value not in captured.out + captured.err
    assert "candidate.txt" in captured.err
    assert "email" in captured.err


def test_installer_preserves_existing_hook(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    hooks = tmp_path / ".git" / "hooks"
    existing = hooks / "pre-push"
    existing.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    existing.chmod(0o755)

    installed = install_hook(tmp_path, python_executable="/usr/bin/python3")

    assert installed == hooks / "pre-push"
    assert HOOK_MARKER in installed.read_text(encoding="utf-8")
    assert (hooks / "pre-push.openmed-original").read_text(encoding="utf-8") == (
        "#!/bin/sh\nexit 7\n"
    )
    assert installed.stat().st_mode & 0o111


def test_value_free_finding_has_no_value_field() -> None:
    finding = Finding("note.txt", "email", 2)

    assert finding.path == "note.txt"
    assert finding.category == "email"
    assert finding.line == 2
    assert "value" not in finding.__dict__
