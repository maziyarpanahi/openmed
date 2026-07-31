"""Repository file policy tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_repo_policy.py"

spec = importlib.util.spec_from_file_location("check_repo_policy", SCRIPT)
assert spec is not None
repo_policy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = repo_policy
spec.loader.exec_module(repo_policy)


def test_tracked_ignored_files_use_standard_ignore_rules(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        assert kwargs["cwd"] == repo_policy.ROOT
        assert kwargs["text"] is True
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["check"] is False
        return SimpleNamespace(
            returncode=0,
            stdout="PLANS/V2/example.md\n",
            stderr="",
        )

    monkeypatch.setattr(repo_policy.subprocess, "run", fake_run)

    assert repo_policy.git_tracked_ignored_files() == ["PLANS/V2/example.md"]
    assert calls == [["git", "ls-files", "--cached", "--ignored", "--exclude-standard"]]


def test_main_fails_for_tracked_ignored_files(monkeypatch, capsys):
    monkeypatch.setattr(repo_policy, "git_ls_files", lambda pattern: [])
    monkeypatch.setattr(repo_policy, "git_deleted_files", lambda pattern: set())
    monkeypatch.setattr(
        repo_policy,
        "git_tracked_ignored_files",
        lambda: ["PLANS/V2/example.md"],
    )

    assert repo_policy.main() == 1

    captured = capsys.readouterr()
    assert "tracked ignored files are not allowed" in captured.err
    assert "PLANS/V2/example.md" in captured.err


def test_current_repo_has_no_tracked_ignored_files():
    deleted_files = repo_policy.git_deleted_files(".")

    assert [
        path
        for path in repo_policy.git_tracked_ignored_files()
        if path not in deleted_files
    ] == []
