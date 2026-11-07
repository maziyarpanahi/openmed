"""Repository file policy tests."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_tracked_dua_relation_payloads_are_detected(monkeypatch):
    monkeypatch.setattr(
        repo_policy,
        "git_ls_files",
        lambda pattern: (
            [
                "openmed/eval/fixtures/BioRED/train.xml",
                "openmed/eval/fixtures/BioRED.zip",
                "openmed/eval/fixtures/n2c2-2022/record.ann",
                "openmed/eval/fixtures/shac/record.txt",
                "openmed/eval/datasets/i2b2.py",
                "docs/n2c2.md",
            ]
            if pattern == "."
            else []
        ),
    )

    assert repo_policy.tracked_dua_relation_corpus_files() == [
        "openmed/eval/fixtures/BioRED/train.xml",
        "openmed/eval/fixtures/BioRED.zip",
        "openmed/eval/fixtures/n2c2-2022/record.ann",
        "openmed/eval/fixtures/shac/record.txt",
    ]


def test_main_fails_for_tracked_dua_relation_payload(monkeypatch, capsys):
    monkeypatch.setattr(
        repo_policy,
        "git_ls_files",
        lambda pattern: (
            ["openmed/eval/fixtures/n2c2-2018/record.ann"] if pattern == "." else []
        ),
    )
    monkeypatch.setattr(repo_policy, "git_deleted_files", lambda pattern: set())
    monkeypatch.setattr(repo_policy, "git_tracked_ignored_files", lambda: [])

    assert repo_policy.main() == 1

    captured = capsys.readouterr()
    assert "relation corpus text and annotations" in captured.err
    assert "record.ann" in captured.err


def test_current_repo_has_no_tracked_ignored_files():
    deleted_files = repo_policy.git_deleted_files(".")

    assert [
        path
        for path in repo_policy.git_tracked_ignored_files()
        if path not in deleted_files
    ] == []


def test_commit_attribution_accepts_humans_and_rejects_forbidden_metadata(subtests):
    human = "Test Maintainer <maintainer@example.org>"
    cases = [
        (human, "GitHub <noreply@github.com>", "Fix offsets\n", False),
        (human, human, "Support a new model adapter\n", False),
        ("Claude Fable 5.1 <noreply@anthropic.com>", human, "Fix offsets", True),
        (human, "OpenAI Codex <owner@example.org>", "Fix offsets", True),
        ("Copilot <198982749+Copilot@users.noreply.github.com>", human, "Fix", True),
        ("Cursor Agent <owner@example.org>", human, "Fix", True),
        (
            human,
            human,
            "Fix\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n",
            True,
        ),
        (
            human,
            human,
            "Fix\n\nco-authored-by: Other Human <other@example.org>\n",
            True,
        ),
        (human, human, "Fix\n\nCo‐Authored‐By: Hidden\n", True),
        (human, human, "Fix\n\nCo-Authored-By: Hidden\n\nMore prose\n", True),
        (human, human, "Fix\n\n🤖 Generated with an assistant\n", True),
        ("Clau\u200bde <owner@example.org>", human, "Fix", True),
    ]
    for author, committer, message, forbidden in cases:
        with subtests.test(author=author, committer=committer, message=message):
            raw = (
                f"tree {'1' * 40}\nauthor {author} 1 +0000\n"
                f"committer {committer} 1 +0000\n\n{message}"
            ).encode()
            assert bool(repo_policy.commit_metadata_errors(raw)) is forbidden
    assert repo_policy.commit_metadata_errors(b"not a commit")


@pytest.mark.slow
def test_commit_policy_checks_merged_side_history_and_fails_closed(
    tmp_path, monkeypatch
):
    """A clean tip/first-parent history must not hide an attributed side commit."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test Maintainer",
        "GIT_AUTHOR_EMAIL": "maintainer@example.org",
        "GIT_COMMITTER_NAME": "Test Maintainer",
        "GIT_COMMITTER_EMAIL": "maintainer@example.org",
    }

    def git(*args, data=None):
        return (
            subprocess.check_output(
                ["git", "-C", str(tmp_path), *args], input=data, env=env
            )
            .decode()
            .strip()
        )

    git("init", "--quiet")
    tree = git("mktree", data=b"")

    def commit(message, *parents):
        args = ["-c", "commit.gpgsign=false", "commit-tree", tree]
        for parent in parents:
            args.extend(["-p", parent])
        return git(*args, data=message.encode())

    base = commit("Initial human commit\n")
    clean = commit("Clean first parent\n", base)
    bad = commit("Fix\n\nCo-Authored-By: Assistant <noreply@anthropic.com>\n", base)
    merged = commit("Merge reviewed change\n", clean, bad)
    tip = commit("Clean follow-up\n", merged)
    monkeypatch.setattr(repo_policy, "ROOT", tmp_path)
    assert repo_policy.check_commit_range(base, clean) == 0
    assert repo_policy.check_commit_range(base, tip) == 1
    assert repo_policy.check_commit_range("missing-base", tip) == 1
    # A truncated clone must never turn an incomplete scan into a green status.
    (tmp_path / ".git" / "shallow").write_text(f"{tip}\n")
    assert repo_policy.check_commit_range(base, tip) == 1


def test_attribution_workflow_uses_trusted_code_and_reports_failures():
    workflow = yaml.safe_load((ROOT / ".github/workflows/secret-scan.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert "pull_request_target" in triggers
    assert "paths" not in triggers["pull_request_target"]
    guard = workflow["jobs"]["commit-attribution"]
    assert guard["if"] == "github.event_name != 'pull_request'"
    assert guard["permissions"] == {"contents": "read", "statuses": "write"}
    checkout = next(
        step for step in guard["steps"] if "actions/checkout@" in step.get("uses", "")
    )
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.base.sha || github.sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert (
        workflow["jobs"]["secret-scan"]["if"]
        == "github.event_name != 'pull_request_target'"
    )
    publisher = guard["steps"][-1]
    assert publisher["if"] == "always()"
    assert "state=failure" in publisher["run"]
    assert "\"$POLICY_OUTCOME\" = 'success'" in publisher["run"]
    assert "statuses/$POLICY_HEAD" in publisher["run"]
