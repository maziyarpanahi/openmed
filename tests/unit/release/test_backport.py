"""Offline backport checks in a temporary Git repository."""

import subprocess

import pytest

from scripts.release.backport import BackportError, backport


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-global-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git(tmp_path, "init", "-q", "-b", "master")
    git(tmp_path, "config", "user.name", "Test Maintainer")
    git(tmp_path, "config", "user.email", "maintainer@example.test")
    package = tmp_path / "openmed"
    package.mkdir()
    (package / "__init__.py").write_text("def existing():\n    return 1\n")
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n### Fixed\n\n- Earlier fix.\n"
    )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    git(tmp_path, "tag", "v4.3.0")
    git(tmp_path, "branch", "release/4.3")
    (package / "__init__.py").write_text("def existing():\n    return 2\n")
    with (tmp_path / "CHANGELOG.md").open("a") as f:
        f.write("- Fix observed value.\n")
    git(tmp_path, "commit", "-qam", "fix observed value")
    return tmp_path


def eligible(*args, **kwargs):
    return [{"merged_at": "2026-11-01", "labels": [{"name": "backport-4.3"}]}]


def test_dry_run_and_local_apply(repo):
    head = git(repo, "rev-parse", "HEAD")
    base = git(repo, "rev-parse", "release/4.3")
    args = dict(target="release/4.3", commits=[head], lookup=eligible)
    plan = backport(repo, **args)
    assert plan == {
        "target": "release/4.3",
        "base": base,
        "lts_tag": "v4.3.0",
        "commits": [head],
        "applied": False,
    }
    assert git(repo, "rev-parse", "release/4.3") == base
    assert git(repo, "branch", "--show-current") == "master"
    git(repo, "checkout", "-q", "release/4.3")
    assert backport(repo, apply=True, **args)["applied"]
    assert f"(cherry picked from commit {head})" in git(
        repo, "log", "-1", "--format=%B"
    )
    assert git(repo, "remote") == ""


def test_pr_lookup_and_policy_refusals(repo):
    head = git(repo, "rev-parse", "HEAD")

    def lookup(endpoint, **opts):
        if endpoint == "pulls/12":
            return eligible()[0]
        assert endpoint == "pulls/12/commits?per_page=100" and opts["paginate"]
        return [{"sha": head}]

    assert backport(repo, target="release/4.3", pr=12, lookup=lookup)["commits"] == [
        head
    ]
    with pytest.raises(BackportError, match="eligible_merged_pr"):
        backport(repo, target="release/4.3", commits=[head], lookup=lambda *a, **k: [])
    for file, extra, reason in [
        ("openmed/__init__.py", "\ndef new_api():\n    pass\n", "public_api_added"),
        ("CHANGELOG.md", "\n### Added\n- Feature.\n", "changelog_section_refused"),
    ]:
        with (repo / file).open("a") as stream:
            stream.write(extra)
        git(repo, "commit", "-qam", "amend candidate", "--amend")
        with pytest.raises(BackportError, match=reason):
            backport(repo, target="release/4.3", commits=["HEAD"], lookup=eligible)


def test_forbidden_attribution_and_dirty_checkout(repo):
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "release/4.3")
    (repo / "untracked.txt").write_text("local work")
    with pytest.raises(BackportError, match="clean_target_checkout"):
        backport(
            repo, target="release/4.3", commits=[head], apply=True, lookup=eligible
        )
    assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", "v4.3.0")
    git(repo, "checkout", "-q", "master")
    git(
        repo,
        "commit",
        "--amend",
        "-qm",
        "fix\n\nCo-authored-by: Extra <extra@example.test>",
    )
    with pytest.raises(BackportError, match="attribution_refused"):
        backport(repo, target="release/4.3", commits=["HEAD"], lookup=eligible)
