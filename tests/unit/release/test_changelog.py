"""Release changelog generation tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CHANGELOG_SCRIPT = ROOT / "scripts" / "release" / "changelog.py"

spec = importlib.util.spec_from_file_location("release_changelog", CHANGELOG_SCRIPT)
release_changelog = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = release_changelog
spec.loader.exec_module(release_changelog)


def parse(subject: str, body: str = ""):
    parsed = release_changelog.parse_conventional_commit(subject, body)
    assert parsed is not None
    return parsed


def test_renders_changelog_section_grouped_by_change_type():
    commits = [
        parse("feat(api): add batch exports"),
        parse("fix: repair tag comparison"),
        parse("deprecated(cli): deprecate manual release flag"),
        parse("security: harden release token handling"),
    ]

    notes = release_changelog.build_release_notes("1.5.5", commits, "2026-06-14")

    assert notes.bump == "minor"
    assert notes.next_version == "1.6.0"
    assert notes.markdown == (
        "## [1.6.0] - 2026-06-14\n"
        "\n"
        "### Added\n"
        "\n"
        "- **api:** add batch exports\n"
        "\n"
        "### Deprecated\n"
        "\n"
        "- **cli:** deprecate manual release flag\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- repair tag comparison\n"
        "\n"
        "### Security\n"
        "\n"
        "- harden release token handling\n"
    )


def test_fix_only_commits_compute_patch_bump():
    commits = [parse("fix: repair changelog ordering")]

    notes = release_changelog.build_release_notes("1.5.5", commits, "2026-06-14")

    assert notes.bump == "patch"
    assert notes.next_version == "1.5.6"


def test_feat_commit_computes_minor_bump():
    commits = [
        parse("fix: repair changelog ordering"),
        parse("feat(release): expose computed version"),
    ]

    notes = release_changelog.build_release_notes("1.5.5", commits, "2026-06-14")

    assert notes.bump == "minor"
    assert notes.next_version == "1.6.0"


def test_breaking_change_footer_computes_major_bump():
    commits = [
        parse(
            "fix(api): require explicit release version",
            "BREAKING CHANGE: release callers must pass an explicit version.",
        ),
        parse("feat(release): expose computed version"),
    ]

    notes = release_changelog.build_release_notes("1.5.5", commits, "2026-06-14")

    assert notes.bump == "major"
    assert notes.next_version == "2.0.0"
    assert "- **api:** require explicit release version (BREAKING)" in notes.markdown


def test_bang_subject_computes_major_bump():
    commits = [parse("feat!: remove legacy release command")]

    notes = release_changelog.build_release_notes("1.5.5", commits, "2026-06-14")

    assert notes.bump == "major"
    assert notes.next_version == "2.0.0"


def test_release_version_accepts_intentional_major_bump_above_minimum():
    release_changelog.validate_release_version("1.9.1", "1.10.0", "2.0.0")


def test_release_version_rejects_stale_or_insufficient_bump(subtests):
    for current_version, minimum_version, release_version in [
        ("1.9.1", "1.10.0", "1.9.2"),
        ("1.9.1", "1.9.2", "1.9.1"),
    ]:
        with subtests.test(
            current_version=current_version,
            minimum_version=minimum_version,
            release_version=release_version,
        ):
            with pytest.raises(ValueError):
                release_changelog.validate_release_version(
                    current_version,
                    minimum_version,
                    release_version,
                )


def test_writes_github_outputs(tmp_path):
    output_file = tmp_path / "github-output.txt"
    release_changelog.write_github_output(
        {
            "bump": "minor",
            "next_version": "1.6.0",
            "changelog": "## [1.6.0] - 2026-06-14\n\n### Added\n\n- add release metadata\n",
        },
        output_file,
    )

    assert output_file.read_text(encoding="utf-8") == (
        "bump=minor\n"
        "next_version=1.6.0\n"
        "changelog<<__OPENMED_CHANGELOG__\n"
        "## [1.6.0] - 2026-06-14\n"
        "\n"
        "### Added\n"
        "\n"
        "- add release metadata\n"
        "__OPENMED_CHANGELOG__\n"
    )


@pytest.fixture
def lts_repo(tmp_path):
    def git(*args):
        result = release_changelog.run_git(tmp_path, list(args))
        assert result.returncode == 0, result.stderr

    git("init")
    git("config", "user.name", "Test Contributor")
    git("config", "user.email", "test@example.invalid")
    git("commit", "--allow-empty", "-m", "fix: baseline")
    git("tag", "v4.3.2")
    git("tag", "v4.3.10")
    git("tag", "v4.3.11rc1")
    git("tag", "v6.2.0")
    return tmp_path, git


def test_lts_uses_highest_stable_patch_on_selected_line(lts_repo, capsys):
    repo, git = lts_repo
    git("commit", "--allow-empty", "-m", "fix: repair recall backstop")
    assert (
        release_changelog.main(
            ["--repo", str(repo), "--lts-line", "4.3", "--format", "json"]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["current_version"] == "4.3.10"
    assert report["next_version"] == "4.3.11"
    assert report["bump"] == "patch"


@pytest.mark.parametrize(
    "subject,body",
    [
        ("feat: new feature", ""),
        ("fix!: breaking fix", ""),
        ("chore: changed contract", "BREAKING CHANGE: remove old API"),
    ],
)
def test_lts_refuses_and_names_offending_commits(lts_repo, capsys, subject, body):
    repo, git = lts_repo
    git("commit", "--allow-empty", "-m", subject, "-m", body)
    assert release_changelog.main(["--repo", str(repo), "--lts-line", "4.3"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert subject in captured.err


@pytest.mark.parametrize(
    "extra",
    [
        ["--release-version", "5.0.0"],
        ["--range", "HEAD"],
        ["--current-version", "6.2.0"],
    ],
)
def test_lts_rejects_overrides_that_bypass_the_line(lts_repo, capsys, extra):
    repo, git = lts_repo
    git("commit", "--allow-empty", "-m", "fix: patch")
    assert (
        release_changelog.main(["--repo", str(repo), "--lts-line", "4.3", *extra]) == 2
    )
    assert not capsys.readouterr().out


@pytest.mark.parametrize("line", ["", "04.3", "4.3.0", "4.*", "9.9"])
def test_lts_rejects_invalid_or_missing_line(lts_repo, capsys, line):
    repo, _ = lts_repo
    assert release_changelog.main(["--repo", str(repo), "--lts-line", line]) == 2
    assert not capsys.readouterr().out
