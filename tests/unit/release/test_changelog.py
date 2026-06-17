"""Release changelog generation tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
