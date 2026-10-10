"""Breaking commits must be visible even when their type is not mapped."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def changelog():
    path = Path(__file__).resolve().parents[3] / "scripts/release/changelog.py"
    name = "openmed_changelog_unmapped_breaking_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", ["build", "chore", "ci", "custom"])
def test_bang_breaking_type_is_rendered(changelog, kind):
    commit = changelog.parse_conventional_commit(f"{kind}!: remove legacy behavior")
    assert commit is not None
    notes = changelog.build_release_notes("2.5.0", [commit], "2026-01-01")
    assert notes.bump == "major" and notes.next_version == "3.0.0"
    assert "### Changed" in notes.markdown
    assert "remove legacy behavior (BREAKING)" in notes.markdown
    assert "No user-facing" not in notes.markdown


@pytest.mark.parametrize("footer", ["BREAKING CHANGE:", "BREAKING-CHANGE:"])
def test_footer_breaking_type_is_rendered(changelog, footer):
    commit = changelog.parse_conventional_commit(
        "build(runtime): update support", f"{footer} remove old runtime"
    )
    assert commit is not None
    assert changelog.changelog_section_for(commit) == "Changed"
    assert "update support (BREAKING)" in changelog.render_changelog("3.0.0", [commit])


@pytest.mark.parametrize(
    "kind,section", [("feat", "Added"), ("fix", "Fixed"), ("security", "Security")]
)
def test_mapped_breaking_types_keep_their_section(changelog, kind, section):
    commit = changelog.parse_conventional_commit(f"{kind}!: change behavior")
    assert commit is not None
    assert changelog.changelog_section_for(commit) == section


def test_unmapped_nonbreaking_commit_remains_hidden(changelog):
    commit = changelog.parse_conventional_commit("chore: tidy fixture comments")
    assert commit is not None
    assert changelog.changelog_section_for(commit) is None
    assert changelog.commit_bump(commit) == "none"
