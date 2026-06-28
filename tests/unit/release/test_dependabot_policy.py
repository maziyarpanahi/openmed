"""Dependabot configuration policy tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEPENDABOT_CONFIG = ROOT / ".github" / "dependabot.yml"


def _github_actions_section() -> str:
    text = DEPENDABOT_CONFIG.read_text(encoding="utf-8")
    _, section = text.split('package-ecosystem: "github-actions"', 1)
    return section.split('package-ecosystem: "pre-commit"', 1)[0]


def test_github_actions_group_excludes_major_updates():
    section = _github_actions_section()

    assert "applies-to: version-updates" in section
    assert "update-types:" in section
    assert '- "minor"' in section
    assert '- "patch"' in section
    assert '- "major"' not in section
