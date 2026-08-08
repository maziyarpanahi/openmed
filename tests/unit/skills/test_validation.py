"""Focused offline tests for the repository Agent Skills validation gate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "scripts/skills/validate.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("skills_validation", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture_repo(
    root: Path,
    *,
    body: str,
    pack_entries: list[str],
    frontmatter: str | None = None,
) -> None:
    skill_dir = root / "skills" / "synthetic-check"
    skill_dir.mkdir(parents=True)
    frontmatter = frontmatter or (
        "---\n"
        "name: synthetic-check\n"
        'description: "Synthetic offline validation fixture."\n'
        "---\n"
    )
    (skill_dir / "SKILL.md").write_text(
        frontmatter + body,
        encoding="utf-8",
    )
    marketplace = root / ".claude-plugin"
    marketplace.mkdir()
    (marketplace / "marketplace.json").write_text(
        json.dumps({"plugins": [{"skills": pack_entries}]}),
        encoding="utf-8",
    )


def test_repository_catalog_passes_the_offline_gate() -> None:
    validator = _load_validator()

    report = validator.validate_repository(REPO_ROOT)

    assert report.ok, validator.format_report(report)
    assert report.skill_count == 72
    assert report.link_count == 13
    assert report.helper_count >= 2


def test_missing_reference_and_pack_membership_are_path_based(tmp_path: Path) -> None:
    validator = _load_validator()
    secret_marker = "synthetic-sensitive-value"
    _write_fixture_repo(
        tmp_path,
        body=f"\n[local reference]({secret_marker}.md)\n",
        pack_entries=[],
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "skills/synthetic-check/SKILL.md" in output
    assert "internal link target is missing" in output
    assert "skill is not present in a pack" in output
    assert secret_marker not in output


def test_invalid_frontmatter_does_not_echo_raw_values(tmp_path: Path) -> None:
    validator = _load_validator()
    secret_marker = "synthetic-frontmatter-value"
    _write_fixture_repo(
        tmp_path,
        body="\n# Synthetic fixture\n",
        pack_entries=["./skills/synthetic-check"],
        frontmatter=(f"---\ndescription: [{secret_marker}\n---\n"),
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "frontmatter is not valid YAML" in output
    assert secret_marker not in output


def test_validator_help_command_is_successful_and_local() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "without network access" in result.stdout
