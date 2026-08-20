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
    expected_skills = sum(
        1
        for path in (REPO_ROOT / "skills").iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )
    assert report.skill_count == expected_skills
    assert report.link_count > 0
    assert report.helper_count == len(validator._executable_helpers(REPO_ROOT))


def test_interpreter_helpers_do_not_depend_on_posix_mode_bits(tmp_path: Path) -> None:
    validator = _load_validator()
    python_helper = tmp_path / "skills" / "helper.py"
    shell_helper = tmp_path / "scripts" / "skills" / "helper.sh"
    python_helper.parent.mkdir(parents=True)
    shell_helper.parent.mkdir(parents=True)
    python_helper.write_text("print('synthetic')\n", encoding="utf-8")
    shell_helper.write_text("printf 'synthetic\\n'\n", encoding="utf-8")

    assert validator._executable_helpers(tmp_path) == [
        python_helper,
        shell_helper,
    ]


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


def test_missing_image_reference_is_validated(tmp_path: Path) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body="\n![synthetic diagram](assets/missing.png)\n",
        pack_entries=["./skills/synthetic-check"],
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)

    assert not report.ok
    assert any("internal link target is missing" in error for error in report.errors)


def test_malformed_link_target_is_a_path_only_error(tmp_path: Path) -> None:
    validator = _load_validator()
    secret_marker = "synthetic-sensitive-value"
    _write_fixture_repo(
        tmp_path,
        body=f"\n[malformed](//[{secret_marker})\n",
        pack_entries=["./skills/synthetic-check"],
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "internal link target is invalid" in output
    assert secret_marker not in output


def test_catalog_infrastructure_directory_is_not_treated_as_a_skill(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body="\n# Synthetic fixture\n",
        pack_entries=["./skills/synthetic-check"],
    )
    packs = tmp_path / "skills" / "packs"
    packs.mkdir()
    (packs / "manifest.json").write_text("{}\n", encoding="utf-8")

    report = validator.validate_repository(tmp_path, run_helper_help=False)

    assert report.ok, validator.format_report(report)
    assert report.skill_count == 1


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


def test_helper_environment_drops_ambient_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validator = _load_validator()
    secret_marker = "synthetic-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret_marker)
    monkeypatch.setenv("GITHUB_TOKEN", secret_marker)

    env = validator._helper_environment(tmp_path)

    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert secret_marker not in env.values()
    assert env["HOME"] == str(tmp_path)
    assert env["OPENMED_OFFLINE"] == "1"
    assert env["PIP_NO_INDEX"] == "1"


def test_pack_builder_uses_its_existing_focused_test() -> None:
    validator = _load_validator()
    helper = REPO_ROOT / "scripts" / "skills" / "build_packs.py"

    assert validator._focused_test_for_helper(REPO_ROOT, helper) == Path(
        "tests/unit/skills/test_packs.py"
    )
