"""Focused offline tests for the repository Agent Skills validation gate."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = REPO_ROOT / "scripts/skills/validate.py"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/skills.yml"


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


@pytest.mark.parametrize(
    "target",
    (
        "file:///synthetic-sensitive-value",
        "C:\\synthetic-sensitive-value\\policy.md",
        "..\\synthetic-sensitive-value.md",
    ),
)
def test_local_link_bypasses_are_rejected_without_echoing_targets(
    tmp_path: Path,
    target: str,
) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body=f"\n[unsafe local reference]({target})\n",
        pack_entries=["./skills/synthetic-check"],
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "internal link target is invalid" in output
    assert "synthetic-sensitive-value" not in output


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


def test_oversized_skill_is_rejected_before_parsing(tmp_path: Path) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body="\n" + ("x" * validator.MAX_SKILL_BYTES),
        pack_entries=["./skills/synthetic-check"],
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)

    assert not report.ok
    assert any("skill file exceeds size limit" in error for error in report.errors)


def test_duplicate_marketplace_keys_are_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body="\n# Synthetic fixture\n",
        pack_entries=["./skills/synthetic-check"],
    )
    (tmp_path / ".claude-plugin" / "marketplace.json").write_text(
        '{"plugins": [], "plugins": []}\n',
        encoding="utf-8",
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)

    assert not report.ok
    assert any("manifest is invalid JSON" in error for error in report.errors)


def test_external_display_paths_use_a_fixed_placeholder(tmp_path: Path) -> None:
    validator = _load_validator()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external = tmp_path / "synthetic-sensitive-directory" / "SKILL.md"

    displayed = validator._display_path(repo_root, external)

    assert displayed == "<outside-repository>"
    assert str(tmp_path) not in displayed


@pytest.mark.skipif(
    os.name == "nt",
    reason="file symlinks require elevated privileges on Windows",
)
def test_symlinked_skill_file_is_rejected_without_reading_target(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    skill_dir = tmp_path / "skills" / "synthetic-check"
    skill_dir.mkdir(parents=True)
    outside = tmp_path / "synthetic-sensitive-target.md"
    outside.write_text(
        "---\nname: synthetic-check\ndescription: leaked\n---\n# Secret\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").symlink_to(outside)
    marketplace = tmp_path / ".claude-plugin"
    marketplace.mkdir()
    (marketplace / "marketplace.json").write_text(
        json.dumps({"plugins": [{"skills": ["./skills/synthetic-check"]}]}),
        encoding="utf-8",
    )

    report = validator.validate_repository(tmp_path, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "SKILL.md must not be a symlink" in output
    assert "synthetic-sensitive-target" not in output
    assert str(tmp_path) not in output


@pytest.mark.skipif(
    os.name == "nt",
    reason="file symlinks require elevated privileges on Windows",
)
def test_extensionless_symlinked_helper_is_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body="\n# Synthetic fixture\n",
        pack_entries=["./skills/synthetic-check"],
    )
    target = tmp_path / "synthetic-sensitive-helper-target"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o700)
    helper = tmp_path / "scripts" / "skills" / "linked-helper"
    helper.parent.mkdir(parents=True)
    helper.symlink_to(target)

    report = validator.validate_repository(tmp_path, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "executable helper must not be a symlink" in output
    assert "synthetic-sensitive-helper-target" not in output


@pytest.mark.skipif(
    os.name == "nt",
    reason="directory symlinks require elevated privileges on Windows",
)
def test_unresolvable_repository_root_returns_a_fixed_error(tmp_path: Path) -> None:
    validator = _load_validator()
    loop = tmp_path / "synthetic-sensitive-loop"
    loop.symlink_to(loop, target_is_directory=True)

    report = validator.validate_repository(loop, run_helper_help=False)
    output = validator.format_report(report)

    assert not report.ok
    assert "repository root cannot be resolved" in output
    assert "synthetic-sensitive-loop" not in output
    assert str(tmp_path) not in output


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
    monkeypatch.setenv("TEMP", secret_marker)
    monkeypatch.setenv("TMP", secret_marker)
    monkeypatch.setenv("TMPDIR", secret_marker)

    env = validator._helper_environment(tmp_path)

    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert secret_marker not in env.values()
    assert env["HOME"] == str(tmp_path)
    assert env["TEMP"] == str(tmp_path)
    assert env["TMP"] == str(tmp_path)
    assert env["TMPDIR"] == str(tmp_path)
    assert env["OPENMED_OFFLINE"] == "1"
    assert env["PIP_NO_INDEX"] == "1"


def test_pack_builder_uses_its_existing_focused_test() -> None:
    validator = _load_validator()
    helper = REPO_ROOT / "scripts" / "skills" / "build_packs.py"

    assert validator._focused_test_for_helper(REPO_ROOT, helper) == Path(
        "tests/unit/skills/test_packs.py"
    )


@pytest.mark.parametrize(
    ("test_source", "expected_error"),
    (
        (
            "def test_helper_contract():\n    pass\n",
            "focused helper test has no assertion",
        ),
        (
            "def test_unrelated_contract():\n    assert True\n",
            "focused helper test does not reference helper",
        ),
    ),
)
def test_helper_tests_require_assertions_and_a_helper_reference(
    tmp_path: Path,
    test_source: str,
    expected_error: str,
) -> None:
    validator = _load_validator()
    _write_fixture_repo(
        tmp_path,
        body="\n# Synthetic fixture\n",
        pack_entries=["./skills/synthetic-check"],
    )
    helper = tmp_path / "scripts" / "skills" / "helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("def main():\n    return 0\n", encoding="utf-8")
    test_path = tmp_path / "tests" / "unit" / "skills" / "test_helper.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text(test_source, encoding="utf-8")

    report = validator.validate_repository(tmp_path, run_helper_help=False)

    assert not report.ok
    assert any(expected_error in error for error in report.errors)


def test_workflow_runs_every_focused_skill_test() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python -m pytest tests/unit/skills -q" in workflow
