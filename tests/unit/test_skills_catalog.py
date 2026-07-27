"""Guard the shipped Agent Skills catalog under ``skills/``.

Skills are portable ``SKILL.md`` folders (the open Agent Skills standard) that
Claude Code and OpenAI Codex load. This test keeps them valid so a broken
frontmatter, a folder/name mismatch, or accidental vendor attribution can't ship.
It reuses the repo's own ``skills/build_catalog.py`` validator (standard-library
only) and additionally enforces strict-YAML parseability when PyYAML is present.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
BUILDER = SKILLS_DIR / "build_catalog.py"
INSTALLER = REPO_ROOT / "install-skills.sh"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "openmed_skills_build_catalog", BUILDER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skills_directory_exists():
    assert SKILLS_DIR.is_dir(), "skills/ directory is missing"
    assert BUILDER.exists(), "skills/build_catalog.py is missing"


def test_all_skills_validate():
    builder = _load_builder()
    skills, errors = builder.load_skills()
    assert not errors, "skill validation errors:\n" + "\n".join(errors)
    # A healthy catalog; guards against an empty/half-written checkout.
    assert len(skills) >= 50, f"expected the full skills catalog, found {len(skills)}"


def test_every_skill_has_name_and_description():
    builder = _load_builder()
    skills, _ = builder.load_skills()
    for s in skills:
        assert s["name"], f"skill missing name: {s['path']}"
        assert s["description"], f"{s['name']}: missing description"
        assert len(s["description"]) <= 1024, f"{s['name']}: description too long"


def test_frontmatter_is_strict_yaml():
    """Real agents use strict YAML; an unquoted colon in a description breaks them."""
    yaml = __import__("importlib").import_module("yaml") if _has_yaml() else None
    if yaml is None:  # pragma: no cover - environment without PyYAML
        pytest.skip("PyYAML not installed")
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        assert text.startswith("---"), f"{skill_md}: no frontmatter"
        end = text.find("\n---", 3)
        data = yaml.safe_load(text[3:end])
        assert isinstance(data, dict), f"{skill_md}: frontmatter is not a mapping"
        assert data.get("name") and data.get("description"), f"{skill_md}: missing keys"


def _has_yaml() -> bool:
    return importlib.util.find_spec("yaml") is not None


def _run_installer(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _skill_names() -> set[str]:
    return {path.parent.name for path in SKILLS_DIR.glob("*/SKILL.md")}


def test_installer_default_links_every_supported_target(tmp_path: Path) -> None:
    result = _run_installer(tmp_path)

    assert result.returncode == 0, result.stderr
    expected = _skill_names()
    for relative in (
        ".claude/skills",
        ".codex/skills",
        ".config/opencode/skills",
        ".agents/skills",
    ):
        destination = tmp_path / relative
        assert {path.name for path in destination.iterdir()} == expected
        for name in expected:
            link = destination / name
            assert link.is_symlink()
            assert link.resolve() == (SKILLS_DIR / name).resolve()

    repeated = _run_installer(tmp_path)
    assert repeated.returncode == 0, repeated.stderr
    assert "skip " not in repeated.stderr


def test_installer_preserves_existing_entries(tmp_path: Path) -> None:
    blocked_name = min(_skill_names())
    blocked = tmp_path / ".codex" / "skills" / blocked_name
    blocked.parent.mkdir(parents=True)
    blocked.write_text("user-owned\n", encoding="utf-8")

    result = _run_installer(tmp_path, "codex")

    assert result.returncode == 0, result.stderr
    assert blocked.read_text(encoding="utf-8") == "user-owned\n"
    assert not blocked.is_symlink()
    assert f"skip {blocked}" in result.stderr
    assert "1 existing entries preserved" in result.stderr


def test_catalog_installer_supports_the_same_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    builder = _load_builder()
    skills, errors = builder.load_skills()

    assert not errors
    builder.do_install("all", skills)

    expected = _skill_names()
    for relative in (
        ".claude/skills",
        ".codex/skills",
        ".config/opencode/skills",
        ".agents/skills",
    ):
        destination = tmp_path / relative
        assert {path.name for path in destination.iterdir()} == expected
