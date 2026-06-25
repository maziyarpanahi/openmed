"""Guard the shipped Agent Skills catalog under ``skills/``.

Skills are portable ``SKILL.md`` folders (the open Agent Skills standard) that
Claude Code and OpenAI Codex load. This test keeps them valid so a broken
frontmatter, a folder/name mismatch, or accidental vendor attribution can't ship.
It reuses the repo's own ``skills/build_catalog.py`` validator (standard-library
only) and additionally enforces strict-YAML parseability when PyYAML is present.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
BUILDER = SKILLS_DIR / "build_catalog.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("openmed_skills_build_catalog", BUILDER)
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
        import pytest

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
