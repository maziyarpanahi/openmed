"""Validate the consumer-facing repository skills required by OM-401."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"
EXPECTED_SKILLS = (
    "benchmark-pii-recall",
    "deidentify-a-dataset",
    "extract-clinical-entities-to-fhir",
    "pick-a-pii-model",
)
FRONTMATTER_RE = re.compile(
    r"\A---\n(?P<frontmatter>.*?)\n---\n(?P<body>.*)\Z",
    re.DOTALL,
)
LOCAL_PYTHON_LINK_RE = re.compile(r"\[[^\]]+\]\((?P<path>(?!https?://|#)[^)]+\.py)\)")


def _skill_path(name: str) -> Path:
    return SKILLS_ROOT / name / "SKILL.md"


def _read_skill(name: str) -> tuple[Path, dict[str, object], str]:
    skill_path = _skill_path(name)
    assert skill_path.is_file(), f"missing repository skill: {skill_path}"

    text = skill_path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.fullmatch(text)
    assert match is not None, f"{skill_path}: invalid frontmatter delimiters"

    frontmatter = yaml.safe_load(match.group("frontmatter"))
    assert isinstance(frontmatter, dict), f"{skill_path}: frontmatter is not a mapping"
    return skill_path, frontmatter, match.group("body")


def test_required_repo_skills_have_valid_frontmatter_and_runnable_snippets():
    for name in EXPECTED_SKILLS:
        skill_path, frontmatter, body = _read_skill(name)

        assert frontmatter.get("name") == name
        description = frontmatter.get("description")
        assert isinstance(description, str) and description.strip(), (
            f"{skill_path}: description must be a non-empty string"
        )
        assert "```python" in body, f"{skill_path}: missing runnable Python snippet"


def test_required_repo_skills_reference_existing_python_examples():
    for name in EXPECTED_SKILLS:
        skill_path, _, body = _read_skill(name)
        references = [
            match.group("path") for match in LOCAL_PYTHON_LINK_RE.finditer(body)
        ]
        assert references, f"{skill_path}: no referenced Python example"

        for reference in references:
            resolved = (skill_path.parent / reference).resolve()
            assert resolved.is_relative_to(REPO_ROOT), (
                f"{skill_path}: reference escapes repository: {reference}"
            )
            assert resolved.is_file(), (
                f"{skill_path}: referenced example does not exist: {reference}"
            )
