"""Focused, offline checks for the setup-openmed policy skill."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL = REPO_ROOT / "skills" / "setup-openmed" / "SKILL.md"
TEMPLATE = (
    REPO_ROOT / "skills" / "setup-openmed" / "assets" / ("DEID-POLICY.template.md")
)
DOC = REPO_ROOT / "docs" / "agent-skills" / "setup.md"

DECISION_VALUES = {
    "jurisdiction": "eu",
    "recall_floor": "0.99",
    "surrogate_strategy": "mask",
    "model_policy": "local-preinstalled",
    "audit_location": "separate-local-directory",
}
PLACEHOLDER_RE = re.compile(r"{{\s*(\w+)\s*}}")


def _read(path: Path) -> str:
    assert path.is_file(), f"missing expected file: {path}"
    return path.read_text(encoding="utf-8")


def _frontmatter(text: str) -> tuple[str, str]:
    assert text.startswith("---\n")
    end = text.find("\n---\n", 4)
    assert end != -1
    return text[4:end], text[end + len("\n---\n") :]


def _render(template: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        assert key in values, f"unexpected template placeholder: {key}"
        return values[key]

    return PLACEHOLDER_RE.sub(replace, template)


def test_setup_skill_has_valid_frontmatter_and_bounded_contract() -> None:
    frontmatter, body = _frontmatter(_read(SKILL))

    assert "name: setup-openmed" in frontmatter
    assert "description:" in frontmatter
    for phrase in (
        "bounded",
        "jurisdiction",
        "recall floor",
        "surrogate strategy",
        "model policy",
        "audit location",
        "no mandatory network call",
        "human approval",
    ):
        assert phrase in (frontmatter + body).lower()

    for value in (
        "us",
        "eu",
        "canada",
        "research",
        "organization-defined",
        "0.90",
        "0.95",
        "0.99",
        "mask",
        "remove",
        "replace",
        "hash",
        "local-preinstalled",
        "local-user-supplied",
        "rules-only",
        "separate-local-directory",
        "controlled-artifact-store",
        "no-retention",
    ):
        assert f"`{value}`" in body


def test_skill_links_to_local_template_and_requires_safe_writing() -> None:
    body = _read(SKILL)

    assert "(assets/DEID-POLICY.template.md)" in body
    assert "DEID-POLICY.md" in body
    assert re.search(r"Do not accept a\s+free-form value", body)
    assert "never echo the" in body.lower()
    assert "Do not add a timestamp" in body
    assert "byte-for-byte identical" in body
    assert "DRAFT — HUMAN APPROVAL REQUIRED" in body
    assert "not a compliance certification" in body

    linked = (SKILL.parent / "assets/DEID-POLICY.template.md").resolve()
    assert linked == TEMPLATE.resolve()
    assert linked.is_file()


def test_template_is_versioned_and_contains_only_decision_placeholders() -> None:
    template = _read(TEMPLATE)
    placeholders = set(PLACEHOLDER_RE.findall(template))

    assert "Template version: 1.0" in template
    assert "Policy schema: 1" in template
    assert placeholders == set(DECISION_VALUES)
    assert "DRAFT — HUMAN APPROVAL REQUIRED" in template
    assert "Human approval:** `PENDING`" in template
    assert "not a compliance certification" in template
    assert "guarantee" in template
    assert "raw source values" in template


def test_rendering_is_deterministic_and_does_not_copy_payloads() -> None:
    template = _read(TEMPLATE)

    rendered_once = _render(template, DECISION_VALUES)
    rendered_twice = _render(template, DECISION_VALUES)

    assert rendered_once == rendered_twice
    assert not PLACEHOLDER_RE.search(rendered_once)
    assert all(value in rendered_once for value in DECISION_VALUES.values())
    assert "source data" in rendered_once
    assert "download a model" in rendered_once
    assert "remote service" in rendered_once


def test_setup_document_explains_local_artifact_and_approval_gate() -> None:
    document = _read(DOC)

    for target in (
        "../../skills/setup-openmed/SKILL.md",
        "../../skills/setup-openmed/assets/DEID-POLICY.template.md",
    ):
        assert f"]({target})" in document
        assert (DOC.parent / target).resolve().is_file()

    lowered = document.lower()
    assert "no mandatory network call" in lowered
    assert "deid-policy.md" in lowered
    assert "human approval" in lowered
    assert re.search(r"raw\s+sensitive\s+values", lowered)
