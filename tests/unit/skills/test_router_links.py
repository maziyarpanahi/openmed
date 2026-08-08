"""Validate the OpenMed workflow router's local skill links and guardrails."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTER = REPO_ROOT / "skills" / "ask-openmed" / "SKILL.md"
DOC = REPO_ROOT / "docs" / "agent-skills" / "router.md"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _local_links(path: Path) -> list[Path]:
    links: list[Path] = []
    for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("#", "http://", "https://")):
            continue
        links.append((path.parent / target).resolve())
    return links


def test_router_links_resolve_to_existing_skill_identifiers() -> None:
    """Every local router link must target a shipped ``SKILL.md``."""
    for source in (ROUTER, DOC):
        links = _local_links(source)
        assert links, f"{source} has no local skill links"
        for target in links:
            assert target.name == "SKILL.md", f"unexpected router target: {target}"
            assert target.is_file(), f"router link does not exist: {target}"
            assert target.parent.parent == REPO_ROOT / "skills", (
                f"router link escapes the skills catalog: {target}"
            )


def test_router_has_all_goal_sections_and_a_privacy_override() -> None:
    """The route map stays organized around the five supported goal families."""
    body = ROUTER.read_text(encoding="utf-8")
    for section in ("Intake", "Privacy", "Extraction", "Exchange", "Verification"):
        assert f"## {section}" in body

    assert "privacy override" in body.lower()
    lowered = body.lower()
    assert "no mandatory" in lowered and "network call" in lowered
    assert "Do not inspect or copy the data payload" in body
    assert "Never echo the request" in body


def test_router_examples_escalate_ambiguous_sensitive_work() -> None:
    """Unstated sensitivity must route through the de-identification gate."""
    body = ROUTER.read_text(encoding="utf-8")
    assert "The sensitivity is unstated" in body
    assert (
        "[deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md)" in body
    )
    assert "explicitly" in body and "synthetic" in body
