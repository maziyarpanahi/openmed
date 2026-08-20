"""Validate the OpenMed workflow router's local skill links and guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTER = REPO_ROOT / "skills" / "ask-openmed" / "SKILL.md"
DOC = REPO_ROOT / "docs" / "agent-skills" / "router.md"
CATALOG = REPO_ROOT / "skills" / "README.md"
MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
GITHUB_SKILLS_PREFIX = "https://github.com/maziyarpanahi/openmed/blob/master/skills/"


def _skill_links(path: Path) -> list[tuple[str, Path]]:
    links: list[tuple[str, Path]] = []
    for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        if target.startswith("#"):
            continue
        if target.startswith(GITHUB_SKILLS_PREFIX):
            relative = target.removeprefix(GITHUB_SKILLS_PREFIX)
            resolved = (REPO_ROOT / "skills" / relative).resolve()
        else:
            assert not target.startswith(("http://", "https://")), (
                f"noncanonical router URL: {target}"
            )
            resolved = (path.parent / target).resolve()
        links.append((target, resolved))
    return links


def test_router_links_resolve_to_existing_skill_identifiers() -> None:
    """Every local router link must target a shipped ``SKILL.md``."""
    for source in (ROUTER, DOC):
        links = _skill_links(source)
        assert links, f"{source} has no local skill links"
        for raw_target, target in links:
            assert target.name == "SKILL.md", f"unexpected router target: {target}"
            assert target.is_file(), f"router link does not exist: {target}"
            assert target.parent.parent == REPO_ROOT / "skills", (
                f"router link escapes the skills catalog: {target}"
            )
            if source == DOC:
                assert raw_target.startswith(GITHUB_SKILLS_PREFIX), (
                    f"docs link must use the canonical GitHub URL: {raw_target}"
                )


def test_router_has_all_goal_sections_and_a_privacy_override() -> None:
    """The route map stays organized around the five supported goal families."""
    body = ROUTER.read_text(encoding="utf-8")
    for section in ("Intake", "Privacy", "Extraction", "Exchange", "Verification"):
        assert f"## {section}" in body

    assert "privacy override" in body.lower()
    lowered = body.lower()
    assert "no mandatory" in lowered and "network call" in lowered
    assert "case-insensitive substring" in lowered
    assert "stage order and row order break every tie" in lowered
    assert "Do not inspect or copy the data payload" in body
    assert "Never echo the request" in body
    assert "matched rule index" in body


def test_router_examples_escalate_ambiguous_sensitive_work() -> None:
    """Unstated sensitivity must route through the de-identification gate."""
    body = ROUTER.read_text(encoding="utf-8")
    assert "The sensitivity is unstated" in body
    assert (
        "[deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md)" in body
    )
    assert "explicitly" in body and "synthetic" in body


def test_router_applies_intake_before_the_privacy_gate() -> None:
    """Structured intake must precede text privacy and downstream stages."""
    body = ROUTER.read_text(encoding="utf-8")
    normalized = " ".join(body.split())

    intake_rule = body.index("2. Apply the intake boundary")
    privacy_rule = body.index("3. For a goal with no intake cue")
    assert intake_rule < privacy_rule
    assert "select the first matching intake skill even when the goal" in normalized
    assert "before extraction, exchange, or verification" in normalized


def test_router_is_present_in_generated_catalogs() -> None:
    """The new router must be discoverable through both shipped catalogs."""
    assert "[`ask-openmed`](ask-openmed/SKILL.md)" in CATALOG.read_text(
        encoding="utf-8"
    )
    marketplace = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    assert "./skills/ask-openmed" in marketplace["plugins"][0]["skills"]
