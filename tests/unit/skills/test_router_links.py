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


def _router_rows() -> list[tuple[list[str], str]]:
    """Return normalized cue alternatives and their linked skill identifier."""
    rows: list[tuple[list[str], str]] = []
    for line in ROUTER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "](../" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 3, f"unexpected router row: {line}"
        target = LINK_RE.search(cells[1])
        assert target is not None, f"router row has no skill link: {line}"
        skill_id = Path(target.group(1)).parent.name
        cues = [cue.strip().casefold() for cue in cells[0].split(",")]
        rows.append((cues, skill_id))
    return rows


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
    normalized = " ".join(body.split())
    assert "no mandatory" in lowered and "network call" in lowered
    assert "case-insensitive substring" in lowered
    assert "stage order and row order break every tie" in normalized.lower()
    assert "Do not inspect or copy the data payload" in body
    assert "Never echo the request" in body
    assert "matched rule index" in body


def test_router_cues_are_explicit_and_unambiguous() -> None:
    """Cue parsing must not depend on interpreting natural-language conjunctions."""
    rows = _router_rows()
    assert rows
    cues = [cue for row_cues, _skill_id in rows for cue in row_cues]

    assert all(cue and " or " not in cue for cue in cues)
    assert len(cues) == len(set(cues))

    lab_row = next(
        index
        for index, (row_cues, skill_id) in enumerate(rows)
        if skill_id == "extracting-lab-tables" and "lab table" in row_cues
    )
    generic_table_row = next(
        index
        for index, (row_cues, skill_id) in enumerate(rows)
        if skill_id == "ingesting-clinical-documents" and "table" in row_cues
    )
    assert lab_row < generic_table_row


def test_router_examples_escalate_ambiguous_sensitive_work() -> None:
    """Unstated sensitivity must route through the de-identification gate."""
    body = ROUTER.read_text(encoding="utf-8")
    assert "The sensitivity is unstated" in body
    assert (
        "[deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md)" in body
    )
    assert "explicitly" in body and "synthetic" in body
    assert "it is not de-identified" in body
    assert "negated" in body and "does not count as a safe marker" in body


def test_router_safety_markers_fail_closed_on_negation() -> None:
    """Only fixed positive markers may bypass the privacy override."""
    body = ROUTER.read_text(encoding="utf-8")
    normalized = " ".join(body.split())
    for marker in (
        "`synthetic input`",
        "`synthetic note`",
        "`already de-identified`",
        "`already deidentified`",
    ):
        assert marker in body
    for negation in ("`no`", "`not`", "`never`", "`unknown`", "`uncertain`"):
        assert negation in body
    assert "four normalized words before it" in normalized
    assert "Treat every ambiguous safety statement as sensitive" in normalized


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
