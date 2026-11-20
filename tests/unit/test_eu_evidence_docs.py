"""Checks for the EU evidence index page in the Compliance docs."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPLIANCE = ROOT / "docs" / "compliance"
INDEX = COMPLIANCE / "eu-evidence.md"
NAV_ENTRY = "EU Evidence Index: compliance/eu-evidence.md"

EU_PAGES = (
    "dpia-template.md",
    "dpa-template.md",
    "gdpr-dsar-export.md",
    "model-card-eu-ai-act-fields.md",
    "iso-27701-control-evidence.md",
)

_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def _relative_links(markdown: str) -> list[str]:
    return [
        target
        for target in _LINK.findall(markdown)
        if "://" not in target and not target.startswith("mailto:")
    ]


def _missing_targets(markdown: str, base: Path) -> list[str]:
    return [
        target
        for target in _relative_links(markdown)
        if not (base / target).resolve().is_file()
    ]


def test_index_links_every_existing_eu_page() -> None:
    links = _relative_links(INDEX.read_text(encoding="utf-8"))

    for page in EU_PAGES:
        assert (COMPLIANCE / page).is_file()
        assert page in links


def test_every_relative_link_resolves() -> None:
    markdown = INDEX.read_text(encoding="utf-8")

    assert _relative_links(markdown)
    assert _missing_targets(markdown, COMPLIANCE) == []


def test_link_check_rejects_a_missing_target() -> None:
    markdown = "[Present](dpia-template.md) and [Absent](no-such-eu-page.md)"

    assert _missing_targets(markdown, COMPLIANCE) == ["no-such-eu-page.md"]


def test_index_states_it_is_an_index_and_not_legal_advice() -> None:
    lead = INDEX.read_text(encoding="utf-8").split("\n## ", 1)[0].lower()

    assert "index only" in lead
    assert "not legal advice" in lead


def test_index_is_in_compliance_nav_above_linked_pages() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8").splitlines()
    entries = [line.strip().removeprefix("- ") for line in nav]

    assert NAV_ENTRY in entries
    index_line = entries.index(NAV_ENTRY)
    assert entries.index("Compliance Posture: compliance.md") < index_line
    for page in EU_PAGES:
        listed = [
            number
            for number, entry in enumerate(entries)
            if entry.endswith(f": compliance/{page}")
        ]
        assert all(number > index_line for number in listed)
