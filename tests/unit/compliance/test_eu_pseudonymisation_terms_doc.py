"""Guards for the EU pseudonymisation and anonymisation terms page."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import yaml

from openmed.core.pii import DeidentificationMethod
from openmed.risk.policy_schema import SUPPORTED_ACTIONS

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
PAGE = DOCS / "compliance" / "eu-pseudonymisation-terms.md"
BANNER = "> **Template — requires legal review. Not legal advice.**"
ACTION_HEADER = (
    "| OpenMed action | What the output keeps | Default category "
    "| Evidence an anonymisation claim would need |"
)
_CITATION_RE = re.compile(
    r"(Article|Recital) \d+|EDPB Guidelines 01/2025|WP216|Case C-\d+/\d+"
)
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_EMOJI_RE = re.compile("[☀-➿\U0001f000-\U0001faff]")


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _table(header: str) -> list[list[str]]:
    lines = _text().splitlines()
    start = lines.index(header)
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


def _policy_doc_actions() -> set[str]:
    text = " ".join(
        (DOCS / "security" / "policy-as-data.md").read_text(encoding="utf-8").split()
    )
    sentence = re.search(r"Supported actions are (.+?)\. ", text)
    assert sentence is not None
    return set(re.findall(r"`([a-z_]+)`", sentence.group(1)))


def _anonymization_doc_methods() -> set[str]:
    """Return the methods in the first table of ``anonymization.md``."""

    lines = (DOCS / "anonymization.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Method"))
    methods: set[str] = set()
    for line in lines[start + 2 :]:
        match = re.match(r"\| `([a-z_]+)`\s+\|", line)
        if not match:
            break
        methods.add(match.group(1))
    return methods


def test_page_carries_the_banner_a_checked_date_and_no_emoji() -> None:
    text = _text()

    assert text.splitlines()[2].startswith(BANNER)
    assert re.search(r"Sources were checked on \d{4}-\d{2}-\d{2}", text)
    assert not _EMOJI_RE.search(text)


def test_every_documented_and_runtime_action_is_in_the_table() -> None:
    rows = _table(ACTION_HEADER)
    covered = {name for row in rows for name in re.findall(r"`([a-z_]+)`", row[0])}
    expected = (
        _policy_doc_actions()
        | _anonymization_doc_methods()
        | set(SUPPORTED_ACTIONS)
        | set(get_args(DeidentificationMethod))
    )

    assert {"keep", "redact", "remove", "shift_dates"} <= expected
    assert expected <= covered, sorted(expected - covered)
    first_cells = " ".join(row[0] for row in rows)
    assert "k-anonymity" in first_cells
    assert "Differential-privacy" in first_cells


def test_every_category_and_claim_carries_a_citation() -> None:
    for action, keeps, category, evidence in _table(ACTION_HEADER):
        assert keeps, action
        if action == "`keep`":
            assert _CITATION_RE.search(category), action
            continue
        assert _CITATION_RE.search(category), action
        assert _CITATION_RE.search(evidence) or "for `mask`" in evidence, action


def test_terms_table_and_legal_statements_cite_sources() -> None:
    for term, meaning, source in _table("| Term | Meaning | Source |"):
        assert meaning and source, term
    text = " ".join(_text().split())
    section = text.split("## What the sources say", 1)[1].split("## ", 1)[0]
    bullets = [item for item in section.split(" - ") if item.strip()]
    assert len(bullets) >= 5
    for bullet in bullets:
        assert _CITATION_RE.search(bullet) or "Opinion 05/2014" in bullet, bullet


def test_case_law_and_guidance_are_cited_by_title_and_number() -> None:
    text = " ".join(_text().split())

    for citation in (
        "Case C-582/14, Breyer (judgment of 19 October 2016)",
        "Case C-413/23 P, EDPS v SRB (judgment of 4 September 2025)",
        "Guidelines 01/2025 on Pseudonymisation, adopted on 16 January 2025",
        "Opinion 05/2014 on Anonymisation Techniques (WP216, adopted on 10 April 2014)",
    ):
        assert citation in text


def test_links_resolve_and_anonymization_page_links_back() -> None:
    publication = yaml.safe_load(
        (DOCS / "brand" / "system" / "publication.yml").read_text(encoding="utf-8")
    )
    classification = publication["classification"]
    published = set(classification["navigated"]) | set(classification["link_only"])
    links = [
        link
        for link in _LINK_RE.findall(_text())
        if not link.startswith(("http://", "https://", "#"))
    ]

    assert links
    for link in links:
        target = (PAGE.parent / link.split("#", 1)[0]).resolve()
        assert target.is_file(), link
        assert target.relative_to(DOCS.resolve()).as_posix() in published, link
    assert "compliance/eu-pseudonymisation-terms.md" in classification["link_only"]
    anonymization = (DOCS / "anonymization.md").read_text(encoding="utf-8")
    compliance = (DOCS / "compliance.md").read_text(encoding="utf-8")
    assert "(compliance/eu-pseudonymisation-terms.md)" in anonymization
    assert "(compliance/eu-pseudonymisation-terms.md)" in compliance
