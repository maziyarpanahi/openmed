"""Guards for the EHDS secondary-use evidence map."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
PAGE = DOCS / "compliance" / "eu-ehds-secondary-use.md"
BANNER = "> **Template — requires legal review. Not legal advice.**"
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_ARTICLE_RE = re.compile(r"\bArticles? \d+")
_EMOJI_RE = re.compile("[☀-➿\U0001f000-\U0001faff]")


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _table_rows(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = lines.index(header)
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


def test_page_carries_the_legal_review_banner_and_no_emoji() -> None:
    text = _text()

    assert text.splitlines()[2].startswith(BANNER)
    assert "Not legal advice" in text
    assert not _EMOJI_RE.search(text)


def test_application_dates_are_quoted_with_a_checked_on_date() -> None:
    text = _text()

    assert "Article 105" in text
    assert re.search(
        r"checked against the Official Journal text on \d{4}-\d{2}-\d{2}", text
    )
    assert '"This Regulation shall apply from 26 March 2027."' in text
    assert '"Chapter IV shall apply from 26 March 2029"' in text
    for date in ("26 March 2031", "26 March 2035"):
        assert date in text


def test_every_mapping_row_cites_an_article() -> None:
    rows = _table_rows(
        _text(),
        "| EHDS concept | Article | OpenMed evidence | Not covered by OpenMed |",
    )

    assert len(rows) >= 10
    for concept, article, evidence, remainder in rows:
        assert _ARTICLE_RE.search(article), concept
        assert evidence and remainder, concept


def test_page_states_what_openmed_is_not() -> None:
    text = " ".join(_text().split())

    for statement in (
        "OpenMed is not a health data access body.",
        "OpenMed issues no data permits",
        "OpenMed is not a secure processing environment.",
        "does not make anyone compliant",
    ):
        assert statement in text


def test_local_links_resolve_to_published_pages() -> None:
    publication = yaml.safe_load(
        (DOCS / "brand" / "system" / "publication.yml").read_text(encoding="utf-8")
    )
    published = set(publication["classification"]["navigated"]) | set(
        publication["classification"]["link_only"]
    )
    links = [
        link
        for link in _LINK_RE.findall(_text())
        if not link.startswith(("http://", "https://", "#"))
    ]

    assert links
    for link in links:
        target = (PAGE.parent / link.split("#", 1)[0]).resolve()
        relative = target.relative_to(DOCS.resolve()).as_posix()
        assert target.is_file(), link
        assert relative in published, link


def test_page_is_published_and_linked_from_the_compliance_posture() -> None:
    publication = yaml.safe_load(
        (DOCS / "brand" / "system" / "publication.yml").read_text(encoding="utf-8")
    )
    compliance = (DOCS / "compliance.md").read_text(encoding="utf-8")

    assert (
        "compliance/eu-ehds-secondary-use.md"
        in publication["classification"]["link_only"]
    )
    assert "(compliance/eu-ehds-secondary-use.md)" in compliance
