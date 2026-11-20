"""Guards for the AI Act technical-documentation evidence map."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
PAGE = DOCS / "compliance" / "eu-ai-act-technical-documentation.md"
BANNER = "> **Template — requires legal review. Not legal advice.**"
TABLE_HEADER = (
    "| Annex IV section | What it asks for | Supporting OpenMed artifact "
    "| Who owns the remainder |"
)
ANNEX_IV_SECTIONS = (
    *(f"1({letter})" for letter in "abcdefgh"),
    *(f"2({letter})" for letter in "abcdefgh"),
    *(str(point) for point in range(3, 10)),
)
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_EMOJI_RE = re.compile("[☀-➿\U0001f000-\U0001faff]")


def _text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _table_rows() -> list[list[str]]:
    lines = _text().splitlines()
    start = lines.index(TABLE_HEADER)
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


def _published() -> set[str]:
    publication = yaml.safe_load(
        (DOCS / "brand" / "system" / "publication.yml").read_text(encoding="utf-8")
    )
    classification = publication["classification"]
    return set(classification["navigated"]) | set(classification["link_only"])


def test_page_carries_the_banner_a_checked_on_date_and_no_emoji() -> None:
    text = _text()

    assert text.splitlines()[2].startswith(BANNER)
    assert re.search(r"checked on \d{4}-\d{2}-\d{2}", text)
    assert "Regulation (EU) 2026/1744" in text
    assert "check the current consolidated text" in text
    assert not _EMOJI_RE.search(text)


def test_every_annex_iv_section_appears_once_with_an_owner() -> None:
    rows = _table_rows()

    assert [row[0] for row in rows] == list(ANNEX_IV_SECTIONS)
    for section, asks_for, artifact, owner in rows:
        assert asks_for, section
        assert artifact, section
        assert owner, section
    unsupported = [row[0] for row in rows if row[2].startswith("None")]
    assert {"1(f)", "1(g)", "2(f)", "7", "8"} <= set(unsupported)


def test_application_dates_follow_article_113() -> None:
    text = _text()

    assert "Article 113" in text
    assert '"It shall apply from 2 August 2026."' in text
    for date in (
        "2 February 2025",
        "2 August 2025",
        "2 December 2026",
        "2 December 2027",
        "2 August 2028",
        "27 July 2026",
    ):
        assert date in text


def test_page_does_not_classify_openmed_components() -> None:
    text = " ".join(_text().split())

    assert (
        "These docs do not classify OpenMed components, or any system built with "
        "them, as medical devices or as high-risk AI systems."
    ) in text


def test_artifact_links_resolve_to_published_pages() -> None:
    published = _published()
    links = [
        link
        for link in _LINK_RE.findall(_text())
        if not link.startswith(("http://", "https://", "#"))
    ]

    assert links
    for link in links:
        path, _, anchor = link.partition("#")
        target = (PAGE.parent / path).resolve()
        assert target.is_file(), link
        assert target.relative_to(DOCS.resolve()).as_posix() in published, link
        if anchor:
            headings = {
                re.sub(r"[^a-z0-9]+", "-", line.lstrip("#").strip().lower()).strip("-")
                for line in target.read_text(encoding="utf-8").splitlines()
                if line.startswith("#")
            }
            assert anchor in headings, link


def test_page_is_published_and_linked_from_the_compliance_posture() -> None:
    compliance = (DOCS / "compliance.md").read_text(encoding="utf-8")

    assert "compliance/eu-ai-act-technical-documentation.md" in _published()
    assert "(compliance/eu-ai-act-technical-documentation.md)" in compliance
