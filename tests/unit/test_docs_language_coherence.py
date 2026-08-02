"""Source-of-truth guards for the per-language de-identification guide.

``docs/languages.md`` must document exactly the languages wired in
``openmed.core.pii_i18n.SUPPORTED_LANGUAGES``, and for each one it must list the
default PII model (``DEFAULT_PII_MODELS``), the Faker locale
(``LANG_TO_LOCALE``), and a per-language worked example. These assertions fail
loudly when a new language is wired in code but not documented here (or vice
versa), keeping the page coherent with the constants.
"""

from __future__ import annotations

import re
from pathlib import Path

from openmed.core.anonymizer.locales import LANG_TO_LOCALE
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    INDIC_NER_LANGUAGES,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "languages.md"
ANONYMIZATION = ROOT / "docs" / "anonymization.md"
README = ROOT / "README.md"
MKDOCS = ROOT / "mkdocs.yml"

# Matches a summary-table row: | `xx` | Language | `model` | `locale` | ... |
_TABLE_ROW = re.compile(
    r"^\|\s*`([a-z]{2})`\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|"
)
# Matches a per-language subsection heading: ### Language — `xx`
_HEADING = re.compile(r"^###\s+.*`([a-z]{2})`")


def _table_rows(text: str) -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    for line in text.splitlines():
        match = _TABLE_ROW.match(line)
        if match:
            code, name, model, locale = match.groups()
            rows[code] = (name.strip(), model.strip(), locale.strip())
    return rows


def _heading_codes(text: str) -> set[str]:
    return {
        match.group(1) for line in text.splitlines() if (match := _HEADING.match(line))
    }


def test_languages_doc_table_matches_supported_languages() -> None:
    rows = _table_rows(DOC.read_text(encoding="utf-8"))
    documented_languages = SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES

    assert set(rows) == documented_languages

    for lang in documented_languages:
        name, model, locale = rows[lang]
        assert name == LANGUAGE_NAMES[lang], lang
        assert model == DEFAULT_PII_MODELS[lang], lang
        assert locale == LANG_TO_LOCALE[lang], lang


def test_languages_doc_has_a_worked_example_per_language() -> None:
    text = DOC.read_text(encoding="utf-8")
    documented_languages = SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES

    assert _heading_codes(text) == documented_languages
    # Each worked example shows a before/after de-identification.
    assert text.count("Before:") >= len(documented_languages)
    assert text.count("After:") >= len(documented_languages)


def test_languages_doc_is_in_nav_and_cross_linked() -> None:
    nav = MKDOCS.read_text(encoding="utf-8")
    anonymization = ANONYMIZATION.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "languages.md" in nav
    assert "languages.md" in anonymization
    assert "docs/languages.md" in readme
