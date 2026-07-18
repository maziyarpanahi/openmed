"""Guard documented PII language claims against source-of-truth drift."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

REPO_ROOT = Path(__file__).resolve().parents[3]
LANGUAGE_CLAIM_PATHS = (
    "README.md",
    "docs/anonymization.md",
    "docs/faq.md",
    "docs/rest-recipes.md",
    "docs/feature-map.md",
    "docs/index.md",
    "docs/troubleshooting.md",
    "docs/website/index.html",
)
STALE_LANGUAGE_COUNT_PATHS = LANGUAGE_CLAIM_PATHS + ("docs/website/assets/script.js",)

EXPECTED_LANGUAGE_CODES = sorted(SUPPORTED_LANGUAGES)
EXPECTED_LANGUAGE_COUNT = len(EXPECTED_LANGUAGE_CODES)
EXPECTED_LANGUAGE_CODES_TEXT = (
    ", ".join(EXPECTED_LANGUAGE_CODES[:-1]) + f", and {EXPECTED_LANGUAGE_CODES[-1]}"
)

LANGUAGE_CODE_PATTERN = r"\b[a-z]{2}\b"
LANGUAGE_CODES_PATTERN = (
    rf"{LANGUAGE_CODE_PATTERN}(?:,\s+{LANGUAGE_CODE_PATTERN})*"
    rf"(?:,?\s+and\s+{LANGUAGE_CODE_PATTERN})?"
)
LANGUAGE_CLAIM_PATTERN = re.compile(
    rf"(?P<count>\d+)\s+supported\s+PII\s+language\s+codes\s*:\s+"
    rf"(?P<codes>{LANGUAGE_CODES_PATTERN})",
    flags=re.IGNORECASE,
)
_STALE_COUNT = r"(?:15|16|fifteen|sixteen)"
_LANGUAGE_DESCRIPTOR = r"(?:(?:supported|model-backed|PII)[\s_-]+){0,3}languages?"
STALE_PII_LANGUAGE_COUNT_PATTERN = re.compile(
    rf"\b(?:{_STALE_COUNT}[\s_-]+{_LANGUAGE_DESCRIPTOR}|"
    rf"{_LANGUAGE_DESCRIPTOR}[^A-Za-z0-9]{{0,48}}{_STALE_COUNT})\b",
    flags=re.IGNORECASE,
)


def _visible_text(raw_text: str) -> str:
    decoded = html.unescape(raw_text)
    without_tags = re.sub(r"<[^>]+>", " ", decoded)
    without_markup = without_tags.translate(str.maketrans("", "", "`*"))
    return re.sub(r"\s+", " ", without_markup)


def _assert_supported_language_claim_matches(name: str, text: str) -> None:
    normalized = _visible_text(text)
    claims = list(LANGUAGE_CLAIM_PATTERN.finditer(normalized))
    assert claims, f"{name} must publish a parseable supported PII language claim"

    for claim in claims:
        count = int(claim.group("count"))
        codes = re.findall(LANGUAGE_CODE_PATTERN, claim.group("codes"))

        assert count == EXPECTED_LANGUAGE_COUNT
        assert codes == EXPECTED_LANGUAGE_CODES


@pytest.mark.parametrize("relative_path", LANGUAGE_CLAIM_PATHS)
def test_documented_pii_language_claim_matches_supported_languages(
    relative_path: str,
) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    _assert_supported_language_claim_matches(relative_path, text)


@pytest.mark.parametrize("relative_path", STALE_LANGUAGE_COUNT_PATHS)
def test_user_facing_docs_do_not_reintroduce_pre_17_language_claims(
    relative_path: str,
) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert not STALE_PII_LANGUAGE_COUNT_PATTERN.search(text)


@pytest.mark.parametrize(
    "claim",
    (
        "15 languages",
        "fifteen supported PII languages",
        "16 model-backed PII languages",
        "sixteen supported languages",
        "Model-backed PII languages | 16",
        "15-pii-languages",
    ),
)
def test_stale_language_count_guard_catches_pre_17_claims(claim: str) -> None:
    assert STALE_PII_LANGUAGE_COUNT_PATTERN.search(claim)


@pytest.mark.parametrize(
    "fixture_text",
    (
        (
            f"{EXPECTED_LANGUAGE_COUNT + 1} supported PII language codes: "
            f"{EXPECTED_LANGUAGE_CODES_TEXT}"
        ),
        (
            f"{EXPECTED_LANGUAGE_COUNT} supported PII language codes: "
            f"{', '.join(EXPECTED_LANGUAGE_CODES[:-1])}, and zz"
        ),
    ),
)
def test_documented_pii_language_claim_guard_rejects_mismatches(
    fixture_text: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_supported_language_claim_matches("fixture", fixture_text)
