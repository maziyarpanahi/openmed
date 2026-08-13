import re
from pathlib import Path

from openmed.core.model_registry import (
    get_entity_types_by_category,
    get_models_by_category,
    list_model_categories,
)
from openmed.ner import available_domains

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "ner-families.md"
MKDOCS_PATH = ROOT / "mkdocs.yml"
SIZE_ORDER = {
    "Tiny": 0,
    "Small": 1,
    "Medium": 2,
    "Large": 3,
    "XLarge": 4,
    "Unknown": 5,
}
DOMAIN_ALIASES = {"Medical": "biomedical"}


def _sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## ([A-Za-z]+)\n", markdown, re.MULTILINE))
    return {
        match.group(1): markdown[
            match.end() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(markdown)
        ]
        for index, match in enumerate(matches)
    }


def _inline_code_values(section: str, field: str) -> list[str]:
    prefix = f"- **{field}:**"
    line = next(
        (line for line in section.splitlines() if line.startswith(prefix)),
        None,
    )
    assert line is not None, f"Missing {field!r} field"
    return re.findall(r"`([^`]+)`", line)


def test_ner_family_catalog_matches_model_registry() -> None:
    markdown = DOC_PATH.read_text(encoding="utf-8")
    sections = _sections(markdown)
    expected_categories = [
        category for category in list_model_categories() if category != "Privacy"
    ]

    assert list(sections) == expected_categories

    domains = set(available_domains())
    for category, section in sections.items():
        models = get_models_by_category(category)
        assert models, f"Registry category {category!r} has no models"

        assert _inline_code_values(section, "Entity types") == (
            get_entity_types_by_category(category)
        )

        expected_sizes = sorted(
            {model.size_category for model in models},
            key=lambda size: (SIZE_ORDER.get(size, len(SIZE_ORDER)), size),
        )
        assert _inline_code_values(section, "Available size categories") == (
            expected_sizes
        )

        expected_confidences = [
            f"{confidence:.2f}"
            for confidence in sorted({model.recommended_confidence for model in models})
        ]
        assert _inline_code_values(section, "Recommended confidence") == (
            expected_confidences
        )

        expected_domain = DOMAIN_ALIASES.get(category, category.lower())
        assert expected_domain in domains
        assert _inline_code_values(section, "Zero-shot domain") == [expected_domain]
        anchor = expected_domain.replace("_", "-")
        assert f"(./clinical-domains.md#{anchor})" in section


def test_ner_family_catalog_is_in_docs_navigation() -> None:
    nav = MKDOCS_PATH.read_text(encoding="utf-8")
    assert "- NER Families: ner-families.md" in nav
