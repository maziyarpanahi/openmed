from __future__ import annotations

import re
from pathlib import Path

from openmed.core.model_registry import list_model_categories


DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "ner-families.md"


def _family_section_headings() -> set[str]:
    text = DOC_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- ner-family-categories:start -->(.*?)<!-- ner-family-categories:end -->",
        text,
        re.S,
    )
    assert match is not None, "NER family doc must mark the generated category section"

    return set(re.findall(r"^## ([^\n]+)$", match.group(1), flags=re.M))


def test_ner_family_doc_lists_every_non_privacy_model_category() -> None:
    expected = set(list_model_categories()) - {"Privacy"}

    assert _family_section_headings() == expected
