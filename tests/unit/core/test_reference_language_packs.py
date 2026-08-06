"""Cross-script gates for the built-in language-pack reference declarations."""

from __future__ import annotations

from pathlib import Path

from openmed.core import LanguagePackRegistry, get_language_pack, pack_coherence_report
from openmed.core.language_pack_catalog import BUILTIN_LANGUAGE_PACKS
from openmed.core.language_pack_coherence import APPROXIMATED, CAPABILITY_SLOTS
from openmed.core.language_packs import (
    CHINESE_LANGUAGE_PACK,
    HINDI_LANGUAGE_PACK,
    REFERENCE_LANGUAGE_PACKS,
    TELUGU_LANGUAGE_PACK,
)

ROOT = Path(__file__).resolve().parents[3]


def test_reference_packs_register_only_through_the_public_contract() -> None:
    registry = LanguagePackRegistry()

    for pack in REFERENCE_LANGUAGE_PACKS:
        registry.register(pack)

    assert tuple(registry.iter_codes()) == ("hi", "te", "zh")
    assert registry.get("hi") is HINDI_LANGUAGE_PACK
    assert registry.get("te") is TELUGU_LANGUAGE_PACK
    assert registry.get("zh") is CHINESE_LANGUAGE_PACK


def test_builtin_bootstrap_registers_the_reference_declarations_once() -> None:
    for pack in REFERENCE_LANGUAGE_PACKS:
        assert sum(candidate is pack for candidate in BUILTIN_LANGUAGE_PACKS) == 1
        assert get_language_pack(pack.code) is pack


def test_cross_script_reference_packs_have_no_empty_capability_slots() -> None:
    rows = {row["language"]: row for row in pack_coherence_report()}

    for pack in REFERENCE_LANGUAGE_PACKS:
        row = rows[pack.code]
        coverage = row["coverage"]
        assert row["coherent"] is True
        assert set(coverage["slots"]) == set(CAPABILITY_SLOTS)
        assert coverage["populated"] == len(CAPABILITY_SLOTS)
        assert coverage["missing"] == 0
        assert coverage["filled"] + coverage["approximated"] == len(CAPABILITY_SLOTS)

    assert rows["hi"]["coverage"]["filled"] == 5
    assert rows["zh"]["coverage"]["filled"] == 5
    assert rows["te"]["coverage"]["filled"] == 4
    assert rows["te"]["coverage"]["approximated"] == 1
    assert rows["te"]["surrogate_locale"]["status"] == APPROXIMATED
    assert rows["te"]["surrogate_locale"]["approximation"]


def test_language_pack_onboarding_page_covers_the_complete_public_flow() -> None:
    page = (ROOT / "docs" / "language-packs.md").read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "Language Pack Plugins: language-packs.md" in nav
    assert "register_language_pack(pack)" in page
    assert "require_language_pack_coherence()" in page
    assert "pack_coherence_report()" in page
    assert "surrogate_locale_approximation" in page
    assert "replace=True" in page
