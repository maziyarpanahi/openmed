"""Unit tests for eval dataset cards."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from openmed.eval import (
    DATASET_CARD_SUITES,
    build_all_dataset_cards,
    build_dataset_card,
)
from openmed.eval.datasets import license_for
from openmed.eval.golden import GOLDEN_CATEGORIES, load_golden_fixtures
from openmed.eval.suites import DRUGPROT, GOLDEN, SHIELD
from openmed.eval.suites.shield import (
    PUBLIC_SAMPLE_NOTES_CONFIG,
    PUBLIC_SAMPLE_SPANS_CONFIG,
)

DRUGPROT_FIXTURE_DIR = (
    Path(__file__).parents[2] / "fixtures" / "drugprot_synthetic" / "training"
)


def test_build_all_dataset_cards_is_offline_and_covers_concrete_suites() -> None:
    cards = build_all_dataset_cards()

    assert tuple(card.dataset for card in cards) == DATASET_CARD_SUITES
    assert tuple(card.dataset for card in cards) == (GOLDEN, SHIELD, DRUGPROT)
    for card in cards:
        dataset_license = license_for(card.dataset)
        assert card.license_id == dataset_license.license_id
        assert card.source_url == dataset_license.source_url
        assert card.redistribution == dataset_license.redistribution

    external_counts = {
        card.dataset: card.record_count
        for card in cards
        if card.dataset in {SHIELD, DRUGPROT}
    }
    assert external_counts == {SHIELD: 0, DRUGPROT: 0}


def test_golden_card_counts_committed_fixtures_without_text() -> None:
    fixtures = load_golden_fixtures()
    card = build_dataset_card(GOLDEN)

    assert card.record_count == len(fixtures)
    assert card.splits == tuple(sorted(GOLDEN_CATEGORIES))
    assert "SSN" in card.labels
    assert "en" in card.languages

    rendered = card.to_json() + card.to_markdown()
    for fixture in fixtures:
        assert fixture.text not in rendered


def test_dataset_card_markdown_and_json_are_byte_stable() -> None:
    first = build_dataset_card(GOLDEN)
    second = build_dataset_card(GOLDEN)

    assert first.to_markdown() == second.to_markdown()
    assert first.to_json() == second.to_json()


def test_shield_card_counts_explicit_rows_and_uses_license_registry() -> None:
    text = "Jordan Smith 555-0100"
    spans = [
        _shield_span(text, "patient", "Jordan Smith"),
        _shield_span(text, "phone", "555-0100"),
    ]

    def rows_loader(
        repository: str,
        config: str,
        split: str,
    ) -> list[Mapping[str, object]]:
        del repository, split
        if config == PUBLIC_SAMPLE_NOTES_CONFIG:
            return [
                {
                    "note_id": "shield-card-1",
                    "note_text": text,
                    "note_type": "synthetic_unit",
                }
            ]
        if config == PUBLIC_SAMPLE_SPANS_CONFIG:
            return spans
        raise AssertionError(f"unexpected config: {config}")

    card = build_dataset_card(SHIELD, rows_loader=rows_loader)
    dataset_license = license_for(SHIELD)

    assert card.record_count == 1
    assert card.license_id == dataset_license.license_id
    assert card.source_url == dataset_license.source_url
    assert card.labels == (
        "AGE",
        "DATE",
        "ID_NUM",
        "LOCATION",
        "ORGANIZATION",
        "PERSON",
        "PHONE",
        "URL",
    )
    assert card.languages == ("en",)
    assert card.splits == ("train",)
    assert text not in card.to_json()
    assert text not in card.to_markdown()


def test_drugprot_card_counts_explicit_fixture_path_without_text() -> None:
    card = build_dataset_card(DRUGPROT, path=DRUGPROT_FIXTURE_DIR)
    dataset_license = license_for(DRUGPROT)

    assert card.record_count == 1
    assert card.license_id == dataset_license.license_id
    assert card.source_url == dataset_license.source_url
    assert card.labels == ("OTHER",)
    assert card.languages == ("en",)
    assert card.splits == ("training",)

    rendered = card.to_json() + card.to_markdown()
    assert "Aspirin inhibits TP53" not in rendered
    assert "Metformin activates EGFR" not in rendered


def _shield_span(text: str, label: str, value: str) -> dict[str, object]:
    start = text.index(value)
    return {
        "note_id": "shield-card-1",
        "span_end": start + len(value),
        "span_label": label,
        "span_start": start,
    }
