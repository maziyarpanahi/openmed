"""Offline round trip for augmentation-only locale PHI notes."""

from __future__ import annotations

import json
import re

import pytest

from openmed.core.labels import ID_NUM
from openmed.core.pii_i18n import get_patterns_for_language
from openmed.training.synthetic.locale_phi import (
    AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES,
    NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES,
    LocalePhiGenerator,
    generate_locale_phi_examples,
)

pytestmark = pytest.mark.integration


def test_training_items_round_trip_through_jsonl_with_exact_offsets(tmp_path):
    examples = generate_locale_phi_examples(
        languages=AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES, seed=3869
    )
    path = tmp_path / "locale_phi_augmentation.jsonl"
    path.write_text(
        "".join(
            json.dumps(example.to_training_item(), ensure_ascii=False) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [row["language"] for row in rows] == list(
        AUGMENTATION_ONLY_LOCALE_PHI_LANGUAGES
    )
    for row in rows:
        assert row["is_synthetic"] is True
        assert row["metadata"]["augmentation_only"] is True
        assert row["metadata"]["bundled_model"] is False
        for label in row["labels"]:
            assert row["text"][label["start"] : label["end"]] == label["text"]


@pytest.mark.parametrize("language", NATIONAL_ID_ONLY_LOCALE_PHI_LANGUAGES)
def test_national_id_span_matches_the_deterministic_pattern_pack(language):
    # The templates carry the same national-ID cues the deterministic pattern
    # packs use, so the generated ID is recoverable at its exact gold offsets.
    for seed in range(5):
        example = LocalePhiGenerator(seed=seed).generate(language)
        id_span = next(span for span in example.gold_spans if span.label == ID_NUM)

        recovered = {
            (match.start(), match.end())
            for pattern in get_patterns_for_language(language)
            if pattern.entity_type == "national_id"
            for match in re.finditer(pattern.pattern, example.text, pattern.flags)
            if pattern.validator is None or pattern.validator(match.group(0))
        }

        assert (id_span.start, id_span.end) in recovered
