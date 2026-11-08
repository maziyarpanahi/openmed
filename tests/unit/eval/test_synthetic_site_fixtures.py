"""Deterministic synthetic site holdouts for federated evaluation tests."""

from __future__ import annotations

import random

import pytest

from openmed.eval.metrics import compute_exact_span_f1
from tests.fixtures.synthetic_site_holdouts import (
    DOCUMENTED_FAKE_VALUES,
    LABEL_VALUE_LISTS,
    SYNTHETIC_SITE_IDS,
    build_synthetic_site_holdouts,
)


def test_same_seed_returns_identical_fixtures() -> None:
    random.seed(0)
    first = build_synthetic_site_holdouts(4, seed=9, documents_per_site=2)
    random.seed(99)
    second = build_synthetic_site_holdouts(4, seed=9, documents_per_site=2)

    assert first == second
    assert [fixture.fixture_id for fixture in first] == [
        f"{site_id}-holdout-{document_index:02d}"
        for site_id in SYNTHETIC_SITE_IDS[:4]
        for document_index in range(2)
    ]


def test_different_seed_changes_holdout_text() -> None:
    first = build_synthetic_site_holdouts(4, seed=9, documents_per_site=2)
    changed = build_synthetic_site_holdouts(4, seed=10, documents_per_site=2)

    assert changed != first
    assert [fixture.text for fixture in changed] != [fixture.text for fixture in first]
    assert [fixture.language for fixture in changed] != [
        fixture.language for fixture in first
    ]
    assert [fixture.metadata["label_mix"] for fixture in changed] != [
        fixture.metadata["label_mix"] for fixture in first
    ]


def test_gold_spans_slice_documented_fake_values() -> None:
    fixtures = build_synthetic_site_holdouts(
        len(SYNTHETIC_SITE_IDS),
        seed=3,
        documents_per_site=2,
    )

    assert fixtures
    for fixture in fixtures:
        assert fixture.metadata["synthetic"] is True
        assert fixture.metadata["site_id"] in SYNTHETIC_SITE_IDS
        assert (
            tuple(span.label for span in fixture.gold_spans)
            == (fixture.metadata["label_mix"])
        )
        covered: list[tuple[int, int]] = []
        for span in fixture.gold_spans:
            assert 0 <= span.start < span.end <= len(fixture.text)
            assert fixture.text[span.start : span.end] == span.text
            assert span.text in LABEL_VALUE_LISTS[span.label]
            assert span.text in DOCUMENTED_FAKE_VALUES
            assert span.language == fixture.language
            assert span.text not in str(fixture.metadata["site_id"])
            assert all(
                span.text not in label for label in fixture.metadata["label_mix"]
            )
            covered.append((span.start, span.end))
        covered.sort()
        for previous, following in zip(covered, covered[1:], strict=False):
            assert previous[1] <= following[0]


def test_sites_vary_label_mix_and_language() -> None:
    fixtures = build_synthetic_site_holdouts(len(SYNTHETIC_SITE_IDS), seed=0)
    by_site: dict[str, list] = {}
    for fixture in fixtures:
        by_site.setdefault(fixture.metadata["site_id"], []).append(fixture)

    assert list(by_site) == list(SYNTHETIC_SITE_IDS)
    languages = {fixture.language for fixture in fixtures}
    mixes = {fixture.metadata["label_mix"] for fixture in fixtures}
    assert len(languages) > 1
    assert len(mixes) > 1
    for site_fixtures in by_site.values():
        assert len({fixture.language for fixture in site_fixtures}) == 1
        assert len({fixture.metadata["label_mix"] for fixture in site_fixtures}) == 1


def test_documents_at_one_site_use_different_placeholders() -> None:
    fixtures = build_synthetic_site_holdouts(2, seed=1, documents_per_site=2)
    by_site: dict[str, list[str]] = {}
    for fixture in fixtures:
        by_site.setdefault(fixture.metadata["site_id"], []).append(fixture.text)

    assert len(by_site) == 2
    for texts in by_site.values():
        assert len(set(texts)) == len(texts)


def test_matching_predictions_score_perfect_exact_span_f1() -> None:
    fixtures = build_synthetic_site_holdouts(3, seed=4, documents_per_site=2)
    for fixture in fixtures:
        matched = compute_exact_span_f1(
            fixture.gold_spans,
            fixture.gold_spans,
            default_language=fixture.language,
            source_text=fixture.text,
        )
        missed = compute_exact_span_f1(
            fixture.gold_spans,
            (),
            default_language=fixture.language,
            source_text=fixture.text,
        )
        assert matched.f1 == 1.0
        assert matched.false_negatives == 0
        assert matched.false_positives == 0
        assert missed.recall == 0.0
        assert missed.false_negatives == len(fixture.gold_spans)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"site_count": 0, "seed": 0}, "site_count"),
        ({"site_count": len(SYNTHETIC_SITE_IDS) + 1, "seed": 0}, "site_count"),
        ({"site_count": 1, "seed": -1}, "seed"),
        (
            {"site_count": 1, "seed": 0, "documents_per_site": 0},
            "documents_per_site",
        ),
    ],
)
def test_rejects_invalid_counts(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_synthetic_site_holdouts(**kwargs)


def test_rejects_boolean_counts() -> None:
    with pytest.raises(TypeError, match="site_count"):
        build_synthetic_site_holdouts(True, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed"):
        build_synthetic_site_holdouts(1, False)  # type: ignore[arg-type]
