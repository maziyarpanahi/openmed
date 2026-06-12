"""Synthetic-only golden de-identification fixtures for eval suites."""

from .loader import (
    GOLDEN_CATEGORIES,
    GoldenFixture,
    fixture_languages,
    fixtures_by_category,
    list_fixture_paths,
    load_benchmark_fixtures,
    load_golden_fixtures,
)

__all__ = [
    "GOLDEN_CATEGORIES",
    "GoldenFixture",
    "fixture_languages",
    "fixtures_by_category",
    "list_fixture_paths",
    "load_benchmark_fixtures",
    "load_golden_fixtures",
]
