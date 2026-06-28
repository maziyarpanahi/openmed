"""Coverage reporting for committed golden de-identification fixtures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval.golden import (
    GOLDEN_CATEGORIES,
    GoldenFixture,
    load_golden_fixtures,
)

GOLDEN_EDGE_CASE_CATEGORIES: tuple[str, ...] = GOLDEN_CATEGORIES


@dataclass(frozen=True)
class FixtureCoverageReport:
    """Golden fixture coverage across labels, languages, and edge cases."""

    fixture_count: int
    covered_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    covered_languages: tuple[str, ...]
    missing_languages: tuple[str, ...]
    covered_categories: tuple[str, ...]
    missing_categories: tuple[str, ...]
    category_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready mapping."""
        category_counts = {
            category: int(self.category_counts.get(category, 0))
            for category in GOLDEN_EDGE_CASE_CATEGORIES
        }
        return {
            "fixture_count": self.fixture_count,
            "labels": {
                "covered": list(self.covered_labels),
                "missing": list(self.missing_labels),
            },
            "languages": {
                "covered": list(self.covered_languages),
                "missing": list(self.missing_languages),
            },
            "categories": {
                "covered": list(self.covered_categories),
                "missing": list(self.missing_categories),
            },
            "category_counts": category_counts,
        }

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown coverage report."""
        label_status = _status_by_value(
            supported=sorted(CANONICAL_LABELS),
            covered=self.covered_labels,
        )
        language_status = _status_by_value(
            supported=sorted(SUPPORTED_LANGUAGES),
            covered=self.covered_languages,
        )

        lines = [
            "# Golden Fixture Coverage",
            "",
            "## Summary",
            "",
            "| Scope | Covered | Missing | Total |",
            "|---|---:|---:|---:|",
            (
                f"| Labels | {len(self.covered_labels)} | "
                f"{len(self.missing_labels)} | {len(CANONICAL_LABELS)} |"
            ),
            (
                f"| Languages | {len(self.covered_languages)} | "
                f"{len(self.missing_languages)} | {len(SUPPORTED_LANGUAGES)} |"
            ),
            (
                f"| Categories | {len(self.covered_categories)} | "
                f"{len(self.missing_categories)} | "
                f"{len(GOLDEN_EDGE_CASE_CATEGORIES)} |"
            ),
            f"| Fixtures | {self.fixture_count} | 0 | {self.fixture_count} |",
            "",
            "## Labels",
            "",
            "| Label | Status |",
            "|---|---|",
        ]
        for label, status in label_status:
            lines.append(f"| `{label}` | {status} |")

        lines.extend(
            [
                "",
                "## Languages",
                "",
                "| Language | Status |",
                "|---|---|",
            ]
        )
        for language, status in language_status:
            lines.append(f"| `{language}` | {status} |")

        lines.extend(
            [
                "",
                "## Categories",
                "",
                "| Category | Fixture Count | Status |",
                "|---|---:|---|",
            ]
        )
        covered_categories = set(self.covered_categories)
        for category in GOLDEN_EDGE_CASE_CATEGORIES:
            status = "covered" if category in covered_categories else "missing"
            count = int(self.category_counts.get(category, 0))
            lines.append(f"| `{category}` | {count} | {status} |")

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def fixture_coverage_report(
    path: str | Path | None = None,
    *,
    fixtures: Sequence[GoldenFixture] | None = None,
) -> FixtureCoverageReport:
    """Compute coverage for golden fixtures.

    Args:
        path: Optional fixture JSON file or directory. Defaults to the committed
            golden fixture directory.
        fixtures: Optional preloaded fixtures. Pass this in tests or callers
            that need to report a filtered fixture set.

    Returns:
        A report listing covered and missing canonical labels, supported
        languages, and golden edge-case categories.
    """
    if path is not None and fixtures is not None:
        raise ValueError("pass either path or fixtures, not both")

    source = list(fixtures) if fixtures is not None else load_golden_fixtures(path)

    observed_labels = {span.label for fixture in source for span in fixture.gold_spans}
    observed_languages = {fixture.language for fixture in source}
    category_counts = Counter(fixture.category for fixture in source)
    ordered_category_counts = {
        category: int(category_counts.get(category, 0))
        for category in GOLDEN_EDGE_CASE_CATEGORIES
    }

    covered_labels = tuple(
        label for label in sorted(CANONICAL_LABELS) if label in observed_labels
    )
    missing_labels = tuple(
        label for label in sorted(CANONICAL_LABELS) if label not in observed_labels
    )
    covered_languages = tuple(
        language
        for language in sorted(SUPPORTED_LANGUAGES)
        if language in observed_languages
    )
    missing_languages = tuple(
        language
        for language in sorted(SUPPORTED_LANGUAGES)
        if language not in observed_languages
    )
    covered_categories = tuple(
        category
        for category in GOLDEN_EDGE_CASE_CATEGORIES
        if ordered_category_counts[category] > 0
    )
    missing_categories = tuple(
        category
        for category in GOLDEN_EDGE_CASE_CATEGORIES
        if ordered_category_counts[category] == 0
    )

    return FixtureCoverageReport(
        fixture_count=len(source),
        covered_labels=covered_labels,
        missing_labels=missing_labels,
        covered_languages=covered_languages,
        missing_languages=missing_languages,
        covered_categories=covered_categories,
        missing_categories=missing_categories,
        category_counts=ordered_category_counts,
    )


def golden_fixture_coverage_report(
    path: str | Path | None = None,
    *,
    fixtures: Sequence[GoldenFixture] | None = None,
) -> FixtureCoverageReport:
    """Alias for :func:`fixture_coverage_report` with explicit golden naming."""
    return fixture_coverage_report(path, fixtures=fixtures)


def _status_by_value(
    *,
    supported: Sequence[str],
    covered: Sequence[str],
) -> list[tuple[str, str]]:
    covered_values = set(covered)
    return [
        (value, "covered" if value in covered_values else "missing")
        for value in supported
    ]


__all__ = [
    "FixtureCoverageReport",
    "GOLDEN_EDGE_CASE_CATEGORIES",
    "fixture_coverage_report",
    "golden_fixture_coverage_report",
]
