"""Loader for synthetic golden de-identification fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_i18n import NATIONAL_ID_ONLY_LANGUAGES, SUPPORTED_LANGUAGES
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan, normalize_eval_spans

GOLDEN_CATEGORIES: tuple[str, ...] = (
    "nested_overlapping",
    "chunk_boundary",
    "multilingual",
    "checksum_ids",
    "date_arithmetic",
    "policy_profile_actions",
)

_FIXTURE_VERSION = 1
_FIXTURE_DIR = Path(__file__).with_name("fixtures")


@dataclass(frozen=True)
class GoldenFixture:
    """One validated golden fixture with expected post-action output."""

    fixture_id: str
    category: str
    language: str
    text: str
    gold_spans: tuple[EvalSpan, ...]
    expected_output: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GoldenFixture":
        """Build and validate a golden fixture from a JSON-ready mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("golden fixture must be a mapping")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("golden fixture metadata must be a mapping")
        metadata = dict(metadata)

        if metadata.get("synthetic") is not True:
            raise ValueError("golden fixture metadata.synthetic must be true")

        category = str(metadata.get("category", ""))
        if category not in GOLDEN_CATEGORIES:
            raise ValueError(f"unknown golden fixture category: {category!r}")

        expected_output = metadata.get("expected_output")
        if not isinstance(expected_output, Mapping):
            raise ValueError(
                "golden fixture metadata.expected_output must be a mapping"
            )
        if not str(expected_output.get("method", "")):
            raise ValueError("golden fixture expected_output.method is required")
        if not isinstance(expected_output.get("text"), str):
            raise ValueError("golden fixture expected_output.text is required")

        language = str(data.get("language") or data.get("lang") or "en")
        fixture_languages = SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES
        if language not in fixture_languages:
            raise ValueError(f"unsupported golden fixture language: {language!r}")

        text = str(data.get("text", ""))
        if not text:
            raise ValueError("golden fixture text is required")

        raw_spans = data.get("gold_spans") or []
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ValueError("golden fixture must include at least one gold span")
        _validate_raw_span_labels(raw_spans, language)

        gold_spans = tuple(
            normalize_eval_spans(raw_spans, default_language=language, source_text=text)
        )
        _validate_offsets(text, gold_spans)

        fixture_id = str(data.get("id") or data.get("fixture_id") or "")
        if not fixture_id:
            raise ValueError("golden fixture id is required")

        return cls(
            fixture_id=fixture_id,
            category=category,
            language=language,
            text=text,
            gold_spans=gold_spans,
            expected_output=dict(expected_output),
            metadata=metadata,
        )

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Return the harness-compatible fixture view."""
        return BenchmarkFixture(
            fixture_id=self.fixture_id,
            text=self.text,
            gold_spans=self.gold_spans,
            language=self.language,
            metadata=dict(self.metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable JSON-ready mapping."""
        return {
            "id": self.fixture_id,
            "language": self.language,
            "text": self.text,
            "gold_spans": [_span_to_mapping(span) for span in self.gold_spans],
            "metadata": _plain_mapping(self.metadata),
        }


def list_fixture_paths(path: str | Path | None = None) -> tuple[Path, ...]:
    """Return fixture paths in deterministic order."""
    fixture_path = Path(path) if path is not None else _FIXTURE_DIR
    if fixture_path.is_file():
        return (fixture_path,)
    return tuple(
        sorted((*fixture_path.glob("*.json"), *fixture_path.glob("**/*.jsonl")))
    )


def load_golden_fixtures(path: str | Path | None = None) -> list[GoldenFixture]:
    """Load and validate all golden fixtures under *path*."""
    fixtures: list[GoldenFixture] = []
    for fixture_path in list_fixture_paths(path):
        if fixture_path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in fixture_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            fixtures.extend(GoldenFixture.from_mapping(row) for row in rows)
            continue

        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"{fixture_path} must contain a mapping")
        if raw.get("version") != _FIXTURE_VERSION:
            raise ValueError(f"{fixture_path} has unsupported fixture version")
        if raw.get("synthetic") is not True:
            raise ValueError(f"{fixture_path} must be marked synthetic")
        rows = raw.get("fixtures")
        if not isinstance(rows, list):
            raise ValueError(f"{fixture_path} must contain a fixtures list")
        fixtures.extend(GoldenFixture.from_mapping(row) for row in rows)
    return fixtures


def load_benchmark_fixtures(path: str | Path | None = None) -> list[BenchmarkFixture]:
    """Load golden fixtures as eval harness benchmark fixtures."""
    return [fixture.to_benchmark_fixture() for fixture in load_golden_fixtures(path)]


def fixtures_by_category(
    fixtures: list[GoldenFixture] | None = None,
) -> dict[str, list[GoldenFixture]]:
    """Group fixtures by golden category."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    grouped: defaultdict[str, list[GoldenFixture]] = defaultdict(list)
    for fixture in source:
        grouped[fixture.category].append(fixture)
    return dict(grouped)


def fixtures_by_language(
    fixtures: list[GoldenFixture] | None = None,
    *,
    category: str | None = None,
) -> dict[str, list[GoldenFixture]]:
    """Group fixtures by language, optionally restricted to one category."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    grouped: defaultdict[str, list[GoldenFixture]] = defaultdict(list)
    for fixture in source:
        if category is None or fixture.category == category:
            grouped[fixture.language].append(fixture)
    return dict(grouped)


def fixture_languages(
    fixtures: list[GoldenFixture] | None = None,
    *,
    category: str | None = None,
) -> set[str]:
    """Return languages covered by loaded fixtures."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    return {
        fixture.language
        for fixture in source
        if category is None or fixture.category == category
    }


def _validate_raw_span_labels(raw_spans: list[Any], language: str) -> None:
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("gold span must be a mapping")
        raw_label = raw_span.get("label") or raw_span.get("canonical_label")
        if not isinstance(raw_label, str):
            raise ValueError("gold span label is required")
        canonical = normalize_label(raw_label, language)
        if canonical != raw_label or canonical not in CANONICAL_LABELS:
            raise ValueError(f"gold span label must be canonical: {raw_label!r}")


def _validate_offsets(text: str, spans: tuple[EvalSpan, ...]) -> None:
    for span in spans:
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError(f"gold span has invalid offsets: {span!r}")
        actual_text = text[span.start : span.end]
        if span.text and actual_text != span.text:
            raise ValueError(
                f"gold span text mismatch for {span.label}: "
                f"{span.text!r} != {actual_text!r}"
            )


def _span_to_mapping(span: EvalSpan) -> dict[str, Any]:
    row: dict[str, Any] = {
        "start": span.start,
        "end": span.end,
        "label": span.label,
        "text": span.text,
    }
    metadata = dict(span.metadata)
    group = metadata.pop("group", None)
    if group is not None and str(group).strip():
        row["group"] = str(group).strip()
    if metadata:
        row["metadata"] = _plain_mapping(metadata)
    return row


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(value[key]) for key in sorted(value, key=str)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "GOLDEN_CATEGORIES",
    "GoldenFixture",
    "fixture_languages",
    "fixtures_by_category",
    "fixtures_by_language",
    "list_fixture_paths",
    "load_benchmark_fixtures",
    "load_golden_fixtures",
]
