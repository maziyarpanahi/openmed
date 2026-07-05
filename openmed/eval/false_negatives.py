"""False-negative explorer over benchmark error-analysis reports.

The explorer surfaces the gold PHI spans a model missed (false negatives) from
a persisted :class:`~openmed.eval.error_analysis.ErrorAnalysisReport`. Span
matching is never re-implemented here: the misses are the ``kind == "missed"``
examples already produced by the harness span comparison.

By default the explorer emits only offsets, labels, and span hashes, keeping
raw PHI out of the output. Span text and a surrounding-context window are shown
only when the caller supplies the matching *synthetic* gold fixtures, whose text
is safe to display by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.eval.error_analysis import (
    MISSED,
    ErrorAnalysisReport,
    ErrorSpanExample,
)
from openmed.eval.harness import BenchmarkFixture, load_fixtures

__all__ = [
    "FalseNegativeRecord",
    "FalseNegativeExploration",
    "explore_false_negatives",
    "load_fixture_texts",
]


@dataclass(frozen=True)
class FalseNegativeRecord:
    """One missed gold PHI span, grouped under its label.

    ``span_text`` and ``context`` are populated only when synthetic fixture text
    is available; otherwise they stay ``None`` and only offsets/hash are shown.
    """

    label: str
    fixture_id: str
    start: int
    end: int
    context_start: int
    context_end: int
    text_hash: str
    span_text: str | None = None
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-ready record."""
        payload: dict[str, Any] = {
            "context_end": self.context_end,
            "context_start": self.context_start,
            "end": self.end,
            "fixture_id": self.fixture_id,
            "label": self.label,
            "start": self.start,
            "text_hash": self.text_hash,
        }
        if self.span_text is not None:
            payload["span_text"] = self.span_text
        if self.context is not None:
            payload["context"] = self.context
        return payload


@dataclass(frozen=True)
class FalseNegativeGroup:
    """All missed spans for a single label, most-frequent labels first."""

    label: str
    count: int
    records: tuple[FalseNegativeRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-ready group."""
        return {
            "count": self.count,
            "label": self.label,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class FalseNegativeExploration:
    """Grouped false-negative view derived from an error-analysis report."""

    suite: str
    model_name: str
    device: str
    total_missed: int
    shown: int
    groups: tuple[FalseNegativeGroup, ...]
    label_filter: str | None = None
    limit: int | None = None
    has_text: bool = False

    def iter_records(self) -> Iterable[FalseNegativeRecord]:
        """Yield every record across groups in display order."""
        for group in self.groups:
            yield from group.records

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-ready exploration payload."""
        return {
            "device": self.device,
            "groups": [group.to_dict() for group in self.groups],
            "has_text": self.has_text,
            "label_filter": self.label_filter,
            "limit": self.limit,
            "model_name": self.model_name,
            "shown": self.shown,
            "suite": self.suite,
            "total_missed": self.total_missed,
        }


def load_fixture_texts(
    sources: Iterable[str | Path | BenchmarkFixture | Mapping[str, Any]],
) -> dict[str, str]:
    """Build a ``fixture_id -> text`` map from synthetic gold fixtures.

    Each source may be a path to a fixture file, an already-loaded
    :class:`BenchmarkFixture`, or a fixture mapping. Only synthetic gold data
    should be passed; the returned text is used to render span context.
    """
    texts: dict[str, str] = {}
    for source in sources:
        for fixture in _coerce_fixture_sources(source):
            texts[fixture.fixture_id] = fixture.text
    return texts


def explore_false_negatives(
    report: ErrorAnalysisReport,
    *,
    fixture_texts: Mapping[str, str] | None = None,
    label: str | None = None,
    limit: int | None = None,
) -> FalseNegativeExploration:
    """Group the report's missed gold spans by label.

    Args:
        report: A persisted error-analysis report.
        fixture_texts: Optional ``fixture_id -> text`` map of *synthetic* gold
            fixtures. When present, span text and context windows are rendered;
            otherwise only offsets and hashes are shown.
        label: Optional label filter (case-insensitive). Only misses for this
            label are returned.
        limit: Optional cap on the total number of records shown.

    Returns:
        A :class:`FalseNegativeExploration` with labels ordered by miss
        frequency (descending), ties broken alphabetically, and records within
        a label ordered by fixture id then span offset.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    wanted_label = label.strip().upper() if label else None
    texts = dict(fixture_texts or {})

    grouped: dict[str, list[FalseNegativeRecord]] = {}
    total_missed = 0
    for group_label, examples in report.false_negatives.items():
        for example in examples:
            if example.kind != MISSED:
                # ``label_confusion`` spans are recall losses too, but a missed
                # gold span (kind == MISSED) is the leaked-PHI case this
                # explorer is scoped to. Skip mislabel confusions here.
                continue
            if wanted_label is not None and group_label.upper() != wanted_label:
                continue
            total_missed += 1
            grouped.setdefault(group_label, []).append(
                _record(group_label, example, texts)
            )

    ordered_labels = sorted(
        grouped,
        key=lambda name: (-len(grouped[name]), name),
    )

    groups: list[FalseNegativeGroup] = []
    shown = 0
    remaining = limit
    for name in ordered_labels:
        if remaining is not None and remaining <= 0:
            break
        records = sorted(
            grouped[name],
            key=lambda record: (record.fixture_id, record.start, record.end),
        )
        if remaining is not None:
            records = records[:remaining]
            remaining -= len(records)
        if not records:
            continue
        shown += len(records)
        groups.append(
            FalseNegativeGroup(
                label=name,
                count=len(grouped[name]),
                records=tuple(records),
            )
        )

    return FalseNegativeExploration(
        suite=report.suite,
        model_name=report.model_name,
        device=report.device,
        total_missed=total_missed,
        shown=shown,
        groups=tuple(groups),
        label_filter=wanted_label,
        limit=limit,
        has_text=bool(texts),
    )


def _record(
    label: str,
    example: ErrorSpanExample,
    texts: Mapping[str, str],
) -> FalseNegativeRecord:
    span_text: str | None = None
    context: str | None = None
    text = texts.get(example.fixture_id)
    if text is not None:
        span_text = _slice(text, example.start, example.end)
        context = _slice(text, example.context_start, example.context_end)
    return FalseNegativeRecord(
        label=label,
        fixture_id=example.fixture_id,
        start=example.start,
        end=example.end,
        context_start=example.context_start,
        context_end=example.context_end,
        text_hash=example.text_hash,
        span_text=span_text,
        context=context,
    )


def _slice(text: str, start: int, end: int) -> str | None:
    if 0 <= start <= end <= len(text):
        return text[start:end]
    return None


def _coerce_fixture_sources(
    source: str | Path | BenchmarkFixture | Mapping[str, Any],
) -> Sequence[BenchmarkFixture]:
    if isinstance(source, BenchmarkFixture):
        return (source,)
    if isinstance(source, Mapping):
        return (BenchmarkFixture.from_mapping(source),)
    return load_fixtures(source)
