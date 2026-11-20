"""Project local credentialed benchmark results onto a small aggregate contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from openmed.core.labels import CANONICAL_LABELS
from openmed.eval.cache import hash_fixture_set
from openmed.eval.report import BenchmarkReport
from openmed.eval.synthetic_fidelity import FidelityCounts, wilson_interval

if TYPE_CHECKING:
    from openmed.eval.harness import BenchmarkFixture


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative aggregate count")
    return value


def _rate(numerator: int, denominator: int) -> dict:
    counts = FidelityCounts(numerator, denominator)
    return {
        "rate": numerator / denominator if denominator else None,
        "wilson_95": wilson_interval(counts),
    }


def project_aggregate_report(
    report: BenchmarkReport,
    fixtures: Sequence[BenchmarkFixture],
    *,
    corpus: str,
    year: int,
    minimum_gold_spans: int = 5,
) -> dict:
    """Return a deterministic, aggregate-only credentialed de-identification report.

    Args:
        report: Span benchmark with per-label exact-match and leakage tallies.
        fixtures: The ordered local inputs used to generate the report.
        corpus: Controlled corpus family (i2b2, n2c2, cegs-n-grid or synthetic).
        year: Four-digit corpus edition year.
        minimum_gold_spans: Minimum gold spans needed to release a label cell.

    Returns:
        The corpus edition and dataset digest, suppression threshold and label
        cells. Each unsuppressed rate includes its descriptive Wilson interval.
    """
    if (
        corpus not in {"i2b2", "n2c2", "cegs-n-grid", "synthetic"}
        or type(year) is not int
        or not 1000 <= year <= 9999
    ):
        raise ValueError("invalid corpus edition")
    if type(minimum_gold_spans) is not int or minimum_gold_spans < 1:
        raise ValueError("invalid minimum gold span count")
    if report.fixture_count != len(fixtures) or (
        "fixture_ids" in report.metadata
        and report.metadata["fixture_ids"] != [f.fixture_id for f in fixtures]
    ):
        raise ValueError("fixture set does not match report")
    cells = report.metrics.get("exact_span_f1_by_label")
    leakage = report.metrics.get("leakage")
    if not isinstance(cells, Mapping) or not isinstance(leakage, Mapping):
        raise ValueError("per-label aggregate inputs missing")
    if not all(
        isinstance(leakage.get(k), Mapping)
        for k in ("leaked_chars_by_label", "total_chars_by_label")
    ):
        raise ValueError("per-label leakage counts missing")
    if not set(cells).issubset(CANONICAL_LABELS):
        raise ValueError("unsupported aggregate label")

    labels = {}
    for label in sorted(cells):
        cell = cells[label]
        if not isinstance(cell, Mapping):
            raise ValueError("invalid aggregate cell")
        tp, fn, fp = (
            _count(cell.get(k))
            for k in ("true_positives", "false_negatives", "false_positives")
        )
        if tp + fn < minimum_gold_spans:
            labels[label] = {"status": "suppressed"}
            continue
        total = _count(leakage["total_chars_by_label"].get(label))
        # The harness omits zero-leak labels from the numerator mapping.
        leaked = _count(leakage["leaked_chars_by_label"].get(label, 0))
        labels[label] = {
            "status": "reported",
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": _rate(tp, tp + fn),
            "precision": _rate(tp, tp + fp),
            "leakage": _rate(leaked, total),
        }
    return {
        "corpus": corpus,
        "year": year,
        "dataset_digest": hash_fixture_set(fixtures),
        "minimum_gold_spans": minimum_gold_spans,
        "by_label": labels,
    }
