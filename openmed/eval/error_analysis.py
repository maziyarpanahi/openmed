"""Per-entity confusion matrices and error examples for eval suites."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.quality_gates import (
    detect_overlapping_entities,
    validate_entity_spans,
)
from openmed.eval.golden.hard_negatives import HARD_NEGATIVE_CATEGORY
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
    load_fixtures,
)
from openmed.eval.metrics import EvalSpan, normalize_eval_spans
from openmed.eval.report import _format_value, _plain

MISSED = "missed"
SPURIOUS = "spurious"
ERROR_BUCKETS: tuple[str, str] = (MISSED, SPURIOUS)
LABELS: tuple[str, ...] = tuple(sorted(CANONICAL_LABELS))
MATRIX_LABELS: tuple[str, ...] = LABELS + ERROR_BUCKETS
DEFAULT_EXAMPLE_CAP = 5
DEFAULT_CONTEXT_WINDOW = 24


@dataclass(frozen=True)
class ErrorSpanExample:
    """One false-negative or false-positive span without plaintext PHI."""

    kind: str
    fixture_id: str
    label: str
    start: int
    end: int
    context_start: int
    context_end: int
    text_hash: str
    matched_label: str | None = None
    matched_start: int | None = None
    matched_end: int | None = None
    matched_text_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready example payload."""
        payload: dict[str, Any] = {
            "context_end": self.context_end,
            "context_start": self.context_start,
            "end": self.end,
            "fixture_id": self.fixture_id,
            "kind": self.kind,
            "label": self.label,
            "start": self.start,
            "text_hash": self.text_hash,
        }
        if self.matched_label is not None:
            payload["matched_label"] = self.matched_label
        if self.matched_start is not None:
            payload["matched_start"] = self.matched_start
        if self.matched_end is not None:
            payload["matched_end"] = self.matched_end
        if self.matched_text_hash is not None:
            payload["matched_text_hash"] = self.matched_text_hash
        return payload


@dataclass(frozen=True)
class ErrorAnalysisReport:
    """Serializable per-entity error-analysis report."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    confusion_matrix: Mapping[str, Mapping[str, int]]
    false_negatives: Mapping[str, Sequence[ErrorSpanExample]]
    false_positives: Mapping[str, Sequence[ErrorSpanExample]]
    example_cap: int = DEFAULT_EXAMPLE_CAP
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary with stable keys."""
        return {
            "confusion_matrix": {
                label: _matrix_row(self.confusion_matrix.get(label, {}))
                for label in MATRIX_LABELS
            },
            "device": self.device,
            "example_cap": self.example_cap,
            "false_negatives": _examples_to_dict(self.false_negatives),
            "false_positives": _examples_to_dict(self.false_positives),
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
            "model_name": self.model_name,
            "suite": self.suite,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report to deterministic Markdown."""
        lines = [
            f"# Error Analysis Report: {self.suite}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Example Cap | {self.example_cap} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Confusion Matrix",
                "",
                "| Gold Label/Bucket | Predicted Label/Bucket | Count |",
                "|---|---|---:|",
            ]
        )
        for gold_label in MATRIX_LABELS:
            row = self.confusion_matrix.get(gold_label, {})
            for predicted_label in MATRIX_LABELS:
                count = int(row.get(predicted_label, 0))
                if count:
                    lines.append(f"| `{gold_label}` | `{predicted_label}` | {count} |")

        lines.extend(_examples_markdown("False Negatives", self.false_negatives))
        lines.extend(_examples_markdown("False Positives", self.false_positives))

        if self.metadata:
            lines.extend(["", "## Metadata", "", "| Key | Value |", "|---|---|"])
            for key, value in _flatten(_plain(self.metadata)):
                lines.append(f"| `{key}` | {_format_value(value)} |")

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


@dataclass(frozen=True)
class HardNegativeOverRedactionReport:
    """Aggregate false-positive rate over synthetic hard-negative fixtures."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    candidate_count: int
    over_redacted_candidates: int
    over_redaction_rate: float
    by_label: Mapping[str, Mapping[str, int | float]]
    generated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report dictionary."""
        return {
            "by_label": {
                label: {
                    "candidate_count": int(row.get("candidate_count", 0)),
                    "over_redacted_candidates": int(
                        row.get("over_redacted_candidates", 0)
                    ),
                    "over_redaction_rate": float(row.get("over_redaction_rate", 0.0)),
                }
                for label, row in sorted(self.by_label.items())
            },
            "candidate_count": self.candidate_count,
            "device": self.device,
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "model_name": self.model_name,
            "over_redacted_candidates": self.over_redacted_candidates,
            "over_redaction_rate": self.over_redaction_rate,
            "suite": self.suite,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Serialize the report to deterministic Markdown."""
        lines = [
            f"# Hard Negative Over-Redaction Report: {self.suite}",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Fixtures | {self.fixture_count} |",
            f"| Candidates | {self.candidate_count} |",
            f"| Over-Redacted Candidates | {self.over_redacted_candidates} |",
            f"| Over-Redaction Rate | {self.over_redaction_rate:.6f} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")
        lines.extend(
            [
                "",
                "## Labels",
                "",
                "| Label | Candidates | Over-Redacted | Rate |",
                "|---|---:|---:|---:|",
            ]
        )
        if not self.by_label:
            lines.append("| _None_ | 0 | 0 | 0.000000 |")
        for label, row in sorted(self.by_label.items()):
            lines.append(
                f"| `{label}` | "
                f"{int(row.get('candidate_count', 0))} | "
                f"{int(row.get('over_redacted_candidates', 0))} | "
                f"{float(row.get('over_redaction_rate', 0.0)):.6f} |"
            )
        return "\n".join(lines) + "\n"


def error_report(
    model: str | ModelRunner,
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    suite_name: str | None = None,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    example_cap: int = DEFAULT_EXAMPLE_CAP,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ErrorAnalysisReport:
    """Run *model* on *suite* and build a per-label error-analysis report.

    ``model`` may be a model name or a callable using the benchmark runner
    signature. Passing ``runner`` keeps tests and local evals offline-friendly
    while preserving the default harness behavior for real model names.
    """
    if example_cap < 0:
        raise ValueError("example_cap must be non-negative")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    fixtures = _coerce_fixtures(suite)
    model_name, model_runner = _resolve_model_runner(model, runner)
    report_suite = suite_name or _suite_name(suite)

    matrix = _empty_matrix()
    false_negatives = _empty_examples()
    false_positives = _empty_examples()

    for fixture in fixtures:
        raw_predictions = list(model_runner(fixture, model_name, device))
        predicted_spans = normalize_eval_spans(
            raw_predictions,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        _accumulate_fixture_errors(
            fixture=fixture,
            predicted_spans=predicted_spans,
            matrix=matrix,
            false_negatives=false_negatives,
            false_positives=false_positives,
            example_cap=example_cap,
            context_window=context_window,
        )

    return ErrorAnalysisReport(
        suite=report_suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        confusion_matrix=matrix,
        false_negatives=false_negatives,
        false_positives=false_positives,
        example_cap=example_cap,
        generated_at=generated_at,
        metadata=dict(metadata or {}),
    )


def hard_negative_over_redaction_report(
    model: str | ModelRunner,
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    suite_name: str | None = None,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
) -> HardNegativeOverRedactionReport:
    """Measure false positives over synthetic hard-negative fixture candidates."""
    fixtures = _coerce_fixtures(suite)
    model_name, model_runner = _resolve_model_runner(model, runner)
    report_suite = suite_name or _suite_name(suite)
    by_label_counts: dict[str, dict[str, int]] = {}
    candidate_count = 0
    over_redacted = 0

    for fixture in fixtures:
        candidates = _hard_negative_candidate_rows(fixture)
        if not candidates:
            continue
        raw_predictions = list(model_runner(fixture, model_name, device))
        predicted_spans = normalize_eval_spans(
            raw_predictions,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        for candidate in candidates:
            label = str(candidate["label"])
            start = int(candidate["start"])
            end = int(candidate["end"])
            candidate_count += 1
            row = by_label_counts.setdefault(
                label,
                {"candidate_count": 0, "over_redacted_candidates": 0},
            )
            row["candidate_count"] += 1
            if any(
                _intervals_overlap(start, end, prediction.start, prediction.end)
                for prediction in predicted_spans
            ):
                over_redacted += 1
                row["over_redacted_candidates"] += 1

    by_label = {
        label: {
            "candidate_count": row["candidate_count"],
            "over_redacted_candidates": row["over_redacted_candidates"],
            "over_redaction_rate": _rate(
                row["over_redacted_candidates"],
                row["candidate_count"],
            ),
        }
        for label, row in by_label_counts.items()
    }
    return HardNegativeOverRedactionReport(
        suite=report_suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        candidate_count=candidate_count,
        over_redacted_candidates=over_redacted,
        over_redaction_rate=_rate(over_redacted, candidate_count),
        by_label=by_label,
        generated_at=generated_at,
    )


def _accumulate_fixture_errors(
    *,
    fixture: BenchmarkFixture,
    predicted_spans: Sequence[EvalSpan],
    matrix: dict[str, dict[str, int]],
    false_negatives: dict[str, list[ErrorSpanExample]],
    false_positives: dict[str, list[ErrorSpanExample]],
    example_cap: int,
    context_window: int,
) -> None:
    matched_predictions: set[int] = set()
    gold_spans = _ordered_spans(fixture.gold_spans)
    ordered_predictions = _ordered_indexed_spans(predicted_spans)

    for gold_span in gold_spans:
        match = _best_overlapping_prediction(
            gold_span,
            ordered_predictions,
            matched_predictions,
        )
        if match is None:
            matrix[gold_span.label][MISSED] += 1
            _append_example(
                false_negatives[gold_span.label],
                _example(
                    kind=MISSED,
                    fixture=fixture,
                    span=gold_span,
                    context_window=context_window,
                ),
                example_cap,
            )
            continue

        pred_index, pred_span = match
        matched_predictions.add(pred_index)
        matrix[gold_span.label][pred_span.label] += 1
        if gold_span.label == pred_span.label:
            continue

        _append_example(
            false_negatives[gold_span.label],
            _example(
                kind="label_confusion",
                fixture=fixture,
                span=gold_span,
                context_window=context_window,
                matched_span=pred_span,
            ),
            example_cap,
        )
        _append_example(
            false_positives[pred_span.label],
            _example(
                kind="label_confusion",
                fixture=fixture,
                span=pred_span,
                context_window=context_window,
                matched_span=gold_span,
            ),
            example_cap,
        )

    for pred_index, pred_span in ordered_predictions:
        if pred_index in matched_predictions:
            continue
        matrix[SPURIOUS][pred_span.label] += 1
        _append_example(
            false_positives[pred_span.label],
            _example(
                kind=SPURIOUS,
                fixture=fixture,
                span=pred_span,
                context_window=context_window,
            ),
            example_cap,
        )


def _best_overlapping_prediction(
    gold_span: EvalSpan,
    predictions: Sequence[tuple[int, EvalSpan]],
    matched_predictions: set[int],
) -> tuple[int, EvalSpan] | None:
    candidates = [
        (index, pred_span)
        for index, pred_span in predictions
        if index not in matched_predictions and _spans_overlap(gold_span, pred_span)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: _match_key(gold_span, item[0], item[1]),
    )


def _match_key(gold_span: EvalSpan, index: int, pred_span: EvalSpan) -> tuple[Any, ...]:
    return (
        _overlap_len(gold_span, pred_span),
        pred_span.start == gold_span.start and pred_span.end == gold_span.end,
        pred_span.label == gold_span.label,
        -abs(pred_span.start - gold_span.start),
        -abs(pred_span.end - gold_span.end),
        -index,
    )


def _spans_overlap(a: EvalSpan, b: EvalSpan) -> bool:
    return bool(detect_overlapping_entities([a.to_entity(), b.to_entity()]))


def _intervals_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return left_start < right_end and right_start < left_end


def _overlap_len(a: EvalSpan, b: EvalSpan) -> int:
    if not _spans_overlap(a, b):
        return 0
    return max(min(a.end, b.end) - max(a.start, b.start), 0)


def _example(
    *,
    kind: str,
    fixture: BenchmarkFixture,
    span: EvalSpan,
    context_window: int,
    matched_span: EvalSpan | None = None,
) -> ErrorSpanExample:
    context_start = max(0, span.start - context_window)
    context_end = min(len(fixture.text), span.end + context_window)
    return ErrorSpanExample(
        kind=kind,
        fixture_id=fixture.fixture_id,
        label=span.label,
        start=span.start,
        end=span.end,
        context_start=context_start,
        context_end=context_end,
        text_hash=_span_hash(fixture.text, span),
        matched_label=(matched_span.label if matched_span is not None else None),
        matched_start=(matched_span.start if matched_span is not None else None),
        matched_end=(matched_span.end if matched_span is not None else None),
        matched_text_hash=(
            _span_hash(fixture.text, matched_span) if matched_span is not None else None
        ),
    )


def _span_hash(text: str, span: EvalSpan) -> str:
    value = span.text
    if 0 <= span.start <= span.end <= len(text):
        value = text[span.start : span.end]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _append_example(
    examples: list[ErrorSpanExample],
    example: ErrorSpanExample,
    example_cap: int,
) -> None:
    if len(examples) < example_cap:
        examples.append(example)


def _coerce_fixtures(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
) -> list[BenchmarkFixture]:
    if isinstance(suite, (str, Path)):
        return load_fixtures(suite)
    return [
        fixture
        if isinstance(fixture, BenchmarkFixture)
        else BenchmarkFixture.from_mapping(fixture)
        for fixture in suite
    ]


def _hard_negative_candidate_rows(
    fixture: BenchmarkFixture,
) -> list[Mapping[str, Any]]:
    if fixture.metadata.get("category") != HARD_NEGATIVE_CATEGORY:
        return []
    rows = fixture.metadata.get("hard_negative_candidates")
    if not isinstance(rows, list):
        return []
    candidates: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if {"start", "end", "label"} <= set(row):
            candidates.append(row)
    return candidates


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner]:
    if runner is not None:
        return str(model), runner
    if callable(model):
        return _callable_name(model), model
    return str(model), default_model_runner


def _callable_name(value: ModelRunner) -> str:
    name = getattr(value, "__name__", "")
    if name and name != "<lambda>":
        return str(name)
    try:
        return value.__class__.__name__
    except AttributeError:
        return "model"


def _suite_name(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
) -> str:
    if isinstance(suite, (str, Path)):
        path = Path(suite)
        return path.stem or str(path)
    return "suite"


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {
        row_label: {column_label: 0 for column_label in MATRIX_LABELS}
        for row_label in MATRIX_LABELS
    }


def _matrix_row(row: Mapping[str, int]) -> dict[str, int]:
    return {
        column_label: int(row.get(column_label, 0)) for column_label in MATRIX_LABELS
    }


def _empty_examples() -> dict[str, list[ErrorSpanExample]]:
    return {label: [] for label in LABELS}


def _ordered_spans(spans: Iterable[EvalSpan]) -> list[EvalSpan]:
    return sorted(
        spans,
        key=lambda span: (span.start, span.end, span.label, span.text),
    )


def _ordered_indexed_spans(spans: Sequence[EvalSpan]) -> list[tuple[int, EvalSpan]]:
    return sorted(
        enumerate(spans),
        key=lambda item: (
            item[1].start,
            item[1].end,
            item[1].label,
            item[1].text,
            item[0],
        ),
    )


def _examples_to_dict(
    examples: Mapping[str, Sequence[ErrorSpanExample]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        label: [example.to_dict() for example in examples.get(label, ())]
        for label in LABELS
    }


def _examples_markdown(
    title: str,
    examples: Mapping[str, Sequence[ErrorSpanExample]],
) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Label | Kind | Fixture | Span | Context | Matched | Text Hash |",
        "|---|---|---|---|---|---|---|",
    ]
    has_rows = False
    for label in LABELS:
        for example in examples.get(label, ()):
            has_rows = True
            matched = ""
            if example.matched_label is not None:
                matched = (
                    f"`{example.matched_label}` "
                    f"{example.matched_start}:{example.matched_end}"
                )
            lines.append(
                "| "
                f"`{label}` | "
                f"`{example.kind}` | "
                f"`{example.fixture_id}` | "
                f"{example.start}:{example.end} | "
                f"{example.context_start}:{example.context_end} | "
                f"{matched} | "
                f"`{example.text_hash}` |"
            )
    if not has_rows:
        lines.append("| _None_ |  |  |  |  |  |  |")
    return lines


def _flatten(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    for key in sorted(payload):
        value = payload[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten(value, prefix=path)
        else:
            yield path, value


__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_EXAMPLE_CAP",
    "ERROR_BUCKETS",
    "LABELS",
    "MATRIX_LABELS",
    "MISSED",
    "SPURIOUS",
    "ErrorAnalysisReport",
    "ErrorSpanExample",
    "HardNegativeOverRedactionReport",
    "error_report",
    "hard_negative_over_redaction_report",
]
