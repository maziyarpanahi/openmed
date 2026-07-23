"""Clinical utility-loss reports for over-redaction evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    CLINICAL_CONCEPT,
    LABEL_METADATA,
    normalize_label,
    policy_label_for,
)
from openmed.core.quality_gates import (
    detect_overlapping_entities,
    validate_entity_spans,
)
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
)
from openmed.eval.metrics import (
    EvalSpan,
    RateMetric,
    compute_over_redaction_loss,
    normalize_eval_spans,
)
from openmed.eval.report import _format_value, _plain

DEFAULT_EXAMPLE_CAP = 5
DEFAULT_CONTEXT_WINDOW = 24
CLINICAL_SPAN_KEYS: tuple[str, ...] = (
    "clinical_spans",
    "gold_clinical_spans",
    "non_phi_clinical_spans",
    "non_phi_spans",
    "utility_spans",
)
CLINICAL_CATEGORIES: tuple[str, ...] = tuple(
    sorted(
        label
        for label, metadata in LABEL_METADATA.items()
        if metadata["policy_label"] == CLINICAL_CONCEPT
    )
)


@dataclass(frozen=True)
class UtilityLossExample:
    """One clinical span that a prediction would wrongly redact."""

    fixture_id: str
    category: str
    start: int
    end: int
    predicted_label: str
    predicted_start: int
    predicted_end: int
    context_start: int
    context_end: int
    text_hash: str
    predicted_text_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready example payload."""
        return {
            "category": self.category,
            "context_end": self.context_end,
            "context_start": self.context_start,
            "end": self.end,
            "fixture_id": self.fixture_id,
            "predicted_end": self.predicted_end,
            "predicted_label": self.predicted_label,
            "predicted_start": self.predicted_start,
            "predicted_text_hash": self.predicted_text_hash,
            "start": self.start,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class UtilityCategoryLoss:
    """Clinical utility loss for one non-PHI clinical category."""

    category: str
    rate: float
    over_redacted_chars: int
    total_chars: int
    over_redacted_spans: int
    total_spans: int

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a deterministic JSON-ready category payload."""
        return {
            "category": self.category,
            "over_redacted_chars": self.over_redacted_chars,
            "over_redacted_spans": self.over_redacted_spans,
            "rate": self.rate,
            "total_chars": self.total_chars,
            "total_spans": self.total_spans,
        }


@dataclass(frozen=True)
class UtilityLossReport:
    """Serializable report for clinical content lost to over-redaction."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    overall_over_redaction: RateMetric
    by_category: Mapping[str, UtilityCategoryLoss]
    examples: Mapping[str, Sequence[UtilityLossExample]]
    example_cap: int = DEFAULT_EXAMPLE_CAP
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary with stable keys."""
        return {
            "by_category": {
                category: self.by_category.get(
                    category,
                    _empty_category_loss(category),
                ).to_dict()
                for category in _category_keys(self.by_category, self.examples)
            },
            "device": self.device,
            "example_cap": self.example_cap,
            "examples": _examples_to_dict(self.examples),
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
            "model_name": self.model_name,
            "overall_over_redaction": self.overall_over_redaction.to_dict(),
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
        overall = self.overall_over_redaction
        lines = [
            f"# Utility Loss Report: {self.suite}",
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
                "## Overall Over-Redaction",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Rate | {_format_value(overall.rate)} |",
                f"| Numerator | {_format_value(overall.numerator)} |",
                f"| Denominator | {_format_value(overall.denominator)} |",
                "",
                "## Clinical Categories",
                "",
                (
                    "| Category | Rate | Over-Redacted Chars | Total Chars | "
                    "Over-Redacted Spans | Total Spans |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for category in _category_keys(self.by_category, self.examples):
            row = self.by_category.get(category, _empty_category_loss(category))
            lines.append(
                "| "
                f"`{category}` | "
                f"{_format_value(row.rate)} | "
                f"{row.over_redacted_chars} | "
                f"{row.total_chars} | "
                f"{row.over_redacted_spans} | "
                f"{row.total_spans} |"
            )

        lines.extend(_examples_markdown(self.examples))

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


def utility_loss_report(
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
) -> UtilityLossReport:
    """Run *model* on *suite* and report clinical utility lost to redaction.

    Gold PHI spans stay in ``fixture.gold_spans``. Gold non-PHI clinical spans
    can be supplied in fixture metadata or top-level fixture mappings under any
    of ``CLINICAL_SPAN_KEYS``. Clinical labels present in ``gold_spans`` are
    treated as preserve annotations for utility reporting rather than PHI.
    """
    if example_cap < 0:
        raise ValueError("example_cap must be non-negative")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    fixtures = _coerce_fixtures(suite)
    model_name, model_runner = _resolve_model_runner(model, runner)
    report_suite = suite_name or _suite_name(suite)

    over_redacted_chars: defaultdict[str, int] = defaultdict(int)
    total_chars: defaultdict[str, int] = defaultdict(int)
    over_redacted_spans: defaultdict[str, int] = defaultdict(int)
    total_spans: defaultdict[str, int] = defaultdict(int)
    examples: dict[str, list[UtilityLossExample]] = _empty_examples()
    overall_numerator = 0
    overall_denominator = 0

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
        gold_phi_spans = _phi_spans(fixture)
        overall = compute_over_redaction_loss(
            gold_phi_spans,
            predicted_spans,
            text_length=len(fixture.text),
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
        overall_numerator += int(overall.numerator)
        overall_denominator += int(overall.denominator)
        _accumulate_fixture_utility(
            fixture=fixture,
            predicted_spans=predicted_spans,
            clinical_spans=_clinical_spans(fixture),
            over_redacted_chars=over_redacted_chars,
            total_chars=total_chars,
            over_redacted_spans=over_redacted_spans,
            total_spans=total_spans,
            examples=examples,
            example_cap=example_cap,
            context_window=context_window,
        )

    by_category = {
        category: UtilityCategoryLoss(
            category=category,
            rate=_rate(over_redacted_chars[category], total_chars[category]),
            over_redacted_chars=over_redacted_chars[category],
            total_chars=total_chars[category],
            over_redacted_spans=over_redacted_spans[category],
            total_spans=total_spans[category],
        )
        for category in _category_keys(
            over_redacted_chars,
            total_chars,
            over_redacted_spans,
            total_spans,
            examples,
        )
    }

    return UtilityLossReport(
        suite=report_suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        overall_over_redaction=RateMetric(
            rate=_rate(overall_numerator, overall_denominator),
            numerator=overall_numerator,
            denominator=overall_denominator,
        ),
        by_category=by_category,
        examples=examples,
        example_cap=example_cap,
        generated_at=generated_at,
        metadata=dict(metadata or {}),
    )


def _accumulate_fixture_utility(
    *,
    fixture: BenchmarkFixture,
    predicted_spans: Sequence[EvalSpan],
    clinical_spans: Sequence[EvalSpan],
    over_redacted_chars: dict[str, int],
    total_chars: dict[str, int],
    over_redacted_spans: dict[str, int],
    total_spans: dict[str, int],
    examples: dict[str, list[UtilityLossExample]],
    example_cap: int,
    context_window: int,
) -> None:
    ordered_predictions = _ordered_spans(predicted_spans)
    for clinical_span in _ordered_spans(clinical_spans):
        category = _clinical_category(clinical_span.label)
        total_chars[category] += clinical_span.length
        total_spans[category] += 1

        matches = [
            prediction
            for prediction in ordered_predictions
            if _spans_overlap(clinical_span, prediction)
        ]
        intervals = [
            (
                max(clinical_span.start, prediction.start),
                min(clinical_span.end, prediction.end),
            )
            for prediction in matches
        ]
        overlap_chars = _merged_interval_length(
            [(start, end) for start, end in intervals if start < end]
        )
        if overlap_chars == 0:
            continue

        over_redacted_chars[category] += overlap_chars
        over_redacted_spans[category] += 1
        _append_example(
            examples[category],
            _example(
                fixture=fixture,
                span=clinical_span,
                prediction=_best_prediction(clinical_span, matches),
                context_window=context_window,
            ),
            example_cap,
        )


def _coerce_fixtures(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
) -> list[BenchmarkFixture]:
    if isinstance(suite, (str, Path)):
        return _load_fixtures(suite)
    return [
        fixture
        if isinstance(fixture, BenchmarkFixture)
        else _fixture_from_mapping(fixture)
        for fixture in suite
    ]


def _load_fixtures(path: str | Path) -> list[BenchmarkFixture]:
    fixture_path = Path(path)
    try:
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{fixture_path}: invalid JSON: {exc}") from exc
    rows = raw.get("fixtures") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise ValueError(
            "benchmark fixture JSON must be a list or contain a fixtures list"
        )
    fixtures = [_fixture_from_mapping(row) for row in rows]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def _fixture_from_mapping(data: Mapping[str, Any]) -> BenchmarkFixture:
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    merged_metadata = dict(metadata)
    for key in CLINICAL_SPAN_KEYS:
        if key in data and key not in merged_metadata:
            merged_metadata[key] = data[key]
    payload = dict(data)
    payload["metadata"] = merged_metadata
    return BenchmarkFixture.from_mapping(payload)


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


def _phi_spans(fixture: BenchmarkFixture) -> tuple[EvalSpan, ...]:
    return tuple(span for span in fixture.gold_spans if not _is_clinical(span.label))


def _clinical_spans(fixture: BenchmarkFixture) -> tuple[EvalSpan, ...]:
    raw_spans: list[Any] = []
    for key in CLINICAL_SPAN_KEYS:
        raw_spans.extend(_raw_spans(fixture.metadata.get(key)))

    normalized = normalize_eval_spans(
        raw_spans,
        default_language=fixture.language,
        source_text=fixture.text,
    )
    combined = [
        span for span in (*fixture.gold_spans, *normalized) if _is_clinical(span.label)
    ]

    deduped: dict[tuple[int, int, str, str], EvalSpan] = {}
    for span in _ordered_spans(combined):
        deduped.setdefault((span.start, span.end, span.label, span.text), span)
    return tuple(deduped.values())


def _raw_spans(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        nested = value.get("spans")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return list(nested)
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise ValueError("clinical span annotations must be mappings or sequences")


def _is_clinical(label: str) -> bool:
    return policy_label_for(label) == CLINICAL_CONCEPT


def _clinical_category(label: str) -> str:
    category = normalize_label(label)
    if category in CLINICAL_CATEGORIES:
        return category
    return "OTHER"


def _spans_overlap(a: EvalSpan, b: EvalSpan) -> bool:
    return bool(detect_overlapping_entities([a.to_entity(), b.to_entity()]))


def _best_prediction(span: EvalSpan, predictions: Sequence[EvalSpan]) -> EvalSpan:
    return max(
        enumerate(predictions),
        key=lambda item: (
            _overlap_len(span, item[1]),
            item[1].start == span.start and item[1].end == span.end,
            -abs(item[1].start - span.start),
            -abs(item[1].end - span.end),
            -item[0],
        ),
    )[1]


def _overlap_len(a: EvalSpan, b: EvalSpan) -> int:
    if not _spans_overlap(a, b):
        return 0
    return max(min(a.end, b.end) - max(a.start, b.start), 0)


def _merged_interval_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        old_start, old_end = merged[-1]
        merged[-1] = (old_start, max(old_end, end))
    return sum(end - start for start, end in merged)


def _example(
    *,
    fixture: BenchmarkFixture,
    span: EvalSpan,
    prediction: EvalSpan,
    context_window: int,
) -> UtilityLossExample:
    context_start = max(0, span.start - context_window)
    context_end = min(len(fixture.text), span.end + context_window)
    return UtilityLossExample(
        fixture_id=fixture.fixture_id,
        category=_clinical_category(span.label),
        start=span.start,
        end=span.end,
        predicted_label=prediction.label,
        predicted_start=prediction.start,
        predicted_end=prediction.end,
        context_start=context_start,
        context_end=context_end,
        text_hash=_span_hash(fixture.text, span),
        predicted_text_hash=_span_hash(fixture.text, prediction),
    )


def _span_hash(text: str, span: EvalSpan) -> str:
    value = span.text
    if 0 <= span.start <= span.end <= len(text):
        value = text[span.start : span.end]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _append_example(
    examples: list[UtilityLossExample],
    example: UtilityLossExample,
    example_cap: int,
) -> None:
    if len(examples) < example_cap:
        examples.append(example)


def _empty_examples() -> dict[str, list[UtilityLossExample]]:
    return {category: [] for category in CLINICAL_CATEGORIES}


def _examples_to_dict(
    examples: Mapping[str, Sequence[UtilityLossExample]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        category: [example.to_dict() for example in examples.get(category, ())]
        for category in _category_keys(examples)
    }


def _examples_markdown(
    examples: Mapping[str, Sequence[UtilityLossExample]],
) -> list[str]:
    lines = [
        "",
        "## Examples",
        "",
        (
            "| Category | Fixture | Clinical Span | Predicted Label | "
            "Predicted Span | Context | Text Hash | Predicted Text Hash |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    has_rows = False
    for category in _category_keys(examples):
        for example in examples.get(category, ()):
            has_rows = True
            lines.append(
                "| "
                f"`{category}` | "
                f"`{example.fixture_id}` | "
                f"{example.start}:{example.end} | "
                f"`{example.predicted_label}` | "
                f"{example.predicted_start}:{example.predicted_end} | "
                f"{example.context_start}:{example.context_end} | "
                f"`{example.text_hash}` | "
                f"`{example.predicted_text_hash}` |"
            )
    if not has_rows:
        lines.append("| _None_ |  |  |  |  |  |  |  |")
    return lines


def _ordered_spans(spans: Iterable[EvalSpan]) -> list[EvalSpan]:
    return sorted(
        spans,
        key=lambda span: (span.start, span.end, span.label, span.text),
    )


def _category_keys(*maps: Mapping[str, Any]) -> list[str]:
    keys = set(CLINICAL_CATEGORIES)
    for item in maps:
        keys.update(item)
    return sorted(keys)


def _empty_category_loss(category: str) -> UtilityCategoryLoss:
    return UtilityCategoryLoss(
        category=category,
        rate=0.0,
        over_redacted_chars=0,
        total_chars=0,
        over_redacted_spans=0,
        total_spans=0,
    )


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


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


def _validate_unique_fixture_ids(fixtures: Sequence[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for fixture in fixtures:
        if fixture.fixture_id in seen and fixture.fixture_id not in duplicates:
            duplicates.append(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        quoted = ", ".join(repr(value) for value in duplicates)
        raise ValueError(f"duplicate benchmark fixture id(s): {quoted}")


__all__ = [
    "CLINICAL_CATEGORIES",
    "CLINICAL_SPAN_KEYS",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_EXAMPLE_CAP",
    "UtilityCategoryLoss",
    "UtilityLossExample",
    "UtilityLossReport",
    "utility_loss_report",
]
