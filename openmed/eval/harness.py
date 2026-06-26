"""Suite runner for OpenMed benchmark fixtures."""

from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.cache import build_report_key, hash_fixture_set, load_or_compute
from openmed.eval.metrics import (
    EvalSpan,
    compute_confidence_intervals,
    compute_metrics_bundle,
    normalize_eval_spans,
)
from openmed.eval.report import BenchmarkReport

ModelRunner = Callable[["BenchmarkFixture", str, str], Iterable[Any]]


@dataclass(frozen=True)
class BenchmarkFixture:
    """One benchmark document with gold PHI spans."""

    fixture_id: str
    text: str
    gold_spans: tuple[EvalSpan, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BenchmarkFixture":
        """Build a fixture from a JSON-ready mapping."""
        text = str(data.get("text", ""))
        language = str(data.get("language") or data.get("lang") or "en")
        fixture_id = str(data.get("id") or data.get("fixture_id") or "fixture")
        gold_spans = tuple(
            normalize_eval_spans(
                data.get("gold_spans") or data.get("entities") or [],
                default_language=language,
                source_text=text,
            )
        )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            fixture_id=fixture_id,
            text=text,
            gold_spans=gold_spans,
            language=language,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class FixtureResult:
    """Predictions and timing for one benchmark fixture."""

    fixture_id: str
    predicted_spans: tuple[EvalSpan, ...]
    latency_ms: float


def load_fixtures(path: str | Path) -> list[BenchmarkFixture]:
    """Load benchmark fixtures from a JSON file.

    Accepted top-level shapes are either a list of fixture objects or a mapping
    containing a ``fixtures`` list.
    """
    fixture_path = Path(path)
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = raw.get("fixtures") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise ValueError(
            "benchmark fixture JSON must be a list or contain a fixtures list"
        )
    fixtures = [BenchmarkFixture.from_mapping(row) for row in rows]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def default_model_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
    *,
    loader: Any | None = None,
) -> Iterable[Any]:
    """Run a fixture through the existing PII runtime."""
    from openmed.core.pii import extract_pii

    result = extract_pii(
        fixture.text,
        model_name=model_name,
        lang=fixture.language,
        loader=loader,
    )
    for entity in result.entities:
        metadata = dict(entity.metadata or {})
        metadata.setdefault("device", device)
        entity.metadata = metadata
    return result.entities


def run_benchmark(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_code_hash: str | None = None,
) -> BenchmarkReport:
    """Run *model_name* over fixtures and return a benchmark report.

    When ``confidence_intervals`` is enabled (off by default to keep fast runs
    cheap), a non-parametric bootstrap over documents attaches a
    ``confidence_interval`` payload to the leakage, character recall, and exact
    and relaxed span F1 metrics. The bootstrap is deterministic for a fixed
    ``ci_seed``. Passing ``cache_dir`` opts into a local filesystem cache keyed
    by model, suite, device, fixture-set hash, and eval code hash.
    """
    _validate_unique_fixture_ids(fixtures)
    if cache_dir is not None:
        report_key = build_report_key(
            model_name=model_name,
            suite=suite,
            fixture_set_hash=hash_fixture_set(fixtures),
            code_hash=cache_code_hash,
            device=device,
        )
        return load_or_compute(
            report_key,
            lambda: run_benchmark(
                fixtures,
                suite=suite,
                model_name=model_name,
                device=device,
                runner=runner,
                generated_at=generated_at,
                metadata=metadata,
                confidence_intervals=confidence_intervals,
                ci_resamples=ci_resamples,
                ci_alpha=ci_alpha,
                ci_seed=ci_seed,
            ),
            cache_dir=cache_dir,
        )

    model_runner = runner or _shared_default_model_runner()
    results: list[FixtureResult] = []
    peak_rss_start = _peak_rss_bytes()

    for fixture in fixtures:
        started = time.perf_counter()
        raw_predictions = list(model_runner(fixture, model_name, device))
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_spans = tuple(
            normalize_eval_spans(
                raw_predictions,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        results.append(
            FixtureResult(
                fixture_id=fixture.fixture_id,
                predicted_spans=predicted_spans,
                latency_ms=latency_ms,
            )
        )

    gold_spans, predicted_spans, corpus_text = _corpus_coordinates(fixtures, results)
    peak_rss_end = _peak_rss_bytes()
    rss_values = [
        value for value in (peak_rss_start, peak_rss_end) if value is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    metrics = compute_metrics_bundle(
        gold_spans,
        predicted_spans,
        latencies_ms=[result.latency_ms for result in results[1:]],
        cold_start_ms=(results[0].latency_ms if results else None),
        peak_rss_bytes=peak_rss,
        default_device=device,
        source_text=corpus_text,
    )
    if confidence_intervals:
        metrics = _attach_confidence_intervals(
            metrics,
            fixtures,
            results,
            device=device,
            n_resamples=ci_resamples,
            alpha=ci_alpha,
            seed=ci_seed,
        )

    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids", [fixture.fixture_id for fixture in fixtures]
    )
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def run_suite(
    fixture_path: str | Path,
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_code_hash: str | None = None,
) -> BenchmarkReport:
    """Load fixtures, run the benchmark, and optionally write reports."""
    report = run_benchmark(
        load_fixtures(fixture_path),
        suite=suite,
        model_name=model_name,
        device=device,
        runner=runner,
        generated_at=generated_at,
        metadata=metadata,
        confidence_intervals=confidence_intervals,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
        cache_dir=cache_dir,
        cache_code_hash=cache_code_hash,
    )
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        report.write_markdown(output_markdown)
    return report


def _shared_default_model_runner() -> ModelRunner:
    shared_loader: Any | None = None
    accepts_loader = _runner_accepts_loader(default_model_runner)

    def run_fixture(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> Iterable[Any]:
        nonlocal shared_loader
        if not accepts_loader:
            return default_model_runner(fixture, model_name, device)
        if shared_loader is None:
            from openmed.core.models import ModelLoader

            shared_loader = ModelLoader()
        return default_model_runner(
            fixture,
            model_name,
            device,
            loader=shared_loader,
        )

    return run_fixture


def _runner_accepts_loader(runner: Callable[..., Iterable[Any]]) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True

    return any(
        parameter.name == "loader" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _attach_confidence_intervals(
    metrics: Mapping[str, Any],
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
    *,
    device: str,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap per-document CIs and merge them into the metric bundle."""
    result_by_id = {result.fixture_id: result for result in results}
    per_document_spans = [
        (
            fixture.gold_spans,
            getattr(result_by_id.get(fixture.fixture_id), "predicted_spans", ()),
        )
        for fixture in fixtures
    ]
    intervals = compute_confidence_intervals(
        per_document_spans,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
        default_device=device,
    )
    merged = dict(metrics)
    for key, interval in intervals.items():
        metric = merged.get(key)
        if isinstance(metric, Mapping):
            merged[key] = {**metric, "confidence_interval": interval}
    return merged


def _corpus_coordinates(
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
) -> tuple[list[EvalSpan], list[EvalSpan], str]:
    result_by_id = {result.fixture_id: result for result in results}
    gold: list[EvalSpan] = []
    predicted: list[EvalSpan] = []
    text_parts: list[str] = []
    offset = 0
    for fixture in fixtures:
        text_parts.append(fixture.text)
        gold.extend(_shift_spans(fixture.gold_spans, offset))
        result = result_by_id.get(fixture.fixture_id)
        if result is not None:
            predicted.extend(_shift_spans(result.predicted_spans, offset))
        offset += len(fixture.text) + 1
    return gold, predicted, "\n".join(text_parts)


def _shift_spans(spans: Iterable[EvalSpan], offset: int) -> list[EvalSpan]:
    return [
        replace(span, start=span.start + offset, end=span.end + offset)
        for span in spans
    ]


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


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


__all__ = [
    "ModelRunner",
    "BenchmarkFixture",
    "FixtureResult",
    "load_fixtures",
    "default_model_runner",
    "run_benchmark",
    "run_suite",
]
