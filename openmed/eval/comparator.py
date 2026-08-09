"""Reproducible, offline comparator benchmarking for de-identification."""

from __future__ import annotations

import inspect
import json
import math
import random
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

from openmed.core.audit import stable_hash
from openmed.core.labels import normalize_label
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import (
    EvalSpan,
    LatencyMetrics,
    compute_character_recall,
    compute_exact_span_f1,
    compute_latency_summary,
    compute_leakage_rate,
    normalize_eval_spans,
)

COMPARATOR_SCHEMA_VERSION = "openmed.eval.comparator.v1"
STATUS_SCORED = "scored"
STATUS_NOT_AVAILABLE = "not_available"
_DEFAULT_CRITICAL_LABELS = frozenset(
    {
        "SSN",
        "ID_NUM",
        "API_KEY",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "CREDIT_CARD",
        "CVV",
        "IBAN",
        "BIC",
    }
)

Clock = Callable[[], float]
MemorySampler = Callable[[], int | None]
AdapterRunner = Callable[..., Iterable[Any]]


class ComparatorError(RuntimeError):
    """Base error for comparator runs with source-safe messages."""


class ComparatorFixtureError(ComparatorError, ValueError):
    """Raised when a comparator fixture is not valid synthetic input."""


class ComparatorExecutionError(ComparatorError):
    """Raised when a configured adapter cannot produce valid predictions."""


class ComparatorAdapterUnavailable(ImportError):
    """Raised by an adapter when its optional local dependency is unavailable."""


@dataclass(frozen=True)
class ComparatorFixture:
    """One synthetic document and its gold de-identification spans.

    The raw text is accepted only for the in-process adapter call. It is never
    included in :meth:`to_dict`, reports, or error messages. ``synthetic`` and
    ``phi_free`` default to ``True`` for inline test fixtures; callers loading
    external data should set both flags explicitly and keep the source corpus
    synthetic.
    """

    fixture_id: str
    text: str
    gold_spans: tuple[EvalSpan, ...] = ()
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    synthetic: bool = True
    phi_free: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ComparatorFixtureError("invalid comparator fixture")
        fixture_id = str(self.fixture_id).strip()
        if not fixture_id:
            raise ComparatorFixtureError("invalid comparator fixture")
        language = str(self.language or "en").strip().lower() or "en"
        if not isinstance(self.metadata, Mapping):
            raise ComparatorFixtureError("invalid comparator fixture")

        metadata_synthetic = self.metadata.get("synthetic")
        metadata_phi_free = self.metadata.get("phi_free")
        if metadata_synthetic is not None and not _flag(metadata_synthetic):
            raise ComparatorFixtureError("comparator fixtures must be synthetic")
        if metadata_phi_free is not None and not _flag(metadata_phi_free):
            raise ComparatorFixtureError("comparator fixtures must be PHI-free")
        if not self.synthetic:
            raise ComparatorFixtureError("comparator fixtures must be synthetic")
        if not self.phi_free:
            raise ComparatorFixtureError("comparator fixtures must be PHI-free")

        try:
            spans = _normalize_spans(
                self.gold_spans,
                language=language,
                text=self.text,
            )
        except Exception:
            raise ComparatorFixtureError("invalid comparator fixture") from None

        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "gold_spans", spans)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComparatorFixture":
        """Build a fixture from a JSON-compatible mapping."""

        if not isinstance(data, Mapping):
            raise ComparatorFixtureError("invalid comparator fixture")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ComparatorFixtureError("invalid comparator fixture")
        synthetic = data.get("synthetic", metadata.get("synthetic", True))
        phi_free = data.get("phi_free", metadata.get("phi_free", True))
        return cls(
            fixture_id=data.get("fixture_id") or data.get("id") or "",
            text=data.get("text", ""),
            gold_spans=data.get("gold_spans") or data.get("entities") or (),
            language=data.get("language") or data.get("lang") or "en",
            metadata=metadata,
            synthetic=_flag(synthetic),
            phi_free=_flag(phi_free),
        )

    @classmethod
    def from_benchmark_fixture(cls, fixture: BenchmarkFixture) -> "ComparatorFixture":
        """Convert an existing local harness fixture without retaining it in a report."""

        return cls(
            fixture_id=fixture.fixture_id,
            text=fixture.text,
            gold_spans=fixture.gold_spans,
            language=fixture.language,
            metadata=fixture.metadata,
        )

    @property
    def digest(self) -> str:
        """Return a content digest without exposing fixture identifiers or text."""

        return stable_hash(
            {
                "gold_spans": [
                    {"end": span.end, "label": span.label, "start": span.start}
                    for span in self.gold_spans
                ],
                "language": self.language,
                "text": self.text,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free fixture descriptor for diagnostics."""

        return {
            "fixture_digest": self.digest,
            "gold_span_count": len(self.gold_spans),
            "language": self.language,
            "phi_free": self.phi_free,
            "synthetic": self.synthetic,
            "text_length": len(self.text),
        }

    def as_benchmark_fixture(self) -> BenchmarkFixture:
        """Return the existing harness shape for legacy three-argument runners."""

        return BenchmarkFixture(
            fixture_id=self.fixture_id,
            text=self.text,
            gold_spans=self.gold_spans,
            language=self.language,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ComparatorAdapter:
    """A local baseline adapter used by :func:`run_comparator_benchmark`.

    The preferred runner contract is ``runner(text, language)`` and it must
    return an iterable of span-like mappings with ``start``, ``end``, and
    ``label`` fields. A one-argument runner receives ``text``. Existing
    OpenMed benchmark runners taking ``(fixture, model_name, device)`` are
    also accepted so local adapters can be reused without network access.
    """

    name: str
    runner: AdapterRunner | None = None
    version: str = "local"
    model_name: str | None = None
    device: str = "cpu"
    requires_network: bool = False
    unavailable_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    predict: AdapterRunner | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("comparator adapter name must be non-empty")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("comparator adapter metadata must be a mapping")
        runner = self.runner or self.predict
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "runner", runner)
        object.__setattr__(self, "version", str(self.version or "local"))
        object.__setattr__(self, "device", str(self.device or "cpu"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ComparatorBudget:
    """Common optional resource limits applied to every adapter."""

    max_latency_ms: float | None = None
    max_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_latency_ms is not None and (
            not math.isfinite(float(self.max_latency_ms))
            or float(self.max_latency_ms) <= 0
        ):
            raise ValueError("max_latency_ms must be positive")
        if self.max_memory_bytes is not None and (
            isinstance(self.max_memory_bytes, bool) or self.max_memory_bytes <= 0
        ):
            raise ValueError("max_memory_bytes must be positive")

    def to_dict(self) -> dict[str, int | float | None]:
        """Return the stable budget payload."""

        return {
            "max_latency_ms": self.max_latency_ms,
            "max_memory_bytes": self.max_memory_bytes,
        }


@dataclass(frozen=True)
class ComparatorMemoryMetrics:
    """Aggregate process-memory samples for one adapter run."""

    peak_bytes: int | None
    baseline_bytes: int | None
    delta_bytes: int | None
    sample_count: int

    @property
    def peak_rss_bytes(self) -> int | None:
        """Compatibility alias for callers using RSS terminology."""

        return self.peak_bytes

    def to_dict(self) -> dict[str, int | None]:
        """Return the stable memory payload."""

        return {
            "baseline_bytes": self.baseline_bytes,
            "delta_bytes": self.delta_bytes,
            "peak_bytes": self.peak_bytes,
            "peak_rss_bytes": self.peak_bytes,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class ComparatorMetrics:
    """Aggregate quality, privacy, latency, and memory measurements."""

    precision: float
    recall: float
    f1: float
    character_recall: float
    critical_leakage: float
    critical_leakage_count: int
    critical_span_count: int
    latency: LatencyMetrics
    memory: ComparatorMemoryMetrics
    within_budget: bool | None = None

    @property
    def critical_leakage_rate(self) -> float:
        """Compatibility alias for the critical leakage rate."""

        return self.critical_leakage

    @property
    def peak_memory_bytes(self) -> int | None:
        """Return the observed peak memory in bytes."""

        return self.memory.peak_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate-only metrics with no source surfaces."""

        return {
            "character_recall": self.character_recall,
            "critical_leakage": self.critical_leakage,
            "critical_leakage_count": self.critical_leakage_count,
            "critical_leakage_rate": self.critical_leakage,
            "critical_span_count": self.critical_span_count,
            "f1": self.f1,
            "latency": self.latency.to_dict(),
            "memory": self.memory.to_dict(),
            "peak_memory_bytes": self.memory.peak_bytes,
            "precision": self.precision,
            "recall": self.recall,
            "within_budget": self.within_budget,
        }

    def __getitem__(self, key: str) -> Any:
        """Support mapping-style metric access."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class ComparatorResult:
    """One adapter row in a comparator report."""

    adapter: str
    status: str
    fixture_count: int
    version: str = "local"
    metrics: ComparatorMetrics | None = None
    reason: str | None = None
    metadata_digest: str | None = None

    @property
    def system(self) -> str:
        """Compatibility alias for matrix-style consumers."""

        return self.adapter

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-free adapter row."""

        return {
            "adapter": self.adapter,
            "fixture_count": self.fixture_count,
            "metadata_digest": self.metadata_digest,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "reason": self.reason,
            "status": self.status,
            "system": self.adapter,
            "version": self.version,
        }

    def __getitem__(self, key: str) -> Any:
        """Support mapping-style result access."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class ComparatorReport:
    """Deterministic, aggregate-only report for a comparator benchmark."""

    suite: str
    fixture_count: int
    results: tuple[ComparatorResult, ...]
    fixture_digests: tuple[str, ...]
    critical_labels: tuple[str, ...]
    budget: ComparatorBudget
    seed: int = 0
    generated_at: str | None = None
    metadata_digest: str | None = None
    reproducibility_hash: str = ""
    schema_version: str = COMPARATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "fixture_digests", tuple(self.fixture_digests))
        object.__setattr__(
            self,
            "critical_labels",
            tuple(sorted(str(label) for label in self.critical_labels)),
        )
        if not self.reproducibility_hash:
            object.__setattr__(
                self,
                "reproducibility_hash",
                stable_hash(self._reproducibility_payload()),
            )

    @property
    def rows(self) -> tuple[ComparatorResult, ...]:
        """Compatibility alias for row-oriented consumers."""

        return self.results

    @property
    def scored_results(self) -> tuple[ComparatorResult, ...]:
        """Return adapters that produced measurements."""

        return tuple(row for row in self.results if row.status == STATUS_SCORED)

    @property
    def skipped_results(self) -> tuple[ComparatorResult, ...]:
        """Return adapters that were unavailable locally."""

        return tuple(row for row in self.results if row.status == STATUS_NOT_AVAILABLE)

    def result(self, adapter: str) -> ComparatorResult:
        """Return the result named *adapter*."""

        for row in self.results:
            if row.adapter == adapter:
                return row
        raise KeyError(adapter)

    def _reproducibility_payload(self) -> dict[str, Any]:
        return {
            "adapters": [
                {
                    "adapter": row.adapter,
                    "metadata_digest": row.metadata_digest,
                    "status": row.status,
                    "version": row.version,
                }
                for row in self.results
            ],
            "budget": self.budget.to_dict(),
            "critical_labels": list(self.critical_labels),
            "fixture_digests": list(self.fixture_digests),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "suite": self.suite,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free report payload."""

        return {
            "budget": self.budget.to_dict(),
            "critical_labels": list(self.critical_labels),
            "fixture_count": self.fixture_count,
            "fixture_digests": list(self.fixture_digests),
            "generated_at": self.generated_at,
            "metadata_digest": self.metadata_digest,
            "reproducibility_hash": self.reproducibility_hash,
            "results": [row.to_dict() for row in self.results],
            "schema_version": self.schema_version,
            "seed": self.seed,
            "suite": self.suite,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a JSON report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render aggregate measurements as deterministic Markdown."""

        lines = [
            f"# Comparator Benchmark: {_markdown_cell(self.suite)}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Fixtures | {self.fixture_count} |",
            f"| Seed | {self.seed} |",
            f"| Reproducibility hash | `{self.reproducibility_hash}` |",
            "",
            "## Adapters",
            "",
            (
                "| Adapter | Status | Recall | Precision | Critical leakage | "
                "P95 latency (ms) | Peak memory (bytes) | Budget |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in self.results:
            if row.metrics is None:
                lines.append(
                    f"| `{_markdown_cell(row.adapter)}` | {row.status} | n/a | n/a | "
                    f"n/a | n/a | n/a | n/a |"
                )
                continue
            metrics = row.metrics
            budget = (
                "pass"
                if metrics.within_budget is True
                else "fail"
                if metrics.within_budget is False
                else "n/a"
            )
            lines.append(
                f"| `{_markdown_cell(row.adapter)}` | {row.status} | "
                f"{metrics.recall:.2%} | {metrics.precision:.2%} | "
                f"{metrics.critical_leakage:.2%} | "
                f"{metrics.latency.p95_ms:.3f} | "
                f"{_markdown_value(metrics.memory.peak_bytes)} | {budget} |"
            )
        lines.extend(
            [
                "",
                "All fixture identifiers, source text, predicted surfaces, and "
                "adapter exception text are excluded from this report.",
                "",
            ]
        )
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        """Write a Markdown report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Support mapping-style report access."""

        return self.to_dict()[key]


def load_comparator_fixtures(path: str | Path) -> tuple[ComparatorFixture, ...]:
    """Load local JSON or JSONL synthetic fixtures without network access."""

    fixture_path = Path(path)
    try:
        raw = fixture_path.read_text(encoding="utf-8")
        if fixture_path.suffix.lower() == ".jsonl":
            rows: Any = [
                json.loads(line)
                for line in raw.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            rows = json.loads(raw)
    except Exception:
        raise ComparatorFixtureError("could not load comparator fixtures") from None

    if isinstance(rows, Mapping):
        rows = rows.get("fixtures")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ComparatorFixtureError("invalid comparator fixture collection")
    try:
        return tuple(ComparatorFixture.from_mapping(row) for row in rows)
    except ComparatorFixtureError:
        raise
    except Exception:
        raise ComparatorFixtureError("invalid comparator fixture collection") from None


def run_comparator_benchmark(
    fixtures: (
        str
        | Path
        | ComparatorFixture
        | BenchmarkFixture
        | Mapping[str, Any]
        | Sequence[ComparatorFixture | BenchmarkFixture | Mapping[str, Any]]
    ),
    adapters: Iterable[ComparatorAdapter | Mapping[str, Any] | Any],
    *,
    suite: str = "synthetic-comparator",
    budget: ComparatorBudget | None = None,
    critical_labels: Iterable[str] | None = None,
    clock: Clock | None = None,
    memory_sampler: MemorySampler | None = None,
    generated_at: str | None = None,
    seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> ComparatorReport:
    """Run every local adapter over the same synthetic fixtures.

    No model, package, or network resource is downloaded by this function. The
    adapter owns its local setup and receives only fixture text and language.
    Reports contain aggregate measurements and hashes; source text and adapter
    exception messages are deliberately discarded.

    Args:
        fixtures: Inline fixtures, existing benchmark fixtures, or a local JSON
            / JSONL path.
        adapters: User-supplied :class:`ComparatorAdapter` values, compatible
            mappings, or adapter objects exposing ``name`` and ``runner``.
        budget: Optional limits shared by every adapter. ``p95`` latency and
            observed peak memory are checked against the limits.
        critical_labels: Labels included in the critical leakage slice.
        clock: Optional monotonic clock for deterministic tests.
        memory_sampler: Optional byte sampler for deterministic tests.
        generated_at: Optional caller-supplied provenance timestamp. It is not
            generated implicitly, so repeated report serialization is stable.
        seed: Seed used for the harness-local standard-library random context.
        metadata: Optional provenance mapping stored only as a digest.
    """

    resolved_fixtures = _resolve_fixtures(fixtures)
    if not resolved_fixtures:
        raise ComparatorFixtureError("comparator benchmark requires fixtures")
    _validate_unique_fixtures(resolved_fixtures)

    resolved_adapters = tuple(_coerce_adapter(adapter) for adapter in adapters)
    if not resolved_adapters:
        raise ValueError("comparator benchmark requires adapters")
    _validate_unique_adapters(resolved_adapters)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("comparator seed must be an integer")
    active_budget = budget or ComparatorBudget()
    labels = _resolve_critical_labels(critical_labels)
    run_clock = clock or time.perf_counter
    sample_memory = memory_sampler or _peak_rss_bytes
    results: list[ComparatorResult] = []

    for adapter in resolved_adapters:
        if adapter.requires_network:
            results.append(_unavailable_result(adapter, len(resolved_fixtures)))
            continue
        if adapter.runner is None:
            results.append(_unavailable_result(adapter, len(resolved_fixtures)))
            continue

        try:
            metrics = _run_adapter(
                adapter,
                resolved_fixtures,
                critical_labels=labels,
                budget=active_budget,
                clock=run_clock,
                memory_sampler=sample_memory,
                seed=seed,
            )
        except (ComparatorAdapterUnavailable, ImportError):
            results.append(_unavailable_result(adapter, len(resolved_fixtures)))
            continue
        except Exception:
            raise ComparatorExecutionError(
                "comparator adapter execution failed"
            ) from None
        results.append(
            ComparatorResult(
                adapter=adapter.name,
                status=STATUS_SCORED,
                fixture_count=len(resolved_fixtures),
                version=adapter.version,
                metrics=metrics,
                metadata_digest=_metadata_digest(adapter.metadata),
            )
        )

    return ComparatorReport(
        suite=str(suite).strip() or "synthetic-comparator",
        fixture_count=len(resolved_fixtures),
        results=tuple(results),
        fixture_digests=tuple(fixture.digest for fixture in resolved_fixtures),
        critical_labels=tuple(labels),
        budget=active_budget,
        seed=seed,
        generated_at=generated_at,
        metadata_digest=_metadata_digest(metadata or {}),
    )


def run_comparator_harness(*args: Any, **kwargs: Any) -> ComparatorReport:
    """Compatibility alias for :func:`run_comparator_benchmark`."""

    return run_comparator_benchmark(*args, **kwargs)


def _resolve_fixtures(
    fixtures: (
        str
        | Path
        | ComparatorFixture
        | BenchmarkFixture
        | Mapping[str, Any]
        | Sequence[ComparatorFixture | BenchmarkFixture | Mapping[str, Any]]
    ),
) -> tuple[ComparatorFixture, ...]:
    if isinstance(fixtures, (str, Path)):
        return load_comparator_fixtures(fixtures)
    if isinstance(fixtures, ComparatorFixture | BenchmarkFixture | Mapping):
        values: Sequence[Any] = (fixtures,)
    else:
        values = fixtures
    try:
        return tuple(_coerce_fixture(value) for value in values)
    except ComparatorFixtureError:
        raise
    except Exception:
        raise ComparatorFixtureError("invalid comparator fixture collection") from None


def _coerce_fixture(value: Any) -> ComparatorFixture:
    if isinstance(value, ComparatorFixture):
        return value
    if isinstance(value, BenchmarkFixture):
        return ComparatorFixture.from_benchmark_fixture(value)
    if isinstance(value, Mapping):
        return ComparatorFixture.from_mapping(value)
    raise ComparatorFixtureError("invalid comparator fixture")


def _coerce_adapter(value: Any) -> ComparatorAdapter:
    if isinstance(value, ComparatorAdapter):
        return value
    if isinstance(value, Mapping):
        runner = value.get("runner") or value.get("predict")
        return ComparatorAdapter(
            name=value.get("name") or value.get("system") or "adapter",
            runner=runner,
            version=value.get("version") or "local",
            model_name=value.get("model_name"),
            device=value.get("device") or "cpu",
            requires_network=_flag(value.get("requires_network", False)),
            unavailable_reason=value.get("unavailable_reason"),
            metadata=value.get("metadata") or {},
        )
    if callable(value):
        return ComparatorAdapter(
            name=getattr(value, "name", None) or getattr(value, "__name__", "adapter"),
            runner=value,
        )

    runner = getattr(value, "runner", None) or getattr(value, "predict", None)
    if runner is None and not hasattr(value, "name"):
        raise ValueError("invalid comparator adapter")
    return ComparatorAdapter(
        name=getattr(value, "name", None) or "adapter",
        runner=runner,
        version=getattr(value, "version", None) or "local",
        model_name=getattr(value, "model_name", None),
        device=getattr(value, "device", None) or "cpu",
        requires_network=_flag(getattr(value, "requires_network", False)),
        unavailable_reason=getattr(value, "unavailable_reason", None),
        metadata=getattr(value, "metadata", None) or {},
    )


def _run_adapter(
    adapter: ComparatorAdapter,
    fixtures: Sequence[ComparatorFixture],
    *,
    critical_labels: frozenset[str],
    budget: ComparatorBudget,
    clock: Clock,
    memory_sampler: MemorySampler,
    seed: int,
) -> ComparatorMetrics:
    true_positives = 0
    predicted_count = 0
    gold_count = 0
    covered_characters = 0
    gold_characters = 0
    leaked_characters = 0
    critical_characters = 0
    critical_leakage_count = 0
    critical_span_count = 0
    latencies: list[float] = []
    memory_samples: list[int] = []

    with _random_context(seed):
        for fixture in fixtures:
            before_memory = _sample_memory(memory_sampler)
            started = clock()
            try:
                raw_predictions = _invoke_runner(adapter, fixture)
                predictions = _normalize_predictions(raw_predictions, fixture)
            except ComparatorAdapterUnavailable:
                raise
            except ImportError:
                raise
            except Exception:
                raise ComparatorExecutionError(
                    "invalid comparator adapter output"
                ) from None
            elapsed = float(clock()) - float(started)
            if not math.isfinite(elapsed):
                raise ComparatorExecutionError("invalid comparator adapter timing")
            latencies.append(max(elapsed * 1000.0, 0.0))
            after_memory = _sample_memory(memory_sampler)
            for value in (before_memory, after_memory):
                if value is not None:
                    memory_samples.append(value)

            exact = compute_exact_span_f1(fixture.gold_spans, predictions)
            character = compute_character_recall(
                fixture.gold_spans,
                predictions,
                source_text=fixture.text,
            )
            leakage = _critical_leakage(
                fixture,
                predictions,
                critical_labels=critical_labels,
            )
            true_positives += exact.true_positives
            predicted_count += exact.true_positives + exact.false_positives
            gold_count += exact.true_positives + exact.false_negatives
            covered_characters += int(character.numerator)
            gold_characters += int(character.denominator)
            leaked_characters += leakage[0]
            critical_characters += leakage[1]
            critical_leakage_count += leakage[2]
            critical_span_count += leakage[3]

    precision = _ratio(true_positives, predicted_count, zero_denominator=1.0)
    recall = _ratio(true_positives, gold_count, zero_denominator=1.0)
    f1 = _f1(precision, recall)
    character_recall = _ratio(
        covered_characters,
        gold_characters,
        zero_denominator=1.0,
    )
    critical_leakage = _ratio(
        leaked_characters,
        critical_characters,
        zero_denominator=0.0,
    )
    latency = compute_latency_summary(latencies)
    memory = _memory_metrics(memory_samples)
    within_budget: bool | None = None
    if budget.max_latency_ms is not None or budget.max_memory_bytes is not None:
        latency_ok = (
            budget.max_latency_ms is None or latency.p95_ms <= budget.max_latency_ms
        )
        memory_ok = (
            budget.max_memory_bytes is None
            or memory.peak_bytes is not None
            and memory.peak_bytes <= budget.max_memory_bytes
        )
        within_budget = bool(latency_ok and memory_ok)
    return ComparatorMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        character_recall=character_recall,
        critical_leakage=critical_leakage,
        critical_leakage_count=critical_leakage_count,
        critical_span_count=critical_span_count,
        latency=latency,
        memory=memory,
        within_budget=within_budget,
    )


def _invoke_runner(adapter: ComparatorAdapter, fixture: ComparatorFixture) -> Any:
    runner = adapter.runner
    if runner is None:
        raise ComparatorAdapterUnavailable()
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(fixture.text, fixture.language)

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    language_keyword = signature.parameters.get("language")
    if has_varargs or len(positional) >= 3:
        return runner(
            fixture.as_benchmark_fixture(),
            adapter.model_name or adapter.name,
            adapter.device,
        )
    if len(positional) >= 2:
        return runner(fixture.text, fixture.language)
    if (
        language_keyword is not None
        and language_keyword.kind is inspect.Parameter.KEYWORD_ONLY
    ):
        return runner(fixture.text, language=fixture.language)
    if positional:
        return runner(fixture.text)
    return runner()


def _normalize_predictions(
    raw_predictions: Any,
    fixture: ComparatorFixture,
) -> tuple[EvalSpan, ...]:
    if raw_predictions is None:
        values: Iterable[Any] = ()
    elif isinstance(raw_predictions, Mapping):
        values = (raw_predictions,)
    elif isinstance(raw_predictions, (str, bytes, bytearray)):
        raise ComparatorExecutionError("invalid comparator adapter output")
    else:
        values = raw_predictions
    try:
        predictions = _normalize_spans(
            values,
            language=fixture.language,
            text=fixture.text,
        )
    except Exception:
        raise ComparatorExecutionError("invalid comparator adapter output") from None
    return tuple(sorted(predictions, key=_span_sort_key))


def _normalize_spans(
    spans: Iterable[Any],
    *,
    language: str,
    text: str,
) -> tuple[EvalSpan, ...]:
    normalized = normalize_eval_spans(
        spans,
        default_language=language,
        default_device="cpu",
        source_text=text,
    )
    result: list[EvalSpan] = []
    for span in normalized:
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError("span offsets are invalid")
        label = normalize_label(str(span.label), language)
        if not label:
            raise ValueError("span label is empty")
        result.append(replace(span, label=label, language=language))
    return tuple(sorted(result, key=_span_sort_key))


def _critical_leakage(
    fixture: ComparatorFixture,
    predictions: Sequence[EvalSpan],
    *,
    critical_labels: frozenset[str],
) -> tuple[int, int, int, int]:
    gold = [
        span
        for span in fixture.gold_spans
        if normalize_label(span.label, fixture.language) in critical_labels
    ]
    leakage = compute_leakage_rate(
        gold,
        predictions,
        source_text=fixture.text,
    )
    missed = 0
    for gold_span in gold:
        if not any(
            prediction.label == gold_span.label
            and prediction.start <= gold_span.start
            and prediction.end >= gold_span.end
            for prediction in predictions
        ):
            missed += 1
    return (
        leakage.leaked_chars,
        leakage.total_chars,
        missed,
        len(gold),
    )


def _memory_metrics(samples: Sequence[int]) -> ComparatorMemoryMetrics:
    if not samples:
        return ComparatorMemoryMetrics(
            peak_bytes=None,
            baseline_bytes=None,
            delta_bytes=None,
            sample_count=0,
        )
    baseline = samples[0]
    peak = max(samples)
    return ComparatorMemoryMetrics(
        peak_bytes=peak,
        baseline_bytes=baseline,
        delta_bytes=max(peak - baseline, 0),
        sample_count=len(samples),
    )


def _sample_memory(sampler: MemorySampler) -> int | None:
    try:
        value = sampler()
    except Exception:
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


def _resolve_critical_labels(labels: Iterable[str] | None) -> frozenset[str]:
    values = _DEFAULT_CRITICAL_LABELS if labels is None else labels
    try:
        return frozenset(
            normalize_label(str(label), "en") for label in values if str(label).strip()
        )
    except Exception:
        raise ValueError("invalid critical label configuration") from None


def _unavailable_result(
    adapter: ComparatorAdapter,
    fixture_count: int,
) -> ComparatorResult:
    return ComparatorResult(
        adapter=adapter.name,
        status=STATUS_NOT_AVAILABLE,
        fixture_count=fixture_count,
        version=adapter.version,
        reason="adapter is not available for this offline run",
        metadata_digest=_metadata_digest(adapter.metadata),
    )


def _validate_unique_fixtures(fixtures: Sequence[ComparatorFixture]) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ComparatorFixtureError("comparator fixture identifiers must be unique")


def _validate_unique_adapters(adapters: Sequence[ComparatorAdapter]) -> None:
    names = [adapter.name for adapter in adapters]
    if len(names) != len(set(names)):
        raise ValueError("comparator adapter names must be unique")


def _span_sort_key(span: EvalSpan) -> tuple[int, int, str, str]:
    return span.start, span.end, span.label, span.language


def _ratio(numerator: int, denominator: int, *, zero_denominator: float) -> float:
    if denominator == 0:
        return zero_denominator
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _metadata_digest(value: Mapping[str, Any]) -> str:
    return stable_hash(_digest_value(value))


def _digest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _digest_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_digest_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_value(value: Any) -> str:
    return "n/a" if value is None else str(value)


@contextmanager
def _random_context(seed: int) -> Iterator[None]:
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


# Short aliases keep the contract discoverable for callers using baseline
# terminology while leaving the established matrix module's public names intact.
BaselineAdapter = ComparatorAdapter
ComparatorCase = ComparatorFixture
ResourceBudget = ComparatorBudget


__all__ = [
    "BaselineAdapter",
    "ComparatorAdapter",
    "ComparatorAdapterUnavailable",
    "ComparatorBudget",
    "ComparatorCase",
    "ComparatorError",
    "ComparatorExecutionError",
    "ComparatorFixture",
    "ComparatorFixtureError",
    "ComparatorMemoryMetrics",
    "ComparatorMetrics",
    "ComparatorReport",
    "ComparatorResult",
    "COMPARATOR_SCHEMA_VERSION",
    "ResourceBudget",
    "STATUS_NOT_AVAILABLE",
    "STATUS_SCORED",
    "load_comparator_fixtures",
    "run_comparator_benchmark",
    "run_comparator_harness",
]
