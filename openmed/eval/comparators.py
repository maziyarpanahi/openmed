"""Comparator benchmark matrix runner for interop adapters."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    load_fixtures,
    run_benchmark,
)
from openmed.eval.report import (
    BenchmarkReport,
    _display_value,
    _format_percent,
    _format_value,
    _nested_get,
    _plain,
)

DEFAULT_COMPARATOR_ADAPTERS: tuple[str, ...] = (
    "presidio",
    "philter",
    "pydeid",
    "gliner_biomed",
)
OPENMED_SYSTEM_NAME = "OpenMed"
STATUS_SCORED = "scored"
STATUS_NOT_AVAILABLE = "not_available"


class ComparatorUnavailable(ImportError):
    """Raised when a comparator cannot run in the current environment."""


@dataclass(frozen=True)
class ComparatorAdapter:
    """One adapter entry for the comparator matrix.

    Args:
        name: Display name for the adapter row.
        runner: Harness-compatible runner for this adapter. It receives
            ``(fixture, model_name, device)`` and returns span-like predictions.
        model_name: Optional model/report name passed to ``run_benchmark``.
        device: Optional device label for this adapter's report.
        unavailable_reason: Precomputed reason to record as ``not_available``.
        metadata: PHI-free metadata attached to the adapter benchmark report.
    """

    name: str
    runner: ModelRunner | None = None
    model_name: str | None = None
    device: str | None = None
    unavailable_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = str(self.name).strip()
        if not normalized:
            raise ValueError("comparator adapter name must be non-empty")
        object.__setattr__(self, "name", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ComparatorMatrixRow:
    """One system row in a comparator matrix report."""

    system: str
    status: str
    fixture_count: int
    leakage_rate: float | None = None
    character_recall: float | None = None
    exact_span_f1: float | None = None
    relaxed_span_f1: float | None = None
    reason: str | None = None
    benchmark_report: BenchmarkReport | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready row payload."""

        payload: dict[str, Any] = {
            "character_recall": self.character_recall,
            "exact_span_f1": self.exact_span_f1,
            "fixture_count": self.fixture_count,
            "leakage_rate": self.leakage_rate,
            "metadata": _plain(self.metadata),
            "reason": self.reason,
            "relaxed_span_f1": self.relaxed_span_f1,
            "status": self.status,
            "system": self.system,
        }
        if self.benchmark_report is not None:
            payload["benchmark_report"] = self.benchmark_report.to_dict()
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ComparatorMatrixReport:
    """Side-by-side benchmark report for OpenMed and comparator adapters."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    rows: tuple[ComparatorMatrixRow, ...]
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def scored_rows(self) -> tuple[ComparatorMatrixRow, ...]:
        """Return rows that completed scoring."""

        return tuple(row for row in self.rows if row.status == STATUS_SCORED)

    @property
    def skipped_rows(self) -> tuple[ComparatorMatrixRow, ...]:
        """Return rows skipped because their adapter was unavailable."""

        return tuple(row for row in self.rows if row.status == STATUS_NOT_AVAILABLE)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report payload."""

        return {
            "device": self.device,
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
            "model_name": self.model_name,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
            "suite": self.suite,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the matrix to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic matrix JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render the matrix as deterministic Markdown."""

        lines = [
            f"# Comparator Matrix: {self.suite}",
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| OpenMed Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {_format_value(self.fixture_count)} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Systems",
                "",
                (
                    "| System | Status | Leakage Rate | Character Recall | "
                    "Exact F1 | Relaxed F1 | Fixtures | Notes |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in self.rows:
            lines.append(
                "| "
                f"`{_markdown_cell(row.system)}` | "
                f"{_markdown_cell(row.status)} | "
                f"{_format_percent(row.leakage_rate)} | "
                f"{_format_percent(row.character_recall)} | "
                f"{_format_percent(row.exact_span_f1)} | "
                f"{_format_percent(row.relaxed_span_f1)} | "
                f"{_format_value(row.fixture_count)} | "
                f"{_markdown_cell(_display_value(row.reason))} |"
            )

        if self.metadata:
            lines.extend(["", "## Metadata", "", "| Key | Value |", "|---|---|"])
            for key, value in _flatten(_plain(self.metadata)):
                lines.append(
                    f"| `{_markdown_cell(key)}` | {_markdown_cell(_format_value(value))} |"
                )

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic matrix Markdown to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


ComparatorAdapterLike = (
    ComparatorAdapter | str | Mapping[str, Any] | Callable[..., Iterable[Any]]
)


def run_comparator_matrix(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
    adapters: Iterable[ComparatorAdapterLike] | None = None,
    *,
    suite_name: str | None = None,
    model_name: str = OPENMED_SYSTEM_NAME,
    device: str = "cpu",
    openmed_runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ComparatorMatrixReport:
    """Run OpenMed and comparator adapters over the same fixture suite.

    ``suite`` may be a sequence of ``BenchmarkFixture`` objects, a sequence of
    fixture mappings, a fixture JSON/JSONL path, or a registered suite name.
    Named adapters are resolved lazily. Adapter runners that raise
    ``ImportError`` or ``ComparatorUnavailable`` are recorded as
    ``not_available`` instead of failing the matrix run.
    """

    fixtures, resolved_suite_name = _resolve_suite(suite, suite_name=suite_name)
    adapter_rows = tuple(DEFAULT_COMPARATOR_ADAPTERS if adapters is None else adapters)
    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids", [fixture.fixture_id for fixture in fixtures]
    )
    report_metadata.setdefault(
        "adapters", [_adapter_display_name(adapter) for adapter in adapter_rows]
    )

    rows: list[ComparatorMatrixRow] = []
    openmed_report = run_benchmark(
        fixtures,
        suite=resolved_suite_name,
        model_name=model_name,
        device=device,
        runner=openmed_runner,
        generated_at=generated_at,
        metadata={"system": OPENMED_SYSTEM_NAME},
    )
    rows.append(_row_from_report(OPENMED_SYSTEM_NAME, openmed_report))

    for adapter_like in adapter_rows:
        adapter = _coerce_adapter(adapter_like)
        if adapter.unavailable_reason:
            rows.append(_not_available_row(adapter, len(fixtures)))
            continue

        if adapter.runner is None:
            rows.append(
                _not_available_row(
                    adapter,
                    len(fixtures),
                    reason="adapter has no runnable benchmark entrypoint",
                )
            )
            continue

        try:
            adapter_report = run_benchmark(
                fixtures,
                suite=resolved_suite_name,
                model_name=adapter.model_name or adapter.name,
                device=adapter.device or device,
                runner=adapter.runner,
                generated_at=generated_at,
                metadata={
                    "adapter": adapter.name,
                    "system": adapter.name,
                    **dict(adapter.metadata or {}),
                },
            )
        except (ComparatorUnavailable, ImportError, ModuleNotFoundError) as exc:
            rows.append(_not_available_row(adapter, len(fixtures), reason=str(exc)))
            continue
        rows.append(_row_from_report(adapter.name, adapter_report))

    return ComparatorMatrixReport(
        suite=resolved_suite_name,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        rows=tuple(rows),
        generated_at=generated_at,
        metadata=report_metadata,
    )


def _resolve_suite(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    suite_name: str | None,
) -> tuple[list[BenchmarkFixture], str]:
    if isinstance(suite, (str, Path)):
        suite_text = str(suite)
        fixture_path = Path(suite)
        if fixture_path.exists():
            return load_fixtures(fixture_path), suite_name or fixture_path.stem

        from openmed.eval.suites import load_suite_fixtures

        return (
            [
                fixture
                if isinstance(fixture, BenchmarkFixture)
                else BenchmarkFixture.from_mapping(fixture)
                for fixture in load_suite_fixtures(suite_text)
            ],
            suite_name or suite_text,
        )

    fixtures = [
        fixture
        if isinstance(fixture, BenchmarkFixture)
        else BenchmarkFixture.from_mapping(fixture)
        for fixture in suite
    ]
    return fixtures, suite_name or "custom"


def _coerce_adapter(adapter: ComparatorAdapterLike) -> ComparatorAdapter:
    if isinstance(adapter, ComparatorAdapter):
        return adapter
    if isinstance(adapter, str):
        return _named_comparator_adapter(adapter)
    if isinstance(adapter, Mapping):
        runner = adapter.get("runner") or adapter.get("callable")
        return ComparatorAdapter(
            name=str(adapter.get("name") or adapter.get("system") or "adapter"),
            runner=_wrap_runner(runner) if callable(runner) else None,
            model_name=_optional_string(adapter.get("model_name")),
            device=_optional_string(adapter.get("device")),
            unavailable_reason=_optional_string(adapter.get("unavailable_reason")),
            metadata=_mapping_or_empty(adapter.get("metadata")),
        )

    runner = getattr(adapter, "runner", None) or getattr(adapter, "run", None)
    if callable(runner):
        return ComparatorAdapter(
            name=_object_adapter_name(adapter),
            runner=_wrap_runner(runner),
            model_name=_optional_string(getattr(adapter, "model_name", None)),
            device=_optional_string(getattr(adapter, "device", None)),
            unavailable_reason=_optional_string(
                getattr(adapter, "unavailable_reason", None)
            ),
            metadata=_mapping_or_empty(getattr(adapter, "metadata", None)),
        )

    if callable(adapter):
        return ComparatorAdapter(
            name=_object_adapter_name(adapter),
            runner=_wrap_runner(adapter),
        )

    raise TypeError(f"unsupported comparator adapter: {adapter!r}")


def _named_comparator_adapter(name: str) -> ComparatorAdapter:
    key = str(name).strip().lower().replace("-", "_")
    if key == "presidio":
        return ComparatorAdapter(name="presidio", runner=_presidio_runner())
    if key == "gliner_biomed":
        return ComparatorAdapter(name="gliner_biomed", runner=_gliner_biomed_runner())
    if key in {"philter", "pydeid"}:
        return ComparatorAdapter(
            name=key,
            unavailable_reason=(
                f"{key} exposes conversion helpers but no stable in-process text "
                "runner; pass ComparatorAdapter(..., runner=...) with a configured "
                "detector to include it in the matrix"
            ),
        )

    from openmed.interop import adapter_spec

    adapter_spec(key)
    return ComparatorAdapter(
        name=key,
        unavailable_reason=(
            f"{key} is registered for interop conversion but has no comparator runner"
        ),
    )


def _presidio_runner() -> ModelRunner:
    analyzer: Any | None = None

    def run_fixture(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> Iterable[Any]:
        del model_name, device
        nonlocal analyzer
        if analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:
                raise ComparatorUnavailable(
                    "Presidio comparator requires the 'presidio' extra. "
                    "Install with `pip install openmed[presidio]`."
                ) from exc
            analyzer = AnalyzerEngine()

        from openmed.interop import presidio

        language = fixture.language if fixture.language else "en"
        try:
            results = analyzer.analyze(text=fixture.text, language=language)
        except ValueError:
            results = analyzer.analyze(text=fixture.text, language="en")
        return presidio.to_canonical(results, text=fixture.text)

    return run_fixture


def _gliner_biomed_runner() -> ModelRunner:
    model: Any | None = None

    def run_fixture(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> Iterable[Any]:
        del model_name, device
        nonlocal model
        from openmed.interop import gliner_biomed

        if model is None:
            try:
                from gliner import GLiNER
            except ImportError as exc:
                raise ComparatorUnavailable(
                    "GLiNER-BioMed comparator requires the 'gliner' extra. "
                    "Install with `pip install openmed[gliner]`."
                ) from exc
            model = GLiNER.from_pretrained(gliner_biomed.DEFAULT_MODEL_ID)
        return gliner_biomed.predict_to_canonical(fixture.text, model=model)

    return run_fixture


def _wrap_runner(runner: Callable[..., Iterable[Any]]) -> ModelRunner:
    if _accepts_harness_runner_args(runner):
        return runner  # type: ignore[return-value]

    def run_fixture(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> Iterable[Any]:
        del model_name, device
        return runner(fixture)

    return run_fixture


def _accepts_harness_runner_args(runner: Callable[..., Iterable[Any]]) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    return len(positional) >= 3 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )


def _row_from_report(system: str, report: BenchmarkReport) -> ComparatorMatrixRow:
    return ComparatorMatrixRow(
        system=system,
        status=STATUS_SCORED,
        fixture_count=report.fixture_count,
        leakage_rate=_metric_float(report, "leakage.overall"),
        character_recall=_metric_float(report, "character_recall.rate"),
        exact_span_f1=_metric_float(report, "exact_span_f1.f1"),
        relaxed_span_f1=_metric_float(report, "relaxed_span_f1.f1"),
        benchmark_report=report,
        metadata={"suite": report.suite, "device": report.device},
    )


def _not_available_row(
    adapter: ComparatorAdapter,
    fixture_count: int,
    *,
    reason: str | None = None,
) -> ComparatorMatrixRow:
    return ComparatorMatrixRow(
        system=adapter.name,
        status=STATUS_NOT_AVAILABLE,
        fixture_count=fixture_count,
        reason=reason or adapter.unavailable_reason or "adapter is not available",
        metadata={"adapter": adapter.name, **dict(adapter.metadata or {})},
    )


def _metric_float(report: BenchmarkReport, dotted_key: str) -> float | None:
    value = _nested_get(report.metrics, dotted_key)
    if value is None:
        return None
    return float(value)


def _adapter_display_name(adapter: ComparatorAdapterLike) -> str:
    if isinstance(adapter, ComparatorAdapter):
        return adapter.name
    if isinstance(adapter, str):
        return adapter
    if isinstance(adapter, Mapping):
        return str(adapter.get("name") or adapter.get("system") or "adapter")
    return _object_adapter_name(adapter)


def _object_adapter_name(adapter: Any) -> str:
    explicit = getattr(adapter, "name", None)
    if explicit:
        return str(explicit)
    function_name = getattr(adapter, "__name__", None)
    if function_name:
        return str(function_name)
    return adapter.__class__.__name__


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(value[key], child_prefix))
        return rows
    return [(prefix, value)]


__all__ = [
    "ComparatorAdapter",
    "ComparatorMatrixReport",
    "ComparatorMatrixRow",
    "ComparatorUnavailable",
    "DEFAULT_COMPARATOR_ADAPTERS",
    "OPENMED_SYSTEM_NAME",
    "STATUS_NOT_AVAILABLE",
    "STATUS_SCORED",
    "run_comparator_matrix",
]
