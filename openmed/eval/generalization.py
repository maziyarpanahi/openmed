"""Cross-corpus generalization evaluation for benchmark suites.

The evaluator deliberately builds on :func:`openmed.eval.harness.run_benchmark`
so every corpus is scored with the same metric implementation and report
serializer.  Inputs may be local JSON/JSONL fixture paths, registered suite
names, or already-loaded synthetic :class:`BenchmarkFixture` objects.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
    load_fixtures,
    run_benchmark,
)
from openmed.eval.report import BenchmarkReport

GENERALIZATION_SCHEMA_VERSION = 1
GENERALIZATION_ARTIFACT_TYPE = "openmed.eval.generalization"
GENERALIZATION_SUITE = "cross-corpus-generalization"
GENERALIZATION_METRICS = ("leakage_rate", "recall", "f1")

_METRIC_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "leakage_rate": (("leakage", "overall"), ("leakage", "rate")),
    "recall": (
        ("character_recall", "rate"),
        ("character_recall", "overall"),
    ),
    "f1": (("exact_span_f1", "f1"),),
}
_GAP_DIRECTION = {
    "leakage_rate": 1.0,
    "recall": -1.0,
    "f1": -1.0,
}


@dataclass(frozen=True, slots=True)
class GeneralizationReport:
    """Aggregate in-domain versus out-of-domain benchmark evidence.

    ``deltas`` is source-corpus first and contains the signed mathematical
    delta ``out_of_domain - in_domain`` for each metric.  ``metrics`` is
    metric first and retains both scores plus a direction-aware ``gap``:
    leakage increases are gaps, while recall and F1 decreases are gaps.  The
    headline gap is the mean of those direction-aware gaps across all
    out-of-domain corpora and the three reported metrics.
    """

    model_name: str
    device: str
    in_domain_suite: str
    out_of_domain_suites: tuple[str, ...]
    in_domain_report: BenchmarkReport
    out_of_domain_reports: Mapping[str, BenchmarkReport]
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
    deltas: Mapping[str, Mapping[str, float]]
    headline_gap: float
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    suite: str = GENERALIZATION_SUITE

    @property
    def generalization_gap(self) -> float:
        """Return the headline direction-aware generalization gap."""

        return self.headline_gap

    @property
    def headline_generalization_gap(self) -> float:
        """Backward-compatible descriptive alias for :attr:`headline_gap`."""

        return self.headline_gap

    @property
    def fixture_count(self) -> int:
        """Return the total number of fixtures scored across all corpora."""

        return self.in_domain_report.fixture_count + sum(
            report.fixture_count for report in self.out_of_domain_reports.values()
        )

    @property
    def in_domain(self) -> BenchmarkReport:
        """Return the in-domain benchmark report."""

        return self.in_domain_report

    @property
    def out_of_domain(self) -> Mapping[str, BenchmarkReport]:
        """Return benchmark reports keyed by source-corpus name."""

        return self.out_of_domain_reports

    @property
    def reports(self) -> dict[str, BenchmarkReport]:
        """Return all child reports keyed by their evaluation role."""

        return {
            "in_domain": self.in_domain_report,
            **dict(self.out_of_domain_reports),
        }

    @property
    def delta_by_metric(self) -> dict[str, dict[str, float]]:
        """Return signed deltas keyed by metric, then source corpus."""

        return {
            metric: {
                source: float(source_deltas[metric])
                for source, source_deltas in self.deltas.items()
            }
            for metric in GENERALIZATION_METRICS
        }

    @property
    def source_corpus_deltas(self) -> Mapping[str, Mapping[str, float]]:
        """Return the source-corpus-first signed delta mapping."""

        return self.deltas

    def metric_delta(self, metric: str, source_corpus: str) -> Mapping[str, float]:
        """Return the detailed comparison for one metric and corpus."""

        try:
            return self.metrics[metric][source_corpus]
        except KeyError as exc:
            raise KeyError(
                f"unknown metric/corpus pair: {metric}/{source_corpus}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, aggregate-only JSON-ready evidence."""

        return {
            "artifact_type": GENERALIZATION_ARTIFACT_TYPE,
            "schema_version": GENERALIZATION_SCHEMA_VERSION,
            "suite": self.suite,
            "model_name": self.model_name,
            "device": self.device,
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "in_domain_suite": self.in_domain_suite,
            "out_of_domain_suites": list(self.out_of_domain_suites),
            "headline_gap": self.headline_gap,
            "headline_generalization_gap": self.headline_generalization_gap,
            "generalization_gap": self.generalization_gap,
            "metrics": _plain(self.metrics),
            "deltas": _plain(self.deltas),
            "delta_by_metric": _plain(self.delta_by_metric),
            "in_domain_report": self.in_domain_report.to_dict(),
            "out_of_domain_reports": {
                source: report.to_dict()
                for source, report in self.out_of_domain_reports.items()
            },
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneralizationReport":
        """Reconstruct a report from :meth:`to_dict` output."""

        if payload.get("artifact_type") != GENERALIZATION_ARTIFACT_TYPE:
            raise ValueError("unsupported generalization report artifact type")
        if int(payload.get("schema_version", -1)) != GENERALIZATION_SCHEMA_VERSION:
            raise ValueError("unsupported generalization report schema version")

        raw_out_reports = payload.get("out_of_domain_reports") or {}
        if not isinstance(raw_out_reports, Mapping):
            raise ValueError("out_of_domain_reports must be a mapping")
        raw_metrics = payload.get("metrics") or {}
        raw_deltas = payload.get("deltas") or {}
        if not isinstance(raw_metrics, Mapping) or not isinstance(raw_deltas, Mapping):
            raise ValueError("generalization metrics and deltas must be mappings")
        return cls(
            model_name=str(payload["model_name"]),
            device=str(payload["device"]),
            in_domain_suite=str(payload["in_domain_suite"]),
            out_of_domain_suites=tuple(
                str(value) for value in payload.get("out_of_domain_suites", ())
            ),
            in_domain_report=BenchmarkReport.from_dict(
                _mapping(payload.get("in_domain_report"))
            ),
            out_of_domain_reports={
                str(source): BenchmarkReport.from_dict(_mapping(report))
                for source, report in raw_out_reports.items()
            },
            metrics=_coerce_nested_float_mapping(raw_metrics, depth=3),
            deltas=_coerce_nested_float_mapping(raw_deltas, depth=2),
            headline_gap=float(
                payload.get("headline_gap", payload.get("generalization_gap", 0.0))
            ),
            generated_at=payload.get("generated_at"),
            metadata=_mapping(payload.get("metadata")),
            suite=str(payload.get("suite") or GENERALIZATION_SUITE),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "GeneralizationReport":
        """Read a serialized generalization report."""

        report_path = Path(path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"generalization report must be a JSON object: {path}")
        return cls.from_dict(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize deterministic generalization evidence as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the JSON report to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render the aggregate report as stable Markdown."""

        lines = [
            "# Cross-Corpus Generalization Report",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| In-domain corpus | `{self.in_domain_suite}` |",
            f"| Out-of-domain corpora | {len(self.out_of_domain_suites)} |",
            f"| Headline generalization gap | {_format_metric(self.headline_gap)} |",
            "",
            "| Metric | Source corpus | In-domain | Out-of-domain | "
            "Delta (out - in) | Gap |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for metric in GENERALIZATION_METRICS:
            for source in self.out_of_domain_suites:
                detail = self.metrics[metric][source]
                lines.append(
                    f"| `{metric}` | `{source}` | "
                    f"{_format_metric(detail['in_domain'])} | "
                    f"{_format_metric(detail['out_of_domain'])} | "
                    f"{_format_metric(detail['delta'])} | "
                    f"{_format_metric(detail['gap'])} |"
                )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def model_card_evidence(self) -> dict[str, Any]:
        """Return the report under a model-card evidence key."""

        return {"cross_corpus_generalization": self.to_dict()}

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access to serialized report fields."""

        return self.to_dict()[key]


def cross_corpus_report(
    model: str | ModelRunner,
    in_domain_suite: Any,
    out_of_domain_suites: Any,
    *,
    runner: ModelRunner | None = None,
    device: str = "cpu",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GeneralizationReport:
    """Run the benchmark harness across in- and out-of-domain corpora.

    ``in_domain_suite`` and ``out_of_domain_suites`` accept a local fixture
    path, a registered suite name, a sequence of ``BenchmarkFixture`` objects,
    or (for out-of-domain inputs) a mapping from source-corpus name to one of
    those values.  A callable ``model`` is treated as the runner when
    ``runner`` is omitted, which keeps synthetic evaluations fully offline.

    Signed deltas are always ``out_of_domain - in_domain``.  The direction-
    aware gap used for the headline treats lower leakage as better and higher
    recall/F1 as better.
    """

    model_name, model_runner = _resolve_model_runner(model, runner)
    device_name = _require_text(device, "device")
    in_name, in_fixtures = _resolve_suite(in_domain_suite, "in-domain")
    out_specs = _resolve_out_of_domain_suites(out_of_domain_suites)
    if not out_specs:
        raise ValueError("at least one out-of-domain suite is required")

    out_names = [name for name, _ in out_specs]
    if len(out_names) != len(set(out_names)):
        raise ValueError("out-of-domain suite names must be unique")

    base_metadata = dict(metadata or {})
    in_report = _run_suite(
        in_fixtures,
        source_corpus=in_name,
        role="in_domain",
        model_name=model_name,
        model_runner=model_runner,
        device=device_name,
        generated_at=generated_at,
        metadata=base_metadata,
    )
    out_reports: dict[str, BenchmarkReport] = {}
    for source_corpus, fixtures in out_specs:
        out_reports[source_corpus] = _run_suite(
            fixtures,
            source_corpus=source_corpus,
            role="out_of_domain",
            model_name=model_name,
            model_runner=model_runner,
            device=device_name,
            generated_at=generated_at,
            metadata=base_metadata,
        )

    in_values = {
        metric: _metric_value(in_report, metric) for metric in GENERALIZATION_METRICS
    }
    details: dict[str, dict[str, dict[str, float]]] = {
        metric: {} for metric in GENERALIZATION_METRICS
    }
    signed_deltas: dict[str, dict[str, float]] = {}
    headline_values: list[float] = []
    for source_corpus, report in out_reports.items():
        source_deltas: dict[str, float] = {}
        signed_deltas[source_corpus] = source_deltas
        for metric in GENERALIZATION_METRICS:
            out_value = _metric_value(report, metric)
            signed_delta = out_value - in_values[metric]
            gap = signed_delta * _GAP_DIRECTION[metric]
            source_deltas[metric] = signed_delta
            details[metric][source_corpus] = {
                "in_domain": in_values[metric],
                "out_of_domain": out_value,
                "delta": signed_delta,
                "gap": gap,
            }
            headline_values.append(gap)

    headline_gap = sum(headline_values) / len(headline_values)
    report_metadata = dict(base_metadata)
    report_metadata.update(
        {
            "generalization_metrics": list(GENERALIZATION_METRICS),
            "in_domain_source_corpus": in_name,
            "out_of_domain_source_corpora": out_names,
        }
    )
    return GeneralizationReport(
        model_name=model_name,
        device=device_name,
        in_domain_suite=in_name,
        out_of_domain_suites=tuple(out_names),
        in_domain_report=in_report,
        out_of_domain_reports=out_reports,
        metrics=details,
        deltas=signed_deltas,
        headline_gap=headline_gap,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def _run_suite(
    fixtures: Sequence[BenchmarkFixture],
    *,
    source_corpus: str,
    role: str,
    model_name: str,
    model_runner: ModelRunner | None,
    device: str,
    generated_at: str | None,
    metadata: Mapping[str, Any],
) -> BenchmarkReport:
    report_metadata = dict(metadata)
    report_metadata.update(
        {
            "generalization_role": role,
            "source_corpus": source_corpus,
        }
    )
    kwargs: dict[str, Any] = {
        "suite": source_corpus,
        "model_name": model_name,
        "device": device,
        "generated_at": generated_at,
        "metadata": report_metadata,
    }
    if model_runner is not None:
        kwargs["runner"] = model_runner
    return run_benchmark(fixtures, **kwargs)


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner | None]:
    if runner is not None:
        return _model_name(model), runner
    if callable(model) and not isinstance(model, str):
        return _model_name(model), model
    model_name = _model_name(model)
    return model_name, None


def _model_name(model: object) -> str:
    if isinstance(model, str):
        return _require_text(model, "model")
    if callable(model):
        return _require_text(
            str(getattr(model, "__name__", model.__class__.__name__)), "model"
        )
    raise TypeError("model must be a string identifier or callable runner")


def _resolve_suite(
    spec: Any, default_name: str
) -> tuple[str, tuple[BenchmarkFixture, ...]]:
    name, raw_spec = _split_named_spec(spec, default_name)
    if isinstance(raw_spec, Mapping) and "fixtures" in raw_spec:
        name = _require_text(str(raw_spec.get("name") or name), "suite name")
        raw_spec = raw_spec["fixtures"]
    elif isinstance(raw_spec, Mapping) and _is_fixture_mapping(raw_spec):
        raw_spec = (raw_spec,)

    if isinstance(raw_spec, (str, Path)):
        path = Path(raw_spec)
        if path.is_file():
            fixtures = load_fixtures(path)
            name = _path_name(path, name)
        else:
            if isinstance(raw_spec, Path):
                raise FileNotFoundError(f"suite fixture file not found: {raw_spec}")
            from openmed.eval.suites import load_suite_fixtures

            fixtures = load_suite_fixtures(str(raw_spec))
            name = _require_text(str(raw_spec), "suite name")
    else:
        fixtures = raw_spec

    coerced = _coerce_fixtures(fixtures, name)
    return _require_text(name, "suite name"), coerced


def _resolve_out_of_domain_suites(
    spec: Any,
) -> list[tuple[str, tuple[BenchmarkFixture, ...]]]:
    if isinstance(spec, Mapping):
        return [
            _resolve_suite(value, _require_text(str(name), "suite name"))
            for name, value in spec.items()
        ]
    if isinstance(spec, (str, Path)):
        return [_resolve_suite(spec, "out-of-domain-1")]
    if _is_named_spec(spec):
        return [_resolve_suite(spec, "out-of-domain-1")]

    values = list(spec) if isinstance(spec, Iterable) else []
    if not values:
        return []
    if all(_is_fixture_value(value) for value in values):
        return [_resolve_suite(values, "out-of-domain-1")]
    return [
        _resolve_suite(value, f"out-of-domain-{index}")
        for index, value in enumerate(values, start=1)
    ]


def _split_named_spec(spec: Any, default_name: str) -> tuple[str, Any]:
    if _is_named_spec(spec):
        name, value = spec
        return _require_text(str(name), "suite name"), value
    return default_name, spec


def _is_named_spec(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


def _is_fixture_value(value: Any) -> bool:
    return isinstance(value, BenchmarkFixture) or (
        isinstance(value, Mapping) and _is_fixture_mapping(value)
    )


def _is_fixture_mapping(value: Mapping[str, Any]) -> bool:
    return "text" in value and (
        "gold_spans" in value or "entities" in value or "id" in value
    )


def _coerce_fixtures(value: Any, suite_name: str) -> tuple[BenchmarkFixture, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError(f"{suite_name!r} must resolve to benchmark fixtures")
    fixtures: list[BenchmarkFixture] = []
    for item in value:
        if isinstance(item, BenchmarkFixture):
            fixtures.append(item)
        elif isinstance(item, Mapping):
            fixtures.append(BenchmarkFixture.from_mapping(item))
        else:
            raise TypeError(
                f"{suite_name!r} contains an unsupported fixture type "
                f"{type(item).__name__}"
            )
    if not fixtures:
        raise ValueError(f"{suite_name!r} must contain at least one fixture")
    return tuple(fixtures)


def _path_name(path: Path, fallback: str) -> str:
    return _require_text(path.stem or fallback, "suite name")


def _metric_value(report: BenchmarkReport, metric: str) -> float:
    for path in _METRIC_PATHS[metric]:
        value: Any = report.metrics
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if isinstance(value, bool) or not isinstance(value, Real):
                break
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
            break
    raise ValueError(
        f"benchmark report {report.suite!r} is missing finite {metric} metric"
    )


def _require_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_nested_float_mapping(
    value: Mapping[str, Any],
    *,
    depth: int,
) -> dict[str, Any]:
    if depth == 1:
        result: dict[str, float] = {}
        for key, item in value.items():
            result[str(key)] = float(item)
        return result
    return {
        str(key): _coerce_nested_float_mapping(_mapping(item), depth=depth - 1)
        for key, item in value.items()
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


__all__ = [
    "GENERALIZATION_ARTIFACT_TYPE",
    "GENERALIZATION_METRICS",
    "GENERALIZATION_SCHEMA_VERSION",
    "GENERALIZATION_SUITE",
    "GeneralizationReport",
    "cross_corpus_report",
    "default_model_runner",
]
