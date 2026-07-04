"""Per-model benchmark scorecard rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.eval.metrics import DEVICE_TIERS
from openmed.eval.report import (
    BenchmarkReport,
    _display_value,
    _find_manifest_row,
    _format_value,
    _nested_get,
    _plain,
    _slug,
)
from openmed.eval.tiers import TIERS

PLACEHOLDER = "n/a"
BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class ModelScorecard:
    """Aggregate BenchmarkReport objects for one model into a scorecard."""

    model_name: str
    reports: tuple[BenchmarkReport, ...]
    manifest_row: Mapping[str, Any] | None = None
    device_tiers: tuple[str, ...] = DEVICE_TIERS
    tier_budgets: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: TIERS)
    placeholder: str = PLACEHOLDER

    def __post_init__(self) -> None:
        reports = tuple(self.reports)
        object.__setattr__(self, "reports", reports)
        for report in reports:
            if report.model_name != self.model_name:
                raise ValueError("ModelScorecard reports must all share one model_name")

    @classmethod
    def from_reports(
        cls,
        reports: Iterable[BenchmarkReport],
        *,
        manifest_rows: Iterable[Mapping[str, Any]] = (),
        model_name: str | None = None,
        device_tiers: Sequence[str] = DEVICE_TIERS,
    ) -> "ModelScorecard":
        """Build a scorecard from reports and optional manifest rows."""

        report_rows = tuple(reports)
        if model_name is None:
            if not report_rows:
                raise ValueError("model_name is required when reports are empty")
            model_name = report_rows[0].model_name
        for report in report_rows:
            if report.model_name != model_name:
                raise ValueError("ModelScorecard reports must all share one model_name")
        manifest_row = _find_manifest_row(model_name, manifest_rows)
        return cls(
            model_name=model_name,
            reports=report_rows,
            manifest_row=manifest_row,
            device_tiers=tuple(device_tiers),
        )

    @property
    def family(self) -> str | None:
        """Return the manifest family, falling back to report metadata."""

        if self.manifest_row is not None and self.manifest_row.get("family"):
            return str(self.manifest_row["family"])
        for report in self.reports:
            family = report.metadata.get("family")
            if family:
                return str(family)
        return None

    @property
    def model_tier(self) -> str | None:
        """Return the manifest model tier, falling back to report metadata."""

        if self.manifest_row is not None and self.manifest_row.get("tier"):
            return str(self.manifest_row["tier"])
        for report in self.reports:
            tier = report.metadata.get("tier")
            if tier:
                return str(tier)
        return None

    @property
    def target_formats(self) -> tuple[str, ...]:
        """Return manifest formats in deterministic order."""

        formats: list[str] = []
        if self.manifest_row is not None:
            formats.extend(str(item) for item in self.manifest_row.get("formats") or [])
        for report in self.reports:
            if report.device not in formats:
                formats.append(report.device)
        return tuple(formats)

    @property
    def suites(self) -> tuple[str, ...]:
        """Return suite names represented in this scorecard."""

        return tuple(sorted({report.suite for report in self.reports}))

    @property
    def latest_generated_at(self) -> str | None:
        """Return the latest report timestamp when present."""

        timestamps = sorted(
            str(report.generated_at)
            for report in self.reports
            if report.generated_at is not None
        )
        return timestamps[-1] if timestamps else None

    @property
    def fixture_count(self) -> int:
        """Return total fixtures across reports."""

        return sum(report.fixture_count for report in self.reports)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-safe scorecard dictionary."""

        model_tier = self.model_tier
        tier_budget = (
            _plain(self.tier_budgets.get(model_tier))
            if model_tier is not None
            else None
        )
        return {
            "device_tiers": [self._device_row(device) for device in self._devices()],
            "family": self.family,
            "fixture_count": self.fixture_count,
            "latest_generated_at": self.latest_generated_at,
            "model_name": self.model_name,
            "model_tier": model_tier,
            "report_count": len(self.reports),
            "schema_version": 1,
            "suites": list(self.suites),
            "target_formats": list(self.target_formats),
            "tier_budget": tier_budget,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the scorecard to byte-stable JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic scorecard JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic, single-page Markdown scorecard."""

        payload = self.to_dict()
        target_formats = payload["target_formats"]
        suites = payload["suites"]
        tier_budget = payload["tier_budget"] or {}
        latency_budget = self.placeholder
        if tier_budget:
            latency_budget = (
                f"{_display_value(tier_budget.get('p50_ms_max'))} / "
                f"{_display_value(tier_budget.get('p95_ms_max'))}"
            )

        lines = [
            f"# Model Scorecard: {self.model_name}",
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model | `{self.model_name}` |",
            f"| Family | `{_display_value(payload['family'])}` |",
            f"| Model Tier | `{_display_value(payload['model_tier'])}` |",
            (
                "| Target Formats | "
                f"`{', '.join(target_formats) if target_formats else self.placeholder}` |"
            ),
            f"| Suites | `{', '.join(suites) if suites else self.placeholder}` |",
            f"| Reports | {_format_value(payload['report_count'])} |",
            f"| Fixtures | {_format_value(payload['fixture_count'])} |",
            (
                "| Latest Report Timestamp | "
                f"`{_display_value(payload['latest_generated_at'])}` |"
            ),
            f"| Tier RAM Limit MB | {_display_value(tier_budget.get('ram_mb_max'))} |",
            f"| Tier Latency Budget p50/p95 ms | `{latency_budget}` |",
            "",
            "## Device Tiers",
            "",
            (
                "| Device Tier | Targeted | Reports | Fixtures | Recall | "
                "Critical Finding Recall | Leakage Rate | Latency p50/p95 ms | "
                "Peak RSS MB | Model Size MB |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in payload["device_tiers"]:
            lines.append(
                "| "
                f"`{row['device_tier']}` | "
                f"{'yes' if row['targeted'] else 'no'} | "
                f"{row['report_count']} | "
                f"{row['fixture_count']} | "
                f"{_format_percent_or_placeholder(row['recall'], self.placeholder)} | "
                f"{_format_percent_or_placeholder(row['critical_finding_recall'], self.placeholder)} | "
                f"{_format_percent_or_placeholder(row['leakage_rate'], self.placeholder)} | "
                f"{_format_latency(row, self.placeholder)} | "
                f"{_format_number_or_placeholder(row['peak_rss_mb'], self.placeholder)} | "
                f"{_format_number_or_placeholder(row['model_size_mb'], self.placeholder)} |"
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write Markdown scorecard content to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def _devices(self) -> tuple[str, ...]:
        devices = list(self.device_tiers)
        for value in (
            *self.target_formats,
            *(report.device for report in self.reports),
        ):
            if value not in devices:
                devices.append(value)
        return tuple(devices)

    def _device_row(self, device: str) -> dict[str, Any]:
        reports = tuple(report for report in self.reports if report.device == device)
        return {
            "device_tier": device,
            "fixture_count": sum(report.fixture_count for report in reports),
            "critical_finding_recall": _aggregate_critical_finding_recall(reports),
            "leakage_rate": _aggregate_leakage(reports),
            "latency_p50_ms": _aggregate_latency(reports, "p50_ms"),
            "latency_p95_ms": _aggregate_latency(reports, "p95_ms"),
            "model_size_mb": _aggregate_model_size_mb(reports, self.manifest_row),
            "peak_rss_mb": _aggregate_peak_rss_mb(reports, self.manifest_row, device),
            "recall": _aggregate_recall(reports),
            "report_count": len(reports),
            "targeted": device in self.target_formats,
        }


def render_model_scorecard(
    reports: Iterable[BenchmarkReport],
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
    model_name: str | None = None,
) -> str:
    """Render a Markdown scorecard for one model."""

    return ModelScorecard.from_reports(
        reports,
        manifest_rows=manifest_rows,
        model_name=model_name,
    ).to_markdown()


def write_model_scorecard(
    reports: Iterable[BenchmarkReport],
    output_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
    model_name: str | None = None,
) -> Path:
    """Write ``<model>.md`` scorecard content under *output_dir*."""

    scorecard = ModelScorecard.from_reports(
        reports,
        manifest_rows=manifest_rows,
        model_name=model_name,
    )
    return scorecard.write_markdown(
        Path(output_dir) / f"{_slug(scorecard.model_name)}.md"
    )


def write_model_scorecard_json(
    reports: Iterable[BenchmarkReport],
    output_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
    model_name: str | None = None,
    indent: int = 2,
) -> Path:
    """Write ``<model>.json`` scorecard content under *output_dir*."""

    scorecard = ModelScorecard.from_reports(
        reports,
        manifest_rows=manifest_rows,
        model_name=model_name,
    )
    return scorecard.write_json(
        Path(output_dir) / f"{_slug(scorecard.model_name)}.json",
        indent=indent,
    )


def _aggregate_recall(reports: Sequence[BenchmarkReport]) -> float | None:
    return _aggregate_rate(
        reports,
        numerator_denominator_keys=(
            ("character_recall.numerator", "character_recall.denominator"),
            ("recall_slices.covered_chars", "recall_slices.total_chars"),
        ),
        value_keys=(
            "character_recall.rate",
            "character_recall.overall",
            "recall_slices.overall",
            "exact_span_f1.recall",
        ),
    )


def _aggregate_critical_finding_recall(
    reports: Sequence[BenchmarkReport],
) -> float | None:
    covered = 0.0
    total = 0.0
    saw_denominator = False
    for report in reports:
        metrics = _plain(report.metrics)
        raw_covered = _number_at(metrics, "critical_finding_recall.covered")
        raw_total = _number_at(metrics, "critical_finding_recall.total")
        if raw_covered is None or raw_total is None:
            continue
        saw_denominator = True
        covered += raw_covered
        total += raw_total
    if saw_denominator:
        return covered / total if total > 0 else None
    return _aggregate_rate(
        reports,
        numerator_denominator_keys=(),
        value_keys=("critical_finding_recall.overall",),
    )


def _aggregate_leakage(reports: Sequence[BenchmarkReport]) -> float | None:
    return _aggregate_rate(
        reports,
        numerator_denominator_keys=(
            ("leakage.leaked_chars", "leakage.total_chars"),
            ("leakage_rate.leaked_chars", "leakage_rate.total_chars"),
        ),
        value_keys=("leakage.overall", "leakage_rate.overall"),
    )


def _aggregate_rate(
    reports: Sequence[BenchmarkReport],
    *,
    numerator_denominator_keys: Sequence[tuple[str, str]],
    value_keys: Sequence[str],
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for report in reports:
        metrics = _plain(report.metrics)
        for numerator_key, denominator_key in numerator_denominator_keys:
            raw_numerator = _number_at(metrics, numerator_key)
            raw_denominator = _number_at(metrics, denominator_key)
            if raw_numerator is not None and raw_denominator is not None:
                numerator += raw_numerator
                denominator += raw_denominator
                break
    if denominator > 0:
        return numerator / denominator

    values: list[float] = []
    for report in reports:
        metrics = _plain(report.metrics)
        for key in value_keys:
            value = _number_at(metrics, key)
            if value is not None:
                values.append(value)
                break
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_latency(
    reports: Sequence[BenchmarkReport],
    field: str,
) -> float | None:
    weighted_total = 0.0
    sample_count = 0.0
    unweighted: list[float] = []
    for report in reports:
        metrics = _plain(report.metrics)
        value = _number_at(metrics, f"latency.{field}")
        if value is None:
            continue
        count = _number_at(metrics, "latency.count")
        if count is not None and count > 0:
            weighted_total += value * count
            sample_count += count
        else:
            unweighted.append(value)
    if sample_count > 0:
        return weighted_total / sample_count
    if unweighted:
        return sum(unweighted) / len(unweighted)
    return None


def _aggregate_peak_rss_mb(
    reports: Sequence[BenchmarkReport],
    manifest_row: Mapping[str, Any] | None,
    device: str,
) -> float | None:
    values: list[float] = []
    for report in reports:
        metrics = _plain(report.metrics)
        values.extend(
            value
            for value in (
                _number_at(metrics, "resources.peak_rss_mib"),
                _number_at(metrics, "resources.peak_rss_mb"),
                _bytes_to_mib(_number_at(metrics, "resources.peak_rss_bytes")),
            )
            if value is not None
        )
    if manifest_row is not None:
        peak_ram = manifest_row.get("peak_ram_mb")
        if isinstance(peak_ram, Mapping):
            manifest_value = _number_or_none(peak_ram.get(device))
            if manifest_value is not None:
                values.append(manifest_value)
    if not values:
        return None
    return max(values)


def _aggregate_model_size_mb(
    reports: Sequence[BenchmarkReport],
    manifest_row: Mapping[str, Any] | None,
) -> float | None:
    values: list[float] = []
    for report in reports:
        metrics = _plain(report.metrics)
        values.extend(
            value
            for value in (
                _number_at(metrics, "resources.model_size_mib"),
                _number_at(metrics, "resources.model_size_mb"),
                _bytes_to_mib(_number_at(metrics, "resources.model_size_bytes")),
            )
            if value is not None
        )
    if manifest_row is not None:
        values.extend(
            value
            for value in (
                _number_at(manifest_row, "model_size_mib"),
                _number_at(manifest_row, "model_size_mb"),
                _bytes_to_mib(_number_at(manifest_row, "model_size_bytes")),
                _number_at(manifest_row, "resources.model_size_mib"),
                _number_at(manifest_row, "resources.model_size_mb"),
                _bytes_to_mib(_number_at(manifest_row, "resources.model_size_bytes")),
            )
            if value is not None
        )
    if not values:
        return None
    return max(values)


def _number_at(mapping: Mapping[str, Any], dotted_key: str) -> float | None:
    return _number_or_none(_nested_get(mapping, dotted_key))


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bytes_to_mib(value: float | None) -> float | None:
    if value is None:
        return None
    return value / BYTES_PER_MIB


def _format_percent_or_placeholder(value: Any, placeholder: str) -> str:
    if value is None:
        return placeholder
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return str(value)


def _format_number_or_placeholder(value: Any, placeholder: str) -> str:
    if value is None:
        return placeholder
    return _format_value(value)


def _format_latency(row: Mapping[str, Any], placeholder: str) -> str:
    p50 = _format_number_or_placeholder(row["latency_p50_ms"], placeholder)
    p95 = _format_number_or_placeholder(row["latency_p95_ms"], placeholder)
    if p50 == placeholder and p95 == placeholder:
        return placeholder
    return f"{p50} / {p95}"


__all__ = [
    "ModelScorecard",
    "render_model_scorecard",
    "write_model_scorecard",
    "write_model_scorecard_json",
]
