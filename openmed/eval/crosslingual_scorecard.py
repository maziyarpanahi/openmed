"""Aggregate-only cross-lingual extraction scorecards.

The scorecard adapts the metric evidence already emitted by OpenMed evaluation
reports.  Reports in older suites do not share one metric layout, so the
adapter accepts :class:`~openmed.eval.report.BenchmarkReport` instances and
their serialized mappings, including nested ``character_recall``, ``leakage``,
``abstention``, and ``latency`` sections.

Only allow-listed language/family labels, counts, rates, and latency summaries
are retained.  Source text, fixture identifiers, model metadata, and arbitrary
report fields are deliberately discarded before JSON or Markdown rendering.
The implementation is local and deterministic: it does not load models, read
network resources, or use wall-clock time.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

CROSS_LINGUAL_SCORECARD_SCHEMA_VERSION = 1
CROSS_LINGUAL_SCORECARD_ARTIFACT_TYPE = "openmed.eval.crosslingual_scorecard"
CROSS_LINGUAL_SCORECARD_SCHEMA = CROSS_LINGUAL_SCORECARD_SCHEMA_VERSION
CROSS_LINGUAL_SCORECARD_ARTIFACT = CROSS_LINGUAL_SCORECARD_ARTIFACT_TYPE

# Backwards-friendly aliases use the spelling from the module name as well as
# the spaced spelling used in prose and documentation.
CROSSLINGUAL_SCORECARD_SCHEMA_VERSION = CROSS_LINGUAL_SCORECARD_SCHEMA_VERSION
CROSSLINGUAL_SCORECARD_ARTIFACT_TYPE = CROSS_LINGUAL_SCORECARD_ARTIFACT_TYPE

SCORECARD_METRICS: tuple[str, ...] = (
    "recall",
    "critical_leakage",
    "abstention",
    "latency_ms",
)
METRIC_NAMES = SCORECARD_METRICS
UNSPECIFIED_FAMILY = "unspecified"

_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+\-]{0,63}$")
_MISSING = object()

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "recall": (
        "recall",
        "character_recall",
        "span_recall",
        "extraction_recall",
        "recall_rate",
        "micro_recall",
        "macro_recall",
    ),
    "critical_leakage": (
        "critical_leakage_count",
        "critical_leakage",
        "critical_leakage_events",
        "critical_leakage_rate",
        "critical_leakage_total",
    ),
    "abstention": (
        "abstention_rate",
        "abstention",
        "abstentions",
        "abstained_rate",
    ),
    "latency_ms": (
        "latency_ms",
        "latency",
        "latency_summary",
        "inference_latency_ms",
    ),
}

_LANGUAGE_CONTAINER_KEYS = frozenset(
    {
        "by_language",
        "language_metrics",
        "languages",
        "metrics_by_language",
        "per_language",
    }
)
_NESTED_METRIC_CONTAINER_KEYS = frozenset(
    {
        "abstention",
        "calibration",
        "extraction",
        "gate",
        "latency",
        "leakage",
        "metrics",
        "performance",
        "quality",
        "release_gate",
        "resources",
        "summary",
    }
)
_NON_LANGUAGE_KEYS = frozenset(
    {
        "accepted",
        "abstained",
        "abstention",
        "abstention_rate",
        "average",
        "character_recall",
        "count",
        "covered",
        "critical_leakage",
        "critical_leakage_count",
        "denominator",
        "fixture_count",
        "hits",
        "latency",
        "latency_ms",
        "mean",
        "mean_ms",
        "numerator",
        "p50",
        "p50_ms",
        "p95",
        "p95_ms",
        "rate",
        "recall",
        "score",
        "support",
        "total",
        "value",
    }
)

__all__ = [
    "CROSS_LINGUAL_SCORECARD_ARTIFACT_TYPE",
    "CROSS_LINGUAL_SCORECARD_ARTIFACT",
    "CROSS_LINGUAL_SCORECARD_SCHEMA",
    "CROSS_LINGUAL_SCORECARD_SCHEMA_VERSION",
    "CROSSLINGUAL_SCORECARD_ARTIFACT_TYPE",
    "CROSSLINGUAL_SCORECARD_SCHEMA_VERSION",
    "CrossLingualScorecard",
    "CrossLingualScorecardRenderer",
    "CrossLingualScorecardRow",
    "METRIC_NAMES",
    "SCORECARD_METRICS",
    "UNSPECIFIED_FAMILY",
    "aggregate_crosslingual_reports",
    "build_cross_lingual_scorecard",
    "build_crosslingual_scorecard",
    "render_crosslingual_scorecard_json",
    "render_crosslingual_scorecard_markdown",
    "render_crosslingual_scorecard",
    "write_crosslingual_scorecard",
]


@dataclass(frozen=True)
class CrossLingualScorecardRow:
    """One aggregate language or family row.

    ``metrics`` contains only the four scorecard metrics and their aggregate
    count evidence.  The grouping label is held by the surrounding mapping and
    is therefore not repeated in the serialized row.
    """

    report_count: int
    fixture_count: int
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready aggregate-only row."""

        payload: dict[str, Any] = {
            "fixture_count": int(self.fixture_count),
            "report_count": int(self.report_count),
        }
        payload.update(_plain(self.metrics))
        return payload

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access used by other eval reports."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class CrossLingualScorecard:
    """Deterministic per-language and per-family extraction scorecard."""

    report_count: int
    fixture_count: int
    languages: tuple[str, ...]
    families: tuple[str, ...]
    per_language: Mapping[str, Mapping[str, Any]]
    per_family: Mapping[str, Mapping[str, Any]]
    expected_languages: tuple[str, ...] = ()
    missing_languages: tuple[str, ...] = ()
    unlabeled_report_count: int = 0
    unlabeled_fixture_count: int = 0
    missing_family_report_count: int = 0

    def __post_init__(self) -> None:
        """Snapshot mappings and ordering so later input mutation cannot drift."""

        object.__setattr__(self, "languages", tuple(sorted(self.languages)))
        object.__setattr__(self, "families", tuple(sorted(self.families)))
        object.__setattr__(
            self,
            "expected_languages",
            tuple(sorted(self.expected_languages)),
        )
        object.__setattr__(
            self,
            "missing_languages",
            tuple(sorted(self.missing_languages)),
        )
        object.__setattr__(
            self,
            "per_language",
            _snapshot_rows(self.per_language),
        )
        object.__setattr__(self, "per_family", _snapshot_rows(self.per_family))

    @classmethod
    def from_reports(
        cls,
        reports: Iterable[Any],
        *,
        expected_languages: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        required_languages: Sequence[str] | None = None,
        family_by_model: Mapping[str, Any] | None = None,
        manifest_rows: Iterable[Mapping[str, Any]] | None = None,
    ) -> "CrossLingualScorecard":
        """Build a scorecard from existing evaluation reports.

        ``expected_languages``, ``languages``, and ``required_languages`` are
        equivalent ways to declare the language evidence a caller expects.  If
        none is supplied, the observed language set becomes the expectation;
        unlabeled reports are still reported as missing evidence.
        """

        return build_crosslingual_scorecard(
            reports,
            expected_languages=expected_languages,
            languages=languages,
            required_languages=required_languages,
            family_by_model=family_by_model,
            manifest_rows=manifest_rows,
        )

    @property
    def by_language(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the per-language rows under a familiar dashboard alias."""

        return self.per_language

    @property
    def by_family(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the per-family rows under a familiar dashboard alias."""

        return self.per_family

    @property
    def language_coverage_complete(self) -> bool:
        """Whether expected languages are present and every report is labeled."""

        return not self.missing_languages and self.unlabeled_report_count == 0

    @property
    def missing_language_evidence(self) -> tuple[str, ...]:
        """Return expected languages for which no metric evidence was found."""

        return self.missing_languages

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe, deterministic JSON-compatible payload."""

        return {
            "aggregate_only": True,
            "artifact_type": CROSS_LINGUAL_SCORECARD_ARTIFACT_TYPE,
            "families": list(self.families),
            "family_coverage": {
                "missing_family_report_count": self.missing_family_report_count,
                "observed_family_count": len(self.families),
            },
            "fixture_count": int(self.fixture_count),
            "language_coverage_complete": self.language_coverage_complete,
            "languages": list(self.languages),
            "missing_language_evidence": {
                "complete": self.language_coverage_complete,
                "expected": list(self.expected_languages),
                "missing": list(self.missing_languages),
                "observed": list(self.languages),
                "unlabeled_fixture_count": int(self.unlabeled_fixture_count),
                "unlabeled_report_count": int(self.unlabeled_report_count),
            },
            "missing_languages": list(self.missing_languages),
            "by_family": {
                family: dict(self.per_family[family]) for family in self.families
            },
            "by_language": {
                language: dict(self.per_language[language])
                for language in self.languages
            },
            "per_family": {
                family: dict(self.per_family[family]) for family in self.families
            },
            "per_language": {
                language: dict(self.per_language[language])
                for language in self.languages
            },
            "report_count": int(self.report_count),
            "schema_version": CROSS_LINGUAL_SCORECARD_SCHEMA_VERSION,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the scorecard with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON scorecard evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render aggregate-only Markdown for human review."""

        lines = [
            "# Cross-Lingual Extraction Scorecard",
            "",
            (
                "This report contains aggregate counts, rates, and latency only; "
                "source text, spans, fixture identifiers, and arbitrary metadata "
                "are excluded."
            ),
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Reports | {self.report_count} |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Languages with evidence | {len(self.languages)} |",
            f"| Expected languages | {len(self.expected_languages)} |",
            f"| Missing languages | {len(self.missing_languages)} |",
            f"| Unlabeled reports | {self.unlabeled_report_count} |",
            f"| Families with evidence | {len(self.families)} |",
            f"| Reports without family evidence | {self.missing_family_report_count} |",
            "",
            "## Language evidence",
            "",
            (
                "| Language | Reports | Fixtures | Recall | Critical leakage | "
                "Abstention | Latency ms | Missing metrics |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        if self.languages:
            for language in self.languages:
                lines.append(_markdown_group_row(language, self.per_language[language]))
        else:
            lines.append("| `none` | 0 | 0 | n/a | n/a | n/a | n/a | all |")

        lines.extend(
            [
                "",
                "## Family evidence",
                "",
                (
                    "| Family | Reports | Fixtures | Recall | Critical leakage | "
                    "Abstention | Latency ms | Missing metrics |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        if self.families:
            for family in self.families:
                lines.append(_markdown_group_row(family, self.per_family[family]))
        else:
            lines.append("| `none` | 0 | 0 | n/a | n/a | n/a | n/a | all |")

        lines.extend(["", "## Missing language evidence", ""])
        if self.missing_languages:
            lines.append(
                "Expected languages without evidence: "
                + ", ".join(
                    f"`{_markdown_cell(language)}`"
                    for language in self.missing_languages
                )
                + "."
            )
        else:
            lines.append("No expected language is missing from the aggregate evidence.")
        if self.unlabeled_report_count:
            lines.append(
                f"{self.unlabeled_report_count} report(s) did not provide a safe "
                "language label; their language metrics are not attributed."
            )
        else:
            lines.append("Every report had a safe language label or language slice.")

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown scorecard evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def model_card_evidence(self) -> dict[str, Any]:
        """Return the compact scorecard payload for a model-card consumer."""

        payload = self.to_dict()
        return {
            "artifact_type": payload["artifact_type"],
            "schema_version": payload["schema_version"],
            "fixture_count": payload["fixture_count"],
            "language_count": len(self.languages),
            "family_count": len(self.families),
            "missing_language_count": len(self.missing_languages),
            "unlabeled_report_count": self.unlabeled_report_count,
        }

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access to serialized fields."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class CrossLingualScorecardRenderer:
    """Reusable renderer configured with expected local evidence labels."""

    expected_languages: Sequence[str] | None = None
    family_by_model: Mapping[str, Any] | None = None
    manifest_rows: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        """Snapshot renderer inputs that could otherwise be mutated."""

        if self.expected_languages is not None:
            object.__setattr__(
                self,
                "expected_languages",
                tuple(_normalise_labels(self.expected_languages)),
            )
        object.__setattr__(self, "manifest_rows", tuple(self.manifest_rows))

    def build(self, reports: Any) -> CrossLingualScorecard:
        """Build a scorecard from report objects or report mappings."""

        return build_crosslingual_scorecard(
            reports,
            expected_languages=self.expected_languages,
            family_by_model=self.family_by_model,
            manifest_rows=self.manifest_rows,
        )

    def render(
        self,
        reports: Any,
        *,
        format: str = "markdown",
        output_format: str | None = None,
        indent: int = 2,
    ) -> str:
        """Render a scorecard as Markdown or JSON."""

        return render_crosslingual_scorecard(
            self.build(reports),
            format=format,
            output_format=output_format,
            indent=indent,
        )

    def render_json(self, reports: Any, *, indent: int = 2) -> str:
        """Render deterministic JSON from report objects or mappings."""

        return self.build(reports).to_json(indent=indent)

    def render_markdown(self, reports: Any) -> str:
        """Render deterministic Markdown from report objects or mappings."""

        return self.build(reports).to_markdown()


def build_crosslingual_scorecard(
    reports: Iterable[Any],
    *,
    expected_languages: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    required_languages: Sequence[str] | None = None,
    family_by_model: Mapping[str, Any] | None = None,
    manifest_rows: Iterable[Mapping[str, Any]] | None = None,
) -> CrossLingualScorecard:
    """Aggregate existing evaluation reports by language and model family.

    A report may provide one language in ``language`` or
    ``metadata.language``, or several language slices in ``per_language`` /
    ``by_language``.  Metric rates are weighted by explicit numerator and
    denominator evidence when available, otherwise by fixture count.  Family
    aggregation counts each source report once even when it contains multiple
    language slices.

    Args:
        reports: ``BenchmarkReport`` objects, serialized reports, or mappings
            containing the same ``metrics``/metadata fields.
        expected_languages: Optional required language codes.
        languages: Alias for ``expected_languages``.
        required_languages: Alias for ``expected_languages``.
        family_by_model: Optional model-name-to-family fallback mapping.
        manifest_rows: Optional manifest mappings used as a family fallback.

    Returns:
        A deterministic scorecard whose artifacts contain aggregate evidence
        only.

    Raises:
        ValueError: If no reports are supplied or conflicting language aliases
            are provided.
        TypeError: If an input item is neither a report-like object nor a
            mapping.
    """

    expected = _resolve_expected_languages(
        expected_languages,
        languages=languages,
        required_languages=required_languages,
    )
    family_lookup = _resolve_family_lookup(
        family_by_model=family_by_model,
        manifest_rows=manifest_rows,
    )
    report_items = _coerce_report_items(reports)
    normalized_reports = tuple(_normalize_report(item) for item in report_items)
    if not normalized_reports:
        raise ValueError("at least one evaluation report is required")

    language_groups: dict[str, _GroupAccumulator] = {}
    family_groups: dict[str, _GroupAccumulator] = {}
    observed_languages: set[str] = set()
    observed_families: set[str] = set()
    unlabeled_report_count = 0
    unlabeled_fixture_count = 0
    missing_family_report_count = 0
    total_fixture_count = 0

    for report in normalized_reports:
        total_fixture_count += report.fixture_count
        family = report.family or family_lookup.get(report.model_name)
        if family is None:
            family = UNSPECIFIED_FAMILY
            missing_family_report_count += 1
        observed_families.add(family)
        family_group = family_groups.setdefault(family, _GroupAccumulator())
        family_group.add_report(report.fixture_count)

        language_payloads = _collect_language_payloads(report.root, report.metrics)
        if not language_payloads and report.language is not None:
            language_payloads = {report.language: {}}

        if not language_payloads:
            unlabeled_report_count += 1
            unlabeled_fixture_count += report.fixture_count

        observations: list[tuple[str, Mapping[str, Any], int]] = []
        for language, language_payload in sorted(language_payloads.items()):
            fixture_count = _language_fixture_count(
                language_payload,
                report,
                language_count=len(language_payloads),
            )
            observations.append((language, language_payload, fixture_count))
            observed_languages.add(language)
            group = language_groups.setdefault(language, _GroupAccumulator())
            group.add_report(fixture_count)
            source = language_payload or report.metrics
            samples = _samples_for_source(source, fixture_count)
            if len(observations) == 1 and len(language_payloads) == 1:
                samples = _fill_missing_samples(
                    samples, report.metrics, report.fixture_count
                )
            group.add_samples(samples)

        family_samples: dict[str, list[_MetricSample]] = {
            metric: [] for metric in SCORECARD_METRICS
        }
        for _, language_payload, language_fixture_count in observations:
            samples = _samples_for_source(language_payload, language_fixture_count)
            for metric in SCORECARD_METRICS:
                if samples[metric] is not None:
                    family_samples[metric].append(samples[metric])
        root_samples = _samples_for_source(report.metrics, report.fixture_count)
        for metric in SCORECARD_METRICS:
            if not family_samples[metric] and root_samples[metric] is not None:
                family_samples[metric].append(root_samples[metric])
        family_group.add_samples(
            {
                metric: _merge_samples(metric, samples)
                for metric, samples in family_samples.items()
            }
        )

    if expected is None:
        resolved_expected = tuple(sorted(observed_languages))
    else:
        resolved_expected = expected
    missing_languages = tuple(
        language for language in resolved_expected if language not in observed_languages
    )

    return CrossLingualScorecard(
        report_count=len(normalized_reports),
        fixture_count=total_fixture_count,
        languages=tuple(sorted(observed_languages)),
        families=tuple(sorted(observed_families)),
        per_language={
            language: language_groups[language].to_dict()
            for language in sorted(language_groups)
        },
        per_family={
            family: family_groups[family].to_dict() for family in sorted(family_groups)
        },
        expected_languages=resolved_expected,
        missing_languages=missing_languages,
        unlabeled_report_count=unlabeled_report_count,
        unlabeled_fixture_count=unlabeled_fixture_count,
        missing_family_report_count=missing_family_report_count,
    )


def aggregate_crosslingual_reports(
    reports: Iterable[Any],
    **kwargs: Any,
) -> CrossLingualScorecard:
    """Alias for :func:`build_crosslingual_scorecard`."""

    return build_crosslingual_scorecard(reports, **kwargs)


def build_cross_lingual_scorecard(
    reports: Iterable[Any],
    **kwargs: Any,
) -> CrossLingualScorecard:
    """Spelled-out alias for :func:`build_crosslingual_scorecard`."""

    return build_crosslingual_scorecard(reports, **kwargs)


def render_crosslingual_scorecard(
    reports: Any,
    *,
    format: str = "markdown",
    output_format: str | None = None,
    expected_languages: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    required_languages: Sequence[str] | None = None,
    family_by_model: Mapping[str, Any] | None = None,
    manifest_rows: Iterable[Mapping[str, Any]] | None = None,
    indent: int = 2,
) -> str:
    """Build and render report evidence in the requested deterministic format."""

    if isinstance(reports, CrossLingualScorecard) and all(
        value is None
        for value in (
            expected_languages,
            languages,
            required_languages,
            family_by_model,
            manifest_rows,
        )
    ):
        scorecard = reports
    else:
        scorecard = build_crosslingual_scorecard(
            reports,
            expected_languages=expected_languages,
            languages=languages,
            required_languages=required_languages,
            family_by_model=family_by_model,
            manifest_rows=manifest_rows,
        )
    selected_format = (output_format or format).casefold()
    if selected_format == "json":
        return scorecard.to_json(indent=indent)
    if selected_format in {"markdown", "md"}:
        return scorecard.to_markdown()
    raise ValueError("scorecard format must be markdown or json")


def render_crosslingual_scorecard_json(
    reports: Any,
    *,
    expected_languages: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    required_languages: Sequence[str] | None = None,
    family_by_model: Mapping[str, Any] | None = None,
    manifest_rows: Iterable[Mapping[str, Any]] | None = None,
    indent: int = 2,
) -> str:
    """Render deterministic JSON for reports or a built scorecard."""

    return render_crosslingual_scorecard(
        reports,
        format="json",
        expected_languages=expected_languages,
        languages=languages,
        required_languages=required_languages,
        family_by_model=family_by_model,
        manifest_rows=manifest_rows,
        indent=indent,
    )


def render_crosslingual_scorecard_markdown(
    reports: Any,
    *,
    expected_languages: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    required_languages: Sequence[str] | None = None,
    family_by_model: Mapping[str, Any] | None = None,
    manifest_rows: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    """Render deterministic Markdown for reports or a built scorecard."""

    return render_crosslingual_scorecard(
        reports,
        format="markdown",
        expected_languages=expected_languages,
        languages=languages,
        required_languages=required_languages,
        family_by_model=family_by_model,
        manifest_rows=manifest_rows,
    )


def write_crosslingual_scorecard(
    reports: Any,
    path: str | Path,
    *,
    format: str | None = None,
    expected_languages: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    required_languages: Sequence[str] | None = None,
    family_by_model: Mapping[str, Any] | None = None,
    manifest_rows: Iterable[Mapping[str, Any]] | None = None,
    indent: int = 2,
) -> Path:
    """Write deterministic JSON or Markdown selected by *path* suffix."""

    output_path = Path(path)
    selected_format = format or (
        "json" if output_path.suffix.casefold() == ".json" else "markdown"
    )
    content = render_crosslingual_scorecard(
        reports,
        format=selected_format,
        expected_languages=expected_languages,
        languages=languages,
        required_languages=required_languages,
        family_by_model=family_by_model,
        manifest_rows=manifest_rows,
        indent=indent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        content if content.endswith("\n") else content + "\n",
        encoding="utf-8",
    )
    return output_path


@dataclass(frozen=True)
class _NormalizedReport:
    root: Mapping[str, Any]
    metrics: Mapping[str, Any]
    language: str | None
    family: str | None
    model_name: str | None
    fixture_count: int


@dataclass(frozen=True)
class _MetricSample:
    metric: str
    value: float
    weight: float = 1.0
    numerator: float | None = None
    denominator: float | None = None
    latency_stat: str | None = None


class _MetricAccumulator:
    """Collect raw numeric samples before deterministic reduction."""

    def __init__(self, metric: str) -> None:
        self.metric = metric
        self.samples: list[_MetricSample] = []

    def add(self, sample: _MetricSample | None) -> None:
        if isinstance(sample, _LatencyBundle):
            self.samples.extend(sample.samples)
        elif sample is not None:
            self.samples.append(sample)

    def summary(self) -> Any:
        samples = tuple(
            sorted(
                self.samples,
                key=lambda item: (
                    item.latency_stat or "",
                    item.value,
                    item.weight,
                    item.numerator if item.numerator is not None else -1.0,
                    item.denominator if item.denominator is not None else -1.0,
                ),
            )
        )
        if not samples:
            return None
        if self.metric == "critical_leakage":
            total = math.fsum(item.value for item in samples)
            return _integer_or_float(total)
        if self.metric in {"recall", "abstention"}:
            numerator = math.fsum(
                item.numerator
                if item.numerator is not None
                else item.value * max(item.weight, 1.0)
                for item in samples
            )
            denominator = math.fsum(
                item.denominator
                if item.denominator is not None
                else max(item.weight, 1.0)
                for item in samples
            )
            if denominator <= 0:
                return None
            return _clean_float(numerator / denominator)
        values = [item for item in samples if item.latency_stat == "value"]
        if self.metric == "latency_ms":
            # The scalar headline is a mean when a mean/scalar was supplied,
            # then p50, then p95.  Percentile summaries are never mixed into a
            # mean because that would imply a distribution we do not possess.
            mean = _weighted_latency(values)
            p50 = _weighted_latency(
                [item for item in samples if item.latency_stat == "p50"]
            )
            p95 = _weighted_latency(
                [item for item in samples if item.latency_stat == "p95"]
            )
            return {
                "mean_ms": mean,
                "p50_ms": p50,
                "p95_ms": p95,
                "headline_ms": (
                    mean if mean is not None else p50 if p50 is not None else p95
                ),
                "observations": len(samples),
            }
        return None

    def count_evidence(self) -> dict[str, Any]:
        """Return only count evidence associated with this metric."""

        samples = self.samples
        evidence: dict[str, Any] = {"observations": len(samples)}
        numerators = [item.numerator for item in samples if item.numerator is not None]
        denominators = [
            item.denominator for item in samples if item.denominator is not None
        ]
        if numerators:
            evidence["numerator"] = _integer_or_float(math.fsum(numerators))
        if denominators:
            evidence["denominator"] = _integer_or_float(math.fsum(denominators))
        if self.metric == "critical_leakage" and samples:
            evidence["count"] = _integer_or_float(
                math.fsum(item.value for item in samples)
            )
        return evidence


class _GroupAccumulator:
    """Mutable construction helper for one language or family row."""

    def __init__(self) -> None:
        self.report_count = 0
        self.fixture_count = 0
        self.metrics = {
            metric: _MetricAccumulator(
                "latency_ms" if metric == "latency_ms" else metric
            )
            for metric in SCORECARD_METRICS
        }

    def add_report(self, fixture_count: int) -> None:
        self.report_count += 1
        self.fixture_count += max(int(fixture_count), 0)

    def add_samples(self, samples: Mapping[str, _MetricSample | None]) -> None:
        for metric in SCORECARD_METRICS:
            self.metrics[metric].add(samples.get(metric))

    def to_dict(self) -> dict[str, Any]:
        recall = self.metrics["recall"].summary()
        critical = self.metrics["critical_leakage"].summary()
        abstention = self.metrics["abstention"].summary()
        latency = self.metrics["latency_ms"].summary()
        missing_metrics = [
            metric
            for metric, accumulator in self.metrics.items()
            if not accumulator.samples
        ]
        observed_metrics = [
            metric for metric in SCORECARD_METRICS if metric not in missing_metrics
        ]
        latency_mapping = latency if isinstance(latency, Mapping) else {}
        counts = {
            "abstention": self.metrics["abstention"].count_evidence(),
            "critical_leakage": self.metrics["critical_leakage"].count_evidence(),
            "fixtures": int(self.fixture_count),
            "latency": self.metrics["latency_ms"].count_evidence(),
            "recall": self.metrics["recall"].count_evidence(),
            "reports": int(self.report_count),
        }
        metric_counts = {
            metric: self.metrics[metric].count_evidence()
            for metric in SCORECARD_METRICS
        }
        metric_values = {
            "abstention": abstention,
            "critical_leakage": critical,
            "latency_ms": latency_mapping.get("headline_ms"),
            "recall": recall,
        }
        return {
            "abstention": abstention,
            "abstention_rate": abstention,
            "critical_leakage": critical,
            "critical_leakage_count": critical,
            "counts": counts,
            "latency_ms": latency_mapping.get("headline_ms"),
            "latency_mean_ms": latency_mapping.get("mean_ms"),
            "latency_p50_ms": latency_mapping.get("p50_ms"),
            "latency_p95_ms": latency_mapping.get("p95_ms"),
            "metric_counts": metric_counts,
            "metrics": metric_values,
            "missing_metrics": missing_metrics,
            "observed_metrics": observed_metrics,
            "recall": recall,
        }


def _normalize_report(report: Any) -> _NormalizedReport:
    if isinstance(report, Mapping):
        root = dict(report)
    elif hasattr(report, "to_dict") and callable(report.to_dict):
        candidate = report.to_dict()
        if not isinstance(candidate, Mapping):
            raise TypeError("evaluation reports must serialize to mappings")
        root = dict(candidate)
    else:
        raise TypeError("evaluation reports must be mappings or report objects")

    raw_metrics = root.get("metrics")
    metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else root
    metadata = root.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    language = _first_label(
        root.get("language"),
        root.get("lang"),
        root.get("locale"),
        root.get("language_code"),
        root.get("source_language"),
        root.get("target_language"),
        metadata_mapping.get("language"),
        metadata_mapping.get("lang"),
        metadata_mapping.get("locale"),
        metadata_mapping.get("language_code"),
    )
    family = _first_label(
        root.get("family"),
        root.get("model_family"),
        metadata_mapping.get("family"),
        metadata_mapping.get("model_family"),
    )
    model_name = _first_text(
        root.get("model_name"),
        metadata_mapping.get("model_name"),
    )
    fixture_count = _safe_count(root.get("fixture_count"), default=1)
    return _NormalizedReport(
        root=root,
        metrics=metrics,
        language=language,
        family=family,
        model_name=model_name,
        fixture_count=fixture_count,
    )


def _coerce_report_items(reports: Any) -> tuple[Any, ...]:
    """Accept one report, a report collection, or a wrapper mapping."""

    if isinstance(reports, CrossLingualScorecard):
        return (reports.to_dict(),)
    if isinstance(reports, Mapping):
        for key in ("reports", "results", "evaluation_reports"):
            nested = reports.get(key)
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, bytearray)
            ):
                return tuple(nested)
        if any(
            key in reports
            for key in ("metrics", "suite", "model_name", "language", "metadata")
        ):
            return (reports,)
        values = tuple(reports.values())
        if values and all(isinstance(value, Mapping) for value in values):
            return values
        return (reports,)
    if hasattr(reports, "to_dict") and callable(reports.to_dict):
        return (reports,)
    if isinstance(reports, (str, bytes, bytearray)):
        raise TypeError("evaluation reports must be report objects or collections")
    try:
        return tuple(reports)
    except TypeError as exc:
        raise TypeError(
            "evaluation reports must be report objects or collections"
        ) from exc


def _resolve_expected_languages(
    expected_languages: Sequence[str] | None,
    *,
    languages: Sequence[str] | None,
    required_languages: Sequence[str] | None,
) -> tuple[str, ...] | None:
    provided = [
        value
        for value in (expected_languages, languages, required_languages)
        if value is not None
    ]
    if len(provided) > 1:
        first = _normalise_labels(provided[0])
        if any(_normalise_labels(value) != first for value in provided[1:]):
            raise ValueError("language expectations were provided more than once")
    if not provided:
        return None
    return tuple(sorted(_normalise_labels(provided[0])))


def _resolve_family_lookup(
    *,
    family_by_model: Mapping[str, Any] | None,
    manifest_rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if family_by_model is not None:
        for model, value in family_by_model.items():
            family = value.get("family") if isinstance(value, Mapping) else value
            safe_family = _safe_label(family)
            if safe_family is not None:
                lookup[str(model)] = safe_family
    if manifest_rows is not None:
        for row in manifest_rows:
            if not isinstance(row, Mapping):
                continue
            model = _first_text(row.get("model_name"), row.get("repo_id"))
            family = _safe_label(row.get("family"))
            if model is not None and family is not None and model not in lookup:
                lookup[model] = family
    return lookup


def _collect_language_payloads(
    root: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def merge(language_value: Any, payload: Any) -> None:
        language = _safe_label(language_value)
        if language is None:
            return
        target = result.setdefault(language, {})
        if isinstance(payload, Mapping):
            target.update({str(key): value for key, value in payload.items()})

    def scan(source: Mapping[str, Any], depth: int) -> None:
        if depth > 3:
            return
        for raw_key, value in source.items():
            key = _normalise_key(raw_key)
            if key in _LANGUAGE_CONTAINER_KEYS and isinstance(value, Mapping):
                for language, payload in value.items():
                    if _safe_label(language) is not None and isinstance(
                        payload, Mapping
                    ):
                        merge(language, payload)
                continue
            if key in _METRIC_ALIASES_BY_KEY:
                if isinstance(value, Mapping):
                    by_language = _language_mapping(value)
                    if by_language is not None:
                        for language, metric_value in by_language.items():
                            merge(language, {str(raw_key): metric_value})
                    scan(value, depth + 1)
                continue
            if key.endswith("_by_language") or key.endswith("_per_language"):
                if isinstance(value, Mapping):
                    base_key = key.rsplit("_", 2)[0]
                    for language, metric_value in value.items():
                        merge(language, {base_key: metric_value})
                continue
            if key in _NESTED_METRIC_CONTAINER_KEYS and isinstance(value, Mapping):
                scan(value, depth + 1)

    _METRIC_ALIASES_BY_KEY = {
        _normalise_key(alias)
        for aliases in _METRIC_ALIASES.values()
        for alias in aliases
    }
    scan(root, 0)
    if metrics is not root:
        scan(metrics, 0)
    return {language: dict(payload) for language, payload in sorted(result.items())}


def _language_mapping(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not value:
        return None
    candidate: dict[str, Any] = {}
    for raw_key, item in value.items():
        if _normalise_key(raw_key) in _LANGUAGE_CONTAINER_KEYS and isinstance(
            item, Mapping
        ):
            nested = _language_mapping(item)
            if nested is not None:
                candidate.update(nested)
            continue
        language = _safe_label(raw_key)
        if language is None or _normalise_key(raw_key) in _NON_LANGUAGE_KEYS:
            return None
        candidate[language] = item
    return candidate or None


def _language_fixture_count(
    payload: Mapping[str, Any],
    report: _NormalizedReport,
    *,
    language_count: int,
) -> int:
    direct = _safe_count(payload.get("fixture_count"), default=0)
    if direct:
        return direct
    if language_count == 1:
        return report.fixture_count
    for metric in ("recall", "abstention"):
        sample = _sample_for_metric(payload, metric, report.fixture_count)
        if sample is not None and sample.denominator is not None:
            return max(int(round(sample.denominator)), 1)
    return 1


def _samples_for_source(
    source: Mapping[str, Any],
    default_weight: int,
) -> dict[str, _MetricSample | None]:
    return {
        metric: _sample_for_metric(source, metric, default_weight)
        for metric in SCORECARD_METRICS
    }


def _fill_missing_samples(
    samples: Mapping[str, _MetricSample | None],
    root_source: Mapping[str, Any],
    default_weight: int,
) -> dict[str, _MetricSample | None]:
    result = dict(samples)
    root_samples = _samples_for_source(root_source, default_weight)
    for metric in SCORECARD_METRICS:
        if result[metric] is None:
            result[metric] = root_samples[metric]
    return result


def _merge_samples(
    metric: str,
    samples: Sequence[_MetricSample],
) -> _MetricSample | None:
    if not samples:
        return None
    if metric == "latency_ms":
        expanded: list[_MetricSample] = []
        for sample in samples:
            if isinstance(sample, _LatencyBundle):
                expanded.extend(sample.samples)
            else:
                expanded.append(sample)
        return _merge_latency_samples(expanded)
    if metric == "critical_leakage":
        return _MetricSample(metric, math.fsum(item.value for item in samples))
    numerator_values = [
        item.numerator for item in samples if item.numerator is not None
    ]
    denominator_values = [
        item.denominator for item in samples if item.denominator is not None
    ]
    numerator = math.fsum(numerator_values) if numerator_values else None
    denominator = math.fsum(denominator_values) if denominator_values else None
    if numerator is not None and denominator is not None and denominator > 0:
        value = numerator / denominator
    else:
        weight = math.fsum(item.weight for item in samples)
        value = math.fsum(item.value * item.weight for item in samples) / max(
            weight, 1.0
        )
    return _MetricSample(
        metric=metric,
        value=value,
        weight=denominator or math.fsum(item.weight for item in samples),
        numerator=numerator,
        denominator=denominator,
    )


def _sample_for_metric(
    source: Mapping[str, Any],
    metric: str,
    default_weight: int,
) -> _MetricSample | None:
    value = _find_metric_value(source, metric)
    if value is _MISSING:
        return None
    if metric in {"recall", "abstention"}:
        return _rate_sample(metric, value, default_weight)
    if metric == "critical_leakage":
        return _count_sample(value, default_weight)
    return _latency_sample(value, default_weight)


def _find_metric_value(source: Mapping[str, Any], metric: str) -> Any:
    aliases = {_normalise_key(alias) for alias in _METRIC_ALIASES[metric]}
    candidates: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def visit(mapping: Mapping[str, Any], depth: int) -> None:
        if id(mapping) in seen or depth > 3:
            return
        seen.add(id(mapping))
        candidates.append(mapping)
        for key, value in mapping.items():
            if _normalise_key(key) in _NESTED_METRIC_CONTAINER_KEYS and isinstance(
                value, Mapping
            ):
                visit(value, depth + 1)

    visit(source, 0)
    for mapping in candidates:
        for key, value in mapping.items():
            if _normalise_key(key) in aliases:
                return value
    return _MISSING


def _rate_sample(
    metric: str,
    value: Any,
    default_weight: int,
) -> _MetricSample | None:
    numerator: float | None = None
    denominator: float | None = None
    rate: float | None = None
    if isinstance(value, Mapping):
        numerator = _number_from_keys(
            value,
            (
                "abstained",
                "count",
                "covered",
                "correct",
                "hits",
                "numerator",
                "true_positives",
            ),
        )
        denominator = _number_from_keys(
            value,
            ("denominator", "gold", "support", "total"),
        )
        if metric == "abstention" and numerator is None:
            accepted = _number_from_keys(value, ("accepted",))
            if accepted is not None and denominator is not None:
                numerator = denominator - accepted
        rate = _number_from_keys(value, ("rate", "score", "value"))
        if rate is None:
            rate = _number_from_keys(value, _METRIC_ALIASES[metric])
    else:
        rate = _finite_number(value)
    if numerator is not None and denominator is not None and denominator > 0:
        rate = numerator / denominator
    if rate is None or not 0.0 <= rate <= 1.0:
        return None
    if denominator is None or denominator <= 0:
        denominator = float(max(default_weight, 1))
    if numerator is None:
        numerator = rate * denominator
    return _MetricSample(
        metric=metric,
        value=rate,
        weight=denominator,
        numerator=numerator,
        denominator=denominator,
    )


def _count_sample(value: Any, default_weight: int) -> _MetricSample | None:
    count: float | None
    if isinstance(value, Mapping):
        count = _number_from_keys(
            value,
            (
                "critical_leakage_count",
                "critical_leakage_total",
                "critical_count",
                "count",
                "leaked",
                "residual_count",
                "value",
            ),
        )
        if count is None:
            rate = _number_from_keys(value, ("rate", "critical_leakage_rate"))
            denominator = _number_from_keys(value, ("denominator", "total"))
            if rate is not None and denominator is not None:
                count = rate * denominator
    else:
        count = _finite_number(value)
    if count is None or count < 0:
        return None
    return _MetricSample(
        metric="critical_leakage",
        value=count,
        weight=max(default_weight, 1),
    )


def _latency_sample(value: Any, default_weight: int) -> _MetricSample | None:
    if isinstance(value, Mapping):
        samples: list[_MetricSample] = []
        for stat, keys in (
            ("value", ("mean_ms", "average_ms", "latency_ms", "mean", "average")),
            ("p50", ("p50_ms", "p50", "median_ms", "median")),
            ("p95", ("p95_ms", "p95")),
        ):
            number = _number_from_keys(value, keys)
            if number is not None and number >= 0:
                samples.append(
                    _MetricSample(
                        metric="latency_ms",
                        value=number,
                        weight=max(default_weight, 1),
                        latency_stat=stat,
                    )
                )
        if samples:
            # The scorecard row stores all latency statistics in one
            # accumulator; one source can contribute one sample per statistic.
            return _merge_latency_samples(samples)
        return None
    number = _finite_number(value)
    if number is None or number < 0:
        return None
    return _MetricSample(
        metric="latency_ms",
        value=number,
        weight=max(default_weight, 1),
        latency_stat="value",
    )


def _merge_latency_samples(samples: Sequence[_MetricSample]) -> _MetricSample:
    """Keep one scalar sample while retaining all stats in a private tuple."""

    # ``_MetricAccumulator.add`` accepts one sample, so a latency mapping is
    # represented by a scalar headline here.  The source's p50/p95 values are
    # also copied into private attributes by the lightweight tuple subclass
    # below; this avoids exposing the raw mapping in output.
    first = samples[0]
    return _LatencyBundle(
        metric="latency_ms",
        value=first.value,
        weight=first.weight,
        latency_stat=first.latency_stat,
        samples=tuple(samples),
    )


@dataclass(frozen=True)
class _LatencyBundle(_MetricSample):
    samples: tuple[_MetricSample, ...] = ()


def _weighted_latency(samples: Sequence[_MetricSample]) -> float | None:
    if not samples:
        return None
    weights = math.fsum(max(item.weight, 1.0) for item in samples)
    return _clean_float(
        math.fsum(item.value * max(item.weight, 1.0) for item in samples)
        / max(weights, 1.0)
    )


def _clean_float(value: float) -> float:
    return round(float(value), 12)


def _integer_or_float(value: float) -> int | float:
    cleaned = _clean_float(value)
    if cleaned.is_integer():
        return int(cleaned)
    return cleaned


def _number_from_keys(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    targets = {_normalise_key(key) for key in keys}
    for key, value in mapping.items():
        if _normalise_key(key) in targets:
            number = _finite_number(value)
            if number is not None:
                return number
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _safe_count(value: Any, *, default: int) -> int:
    number = _finite_number(value)
    if number is None or number < 0:
        return default
    return int(number)


def _first_label(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if len(value) == 1:
                safe = _safe_label(value[0])
            else:
                safe = None
        else:
            safe = _safe_label(value)
        if safe is not None:
            return safe
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if _LABEL_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


def _normalise_labels(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (str(values),)
    result = {_safe_label(value) for value in values}
    result.discard(None)
    return tuple(sorted(result))  # type: ignore[arg-type]


def _normalise_key(value: Any) -> str:
    text = str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _snapshot_rows(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(key): _plain(value)
        for key, value in sorted(rows.items(), key=lambda item: str(item[0]))
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _markdown_group_row(label: str, row: Mapping[str, Any]) -> str:
    missing = row.get("missing_metrics") or []
    missing_text = ", ".join(str(item) for item in missing) or "none"
    return (
        f"| `{_markdown_cell(label)}` | {row.get('report_count', 0)} | "
        f"{row.get('fixture_count', 0)} | {_format_rate(row.get('recall'))} | "
        f"{_format_number(row.get('critical_leakage'))} | "
        f"{_format_rate(row.get('abstention'))} | "
        f"{_format_number(row.get('latency_ms'))} | `{_markdown_cell(missing_text)}` |"
    )


def _format_rate(value: Any) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number:.2%}"


def _format_number(value: Any) -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    return str(_integer_or_float(number))


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("`", "'").replace("\n", " ")
