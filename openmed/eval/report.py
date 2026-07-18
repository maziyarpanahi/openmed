"""Benchmark report serialization."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from openmed.core.audit import stable_hash
from openmed.core.baseline import baseline_key, get_baseline, load_baseline_store
from openmed.core.model_registry import MANIFEST_PATH, load_manifest_rows
from openmed.eval.metrics import inter_annotator_agreement

if TYPE_CHECKING:
    from openmed.eval.golden.loader import ConsensusDocument


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable benchmark report emitted by the eval harness."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    metrics: Mapping[str, Any]
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary with stable keys."""
        result: dict[str, Any] = {
            "device": self.device,
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
            "metrics": _plain(self.metrics),
            "model_name": self.model_name,
            "suite": self.suite,
        }
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkReport":
        """Create a report from a JSON-compatible mapping."""
        return cls(
            suite=str(payload["suite"]),
            model_name=str(payload["model_name"]),
            device=str(payload["device"]),
            fixture_count=int(payload["fixture_count"]),
            metrics=payload.get("metrics", {}) or {},
            generated_at=payload.get("generated_at"),
            metadata=payload.get("metadata", {}) or {},
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "BenchmarkReport":
        """Read a benchmark report JSON file."""
        report_path = Path(path)
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Benchmark report must be a JSON object: {report_path}")
        return cls.from_dict(payload)

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
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report to deterministic Markdown."""
        lines = [
            f"# Benchmark Report: {self.suite}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(["", "## Metrics", "", "| Metric | Value |", "|---|---:|"])
        for key, value in _flatten(_plain(self.metrics)):
            lines.append(f"| `{key}` | {_format_value(value)} |")

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


DEFAULT_COMPETITORS = ("SHIELD", "Declared competitors")


@dataclass(frozen=True)
class GoldCorpusQualityReport:
    """Agreement and adjudication quality of a multi-annotator gold corpus.

    Low-agreement examples reference offsets, labels, document ids, and content
    hashes only -- never raw clinical text -- so reviewers can triage before
    promotion without handling PHI.
    """

    n_documents: int
    overall_agreement: float
    per_label: Mapping[str, float]
    relation_agreement: Mapping[str, float]
    relation_types: Mapping[str, int]
    adjudication_coverage: float
    low_agreement_examples: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_documents": self.n_documents,
            "overall_agreement": self.overall_agreement,
            "per_label": dict(self.per_label),
            "relation_agreement": dict(self.relation_agreement),
            "relation_types": dict(self.relation_types),
            "adjudication_coverage": self.adjudication_coverage,
            "low_agreement_examples": [
                {
                    "document_id": example["document_id"],
                    "offset": list(example["offset"]),
                    "labels": list(example["labels"]),
                    "resolved": example["resolved"],
                    "hash": example["hash"],
                }
                for example in self.low_agreement_examples
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Gold Corpus Quality Report",
            "",
            f"- Documents: {self.n_documents}",
            f"- Overall agreement: {self.overall_agreement:.4f}",
            f"- Adjudication coverage: {self.adjudication_coverage:.4f}",
            "",
            "## Agreement by label",
            "",
            "| Label | Agreement |",
            "| --- | --- |",
        ]
        for label, score in self.per_label.items():
            lines.append(f"| {label} | {score:.4f} |")
        lines.extend(["", "## Agreement by relation type", ""])
        if self.relation_agreement:
            lines.append("| Relation type | Agreement |")
            lines.append("| --- | --- |")
            for relation_type, score in self.relation_agreement.items():
                lines.append(f"| {relation_type} | {score:.4f} |")
        else:
            lines.append("_None._")
        lines.extend(["", "## Consensus relations by type", ""])
        if self.relation_types:
            lines.append("| Relation type | Count |")
            lines.append("| --- | --- |")
            for relation_type, count in self.relation_types.items():
                lines.append(f"| {relation_type} | {count} |")
        else:
            lines.append("_None._")
        lines.extend(
            ["", f"## Low-agreement examples ({len(self.low_agreement_examples)})", ""]
        )
        for example in self.low_agreement_examples:
            offset = tuple(example["offset"])
            labels = ", ".join(example["labels"])
            resolved = "resolved" if example["resolved"] else "unresolved"
            lines.append(
                f"- {example['document_id']} {offset} [{labels}] "
                f"({resolved}, {example['hash']})"
            )
        return "\n".join(lines) + "\n"


def _relation_key(relation: Any) -> tuple[Any, ...]:
    return (
        relation.head.start,
        relation.head.end,
        relation.head.label,
        relation.tail.start,
        relation.tail.end,
        relation.tail.label,
        relation.label,
    )


def _relation_agreement_by_type(
    documents: Sequence["ConsensusDocument"],
) -> dict[str, float]:
    agreement_flags: dict[str, list[float]] = defaultdict(list)
    for document in documents:
        annotator_relations = list(document.annotator_relations.values())
        relation_types = sorted(
            {
                relation.relation_type
                for relations in annotator_relations
                for relation in relations
            }
        )
        for relation_type in relation_types:
            relation_sets = [
                {
                    _relation_key(relation)
                    for relation in relations
                    if relation.relation_type == relation_type
                }
                for relations in annotator_relations
            ]
            relation_items = set().union(*relation_sets)
            for item in relation_items:
                presence = {item in relations for relations in relation_sets}
                agreement_flags[relation_type].append(
                    1.0 if len(presence) == 1 else 0.0
                )

    scores = {
        relation_type: sum(flags) / len(flags)
        for relation_type, flags in agreement_flags.items()
    }
    return dict(sorted(scores.items(), key=lambda item: (item[1], item[0])))


def gold_corpus_quality_report(
    documents: Sequence["ConsensusDocument"],
) -> GoldCorpusQualityReport:
    """Build a quality report from a multi-annotator consensus corpus.

    Agreement is computed per document from the annotator span sets (Cohen or
    Fleiss kappa, and per-label agreement), relation types are counted from the
    adjudicated consensus, and adjudication coverage is the fraction of
    annotator disagreements represented in the consensus view.
    """

    kappas: list[float] = []
    label_scores: dict[str, list[float]] = defaultdict(list)
    relation_counts: dict[str, int] = defaultdict(int)
    low_agreement: list[Mapping[str, Any]] = []
    resolved_count = 0
    disagreement_count = 0

    for document in documents:
        annotator_spans = [
            [(span.start, span.end, span.label) for span in spans]
            for spans in document.annotators.values()
        ]
        agreement = inter_annotator_agreement(annotator_spans)
        kappa = (
            agreement.cohen_kappa
            if agreement.cohen_kappa is not None
            else agreement.fleiss_kappa
        )
        if kappa is not None:
            kappas.append(kappa)
        for label, score in agreement.per_label.items():
            label_scores[label].append(score)
        for relation in document.consensus_relations:
            relation_counts[relation.relation_type] += 1

        consensus_offsets = {
            (span.start, span.end) for span in document.consensus_spans
        }
        for item in agreement.disagreements:
            disagreement_count += 1
            offset = tuple(item["offset"])
            labels = tuple(item["labels"])
            resolved = offset in consensus_offsets
            if resolved:
                resolved_count += 1
            low_agreement.append(
                {
                    "document_id": document.doc_id,
                    "offset": offset,
                    "labels": labels,
                    "resolved": resolved,
                    "hash": stable_hash(
                        {
                            "document_id": document.doc_id,
                            "offset": list(offset),
                            "labels": list(labels),
                        }
                    ),
                }
            )

    overall = sum(kappas) / len(kappas) if kappas else 1.0
    label_averages = {
        label: sum(scores) / len(scores) for label, scores in label_scores.items()
    }
    per_label = dict(
        sorted(label_averages.items(), key=lambda item: (item[1], item[0]))
    )
    coverage = resolved_count / disagreement_count if disagreement_count else 1.0

    return GoldCorpusQualityReport(
        n_documents=len(documents),
        overall_agreement=overall,
        per_label=per_label,
        relation_agreement=_relation_agreement_by_type(documents),
        relation_types=dict(sorted(relation_counts.items())),
        adjudication_coverage=coverage,
        low_agreement_examples=tuple(low_agreement),
    )


def render_benchmark_card(
    report: BenchmarkReport,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Render a benchmark card sourced from a BenchmarkReport and manifest."""

    manifest_row = _find_manifest_row(report.model_name, manifest_rows)
    lines = [
        f"# Benchmark Card: {report.suite}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Suite | `{report.suite}` |",
        f"| Model | `{report.model_name}` |",
        f"| Device | `{report.device}` |",
        f"| Fixtures | {report.fixture_count} |",
    ]
    if report.generated_at is not None:
        lines.append(f"| Report Timestamp | `{report.generated_at}` |")
    if manifest_row is not None:
        lines.extend(
            [
                f"| Family | `{manifest_row.get('family')}` |",
                f"| Tier | `{_display_value(manifest_row.get('tier'))}` |",
                f"| Formats | `{', '.join(manifest_row.get('formats') or [])}` |",
                f"| Released | `{_display_value(manifest_row.get('released'))}` |",
                (
                    "| Reproducibility Hash | "
                    f"`{_display_value(manifest_row.get('reproducibility_hash'))}` |"
                ),
            ]
        )

    lines.extend(["", "## Metrics", "", "| Metric | Value |", "|---|---:|"])
    for key, value in _flatten(_plain(report.metrics)):
        lines.append(f"| `{key}` | {_format_value(value)} |")

    lines.extend(
        [
            "",
            "## Source Inputs",
            "",
            "- BenchmarkReport JSON",
            "- Canonical model manifest",
        ]
    )
    return "\n".join(lines) + "\n"


def write_benchmark_card(
    report: BenchmarkReport,
    output_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
) -> Path:
    """Write ``docs/benchmarks/<suite>.md`` for one report."""

    output_path = Path(output_dir) / f"{_slug(report.suite)}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_benchmark_card(report, manifest_rows=manifest_rows),
        encoding="utf-8",
    )
    return output_path


def write_benchmark_cards(
    reports: Iterable[BenchmarkReport],
    output_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
) -> list[Path]:
    """Write benchmark cards for all supplied reports."""

    rows = list(manifest_rows)
    return [
        write_benchmark_card(report, output_dir, manifest_rows=rows)
        for report in reports
    ]


def render_leaderboard(
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
    reports: Iterable[BenchmarkReport] = (),
    baseline_store: Mapping[str, Any] | None = None,
    competitors: Sequence[str] = DEFAULT_COMPETITORS,
) -> str:
    """Render a manifest/report-sourced benchmark leaderboard."""

    rows = list(manifest_rows)
    report_rows = list(reports)
    store = baseline_store or {"schema_version": 1, "entries": {}}
    aggregates = _manifest_aggregates(rows)
    reports_by_key = _reports_by_key(report_rows, rows)

    lines = [
        "# Open Benchmark Leaderboard",
        "",
        "OpenMed rows are aggregated from the canonical model manifest and joined",
        "with the latest BenchmarkReport for each family, tier, and format.",
        "",
        "| System | Family | Tier | Format | Models | Current Leakage | Last Green Release | Harness Freshness | Evidence |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ]

    for key in sorted(aggregates):
        aggregate = aggregates[key]
        report = reports_by_key.get(key)
        baseline = get_baseline(
            aggregate["family"],
            aggregate["tier"],
            aggregate["format"],
            store=store,
        )
        lines.append(
            "| "
            f"OpenMed | "
            f"`{aggregate['family']}` | "
            f"`{_display_value(aggregate['tier'])}` | "
            f"`{aggregate['format']}` | "
            f"{aggregate['model_count']} | "
            f"{_format_percent(_metric_lookup(report, ('leakage.overall', 'leakage_rate.overall')))} | "
            f"`{_display_value((baseline or {}).get('released'))}` | "
            f"`{_display_value(report.generated_at if report else None)}` | "
            f"`{_display_value(report.suite if report else None)}` |"
        )

    for competitor in competitors:
        lines.append(
            "| "
            f"{competitor} | `declared competitor` | `n/a` | `n/a` | "
            "n/a | n/a | n/a | n/a | `no BenchmarkReport source` |"
        )

    return "\n".join(lines) + "\n"


def write_leaderboard(
    output_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
    reports: Iterable[BenchmarkReport] = (),
    baseline_store: Mapping[str, Any] | None = None,
    competitors: Sequence[str] = DEFAULT_COMPETITORS,
) -> Path:
    """Write ``docs/leaderboard/index.md``."""

    output_path = Path(output_dir) / "index.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_leaderboard(
            manifest_rows=manifest_rows,
            reports=reports,
            baseline_store=baseline_store,
            competitors=competitors,
        ),
        encoding="utf-8",
    )
    return output_path


def read_reports(paths: Iterable[str | Path]) -> list[BenchmarkReport]:
    """Read all supplied BenchmarkReport JSON files."""

    return [BenchmarkReport.read_json(path) for path in paths]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the report surface generator CLI parser."""

    parser = argparse.ArgumentParser(
        description="Render benchmark cards and leaderboard from manifest reports.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--report", action="append", type=Path, default=[])
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument(
        "--leaderboard-dir", type=Path, default=Path("docs/leaderboard")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render requested benchmark report artifacts."""

    args = build_arg_parser().parse_args(argv)
    manifest_rows = load_manifest_rows(args.manifest)
    reports = read_reports(args.report)
    write_benchmark_cards(reports, args.benchmarks_dir, manifest_rows=manifest_rows)
    if args.leaderboard:
        baseline_store = (
            load_baseline_store(args.baseline)
            if args.baseline is not None and args.baseline.exists()
            else None
        )
        write_leaderboard(
            args.leaderboard_dir,
            manifest_rows=manifest_rows,
            reports=reports,
            baseline_store=baseline_store,
        )
    return 0


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(value[key], child_prefix))
        return rows
    return [(prefix, value)]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _find_manifest_row(
    model_name: str,
    rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("repo_id") == model_name:
            return row
    return None


def _manifest_aggregates(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row.get("family") or "Unknown")
        tier = row.get("tier")
        for format_name in row.get("formats") or ["unknown"]:
            key = baseline_key(family, tier, str(format_name))
            aggregate = aggregates.setdefault(
                key,
                {
                    "family": family,
                    "tier": tier,
                    "format": str(format_name),
                    "model_count": 0,
                    "latest_release": None,
                },
            )
            aggregate["model_count"] += 1
            released = row.get("released")
            if released and (
                aggregate["latest_release"] is None
                or str(released) > str(aggregate["latest_release"])
            ):
                aggregate["latest_release"] = str(released)
    return aggregates


def _reports_by_key(
    reports: Iterable[BenchmarkReport],
    manifest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, BenchmarkReport]:
    rows = list(manifest_rows)
    reports_by_key: dict[str, BenchmarkReport] = {}
    for report in reports:
        row = _find_manifest_row(report.model_name, rows)
        if row is not None:
            family = str(
                row.get("family") or report.metadata.get("family") or "Unknown"
            )
            tier = row.get("tier")
            formats = row.get("formats") or [report.device]
            format_name = report.device if report.device in formats else formats[0]
        else:
            family = str(report.metadata.get("family") or "Unknown")
            tier = report.metadata.get("tier")
            format_name = str(report.metadata.get("format") or report.device)
        reports_by_key[baseline_key(family, tier, str(format_name))] = report
    return reports_by_key


def _metric_lookup(report: BenchmarkReport | None, keys: Sequence[str]) -> Any:
    if report is None:
        return None
    for key in keys:
        value = _nested_get(report.metrics, key)
        if value is not None:
            return value
    return None


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return str(value)


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _slug(value: str) -> str:
    return (
        "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
        or "report"
    )


__all__ = [
    "BenchmarkReport",
    "DEFAULT_COMPETITORS",
    "GoldCorpusQualityReport",
    "gold_corpus_quality_report",
    "read_reports",
    "render_benchmark_card",
    "render_leaderboard",
    "write_benchmark_card",
    "write_benchmark_cards",
    "write_leaderboard",
]


if __name__ == "__main__":
    raise SystemExit(main())
