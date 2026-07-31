"""Deterministic public leaderboard rendering for archived benchmark reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from openmed.core.model_registry import MANIFEST_PATH, load_manifest_rows
from openmed.eval.report import BenchmarkReport

LEADERBOARD_SCHEMA_VERSION = 1
DEFAULT_REPORTS_DIR = Path("docs/benchmarks")
DEFAULT_OUTPUT_DIR = Path("docs/eval/benchmark-leaderboard")

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_REPORT_DISCRIMINATORS = frozenset({"suite", "model_name", "fixture_count", "metrics"})
_LEAKAGE_METRICS = (
    "leakage.overall",
    "leakage_rate.overall",
    "leakage_rate",
    "critical_leakage_rate",
)
_RECALL_METRICS = (
    "exact_span_f1.recall",
    "span.recall",
    "micro.recall",
    "recall",
)
_F1_METRICS = (
    "exact_span_f1.f1",
    "span.f1",
    "micro.f1",
    "micro_f1",
    "f1",
)


class LeaderboardError(ValueError):
    """Raised when archived reports cannot form a trustworthy leaderboard."""


@dataclass(frozen=True)
class LeaderboardRow:
    """One public leaderboard row derived from an archived report."""

    suite: str
    model_family: str
    model_name: str
    device: str
    leakage: float
    recall: float
    f1: float
    release_tag: str
    run_date: str
    reproducibility_hash: str
    report_path: str
    report_url: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable machine-readable row representation."""

        return {
            "device": self.device,
            "f1": self.f1,
            "leakage": self.leakage,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "recall": self.recall,
            "release_tag": self.release_tag,
            "report_path": self.report_path,
            "report_url": self.report_url,
            "reproducibility_hash": self.reproducibility_hash,
            "run_date": self.run_date,
            "suite": self.suite,
        }


@dataclass(frozen=True)
class LeaderboardArtifacts:
    """Paths written for one public leaderboard render."""

    html_path: Path
    json_path: Path
    report_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _ArchivedReport:
    path: Path
    relative_path: Path
    payload: Mapping[str, Any]
    report: BenchmarkReport


def load_leaderboard_rows(
    reports_dir: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
    release_tag: str | None = None,
) -> list[LeaderboardRow]:
    """Load and rank archived reports for public rendering.

    Args:
        reports_dir: Directory recursively containing archived report JSON.
        manifest_rows: Optional model manifest rows used to fill family and
            reproducibility metadata absent from older reports.
        release_tag: Fallback release tag for reports that predate that field.

    Returns:
        Rows sorted by leakage ascending, recall descending, then stable
        identity fields.

    Raises:
        LeaderboardError: If the archive is empty or a required field is
            missing or invalid.
    """

    archives = _load_archived_reports(reports_dir)
    manifests = {
        str(row.get("repo_id")): row
        for row in manifest_rows
        if row.get("repo_id") is not None
    }
    rows = [
        _row_from_archive(
            archive,
            manifest_row=manifests.get(archive.report.model_name),
            release_tag=release_tag,
        )
        for archive in archives
    ]
    return sorted(rows, key=_row_sort_key)


def render_leaderboard_json(rows: Iterable[LeaderboardRow]) -> str:
    """Render deterministic machine-readable leaderboard JSON."""

    ranked_rows = sorted(rows, key=_row_sort_key)
    payload = {
        "row_count": len(ranked_rows),
        "rows": [row.to_dict() for row in ranked_rows],
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "sort": ["leakage:asc", "recall:desc"],
        "suites": _suite_payload(ranked_rows),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_leaderboard_html(rows: Iterable[LeaderboardRow]) -> str:
    """Render deterministic leaderboard HTML using the shared docs system."""

    ranked_rows = sorted(rows, key=_row_sort_key)
    suites = _group_rows(ranked_rows)
    tab_buttons: list[str] = []
    panels: list[str] = []

    for suite_index, (suite, families) in enumerate(suites.items()):
        tab_id = f"suite-{suite_index}"
        selected = suite_index == 0
        suite_row_count = sum(len(family_rows) for family_rows in families.values())
        suite_row_noun = "row" if suite_row_count == 1 else "rows"
        escaped_suite = html.escape(suite)
        tab_buttons.append(
            '<button class="tab" role="tab" '
            f'id="{tab_id}-tab" aria-controls="{tab_id}" '
            f'aria-selected="{"true" if selected else "false"}" '
            f'tabindex="{"0" if selected else "-1"}" '
            f'data-target="{tab_id}" data-label="{escaped_suite}" '
            f'aria-label="{escaped_suite}, {suite_row_count} benchmark '
            f'{suite_row_noun}">'
            f"{escaped_suite} "
            '<span data-leaderboard-tab-count aria-hidden="true">'
            f"({suite_row_count})</span></button>"
        )
        family_sections = [
            _render_family_table(family, family_rows)
            for family, family_rows in families.items()
        ]
        panels.append(
            f'<section class="suite-panel" id="{tab_id}" '
            f'role="tabpanel" aria-labelledby="{tab_id}-tab" tabindex="0" '
            f'data-match-count="{suite_row_count}"'
            f"{' hidden' if not selected else ''}>"
            f"{''.join(family_sections)}</section>"
        )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '  <meta name="referrer" content="no-referrer">\n'
        '  <meta name="description" content="Versioned OpenMed synthetic '
        'benchmark results ranked by leakage first and recall second.">\n'
        '  <link rel="canonical" href="https://openmed.life/docs/eval/'
        'benchmark-leaderboard/">\n'
        '  <meta property="og:type" content="website">\n'
        '  <meta property="og:site_name" content="OpenMed">\n'
        '  <meta property="og:title" content="OpenMed Public Benchmark '
        'Leaderboard">\n'
        '  <meta property="og:description" content="Versioned OpenMed synthetic '
        'benchmark results ranked by leakage first and recall second.">\n'
        '  <meta property="og:url" content="https://openmed.life/docs/eval/'
        'benchmark-leaderboard/">\n'
        '  <meta property="og:image" content="https://openmed.life/og.png">\n'
        '  <meta property="og:image:alt" content="OpenMed local-first '
        'healthcare NLP">\n'
        '  <meta name="twitter:card" content="summary_large_image">\n'
        '  <meta name="twitter:title" content="OpenMed Public Benchmark '
        'Leaderboard">\n'
        '  <meta name="twitter:description" content="Versioned OpenMed synthetic '
        'benchmark results ranked by leakage first and recall second.">\n'
        '  <meta name="twitter:image" content="https://openmed.life/og.png">\n'
        '  <meta name="twitter:image:alt" content="OpenMed local-first '
        'healthcare NLP">\n'
        '  <link rel="icon" href="../../assets/openmed-favicon.svg">\n'
        '  <link rel="stylesheet" href="../../stylesheets/openmed-system.css">\n'
        '  <link rel="stylesheet" '
        'href="../../stylesheets/openmed-standalone.css">\n'
        "  <title>OpenMed Public Benchmark Leaderboard</title>\n"
        "</head>\n"
        "<body>\n"
        '<a class="om-skip-link" href="#main">Skip to content</a>\n'
        '<header class="om-site-header"><div class="om-site-header__inner">'
        '<a class="om-brand" href="/docs/" '
        'aria-label="OpenMed documentation">'
        '<img src="../../assets/openmed-mark.svg" alt="" width="28" height="28">'
        "<span>openmed</span>"
        '<span class="om-brand__product">benchmark evidence</span></a>'
        '<nav class="om-site-nav" aria-label="Page links">'
        '<a href="/docs/">Documentation</a>'
        '<a href="/docs/eval/leaderboard/">Leaderboard contract</a>'
        '<a href="https://github.com/maziyarpanahi/openmed/tree/master/'
        'docs/benchmarks">Reports</a>'
        '<button class="om-theme-toggle" type="button" '
        "data-openmed-theme>Theme: light</button>"
        "</nav></div></header>\n"
        '<main id="main" class="om-page">\n'
        '<section class="om-hero" aria-labelledby="page-title">'
        '<p class="om-eyebrow">Public synthetic evidence</p>'
        '<h1 id="page-title">OpenMed Public Benchmark Leaderboard</h1>'
        '<p class="om-lede">Ranked leakage ascending, then recall descending. '
        "Every row links to its archived synthetic report.</p>"
        '<div class="om-actions">'
        '<a class="om-link-button" href="leaderboard.json" download>'
        "Download leaderboard.json</a>"
        "</div></section>\n"
        '<div class="om-grid"><div class="om-card om-card--full" role="search" '
        'aria-label="Filter benchmark results" data-leaderboard-filter hidden>'
        '<label for="leaderboard-filter">Filter benchmark results</label>'
        '<input id="leaderboard-filter" type="search" autocomplete="off" '
        'spellcheck="false" aria-controls="leaderboard-results" '
        'aria-describedby="leaderboard-filter-status" '
        'placeholder="Model, family, device, release, date, or hash">'
        '<p id="leaderboard-filter-status" class="status" role="status" '
        'aria-live="polite" aria-atomic="true">'
        f"Showing {len(ranked_rows)} of {len(ranked_rows)} benchmark "
        f"{'row' if len(ranked_rows) == 1 else 'rows'}.</p>"
        "</div></div>\n"
        '<section id="leaderboard-results" aria-label="Benchmark results">\n'
        f'<div class="tabs" role="tablist" aria-label="Benchmark suites">{"".join(tab_buttons)}</div>\n'
        f"{''.join(panels)}\n"
        "</section>\n"
        "</main>\n"
        '<footer class="om-site-footer"><div class="om-site-footer__inner">'
        "<span>OpenMed documentation</span>"
        "<span>Synthetic reports · deterministic publication</span>"
        "</div></footer>\n"
        '<script src="../../javascripts/openmed-standalone.js"></script>\n'
        "<script>\n"
        "  (() => {\n"
        "    const tabs = Array.from(document.querySelectorAll('[role=\"tab\"]'));\n"
        "    const panels = Array.from(document.querySelectorAll('[role=\"tabpanel\"]'));\n"
        "    const filterRegion = document.querySelector('[data-leaderboard-filter]');\n"
        "    const filter = document.getElementById('leaderboard-filter');\n"
        "    const filterStatus = document.getElementById('leaderboard-filter-status');\n"
        "    const rows = Array.from(document.querySelectorAll('[data-leaderboard-row]'));\n"
        "    const groups = Array.from(document.querySelectorAll('[data-leaderboard-group]'));\n"
        "    const normalize = (value) => value.trim().replace(/\\s+/g, ' ').toLowerCase();\n"
        "    const activate = (tab, moveFocus = false) => {\n"
        "      tabs.forEach((item) => {\n"
        "        const selected = item === tab;\n"
        "        item.setAttribute('aria-selected', String(selected));\n"
        "        item.tabIndex = selected ? 0 : -1;\n"
        "      });\n"
        "      panels.forEach((panel) => {\n"
        "        panel.hidden = panel.id !== tab.dataset.target;\n"
        "      });\n"
        "      if (moveFocus) tab.focus();\n"
        "    };\n"
        "    tabs.forEach((tab) => {\n"
        "      tab.addEventListener('click', () => activate(tab));\n"
        "      tab.addEventListener('keydown', (event) => {\n"
        "        const current = tabs.indexOf(tab);\n"
        "        let next = current;\n"
        "        if (event.key === 'ArrowRight') next = (current + 1) % tabs.length;\n"
        "        if (event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;\n"
        "        if (event.key === 'Home') next = 0;\n"
        "        if (event.key === 'End') next = tabs.length - 1;\n"
        "        if (next === current) return;\n"
        "        event.preventDefault();\n"
        "        activate(tabs[next], true);\n"
        "      });\n"
        "    });\n"
        "    const applyFilter = () => {\n"
        "      const rawQuery = filter.value.trim();\n"
        "      const query = normalize(rawQuery);\n"
        "      let matchCount = 0;\n"
        "      rows.forEach((row) => {\n"
        "        const matches = !query || normalize(row.dataset.search).includes(query);\n"
        "        row.hidden = !matches;\n"
        "        if (matches) matchCount += 1;\n"
        "      });\n"
        "      groups.forEach((group) => {\n"
        "        group.hidden = !group.querySelector('[data-leaderboard-row]:not([hidden])');\n"
        "      });\n"
        "      panels.forEach((panel) => {\n"
        "        const count = panel.querySelectorAll('[data-leaderboard-row]:not([hidden])').length;\n"
        "        panel.dataset.matchCount = String(count);\n"
        "        const tab = document.getElementById(`${panel.id}-tab`);\n"
        "        tab.querySelector('[data-leaderboard-tab-count]').textContent = `(${count})`;\n"
        "        const noun = count === 1 ? 'row' : 'rows';\n"
        "        tab.setAttribute('aria-label', `${tab.dataset.label}, ${count} matching benchmark ${noun}`);\n"
        "      });\n"
        "      const selected = tabs.find((tab) => tab.getAttribute('aria-selected') === 'true');\n"
        "      if (query && matchCount > 0 && selected) {\n"
        "        const selectedPanel = document.getElementById(selected.dataset.target);\n"
        "        if (Number(selectedPanel.dataset.matchCount) === 0) {\n"
        "          const matchingPanel = panels.find((panel) => Number(panel.dataset.matchCount) > 0);\n"
        "          activate(document.getElementById(`${matchingPanel.id}-tab`));\n"
        "        }\n"
        "      }\n"
        "      if (!query) {\n"
        "        const noun = rows.length === 1 ? 'row' : 'rows';\n"
        "        filterStatus.textContent = `Showing ${rows.length} of ${rows.length} benchmark ${noun}.`;\n"
        "      } else if (matchCount === 0) {\n"
        "        filterStatus.textContent = `No benchmark rows match “${rawQuery}”.`;\n"
        "      } else {\n"
        "        const noun = rows.length === 1 ? 'row' : 'rows';\n"
        "        filterStatus.textContent = `Showing ${matchCount} of ${rows.length} benchmark ${noun} for “${rawQuery}”.`;\n"
        "      }\n"
        "    };\n"
        "    filter.addEventListener('input', applyFilter);\n"
        "    filter.addEventListener('keydown', (event) => {\n"
        "      if (event.key !== 'Escape' || !filter.value) return;\n"
        "      event.preventDefault();\n"
        "      filter.value = '';\n"
        "      applyFilter();\n"
        "    });\n"
        "    filterRegion.hidden = false;\n"
        "    applyFilter();\n"
        "  })();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_leaderboard(
    reports_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    manifest_rows: Iterable[Mapping[str, Any]] = (),
    release_tag: str | None = None,
) -> LeaderboardArtifacts:
    """Render HTML, JSON, and downloadable reports into *output_dir*.

    Args:
        reports_dir: Directory recursively containing archived report JSON.
        output_dir: Destination below the documentation tree.
        manifest_rows: Optional model manifest rows used for metadata fallback.
        release_tag: Fallback release tag for legacy reports.

    Returns:
        Paths of the generated HTML, JSON, and report copies.
    """

    archives = _load_archived_reports(reports_dir)
    manifests = {
        str(row.get("repo_id")): row
        for row in manifest_rows
        if row.get("repo_id") is not None
    }
    rows = sorted(
        (
            _row_from_archive(
                archive,
                manifest_row=manifests.get(archive.report.model_name),
                release_tag=release_tag,
            )
            for archive in archives
        ),
        key=_row_sort_key,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reports_output = destination / "reports"
    _clear_generated_reports(reports_output)

    copied_reports: list[Path] = []
    for archive in archives:
        report_output = reports_output / archive.relative_path
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(_json_text(archive.payload), encoding="utf-8")
        copied_reports.append(report_output)

    html_path = destination / "index.html"
    json_path = destination / "leaderboard.json"
    html_path.write_text(render_leaderboard_html(rows), encoding="utf-8")
    json_path.write_text(render_leaderboard_json(rows), encoding="utf-8")
    return LeaderboardArtifacts(
        html_path=html_path,
        json_path=json_path,
        report_paths=tuple(copied_reports),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the public leaderboard command-line parser."""

    parser = argparse.ArgumentParser(
        description="Render a deterministic public leaderboard from archived reports."
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--release-tag",
        help="Fallback release tag for archived reports that do not contain one.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render the public leaderboard from command-line arguments."""

    args = build_arg_parser().parse_args(argv)
    manifest_rows = load_manifest_rows(args.manifest) if args.manifest.is_file() else []
    artifacts = write_leaderboard(
        args.reports_dir,
        args.output_dir,
        manifest_rows=manifest_rows,
        release_tag=args.release_tag,
    )
    print(artifacts.html_path)
    print(artifacts.json_path)
    return 0


def _load_archived_reports(reports_dir: str | Path) -> list[_ArchivedReport]:
    root = Path(reports_dir)
    if not root.is_dir():
        raise LeaderboardError(f"Archived report directory does not exist: {root}")

    paths = sorted(
        (
            path
            for path in root.rglob("*.json")
            if path.name != "leaderboard.json" and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise LeaderboardError(f"No archived report JSON found in {root}")

    archives: list[_ArchivedReport] = []
    for path in paths:
        relative_path = path.relative_to(root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LeaderboardError(
                f"Could not read archived report {path}: {exc}"
            ) from exc
        is_named_report = path.name.endswith(".report.json")
        if not isinstance(payload, Mapping):
            if is_named_report:
                raise LeaderboardError(f"Archived report must be a JSON object: {path}")
            continue
        if not is_named_report and not _REPORT_DISCRIMINATORS.intersection(payload):
            continue
        source = relative_path.as_posix()
        _validate_report_payload(payload, source=source)
        try:
            report = BenchmarkReport.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise LeaderboardError(f"Invalid BenchmarkReport {path}: {exc}") from exc
        archives.append(
            _ArchivedReport(
                path=path,
                relative_path=relative_path,
                payload=payload,
                report=report,
            )
        )
    if not archives:
        raise LeaderboardError(f"No archived BenchmarkReport JSON found in {root}")
    return archives


def _validate_report_payload(payload: Mapping[str, Any], *, source: str) -> None:
    for field in ("suite", "model_name", "device"):
        _required_text(payload.get(field), field=field, source=source)

    fixture_count = payload.get("fixture_count")
    if (
        isinstance(fixture_count, bool)
        or not isinstance(fixture_count, int)
        or fixture_count <= 0
    ):
        raise LeaderboardError(
            f"Report {source} requires a positive integer fixture_count"
        )
    if not isinstance(payload.get("metrics"), Mapping):
        raise LeaderboardError(f"Report {source} requires a metrics object")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise LeaderboardError(f"Report {source} requires a metadata object")
    generated_at = payload.get("generated_at")
    if generated_at is not None and not isinstance(generated_at, str):
        raise LeaderboardError(f"Report {source} has a non-string generated_at")


def _row_from_archive(
    archive: _ArchivedReport,
    *,
    manifest_row: Mapping[str, Any] | None,
    release_tag: str | None,
) -> LeaderboardRow:
    report = archive.report
    source = archive.relative_path.as_posix()
    metadata = report.metadata
    if metadata.get("synthetic") is not True:
        raise LeaderboardError(f"Report {source} is not explicitly marked as synthetic")
    family = metadata.get("model_family") or metadata.get("family")
    if family is None and manifest_row is not None:
        family = manifest_row.get("family")
    family_text = _required_text(
        family,
        field="model family metadata",
        source=source,
    )

    row_release = metadata.get("release_tag") or metadata.get("release") or release_tag
    release_text = _required_text(
        row_release,
        field="release tag",
        source=source,
    )

    run_date = metadata.get("run_date") or report.generated_at
    normalized_run_date = _normalize_run_date(run_date, source=source)

    repro_hash = metadata.get("reproducibility_hash") or metadata.get("repro_hash")
    if repro_hash is None and manifest_row is not None:
        repro_hash = manifest_row.get("reproducibility_hash")
    if not isinstance(repro_hash, str) or not _SHA256_RE.fullmatch(repro_hash):
        raise LeaderboardError(
            f"Report {source} has no valid sha256 reproducibility hash"
        )

    return LeaderboardRow(
        suite=_required_text(report.suite, field="suite", source=source),
        model_family=family_text,
        model_name=_required_text(
            report.model_name,
            field="model_name",
            source=source,
        ),
        device=_required_text(report.device, field="device", source=source),
        leakage=_metric(report.metrics, _LEAKAGE_METRICS, "leakage", source),
        recall=_metric(report.metrics, _RECALL_METRICS, "recall", source),
        f1=_metric(report.metrics, _F1_METRICS, "F1", source),
        release_tag=release_text,
        run_date=normalized_run_date,
        reproducibility_hash=repro_hash,
        report_path=source,
        report_url=f"reports/{quote(source)}",
    )


def _metric(
    metrics: Mapping[str, Any],
    candidates: Sequence[str],
    label: str,
    source: str,
) -> float:
    for key in candidates:
        value = _nested_get(metrics, key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LeaderboardError(f"Report {source} has non-numeric {label}: {key}")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise LeaderboardError(
                f"Report {source} has {label} outside [0, 1]: {key}={value!r}"
            )
        return number
    raise LeaderboardError(
        f"Report {source} is missing {label}; expected one of {', '.join(candidates)}"
    )


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _normalize_run_date(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeaderboardError(f"Report {source} has no run date")
    candidate = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            return date.fromisoformat(candidate).isoformat()
        normalized = f"{candidate[:-1]}+00:00" if candidate.endswith("Z") else candidate
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError as exc:
        raise LeaderboardError(
            f"Report {source} has invalid run date: {value!r}"
        ) from exc


def _required_text(value: Any, *, field: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeaderboardError(f"Report {source} has no valid {field}")
    return value.strip()


def _row_sort_key(row: LeaderboardRow) -> tuple[Any, ...]:
    return (
        row.leakage,
        -row.recall,
        row.suite.casefold(),
        row.model_family.casefold(),
        row.model_name.casefold(),
        row.device.casefold(),
        row.release_tag.casefold(),
        row.report_path,
    )


def _group_rows(
    rows: Iterable[LeaderboardRow],
) -> dict[str, dict[str, list[LeaderboardRow]]]:
    grouped: dict[str, dict[str, list[LeaderboardRow]]] = {}
    for row in rows:
        grouped.setdefault(row.suite, {}).setdefault(row.model_family, []).append(row)
    return {
        suite: {
            family: sorted(family_rows, key=_row_sort_key)
            for family, family_rows in sorted(
                families.items(), key=lambda item: item[0].casefold()
            )
        }
        for suite, families in sorted(
            grouped.items(), key=lambda item: item[0].casefold()
        )
    }


def _suite_payload(rows: Iterable[LeaderboardRow]) -> list[dict[str, Any]]:
    return [
        {
            "families": [
                {
                    "name": family,
                    "rows": [row.to_dict() for row in family_rows],
                }
                for family, family_rows in families.items()
            ],
            "name": suite,
        }
        for suite, families in _group_rows(rows).items()
    ]


def _render_family_table(family: str, rows: Sequence[LeaderboardRow]) -> str:
    body: list[str] = []
    for rank, row in enumerate(rows, start=1):
        search_text = _row_search_text(row)
        body.append(
            "<tr data-leaderboard-row "
            f'data-search="{html.escape(search_text)}">'
            f'<td class="numeric">{rank}</td>'
            f'<th scope="row">{html.escape(row.model_name)}</th>'
            f"<td>{html.escape(row.device)}</td>"
            f'<td class="numeric">{_format_percent(row.leakage)}</td>'
            f'<td class="numeric">{_format_percent(row.recall)}</td>'
            f'<td class="numeric">{_format_percent(row.f1)}</td>'
            f"<td><code>{html.escape(row.release_tag)}</code></td>"
            f'<td><time datetime="{html.escape(row.run_date)}">'
            f"{html.escape(row.run_date)}</time></td>"
            f"<td><code>{html.escape(row.reproducibility_hash)}</code></td>"
            f'<td><a href="{html.escape(row.report_url)}" download>Download JSON</a></td>'
            "</tr>"
        )
    return (
        "<section data-leaderboard-group "
        f'aria-label="{html.escape(family)} benchmark results">'
        f"<h2>{html.escape(family)}</h2>"
        '<div class="table-wrap"><table>'
        f'<caption class="om-sr-only">{html.escape(family)} benchmark '
        "results</caption>"
        '<thead><tr><th class="numeric" scope="col">Rank</th>'
        '<th scope="col">Model</th><th scope="col">Device</th>'
        '<th class="numeric" scope="col">Leakage</th>'
        '<th class="numeric" scope="col">Recall</th>'
        '<th class="numeric" scope="col">F1</th>'
        '<th scope="col">Release</th><th scope="col">Run date</th>'
        '<th scope="col">Reproducibility hash</th>'
        '<th scope="col">Report</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></section>"
    )


def _row_search_text(row: LeaderboardRow) -> str:
    values = (
        row.suite,
        row.model_family,
        row.model_name,
        row.device,
        _format_percent(row.leakage),
        _format_percent(row.recall),
        _format_percent(row.f1),
        row.release_tag,
        row.run_date,
        row.reproducibility_hash,
        row.report_path,
    )
    return " ".join(" ".join(values).split())


def _format_percent(value: float) -> str:
    return f"{value:.4%}"


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _clear_generated_reports(reports_dir: Path) -> None:
    if not reports_dir.exists():
        return
    if not reports_dir.is_dir() or reports_dir.is_symlink():
        raise LeaderboardError(
            f"Generated reports path is not a directory: {reports_dir}"
        )
    paths = sorted(
        reports_dir.rglob("*"), key=lambda path: len(path.parts), reverse=True
    )
    for path in paths:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORTS_DIR",
    "LEADERBOARD_SCHEMA_VERSION",
    "LeaderboardArtifacts",
    "LeaderboardError",
    "LeaderboardRow",
    "load_leaderboard_rows",
    "render_leaderboard_html",
    "render_leaderboard_json",
    "write_leaderboard",
]


if __name__ == "__main__":
    raise SystemExit(main())
