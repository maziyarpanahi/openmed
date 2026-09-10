"""Coverage and consistency gates for synthetic clinical-domain fixtures.

The gate checks the fixture-backed clinical domains shipped with the zero-shot
label maps.  Its evidence is deliberately leakage-first: only domain names,
labels, offsets, and counts leave this module.  Fixture text and row metadata
are never copied into reports or failure messages.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.labels import CANONICAL_LABELS, OTHER, normalize_label
from openmed.ner.labels import load_default_label_map

CLINICAL_DOMAIN_COVERAGE = "clinical_domain_coverage"
DOMAIN_COVERAGE = CLINICAL_DOMAIN_COVERAGE
SCHEMA_VERSION = 1

DEFAULT_LABEL_MAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "zero_shot"
    / "data"
    / "label_maps"
    / "defaults.json"
)
DEFAULT_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "clinical"
)

# These are the granular clinical maps with committed span fixtures.  The
# remaining packaged maps are legacy or explicitly marked "Not shipped" in
# docs/clinical-domains.md and are not part of this clinical fixture gate.
CLINICAL_DOMAIN_FIXTURE_NAMES: Mapping[str, str] = {
    "anesthesia": "anesthesia.jsonl",
    "endocrinology": "endocrinology.jsonl",
    "gastroenterology": "gastroenterology.jsonl",
    "genomic_variant": "genomic_variant.jsonl",
    "immunization": "immunization.jsonl",
    "nephrology_renal": "nephrology_renal.jsonl",
    "nursing_observation": "nursing_observation.jsonl",
    "nutrition_diet": "nutrition_diet.jsonl",
    "pediatrics_growth": "pediatrics_growth.jsonl",
    "pulmonology": "pulmonology.jsonl",
    "radiology": "radiology.jsonl",
}


@dataclass(frozen=True)
class CoverageOffset:
    """One validated, text-free fixture offset."""

    line_number: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the span length without exposing its surface text."""

        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        """Return the machine-readable offset evidence."""

        return {
            "line": self.line_number,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


@dataclass(frozen=True)
class LabelCoverage:
    """Aggregate span coverage for one expected display label."""

    label: str
    canonical_label: str
    span_count: int
    fixture_count: int
    offsets: tuple[CoverageOffset, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return counts and offsets without fixture text."""

        return {
            "label": self.label,
            "canonical_label": self.canonical_label,
            "span_count": self.span_count,
            "fixture_count": self.fixture_count,
            "offsets": [offset.to_dict() for offset in self.offsets],
        }


@dataclass(frozen=True)
class CoverageIssue:
    """A text-free orphan-label or malformed-span finding."""

    domain: str
    label: str
    line_number: int
    reason: str
    canonical_label: str = OTHER
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return stable issue evidence without raw fixture content."""

        payload: dict[str, Any] = {
            "domain": self.domain,
            "label": self.label,
            "line": self.line_number,
            "reason": self.reason,
            "canonical_label": self.canonical_label,
        }
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        return payload


@dataclass(frozen=True)
class CoverageError:
    """A text-free fixture loading error."""

    domain: str
    line_number: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return stable loading-error evidence."""

        return {
            "domain": self.domain,
            "line": self.line_number,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DomainCoverage:
    """Coverage report for one fixture-backed clinical domain."""

    domain: str
    fixture: str
    missing_fixture: bool
    fixture_count: int
    span_count: int
    per_label: tuple[LabelCoverage, ...]
    missing_labels: tuple[str, ...]
    orphan_labels: tuple[CoverageIssue, ...]
    invalid_spans: tuple[CoverageIssue, ...]
    errors: tuple[CoverageError, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether this domain satisfies the coverage gate."""

        return not (
            self.missing_fixture
            or self.missing_labels
            or self.orphan_labels
            or self.invalid_spans
            or self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a text-free per-domain report."""

        return {
            "domain": self.domain,
            "fixture": self.fixture,
            "missing_fixture": self.missing_fixture,
            "fixture_count": self.fixture_count,
            "span_count": self.span_count,
            "per_label": [coverage.to_dict() for coverage in self.per_label],
            "missing_labels": list(self.missing_labels),
            "orphan_labels": [issue.to_dict() for issue in self.orphan_labels],
            "invalid_spans": [issue.to_dict() for issue in self.invalid_spans],
            "errors": [error.to_dict() for error in self.errors],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DomainCoverageReport:
    """Aggregate clinical-domain coverage and gate evidence."""

    per_domain: tuple[DomainCoverage, ...]
    missing_fixtures: tuple[str, ...] = ()
    errors: tuple[CoverageError, ...] = ()

    @property
    def suite(self) -> str:
        """Return the registered suite identifier."""

        return CLINICAL_DOMAIN_COVERAGE

    @property
    def fixture_count(self) -> int:
        """Return the total number of fixture rows inspected."""

        return sum(domain.fixture_count for domain in self.per_domain)

    @property
    def span_count(self) -> int:
        """Return the total number of annotated spans inspected."""

        return sum(domain.span_count for domain in self.per_domain)

    @property
    def orphan_labels(self) -> tuple[CoverageIssue, ...]:
        """Return all orphan-label findings in stable domain order."""

        return tuple(
            issue for domain in self.per_domain for issue in domain.orphan_labels
        )

    @property
    def missing_labels(self) -> tuple[dict[str, str], ...]:
        """Return all expected labels with no valid fixture span."""

        return tuple(
            {"domain": domain.domain, "label": label}
            for domain in self.per_domain
            for label in domain.missing_labels
        )

    @property
    def invalid_spans(self) -> tuple[CoverageIssue, ...]:
        """Return all malformed or out-of-bounds span findings."""

        return tuple(
            issue for domain in self.per_domain for issue in domain.invalid_spans
        )

    @property
    def passed(self) -> bool:
        """Return the aggregate gate verdict."""

        return not (
            self.missing_fixtures
            or self.errors
            or any(not domain.passed for domain in self.per_domain)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable, raw-text-free summary."""

        return {
            "suite": self.suite,
            "schema_version": SCHEMA_VERSION,
            "passed": self.passed,
            "summary": {
                "domain_count": len(self.per_domain),
                "fixture_count": self.fixture_count,
                "span_count": self.span_count,
                "missing_fixture_count": len(self.missing_fixtures),
                "missing_label_count": len(self.missing_labels),
                "orphan_label_count": len(self.orphan_labels),
                "invalid_span_count": len(self.invalid_spans),
                "error_count": len(self.errors),
            },
            "missing_fixtures": list(self.missing_fixtures),
            "missing_labels": list(self.missing_labels),
            "orphan_labels": [issue.to_dict() for issue in self.orphan_labels],
            "invalid_spans": [issue.to_dict() for issue in self.invalid_spans],
            "errors": [error.to_dict() for error in self.errors],
            "per_domain": [domain.to_dict() for domain in self.per_domain],
        }

    def to_json(self) -> str:
        """Serialize the report deterministically as JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        """Render a leakage-first Markdown report."""

        lines = [
            "# Clinical Domain Coverage",
            "",
            f"**Status:** {'PASS' if self.passed else 'FAIL'}",
            "",
            "| Domain | Fixtures | Spans | Missing Labels | Orphans | Status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for domain in self.per_domain:
            lines.append(
                f"| {_markdown_cell(domain.domain)} | {domain.fixture_count} | "
                f"{domain.span_count} | {len(domain.missing_labels)} | "
                f"{len(domain.orphan_labels)} | "
                f"{'PASS' if domain.passed else 'FAIL'} |"
            )

        lines.extend(
            [
                "",
                "## Per-label offsets",
                "",
                "| Domain | Label | Canonical | Spans | Fixtures | Offsets |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for domain in self.per_domain:
            for coverage in domain.per_label:
                offsets = (
                    ", ".join(
                        f"{offset.line_number}:{offset.start}-{offset.end}"
                        for offset in coverage.offsets
                    )
                    or "—"
                )
                lines.append(
                    f"| {_markdown_cell(domain.domain)} | "
                    f"{_markdown_cell(coverage.label)} | "
                    f"{_markdown_cell(coverage.canonical_label)} | "
                    f"{coverage.span_count} | {coverage.fixture_count} | {offsets} |"
                )

        lines.extend(["", "## Gate findings", ""])
        if self.missing_fixtures:
            lines.append(
                "- Missing fixtures: "
                + ", ".join(_markdown_cell(value) for value in self.missing_fixtures)
            )
        if self.missing_labels:
            lines.append(
                "- Missing labels: "
                + ", ".join(
                    f"{item['domain']}/{item['label']}" for item in self.missing_labels
                )
            )
        if self.orphan_labels:
            lines.append(
                "- Orphan labels: "
                + ", ".join(
                    f"{issue.domain}/{issue.label}@{issue.line_number}"
                    for issue in self.orphan_labels
                )
            )
        if self.invalid_spans:
            lines.append(f"- Invalid spans: {len(self.invalid_spans)}")
        if self.errors:
            lines.append(
                "- Fixture errors: "
                + ", ".join(
                    f"{error.domain}@{error.line_number}: {error.reason}"
                    for error in self.errors
                )
            )
        if not (
            self.missing_fixtures
            or self.missing_labels
            or self.orphan_labels
            or self.invalid_spans
            or self.errors
        ):
            lines.append("- None")
        return "\n".join(lines) + "\n"

    def write_json(self, path: str | Path) -> Path:
        """Write the machine-readable report to *path*."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report to *path*."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_markdown(), encoding="utf-8")
        return output


def run_domain_coverage(
    *,
    label_map: Mapping[str, Sequence[str]] | None = None,
    label_map_path: str | Path | None = None,
    fixture_dir: str | Path | None = None,
    domains: Sequence[str] | None = None,
    fixture_names: Mapping[str, str] | None = None,
) -> DomainCoverageReport:
    """Run the offline clinical-domain label and span coverage gate.

    The default run covers the fixture-backed clinical domains registered in
    :data:`CLINICAL_DOMAIN_FIXTURE_NAMES`.  Supplying a label map or fixture
    directory makes the selected map domains the default scope; callers can
    always provide ``domains`` explicitly to exercise missing-fixture cases.

    Args:
        label_map: Optional in-memory domain-to-display-label mapping.
        label_map_path: Optional path to a JSON label map.
        fixture_dir: Directory containing one JSONL fixture per selected domain.
        domains: Optional domain identifiers to inspect.
        fixture_names: Optional domain-to-filename overrides for test fixtures.
    """

    if label_map is not None and label_map_path is not None:
        raise ValueError("label_map and label_map_path are mutually exclusive")
    if label_map_path is not None:
        active_map = load_default_label_map(Path(label_map_path))
    elif label_map is not None:
        active_map = _normalize_label_map(label_map)
    else:
        active_map = load_default_label_map(DEFAULT_LABEL_MAP_PATH)

    active_fixture_dir = (
        Path(fixture_dir) if fixture_dir is not None else DEFAULT_FIXTURE_DIR
    )
    active_fixture_names = dict(fixture_names or CLINICAL_DOMAIN_FIXTURE_NAMES)
    if domains is None:
        if (
            label_map is not None
            or label_map_path is not None
            or fixture_dir is not None
        ):
            selected_domains = tuple(sorted(active_map))
        else:
            selected_domains = tuple(sorted(active_fixture_names))
    else:
        selected_domains = tuple(
            sorted(
                {_normalize_domain(domain) for domain in domains if str(domain).strip()}
            )
        )

    all_display_labels = {
        _label_key(label) for labels in active_map.values() for label in labels
    }
    domain_reports: list[DomainCoverage] = []
    missing_fixtures: list[str] = []
    loading_errors: list[CoverageError] = []
    for domain in selected_domains:
        expected_labels = tuple(active_map.get(domain, ()))
        filename = active_fixture_names.get(domain, f"{domain}.jsonl")
        path = Path(filename)
        if not path.is_absolute():
            path = active_fixture_dir / path
        if domain not in active_map:
            loading_errors.append(
                CoverageError(domain, 0, "domain_missing_from_label_map")
            )
        report, missing, errors = _load_domain_fixture(
            domain,
            path,
            expected_labels,
            all_display_labels,
        )
        domain_reports.append(report)
        if missing:
            missing_fixtures.append(domain)
        loading_errors.extend(errors)

    return DomainCoverageReport(
        per_domain=tuple(domain_reports),
        missing_fixtures=tuple(sorted(missing_fixtures)),
        errors=tuple(loading_errors),
    )


def run_clinical_domain_coverage(**kwargs: Any) -> DomainCoverageReport:
    """Alias for :func:`run_domain_coverage` with explicit clinical naming."""

    return run_domain_coverage(**kwargs)


def assert_domain_coverage_gate(**kwargs: Any) -> DomainCoverageReport:
    """Run the gate and raise a text-free assertion on any failure."""

    report = run_domain_coverage(**kwargs)
    if not report.passed:
        raise AssertionError(_failure_message(report))
    return report


def assert_clinical_domain_coverage_gate(**kwargs: Any) -> DomainCoverageReport:
    """Alias for :func:`assert_domain_coverage_gate`."""

    return assert_domain_coverage_gate(**kwargs)


def domain_coverage_metadata() -> dict[str, Any]:
    """Return static, raw-text-free metadata for the registered suite."""

    return {
        "suite": CLINICAL_DOMAIN_COVERAGE,
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "offline": True,
        "domains": list(sorted(CLINICAL_DOMAIN_FIXTURE_NAMES)),
    }


def _load_domain_fixture(
    domain: str,
    path: Path,
    expected_labels: Sequence[str],
    all_display_labels: set[str],
) -> tuple[DomainCoverage, bool, tuple[CoverageError, ...]]:
    expected = tuple(
        str(label).strip() for label in expected_labels if str(label).strip()
    )
    fixture_name = path.name
    empty_offsets: dict[str, list[CoverageOffset]] = {label: [] for label in expected}
    if not path.is_file():
        return (
            DomainCoverage(
                domain=domain,
                fixture=fixture_name,
                missing_fixture=True,
                fixture_count=0,
                span_count=0,
                per_label=_label_coverages(expected, empty_offsets),
                missing_labels=(),
                orphan_labels=(),
                invalid_spans=(),
                errors=(),
            ),
            True,
            (),
        )

    fixture_count = 0
    span_count = 0
    observed: dict[str, list[CoverageOffset]] = {label: [] for label in expected}
    orphan_labels: list[CoverageIssue] = []
    invalid_spans: list[CoverageIssue] = []
    errors: list[CoverageError] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        error = CoverageError(domain, 0, "fixture_unreadable")
        return (
            DomainCoverage(
                domain=domain,
                fixture=fixture_name,
                missing_fixture=False,
                fixture_count=0,
                span_count=0,
                per_label=_label_coverages(expected, observed),
                missing_labels=expected,
                orphan_labels=(),
                invalid_spans=(),
                errors=(error,),
            ),
            False,
            (error,),
        )

    expected_keys = {_label_key(label) for label in expected}
    expected_canonical = {normalize_label(label) for label in expected}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(CoverageError(domain, line_number, "invalid_json"))
            continue
        if not isinstance(row, Mapping):
            errors.append(CoverageError(domain, line_number, "fixture_row_not_object"))
            continue
        fixture_count += 1
        source_text = row.get("text") if isinstance(row.get("text"), str) else ""
        raw_spans = _span_rows(row)
        for raw_span in raw_spans:
            span_count += 1
            if not isinstance(raw_span, Mapping):
                invalid_spans.append(
                    CoverageIssue(domain, "<invalid>", line_number, "span_not_object")
                )
                continue
            raw_label = raw_span.get("label")
            label = str(raw_label).strip() if raw_label is not None else ""
            canonical = normalize_label(label)
            start = _integer_offset(raw_span.get("start"))
            end = _integer_offset(raw_span.get("end"))
            if (
                start is None
                or end is None
                or start < 0
                or end <= start
                or end > len(source_text)
            ):
                invalid_spans.append(
                    CoverageIssue(
                        domain,
                        label or "<missing>",
                        line_number,
                        "invalid_offsets",
                        canonical,
                        start,
                        end,
                    )
                )
                continue
            surface = raw_span.get("text")
            if isinstance(surface, str) and source_text[start:end] != surface:
                invalid_spans.append(
                    CoverageIssue(
                        domain,
                        label or "<missing>",
                        line_number,
                        "surface_offset_mismatch",
                        canonical,
                        start,
                        end,
                    )
                )
                continue

            if not _is_catalog_label(
                label,
                canonical,
                all_display_labels=all_display_labels,
                expected_keys=expected_keys,
                expected_canonical=expected_canonical,
            ):
                orphan_labels.append(
                    CoverageIssue(
                        domain,
                        label or "<missing>",
                        line_number,
                        "label_not_in_catalog",
                        canonical,
                        start,
                        end,
                    )
                )
                continue

            exact_matches = [
                expected_label
                for expected_label in expected
                if _label_key(label) == _label_key(expected_label)
            ]
            canonical_matches = [
                expected_label
                for expected_label in expected
                if canonical != OTHER and canonical == normalize_label(expected_label)
            ]
            matching_labels = exact_matches or canonical_matches
            if matching_labels:
                observed[matching_labels[0]].append(
                    CoverageOffset(line_number, start, end)
                )
            else:
                orphan_labels.append(
                    CoverageIssue(
                        domain,
                        label or "<missing>",
                        line_number,
                        "label_not_in_domain_map",
                        canonical,
                        start,
                        end,
                    )
                )

    per_label = _label_coverages(expected, observed)
    missing_labels = tuple(
        coverage.label for coverage in per_label if coverage.span_count == 0
    )
    report = DomainCoverage(
        domain=domain,
        fixture=fixture_name,
        missing_fixture=False,
        fixture_count=fixture_count,
        span_count=span_count,
        per_label=per_label,
        missing_labels=missing_labels,
        orphan_labels=tuple(orphan_labels),
        invalid_spans=tuple(invalid_spans),
        errors=tuple(errors),
    )
    return report, False, tuple(errors)


def _label_coverages(
    expected: Sequence[str],
    offsets: Mapping[str, Sequence[CoverageOffset]],
) -> tuple[LabelCoverage, ...]:
    return tuple(
        LabelCoverage(
            label=label,
            canonical_label=normalize_label(label),
            span_count=len(offsets.get(label, ())),
            fixture_count=len(
                {offset.line_number for offset in offsets.get(label, ())}
            ),
            offsets=tuple(offsets.get(label, ())),
        )
        for label in expected
    )


def _span_rows(row: Mapping[str, Any]) -> Sequence[Any]:
    for key in ("entities", "spans", "gold_spans"):
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return value
    return ()


def _is_catalog_label(
    label: str,
    canonical: str,
    *,
    all_display_labels: set[str],
    expected_keys: set[str],
    expected_canonical: set[str],
) -> bool:
    if not label or canonical not in CANONICAL_LABELS:
        return False
    key = _label_key(label)
    domain_match = key in expected_keys or (
        canonical != OTHER and canonical in expected_canonical
    )
    if not domain_match:
        return False
    # Several existing display labels intentionally normalize to OTHER.  They
    # are valid only when explicitly declared in a shipped label map; an
    # arbitrary unknown label must still fail the orphan gate.
    return canonical != OTHER or key == _label_key(OTHER) or key in all_display_labels


def _normalize_label_map(
    label_map: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_domain, labels in label_map.items():
        if isinstance(labels, (str, bytes, bytearray)):
            raise ValueError(f"label map domain {raw_domain!r} must contain labels")
        domain = _normalize_domain(raw_domain)
        normalized[domain] = tuple(
            str(label).strip() for label in labels if str(label).strip()
        )
    return normalized


def _normalize_domain(domain: object) -> str:
    return str(domain).strip().lower().replace(" ", "_")


def _label_key(label: object) -> str:
    return "".join(
        character for character in str(label).casefold() if character.isalnum()
    )


def _integer_offset(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _failure_message(report: DomainCoverageReport) -> str:
    findings: list[str] = []
    if report.missing_fixtures:
        findings.append("missing fixtures: " + ", ".join(report.missing_fixtures))
    if report.missing_labels:
        findings.append(
            "missing labels: "
            + ", ".join(
                f"{item['domain']}/{item['label']}" for item in report.missing_labels
            )
        )
    if report.orphan_labels:
        findings.append(
            "orphan labels: "
            + ", ".join(
                f"{issue.domain}/{issue.label} (line {issue.line_number})"
                for issue in report.orphan_labels
            )
        )
    if report.invalid_spans:
        findings.append(f"invalid spans: {len(report.invalid_spans)}")
    if report.errors:
        findings.append(
            "fixture errors: "
            + ", ".join(
                f"{error.domain}@{error.line_number}: {error.reason}"
                for error in report.errors
            )
        )
    return "Clinical domain coverage gate failed: " + "; ".join(findings)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-map", type=Path, default=None)
    parser.add_argument("--fixture-dir", type=Path, default=None)
    parser.add_argument("--domain", action="append", dest="domains", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone offline coverage gate."""

    args = _build_parser().parse_args(argv)
    report = run_domain_coverage(
        label_map_path=args.label_map,
        fixture_dir=args.fixture_dir,
        domains=args.domains,
    )
    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        report.write_markdown(args.markdown_output)
    print(report.to_json(), end="")
    if not report.passed:
        print(_failure_message(report), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the workflow
    raise SystemExit(main())


__all__ = [
    "CLINICAL_DOMAIN_COVERAGE",
    "CLINICAL_DOMAIN_FIXTURE_NAMES",
    "CoverageError",
    "CoverageIssue",
    "CoverageOffset",
    "DEFAULT_FIXTURE_DIR",
    "DEFAULT_LABEL_MAP_PATH",
    "DOMAIN_COVERAGE",
    "DomainCoverage",
    "DomainCoverageReport",
    "LabelCoverage",
    "SCHEMA_VERSION",
    "assert_clinical_domain_coverage_gate",
    "assert_domain_coverage_gate",
    "domain_coverage_metadata",
    "main",
    "run_clinical_domain_coverage",
    "run_domain_coverage",
]
