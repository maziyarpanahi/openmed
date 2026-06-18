"""Adversarial re-identification attack for PII benchmark fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.eval.golden import GoldenFixture, load_golden_fixtures
from openmed.eval.report import BenchmarkReport
from openmed.risk import risk_report


@dataclass(frozen=True)
class ReidAttackResult:
    """Leakage-style adversarial re-identification score."""

    rate: float
    numerator: int
    denominator: int
    risk: Mapping[str, Any]
    surrogate_findings: tuple[dict[str, Any], ...] = ()
    date_shift_findings: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metric(self) -> dict[str, Any]:
        return {
            "rate": float(self.rate),
            "numerator": int(self.numerator),
            "denominator": int(self.denominator),
            "leakage_rate": float(self.risk.get("leakage_rate", 0.0)),
            "aux_linkage_rate": float(self.risk.get("reid_rate", 0.0)),
            "surrogate_consistency_rate": _rate(
                len(self.surrogate_findings),
                self.denominator,
            ),
            "date_shift_inversion_rate": _rate(
                len(self.date_shift_findings),
                self.denominator,
            ),
            "k_min": int(self.risk.get("k_min", 0) or 0),
            "singleton_count": len(self.risk.get("singleton_records") or ()),
            "quasi_identifier_count": len(self.risk.get("quasi_identifiers") or ()),
            "surrogate_findings": [dict(item) for item in self.surrogate_findings],
            "date_shift_findings": [dict(item) for item in self.date_shift_findings],
            "metadata": dict(self.metadata),
        }


def run_reid_benchmark(
    *,
    suite: str = "golden",
    model_name: str = "privacy-filter",
    deidentified_records: Sequence[Mapping[str, Any]] | None = None,
    auxiliary_records: Sequence[Mapping[str, Any]] | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run the re-identification attack and return a BenchmarkReport."""

    fixtures = _load_suite_fixtures(suite)
    result = run_reid_attack(
        fixtures,
        deidentified_records=deidentified_records,
        auxiliary_records=auxiliary_records,
    )
    report = BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device="attack",
        fixture_count=result.denominator,
        generated_at=generated_at or _utc_now(),
        metrics={
            "reid_leakage": {
                "rate": result.rate,
                "numerator": result.numerator,
                "denominator": result.denominator,
            },
            "reidentification": result.to_metric(),
        },
        metadata={
            "attack": "reid",
            "leaderboard_metric": "reid_leakage_rate",
        },
    )
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        Path(output_markdown).write_text(
            render_reid_leaderboard([report]),
            encoding="utf-8",
        )
    return report


def run_reid_attack(
    fixtures: Sequence[GoldenFixture | Mapping[str, Any]],
    *,
    deidentified_records: Sequence[Mapping[str, Any]] | None = None,
    auxiliary_records: Sequence[Mapping[str, Any]] | None = None,
) -> ReidAttackResult:
    """Attempt re-identification against fixture originals and outputs."""

    normalized_fixtures = [_coerce_fixture(fixture) for fixture in fixtures]
    original_records = [_original_record(fixture) for fixture in normalized_fixtures]
    deidentified = (
        [dict(record) for record in deidentified_records]
        if deidentified_records is not None
        else [_deidentified_record(fixture) for fixture in normalized_fixtures]
    )
    aux = [dict(record) for record in (auxiliary_records or ())]
    risk = risk_report(deidentified, original=original_records, aux=aux)
    surrogate_findings = tuple(_surrogate_consistency_findings(deidentified))
    date_shift_findings = tuple(_date_shift_findings(deidentified))

    denominator = max(len(deidentified), len(normalized_fixtures), 1)
    leakage_successes = round(float(risk.get("leakage_rate", 0.0)) * denominator)
    linkage_successes = round(float(risk.get("reid_rate", 0.0)) * denominator)
    numerator = min(
        denominator,
        max(
            leakage_successes,
            linkage_successes,
            len(surrogate_findings),
            len(date_shift_findings),
        ),
    )
    rate = _rate(numerator, denominator)
    return ReidAttackResult(
        rate=rate,
        numerator=numerator,
        denominator=denominator,
        risk=risk,
        surrogate_findings=surrogate_findings,
        date_shift_findings=date_shift_findings,
        metadata={"fixture_ids": [fixture.fixture_id for fixture in normalized_fixtures]},
    )


def generate_reid_leaderboard(
    reports: Sequence[BenchmarkReport | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic leaderboard rows with the re-id score surfaced."""

    rows: list[dict[str, Any]] = []
    for report in reports:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        metrics = payload.get("metrics") or {}
        reid = metrics.get("reid_leakage") or metrics.get("reidentification") or {}
        rows.append(
            {
                "model_name": payload.get("model_name"),
                "suite": payload.get("suite"),
                "attack": (payload.get("metadata") or {}).get("attack", "reid"),
                "reid_leakage_rate": float(reid.get("rate", 0.0)),
                "reid_successes": int(reid.get("numerator", 0) or 0),
                "fixture_count": int(payload.get("fixture_count", 0) or 0),
            }
        )
    return sorted(
        rows,
        key=lambda row: (str(row["suite"]), str(row["model_name"]), str(row["attack"])),
    )


def render_reid_leaderboard(
    reports: Sequence[BenchmarkReport | Mapping[str, Any]],
    *,
    output_format: str = "markdown",
) -> str:
    """Render leaderboard rows as Markdown or JSON."""

    rows = generate_reid_leaderboard(reports)
    if output_format == "json":
        return json.dumps(rows, indent=2, sort_keys=True) + "\n"
    if output_format != "markdown":
        raise ValueError("output_format must be markdown or json")

    lines = [
        "| Model | Suite | Attack | reid_leakage_rate | Re-id Successes | Fixtures |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {suite} | {attack} | {rate:.6g} | {successes} | {count} |".format(
                model_name=row["model_name"],
                suite=row["suite"],
                attack=row["attack"],
                rate=row["reid_leakage_rate"],
                successes=row["reid_successes"],
                count=row["fixture_count"],
            )
        )
    return "\n".join(lines) + "\n"


def _load_suite_fixtures(suite: str) -> list[GoldenFixture]:
    if suite != "golden":
        raise ValueError("re-identification attack currently supports the golden suite")
    return load_golden_fixtures()


def _coerce_fixture(fixture: GoldenFixture | Mapping[str, Any]) -> GoldenFixture:
    if isinstance(fixture, GoldenFixture):
        return fixture
    return GoldenFixture.from_mapping(fixture)


def _original_record(fixture: GoldenFixture) -> dict[str, Any]:
    return {
        "record_id": fixture.fixture_id,
        "text": fixture.text,
        "entities": [
            {
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "text": span.text,
                "metadata": dict(span.metadata),
            }
            for span in fixture.gold_spans
        ],
        "metadata": dict(fixture.metadata),
    }


def _deidentified_record(fixture: GoldenFixture) -> dict[str, Any]:
    expected = dict(fixture.expected_output)
    return {
        "record_id": fixture.fixture_id,
        "text": str(expected.get("text", "")),
        "metadata": {
            "category": fixture.metadata.get("category"),
            "method": expected.get("method"),
        },
    }


def _surrogate_consistency_findings(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_original: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_surrogate: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        record_id = _record_id(record)
        for span in _iter_audit_spans(record):
            label = str(
                span.get("canonical_label")
                or span.get("label")
                or span.get("entity_type")
                or ""
            )
            surrogate = span.get("surrogate") or span.get("replacement")
            original_hash = (
                span.get("original_hash")
                or span.get("text_hash")
                or (span.get("evidence") or {}).get("text_hash")
            )
            if not label or surrogate is None or original_hash is None:
                continue
            by_original[(record_id, f"{label}:{original_hash}")].add(str(surrogate))
            by_surrogate[(record_id, f"{label}:{surrogate}")].add(str(original_hash))

    findings: list[dict[str, Any]] = []
    for (record_id, key), surrogates in sorted(by_original.items()):
        if len(surrogates) > 1:
            findings.append(
                {
                    "record_id": record_id,
                    "type": "one_original_multiple_surrogates",
                    "key": key,
                    "surrogate_count": len(surrogates),
                }
            )
    for (record_id, key), originals in sorted(by_surrogate.items()):
        if len(originals) > 1:
            findings.append(
                {
                    "record_id": record_id,
                    "type": "one_surrogate_multiple_originals",
                    "key": key,
                    "original_count": len(originals),
                }
            )
    return findings


def _date_shift_findings(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        date_chain = metadata.get("date_chain")
        if not isinstance(date_chain, Mapping):
            continue
        original_dates = date_chain.get("original_dates") or ()
        shifted_dates = date_chain.get("shifted_dates") or ()
        if len(original_dates) < 2 or len(original_dates) != len(shifted_dates):
            continue
        original_intervals = _intervals(original_dates)
        shifted_intervals = _intervals(shifted_dates)
        if original_intervals and original_intervals == shifted_intervals:
            findings.append(
                {
                    "record_id": _record_id(record),
                    "type": "preserved_date_intervals",
                    "interval_days": original_intervals,
                }
            )
    return findings


def _iter_audit_spans(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    spans: list[Mapping[str, Any]] = []
    for key in ("audit_spans", "spans", "entities"):
        value = record.get(key)
        if isinstance(value, Mapping):
            spans.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            spans.extend(item for item in value if isinstance(item, Mapping))

    audit = record.get("audit") or record.get("audit_report")
    if hasattr(audit, "to_dict"):
        audit = audit.to_dict()
    if isinstance(audit, Mapping):
        value = audit.get("spans")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            spans.extend(item for item in value if isinstance(item, Mapping))
    return spans


def _record_id(record: Mapping[str, Any]) -> str:
    return str(
        record.get("record_id")
        or record.get("doc_id")
        or record.get("id")
        or "record"
    )


def _intervals(values: Sequence[Any]) -> list[int]:
    dates: list[datetime] = []
    for value in values:
        try:
            dates.append(datetime.fromisoformat(str(value)))
        except ValueError:
            return []
    return [
        (dates[index + 1] - dates[index]).days
        for index in range(len(dates) - 1)
    ]


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "ReidAttackResult",
    "generate_reid_leaderboard",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
]
