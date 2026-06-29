"""Adversarial re-identification attack for PII benchmark fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.core.safety_sweep import hashed_span_surface
from openmed.eval.golden import GoldenFixture, load_golden_fixtures
from openmed.eval.report import BenchmarkReport
from openmed.risk import risk_report

from .linkage import linkage_attack


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
    attack_mode: str = "reid",
    deidentified_records: Sequence[Mapping[str, Any]] | None = None,
    auxiliary_records: Sequence[Mapping[str, Any]] | None = None,
    quasi_id_table: Sequence[Mapping[str, Any]] | None = None,
    quasi_identifiers: Sequence[str] | None = None,
    candidate_members: Sequence[Mapping[str, Any]] | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run the re-identification attack and return a BenchmarkReport.

    ``attack_mode="linkage"`` runs a first-class external quasi-identifier
    linkage attack against ``quasi_id_table``. When ``candidate_members`` is
    provided in the default ``"reid"`` mode, the membership-inference probe
    runs against the de-identified records and its result is added to the
    report metrics under ``"membership_inference"``.
    """

    fixtures = _load_suite_fixtures(suite)
    if attack_mode not in {"reid", "linkage"}:
        raise ValueError("attack_mode must be 'reid' or 'linkage'")

    if attack_mode == "linkage":
        if quasi_id_table is None:
            raise ValueError("quasi_id_table is required for linkage mode")
        deidentified = (
            list(deidentified_records)
            if deidentified_records is not None
            else [_deidentified_record(fixture) for fixture in fixtures]
        )
        linkage_result = linkage_attack(
            deidentified,
            quasi_id_table,
            quasi_identifiers=quasi_identifiers,
        )
        report = BenchmarkReport(
            suite=suite,
            model_name=model_name,
            device="attack",
            fixture_count=linkage_result.record_count,
            generated_at=generated_at or _utc_now(),
            metrics={
                "linkage_unique_match_rate": linkage_result.unique_match_rate,
                "linkage_attack": linkage_result.to_metric(),
            },
            metadata={
                "attack": "linkage",
                "leaderboard_metric": "linkage_unique_match_rate",
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

    result = run_reid_attack(
        fixtures,
        deidentified_records=deidentified_records,
        auxiliary_records=auxiliary_records,
    )
    metrics: dict[str, Any] = {
        "reid_leakage": {
            "rate": result.rate,
            "numerator": result.numerator,
            "denominator": result.denominator,
        },
        "reidentification": result.to_metric(),
    }
    if candidate_members is not None:
        deidentified = (
            list(deidentified_records)
            if deidentified_records is not None
            else [_deidentified_record(fixture) for fixture in fixtures]
        )
        metrics["membership_inference"] = membership_inference_attack(
            deidentified, candidate_members
        ).to_metric()
    report = BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device="attack",
        fixture_count=result.denominator,
        generated_at=generated_at or _utc_now(),
        metrics=metrics,
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
        metadata={
            "fixture_ids": [fixture.fixture_id for fixture in normalized_fixtures]
        },
    )


_ID_FIELDS = ("record_id", "doc_id", "id")
_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")


@dataclass(frozen=True)
class MembershipInferenceResult:
    """Membership-inference probe score over de-identified records.

    ``advantage`` is the attacker accuracy above the 0.5 chance baseline:
    confident-correct decisions count 1.0, confident-wrong 0.0, and records
    with no distinguishing residual signal count as chance (0.5).
    """

    advantage: float
    accuracy: float
    record_count: int
    confident_count: int
    per_record: tuple[dict[str, Any], ...]
    baseline: float = 0.5

    def to_metric(self) -> dict[str, Any]:
        return {
            "advantage": float(self.advantage),
            "accuracy": float(self.accuracy),
            "baseline": float(self.baseline),
            "record_count": int(self.record_count),
            "confident_count": int(self.confident_count),
            "per_record": [dict(row) for row in self.per_record],
        }


@dataclass(frozen=True)
class SideChannelProbeResult:
    """Mutual-information style probe for PHI encoded in timing metadata."""

    flagged: bool
    estimate_bits: float
    threshold_bits: float
    sample_count: int
    findings: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metric(self) -> dict[str, Any]:
        return {
            "flagged": bool(self.flagged),
            "estimate_bits": float(self.estimate_bits),
            "threshold_bits": float(self.threshold_bits),
            "sample_count": int(self.sample_count),
            "findings": [dict(item) for item in self.findings],
            "metadata": dict(self.metadata),
        }


def membership_inference_attack(
    deidentified_records: Sequence[Mapping[str, Any]],
    candidate_members: Sequence[Mapping[str, Any]],
    *,
    threshold: int = 1,
) -> MembershipInferenceResult:
    """Score whether each de-identified record's source is in the candidates.

    Each record is matched to candidates by residual quasi-identifier overlap
    and surviving rare tokens. A token held by exactly one candidate is
    *distinguishing*; when a record's distinguishing tokens point to a single
    candidate (with at least ``threshold`` hits), the attacker confidently
    predicts that candidate. The ground-truth source id is used only to score
    correctness, never as an attack feature.
    """
    deidentified = list(deidentified_records)
    candidates = list(candidate_members)

    candidate_ids = [_record_id(record) for record in candidates]
    candidate_features = _feature_sets(candidates)
    token_to_candidates: defaultdict[str, set[str]] = defaultdict(set)
    for cid, features in zip(candidate_ids, candidate_features):
        for token in features:
            token_to_candidates[token].add(cid)

    deid_features = _feature_sets(deidentified)
    known_ids = set(candidate_ids)

    per_record: list[dict[str, Any]] = []
    outcomes: list[float] = []
    confident_count = 0
    for record, features in zip(deidentified, deid_features):
        record_id = _record_id(record)
        hits: Counter[str] = Counter()
        for token in features:
            owners = token_to_candidates.get(token, set())
            if len(owners) == 1:
                hits[next(iter(owners))] += 1

        best = _best_candidate(hits)
        confident = best is not None and hits[best] >= threshold
        true_source = record_id if record_id in known_ids else None

        if confident:
            confident_count += 1
            outcome = 1.0 if best == true_source else 0.0
        else:
            outcome = 0.5  # no distinguishing signal -> chance
        outcomes.append(outcome)

        per_record.append(
            {
                "record_id": record_id,
                "matched_candidate": best if confident else None,
                "confidence": (hits[best] / len(features))
                if confident and features
                else 0.0,
                "distinguishing_hits": int(hits[best]) if confident else 0,
                "outcome": outcome,
            }
        )

    accuracy = sum(outcomes) / len(outcomes) if outcomes else 0.5
    return MembershipInferenceResult(
        advantage=accuracy - 0.5,
        accuracy=accuracy,
        record_count=len(deidentified),
        confident_count=confident_count,
        per_record=tuple(per_record),
    )


def probe_span_timing_side_channel(
    fixtures: Sequence[Any],
    timing_records: Sequence[Mapping[str, Any]],
    *,
    threshold_bits: float = 0.30,
    min_samples: int = 4,
) -> SideChannelProbeResult:
    """Estimate whether span timings encode gold PHI beyond detection outputs.

    The probe discretizes per-span duration around the median and compares that
    bucket with a deterministic secret bit derived from the gold span surface.
    Findings include only fixture ids, offsets, labels, and hashes.
    """
    gold_by_fixture = _gold_span_index(fixtures)
    samples: list[tuple[int, float, dict[str, Any]]] = []

    for record in timing_records:
        fixture_id = str(record.get("fixture_id") or "")
        duration = _optional_float(record.get("duration_ms"))
        start = _optional_int(record.get("start"))
        end = _optional_int(record.get("end"))
        if duration is None or start is None or end is None:
            continue
        gold = _matching_gold_span(gold_by_fixture.get(fixture_id, ()), start, end)
        if gold is None:
            continue
        surface = str(gold["surface"])
        evidence = {
            "fixture_id": fixture_id,
            **hashed_span_surface(
                str(gold["fixture_text"]),
                int(gold["start"]),
                int(gold["end"]),
                label=str(gold["label"]),
            ),
        }
        samples.append((_surface_secret_bit(surface), duration, evidence))

    if len(samples) < min_samples:
        return SideChannelProbeResult(
            flagged=False,
            estimate_bits=0.0,
            threshold_bits=threshold_bits,
            sample_count=len(samples),
            metadata={"reason": "insufficient_timing_samples"},
        )

    durations = [duration for _secret_bit, duration, _evidence in samples]
    median_duration = _median(durations)
    bucketed = [
        (secret_bit, int(duration > median_duration), evidence)
        for secret_bit, duration, evidence in samples
    ]
    estimate = _mutual_information_bits(
        [(secret_bit, timing_bucket) for secret_bit, timing_bucket, _ in bucketed]
    )
    flagged = estimate >= threshold_bits
    findings = tuple(
        {
            **evidence,
            "secret_bucket": secret_bit,
            "timing_bucket": timing_bucket,
        }
        for secret_bit, timing_bucket, evidence in bucketed
        if flagged
    )
    return SideChannelProbeResult(
        flagged=flagged,
        estimate_bits=estimate,
        threshold_bits=threshold_bits,
        sample_count=len(bucketed),
        findings=findings,
        metadata={"median_duration_ms": median_duration},
    )


def _best_candidate(hits: Counter[str]) -> str | None:
    if not hits:
        return None
    # Deterministic: most hits, ties broken by sorted candidate id.
    return min(sorted(hits), key=lambda cid: (-hits[cid], cid))


def _feature_sets(records: Sequence[Mapping[str, Any]]) -> list[set[str]]:
    """Residual features per record: reused QI values plus surviving tokens.

    Identifier fields are excluded so the ground-truth id never leaks into the
    attack features.
    """
    sanitized = [_without_id_fields(record) for record in records]
    risk = risk_report(sanitized)
    qi_by_index: defaultdict[int, set[str]] = defaultdict(set)
    for qi in risk.get("quasi_identifiers") or ():
        value = str(qi.get("normalized_value") or "").strip()
        if value:
            qi_by_index[int(qi.get("record_index", -1))].add(value)

    feature_sets: list[set[str]] = []
    for index, record in enumerate(records):
        tokens = _residual_tokens(record)
        feature_sets.append(tokens | qi_by_index.get(index, set()))
    return feature_sets


def _without_id_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _ID_FIELDS}


def _residual_tokens(record: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        str(value) for key, value in record.items() if key not in _ID_FIELDS
    )
    return set(_TOKEN_RE.findall(text.lower()))


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
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            spans.extend(item for item in value if isinstance(item, Mapping))

    audit = record.get("audit") or record.get("audit_report")
    if hasattr(audit, "to_dict"):
        audit = audit.to_dict()
    if isinstance(audit, Mapping):
        value = audit.get("spans")
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            spans.extend(item for item in value if isinstance(item, Mapping))
    return spans


def _record_id(record: Mapping[str, Any]) -> str:
    return str(
        record.get("record_id") or record.get("doc_id") or record.get("id") or "record"
    )


def _intervals(values: Sequence[Any]) -> list[int]:
    dates: list[datetime] = []
    for value in values:
        try:
            dates.append(datetime.fromisoformat(str(value)))
        except ValueError:
            return []
    return [(dates[index + 1] - dates[index]).days for index in range(len(dates) - 1)]


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _gold_span_index(fixtures: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        fixture_id = str(
            _fixture_value(fixture, "fixture_id")
            or _fixture_value(fixture, "id")
            or "fixture"
        )
        text = str(_fixture_value(fixture, "text") or "")
        spans = (
            _fixture_value(fixture, "gold_spans")
            or _fixture_value(fixture, "entities")
            or ()
        )
        entries: list[dict[str, Any]] = []
        for span in spans:
            start = _optional_int(_span_value(span, "start"))
            end = _optional_int(_span_value(span, "end"))
            if start is None or end is None or not (0 <= start < end <= len(text)):
                continue
            entries.append(
                {
                    "start": start,
                    "end": end,
                    "label": _span_value(span, "label")
                    or _span_value(span, "entity_type")
                    or "OTHER",
                    "surface": text[start:end],
                    "fixture_text": text,
                }
            )
        indexed[fixture_id] = entries
    return indexed


def _matching_gold_span(
    spans: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> Mapping[str, Any] | None:
    for span in spans:
        if int(span["start"]) == start and int(span["end"]) == end:
            return span
    for span in spans:
        if start < int(span["end"]) and end > int(span["start"]):
            return span
    return None


def _surface_secret_bit(surface: str) -> int:
    return hashlib.sha256(surface.encode("utf-8")).digest()[0] & 1


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _mutual_information_bits(samples: Sequence[tuple[int, int]]) -> float:
    total = len(samples)
    if total == 0:
        return 0.0
    joint = Counter(samples)
    x_counts = Counter(secret for secret, _timing in samples)
    y_counts = Counter(timing for _secret, timing in samples)
    estimate = 0.0
    for (secret, timing), count in joint.items():
        p_xy = count / total
        p_x = x_counts[secret] / total
        p_y = y_counts[timing] / total
        estimate += p_xy * math.log2(p_xy / (p_x * p_y))
    return float(max(0.0, estimate))


def _fixture_value(fixture: Any, key: str) -> Any:
    if isinstance(fixture, Mapping):
        return fixture.get(key)
    return getattr(fixture, key, None)


def _span_value(span: Any, key: str) -> Any:
    if isinstance(span, Mapping):
        return span.get(key)
    return getattr(span, key, None)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "ReidAttackResult",
    "MembershipInferenceResult",
    "SideChannelProbeResult",
    "generate_reid_leaderboard",
    "membership_inference_attack",
    "probe_span_timing_side_channel",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
]
