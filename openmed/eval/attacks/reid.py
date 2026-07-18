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

from openmed.core.audit import stable_hash
from openmed.core.labels import normalize_label
from openmed.core.safety_sweep import hashed_span_surface
from openmed.core.thresholds import MembershipDefensePolicy
from openmed.eval.golden import GoldenFixture, load_golden_fixtures
from openmed.eval.metrics import compute_extraction_reemission_leakage
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
    shadow_member_records: Sequence[Mapping[str, Any]] | None = None,
    shadow_heldout_records: Sequence[Mapping[str, Any]] | None = None,
    membership_defense: Mapping[str, Any] | MembershipDefensePolicy | None = None,
    membership_advantage_ceiling: float | None = None,
    extraction_outputs: Any | None = None,
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
    if shadow_member_records is not None or shadow_heldout_records is not None:
        if shadow_member_records is None or shadow_heldout_records is None:
            raise ValueError(
                "shadow_member_records and shadow_heldout_records must be provided "
                "together"
            )
        metrics["membership_leakage"] = shadow_membership_inference_attack(
            shadow_member_records,
            shadow_heldout_records,
            defense_policy=membership_defense,
            advantage_ceiling=membership_advantage_ceiling,
        ).to_metric()
    if extraction_outputs is not None:
        metrics["extraction_reemission_leakage"] = (
            compute_extraction_reemission_leakage(
                [span for fixture in fixtures for span in fixture.gold_spans],
                extraction_outputs,
            ).to_dict()
        )
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
class ShadowMembershipInferenceResult:
    """Trained shadow-model membership-inference score over detector outputs."""

    attacker_auc: float
    attacker_advantage: float
    accuracy: float
    decision_threshold: float
    train_record_count: int
    eval_record_count: int
    member_count: int
    heldout_count: int
    advantage_ceiling: float
    defense: Mapping[str, Any] = field(default_factory=dict)
    per_label: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    feature_hash: str = ""

    def to_metric(self) -> dict[str, Any]:
        return {
            "attacker_auc": float(self.attacker_auc),
            "attacker_advantage": float(self.attacker_advantage),
            "advantage": float(self.attacker_advantage),
            "accuracy": float(self.accuracy),
            "decision_threshold": float(self.decision_threshold),
            "advantage_ceiling": float(self.advantage_ceiling),
            "train_record_count": int(self.train_record_count),
            "eval_record_count": int(self.eval_record_count),
            "member_count": int(self.member_count),
            "heldout_count": int(self.heldout_count),
            "feature_hash": self.feature_hash,
            "defense": dict(self.defense),
            "per_label": {
                label: dict(values) for label, values in sorted(self.per_label.items())
            },
        }


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
        matched_count = sum(
            1 for row in self.per_record if row.get("matched_candidate") is not None
        )
        return {
            "advantage": float(self.advantage),
            "accuracy": float(self.accuracy),
            "baseline": float(self.baseline),
            "record_count": int(self.record_count),
            "confident_count": int(self.confident_count),
            "matched_count": int(matched_count),
            "record_hashes": [
                stable_hash({"record_id": row.get("record_id")})
                for row in self.per_record
            ],
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


@dataclass(frozen=True)
class _ShadowExample:
    record_hash: str
    member: bool
    features: tuple[float, ...]
    label_scores: Mapping[str, tuple[float, ...]]


def shadow_membership_inference_attack(
    member_records: Sequence[Mapping[str, Any]],
    heldout_records: Sequence[Mapping[str, Any]],
    *,
    defense_policy: Mapping[str, Any] | MembershipDefensePolicy | None = None,
    advantage_ceiling: float | None = None,
    train_fraction: float = 0.5,
) -> ShadowMembershipInferenceResult:
    """Train and evaluate a shadow attacker over synthetic score records.

    Records may expose detector scores as top-level ``score``/``confidence``/
    ``logit`` values, a ``scores`` mapping, or span-like ``entities``/
    ``spans``/``detections`` rows. Raw note text is ignored; the report
    contains only aggregate metrics and stable hashes over score metadata.
    """

    if not member_records or not heldout_records:
        raise ValueError("shadow membership attack requires member and held-out rows")
    train_fraction = _bounded_fraction(train_fraction, "train_fraction")
    defense = MembershipDefensePolicy.from_mapping(defense_policy)
    ceiling = (
        defense.advantage_ceiling
        if advantage_ceiling is None
        else _bounded_fraction(advantage_ceiling, "advantage_ceiling")
    )

    members = [
        _shadow_example(record, member=True, defense=defense)
        for record in member_records
    ]
    heldout = [
        _shadow_example(record, member=False, defense=defense)
        for record in heldout_records
    ]
    train, evaluation = _train_eval_split(
        members,
        heldout,
        train_fraction=train_fraction,
    )
    attack = _evaluate_shadow_examples(train, evaluation)

    per_label: dict[str, Mapping[str, Any]] = {}
    labels = sorted(
        {label for example in (*members, *heldout) for label in example.label_scores}
    )
    for label in labels:
        label_train = [_label_example(example, label) for example in train]
        label_eval = [_label_example(example, label) for example in evaluation]
        label_attack = _evaluate_shadow_examples(label_train, label_eval)
        per_label[label] = {
            "attacker_auc": label_attack["auc"],
            "attacker_advantage": label_attack["advantage"],
            "member_count": sum(1 for example in label_eval if example.member),
            "heldout_count": sum(1 for example in label_eval if not example.member),
            "feature_hash": stable_hash(
                {
                    "label": label,
                    "records": [example.record_hash for example in label_eval],
                }
            ),
        }

    return ShadowMembershipInferenceResult(
        attacker_auc=attack["auc"],
        attacker_advantage=attack["advantage"],
        accuracy=attack["accuracy"],
        decision_threshold=attack["threshold"],
        train_record_count=len(train),
        eval_record_count=len(evaluation),
        member_count=len(members),
        heldout_count=len(heldout),
        advantage_ceiling=ceiling,
        defense=defense.to_dict(),
        per_label=per_label,
        feature_hash=stable_hash(
            {
                "members": [example.record_hash for example in members],
                "heldout": [example.record_hash for example in heldout],
                "defense": defense.to_dict(),
            }
        ),
    )


def _shadow_example(
    record: Mapping[str, Any],
    *,
    member: bool,
    defense: MembershipDefensePolicy,
) -> _ShadowExample:
    events = _score_events(record, defense)
    label_scores: defaultdict[str, list[float]] = defaultdict(list)
    for label, score in events:
        label_scores[label].append(score)
    scores = [score for _, score in events]
    return _ShadowExample(
        record_hash=_record_hash(record),
        member=member,
        features=_aggregate_score_features(scores),
        label_scores={
            label: tuple(values) for label, values in sorted(label_scores.items())
        },
    )


def _label_example(example: _ShadowExample, label: str) -> _ShadowExample:
    return _ShadowExample(
        record_hash=example.record_hash,
        member=example.member,
        features=_aggregate_score_features(example.label_scores.get(label, ())),
        label_scores={label: example.label_scores.get(label, ())},
    )


def _score_events(
    record: Mapping[str, Any],
    defense: MembershipDefensePolicy,
) -> list[tuple[str, float]]:
    events: list[tuple[str, float]] = []
    direct_score = _score_from_mapping(record)
    if direct_score is not None:
        events.append(
            (
                _label_from_mapping(record),
                defense.apply_score(direct_score),
            )
        )

    scores = record.get("scores")
    if isinstance(scores, Mapping):
        for label, value in scores.items():
            _append_score_value(events, str(label), value, defense)
    elif isinstance(scores, Sequence) and not isinstance(
        scores,
        (str, bytes, bytearray),
    ):
        for item in scores:
            _append_score_value(events, "OVERALL", item, defense)

    for key in ("spans", "entities", "detections", "predictions", "outputs"):
        rows = record.get(key)
        if not isinstance(rows, Sequence) or isinstance(
            rows,
            (str, bytes, bytearray),
        ):
            continue
        for item in rows:
            _append_score_value(events, "OVERALL", item, defense)

    return events or [("OVERALL", defense.apply_score(0.0))]


def _append_score_value(
    events: list[tuple[str, float]],
    fallback_label: str,
    value: Any,
    defense: MembershipDefensePolicy,
) -> None:
    if isinstance(value, Mapping):
        score = _score_from_mapping(value)
        if score is None:
            nested_scores = value.get("scores")
            if isinstance(nested_scores, Mapping):
                for label, nested in nested_scores.items():
                    _append_score_value(events, str(label), nested, defense)
            return
        events.append(
            (
                _label_from_mapping(value, fallback_label),
                defense.apply_score(score),
            )
        )
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            _append_score_value(events, fallback_label, item, defense)
        return
    score = _coerce_score(value)
    if score is not None:
        events.append(
            (
                _normalise_attack_label(fallback_label),
                defense.apply_score(score),
            )
        )


def _score_from_mapping(value: Mapping[str, Any]) -> float | None:
    if value.get("logit") is not None:
        try:
            return _sigmoid(float(value["logit"]))
        except (TypeError, ValueError):
            return None
    for key in ("score", "confidence", "probability", "prob"):
        score = _coerce_score(value.get(key))
        if score is not None:
            return score
    return None


def _label_from_mapping(
    value: Mapping[str, Any],
    fallback: str = "OVERALL",
) -> str:
    raw = (
        value.get("canonical_label")
        or value.get("label")
        or value.get("entity_type")
        or value.get("entity")
        or fallback
    )
    language = str(value.get("language") or value.get("lang") or "en")
    return _normalise_attack_label(str(raw), language=language)


def _normalise_attack_label(label: str, *, language: str = "en") -> str:
    if label.strip().upper() == "OVERALL":
        return "OVERALL"
    return normalize_label(label, language)


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return min(1.0, max(0.0, score))


def _aggregate_score_features(scores: Sequence[float]) -> tuple[float, ...]:
    if not scores:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [min(1.0, max(0.0, float(score))) for score in scores]
    count = len(values)
    mean = sum(values) / count
    variance = sum((score - mean) ** 2 for score in values) / count
    margins = [abs(score - 0.5) * 2.0 for score in values]
    return (
        max(values),
        mean,
        min(values),
        min(1.0, count / 10.0),
        min(1.0, variance * 4.0),
        sum(margins) / count,
    )


def _train_eval_split(
    members: Sequence[_ShadowExample],
    heldout: Sequence[_ShadowExample],
    *,
    train_fraction: float,
) -> tuple[list[_ShadowExample], list[_ShadowExample]]:
    train: list[_ShadowExample] = []
    evaluation: list[_ShadowExample] = []
    for examples in (members, heldout):
        ordered = sorted(examples, key=lambda example: example.record_hash)
        train_count = max(1, int(round(len(ordered) * train_fraction)))
        if len(ordered) > 1:
            train_count = min(train_count, len(ordered) - 1)
        train.extend(ordered[:train_count])
        evaluation.extend(ordered[train_count:] or ordered[:train_count])
    return train, evaluation


def _evaluate_shadow_examples(
    train: Sequence[_ShadowExample],
    evaluation: Sequence[_ShadowExample],
) -> dict[str, float]:
    weights = _train_logistic(train)
    train_scores = [_predict_membership(weights, example.features) for example in train]
    train_labels = [1 if example.member else 0 for example in train]
    eval_scores = [
        _predict_membership(weights, example.features) for example in evaluation
    ]
    eval_labels = [1 if example.member else 0 for example in evaluation]
    threshold = _best_score_threshold(train_scores, train_labels)
    accuracy = _threshold_accuracy(eval_scores, eval_labels, threshold)
    return {
        "auc": _auc(eval_scores, eval_labels),
        "advantage": _max_tpr_minus_fpr(eval_scores, eval_labels),
        "accuracy": accuracy,
        "threshold": threshold,
    }


def _train_logistic(
    examples: Sequence[_ShadowExample],
    *,
    epochs: int = 300,
    learning_rate: float = 0.2,
    l2: float = 0.001,
) -> tuple[float, ...]:
    width = max((len(example.features) for example in examples), default=0)
    weights = [0.0] * (width + 1)
    ordered = sorted(examples, key=lambda example: example.record_hash)
    for _ in range(epochs):
        for example in ordered:
            label = 1.0 if example.member else 0.0
            score = _predict_membership(tuple(weights), example.features)
            error = score - label
            weights[0] -= learning_rate * error
            for index, value in enumerate(example.features, start=1):
                weights[index] -= learning_rate * (error * value + l2 * weights[index])
    return tuple(weights)


def _predict_membership(weights: Sequence[float], features: Sequence[float]) -> float:
    if not weights:
        return 0.5
    logit = float(weights[0])
    for index, value in enumerate(features, start=1):
        if index >= len(weights):
            break
        logit += float(weights[index]) * float(value)
    return _sigmoid(logit)


def _best_score_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.5
    candidates = sorted(set(float(score) for score in scores))
    return max(
        candidates,
        key=lambda threshold: (
            _threshold_accuracy(scores, labels, threshold),
            _tpr_minus_fpr(scores, labels, threshold),
            -threshold,
        ),
    )


def _threshold_accuracy(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> float:
    if not scores:
        return 0.5
    correct = 0
    for score, label in zip(scores, labels):
        prediction = 1 if score >= threshold else 0
        correct += int(prediction == label)
    return correct / len(scores)


def _max_tpr_minus_fpr(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.0
    return max(
        0.0,
        *(_tpr_minus_fpr(scores, labels, threshold) for threshold in set(scores)),
    )


def _tpr_minus_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> float:
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return 0.0
    true_positive = sum(
        1 for score, label in zip(scores, labels) if label == 1 and score >= threshold
    )
    false_positive = sum(
        1 for score, label in zip(scores, labels) if label == 0 and score >= threshold
    )
    return (true_positive / positives) - (false_positive / negatives)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _bounded_fraction(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _record_hash(record: Mapping[str, Any]) -> str:
    return stable_hash(_record_hash_material(record))


def _record_hash_material(record: Mapping[str, Any]) -> dict[str, Any]:
    material: dict[str, Any] = {}
    for key, value in sorted(record.items()):
        if str(key) in {"text", "note", "source_text", "raw_text"}:
            material[str(key)] = {"sha256": stable_hash({"value": str(value)})}
        elif str(key) in _ID_FIELDS:
            material[str(key)] = str(value)
        elif str(key) in {
            "score",
            "confidence",
            "probability",
            "prob",
            "logit",
            "scores",
            "spans",
            "entities",
            "detections",
            "predictions",
            "outputs",
            "label",
            "canonical_label",
            "entity_type",
            "language",
            "lang",
        }:
            material[str(key)] = _hash_safe_value(value)
    return material


def _hash_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _hash_safe_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {"text", "note", "source_text", "raw_text"}
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_hash_safe_value(item) for item in value]
    return value


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
    "ShadowMembershipInferenceResult",
    "generate_reid_leaderboard",
    "membership_inference_attack",
    "probe_span_timing_side_channel",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
    "shadow_membership_inference_attack",
]
