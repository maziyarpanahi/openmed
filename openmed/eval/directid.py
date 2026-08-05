"""Auditable DirectID model and deterministic safety-sweep evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from openmed.core.audit import hash_text, stable_hash
from openmed.core.labels import normalize_label, policy_label_for, risk_level_for
from openmed.core.pii_entity_merger import PIIPattern
from openmed.core.quality_gates import (
    detect_overlapping_entities,
    resolve_overlapping_entities,
)
from openmed.core.safety_sweep import (
    SAFETY_SWEEP_PATTERNS_VERSION,
    SAFETY_SWEEP_SOURCE,
    hashed_span_surface,
    safety_sweep,
)
from openmed.eval.report import BenchmarkReport
from openmed.processing.outputs import EntityPrediction
from openmed.training.directid import (
    DIRECTID_FAMILY,
    DIRECTID_TIER,
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDHeadContract,
    validate_directid_contract,
)

DIRECTID_EVIDENCE_SCHEMA_VERSION = "openmed.eval.directid.v1"
DIRECTID_MODEL_SOURCE = "directid_model"
DIRECTID_GOLD_SOURCE = "gold_fixture"


class DirectIDEvidenceError(ValueError):
    """Raised when DirectID evidence inputs cannot be scored safely."""


@dataclass(frozen=True)
class DirectIDSpanProvenance:
    """Raw-text-free provenance shared by model and sweep spans."""

    source: str
    patterns_version: str | None
    start: int
    end: int
    text_hash: str

    def to_dict(self) -> dict[str, object]:
        """Return the contract's source/version/offset/hash provenance fields."""
        return {
            "source": self.source,
            "patterns_version": self.patterns_version,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class DirectIDRiskEvidence:
    """Policy-spine risk evidence for a canonical DirectID label."""

    policy_label: str
    risk_level: str
    critical: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready risk evidence without an identifier surface."""
        return {
            "policy_label": self.policy_label,
            "risk_level": self.risk_level,
            "critical": self.critical,
        }


@dataclass(frozen=True)
class DirectIDSpanEvidence:
    """One canonical prediction with PHI-safe provenance and attribution."""

    label: str
    provenance: DirectIDSpanProvenance
    risk: DirectIDRiskEvidence
    matched_gold: bool
    recovered_model_miss: bool

    def to_dict(self) -> dict[str, object]:
        """Return a raw-text-free serialized prediction."""
        return {
            "label": self.label,
            "provenance": self.provenance.to_dict(),
            "risk": self.risk.to_dict(),
            "matched_gold": self.matched_gold,
            "recovered_model_miss": self.recovered_model_miss,
        }


@dataclass(frozen=True)
class DirectIDModelMissEvidence:
    """A gold DirectID span missed by the model, recovered or residual."""

    label: str
    provenance: DirectIDSpanProvenance
    risk: DirectIDRiskEvidence
    recovered_by_safety_sweep: bool
    recovery_source: str | None
    recovery_patterns_version: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a miss record that never serializes the identifier value."""
        return {
            "label": self.label,
            "provenance": self.provenance.to_dict(),
            "risk": self.risk.to_dict(),
            "recovered_by_safety_sweep": self.recovered_by_safety_sweep,
            "recovery_source": self.recovery_source,
            "recovery_patterns_version": self.recovery_patterns_version,
        }


@dataclass(frozen=True)
class DirectIDRecallSummary:
    """Exact-span recall and precision for one DirectID detector stage."""

    prediction_count: int
    true_positive_count: int
    per_label_recall: Mapping[str, float]
    per_label_precision: Mapping[str, float]
    structured_id_recall: float

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready recall evidence."""
        return {
            "prediction_count": self.prediction_count,
            "true_positive_count": self.true_positive_count,
            "per_label_recall": dict(sorted(self.per_label_recall.items())),
            "per_label_precision": dict(sorted(self.per_label_precision.items())),
            "structured_id_recall": self.structured_id_recall,
        }


@dataclass(frozen=True)
class DirectIDEvidenceReport:
    """Combined DirectID/safety-sweep evidence for G1b, G3, and span integrity."""

    family: str
    contract_ref: str
    patterns_version: str
    eval_set_hash: str
    leakage_fixture_hash: str
    gold_span_count: int
    per_label_denominators: Mapping[str, int]
    model: DirectIDRecallSummary
    combined: DirectIDRecallSummary
    model_spans: tuple[DirectIDSpanEvidence, ...]
    safety_sweep_spans: tuple[DirectIDSpanEvidence, ...]
    combined_spans: tuple[DirectIDSpanEvidence, ...]
    model_misses: tuple[DirectIDModelMissEvidence, ...]
    recovered_per_label: Mapping[str, int]
    safety_sweep_recovered_count: int
    safety_sweep_structured_recovered_count: int
    structured_id_recall_gain: float
    critical_leakage_count: int
    residual_leakage_rate: float
    span_integrity: Mapping[str, int | bool]

    @property
    def gate_evidence(self) -> dict[str, dict[str, object]]:
        """Return release-gate fields with model misses kept independently visible."""
        unrecovered = [
            miss.to_dict()
            for miss in self.model_misses
            if not miss.recovered_by_safety_sweep
        ]
        return {
            "G1b": {
                "per_label_recall": dict(self.combined.per_label_recall),
                "structured_id_recall": self.combined.structured_id_recall,
                "model_per_label_recall": dict(self.model.per_label_recall),
                "model_structured_id_recall": self.model.structured_id_recall,
                "safety_sweep_recovered_count": self.safety_sweep_recovered_count,
                "safety_sweep_structured_recovered_count": (
                    self.safety_sweep_structured_recovered_count
                ),
                "structured_id_recall_gain": self.structured_id_recall_gain,
                "eval_set_hash": self.eval_set_hash,
            },
            "G3": {
                "critical_leakage_count": self.critical_leakage_count,
                "residual_leakage_rate": self.residual_leakage_rate,
                "leakage_fixture_hash": self.leakage_fixture_hash,
                "unrecovered_model_misses": unrecovered,
            },
            "G8": dict(self.span_integrity),
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the report without raw source text or identifier surfaces."""
        return {
            "schema_version": DIRECTID_EVIDENCE_SCHEMA_VERSION,
            "family": self.family,
            "contract_ref": self.contract_ref,
            "patterns_version": self.patterns_version,
            "eval_set_hash": self.eval_set_hash,
            "leakage_fixture_hash": self.leakage_fixture_hash,
            "gold_span_count": self.gold_span_count,
            "per_label_denominators": dict(sorted(self.per_label_denominators.items())),
            "model": {
                **self.model.to_dict(),
                "source": DIRECTID_MODEL_SOURCE,
                "spans": [span.to_dict() for span in self.model_spans],
            },
            "safety_sweep": {
                "source": SAFETY_SWEEP_SOURCE,
                "patterns_version": self.patterns_version,
                "spans_added": len(self.safety_sweep_spans),
                "recovered_model_misses": self.safety_sweep_recovered_count,
                "structured_ids_recovered": (
                    self.safety_sweep_structured_recovered_count
                ),
                "recovered_per_label": dict(sorted(self.recovered_per_label.items())),
                "structured_id_recall_gain": self.structured_id_recall_gain,
                "spans": [span.to_dict() for span in self.safety_sweep_spans],
            },
            "combined": {
                **self.combined.to_dict(),
                "spans": [span.to_dict() for span in self.combined_spans],
            },
            "model_misses": [miss.to_dict() for miss in self.model_misses],
            "critical_leakage_count": self.critical_leakage_count,
            "residual_leakage_rate": self.residual_leakage_rate,
            "span_integrity": dict(self.span_integrity),
            "gate_evidence": self.gate_evidence,
        }

    def to_benchmark_report(
        self,
        *,
        model_name: str = DIRECTID_FAMILY,
        device: str = "offline",
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkReport:
        """Adapt this evidence to the release-gate ``BenchmarkReport`` surface."""
        report_metadata = dict(metadata or {})
        report_metadata.update(
            {
                "family": "DirectID",
                "tier": DIRECTID_TIER,
                "eval_set_hash": self.eval_set_hash,
                "leakage_fixture_hash": self.leakage_fixture_hash,
                "per_label_denominators": dict(self.per_label_denominators),
                "directid_evidence": self.to_dict(),
            }
        )
        metrics = {
            "per_label_recall": dict(self.combined.per_label_recall),
            "per_label_precision": dict(self.combined.per_label_precision),
            "structured_id_recall": self.combined.structured_id_recall,
            "model_per_label_recall": dict(self.model.per_label_recall),
            "model_structured_id_recall": self.model.structured_id_recall,
            "safety_sweep_recovered_count": self.safety_sweep_recovered_count,
            "safety_sweep_structured_recovered_count": (
                self.safety_sweep_structured_recovered_count
            ),
            "structured_id_recall_gain": self.structured_id_recall_gain,
            "critical_leakage_count": self.critical_leakage_count,
            "residual_leakage_rate": self.residual_leakage_rate,
            "eval_set_hash": self.eval_set_hash,
            "leakage_fixture_hash": self.leakage_fixture_hash,
        }
        return BenchmarkReport(
            suite="directid_safety_sweep",
            model_name=model_name,
            device=device,
            fixture_count=1,
            metrics=metrics,
            generated_at=None,
            metadata=report_metadata,
        )


def build_directid_evidence(
    text: str,
    gold_spans: Sequence[Any],
    model_spans: Sequence[Any],
    *,
    lang: str = "en",
    locale: str | None = None,
    patterns: Sequence[PIIPattern] | None = None,
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
) -> DirectIDEvidenceReport:
    """Score DirectID output before and after the deterministic safety sweep.

    Gold and model inputs may be mappings or entity-like objects with ``label``,
    ``start``, and ``end`` fields. Identifier text is used only in memory to run
    the sweep and compute hashes; the returned report contains no raw surfaces.

    Args:
        text: Source text evaluated locally.
        gold_spans: Synthetic or authorized gold DirectID spans.
        model_spans: DirectID model predictions before the safety sweep.
        lang: Language passed to label normalization and the sweep.
        locale: Optional locale passed to the sweep.
        patterns: Optional deterministic pattern set, primarily for offline tests.
        contract: DirectID head contract governing labels and gate fields.

    Returns:
        A raw-text-free evidence report with model, recovery, and combined recall.

    Raises:
        DirectIDEvidenceError: If a span is invalid, unsupported, or gold overlaps.
    """
    validate_directid_contract(contract)
    if not isinstance(text, str):
        raise DirectIDEvidenceError("text must be a string")

    gold_entities = _coerce_spans(
        text,
        gold_spans,
        role="gold",
        lang=lang,
        contract=contract,
    )
    if not gold_entities:
        raise DirectIDEvidenceError("gold_spans must contain DirectID evidence")
    if detect_overlapping_entities(gold_entities):
        raise DirectIDEvidenceError("gold_spans must not overlap")

    model_entities = _coerce_spans(
        text,
        model_spans,
        role="model",
        lang=lang,
        contract=contract,
    )
    input_model_overlaps = len(detect_overlapping_entities(model_entities))
    resolved_model = resolve_overlapping_entities(model_entities)
    combined = safety_sweep(
        text,
        model_entities,
        lang=lang,
        locale=locale,
        patterns=patterns,
    )
    combined = [
        entity
        for entity in combined
        if _canonical_label(entity, lang=lang) in set(contract.labels)
    ]
    residual_overlaps = len(detect_overlapping_entities(combined))

    gold_keys = {_span_key(entity, lang=lang) for entity in gold_entities}
    if len(gold_keys) != len(gold_entities):
        raise DirectIDEvidenceError("gold_spans must not contain duplicates")
    model_keys = {_span_key(entity, lang=lang) for entity in resolved_model}
    combined_keys = {_span_key(entity, lang=lang) for entity in combined}

    safety_entities = [
        entity
        for entity in combined
        if _metadata(entity).get("source") == SAFETY_SWEEP_SOURCE
    ]
    safety_by_key = {_span_key(entity, lang=lang): entity for entity in safety_entities}
    recovered_keys = (gold_keys - model_keys) & set(safety_by_key)
    residual_keys = {
        key
        for key in gold_keys - combined_keys
        if key[2] in set(contract.critical_labels)
    }

    model_summary = _recall_summary(
        gold_entities,
        resolved_model,
        lang=lang,
        contract=contract,
    )
    combined_summary = _recall_summary(
        gold_entities,
        combined,
        lang=lang,
        contract=contract,
    )
    model_evidence = tuple(
        _prediction_evidence(
            text,
            entity,
            source=DIRECTID_MODEL_SOURCE,
            matched_gold=_span_key(entity, lang=lang) in gold_keys,
            recovered_model_miss=False,
            lang=lang,
            contract=contract,
        )
        for entity in resolved_model
    )
    sweep_evidence = tuple(
        _prediction_evidence(
            text,
            entity,
            source=SAFETY_SWEEP_SOURCE,
            matched_gold=_span_key(entity, lang=lang) in gold_keys,
            recovered_model_miss=_span_key(entity, lang=lang) in recovered_keys,
            lang=lang,
            contract=contract,
        )
        for entity in safety_entities
    )
    combined_evidence = tuple(
        _prediction_evidence(
            text,
            entity,
            source=(
                SAFETY_SWEEP_SOURCE
                if _metadata(entity).get("source") == SAFETY_SWEEP_SOURCE
                else DIRECTID_MODEL_SOURCE
            ),
            matched_gold=_span_key(entity, lang=lang) in gold_keys,
            recovered_model_miss=_span_key(entity, lang=lang) in recovered_keys,
            lang=lang,
            contract=contract,
        )
        for entity in combined
    )
    model_misses = tuple(
        _miss_evidence(
            text,
            entity,
            recovered=_span_key(entity, lang=lang) in recovered_keys,
            lang=lang,
            contract=contract,
        )
        for entity in gold_entities
        if _span_key(entity, lang=lang) not in model_keys
    )

    recovered_per_label = Counter(key[2] for key in recovered_keys)
    structured_labels = set(contract.structured_id_labels)
    structured_recovered = sum(
        count
        for label, count in recovered_per_label.items()
        if label in structured_labels
    )
    denominators = Counter(
        _canonical_label(entity, lang=lang) for entity in gold_entities
    )
    structured_gain = _rate(
        combined_summary.structured_id_recall - model_summary.structured_id_recall
    )
    critical_denominator = sum(
        count
        for label, count in denominators.items()
        if label in set(contract.critical_labels)
    )
    residual_leakage_rate = _rate(
        len(residual_keys) / critical_denominator if critical_denominator else 0.0
    )
    span_integrity: dict[str, int | bool] = {
        "passed": residual_overlaps == 0,
        "input_model_overlaps": input_model_overlaps,
        "model_overlaps_resolved": len(model_entities) - len(resolved_model),
        "combined_residual_overlaps": residual_overlaps,
    }

    gold_descriptors = [
        _gold_descriptor(text, entity, lang=lang, contract=contract)
        for entity in gold_entities
    ]
    eval_set_hash = stable_hash(
        {
            "schema_version": DIRECTID_EVIDENCE_SCHEMA_VERSION,
            "document_hash": hash_text(text),
            "gold_spans": gold_descriptors,
        }
    )
    leakage_fixture_hash = stable_hash(
        {
            "eval_set_hash": eval_set_hash,
            "critical_labels": sorted(contract.critical_labels),
        }
    )

    return DirectIDEvidenceReport(
        family=contract.family,
        contract_ref=contract.contract_ref,
        patterns_version=SAFETY_SWEEP_PATTERNS_VERSION,
        eval_set_hash=eval_set_hash,
        leakage_fixture_hash=leakage_fixture_hash,
        gold_span_count=len(gold_entities),
        per_label_denominators=dict(sorted(denominators.items())),
        model=model_summary,
        combined=combined_summary,
        model_spans=model_evidence,
        safety_sweep_spans=sweep_evidence,
        combined_spans=combined_evidence,
        model_misses=model_misses,
        recovered_per_label=dict(sorted(recovered_per_label.items())),
        safety_sweep_recovered_count=len(recovered_keys),
        safety_sweep_structured_recovered_count=structured_recovered,
        structured_id_recall_gain=structured_gain,
        critical_leakage_count=len(residual_keys),
        residual_leakage_rate=residual_leakage_rate,
        span_integrity=span_integrity,
    )


def _coerce_spans(
    text: str,
    spans: Sequence[Any],
    *,
    role: str,
    lang: str,
    contract: DirectIDHeadContract,
) -> list[EntityPrediction]:
    entities: list[EntityPrediction] = []
    for index, span in enumerate(spans):
        start = _value(span, "start")
        end = _value(span, "end")
        if type(start) is not int or type(end) is not int:
            raise DirectIDEvidenceError(f"{role} span {index} requires integer offsets")
        if start < 0 or start >= end or end > len(text):
            raise DirectIDEvidenceError(f"{role} span {index} has invalid offsets")

        label = str(
            _value(span, "label")
            or _value(span, "entity_type")
            or _value(span, "entity_group")
            or ""
        )
        canonical = normalize_label(label, lang=lang)
        if canonical not in set(contract.labels):
            raise DirectIDEvidenceError(
                f"{role} span {index} has unsupported DirectID label {canonical!r}"
            )

        supplied_text = _value(span, "text") or _value(span, "word")
        surface = text[start:end]
        if supplied_text is not None and str(supplied_text) != surface:
            raise DirectIDEvidenceError(
                f"{role} span {index} text does not match its offsets"
            )
        confidence = _value(span, "confidence")
        if confidence is None:
            confidence = _value(span, "score")
        try:
            score = float(confidence) if confidence is not None else 1.0
        except (TypeError, ValueError) as exc:
            raise DirectIDEvidenceError(
                f"{role} span {index} has invalid confidence"
            ) from exc

        metadata = dict(_metadata(span))
        metadata["source"] = (
            DIRECTID_GOLD_SOURCE if role == "gold" else DIRECTID_MODEL_SOURCE
        )
        entities.append(
            EntityPrediction(
                text=surface,
                label=label,
                start=start,
                end=end,
                confidence=score,
                metadata=metadata,
            )
        )
    return entities


def _recall_summary(
    gold: Sequence[EntityPrediction],
    predicted: Sequence[EntityPrediction],
    *,
    lang: str,
    contract: DirectIDHeadContract,
) -> DirectIDRecallSummary:
    gold_keys = {_span_key(entity, lang=lang) for entity in gold}
    predicted_keys = {_span_key(entity, lang=lang) for entity in predicted}
    true_positive_keys = gold_keys & predicted_keys
    gold_counts = Counter(key[2] for key in gold_keys)
    predicted_counts = Counter(key[2] for key in predicted_keys)
    true_positive_counts = Counter(key[2] for key in true_positive_keys)

    per_label_recall = {
        label: _rate(true_positive_counts[label] / count)
        for label, count in sorted(gold_counts.items())
    }
    labels = sorted(set(gold_counts) | set(predicted_counts))
    per_label_precision = {
        label: _rate(
            true_positive_counts[label] / predicted_counts[label]
            if predicted_counts[label]
            else 0.0
        )
        for label in labels
    }
    structured_labels = set(contract.structured_id_labels)
    structured_total = sum(
        count for label, count in gold_counts.items() if label in structured_labels
    )
    structured_true_positives = sum(
        count
        for label, count in true_positive_counts.items()
        if label in structured_labels
    )
    structured_recall = _rate(
        structured_true_positives / structured_total if structured_total else 0.0
    )
    return DirectIDRecallSummary(
        prediction_count=len(predicted_keys),
        true_positive_count=len(true_positive_keys),
        per_label_recall=per_label_recall,
        per_label_precision=per_label_precision,
        structured_id_recall=structured_recall,
    )


def _prediction_evidence(
    text: str,
    entity: EntityPrediction,
    *,
    source: str,
    matched_gold: bool,
    recovered_model_miss: bool,
    lang: str,
    contract: DirectIDHeadContract,
) -> DirectIDSpanEvidence:
    label = _canonical_label(entity, lang=lang)
    metadata = _metadata(entity)
    patterns_version = (
        str(metadata.get("patterns_version") or SAFETY_SWEEP_PATTERNS_VERSION)
        if source == SAFETY_SWEEP_SOURCE
        else None
    )
    return DirectIDSpanEvidence(
        label=label,
        provenance=_provenance(
            text,
            entity,
            source=source,
            patterns_version=patterns_version,
        ),
        risk=_risk_evidence(label, contract=contract),
        matched_gold=matched_gold,
        recovered_model_miss=recovered_model_miss,
    )


def _miss_evidence(
    text: str,
    entity: EntityPrediction,
    *,
    recovered: bool,
    lang: str,
    contract: DirectIDHeadContract,
) -> DirectIDModelMissEvidence:
    label = _canonical_label(entity, lang=lang)
    return DirectIDModelMissEvidence(
        label=label,
        provenance=_provenance(
            text,
            entity,
            source=DIRECTID_GOLD_SOURCE,
            patterns_version=None,
        ),
        risk=_risk_evidence(label, contract=contract),
        recovered_by_safety_sweep=recovered,
        recovery_source=SAFETY_SWEEP_SOURCE if recovered else None,
        recovery_patterns_version=(
            SAFETY_SWEEP_PATTERNS_VERSION if recovered else None
        ),
    )


def _gold_descriptor(
    text: str,
    entity: EntityPrediction,
    *,
    lang: str,
    contract: DirectIDHeadContract,
) -> dict[str, object]:
    label = _canonical_label(entity, lang=lang)
    return {
        "label": label,
        "provenance": _provenance(
            text,
            entity,
            source=DIRECTID_GOLD_SOURCE,
            patterns_version=None,
        ).to_dict(),
        "risk": _risk_evidence(label, contract=contract).to_dict(),
    }


def _provenance(
    text: str,
    entity: EntityPrediction,
    *,
    source: str,
    patterns_version: str | None,
) -> DirectIDSpanProvenance:
    start = int(entity.start or 0)
    end = int(entity.end or start)
    hashed = hashed_span_surface(text, start, end)
    return DirectIDSpanProvenance(
        source=source,
        patterns_version=patterns_version,
        start=start,
        end=end,
        text_hash=str(hashed["text_hash"]),
    )


def _risk_evidence(
    label: str,
    *,
    contract: DirectIDHeadContract,
) -> DirectIDRiskEvidence:
    return DirectIDRiskEvidence(
        policy_label=policy_label_for(label),
        risk_level=risk_level_for(label),
        critical=label in set(contract.critical_labels),
    )


def _span_key(entity: EntityPrediction, *, lang: str) -> tuple[int, int, str]:
    return (
        int(entity.start or 0),
        int(entity.end or entity.start or 0),
        _canonical_label(entity, lang=lang),
    )


def _canonical_label(entity: Any, *, lang: str) -> str:
    return normalize_label(
        str(
            _value(entity, "label")
            or _value(entity, "entity_type")
            or _value(entity, "entity_group")
            or ""
        ),
        lang=lang,
    )


def _value(span: Any, key: str) -> Any:
    if isinstance(span, Mapping):
        return span.get(key)
    return getattr(span, key, None)


def _metadata(span: Any) -> Mapping[str, Any]:
    metadata = _value(span, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _rate(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 12)


__all__ = [
    "DIRECTID_EVIDENCE_SCHEMA_VERSION",
    "DIRECTID_GOLD_SOURCE",
    "DIRECTID_MODEL_SOURCE",
    "DirectIDEvidenceError",
    "DirectIDEvidenceReport",
    "DirectIDModelMissEvidence",
    "DirectIDRecallSummary",
    "DirectIDRiskEvidence",
    "DirectIDSpanEvidence",
    "DirectIDSpanProvenance",
    "build_directid_evidence",
]
