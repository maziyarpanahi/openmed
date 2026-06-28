"""Reviewer-facing trace reports for pipeline span decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .pipeline import PipelineResult, PipelineStageResult
from .schemas.span import OpenMedSpan

EXPLAIN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpanTraceKey:
    """Safe key for a span without exposing its surface text."""

    start: int
    end: int
    text_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class DetectionTrace:
    """Detector provenance for a candidate span."""

    stage: str
    detector: str | None
    score: float | None
    canonical_label: str
    entity_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "detector": self.detector,
            "score": self.score,
            "canonical_label": self.canonical_label,
            "entity_type": self.entity_type,
        }


@dataclass(frozen=True)
class ArbitrationTrace:
    """Arbitration outcome for a retained or dropped span."""

    outcome: str
    rule: str
    reason: str | None = None
    mode: str | None = None
    policy_profile: str | None = None
    winning_span: SpanTraceKey | None = None
    losing_spans: tuple[SpanTraceKey, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "rule": self.rule,
            "reason": self.reason,
            "mode": self.mode,
            "policy_profile": self.policy_profile,
            "winning_span": (
                self.winning_span.to_dict() if self.winning_span is not None else None
            ),
            "losing_spans": [span.to_dict() for span in self.losing_spans],
        }


@dataclass(frozen=True)
class ThresholdTrace:
    """Threshold details that affected retention or final action."""

    keep_floor: float | None = None
    escalate_below: float | None = None
    action: str | None = None
    policy_profile: str | None = None
    source: str | None = None
    schema_version: int | None = None
    canonical_label: str | None = None
    language: str | None = None
    calibration: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_floor": self.keep_floor,
            "escalate_below": self.escalate_below,
            "action": self.action,
            "policy_profile": self.policy_profile,
            "source": self.source,
            "schema_version": self.schema_version,
            "canonical_label": self.canonical_label,
            "language": self.language,
            "calibration": _json_safe(self.calibration),
        }


@dataclass(frozen=True)
class PolicyTrace:
    """Policy rule that selected the final span action."""

    policy_label: str | None
    action: str
    rule: str
    policy: str | None = None
    source: str | None = None
    schema_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_label": self.policy_label,
            "action": self.action,
            "rule": self.rule,
            "policy": self.policy,
            "source": self.source,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class SpanTraceEntry:
    """Full explain trace for one emitted or dropped span."""

    key: SpanTraceKey
    detections: tuple[DetectionTrace, ...]
    arbitration: ArbitrationTrace
    threshold: ThresholdTrace | None
    policy: PolicyTrace | None
    final_action: str
    emitted: bool = True
    normalized_key: SpanTraceKey | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "normalized_key": (
                self.normalized_key.to_dict()
                if self.normalized_key is not None
                else None
            ),
            "emitted": self.emitted,
            "detections": [detection.to_dict() for detection in self.detections],
            "arbitration": self.arbitration.to_dict(),
            "threshold": (
                self.threshold.to_dict() if self.threshold is not None else None
            ),
            "policy": self.policy.to_dict() if self.policy is not None else None,
            "final_action": self.final_action,
        }


@dataclass(frozen=True)
class ExplainReport:
    """Structured explain report for a completed pipeline result."""

    entries: tuple[SpanTraceEntry, ...]
    dropped_spans: tuple[SpanTraceEntry, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = EXPLAIN_SCHEMA_VERSION

    def render(self, fmt: Literal["text", "dict"] = "text") -> str | dict[str, Any]:
        """Render this report as compact text or a JSON-serializable dictionary."""

        return render(self, fmt=fmt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": _json_safe(self.metadata),
            "spans": [entry.to_dict() for entry in self.entries],
            "dropped_spans": [entry.to_dict() for entry in self.dropped_spans],
        }


def explain(result: PipelineResult) -> ExplainReport:
    """Build a safe span-decision trace from a pipeline result.

    The report is derived from ``stage_results`` and ``OpenMedSpan`` provenance.
    It deliberately emits offsets and HMAC hashes, never span surface text.
    """

    stages = {stage.name: stage for stage in result.stage_results}
    detection_index = _detection_index(result.stage_results)
    policy_index = _span_index(stages.get("policy_actions"))
    sweep_index = _span_index(stages.get("safety_sweep"))
    arbitration_index, dropped_entries = _arbitration_index(
        stages.get("span_arbitration"),
        detection_index,
    )

    entries: list[SpanTraceEntry] = []
    for span in result.spans:
        key = _span_key(span)
        normalized_key = _normalized_key(span)
        source_span = (
            policy_index.get(normalized_key) or sweep_index.get(normalized_key) or span
        )
        detections = detection_index.get(normalized_key, ())
        arbitration = arbitration_index.get(
            normalized_key,
            ArbitrationTrace(outcome="unknown", rule="not_recorded"),
        )
        threshold = _threshold_trace(source_span)
        policy = _policy_trace(source_span, threshold)
        final_action = policy.action if policy is not None else source_span.action
        entries.append(
            SpanTraceEntry(
                key=key,
                normalized_key=normalized_key if normalized_key != key else None,
                detections=detections,
                arbitration=arbitration,
                threshold=threshold,
                policy=policy,
                final_action=final_action,
                emitted=True,
            )
        )

    metadata = {
        "doc_id": result.audit_record.get("doc_id"),
        "language": result.audit_record.get("language"),
        "stage_count": len(result.stage_results),
        "span_count": len(entries),
        "dropped_span_count": len(dropped_entries),
        "arbitration_trace_recorded": bool(
            stages.get("span_arbitration")
            and stages["span_arbitration"].metadata.get("arbitration_trace")
        ),
    }
    return ExplainReport(
        entries=tuple(entries),
        dropped_spans=tuple(dropped_entries),
        metadata=metadata,
    )


def render(
    report: ExplainReport,
    fmt: Literal["text", "dict"] = "text",
) -> str | dict[str, Any]:
    """Render an explain report as text or a JSON-serializable dictionary."""

    if fmt == "dict":
        return report.to_dict()
    if fmt != "text":
        raise ValueError("fmt must be 'text' or 'dict'")

    lines = [
        (
            "OpenMed explain trace "
            f"(spans={len(report.entries)}, dropped={len(report.dropped_spans)})"
        )
    ]
    for entry in report.entries:
        lines.extend(_render_entry(entry, prefix="span"))
    for entry in report.dropped_spans:
        lines.extend(_render_entry(entry, prefix="dropped"))
    return "\n".join(lines)


def _render_entry(entry: SpanTraceEntry, *, prefix: str) -> list[str]:
    key = entry.key
    policy_label = entry.policy.policy_label if entry.policy is not None else None
    lines = [
        (
            f"{prefix} {key.start}-{key.end} {key.text_hash} "
            f"action={entry.final_action} policy_label={policy_label}"
        )
    ]
    if entry.detections:
        detectors = ", ".join(
            (
                f"{detection.stage}/{detection.detector or 'unknown'}"
                f" score={detection.score}"
            )
            for detection in entry.detections
        )
        lines.append(f"  detectors: {detectors}")
    arbitration = entry.arbitration
    arbitration_line = (
        f"  arbitration: outcome={arbitration.outcome} rule={arbitration.rule}"
    )
    if arbitration.winning_span is not None:
        winner = arbitration.winning_span
        arbitration_line += f" winner={winner.start}-{winner.end} {winner.text_hash}"
    if arbitration.losing_spans:
        losses = ", ".join(
            f"{span.start}-{span.end} {span.text_hash}"
            for span in arbitration.losing_spans
        )
        arbitration_line += f" losing_spans={losses}"
    lines.append(arbitration_line)
    if entry.threshold is not None:
        threshold = entry.threshold
        lines.append(
            (
                "  threshold: "
                f"keep_floor={threshold.keep_floor} "
                f"action={threshold.action} "
                f"profile={threshold.policy_profile} "
                f"source={threshold.source}"
            )
        )
    if entry.policy is not None:
        policy = entry.policy
        lines.append(
            (
                "  policy: "
                f"rule={policy.rule} action={policy.action} source={policy.source}"
            )
        )
    return lines


def _detection_index(
    stages: Sequence[PipelineStageResult],
) -> dict[SpanTraceKey, tuple[DetectionTrace, ...]]:
    detection_stages = {
        "deterministic_detectors",
        "fast_pii_model",
        "clinical_phi_model",
        "safety_sweep",
    }
    grouped: dict[SpanTraceKey, list[DetectionTrace]] = {}
    for stage in stages:
        if stage.name not in detection_stages:
            continue
        for span in stage.spans:
            key = _span_key(span)
            grouped.setdefault(key, []).append(
                DetectionTrace(
                    stage=stage.name,
                    detector=span.detector,
                    score=span.score,
                    canonical_label=span.canonical_label,
                    entity_type=span.entity_type,
                )
            )
    return {key: tuple(value) for key, value in grouped.items()}


def _span_index(stage: PipelineStageResult | None) -> dict[SpanTraceKey, OpenMedSpan]:
    if stage is None:
        return {}
    return {_span_key(span): span for span in stage.spans}


def _arbitration_index(
    stage: PipelineStageResult | None,
    detection_index: Mapping[SpanTraceKey, tuple[DetectionTrace, ...]],
) -> tuple[dict[SpanTraceKey, ArbitrationTrace], list[SpanTraceEntry]]:
    if stage is None:
        return {}, []

    trace = stage.metadata.get("arbitration_trace")
    if not isinstance(trace, Mapping):
        return {
            _span_key(span): ArbitrationTrace(outcome="winner", rule="retained")
            for span in stage.spans
        }, []

    mode = _optional_str(trace.get("mode"))
    policy_profile = _optional_str(trace.get("policy_profile"))
    decisions = trace.get("decisions") or ()
    traces: dict[SpanTraceKey, ArbitrationTrace] = {}
    dropped_entries: list[SpanTraceEntry] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        span_key = _span_key_from_mapping(decision.get("span"))
        if span_key is None:
            continue
        winning_key = _span_key_from_mapping(decision.get("winning_span"))
        losing_keys = tuple(
            key
            for key in (
                _span_key_from_mapping(loser)
                for loser in decision.get("losing_spans") or ()
            )
            if key is not None
        )
        arbitration = ArbitrationTrace(
            outcome=str(decision.get("outcome") or "unknown"),
            rule=str(decision.get("rule") or "unknown"),
            reason=_optional_str(decision.get("reason")),
            mode=mode,
            policy_profile=policy_profile,
            winning_span=winning_key,
            losing_spans=losing_keys,
        )
        if arbitration.outcome == "winner" or traces.get(span_key) is None:
            traces[span_key] = arbitration
        if arbitration.outcome == "loser":
            threshold = _threshold_trace_from_mapping(decision.get("threshold"))
            dropped_entries.append(
                SpanTraceEntry(
                    key=span_key,
                    normalized_key=None,
                    detections=detection_index.get(span_key, ()),
                    arbitration=arbitration,
                    threshold=threshold,
                    policy=None,
                    final_action="dropped",
                    emitted=False,
                )
            )
    return traces, dropped_entries


def _threshold_trace(span: OpenMedSpan) -> ThresholdTrace | None:
    metadata = dict(span.metadata)
    threshold_action = metadata.get("threshold_action")
    if isinstance(threshold_action, Mapping):
        return _threshold_trace_from_mapping(threshold_action)

    calibration_threshold = metadata.get("calibration_threshold")
    if isinstance(calibration_threshold, Mapping):
        return ThresholdTrace(
            keep_floor=_optional_float(calibration_threshold.get("threshold")),
            source=_optional_str(calibration_threshold.get("source")),
            schema_version=_optional_int(calibration_threshold.get("schema_version")),
            language=_optional_str(calibration_threshold.get("language")),
            calibration=_whitelisted_mapping(
                calibration_threshold,
                {
                    "threshold",
                    "source",
                    "schema_version",
                    "model_id",
                    "language",
                },
            ),
        )
    return None


def _threshold_trace_from_mapping(value: Any) -> ThresholdTrace | None:
    if not isinstance(value, Mapping):
        return None
    return ThresholdTrace(
        keep_floor=_optional_float(value.get("keep_floor")),
        escalate_below=_optional_float(value.get("escalate_below")),
        action=_optional_str(value.get("action")),
        policy_profile=_optional_str(value.get("policy_profile")),
        source=_optional_str(value.get("source")),
        schema_version=_optional_int(value.get("schema_version")),
        canonical_label=_optional_str(value.get("canonical_label")),
        language=_optional_str(value.get("language")),
    )


def _policy_trace(
    span: OpenMedSpan,
    threshold: ThresholdTrace | None,
) -> PolicyTrace | None:
    metadata = dict(span.metadata)
    policy_action = metadata.get("policy_action")
    if isinstance(policy_action, Mapping):
        return PolicyTrace(
            policy_label=span.policy_label,
            action=str(policy_action.get("action") or span.action),
            rule="policy_profile",
            policy=_optional_str(policy_action.get("policy")),
            source=_optional_str(policy_action.get("source")),
            schema_version=_optional_int(policy_action.get("schema_version")),
        )
    if threshold is not None:
        return PolicyTrace(
            policy_label=span.policy_label,
            action=threshold.action or span.action,
            rule="threshold_matrix",
            policy=threshold.policy_profile,
            source=threshold.source,
            schema_version=threshold.schema_version,
        )
    return None


def _span_key(span: OpenMedSpan) -> SpanTraceKey:
    return SpanTraceKey(
        start=span.start,
        end=span.end,
        text_hash=span.text_hash,
    )


def _normalized_key(span: OpenMedSpan) -> SpanTraceKey:
    metadata = dict(span.metadata)
    start = metadata.get("normalized_start")
    end = metadata.get("normalized_end")
    text_hash = metadata.get("normalized_text_hash")
    if isinstance(start, int) and isinstance(end, int) and isinstance(text_hash, str):
        return SpanTraceKey(start=start, end=end, text_hash=text_hash)
    return _span_key(span)


def _span_key_from_mapping(value: Any) -> SpanTraceKey | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("start")
    end = value.get("end")
    text_hash = value.get("text_hash")
    if isinstance(start, int) and isinstance(end, int) and isinstance(text_hash, str):
        return SpanTraceKey(start=start, end=end, text_hash=text_hash)
    return None


def _whitelisted_mapping(
    mapping: Mapping[str, Any],
    keys: set[str],
) -> Mapping[str, Any]:
    return {key: _json_safe(mapping[key]) for key in keys if key in mapping}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ArbitrationTrace",
    "DetectionTrace",
    "EXPLAIN_SCHEMA_VERSION",
    "ExplainReport",
    "PolicyTrace",
    "SpanTraceEntry",
    "SpanTraceKey",
    "ThresholdTrace",
    "explain",
    "render",
]
