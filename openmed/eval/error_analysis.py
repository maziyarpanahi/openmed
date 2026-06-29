"""Per-entity confusion matrices and error examples for eval suites."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.core.audit import stable_hash
from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.quality_gates import (
    detect_overlapping_entities,
    validate_entity_spans,
)
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
    load_fixtures,
)
from openmed.eval.metrics import EvalSpan, normalize_eval_spans
from openmed.eval.report import _format_value, _plain

MISSED = "missed"
SPURIOUS = "spurious"
ERROR_BUCKETS: tuple[str, str] = (MISSED, SPURIOUS)
LABELS: tuple[str, ...] = tuple(sorted(CANONICAL_LABELS))
MATRIX_LABELS: tuple[str, ...] = LABELS + ERROR_BUCKETS
DEFAULT_EXAMPLE_CAP = 5
DEFAULT_CONTEXT_WINDOW = 24
DEFAULT_DEDUPE_SIMILARITY = 0.92
LABELING_QUEUE_SCHEMA_VERSION = "openmed.eval.labeling_queue.v1"

_RAW_TEXT_KEYS = frozenset(
    {
        "context",
        "raw_context",
        "raw_source",
        "raw_text",
        "source_text",
        "span_text",
        "text",
    }
)
_PLACEHOLDER_RE = re.compile(r"(<[A-Za-z][A-Za-z0-9_:-]*>)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DATE_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")
_ID_RE = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d .()/-]{6,}\d)\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[./:-]\d+)*\b")
_WORD_RE = re.compile(r"(?u)\b[^\W\d_][\w'.-]*\b")
_TOKEN_RE = re.compile(r"<[A-Z0-9_:-]+>|[A-Za-z0-9_:-]+")


@dataclass(frozen=True)
class ErrorSpanExample:
    """One false-negative or false-positive span without plaintext PHI."""

    kind: str
    fixture_id: str
    label: str
    start: int
    end: int
    context_start: int
    context_end: int
    text_hash: str
    matched_label: str | None = None
    matched_start: int | None = None
    matched_end: int | None = None
    matched_text_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready example payload."""
        payload: dict[str, Any] = {
            "context_end": self.context_end,
            "context_start": self.context_start,
            "end": self.end,
            "fixture_id": self.fixture_id,
            "kind": self.kind,
            "label": self.label,
            "start": self.start,
            "text_hash": self.text_hash,
        }
        if self.matched_label is not None:
            payload["matched_label"] = self.matched_label
        if self.matched_start is not None:
            payload["matched_start"] = self.matched_start
        if self.matched_end is not None:
            payload["matched_end"] = self.matched_end
        if self.matched_text_hash is not None:
            payload["matched_text_hash"] = self.matched_text_hash
        return payload


@dataclass(frozen=True)
class ErrorAnalysisReport:
    """Serializable per-entity error-analysis report."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    confusion_matrix: Mapping[str, Mapping[str, int]]
    false_negatives: Mapping[str, Sequence[ErrorSpanExample]]
    false_positives: Mapping[str, Sequence[ErrorSpanExample]]
    example_cap: int = DEFAULT_EXAMPLE_CAP
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary with stable keys."""
        return {
            "confusion_matrix": {
                label: _matrix_row(self.confusion_matrix.get(label, {}))
                for label in MATRIX_LABELS
            },
            "device": self.device,
            "example_cap": self.example_cap,
            "false_negatives": _examples_to_dict(self.false_negatives),
            "false_positives": _examples_to_dict(self.false_positives),
            "fixture_count": self.fixture_count,
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
            "model_name": self.model_name,
            "suite": self.suite,
        }

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report to deterministic Markdown."""
        lines = [
            f"# Error Analysis Report: {self.suite}",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Example Cap | {self.example_cap} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Confusion Matrix",
                "",
                "| Gold Label/Bucket | Predicted Label/Bucket | Count |",
                "|---|---|---:|",
            ]
        )
        for gold_label in MATRIX_LABELS:
            row = self.confusion_matrix.get(gold_label, {})
            for predicted_label in MATRIX_LABELS:
                count = int(row.get(predicted_label, 0))
                if count:
                    lines.append(f"| `{gold_label}` | `{predicted_label}` | {count} |")

        lines.extend(_examples_markdown("False Negatives", self.false_negatives))
        lines.extend(_examples_markdown("False Positives", self.false_positives))

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


@dataclass(frozen=True)
class LabelingQueueItem:
    """One PHI-free gate-failure candidate for human labeling."""

    span_hash: str
    surrogate_context: str
    label: str
    language: str
    priority: float
    uncertainty: float
    gate_impact: float
    provenance: Mapping[str, Any]
    _dedupe_context: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "span_hash", str(self.span_hash))
        object.__setattr__(self, "surrogate_context", str(self.surrogate_context))
        object.__setattr__(self, "label", normalize_label(str(self.label)))
        object.__setattr__(self, "language", str(self.language or "unknown"))
        object.__setattr__(self, "priority", float(self.priority))
        object.__setattr__(self, "uncertainty", float(self.uncertainty))
        object.__setattr__(self, "gate_impact", float(self.gate_impact))
        object.__setattr__(self, "provenance", _plain(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready PHI-free queue item."""
        return {
            "gate_impact": round(self.gate_impact, 6),
            "label": self.label,
            "language": self.language,
            "priority": round(self.priority, 6),
            "provenance": _plain(self.provenance),
            "span_hash": self.span_hash,
            "surrogate_context": self.surrogate_context,
            "uncertainty": round(self.uncertainty, 6),
        }


@dataclass(frozen=True)
class LabelingQueueArtifact:
    """PHI-free labeling queue mined from gate-failure error examples."""

    gate_run_hash: str
    report_hash: str
    items: Sequence[LabelingQueueItem]
    raw_candidate_count: int
    dropped_duplicate_count: int
    schema_version: str = LABELING_QUEUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_run_hash", str(self.gate_run_hash))
        object.__setattr__(self, "report_hash", str(self.report_hash))
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "raw_candidate_count", int(self.raw_candidate_count))
        object.__setattr__(
            self,
            "dropped_duplicate_count",
            int(self.dropped_duplicate_count),
        )

    @property
    def duplicate_reduction_rate(self) -> float:
        """Fraction of raw candidates removed by deduplication."""
        if self.raw_candidate_count <= 0:
            return 0.0
        return self.dropped_duplicate_count / self.raw_candidate_count

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready artifact payload."""
        payload: dict[str, Any] = {
            "dropped_duplicate_count": self.dropped_duplicate_count,
            "duplicate_reduction_rate": round(self.duplicate_reduction_rate, 6),
            "gate_run_hash": self.gate_run_hash,
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "raw_candidate_count": self.raw_candidate_count,
            "report_hash": self.report_hash,
            "schema_version": self.schema_version,
        }
        payload["artifact_hash"] = stable_hash(payload)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the artifact to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


@dataclass(frozen=True)
class _QueueCandidate:
    item: LabelingQueueItem
    kind: str
    matched_label: str


def mine_gate_failure_labeling_queue(
    report: ErrorAnalysisReport | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    gate_run_hash: str,
    report_hash: str | None = None,
    label_gate_impacts: Mapping[str, float] | None = None,
    max_items: int | None = None,
    dedupe_similarity: float = DEFAULT_DEDUPE_SIMILARITY,
) -> LabelingQueueArtifact:
    """Mine a PHI-free labeling queue from error-analysis examples.

    ``report`` may be an :class:`ErrorAnalysisReport`, a serialized report
    mapping, or a sequence of candidate mappings. Raw text-like fields are
    ignored; queue items keep only span hashes, surrogate-normalized context,
    label/language, score metadata, and audit provenance.
    """

    if not gate_run_hash:
        raise ValueError("gate_run_hash must be non-empty")
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative")
    if not 0.0 <= dedupe_similarity <= 1.0:
        raise ValueError("dedupe_similarity must be between 0 and 1")

    payload, raw_candidates = _coerce_queue_source(report)
    source_report_hash = report_hash or _source_report_hash(payload, raw_candidates)
    impacts = _label_gate_impacts(payload, label_gate_impacts)
    candidates = [
        _queue_candidate(
            candidate,
            gate_run_hash=gate_run_hash,
            report_hash=source_report_hash,
            label_gate_impacts=impacts,
        )
        for candidate in raw_candidates
    ]
    ranked = _rank_queue_candidates(candidates)
    deduped = _dedupe_queue_candidates(ranked, similarity=dedupe_similarity)
    dropped_duplicate_count = len(raw_candidates) - len(deduped)
    if max_items is not None:
        deduped = deduped[:max_items]

    return LabelingQueueArtifact(
        gate_run_hash=gate_run_hash,
        report_hash=source_report_hash,
        items=tuple(candidate.item for candidate in deduped),
        raw_candidate_count=len(raw_candidates),
        dropped_duplicate_count=dropped_duplicate_count,
    )


def _coerce_queue_source(
    report: ErrorAnalysisReport | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]]]:
    if isinstance(report, ErrorAnalysisReport):
        payload = report.to_dict()
        return payload, list(_iter_report_queue_candidates(payload))

    if isinstance(report, Mapping):
        payload = _plain(report)
        if _looks_like_error_report(payload):
            return payload, list(_iter_report_queue_candidates(payload))
        return None, [
            _candidate_with_source(payload, source_bucket="candidates", index=0)
        ]

    if isinstance(report, Sequence) and not isinstance(report, (str, bytes)):
        candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(report):
            if not isinstance(candidate, Mapping):
                raise TypeError("candidate sequences must contain mappings")
            candidates.append(
                _candidate_with_source(
                    _plain(candidate),
                    source_bucket="candidates",
                    index=index,
                )
            )
        return None, candidates

    raise TypeError(
        "report must be an ErrorAnalysisReport, a mapping, or a sequence of mappings"
    )


def _looks_like_error_report(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in ("false_negatives", "false_positives"))


def _iter_report_queue_candidates(
    payload: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    language = _report_language(payload)
    suite = str(payload.get("suite", ""))
    model_name = str(payload.get("model_name", ""))
    index = 0
    for source_bucket in ("false_negatives", "false_positives"):
        by_label = payload.get(source_bucket) or {}
        if not isinstance(by_label, Mapping):
            continue
        for label in sorted(by_label):
            examples = by_label.get(label) or ()
            if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
                continue
            for example in examples:
                if not isinstance(example, Mapping):
                    continue
                candidate = _candidate_with_source(
                    example,
                    source_bucket=source_bucket,
                    index=index,
                    language=language,
                    suite=suite,
                    model_name=model_name,
                )
                candidate.setdefault("label", label)
                yield candidate
                index += 1


def _candidate_with_source(
    candidate: Mapping[str, Any],
    *,
    source_bucket: str,
    index: int,
    language: str = "unknown",
    suite: str = "",
    model_name: str = "",
) -> dict[str, Any]:
    payload = dict(candidate)
    payload["_source_bucket"] = source_bucket
    payload["_example_index"] = index
    payload.setdefault("_language", language)
    if suite:
        payload["_suite"] = suite
    if model_name:
        payload["_model_name"] = model_name
    return payload


def _report_language(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, Mapping):
        for key in ("language", "lang"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        languages = metadata.get("languages")
        if isinstance(languages, Sequence) and not isinstance(languages, (str, bytes)):
            values = [str(value) for value in languages if str(value)]
            if len(values) == 1:
                return values[0]
    return "unknown"


def _source_report_hash(
    payload: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    if payload is not None:
        return stable_hash(payload)
    return stable_hash([_candidate_safe_payload(candidate) for candidate in candidates])


def _label_gate_impacts(
    payload: Mapping[str, Any] | None,
    overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    impacts: dict[str, float] = {}
    if payload is not None:
        matrix = payload.get("confusion_matrix") or {}
        if isinstance(matrix, Mapping):
            counts: dict[str, int] = {}
            for label, row in matrix.items():
                if label in ERROR_BUCKETS or not isinstance(row, Mapping):
                    continue
                normal_label = normalize_label(str(label))
                errors = sum(
                    int(count)
                    for predicted, count in row.items()
                    if predicted != normal_label
                )
                counts[normal_label] = counts.get(normal_label, 0) + errors
            spurious_row = matrix.get(SPURIOUS) or {}
            if isinstance(spurious_row, Mapping):
                for label, count in spurious_row.items():
                    normal_label = normalize_label(str(label))
                    counts[normal_label] = counts.get(normal_label, 0) + int(count)
            max_count = max(counts.values(), default=0)
            if max_count:
                impacts.update(
                    {
                        label: 1.0 + (count / max_count)
                        for label, count in counts.items()
                    }
                )

    for label, impact in (overrides or {}).items():
        impacts[normalize_label(str(label))] = max(float(impact), 0.0)
    return impacts


def _queue_candidate(
    data: Mapping[str, Any],
    *,
    gate_run_hash: str,
    report_hash: str,
    label_gate_impacts: Mapping[str, float],
) -> _QueueCandidate:
    label = normalize_label(str(data.get("label") or "OTHER"))
    kind = str(data.get("kind") or data.get("failure_kind") or "failure")
    matched_label = _normalised_optional_label(data.get("matched_label"))
    language = str(data.get("language") or data.get("lang") or data.get("_language"))
    span_hash = _candidate_span_hash(data, label=label, kind=kind)
    uncertainty = _candidate_uncertainty(data, kind=kind)
    gate_impact = _candidate_gate_impact(
        data,
        label=label,
        label_gate_impacts=label_gate_impacts,
    )
    surrogate_context = _candidate_surrogate_context(
        data,
        label=label,
        kind=kind,
        matched_label=matched_label,
    )
    provenance = _candidate_provenance(
        data,
        gate_run_hash=gate_run_hash,
        report_hash=report_hash,
        span_hash=span_hash,
        label=label,
        kind=kind,
        matched_label=matched_label,
    )
    item = LabelingQueueItem(
        span_hash=span_hash,
        surrogate_context=surrogate_context,
        label=label,
        language=language,
        priority=uncertainty * gate_impact,
        uncertainty=uncertainty,
        gate_impact=gate_impact,
        provenance=provenance,
        _dedupe_context=_normalise_context_for_dedupe(surrogate_context),
    )
    return _QueueCandidate(item=item, kind=kind, matched_label=matched_label)


def _normalised_optional_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return normalize_label(text) if text else ""


def _candidate_span_hash(
    data: Mapping[str, Any],
    *,
    label: str,
    kind: str,
) -> str:
    for key in ("span_hash", "text_hash"):
        value = data.get(key)
        if value:
            return str(value)
    return stable_hash(
        {
            "end": _optional_int(data.get("end")),
            "kind": kind,
            "label": label,
            "start": _optional_int(data.get("start")),
        }
    )


def _candidate_uncertainty(data: Mapping[str, Any], *, kind: str) -> float:
    for key in ("uncertainty", "model_uncertainty"):
        if key in data:
            return _bounded_float(data.get(key), default=1.0, minimum=0.0)
    if kind == MISSED:
        return 1.0
    if kind == "label_confusion":
        return 0.85
    if kind == SPURIOUS:
        return 0.65
    return 0.75


def _candidate_gate_impact(
    data: Mapping[str, Any],
    *,
    label: str,
    label_gate_impacts: Mapping[str, float],
) -> float:
    if "gate_impact" in data:
        return _bounded_float(data.get("gate_impact"), default=1.0, minimum=0.0)
    return float(label_gate_impacts.get(label, 1.0))


def _bounded_float(value: Any, *, default: float, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(number, minimum)


def _candidate_surrogate_context(
    data: Mapping[str, Any],
    *,
    label: str,
    kind: str,
    matched_label: str,
) -> str:
    for key in (
        "surrogate_context",
        "context_template",
        "normalized_context",
        "normalised_context",
    ):
        value = data.get(key)
        if value:
            return _safe_surrogate_context(str(value))
    return _default_surrogate_context(
        data,
        label=label,
        kind=kind,
        matched_label=matched_label,
    )


def _safe_surrogate_context(value: str) -> str:
    parts = _PLACEHOLDER_RE.split(value)
    safe_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if _PLACEHOLDER_RE.fullmatch(part):
            safe_parts.append(_canonical_placeholder(part[1:-1]))
        else:
            safe_parts.append(_mask_context_text(part))
    safe = " ".join(part.strip() for part in safe_parts if part.strip())
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe or "<CONTEXT>"


def _mask_context_text(value: str) -> str:
    safe = _EMAIL_RE.sub("<EMAIL>", value)
    safe = _ID_RE.sub("<ID_NUM>", safe)
    safe = _DATE_RE.sub("<DATE>", safe)
    safe = _PHONE_RE.sub("<PHONE>", safe)
    safe = _NUMBER_RE.sub("<NUM>", safe)
    safe = _mask_words_outside_placeholders(safe)
    return re.sub(r"\s+", " ", safe)


def _mask_words_outside_placeholders(value: str) -> str:
    parts = _PLACEHOLDER_RE.split(value)
    safe_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if _PLACEHOLDER_RE.fullmatch(part):
            safe_parts.append(_canonical_placeholder(part[1:-1]))
        else:
            safe_parts.append(_WORD_RE.sub("<WORD>", part))
    return "".join(safe_parts)


def _canonical_placeholder(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_:-]+", "_", value).strip("_").upper()
    return f"<{token or 'TOKEN'}>"


def _default_surrogate_context(
    data: Mapping[str, Any],
    *,
    label: str,
    kind: str,
    matched_label: str,
) -> str:
    span_len = _span_length(data.get("start"), data.get("end"))
    context_len = _span_length(data.get("context_start"), data.get("context_end"))
    parts = [
        _canonical_placeholder(label),
        _canonical_placeholder(kind),
        f"span_chars:{span_len}",
        f"context_chars:{context_len}",
    ]
    if matched_label:
        parts.append(_canonical_placeholder(f"matched_{matched_label}"))
    return " ".join(parts)


def _span_length(start: Any, end: Any) -> int:
    start_int = _optional_int(start)
    end_int = _optional_int(end)
    if start_int is None or end_int is None:
        return 0
    return max(end_int - start_int, 0)


def _candidate_provenance(
    data: Mapping[str, Any],
    *,
    gate_run_hash: str,
    report_hash: str,
    span_hash: str,
    label: str,
    kind: str,
    matched_label: str,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "example_hash": stable_hash(_candidate_safe_payload(data)),
        "example_index": int(data.get("_example_index", 0)),
        "gate_run_hash": gate_run_hash,
        "kind": kind,
        "label": label,
        "report_hash": report_hash,
        "source": "error_analysis",
        "source_bucket": str(data.get("_source_bucket") or "candidates"),
        "span_hash": span_hash,
    }
    fixture_id = data.get("fixture_id")
    if fixture_id:
        provenance["fixture_hash"] = stable_hash({"fixture_id": str(fixture_id)})
    for key in ("start", "end", "context_start", "context_end"):
        value = _optional_int(data.get(key))
        if value is not None:
            provenance[key] = value
    if matched_label:
        provenance["matched_label"] = matched_label
    for key in ("matched_start", "matched_end"):
        value = _optional_int(data.get(key))
        if value is not None:
            provenance[key] = value
    matched_hash = data.get("matched_text_hash")
    if matched_hash:
        provenance["matched_span_hash"] = str(matched_hash)
    suite = data.get("_suite")
    if suite:
        provenance["suite_hash"] = stable_hash({"suite": str(suite)})
    model_name = data.get("_model_name")
    if model_name:
        provenance["model_hash"] = stable_hash({"model_name": str(model_name)})
    return provenance


def _candidate_safe_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "_example_index",
        "_source_bucket",
        "context_end",
        "context_start",
        "end",
        "kind",
        "label",
        "matched_end",
        "matched_label",
        "matched_start",
        "matched_text_hash",
        "span_hash",
        "start",
        "text_hash",
    )
    return {
        key: _plain(data[key])
        for key in allowed
        if key in data and key not in _RAW_TEXT_KEYS
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rank_queue_candidates(
    candidates: Sequence[_QueueCandidate],
) -> list[_QueueCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.item.priority,
            candidate.item.label,
            candidate.kind,
            candidate.matched_label,
            candidate.item.span_hash,
            stable_hash(candidate.item.provenance),
        ),
    )


def _dedupe_queue_candidates(
    candidates: Sequence[_QueueCandidate],
    *,
    similarity: float,
) -> list[_QueueCandidate]:
    kept: list[_QueueCandidate] = []
    for candidate in candidates:
        if any(
            _same_failure_mode(candidate, existing, similarity=similarity)
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _same_failure_mode(
    candidate: _QueueCandidate,
    existing: _QueueCandidate,
    *,
    similarity: float,
) -> bool:
    if candidate.item.span_hash == existing.item.span_hash:
        return True
    if (
        candidate.item.label,
        candidate.item.language,
        candidate.kind,
        candidate.matched_label,
    ) != (
        existing.item.label,
        existing.item.language,
        existing.kind,
        existing.matched_label,
    ):
        return False
    return (
        _context_similarity(
            candidate.item._dedupe_context, existing.item._dedupe_context
        )
        >= similarity
    )


def _normalise_context_for_dedupe(value: str) -> str:
    return " ".join(_context_tokens(value))


def _context_similarity(a: str, b: str) -> float:
    left = set(_context_tokens(a))
    right = set(_context_tokens(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _context_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.upper()))


def error_report(
    model: str | ModelRunner,
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    suite_name: str | None = None,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    example_cap: int = DEFAULT_EXAMPLE_CAP,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ErrorAnalysisReport:
    """Run *model* on *suite* and build a per-label error-analysis report.

    ``model`` may be a model name or a callable using the benchmark runner
    signature. Passing ``runner`` keeps tests and local evals offline-friendly
    while preserving the default harness behavior for real model names.
    """
    if example_cap < 0:
        raise ValueError("example_cap must be non-negative")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    fixtures = _coerce_fixtures(suite)
    model_name, model_runner = _resolve_model_runner(model, runner)
    report_suite = suite_name or _suite_name(suite)

    matrix = _empty_matrix()
    false_negatives = _empty_examples()
    false_positives = _empty_examples()

    for fixture in fixtures:
        raw_predictions = list(model_runner(fixture, model_name, device))
        predicted_spans = normalize_eval_spans(
            raw_predictions,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        _accumulate_fixture_errors(
            fixture=fixture,
            predicted_spans=predicted_spans,
            matrix=matrix,
            false_negatives=false_negatives,
            false_positives=false_positives,
            example_cap=example_cap,
            context_window=context_window,
        )

    return ErrorAnalysisReport(
        suite=report_suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        confusion_matrix=matrix,
        false_negatives=false_negatives,
        false_positives=false_positives,
        example_cap=example_cap,
        generated_at=generated_at,
        metadata=dict(metadata or {}),
    )


def _accumulate_fixture_errors(
    *,
    fixture: BenchmarkFixture,
    predicted_spans: Sequence[EvalSpan],
    matrix: dict[str, dict[str, int]],
    false_negatives: dict[str, list[ErrorSpanExample]],
    false_positives: dict[str, list[ErrorSpanExample]],
    example_cap: int,
    context_window: int,
) -> None:
    matched_predictions: set[int] = set()
    gold_spans = _ordered_spans(fixture.gold_spans)
    ordered_predictions = _ordered_indexed_spans(predicted_spans)

    for gold_span in gold_spans:
        match = _best_overlapping_prediction(
            gold_span,
            ordered_predictions,
            matched_predictions,
        )
        if match is None:
            matrix[gold_span.label][MISSED] += 1
            _append_example(
                false_negatives[gold_span.label],
                _example(
                    kind=MISSED,
                    fixture=fixture,
                    span=gold_span,
                    context_window=context_window,
                ),
                example_cap,
            )
            continue

        pred_index, pred_span = match
        matched_predictions.add(pred_index)
        matrix[gold_span.label][pred_span.label] += 1
        if gold_span.label == pred_span.label:
            continue

        _append_example(
            false_negatives[gold_span.label],
            _example(
                kind="label_confusion",
                fixture=fixture,
                span=gold_span,
                context_window=context_window,
                matched_span=pred_span,
            ),
            example_cap,
        )
        _append_example(
            false_positives[pred_span.label],
            _example(
                kind="label_confusion",
                fixture=fixture,
                span=pred_span,
                context_window=context_window,
                matched_span=gold_span,
            ),
            example_cap,
        )

    for pred_index, pred_span in ordered_predictions:
        if pred_index in matched_predictions:
            continue
        matrix[SPURIOUS][pred_span.label] += 1
        _append_example(
            false_positives[pred_span.label],
            _example(
                kind=SPURIOUS,
                fixture=fixture,
                span=pred_span,
                context_window=context_window,
            ),
            example_cap,
        )


def _best_overlapping_prediction(
    gold_span: EvalSpan,
    predictions: Sequence[tuple[int, EvalSpan]],
    matched_predictions: set[int],
) -> tuple[int, EvalSpan] | None:
    candidates = [
        (index, pred_span)
        for index, pred_span in predictions
        if index not in matched_predictions and _spans_overlap(gold_span, pred_span)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: _match_key(gold_span, item[0], item[1]),
    )


def _match_key(gold_span: EvalSpan, index: int, pred_span: EvalSpan) -> tuple[Any, ...]:
    return (
        _overlap_len(gold_span, pred_span),
        pred_span.start == gold_span.start and pred_span.end == gold_span.end,
        pred_span.label == gold_span.label,
        -abs(pred_span.start - gold_span.start),
        -abs(pred_span.end - gold_span.end),
        -index,
    )


def _spans_overlap(a: EvalSpan, b: EvalSpan) -> bool:
    return bool(detect_overlapping_entities([a.to_entity(), b.to_entity()]))


def _overlap_len(a: EvalSpan, b: EvalSpan) -> int:
    if not _spans_overlap(a, b):
        return 0
    return max(min(a.end, b.end) - max(a.start, b.start), 0)


def _example(
    *,
    kind: str,
    fixture: BenchmarkFixture,
    span: EvalSpan,
    context_window: int,
    matched_span: EvalSpan | None = None,
) -> ErrorSpanExample:
    context_start = max(0, span.start - context_window)
    context_end = min(len(fixture.text), span.end + context_window)
    return ErrorSpanExample(
        kind=kind,
        fixture_id=fixture.fixture_id,
        label=span.label,
        start=span.start,
        end=span.end,
        context_start=context_start,
        context_end=context_end,
        text_hash=_span_hash(fixture.text, span),
        matched_label=(matched_span.label if matched_span is not None else None),
        matched_start=(matched_span.start if matched_span is not None else None),
        matched_end=(matched_span.end if matched_span is not None else None),
        matched_text_hash=(
            _span_hash(fixture.text, matched_span) if matched_span is not None else None
        ),
    )


def _span_hash(text: str, span: EvalSpan) -> str:
    value = span.text
    if 0 <= span.start <= span.end <= len(text):
        value = text[span.start : span.end]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _append_example(
    examples: list[ErrorSpanExample],
    example: ErrorSpanExample,
    example_cap: int,
) -> None:
    if len(examples) < example_cap:
        examples.append(example)


def _coerce_fixtures(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
) -> list[BenchmarkFixture]:
    if isinstance(suite, (str, Path)):
        return load_fixtures(suite)
    return [
        fixture
        if isinstance(fixture, BenchmarkFixture)
        else BenchmarkFixture.from_mapping(fixture)
        for fixture in suite
    ]


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner]:
    if runner is not None:
        return str(model), runner
    if callable(model):
        return _callable_name(model), model
    return str(model), default_model_runner


def _callable_name(value: ModelRunner) -> str:
    name = getattr(value, "__name__", "")
    if name and name != "<lambda>":
        return str(name)
    try:
        return value.__class__.__name__
    except AttributeError:
        return "model"


def _suite_name(
    suite: str | Path | Sequence[BenchmarkFixture | Mapping[str, Any]],
) -> str:
    if isinstance(suite, (str, Path)):
        path = Path(suite)
        return path.stem or str(path)
    return "suite"


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {
        row_label: {column_label: 0 for column_label in MATRIX_LABELS}
        for row_label in MATRIX_LABELS
    }


def _matrix_row(row: Mapping[str, int]) -> dict[str, int]:
    return {
        column_label: int(row.get(column_label, 0)) for column_label in MATRIX_LABELS
    }


def _empty_examples() -> dict[str, list[ErrorSpanExample]]:
    return {label: [] for label in LABELS}


def _ordered_spans(spans: Iterable[EvalSpan]) -> list[EvalSpan]:
    return sorted(
        spans,
        key=lambda span: (span.start, span.end, span.label, span.text),
    )


def _ordered_indexed_spans(spans: Sequence[EvalSpan]) -> list[tuple[int, EvalSpan]]:
    return sorted(
        enumerate(spans),
        key=lambda item: (
            item[1].start,
            item[1].end,
            item[1].label,
            item[1].text,
            item[0],
        ),
    )


def _examples_to_dict(
    examples: Mapping[str, Sequence[ErrorSpanExample]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        label: [example.to_dict() for example in examples.get(label, ())]
        for label in LABELS
    }


def _examples_markdown(
    title: str,
    examples: Mapping[str, Sequence[ErrorSpanExample]],
) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Label | Kind | Fixture | Span | Context | Matched | Text Hash |",
        "|---|---|---|---|---|---|---|",
    ]
    has_rows = False
    for label in LABELS:
        for example in examples.get(label, ()):
            has_rows = True
            matched = ""
            if example.matched_label is not None:
                matched = (
                    f"`{example.matched_label}` "
                    f"{example.matched_start}:{example.matched_end}"
                )
            lines.append(
                "| "
                f"`{label}` | "
                f"`{example.kind}` | "
                f"`{example.fixture_id}` | "
                f"{example.start}:{example.end} | "
                f"{example.context_start}:{example.context_end} | "
                f"{matched} | "
                f"`{example.text_hash}` |"
            )
    if not has_rows:
        lines.append("| _None_ |  |  |  |  |  |  |")
    return lines


def _flatten(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    for key in sorted(payload):
        value = payload[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten(value, prefix=path)
        else:
            yield path, value


__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_DEDUPE_SIMILARITY",
    "DEFAULT_EXAMPLE_CAP",
    "ERROR_BUCKETS",
    "LABELS",
    "LABELING_QUEUE_SCHEMA_VERSION",
    "MATRIX_LABELS",
    "MISSED",
    "SPURIOUS",
    "ErrorAnalysisReport",
    "ErrorSpanExample",
    "LabelingQueueArtifact",
    "LabelingQueueItem",
    "error_report",
    "mine_gate_failure_labeling_queue",
]
