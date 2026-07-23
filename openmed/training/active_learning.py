"""Active-learning queue runtime for release-gate and adjudication failures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label

DEFAULT_EVENT_LOG = Path(".openmed") / "active-learning-events.jsonl"
DEFAULT_RECALL_FLOOR = 0.99

CRITICAL_LABELS = frozenset(
    {
        "SSN",
        "ID_NUM",
        "API_KEY",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "CREDIT_CARD",
        "CVV",
        "IBAN",
        "BIC",
    }
)

_GATE_SPAN_KEYS = (
    "critical_leakage_spans",
    "gate_failure_spans",
    "failure_spans",
    "missed_spans",
    "candidate_spans",
    "spans",
    "predicted_spans",
)
_FIXTURE_KEYS = ("span_fixtures", "fixtures")


@dataclass(frozen=True)
class ActiveLearningCandidate:
    """A PHI-free span candidate queued for human labeling."""

    start: int
    end: int
    labels: tuple[str, ...]
    source: str
    priority: float
    reason: str
    record_id: str = ""
    model_sources: tuple[str, ...] = ()
    span_hash: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        labels = tuple(normalize_label(label) for label in self.labels)
        model_sources = tuple(
            sorted({source for source in self.model_sources if source})
        )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "model_sources", model_sources)
        if not self.span_hash:
            object.__setattr__(
                self,
                "span_hash",
                span_hash(
                    record_id=self.record_id,
                    start=self.start,
                    end=self.end,
                    labels=labels,
                ),
            )

    @property
    def label(self) -> str:
        """Return the primary label for callers expecting a single label."""

        return self.labels[0] if self.labels else "OTHER"

    def to_dict(self) -> dict[str, Any]:
        """Serialize without raw text or unbounded metadata."""

        return {
            "end": int(self.end),
            "label": self.label,
            "labels": list(self.labels),
            "model_sources": list(self.model_sources),
            "priority": round(float(self.priority), 6),
            "reason": self.reason,
            "record_id": self.record_id,
            "source": self.source,
            "span_hash": self.span_hash,
            "start": int(self.start),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActiveLearningCandidate":
        labels = data.get("labels") or [data.get("label", "OTHER")]
        if isinstance(labels, str):
            labels = [labels]
        model_sources = data.get("model_sources") or ()
        if isinstance(model_sources, str):
            model_sources = [model_sources]
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            labels=tuple(str(label) for label in labels),
            source=str(data.get("source", "")),
            priority=float(data.get("priority", 0.0)),
            reason=str(data.get("reason", "")),
            record_id=str(data.get("record_id", "")),
            model_sources=tuple(str(source) for source in model_sources),
            span_hash=str(data.get("span_hash", "")),
        )


class ActiveLearningQueue:
    """Persist and rank PHI-free active-learning candidates.

    The queue stores an append-only JSONL event log. ``queued`` events contain
    only coordinates, labels, source identifiers, and ranking evidence.
    ``labeled`` events tombstone a span hash so later ingestion does not put the
    same span back into the next labeling batch.
    """

    def __init__(
        self,
        event_log: str | Path | None = DEFAULT_EVENT_LOG,
        *,
        recall_floor: float = DEFAULT_RECALL_FLOOR,
    ) -> None:
        self.event_log = Path(event_log) if event_log is not None else None
        self.recall_floor = float(recall_floor)
        self._pending: dict[str, ActiveLearningCandidate] = {}
        self._labeled: set[str] = set()
        self._load()

    @property
    def pending(self) -> tuple[ActiveLearningCandidate, ...]:
        """Return pending candidates in deterministic priority order."""

        return tuple(self._ranked(self._pending.values()))

    @property
    def labeled_hashes(self) -> frozenset[str]:
        """Span hashes already marked labeled in the event log."""

        return frozenset(self._labeled)

    def add(self, candidate: ActiveLearningCandidate) -> bool:
        """Queue *candidate* if it has not already been queued or labeled."""

        if candidate.span_hash in self._labeled or candidate.span_hash in self._pending:
            return False
        self._pending[candidate.span_hash] = candidate
        self._append_event({"event": "queued", "candidate": candidate.to_dict()})
        return True

    def ingest_gate_report(self, report: Any) -> tuple[str, ...]:
        """Ingest release-gate report failures and return queued span hashes."""

        report = report.to_dict() if hasattr(report, "to_dict") else report
        if not isinstance(report, Mapping):
            raise TypeError("gate report must be a mapping or expose to_dict()")

        per_label_recall = _per_label_recall(report)
        critical_leakage_count = _critical_leakage_count(report)
        report_identity = _report_identity(report)
        queued: list[str] = []

        spans = list(_iter_report_spans(report))
        if not spans:
            spans = _label_level_spans(
                per_label_recall,
                critical_leakage_count=critical_leakage_count,
                record_id=report_identity,
                recall_floor=self.recall_floor,
            )

        for raw_span in spans:
            candidate = _candidate_from_gate_span(
                raw_span,
                per_label_recall=per_label_recall,
                critical_leakage_count=critical_leakage_count,
                recall_floor=self.recall_floor,
                fallback_record_id=report_identity,
            )
            if candidate is None:
                continue
            if self.add(candidate):
                queued.append(candidate.span_hash)
        return tuple(queued)

    def ingest_adjudication(
        self,
        item: Any,
        *,
        source: str = "weak_labeling_adjudication",
    ) -> tuple[str, ...]:
        """Ingest one OM-038c adjudication hook item."""

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, Mapping)):
            queued: list[str] = []
            for row in item:
                queued.extend(self.ingest_adjudication(row, source=source))
            return tuple(queued)

        data = item.to_dict() if hasattr(item, "to_dict") else item
        if not isinstance(data, Mapping):
            raise TypeError("adjudication item must be a mapping or expose to_dict()")

        priority = float(data.get("active_learning_priority", 1.0))
        reason = str(data.get("reason") or "inter_model_disagreement")
        record_id = str(data.get("record_id", ""))
        candidates = data.get("candidates") or data.get("spans") or ()
        grouped = _group_adjudication_candidates(candidates)

        queued: list[str] = []
        for start, end, labels, model_sources, max_score in grouped:
            score = _adjudication_priority(
                labels=labels,
                base_priority=priority,
                max_score=max_score,
            )
            candidate = ActiveLearningCandidate(
                start=start,
                end=end,
                labels=labels,
                source=source,
                priority=score,
                reason=reason,
                record_id=record_id,
                model_sources=model_sources,
            )
            if self.add(candidate):
                queued.append(candidate.span_hash)
        return tuple(queued)

    def mark_labeled(
        self,
        candidate: ActiveLearningCandidate | Mapping[str, Any] | str | None = None,
        *,
        record_id: str = "",
        start: int | None = None,
        end: int | None = None,
        label: str | None = None,
        labels: Iterable[str] | None = None,
    ) -> str:
        """Mark a candidate as labeled and return its span hash."""

        resolved_hash = _resolve_labeled_hash(
            candidate,
            record_id=record_id,
            start=start,
            end=end,
            label=label,
            labels=labels,
        )
        self._labeled.add(resolved_hash)
        self._pending.pop(resolved_hash, None)
        self._append_event({"event": "labeled", "span_hash": resolved_hash})
        return resolved_hash

    def next_batch(self, size: int) -> tuple[ActiveLearningCandidate, ...]:
        """Return at most *size* pending candidates by priority."""

        if size < 0:
            raise ValueError("size must be non-negative")
        return tuple(self._ranked(self._pending.values())[:size])

    def next_batch_dicts(self, size: int) -> list[dict[str, Any]]:
        """Return the next batch as JSON-ready PHI-free dictionaries."""

        return [candidate.to_dict() for candidate in self.next_batch(size)]

    def next_batch_jsonl(self, size: int) -> str:
        """Render the next batch as newline-delimited JSON."""

        lines = (
            json.dumps(candidate, sort_keys=True)
            for candidate in self.next_batch_dicts(size)
        )
        return "\n".join(lines)

    def write_next_batch_jsonl(self, path: str | Path, *, size: int) -> Path:
        """Write the next labeling batch JSONL to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.next_batch_jsonl(size)
        output_path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
        return output_path

    def _load(self) -> None:
        if self.event_log is None or not self.event_log.exists():
            return
        for line in self.event_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("event")
            if event_type == "labeled":
                span = str(event.get("span_hash", ""))
                if span:
                    self._labeled.add(span)
                    self._pending.pop(span, None)
                continue
            if event_type == "queued":
                candidate_data = event.get("candidate")
                if not isinstance(candidate_data, Mapping):
                    continue
                candidate = ActiveLearningCandidate.from_dict(candidate_data)
                if candidate.span_hash not in self._labeled:
                    self._pending[candidate.span_hash] = candidate

    def _append_event(self, event: Mapping[str, Any]) -> None:
        if self.event_log is None:
            return
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _ranked(
        candidates: Iterable[ActiveLearningCandidate],
    ) -> list[ActiveLearningCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.priority,
                candidate.record_id,
                candidate.start,
                candidate.end,
                candidate.labels,
                candidate.source,
            ),
        )


def span_hash(
    *,
    record_id: str,
    start: int,
    end: int,
    labels: Iterable[str],
) -> str:
    """Create a deterministic hash from PHI-free span identity fields."""

    payload = {
        "end": int(end),
        "labels": sorted({normalize_label(label) for label in labels}),
        "record_id": str(record_id),
        "start": int(start),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _candidate_from_gate_span(
    span: Mapping[str, Any],
    *,
    per_label_recall: Mapping[str, float],
    critical_leakage_count: int,
    recall_floor: float,
    fallback_record_id: str,
) -> ActiveLearningCandidate | None:
    try:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
    except (TypeError, ValueError):
        return None

    label = _span_label(span)
    if label is None:
        return None

    reason, priority = _gate_priority(
        label=label,
        explicit_reason=str(span.get("reason", "")),
        per_label_recall=per_label_recall,
        critical_leakage_count=critical_leakage_count,
        recall_floor=recall_floor,
    )
    if reason is None:
        return None

    record_id = str(
        span.get("record_id")
        or span.get("fixture_id")
        or span.get("document_id")
        or fallback_record_id
    )
    return ActiveLearningCandidate(
        start=start,
        end=end,
        labels=(label,),
        source="release_gate",
        priority=priority,
        reason=reason,
        record_id=record_id,
    )


def _gate_priority(
    *,
    label: str,
    explicit_reason: str,
    per_label_recall: Mapping[str, float],
    critical_leakage_count: int,
    recall_floor: float,
) -> tuple[str | None, float]:
    lowered_reason = explicit_reason.lower()
    recall = per_label_recall.get(label)
    gap = 0.0 if recall is None else max(0.0, recall_floor - recall)
    if "critical" in lowered_reason or (
        critical_leakage_count > 0 and label in CRITICAL_LABELS
    ):
        return "critical_leakage", 3000.0 + critical_leakage_count + gap * 100.0
    if recall is not None and recall < recall_floor:
        return "low_recall", 1000.0 + gap * 100.0
    if explicit_reason:
        return explicit_reason, 900.0
    return None, 0.0


def _adjudication_priority(
    *,
    labels: Sequence[str],
    base_priority: float,
    max_score: float,
) -> float:
    critical_bonus = 75.0 if any(label in CRITICAL_LABELS for label in labels) else 0.0
    uncertainty = max(0.0, 1.0 - max_score)
    return 2000.0 + float(base_priority) + critical_bonus + uncertainty * 100.0


def _group_adjudication_candidates(
    candidates: Any,
) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...], float]]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []

    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in candidates:
        data = raw if isinstance(raw, Mapping) else vars(raw)
        if not isinstance(data, Mapping):
            continue
        try:
            start = int(data["start"])
            end = int(data["end"])
        except (KeyError, TypeError, ValueError):
            continue
        label = _span_label(data)
        if label is None:
            continue
        sources = data.get("sources") or data.get("model_sources") or ()
        if isinstance(sources, str):
            sources = [sources]
        source = data.get("source")
        if source:
            sources = [*sources, source]
        score = _optional_float(data.get("score", data.get("confidence"))) or 0.0
        group = grouped.setdefault(
            (start, end),
            {"labels": set(), "sources": set(), "score": 0.0},
        )
        group["labels"].add(label)
        group["sources"].update(str(item) for item in sources if item)
        group["score"] = max(float(group["score"]), score)

    rows = []
    for (start, end), group in grouped.items():
        rows.append(
            (
                start,
                end,
                tuple(sorted(group["labels"])),
                tuple(sorted(group["sources"])),
                float(group["score"]),
            )
        )
    return rows


def _iter_report_spans(report: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = [report]
    metadata = report.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)

    for container in containers:
        for key in _GATE_SPAN_KEYS:
            yield from _iter_span_sequence(container.get(key), parent_record_id="")
        for fixture_key in _FIXTURE_KEYS:
            fixtures = container.get(fixture_key)
            if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
                continue
            for fixture in fixtures:
                if not isinstance(fixture, Mapping):
                    continue
                record_id = str(
                    fixture.get("record_id")
                    or fixture.get("fixture_id")
                    or fixture.get("id")
                    or ""
                )
                for key in _GATE_SPAN_KEYS:
                    yield from _iter_span_sequence(
                        fixture.get(key),
                        parent_record_id=record_id,
                    )


def _iter_span_sequence(
    value: Any,
    *,
    parent_record_id: str,
) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping) and {"start", "end"} <= set(value):
        payload = dict(value)
        payload.setdefault("record_id", parent_record_id)
        yield payload
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return
    for raw in value:
        data = raw if isinstance(raw, Mapping) else vars(raw)
        if not isinstance(data, Mapping):
            continue
        payload = dict(data)
        payload.setdefault("record_id", parent_record_id)
        yield payload


def _label_level_spans(
    per_label_recall: Mapping[str, float],
    *,
    critical_leakage_count: int,
    record_id: str,
    recall_floor: float,
) -> list[dict[str, Any]]:
    spans = [
        {
            "end": 0,
            "label": label,
            "reason": "low_recall",
            "record_id": record_id,
            "start": 0,
        }
        for label, recall in per_label_recall.items()
        if recall < recall_floor
    ]
    if critical_leakage_count > 0 and not any(
        span["label"] in CRITICAL_LABELS for span in spans
    ):
        spans.insert(
            0,
            {
                "end": 0,
                "label": "ID_NUM",
                "reason": "critical_leakage",
                "record_id": record_id,
                "start": 0,
            },
        )
    return spans


def _per_label_recall(report: Mapping[str, Any]) -> dict[str, float]:
    metrics = _mapping(report.get("metrics"))
    metadata = _mapping(report.get("metadata"))
    recall = (
        _mapping(report.get("per_label_recall"))
        or _mapping(metadata.get("per_label_recall"))
        or _mapping(metrics.get("per_label_recall"))
        or _mapping(metrics.get("recall_by_label"))
    )
    return {
        normalize_label(str(label)): float(value)
        for label, value in recall.items()
        if _optional_float(value) is not None
    }


def _critical_leakage_count(report: Mapping[str, Any]) -> int:
    metrics = _mapping(report.get("metrics"))
    metadata = _mapping(report.get("metadata"))
    for value in (
        report.get("critical_leakage_count"),
        metadata.get("critical_leakage_count"),
        metrics.get("critical_leakage_count"),
        _mapping(metrics.get("leakage")).get("critical_leakage_count"),
    ):
        parsed = _optional_float(value)
        if parsed is not None:
            return int(parsed)

    leaked_by_label = _mapping(
        _mapping(metrics.get("leakage")).get("leaked_chars_by_label")
    )
    return int(
        sum(
            float(value)
            for label, value in leaked_by_label.items()
            if normalize_label(str(label)) in CRITICAL_LABELS
            and _optional_float(value) is not None
        )
    )


def _report_identity(report: Mapping[str, Any]) -> str:
    metadata = _mapping(report.get("metadata"))
    return str(
        report.get("repo_id")
        or metadata.get("repo_id")
        or report.get("model_name")
        or metadata.get("model_name")
        or ""
    )


def _resolve_labeled_hash(
    candidate: ActiveLearningCandidate | Mapping[str, Any] | str | None,
    *,
    record_id: str,
    start: int | None,
    end: int | None,
    label: str | None,
    labels: Iterable[str] | None,
) -> str:
    if isinstance(candidate, ActiveLearningCandidate):
        return candidate.span_hash
    if isinstance(candidate, str) and candidate:
        return candidate
    if isinstance(candidate, Mapping):
        return ActiveLearningCandidate.from_dict(candidate).span_hash
    resolved_labels = tuple(labels or ([label] if label else ()))
    if start is None or end is None or not resolved_labels:
        raise ValueError("provide a candidate, span_hash, or record_id/start/end/label")
    return span_hash(
        record_id=record_id,
        start=start,
        end=end,
        labels=resolved_labels,
    )


def _span_label(data: Mapping[str, Any]) -> str | None:
    label = (
        data.get("canonical_label")
        or data.get("label")
        or data.get("entity_type")
        or data.get("entity_group")
    )
    if label is None:
        return None
    return normalize_label(str(label))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ActiveLearningCandidate",
    "ActiveLearningQueue",
    "CRITICAL_LABELS",
    "DEFAULT_EVENT_LOG",
    "DEFAULT_RECALL_FLOOR",
    "span_hash",
]
