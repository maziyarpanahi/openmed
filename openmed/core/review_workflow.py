"""Human-in-the-loop review workflow for extraction and de-identification output.

This module turns an extraction or de-identification result into an actionable,
PHI-safe **review queue** and captures a reviewer's decision as an appendable
**feedback record**. It is a decision *aid*, not a decision *maker*.

Design posture (medical-device disclaimer):
    OpenMed de-identification and clinical extraction are assistive. This module
    exists precisely so that a human stays in the loop for the spans where
    automation is weakest — low-confidence predictions and high-stakes direct
    identifiers. Nothing here auto-triggers a clinical decision, a release, or a
    threshold change. The queue *surfaces* candidates and the feedback API
    *records* what a human decided; acting on those decisions remains a human
    responsibility.

Privacy posture (local-first, no raw PHI):
    Neither the review queue nor the feedback record stores raw PHI. Following
    the same offsets-first contract as :mod:`openmed.core.audit` and
    :mod:`openmed.core.redaction_preview`, items and records carry offsets,
    (canonical) labels, confidence, thresholds, SHA-256 hashes, provenance, and
    a *redacted* context window whose identifier span is masked. Feedback
    records are JSONL-appendable so a local stream of reviewer decisions can be
    persisted without ever writing plaintext identifiers to disk.

Example:
    >>> from datetime import datetime
    >>> from openmed.core.pii import DeidentificationResult, PIIEntity
    >>> from openmed.core.review_workflow import build_review_queue
    >>> result = DeidentificationResult(
    ...     original_text="Patient Jane Roe, SSN 123-45-6789.",
    ...     deidentified_text="Patient [NAME], SSN [SSN].",
    ...     pii_entities=[
    ...         PIIEntity(
    ...             text="Jane Roe", label="NAME", start=8, end=16,
    ...             confidence=0.40, entity_type="NAME",
    ...         ),
    ...         PIIEntity(
    ...             text="123-45-6789", label="SSN", start=22, end=33,
    ...             confidence=0.99, entity_type="SSN",
    ...         ),
    ...     ],
    ...     method="mask",
    ...     timestamp=datetime(2026, 7, 5),
    ... )
    >>> queue = build_review_queue(result, confidence_threshold=0.7)
    >>> [(item.label, sorted(item.reasons)) for item in queue.items]
    [('NAME', ['critical_label', 'low_confidence']), ('SSN', ['critical_label'])]
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .audit import hash_text
from .labels import (
    DIRECT_IDENTIFIER,
    LABEL_METADATA,
    RISK_HIGH,
    normalize_label,
)

__all__ = [
    "REVIEW_REASON_LOW_CONFIDENCE",
    "REVIEW_REASON_CRITICAL_LABEL",
    "REVIEW_DECISIONS",
    "ReviewItem",
    "ReviewQueue",
    "ReviewFeedback",
    "critical_labels",
    "build_review_queue",
    "record_review_decision",
    "append_feedback",
]

_MISSING = object()

#: Reason emitted when an entity's confidence is below the review threshold.
REVIEW_REASON_LOW_CONFIDENCE = "low_confidence"

#: Reason emitted when an entity carries a critical (direct-identifier) label.
REVIEW_REASON_CRITICAL_LABEL = "critical_label"

#: Valid reviewer decisions captured by :func:`record_review_decision`.
REVIEW_DECISIONS = ("accept", "reject", "correct")

_CONTEXT_WINDOW = 32
_MASK = "[REDACTED]"


def critical_labels() -> frozenset[str]:
    """Return the default set of critical (high-stakes) canonical labels.

    A label is critical when its policy class is ``DIRECT_IDENTIFIER`` or its
    residual-risk level is ``high`` in
    :data:`openmed.core.labels.LABEL_METADATA`. These are the identifiers whose
    leakage is most damaging, so they always warrant a human look regardless of
    model confidence.

    Returns:
        A frozenset of canonical ``UPPER_SNAKE_CASE`` labels.
    """
    return frozenset(
        label
        for label, metadata in LABEL_METADATA.items()
        if metadata["policy_label"] == DIRECT_IDENTIFIER
        or metadata["risk_level"] == RISK_HIGH
    )


@dataclass(frozen=True)
class ReviewItem:
    """One extraction span flagged for human review.

    Attributes:
        start: Character start offset in the source document.
        end: Character end offset in the source document.
        label: The entity's source label as emitted by the detector.
        canonical_label: Normalized ``UPPER_SNAKE_CASE`` label.
        confidence: Model confidence for the span (0-1).
        threshold: The confidence threshold the span was compared against.
        reasons: Machine-readable review reasons (``low_confidence`` and/or
            ``critical_label``); always sorted and non-empty unless
            ``include_all`` was requested.
        text_hash: SHA-256 hash of the original span text (never the plaintext).
        context: A redacted context window ``{"before": ..., "after": ...,
            "span": "[REDACTED]"}`` around the span.
        action: The redaction action applied to the span, when known.
    """

    start: int
    end: int
    label: str
    canonical_label: str
    confidence: float
    threshold: float
    reasons: tuple[str, ...]
    text_hash: str
    context: dict[str, str] = field(default_factory=dict)
    action: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, PHI-safe representation of the item."""
        payload: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
            "reasons": list(self.reasons),
            "text_hash": self.text_hash,
            "context": dict(self.context),
        }
        if self.action is not None:
            payload["action"] = self.action
        return payload


@dataclass(frozen=True)
class ReviewQueue:
    """An ordered, PHI-safe queue of spans needing human review.

    Attributes:
        items: Review items in deterministic ``(start, end, canonical_label)``
            order.
        confidence_threshold: The threshold used to flag low-confidence spans.
        critical_labels: The critical canonical labels used for selection.
        document_hash: SHA-256 hash of the source document text, when available.
        total_spans: Number of candidate spans considered before filtering.
    """

    items: tuple[ReviewItem, ...]
    confidence_threshold: float
    critical_labels: frozenset[str]
    document_hash: Optional[str] = None
    total_spans: int = 0

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def review_count(self) -> int:
        """Number of spans flagged for review."""
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, PHI-safe representation of the queue."""
        return {
            "confidence_threshold": float(self.confidence_threshold),
            "critical_labels": sorted(self.critical_labels),
            "document_hash": self.document_hash,
            "total_spans": int(self.total_spans),
            "review_count": self.review_count,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class ReviewFeedback:
    """A PHI-safe, JSONL-appendable record of one reviewer decision.

    The record captures *what a human decided* about a flagged span without ever
    storing raw PHI. Corrections carry the corrected label and, for boundary
    fixes, corrected offsets — never corrected plaintext.

    Attributes:
        start: Character start offset of the reviewed span.
        end: Character end offset of the reviewed span.
        label: The label originally proposed by the detector.
        canonical_label: Normalized ``UPPER_SNAKE_CASE`` original label.
        confidence: The original model confidence.
        reasons: Why the span was queued for review.
        decision: The reviewer's decision (``accept``/``reject``/``correct``).
        text_hash: SHA-256 hash of the original span text.
        corrected_label: For ``correct`` decisions, the reviewer's label (source
            form). ``None`` otherwise.
        corrected_canonical_label: Normalized corrected label. ``None`` when no
            label correction was made.
        corrected_start: Corrected start offset for a boundary fix, if any.
        corrected_end: Corrected end offset for a boundary fix, if any.
        reviewer_id: Optional non-PHI reviewer identifier (e.g. a role or a
            pseudonymous handle). Not intended to carry patient data.
        note_hash: SHA-256 hash of an optional reviewer note. The plaintext note
            is never stored, so a note cannot leak PHI into the record.
        timestamp: ISO-8601 UTC timestamp of when the decision was recorded.
    """

    start: int
    end: int
    label: str
    canonical_label: str
    confidence: float
    reasons: tuple[str, ...]
    decision: str
    text_hash: str
    corrected_label: Optional[str] = None
    corrected_canonical_label: Optional[str] = None
    corrected_start: Optional[int] = None
    corrected_end: Optional[int] = None
    reviewer_id: Optional[str] = None
    note_hash: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, PHI-safe representation of the record."""
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "confidence": float(self.confidence),
            "reasons": list(self.reasons),
            "decision": self.decision,
            "text_hash": self.text_hash,
            "corrected_label": self.corrected_label,
            "corrected_canonical_label": self.corrected_canonical_label,
            "corrected_start": self.corrected_start,
            "corrected_end": self.corrected_end,
            "reviewer_id": self.reviewer_id,
            "note_hash": self.note_hash,
            "timestamp": self.timestamp,
        }

    def to_jsonl_line(self) -> str:
        """Return a single canonical JSON line (no trailing newline)."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewFeedback":
        """Reconstruct a feedback record from its serialized form."""
        reasons = data.get("reasons") or ()
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            label=str(data.get("label", "")),
            canonical_label=str(data.get("canonical_label", "")),
            confidence=float(data.get("confidence", 0.0)),
            reasons=tuple(str(reason) for reason in reasons),
            decision=str(data.get("decision", "")),
            text_hash=str(data.get("text_hash", "")),
            corrected_label=_optional_str(data.get("corrected_label")),
            corrected_canonical_label=_optional_str(
                data.get("corrected_canonical_label")
            ),
            corrected_start=_optional_int(data.get("corrected_start")),
            corrected_end=_optional_int(data.get("corrected_end")),
            reviewer_id=_optional_str(data.get("reviewer_id")),
            note_hash=_optional_str(data.get("note_hash")),
            timestamp=str(data.get("timestamp", "")),
        )


def build_review_queue(
    result: Any,
    *,
    confidence_threshold: float = 0.5,
    critical_labels: Optional[Iterable[str]] = None,
    lang: str = "en",
    include_all: bool = False,
) -> ReviewQueue:
    """Build a review queue from an extraction or de-identification result.

    Surfaces the spans a human should look at before the output is trusted:

    - **Low confidence** — the span's confidence is below
      ``confidence_threshold``.
    - **Critical label** — the span's canonical label is a critical
      (direct-identifier / high-risk) class.

    Args:
        result: A :class:`~openmed.core.pii.DeidentificationResult`, a
            :class:`~openmed.processing.outputs.PredictionResult`, either one's
            ``to_dict()`` payload, or any compatible object/mapping exposing
            ``pii_entities`` or ``entities`` (each entity exposing ``label``,
            ``confidence``, ``start``, and ``end``). This is the same
            duck-typed input contract used by
            :func:`~openmed.core.redaction_preview.redaction_preview`.
        confidence_threshold: Spans with confidence strictly below this value
            are flagged ``low_confidence``.
        critical_labels: Canonical labels treated as critical. When ``None``,
            defaults to :func:`critical_labels`. Values are normalized, so
            source-form labels (``ssn``, ``B-SSN``) are accepted.
        lang: ISO 639-1 language hint used when normalizing labels.
        include_all: When ``True``, include every span in the queue (each item
            still carries its computed reasons, which may be empty). When
            ``False`` (default), only spans with at least one reason are kept.

    Returns:
        A :class:`ReviewQueue` in deterministic order.

    Raises:
        TypeError: If ``result`` does not expose a span sequence.
        ValueError: If a span carries invalid offsets.
    """
    text = _result_text(result)
    critical = (
        globals()["critical_labels"]()
        if critical_labels is None
        else frozenset(normalize_label(label, lang) for label in critical_labels)
    )

    entities = _result_entities(result)

    # Collect every span up front so a context window can mask *other* nearby
    # identifier spans, not just the item's own span. A window around one span
    # may straddle a neighbouring name/ID; leaving that in plaintext would leak
    # PHI, so all spans are redacted from every window.
    spans: list[tuple[int, int]] = []
    for entity in entities:
        start, end = _entity_offsets(entity)
        if start < 0 or end < start:
            raise ValueError("entity offsets must be non-negative with end >= start")
        if text is not None and end > len(text):
            raise ValueError("entity offsets must fall within text")
        spans.append((start, end))

    items: list[ReviewItem] = []
    for entity, (start, end) in zip(entities, spans):
        label = str(_get_value(entity, "label", default=""))
        canonical = normalize_label(label, lang)
        confidence = float(_get_value(entity, "confidence", default=0.0) or 0.0)

        reasons: list[str] = []
        if confidence < confidence_threshold:
            reasons.append(REVIEW_REASON_LOW_CONFIDENCE)
        if canonical in critical:
            reasons.append(REVIEW_REASON_CRITICAL_LABEL)

        if not reasons and not include_all:
            continue

        span_text = text[start:end] if text is not None else ""
        items.append(
            ReviewItem(
                start=start,
                end=end,
                label=label,
                canonical_label=canonical,
                confidence=confidence,
                threshold=float(confidence_threshold),
                reasons=tuple(sorted(reasons)),
                text_hash=hash_text(span_text),
                context=_redacted_context(text, start, end, spans=spans),
                action=_optional_str(_get_value(entity, "action", default=None)),
            )
        )

    items.sort(key=lambda item: (item.start, item.end, item.canonical_label))
    return ReviewQueue(
        items=tuple(items),
        confidence_threshold=float(confidence_threshold),
        critical_labels=critical,
        document_hash=hash_text(text) if text is not None else None,
        total_spans=len(entities),
    )


def record_review_decision(
    item: ReviewItem,
    decision: str,
    *,
    corrected_label: Optional[str] = None,
    corrected_start: Optional[int] = None,
    corrected_end: Optional[int] = None,
    reviewer_id: Optional[str] = None,
    note: Optional[str] = None,
    lang: str = "en",
    timestamp: Optional[datetime] = None,
) -> ReviewFeedback:
    """Capture a reviewer's decision about a review item as a feedback record.

    Args:
        item: The :class:`ReviewItem` being reviewed.
        decision: One of :data:`REVIEW_DECISIONS` (``accept``/``reject``/
            ``correct``).
        corrected_label: For a ``correct`` decision, the reviewer's label
            (source form; normalized into the record). Required for ``correct``
            unless a boundary-only correction is being made.
        corrected_start: Optional corrected start offset for a boundary fix.
        corrected_end: Optional corrected end offset for a boundary fix.
        reviewer_id: Optional non-PHI reviewer identifier. Do not pass patient
            data here.
        note: Optional free-text reviewer note. **Never stored as plaintext** —
            only its SHA-256 hash is kept, so a note cannot leak PHI into the
            record.
        lang: ISO 639-1 language hint for normalizing the corrected label.
        timestamp: Optional decision timestamp; defaults to ``now`` in UTC.

    Returns:
        A PHI-safe :class:`ReviewFeedback` record.

    Raises:
        TypeError: If ``item`` is not a :class:`ReviewItem`.
        ValueError: If ``decision`` is invalid, if a ``correct`` decision
            carries neither a corrected label nor corrected offsets, or if
            corrected offsets are inconsistent.
    """
    if not isinstance(item, ReviewItem):
        raise TypeError("item must be a ReviewItem")
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {REVIEW_DECISIONS!r}")

    corrected_canonical: Optional[str] = None
    if decision == "correct":
        has_label = corrected_label is not None
        has_offsets = corrected_start is not None or corrected_end is not None
        if not has_label and not has_offsets:
            raise ValueError(
                "a 'correct' decision requires a corrected_label and/or "
                "corrected offsets"
            )
        if has_label:
            corrected_canonical = normalize_label(str(corrected_label), lang)
    else:
        if corrected_label is not None or (
            corrected_start is not None or corrected_end is not None
        ):
            raise ValueError(
                "corrected_label/offsets are only valid for a 'correct' decision"
            )

    if (corrected_start is None) != (corrected_end is None):
        raise ValueError("corrected_start and corrected_end must be set together")
    if (
        corrected_start is not None
        and corrected_end is not None
        and (corrected_start < 0 or corrected_end < corrected_start)
    ):
        raise ValueError("corrected offsets must be non-negative with end >= start")

    moment = timestamp or datetime.now(timezone.utc)
    return ReviewFeedback(
        start=item.start,
        end=item.end,
        label=item.label,
        canonical_label=item.canonical_label,
        confidence=item.confidence,
        reasons=item.reasons,
        decision=decision,
        text_hash=item.text_hash,
        corrected_label=(str(corrected_label) if corrected_label is not None else None),
        corrected_canonical_label=corrected_canonical,
        corrected_start=corrected_start,
        corrected_end=corrected_end,
        reviewer_id=_optional_str(reviewer_id),
        note_hash=(hash_text(note) if note is not None else None),
        timestamp=moment.isoformat(),
    )


def append_feedback(path: str | Path, record: ReviewFeedback) -> Path:
    """Append one feedback record to a local JSONL file.

    The file is created if it does not exist. Each call appends a single
    canonical JSON line. The written content contains no raw PHI.

    Args:
        path: Destination JSONL file path.
        record: The :class:`ReviewFeedback` record to append.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(record.to_jsonl_line())
        handle.write("\n")
    return destination


# ---------------------------------------------------------------------------
# Internal helpers (duck-typed accessors mirror redaction_preview.py)
# ---------------------------------------------------------------------------


def _redacted_context(
    text: Optional[str],
    start: int,
    end: int,
    *,
    size: int = _CONTEXT_WINDOW,
    spans: Optional[Sequence[tuple[int, int]]] = None,
) -> dict[str, str]:
    """Return a context window with all identifier spans masked.

    The window around one span may straddle a neighbouring identifier span, so
    every span in ``spans`` (plus the item's own span) is replaced with
    ``[REDACTED]``. This guarantees the ``before``/``after`` surfaces never carry
    plaintext PHI.
    """
    if text is None:
        return {"before": "", "after": "", "span": _MASK}

    before_start = max(0, start - size)
    after_end = min(len(text), end + size)

    all_spans = list(spans or ())
    all_spans.append((start, end))

    before = _mask_window(text, before_start, start, all_spans)
    after = _mask_window(text, end, after_end, all_spans)
    return {"before": before, "after": after, "span": _MASK}


def _mask_window(
    text: str,
    window_start: int,
    window_end: int,
    spans: Sequence[tuple[int, int]],
) -> str:
    """Return ``text[window_start:window_end]`` with overlapping spans masked."""
    if window_end <= window_start:
        return ""

    # Clip spans to the window, then mask each clipped region right-to-left so
    # earlier offsets stay valid while rebuilding the substring.
    clipped = sorted(
        {
            (max(span_start, window_start), min(span_end, window_end))
            for span_start, span_end in spans
            if span_start < window_end and span_end > window_start
        }
    )
    window = text[window_start:window_end]
    for span_start, span_end in reversed(clipped):
        rel_start = span_start - window_start
        rel_end = span_end - window_start
        window = window[:rel_start] + _MASK + window[rel_end:]
    return window


def _result_text(result: Any) -> Optional[str]:
    for key in ("original_text", "text"):
        value = _get_value(result, key, default=None)
        if isinstance(value, str):
            return value
    return None


def _result_entities(result: Any) -> list[Any]:
    entities = _get_value(result, "pii_entities", default=None)
    if entities is None:
        entities = _get_value(result, "entities", default=None)
    if entities is None:
        raise TypeError("result must expose 'pii_entities' or 'entities'")
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
        return list(entities)
    raise TypeError("result entities must be a sequence")


def _entity_offsets(entity: Any) -> tuple[int, int]:
    start = _get_value(entity, "start", default=None)
    end = _get_value(entity, "end", default=None)
    if start is None or end is None:
        raise ValueError("entity offsets are required")
    return int(start), int(end)


def _get_value(obj: Any, key: str, *, default: Any = _MISSING) -> Any:
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
    elif hasattr(obj, key):
        return getattr(obj, key)

    if default is _MISSING:
        raise KeyError(key)
    return default


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)
