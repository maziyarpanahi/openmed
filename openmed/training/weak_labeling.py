"""Teacher weak-labeling with agreement and adjudication routing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label

from .adjudication import (
    AdjudicationCandidate,
    AdjudicationItem,
    make_adjudication_item,
)

SpanValidator = Callable[["WeakLabelSpan"], bool]
ActiveLearningHook = Callable[[AdjudicationItem], None]


@dataclass(frozen=True)
class WeakLabelSpan:
    start: int
    end: int
    label: str
    text: str = ""
    score: float = 0.0
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int, str]:
        return self.start, self.end, self.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "label": self.label,
            "metadata": dict(self.metadata),
            "score": self.score,
            "source": self.source,
            "start": self.start,
            "text": self.text,
        }

    def to_candidate(
        self, sources: Iterable[str] | None = None
    ) -> AdjudicationCandidate:
        return AdjudicationCandidate(
            start=self.start,
            end=self.end,
            label=self.label,
            text=self.text,
            score=self.score,
            sources=tuple(
                sorted(set(sources or ([self.source] if self.source else [])))
            ),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class WeakLabelDecision:
    accepted_spans: tuple[WeakLabelSpan, ...]
    rejected_spans: tuple[WeakLabelSpan, ...]
    adjudication_items: tuple[AdjudicationItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_spans": [span.to_dict() for span in self.accepted_spans],
            "adjudication_items": [item.to_dict() for item in self.adjudication_items],
            "rejected_spans": [span.to_dict() for span in self.rejected_spans],
        }


def weak_label_document(
    text: str,
    detector_outputs: Mapping[str, Iterable[Any]],
    *,
    record_id: str = "",
    min_agreeing_models: int = 2,
    validators: Sequence[SpanValidator] = (),
    human_reviewed_spans: Iterable[Any] = (),
    active_learning_hook: ActiveLearningHook | None = None,
) -> WeakLabelDecision:
    """Accept only agreed or reviewed spans; route conflicts to adjudication."""

    if min_agreeing_models < 2:
        raise ValueError("min_agreeing_models must be at least 2")

    spans = _normalize_detector_outputs(detector_outputs, text=text)
    reviewed_keys = {
        span.key
        for span in (
            normalize_weak_span(raw, source="human_review", text=text)
            for raw in human_reviewed_spans
        )
    }

    by_key: dict[tuple[int, int, str], list[WeakLabelSpan]] = defaultdict(list)
    for span in spans:
        by_key[span.key].append(span)

    accepted: list[WeakLabelSpan] = []
    rejected: list[WeakLabelSpan] = []
    routed_keys: set[tuple[int, int, str]] = set()

    for key, grouped in sorted(by_key.items()):
        sources = {span.source for span in grouped if span.source}
        representative = _combine_group(grouped)
        if key in reviewed_keys or bool(representative.metadata.get("human_reviewed")):
            accepted.append(_with_metadata(representative, accepted_by="human_review"))
            continue
        if len(sources) >= min_agreeing_models and _validators_accept(
            representative, validators
        ):
            accepted.append(
                _with_metadata(representative, accepted_by="inter_model_agreement")
            )
            continue
        if len(sources) < min_agreeing_models:
            rejected.append(
                _with_metadata(representative, rejection_reason="single_model")
            )
        else:
            rejected.append(
                _with_metadata(representative, rejection_reason="validator_rejected")
            )

    adjudication_items: list[AdjudicationItem] = []
    for group in _disagreement_groups(spans):
        group_keys = {span.key for span in group}
        if group_keys & {span.key for span in accepted}:
            continue
        if group_keys <= routed_keys:
            continue
        item = make_adjudication_item(
            text=text,
            candidates=[
                span.to_candidate(_sources_for_key(by_key, span.key)) for span in group
            ],
            reason="inter_model_disagreement",
            record_id=record_id,
            metadata={"hard_negative_seed": True},
        )
        adjudication_items.append(item)
        routed_keys.update(group_keys)
        if active_learning_hook is not None:
            active_learning_hook(item)

    return WeakLabelDecision(
        accepted_spans=tuple(accepted),
        rejected_spans=tuple(rejected),
        adjudication_items=tuple(adjudication_items),
    )


def normalize_weak_span(raw: Any, *, source: str = "", text: str = "") -> WeakLabelSpan:
    if isinstance(raw, WeakLabelSpan):
        return raw
    data = raw if isinstance(raw, Mapping) else vars(raw)
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}

    start = int(data["start"])
    end = int(data["end"])
    label = normalize_label(
        str(
            data.get("canonical_label")
            or data.get("label")
            or data.get("entity_type")
            or data.get("entity_group")
            or "OTHER"
        )
    )
    span_text = str(data.get("text") or text[start:end])
    score = data.get("score", data.get("confidence", 0.0))
    return WeakLabelSpan(
        start=start,
        end=end,
        label=label,
        text=span_text,
        score=float(score),
        source=str(data.get("source") or source),
        metadata=dict(metadata),
    )


def _normalize_detector_outputs(
    detector_outputs: Mapping[str, Iterable[Any]],
    *,
    text: str,
) -> list[WeakLabelSpan]:
    spans: list[WeakLabelSpan] = []
    for source, raw_spans in detector_outputs.items():
        spans.extend(
            normalize_weak_span(raw_span, source=source, text=text)
            for raw_span in raw_spans
        )
    return spans


def _combine_group(grouped: Sequence[WeakLabelSpan]) -> WeakLabelSpan:
    best = max(grouped, key=lambda span: span.score)
    sources = tuple(sorted({span.source for span in grouped if span.source}))
    score = sum(span.score for span in grouped) / len(grouped)
    return WeakLabelSpan(
        start=best.start,
        end=best.end,
        label=best.label,
        text=best.text,
        score=score,
        source="+".join(sources),
        metadata={**dict(best.metadata), "sources": sources},
    )


def _with_metadata(span: WeakLabelSpan, **updates: Any) -> WeakLabelSpan:
    return WeakLabelSpan(
        start=span.start,
        end=span.end,
        label=span.label,
        text=span.text,
        score=span.score,
        source=span.source,
        metadata={**dict(span.metadata), **updates},
    )


def _validators_accept(
    span: WeakLabelSpan, validators: Sequence[SpanValidator]
) -> bool:
    return all(validator(span) for validator in validators)


def _disagreement_groups(
    spans: Sequence[WeakLabelSpan],
) -> list[tuple[WeakLabelSpan, ...]]:
    groups: list[tuple[WeakLabelSpan, ...]] = []
    for index, span in enumerate(spans):
        conflicts = [
            other
            for other in spans[index + 1 :]
            if _overlap(span, other) and span.key != other.key
        ]
        if conflicts:
            groups.append(tuple([span, *conflicts]))
    return groups


def _overlap(left: WeakLabelSpan, right: WeakLabelSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _sources_for_key(
    by_key: Mapping[tuple[int, int, str], Sequence[WeakLabelSpan]],
    key: tuple[int, int, str],
) -> tuple[str, ...]:
    return tuple(sorted({span.source for span in by_key.get(key, ()) if span.source}))


__all__ = [
    "ActiveLearningHook",
    "SpanValidator",
    "WeakLabelDecision",
    "WeakLabelSpan",
    "normalize_weak_span",
    "weak_label_document",
]
