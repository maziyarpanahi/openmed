"""Relation extraction metrics for synthetic clinical eval suites."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from openmed.eval.metrics import (
    EvalSpan,
    F1Metrics,
    bootstrap_ci,
    normalize_eval_span,
)

RELATION_SCOPE_SENTENCE = "sentence"
RELATION_SCOPE_DOCUMENT = "document"
RELATION_SCOPES: tuple[str, ...] = (
    RELATION_SCOPE_SENTENCE,
    RELATION_SCOPE_DOCUMENT,
)

MATCH_STRICT = "strict"
MATCH_RELAXED = "relaxed"
MATCH_MODES: tuple[str, ...] = (MATCH_STRICT, MATCH_RELAXED)


@dataclass(frozen=True)
class EvalRelation:
    """Normalized directed relation between two eval spans."""

    relation_type: str
    head: EvalSpan
    tail: EvalSpan
    scope: str = RELATION_SCOPE_SENTENCE
    relation_id: str = ""
    fixture_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready relation payload."""
        payload = {
            "id": self.relation_id,
            "fixture_id": self.fixture_id,
            "type": self.relation_type,
            "scope": self.scope,
            "head": _span_to_dict(self.head),
            "tail": _span_to_dict(self.tail),
            "metadata": _plain_mapping(self.metadata),
        }
        return {key: value for key, value in payload.items() if value not in ("", {})}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def normalize_relation_type(relation_type: Any) -> str:
    """Normalize relation labels without touching span labels."""
    value = str(relation_type or "").strip()
    if not value:
        raise ValueError("relation type is required")
    normalized = value.replace("-", "_").replace(" ", "_").upper()
    return "_".join(part for part in normalized.split("_") if part)


def normalize_relation_scope(scope: Any) -> str:
    """Return a supported relation scope."""
    value = str(scope or RELATION_SCOPE_SENTENCE).strip().lower()
    if value not in RELATION_SCOPES:
        allowed = ", ".join(RELATION_SCOPES)
        raise ValueError(f"relation scope {value!r} must be one of: {allowed}")
    return value


def normalize_eval_relation(
    relation: Any,
    *,
    entity_spans: Mapping[str, EvalSpan] | Iterable[Any] | None = None,
    fixture_id: str = "",
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> EvalRelation:
    """Normalize a relation-like mapping or object for scoring."""
    if isinstance(relation, EvalRelation):
        if fixture_id and relation.fixture_id and relation.fixture_id != fixture_id:
            raise ValueError(
                "relation fixture_id does not match the containing fixture: "
                f"{relation.fixture_id!r} != {fixture_id!r}"
            )
        if fixture_id and not relation.fixture_id:
            return replace(relation, fixture_id=fixture_id)
        return relation

    if isinstance(relation, Sequence) and not isinstance(relation, str | bytes):
        relation = _relation_sequence_mapping(relation)

    data = relation if isinstance(relation, Mapping) else vars(relation)
    entities = _entity_lookup(
        entity_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    metadata = _metadata(data)
    relation_type = normalize_relation_type(
        _read_value(data, "type", "relation_type", "label", "predicate")
    )
    scope = normalize_relation_scope(
        _read_value(data, "scope", "level") or metadata.get("scope")
    )
    relation_id = str(_read_value(data, "id", "relation_id") or "")
    supplied_fixture = str(_read_value(data, "fixture_id") or "")
    if fixture_id and supplied_fixture and supplied_fixture != fixture_id:
        raise ValueError(
            "relation fixture_id does not match the containing fixture: "
            f"{supplied_fixture!r} != {fixture_id!r}"
        )
    fixture = supplied_fixture or fixture_id

    return EvalRelation(
        relation_type=relation_type,
        head=_resolve_argument(
            data,
            ("head", "head_id", "arg1", "source", "subject"),
            entities=entities,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ),
        tail=_resolve_argument(
            data,
            ("tail", "tail_id", "arg2", "target", "object"),
            entities=entities,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ),
        scope=scope,
        relation_id=relation_id,
        fixture_id=fixture,
        metadata=metadata,
    )


def normalize_eval_relations(
    relations: Iterable[Any],
    *,
    entity_spans: Mapping[str, EvalSpan] | Iterable[Any] | None = None,
    fixture_id: str = "",
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> list[EvalRelation]:
    """Normalize relation-like records for scoring."""
    return [
        normalize_eval_relation(
            relation,
            entity_spans=entity_spans,
            fixture_id=fixture_id,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        )
        for relation in relations
    ]


def compute_relation_f1(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
    *,
    match: str = MATCH_STRICT,
) -> F1Metrics:
    """Compute directed relation F1 with strict or relaxed argument matching."""
    mode = _validate_match_mode(match)
    gold = _as_relations(gold_relations)
    predicted = _as_relations(predicted_relations)
    true_positives = _matching_count(gold, predicted, match=mode)
    return _f1_from_counts(true_positives, len(predicted), len(gold))


def compute_strict_relation_f1(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> F1Metrics:
    """Compute strict relation F1."""
    return compute_relation_f1(gold_relations, predicted_relations, match=MATCH_STRICT)


def compute_relaxed_relation_f1(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> F1Metrics:
    """Compute relaxed relation F1."""
    return compute_relation_f1(gold_relations, predicted_relations, match=MATCH_RELAXED)


def compute_relation_metrics(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> dict[str, Any]:
    """Return strict and relaxed relation metrics with required breakdowns."""
    gold = _as_relations(gold_relations)
    predicted = _as_relations(predicted_relations)
    relation_types = sorted(
        {relation.relation_type for relation in gold}
        | {relation.relation_type for relation in predicted}
    )
    languages = sorted(
        {_relation_language(relation) for relation in gold}
        | {_relation_language(relation) for relation in predicted}
    )
    scopes = [
        scope
        for scope in RELATION_SCOPES
        if any(relation.scope == scope for relation in (*gold, *predicted))
    ]
    extra_scopes = sorted(
        {
            relation.scope
            for relation in (*gold, *predicted)
            if relation.scope not in RELATION_SCOPES
        }
    )
    scopes.extend(extra_scopes)

    return {
        "strict": compute_strict_relation_f1(gold, predicted).to_dict(),
        "relaxed": compute_relaxed_relation_f1(gold, predicted).to_dict(),
        "by_type": {
            relation_type: _metric_pair(
                [
                    relation
                    for relation in gold
                    if relation.relation_type == relation_type
                ],
                [
                    relation
                    for relation in predicted
                    if relation.relation_type == relation_type
                ],
            )
            for relation_type in relation_types
        },
        "by_scope": {
            scope: _metric_pair(
                [relation for relation in gold if relation.scope == scope],
                [relation for relation in predicted if relation.scope == scope],
            )
            for scope in scopes
        },
        "by_language": {
            language: _metric_pair(
                [
                    relation
                    for relation in gold
                    if _relation_language(relation) == language
                ],
                [
                    relation
                    for relation in predicted
                    if _relation_language(relation) == language
                ],
            )
            for language in languages
        },
        "counts": {
            "gold": len(gold),
            "languages": languages,
            "predicted": len(predicted),
            "relation_types": relation_types,
            "scopes": scopes,
        },
    }


def compute_relation_metrics_bundle(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> dict[str, Any]:
    """Return relation metrics in the release-report payload shape."""
    gold = _as_relations(gold_relations)
    predicted = _as_relations(predicted_relations)
    metrics = compute_relation_metrics(gold, predicted)
    return {
        "gold_relation_count": len(gold),
        "per_relation_type": metrics["by_type"],
        "predicted_relation_count": len(predicted),
        "relaxed": metrics["relaxed"],
        "strict": metrics["strict"],
        "by_scope": metrics["by_scope"],
        "per_language": metrics["by_language"],
    }


def compute_relation_confidence_intervals(
    per_document_relations: Sequence[tuple[Iterable[Any], Iterable[Any]]],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Bootstrap document-level confidence intervals for relation F1."""
    strict_docs: list[tuple[int, int, int]] = []
    relaxed_docs: list[tuple[int, int, int]] = []
    for gold_relations, predicted_relations in per_document_relations:
        strict = compute_strict_relation_f1(gold_relations, predicted_relations)
        relaxed = compute_relaxed_relation_f1(gold_relations, predicted_relations)
        strict_docs.append(
            (strict.true_positives, strict.false_positives, strict.false_negatives)
        )
        relaxed_docs.append(
            (relaxed.true_positives, relaxed.false_positives, relaxed.false_negatives)
        )

    return {
        "relaxed": bootstrap_ci(
            relaxed_docs,
            _f1_over_documents,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
        "strict": bootstrap_ci(
            strict_docs,
            _f1_over_documents,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
    }


def _metric_pair(
    gold: list[EvalRelation], predicted: list[EvalRelation]
) -> dict[str, Any]:
    return {
        "strict": compute_strict_relation_f1(gold, predicted).to_dict(),
        "relaxed": compute_relaxed_relation_f1(gold, predicted).to_dict(),
    }


def _relation_language(relation: EvalRelation) -> str:
    metadata_language = relation.metadata.get("language")
    if metadata_language:
        return str(metadata_language)
    return relation.head.language or relation.tail.language or "en"


def _as_relations(relations: Iterable[Any]) -> list[EvalRelation]:
    normalized: list[EvalRelation] = []
    for relation in relations:
        if isinstance(relation, EvalRelation):
            normalized.append(relation)
        else:
            normalized.append(normalize_eval_relation(relation))
    return normalized


def _matching_count(
    gold: list[EvalRelation],
    predicted: list[EvalRelation],
    *,
    match: str,
) -> int:
    matched_predictions: set[int] = set()
    true_positives = 0
    for gold_relation in gold:
        candidates = [
            (index, prediction)
            for index, prediction in enumerate(predicted)
            if index not in matched_predictions
            and _relation_matches(gold_relation, prediction, match=match)
        ]
        if not candidates:
            continue
        best_index, _ = max(
            candidates,
            key=lambda item: _match_score(gold_relation, item[1], match=match),
        )
        matched_predictions.add(best_index)
        true_positives += 1
    return true_positives


def _relation_matches(
    gold: EvalRelation,
    predicted: EvalRelation,
    *,
    match: str,
) -> bool:
    if gold.fixture_id != predicted.fixture_id:
        return False
    if gold.relation_type != predicted.relation_type:
        return False
    if match == MATCH_STRICT:
        return _span_exact(gold.head, predicted.head) and _span_exact(
            gold.tail, predicted.tail
        )
    return _span_overlaps(gold.head, predicted.head) and _span_overlaps(
        gold.tail, predicted.tail
    )


def _match_score(
    gold: EvalRelation,
    predicted: EvalRelation,
    *,
    match: str,
) -> tuple[int, int, int]:
    if match == MATCH_STRICT:
        return (1, 0, 0)
    return (
        _overlap_len(gold.head, predicted.head)
        + _overlap_len(gold.tail, predicted.tail),
        _overlap_len(gold.head, predicted.head),
        _overlap_len(gold.tail, predicted.tail),
    )


def _span_exact(left: EvalSpan, right: EvalSpan) -> bool:
    return left.start == right.start and left.end == right.end


def _span_overlaps(left: EvalSpan, right: EvalSpan) -> bool:
    return _overlap_len(left, right) > 0


def _overlap_len(left: EvalSpan, right: EvalSpan) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _resolve_argument(
    data: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    entities: Mapping[str, EvalSpan],
    default_language: str,
    default_device: str,
    source_text: str | None,
) -> EvalSpan:
    value = _read_value(data, *keys)
    if value is None:
        value = _flat_argument_mapping(data, keys)
    if value is None:
        names = "/".join(keys)
        raise ValueError(f"relation must include {names}")
    if isinstance(value, EvalSpan):
        return value
    if isinstance(value, str):
        try:
            return entities[value]
        except KeyError as exc:
            raise ValueError(f"unknown relation argument span id: {value!r}") from exc
    if (
        isinstance(value, Mapping)
        and "id" in value
        and not {"start", "end"} <= set(value)
    ):
        span_id = str(value["id"])
        try:
            return entities[span_id]
        except KeyError as exc:
            raise ValueError(f"unknown relation argument span id: {span_id!r}") from exc
    return normalize_eval_span(
        value,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )


def _flat_argument_mapping(
    data: Mapping[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any] | None:
    for prefix in prefixes:
        start = _read_value(data, f"{prefix}_start", f"{prefix}Start")
        end = _read_value(data, f"{prefix}_end", f"{prefix}End")
        if start is None or end is None:
            continue
        label = _read_value(
            data,
            f"{prefix}_label",
            f"{prefix}Label",
            f"{prefix}_type",
        )
        text = _read_value(data, f"{prefix}_text", f"{prefix}Text")
        return {
            "start": start,
            "end": end,
            "label": label or "OTHER",
            "text": text or "",
        }
    return None


def _relation_sequence_mapping(data: Sequence[Any]) -> dict[str, Any]:
    if len(data) == 3:
        relation_type, head, tail = data
        return {"type": relation_type, "head": head, "tail": tail}
    if len(data) == 5:
        relation_type, head_start, head_end, tail_start, tail_end = data
        return {
            "type": relation_type,
            "head": {"start": head_start, "end": head_end},
            "tail": {"start": tail_start, "end": tail_end},
        }
    if len(data) == 6:
        relation_id, relation_type, head_start, head_end, tail_start, tail_end = data
        return {
            "id": relation_id,
            "type": relation_type,
            "head": {"start": head_start, "end": head_end},
            "tail": {"start": tail_start, "end": tail_end},
        }
    raise ValueError(
        "relation tuple must be (type, head, tail), "
        "(type, head_start, head_end, tail_start, tail_end), or "
        "(id, type, head_start, head_end, tail_start, tail_end)"
    )


def _entity_lookup(
    entity_spans: Mapping[str, EvalSpan] | Iterable[Any] | None,
    *,
    default_language: str,
    default_device: str,
    source_text: str | None,
) -> dict[str, EvalSpan]:
    if entity_spans is None:
        return {}
    if isinstance(entity_spans, Mapping):
        return {
            str(span_id): (
                span
                if isinstance(span, EvalSpan)
                else normalize_eval_span(
                    span,
                    default_language=default_language,
                    default_device=default_device,
                    source_text=source_text,
                )
            )
            for span_id, span in entity_spans.items()
        }

    lookup: dict[str, EvalSpan] = {}
    for span in entity_spans:
        data = span if isinstance(span, Mapping) else vars(span)
        span_id = _read_value(data, "id", "span_id", "entity_id")
        if span_id is None:
            continue
        lookup[str(span_id)] = normalize_eval_span(
            span,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        )
    return lookup


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _read_value(data, "metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    return dict(metadata)


def _read_value(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _validate_match_mode(match: str) -> str:
    mode = str(match).strip().lower()
    if mode not in MATCH_MODES:
        allowed = ", ".join(MATCH_MODES)
        raise ValueError(f"relation match mode {match!r} must be one of: {allowed}")
    return mode


def _f1_from_counts(
    true_positives: int,
    predicted_count: int,
    gold_count: int,
) -> F1Metrics:
    false_positives = predicted_count - true_positives
    false_negatives = gold_count - true_positives
    precision = _safe_rate(true_positives, predicted_count, zero_denominator=1.0)
    recall = _safe_rate(true_positives, gold_count, zero_denominator=1.0)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return F1Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _f1_over_documents(rows: Sequence[tuple[int, int, int]]) -> float:
    true_positives = sum(row[0] for row in rows)
    false_positives = sum(row[1] for row in rows)
    false_negatives = sum(row[2] for row in rows)
    return _f1_from_counts(
        true_positives,
        true_positives + false_positives,
        true_positives + false_negatives,
    ).f1


def _safe_rate(
    numerator: int | float,
    denominator: int | float,
    *,
    zero_denominator: float,
) -> float:
    if denominator == 0:
        return zero_denominator
    rate = float(numerator) / float(denominator)
    if not isfinite(rate):
        return zero_denominator
    return rate


def _span_to_dict(span: EvalSpan) -> dict[str, Any]:
    return {
        "start": span.start,
        "end": span.end,
        "label": span.label,
        "text": span.text,
        "language": span.language,
        "device": span.device,
        "metadata": _plain_mapping(span.metadata),
    }


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            plain[str(key)] = _plain_mapping(item)
        elif isinstance(item, (list, tuple)):
            plain[str(key)] = [
                _plain_mapping(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            ]
        else:
            plain[str(key)] = item
    return plain
