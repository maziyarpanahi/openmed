"""Span-boundary guards for entity predictions.

Runtime validation functions that catch stale spans, off-by-one errors,
and overlapping entities *after* tokenizer repair and smart merging.

Validation guards are **warn-only** — they log diagnostics and set metadata
flags but never silently drop entities. Callers that need release-gate style
output can opt into deterministic overlap resolution explicitly.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple, TypeVar

from .labels import normalize_label

if TYPE_CHECKING:
    from openmed.processing.outputs import EntityPrediction

logger = logging.getLogger(__name__)

EntityT = TypeVar("EntityT")

_RISK_LEVEL_RANKS = {
    "none": 0,
    "minimal": 0,
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "high": 3,
    "critical": 4,
    "severe": 4,
}

_HIGH_RISK_LABEL_RANKS = {
    "PERSON": 2,
    "FIRST_NAME": 2,
    "LAST_NAME": 2,
    "MIDDLE_NAME": 2,
    "USERNAME": 2,
    "EMAIL": 2,
    "PHONE": 2,
    "STREET_ADDRESS": 2,
    "GPS_COORDINATES": 2,
    "DATE_OF_BIRTH": 3,
    "ID_NUM": 3,
    "SSN": 4,
    "ACCOUNT_NUMBER": 3,
    "PASSWORD": 4,
    "PIN": 4,
    "API_KEY": 4,
    "CREDIT_CARD": 4,
    "CVV": 4,
    "IBAN": 3,
    "BIC": 3,
    "BITCOIN_ADDRESS": 3,
    "ETHEREUM_ADDRESS": 3,
    "LITECOIN_ADDRESS": 3,
    "IP_ADDRESS": 3,
    "MAC_ADDRESS": 3,
    "VIN": 3,
    "VEHICLE_REGISTRATION": 3,
    "IMEI": 3,
}


class SpanValidationWarning(UserWarning):
    """Raised when an entity span fails a boundary check."""


def _span_value(entity: Any, key: str) -> Any:
    if isinstance(entity, Mapping):
        return entity.get(key)
    return getattr(entity, key, None)


def _span_bounds(entity: Any) -> tuple[int | None, int | None]:
    start = _span_value(entity, "start")
    end = _span_value(entity, "end")
    return start, end


def _entity_label(entity: Any) -> str:
    label = (
        _span_value(entity, "label")
        or _span_value(entity, "entity_type")
        or _span_value(entity, "entity_group")
        or _span_value(entity, "entity")
        or ""
    )
    return str(label)


def _entity_metadata(entity: Any) -> Mapping[str, Any]:
    metadata = _span_value(entity, "metadata")
    if isinstance(metadata, Mapping):
        return metadata
    return {}


def _entity_confidence(entity: Any) -> float:
    value = _span_value(entity, "confidence")
    if value is None:
        value = _span_value(entity, "score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _risk_rank_from_value(value: Any) -> int:
    if isinstance(value, (int, float)):
        return max(0, min(4, int(value)))
    if value is None:
        return 0
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _RISK_LEVEL_RANKS.get(normalized, 0)


def _entity_risk_rank(entity: Any) -> int:
    metadata = _entity_metadata(entity)
    metadata_rank = max(
        _risk_rank_from_value(metadata.get("risk_level")),
        _risk_rank_from_value(metadata.get("risk")),
        _risk_rank_from_value(metadata.get("severity")),
    )
    label_rank = _HIGH_RISK_LABEL_RANKS.get(
        normalize_label(_entity_label(entity)),
        0,
    )
    return max(metadata_rank, label_rank)


def _spans_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> bool:
    return a_start < b_end and a_end > b_start


def _valid_offsets(start: Any, end: Any) -> bool:
    return isinstance(start, int) and isinstance(end, int) and start < end


def validate_entity_spans(
    entities: List["EntityPrediction"],
    text: str,
) -> List["EntityPrediction"]:
    """Validate span boundaries for every entity against *text*.

    Checks performed per entity:
    - ``start < end`` (no inverted or zero-length spans)
    - ``start >= 0`` and ``end <= len(text)``
    - ``text[start:end]`` matches ``entity.text`` (catch stale spans)

    Violations are logged at WARNING level and a
    :class:`SpanValidationWarning` is emitted so callers can
    ``warnings.filterwarnings`` as needed.  A ``span_valid`` flag is
    written into ``entity.metadata`` for downstream consumers.

    Returns the *same* list (never filters entities out).
    """
    text_len = len(text)

    for entity in entities:
        problems: list[str] = []
        start = entity.start
        end = entity.end

        if start is None or end is None:
            # Entities without offsets cannot be validated.
            continue

        # --- invariant checks ---
        if start >= end:
            if start == end:
                problems.append("zero-length span")
            else:
                problems.append(f"inverted span (start={start} >= end={end})")

        if start < 0:
            problems.append(f"negative start ({start})")

        if end > text_len:
            problems.append(f"end ({end}) exceeds text length ({text_len})")

        # --- text-match check (only when bounds are sane) ---
        if not problems and start >= 0 and end <= text_len:
            actual = text[start:end]
            if actual != entity.text:
                # Allow whitespace-only differences (common after span
                # trimming) — downgrade to INFO instead of a full warning.
                if " ".join(actual.split()) == " ".join(entity.text.split()):
                    logger.info(
                        "SpanValidation: Entity %r @ [%d:%d]: "
                        "whitespace-only text difference (span=%r, stored=%r)",
                        entity.label, start, end, actual, entity.text,
                    )
                else:
                    problems.append(
                        f"text mismatch: span gives {actual!r}, "
                        f"entity stores {entity.text!r}"
                    )

        # --- report ---
        if problems:
            msg = (
                f"Entity {entity.label!r} @ [{start}:{end}]: "
                + "; ".join(problems)
            )
            logger.warning("SpanValidation: %s", msg)
            warnings.warn(msg, SpanValidationWarning, stacklevel=2)

        # Tag metadata so downstream code can inspect validity.
        meta = entity.metadata if entity.metadata is not None else {}
        meta["span_valid"] = len(problems) == 0
        entity.metadata = meta

    return entities


def resolve_overlapping_entities(
    entities: Sequence[EntityT],
) -> list[EntityT]:
    """Return a deterministic, non-overlapping span set.

    The resolver is opt-in and leaves the warn-only validation functions
    unchanged. Offset-less or invalid spans are preserved but do not participate
    in overlap resolution.

    Conflict policy, in order:
    - critical/high-risk labels or explicit metadata risk levels win;
    - within the same risk tier, the longest span wins;
    - within the same span length, the highest confidence/score wins;
    - remaining ties use earliest span position and original input order.
    """
    ranked: list[tuple[int, EntityT, int, int]] = []
    unresolved: list[tuple[int, EntityT]] = []

    for index, entity in enumerate(entities):
        start, end = _span_bounds(entity)
        if not _valid_offsets(start, end):
            unresolved.append((index, entity))
            continue
        ranked.append((index, entity, start, end))

    ranked.sort(
        key=lambda item: (
            -_entity_risk_rank(item[1]),
            -(item[3] - item[2]),
            -_entity_confidence(item[1]),
            item[2],
            item[3],
            item[0],
        )
    )

    selected: list[tuple[int, EntityT, int, int]] = []
    selected_bounds: list[tuple[int, int]] = []

    for index, entity, start, end in ranked:
        if any(
            _spans_overlap(start, end, kept_start, kept_end)
            for kept_start, kept_end in selected_bounds
        ):
            continue
        selected.append((index, entity, start, end))
        selected_bounds.append((start, end))

    resolved: list[tuple[int, EntityT, int | None, int | None]] = [
        (index, entity, start, end)
        for index, entity, start, end in selected
    ]
    resolved.extend((index, entity, None, None) for index, entity in unresolved)

    resolved.sort(
        key=lambda item: (
            item[2] is None,
            item[2] if item[2] is not None else 0,
            item[3] if item[3] is not None else 0,
            item[0],
        )
    )
    return [entity for _, entity, _, _ in resolved]


def detect_overlapping_entities(
    entities: List["EntityPrediction"],
) -> List[Tuple["EntityPrediction", "EntityPrediction"]]:
    """Return pairs of entities whose character spans overlap.

    Entities without ``start``/``end`` offsets are skipped.
    """
    # Filter to entities that have offsets.
    with_offsets = [
        e for e in entities
        if e.start is not None and e.end is not None
    ]
    sorted_ents = sorted(with_offsets, key=lambda e: (e.start, e.end))

    overlaps: List[Tuple["EntityPrediction", "EntityPrediction"]] = []
    for i in range(len(sorted_ents) - 1):
        a = sorted_ents[i]
        # Check against all subsequent entities that could overlap.
        for j in range(i + 1, len(sorted_ents)):
            b = sorted_ents[j]
            if b.start < a.end:
                overlaps.append((a, b))
            else:
                break  # No further overlaps possible for *a*.
    return overlaps
