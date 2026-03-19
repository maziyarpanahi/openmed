"""Span-boundary guards for entity predictions.

Runtime validation functions that catch stale spans, off-by-one errors,
and overlapping entities *after* tokenizer repair and smart merging.

All guards are **warn-only** — they log diagnostics and set metadata flags
but never silently drop entities.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from openmed.processing.outputs import EntityPrediction

logger = logging.getLogger(__name__)


class SpanValidationWarning(UserWarning):
    """Raised when an entity span fails a boundary check."""


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
