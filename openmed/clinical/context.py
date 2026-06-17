"""ConText temporality axis for clinical spans (OM-141, roadmap 5.2).

This module is the narrow temporality decision layer of the shared ConText
engine.  It classifies the temporal status of a clinical span onto the
original ConText three-value temporality scale:

* ``"recent"``       -- the finding is current / active (the default).
* ``"historical"``   -- the finding belongs to the patient's past history.
* ``"hypothetical"`` -- the finding is conditional and not asserted as present.

The historical-versus-current distinction is what separates a resolved past
problem from an active one (``"history of MI"`` versus ``"acute MI"``).
Downstream FHIR/OMOP records consume this axis to drive
``clinicalStatus`` / ``onset``: a ``"historical"`` span maps to an inactive or
resolved ``clinicalStatus`` and a past ``onset``, whereas a ``"recent"`` span
maps to an active ``clinicalStatus``.  A ``"hypothetical"`` span is not asserted
to be present at all and should not be recorded as an active condition.

Sibling axes (negation, uncertainty, experiencer) and absolute-date timeline
normalization (TIMEX3) are handled by separate layers and are out of scope
here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

RECENT = "recent"
HISTORICAL = "historical"
HYPOTHETICAL = "hypothetical"

#: The temporality values, ordered from default to most specific.
TEMPORALITY_VALUES = (RECENT, HISTORICAL, HYPOTHETICAL)

# Ported ConText temporal lexicon.  Cues are matched case-insensitively and on
# token boundaries so abbreviations such as ``h/o`` and ``s/p`` match cleanly
# without firing inside unrelated words.
HISTORICAL_CUES = (
    "history of",
    "hx of",
    "h/o",
    "status post",
    "s/p",
    "previous",
    "previously",
    "prior",
    "in the past",
    "past medical history",
    "pmh",
    "resolved",
)

HYPOTHETICAL_CUES = (
    "if",
    "should",
    "in case of",
    "in case",
    "in the event of",
    "unless",
)


def _cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(cues, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


_HISTORICAL_RE = _cue_pattern(HISTORICAL_CUES)
_HYPOTHETICAL_RE = _cue_pattern(HYPOTHETICAL_CUES)


def _text_of(obj: Any) -> str:
    """Best-effort extraction of cue/span text from heterogeneous inputs."""

    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        for key in ("text", "phrase", "cue", "literal", "word", "value", "surface"):
            value = obj.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("text", "phrase", "cue", "literal", "word", "value", "surface"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    return ""


def _text_parts(span: Any, modifier_hits: Any) -> tuple[str, ...]:
    parts = [_text_of(span)]
    if modifier_hits is not None and not isinstance(modifier_hits, (str, Mapping)):
        if isinstance(modifier_hits, Iterable):
            parts.extend(_text_of(hit) for hit in modifier_hits)
        else:
            parts.append(_text_of(modifier_hits))
    else:
        parts.append(_text_of(modifier_hits))
    return tuple(part for part in parts if part)


def resolve_temporality(span: Any, modifier_hits: Any = None) -> str:
    """Classify the ConText temporality of ``span``.

    ``span`` is the target clinical span -- a string, a span mapping with a
    ``text``-like key, or any object exposing a ``text`` attribute.
    ``modifier_hits`` is the optional collection of ConText modifier cues the
    shared engine matched in the span's window (cue strings, or mappings/objects
    exposing a ``text``-like field).  The span surface itself is also scanned so
    the layer is usable standalone when no separate modifier hits are supplied.

    Returns one of ``"recent"``, ``"historical"`` or ``"hypothetical"``.
    ``"recent"`` is the default when no temporal cue is found.  When both a
    hypothetical and a historical cue are present the span is treated as
    ``"hypothetical"``: a conditional statement is not asserted to have
    occurred, which takes precedence over where in time it would sit.
    """

    parts = _text_parts(span, modifier_hits)

    if any(_HYPOTHETICAL_RE.search(part) for part in parts):
        return HYPOTHETICAL
    if any(_HISTORICAL_RE.search(part) for part in parts):
        return HISTORICAL
    return RECENT


__all__ = [
    "RECENT",
    "HISTORICAL",
    "HYPOTHETICAL",
    "TEMPORALITY_VALUES",
    "HISTORICAL_CUES",
    "HYPOTHETICAL_CUES",
    "resolve_temporality",
]
