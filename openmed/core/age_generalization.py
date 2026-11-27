"""HIPAA Safe Harbor age generalization for free-text spans.

45 CFR 164.514(b)(2)(i)(C) treats an age as an identifier only when it is over
89, and allows those ages to be aggregated into one "90 or older" category.
This module applies that rule to a detected ``AGE`` span: ages 89 and below are
kept, ages 90 and above are rendered as the top band, and a span whose age
cannot be read with confidence is reported as unparseable so the caller can
fall back to a stronger action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

#: First age that Safe Harbor requires to be aggregated.
AGE_TOP_THRESHOLD: Final = 90
#: Rendering of the aggregated "90 or older" category.
AGE_TOP_BAND: Final = "90+"
#: Largest value accepted as an age; larger numbers are not treated as ages.
AGE_MAX: Final = 150

AGE_OUTCOME_KEPT: Final = "kept"
AGE_OUTCOME_TOP_BAND: Final = "top_band"
AGE_OUTCOME_UNPARSEABLE: Final = "unparseable"

_NUMBER_RE: Final = re.compile(r"\d+")


@dataclass(frozen=True)
class AgeGeneralization:
    """Result of applying the Safe Harbor age rule to one span.

    Attributes:
        outcome: ``"kept"``, ``"top_band"``, or ``"unparseable"``.
        text: Replacement text for the span, or ``None`` when unparseable.
    """

    outcome: str
    text: str | None


def generalize_age_span(text: str) -> AgeGeneralization:
    """Apply the Safe Harbor age rule to the surface text of an ``AGE`` span.

    The span must contain exactly one number in ``0..AGE_MAX``. Spans with no
    digits (``"ninety-three"``), several numbers (``"65-70"``, ``"2.5"``), or
    an implausible value are unparseable; the rule never guesses.

    Args:
        text: Surface text of the detected span, such as ``"93-year-old"``.

    Returns:
        The outcome and the replacement text. Ages 89 and below return the
        span unchanged. Ages 90 and above replace only the number, so the
        surrounding token shape survives (``"93-year-old"`` becomes
        ``"90+-year-old"``).
    """

    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return AgeGeneralization(AGE_OUTCOME_UNPARSEABLE, None)
    match = matches[0]
    try:
        age = int(match.group())
    except ValueError:
        return AgeGeneralization(AGE_OUTCOME_UNPARSEABLE, None)
    if age > AGE_MAX:
        return AgeGeneralization(AGE_OUTCOME_UNPARSEABLE, None)
    if age < AGE_TOP_THRESHOLD:
        return AgeGeneralization(AGE_OUTCOME_KEPT, text)
    return AgeGeneralization(
        AGE_OUTCOME_TOP_BAND,
        text[: match.start()] + AGE_TOP_BAND + text[match.end() :],
    )


__all__ = [
    "AGE_MAX",
    "AGE_OUTCOME_KEPT",
    "AGE_OUTCOME_TOP_BAND",
    "AGE_OUTCOME_UNPARSEABLE",
    "AGE_TOP_BAND",
    "AGE_TOP_THRESHOLD",
    "AgeGeneralization",
    "generalize_age_span",
]
