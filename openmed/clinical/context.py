"""ConText decision axes for clinical spans (roadmap 5.2).

This module holds narrow, deterministic ConText layers that classify clinical
spans from a target span plus optional modifier hits produced by the shared
context engine.

Negation classifies whether a clinical span is affirmed or negated. Downstream
grounding layers map ``"negated"`` spans to FHIR
``verificationStatus=refuted``; this module only exposes the per-span context
axis and does not build grounded records.

Temporality classifies the temporal status of a clinical span onto the original
ConText three-value temporality scale:

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

Uncertainty resolves whether a span is asserted as certain or remains hedged,
hypothetical, or conditional. Uncertain spans are flagged, not dropped, so
grounding layers can downweight or annotate unconfirmed conditions while still
receiving the original span.

Sibling axes such as experiencer and absolute-date timeline normalization
(TIMEX3) are handled by separate layers and are out of scope here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

Negation = Literal["affirmed", "negated"]

AFFIRMED: Negation = "affirmed"
NEGATED: Negation = "negated"

#: The negation values, ordered from default asserted polarity to refuted.
NEGATION_VALUES = (AFFIRMED, NEGATED)

RECENT = "recent"
HISTORICAL = "historical"
HYPOTHETICAL = "hypothetical"

#: The temporality values, ordered from default to most specific.
TEMPORALITY_VALUES = (RECENT, HISTORICAL, HYPOTHETICAL)

Certainty = Literal["certain", "uncertain"]

CERTAIN: Certainty = "certain"
UNCERTAIN: Certainty = "uncertain"

#: The certainty values, ordered from asserted to less certain.
CERTAINTY_VALUES = (CERTAIN, UNCERTAIN)

# Ported ConText temporal lexicon. Cues are matched case-insensitively and on
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

# ConText hypothetical/uncertainty trigger lexicon. It intentionally overlaps
# with HYPOTHETICAL_CUES because a conditional span is both temporally
# hypothetical and not clinically asserted as certain.
UNCERTAINTY_CUES = (
    "cannot exclude",
    "can't exclude",
    "concern for",
    "concerning for",
    "suspicious for",
    "suspicion for",
    "worrisome for",
    "question of",
    "to rule out",
    "rule out",
    "in the event of",
    "in case of",
    "in case",
    "versus",
    "probable",
    "probably",
    "possible",
    "possibly",
    "suspected",
    "suspect",
    "unlikely",
    "likely",
    "unless",
    "should",
    "might",
    "could",
    "may",
    "r/o",
    "vs",
    "if",
)

# ConText/NegEx-style negation cues. Phrases are matched before shorter cues,
# so "no evidence of" is counted as one negation cue rather than "no" plus a
# second incidental phrase.
NEGATION_CUES = (
    "no evidence of",
    "no evidence",
    "no sign of",
    "no signs of",
    "negative for",
    "absence of",
    "free of",
    "ruled out",
    "not present",
    "denies",
    "denied",
    "deny",
    "without",
    "absent",
    "never",
    "none",
    "not",
    "no",
)

# Pseudo-negation cues contain negation words but do not refute the target
# concept. They are masked before true negation cues are counted.
PSEUDO_NEGATION_CUES = (
    "no increase",
    "no interval increase",
    "no significant increase",
    "not ruled out",
    "not yet ruled out",
    "not completely ruled out",
    "not been ruled out",
    "cannot be excluded",
    "can't be excluded",
    "cannot exclude",
    "can't exclude",
)


def _cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(cues, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


_HISTORICAL_RE = _cue_pattern(HISTORICAL_CUES)
_HYPOTHETICAL_RE = _cue_pattern(HYPOTHETICAL_CUES)
_UNCERTAINTY_RE = _cue_pattern(UNCERTAINTY_CUES)
_NEGATION_RE = _cue_pattern(NEGATION_CUES)
_PSEUDO_NEGATION_RE = _cue_pattern(PSEUDO_NEGATION_CUES)


@dataclass(frozen=True)
class ClinicalContextResult:
    """Per-span ConText decision result.

    ``negation="negated"`` is the downstream signal for
    ``verificationStatus=refuted`` when a grounding/FHIR layer materializes the
    clinical span as a Condition. This object deliberately stays at the context
    axis layer and does not construct that grounded record.
    """

    temporality: str
    certainty: Certainty
    negation: Negation


@dataclass(frozen=True)
class ClinicalAssertion:
    """Advisory per-span clinical assertion axes for downstream grounding.

    This record composes the ConText axis decisions currently needed by
    downstream FHIR/OMOP grounding without building FHIR, OMOP, or other
    clinical records here. A ``"historical"`` temporality maps downstream to an
    inactive or resolved FHIR ``clinicalStatus``. A ``"hypothetical"`` span is
    not asserted as present. An ``"uncertain"`` certainty maps downstream to
    ``verificationStatus=provisional``.

    ``negation`` and ``experiencer`` are optional extension points for sibling
    axes and remain unset when those axes are not part of this composition.
    Outputs are advisory annotations for review and downstream processing, not
    clinical decisions and not medical-device instructions.
    """

    temporality: str
    certainty: Certainty
    negation: Negation | None = None
    experiencer: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Return a compact dictionary, omitting axes that are not set."""

        values = {
            "temporality": self.temporality,
            "certainty": self.certainty,
            "negation": self.negation,
            "experiencer": self.experiencer,
        }
        return {key: value for key, value in values.items() if value is not None}


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


def resolve_uncertainty(span: Any, modifier_hits: Any = None) -> Certainty:
    """Classify a clinical span as certain or uncertain/hypothetical.

    ``span`` is the target clinical span -- a string, a span mapping with a
    ``text``-like key, or any object exposing a ``text`` attribute.
    ``modifier_hits`` is the optional collection of ConText uncertainty cues
    matched in the span's window. Each part is scanned independently so cues
    are never created by concatenating unrelated fragments.

    Returns ``"uncertain"`` for hedged, hypothetical, or conditional concepts
    and ``"certain"`` otherwise. Uncertain spans are flagged for grounding
    consumers; this helper does not filter or drop spans.
    """

    parts = _text_parts(span, modifier_hits)
    if any(_UNCERTAINTY_RE.search(part) for part in parts):
        return UNCERTAIN
    return CERTAIN


def _mask_pseudo_negation(text: str) -> str:
    return _PSEUDO_NEGATION_RE.sub(lambda match: " " * len(match.group(0)), text)


def resolve_negation(span: Any, modifier_hits: Any = None) -> Negation:
    """Classify a clinical span as affirmed or negated.

    ``span`` is the target clinical span -- a string, a span mapping with a
    ``text``-like key, or any object exposing a ``text`` attribute.
    ``modifier_hits`` is the optional collection of ConText negation cues
    matched in the span's window. Each part is scanned independently so cues
    are never created by concatenating unrelated fragments.

    Pseudo-negation cues such as ``"no increase"``, ``"not ruled out"``, and
    ``"cannot be excluded"`` are masked before true negation cues are counted.
    An odd number of true cues returns ``"negated"``; an even number returns
    ``"affirmed"``, which makes double-negation deterministic.
    """

    negation_cue_count = 0
    for part in _text_parts(span, modifier_hits):
        masked = _mask_pseudo_negation(part)
        negation_cue_count += sum(1 for _ in _NEGATION_RE.finditer(masked))
    return NEGATED if negation_cue_count % 2 else AFFIRMED


def resolve_span_context(
    span: Any,
    modifier_hits: Any = None,
) -> ClinicalContextResult:
    """Return all currently implemented ConText decision axes for ``span``."""

    return ClinicalContextResult(
        temporality=resolve_temporality(span, modifier_hits),
        certainty=resolve_uncertainty(span, modifier_hits),
        negation=resolve_negation(span, modifier_hits),
    )


def assert_context_axes(
    span: Any,
    modifier_hits: Any = None,
) -> ClinicalAssertion:
    """Return the composed clinical assertion axes for ``span``."""

    return ClinicalAssertion(
        temporality=resolve_temporality(span, modifier_hits),
        certainty=resolve_uncertainty(span, modifier_hits),
    )


__all__ = [
    "AFFIRMED",
    "NEGATED",
    "Negation",
    "NEGATION_VALUES",
    "NEGATION_CUES",
    "PSEUDO_NEGATION_CUES",
    "ClinicalContextResult",
    "ClinicalAssertion",
    "RECENT",
    "HISTORICAL",
    "HYPOTHETICAL",
    "TEMPORALITY_VALUES",
    "HISTORICAL_CUES",
    "HYPOTHETICAL_CUES",
    "resolve_temporality",
    "Certainty",
    "CERTAIN",
    "UNCERTAIN",
    "CERTAINTY_VALUES",
    "UNCERTAINTY_CUES",
    "resolve_uncertainty",
    "resolve_negation",
    "resolve_span_context",
    "assert_context_axes",
]
