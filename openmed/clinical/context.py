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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
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

ContextCueCategory = Literal["historical", "hypothetical", "uncertainty", "negation"]
ContextCueDirection = Literal["forward", "backward"]

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

RECENT_TEMPORALITY_CUES = (
    "active",
    "acute",
    "current",
    "currently",
    "new",
    "new onset",
    "ongoing",
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
    "not ruled out",
    "not yet ruled out",
    "not completely ruled out",
    "not been ruled out",
    "cannot be excluded",
    "can't be excluded",
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
_RECENT_TEMPORALITY_RE = _cue_pattern(RECENT_TEMPORALITY_CUES)
_UNCERTAINTY_RE = _cue_pattern(UNCERTAINTY_CUES)
_NEGATION_RE = _cue_pattern(NEGATION_CUES)
_PSEUDO_NEGATION_RE = _cue_pattern(PSEUDO_NEGATION_CUES)
_SCOPE_TERMINATOR_RE = re.compile(
    r"(?:[.!?;]|\b(?:and|but|however|or)\b)",
    re.IGNORECASE,
)
_CONTEXT_CUE_RE = _cue_pattern(
    (*HISTORICAL_CUES, *HYPOTHETICAL_CUES, *UNCERTAINTY_CUES, *NEGATION_CUES)
)
_CONJUNCTION_TERMINATOR_RE = re.compile(
    r"(?<!\w)(?:but|however|although)(?!\w)",
    re.IGNORECASE,
)
_CONTEXT_TEXT_KEYS = (
    "document_text",
    "context_text",
    "source_text",
    "full_text",
    "note_text",
    "context",
    "document",
)
_START_KEYS = ("start", "start_char", "start_offset", "begin", "offset")
_END_KEYS = ("end", "end_char", "end_offset", "stop")
_OCCURRENCE_KEYS = ("occurrence", "text_occurrence", "occurrence_index")
_BACKWARD_CONTEXT_CUES = {
    "absent",
    "can't be excluded",
    "cannot be excluded",
    "not been ruled out",
    "not completely ruled out",
    "not present",
    "not ruled out",
    "not yet ruled out",
    "resolved",
    "ruled out",
}


def _normalize_cue_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _cue_category_lookup() -> dict[str, ContextCueCategory]:
    category_by_cue: dict[str, ContextCueCategory] = {}
    for category, cues in (
        ("historical", HISTORICAL_CUES),
        ("hypothetical", HYPOTHETICAL_CUES),
        ("uncertainty", UNCERTAINTY_CUES),
        ("negation", NEGATION_CUES),
    ):
        for cue in cues:
            category_by_cue.setdefault(_normalize_cue_text(cue), category)
    return category_by_cue


_CONTEXT_CUE_CATEGORY_BY_TEXT = _cue_category_lookup()

PATIENT_EXPERIENCER = "patient"
FAMILY_EXPERIENCER = "family"

# Canonical names are lower_snake_case keys expected to compose with the
# user-facing labels OM-086's detect_sections will emit. Aliases are normalized
# with _normalize_section_label before lookup.
CANONICAL_SECTION_LABELS = {
    "past_medical_history": (
        "Past Medical History",
        "PMH",
        "Medical History",
    ),
    "history": ("History",),
    "family_history": (
        "Family History",
        "Family Medical History",
        "FH",
    ),
    "social_history": (
        "Social History",
        "Social Hx",
        "SH",
    ),
    "history_of_present_illness": (
        "History of Present Illness",
        "HPI",
    ),
    "assessment": ("Assessment",),
    "plan": ("Plan",),
}

SECTION_LABEL_ALIASES = {
    "past medical history": "past_medical_history",
    "pmh": "past_medical_history",
    "medical history": "past_medical_history",
    "history": "history",
    "family history": "family_history",
    "family medical history": "family_history",
    "fh": "family_history",
    "social history": "social_history",
    "social hx": "social_history",
    "sh": "social_history",
    "history of present illness": "history_of_present_illness",
    "hpi": "history_of_present_illness",
    "assessment": "assessment",
    "plan": "plan",
}

SECTION_CONTEXT_PRIORS = {
    "past_medical_history": {"temporality": HISTORICAL},
    "history": {"temporality": HISTORICAL},
    "family_history": {"experiencer": FAMILY_EXPERIENCER},
    "social_history": {"experiencer": PATIENT_EXPERIENCER},
}


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


@dataclass(frozen=True)
class ModifierHit:
    """Scoped ConText cue matched around a target clinical span.

    ``cue`` preserves the matched document text and ``start``/``end`` are
    document offsets. ``direction`` records whether the cue was allowed to
    scope forward to a following target or backward to a preceding target. The
    ``text`` property makes the object directly consumable by the existing
    resolver helpers without changing their signatures.
    """

    cue: str
    category: ContextCueCategory
    start: int
    end: int
    direction: ContextCueDirection

    @property
    def text(self) -> str:
        """Return the matched cue text for resolver compatibility."""

        return self.cue


@dataclass(frozen=True)
class _SentenceWindow:
    start: int
    end: int


class _ContextCueResults(dict[Any, tuple[ModifierHit, ...]]):
    """Dictionary-like result that can be looked up with unhashable spans."""

    def __init__(self) -> None:
        super().__init__()
        self._unhashable_entries: list[tuple[Any, tuple[ModifierHit, ...]]] = []

    def add(self, span: Any, hits: tuple[ModifierHit, ...]) -> None:
        try:
            super().__setitem__(span, hits)
        except TypeError:
            self._unhashable_entries.append((span, hits))

    def __getitem__(self, span: Any) -> tuple[ModifierHit, ...]:
        try:
            return super().__getitem__(span)
        except TypeError:
            result = self._unhashable_lookup(span)
            if result is not None:
                return result
            raise KeyError(span) from None
        except KeyError:
            result = self._unhashable_lookup(span)
            if result is not None:
                return result
            raise

    def __contains__(self, span: object) -> bool:
        try:
            if super().__contains__(span):
                return True
        except TypeError:
            pass
        return self._unhashable_lookup(span) is not None

    def __iter__(self) -> Iterable[Any]:
        yield from super().__iter__()
        for span, _ in self._unhashable_entries:
            yield span

    def __len__(self) -> int:
        return super().__len__() + len(self._unhashable_entries)

    def get(
        self,
        span: Any,
        default: tuple[ModifierHit, ...] | None = None,
    ) -> tuple[ModifierHit, ...] | None:
        try:
            return self[span]
        except KeyError:
            return default

    def items(self) -> Iterable[tuple[Any, tuple[ModifierHit, ...]]]:
        yield from super().items()
        yield from self._unhashable_entries

    def keys(self) -> Iterable[Any]:
        for span, _ in self.items():
            yield span

    def values(self) -> Iterable[tuple[ModifierHit, ...]]:
        for _, hits in self.items():
            yield hits

    def _unhashable_lookup(self, span: object) -> tuple[ModifierHit, ...] | None:
        for candidate, hits in self._unhashable_entries:
            if _same_span_key(candidate, span):
                return hits
        return None


def _same_span_key(left: Any, right: object) -> bool:
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


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


def _field_value(obj: Any, keys: Iterable[str]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for key in keys:
            value = obj.get(key)
            if value is not None:
                return value
        return None
    for key in keys:
        value = getattr(obj, key, None)
        if value is not None:
            return value
    return None


def _int_field(obj: Any, keys: Iterable[str]) -> int | None:
    value = _field_value(obj, keys)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _offsets_of(obj: Any) -> tuple[int, int] | None:
    start = _int_field(obj, _START_KEYS)
    end = _int_field(obj, _END_KEYS)
    if start is None or end is None or start < 0 or end < start:
        return None
    return start, end


def _context_text_of(span: Any) -> str:
    context = _field_value(span, _CONTEXT_TEXT_KEYS)
    if isinstance(context, str):
        return context

    text = _text_of(span)
    offsets = _offsets_of(span)
    if offsets is not None and offsets[1] <= len(text):
        return text
    return ""


def _occurrence_of(obj: Any) -> int | None:
    occurrence = _int_field(obj, _OCCURRENCE_KEYS)
    if occurrence is None or occurrence < 0:
        return None
    return occurrence


def _text_offsets_in_context(
    context: str, text: str, occurrence: int | None
) -> tuple[int, int] | None:
    if not context or not text:
        return None

    haystack = context.casefold()
    needle = text.casefold()
    matches: list[tuple[int, int]] = []
    start = haystack.find(needle)
    while start != -1:
        matches.append((start, start + len(text)))
        start = haystack.find(needle, start + 1)

    if occurrence is not None:
        if occurrence < len(matches):
            return matches[occurrence]
        return None
    if len(matches) == 1:
        return matches[0]
    return None


def _text_appears_in_context(context: str, text: str) -> bool:
    return bool(context and text and text.casefold() in context.casefold())


def _target_offsets_in_context(span: Any, context: str) -> tuple[int, int] | None:
    offsets = _offsets_of(span)
    if offsets is not None and offsets[1] <= len(context):
        return offsets
    return _text_offsets_in_context(context, _text_of(span), _occurrence_of(span))


def _scope_bounds(
    context: str,
    target_start: int,
    target_end: int,
) -> tuple[int, int]:
    scope_start = 0
    for match in _SCOPE_TERMINATOR_RE.finditer(context, 0, target_start):
        scope_start = match.end()

    scope_end = len(context)
    for match in _SCOPE_TERMINATOR_RE.finditer(context, target_end):
        scope_end = match.start()
        break

    return scope_start, scope_end


def _cue_reaches_target(
    context: str,
    cue_start: int,
    cue_end: int,
    target_start: int,
    target_end: int,
) -> bool:
    if cue_end <= target_start:
        between = context[cue_end:target_start]
    elif target_end <= cue_start:
        between = context[target_end:cue_start]
    else:
        between = ""
    return _SCOPE_TERMINATOR_RE.search(between) is None


def _scoped_span_text(span: Any) -> str:
    context = _context_text_of(span)
    target_offsets = _target_offsets_in_context(span, context)
    if target_offsets is None:
        return _text_of(span)

    scope_start, scope_end = _scope_bounds(context, *target_offsets)
    return context[scope_start:scope_end]


def _modifier_hit_offsets(hit: Any, context: str) -> tuple[int, int] | None:
    offsets = _offsets_of(hit)
    if offsets is not None and offsets[1] <= len(context):
        return offsets
    return _text_offsets_in_context(context, _text_of(hit), _occurrence_of(hit))


def _iter_modifier_hits(modifier_hits: Any) -> Iterable[Any]:
    if modifier_hits is None:
        return ()
    if isinstance(modifier_hits, (str, Mapping)):
        return (modifier_hits,)
    if isinstance(modifier_hits, Iterable):
        return modifier_hits
    return (modifier_hits,)


def _scoped_modifier_texts(span: Any, modifier_hits: Any) -> tuple[str, ...]:
    context = _context_text_of(span)
    target_offsets = _target_offsets_in_context(span, context)
    parts: list[str] = []

    for hit in _iter_modifier_hits(modifier_hits):
        text = _text_of(hit)
        if not text:
            continue
        if target_offsets is not None:
            hit_offsets = _modifier_hit_offsets(hit, context)
            if hit_offsets is None and _text_appears_in_context(context, text):
                continue
            if hit_offsets is not None and not _cue_reaches_target(
                context,
                *hit_offsets,
                *target_offsets,
            ):
                continue
        parts.append(text)

    return tuple(parts)


def _text_parts(span: Any, modifier_hits: Any) -> tuple[str, ...]:
    span_text = _scoped_span_text(span)
    parts = [span_text]
    span_text_casefold = span_text.casefold()
    for modifier_text in _scoped_modifier_texts(span, modifier_hits):
        if span_text_casefold and modifier_text.casefold() in span_text_casefold:
            continue
        parts.append(modifier_text)
    return tuple(part for part in parts if part)


def _normalize_section_label(section: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", section.casefold()).strip()
    return re.sub(r"\s+", " ", normalized)


def _section_text_of(obj: Any) -> str:
    """Best-effort extraction of a section label from heterogeneous inputs."""

    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        for key in ("section", "label", "name", "section_label", "section_name"):
            value = obj.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("section", "label", "name", "section_label", "section_name"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    return ""


def _canonical_section_name(section: Any) -> str | None:
    section_text = _section_text_of(section)
    if not section_text:
        return None
    normalized = _normalize_section_label(section_text)
    return SECTION_LABEL_ALIASES.get(normalized)


def _has_explicit_temporality_cue(span: Any) -> bool:
    return any(
        _HYPOTHETICAL_RE.search(part)
        or _HISTORICAL_RE.search(part)
        or _RECENT_TEMPORALITY_RE.search(part)
        for part in _text_parts(span, None)
    )


def scan_context_cues(
    text: str,
    spans: Iterable[Any],
    sentences: Iterable[Any] | None = None,
) -> dict[Any, tuple[ModifierHit, ...]]:
    """Return scoped ConText cue hits for each target span.

    This is the lightweight cue producer consumed by the deterministic
    ConText axis resolvers in this module. It scans the already-committed
    temporal, uncertainty, and negation cue lexicons; the full medspaCy-style
    rule-engine port remains tracked separately.

    Args:
        text: Source document text.
        spans: Target spans. Each span may expose ``start``/``end`` offsets via
            mapping keys or object attributes. When offsets are absent, a
            text-like value is located in ``text``.
        sentences: Optional sentence spans. Items may expose ``start`` and
            ``end`` like the target spans, or may be sentence strings located
            sequentially in ``text``. When omitted, pySBD sentence segmentation
            is used if available, with a regex fallback.

    Returns:
        A dictionary keyed by the original span objects. Values are tuples of
        ``ModifierHit`` objects whose cue scope reaches that span.
    """

    sentence_windows = _sentence_windows(text, sentences)
    cue_hits = tuple(_iter_context_cue_hits(text))
    scoped_hits = _ContextCueResults()

    for span in spans:
        target_start, target_end = _target_offsets(text, span)
        target_sentence = _window_index(sentence_windows, target_start, target_end)
        if target_sentence is None:
            scoped_hits.add(span, ())
            continue

        span_hits = [
            hit
            for hit in cue_hits
            if _cue_reaches_span(
                text=text,
                cue_sentence=_window_index(sentence_windows, hit.start, hit.end),
                target_sentence=target_sentence,
                cue=hit,
                target_start=target_start,
                target_end=target_end,
            )
        ]
        scoped_hits.add(
            span,
            tuple(
                sorted(span_hits, key=lambda hit: (hit.start, hit.end, hit.category))
            ),
        )

    return scoped_hits


def _iter_context_cue_hits(text: str) -> Iterable[ModifierHit]:
    for match in _CONTEXT_CUE_RE.finditer(text):
        cue = match.group(0)
        normalized_cue = _normalize_cue_text(cue)
        category = _CONTEXT_CUE_CATEGORY_BY_TEXT[normalized_cue]
        direction: ContextCueDirection = (
            "backward" if normalized_cue in _BACKWARD_CONTEXT_CUES else "forward"
        )
        yield ModifierHit(
            cue=cue,
            category=category,
            start=match.start(),
            end=match.end(),
            direction=direction,
        )


def _sentence_windows(
    text: str,
    sentences: Iterable[Any] | None,
) -> tuple[_SentenceWindow, ...]:
    if sentences is not None:
        windows = tuple(_coerce_sentence_windows(text, sentences))
    else:
        windows = _pysbd_sentence_windows(text)
        if windows is None:
            windows = _regex_sentence_windows(text)

    if not windows and text:
        return (_SentenceWindow(0, len(text)),)
    return windows


def _pysbd_sentence_windows(text: str) -> tuple[_SentenceWindow, ...] | None:
    try:
        from openmed.processing.sentences import segment_text
    except ImportError:  # pragma: no cover - import layout depends on install mode
        return None

    try:
        return tuple(
            _SentenceWindow(segment.start, segment.end)
            for segment in segment_text(text)
            if segment.start < segment.end
        )
    except ImportError:
        return None


def _regex_sentence_windows(text: str) -> tuple[_SentenceWindow, ...]:
    if not text:
        return ()

    windows: list[_SentenceWindow] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)|\n+", text):
        end = match.end()
        _append_trimmed_window(windows, text, start, end)
        start = end
    _append_trimmed_window(windows, text, start, len(text))
    return tuple(windows)


def _append_trimmed_window(
    windows: list[_SentenceWindow],
    text: str,
    start: int,
    end: int,
) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        windows.append(_SentenceWindow(start, end))


def _coerce_sentence_windows(
    text: str,
    sentences: Iterable[Any],
) -> Iterable[_SentenceWindow]:
    cursor = 0
    for sentence in sentences:
        if isinstance(sentence, str):
            start = text.find(sentence, cursor)
            if start == -1:
                stripped = sentence.strip()
                start = text.find(stripped, cursor) if stripped else -1
                length = len(stripped)
            else:
                length = len(sentence)
            if start == -1:
                raise ValueError("sentence text is not present in source text")
            end = start + length
            cursor = end
        else:
            start, end = _offsets_from_obj(sentence)
        yield _validate_offsets(text, start, end, name="sentence")


def _target_offsets(text: str, span: Any) -> tuple[int, int]:
    offsets = _try_offsets_from_obj(span)
    if offsets is not None:
        return _validate_offsets(text, *offsets, name="span")

    span_text = _text_of(span)
    if not span_text:
        raise ValueError("span must expose start/end offsets or a text-like value")

    start = text.find(span_text)
    if start == -1:
        raise ValueError("span text is not present in source text")
    return start, start + len(span_text)


def _try_offsets_from_obj(obj: Any) -> tuple[int, int] | None:
    if isinstance(obj, Mapping):
        start = _first_mapping_value(obj, ("start", "span_start", "sentence_start"))
        end = _first_mapping_value(obj, ("end", "span_end", "sentence_end"))
        if start is not None and end is not None:
            return int(start), int(end)
        return None

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if len(obj) >= 2 and all(isinstance(value, int) for value in obj[:2]):
            return int(obj[0]), int(obj[1])

    start = _first_attr(obj, ("start", "span_start", "sentence_start"))
    end = _first_attr(obj, ("end", "span_end", "sentence_end"))
    if start is not None and end is not None:
        return int(start), int(end)
    return None


def _offsets_from_obj(obj: Any) -> tuple[int, int]:
    offsets = _try_offsets_from_obj(obj)
    if offsets is None:
        raise ValueError("sentence must expose start/end offsets or be text")
    return offsets


def _first_mapping_value(obj: Mapping[Any, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return None


def _first_attr(obj: Any, attrs: tuple[str, ...]) -> Any:
    for attr in attrs:
        value = getattr(obj, attr, None)
        if value is not None:
            return value
    return None


def _validate_offsets(
    text: str,
    start: int,
    end: int,
    *,
    name: str,
) -> tuple[int, int]:
    if start < 0 or end < start or end > len(text):
        raise ValueError(f"{name} offsets must satisfy 0 <= start <= end <= len(text)")
    return start, end


def _window_index(
    sentence_windows: tuple[_SentenceWindow, ...],
    start: int,
    end: int,
) -> int | None:
    for index, sentence in enumerate(sentence_windows):
        if start >= sentence.start and end <= sentence.end:
            return index
    return None


def _cue_reaches_span(
    *,
    text: str,
    cue_sentence: int | None,
    target_sentence: int,
    cue: ModifierHit,
    target_start: int,
    target_end: int,
) -> bool:
    if cue_sentence is None or cue_sentence != target_sentence:
        return False

    if cue.direction == "forward":
        if cue.end > target_start:
            return False
        between = text[cue.end : target_start]
    else:
        if target_end > cue.start:
            return False
        between = text[target_end : cue.start]

    return _CONJUNCTION_TERMINATOR_RE.search(between) is None


def resolve_temporality(span: Any, modifier_hits: Any = None) -> str:
    """Classify the ConText temporality of ``span``.

    ``span`` is the target clinical span -- a string, a span mapping with a
    ``text``-like key, or any object exposing a ``text`` attribute.
    Span mappings/objects may also expose note text via ``context`` or
    ``document_text`` plus ``start``/``end`` offsets; when present, cue matching
    is bounded to the target's sentence or clause.
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
    Span mappings/objects may also expose note text via ``context`` or
    ``document_text`` plus ``start``/``end`` offsets; when present, cue matching
    is bounded to the target's sentence or clause.
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
    Span mappings/objects may also expose note text via ``context`` or
    ``document_text`` plus ``start``/``end`` offsets; when present, cue matching
    is bounded to the target's sentence or clause.
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
    section: Any = None,
) -> ClinicalAssertion:
    """Return the composed clinical assertion axes for ``span``."""

    assertion = ClinicalAssertion(
        temporality=resolve_temporality(span, modifier_hits),
        certainty=resolve_uncertainty(span, modifier_hits),
    )
    return apply_section_context(span, section, assertion)


def apply_section_context(
    span: Any,
    section: Any = None,
    assertion: ClinicalAssertion | None = None,
) -> ClinicalAssertion:
    """Apply conservative section priors to a span assertion.

    ``section`` may be an explicit section label or mapping from OM-086's
    future ``detect_sections`` output. When it is omitted, the helper falls
    back to ``span.section`` or mapping-style ``span["section"]``. Section
    priors only fill unset/default axes for the current span and never override
    explicit in-sentence temporal cues such as ``"acute"``, ``"history of"``,
    or ``"if"``.
    """

    resolved_assertion = assertion or ClinicalAssertion(
        temporality=resolve_temporality(span),
        certainty=resolve_uncertainty(span),
    )
    canonical_section = _canonical_section_name(section) or _canonical_section_name(
        span
    )
    if canonical_section is None:
        return resolved_assertion

    prior = SECTION_CONTEXT_PRIORS.get(canonical_section)
    if not prior:
        return resolved_assertion

    changes: dict[str, str] = {}
    temporality_prior = prior.get("temporality")
    if (
        temporality_prior
        and resolved_assertion.temporality == RECENT
        and not _has_explicit_temporality_cue(span)
    ):
        changes["temporality"] = temporality_prior

    experiencer_prior = prior.get("experiencer")
    if experiencer_prior and resolved_assertion.experiencer is None:
        changes["experiencer"] = experiencer_prior

    if not changes:
        return resolved_assertion
    return replace(resolved_assertion, **changes)


__all__ = [
    "AFFIRMED",
    "NEGATED",
    "Negation",
    "NEGATION_VALUES",
    "NEGATION_CUES",
    "PSEUDO_NEGATION_CUES",
    "ClinicalContextResult",
    "ClinicalAssertion",
    "ContextCueCategory",
    "ContextCueDirection",
    "ModifierHit",
    "scan_context_cues",
    "RECENT",
    "HISTORICAL",
    "HYPOTHETICAL",
    "TEMPORALITY_VALUES",
    "HISTORICAL_CUES",
    "HYPOTHETICAL_CUES",
    "RECENT_TEMPORALITY_CUES",
    "resolve_temporality",
    "Certainty",
    "CERTAIN",
    "UNCERTAIN",
    "CERTAINTY_VALUES",
    "UNCERTAINTY_CUES",
    "resolve_uncertainty",
    "resolve_negation",
    "PATIENT_EXPERIENCER",
    "FAMILY_EXPERIENCER",
    "CANONICAL_SECTION_LABELS",
    "SECTION_LABEL_ALIASES",
    "SECTION_CONTEXT_PRIORS",
    "apply_section_context",
    "resolve_span_context",
    "assert_context_axes",
]
