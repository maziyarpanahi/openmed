"""Project clinical labels through configurable morphology with exact offsets.

Cross-lingual pipelines often project an annotation from a high-resource
language onto a target-language word whose surface was changed by case
endings, clitics, agglutinative suffix chains, noun-class prefixes or stem
alternations. Plain string matching then misses the word, picks the wrong
occurrence, or emits a span that cuts through an affix or a grapheme cluster.

:func:`project_labels` matches each projected base form (lemma) against the
target text through a caller-supplied :class:`MorphologyProfile`. Every
emitted label is a half-open Unicode code-point span into the exact,
unnormalized source text (see ``openmed/core/decoding/OFFSET_CONTRACT.md``)
together with the matching extended-grapheme and UTF-8 byte coordinates.
Projections that cannot be resolved to exactly one grapheme-safe span are
routed to review with a controlled reason code instead of being emitted as
training labels.

Tokenization, grapheme segmentation and Unicode normalization reuse the shared
offset-preserving helpers in :mod:`openmed.processing.tokenization`,
:mod:`openmed.core.decoding.spans` and :mod:`openmed.core.text_normalize`.
The module is deterministic, runs no models, makes no network calls and never
logs. Decisions and reports carry offsets, controlled codes, counts and a
profile digest only; they never carry source text or lemmas.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

from openmed.core.decoding.spans import (
    grapheme_break_checker,
    iter_grapheme_cluster_spans,
    snap_span_to_grapheme_boundaries,
)
from openmed.core.locale_tag import normalize_locale_tag
from openmed.core.text_normalize import normalize_unicode_with_offsets
from openmed.processing.tokenization import SpanToken, indic_word_tokenize

SPAN_POLICIES: Final[tuple[str, ...]] = ("token", "host", "stem")
AFFIX_POSITIONS: Final[tuple[str, ...]] = ("prefix", "suffix")
AFFIX_KINDS: Final[tuple[str, ...]] = ("inflection", "clitic")
CASING_MODES: Final[tuple[str, ...]] = ("auto", "unicode", "turkic")
PROJECTION_STATUSES: Final[tuple[str, ...]] = (
    "projected",
    "needs_review",
    "unmatched",
)
PROJECTED_REASONS: Final[tuple[str, ...]] = (
    "exact_match",
    "affix_match",
    "alternation_match",
)
REVIEW_REASONS: Final[tuple[str, ...]] = (
    "ambiguous_segmentation",
    "multiple_matches",
    "grapheme_bisection",
    "unanalyzed_affix",
    "overlapping_projection",
)
UNMATCHED_REASONS: Final[tuple[str, ...]] = ("no_match",)

MAX_PROJECTIONS: Final[int] = 10_000
MAX_LEMMA_LENGTH: Final[int] = 256
MAX_LEMMA_TOKENS: Final[int] = 16
MAX_AFFIX_RULES: Final[int] = 512
MAX_AFFIX_LENGTH: Final[int] = 32
MAX_ALTERNATIONS: Final[int] = 64
MAX_LEXICALIZED_FORMS: Final[int] = 10_000
MAX_AFFIXES_PER_TOKEN: Final[int] = 8
MAX_ANALYZED_TOKEN_LENGTH: Final[int] = 256
MAX_REVIEW_CANDIDATES: Final[int] = 32

_LABEL: Final = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_TURKIC_LANGUAGES: Final = frozenset({"az", "tr"})
_JOINERS: Final = frozenset({"\u200c", "\u200d"})
_PROFILE_FIELDS: Final = frozenset(
    {
        "language",
        "affixes",
        "alternations",
        "lexicalized_forms",
        "casing",
        "max_affixes",
        "min_stem_graphemes",
    }
)
_STATUS_REASONS: Final[Mapping[str, tuple[str, ...]]] = {
    "projected": PROJECTED_REASONS,
    "needs_review": REVIEW_REASONS,
    "unmatched": UNMATCHED_REASONS,
}
_EXACT, _AFFIX, _ALTERNATION = range(3)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_surface(value: object, *, allow_empty: bool = False) -> str:
    """Validate configured affix or lemma-edge material without echoing it."""
    if not isinstance(value, str):
        raise TypeError("morphology surfaces must be strings")
    if not allow_empty and not value:
        raise ValueError("morphology surfaces must not be empty")
    if len(value) > MAX_AFFIX_LENGTH:
        raise ValueError("morphology surface exceeds the length limit")
    for char in value:
        # Zero-width (non-)joiners are legitimate inside Indic affixes.
        if char.isspace() or (not char.isprintable() and char not in _JOINERS):
            raise ValueError(
                "morphology surfaces must not contain whitespace or controls"
            )
    return value


def _string_tuple(values: object, *, what: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a sequence")
    return tuple(values)


@dataclass(frozen=True)
class AffixRule:
    """One configurable prefix or suffix surface.

    Attributes:
        surface: Affix surface in the target script. Matching happens after
            the profile's NFC and case folding, so either case may be given.
        position: ``"prefix"`` or ``"suffix"``.
        kind: ``"inflection"`` for case, number, possessive or noun-class
            material that belongs to the word, or ``"clitic"`` for attached
            function words such as conjunctions, prepositions or focus
            particles. Clitics are always outermost and the ``"host"`` span
            policy excludes them from the label.
    """

    surface: str
    position: str = "suffix"
    kind: str = "inflection"

    def __post_init__(self) -> None:
        _validate_surface(self.surface)
        if not isinstance(self.position, str) or self.position not in AFFIX_POSITIONS:
            raise ValueError("affix position must be 'prefix' or 'suffix'")
        if not isinstance(self.kind, str) or self.kind not in AFFIX_KINDS:
            raise ValueError("affix kind must be 'inflection' or 'clitic'")

    def to_dict(self) -> dict[str, str]:
        """Serialize the rule as configuration data."""
        return {"surface": self.surface, "position": self.position, "kind": self.kind}


@dataclass(frozen=True)
class StemAlternation:
    """Rewrite lemma-edge material into the stem form used before an affix.

    Examples are Finnish ``diabetes`` -> ``diabetekse-`` (``"s"`` to
    ``"kse"``), Turkish ``kalp`` -> ``kalb-`` (``"p"`` to ``"b"``), Tamil
    ``மருந்து`` -> ``மருந்த-`` (drop the final vowel sign) and Swahili
    ``ugonjwa`` -> ``-gonjwa`` after a plural class prefix (``"u"`` to ``""``
    at the prefix edge). An alternation only applies when at least one affix
    was removed on the same side, so a bare alternated stem never matches.

    Attributes:
        lemma_edge: Material at the lemma edge that is replaced.
        stem_edge: Replacement material; may be empty.
        position: ``"suffix"`` for the word end or ``"prefix"`` for its start.
    """

    lemma_edge: str
    stem_edge: str
    position: str = "suffix"

    def __post_init__(self) -> None:
        _validate_surface(self.lemma_edge, allow_empty=True)
        _validate_surface(self.stem_edge, allow_empty=True)
        if self.lemma_edge == self.stem_edge:
            raise ValueError("a stem alternation must change the lemma edge")
        if not isinstance(self.position, str) or self.position not in AFFIX_POSITIONS:
            raise ValueError("alternation position must be 'prefix' or 'suffix'")

    def to_dict(self) -> dict[str, str]:
        """Serialize the alternation as configuration data."""
        return {
            "lemma_edge": self.lemma_edge,
            "stem_edge": self.stem_edge,
            "position": self.position,
        }


@dataclass(frozen=True)
class MorphologyProfile:
    """Caller-configured morphology for one target language.

    Profiles are configuration, not linguistic authority: they should list
    only inflectional affixes and clitics that a native reviewer has approved
    for label projection. Rules are canonicalized (deduplicated and sorted),
    so equivalent profiles share one :attr:`digest`.

    Attributes:
        language: BCP 47 language tag, normalized on construction.
        affixes: Prefix and suffix rules.
        alternations: Lemma-to-stem edge rewrites.
        lexicalized_forms: Whole words that must never be split by an affix
            boundary, for example a noun that starts with the same letters as
            a prepositional clitic.
        casing: ``"unicode"`` full case folding, ``"turkic"`` dotted and
            dotless I folding, or ``"auto"`` to pick ``"turkic"`` for Turkish
            and Azerbaijani and ``"unicode"`` otherwise.
        max_affixes: Maximum affixes removed from one token, in ``0..8``.
        min_stem_graphemes: Minimum graphemes left after removing affixes.
    """

    language: str
    affixes: tuple[AffixRule, ...] = ()
    alternations: tuple[StemAlternation, ...] = ()
    lexicalized_forms: tuple[str, ...] = ()
    casing: str = "auto"
    max_affixes: int = 4
    min_stem_graphemes: int = 2

    def __post_init__(self) -> None:
        language = normalize_locale_tag(self.language)
        affixes = _string_tuple(self.affixes, what="affixes")
        alternations = _string_tuple(self.alternations, what="alternations")
        lexicalized = _string_tuple(self.lexicalized_forms, what="lexicalized_forms")
        if not all(isinstance(rule, AffixRule) for rule in affixes):
            raise TypeError("affixes must contain AffixRule values")
        if not all(isinstance(rule, StemAlternation) for rule in alternations):
            raise TypeError("alternations must contain StemAlternation values")
        if len(affixes) > MAX_AFFIX_RULES or len(alternations) > MAX_ALTERNATIONS:
            raise ValueError("morphology profile exceeds its rule limit")
        if len(lexicalized) > MAX_LEXICALIZED_FORMS:
            raise ValueError("morphology profile exceeds its lexicalized-form limit")
        for form in lexicalized:
            if not isinstance(form, str):
                raise TypeError("lexicalized forms must be strings")
            if (
                not form
                or len(form) > MAX_LEMMA_LENGTH
                or any(char.isspace() for char in form)
            ):
                raise ValueError("lexicalized forms must be single non-empty words")
        if not isinstance(self.casing, str) or self.casing not in CASING_MODES:
            raise ValueError("casing must be 'auto', 'unicode' or 'turkic'")
        if not _is_int(self.max_affixes) or not (
            0 <= self.max_affixes <= MAX_AFFIXES_PER_TOKEN
        ):
            raise ValueError("max_affixes must be an integer between 0 and 8")
        if not _is_int(self.min_stem_graphemes) or not (
            1 <= self.min_stem_graphemes <= 16
        ):
            raise ValueError("min_stem_graphemes must be an integer between 1 and 16")
        casing = self.casing
        if casing == "auto":
            primary = language.split("-", 1)[0]
            casing = "turkic" if primary in _TURKIC_LANGUAGES else "unicode"
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "casing", casing)
        object.__setattr__(
            self,
            "affixes",
            tuple(
                sorted(set(affixes), key=lambda rule: tuple(rule.to_dict().values()))
            ),
        )
        object.__setattr__(
            self,
            "alternations",
            tuple(
                sorted(
                    set(alternations), key=lambda rule: tuple(rule.to_dict().values())
                )
            ),
        )
        object.__setattr__(self, "lexicalized_forms", tuple(sorted(set(lexicalized))))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MorphologyProfile:
        """Build a profile from JSON- or YAML-style configuration data.

        Args:
            data: Mapping with ``language`` and optional ``affixes`` (mappings
                with ``surface``, ``position`` and ``kind``), ``alternations``
                (mappings with ``lemma_edge``, ``stem_edge`` and ``position``),
                ``lexicalized_forms``, ``casing``, ``max_affixes`` and
                ``min_stem_graphemes``.

        Returns:
            A validated, canonical profile.

        Raises:
            TypeError: If the configuration has the wrong shape.
            ValueError: If a field is unknown or a value is invalid.
        """
        if not isinstance(data, Mapping):
            raise TypeError("morphology profile configuration must be a mapping")
        if not all(isinstance(key, str) for key in data):
            raise TypeError("morphology profile keys must be strings")
        if set(data) - _PROFILE_FIELDS:
            raise ValueError("unknown morphology profile field")
        if "language" not in data:
            raise ValueError("morphology profile requires a language")
        affixes = _string_tuple(data.get("affixes", ()), what="affixes")
        alternations = _string_tuple(data.get("alternations", ()), what="alternations")
        if not all(isinstance(item, Mapping) for item in (*affixes, *alternations)):
            raise TypeError("affix and alternation entries must be mappings")
        try:
            affix_rules = tuple(AffixRule(**item) for item in affixes)
            alternation_rules = tuple(StemAlternation(**item) for item in alternations)
        except TypeError:
            raise ValueError("invalid affix or alternation fields") from None
        return cls(
            language=data["language"],
            affixes=affix_rules,
            alternations=alternation_rules,
            lexicalized_forms=_string_tuple(
                data.get("lexicalized_forms", ()), what="lexicalized_forms"
            ),
            casing=data.get("casing", "auto"),
            max_affixes=data.get("max_affixes", 4),
            min_stem_graphemes=data.get("min_stem_graphemes", 2),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical profile configuration."""
        return {
            "language": self.language,
            "affixes": [rule.to_dict() for rule in self.affixes],
            "alternations": [rule.to_dict() for rule in self.alternations],
            "lexicalized_forms": list(self.lexicalized_forms),
            "casing": self.casing,
            "max_affixes": self.max_affixes,
            "min_stem_graphemes": self.min_stem_graphemes,
        }

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest of the canonical configuration."""
        payload = json.dumps(
            self.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class LabelProjection:
    """One label projected from a high-resource annotation.

    Attributes:
        label: Controlled label code such as ``"DISEASE"``.
        lemma: Target-language base form of the projected mention, for
            example from a bilingual lexicon or an aligned translation. It may
            contain several words. It is excluded from ``repr`` and from every
            decision and report.
        window: Optional ``(start, end)`` code-point window from an aligner.
            Only matches with at least one token inside the window count.
    """

    label: str
    lemma: str = field(repr=False)
    window: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not _LABEL.fullmatch(self.label):
            raise ValueError("label must be a controlled upper-case code")
        if not isinstance(self.lemma, str):
            raise TypeError("lemma must be a string")
        if not self.lemma.strip() or len(self.lemma) > MAX_LEMMA_LENGTH:
            raise ValueError("lemma must be non-empty and within the length limit")
        if not 1 <= len(indic_word_tokenize(self.lemma)) <= MAX_LEMMA_TOKENS:
            raise ValueError("lemma exceeds the token limit")
        if self.window is not None:
            window = self.window
            if (
                not isinstance(window, (tuple, list))
                or len(window) != 2
                or not all(_is_int(value) for value in window)
                or not 0 <= window[0] < window[1]
            ):
                raise ValueError("window must be a non-empty (start, end) pair")
            object.__setattr__(self, "window", (window[0], window[1]))


@dataclass(frozen=True)
class ProjectionDecision:
    """Metadata-only outcome for one :class:`LabelProjection`.

    Offsets are present only for ``"projected"`` decisions. ``start`` and
    ``end`` are code-point offsets into the exact source text; the grapheme
    and byte fields describe the same span as extended-grapheme indices and
    UTF-8 byte offsets. Review decisions list grapheme-safe candidate
    code-point spans for a reviewer instead of a label span.

    Attributes:
        index: Position of the projection in the input sequence.
        label: Controlled label code.
        status: ``"projected"``, ``"needs_review"`` or ``"unmatched"``.
        reason: Controlled reason code for the status.
        start: Inclusive code-point offset of a projected label.
        end: Exclusive code-point offset of a projected label.
        grapheme_start: Index of the first grapheme cluster in the label.
        grapheme_end: Index one past the last grapheme cluster in the label.
        byte_start: Inclusive UTF-8 byte offset of a projected label.
        byte_end: Exclusive UTF-8 byte offset of a projected label.
        affix_count: Affixes removed to reach the lemma.
        candidate_spans: Sorted code-point spans to review, at most 32.
        candidate_count: Number of distinct candidates before truncation.
    """

    index: int
    label: str
    status: str
    reason: str
    start: int | None = None
    end: int | None = None
    grapheme_start: int | None = None
    grapheme_end: int | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    affix_count: int = 0
    candidate_spans: tuple[tuple[int, int], ...] = ()
    candidate_count: int = 0

    def __post_init__(self) -> None:
        if not _is_int(self.index) or self.index < 0:
            raise ValueError("decision index must be a non-negative integer")
        if not isinstance(self.label, str) or not _LABEL.fullmatch(self.label):
            raise ValueError("decision label must be a controlled code")
        if not isinstance(self.status, str) or not isinstance(self.reason, str):
            raise ValueError("invalid projection disposition")
        if self.reason not in _STATUS_REASONS.get(self.status, ()):
            raise ValueError("invalid projection disposition")
        offsets = (
            self.start,
            self.end,
            self.grapheme_start,
            self.grapheme_end,
            self.byte_start,
            self.byte_end,
        )
        spans = self.candidate_spans
        if not isinstance(spans, tuple) or len(spans) > MAX_REVIEW_CANDIDATES:
            raise ValueError("invalid review candidate spans")
        for span in spans:
            if (
                not isinstance(span, tuple)
                or len(span) != 2
                or not all(_is_int(value) for value in span)
                or not 0 <= span[0] < span[1]
            ):
                raise ValueError("invalid review candidate spans")
        if (
            not _is_int(self.affix_count)
            or not _is_int(self.candidate_count)
            or self.affix_count < 0
            or self.candidate_count < len(spans)
        ):
            raise ValueError("invalid projection counts")
        if self.status == "projected":
            if (
                not all(_is_int(value) for value in offsets)
                or not 0 <= self.start < self.end
                or not 0 <= self.grapheme_start < self.grapheme_end
                or not 0 <= self.byte_start < self.byte_end
                or spans
            ):
                raise ValueError("inconsistent projected label coordinates")
        elif any(value is not None for value in offsets) or (
            (self.status == "needs_review") != bool(spans)
        ):
            raise ValueError("inconsistent review or unmatched metadata")

    def to_dict(self) -> dict[str, Any]:
        """Serialize controlled codes, offsets and counts only."""
        return {
            "index": self.index,
            "label": self.label,
            "status": self.status,
            "reason": self.reason,
            "start": self.start,
            "end": self.end,
            "grapheme_start": self.grapheme_start,
            "grapheme_end": self.grapheme_end,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "affix_count": self.affix_count,
            "candidate_spans": [list(span) for span in self.candidate_spans],
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True)
class MorphologyProjectionResult:
    """Deterministic, metadata-only result of :func:`project_labels`.

    Attributes:
        language: Normalized profile language tag.
        span_policy: Span policy used for every decision.
        profile_digest: SHA-256 digest of the canonical profile.
        decisions: One decision per input projection, in input order.
    """

    language: str
    span_policy: str
    profile_digest: str
    decisions: tuple[ProjectionDecision, ...]

    @property
    def projected(self) -> tuple[ProjectionDecision, ...]:
        """Return decisions that are safe to export as training labels."""
        return tuple(item for item in self.decisions if item.status == "projected")

    @property
    def review_queue(self) -> tuple[ProjectionDecision, ...]:
        """Return decisions that a reviewer must resolve before export."""
        return tuple(item for item in self.decisions if item.status == "needs_review")

    def reason_counts(self) -> dict[str, int]:
        """Return decision counts keyed by reason code, sorted by code."""
        counts: dict[str, int] = {}
        for item in self.decisions:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        """Serialize codes, offsets, counts and the profile digest only."""
        statuses = {status: 0 for status in PROJECTION_STATUSES}
        for item in self.decisions:
            statuses[item.status] += 1
        return {
            "language": self.language,
            "span_policy": self.span_policy,
            "profile_digest": self.profile_digest,
            "status_counts": statuses,
            "reason_counts": self.reason_counts(),
            "decisions": [item.to_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class _Fold:
    """Folded token text with an absolute source span per folded code point."""

    text: str
    origins: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Analysis:
    """Stem and clitic-host boundaries inside one folded token."""

    lo: int
    hi: int
    host_lo: int
    host_hi: int
    prefixes: int
    suffixes: int


@dataclass(frozen=True)
class _CompiledProfile:
    prefixes: tuple[tuple[str, bool], ...]
    suffixes: tuple[tuple[str, bool], ...]
    alternations: tuple[tuple[str, str, str], ...]
    lexicalized: Mapping[int, frozenset[str]]
    turkic: bool
    max_affixes: int
    min_stem_graphemes: int


@dataclass(frozen=True)
class _LemmaToken:
    fold: str
    variants: tuple[tuple[str, str | None], ...]
    gap_before: bool


@dataclass(frozen=True)
class _Draft:
    index: int
    label: str
    status: str
    reason: str
    span: tuple[int, int] | None = None
    affix_count: int = 0
    candidates: tuple[tuple[int, int], ...] = ()
    candidate_count: int = 0


_Hit = tuple[_Analysis, int]


def _fold_char(char: str, *, turkic: bool) -> str:
    if turkic:
        if char == "I":
            return "\u0131"  # dotless i
        if char == "\u0130":  # dotted capital I
            return "i"
    return char.casefold()


def _fold(text: str, start: int, end: int, *, turkic: bool) -> _Fold:
    """NFC-normalize and case-fold ``text[start:end]`` with source origins."""
    normalized, offset_map = normalize_unicode_with_offsets(text[start:end], "NFC")
    parts: list[str] = []
    origins: list[tuple[int, int]] = []
    for char, (source_start, source_end) in zip(
        normalized, offset_map.source_spans, strict=True
    ):
        folded = _fold_char(char, turkic=turkic)
        parts.append(folded)
        origins.extend([(start + source_start, start + source_end)] * len(folded))
    return _Fold("".join(parts), tuple(origins))


def _fold_text(value: str, *, turkic: bool) -> str:
    return _fold(value, 0, len(value), turkic=turkic).text


def _compile(profile: MorphologyProfile) -> _CompiledProfile:
    turkic = profile.casing == "turkic"

    def side(position: str) -> tuple[tuple[str, bool], ...]:
        rules = {
            (_fold_text(rule.surface, turkic=turkic), rule.kind == "clitic")
            for rule in profile.affixes
            if rule.position == position
        }
        return tuple(sorted(rules))

    # Single-code-point forms cannot be split, so they never block a cut.
    buckets: dict[int, set[str]] = {}
    for form in profile.lexicalized_forms:
        folded = _fold_text(form, turkic=turkic)
        if len(folded) > 1:
            buckets.setdefault(len(folded), set()).add(folded)

    return _CompiledProfile(
        prefixes=side("prefix"),
        suffixes=side("suffix"),
        alternations=tuple(
            sorted(
                {
                    (
                        _fold_text(rule.lemma_edge, turkic=turkic),
                        _fold_text(rule.stem_edge, turkic=turkic),
                        rule.position,
                    )
                    for rule in profile.alternations
                }
            )
        ),
        lexicalized={
            length: frozenset(forms) for length, forms in sorted(buckets.items())
        },
        turkic=turkic,
        max_affixes=profile.max_affixes,
        min_stem_graphemes=profile.min_stem_graphemes,
    )


def _cut_guard(
    fold: str, lexicalized: Mapping[int, frozenset[str]]
) -> Callable[[int], bool]:
    """Return a memoized predicate: does ``cut`` split a lexicalized form?

    Only proposed affix cuts are checked, against forms bucketed by length, so
    a large lexicon does not cost a full scan of every token.
    """
    memo: dict[int, bool] = {}

    def splits(cut: int) -> bool:
        if cut not in memo:
            memo[cut] = any(
                fold[start : start + length] in forms
                for length, forms in lexicalized.items()
                for start in range(
                    max(0, cut - length + 1), min(cut, len(fold) - length + 1)
                )
            )
        return memo[cut]

    return splits


def _side_states(
    fold: str,
    rules: Sequence[tuple[str, bool]],
    *,
    suffix: bool,
    limit: int,
    forbidden: Callable[[int], bool],
) -> tuple[tuple[int, int, int], ...]:
    """Enumerate ``(boundary, host_boundary, count)`` affix-stripping states.

    Clitics may only be removed while every removed affix so far is a clitic,
    which keeps clitics outermost; the host boundary stops at the last clitic.
    """
    origin = len(fold) if suffix else 0
    initial = (origin, origin, True, 0)
    seen = {initial}
    pending = [initial]
    while pending:
        boundary, host, clitic_phase, count = pending.pop()
        if count >= limit:
            continue
        for surface, is_clitic in rules:
            if is_clitic and not clitic_phase:
                continue
            if suffix:
                if not fold.endswith(surface, 0, boundary):
                    continue
                cut = boundary - len(surface)
            else:
                if not fold.startswith(surface, boundary):
                    continue
                cut = boundary + len(surface)
            if forbidden(cut):
                continue
            state = (cut, cut if is_clitic else host, is_clitic, count + 1)
            if state not in seen:
                seen.add(state)
                pending.append(state)
    return tuple(sorted({(boundary, host, count) for boundary, host, _, count in seen}))


def _grapheme_count(value: str) -> int:
    return sum(1 for _ in iter_grapheme_cluster_spans(value))


def _token_analyses(fold: str, compiled: _CompiledProfile) -> tuple[_Analysis, ...]:
    """Return every bounded affix analysis of one folded token."""
    exact = _Analysis(0, len(fold), 0, len(fold), 0, 0)
    if (
        compiled.max_affixes == 0
        or not (compiled.prefixes or compiled.suffixes)
        or len(fold) > MAX_ANALYZED_TOKEN_LENGTH
    ):
        return (exact,)
    forbidden = _cut_guard(fold, compiled.lexicalized)
    prefix_states = _side_states(
        fold,
        compiled.prefixes,
        suffix=False,
        limit=compiled.max_affixes,
        forbidden=forbidden,
    )
    suffix_states = _side_states(
        fold,
        compiled.suffixes,
        suffix=True,
        limit=compiled.max_affixes,
        forbidden=forbidden,
    )
    analyses = []
    for lo, host_lo, prefix_count in prefix_states:
        for hi, host_hi, suffix_count in suffix_states:
            if lo >= hi or prefix_count + suffix_count > compiled.max_affixes:
                continue
            if prefix_count + suffix_count and (
                _grapheme_count(fold[lo:hi]) < compiled.min_stem_graphemes
            ):
                continue
            analyses.append(
                _Analysis(lo, hi, host_lo, host_hi, prefix_count, suffix_count)
            )
    return tuple(analyses)


def _build_index(
    folds: Sequence[_Fold], compiled: _CompiledProfile
) -> dict[str, list[tuple[int, _Analysis]]]:
    index: dict[str, list[tuple[int, _Analysis]]] = {}
    cache: dict[str, tuple[_Analysis, ...]] = {}
    for token_index, fold in enumerate(folds):
        analyses = cache.get(fold.text)
        if analyses is None:
            analyses = _token_analyses(fold.text, compiled)
            cache[fold.text] = analyses
        for analysis in analyses:
            stem = fold.text[analysis.lo : analysis.hi]
            index.setdefault(stem, []).append((token_index, analysis))
    return index


def _lemma_tokens(lemma: str, compiled: _CompiledProfile) -> tuple[_LemmaToken, ...]:
    tokens: list[_LemmaToken] = []
    previous_end: int | None = None
    for token in indic_word_tokenize(lemma):
        folded = _fold_text(token.text, turkic=compiled.turkic)
        variants: list[tuple[str, str | None]] = [(folded, None)]
        for lemma_edge, stem_edge, position in compiled.alternations:
            if len(folded) <= len(lemma_edge):
                continue
            if position == "suffix" and folded.endswith(lemma_edge):
                kept = folded[: len(folded) - len(lemma_edge)]
                variants.append((kept + stem_edge, "suffix"))
            elif position == "prefix" and folded.startswith(lemma_edge):
                variants.append((stem_edge + folded[len(lemma_edge) :], "prefix"))
        tokens.append(
            _LemmaToken(
                fold=folded,
                variants=tuple(
                    sorted(set(variants), key=lambda item: (item[0], item[1] or ""))
                ),
                gap_before=previous_end is not None and token.start > previous_end,
            )
        )
        previous_end = token.end
    return tuple(tokens)


def _token_hits(
    lemma_token: _LemmaToken,
    index: Mapping[str, Sequence[tuple[int, _Analysis]]],
) -> dict[int, list[_Hit]]:
    hits: dict[int, list[_Hit]] = {}
    for form, side in lemma_token.variants:
        for token_index, analysis in index.get(form, ()):
            if side == "suffix" and not analysis.suffixes:
                continue
            if side == "prefix" and not analysis.prefixes:
                continue
            if side is not None:
                rank = _ALTERNATION
            elif analysis.prefixes or analysis.suffixes:
                rank = _AFFIX
            else:
                rank = _EXACT
            hits.setdefault(token_index, []).append((analysis, rank))
    return hits


def _boundary_start(fold: _Fold, boundary: int, token: SpanToken) -> tuple[int, bool]:
    if boundary <= 0:
        return token.start, True
    if boundary >= len(fold.origins):
        return token.end, True
    exact = fold.origins[boundary - 1][1] <= fold.origins[boundary][0]
    return fold.origins[boundary][0], exact


def _boundary_end(fold: _Fold, boundary: int, token: SpanToken) -> tuple[int, bool]:
    if boundary >= len(fold.origins):
        return token.end, True
    if boundary <= 0:
        return token.start, True
    exact = fold.origins[boundary - 1][1] <= fold.origins[boundary][0]
    return fold.origins[boundary - 1][1], exact


def _span_edge(
    fold: _Fold, token: SpanToken, analysis: _Analysis, policy: str, *, end: bool
) -> tuple[int, bool]:
    if policy == "token":
        return (token.end if end else token.start), True
    if end:
        boundary = analysis.host_hi if policy == "host" else analysis.hi
        return _boundary_end(fold, boundary, token)
    boundary = analysis.host_lo if policy == "host" else analysis.lo
    return _boundary_start(fold, boundary, token)


def _gaps_match(
    text: str, tokens: Sequence[SpanToken], first: int, lemma: Sequence[_LemmaToken]
) -> bool:
    for offset in range(1, len(lemma)):
        gap = text[tokens[first + offset - 1].end : tokens[first + offset].start]
        if bool(gap) != lemma[offset].gap_before:
            return False
    return True


def _touches_window(
    tokens: Sequence[SpanToken], first: int, count: int, window: tuple[int, int] | None
) -> bool:
    if window is None:
        return True
    return any(
        token.start < window[1] and token.end > window[0]
        for token in tokens[first : first + count]
    )


def _location_spans(
    *,
    first: int,
    lemma: Sequence[_LemmaToken],
    hits: Sequence[Mapping[int, list[_Hit]]],
    tokens: Sequence[SpanToken],
    folds: Sequence[_Fold],
    policy: str,
    has_break: Callable[[int], bool],
) -> dict[tuple[int, int], tuple[bool, int, int]]:
    """Map each label span at one location to ``(aligned, rank, affixes)``."""
    last = first + len(lemma) - 1
    middle_rank = 0
    middle_affixes = 0
    for offset in range(1, len(lemma) - 1):
        best = min(
            (rank, analysis.prefixes + analysis.suffixes)
            for analysis, rank in hits[offset][first + offset]
        )
        middle_rank = max(middle_rank, best[0])
        middle_affixes += best[1]
    if len(lemma) == 1:
        pairs = [(hit, hit) for hit in hits[0][first]]
    else:
        pairs = [(head, tail) for head in hits[0][first] for tail in hits[-1][last]]
    spans: dict[tuple[int, int], tuple[bool, int, int]] = {}
    for (head, head_rank), (tail, tail_rank) in pairs:
        start, start_exact = _span_edge(
            folds[first], tokens[first], head, policy, end=False
        )
        end, end_exact = _span_edge(folds[last], tokens[last], tail, policy, end=True)
        if start >= end:
            continue
        aligned = start_exact and end_exact and has_break(start) and has_break(end)
        rank = max(head_rank, tail_rank, middle_rank)
        affixes = head.prefixes + head.suffixes + middle_affixes
        if len(lemma) > 1:
            affixes += tail.prefixes + tail.suffixes
        previous = spans.get((start, end))
        if previous is None:
            spans[(start, end)] = (aligned, rank, affixes)
        else:
            spans[(start, end)] = (
                previous[0] and aligned,
                *min(previous[1:], (rank, affixes)),
            )
    return spans


def _review(
    index: int,
    label: str,
    reason: str,
    spans: Iterable[tuple[int, int]],
    *,
    text: str,
    has_break: Callable[[int], bool],
) -> _Draft:
    # Only spans that bisect a cluster pay for the shared outward snap.
    candidates = sorted(
        {
            (start, end)
            if has_break(start) and has_break(end)
            else snap_span_to_grapheme_boundaries(start, end, text)
            for start, end in spans
        }
    )
    return _Draft(
        index=index,
        label=label,
        status="needs_review",
        reason=reason,
        candidates=tuple(candidates[:MAX_REVIEW_CANDIDATES]),
        candidate_count=len(candidates),
    )


def _decide(
    index: int,
    projection: LabelProjection,
    lemma: Sequence[_LemmaToken],
    *,
    text: str,
    tokens: Sequence[SpanToken],
    folds: Sequence[_Fold],
    term_index: Mapping[str, Sequence[tuple[int, _Analysis]]],
    policy: str,
    has_break: Callable[[int], bool],
) -> _Draft:
    hits = [_token_hits(token, term_index) for token in lemma]
    count = len(lemma)
    locations: dict[int, dict[tuple[int, int], tuple[bool, int, int]]] = {}
    for first in sorted(hits[0]):
        if first + count > len(tokens):
            continue
        if not all(first + offset in hits[offset] for offset in range(count)):
            continue
        if not _gaps_match(text, tokens, first, lemma) or not _touches_window(
            tokens, first, count, projection.window
        ):
            continue
        spans = _location_spans(
            first=first,
            lemma=lemma,
            hits=hits,
            tokens=tokens,
            folds=folds,
            policy=policy,
            has_break=has_break,
        )
        if spans:
            locations[first] = spans

    label = projection.label
    review = {"text": text, "has_break": has_break}
    if not locations:
        partial = []
        for first in range(len(tokens) - count + 1):
            if not _touches_window(tokens, first, count, projection.window):
                continue
            if not _gaps_match(text, tokens, first, lemma):
                continue
            contained = [
                any(form in folds[first + offset].text for form, _ in token.variants)
                for offset, token in enumerate(lemma)
            ]
            full = [first + offset in hits[offset] for offset in range(count)]
            if all(c or f for c, f in zip(contained, full, strict=True)):
                partial.append((tokens[first].start, tokens[first + count - 1].end))
        if partial:
            return _review(index, label, "unanalyzed_affix", partial, **review)
        return _Draft(index=index, label=label, status="unmatched", reason="no_match")

    if len(locations) > 1:
        every_span = [span for spans in locations.values() for span in spans]
        return _review(index, label, "multiple_matches", every_span, **review)
    (spans,) = locations.values()
    if len(spans) > 1:
        return _review(index, label, "ambiguous_segmentation", spans, **review)
    ((span, (aligned, rank, affixes)),) = spans.items()
    if not aligned:
        return _review(index, label, "grapheme_bisection", [span], **review)
    return _Draft(
        index=index,
        label=label,
        status="projected",
        reason=PROJECTED_REASONS[rank],
        span=span,
        affix_count=affixes,
    )


def _flag_overlaps(
    drafts: Sequence[_Draft], *, text: str, has_break: Callable[[int], bool]
) -> list[_Draft]:
    """Route overlapping projected spans with different extents or labels.

    Identical duplicates (same span and label) stay projected; nested,
    crossing or relabelled spans would corrupt flat token-classification
    labels, so every participant goes to review.
    """
    projected = sorted(
        (draft for draft in drafts if draft.span is not None),
        key=lambda draft: (draft.span[0], draft.span[1], draft.index),
    )
    conflicted: set[int] = set()
    active: list[_Draft] = []
    for draft in projected:
        active = [other for other in active if other.span[1] > draft.span[0]]
        for other in active:
            if other.span != draft.span or other.label != draft.label:
                conflicted.update((other.index, draft.index))
        active.append(draft)
    return [
        _review(
            draft.index,
            draft.label,
            "overlapping_projection",
            [draft.span],
            text=text,
            has_break=has_break,
        )
        if draft.index in conflicted
        else draft
        for draft in drafts
    ]


def _coordinate_maps(
    text: str, offsets: Iterable[int]
) -> tuple[dict[int, int], dict[int, int]]:
    """Return UTF-8 byte and grapheme indices for grapheme-boundary offsets."""
    targets = sorted(set(offsets))
    byte_map: dict[int, int] = {}
    position = 0
    byte_count = 0
    for offset in targets:
        byte_count += len(text[position:offset].encode("utf-8"))
        position = offset
        byte_map[offset] = byte_count
    grapheme_map: dict[int, int] = {}
    pointer = 0
    while pointer < len(targets) and targets[pointer] == 0:
        grapheme_map[0] = 0
        pointer += 1
    for cluster_count, (_, cluster_end) in enumerate(
        iter_grapheme_cluster_spans(text), start=1
    ):
        if pointer == len(targets):
            break
        while pointer < len(targets) and targets[pointer] <= cluster_end:
            if targets[pointer] == cluster_end:
                grapheme_map[cluster_end] = cluster_count
            pointer += 1
    return byte_map, grapheme_map


def project_labels(
    text: str,
    projections: Sequence[LabelProjection],
    profile: MorphologyProfile,
    *,
    span_policy: str = "host",
) -> MorphologyProjectionResult:
    """Project labels through a morphology profile with exact source offsets.

    Each projection lemma is tokenized and folded like the target text (NFC,
    then profile case folding). A text token matches a lemma word when
    removing at most ``max_affixes`` configured affixes leaves the lemma word
    or one of its alternated stems. Multi-word lemmas must match consecutive
    tokens with the same spacing. The label span depends on ``span_policy``:

    * ``"token"`` labels whole matched tokens, affixes and clitics included.
    * ``"host"`` (default) excludes outer clitics but keeps inflection.
    * ``"stem"`` labels only the material left after removing every affix.

    A projection is emitted only when exactly one location and one
    grapheme-safe span remain. Otherwise it is routed to review:
    ``multiple_matches`` (several locations), ``ambiguous_segmentation``
    (several spans at one location), ``grapheme_bisection`` (a label edge
    would cut a grapheme cluster or a case-folding expansion),
    ``unanalyzed_affix`` (the lemma only occurs inside a token with
    unconfigured material, such as a compound) or ``overlapping_projection``
    (projected spans overlap with a different extent or label). Projections
    with no candidate at all are ``unmatched``.

    Args:
        text: Exact, unnormalized target-language source text.
        projections: Labels to project, at most 10,000.
        profile: Target-language morphology configuration.
        span_policy: ``"token"``, ``"host"`` or ``"stem"``.

    Returns:
        One metadata-only decision per projection, in input order.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a window falls outside the text, the batch is too
            large, or the span policy is unknown.
        openmed.core.errors.InputError: If ``text`` exceeds the shared
            tokenizer's validated input budget.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(profile, MorphologyProfile):
        raise TypeError("profile must be a MorphologyProfile")
    if not isinstance(span_policy, str) or span_policy not in SPAN_POLICIES:
        raise ValueError("span_policy must be 'token', 'host' or 'stem'")
    if isinstance(projections, (str, bytes)) or not isinstance(projections, Sequence):
        raise TypeError("projections must be a sequence of LabelProjection values")
    if len(projections) > MAX_PROJECTIONS:
        raise ValueError("provide at most 10000 label projections")
    for projection in projections:
        if not isinstance(projection, LabelProjection):
            raise TypeError("projections must be a sequence of LabelProjection values")
        if projection.window is not None and projection.window[1] > len(text):
            raise ValueError("projection window must lie inside the text")

    compiled = _compile(profile)
    tokens = tuple(indic_word_tokenize(text))
    folds = tuple(
        _fold(text, token.start, token.end, turkic=compiled.turkic) for token in tokens
    )
    term_index = _build_index(folds, compiled)
    has_break = grapheme_break_checker(text)
    lemma_cache: dict[str, tuple[_LemmaToken, ...]] = {}
    drafts = []
    for index, projection in enumerate(projections):
        lemma = lemma_cache.get(projection.lemma)
        if lemma is None:
            lemma = _lemma_tokens(projection.lemma, compiled)
            lemma_cache[projection.lemma] = lemma
        drafts.append(
            _decide(
                index,
                projection,
                lemma,
                text=text,
                tokens=tokens,
                folds=folds,
                term_index=term_index,
                policy=span_policy,
                has_break=has_break,
            )
        )
    drafts = _flag_overlaps(drafts, text=text, has_break=has_break)
    byte_map, grapheme_map = _coordinate_maps(
        text,
        (edge for draft in drafts if draft.span is not None for edge in draft.span),
    )
    decisions = []
    for draft in drafts:
        if draft.span is None:
            decisions.append(
                ProjectionDecision(
                    index=draft.index,
                    label=draft.label,
                    status=draft.status,
                    reason=draft.reason,
                    candidate_spans=draft.candidates,
                    candidate_count=draft.candidate_count,
                )
            )
            continue
        start, end = draft.span
        decisions.append(
            ProjectionDecision(
                index=draft.index,
                label=draft.label,
                status=draft.status,
                reason=draft.reason,
                start=start,
                end=end,
                grapheme_start=grapheme_map[start],
                grapheme_end=grapheme_map[end],
                byte_start=byte_map[start],
                byte_end=byte_map[end],
                affix_count=draft.affix_count,
            )
        )
    return MorphologyProjectionResult(
        language=profile.language,
        span_policy=span_policy,
        profile_digest=profile.digest,
        decisions=tuple(decisions),
    )


__all__ = [
    "AFFIX_KINDS",
    "AFFIX_POSITIONS",
    "CASING_MODES",
    "PROJECTED_REASONS",
    "PROJECTION_STATUSES",
    "REVIEW_REASONS",
    "SPAN_POLICIES",
    "UNMATCHED_REASONS",
    "AffixRule",
    "LabelProjection",
    "MorphologyProfile",
    "MorphologyProjectionResult",
    "ProjectionDecision",
    "StemAlternation",
    "project_labels",
]
