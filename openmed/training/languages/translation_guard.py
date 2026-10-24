"""Clinical terminology guard for local translation round trips.

Back-translation augmentation can return fluent text that silently changes a
dose, drops a negation, flips laterality, expands an abbreviation, or rewrites
a de-identification placeholder. This module wraps a caller-supplied local
translator and makes those changes detectable before an example reaches a
training set:

* Invariant items (annotated spans, PHI placeholders, glossary and protected
  terms, doses and measurements, identifiers, numbers, and abbreviations) are
  replaced with opaque placeholders before translation, so the translator never
  sees them, and are restored afterwards. A missing, duplicated, unknown, or
  altered placeholder rejects the translation.
* Translatable assertion cues (negation, uncertainty, hypothetical, historical)
  are verified by per-axis cue counts from the shared ConText language packs.
  English round trips also verify severity, laterality, and position modifiers.
* Digits outside protected items must survive as a multiset.

The guard never translates text itself and never opens a network connection; it
only calls the injected translator. Results keep the translated text for the
caller, but reports carry counts, reason codes, labels, and offsets only.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Final, Protocol

from openmed.clinical.lexicons import (
    ClinicalCueLexicon,
    available_clinical_cue_languages,
    canonical_unit_alias,
    get_clinical_cue_lexicon,
    get_clinical_norm_lexicon,
    normalize_language,
)
from openmed.clinical.medication_sig import normalize_frequency
from openmed.clinical.severity_laterality import extract_severity_laterality
from openmed.clinical.units import parse_unit
from openmed.training.synthetic.offset_projection import (
    SpanAnnotation,
    SpanProjectionError,
    normalize_span_annotations,
    validate_span_integrity,
)

TRANSLATION_GUARD_SCHEMA_VERSION: Final = 1
DEFAULT_PLACEHOLDER_TEMPLATE: Final = "__OMP{index}__"
GUARDED_SYNTHETIC_SOURCE: Final = "guarded_backtranslation"
TRANSLATION_GUARD_ADVISORY: Final = (
    "Translation guard checks are deterministic surface checks for training-data "
    "augmentation. They do not certify translation quality or clinical "
    "equivalence and must not drive autonomous clinical decisions."
)
ASSERTION_AXES: Final[tuple[str, ...]] = (
    "negation",
    "uncertainty",
    "hypothetical",
    "historical",
)
DEFAULT_CROSS_LINGUAL_ASSERTION_AXES: Final[tuple[str, ...]] = ("negation",)
MODIFIER_KINDS: Final[tuple[str, ...]] = ("laterality", "severity", "position")

# Severity/laterality/position cues ship as an English lexicon only.
_MODIFIER_LANGUAGES: Final = frozenset({"en"})


class TranslationGuardError(ValueError):
    """Raised for invalid guard configuration or input.

    Messages name the failing field or offsets only and never echo text.
    """


class ProtectedKind(str, Enum):
    """Category of an item frozen behind a placeholder during translation."""

    ANNOTATED_SPAN = "annotated_span"
    PHI_PLACEHOLDER = "phi_placeholder"
    CLINICAL_TERM = "clinical_term"
    MEASUREMENT = "measurement"
    IDENTIFIER = "identifier"
    NUMBER = "number"
    ABBREVIATION = "abbreviation"


DEFAULT_PROTECTED_KINDS: Final[frozenset[ProtectedKind]] = frozenset(ProtectedKind)


class TranslationGuardReason(str, Enum):
    """Stable reason codes for rejected or flagged translations."""

    INVALID_INPUT = "invalid_input"
    SAME_LANGUAGE = "same_language"
    PLACEHOLDER_COLLISION = "placeholder_collision"
    TRANSLATOR_ERROR = "translator_error"
    EMPTY_TRANSLATION = "empty_translation"
    PROTECTED_ITEM_DROPPED = "protected_item_dropped"
    PROTECTED_ITEM_DUPLICATED = "protected_item_duplicated"
    UNKNOWN_PLACEHOLDER = "unknown_placeholder"
    PLACEHOLDER_RESIDUE = "placeholder_residue"
    NUMBER_CHANGED = "number_changed"
    NEGATION_CHANGED = "negation_changed"
    UNCERTAINTY_CHANGED = "uncertainty_changed"
    HYPOTHETICAL_CHANGED = "hypothetical_changed"
    HISTORICAL_CHANGED = "historical_changed"
    LATERALITY_CHANGED = "laterality_changed"
    SEVERITY_CHANGED = "severity_changed"
    POSITION_CHANGED = "position_changed"
    ASSERTION_UNVERIFIABLE = "assertion_unverifiable"
    MODIFIER_UNVERIFIABLE = "modifier_unverifiable"
    SPAN_INTEGRITY_FAILED = "span_integrity_failed"


_REASON_ORDER: Final = {
    reason: rank for rank, reason in enumerate(TranslationGuardReason)
}
_AXIS_REASONS: Final[Mapping[str, TranslationGuardReason]] = {
    "negation": TranslationGuardReason.NEGATION_CHANGED,
    "uncertainty": TranslationGuardReason.UNCERTAINTY_CHANGED,
    "hypothetical": TranslationGuardReason.HYPOTHETICAL_CHANGED,
    "historical": TranslationGuardReason.HISTORICAL_CHANGED,
}
_MODIFIER_REASONS: Final[Mapping[str, TranslationGuardReason]] = {
    "laterality": TranslationGuardReason.LATERALITY_CHANGED,
    "severity": TranslationGuardReason.SEVERITY_CHANGED,
    "position": TranslationGuardReason.POSITION_CHANGED,
}


class LocalTranslator(Protocol):
    """Local translation backend contract accepted by the guard."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate ``text`` from ``source_lang`` to ``target_lang``."""


TranslateFn = Callable[[str, str, str], str]


@dataclass(frozen=True)
class TranslationGuardConfig:
    """Protection and verification policy for guarded translation.

    Attributes:
        placeholder_template: Placeholder shape with exactly one ``{index}``
            field, a non-empty prefix and suffix, and no whitespace.
        protected_kinds: Detected item kinds to freeze. Caller-annotated spans
            are always protected.
        protected_terms: Clinical terms frozen verbatim, matched
            case-insensitively on token boundaries.
        glossary: Terminology constraints keyed by ``(source, target)``
            language pair. A matching term or annotated span is frozen and
            restored as the glossary target term.
        assertion_axes: Cue axes compared when source and output share a
            language, including the end of a round trip.
        cross_lingual_assertion_axes: Cue axes compared across languages.
        verify_modifiers: Compare severity, laterality, and position cues on
            English round trips.
        verify_numbers: Require digits outside protected items to survive.
        fail_on_unverifiable: Reject when a required assertion check has no
            language pack instead of only flagging it.
    """

    placeholder_template: str = DEFAULT_PLACEHOLDER_TEMPLATE
    protected_kinds: frozenset[ProtectedKind] = DEFAULT_PROTECTED_KINDS
    protected_terms: tuple[str, ...] = ()
    glossary: Mapping[tuple[str, str], Mapping[str, str]] = field(default_factory=dict)
    assertion_axes: tuple[str, ...] = ASSERTION_AXES
    cross_lingual_assertion_axes: tuple[str, ...] = DEFAULT_CROSS_LINGUAL_ASSERTION_AXES
    verify_modifiers: bool = True
    verify_numbers: bool = True
    fail_on_unverifiable: bool = True

    def __post_init__(self) -> None:
        _placeholder_codec(_validated_template(self.placeholder_template))
        try:
            kinds = frozenset(ProtectedKind(kind) for kind in self.protected_kinds)
        except (TypeError, ValueError):
            raise TranslationGuardError("protected_kinds has an unknown kind") from None
        object.__setattr__(self, "protected_kinds", kinds)
        if isinstance(self.protected_terms, str):
            raise TranslationGuardError("protected_terms must be a sequence of terms")
        object.__setattr__(
            self,
            "protected_terms",
            tuple(
                dict.fromkeys(_validated_term(term) for term in self.protected_terms)
            ),
        )
        object.__setattr__(self, "glossary", _normalized_glossary(self.glossary))
        object.__setattr__(
            self, "assertion_axes", _validated_axes(self.assertion_axes, "assertion")
        )
        object.__setattr__(
            self,
            "cross_lingual_assertion_axes",
            _validated_axes(self.cross_lingual_assertion_axes, "cross_lingual"),
        )

    def glossary_for(
        self, source_language: str, target_language: str
    ) -> dict[str, str]:
        """Return the case-folded glossary for one language pair.

        Args:
            source_language: Language of the text being protected.
            target_language: Language the glossary terms are restored in.

        Returns:
            A mapping from case-folded source term to target term.
        """

        key = (normalize_language(source_language), normalize_language(target_language))
        return dict(self.glossary.get(key, {}))


@dataclass(frozen=True)
class ProtectedItem:
    """One invariant item frozen behind a placeholder.

    ``surface`` and ``replacement`` hold raw text for restoration and are
    excluded from ``repr`` and from :meth:`to_dict`.
    """

    index: int
    kind: ProtectedKind
    start: int
    end: int
    label: str | None = None
    surface: str = field(default="", repr=False)
    replacement: str = field(default="", repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return an offsets-only record."""

        return {
            "end": self.end,
            "index": self.index,
            "kind": self.kind.value,
            "label": self.label,
            "mapped": self.replacement != self.surface,
            "start": self.start,
        }


@dataclass(frozen=True)
class ProtectedText:
    """Text with every protected item replaced by its placeholder."""

    language: str
    items: tuple[ProtectedItem, ...]
    masked_text: str = field(repr=False)

    def kind_counts(self) -> dict[str, int]:
        """Return protected item counts keyed by kind value."""

        return _kind_counts(self.items)


@dataclass(frozen=True)
class SpanMapEntry:
    """Offsets of one protected item before and after a translation leg."""

    index: int
    kind: ProtectedKind
    label: str | None
    source_start: int
    source_end: int
    target_start: int
    target_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready offsets record."""

        return {
            "index": self.index,
            "kind": self.kind.value,
            "label": self.label,
            "source": [self.source_start, self.source_end],
            "target": [self.target_start, self.target_end],
        }


@dataclass(frozen=True)
class RoundTripSpanMapEntry:
    """Offsets of one protected item across source, pivot, and back text."""

    index: int
    kind: ProtectedKind
    label: str | None
    source_start: int
    source_end: int
    pivot_start: int
    pivot_end: int
    back_start: int
    back_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready offsets record."""

        return {
            "back": [self.back_start, self.back_end],
            "index": self.index,
            "kind": self.kind.value,
            "label": self.label,
            "pivot": [self.pivot_start, self.pivot_end],
            "source": [self.source_start, self.source_end],
        }


@dataclass(frozen=True)
class GuardedTranslation:
    """Outcome of one guarded translation leg.

    ``text`` and ``spans`` are populated only when the translation is
    accepted, so a rejected translation fails closed.
    """

    source_language: str
    target_language: str
    reasons: tuple[TranslationGuardReason, ...]
    warnings: tuple[TranslationGuardReason, ...]
    items: tuple[ProtectedItem, ...]
    span_map: tuple[SpanMapEntry, ...] = ()
    problem_indices: tuple[int, ...] = ()
    spans: tuple[SpanAnnotation, ...] = field(default=(), repr=False)
    text: str | None = field(default=None, repr=False)

    @property
    def accepted(self) -> bool:
        """Whether every protection and verification check passed."""

        return not self.reasons

    @property
    def protected_counts(self) -> dict[str, int]:
        """Protected item counts keyed by kind value."""

        return _kind_counts(self.items)

    def to_report(self) -> dict[str, Any]:
        """Return a counts, codes, and offsets report without raw text."""

        return {
            "accepted": self.accepted,
            "advisory": TRANSLATION_GUARD_ADVISORY,
            "problem_indices": list(self.problem_indices),
            "protected_counts": self.protected_counts,
            "protected_items": [item.to_dict() for item in self.items],
            "reasons": _codes(self.reasons),
            "schema_version": TRANSLATION_GUARD_SCHEMA_VERSION,
            "source_language": self.source_language,
            "span_map": [entry.to_dict() for entry in self.span_map],
            "target_language": self.target_language,
            "warnings": _codes(self.warnings),
        }


@dataclass(frozen=True)
class GuardedRoundTrip:
    """Outcome of a guarded source to pivot to source round trip.

    ``text`` and ``spans`` describe the back-translated source-language text
    and are populated only when the full round trip is accepted.
    """

    source_language: str
    pivot_language: str
    reasons: tuple[TranslationGuardReason, ...]
    warnings: tuple[TranslationGuardReason, ...]
    forward: GuardedTranslation
    backward: GuardedTranslation | None = None
    round_trip_reasons: tuple[TranslationGuardReason, ...] = ()
    span_map: tuple[RoundTripSpanMapEntry, ...] = ()
    spans: tuple[SpanAnnotation, ...] = field(default=(), repr=False)
    text: str | None = field(default=None, repr=False)

    @property
    def accepted(self) -> bool:
        """Whether both legs and the end-to-end checks passed."""

        return not self.reasons

    @property
    def pivot_text(self) -> str | None:
        """Accepted pivot-language text, or ``None`` when the round trip failed."""

        return self.forward.text if self.accepted else None

    @property
    def protected_counts(self) -> dict[str, int]:
        """Protected item counts keyed by kind value."""

        return self.forward.protected_counts

    def to_report(self) -> dict[str, Any]:
        """Return a counts, codes, and offsets report without raw text."""

        backward = self.backward
        return {
            "accepted": self.accepted,
            "advisory": TRANSLATION_GUARD_ADVISORY,
            "pivot_language": self.pivot_language,
            "protected_counts": self.protected_counts,
            "protected_items": [item.to_dict() for item in self.forward.items],
            "reasons": _codes(self.reasons),
            "schema_version": TRANSLATION_GUARD_SCHEMA_VERSION,
            "source_language": self.source_language,
            "span_map": [entry.to_dict() for entry in self.span_map],
            "stages": {
                "backward": None if backward is None else _stage_report(backward),
                "forward": _stage_report(self.forward),
                "round_trip": {"reasons": _codes(self.round_trip_reasons)},
            },
            "warnings": _codes(self.warnings),
        }


@dataclass(frozen=True)
class GuardedExample:
    """An accepted back-translated training example with its span map."""

    index: int
    example_id: str
    language: str
    pivot_language: str
    span_map: tuple[RoundTripSpanMapEntry, ...]
    text: str = field(repr=False)
    spans: tuple[SpanAnnotation, ...] = field(repr=False)
    pivot_text: str = field(repr=False)
    pivot_spans: tuple[SpanAnnotation, ...] = field(repr=False)

    @property
    def transform(self) -> str:
        """Transform label recorded in training metadata."""

        return (
            f"backtranslation:{self.language}->{self.pivot_language}->{self.language}"
        )

    def to_training_item(self) -> dict[str, Any]:
        """Return a JSONL-ready span example that keeps the guard span map."""

        spans = [span.to_dict() for span in self.spans]
        return {
            "gold_spans": spans,
            "id": f"{self.example_id}-guarded-bt-{self.pivot_language}",
            "is_synthetic": True,
            "labels": spans,
            "language": self.language,
            "metadata": {
                "augmentation_only": True,
                "synthetic": True,
                "synthetic_source": GUARDED_SYNTHETIC_SOURCE,
                "transform": self.transform,
                "translation_guard": {
                    "pivot_language": self.pivot_language,
                    "schema_version": TRANSLATION_GUARD_SCHEMA_VERSION,
                    "span_map": [entry.to_dict() for entry in self.span_map],
                },
            },
            "source_id": self.example_id,
            "synthetic_source": GUARDED_SYNTHETIC_SOURCE,
            "text": self.text,
        }


@dataclass(frozen=True)
class ExampleDecision:
    """Report row for one example; it carries no example text or identifier."""

    index: int
    accepted: bool
    reasons: tuple[TranslationGuardReason, ...]
    warnings: tuple[TranslationGuardReason, ...]
    protected_counts: Mapping[str, int]
    span_map: tuple[RoundTripSpanMapEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready decision record."""

        return {
            "accepted": self.accepted,
            "index": self.index,
            "protected_counts": dict(sorted(self.protected_counts.items())),
            "reasons": _codes(self.reasons),
            "span_map": [entry.to_dict() for entry in self.span_map],
            "warnings": _codes(self.warnings),
        }


@dataclass(frozen=True)
class TranslationGuardReport:
    """Aggregate counts, reason codes, and offsets for a guarded batch."""

    pivot_language: str
    total: int
    accepted: int
    rejected: int
    reason_counts: Mapping[str, int]
    warning_counts: Mapping[str, int]
    protected_counts: Mapping[str, int]
    decisions: tuple[ExampleDecision, ...]
    schema_version: int = TRANSLATION_GUARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report without raw text or example identifiers."""

        return {
            "accepted": self.accepted,
            "advisory": TRANSLATION_GUARD_ADVISORY,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "pivot_language": self.pivot_language,
            "protected_counts": dict(sorted(self.protected_counts.items())),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "rejected": self.rejected,
            "schema_version": self.schema_version,
            "total": self.total,
            "warning_counts": dict(sorted(self.warning_counts.items())),
        }

    def to_json(self) -> str:
        """Return canonical sorted-key JSON for the report."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """SHA-256 reference of the canonical report JSON."""

        digest = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True)
class TranslationGuardBatch:
    """Accepted examples plus the aggregate report for a guarded batch."""

    accepted: tuple[GuardedExample, ...]
    report: TranslationGuardReport


class TranslationGuard:
    """Protect and verify clinical facts around an injected local translator.

    The guard calls ``translator`` exactly once per translation leg with
    placeholder-masked text. It never translates text itself, never contacts a
    service, and never logs. Frozen surfaces are not passed to the translator.

    Args:
        translator: A local backend exposing
            ``translate(text, source_lang, target_lang)`` or a callable with the
            same arguments.
        config: Optional protection and verification policy.

    Raises:
        TypeError: If ``translator`` is neither callable nor a translator.
    """

    def __init__(
        self,
        translator: LocalTranslator | TranslateFn,
        *,
        config: TranslationGuardConfig | None = None,
    ) -> None:
        self._translate_fn = _resolve_translator(translator)
        self.config = config or TranslationGuardConfig()

    def protect(
        self,
        text: str,
        *,
        language: str,
        target_language: str | None = None,
        spans: Iterable[Any] = (),
    ) -> ProtectedText:
        """Detect protected items and mask them with placeholders.

        Args:
            text: Source text.
            language: Language of ``text``.
            target_language: Language glossary terms are restored in; defaults
                to ``language``.
            spans: Caller-annotated spans (mappings or ``SpanAnnotation``) that
                must survive translation. They must not overlap.

        Returns:
            The masked text and protected items ordered by offset.

        Raises:
            TranslationGuardError: If the text or spans are invalid.
        """

        _require_text(text)
        source = normalize_language(language)
        target = normalize_language(target_language) if target_language else source
        annotated = _normalized_input_spans(text, spans)
        candidates = _detect_candidates(text, source, target, annotated, self.config)
        items = tuple(
            ProtectedItem(
                index=index,
                kind=candidate.kind,
                start=candidate.start,
                end=candidate.end,
                label=candidate.label,
                surface=text[candidate.start : candidate.end],
                replacement=candidate.replacement,
                metadata=candidate.metadata,
            )
            for index, candidate in enumerate(candidates)
        )
        codec = _placeholder_codec(self.config.placeholder_template)
        return ProtectedText(
            language=source,
            items=items,
            masked_text=_mask(text, items, codec),
        )

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        spans: Iterable[Any] = (),
    ) -> GuardedTranslation:
        """Translate once with protection and verification.

        Args:
            text: Source text.
            source_language: Language of ``text``.
            target_language: Output language.
            spans: Caller-annotated spans to protect and project.

        Returns:
            A guarded translation; ``text`` is ``None`` when rejected.

        Raises:
            TranslationGuardError: If the text or spans are invalid.
        """

        source = normalize_language(source_language)
        target = normalize_language(target_language)
        protected = self.protect(
            text, language=source, target_language=target, spans=spans
        )
        same_language = source == target
        result = self._run_leg(
            text,
            protected.items,
            source_language=source,
            target_language=target,
            assertion_axes=(
                self.config.assertion_axes
                if same_language
                else self.config.cross_lingual_assertion_axes
            ),
            unverifiable_rejects=self.config.fail_on_unverifiable,
        )
        if not result.accepted or not self.config.verify_modifiers:
            return result
        source_regions = _regions((item.start, item.end) for item in protected.items)
        if same_language:
            changed = _modifier_changes(
                text,
                source_regions,
                result.text or "",
                _regions((e.target_start, e.target_end) for e in result.span_map),
                source,
            )
            return _rejected_leg(result, changed) if changed else result
        if source in _MODIFIER_LANGUAGES and _modifier_counts(text, source_regions):
            return _with_warnings(
                result, (TranslationGuardReason.MODIFIER_UNVERIFIABLE,)
            )
        return result

    def round_trip(
        self,
        text: str,
        *,
        source_language: str,
        pivot_language: str,
        spans: Iterable[Any] = (),
    ) -> GuardedRoundTrip:
        """Translate to a pivot language and back, verifying both legs.

        Protected items are restored as glossary terms (or verbatim) in the
        pivot text, then re-protected at their pivot offsets and restored as
        their original source surfaces in the back translation.

        Args:
            text: Source text.
            source_language: Language of ``text``.
            pivot_language: Intermediate language; must differ from the source.
            spans: Caller-annotated spans to protect and project.

        Returns:
            A guarded round trip; ``text`` is ``None`` when rejected.

        Raises:
            TranslationGuardError: If the input is invalid or both languages
                are the same.
        """

        source = normalize_language(source_language)
        pivot = normalize_language(pivot_language)
        if source == pivot:
            raise TranslationGuardError(
                "pivot_language must differ from source_language"
            )
        protected = self.protect(
            text, language=source, target_language=pivot, spans=spans
        )
        forward = self._run_leg(
            text,
            protected.items,
            source_language=source,
            target_language=pivot,
            assertion_axes=self.config.cross_lingual_assertion_axes,
            unverifiable_rejects=False,
        )
        if not forward.accepted or forward.text is None:
            return GuardedRoundTrip(
                source_language=source,
                pivot_language=pivot,
                reasons=forward.reasons,
                warnings=forward.warnings,
                forward=forward,
            )

        pivot_positions = {entry.index: entry for entry in forward.span_map}
        back_items = tuple(
            ProtectedItem(
                index=item.index,
                kind=item.kind,
                start=pivot_positions[item.index].target_start,
                end=pivot_positions[item.index].target_end,
                label=item.label,
                surface=item.replacement,
                replacement=item.surface,
                metadata=item.metadata,
            )
            for item in protected.items
        )
        backward = self._run_leg(
            forward.text,
            back_items,
            source_language=pivot,
            target_language=source,
            assertion_axes=(),
            unverifiable_rejects=False,
        )
        warnings = _ordered((*forward.warnings, *backward.warnings))
        if not backward.accepted or backward.text is None:
            return GuardedRoundTrip(
                source_language=source,
                pivot_language=pivot,
                reasons=backward.reasons,
                warnings=warnings,
                forward=forward,
                backward=backward,
            )

        back_positions = {entry.index: entry for entry in backward.span_map}
        span_map = tuple(
            RoundTripSpanMapEntry(
                index=item.index,
                kind=item.kind,
                label=item.label,
                source_start=item.start,
                source_end=item.end,
                pivot_start=pivot_positions[item.index].target_start,
                pivot_end=pivot_positions[item.index].target_end,
                back_start=back_positions[item.index].target_start,
                back_end=back_positions[item.index].target_end,
            )
            for item in protected.items
        )
        source_regions = _regions((e.source_start, e.source_end) for e in span_map)
        back_regions = _regions((e.back_start, e.back_end) for e in span_map)
        end_reasons: list[TranslationGuardReason] = []
        end_warnings: list[TranslationGuardReason] = []
        self._check_assertions(
            text,
            source_regions,
            backward.text,
            back_regions,
            source_language=source,
            target_language=source,
            axes=self.config.assertion_axes,
            unverifiable_rejects=self.config.fail_on_unverifiable,
            reasons=end_reasons,
            warnings=end_warnings,
        )
        if self.config.verify_modifiers:
            changed = _modifier_changes(
                text, source_regions, backward.text, back_regions, source
            )
            if changed is None:
                end_warnings.append(TranslationGuardReason.MODIFIER_UNVERIFIABLE)
            else:
                end_reasons.extend(changed)

        spans_out: tuple[SpanAnnotation, ...] = ()
        if not end_reasons:
            try:
                spans_out = _project_round_trip_spans(
                    backward.text, protected.items, back_positions
                )
            except SpanProjectionError:
                end_reasons.append(TranslationGuardReason.SPAN_INTEGRITY_FAILED)

        round_trip_reasons = _ordered(end_reasons)
        all_warnings = _ordered((*warnings, *end_warnings))
        if round_trip_reasons:
            return GuardedRoundTrip(
                source_language=source,
                pivot_language=pivot,
                reasons=round_trip_reasons,
                warnings=all_warnings,
                forward=forward,
                backward=backward,
                round_trip_reasons=round_trip_reasons,
                span_map=span_map,
            )
        return GuardedRoundTrip(
            source_language=source,
            pivot_language=pivot,
            reasons=(),
            warnings=all_warnings,
            forward=forward,
            backward=backward,
            span_map=span_map,
            spans=spans_out,
            text=backward.text,
        )

    def guard_examples(
        self,
        examples: Iterable[Mapping[str, Any]],
        *,
        pivot_language: str,
        default_language: str = "en",
    ) -> TranslationGuardBatch:
        """Round-trip span examples and reject any whose facts changed.

        Each example is a mapping with ``text``, an optional ``language`` (or
        ``lang``), an optional ``id``, and spans under ``gold_spans``,
        ``labels``, ``entities``, or ``spans``. Invalid examples are rejected
        with ``invalid_input`` instead of raising.

        Args:
            examples: Span-annotated examples.
            pivot_language: Intermediate language for the round trip.
            default_language: Language assumed when an example has none.

        Returns:
            Accepted examples and an aggregate report keyed by input position.
        """

        pivot = normalize_language(pivot_language)
        accepted: list[GuardedExample] = []
        decisions: list[ExampleDecision] = []
        for index, raw_example in enumerate(examples):
            try:
                example_id, text, language, spans = _example_fields(
                    raw_example, index=index, default_language=default_language
                )
                if language == pivot:
                    decisions.append(
                        _decision(index, (TranslationGuardReason.SAME_LANGUAGE,))
                    )
                    continue
                result = self.round_trip(
                    text,
                    source_language=language,
                    pivot_language=pivot,
                    spans=spans,
                )
            except TranslationGuardError:
                decisions.append(
                    _decision(index, (TranslationGuardReason.INVALID_INPUT,))
                )
                continue

            decisions.append(
                ExampleDecision(
                    index=index,
                    accepted=result.accepted,
                    reasons=result.reasons,
                    warnings=result.warnings,
                    protected_counts=result.protected_counts,
                    span_map=result.span_map,
                )
            )
            if result.accepted and result.text is not None:
                accepted.append(
                    GuardedExample(
                        index=index,
                        example_id=example_id,
                        language=language,
                        pivot_language=pivot,
                        span_map=result.span_map,
                        text=result.text,
                        spans=result.spans,
                        pivot_text=result.forward.text or "",
                        pivot_spans=result.forward.spans,
                    )
                )
        return TranslationGuardBatch(
            accepted=tuple(accepted),
            report=_build_report(pivot, decisions),
        )

    def _run_leg(
        self,
        text: str,
        items: tuple[ProtectedItem, ...],
        *,
        source_language: str,
        target_language: str,
        assertion_axes: tuple[str, ...],
        unverifiable_rejects: bool,
    ) -> GuardedTranslation:
        codec = _placeholder_codec(self.config.placeholder_template)
        source_regions = _regions((item.start, item.end) for item in items)

        def rejected(
            reasons: Iterable[TranslationGuardReason],
            problems: Iterable[int] = (),
        ) -> GuardedTranslation:
            return GuardedTranslation(
                source_language=source_language,
                target_language=target_language,
                reasons=_ordered(reasons),
                warnings=(),
                items=items,
                problem_indices=tuple(sorted(set(problems))),
            )

        if codec.loose.search(_blank(text, source_regions)):
            return rejected((TranslationGuardReason.PLACEHOLDER_COLLISION,))
        try:
            output = self._translate_fn(
                _mask(text, items, codec), source_language, target_language
            )
        except Exception:
            # Backend failures become a reason code; the exception message may
            # quote input text, so it is deliberately discarded.
            return rejected((TranslationGuardReason.TRANSLATOR_ERROR,))
        if not isinstance(output, str):
            return rejected((TranslationGuardReason.TRANSLATOR_ERROR,))
        output = output.strip()
        if not output:
            return rejected((TranslationGuardReason.EMPTY_TRANSLATION,))

        restored, positions, restore_reasons, problems = _restore(output, items, codec)
        if restore_reasons or restored is None:
            return rejected(restore_reasons, problems)

        target_regions = _regions(positions.values())
        reasons: list[TranslationGuardReason] = []
        warnings: list[TranslationGuardReason] = []
        if codec.loose.search(_blank(restored, target_regions)):
            reasons.append(TranslationGuardReason.PLACEHOLDER_RESIDUE)
        if self.config.verify_numbers and _digit_runs(
            text, source_regions
        ) != _digit_runs(restored, target_regions):
            reasons.append(TranslationGuardReason.NUMBER_CHANGED)
        self._check_assertions(
            text,
            source_regions,
            restored,
            target_regions,
            source_language=source_language,
            target_language=target_language,
            axes=assertion_axes,
            unverifiable_rejects=unverifiable_rejects,
            reasons=reasons,
            warnings=warnings,
        )

        span_map = tuple(
            SpanMapEntry(
                index=item.index,
                kind=item.kind,
                label=item.label,
                source_start=item.start,
                source_end=item.end,
                target_start=positions[item.index][0],
                target_end=positions[item.index][1],
            )
            for item in items
        )
        spans: tuple[SpanAnnotation, ...] = ()
        if not reasons:
            try:
                spans = _project_leg_spans(restored, items, positions)
            except SpanProjectionError:
                reasons.append(TranslationGuardReason.SPAN_INTEGRITY_FAILED)
        if reasons:
            return GuardedTranslation(
                source_language=source_language,
                target_language=target_language,
                reasons=_ordered(reasons),
                warnings=_ordered(warnings),
                items=items,
                span_map=span_map,
            )
        return GuardedTranslation(
            source_language=source_language,
            target_language=target_language,
            reasons=(),
            warnings=_ordered(warnings),
            items=items,
            span_map=span_map,
            spans=spans,
            text=restored,
        )

    def _check_assertions(
        self,
        source_text: str,
        source_regions: tuple[tuple[int, int], ...],
        target_text: str,
        target_regions: tuple[tuple[int, int], ...],
        *,
        source_language: str,
        target_language: str,
        axes: tuple[str, ...],
        unverifiable_rejects: bool,
        reasons: list[TranslationGuardReason],
        warnings: list[TranslationGuardReason],
    ) -> None:
        if not axes:
            return
        before = _assertion_counts(source_text, source_language, source_regions, axes)
        after = _assertion_counts(target_text, target_language, target_regions, axes)
        if before is None or after is None:
            bucket = reasons if unverifiable_rejects else warnings
            bucket.append(TranslationGuardReason.ASSERTION_UNVERIFIABLE)
            return
        for axis in axes:
            if before[axis] != after[axis]:
                reasons.append(_AXIS_REASONS[axis])


def guard_round_trip_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    translator: LocalTranslator | TranslateFn,
    pivot_language: str,
    config: TranslationGuardConfig | None = None,
    default_language: str = "en",
) -> TranslationGuardBatch:
    """Round-trip span examples through a local translator and gate them.

    Args:
        examples: Span-annotated examples (see
            :meth:`TranslationGuard.guard_examples`).
        translator: Caller-supplied local translator or callable.
        pivot_language: Intermediate language for the round trip.
        config: Optional protection and verification policy.
        default_language: Language assumed when an example has none.

    Returns:
        Accepted examples and a counts, codes, and offsets report.
    """

    return TranslationGuard(translator, config=config).guard_examples(
        examples, pivot_language=pivot_language, default_language=default_language
    )


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlaceholderCodec:
    prefix: str
    suffix: str
    strict: re.Pattern[str]
    loose: re.Pattern[str]

    def render(self, index: int) -> str:
        return f"{self.prefix}{index}{self.suffix}"


def _validated_template(template: object) -> str:
    if not isinstance(template, str) or template.count("{index}") != 1:
        raise TranslationGuardError(
            "placeholder_template must contain '{index}' exactly once"
        )
    prefix, suffix = template.split("{index}")
    if (
        not prefix
        or not suffix
        or prefix[-1].isdigit()
        or suffix[0].isdigit()
        or any(char.isspace() for char in template)
    ):
        raise TranslationGuardError(
            "placeholder_template needs a non-digit prefix and suffix without spaces"
        )
    return template


@lru_cache(maxsize=16)
def _placeholder_codec(template: str) -> _PlaceholderCodec:
    prefix, suffix = template.split("{index}")
    strict = re.compile(re.escape(prefix) + r"([0-9]+)" + re.escape(suffix))
    core = re.sub(r"[\W_]+", "", prefix) or re.sub(r"[\W_]+", "", suffix)
    if len(core) >= 2:
        # A translator that splits or re-cases the token ("OMP 3") leaves residue.
        loose = re.compile(re.escape(core) + r"[\W_]*\d", re.IGNORECASE)
    else:
        fragments = sorted({prefix, suffix}, key=len, reverse=True)
        loose = re.compile("|".join(re.escape(fragment) for fragment in fragments))
    return _PlaceholderCodec(prefix=prefix, suffix=suffix, strict=strict, loose=loose)


def _mask(text: str, items: Sequence[ProtectedItem], codec: _PlaceholderCodec) -> str:
    parts: list[str] = []
    cursor = 0
    for item in sorted(items, key=lambda value: value.start):
        parts.append(text[cursor : item.start])
        parts.append(codec.render(item.index))
        cursor = item.end
    parts.append(text[cursor:])
    return "".join(parts)


def _restore(
    output: str,
    items: Sequence[ProtectedItem],
    codec: _PlaceholderCodec,
) -> tuple[
    str | None,
    dict[int, tuple[int, int]],
    tuple[TranslationGuardReason, ...],
    tuple[int, ...],
]:
    by_index = {item.index: item for item in items}
    matches = list(codec.strict.finditer(output))
    counts: Counter[int] = Counter()
    unknown = False
    for match in matches:
        raw_index = match.group(1)
        index = int(raw_index)
        if index not in by_index or raw_index != str(index):
            unknown = True
            continue
        counts[index] += 1
    dropped = sorted(index for index in by_index if counts[index] == 0)
    duplicated = sorted(index for index, count in counts.items() if count > 1)
    reasons: list[TranslationGuardReason] = []
    if dropped:
        reasons.append(TranslationGuardReason.PROTECTED_ITEM_DROPPED)
    if duplicated:
        reasons.append(TranslationGuardReason.PROTECTED_ITEM_DUPLICATED)
    if unknown:
        reasons.append(TranslationGuardReason.UNKNOWN_PLACEHOLDER)
    if reasons:
        return None, {}, _ordered(reasons), tuple(dropped + duplicated)

    parts: list[str] = []
    positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    length = 0
    for match in matches:
        segment = output[cursor : match.start()]
        parts.append(segment)
        length += len(segment)
        item = by_index[int(match.group(1))]
        positions[item.index] = (length, length + len(item.replacement))
        parts.append(item.replacement)
        length += len(item.replacement)
        cursor = match.end()
    parts.append(output[cursor:])
    return "".join(parts), positions, (), ()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_PHI_PLACEHOLDER_RE: Final = re.compile(
    r"<<[A-Za-z0-9_:-]{1,96}>>"
    r"|\[\[?[A-Za-z][A-Za-z0-9_-]{0,63}\]?\]"
    r"|\{\{[A-Za-z][A-Za-z0-9_-]{0,63}\}\}"
    r"|<[A-Z][A-Z0-9_-]{0,63}>"
    r"|\*{3,}"
)
_NUMBER_CORE: Final = r"\d+(?:[.,]\d+)*"
# Ranges, ratios, dates, and products such as 5-10, 120/80, 2024-03-01, 2 x 5.
_NUMBER_BODY: Final = (
    _NUMBER_CORE
    + r"(?:[ \t]?(?:/|:|-|\N{EN DASH}|x|\N{MULTIPLICATION SIGN})[ \t]?"
    + _NUMBER_CORE
    + r")*"
)
_UNIT_TOKEN_RE: Final = re.compile(
    r"[ \t\N{NO-BREAK SPACE}]?(?P<unit>%|\N{DEGREE SIGN}[ \t]?[CFcf](?!\w)"
    r"|[^\W\d_][\w/^*]*(?:[ \t][Hh]g)?)"
)
_IDENTIFIER_SEGMENT_RE: Final = re.compile(r"[./#-]+")
_DIGIT_RE: Final = re.compile(r"\d+")


@dataclass(frozen=True)
class _ScriptPatterns:
    bounded: bool
    number: re.Pattern[str]
    identifier: re.Pattern[str]
    all_caps: re.Pattern[str]
    short_token: re.Pattern[str]


_BOUNDED_PATTERNS: Final = _ScriptPatterns(
    bounded=True,
    number=re.compile(r"(?<!\w)" + _NUMBER_BODY),
    identifier=re.compile(r"(?<![\w#])[^\W_][\w.#/-]*[^\W_]"),
    all_caps=re.compile(r"(?<!\w)[A-Z]{2,8}(?!\w)"),
    short_token=re.compile(r"(?<![\w/])[A-Za-z][A-Za-z0-9/]{1,4}(?![\w/])"),
)
# Scripts written without spaces (Chinese, Japanese) put Latin codes and digits
# directly next to letters, so boundaries only consider ASCII word characters.
_UNBOUNDED_PATTERNS: Final = _ScriptPatterns(
    bounded=False,
    number=re.compile(r"(?<![\dA-Za-z_])" + _NUMBER_BODY),
    identifier=re.compile(r"(?<![A-Za-z0-9_#])[A-Za-z0-9][A-Za-z0-9.#/_-]*[A-Za-z0-9]"),
    all_caps=re.compile(r"(?<![A-Za-z0-9_])[A-Z]{2,8}(?![A-Za-z0-9_])"),
    short_token=re.compile(
        r"(?<![A-Za-z0-9_/])[A-Za-z][A-Za-z0-9/]{1,4}(?![A-Za-z0-9_/])"
    ),
)
_MAX_UNBOUNDED_UNIT_LENGTH: Final = 8


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    kind: ProtectedKind
    label: str | None
    replacement: str
    metadata: Mapping[str, Any]


def _detect_candidates(
    text: str,
    language: str,
    target_language: str,
    annotated: Sequence[SpanAnnotation],
    config: TranslationGuardConfig,
) -> list[_Candidate]:
    claimed = bytearray(len(text))
    found: list[_Candidate] = []
    glossary = config.glossary_for(language, target_language)

    def claim(
        start: int,
        end: int,
        kind: ProtectedKind,
        *,
        label: str | None = None,
        replacement: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if start >= end or any(claimed[start:end]):
            return
        claimed[start:end] = b"\x01" * (end - start)
        found.append(
            _Candidate(
                start=start,
                end=end,
                kind=kind,
                label=label,
                replacement=text[start:end] if replacement is None else replacement,
                metadata=dict(metadata or {}),
            )
        )

    for span in annotated:
        claim(
            span.start,
            span.end,
            ProtectedKind.ANNOTATED_SPAN,
            label=span.label,
            replacement=glossary.get(_term_key(span.text), span.text),
            metadata=span.metadata,
        )

    kinds = config.protected_kinds
    patterns = (
        _BOUNDED_PATTERNS if _uses_token_boundaries(language) else _UNBOUNDED_PATTERNS
    )
    if ProtectedKind.PHI_PLACEHOLDER in kinds:
        for match in _PHI_PLACEHOLDER_RE.finditer(text):
            claim(match.start(), match.end(), ProtectedKind.PHI_PLACEHOLDER)
    if ProtectedKind.CLINICAL_TERM in kinds:
        for start, end, replacement in _term_matches(
            text, patterns.bounded, glossary, config.protected_terms
        ):
            claim(start, end, ProtectedKind.CLINICAL_TERM, replacement=replacement)
    if ProtectedKind.MEASUREMENT in kinds:
        for start, end in _measurement_matches(text, language, patterns):
            claim(start, end, ProtectedKind.MEASUREMENT)
    if ProtectedKind.IDENTIFIER in kinds:
        for match in patterns.identifier.finditer(text):
            if _looks_like_identifier(match.group(0)):
                claim(match.start(), match.end(), ProtectedKind.IDENTIFIER)
    if ProtectedKind.NUMBER in kinds:
        for match in patterns.number.finditer(text):
            claim(match.start(), match.end(), ProtectedKind.NUMBER)
    if ProtectedKind.ABBREVIATION in kinds:
        for match in patterns.all_caps.finditer(text):
            claim(match.start(), match.end(), ProtectedKind.ABBREVIATION)
        compact = _compact_abbreviations(language)
        for match in patterns.short_token.finditer(text):
            key = match.group(0).casefold()
            if key in compact or _is_sig_abbreviation(key):
                claim(match.start(), match.end(), ProtectedKind.ABBREVIATION)
    return sorted(found, key=lambda candidate: candidate.start)


def _term_matches(
    text: str,
    boundaries: bool,
    glossary: Mapping[str, str],
    protected_terms: Sequence[str],
) -> list[tuple[int, int, str | None]]:
    terms: dict[str, str | None] = {_term_key(term): None for term in protected_terms}
    terms.update(glossary)
    matches: list[tuple[int, int, str | None]] = []
    for term in sorted(terms, key=lambda value: (-len(value), value)):
        for match in _term_pattern(term, boundaries).finditer(text):
            matches.append((match.start(), match.end(), terms[term]))
    return matches


@lru_cache(maxsize=512)
def _term_pattern(term: str, boundaries: bool) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in term.split())
    prefix = r"(?<!\w)" if boundaries and _is_word_char(term[0]) else ""
    suffix = r"(?!\w)" if boundaries and _is_word_char(term[-1]) else ""
    return re.compile(prefix + body + suffix, re.IGNORECASE)


def _measurement_matches(
    text: str,
    language: str,
    patterns: _ScriptPatterns,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for number in patterns.number.finditer(text):
        unit = _UNIT_TOKEN_RE.match(text, number.end())
        if unit is None:
            continue
        unit_start = unit.start("unit")
        for candidate in _unit_candidates(unit.group("unit"), patterns.bounded):
            end = unit_start + len(candidate)
            if patterns.bounded and end < len(text) and _is_word_char(text[end]):
                continue
            if _is_unit(candidate, language):
                matches.append((number.start(), end))
                break
    return matches


def _unit_candidates(token: str, bounded: bool) -> tuple[str, ...]:
    if not bounded:
        # Unspaced scripts run a unit straight into the next word, so try each
        # prefix from longest to shortest.
        longest = min(len(token), _MAX_UNBOUNDED_UNIT_LENGTH)
        return tuple(token[:size] for size in range(longest, 0, -1))
    candidates = [token]
    cut = token
    while "/" in cut:
        cut = cut.rsplit("/", 1)[0]
        if cut:
            candidates.append(cut)
    leading = re.match(r"[^\W\d_]+", token)
    if leading is not None:
        candidates.append(leading.group(0))
    return tuple(dict.fromkeys(candidates))


@lru_cache(maxsize=1024)
def _is_unit(candidate: str, language: str) -> bool:
    if canonical_unit_alias(candidate, language=language) is not None:
        return True
    return parse_unit(candidate, language=language).get("status") == "ok"


def _looks_like_identifier(token: str) -> bool:
    """Return whether a token is a Latin-alphanumeric code such as MRN-0042817.

    A token qualifies when it mixes ASCII letters and digits within one
    segment (``HbA1c``, ``q8h``, ``E11.9``) or carries an upper-case letter
    next to a digit segment (``MRN-0042817``, ``COVID-19``). Lower-case
    phrases such as ``5-year-old`` stay translatable.
    """

    if not any(char.isdigit() for char in token):
        return False
    if not any(_is_ascii_letter(char) for char in token):
        return False
    for segment in _IDENTIFIER_SEGMENT_RE.split(token):
        if any(char.isdigit() for char in segment) and any(
            _is_ascii_letter(char) for char in segment
        ):
            return True
    return any(_is_ascii_letter(char) and char.isupper() for char in token)


def _is_ascii_letter(char: str) -> bool:
    return char.isascii() and char.isalpha()


def _compact_abbreviations(language: str) -> frozenset[str]:
    lexicon = get_clinical_norm_lexicon(language)
    return frozenset(
        key
        for key in lexicon.abbreviation_expansions
        if key.isascii() and " " not in key and 2 <= len(key) <= 5
    )


@lru_cache(maxsize=1024)
def _is_sig_abbreviation(token: str) -> bool:
    if not token.isalpha() or len(token) > 4:
        return False
    frequency = normalize_frequency(token)
    return bool(frequency.get("recognized")) and frequency.get("cue") == token


def _uses_token_boundaries(language: str) -> bool:
    code = normalize_language(language)
    if code not in available_clinical_cue_languages():
        return True
    return get_clinical_cue_lexicon(code).token_boundaries


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CuePack:
    pattern: re.Pattern[str] | None
    axes_by_cue: Mapping[str, frozenset[str]]


@lru_cache(maxsize=32)
def _compiled_cue_pack(lexicon: ClinicalCueLexicon) -> _CuePack:
    axes_by_cue: dict[str, set[str]] = {}
    # Pseudo-negation phrases are matched so their inner "no"/"not" never
    # counts as a negation, but they only count on axes they also belong to.
    for axis in (*ASSERTION_AXES, "pseudo_negation"):
        for cue in getattr(lexicon, axis):
            key = _cue_key(cue)
            if key:
                axes_by_cue.setdefault(key, set()).add(axis)
    if not axes_by_cue:
        return _CuePack(pattern=None, axes_by_cue={})
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(axes_by_cue, key=lambda value: (-len(value), value))
    )
    if lexicon.token_boundaries:
        pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)
    else:
        pattern = re.compile(rf"(?:{alternation})", re.IGNORECASE)
    return _CuePack(
        pattern=pattern,
        axes_by_cue={cue: frozenset(axes) for cue, axes in axes_by_cue.items()},
    )


def _assertion_counts(
    text: str,
    language: str,
    regions: tuple[tuple[int, int], ...],
    axes: tuple[str, ...],
) -> dict[str, int] | None:
    code = normalize_language(language)
    if code not in available_clinical_cue_languages():
        return None
    pack = _compiled_cue_pack(get_clinical_cue_lexicon(code))
    counts = {axis: 0 for axis in axes}
    if pack.pattern is None:
        return counts
    for match in pack.pattern.finditer(_blank(text, regions)):
        for axis in pack.axes_by_cue.get(_cue_key(match.group(0)), ()):
            if axis in counts:
                counts[axis] += 1
    return counts


def _modifier_counts(
    text: str, regions: tuple[tuple[int, int], ...]
) -> Counter[tuple[str, str]]:
    blanked = _blank(text, regions)
    if not blanked.strip():
        return Counter()
    hits = extract_severity_laterality(
        [{"start": 0, "end": len(blanked), "label": "text"}],
        blanked,
        max_distance=len(blanked),
    )
    return Counter((str(hit["kind"]), str(hit["normalized"])) for hit in hits)


def _modifier_changes(
    source_text: str,
    source_regions: tuple[tuple[int, int], ...],
    target_text: str,
    target_regions: tuple[tuple[int, int], ...],
    language: str,
) -> tuple[TranslationGuardReason, ...] | None:
    if normalize_language(language) not in _MODIFIER_LANGUAGES:
        return None
    before = _modifier_counts(source_text, source_regions)
    after = _modifier_counts(target_text, target_regions)
    changed: list[TranslationGuardReason] = []
    for kind in MODIFIER_KINDS:
        left = {key: count for key, count in before.items() if key[0] == kind}
        right = {key: count for key, count in after.items() if key[0] == kind}
        if left != right:
            changed.append(_MODIFIER_REASONS[kind])
    return tuple(changed)


def _digit_runs(text: str, regions: tuple[tuple[int, int], ...]) -> Counter[str]:
    return Counter(
        "".join(str(unicodedata.decimal(char)) for char in match.group(0))
        for match in _DIGIT_RE.finditer(_blank(text, regions))
    )


def _project_leg_spans(
    text: str,
    items: Sequence[ProtectedItem],
    positions: Mapping[int, tuple[int, int]],
) -> tuple[SpanAnnotation, ...]:
    spans = tuple(
        SpanAnnotation(
            start=positions[item.index][0],
            end=positions[item.index][1],
            label=item.label or "",
            text=item.replacement,
            metadata={
                **dict(item.metadata),
                "projected_from_end": item.end,
                "projected_from_start": item.start,
            },
        )
        for item in items
        if item.kind is ProtectedKind.ANNOTATED_SPAN
    )
    validate_span_integrity(text, spans)
    return tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))


def _project_round_trip_spans(
    text: str,
    items: Sequence[ProtectedItem],
    back_positions: Mapping[int, SpanMapEntry],
) -> tuple[SpanAnnotation, ...]:
    spans = tuple(
        SpanAnnotation(
            start=back_positions[item.index].target_start,
            end=back_positions[item.index].target_end,
            label=item.label or "",
            text=item.surface,
            metadata={
                **dict(item.metadata),
                "projected_from_end": item.end,
                "projected_from_start": item.start,
            },
        )
        for item in items
        if item.kind is ProtectedKind.ANNOTATED_SPAN
    )
    validate_span_integrity(text, spans)
    return tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _resolve_translator(translator: object) -> TranslateFn:
    translate = getattr(translator, "translate", None)
    if callable(translate):
        return translate
    if callable(translator):
        return translator  # type: ignore[return-value]
    raise TypeError(
        "translator must be callable or expose translate(text, source, target)"
    )


def _require_text(text: object) -> None:
    if not isinstance(text, str) or not text.strip():
        raise TranslationGuardError("text must be a non-empty string")


def _normalized_input_spans(
    text: str, spans: Iterable[Any]
) -> tuple[SpanAnnotation, ...]:
    try:
        annotated = normalize_span_annotations(list(spans), source_text=text)
        validate_span_integrity(text, annotated)
    except (SpanProjectionError, TypeError, AttributeError):
        raise TranslationGuardError(
            "annotated spans must be in range, match the text, and not overlap"
        ) from None
    return annotated


def _example_fields(
    raw_example: object,
    *,
    index: int,
    default_language: str,
) -> tuple[str, str, str, tuple[SpanAnnotation, ...]]:
    if not isinstance(raw_example, Mapping):
        raise TranslationGuardError("example must be a mapping")
    text = raw_example.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TranslationGuardError("example text must be a non-empty string")
    language = normalize_language(
        raw_example.get("language") or raw_example.get("lang") or default_language
    )
    try:
        spans = normalize_span_annotations(raw_example, source_text=text)
    except (SpanProjectionError, TypeError, AttributeError):
        raise TranslationGuardError("example spans are invalid") from None
    example_id = str(
        raw_example.get("id") or raw_example.get("fixture_id") or f"example-{index}"
    )
    return example_id, text, language, spans


def _validated_term(term: object) -> str:
    if not isinstance(term, str) or not term.strip():
        raise TranslationGuardError("protected terms must be non-empty strings")
    return term.strip()


def _normalized_glossary(
    glossary: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    if not isinstance(glossary, Mapping):
        raise TranslationGuardError("glossary must be a mapping")
    normalized: dict[tuple[str, str], dict[str, str]] = {}
    for pair, entries in glossary.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(code, str) and code.strip() for code in pair)
        ):
            raise TranslationGuardError(
                "glossary keys must be (source_language, target_language) pairs"
            )
        if not isinstance(entries, Mapping):
            raise TranslationGuardError("glossary entries must be mappings")
        key = (normalize_language(pair[0]), normalize_language(pair[1]))
        bucket = normalized.setdefault(key, {})
        for term, target in entries.items():
            if not isinstance(target, str) or not target.strip():
                raise TranslationGuardError(
                    "glossary targets must be non-empty strings"
                )
            bucket[_term_key(_validated_term(term))] = target.strip()
    return normalized


def _validated_axes(axes: Iterable[str], name: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(axes))
    if any(axis not in ASSERTION_AXES for axis in values):
        raise TranslationGuardError(f"{name} axes must be drawn from {ASSERTION_AXES}")
    return values


def _term_key(term: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", term).casefold().split())


def _cue_key(cue: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", cue).casefold().split())


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _regions(pairs: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(start), int(end)) for start, end in pairs))


def _blank(text: str, regions: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in regions:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _ordered(
    reasons: Iterable[TranslationGuardReason],
) -> tuple[TranslationGuardReason, ...]:
    return tuple(sorted(set(reasons), key=_REASON_ORDER.__getitem__))


def _codes(reasons: Iterable[TranslationGuardReason]) -> list[str]:
    return [reason.value for reason in reasons]


def _kind_counts(items: Iterable[ProtectedItem]) -> dict[str, int]:
    return dict(sorted(Counter(item.kind.value for item in items).items()))


def _stage_report(result: GuardedTranslation) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "problem_indices": list(result.problem_indices),
        "reasons": _codes(result.reasons),
        "warnings": _codes(result.warnings),
    }


def _rejected_leg(
    result: GuardedTranslation,
    reasons: Iterable[TranslationGuardReason],
) -> GuardedTranslation:
    return GuardedTranslation(
        source_language=result.source_language,
        target_language=result.target_language,
        reasons=_ordered((*result.reasons, *reasons)),
        warnings=result.warnings,
        items=result.items,
        span_map=result.span_map,
        problem_indices=result.problem_indices,
    )


def _with_warnings(
    result: GuardedTranslation,
    warnings: Iterable[TranslationGuardReason],
) -> GuardedTranslation:
    return GuardedTranslation(
        source_language=result.source_language,
        target_language=result.target_language,
        reasons=result.reasons,
        warnings=_ordered((*result.warnings, *warnings)),
        items=result.items,
        span_map=result.span_map,
        problem_indices=result.problem_indices,
        spans=result.spans,
        text=result.text,
    )


def _decision(
    index: int, reasons: tuple[TranslationGuardReason, ...]
) -> ExampleDecision:
    return ExampleDecision(
        index=index,
        accepted=False,
        reasons=reasons,
        warnings=(),
        protected_counts={},
    )


def _build_report(
    pivot_language: str, decisions: Sequence[ExampleDecision]
) -> TranslationGuardReport:
    reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    protected_counts: Counter[str] = Counter()
    for decision in decisions:
        reason_counts.update(reason.value for reason in decision.reasons)
        warning_counts.update(reason.value for reason in decision.warnings)
        protected_counts.update(decision.protected_counts)
    accepted = sum(1 for decision in decisions if decision.accepted)
    return TranslationGuardReport(
        pivot_language=pivot_language,
        total=len(decisions),
        accepted=accepted,
        rejected=len(decisions) - accepted,
        reason_counts=dict(sorted(reason_counts.items())),
        warning_counts=dict(sorted(warning_counts.items())),
        protected_counts=dict(sorted(protected_counts.items())),
        decisions=tuple(decisions),
    )


__all__ = [
    "ASSERTION_AXES",
    "DEFAULT_CROSS_LINGUAL_ASSERTION_AXES",
    "DEFAULT_PLACEHOLDER_TEMPLATE",
    "DEFAULT_PROTECTED_KINDS",
    "GUARDED_SYNTHETIC_SOURCE",
    "MODIFIER_KINDS",
    "TRANSLATION_GUARD_ADVISORY",
    "TRANSLATION_GUARD_SCHEMA_VERSION",
    "ExampleDecision",
    "GuardedExample",
    "GuardedRoundTrip",
    "GuardedTranslation",
    "LocalTranslator",
    "ProtectedItem",
    "ProtectedKind",
    "ProtectedText",
    "RoundTripSpanMapEntry",
    "SpanMapEntry",
    "TranslateFn",
    "TranslationGuard",
    "TranslationGuardBatch",
    "TranslationGuardConfig",
    "TranslationGuardError",
    "TranslationGuardReason",
    "TranslationGuardReport",
    "guard_round_trip_examples",
]
