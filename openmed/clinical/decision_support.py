"""Decision-support guardrail layer for clinical suggestions (OM-802).

Any clinical-suggestion or decision-support output produced by OpenMed must be
wrapped so it is transparent and clinician-reviewable, keeping OpenMed within
FDA Clinical Decision Support (CDS) transparency expectations. Concretely, every
guarded suggestion carries:

- a mandatory medical-device disclaimer (assist, do not decide; a clinician must
  independently review the basis before acting),
- source-span traceability -- the input character span(s) each suggestion
  derives from, so the clinician can independently review the basis rather than
  relying on the software's conclusion,
- a confidence value in ``[0, 1]``,
- an autonomous-decision flag that is structurally always ``False`` -- the
  software never makes an autonomous clinical decision.

The layer provides a typed :class:`GuardedSuggestion` envelope, a
:func:`guarded_suggestion` decorator/wrapper that routes any suggestion-producing
callable through the envelope, and a :func:`validate_guarded_suggestion` check
that rejects any output lacking a disclaimer or source traceability.

This module performs no clinical inference of its own; it wraps and validates the
output of upstream producers. It is local-first and holds no plaintext
identifiers -- only caller-supplied span text (synthetic in examples and tests),
offsets, confidences, and provenance labels.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

CLINICAL_DECISION_SUPPORT_SCHEMA_VERSION = 1

#: Mandatory medical-device disclaimer attached to every guarded suggestion.
#:
#: The wording asserts the two non-negotiable CDS transparency properties: the
#: output only assists (it is not an autonomous clinical decision) and the
#: clinician must independently review the cited source basis before acting.
CLINICAL_DECISION_SUPPORT_DISCLAIMER = (
    "Assistive decision support for clinician review only. This output is not a "
    "diagnosis, treatment decision, or autonomous clinical decision. A qualified "
    "clinician must independently review the cited source basis and confirm "
    "before any clinical action."
)

#: Human-readable flag text for the mandatory clinician-review requirement.
CLINICIAN_REVIEW_REQUIRED_NOTE = "Clinician must review -- not an autonomous decision."

_ConfidenceError = "confidence must be a real number in the closed interval [0, 1]"

T = TypeVar("T")


@dataclass(frozen=True)
class SourceSpan:
    """One input character span a clinical suggestion is traceable to.

    The span is expressed as character offsets into the source document plus an
    optional label and short excerpt so a clinician can locate and independently
    review the basis. Excerpts are expected to be synthetic in tests/examples;
    the layer never fabricates plaintext identifiers of its own.

    Attributes:
        start: Inclusive character offset where the source basis begins.
        end: Exclusive character offset where the source basis ends.
        label: Optional provenance label for the span (e.g. ``"problem"``).
        text: Optional short excerpt of the source span for reviewer context.
    """

    start: int
    end: int
    label: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("SourceSpan.start must be an integer offset")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("SourceSpan.end must be an integer offset")
        if self.start < 0 or self.end < 0:
            raise ValueError("SourceSpan offsets must be non-negative")
        if self.end < self.start:
            raise ValueError("SourceSpan.end must not precede SourceSpan.start")
        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))
        if self.text is not None:
            object.__setattr__(self, "text", str(self.text))

    def offset_key(self) -> tuple[int, int]:
        """Return the character-offset identity for this span."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible span representation."""

        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": self.text,
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "SourceSpan":
        """Build a :class:`SourceSpan` from a mapping or span-like object.

        Accepts mappings with ``start``/``end`` keys or any object exposing
        ``start`` and ``end`` attributes (for example an ``EntitySpan`` or a
        :class:`~openmed.clinical.relations.SpanReference`). ``label`` and
        ``text`` are copied when present.
        """

        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            start = obj.get("start")
            end = obj.get("end")
            label = obj.get("label")
            text = obj.get("text")
        else:
            start = getattr(obj, "start", None)
            end = getattr(obj, "end", None)
            label = getattr(obj, "label", None)
            text = getattr(obj, "text", None)
        if start is None or end is None:
            raise ValueError("source span requires integer start and end offsets")
        return cls(
            start=int(start),
            end=int(end),
            label=None if label is None else str(label),
            text=None if text is None else str(text),
        )


class GuardrailValidationError(ValueError):
    """Raised when a clinical suggestion fails a guardrail transparency check."""


@dataclass(frozen=True)
class GuardedSuggestion:
    """A single clinical suggestion wrapped with mandatory CDS safety metadata.

    Constructing an instance enforces the guardrail invariants, so a
    ``GuardedSuggestion`` cannot exist without a disclaimer, at least one source
    span, an in-range confidence, and an autonomous-decision flag of ``False``.

    Attributes:
        suggestion: The clinician-facing suggestion payload (opaque to this
            layer; typically a short string or JSON-serializable mapping).
        source_spans: One or more input spans the suggestion is traceable to.
        confidence: Model/heuristic confidence in ``[0, 1]``.
        disclaimer: Mandatory medical-device disclaimer.
        requires_clinician_review: Always ``True`` -- a clinician must review.
        autonomous_decision: Always ``False`` -- never an autonomous decision.
        provenance: Optional PHI-free provenance metadata (producer name,
            model id, rule id, ...).
        schema_version: Envelope schema version.
    """

    suggestion: Any
    source_spans: tuple[SourceSpan, ...]
    confidence: float
    disclaimer: str = CLINICAL_DECISION_SUPPORT_DISCLAIMER
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CLINICAL_DECISION_SUPPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized_spans = _normalize_source_spans(self.source_spans)
        if not normalized_spans:
            raise GuardrailValidationError(
                "a guarded clinical suggestion requires at least one source span "
                "for traceability; untraced suggestions are rejected"
            )
        confidence = _validate_confidence(self.confidence)
        disclaimer = str(self.disclaimer).strip()
        if not disclaimer:
            raise GuardrailValidationError(
                "a guarded clinical suggestion requires a non-empty disclaimer"
            )
        if self.autonomous_decision is not False:
            raise GuardrailValidationError(
                "clinical suggestions are never autonomous decisions; "
                "autonomous_decision must be False"
            )
        if self.requires_clinician_review is not True:
            raise GuardrailValidationError(
                "clinical suggestions always require clinician review; "
                "requires_clinician_review must be True"
            )

        object.__setattr__(self, "source_spans", normalized_spans)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "disclaimer", disclaimer)
        object.__setattr__(self, "provenance", _plain_mapping(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible envelope representation."""

        return {
            "schema_version": self.schema_version,
            "suggestion": copy.deepcopy(self.suggestion),
            "confidence": self.confidence,
            "disclaimer": self.disclaimer,
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "source_spans": [span.to_dict() for span in self.source_spans],
            "provenance": copy.deepcopy(dict(self.provenance)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GuardedSuggestion":
        """Rebuild a :class:`GuardedSuggestion` from :meth:`to_dict` output.

        The same guardrail invariants are re-enforced on load, so a tampered or
        malformed payload (missing disclaimer/traceability, autonomous flag set)
        is rejected rather than silently trusted.
        """

        if not isinstance(data, Mapping):
            raise GuardrailValidationError(
                "guarded suggestion payload must be a mapping"
            )
        return cls(
            suggestion=copy.deepcopy(data.get("suggestion")),
            source_spans=tuple(
                SourceSpan.from_obj(span) for span in data.get("source_spans", ())
            ),
            confidence=_coerce_float(data.get("confidence")),
            disclaimer=str(data.get("disclaimer", "")),
            requires_clinician_review=bool(data.get("requires_clinician_review", True)),
            autonomous_decision=bool(data.get("autonomous_decision", False)),
            provenance=dict(data.get("provenance") or {}),
            schema_version=int(
                data.get("schema_version", CLINICAL_DECISION_SUPPORT_SCHEMA_VERSION)
            ),
        )


def build_guarded_suggestion(
    suggestion: Any,
    source_spans: Iterable[Any],
    confidence: float,
    *,
    disclaimer: str = CLINICAL_DECISION_SUPPORT_DISCLAIMER,
    provenance: Mapping[str, Any] | None = None,
) -> GuardedSuggestion:
    """Build and validate a :class:`GuardedSuggestion`.

    This is the single supported constructor for external producers. It coerces
    span-like inputs (mappings, ``EntitySpan``/``SpanReference`` objects) into
    :class:`SourceSpan` values and enforces every guardrail invariant.

    Args:
        suggestion: The clinician-facing suggestion payload.
        source_spans: One or more spans (mappings or span-like objects) the
            suggestion derives from. At least one is required.
        confidence: Confidence in the closed interval ``[0, 1]``.
        disclaimer: Optional override for the mandatory disclaimer text. Must be
            non-empty; defaults to :data:`CLINICAL_DECISION_SUPPORT_DISCLAIMER`.
        provenance: Optional PHI-free provenance metadata.

    Returns:
        A validated :class:`GuardedSuggestion`.

    Raises:
        GuardrailValidationError: If traceability, disclaimer, or confidence
            invariants are violated.
    """

    return GuardedSuggestion(
        suggestion=suggestion,
        source_spans=tuple(SourceSpan.from_obj(span) for span in source_spans),
        confidence=confidence,
        disclaimer=disclaimer,
        provenance=dict(provenance or {}),
    )


def validate_guarded_suggestion(candidate: Any) -> GuardedSuggestion:
    """Validate that ``candidate`` is a compliant guarded suggestion.

    Accepts a :class:`GuardedSuggestion` or a mapping produced by
    :meth:`GuardedSuggestion.to_dict`. Any output lacking a disclaimer or source
    traceability, carrying an out-of-range confidence, or flagged as an
    autonomous decision fails the check.

    Args:
        candidate: The suggestion envelope to validate.

    Returns:
        The validated :class:`GuardedSuggestion`.

    Raises:
        GuardrailValidationError: If the candidate is not a compliant guarded
            clinical suggestion.
    """

    if isinstance(candidate, GuardedSuggestion):
        suggestion = candidate
    elif isinstance(candidate, Mapping):
        suggestion = GuardedSuggestion.from_dict(candidate)
    else:
        raise GuardrailValidationError(
            "guarded clinical suggestion must be a GuardedSuggestion or its "
            "serialized mapping"
        )

    # Invariants are enforced at construction, but re-check defensively so this
    # function is a hard gate even if the dataclass frozen guarantees are
    # bypassed by object.__setattr__ elsewhere.
    if not str(suggestion.disclaimer).strip():
        raise GuardrailValidationError("guarded suggestion is missing its disclaimer")
    if not suggestion.source_spans:
        raise GuardrailValidationError(
            "guarded suggestion is missing source-span traceability"
        )
    if suggestion.autonomous_decision is not False:
        raise GuardrailValidationError(
            "guarded suggestion must not be an autonomous decision"
        )
    if suggestion.requires_clinician_review is not True:
        raise GuardrailValidationError(
            "guarded suggestion must require clinician review"
        )
    _validate_confidence(suggestion.confidence)
    return suggestion


def guarded_suggestion(
    func: Callable[..., Any] | None = None,
    *,
    disclaimer: str = CLINICAL_DECISION_SUPPORT_DISCLAIMER,
) -> Callable[..., Any]:
    """Decorator/wrapper routing a suggestion producer through the guardrail.

    The wrapped callable may return either a fully built
    :class:`GuardedSuggestion` (or its serialized mapping) or a
    ``(suggestion, source_spans, confidence)`` tuple, optionally with a fourth
    provenance mapping. The wrapper coerces the return value into a validated
    :class:`GuardedSuggestion`, guaranteeing every emitted output carries a
    disclaimer and source traceability.

    Usage::

        @guarded_suggestion
        def suggest(...):
            return "increase monitoring frequency", [span], 0.82

    Args:
        func: The suggestion-producing callable (when used without parentheses).
        disclaimer: Optional disclaimer override applied to tuple-style returns.

    Returns:
        A wrapper that returns a validated :class:`GuardedSuggestion`, or a
        parametrized decorator when ``func`` is omitted.
    """

    def decorate(inner: Callable[..., Any]) -> Callable[..., GuardedSuggestion]:
        @wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> GuardedSuggestion:
            result = inner(*args, **kwargs)
            return _coerce_result(result, disclaimer=disclaimer)

        return wrapper

    if func is not None:
        return decorate(func)
    return decorate


def _coerce_result(
    result: Any,
    *,
    disclaimer: str,
) -> GuardedSuggestion:
    if isinstance(result, GuardedSuggestion):
        return validate_guarded_suggestion(result)
    if isinstance(result, Mapping):
        return validate_guarded_suggestion(result)
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        parts = tuple(result)
        if len(parts) not in (3, 4):
            raise GuardrailValidationError(
                "a guarded suggestion producer must return a GuardedSuggestion, a "
                "serialized mapping, or a (suggestion, source_spans, confidence"
                "[, provenance]) tuple"
            )
        suggestion, source_spans, confidence = parts[0], parts[1], parts[2]
        provenance = parts[3] if len(parts) == 4 else None
        return build_guarded_suggestion(
            suggestion,
            _as_span_iterable(source_spans),
            _coerce_float(confidence),
            disclaimer=disclaimer,
            provenance=provenance,
        )
    raise GuardrailValidationError(
        "a guarded suggestion producer must return a GuardedSuggestion, a "
        "serialized mapping, or a (suggestion, source_spans, confidence"
        "[, provenance]) tuple"
    )


def _as_span_iterable(source_spans: Any) -> Iterable[Any]:
    if isinstance(source_spans, (Mapping, str, bytes)):
        return (source_spans,)
    if isinstance(source_spans, SourceSpan):
        return (source_spans,)
    if isinstance(source_spans, Iterable):
        return source_spans
    return (source_spans,)


def _normalize_source_spans(spans: Any) -> tuple[SourceSpan, ...]:
    if spans is None:
        return ()
    if isinstance(spans, (SourceSpan, Mapping, str, bytes)):
        candidates: Iterable[Any] = (spans,)
    elif isinstance(spans, Iterable):
        candidates = spans
    else:
        candidates = (spans,)
    return tuple(SourceSpan.from_obj(span) for span in candidates)


def _validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailValidationError(_ConfidenceError)
    confidence = float(value)
    if confidence != confidence:  # NaN guard
        raise GuardrailValidationError(_ConfidenceError)
    if not 0.0 <= confidence <= 1.0:
        raise GuardrailValidationError(_ConfidenceError)
    return confidence


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailValidationError(_ConfidenceError)
    return float(value)


def _plain_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise TypeError("provenance must be a mapping")
    return {str(key): copy.deepcopy(value) for key, value in mapping.items()}


__all__ = [
    "CLINICAL_DECISION_SUPPORT_DISCLAIMER",
    "CLINICAL_DECISION_SUPPORT_SCHEMA_VERSION",
    "CLINICIAN_REVIEW_REQUIRED_NOTE",
    "GuardedSuggestion",
    "GuardrailValidationError",
    "SourceSpan",
    "build_guarded_suggestion",
    "guarded_suggestion",
    "validate_guarded_suggestion",
]
