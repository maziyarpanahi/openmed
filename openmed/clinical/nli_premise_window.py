"""Minimal de-identified premise windows for clinical NLI verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

NLI_PREMISE_WINDOW_SCHEMA_VERSION: Final[int] = 1


class PremiseWindowError(ValueError):
    """Raised when premise-window inputs violate the safe input contract."""


class PremiseWindowStatus(str, Enum):
    """Outcome of premise-window selection."""

    READY = "ready"
    REFUSED = "refused"


class PremiseWindowRefusal(str, Enum):
    """Value-free reason why NLI verification must not run."""

    INPUT_NOT_DEIDENTIFIED = "input_not_deidentified"
    MISSING_SOURCE_SPANS = "missing_source_spans"
    WINDOW_LIMIT_EXCEEDED = "window_limit_exceeded"


@dataclass(frozen=True, order=True)
class EvidenceSpan:
    """A half-open source offset required by an NLI hypothesis."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
        ):
            raise PremiseWindowError("invalid evidence span")

    @classmethod
    def from_obj(cls, value: Any) -> EvidenceSpan:
        """Coerce common offset shapes without retaining evidence text.

        Mappings may use ``start``/``end`` or
        ``source_start``/``source_end``. Objects may expose either pair of
        attributes, which permits direct use of citation-like records.
        """

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "start" in value and "end" in value:
                return cls(start=value["start"], end=value["end"])
            if "source_start" in value and "source_end" in value:
                return cls(
                    start=value["source_start"],
                    end=value["source_end"],
                )
            raise PremiseWindowError("invalid evidence span")
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and type(value[0]) is int
            and type(value[1]) is int
        ):
            return cls(start=value[0], end=value[1])

        start = getattr(value, "source_start", None)
        end = getattr(value, "source_end", None)
        if start is None and end is None:
            start = getattr(value, "start", None)
            end = getattr(value, "end", None)
        if start is None or end is None:
            raise PremiseWindowError("invalid evidence span")
        return cls(start=start, end=end)


@dataclass(frozen=True)
class PremiseWindowResult:
    """Fail-closed result for a bounded clinical NLI premise.

    ``premise`` is available only when verification is allowed. It is omitted
    from ``repr`` and :meth:`to_dict` so logs and reports carry only safe
    coordinates and decision metadata.
    """

    status: PremiseWindowStatus
    source_spans: tuple[EvidenceSpan, ...]
    window_start: int | None
    window_end: int | None
    refusal_reason: PremiseWindowRefusal | None = None
    premise: str | None = field(default=None, repr=False)
    schema_version: int = NLI_PREMISE_WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != NLI_PREMISE_WINDOW_SCHEMA_VERSION
        ):
            raise PremiseWindowError("unsupported premise-window schema")
        if not isinstance(self.status, PremiseWindowStatus):
            raise PremiseWindowError("invalid premise-window status")
        if not isinstance(self.source_spans, tuple) or any(
            not isinstance(span, EvidenceSpan) for span in self.source_spans
        ):
            raise PremiseWindowError("invalid evidence span collection")
        if self.source_spans != tuple(sorted(set(self.source_spans))):
            raise PremiseWindowError("noncanonical evidence span collection")

        if self.status is PremiseWindowStatus.READY:
            if (
                not self.source_spans
                or type(self.premise) is not str
                or not self.premise
                or type(self.window_start) is not int
                or type(self.window_end) is not int
                or self.window_start < 0
                or self.window_end <= self.window_start
                or self.refusal_reason is not None
                or len(self.premise) != self.window_end - self.window_start
                or self.window_start != min(span.start for span in self.source_spans)
                or self.window_end != max(span.end for span in self.source_spans)
            ):
                raise PremiseWindowError("invalid ready premise window")
        else:
            if self.premise is not None or not isinstance(
                self.refusal_reason,
                PremiseWindowRefusal,
            ):
                raise PremiseWindowError("invalid refused premise window")
            has_complete_window = (
                type(self.window_start) is int
                and type(self.window_end) is int
                and self.window_start >= 0
                and self.window_end > self.window_start
            )
            if self.refusal_reason is PremiseWindowRefusal.WINDOW_LIMIT_EXCEEDED:
                if (
                    not self.source_spans
                    or not has_complete_window
                    or self.window_start
                    != min(span.start for span in self.source_spans)
                    or self.window_end != max(span.end for span in self.source_spans)
                ):
                    raise PremiseWindowError("invalid refused premise window")
            elif self.window_start is not None or self.window_end is not None:
                raise PremiseWindowError("invalid refused premise window")
            if (
                self.refusal_reason is PremiseWindowRefusal.MISSING_SOURCE_SPANS
                and self.source_spans
            ):
                raise PremiseWindowError("invalid refused premise window")

    @property
    def verification_allowed(self) -> bool:
        """Return whether the bounded premise may be sent to an NLI verifier."""

        return self.status is PremiseWindowStatus.READY

    @property
    def character_count(self) -> int:
        """Return the selected or required window length without its text."""

        if self.window_start is None or self.window_end is None:
            return 0
        return self.window_end - self.window_start

    def to_dict(self) -> dict[str, object]:
        """Return a metadata-only report that never contains premise text."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "verification_allowed": self.verification_allowed,
            "refusal_reason": (
                self.refusal_reason.value if self.refusal_reason is not None else None
            ),
            "window": (
                {"start": self.window_start, "end": self.window_end}
                if self.window_start is not None and self.window_end is not None
                else None
            ),
            "character_count": self.character_count,
            "source_span_count": len(self.source_spans),
        }


def select_minimal_premise_window(
    text: str,
    source_spans: Iterable[object],
    *,
    max_characters: int,
    deidentified: bool,
) -> PremiseWindowResult:
    """Select the smallest allowed window covering every cited source span.

    Args:
        text: Source document addressed by the half-open offsets.
        source_spans: Required citation-like offsets.
        max_characters: Inclusive ceiling for the selected window.
        deidentified: Explicit upstream assertion that ``text`` passed the
            de-identification boundary.

    Returns:
        A ready result containing the exact minimal premise, or a metadata-only
        refusal that prevents verification.

    Raises:
        PremiseWindowError: If types, limits, spans, or offsets are invalid.
            Error messages never include submitted text or values.
    """

    if type(text) is not str:
        raise PremiseWindowError("invalid premise text")
    if type(max_characters) is not int or max_characters < 1:
        raise PremiseWindowError("invalid premise-window ceiling")
    if type(deidentified) is not bool:
        raise PremiseWindowError("invalid de-identification marker")
    spans = _coerce_spans(source_spans)
    if any(span.end > len(text) for span in spans):
        raise PremiseWindowError("evidence span outside premise text")

    if not deidentified:
        return PremiseWindowResult(
            status=PremiseWindowStatus.REFUSED,
            source_spans=spans,
            window_start=None,
            window_end=None,
            refusal_reason=PremiseWindowRefusal.INPUT_NOT_DEIDENTIFIED,
        )
    if not spans:
        return PremiseWindowResult(
            status=PremiseWindowStatus.REFUSED,
            source_spans=(),
            window_start=None,
            window_end=None,
            refusal_reason=PremiseWindowRefusal.MISSING_SOURCE_SPANS,
        )

    window_start = min(span.start for span in spans)
    window_end = max(span.end for span in spans)
    if window_end - window_start > max_characters:
        return PremiseWindowResult(
            status=PremiseWindowStatus.REFUSED,
            source_spans=spans,
            window_start=window_start,
            window_end=window_end,
            refusal_reason=PremiseWindowRefusal.WINDOW_LIMIT_EXCEEDED,
        )

    return PremiseWindowResult(
        status=PremiseWindowStatus.READY,
        source_spans=spans,
        window_start=window_start,
        window_end=window_end,
        premise=text[window_start:window_end],
    )


def select_premise_window(
    text: str,
    source_spans: Iterable[object],
    *,
    max_characters: int,
    deidentified: bool,
) -> PremiseWindowResult:
    """Alias for :func:`select_minimal_premise_window`."""

    return select_minimal_premise_window(
        text,
        source_spans,
        max_characters=max_characters,
        deidentified=deidentified,
    )


def _coerce_spans(source_spans: Iterable[object]) -> tuple[EvidenceSpan, ...]:
    if isinstance(source_spans, (str, bytes, bytearray)):
        raise PremiseWindowError("invalid evidence span collection")
    try:
        spans = tuple(EvidenceSpan.from_obj(span) for span in source_spans)
    except PremiseWindowError:
        raise
    except Exception:
        raise PremiseWindowError("invalid evidence span collection") from None
    return tuple(sorted(set(spans)))


__all__ = [
    "NLI_PREMISE_WINDOW_SCHEMA_VERSION",
    "EvidenceSpan",
    "PremiseWindowError",
    "PremiseWindowRefusal",
    "PremiseWindowResult",
    "PremiseWindowStatus",
    "select_minimal_premise_window",
    "select_premise_window",
]
