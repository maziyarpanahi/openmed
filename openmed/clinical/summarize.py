"""Post-de-identification clinical summarization.

Summarization is deliberately the last generative stage in the clinical
pipeline.  :func:`summarize` accepts a raw note, de-identifies it locally, and
then sends only the de-identified text to a caller-supplied backend or the
deterministic extractive fallback.  :func:`summarize_deidentified` exposes the
guarded stage for callers that already own a :class:`DeidentificationResult`.

The default backend is intentionally small and deterministic.  A trained
on-device SLM, including an MLX implementation, can be supplied through the
``model`` argument without changing the privacy boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from openmed.core.pii import DeidentificationResult, deidentify

DEFAULT_SUMMARIZATION_MODE = "bhc"
SUMMARIZATION_ADVISORY = (
    "Clinical summarization is generative-last assistive output. It runs only "
    "after local de-identification, uses a caller-supplied local or on-device "
    "backend, and requires qualified clinical review."
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")

__all__ = [
    "DEFAULT_SUMMARIZATION_MODE",
    "SUMMARIZATION_ADVISORY",
    "LeakageCheck",
    "SummarizationLeakageError",
    "SummarizationOrderError",
    "SummarizationResult",
    "SummarizerBackend",
    "summarize",
    "summarize_deidentified",
]


class SummarizerBackend(Protocol):
    """Protocol for a local summarizer supplied to :func:`summarize`.

    Implementations may expose ``summarize(text, *, mode=...)`` or be a
    callable accepting the de-identified text.  The backend never receives the
    original note.
    """

    def __call__(self, text: str, *, mode: str) -> str:
        """Return a summary of de-identified ``text``."""


class SummarizationOrderError(ValueError):
    """Raised when the guarded summarization stage receives raw text."""


@dataclass(frozen=True)
class LeakageCheck:
    """PHI-free result of the source-token leakage check.

    Plaintext source identifiers are never retained in the check.  If a
    backend emits one, the result records only its count and SHA-256 digest so
    that diagnostics remain safe to serialize.
    """

    passed: bool
    checked_token_count: int
    leaked_token_count: int
    leaked_token_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        for field_name in ("checked_token_count", "leaked_token_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        hashes = tuple(str(value) for value in self.leaked_token_hashes)
        if self.leaked_token_count != len(hashes):
            raise ValueError(
                "leaked_token_count must match the number of leaked token hashes"
            )
        if self.passed != (self.leaked_token_count == 0):
            raise ValueError("passed must be false when leaked tokens are present")
        object.__setattr__(self, "leaked_token_hashes", hashes)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic PHI-free representation of the check."""

        return {
            "passed": self.passed,
            "checked_token_count": self.checked_token_count,
            "leaked_token_count": self.leaked_token_count,
            "leaked_token_hashes": list(self.leaked_token_hashes),
        }


class SummarizationLeakageError(ValueError):
    """Raised when a backend returns a summary containing source PHI."""

    def __init__(self, check: LeakageCheck) -> None:
        self.check = check
        super().__init__(
            "summary leakage guard rejected backend output: "
            f"{check.leaked_token_count} source token(s) detected"
        )


@dataclass(frozen=True)
class SummarizationResult:
    """Summary output with its mandatory leakage-check result.

    The class is iterable so callers may use either ``result.summary`` and
    ``result.leakage_check`` or unpack ``summary, leakage_check``.
    """

    summary: str
    leakage_check: LeakageCheck
    mode: str = DEFAULT_SUMMARIZATION_MODE
    backend: str = "deterministic-extractive"

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a string")
        if not isinstance(self.leakage_check, LeakageCheck):
            raise TypeError("leakage_check must be a LeakageCheck")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a non-empty string")

    def __iter__(self) -> Iterator[str | LeakageCheck]:
        """Yield the summary and leakage check for tuple-style consumers."""

        yield self.summary
        yield self.leakage_check

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result without source PHI metadata."""

        return {
            "summary": self.summary,
            "leakage_check": self.leakage_check.to_dict(),
            "mode": self.mode,
            "backend": self.backend,
        }


def summarize(
    text: str | DeidentificationResult,
    mode: str = DEFAULT_SUMMARIZATION_MODE,
    model: object | None = None,
) -> SummarizationResult:
    """De-identify and summarize a clinical note.

    Raw strings are always sent through :func:`openmed.core.pii.deidentify`
    before the summarizer backend runs.  Passing an existing
    :class:`DeidentificationResult` is supported for pipeline composition and
    uses the same guarded stage without de-identifying twice.

    Args:
        text: Raw clinical text, or a result already returned by
            :func:`openmed.core.pii.deidentify`.
        mode: Summarization mode forwarded to compatible backends. ``"bhc"``
            denotes a brief hospital-course style summary.
        model: Optional local/on-device backend. It may be callable with the
            de-identified text or expose ``summarize(text, *, mode=...)``.

    Returns:
        A summary and a passing :class:`LeakageCheck`.

    Raises:
        SummarizationLeakageError: If the backend re-emits a source PII span.
        SummarizationOrderError: If a supplied de-identification result still
            contains a detected source PII span in its output.

    The trained SLM backend is intentionally separate from this pipeline
    wiring. No cloud model is selected or contacted by this function.
    """

    normalized_mode = _normalize_mode(mode)
    if isinstance(text, DeidentificationResult):
        return summarize_deidentified(text, mode=normalized_mode, model=model)
    if not isinstance(text, str):
        raise TypeError("text must be a string or DeidentificationResult")
    result = deidentify(text, method="mask")
    return summarize_deidentified(result, mode=normalized_mode, model=model)


def summarize_deidentified(
    deidentified: DeidentificationResult,
    mode: str = DEFAULT_SUMMARIZATION_MODE,
    model: object | None = None,
) -> SummarizationResult:
    """Run the guarded summarization stage on a de-identification result.

    This function is the explicit ordering boundary. A plain string is
    rejected so callers cannot bypass de-identification accidentally. Only
    ``deidentified.deidentified_text`` is passed to the backend; source text
    and source spans remain private to the leakage check.

    Args:
        deidentified: Result produced by the de-identification stage.
        mode: Summarization mode forwarded to compatible backends.
        model: Optional local/on-device summarizer backend.

    Returns:
        A summary paired with a passing leakage check.

    Raises:
        SummarizationOrderError: If the input is not a de-identification
            result or its output still contains a detected source token.
        SummarizationLeakageError: If the backend emits a source token.
    """

    normalized_mode = _normalize_mode(mode)
    source = _require_deidentification_result(deidentified)
    source_check = _build_leakage_check(source, source.deidentified_text)
    if not source_check.passed:
        raise SummarizationOrderError(
            "de-identification ordering guard rejected input: "
            "the de-identified text still contains a source token"
        )

    summary = _invoke_backend(model, source.deidentified_text, normalized_mode)
    leakage_check = _build_leakage_check(source, summary)
    if not leakage_check.passed:
        raise SummarizationLeakageError(leakage_check)

    return SummarizationResult(
        summary=summary,
        leakage_check=leakage_check,
        mode=normalized_mode,
        backend=_backend_name(model),
    )


def _normalize_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    normalized = mode.strip().casefold()
    if not normalized:
        raise ValueError("mode must be a non-empty string")
    return normalized


def _require_deidentification_result(value: object) -> Any:
    """Validate the guarded-stage input without accepting a raw string."""

    if isinstance(value, str) or not all(
        hasattr(value, attribute)
        for attribute in ("original_text", "deidentified_text", "pii_entities")
    ):
        raise SummarizationOrderError(
            "summarization requires a de-identification result; "
            "call summarize() with raw text or deidentify() first"
        )
    if not isinstance(value.deidentified_text, str) or not isinstance(
        value.original_text, str
    ):
        raise SummarizationOrderError(
            "de-identification result must contain string source and output text"
        )
    if value.pii_entities is None:
        raise SummarizationOrderError(
            "de-identification result must expose detected source entities"
        )
    return value


def _invoke_backend(model: object | None, text: str, mode: str) -> str:
    if model is None:
        return _extractive_summary(text)

    callback = getattr(model, "summarize", None)
    if callback is None:
        callback = model
    if not callable(callback):
        raise TypeError("model must be callable or expose a callable summarize method")

    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        output = callback(text, mode=mode)
    else:
        mode_parameter = parameters.get("mode")
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if mode_parameter is not None:
            if mode_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                output = callback(text, mode)
            else:
                output = callback(text, mode=mode)
        elif accepts_keywords:
            output = callback(text, mode=mode)
        else:
            output = callback(text)

    if not isinstance(output, str):
        raise TypeError("summarizer backend must return a string")
    return output.strip()


def _extractive_summary(text: str) -> str:
    """Return a deterministic, local, first-three-sentence stub summary."""

    normalized = " ".join(text.split())
    if not normalized:
        return ""
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(normalized)]
    return " ".join(sentences[:3])


def _backend_name(model: object | None) -> str:
    if model is None:
        return "deterministic-extractive"
    if hasattr(model, "summarize"):
        return f"{type(model).__name__}.summarize"
    return type(model).__name__


def _source_phi_surfaces(deidentified: Any) -> tuple[str, ...]:
    surfaces: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str):
            return
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            surfaces.append(normalized)

    for entity in deidentified.pii_entities:
        add(getattr(entity, "original_text", None))
        add(getattr(entity, "text", None))
        if not getattr(entity, "original_text", None) and not getattr(
            entity, "text", None
        ):
            start = getattr(entity, "start", None)
            end = getattr(entity, "end", None)
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(deidentified.original_text)
            ):
                add(deidentified.original_text[start:end])

    mapping = getattr(deidentified, "mapping", None)
    if isinstance(mapping, Mapping):
        for value in mapping.values():
            add(value)
    return tuple(surfaces)


def _build_leakage_check(deidentified: Any, candidate: str) -> LeakageCheck:
    surfaces = _source_phi_surfaces(deidentified)
    leaked_hashes: list[str] = []
    for surface in surfaces:
        pattern = _surface_pattern(surface)
        if pattern is not None and pattern.search(candidate):
            leaked_hashes.append(_surface_hash(surface))
    return LeakageCheck(
        passed=not leaked_hashes,
        checked_token_count=len(surfaces),
        leaked_token_count=len(leaked_hashes),
        leaked_token_hashes=tuple(leaked_hashes),
    )


def _surface_pattern(surface: str) -> re.Pattern[str] | None:
    parts = surface.split()
    if not parts:
        return None
    return re.compile(
        r"(?<!\w)" + r"\s+".join(re.escape(part) for part in parts) + r"(?!\w)",
        re.IGNORECASE,
    )


def _surface_hash(surface: str) -> str:
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()
