"""Post-de-identification clinical summarization.

Summarization is deliberately the last generative stage in the clinical
pipeline.  :func:`summarize` accepts a raw note, de-identifies it locally, and
then sends only the de-identified text to a local backend or explicitly selected
deterministic extraction. :func:`summarize_deidentified` exposes the
guarded stage for callers that already own a :class:`DeidentificationResult`.

The default is a cache-only MLX backend. Missing artifacts or runtime fail
closed; use ``model="extractive"`` for the deterministic CPU baseline.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from openmed.clinical.brief_cancellation import (
    BriefCancellation,
    BriefInterrupted,
    call_with_cancellation,
    check_cancellation,
)
from openmed.core.pii import DeidentificationResult, deidentify
from openmed.core.script_detect import (
    ZERO_WIDTH_CHARS,
    normalize_for_pii_detection,
    segment_by_script,
)

DEFAULT_SUMMARIZATION_MODE = "bhc"
SUMMARIZATION_ADVISORY = (
    "Clinical summarization is generative-last assistive output. It runs only "
    "after local de-identification, uses a caller-supplied local or on-device "
    "backend, and requires qualified clinical review."
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# Bounded suffixes, not morphological inference. Keep a boundary after the
# suffix so a longer Hangul word does not become a source-name match.
_HANGUL_PARTICLES = (
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "에서",
    "에게",
    "에게서",
    "께",
    "께서",
    "한테",
    "한테서",
    "와",
    "과",
    "랑",
    "이랑",
    "하고",
    "도",
    "만",
    "부터",
    "까지",
    "보다",
    "처럼",
    "으로",
    "로",
    "으로서",
    "로서",
    "으로써",
    "로써",
    "이라고",
    "라고",
    "이나",
    "나",
    "든지",
    "이든지",
    "조차",
    "마저",
    "밖에",
    "뿐",
    "님",
    "씨",
)

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

    summary: str = field(repr=False)
    leakage_check: LeakageCheck
    mode: str = DEFAULT_SUMMARIZATION_MODE
    backend: str = "deterministic-extractive"
    template_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a string")
        if not isinstance(self.leakage_check, LeakageCheck):
            raise TypeError("leakage_check must be a LeakageCheck")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a non-empty string")
        if self.template_digest is not None and (
            not isinstance(self.template_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.template_digest) is None
        ):
            raise ValueError("invalid template digest")

    def __iter__(self) -> Iterator[str | LeakageCheck]:
        """Yield the summary and leakage check for tuple-style consumers."""

        yield self.summary
        yield self.leakage_check

    @property
    def metadata(self) -> dict[str, Any]:
        """Return value-free backend and template provenance for audit logging."""
        return {"backend_id": self.backend, "template_digest": self.template_digest}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result without source PHI metadata."""

        return {
            "summary": self.summary,
            "leakage_check": self.leakage_check.to_dict(),
            "mode": self.mode,
            "backend": self.backend,
            "metadata": self.metadata,
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
        model: Registry alias (default ``mlx``), explicit ``extractive``, or
            a caller-owned local callable accepting de-identified text.

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
    from openmed.clinical.summarize_backends import (
        ExtractiveSummarizerBackend,
        LocalSummarizerError,
        MLXSummarizerBackend,
        _require_runtime,
        _validate_input,
        resolve_summarizer_backend,
    )
    from openmed.core.config import OpenMedConfig
    from openmed.core.offline import network_blocked_if_offline

    backend = resolve_summarizer_backend(model)
    if type(backend) in {ExtractiveSummarizerBackend, MLXSummarizerBackend}:
        _validate_input(text, normalized_mode)
    if isinstance(backend, MLXSummarizerBackend):
        _require_runtime()
    failed = False
    try:
        with network_blocked_if_offline(local_only=True):
            result = deidentify(
                text, method="mask", config=OpenMedConfig(local_only=True)
            )
    except Exception:
        failed = True
    if failed:
        raise LocalSummarizerError(reason="deidentification_unavailable")
    return summarize_deidentified(result, mode=normalized_mode, model=backend)


def summarize_deidentified(
    deidentified: DeidentificationResult,
    mode: str = DEFAULT_SUMMARIZATION_MODE,
    model: object | None = None,
    *,
    cancellation: BriefCancellation | None = None,
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
        cancellation: Optional cooperative brief interruption context.

    Returns:
        A summary paired with a passing leakage check.

    Raises:
        SummarizationOrderError: If the input is not a de-identification
            result or its output still contains a detected source token.
        SummarizationLeakageError: If the backend emits a source token.
    """

    check_cancellation(cancellation)
    normalized_mode = _normalize_mode(mode)
    source = _require_deidentification_result(deidentified)
    source_check = _build_leakage_check(source, source.deidentified_text)
    if not source_check.passed:
        raise SummarizationOrderError(
            "de-identification ordering guard rejected input: "
            "the de-identified text still contains a source token"
        )

    from openmed.clinical.summarize_backends import resolve_summarizer_backend

    backend = resolve_summarizer_backend(model)
    summary = _invoke_backend(
        backend, source.deidentified_text, normalized_mode, cancellation
    )
    leakage_check = _build_leakage_check(source, summary)
    check_cancellation(cancellation)
    if not leakage_check.passed:
        raise SummarizationLeakageError(leakage_check)

    result = SummarizationResult(
        summary=summary,
        leakage_check=leakage_check,
        mode=normalized_mode,
        backend=_backend_name(backend),
        template_digest=(
            backend.template_digest
            if _backend_name(backend) != "caller-supplied-local"
            else None
        ),
    )
    check_cancellation(cancellation)
    return result


def _normalize_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    normalized = mode.strip().casefold()
    if not normalized:
        raise ValueError("mode must be a non-empty string")
    return normalized


def _require_deidentification_result(value: object) -> DeidentificationResult:
    """Validate the guarded-stage input without accepting a raw string."""

    if not isinstance(value, DeidentificationResult):
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


def _invoke_backend(
    model: object | None, text: str, mode: str, cancellation=None
) -> str:
    from openmed.clinical.extractive_selection import ExtractiveSelectionError
    from openmed.clinical.summarize_backends import (
        LocalSummarizerError,
        LocalSummarizerPackageError,
        MLXSummarizerBackend,
    )
    from openmed.core.capabilities import MissingOptionalDependencyError
    from openmed.core.offline import network_blocked_if_offline

    reason = "execution_failed"
    package_code = None
    try:
        with network_blocked_if_offline(local_only=True):
            return _call_backend(model, text, mode, cancellation)
    except BriefInterrupted:
        raise
    except LocalSummarizerPackageError as error:
        if (
            type(model) is MLXSummarizerBackend
            and type(error) is LocalSummarizerPackageError
        ):
            package_code = error.code
    except MissingOptionalDependencyError:
        if type(model) is MLXSummarizerBackend:
            raise
        reason = "runtime_unavailable"
    except ExtractiveSelectionError:
        from openmed.clinical.summarize_backends import ExtractiveSummarizerBackend

        if type(model) is ExtractiveSummarizerBackend:
            raise
    except LocalSummarizerError as error:
        # Reconstruct a closed, value-free error outside the handler instead
        # of retaining a third-party exception or its chained source payload.
        if type(error) is LocalSummarizerError:
            reason = error.reason
    except Exception:
        pass
    if package_code is not None:
        raise LocalSummarizerPackageError(package_code)
    raise LocalSummarizerError(reason=reason)


def _call_backend(model: object | None, text: str, mode: str, cancellation=None) -> str:
    if model is None:
        return _extractive_summary(text)

    callback = getattr(model, "summarize", None)
    if callback is None:
        callback = model
    if not callable(callback):
        raise TypeError("model must be callable or expose a callable summarize method")

    original_callback = callback
    callback = lambda *args, **kwargs: call_with_cancellation(
        original_callback, *args, cancellation=cancellation, **kwargs
    )
    try:
        parameters = inspect.signature(original_callback).parameters
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

    from openmed.clinical.summarize_backends import (
        MAX_OUTPUT_BYTES,
        LocalSummarizerError,
        _utf8_size,
    )

    if not isinstance(output, str):
        raise LocalSummarizerError(reason="invalid_output")
    if _utf8_size(output) > MAX_OUTPUT_BYTES:
        raise LocalSummarizerError(reason="output_limit_exceeded")
    return output.strip()


def _extractive_summary(text: str) -> str:
    """Select the first three source sentences with local script-aware boundaries."""
    from openmed.processing import segment_text

    if not text.strip():
        return ""
    sentences = [text[span.start : span.end].strip() for span in segment_text(text)]
    selected = [sentence for sentence in sentences if sentence][:3]
    return " ".join(selected)


def _backend_name(model: object | None) -> str:
    from openmed.clinical.summarize_backends import (
        ExtractiveSummarizerBackend,
        MLXSummarizerBackend,
    )

    if type(model) is ExtractiveSummarizerBackend:
        return "deterministic-extractive"
    if type(model) is MLXSummarizerBackend:
        return "local-mlx"
    return "caller-supplied-local"


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
    normalized_candidate = _normalize_leakage_text(candidate)
    leaked_hashes: list[str] = []
    for surface in surfaces:
        pattern = _surface_pattern(surface)
        if pattern is not None and any(
            all(
                _hangul_suffix_is_allowed(suffix)
                for suffix in match.groupdict().values()
                if suffix is not None
            )
            for match in pattern.finditer(normalized_candidate)
        ):
            leaked_hashes.append(_surface_hash(surface))
    return LeakageCheck(
        passed=not leaked_hashes,
        checked_token_count=len(surfaces),
        leaked_token_count=len(leaked_hashes),
        leaked_token_hashes=tuple(leaked_hashes),
    )


def _surface_pattern(surface: str) -> re.Pattern[str] | None:
    alternatives = set()
    for index, raw_part in enumerate(dict.fromkeys((surface, *surface.split()))):
        normalized = _normalize_leakage_text(raw_part)
        words = normalized.split()
        if not words:
            continue
        literal = r"\s+".join(re.escape(word) for word in words)
        scripts = {script for _, _, script in segment_by_script(raw_part)}
        unspaced = bool(scripts & {"Han", "Hiragana/Katakana", "Thai"}) or any(
            0x0E80 <= ord(char) <= 0x0EFF  # Lao
            or 0x1780 <= ord(char) <= 0x17FF  # Khmer
            or 0x1000 <= ord(char) <= 0x109F  # Myanmar
            or 0xA9E0 <= ord(char) <= 0xA9FF
            or 0xAA60 <= ord(char) <= 0xAA7F
            for char in raw_part
        )
        if unspaced:
            alternatives.add(literal)
        elif "Hangul" in scripts:
            # Capture a bounded-script run, then segment known suffixes with DP.
            # Repeated overlapping regex alternatives can otherwise backtrack
            # exponentially on an adversarial longer word.
            suffix = (
                rf"(?P<hangul_suffix_{index}>"
                r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f"
                r"\uac00-\ud7af\ud7b0-\ud7ff]*)"
            )
            alternatives.add(r"(?<!\w)" + literal + suffix + r"(?!\w)")
        else:
            alternatives.add(r"(?<!\w)" + literal + r"(?!\w)")
    if not alternatives:
        return None
    return re.compile("(?:" + "|".join(sorted(alternatives)) + ")", re.IGNORECASE)


def _hangul_suffix_is_allowed(suffix: str) -> bool:
    reachable = [False] * (len(suffix) + 1)
    reachable[0] = True
    for index in range(len(suffix)):
        if reachable[index]:
            for particle in _HANGUL_PARTICLES:
                if suffix.startswith(particle, index):
                    reachable[index + len(particle)] = True
    return reachable[-1]


def _normalize_leakage_text(text: str) -> str:
    # Expose combining marks before the detector's mark-stripping defense.
    # Recompose after it, so Hangul particle matching still sees syllables.
    # Remove supported invisible controls first; normalize comparisons only.
    visible = "".join(char for char in text if char not in ZERO_WIDTH_CHARS)
    decomposed = unicodedata.normalize("NFD", visible)
    defended = normalize_for_pii_detection(decomposed).text
    folded = unicodedata.normalize("NFC", defended).casefold()
    # Preserve the prior Python IGNORECASE equivalence for the i-family in
    # the native comparison too. This is separate from the confusable map.
    return folded.replace("\u0131", "i")


def _surface_hash(surface: str) -> str:
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()
