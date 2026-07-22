"""Shared input-normalization and validation gateway.

OpenMed exposes the same clinical NLP and PII capabilities through three
surfaces: the Python library, the REST service (``openmed/service/``), and the
MCP server (``openmed/mcp/``). Historically each surface validated request text
and parameters with slightly different ad-hoc rules, so an input accepted by one
entry point could be rejected (or silently mishandled) by another.

This module centralises that logic. All three surfaces call the same functions
here, so length/size limits, encoding validation, and language guardrails behave
identically everywhere, and every failure raises the same typed error.

Privacy note:
    Validation errors NEVER embed the raw input text (or any excerpt of it).
    Inputs may contain PHI, and error messages can end up in logs, HTTP
    responses, or MCP transcripts. Messages describe *what* was wrong (a limit,
    a type, an encoding fault) using only non-sensitive metadata such as sizes
    and configured caps.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

__all__ = [
    "InputValidationError",
    "GatewayLimits",
    "NormalizedInput",
    "DEFAULT_MAX_TEXT_CHARS",
    "DEFAULT_MAX_TEXT_BYTES",
    "MAX_TEXT_BYTES_ENV_VAR",
    "get_default_limits",
    "ensure_valid_encoding",
    "normalize_text",
    "validate_language",
    "normalize_input",
]

# Character cap. Mirrors ``openmed.service.limits.DEFAULT_MAX_TEXT_LENGTH`` so
# the library and MCP surfaces default to the same bound the REST service uses.
DEFAULT_MAX_TEXT_CHARS = 1_000_000

# Byte cap on the UTF-8 encoding of the text. Guards against payloads that are
# modest in character count but huge on the wire (e.g. multi-byte scripts).
DEFAULT_MAX_TEXT_BYTES = 4_000_000

MAX_TEXT_BYTES_ENV_VAR = "OPENMED_MAX_TEXT_BYTES"

# Control characters that must never appear in text input. Common whitespace
# (tab, newline, carriage return) is intentionally excluded so legitimate
# multi-line clinical notes pass. This mirrors the historical suspicious-content
# check in ``openmed.utils.validation`` but is applied uniformly across surfaces.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Long control-character runs indicate binary or encoded payloads.
_CONTROL_RUN_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]{10,}")

# Extremely long single-character runs indicate abusive or malformed input.
_LONG_REPEAT_RE = re.compile(r"(.)\1{100,}")


class InputValidationError(ValueError):
    """Raised when input fails shared normalization or validation.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers on
    every surface keep working while callers that want to distinguish gateway
    rejections can catch this type directly.

    The message never contains the raw input text. Attributes carry only
    non-sensitive metadata (a stable ``code`` plus optional sizes/limits) so
    callers can build structured responses without touching PHI.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        limit: Optional[int] = None,
        actual: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.limit = limit
        self.actual = actual


@dataclass(frozen=True)
class GatewayLimits:
    """Bounds applied by the shared gateway.

    Attributes:
        max_chars: Maximum number of characters after stripping. ``None``
            disables the character cap.
        max_bytes: Maximum size of the UTF-8 encoding in bytes. ``None``
            disables the byte cap.
    """

    max_chars: Optional[int] = DEFAULT_MAX_TEXT_CHARS
    max_bytes: Optional[int] = DEFAULT_MAX_TEXT_BYTES


@dataclass(frozen=True)
class NormalizedInput:
    """Result of :func:`normalize_input`.

    Attributes:
        text: The normalized, validated text.
        language: The validated language code, or ``None`` when not requested.
    """

    text: str
    language: Optional[str] = None


def _parse_positive_int_env(env_var: str, default: int) -> int:
    """Return a positive int from ``env_var`` or ``default`` on any problem."""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def get_default_limits() -> GatewayLimits:
    """Return the effective gateway limits from the environment.

    The character cap is sourced from the same environment variable the REST
    service already honours (``OPENMED_SERVICE_MAX_TEXT_LENGTH``) so all three
    surfaces agree. The byte cap is read from :data:`MAX_TEXT_BYTES_ENV_VAR`.
    Invalid or non-positive values fall back to the module defaults.
    """
    # Imported lazily to avoid a package import cycle (service imports utils).
    from openmed.service.limits import get_max_text_length

    return GatewayLimits(
        max_chars=get_max_text_length(),
        max_bytes=_parse_positive_int_env(
            MAX_TEXT_BYTES_ENV_VAR, DEFAULT_MAX_TEXT_BYTES
        ),
    )


def _coerce_to_text(value: Any) -> str:
    """Coerce ``value`` to ``str`` while rejecting undecodable bytes.

    Raises:
        InputValidationError: If ``value`` is ``None`` or bytes that are not
            valid UTF-8.
    """
    if value is None:
        raise InputValidationError("Input text is required.", code="text_required")

    if isinstance(value, str):
        # ``str`` can still hold lone surrogates (e.g. from ``surrogateescape``
        # decoding of invalid bytes). Those cannot be encoded as UTF-8 and would
        # blow up deep in tokenization, so reject them here with a clear error.
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InputValidationError(
                "Input text is not valid UTF-8 (contains unpaired surrogates).",
                code="invalid_encoding",
            ) from exc
        return value

    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputValidationError(
                "Input text is not valid UTF-8.",
                code="invalid_encoding",
            ) from exc

    return str(value)


def ensure_valid_encoding(value: Any) -> str:
    """Coerce ``value`` to text, rejecting undecodable bytes or lone surrogates.

    Public shim over the shared encoding check so surfaces that keep their own
    length/empty rules (notably :func:`openmed.utils.validation.validate_input`)
    still reject invalid UTF-8 the same way the full gateway does.

    Args:
        value: ``None``, ``str``, ``bytes``, or any object with a ``str()``.

    Returns:
        The value as ``str``.

    Raises:
        InputValidationError: If ``value`` is ``None`` or not valid UTF-8. The
            message never contains the input text.
    """
    return _coerce_to_text(value)


def _reject_suspicious_content(text: str) -> None:
    """Reject binary/control payloads without echoing the text.

    Raises:
        InputValidationError: If control characters or abusive runs are found.
    """
    if _CONTROL_RUN_RE.search(text):
        raise InputValidationError(
            "Input text contains a run of control characters.",
            code="control_characters",
        )
    if _CONTROL_CHAR_RE.search(text):
        raise InputValidationError(
            "Input text contains disallowed control characters.",
            code="control_characters",
        )
    if _LONG_REPEAT_RE.search(text):
        raise InputValidationError(
            "Input text contains an abusive repeated-character run.",
            code="repeated_characters",
        )


def normalize_text(
    text: Any,
    *,
    limits: Optional[GatewayLimits] = None,
    allow_empty: bool = False,
    strip: bool = True,
) -> str:
    """Normalize and validate free-text input for any OpenMed surface.

    The single normalization path used by the library, REST, and MCP entry
    points. It performs, in order: encoding validation (reject invalid UTF-8 and
    unpaired surrogates), optional whitespace stripping, empty/blank checks,
    control-character rejection, and character/byte length caps.

    Args:
        text: The raw input. ``None``, ``str``, ``bytes``, or any object with a
            useful ``str()`` representation.
        limits: Length/size bounds. Defaults to :func:`get_default_limits`.
        allow_empty: When ``True``, empty/blank input yields ``""`` instead of
            raising.
        strip: When ``True`` (default), leading/trailing whitespace is removed
            before validation.

    Returns:
        The normalized text.

    Raises:
        InputValidationError: If the input is missing, malformed, mis-encoded,
            or exceeds the configured limits. The message never contains the
            input text.
    """
    if limits is None:
        limits = get_default_limits()

    decoded = _coerce_to_text(text)
    normalized = decoded.strip() if strip else decoded

    if not normalized:
        if allow_empty:
            return ""
        raise InputValidationError("Input text must not be empty.", code="empty_text")

    _reject_suspicious_content(normalized)

    if limits.max_chars is not None and len(normalized) > limits.max_chars:
        raise InputValidationError(
            f"Input text exceeds the maximum length of {limits.max_chars} characters.",
            code="max_chars",
            limit=limits.max_chars,
            actual=len(normalized),
        )

    if limits.max_bytes is not None:
        byte_length = len(normalized.encode("utf-8"))
        if byte_length > limits.max_bytes:
            raise InputValidationError(
                f"Input text exceeds the maximum size of {limits.max_bytes} bytes.",
                code="max_bytes",
                limit=limits.max_bytes,
                actual=byte_length,
            )

    return normalized


def validate_language(
    lang: Any,
    *,
    supported: Optional[Iterable[str]] = None,
) -> str:
    """Validate a language code against the supported set.

    Args:
        lang: The requested ISO 639-1 language code.
        supported: Iterable of allowed codes. Defaults to
            :data:`openmed.core.pii_i18n.SUPPORTED_LANGUAGES`.

    Returns:
        The validated language code (lower-cased, stripped).

    Raises:
        InputValidationError: If the code is missing, the wrong type, or not in
            the supported set.
    """
    if lang is None:
        raise InputValidationError(
            "Language code is required.", code="language_required"
        )
    if not isinstance(lang, str):
        raise InputValidationError(
            "Language code must be a string.", code="language_type"
        )

    normalized = lang.strip().lower()
    if not normalized:
        raise InputValidationError(
            "Language code must not be blank.", code="language_required"
        )

    if supported is None:
        from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

        supported = SUPPORTED_LANGUAGES

    allowed = set(supported)
    if normalized not in allowed:
        raise InputValidationError(
            f"Unsupported language '{normalized}'. Supported: {sorted(allowed)}",
            code="unsupported_language",
        )
    return normalized


def normalize_input(
    text: Any,
    *,
    lang: Any = None,
    limits: Optional[GatewayLimits] = None,
    allow_empty: bool = False,
    supported_languages: Optional[Iterable[str]] = None,
) -> NormalizedInput:
    """Normalize text and (optionally) validate a language in one call.

    Convenience wrapper the surfaces use when they need both the normalized text
    and a validated language code.

    Args:
        text: Raw input text (see :func:`normalize_text`).
        lang: Optional language code. When ``None``, no language validation runs
            and :attr:`NormalizedInput.language` is ``None``.
        limits: Optional length/size bounds.
        allow_empty: Whether empty text is allowed.
        supported_languages: Optional override for the allowed language set.

    Returns:
        A :class:`NormalizedInput` with the validated text and language.

    Raises:
        InputValidationError: On any text or language validation failure.
    """
    normalized_text = normalize_text(text, limits=limits, allow_empty=allow_empty)
    normalized_lang = (
        None if lang is None else validate_language(lang, supported=supported_languages)
    )
    return NormalizedInput(text=normalized_text, language=normalized_lang)
