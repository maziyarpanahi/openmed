"""Shared input-normalization and validation gateway.

The Python library, REST service, and MCP server use this module for the same
text length, UTF-8 byte-size, encoding, and language guardrails. Validation
errors intentionally contain only stable codes and non-sensitive metadata;
they never echo input text that may contain PHI.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from openmed.core.errors import InputError

__all__ = [
    "DEFAULT_MAX_TEXT_BYTES",
    "DEFAULT_MAX_TEXT_CHARS",
    "MAX_TEXT_BYTES_ENV_VAR",
    "MAX_TEXT_CHARS_ENV_VAR",
    "GatewayLimits",
    "InputValidationError",
    "get_default_limits",
    "normalize_text",
    "validate_language",
]

DEFAULT_MAX_TEXT_CHARS = 1_000_000
DEFAULT_MAX_TEXT_BYTES = 4_000_000
MAX_TEXT_CHARS_ENV_VAR = "OPENMED_SERVICE_MAX_TEXT_LENGTH"
MAX_TEXT_BYTES_ENV_VAR = "OPENMED_MAX_TEXT_BYTES"


class InputValidationError(InputError):
    """Raised when shared input validation fails.

    Args:
        message: PHI-safe human-readable failure description.
        code: Stable machine-readable failure code.
        metadata: Optional non-sensitive structured details.
        limit: Optional compatibility attribute for a rejected limit.
        actual: Optional compatibility attribute for the observed size.

    Attributes:
        code: Stable machine-readable failure code.
        metadata: Non-sensitive structured details such as sizes and limits.
        limit: Rejected limit when the error concerns a bound.
        actual: Observed size when the error concerns a bound.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        metadata: Optional[Mapping[str, object]] = None,
        limit: Optional[int] = None,
        actual: Optional[int] = None,
    ) -> None:
        details = dict(metadata or {})
        if limit is not None:
            details.setdefault("limit", limit)
        if actual is not None:
            details.setdefault("actual", actual)
        super().__init__(message, code=code, details=details)
        # Preserve the v2.1 constructor-owned public attribute for the static
        # compatibility inventory as well as the runtime value from InputError.
        self.code = code  # type: ignore[misc]
        self.metadata = details
        self.limit = limit
        self.actual = actual


@dataclass(frozen=True)
class GatewayLimits:
    """Character and UTF-8 byte bounds for normalized text.

    Attributes:
        max_chars: Maximum normalized character count, or ``None`` to disable.
        max_bytes: Maximum normalized UTF-8 byte count, or ``None`` to disable.
    """

    max_chars: Optional[int] = DEFAULT_MAX_TEXT_CHARS
    max_bytes: Optional[int] = DEFAULT_MAX_TEXT_BYTES


def _parse_positive_int_env(name: str, default: int) -> int:
    """Return a positive integer environment value or a defensive default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def get_default_limits() -> GatewayLimits:
    """Return current shared text limits.

    The character limit honors the service's existing
    ``OPENMED_SERVICE_MAX_TEXT_LENGTH`` setting. ``OPENMED_MAX_TEXT_BYTES``
    applies the same UTF-8 byte cap to library, REST, and MCP entry points.

    Returns:
        Effective shared gateway limits.
    """
    return GatewayLimits(
        max_chars=_parse_positive_int_env(
            MAX_TEXT_CHARS_ENV_VAR,
            DEFAULT_MAX_TEXT_CHARS,
        ),
        max_bytes=_parse_positive_int_env(
            MAX_TEXT_BYTES_ENV_VAR,
            DEFAULT_MAX_TEXT_BYTES,
        ),
    )


def _coerce_to_text(value: Any) -> str:
    """Return strict UTF-8 text without exposing rejected input in errors."""
    if value is None:
        raise InputValidationError(
            "Input text cannot be None. Pass a str or strict UTF-8 bytes-like value.",
            code="text_required",
        )

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise InputValidationError(
                "Input text is not valid UTF-8. Re-encode the payload as strict "
                "UTF-8 before retrying.",
                code="invalid_encoding",
            ) from None

    if not isinstance(value, str):
        raise InputValidationError(
            "Input text has an unsupported type. Pass a str or strict UTF-8 "
            "bytes-like value.",
            code="text_type",
            metadata={"type": type(value).__name__},
        )

    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise InputValidationError(
            "Input text is not valid UTF-8. Normalize or re-encode it as valid "
            "Unicode before retrying.",
            code="invalid_encoding",
        ) from None
    return value


def normalize_text(
    text: Any,
    *,
    limits: Optional[GatewayLimits] = None,
    min_length: int = 1,
    allow_empty: bool = False,
    strip: bool = True,
) -> str:
    """Normalize text and enforce shared size and encoding guardrails.

    Args:
        text: Text or a strict UTF-8 bytes-like input.
        limits: Optional character and byte limits. Current configured defaults
            are used when omitted.
        min_length: Minimum normalized character count.
        allow_empty: Whether missing, empty, or blank input normalizes to ``""``.
        strip: Whether to remove leading and trailing whitespace.

    Returns:
        Normalized text.

    Raises:
        InputValidationError: If text is missing, empty, too short, too long,
            too large in UTF-8, or invalidly encoded.
    """
    if text is None and allow_empty:
        return ""

    effective_limits = limits or get_default_limits()
    decoded = _coerce_to_text(text)
    normalized = decoded.strip() if strip else decoded

    if not normalized:
        if allow_empty:
            return ""
        raise InputValidationError(
            "Input text cannot be empty. Pass the text to process or explicitly "
            "enable empty input where supported.",
            code="empty_text",
        )

    if len(normalized) < min_length:
        raise InputValidationError(
            f"Input text too short; the minimum is {min_length} characters. "
            "Pass a longer input before retrying.",
            code="min_chars",
            metadata={"unit": "characters"},
            limit=min_length,
            actual=len(normalized),
        )

    if (
        effective_limits.max_chars is not None
        and len(normalized) > effective_limits.max_chars
    ):
        raise InputValidationError(
            f"Input text too long; the limit is {effective_limits.max_chars} "
            "characters. Reduce the input or increase the configured limit.",
            code="max_chars",
            metadata={"unit": "characters"},
            limit=effective_limits.max_chars,
            actual=len(normalized),
        )

    byte_length = len(normalized.encode("utf-8"))
    if (
        effective_limits.max_bytes is not None
        and byte_length > effective_limits.max_bytes
    ):
        raise InputValidationError(
            f"Input text exceeds the {effective_limits.max_bytes}-byte limit. "
            "Reduce the input or increase the configured byte limit.",
            code="max_bytes",
            metadata={"unit": "bytes"},
            limit=effective_limits.max_bytes,
            actual=byte_length,
        )

    return normalized


def _default_supported_languages(*, include_national_id: bool) -> set[str]:
    """Return the canonical PII language set from the core catalog.

    ``USER_SUPPLIED_MODEL_LANGUAGES`` is part of the base set: those codes are
    publicly registered on the REST, MCP, and client language enums and are
    accepted by ``openmed.core.pii._resolve_effective_pii_model``. They ship no
    bundled weights, so the resolver asks for an explicit ``model_name``; the
    gateway must not reject them one layer earlier.
    """
    from openmed.core.pii_i18n import (
        INDIC_NER_LANGUAGES,
        NATIONAL_ID_ONLY_LANGUAGES,
        SUPPORTED_LANGUAGES,
        USER_SUPPLIED_MODEL_LANGUAGES,
    )

    languages = set(
        SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES | USER_SUPPLIED_MODEL_LANGUAGES
    )
    if include_national_id:
        languages.update(NATIONAL_ID_ONLY_LANGUAGES)
    return languages


def validate_language(
    lang: Any,
    *,
    supported: Optional[Iterable[str]] = None,
    include_national_id: bool = True,
) -> str:
    """Normalize and validate a PII language code.

    Args:
        lang: Requested language code.
        supported: Optional caller-specific set. The canonical PII catalog is
            used when omitted.
        include_national_id: Whether default validation includes deterministic
            national-ID-only languages. Ignored when ``supported`` is supplied.
            A code that is both national-ID-only and user-supplied-model (such
            as Urdu) stays accepted either way, because it is registered on the
            public language enums.

    Returns:
        Lowercase validated language code.

    Raises:
        InputValidationError: If the language is missing, invalid, or unsupported.
    """
    if lang is None:
        raise InputValidationError(
            "Language code is required. Pass a supported ISO language code.",
            code="language_required",
        )
    if not isinstance(lang, str):
        raise InputValidationError(
            "Language code must be a string. Pass a supported ISO language code.",
            code="language_type",
        )

    normalized = lang.strip().lower()
    if not normalized:
        raise InputValidationError(
            "Language code is required. Pass a supported ISO language code.",
            code="language_required",
        )

    allowed = (
        _default_supported_languages(include_national_id=include_national_id)
        if supported is None
        else {str(code).strip().lower() for code in supported}
    )
    if normalized not in allowed:
        supported_languages = tuple(sorted(allowed))
        raise InputValidationError(
            "Unsupported language code. Pass one of the documented supported "
            f"codes: {list(supported_languages)}.",
            code="unsupported_language",
            metadata={
                "supported_languages": supported_languages,
                "supported_count": len(supported_languages),
            },
        )
    return normalized
