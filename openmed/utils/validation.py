"""Input validation utilities for OpenMed.

This module is the shared input-normalization and validation gateway used by
every entry point: the library, the REST service schemas, and the MCP tool
handlers. Routing all three surfaces through the same functions keeps their
length, byte-size, encoding, and language guardrails identical, and raises a
single typed error (:class:`InputValidationError`) whose messages and metadata
never echo the raw input so error surfaces cannot leak PHI.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

# UTF-8 byte-size cap applied on top of the character cap. Bounds request
# memory even when a small character count decodes into many bytes (e.g. a
# short string of astral-plane code points). Overridable per deployment.
DEFAULT_MAX_TEXT_BYTES = 4_000_000
SERVICE_MAX_TEXT_BYTES_ENV_VAR = "OPENMED_SERVICE_MAX_TEXT_BYTES"

# Sentinel distinguishing "caller did not specify a cap" (use the configured
# default) from an explicit ``None`` (disable that particular cap).
_UNSET = object()


class InputValidationError(ValueError):
    """Raised when the shared gateway rejects normalized input.

    Subclasses :class:`ValueError` so existing callers that catch ``ValueError``
    keep working. Carries a stable machine-readable ``code`` and a ``metadata``
    mapping restricted to non-sensitive facts (sizes, limits, the supported
    language set) -- never the input text itself -- so validation errors are
    safe to surface without leaking PHI.
    """

    def __init__(self, message: str, *, code: str, **metadata: Any) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = dict(metadata)


def _parse_positive_int(raw_value: Optional[str], default: int) -> int:
    """Parse a positive integer env value, falling back to ``default``."""
    if raw_value is None:
        return default
    raw_value = raw_value.strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def get_max_text_bytes() -> int:
    """Return the current UTF-8 byte-size cap for request text."""
    return _parse_positive_int(
        os.getenv(SERVICE_MAX_TEXT_BYTES_ENV_VAR), DEFAULT_MAX_TEXT_BYTES
    )


def _default_max_length() -> Optional[int]:
    """Resolve the configured character cap (honors the service env var)."""
    try:
        from openmed.service.limits import get_max_text_length

        return get_max_text_length()
    except Exception:  # pragma: no cover - defensive fallback
        return None


def _decode_utf8(text: Any) -> Any:
    """Reject invalid UTF-8 / unpaired surrogates before deeper processing.

    ``bytes`` inputs must decode as strict UTF-8; ``str`` inputs must round-trip
    through UTF-8 (which fails for unpaired surrogates). Raising here keeps
    encoding faults from surfacing deep inside tokenization.
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            return bytes(text).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputValidationError(
                "Input contains invalid UTF-8 bytes",
                code="invalid_encoding",
            ) from exc
    if isinstance(text, str):
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InputValidationError(
                "Input contains unpaired surrogate code points",
                code="invalid_encoding",
            ) from exc
    return text


def validate_text_input(
    text: Any,
    *,
    min_length: int = 1,
    max_length: Any = _UNSET,
    max_bytes: Any = _UNSET,
    allow_empty: bool = False,
    check_suspicious: bool = False,
) -> str:
    """Normalize and validate request text through the shared gateway.

    Enforces, in order: encoding validity (invalid UTF-8 / unpaired
    surrogates), whitespace stripping, emptiness, character length, and UTF-8
    byte size. ``max_length``/``max_bytes`` left at their sentinel default fall
    back to the configured caps; passing ``None`` disables that cap.

    Args:
        text: Raw input (``str``, ``bytes``, or coercible value).
        min_length: Minimum allowed character length.
        max_length: Character cap; sentinel = configured default, ``None`` = off.
        max_bytes: UTF-8 byte cap; sentinel = configured default, ``None`` = off.
        allow_empty: Whether to allow empty strings.
        check_suspicious: Whether to reject suspicious content (library only).

    Returns:
        The normalized text.

    Raises:
        InputValidationError: If any guardrail fails. Carries a stable ``code``
            and non-sensitive ``metadata``; never includes the input text.
    """
    if text is None:
        if allow_empty:
            return ""
        raise InputValidationError("Input text cannot be None", code="text_required")

    text = _decode_utf8(text)

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if not text and not allow_empty:
        raise InputValidationError("Input text cannot be empty", code="text_empty")

    if len(text) < min_length:
        if allow_empty and len(text) == 0:
            return text
        raise InputValidationError(
            f"Input text too short. Minimum length: {min_length}",
            code="text_too_short",
            min_length=min_length,
            length=len(text),
        )

    if max_length is _UNSET:
        max_length = _default_max_length()
    if max_length and len(text) > max_length:
        raise InputValidationError(
            f"Input text too long. Maximum length: {max_length}",
            code="text_too_long",
            max_length=max_length,
            length=len(text),
        )

    if max_bytes is _UNSET:
        max_bytes = get_max_text_bytes()
    if max_bytes:
        byte_size = len(text.encode("utf-8"))
        if byte_size > max_bytes:
            raise InputValidationError(
                f"Input text too large. Maximum size: {max_bytes} bytes",
                code="text_too_large",
                max_bytes=max_bytes,
                byte_size=byte_size,
            )

    if check_suspicious and _contains_suspicious_content(text):
        raise InputValidationError(
            "Input text contains suspicious content",
            code="suspicious_content",
        )

    return text


def validate_language(
    lang: Any,
    *,
    accepted: Optional[Any] = None,
    include_national_id: bool = True,
) -> str:
    """Validate a language code against the supported set (single source of truth).

    When ``accepted`` is not supplied the accepted set is derived from the core
    catalog (built-in ``SUPPORTED_LANGUAGES`` plus optional Indic NER routes,
    and -- unless ``include_national_id`` is false -- the pattern-only
    national-ID languages) so every surface guards against the same set.

    Args:
        lang: Language code to validate.
        accepted: Explicit accepted set; derived from the catalog when ``None``.
        include_national_id: Include pattern-only national-ID languages when
            deriving the default accepted set.

    Returns:
        The validated language code.

    Raises:
        InputValidationError: If ``lang`` is not in the accepted set. The
            ``metadata`` carries the sorted supported set (no input data).
    """
    if accepted is None:
        from openmed.core.pii_i18n import (
            INDIC_NER_LANGUAGES,
            SUPPORTED_LANGUAGES,
        )

        accepted = set(SUPPORTED_LANGUAGES) | set(INDIC_NER_LANGUAGES)
        if include_national_id:
            from openmed.core.pii_i18n import NATIONAL_ID_ONLY_LANGUAGES

            accepted = accepted | set(NATIONAL_ID_ONLY_LANGUAGES)

    if lang not in accepted:
        supported = sorted(accepted)
        raise InputValidationError(
            f"Unsupported language '{lang}'. Supported: {supported}",
            code="unsupported_language",
            supported=supported,
        )
    return lang


def validate_input(
    text: Any,
    min_length: int = 1,
    max_length: Optional[int] = None,
    allow_empty: bool = False,
) -> str:
    """Validate and clean input text.

    Thin backward-compatible wrapper over :func:`validate_text_input`: it adds
    the suspicious-content heuristic and treats ``max_length=None`` as "use the
    configured character cap".

    Args:
        text: Input text to validate.
        min_length: Minimum allowed text length.
        max_length: Maximum allowed text length (``None`` = configured cap).
        allow_empty: Whether to allow empty strings.

    Returns:
        Validated and cleaned text.

    Raises:
        ValueError: If validation fails (an :class:`InputValidationError`).
    """
    resolved_max = _UNSET if max_length is None else max_length
    return validate_text_input(
        text,
        min_length=min_length,
        max_length=resolved_max,
        allow_empty=allow_empty,
        check_suspicious=True,
    )


def validate_model_name(model_name: str) -> str:
    """Validate model name format.

    Args:
        model_name: Model name to validate.

    Returns:
        Validated model name.

    Raises:
        ValueError: If model name is invalid.
    """
    if not isinstance(model_name, str):
        raise ValueError("Model name must be a string")

    model_name = model_name.strip()

    if not model_name:
        raise ValueError("Model name cannot be empty")

    # Allow existing local model directories/files in addition to Hub-style ids.
    expanded_path = Path(model_name).expanduser()
    if expanded_path.exists():
        return str(expanded_path)

    # Check format (organization/model or just model)
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts) != 2:
            raise ValueError("Invalid model name format. Use 'org/model' or 'model'")

        org, model = parts
        if not org or not model:
            raise ValueError("Organization and model name cannot be empty")

        # Validate characters
        if not re.match(r"^[a-zA-Z0-9\-_.]+$", org):
            raise ValueError("Invalid characters in organization name")
        if not re.match(r"^[a-zA-Z0-9\-_.]+$", model):
            raise ValueError("Invalid characters in model name")
    else:
        # Just model name
        if not re.match(r"^[a-zA-Z0-9\-_.]+$", model_name):
            raise ValueError("Invalid characters in model name")

    return model_name


def validate_confidence_threshold(threshold: float) -> float:
    """Validate confidence threshold value.

    Args:
        threshold: Confidence threshold to validate.

    Returns:
        Validated threshold.

    Raises:
        ValueError: If threshold is invalid.
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Confidence threshold must be a number")

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    return float(threshold)


def validate_output_format(format_name: str) -> str:
    """Validate output format name.

    Args:
        format_name: Output format to validate.

    Returns:
        Validated format name.

    Raises:
        ValueError: If format is not supported.
    """
    valid_formats = ["dict", "json", "html", "csv"]

    if not isinstance(format_name, str):
        raise ValueError("Output format must be a string")

    format_name = format_name.lower().strip()

    if format_name not in valid_formats:
        raise ValueError(f"Unsupported output format. Valid formats: {valid_formats}")

    return format_name


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> int:
    """Validate batch size for processing.

    Args:
        batch_size: Batch size to validate.
        max_batch_size: Maximum allowed batch size.

    Returns:
        Validated batch size.

    Raises:
        ValueError: If batch size is invalid.
    """
    if not isinstance(batch_size, int):
        raise ValueError("Batch size must be an integer")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if batch_size > max_batch_size:
        raise ValueError(f"Batch size too large. Maximum: {max_batch_size}")

    return batch_size


def _contains_suspicious_content(text: str) -> bool:
    """Check if text contains suspicious content.

    Args:
        text: Text to check.

    Returns:
        True if suspicious content is found.
    """
    # Check for extremely long repeated characters
    if re.search(r"(.)\1{100,}", text):
        return True

    # Check for excessive special characters
    special_char_ratio = len(re.findall(r"[^\w\s]", text)) / len(text) if text else 0
    if special_char_ratio > 0.5:
        return True

    # Check for binary or encoded content (control characters excluding
    # common whitespace).  Do NOT reject non-ASCII text — CJK, Arabic,
    # Devanagari and other scripts legitimately contain long non-ASCII runs.
    if re.search(r"[\x00-\x08\x0e-\x1f\x7f]{10,}", text):
        return True

    return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations.

    Args:
        filename: Filename to sanitize.

    Returns:
        Sanitized filename.
    """
    if not isinstance(filename, str):
        filename = str(filename)

    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove control characters
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    # Ensure not empty
    if not filename.strip():
        filename = "output"

    return filename.strip()
