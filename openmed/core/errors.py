"""Structured error taxonomy for the OpenMed public API.

This module defines a single rooted exception hierarchy so that callers of the
public de-identify / re-identify / extract surface can reliably distinguish a
bad *input* from a missing *capability*, a broken *configuration*, an exceeded
*budget*, or an unexpected *internal* fault -- rather than pattern-matching on
generic :class:`ValueError` / :class:`RuntimeError` / :class:`ImportError`.

Design goals
------------

* **Rooted hierarchy.** Every OpenMed error is an :class:`OpenMedError`, so a
  caller can ``except OpenMedError`` to catch anything the library raises for an
  expected failure.
* **Backwards compatible.** Each typed error also subclasses the builtin it
  replaces (for example :class:`InputError` is a :class:`ValueError`), so
  existing ``except ValueError`` / ``except RuntimeError`` / ``except
  ImportError`` call sites keep working unchanged.
* **Actionable, PHI-free messages.** Every message states *what* failed, the
  *likely cause*, and a concrete *remediation*. Messages never embed raw input
  text or detected identifiers -- only offsets, labels, hashes, counts, and
  configuration values. Use :func:`redact_detail` for any value whose origin is
  untrusted.
* **Stable machine-readable codes.** Each class carries a ``code`` string used
  by the service and MCP layers to emit a documented, stable error code
  independent of the human-readable message.

Taxonomy
--------

``OpenMedError``  -- base of everything below.

* ``InputError``          (also ``ValueError``)  -- caller supplied malformed or
  invalid input (empty text, unknown language, conflicting parameters, bad
  output format).
* ``ConfigurationError``  (also ``ValueError``)  -- configuration or profile is
  missing, unknown, or inconsistent.
* ``CapabilityError``     (also ``ImportError``) -- a requested capability is
  unavailable in this environment.

    * ``MissingExtraError``  -- an optional dependency / install extra is not
      installed.
    * ``ModelLoadError``     -- a model, tokenizer, or backend could not be
      loaded.

* ``PolicyError``         (also ``ValueError``)  -- an operation violates a
  redaction / privacy policy constraint.
* ``BudgetExceededError`` (also ``RuntimeError``) -- a resource, memory, or time
  budget was exceeded.
* ``InternalError``       (also ``RuntimeError``) -- an unexpected internal
  invariant was violated; indicates a bug, not caller error.

    * ``InferenceError``     -- inference produced a result that violated an
      internal invariant.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

__all__ = [
    "OpenMedError",
    "InputError",
    "ConfigurationError",
    "CapabilityError",
    "MissingExtraError",
    "ModelLoadError",
    "PolicyError",
    "BudgetExceededError",
    "InternalError",
    "InferenceError",
    "redact_detail",
]


def redact_detail(value: Any, *, keep: int = 0) -> str:
    """Return a PHI-safe, stable descriptor for an untrusted value.

    Error messages must never echo raw input text or detected identifiers. When
    a message needs to reference an untrusted value (for example the offending
    span of text), pass it through this helper: it returns the value's length
    and a short truncated SHA-256 digest instead of the plaintext, so the
    descriptor is reproducible for debugging without leaking content.

    Args:
        value: The untrusted value to describe. Coerced to ``str``.
        keep: Number of leading characters to preserve verbatim. Defaults to
            ``0`` (fully redacted). Only raise this for values already known to
            be non-sensitive (for example an enum name).

    Returns:
        A descriptor such as ``"<redacted len=42 sha256=1a2b3c4d>"`` that
        contains no raw PHI.
    """

    text = value if isinstance(value, str) else str(value)
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
    prefix = ""
    if keep > 0 and text:
        prefix = f"prefix={text[:keep]!r} "
    return f"<redacted {prefix}len={len(text)} sha256={digest}>"


class OpenMedError(Exception):
    """Base class for every error raised on the OpenMed public API.

    Catch this to handle any expected OpenMed failure regardless of subtype.
    Carries a stable machine-readable :attr:`code` (used by the service and MCP
    layers) and an optional structured :attr:`details` mapping that must never
    contain raw PHI.

    Args:
        message: Actionable, PHI-free description (what failed, likely cause,
            remediation).
        code: Stable machine-readable error code. Defaults to the class-level
            :attr:`code`.
        details: Optional structured, PHI-free context (offsets, labels,
            counts, configuration values).
    """

    #: Stable machine-readable code for this class. Subclasses override.
    code: str = "openmed_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}


class InputError(OpenMedError, ValueError):
    """Caller supplied malformed or invalid input.

    Also a :class:`ValueError` for backwards compatibility, so existing
    ``except ValueError`` handlers continue to catch it. Raise this for empty or
    wrong-typed text, unknown languages, conflicting parameters, and
    unsupported output formats.
    """

    code = "input_error"


class ConfigurationError(OpenMedError, ValueError):
    """Configuration or profile is missing, unknown, or inconsistent.

    Also a :class:`ValueError` for backwards compatibility.
    """

    code = "configuration_error"


class CapabilityError(OpenMedError, ImportError):
    """A requested capability is unavailable in this environment.

    Also an :class:`ImportError` for backwards compatibility, so callers that
    guard optional features with ``except ImportError`` keep working.
    """

    code = "capability_error"


class MissingExtraError(CapabilityError):
    """An optional dependency or install extra is not installed.

    Args:
        message: Actionable message including the exact install command.
        package: The missing distribution name.
        extra: The OpenMed install extra that provides it, if any.
        feature: The capability that needed it.
        details: Optional extra PHI-free context.
    """

    code = "missing_extra"

    def __init__(
        self,
        message: str,
        *,
        package: Optional[str] = None,
        extra: Optional[str] = None,
        feature: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        merged: dict[str, Any] = {}
        if package is not None:
            merged["package"] = package
        if extra is not None:
            merged["extra"] = extra
        if feature is not None:
            merged["feature"] = feature
        if details:
            merged.update(details)
        super().__init__(message, details=merged)
        self.package = package
        self.extra = extra
        self.feature = feature


class ModelLoadError(CapabilityError):
    """A model, tokenizer, or backend could not be loaded.

    Args:
        message: Actionable, PHI-free message. Do not embed model *inputs*;
            the model *identifier* is safe to include.
        model_name: The model identifier that failed to load, if known.
        details: Optional extra PHI-free context.
    """

    code = "model_load_error"

    def __init__(
        self,
        message: str,
        *,
        model_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        merged: dict[str, Any] = {}
        if model_name is not None:
            merged["model_name"] = model_name
        if details:
            merged.update(details)
        super().__init__(message, details=merged)
        self.model_name = model_name


class PolicyError(OpenMedError, ValueError):
    """An operation violates a redaction or privacy policy constraint.

    Also a :class:`ValueError` for backwards compatibility.
    """

    code = "policy_error"


class BudgetExceededError(OpenMedError, RuntimeError):
    """A resource, memory, or time budget was exceeded.

    Also a :class:`RuntimeError` for backwards compatibility.
    """

    code = "budget_exceeded"


class InternalError(OpenMedError, RuntimeError):
    """An unexpected internal invariant was violated.

    Indicates a bug in OpenMed rather than caller error. Also a
    :class:`RuntimeError` for backwards compatibility.
    """

    code = "internal_error"


class InferenceError(InternalError):
    """Inference produced a result that violated an internal invariant.

    For example, a backend returned a batch whose length did not match its
    input. Subclass of :class:`InternalError`.
    """

    code = "inference_error"
