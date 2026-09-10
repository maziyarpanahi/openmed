"""Structured, PHI-safe errors for OpenMed's public API.

The classes in this module are the stable failure contract for public Python,
REST, and MCP entry points.  Every expected public failure descends from
:class:`OpenMedError`, exposes a machine-readable :attr:`OpenMedError.code`,
and keeps the builtin exception base used by earlier OpenMed releases.

Human-readable messages explain what failed and how to recover.  They must not
contain source text or detected identifiers.  Structured details are limited
to non-sensitive selectors, counts, offsets, labels, hashes, and error types.
Use :func:`redact_detail` when untrusted text must be correlated locally.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Final, Mapping, Optional

__all__ = [
    "ERROR_CODES",
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


def redact_detail(value: Any) -> str:
    """Return a stable descriptor for untrusted text without exposing it.

    Args:
        value: Value to describe. It is converted to text only for hashing and
            is never included verbatim in the returned descriptor.

    Returns:
        A descriptor containing only the UTF-8 byte length and SHA-256 digest.
    """

    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        value_type = type(value)
        text = f"<unprintable:{value_type.__module__}.{value_type.__qualname__}>"
    encoded = text.encode("utf-8", errors="replace")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"<redacted bytes={len(encoded)} sha256={digest}>"


class OpenMedError(Exception):
    """Base class for expected failures on the OpenMed public API.

    Args:
        message: Actionable, PHI-free explanation of the failure.
        code: Optional stable leaf code owned by OpenMed. Callers should not
            invent codes; this hook supports existing specialized errors.
        details: Optional PHI-free structured context.

    Attributes:
        code: Stable machine-readable error code.
        message: Actionable, PHI-free human-readable message.
        details: PHI-free structured context.
    """

    code: ClassVar[str] = "openmed_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def __str__(self) -> str:
        """Return the PHI-free human-readable message."""

        return self.message

    def to_dict(self, *, include_details: bool = True) -> dict[str, Any]:
        """Return a JSON-ready error object.

        Args:
            include_details: Include the structured details mapping. Service
                adapters use ``False`` for server-side failures.

        Returns:
            A mapping with stable ``code`` and actionable ``message`` fields.
        """

        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if include_details:
            payload["details"] = dict(self.details)
        return payload


class InputError(OpenMedError, ValueError, TypeError):
    """Caller input is malformed, conflicting, or unsupported.

    ``ValueError`` and ``TypeError`` remain bases so existing handlers keep
    catching value and type validation failures after adopting the taxonomy.
    """

    code = "input_error"
    reason = "input_rejected"


class ConfigurationError(OpenMedError, ValueError, TypeError, KeyError):
    """Configuration is missing, unknown, or inconsistent.

    The legacy ``ValueError``, ``TypeError``, and ``KeyError`` bases preserve
    compatibility with configuration registries and validators.
    """

    code = "configuration_error"


class CapabilityError(OpenMedError, ImportError):
    """A requested runtime, model, or optional capability is unavailable."""

    code = "capability_error"


class MissingExtraError(CapabilityError):
    """An optional package or OpenMed extra is not installed.

    Args:
        message: Actionable message containing an installation instruction.
        package: Missing distribution name, if known.
        feature: Feature that requires the package, if known.
        extra: OpenMed extra that provides the package, if known.
        details: Additional PHI-free structured context.
    """

    code = "missing_extra"

    def __init__(
        self,
        message: str,
        *,
        package: Optional[str] = None,
        feature: Optional[str] = None,
        extra: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if package is not None:
            merged.setdefault("package", package)
        if feature is not None:
            merged.setdefault("feature", feature)
        if extra is not None:
            merged.setdefault("extra", extra)
        super().__init__(message, details=merged)
        self.package = package
        self.feature = feature
        self.extra = extra


class ModelLoadError(CapabilityError, ValueError):
    """A model, tokenizer, or inference backend could not be loaded.

    ``ValueError`` is retained in addition to ``ImportError`` because older
    model-loading paths used both builtin families.

    Args:
        message: Actionable, PHI-free load failure message.
        model_name: Non-sensitive model identifier or local path, if known.
        details: Additional PHI-free structured context.
    """

    code = "model_load_error"

    def __init__(
        self,
        message: str,
        *,
        model_name: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged = dict(details or {})
        if model_name is not None:
            merged.setdefault("model_name", model_name)
        super().__init__(message, details=merged)
        self.model_name = model_name


class PolicyError(OpenMedError, ValueError, TypeError):
    """A request violates or misconfigures a privacy policy constraint."""

    code = "policy_error"


class BudgetExceededError(OpenMedError, RuntimeError):
    """A request exceeded a configured size, time, or resource budget.

    This class accepts both the generic taxonomy constructor and the historical
    request-budget fields used by :mod:`openmed.core.budget`.

    Args:
        message: Optional actionable message. When omitted, one is built from
            ``kind``, ``limit``, ``observed``, and ``checkpoint``.
        kind: Budget dimension such as ``"wall_time"`` or ``"input_chars"``.
        limit: Configured limit.
        observed: Observed value that exceeded the limit.
        checkpoint: Safe pipeline checkpoint where the limit was observed.
        details: Additional PHI-free structured context.
    """

    code = "budget_exceeded"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        kind: Optional[str] = None,
        limit: Optional[float] = None,
        observed: Optional[float] = None,
        checkpoint: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.observed = observed
        self.checkpoint = checkpoint

        merged = dict(details or {})
        for key, value in (
            ("kind", kind),
            ("limit", limit),
            ("observed", observed),
            ("checkpoint", checkpoint),
        ):
            if value is not None:
                merged.setdefault(key, value)

        if message is None:
            if kind == "wall_time" and limit is not None and observed is not None:
                failure = f"wall-time limit {limit:g}s was exceeded after {observed:g}s"
                remediation = (
                    "Increase max_wall_time or reduce the request workload, then retry."
                )
            elif kind == "input_chars" and limit is not None and observed is not None:
                failure = (
                    f"input-length limit {int(limit)} characters was exceeded "
                    f"by a {int(observed)}-character request"
                )
                remediation = (
                    "Reduce the input or increase max_input_chars, then retry."
                )
            else:
                failure = "a configured request budget was exceeded"
                remediation = "Reduce the request or increase its budget, then retry."
            if checkpoint:
                failure = f"{failure} at checkpoint '{checkpoint}'"
            message = f"Request budget exceeded: {failure}. {remediation}"

        super().__init__(message, details=merged)


class InternalError(OpenMedError, RuntimeError):
    """An internal invariant failed and the request cannot safely continue."""

    code = "internal_error"


class InferenceError(InternalError):
    """A model or backend returned a structurally invalid inference result."""

    code = "inference_error"


# Public class-name registry used by service and MCP adapters. Codes are stable
# API values: changing or reusing one is a compatibility break.
ERROR_CODES: Final[Mapping[str, str]] = {
    "OpenMedError": OpenMedError.code,
    "InputError": InputError.code,
    "ConfigurationError": ConfigurationError.code,
    "CapabilityError": CapabilityError.code,
    "MissingExtraError": MissingExtraError.code,
    "ModelLoadError": ModelLoadError.code,
    "PolicyError": PolicyError.code,
    "BudgetExceededError": BudgetExceededError.code,
    "InternalError": InternalError.code,
    "InferenceError": InferenceError.code,
}
