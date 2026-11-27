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
import re
from typing import Any, ClassVar, Final, Mapping, Optional

__all__ = [
    "ERROR_CODES",
    "LegacyRootedError",
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
    "legacy_error_code",
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


_CODE_WORD_BOUNDARY: Final = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_STABLE_CODE: Final = re.compile(r"[a-z][a-z0-9_]*")


def _snake_case_code(class_name: str) -> str:
    words = _CODE_WORD_BOUNDARY.sub("_", class_name).lower()
    code = re.sub(r"_+", "_", words).strip("_")
    return code if _STABLE_CODE.fullmatch(code) else OpenMedError.code


def legacy_error_code(error_class: type[BaseException]) -> str:
    """Return the stable code a legacy exception class reports once rooted.

    The nearest class in the method resolution order that defines a ``code``
    class attribute wins, so existing explicit codes and the codes subclasses
    inherit from them never change. A class without one reports the
    snake_case form of its own name, for example ``"aecg_error"`` for
    ``AecgError``. Set ``code`` explicitly to keep a code across a rename.

    The function works on classes that are not rooted yet, which lets release
    tooling record the code a class will report before it is rerooted.

    Args:
        error_class: Exception class to inspect.

    Returns:
        The class-level stable code.

    Raises:
        TypeError: If ``error_class`` is not an exception class, or the
            nearest ``code`` attribute is not a string.
    """

    if not isinstance(error_class, type) or not issubclass(error_class, BaseException):
        raise TypeError("legacy_error_code() expects an exception class")
    for klass in error_class.__mro__:
        if klass is LegacyRootedError or klass is OpenMedError:
            break
        if "code" in vars(klass):
            code = vars(klass)["code"]
            if not isinstance(code, str):
                raise TypeError("an exception code must be a string class attribute")
            return code
    return _snake_case_code(error_class.__name__)


def _legacy_member(error_class: type, name: str) -> Any:
    """Return ``name`` as the MRO would resolve it without the OpenMed roots."""

    mro = error_class.__mro__
    for klass in mro[mro.index(LegacyRootedError) + 1 :]:
        if klass is not OpenMedError and name in vars(klass):
            return vars(klass)[name]
    raise AttributeError(name)  # pragma: no cover - BaseException defines both


class _LegacyCode:
    """Class-level ``code`` default for legacy-rooted exceptions.

    It is a non-data descriptor: a ``code`` defined on a subclass, or set on
    an instance, takes precedence exactly as a plain attribute would.
    """

    def __get__(self, instance: object, owner: Optional[type] = None) -> str:
        error_class = owner if owner is not None else type(instance)
        return _snake_case_code(error_class.__name__)


class LegacyRootedError(OpenMedError):
    """Compatibility root for exceptions that predate :class:`OpenMedError`.

    Rerooting an existing exception adds this class *after* its builtin base,
    for example ``class AecgError(ValueError, LegacyRootedError)``, or replaces
    a bare ``Exception`` base (``class ToolError(LegacyRootedError)``). The
    class then satisfies ``except OpenMedError`` and gains :attr:`code` and
    :meth:`to_dict` while everything callers already rely on stays the same:

    * the constructor accepts any positional arguments, including none, and
      ``args`` keeps exactly the values passed;
    * ``str()``, ``repr()``, pickling and copying behave as the builtin base
      defines them, byte for byte;
    * the builtin bases keep their position in the method resolution order,
      so ``except ValueError`` handlers and MRO-based dispatch tables resolve
      as before. ``LegacyRootedError`` and :class:`OpenMedError` sit directly
      in front of ``Exception``;
    * attributes the class or its instances already define, including
      ``code`` and ``to_dict``, keep their values, and no other attribute is
      added: ``hasattr(error, "message")`` and ``hasattr(error, "details")``
      answer as before. Instances get no new instance attributes, so
      ``vars(error)`` and pickled state are unchanged too.

    :attr:`code` is the nearest explicit ``code`` class attribute or, when
    none exists, the snake_case class name (see :func:`legacy_error_code`).
    :meth:`to_dict` reports that code, ``str(error)`` as the message, and
    empty details, because attributes of a legacy class were never reviewed
    against the PHI-safe details contract.

    The REST and MCP adapters keep the transport mapping these errors had
    before they were rooted; moving them onto the taxonomy envelopes is a
    separate, explicit change.
    """

    code = _LegacyCode()  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize exactly as the builtin base would.

        Args:
            *args: Positional arguments stored unchanged in ``args``.
            **kwargs: Keyword arguments forwarded to the builtin initializer,
                which rejects or consumes them as it always has.
        """

        _legacy_member(type(self), "__init__")(self, *args, **kwargs)

    def __str__(self) -> str:
        """Return the message exactly as the builtin base formats it."""

        return _legacy_member(type(self), "__str__")(self)

    def to_dict(self, *, include_details: bool = True) -> dict[str, Any]:
        """Return a JSON-ready error object.

        Args:
            include_details: Include the (always empty) details mapping, for
                parity with :meth:`OpenMedError.to_dict`.

        Returns:
            A mapping with ``code``, ``message`` (``str(error)``) and, when
            requested, an empty ``details`` mapping.
        """

        payload: dict[str, Any] = {"code": self.code, "message": str(self)}
        if include_details:
            payload["details"] = {}
        return payload


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
