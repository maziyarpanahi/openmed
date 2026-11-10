"""PHI-safe, categorical error envelopes for agent execution.

Adapters need machine-readable failures, but an ordinary exception string can
carry tool arguments, model output, credentials, filenames, or clinical
context. An envelope replaces that string with a closed class, a stable code,
the stage the failure belongs to, a derived retryability flag, and optional
opaque correlation identifiers. Exception messages, arguments, class names,
and tracebacks are never read or retained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

from openmed.core.errors import (
    BudgetExceededError,
    CapabilityError,
    ConfigurationError,
    InferenceError,
    InputError,
    InternalError,
    MissingExtraError,
    ModelLoadError,
    OpenMedError,
    PolicyError,
)

from .correlation import ActionId, CorrelationIdError, RunId

ERROR_ENVELOPE_SCHEMA_VERSION: Final[str] = "openmed.agent.error.v1"

_ENVELOPE_FIELDS = (
    "schema_version",
    "error_class",
    "error_code",
    "stage",
    "retryable",
    "run_id",
    "action_id",
)


class ErrorClass(str, Enum):
    """Closed set of agent failure classes.

    Values:
        VALIDATION: The request was malformed or unsupported.
        POLICY: A policy or consent check refused the work.
        AUTHORIZATION: A capability grant was missing, expired, or misscoped.
        RESOURCE: A local budget, dependency, or model was unavailable.
        PROVIDER: A tool or model provider failed or misbehaved.
        TIMEOUT: A deadline elapsed before the work finished.
        INTERNAL: An unexpected failure, including every unmapped exception.
    """

    VALIDATION = "validation"
    POLICY = "policy"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    PROVIDER = "provider"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


class ErrorStage(str, Enum):
    """Closed set of stages at which an agent call can fail.

    Values:
        VALIDATION: Before any work was authorized or planned.
        AUTHORIZATION: While checking grants, consent, or purpose.
        PLANNING: While building the action plan.
        TOOL_CALL: While invoking a tool.
        PROVIDER: Inside a model or provider adapter.
        POST_PROCESSING: While validating or shaping the result.
    """

    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    PROVIDER = "provider"
    POST_PROCESSING = "post_processing"


ERROR_CODES: Final[Mapping[ErrorClass, tuple[str, ...]]] = MappingProxyType(
    {
        ErrorClass.VALIDATION: (
            "invalid_configuration",
            "invalid_input",
            "schema_violation",
            "unsupported_input",
        ),
        ErrorClass.POLICY: (
            "consent_required",
            "policy_denied",
            "purpose_mismatch",
        ),
        ErrorClass.AUTHORIZATION: (
            "audience_mismatch",
            "capability_expired",
            "capability_missing",
        ),
        ErrorClass.RESOURCE: (
            "budget_exceeded",
            "model_unavailable",
            "optional_dependency_missing",
            "resource_exhausted",
        ),
        ErrorClass.PROVIDER: (
            "provider_failed",
            "provider_protocol_error",
            "provider_unavailable",
        ),
        ErrorClass.TIMEOUT: (
            "deadline_exceeded",
            "request_timeout",
        ),
        ErrorClass.INTERNAL: ("internal_error",),
    }
)

RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "deadline_exceeded",
        "provider_unavailable",
        "request_timeout",
        "resource_exhausted",
    }
)

_ALLOWED_STAGES: Final[Mapping[ErrorClass, frozenset[ErrorStage]]] = MappingProxyType(
    {
        ErrorClass.VALIDATION: frozenset(
            {
                ErrorStage.VALIDATION,
                ErrorStage.PLANNING,
                ErrorStage.TOOL_CALL,
                ErrorStage.POST_PROCESSING,
            }
        ),
        ErrorClass.POLICY: frozenset(
            {
                ErrorStage.VALIDATION,
                ErrorStage.AUTHORIZATION,
                ErrorStage.PLANNING,
                ErrorStage.TOOL_CALL,
                ErrorStage.POST_PROCESSING,
            }
        ),
        ErrorClass.AUTHORIZATION: frozenset(
            {ErrorStage.AUTHORIZATION, ErrorStage.TOOL_CALL}
        ),
        ErrorClass.RESOURCE: frozenset(
            {
                ErrorStage.PLANNING,
                ErrorStage.TOOL_CALL,
                ErrorStage.PROVIDER,
                ErrorStage.POST_PROCESSING,
            }
        ),
        ErrorClass.PROVIDER: frozenset({ErrorStage.TOOL_CALL, ErrorStage.PROVIDER}),
        ErrorClass.TIMEOUT: frozenset(
            {ErrorStage.TOOL_CALL, ErrorStage.PROVIDER, ErrorStage.POST_PROCESSING}
        ),
        ErrorClass.INTERNAL: frozenset(ErrorStage),
    }
)

_CODE_CLASSES: Final[Mapping[str, ErrorClass]] = MappingProxyType(
    {code: error_class for error_class, codes in ERROR_CODES.items() for code in codes}
)

_EXCEPTION_CODES: Final[tuple[tuple[type[BaseException], str], ...]] = (
    (InputError, "invalid_input"),
    (ConfigurationError, "invalid_configuration"),
    (PolicyError, "policy_denied"),
    (BudgetExceededError, "budget_exceeded"),
    (MissingExtraError, "optional_dependency_missing"),
    (ModelLoadError, "model_unavailable"),
    (CapabilityError, "optional_dependency_missing"),
    (InferenceError, "provider_failed"),
    (InternalError, "internal_error"),
    (OpenMedError, "internal_error"),
    (TimeoutError, "request_timeout"),
    (MemoryError, "resource_exhausted"),
)


class ErrorEnvelopeError(ValueError):
    """Raised when envelope metadata fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional fixed public field associated with the failure.

    Messages and attributes carry controlled diagnostic metadata only; a
    rejected value is never retained or echoed.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    """Immutable, categorical description of one agent failure.

    Attributes:
        error_class: Closed failure class.
        error_code: Stable code belonging to that class.
        stage: Stage the failure belongs to, valid for the class.
        run_id: Optional opaque run correlation identifier.
        action_id: Optional opaque action correlation identifier.
        schema_version: Envelope schema version.

    There is deliberately no free-text detail field. Retryability is derived
    from the code, so it cannot disagree with it.
    """

    error_class: ErrorClass
    error_code: str
    stage: ErrorStage
    run_id: str | None = None
    action_id: str | None = None
    schema_version: str = ERROR_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ERROR_ENVELOPE_SCHEMA_VERSION
        ):
            raise ErrorEnvelopeError("invalid_schema_version", "schema_version")
        if not isinstance(self.error_class, ErrorClass):
            raise ErrorEnvelopeError("unknown_error_class", "error_class")
        if type(self.error_code) is not str:
            raise ErrorEnvelopeError("unknown_error_code", "error_code")
        if self.error_code not in ERROR_CODES[self.error_class]:
            raise ErrorEnvelopeError("unknown_error_code", "error_code")
        if not isinstance(self.stage, ErrorStage):
            raise ErrorEnvelopeError("unknown_stage", "stage")
        if self.stage not in _ALLOWED_STAGES[self.error_class]:
            raise ErrorEnvelopeError("invalid_stage_for_class", "stage")
        if self.run_id is not None:
            _validate_identifier(RunId, self.run_id, "run_id")
        if self.action_id is not None:
            _validate_identifier(ActionId, self.action_id, "action_id")

    @classmethod
    def for_code(
        cls,
        error_code: str,
        *,
        stage: ErrorStage,
        run_id: str | None = None,
        action_id: str | None = None,
    ) -> "ErrorEnvelope":
        """Build an envelope from a code, deriving its class."""

        return cls(
            error_class=error_class_for_code(error_code),
            error_code=error_code,
            stage=stage,
            run_id=run_id,
            action_id=action_id,
        )

    @property
    def retryable(self) -> bool:
        """Return whether the code is documented as worth retrying."""

        return self.error_code in RETRYABLE_ERROR_CODES

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "error_class": self.error_class.value,
            "error_code": self.error_code,
            "stage": self.stage.value,
            "retryable": self.retryable,
            "run_id": self.run_id,
            "action_id": self.action_id,
        }
        return {field: values[field] for field in _ENVELOPE_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def error_class_for_code(error_code: str) -> ErrorClass:
    """Return the closed class that owns a stable error code."""

    if type(error_code) is not str:
        raise ErrorEnvelopeError("unknown_error_code", "error_code")
    try:
        return _CODE_CLASSES[error_code]
    except KeyError:
        pass
    raise ErrorEnvelopeError("unknown_error_code", "error_code")


def is_retryable(error_code: str) -> bool:
    """Return whether a known code is documented as worth retrying."""

    error_class_for_code(error_code)
    return error_code in RETRYABLE_ERROR_CODES


def envelope_from_exception(
    exception: BaseException,
    *,
    stage: ErrorStage,
    run_id: str | None = None,
    action_id: str | None = None,
) -> ErrorEnvelope:
    """Map an exception onto a categorical envelope.

    The exception's type is matched against the documented OpenMed failure
    contract by method resolution order, so a subclass maps to the most
    specific documented ancestor. Nothing else about the exception is read:
    its message, arguments, class name, and traceback are never inspected, so
    they cannot reach the envelope. Anything unmapped becomes
    ``internal_error``.

    Args:
        exception: The raised exception.
        stage: Stage the failure belongs to; it must be valid for the mapped
            class.
        run_id: Optional opaque run correlation identifier.
        action_id: Optional opaque action correlation identifier.

    Returns:
        A validated :class:`ErrorEnvelope`.

    Raises:
        ErrorEnvelopeError: If ``exception`` is not an exception, or the stage
            is not valid for the mapped class.
    """

    if not isinstance(exception, BaseException):
        raise ErrorEnvelopeError("invalid_exception", "exception")
    code = "internal_error"
    for ancestor in type(exception).__mro__:
        match = _lookup_exception_code(ancestor)
        if match is not None:
            code = match
            break
    return ErrorEnvelope.for_code(code, stage=stage, run_id=run_id, action_id=action_id)


def _lookup_exception_code(ancestor: type) -> str | None:
    for mapped, code in _EXCEPTION_CODES:
        if ancestor is mapped:
            return code
    return None


def _validate_identifier(factory: Any, value: Any, field_name: str) -> None:
    try:
        factory(value)
    except CorrelationIdError:
        pass
    else:
        return
    # Raised outside the handler so the rejected identifier cannot be chained.
    raise ErrorEnvelopeError("invalid_identifier", field_name)


__all__ = [
    "ERROR_CODES",
    "ERROR_ENVELOPE_SCHEMA_VERSION",
    "RETRYABLE_ERROR_CODES",
    "ErrorClass",
    "ErrorEnvelope",
    "ErrorEnvelopeError",
    "ErrorStage",
    "envelope_from_exception",
    "error_class_for_code",
    "is_retryable",
]
