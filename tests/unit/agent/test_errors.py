"""Synthetic unit tests for PHI-safe agent error envelopes."""

from __future__ import annotations

import json

import pytest

from openmed.agent.correlation import ActionId, RunId
from openmed.agent.errors import (
    ERROR_CODES,
    ERROR_ENVELOPE_SCHEMA_VERSION,
    RETRYABLE_ERROR_CODES,
    ErrorClass,
    ErrorEnvelope,
    ErrorEnvelopeError,
    ErrorStage,
    envelope_from_exception,
    error_class_for_code,
    is_retryable,
)
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

RUN_ID = RunId.generate().serialize()
ACTION_ID = ActionId.generate().serialize()

SENTINEL = "JANE DOE MRN 8675309 Bearer abc.def /var/phi/note.txt --prompt=leak"

DOCUMENTED = [
    (InputError, "validation", "invalid_input"),
    (ConfigurationError, "validation", "invalid_configuration"),
    (PolicyError, "policy", "policy_denied"),
    (BudgetExceededError, "resource", "budget_exceeded"),
    (MissingExtraError, "resource", "optional_dependency_missing"),
    (ModelLoadError, "resource", "model_unavailable"),
    (CapabilityError, "resource", "optional_dependency_missing"),
    (InferenceError, "provider", "provider_failed"),
    (InternalError, "internal", "internal_error"),
    (OpenMedError, "internal", "internal_error"),
    (TimeoutError, "timeout", "request_timeout"),
    (MemoryError, "resource", "resource_exhausted"),
]


def build(exception_type):
    if exception_type is MissingExtraError:
        return MissingExtraError(SENTINEL, package="pillow", extra="hf")
    if exception_type is MemoryError:
        return MemoryError()
    if exception_type is TimeoutError:
        return TimeoutError(SENTINEL)
    return exception_type(SENTINEL)


@pytest.mark.parametrize("exception_type,error_class,error_code", DOCUMENTED)
def test_every_documented_failure_class_maps_to_a_stable_code(
    exception_type, error_class, error_code
) -> None:
    envelope = envelope_from_exception(
        build(exception_type), stage=ErrorStage.TOOL_CALL
    )
    assert envelope.error_class.value == error_class
    assert envelope.error_code == error_code
    assert envelope.retryable is (error_code in RETRYABLE_ERROR_CODES)
    assert envelope.schema_version == ERROR_ENVELOPE_SCHEMA_VERSION


@pytest.mark.parametrize(
    "exception",
    [
        ValueError(SENTINEL),
        RuntimeError(SENTINEL),
        KeyError(SENTINEL),
        OSError(SENTINEL),
        ZeroDivisionError(),
        KeyboardInterrupt(),
    ],
)
def test_unknown_exceptions_map_to_a_generic_internal_code(exception) -> None:
    envelope = envelope_from_exception(exception, stage=ErrorStage.PROVIDER)
    assert (envelope.error_class, envelope.error_code) == (
        ErrorClass.INTERNAL,
        "internal_error",
    )
    payload = envelope.to_json()
    assert SENTINEL not in payload
    assert type(exception).__name__ not in payload


def test_subclasses_map_to_their_most_specific_documented_ancestor() -> None:
    class LocalInference(InferenceError):
        pass

    class LocalPolicy(PolicyError):
        pass

    assert (
        envelope_from_exception(
            LocalInference(SENTINEL), stage=ErrorStage.PROVIDER
        ).error_code
        == "provider_failed"
    )
    assert (
        envelope_from_exception(
            LocalPolicy(SENTINEL), stage=ErrorStage.AUTHORIZATION
        ).error_code
        == "policy_denied"
    )


def test_inference_error_does_not_collapse_into_internal_error() -> None:
    assert issubclass(InferenceError, InternalError)
    assert (
        envelope_from_exception(
            InferenceError(SENTINEL), stage=ErrorStage.PROVIDER
        ).error_class
        is ErrorClass.PROVIDER
    )


def test_every_code_belongs_to_exactly_one_class() -> None:
    codes = [code for codes in ERROR_CODES.values() for code in codes]
    assert len(codes) == len(set(codes))
    assert set(ERROR_CODES) == set(ErrorClass)
    for error_class, class_codes in ERROR_CODES.items():
        assert class_codes == tuple(sorted(class_codes))
        for code in class_codes:
            assert error_class_for_code(code) is error_class
            assert is_retryable(code) is (code in RETRYABLE_ERROR_CODES)


def test_every_retryable_code_is_a_known_code() -> None:
    known = {code for codes in ERROR_CODES.values() for code in codes}
    assert RETRYABLE_ERROR_CODES.issubset(known)


@pytest.mark.parametrize("error_class", list(ErrorClass))
def test_every_class_has_at_least_one_valid_stage(error_class) -> None:
    code = ERROR_CODES[error_class][0]
    accepted = [stage for stage in ErrorStage if _accepts(code, stage)]
    assert accepted, error_class
    for stage in accepted:
        envelope = ErrorEnvelope.for_code(code, stage=stage)
        assert envelope.stage is stage


def _accepts(code: str, stage: ErrorStage) -> bool:
    try:
        ErrorEnvelope.for_code(code, stage=stage)
    except ErrorEnvelopeError:
        return False
    return True


def test_internal_errors_are_valid_at_every_stage() -> None:
    for stage in ErrorStage:
        assert ErrorEnvelope.for_code("internal_error", stage=stage).stage is stage


def test_authorization_errors_are_refused_at_unrelated_stages() -> None:
    for stage in (ErrorStage.VALIDATION, ErrorStage.PLANNING, ErrorStage.PROVIDER):
        with pytest.raises(
            ErrorEnvelopeError, match="^stage: invalid_stage_for_class$"
        ):
            ErrorEnvelope.for_code("capability_expired", stage=stage)


def test_a_mapped_class_with_an_invalid_stage_fails_closed() -> None:
    with pytest.raises(ErrorEnvelopeError, match="^stage: invalid_stage_for_class$"):
        envelope_from_exception(InputError(SENTINEL), stage=ErrorStage.PROVIDER)


def test_retryability_is_derived_and_cannot_be_overridden() -> None:
    envelope = ErrorEnvelope.for_code("request_timeout", stage=ErrorStage.TOOL_CALL)
    assert envelope.retryable is True
    with pytest.raises(AttributeError):
        envelope.retryable = False  # type: ignore[misc]
    assert (
        ErrorEnvelope.for_code("invalid_input", stage=ErrorStage.VALIDATION).retryable
        is False
    )


def test_correlation_identifiers_are_optional_and_validated() -> None:
    envelope = ErrorEnvelope.for_code(
        "provider_failed",
        stage=ErrorStage.PROVIDER,
        run_id=RUN_ID,
        action_id=ACTION_ID,
    )
    assert (envelope.run_id, envelope.action_id) == (RUN_ID, ACTION_ID)
    bare = ErrorEnvelope.for_code("provider_failed", stage=ErrorStage.PROVIDER)
    assert (bare.run_id, bare.action_id) == (None, None)


@pytest.mark.parametrize("value", ["", "run_x", ACTION_ID, 7, SENTINEL])
def test_invalid_run_identifiers_fail_closed(value) -> None:
    with pytest.raises(ErrorEnvelopeError) as excinfo:
        ErrorEnvelope.for_code(
            "internal_error", stage=ErrorStage.PLANNING, run_id=value
        )
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "invalid_identifier",
        "run_id",
    )
    assert SENTINEL not in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "act_x", RUN_ID, 7])
def test_invalid_action_identifiers_fail_closed(value) -> None:
    with pytest.raises(ErrorEnvelopeError, match="^action_id: invalid_identifier$"):
        ErrorEnvelope.for_code(
            "internal_error", stage=ErrorStage.PLANNING, action_id=value
        )


@pytest.mark.parametrize("code", ["", "nope", "INVALID_INPUT", 7, None])
def test_unknown_error_codes_fail_closed(code) -> None:
    with pytest.raises(ErrorEnvelopeError, match="^error_code: unknown_error_code$"):
        error_class_for_code(code)
    with pytest.raises(ErrorEnvelopeError, match="^error_code: unknown_error_code$"):
        ErrorEnvelope.for_code(code, stage=ErrorStage.PLANNING)


def test_a_code_from_another_class_fails_closed() -> None:
    with pytest.raises(ErrorEnvelopeError, match="^error_code: unknown_error_code$"):
        ErrorEnvelope(
            error_class=ErrorClass.VALIDATION,
            error_code="provider_failed",
            stage=ErrorStage.VALIDATION,
        )


@pytest.mark.parametrize("value", ["validation", 7, None])
def test_unknown_error_classes_fail_closed(value) -> None:
    with pytest.raises(ErrorEnvelopeError, match="^error_class: unknown_error_class$"):
        ErrorEnvelope(
            error_class=value, error_code="internal_error", stage=ErrorStage.PLANNING
        )


@pytest.mark.parametrize("value", ["tool_call", 7, None])
def test_unknown_stages_fail_closed(value) -> None:
    with pytest.raises(ErrorEnvelopeError, match="^stage: unknown_stage$"):
        ErrorEnvelope(
            error_class=ErrorClass.INTERNAL,
            error_code="internal_error",
            stage=value,
        )


def test_invalid_schema_version_fails_closed() -> None:
    with pytest.raises(ErrorEnvelopeError, match="^schema_version: "):
        ErrorEnvelope(
            error_class=ErrorClass.INTERNAL,
            error_code="internal_error",
            stage=ErrorStage.PLANNING,
            schema_version="openmed.agent.error.v0",
        )


@pytest.mark.parametrize("value", ["boom", 7, None, ValueError])
def test_non_exceptions_fail_closed(value) -> None:
    with pytest.raises(ErrorEnvelopeError, match="^exception: invalid_exception$"):
        envelope_from_exception(value, stage=ErrorStage.TOOL_CALL)


def test_serialization_is_byte_stable_and_field_ordered() -> None:
    envelope = ErrorEnvelope.for_code(
        "provider_unavailable", stage=ErrorStage.PROVIDER, run_id=RUN_ID
    )
    assert list(envelope.to_dict()) == [
        "schema_version",
        "error_class",
        "error_code",
        "stage",
        "retryable",
        "run_id",
        "action_id",
    ]
    assert json.loads(envelope.to_json()) == envelope.to_dict()
    assert (
        envelope.to_json()
        == ErrorEnvelope.for_code(
            "provider_unavailable", stage=ErrorStage.PROVIDER, run_id=RUN_ID
        ).to_json()
    )


def test_envelopes_have_no_free_text_detail_field() -> None:
    payload = json.loads(
        ErrorEnvelope.for_code("internal_error", stage=ErrorStage.PLANNING).to_json()
    )
    assert set(payload) == {
        "schema_version",
        "error_class",
        "error_code",
        "stage",
        "retryable",
        "run_id",
        "action_id",
    }
    assert not {"detail", "message", "reason", "traceback"} & set(payload)


def test_exception_text_and_arguments_never_reach_the_envelope() -> None:
    class Chatty(RuntimeError):
        def __str__(self) -> str:  # pragma: no cover - must never be called
            raise AssertionError("the envelope must not stringify the exception")

    envelope = envelope_from_exception(
        Chatty(SENTINEL), stage=ErrorStage.TOOL_CALL, run_id=RUN_ID
    )
    payload = envelope.to_json()
    assert SENTINEL not in payload
    assert "Chatty" not in payload
    assert envelope.error_code == "internal_error"


def test_envelopes_are_immutable() -> None:
    from dataclasses import FrozenInstanceError

    envelope = ErrorEnvelope.for_code("internal_error", stage=ErrorStage.PLANNING)
    with pytest.raises(FrozenInstanceError):
        envelope.error_code = "invalid_input"  # type: ignore[misc]


def test_error_envelopes_are_available_from_the_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.ErrorEnvelope is ErrorEnvelope
    assert agent.ErrorEnvelopeError is ErrorEnvelopeError
    assert agent.envelope_from_exception is envelope_from_exception
    exported = {
        "ERROR_CODES",
        "ERROR_ENVELOPE_SCHEMA_VERSION",
        "ErrorClass",
        "ErrorEnvelope",
        "ErrorEnvelopeError",
        "ErrorStage",
        "RETRYABLE_ERROR_CODES",
        "envelope_from_exception",
        "error_class_for_code",
        "is_retryable",
    }
    assert exported.issubset(set(agent.__all__))
