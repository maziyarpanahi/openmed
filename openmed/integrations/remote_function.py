"""BigQuery-compatible remote-function handler for batch de-identification.

The HTTP surface accepts BigQuery's batched ``calls`` request envelope and
returns a same-length ``replies`` array.  Request metadata and input text are
never logged.  Errors contain only protocol fields, row positions, or generic
processing state so exception messages cannot disclose source text.

FastAPI belongs to OpenMed's optional ``service`` extra.  Import this module
only in deployments that install ``openmed[service]``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from openmed.core.policy import canonical_policy_name
from openmed.processing.batch import process_batch

DEFAULT_REMOTE_FUNCTION_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
DEFAULT_REMOTE_FUNCTION_POLICY = "hipaa_safe_harbor"
POLICY_HEADER = "X-OpenMed-Policy"

_POLICY_ALIASES = {
    "hipaa": DEFAULT_REMOTE_FUNCTION_POLICY,
    "safe_harbor": DEFAULT_REMOTE_FUNCTION_POLICY,
}

ProcessBatch = Callable[..., Any]
RemoteReply = str | None


class RemoteFunctionRequestError(ValueError):
    """Raised when a remote-function request violates the JSON contract."""


class RemoteFunctionProcessingError(RuntimeError):
    """Raised when a valid remote-function batch cannot be redacted safely."""


@dataclass(frozen=True)
class _RemoteCall:
    index: int
    text: str | None
    policy: str


def redact_remote_function_batch(
    payload: Mapping[str, Any],
    *,
    request_policy: str | None = None,
    process_batch_fn: ProcessBatch | None = None,
) -> dict[str, list[RemoteReply]]:
    """Redact one BigQuery remote-function batch.

    Each element in ``payload["calls"]`` must contain a string-or-null text
    argument and may contain a policy argument.  A request-wide policy can be
    supplied by the HTTP adapter through ``request_policy``.  If both scopes
    select policies, they must agree; this prevents a row from weakening a
    policy enforced by a gateway or deployment.

    Non-empty strings with the same policy are sent through one
    :func:`openmed.processing.process_batch` invocation.  Mixed-policy batches
    are grouped without changing reply order.  SQL ``NULL`` and empty strings
    are preserved without loading a model.

    Args:
        payload: BigQuery request object containing a non-empty ``calls`` list.
        request_policy: Optional request-wide OpenMed policy profile.
        process_batch_fn: Optional batch implementation for embedded runtimes
            and offline tests.  Defaults to :func:`openmed.process_batch`.

    Returns:
        A BigQuery response object with one reply for every input call.

    Raises:
        RemoteFunctionRequestError: If the envelope, row arguments, or policy
            selection is invalid.
        RemoteFunctionProcessingError: If batch redaction fails or returns an
            invalid result.  The error text never contains source values.
    """

    if not isinstance(payload, Mapping):
        raise RemoteFunctionRequestError("request body must be a JSON object")

    calls_value = payload.get("calls")
    if not isinstance(calls_value, list) or not calls_value:
        raise RemoteFunctionRequestError("calls must be a non-empty array")

    envelope_policy = _context_policy(payload)
    normalized_request_policy = _coalesce_request_policies(
        request_policy,
        envelope_policy,
    )
    calls = _parse_calls(
        calls_value,
        request_policy=normalized_request_policy,
    )

    replies: list[RemoteReply] = [None] * len(calls)
    grouped_calls: dict[str, list[_RemoteCall]] = {}
    for call in calls:
        if call.text is None:
            replies[call.index] = None
        elif call.text == "":
            replies[call.index] = ""
        else:
            grouped_calls.setdefault(call.policy, []).append(call)

    if not grouped_calls:
        return {"replies": replies}

    batch_callable = (
        process_batch_fn if process_batch_fn is not None else _default_process_batch()
    )
    for policy, policy_calls in grouped_calls.items():
        redacted = _process_policy_group(
            policy_calls,
            policy=policy,
            process_batch_fn=batch_callable,
        )
        for call, value in zip(policy_calls, redacted, strict=True):
            replies[call.index] = value

    return {"replies": replies}


def create_app(*, process_batch_fn: ProcessBatch | None = None) -> FastAPI:
    """Create the warehouse remote-function HTTP application.

    The root ``POST`` route implements the BigQuery request/response protocol.
    Policy may be selected by the optional second SQL function argument, the
    ``X-OpenMed-Policy`` header, the HTTP ``policy`` query parameter, or
    BigQuery ``userDefinedContext.policy``.  Multiple request-wide selectors
    must resolve to the same canonical profile.

    All request and processing exceptions are converted into bounded
    ``{"errorMessage": ...}`` replies.  The adapter never logs request bodies,
    metadata, policy values, redacted output, or exception messages.

    Args:
        process_batch_fn: Optional batch implementation for embedded runtimes
            and offline tests.  Defaults to :func:`openmed.process_batch`.

    Returns:
        A configured FastAPI application.
    """

    application = FastAPI(
        title="OpenMed warehouse remote function",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/")
    async def remote_function(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return _error_response(
                "request body must be valid JSON",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            request_policy = _request_policy(request, payload)
            reply = redact_remote_function_batch(
                payload,
                request_policy=request_policy,
                process_batch_fn=process_batch_fn,
            )
            return JSONResponse(reply, status_code=status.HTTP_200_OK)
        except RemoteFunctionRequestError as exc:
            return _error_response(str(exc), status.HTTP_400_BAD_REQUEST)
        except RemoteFunctionProcessingError:
            return _error_response(
                "remote-function batch could not be de-identified",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            return _error_response(
                "remote-function request could not be processed",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    return application


def _request_policy(request: Request, payload: Any) -> str | None:
    query_values = request.query_params.getlist("policy")
    if len(query_values) > 1:
        raise RemoteFunctionRequestError(
            "policy query parameter must be provided at most once"
        )

    header_values = request.headers.getlist(POLICY_HEADER)
    if len(header_values) > 1:
        raise RemoteFunctionRequestError("policy header must be provided at most once")

    query_policy = query_values[0] if query_values else None
    header_policy = header_values[0] if header_values else None
    context_policy = _context_policy(payload) if isinstance(payload, Mapping) else None
    return _coalesce_request_policies(
        query_policy,
        header_policy,
        context_policy,
    )


def _context_policy(payload: Mapping[str, Any]) -> str | None:
    context = payload.get("userDefinedContext")
    if context is None:
        return None
    if not isinstance(context, Mapping):
        raise RemoteFunctionRequestError("userDefinedContext must be an object")
    return context.get("policy")


def _coalesce_request_policies(*values: Any) -> str | None:
    normalized = {
        _canonical_policy(value, location="request")
        for value in values
        if value is not None
    }
    if len(normalized) > 1:
        raise RemoteFunctionRequestError("request policy selections conflict")
    return next(iter(normalized), None)


def _parse_calls(
    raw_calls: Sequence[Any],
    *,
    request_policy: str | None,
) -> list[_RemoteCall]:
    calls: list[_RemoteCall] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, list) or len(raw_call) not in {1, 2}:
            raise RemoteFunctionRequestError(
                f"calls[{index}] must contain text and an optional policy"
            )

        text = raw_call[0]
        if text is not None and not isinstance(text, str):
            raise RemoteFunctionRequestError(
                f"calls[{index}] text must be a string or null"
            )

        row_policy = (
            _canonical_policy(raw_call[1], location=f"calls[{index}]")
            if len(raw_call) == 2 and raw_call[1] is not None
            else None
        )
        if (
            request_policy is not None
            and row_policy is not None
            and request_policy != row_policy
        ):
            raise RemoteFunctionRequestError(
                f"calls[{index}] policy conflicts with the request policy"
            )

        calls.append(
            _RemoteCall(
                index=index,
                text=text,
                policy=(request_policy or row_policy or DEFAULT_REMOTE_FUNCTION_POLICY),
            )
        )
    return calls


def _canonical_policy(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RemoteFunctionRequestError(
            f"{location} policy must be a non-empty string"
        )
    try:
        normalized = value.strip().lower().replace("-", "_")
        return canonical_policy_name(_POLICY_ALIASES.get(normalized, normalized))
    except (TypeError, ValueError):
        raise RemoteFunctionRequestError(
            f"{location} policy is not a supported profile"
        ) from None


def _process_policy_group(
    calls: Sequence[_RemoteCall],
    *,
    policy: str,
    process_batch_fn: ProcessBatch,
) -> list[str]:
    texts = [call.text for call in calls]
    if any(not isinstance(text, str) or not text for text in texts):
        raise RemoteFunctionProcessingError(
            "invalid internal remote-function row state"
        )

    try:
        batch_result = process_batch_fn(
            texts,
            model_name=DEFAULT_REMOTE_FUNCTION_MODEL,
            ids=[f"remote:{call.index}" for call in calls],
            operation="deidentify",
            batch_size=len(texts),
            method="mask",
            confidence_threshold=0.7,
            continue_on_error=False,
            use_safety_sweep=True,
            policy=policy,
        )
        return _extract_redacted_texts(batch_result, expected=len(texts))
    except RemoteFunctionProcessingError:
        raise
    except Exception:
        raise RemoteFunctionProcessingError(
            "remote-function batch processing failed"
        ) from None


def _extract_redacted_texts(batch_result: Any, *, expected: int) -> list[str]:
    if isinstance(batch_result, Mapping):
        raw_items = batch_result.get("items")
    else:
        raw_items = getattr(batch_result, "items", batch_result)

    if isinstance(raw_items, (str, bytes, Mapping)):
        raise RemoteFunctionProcessingError(
            "remote-function batch returned an invalid result"
        )
    try:
        items = list(raw_items)
    except TypeError:
        raise RemoteFunctionProcessingError(
            "remote-function batch returned an invalid result"
        ) from None

    if len(items) != expected:
        raise RemoteFunctionProcessingError(
            "remote-function batch returned an unexpected result count"
        )

    values: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            failed = item.get("success") is False or item.get("error") is not None
            result = item.get("result", item)
        else:
            failed = (
                getattr(item, "success", True) is False
                or getattr(item, "error", None) is not None
            )
            result = getattr(item, "result", item)
        if failed:
            raise RemoteFunctionProcessingError(
                f"remote-function batch failed at row offset {index}"
            )

        if isinstance(result, str):
            value = result
        elif isinstance(result, Mapping):
            value = result.get("deidentified_text")
        else:
            value = getattr(result, "deidentified_text", None)
        if not isinstance(value, str):
            raise RemoteFunctionProcessingError(
                "remote-function batch returned an invalid result "
                f"at row offset {index}"
            )
        values.append(value)
    return values


def _default_process_batch() -> ProcessBatch:
    return process_batch


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"errorMessage": message[:1023]},
        status_code=status_code,
    )


app = create_app()


__all__ = [
    "DEFAULT_REMOTE_FUNCTION_MODEL",
    "DEFAULT_REMOTE_FUNCTION_POLICY",
    "POLICY_HEADER",
    "RemoteFunctionProcessingError",
    "RemoteFunctionRequestError",
    "app",
    "create_app",
    "redact_remote_function_batch",
]
