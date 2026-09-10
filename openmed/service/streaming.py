"""NDJSON streaming helpers for REST de-identification responses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from pydantic import Field
from starlette.concurrency import run_in_threadpool

from openmed.core.policy import canonical_policy_name, load_policy
from openmed.core.streaming import deidentify_stream

from .runtime import ServiceRuntime
from .schemas import PIIDeidentifyRequest

DEFAULT_DEIDENTIFY_STREAM_CHUNK_SIZE = 1024
MAX_DEIDENTIFY_STREAM_CHUNK_SIZE = 32768
DEFAULT_DEIDENTIFY_STREAM_MAX_BUFFER = 4096

_END_OF_STREAM = object()


class PIIDeidentifyStreamRequest(PIIDeidentifyRequest):
    """Request schema for ``/pii/deidentify/stream``."""

    chunk_size: int = Field(
        default=DEFAULT_DEIDENTIFY_STREAM_CHUNK_SIZE,
        ge=1,
        le=MAX_DEIDENTIFY_STREAM_CHUNK_SIZE,
    )


def deidentify_ndjson_stream(
    payload: PIIDeidentifyStreamRequest,
    runtime: ServiceRuntime,
) -> AsyncIterator[str]:
    """Return an async NDJSON iterator backed by core streaming de-identification.

    Policy resolution happens before the response starts so invalid policies keep
    the REST service's normal validation/error-envelope behavior.
    """

    policy_name = canonical_policy_name(payload.policy) if payload.policy else None
    policy_profile = load_policy(policy_name) if policy_name is not None else None
    keep_mapping = payload.keep_mapping or bool(
        policy_profile is not None and policy_profile.keep_mapping
    )
    return _iterate_deidentify_ndjson(
        payload,
        runtime,
        policy_name=policy_name,
        keep_mapping=keep_mapping,
    )


async def _iterate_deidentify_ndjson(
    payload: PIIDeidentifyStreamRequest,
    runtime: ServiceRuntime,
    *,
    policy_name: str | None,
    keep_mapping: bool,
) -> AsyncIterator[str]:
    model_key: str | None = None
    error: BaseException | None = None
    iterator: Iterator[Any] | None = None
    chunk_index = 0
    timeout_seconds = float(getattr(runtime.config, "timeout", 0) or 0)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds if timeout_seconds > 0 else None

    # Reversible placeholders must be numbered across the whole document, just
    # as they are for the single-shot endpoint. Keeping that opt-in request in
    # one core window preserves its exact mapping semantics.
    max_buffer = DEFAULT_DEIDENTIFY_STREAM_MAX_BUFFER
    if keep_mapping:
        max_buffer = max(max_buffer, len(payload.text))

    try:
        model_key = runtime.begin_model_request(payload.model_name)
        iterator = iter(
            deidentify_stream(
                _text_chunks(payload.text, payload.chunk_size),
                max_buffer=max_buffer,
                method=payload.method,
                model_name=payload.model_name,
                confidence_threshold=payload.confidence_threshold,
                keep_year=payload.keep_year,
                shift_dates=payload.shift_dates,
                date_shift_days=payload.date_shift_days,
                keep_mapping=keep_mapping,
                config=runtime.config,
                use_smart_merging=payload.use_smart_merging,
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                use_safety_sweep=payload.use_safety_sweep,
                loader=runtime.get_loader(),
                policy=policy_name,
            )
        )

        while True:
            event = await _next_with_deadline(iterator, deadline)
            if event is _END_OF_STREAM:
                break
            if bool(getattr(event, "final", False)):
                final_payload: dict[str, Any] = {
                    "type": "final",
                    "audit": dict(getattr(event, "audit_record", None) or {}),
                    "spans": [
                        _span_to_dict(span)
                        for span in tuple(getattr(event, "spans", ()) or ())
                    ],
                }
                if keep_mapping:
                    mapping = dict(getattr(event, "mapping", None) or {})
                    if mapping:
                        final_payload["mapping"] = mapping
                yield _ndjson(final_payload)
                continue

            redacted_text = str(getattr(event, "redacted_text", ""))
            if redacted_text:
                yield _ndjson(
                    {
                        "type": "chunk",
                        "index": chunk_index,
                        "redacted_text": redacted_text,
                    }
                )
                chunk_index += 1
    except asyncio.TimeoutError as exc:
        error = exc
        yield _ndjson(
            {
                "type": "error",
                "error": {
                    "code": "timeout",
                    "message": (
                        "Request exceeded configured timeout of "
                        f"{timeout_seconds:g} seconds"
                    ),
                    "details": {"timeout_seconds": timeout_seconds},
                },
            }
        )
    except asyncio.CancelledError as exc:
        error = exc
        raise
    except ValueError as exc:
        error = exc
        yield _ndjson(
            {
                "type": "error",
                "error": {
                    "code": "bad_request",
                    "message": "Streaming de-identification request failed",
                    "details": None,
                },
            }
        )
    except Exception as exc:
        error = exc
        yield _ndjson(
            {
                "type": "error",
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "details": None,
                },
            }
        )
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if model_key is not None:
            runtime.finish_model_request(model_key, payload.keep_alive, error)


async def _next_with_deadline(
    iterator: Iterator[Any],
    deadline: float | None,
) -> Any:
    operation = run_in_threadpool(_next_or_end, iterator)
    if deadline is None:
        return await operation

    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return await asyncio.wait_for(operation, timeout=remaining)


def _next_or_end(iterator: Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _END_OF_STREAM


def _text_chunks(text: str, chunk_size: int) -> Iterator[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def _span_to_dict(span: Any) -> Mapping[str, Any]:
    to_dict = getattr(span, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(span, Mapping):
        return dict(span)
    raise TypeError("Streaming span must be a mapping or expose to_dict()")


def _ndjson(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


__all__ = [
    "DEFAULT_DEIDENTIFY_STREAM_CHUNK_SIZE",
    "PIIDeidentifyStreamRequest",
    "deidentify_ndjson_stream",
]
