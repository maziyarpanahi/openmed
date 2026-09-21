"""ASGI request-size enforcement shared by every REST and GraphQL route."""

from __future__ import annotations

import os
from typing import Final

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from openmed.guard.operational_limits import OperationalLimits

MAX_REQUEST_BYTES_ENV_VAR: Final = "OPENMED_SERVICE_MAX_REQUEST_BYTES"


def request_limits_from_env() -> OperationalLimits:
    """Load the bounded request-byte override from the environment."""

    default = OperationalLimits()
    raw = os.getenv(MAX_REQUEST_BYTES_ENV_VAR)
    if raw is None or not raw.strip():
        return default
    try:
        maximum = int(raw)
    except ValueError:
        raise ValueError(
            f"{MAX_REQUEST_BYTES_ENV_VAR} must be an integer number of bytes"
        ) from None
    return OperationalLimits(max_request_bytes=maximum)


class BoundedRequestBodyMiddleware:
    """Reject oversized bodies before application parsing or model work."""

    def __init__(self, app: ASGIApp, *, limits: OperationalLimits) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await _limit_response(400, "request_size_invalid")(scope, receive, send)
                return
            if declared < 0:
                await _limit_response(400, "request_size_invalid")(scope, receive, send)
                return
            if declared > self.limits.max_request_bytes:
                await _limit_response(413, "request_size_exceeded")(
                    scope, receive, send
                )
                return

        messages: list[Message] = []
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            received += len(message.get("body", b""))
            if received > self.limits.max_request_bytes:
                await _limit_response(413, "request_size_exceeded")(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay, send)


def _limit_response(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "details": None,
                "message": "Request rejected by an operational limit",
            }
        },
    )


__all__ = [
    "BoundedRequestBodyMiddleware",
    "MAX_REQUEST_BYTES_ENV_VAR",
    "request_limits_from_env",
]
