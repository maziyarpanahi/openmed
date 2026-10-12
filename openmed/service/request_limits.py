"""ASGI request-size and JSON-structure enforcement for REST and GraphQL routes.

Every ``POST``, ``PUT`` and ``PATCH`` body is bounded by ``max_request_bytes``.
JSON bodies are also streamed through :class:`JsonStructureScanner` while they
arrive, so depth, node and string budgets from :class:`OperationalLimits` are
enforced before any framework parser or Pydantic model sees the body.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from openmed.guard.operational_limits import (
    JSON_STRUCTURE_EXCEEDED,
    JsonStructureLimitError,
    JsonStructureScanner,
    OperationalLimits,
)

MAX_REQUEST_BYTES_ENV_VAR: Final = "OPENMED_SERVICE_MAX_REQUEST_BYTES"
JSON_STRUCTURE_LIMIT_FIELDS: Final = frozenset(
    {"max_json_depth", "max_json_nodes", "max_json_string_bytes"}
)
# Documented per-route JSON ceilings that replace the global values for one
# exact request path. ``/cohort/resolve`` accepts caller-supplied concept
# hierarchies; each ``concept_ancestors`` edge is three JSON values, so the
# default 100,000-value budget would cap a hierarchy near 33,000 edges.
JSON_STRUCTURE_ROUTE_OVERRIDES: Final[Mapping[str, Mapping[str, int]]] = (
    MappingProxyType({"/cohort/resolve": MappingProxyType({"max_json_nodes": 250_000})})
)


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
    """Reject oversized or over-structured bodies before parsing or model work."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limits: OperationalLimits,
        json_route_overrides: Mapping[str, Mapping[str, int]] | None = None,
    ) -> None:
        """Wrap an ASGI app with request-body limits.

        Args:
            app: The downstream ASGI application.
            limits: Global request-byte and JSON-structure budgets.
            json_route_overrides: Exact request paths mapped to replacement
                ``max_json_depth``, ``max_json_nodes`` or
                ``max_json_string_bytes`` values. Defaults to
                ``JSON_STRUCTURE_ROUTE_OVERRIDES``.

        Raises:
            ValueError: An override names another field or leaves the safe
                range enforced by ``OperationalLimits``.
        """

        self.app = app
        self.limits = limits
        overrides = (
            JSON_STRUCTURE_ROUTE_OVERRIDES
            if json_route_overrides is None
            else json_route_overrides
        )
        self.json_route_limits = {
            str(path): _route_limits(limits, fields)
            for path, fields in overrides.items()
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
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

        scanner = (
            JsonStructureScanner(
                self.json_route_limits.get(scope.get("path", ""), self.limits)
            )
            if _is_json_body(headers.get("content-type"))
            else None
        )
        messages: deque[Message] = deque()
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            received += len(body)
            if received > self.limits.max_request_bytes:
                await _limit_response(413, "request_size_exceeded")(
                    scope, receive, send
                )
                return
            more_body = message.get("more_body", False)
            if scanner is not None:
                try:
                    scanner.feed(body)
                    if not more_body:
                        scanner.finish()
                except JsonStructureLimitError as error:
                    await _limit_response(
                        413, JSON_STRUCTURE_EXCEEDED, details=error.to_dict()
                    )(scope, receive, send)
                    return
            if not more_body:
                break

        async def replay() -> Message:
            if messages:
                return messages.popleft()
            return await receive()

        await self.app(scope, replay, send)


def _route_limits(
    limits: OperationalLimits, fields: Mapping[str, int]
) -> OperationalLimits:
    unknown = set(fields) - JSON_STRUCTURE_LIMIT_FIELDS
    if unknown:
        raise ValueError("route overrides may only set JSON structure limits")
    return replace(limits, **dict(fields))


def _is_json_body(content_type: str | None) -> bool:
    """Match the media types FastAPI decodes as JSON, plus legacy FHIR JSON."""

    if content_type is None or not content_type.strip():
        return True
    media_type = content_type.split(";", 1)[0].strip().lower()
    maintype, _, subtype = media_type.partition("/")
    return maintype == "application" and (
        subtype == "json" or subtype.endswith("+json") or subtype.startswith("json+")
    )


def _limit_response(
    status_code: int,
    code: str,
    *,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "details": details,
                "message": "Request rejected by an operational limit",
            }
        },
    )


__all__ = [
    "BoundedRequestBodyMiddleware",
    "JSON_STRUCTURE_LIMIT_FIELDS",
    "JSON_STRUCTURE_ROUTE_OVERRIDES",
    "MAX_REQUEST_BYTES_ENV_VAR",
    "request_limits_from_env",
]
