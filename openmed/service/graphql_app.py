"""FastAPI integration for the OpenMed GraphQL schema."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from strawberry.fastapi import GraphQLRouter
from strawberry.http import GraphQLHTTPResponse
from strawberry.http.ides import GraphQL_IDE
from strawberry.types import ExecutionResult

from openmed.core.config import PROFILE_ENV_VAR

from .graphql_schema import SAFE_RESOLVER_ERROR, OpenMedGraphQLContext, schema
from .journey_resources import JourneyAccessPolicy, JourneyResourceCatalog
from .runtime import ServiceRuntime

GRAPHQL_PATH = "/graphql"
DEFAULT_PROFILE = "prod"
GRAPHQL_IDE_PROFILE = "dev"
GRAPHQL_PRIVACY_HEADERS = (
    ("cache-control", "no-store"),
    ("pragma", "no-cache"),
    ("referrer-policy", "no-referrer"),
)


def graphql_ide_for_profile(profile: str) -> GraphQL_IDE | None:
    """Return the interactive IDE only for an explicit development profile."""

    return "graphiql" if profile == GRAPHQL_IDE_PROFILE else None


class GraphQLPrivacyHeadersMiddleware:
    """Attach no-store privacy headers to every GraphQL transport response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_graphql_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        async def send_with_privacy_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in GRAPHQL_PRIVACY_HEADERS:
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_privacy_headers)


def _is_graphql_path(path: str) -> bool:
    return path == GRAPHQL_PATH or path.startswith(f"{GRAPHQL_PATH}/")


class PrivacySafeGraphQLRouter(GraphQLRouter):
    """GraphQL router that masks all resolver execution failures."""

    async def process_result(
        self,
        request: Request,
        result: ExecutionResult,
    ) -> GraphQLHTTPResponse:
        """Return GraphQL data while removing exception-derived messages."""
        response: dict[str, Any] = {"data": result.data}
        if result.errors:
            response["errors"] = [
                _safe_formatted_error(error) for error in result.errors
            ]
        if result.extensions:
            response["extensions"] = result.extensions
        return response  # type: ignore[return-value]


def mount_graphql(
    app: FastAPI,
    *,
    runtime_getter: Callable[[Request], ServiceRuntime],
    resource_getter: Callable[[Request], JourneyResourceCatalog],
) -> None:
    """Mount the POST-only GraphQL endpoint on an existing OpenMed service app."""

    async def get_context(request: Request) -> OpenMedGraphQLContext:
        policy = getattr(request.app.state, "journey_access_policy", None)
        if not isinstance(policy, JourneyAccessPolicy):
            policy = JourneyAccessPolicy()
        return OpenMedGraphQLContext(
            runtime_getter(request),
            resource_getter(request),
            policy,
        )

    profile = os.getenv(PROFILE_ENV_VAR, DEFAULT_PROFILE)
    router = PrivacySafeGraphQLRouter(
        schema,
        context_getter=get_context,
        graphql_ide=graphql_ide_for_profile(profile),
        allow_queries_via_get=False,
        subscription_protocols=(),
        multipart_uploads_enabled=False,
    )
    app.include_router(router, prefix=GRAPHQL_PATH, include_in_schema=False)


def _safe_formatted_error(error: Any) -> dict[str, Any]:
    formatted = dict(error.formatted)
    if error.original_error is None:
        return formatted

    safe_error: dict[str, Any] = {
        "message": SAFE_RESOLVER_ERROR,
        "extensions": {"code": "OPENMED_RESOLVER_ERROR"},
    }
    if formatted.get("locations") is not None:
        safe_error["locations"] = formatted["locations"]
    if formatted.get("path") is not None:
        safe_error["path"] = formatted["path"]
    return safe_error
