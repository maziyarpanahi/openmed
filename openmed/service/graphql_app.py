"""FastAPI integration for the OpenMed GraphQL schema."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from strawberry.fastapi import GraphQLRouter
from strawberry.http import GraphQLHTTPResponse
from strawberry.types import ExecutionResult

from .graphql_schema import SAFE_RESOLVER_ERROR, OpenMedGraphQLContext, schema
from .runtime import ServiceRuntime

GRAPHQL_PATH = "/graphql"


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
) -> None:
    """Mount the GraphQL endpoint on an existing OpenMed service app."""

    async def get_context(request: Request) -> OpenMedGraphQLContext:
        return OpenMedGraphQLContext(runtime_getter(request))

    router = PrivacySafeGraphQLRouter(
        schema,
        context_getter=get_context,
        graphql_ide="graphiql",
        allow_queries_via_get=True,
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
