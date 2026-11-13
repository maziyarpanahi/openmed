"""FastAPI integration for the OpenMed GraphQL schema."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response
from strawberry.fastapi import GraphQLRouter
from strawberry.http import GraphQLHTTPResponse
from strawberry.types import ExecutionResult
from strawberry.types.unset import UNSET

from .graphql_limits import (
    GraphQLLimitCategory,
    GraphQLLimitError,
    GraphQLLimitViolation,
    batch_violation,
    deadline_violation,
    graphql_limits_for_request,
    graphql_limits_from_env,
    record_graphql_rejection,
    variable_depth_violation,
)
from .graphql_schema import SAFE_RESOLVER_ERROR, OpenMedGraphQLContext, schema
from .journey_resources import JourneyAccessPolicy, JourneyResourceCatalog
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
        if result.errors and any(
            error.extensions.get("code") == "OPENMED_FORBIDDEN"
            for error in result.errors
        ):
            return {
                "data": None,
                "errors": [
                    {
                        "message": "Operation is not permitted.",
                        "extensions": {"code": "OPENMED_FORBIDDEN"},
                    }
                ],
            }
        response: dict[str, Any] = {"data": result.data}
        if result.errors:
            response["errors"] = [
                _safe_formatted_error(error) for error in result.errors
            ]
        if result.extensions:
            response["extensions"] = result.extensions
        return response  # type: ignore[return-value]

    async def run(
        self,
        request: Any,
        context: Any = UNSET,
        root_value: Any = UNSET,
    ) -> Any:
        """Run one HTTP operation under the configured GraphQL deadline.

        The deadline covers body decoding, parsing, validation, resolver
        execution, and result processing. On expiry the operation is cancelled
        and a value-free ``OPENMED_GRAPHQL_TIMEOUT`` error is returned. Model
        work already running in a worker thread keeps its admission permit and
        warm-pool lease until the worker returns, then releases both.
        """
        if self.is_websocket_request(request):
            return await super().run(request, context=context, root_value=root_value)
        limits = graphql_limits_for_request(request)
        deadline = asyncio.timeout(limits.deadline_seconds)
        try:
            async with deadline:
                return await super().run(
                    request, context=context, root_value=root_value
                )
        except GraphQLLimitError as error:
            return _rejection_response(request, error.violation)
        except RecursionError:
            # Only the JSON body decoder recurses on client input before the
            # document budgets run; report it as nested variables.
            return _rejection_response(request, variable_depth_violation(limits))
        except TimeoutError:
            if not deadline.expired():
                raise
            return _rejection_response(request, deadline_violation(limits))

    def _validate_batch_request(self, request_data: Any, protocol: Any) -> None:
        """Reject every batched operation array with a stable error code."""
        raise GraphQLLimitError(batch_violation())


def mount_graphql(
    app: FastAPI,
    *,
    runtime_getter: Callable[[Request], ServiceRuntime],
    resource_getter: Callable[[Request], JourneyResourceCatalog],
) -> None:
    """Mount the GraphQL endpoint on an existing OpenMed service app."""

    async def get_context(request: Request) -> OpenMedGraphQLContext:
        policy = getattr(request.app.state, "journey_access_policy", None)
        if not isinstance(policy, JourneyAccessPolicy):
            policy = JourneyAccessPolicy()
        return OpenMedGraphQLContext(
            runtime_getter(request),
            resource_getter(request),
            policy,
        )

    router = PrivacySafeGraphQLRouter(
        schema,
        context_getter=get_context,
        graphql_ide="graphiql",
        allow_queries_via_get=True,
        subscription_protocols=(),
        multipart_uploads_enabled=False,
    )
    app.include_router(router, prefix=GRAPHQL_PATH, include_in_schema=False)
    app.state.graphql_limits = graphql_limits_from_env()


def _rejection_response(request: Request, violation: GraphQLLimitViolation) -> Response:
    record_graphql_rejection(request, violation)
    error = violation.to_response_error()
    if violation.category is GraphQLLimitCategory.BATCH:
        # A batch is not a single well-formed GraphQL request, so no data key.
        return JSONResponse({"errors": [error]}, status_code=400)
    return JSONResponse({"data": None, "errors": [error]})


def _safe_formatted_error(error: Any) -> dict[str, Any]:
    formatted = dict(error.formatted)
    if error.extensions.get("code") == "model_not_served":
        return {
            "message": "Requested model is not served",
            "extensions": {"code": "model_not_served"},
        }
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
