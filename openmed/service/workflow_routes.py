"""Authenticated, bounded REST adapters for an injected governance service."""

from __future__ import annotations

import asyncio
import hmac
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt

from .auth import AuthPrincipal, scopes_satisfy
from .governed_workflows import (
    MAX_WORKFLOW_REQUEST_BYTES,
    WORKFLOW_ERROR_STATUSES,
    WorkflowGovernanceService,
    WorkflowHTTPPolicy,
    WorkflowReceiptVerification,
    WorkflowReference,
    WorkflowServiceError,
    WorkflowView,
    parse_workflow_json,
    validate_workflow_receipt,
    validate_workflow_view,
)
from .logging import current_request_id
from .schemas import workflow_http_schemas

WORKFLOW_ROUTE_SCOPES = {
    "/v1/workflows/preflight": "workflow:read",
    "/v1/workflows/preview": "workflow:read",
    "/v1/workflows/status": "workflow:read",
    "/v1/workflows/review-receipts": "workflow:review",
    "/v1/workflows/cancel": "workflow:cancel",
}
_SAFE_CORRELATION = re.compile(
    r"(?:req_[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})\Z"
)


def _error(code: str, *, request_id: str | None = None) -> JSONResponse:
    error = WorkflowServiceError(code)
    request_id = request_id or current_request_id()
    headers = {"Cache-Control": "no-store"}
    if error.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer, ApiKey"
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    payload = {"code": error.code, "message": str(error), "details": []}
    if request_id:
        payload["request_id"] = request_id
    return JSONResponse(
        {"error": payload},
        status_code=error.status_code,
        headers=headers,
    )


class WorkflowBoundaryMiddleware:
    """Bound workflow bodies before shared buffering and sanitize log IDs.

    This limit also applies to chunked bodies and misleading Content-Length.
    The request ID is metadata only; it is never mutation idempotency authority.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            "/v1/workflows/"
        ):
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        ids = [value for key, value in headers if key.lower() == b"x-request-id"]
        request_id = str(uuid.uuid4())
        if len(ids) == 1:
            candidate = ids[0].decode("ascii", errors="replace")
            if _SAFE_CORRELATION.fullmatch(candidate):
                request_id = candidate
        scope = dict(scope)
        scope["headers"] = [
            (key, value) for key, value in headers if key.lower() != b"x-request-id"
        ] + [(b"x-request-id", request_id.encode("ascii"))]
        if scope["path"] not in WORKFLOW_ROUTE_SCOPES:
            await self.app(scope, receive, send)
            return

        lengths = [v for k, v in headers if k.lower() == b"content-length"]
        if lengths and (
            len(lengths) != 1 or re.fullmatch(rb"[0-9]{1,20}", lengths[0]) is None
        ):
            await _error("workflow_invalid_input", request_id=request_id)(
                scope, receive, send
            )
            return
        if lengths and int(lengths[0]) > MAX_WORKFLOW_REQUEST_BYTES:
            await _error("workflow_request_too_large", request_id=request_id)(
                scope, receive, send
            )
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_WORKFLOW_REQUEST_BYTES:
                await _error("workflow_request_too_large", request_id=request_id)(
                    scope, receive, send
                )
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def mount_workflow_routes(
    app: FastAPI,
    *,
    service: WorkflowGovernanceService | None,
    policy: WorkflowHTTPPolicy | None,
    clock: Callable[[], int] | None = None,
) -> None:
    """Mount metadata-only routes with explicit server policy and existing auth.

    Args:
        app: Existing REST app, including authentication and access logging.
        service: Trusted local custody implementation; no default dispatcher.
        policy: Explicit server-owned route and mutation opt-ins.
        clock: Trusted integer Unix-seconds source for receipt validity checks.
    """
    settings = WorkflowHTTPPolicy() if policy is None else replace(policy)
    read_clock = clock if clock is not None else lambda: int(time.time())
    if not callable(read_clock):
        raise WorkflowServiceError("workflow_invalid_input")
    schemas = workflow_http_schemas()
    router = APIRouter(tags=["Governed workflows"])

    async def handle(request: Request, operation: str) -> JSONResponse:
        mutation = operation in ("review-receipts", "cancel")
        try:
            principal = request.scope.get("openmed.auth")
            if type(principal) is not AuthPrincipal:
                raise WorkflowServiceError("workflow_authentication_required")
            if not scopes_satisfy(
                principal.scopes, (WORKFLOW_ROUTE_SCOPES[request.url.path],)
            ):
                raise WorkflowServiceError("workflow_forbidden")
            if not settings.enabled or service is None:
                raise WorkflowServiceError("workflow_disabled")
            if (
                operation == "review-receipts" and not settings.allow_review_submissions
            ) or (operation == "cancel" and not settings.allow_cancellation):
                raise WorkflowServiceError("workflow_forbidden")
            if (
                request.scope.get("query_string")
                or request.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
                != "application/json"
            ):
                raise WorkflowServiceError("workflow_invalid_input")
            payload = parse_workflow_json(await request.body())
            receipt = None
            if operation == "review-receipts":
                if "receipt" not in payload:
                    raise WorkflowServiceError("workflow_invalid_input")
                candidate = payload.pop("receipt")
                try:
                    receipt = ApprovalReceipt.from_dict(candidate)
                except (TypeError, ValueError):
                    raise WorkflowServiceError("workflow_invalid_input") from None
            reference = WorkflowReference.from_dict(payload)
            if mutation:
                reference.require_mutation()

            def dispatch() -> WorkflowView:
                if operation != "review-receipts":
                    method = getattr(service, operation)
                    return method(principal, reference)
                if receipt is None:
                    raise WorkflowServiceError("workflow_invalid_input")
                now = read_clock()
                receipt_digest = validate_workflow_receipt(reference, receipt, now=now)
                proof = service.verify_receipt(principal, reference, receipt, now=now)
                try:
                    if type(proof) is not WorkflowReceiptVerification:
                        raise ValueError()
                    proof = replace(proof)
                    if proof.reference != reference or not hmac.compare_digest(
                        proof.receipt_digest, receipt_digest
                    ):
                        raise ValueError()
                except (TypeError, ValueError):
                    raise WorkflowServiceError("workflow_receipt_unverified") from None
                if not proof.verified:
                    raise WorkflowServiceError("workflow_receipt_unverified")
                # A read-only proof is insufficient for mutation. The service
                # rechecks custody, time and state atomically at commit.
                commit_now = read_clock()
                validate_workflow_receipt(reference, receipt, now=commit_now)
                result = validate_workflow_view(
                    service.submit_receipt(
                        principal, reference, receipt, now=commit_now
                    ),
                    reference,
                )
                if result.receipt_digest is None or not hmac.compare_digest(
                    result.receipt_digest, receipt_digest
                ):
                    raise WorkflowServiceError("workflow_invalid_result")
                return result

            try:
                result = await asyncio.wait_for(
                    run_in_threadpool(dispatch), timeout=settings.timeout_seconds
                )
            except asyncio.TimeoutError:
                # Cancelling an await cannot undo a worker's durable mutation.
                raise WorkflowServiceError(
                    "workflow_mutation_unknown" if mutation else "workflow_unavailable"
                ) from None
            except WorkflowServiceError:
                raise
            except Exception:
                raise WorkflowServiceError(
                    "workflow_mutation_unknown"
                    if mutation
                    else "workflow_service_failed"
                ) from None
            result = validate_workflow_view(result, reference)
            if operation == "cancel" and not (
                result.cancellation_requested or result.phase is ActionPhase.ABORTED
            ):
                raise WorkflowServiceError("workflow_invalid_result")
            return JSONResponse(result.to_dict(), headers={"Cache-Control": "no-store"})
        except WorkflowServiceError as error:
            code = (
                error.code
                if type(error) is WorkflowServiceError
                and type(error.code) is str
                and error.code in WORKFLOW_ERROR_STATUSES
                else "workflow_mutation_unknown"
                if mutation
                else "workflow_service_failed"
            )
            return _error(code)
        except Exception:
            return _error("workflow_service_failed")

    for path, scope_name in WORKFLOW_ROUTE_SCOPES.items():
        operation = path.rsplit("/", 1)[1]

        def endpoint_for(
            selected_operation: str,
        ) -> Callable[[Request], Awaitable[JSONResponse]]:
            async def endpoint(request: Request) -> JSONResponse:
                return await handle(request, selected_operation)

            return endpoint

        schema_name = (
            "review"
            if operation == "review-receipts"
            else "mutation"
            if operation == "cancel"
            else "reference"
        )
        router.add_api_route(
            path,
            endpoint_for(operation),
            methods=["POST"],
            name="workflow_" + operation.replace("-", "_"),
            response_model=None,
            openapi_extra={
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/GovernedWorkflow"
                                + schema_name.title()
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Bounded custody snapshot; no effect execution.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/GovernedWorkflowView"
                                }
                            }
                        },
                    },
                    **{
                        str(status): {
                            "description": "Fixed workflow or existing authentication diagnostic.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/GovernedWorkflowError"
                                    }
                                }
                            },
                        }
                        for status in set(WORKFLOW_ERROR_STATUSES.values())
                    },
                },
                "x-required-scope": scope_name,
                "x-max-request-bytes": MAX_WORKFLOW_REQUEST_BYTES,
                "security": [{"WorkflowApiKey": []}, {"WorkflowBearer": []}],
            },
        )
    app.include_router(router)
    previous_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        spec = previous_openapi()
        spec.setdefault("components", {}).setdefault("schemas", {}).update(
            {
                "GovernedWorkflow" + name.title(): schema
                for name, schema in schemas.items()
            }
        )
        spec.setdefault("components", {}).setdefault("securitySchemes", {}).update(
            {
                "WorkflowApiKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
                "WorkflowBearer": {"type": "http", "scheme": "bearer"},
            }
        )
        return spec

    app.openapi = openapi  # type: ignore[method-assign]
