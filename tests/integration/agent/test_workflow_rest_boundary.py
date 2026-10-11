"""Offline app-boundary checks; injected custody remains the authority owner."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from openmed.service.governed_workflows import MAX_WORKFLOW_REQUEST_BYTES
from openmed.service.workflow_routes import WorkflowBoundaryMiddleware
from tests.unit.service.test_workflow_routes import (
    CustodiedService,
    client,
    isolated_service_env,
    post,
    review_payload,
)

pytestmark = pytest.mark.integration
# The shared fixture configures only synthetic API keys and a fake model loader.
_service_fixture = isolated_service_env


def test_two_app_instances_delegate_duplicate_receipts_to_custody():
    service = CustodiedService()
    payload = review_payload(service)
    with client(service) as first, client(service) as second:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda http: post(http, "review-receipts", payload), [first, second]
                )
            )
    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()
    assert service.mutations == 1
    # The adapter never substitutes an in-process acknowledgement cache for the
    # custody service's atomic deduplication. Both requests reach its boundary.
    assert service.calls.count("submit") == 2
    with client(service) as reconstructed_app:
        resumed = post(reconstructed_app, "review-receipts", payload)
        assert resumed.status_code == 200
        assert resumed.json() == responses[0].json()
    assert service.mutations == 1
    service.execute.assert_not_called()
    service.compensate.assert_not_called()


@pytest.mark.parametrize("operation", ["preflight", "preview", "status"])
def test_inspection_remains_effect_free_after_receipt_submission(operation):
    service = CustodiedService()
    with client(service) as http:
        assert post(http, "review-receipts", review_payload(service)).status_code == 200
        result = post(http, operation)
        assert result.status_code == 200
        assert result.json()["committed_effect_count"] == 0
        assert result.json()["receipt_digest"] is not None
    service.execute.assert_not_called()
    service.compensate.assert_not_called()


@pytest.mark.parametrize("advertised_length", [None, b"1"])
def test_chunked_or_understated_body_is_rejected_before_shared_buffering(
    advertised_length,
):
    reached_inner = []
    sent = []
    chunks = iter(
        [
            {
                "type": "http.request",
                "body": b"x" * MAX_WORKFLOW_REQUEST_BYTES,
                "more_body": True,
            },
            {"type": "http.request", "body": b"x", "more_body": False},
        ]
    )

    async def inner(scope, receive, send):
        reached_inner.append(True)

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    headers = (
        [] if advertised_length is None else [(b"content-length", advertised_length)]
    )
    asyncio.run(
        WorkflowBoundaryMiddleware(inner)(
            {"type": "http", "path": "/v1/workflows/status", "headers": headers},
            receive,
            send,
        )
    )
    assert not reached_inner
    assert sent[0]["status"] == 413
    assert b"workflow_request_too_large" in sent[1]["body"]
