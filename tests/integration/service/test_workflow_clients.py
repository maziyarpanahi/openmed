"""Offline client lifecycle using frozen metadata and an injected transport."""

import json
from pathlib import Path

import httpx
import pytest

from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.service.client import OpenMedClient, WorkflowPollPolicy, WorkflowReference


@pytest.mark.integration
def test_inspection_never_implicitly_submits_review_or_cancel():
    vectors = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "tests/fixtures/service/workflow-clients-v1.json"
        ).read_text()
    )
    request = json.loads(vectors["receipt_request_json"])
    receipt = ApprovalReceipt.from_dict(request.pop("receipt"))
    reference = WorkflowReference.from_dict(request)
    calls = []

    def transport(incoming):
        calls.append(incoming)
        response = json.loads(vectors["receipt_response_json"])
        if incoming.url.path.endswith("/cancel"):
            response["cancellation_requested"] = True
        return httpx.Response(
            200,
            content=json.dumps(response),
            headers={"Content-Type": "application/json"},
        )

    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        client.workflow_preflight(reference)
        client.workflow_preview(reference)
        assert (
            client.poll_workflow(
                reference, policy=WorkflowPollPolicy(interval_seconds=0)
            ).phase.value
            == "waiting-review"
        )
        assert [v.url.path for v in calls] == [
            "/v1/workflows/preflight",
            "/v1/workflows/preview",
            "/v1/workflows/status",
        ]
        client.workflow_submit_receipt(reference, receipt)
        client.workflow_cancel(reference)
    assert [v.url.path for v in calls[-2:]] == [
        "/v1/workflows/review-receipts",
        "/v1/workflows/cancel",
    ]
    assert json.loads(calls[-2].content)["receipt"] == receipt.to_dict()
