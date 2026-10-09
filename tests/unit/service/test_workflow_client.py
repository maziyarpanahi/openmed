"""Offline controls for typed, content-free governed workflow consumers."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.service.client import (
    OpenMedClient,
    WorkflowClientError,
    WorkflowPollPolicy,
    WorkflowReadOptions,
    WorkflowReference,
)
from openmed.service.workflow_client import parse_workflow_response

ROOT = Path(__file__).resolve().parents[3]
VECTORS = json.loads(
    (ROOT / "tests/fixtures/service/workflow-clients-v1.json").read_text()
)
CASES = VECTORS["cases"]
REFERENCE = WorkflowReference.from_dict(CASES[0]["reference"])
REVIEW_REQUEST = json.loads(VECTORS["receipt_request_json"])
MUTATION = WorkflowReference.from_dict(
    {k: v for k, v in REVIEW_REQUEST.items() if k != "receipt"}
)
GOOD_BODY = VECTORS["receipt_response_json"].encode()


@pytest.mark.parametrize("case", CASES, ids=lambda v: v["name"])
def test_shared_wire_vectors(case):
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(
            case["status"],
            content=case["body"].encode(),
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": "req_" + "9" * 32,
            },
        )

    reference = WorkflowReference.from_dict(case["reference"])
    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        operation = (
            client.workflow_cancel
            if case["operation"] == "cancel"
            else client.workflow_status
        )
        if "code" in case:
            with pytest.raises(WorkflowClientError) as caught:
                operation(reference)
            error = caught.value
            assert error.code == case["code"]
            assert error.mutation_outcome == case["mutation_outcome"]
            assert error.status_code == case["status"]
            assert error.request_id == "req_" + "9" * 32
            assert "PRIVATE-CANARY" not in repr(error) + str(vars(error))
            assert not hasattr(error, "details") and not hasattr(error, "envelope")
            assert error.__context__ is None and error.__cause__ is None
        else:
            view = operation(reference)
            assert view.to_dict() == case["expected"]
            assert (
                parse_workflow_response(case["body"].encode(), reference).to_dict()
                == case["expected"]
            )
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/workflows/" + case["operation"]
    assert json.loads(requests[0].content) == case["reference"]


def test_existing_receipt_preserves_int64_and_exact_action():
    received = []
    with OpenMedClient(
        transport=httpx.MockTransport(
            lambda request: (
                received.append(request.content),
                httpx.Response(
                    200, content=GOOD_BODY, headers={"Content-Type": "application/json"}
                ),
            )[1]
        )
    ) as client:
        client.workflow_submit_receipt(
            MUTATION, ApprovalReceipt.from_dict(REVIEW_REQUEST["receipt"])
        )
        with pytest.raises(WorkflowClientError) as caught:
            client.workflow_cancel(REFERENCE)
        assert caught.value.mutation_outcome == "not_attempted"
    assert len(received) == 1
    assert received[0].decode() == VECTORS["receipt_request_json"]


@pytest.mark.parametrize(
    "code",
    [
        "workflow_unavailable",
        "workflow_service_failed",
        "workflow_conflict",
        "workflow_forbidden",
        "workflow_mutation_unknown",
    ],
)
def test_retries_only_explicit_inspection(code):
    calls = []
    case = next(v for v in CASES if v["name"] == code)

    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                case["status"],
                content=case["body"],
                headers={"Content-Type": "application/json"},
            )
        return httpx.Response(
            200, content=GOOD_BODY, headers={"Content-Type": "application/json"}
        )

    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        if code in ("workflow_unavailable", "workflow_service_failed"):
            assert (
                client.workflow_status(
                    REFERENCE, options=WorkflowReadOptions(max_attempts=3)
                ).phase.value
                == "waiting-review"
            )
            assert len(calls) == 2
        else:
            with pytest.raises(WorkflowClientError):
                client.workflow_status(
                    REFERENCE, options=WorkflowReadOptions(max_attempts=3)
                )
            assert len(calls) == 1
        calls.clear()
        with pytest.raises(WorkflowClientError):
            client.workflow_cancel(MUTATION)
        assert len(calls) == 1


def test_transport_failure_does_not_replay_a_mutation_or_keep_exception_text():
    calls = []

    def transport(request):
        calls.append(request)
        raise httpx.ReadError("PRIVATE-CANARY", request=request)

    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(WorkflowClientError) as caught:
            client.workflow_cancel(MUTATION)
        assert caught.value.code == "workflow_mutation_unknown"
        assert caught.value.mutation_outcome == "unknown"
        assert "PRIVATE-CANARY" not in str(caught.value)
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert len(calls) == 1
        calls.clear()
        with pytest.raises(WorkflowClientError):
            client.workflow_status(
                REFERENCE, options=WorkflowReadOptions(max_attempts=3)
            )
        assert len(calls) == 3


@pytest.mark.parametrize("phase", ["waiting-review", "completed", "aborted"])
def test_poll_stops_for_review_and_terminal_phase(phase):
    calls = []
    sequence = [
        next(v["body"] for v in CASES if v["name"] == "phase-running"),
        next(v["body"] for v in CASES if v["name"] == "phase-" + phase),
    ]

    def transport(request):
        calls.append(request)
        return httpx.Response(
            200, content=sequence.pop(0), headers={"Content-Type": "application/json"}
        )

    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        assert (
            client.poll_workflow(
                REFERENCE,
                policy=WorkflowPollPolicy(max_requests=3, interval_seconds=0),
                clock=lambda: 0,
            ).phase.value
            == phase
        )
    assert len(calls) == 2
    assert all(request.url.path == "/v1/workflows/status" for request in calls)


def test_poll_request_limit_preserves_last_snapshot_and_does_not_delay_after_limit():
    calls = []
    body = next(v["body"] for v in CASES if v["name"] == "phase-running")
    with OpenMedClient(
        transport=httpx.MockTransport(
            lambda r: (
                calls.append(r),
                httpx.Response(
                    200, content=body, headers={"Content-Type": "application/json"}
                ),
            )[1]
        )
    ) as client:
        with pytest.raises(WorkflowClientError) as caught:
            client.poll_workflow(
                REFERENCE,
                policy=WorkflowPollPolicy(max_requests=1, interval_seconds=30),
                clock=lambda: 0,
            )
        assert caught.value.code == "workflow_poll_timeout"
        assert caught.value.last_view.phase.value == "running"
    assert len(calls) == 1


def test_poll_deadline_and_clock_rollback_stop_without_requests():
    for ticks, code in [
        ([0, 100], "workflow_poll_timeout"),
        ([1, 0], "workflow_clock_failed"),
        ([float("nan")], "workflow_clock_failed"),
    ]:
        calls = []
        times = iter(ticks)
        with OpenMedClient(
            transport=httpx.MockTransport(lambda r: calls.append(r))
        ) as client:
            with pytest.raises(WorkflowClientError) as caught:
                client.poll_workflow(REFERENCE, clock=lambda: next(times))
            assert caught.value.code == code
        assert calls == []


def test_cancellation_before_request_and_during_poll():
    stop = threading.Event()
    stop.set()
    calls = []
    with OpenMedClient(
        transport=httpx.MockTransport(lambda r: calls.append(r))
    ) as client:
        for operation, reference in [
            (client.workflow_status, REFERENCE),
            (client.workflow_cancel, MUTATION),
        ]:
            with pytest.raises(WorkflowClientError) as caught:
                operation(reference, stop_event=stop)
            assert caught.value.code == "workflow_poll_cancelled"
        assert caught.value.mutation_outcome == "not_attempted"
    assert calls == []
    stop.clear()

    def transport(request):
        calls.append(request)
        stop.set()
        return httpx.Response(
            200, content=GOOD_BODY, headers={"Content-Type": "application/json"}
        )

    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(WorkflowClientError) as caught:
            client.poll_workflow(REFERENCE, stop_event=stop)
        assert caught.value.code == "workflow_poll_cancelled"
    assert len(calls) == 1


def test_no_redirect_even_when_injected_http_client_follows_redirects():
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(
            307,
            headers={"Location": "http://private.example/preview"},
            content=b"PRIVATE-CANARY",
        )

    with httpx.Client(
        base_url="http://test.example",
        follow_redirects=True,
        transport=httpx.MockTransport(transport),
    ) as injected:
        with OpenMedClient(client=injected) as client:
            with pytest.raises(WorkflowClientError) as caught:
                client.workflow_cancel(MUTATION)
    assert len(calls) == 1
    assert caught.value.mutation_outcome == "unknown"


@pytest.mark.parametrize(
    "body", [b"{" + b" " * 262145, b"\xff", b'{"value":[' + b"0," * 5000 + b"0]}"]
)
def test_byte_utf8_and_node_bounds(body):
    with pytest.raises(WorkflowClientError):
        parse_workflow_response(body, REFERENCE)


@pytest.mark.parametrize(
    "constructor,values",
    [
        (WorkflowReadOptions, {"max_attempts": True}),
        (WorkflowReadOptions, {"max_attempts": 4}),
        (WorkflowReadOptions, {"retry_delay_seconds": float("inf")}),
        (WorkflowReadOptions, {"timeout_seconds": 31}),
        (WorkflowPollPolicy, {"max_requests": 101}),
        (WorkflowPollPolicy, {"timeout_seconds": 301}),
        (WorkflowPollPolicy, {"interval_seconds": -1}),
    ],
)
def test_closed_resource_policy(constructor, values):
    with pytest.raises(WorkflowClientError):
        constructor(**values)


def test_poll_refusal_keeps_the_last_validated_phase():
    calls = []
    first = next(v["body"] for v in CASES if v["name"] == "phase-running")
    refused = next(v for v in CASES if v["name"] == "workflow_conflict")

    def transport(request):
        calls.append(request)
        return httpx.Response(
            200 if len(calls) == 1 else refused["status"],
            content=first if len(calls) == 1 else refused["body"],
            headers={"Content-Type": "application/json"},
        )

    with OpenMedClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(WorkflowClientError) as caught:
            client.poll_workflow(
                REFERENCE,
                policy=WorkflowPollPolicy(interval_seconds=0),
                clock=lambda: 0,
            )
        assert caught.value.code == "workflow_conflict"
        assert caught.value.last_view.phase.value == "running"
    assert len(calls) == 2


def test_request_deadline_is_checked_during_streaming(monkeypatch):
    ticks = iter([0, 31])
    monkeypatch.setattr(
        "openmed.service.workflow_client.time.monotonic", lambda: next(ticks)
    )
    with OpenMedClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, content=GOOD_BODY, headers={"Content-Type": "application/json"}
            )
        )
    ) as client:
        with pytest.raises(WorkflowClientError) as caught:
            client.workflow_cancel(MUTATION)
    assert caught.value.mutation_outcome == "unknown"
    assert caught.value.code == "workflow_mutation_unknown"


def test_client_construction_and_metadata_parsing_do_not_open_sockets(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("unexpected_network")

    monkeypatch.setattr("socket.create_connection", denied)
    monkeypatch.setattr("socket.socket.connect", denied)
    with OpenMedClient() as client:
        assert client is not None
        assert (
            parse_workflow_response(GOOD_BODY, REFERENCE).phase.value
            == "waiting-review"
        )
