"""Offline CLI/service cancellation parity with synthetic reviewed artifacts."""

import asyncio
import json
from threading import Event

import pytest
from starlette.requests import Request

from openmed.cli.brief import handle_brief
from openmed.clinical.brief_cancellation import BriefCancellation
from openmed.service.app import create_app
from openmed.service.brief import brief_response
from openmed.service.schemas import BriefRequest
from tests.unit.clinical.test_brief import fixture_context
from tests.unit.service.test_brief_surfaces import REVIEW_ID, cli_args

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, asyncio.CancelledError])
def test_cli_and_service_lookup_interruptions_share_empty_outcome(
    tmp_path, capsys, interruption
):
    value, _ = fixture_context()

    def provider(*_, cancellation):
        assert isinstance(cancellation, BriefCancellation)
        raise interruption("SYNTHETIC_PRIVATE_FAILURE")

    expected = brief_response(
        value.original_text,
        model="extractive",
        review_id=REVIEW_ID,
        context_provider=provider,
        cancellation=BriefCancellation(),
    )
    assert expected["refusal_reason"] == "cancelled"
    assert expected["summary"] == ""
    args = cli_args(tmp_path, value.original_text)
    assert handle_brief(args, context_provider=provider) == 1
    assert json.loads(capsys.readouterr().out)["data"] == expected
    assert not args.summary_output.exists()
    assert not args.review_output.exists()


def test_route_cancellation_stops_late_worker_before_generation(monkeypatch):
    asyncio.run(_route_cancellation(monkeypatch))


async def _route_cancellation(monkeypatch):
    value, context = fixture_context()
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    finished = asyncio.Event()
    release = Event()
    contexts = []
    outcomes = []
    actual_response = brief_response

    def provider(*_, cancellation):
        contexts.append(cancellation)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "test provider was not released"
        return value, context

    def response(*args, **kwargs):
        result = actual_response(*args, **kwargs)
        if args[0]:
            outcomes.append(result)
            loop.call_soon_threadsafe(finished.set)
        return result

    monkeypatch.setattr("openmed.service.brief.brief_response", response)
    app = create_app()
    app.state.brief_context_provider = provider
    route = next(route for route in app.routes if route.path == "/brief")
    request = Request({"type": "http", "app": app, "state": {}})
    task = asyncio.create_task(
        route.endpoint(
            BriefRequest(
                text=value.original_text, model="extractive", review_id=REVIEW_ID
            ),
            request,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        result = await asyncio.wait_for(task, 5)
        assert result["refusal_reason"] == "cancelled"
        assert result["summary"] == ""
    finally:
        release.set()
    await asyncio.wait_for(finished.wait(), 5)
    assert outcomes == [result]
    assert outcomes[0]["stages"] == []
    assert len(contexts) == 1


def test_cli_deadline_during_lookup_has_no_output_files(tmp_path, capsys, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("openmed.core.budget.time.perf_counter", lambda: now[0])
    value, context = fixture_context()

    def provider(*_, cancellation):
        now[0] = 2.0
        return value, context

    args = cli_args(tmp_path, value.original_text)
    args.timeout_seconds = 1.0
    assert handle_brief(args, context_provider=provider) == 1
    response = json.loads(capsys.readouterr().out)["data"]
    assert response["refusal_reason"] == "deadline_exceeded"
    assert response["summary"] == ""
    assert not args.summary_output.exists()
    assert not args.review_output.exists()


def test_cli_interrupt_during_output_reservation_removes_owned_files(
    tmp_path, capsys, monkeypatch
):
    import os

    value, context = fixture_context()
    args = cli_args(tmp_path, value.original_text)
    original_open = os.open

    def interrupted_open(path, *args, **kwargs):
        if path == tmp_path / "review.json":
            raise KeyboardInterrupt("SYNTHETIC_PRIVATE_FAILURE")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("openmed.cli.brief.os.open", interrupted_open)
    assert handle_brief(args, context_provider=lambda *_: (value, context)) == 1
    assert not args.summary_output.exists()
    assert not args.review_output.exists()
    response = json.loads(capsys.readouterr().out)["data"]
    assert response["refusal_reason"] == "cancelled"
    assert response["summary"] == ""
