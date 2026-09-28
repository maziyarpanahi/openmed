"""Load-test requests remain connected until streaming response completion."""

import asyncio

import pytest
from starlette.responses import StreamingResponse

import openmed.eval.load_test as load_test


def _run(coroutine):
    async def bounded():
        return await asyncio.wait_for(coroutine, timeout=3.0)

    return asyncio.run(bounded())


def test_receiving_request_body_does_not_report_early_disconnect():
    async def app(scope, receive, send):
        request = await receive()
        assert request["type"] == "http.request"
        assert request["more_body"] is False
        listener = asyncio.create_task(receive())
        try:
            await asyncio.sleep(0)
            assert not listener.done()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {"type": "http.response.body", "body": b"one", "more_body": True}
            )
            await asyncio.sleep(0)
            assert not listener.done()
            await send(
                {"type": "http.response.body", "body": b"two", "more_body": False}
            )
            assert await listener == {"type": "http.disconnect"}
        finally:
            if not listener.done():
                listener.cancel()
                try:
                    await listener
                except asyncio.CancelledError:
                    pass

    assert _run(load_test._post(app, "/synthetic", {"synthetic": True})) == 200


def test_real_streaming_response_finishes_and_timing_includes_generation(monkeypatch):
    emitted = []
    clock = [0.0]
    monkeypatch.setattr(load_test, "perf_counter", lambda: clock[0])

    async def body():
        for item in range(3):
            await asyncio.sleep(0)
            clock[0] += 0.25
            emitted.append(item)
            yield b"synthetic\n"

    async def app(scope, receive, send):
        await StreamingResponse(body())(scope, receive, send)

    latency, failed = _run(load_test._timed_request(app, "/synthetic", {}))
    assert emitted == [0, 1, 2]
    assert latency == pytest.approx(750.0)
    assert failed is False


@pytest.mark.parametrize(
    "body_event",
    [None, {"type": "http.response.body", "body": b"partial", "more_body": True}],
)
def test_unfinished_response_is_counted_as_failure(body_event):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        if body_event is not None:
            await send(body_event)

    _, failed = _run(load_test._timed_request(app, "/synthetic", {}))
    assert failed is True


def test_final_body_without_more_body_flag_is_complete():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    assert _run(load_test._post(app, "/synthetic", {})) == 201


def test_full_load_report_waits_for_all_streamed_requests():
    completed = []

    async def body():
        await asyncio.sleep(0)
        yield b"{}\n"
        await asyncio.sleep(0)
        completed.append(True)

    async def app(scope, receive, send):
        await StreamingResponse(body())(scope, receive, send)

    report = load_test.run_load_test(app, concurrency=2, total_requests=5, payload={})
    assert len(completed) == 5
    assert report.error_rate == 0.0
    assert report.requests_per_second > 0


@pytest.mark.parametrize("status", [400, 500])
def test_complete_error_response_remains_an_error(status):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    _, failed = _run(load_test._timed_request(app, "/synthetic", {}))
    assert failed is True
