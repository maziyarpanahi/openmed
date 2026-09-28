"""Capture real ASGI scope/events with an in-process synthetic app."""

import asyncio
import json

import pytest

from openmed.eval.load_test import _post, run_load_test


@pytest.mark.parametrize(
    "target,path,raw,query",
    [
        ("/synthetic?mode=a", "/synthetic", b"/synthetic", b"mode=a"),
        ("/synthetic?a=1&a=2&x=%2F", "/synthetic", b"/synthetic", b"a=1&a=2&x=%2F"),
        ("/a%20b?q=a+b", "/a b", b"/a%20b", b"q=a+b"),
        ("/a%3Fb?q=%3F", "/a?b", b"/a%3Fb", b"q=%3F"),
        ("/%E7%95%8C", "/界", b"/%E7%95%8C", b""),
        ("/a%2Fb", "/a/b", b"/a%2Fb", b""),
        ("/a%252Fb", "/a%2Fb", b"/a%252Fb", b""),
        ("//synthetic?x=1", "//synthetic", b"//synthetic", b"x=1"),
        ("/plain", "/plain", b"/plain", b""),
        ("/plain?", "/plain", b"/plain", b""),
    ],
)
def test_post_scope_has_path_and_raw_query(tmp_path, target, path, raw, query):
    seen = []

    async def app(scope, receive, send):
        seen.append(scope)
        event = await receive()
        assert json.loads(event["body"]) == {"synthetic": True}
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"ok"})

    assert asyncio.run(_post(app, target, {"synthetic": True})) == 200
    assert seen[0]["path"] == path
    assert seen[0]["raw_path"] == raw
    assert seen[0]["query_string"] == query


def test_public_load_run_routes_query_target_correctly():
    async def app(scope, receive, send):
        good = scope["path"] == "/synthetic" and scope["query_string"] == b"flag=yes"
        await send({"type": "http.response.start", "status": 200 if good else 404})
        await send({"type": "http.response.body", "body": b"ok"})

    result = run_load_test(
        app,
        concurrency=2,
        total_requests=3,
        payload={"synthetic": True},
        path="/synthetic?flag=yes",
    )
    assert result.error_rate == 0


def test_incomplete_response_still_fails():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})

    with pytest.raises(RuntimeError, match="complete"):
        asyncio.run(_post(app, "/synthetic", {}))
