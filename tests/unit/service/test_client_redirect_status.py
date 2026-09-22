"""Unfollowed redirects use the client's documented non-2xx error contract."""

import httpx
import pytest

from openmed.service.client import OpenMedAPIError, OpenMedClient


@pytest.mark.parametrize("status", [301, 302, 303, 304, 307, 308])
@pytest.mark.parametrize("streaming", [False, True])
def test_redirects_are_errors_even_when_the_body_is_valid_json(status, streaming):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={"synthetic": True},
            headers={"Location": "/elsewhere", "X-Request-ID": "synthetic-server-id"},
        )

    with OpenMedClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OpenMedAPIError) as caught:
            if streaming:
                list(client.extract_pii_stream("synthetic"))
            else:
                client.analyze("synthetic")
    assert caught.value.status_code == status
    assert caught.value.code == "http_error"
    assert caught.value.request_id == "synthetic-server-id"
    assert len(requests) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_redirect_with_html_uses_typed_error_and_request_id_fallback(streaming):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            307, text="<p>Moved</p>", headers={"Location": "/other"}
        )
    )
    with OpenMedClient(transport=transport, request_id="synthetic-client-id") as client:
        with pytest.raises(OpenMedAPIError) as caught:
            if streaming:
                list(client.extract_pii_stream("synthetic"))
            else:
                client.analyze("synthetic")
    assert caught.value.status_code == 307
    assert caught.value.request_id == "synthetic-client-id"


def test_redirect_stream_is_closed_when_rejected():
    class TrackedStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b'{"synthetic":true}\n'

        def close(self):
            self.closed = True

    body = TrackedStream()
    transport = httpx.MockTransport(lambda request: httpx.Response(302, stream=body))
    with OpenMedClient(transport=transport) as client:
        with pytest.raises(OpenMedAPIError):
            list(client.extract_pii_stream("synthetic"))
    assert body.closed


@pytest.mark.parametrize("status", [200, 201, 202])
def test_successful_json_responses_are_unchanged(status):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"synthetic": True})
    )
    with OpenMedClient(transport=transport) as client:
        assert client.analyze("synthetic") == {"synthetic": True}


def test_successful_ndjson_response_is_unchanged():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b'{"part":1}\n\n{"part":2}\n')
    )
    with OpenMedClient(transport=transport) as client:
        assert list(client.extract_pii_stream("synthetic")) == [
            {"part": 1},
            {"part": 2},
        ]


def test_existing_error_envelope_is_preserved():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            503,
            json={
                "error": {
                    "code": "synthetic_unavailable",
                    "message": "Synthetic backend unavailable",
                    "details": {"retry": True},
                }
            },
        )
    )
    with OpenMedClient(transport=transport) as client:
        with pytest.raises(OpenMedAPIError) as caught:
            client.analyze("synthetic")
    assert caught.value.code == "synthetic_unavailable"
    assert caught.value.details == {"retry": True}
