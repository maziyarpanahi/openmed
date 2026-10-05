"""Synthetic token endpoint and shared-custody concurrency integration."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from urllib.parse import parse_qs

import httpx
import pytest

from openmed.interop.fhir.smart_refresh import SmartCredentialRefresher
from tests.fixtures.smart_refresh import (
    HANDLE,
    REQUESTED,
    Clock,
    Custody,
    old_credential,
    response,
)

pytestmark = pytest.mark.integration


def test_configured_endpoint_rotation_narrowing_and_dispatch(caplog):
    clock = Clock()
    custody = Custody(old_credential())
    forms = []
    sent = []

    def endpoint(request):
        assert request.url == "https://synthetic-auth.test/token"
        assert request.method == "POST"
        form = parse_qs(request.content.decode())
        forms.append(form)
        expected = f"synthetic-refresh-{len(forms) - 1}"
        if form["refresh_token"] != [expected]:
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json=response(
                scope="system/SyntheticObservation.c",
                refresh_token=f"synthetic-refresh-{len(forms)}",
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as client:

        def transport(form):
            return client.post("https://synthetic-auth.test/token", data=form).json()

        refresher = SmartCredentialRefresher(
            custody,
            transport,
            requested_scopes=REQUESTED,
            clock=clock,
            sender=sent.append,
        )
        report = refresher.ensure(HANDLE)
        assert report.findings == ("scope_narrowed",)
        assert (
            refresher.dispatch(
                HANDLE, required_scopes="system/SyntheticObservation.u"
            ).code
            == "insufficient_scope"
        )
        assert sent == []
        clock.now = 1240
        assert (
            refresher.dispatch(
                HANDLE, required_scopes="system/SyntheticObservation.c"
            ).code
            == "dispatched"
        )
        assert custody.slot.read().refresh_token == "synthetic-refresh-2"
        assert forms[1]["scope"] == ["system/SyntheticObservation.c"]
        assert sent == ["Bearer synthetic-access-1"]
    assert "synthetic-refresh" not in caplog.text
    assert "synthetic-access" not in caplog.text


@pytest.mark.parametrize("fail", [False, True])
def test_two_refresher_instances_serialize_refresh_and_dispatch(fail):
    clock = Clock()
    custody = Custody(old_credential())
    entered = Event()
    release = Event()
    attempting = Event()
    sent = []
    forms = []

    def transport(form):
        forms.append(dict(form))
        entered.set()
        assert release.wait(5)
        return {"error": "invalid_grant"} if fail else response()

    first = SmartCredentialRefresher(
        custody, transport, requested_scopes=REQUESTED, clock=clock
    )
    second = SmartCredentialRefresher(
        custody, transport, requested_scopes=REQUESTED, clock=clock, sender=sent.append
    )

    def dispatch():
        attempting.set()
        return second.dispatch(HANDLE, required_scopes="system/SyntheticObservation.c")

    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh_job = pool.submit(first.ensure, HANDLE)
        try:
            assert entered.wait(5)
            dispatch_job = pool.submit(dispatch)
            assert attempting.wait(5)
            assert sent == []
            assert not dispatch_job.done()
        finally:
            release.set()
        assert refresh_job.result(timeout=5).code == (
            "invalid_grant" if fail else "refreshed"
        )
        assert dispatch_job.result(timeout=5).code == (
            "revoked" if fail else "dispatched"
        )
    assert len(forms) == 1
    assert sent == ([] if fail else ["Bearer synthetic-access-1"])
    assert custody.slot.replacements == (0 if fail else 1)
