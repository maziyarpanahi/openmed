"""Offline composition and cancellation races using synthetic local providers."""

import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from openmed.multimodal.runtime_session import (
    MultimodalSession,
    RuntimeSessionError,
    SessionOutcome,
    SessionState,
)
from openmed.multimodal.vision.providers.base import Prompt
from tests.fixtures.multimodal.runtime_session import (
    PRIVATE,
    CountedStream,
    FixedProvider,
    asset,
    png,
)

pytestmark = pytest.mark.integration


def test_three_modalities_share_lifetime_and_deterministic_metadata_without_network(
    monkeypatch, caplog
):
    def deny(*args, **kwargs):
        pytest.fail("unexpected network access")

    monkeypatch.setattr(socket, "socket", deny)
    reports = []
    for _ in range(2):
        provider = FixedProvider()
        lease = provider.lease()
        with MultimodalSession(
            image=lease, document=lease, speech=lease, clock=lambda: 0
        ) as session:
            results = [
                session.infer_image(asset(), Prompt(text=PRIVATE)),
                session.extract_document(asset("document")),
                session.push_audio(asset("audio"), sequence=0),
                session.finish_audio(),
            ]
            assert all(r.event.outcome is SessionOutcome.SUCCESS for r in results)
            assert [e.sequence for e in session.events] == [0, 1, 2, 3]
        assert provider.closed == 1
        reports.append(session.to_json())
        assert PRIVATE not in session.to_json() + caplog.text + repr(results)
    assert reports[0] == reports[1]


def test_cancellation_discards_late_provider_success_and_cleans_after_unwind():
    entered, unblock = threading.Event(), threading.Event()

    class Blocking(FixedProvider):
        def infer(self, request):
            entered.set()
            assert unblock.wait(10), "test coordination failed"
            return super().infer(request)

    provider = Blocking()
    session = MultimodalSession(image=provider.lease())
    item = asset()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.infer_image, item, Prompt(text=PRIVATE))
        try:
            assert entered.wait(10), "test coordination failed"
            assert session.cancel() is SessionState.CANCELLING
            assert provider.closed == 0 and not item.source.closed
            extra = asset()
            with pytest.raises(RuntimeSessionError):
                session.infer_image(extra, Prompt(text="synthetic"))
            assert extra.source.closed
        finally:
            unblock.set()
        result = future.result(timeout=10)
    assert result.event.outcome is SessionOutcome.CANCELLED
    assert session.state is SessionState.CANCELLED
    assert provider.cancelled == provider.closed == item.source.close_calls == 1
    assert result.event.result is None and all(
        e.outcome is not SessionOutcome.SUCCESS for e in session.events
    )
    with pytest.raises(RuntimeSessionError):
        result.reviewed_output(reviewer_confirmed=True)


def test_cancel_during_asset_read_with_no_provider_does_not_leave_cleanup_pending():
    entered, unblock = threading.Event(), threading.Event()

    class BlockingRead(CountedStream):
        def read(self, size):
            entered.set()
            assert unblock.wait(10), "test coordination failed"
            return super().read(size)

    session = MultimodalSession()
    item = asset(source=BlockingRead(png()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.infer_image, item, Prompt(text="synthetic"))
        try:
            assert entered.wait(10), "test coordination failed"
            assert session.cancel() is SessionState.CANCELLING
        finally:
            unblock.set()
        assert future.result(timeout=10).event.outcome is SessionOutcome.CANCELLED
    assert session.state is SessionState.CANCELLED and item.source.close_calls == 1


def test_hashed_snapshot_cannot_change_when_original_source_is_replaced():
    original = CountedStream(png())

    class MutatingProvider(FixedProvider):
        def infer(self, request):
            original.seek(0)
            original.write(b"modified")
            assert request.image.data == png()
            return super().infer(request)

    with MultimodalSession(image=MutatingProvider().lease()) as session:
        result = session.infer_image(
            asset(source=original, owned=False), Prompt(text="synthetic")
        )
        assert result.event.outcome is SessionOutcome.SUCCESS
    assert not original.closed


def test_cancellation_waits_for_cancel_hook_before_releasing_provider():
    entered, unblock_call = threading.Event(), threading.Event()
    hook_entered, unblock_hook = threading.Event(), threading.Event()

    class Blocking(FixedProvider):
        def infer(self, request):
            entered.set()
            assert unblock_call.wait(10), "test coordination failed"
            return super().infer(request)

        def cancel(self):
            super().cancel()
            hook_entered.set()
            assert unblock_hook.wait(10), "test coordination failed"

    provider = Blocking()
    session = MultimodalSession(image=provider.lease())
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(session.infer_image, asset(), Prompt(text="synthetic"))
        try:
            assert entered.wait(10), "test coordination failed"
            cancelled = pool.submit(session.cancel)
            assert hook_entered.wait(10), "test coordination failed"
            unblock_call.set()
            assert active.result(timeout=10).event.outcome is SessionOutcome.CANCELLED
            assert session.state is SessionState.CANCELLING and provider.closed == 0
        finally:
            unblock_call.set()
            unblock_hook.set()
        assert cancelled.result(timeout=10) is SessionState.CANCELLED
    assert provider.cancelled == provider.closed == 1
