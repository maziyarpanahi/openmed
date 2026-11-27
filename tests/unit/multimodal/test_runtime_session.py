"""Preflight, snapshot ownership, stream ordering and bounded runtime evidence."""

import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import openmed.core as openmed_core
from openmed.multimodal import session_lifecycle
from openmed.multimodal.abstention import AbstentionReason, AbstentionStage
from openmed.multimodal.base import ExtractedDocument, SourceSpan
from openmed.multimodal.runtime_session import (
    SESSION_LIFECYCLE_PROJECTION,
    MultimodalSession,
    ProviderLease,
    RuntimeSessionError,
    SessionLimits,
    SessionOperation,
    SessionOutcome,
    SessionState,
)
from openmed.multimodal.streaming.speech.providers import (
    STREAM_LIFECYCLE_PROJECTION,
    PartialHypothesis,
    SpeechProviderError,
    StreamingAsrSession,
    StreamState,
)
from openmed.multimodal.vision.providers.base import Prompt
from tests.fixtures.multimodal.runtime_session import (
    PRIVATE,
    CountedStream,
    FixedProvider,
    asset,
    png,
)


def test_provider_reuse_order_and_owned_asset_lifetimes():
    provider = FixedProvider()
    first, second = asset(), asset()
    with MultimodalSession(image=provider.lease(), clock=lambda: 0) as session:
        one = session.infer_image(first, Prompt(text=PRIVATE))
        two = session.infer_image(second, Prompt(text=PRIVATE))
        assert [e.sequence for e in session.events] == [0, 1]
        assert all(e.outcome is SessionOutcome.SUCCESS for e in session.events)
        assert provider.calls == ["image", "image"] and provider.closed == 0
        assert first.source.close_calls == second.source.close_calls == 1
        assert one.reviewed_output(reviewer_confirmed=True).text == PRIVATE
        with pytest.raises(RuntimeSessionError):
            two.reviewed_output(reviewer_confirmed=False)
    assert session.state is SessionState.CLOSED and provider.closed == 1
    session.close()
    session.cancel()
    assert provider.closed == 1


def test_borrowed_handles_restore_position_and_borrowed_provider_stays_open():
    provider = FixedProvider()
    source = CountedStream(b"prefix" + png())
    source.seek(6)
    item = asset(source=source, owned=False)
    with MultimodalSession(image=provider.lease(owned=False)) as session:
        assert (
            session.infer_image(item, Prompt(text="synthetic")).event.outcome
            is SessionOutcome.SUCCESS
        )
    assert source.tell() == 6 and not source.closed and provider.closed == 0


def test_nonseekable_borrowed_handle_is_consumed_but_not_closed():
    class ForwardOnly:
        def __init__(self):
            self.inner = io.BytesIO(png())

        def read(self, size):
            return self.inner.read(size)

    source = ForwardOnly()
    with MultimodalSession(image=FixedProvider().lease()) as session:
        assert (
            session.infer_image(
                asset(source=source, owned=False), Prompt(text="synthetic")
            ).event.outcome
            is SessionOutcome.SUCCESS
        )
    assert source.read(1) == b"" and not source.inner.closed


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "f" * 64},
        {"width": 5},
        {"data": b"not an image"},
        {"media_type": "image/jpeg"},
    ],
)
def test_rejected_assets_never_reach_provider_and_owned_sources_close(change):
    class Unreachable:
        @property
        def metadata(self):
            pytest.fail("rejected input reached provider")

        def infer(self, request):
            pytest.fail("rejected input reached inference")

    item = asset(**change)
    with MultimodalSession(image=ProviderLease(provider=Unreachable())) as session:
        result = session.infer_image(item, Prompt(text="synthetic"))
        assert result.event.outcome is SessionOutcome.ABSTAINED
        assert result.event.abstention is not None
    assert item.source.close_calls == 1


def test_input_budget_is_enforced_before_read_and_operation_history_is_bounded():
    provider = FixedProvider()
    item = asset()
    with MultimodalSession(
        image=provider.lease(),
        limits=SessionLimits(max_asset_bytes=8, max_operations=1),
    ) as session:
        result = session.infer_image(item, Prompt(text="synthetic"))
        assert result.event.abstention.reason is AbstentionReason.RESOURCE_LIMIT
        assert item.source.read_calls == 0
        extra = asset()
        with pytest.raises(RuntimeSessionError):
            session.infer_image(extra, Prompt(text="synthetic"))
        assert extra.source.close_calls == 1 and len(session.events) == 1
    assert provider.calls == []


def test_cumulative_input_limit_includes_admitted_attempts():
    with MultimodalSession(
        image=FixedProvider().lease(),
        limits=SessionLimits(max_total_input_bytes=len(png())),
    ) as session:
        assert (
            session.infer_image(asset(), Prompt(text="synthetic")).event.outcome
            is SessionOutcome.SUCCESS
        )
        rejected = asset()
        result = session.infer_image(rejected, Prompt(text="synthetic"))
        assert result.event.abstention.reason is AbstentionReason.RESOURCE_LIMIT
        assert rejected.source.read_calls == 0 and rejected.source.closed


def test_provider_failure_does_not_poison_independent_assets_or_echo_exception(caplog):
    class OnceBroken(FixedProvider):
        def infer(self, request):
            if not self.calls:
                self.calls.append("failed")
                raise RuntimeError(PRIVATE)
            return super().infer(request)

    provider = OnceBroken()
    with MultimodalSession(image=provider.lease()) as session:
        assert (
            session.infer_image(asset(), Prompt(text=PRIVATE)).event.outcome
            is SessionOutcome.ABSTAINED
        )
        assert (
            session.infer_image(asset(), Prompt(text=PRIVATE)).event.outcome
            is SessionOutcome.SUCCESS
        )
        assert PRIVATE not in session.to_json() + repr(session.events) + caplog.text
    assert provider.closed == 1


def test_actual_pdf_geometry_resolves_preflight_and_protected_output_is_copied():
    provider = FixedProvider()
    with MultimodalSession(document=provider.lease(), clock=lambda: 0) as session:
        result = session.extract_document(asset("document"))
        assert result.event.outcome is SessionOutcome.SUCCESS
        assert result.event.pdf_geometry_resolved
        assert provider.document_limits == SessionLimits()
        document = result.reviewed_output(reviewer_confirmed=True)
        provider.document.metadata["format"] = "changed"
        assert document.metadata["format"] == "pdf"
        assert PRIVATE not in repr(result) + result.to_json() + session.to_json()


def test_pdf_page_count_and_raster_budget_fail_before_extraction():
    provider = FixedProvider()
    limits = SessionLimits()
    with MultimodalSession(document=provider.lease()) as session:
        assert (
            session.extract_document(asset("document", pages=2)).event.abstention.reason
            is AbstentionReason.MALFORMED_MEDIA
        )
    tiny = replace(
        limits, preflight=limits.preflight.with_limits(max_pixels=1, max_total_pixels=1)
    )
    with MultimodalSession(
        document=provider.lease(owned=False), limits=tiny
    ) as session:
        result = session.extract_document(asset("document"))
        assert result.event.abstention.reason is AbstentionReason.RESOURCE_LIMIT
    assert provider.calls == []


@pytest.mark.parametrize(
    "document",
    [
        ExtractedDocument("ok", (SourceSpan(0, 3),)),
        ExtractedDocument("ok", (SourceSpan(0, 2, page=8),)),
        ExtractedDocument("ok", metadata={"private": object()}),
        ExtractedDocument("ok", metadata={"private": "x" * 100}),
    ],
)
def test_invalid_or_unbounded_document_output_never_becomes_success(document):
    provider = FixedProvider()
    provider.document = document
    with MultimodalSession(
        document=provider.lease(), limits=SessionLimits(max_output_chars=50)
    ) as session:
        result = session.extract_document(asset("document"))
        assert result.event.outcome is SessionOutcome.ABSTAINED
        with pytest.raises(RuntimeSessionError):
            result.reviewed_output(reviewer_confirmed=True)


def test_wav_stream_executes_adapter_and_validates_order_then_finalizes():
    provider = FixedProvider()
    with MultimodalSession(speech=provider.lease(), clock=lambda: 0) as session:
        wrong = session.push_audio(asset("audio"), sequence=1)
        assert wrong.event.abstention.reason is AbstentionReason.TEMPORAL_INSTABILITY
        assert provider.calls == []
        first = session.push_audio(asset("audio"), sequence=0)
        second = session.push_audio(asset("audio"), sequence=1)
        final = session.finish_audio()
        assert all(
            r.event.outcome is SessionOutcome.SUCCESS for r in (first, second, final)
        )
        assert final.event.operation is SessionOperation.FINISH_AUDIO
        assert isinstance(
            first.reviewed_output(reviewer_confirmed=True)[0], PartialHypothesis
        )
        assert (
            session.push_audio(asset("audio"), sequence=2).event.outcome
            is SessionOutcome.ABSTAINED
        )
        assert provider.calls == ["audio", "audio", "finish_audio"]
    assert provider.closed == 1


def test_wav_duration_mismatch_never_reaches_audio_adapter():
    provider = FixedProvider()
    with MultimodalSession(speech=provider.lease()) as session:
        result = session.push_audio(asset("audio", duration_seconds=1), sequence=0)
        assert result.event.abstention.stage is AbstentionStage.DECODE
        assert provider.calls == []


def test_bad_speech_emission_aborts_only_that_stream_and_releases_input():
    class BrokenSpeech(FixedProvider):
        def push(self, chunk, *, cancellation):
            self.calls.append("audio")
            return (PartialHypothesis(revision=1, text="x" * 200),)

    speech, image = BrokenSpeech(), FixedProvider()
    with MultimodalSession(
        speech=speech.lease(),
        image=image.lease(),
        limits=SessionLimits(max_output_chars=100),
    ) as session:
        audio = asset("audio")
        assert (
            session.push_audio(audio, sequence=0).event.outcome
            is SessionOutcome.ABSTAINED
        )
        assert audio.source.closed
        assert (
            session.push_audio(asset("audio"), sequence=1).event.outcome
            is SessionOutcome.ABSTAINED
        )
        assert speech.calls == ["audio"]
        assert speech.aborted == 1
        assert (
            session.infer_image(asset(), Prompt(text="synthetic")).event.outcome
            is SessionOutcome.SUCCESS
        )
    assert speech.aborted == 1 and speech.closed == 1


def test_unfinished_borrowed_speech_releases_stream_but_not_shared_provider():
    provider = FixedProvider()
    with MultimodalSession(speech=provider.lease(owned=False)) as session:
        assert (
            session.push_audio(asset("audio"), sequence=0).event.outcome
            is SessionOutcome.SUCCESS
        )
    assert provider.aborted == 1 and provider.closed == 0


def test_completed_speech_does_not_abort_twice_at_session_close():
    provider = FixedProvider()
    with MultimodalSession(speech=provider.lease()) as session:
        session.push_audio(asset("audio"), sequence=0)
        assert session.finish_audio().event.outcome is SessionOutcome.SUCCESS
    assert provider.aborted == 0 and provider.closed == 1


def test_failed_speech_cleanup_is_reported_without_blocking_other_cleanup():
    class Broken(FixedProvider):
        def abort(self):
            super().abort()
            raise RuntimeError(PRIVATE)

    provider = Broken()
    with MultimodalSession(speech=provider.lease()) as session:
        session.push_audio(asset("audio"), sequence=0)
    assert provider.aborted == 1 and provider.closed == 1
    report = session.to_json()
    assert json.loads(report)["cleanup_failures"] == 1 and PRIVATE not in report


def test_asset_read_failure_is_private_and_does_not_poison_later_work(caplog):
    class Broken(CountedStream):
        def read(self, size):
            raise OSError(PRIVATE)

    item = asset(source=Broken(png()))
    with MultimodalSession(image=FixedProvider().lease()) as session:
        result = session.infer_image(item, Prompt(text=PRIVATE))
        assert result.event.outcome is SessionOutcome.ABSTAINED and item.source.closed
        assert (
            session.infer_image(asset(), Prompt(text="synthetic")).event.outcome
            is SessionOutcome.SUCCESS
        )
        assert PRIVATE not in result.to_json() + session.to_json() + caplog.text


@pytest.mark.parametrize("operation", ["image", "document", "audio"])
def test_network_declarations_never_reach_provider_execution(operation):
    class NetworkProvider(FixedProvider):
        @property
        def metadata(self):
            return replace(FixedProvider.metadata, requires_network=True)

        @property
        def declaration(self):
            return replace(
                FixedProvider.declaration,
                provenance=replace(
                    FixedProvider.declaration.provenance, requires_network=True
                ),
            )

    provider = NetworkProvider()
    with MultimodalSession(
        **{operation if operation != "audio" else "speech": provider.lease()}
    ) as session:
        if operation == "image":
            result = session.infer_image(asset(), Prompt(text="synthetic"))
        elif operation == "document":
            result = session.extract_document(asset("document"))
        else:
            result = session.push_audio(asset("audio"), sequence=0)
        assert result.event.outcome is SessionOutcome.ABSTAINED
    assert provider.calls == [] and provider.closed == 1


def test_shared_provider_lease_closes_once_and_cleanup_failure_does_not_skip_others():
    provider = FixedProvider()
    lease = provider.lease()
    with MultimodalSession(image=lease, document=lease) as session:
        assert (
            session.infer_image(asset(), Prompt(text="synthetic")).event.outcome
            is SessionOutcome.SUCCESS
        )
        assert (
            session.extract_document(asset("document")).event.outcome
            is SessionOutcome.SUCCESS
        )
    assert provider.closed == 1
    other = FixedProvider()

    def broken_close():
        raise RuntimeError(PRIVATE)

    session = MultimodalSession(
        image=ProviderLease(provider=provider, owned=True, release=broken_close),
        document=other.lease(),
    )
    session.close()
    assert other.closed == 1
    assert json.loads(session.to_json())["cleanup_failures"] == 1
    assert PRIVATE not in session.to_json()


def test_asset_close_failure_withholds_success_and_is_recorded():
    class BrokenClose(CountedStream):
        def close(self):
            super().close()
            raise RuntimeError(PRIVATE)

    item = asset(source=BrokenClose(png()))
    with MultimodalSession(image=FixedProvider().lease()) as session:
        result = session.infer_image(item, Prompt(text="synthetic"))
        assert (
            result.event.outcome is SessionOutcome.ABSTAINED
            and result.event.cleanup_failed
        )
        assert PRIVATE not in result.to_json()


def test_cancelled_or_closed_sessions_release_new_owned_handles_without_execution():
    provider = FixedProvider()
    session = MultimodalSession(image=provider.lease())
    assert session.cancel() is SessionState.CANCELLED
    item = asset()
    with pytest.raises(RuntimeSessionError):
        session.infer_image(item, Prompt(text="synthetic"))
    assert item.source.closed and provider.calls == [] and provider.closed == 1


@pytest.mark.parametrize(
    "change",
    [
        {"max_operations": 0},
        {"max_asset_bytes": True},
        {"max_output_chars": 70000},
        {"pdf_dpi": 601},
    ],
)
def test_invalid_session_limits_fail_without_echo(change):
    with pytest.raises(RuntimeSessionError, match="^invalid runtime session contract$"):
        SessionLimits(**change)


# ---------------------------------------------------------------------------
# Shared session lifecycle vocabulary (openmed.session.lifecycle.v1)
# ---------------------------------------------------------------------------

LIFECYCLE_VECTORS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "parity"
        / "session_lifecycle_v1.json"
    ).read_text(encoding="utf-8")
)
LIFECYCLE_SCHEMA = json.loads(
    (
        Path(openmed_core.__file__).resolve().parent
        / "schemas"
        / "json"
        / "session_lifecycle.schema.json"
    ).read_text(encoding="utf-8")
)
RUNTIME_KIND = "python.runtime_session"
STREAM_KIND = "python.streaming_asr"


class _RuntimeDriver:
    """Drive a MultimodalSession, holding one operation open on request."""

    error = RuntimeSessionError

    def __init__(self):
        self.entered, self.unblock = threading.Event(), threading.Event()
        entered, unblock = self.entered, self.unblock

        class Blocking(FixedProvider):
            def infer(self, request):
                entered.set()
                assert unblock.wait(10), "test coordination failed"
                return super().infer(request)

        self.session = MultimodalSession(image=Blocking().lease())
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def apply(self, action):
        if action in {"cancel", "close"}:
            getattr(self.session, action)()
        elif action in {"cancel_while_active", "close_while_active"}:
            self.future = self.pool.submit(
                self.session.infer_image, asset(), Prompt(text="synthetic")
            )
            assert self.entered.wait(10), "test coordination failed"
            getattr(self.session, action.removesuffix("_while_active"))()
        elif action == "release_active":
            self.unblock.set()
            self.future.result(timeout=10)
        else:
            pytest.fail("unknown runtime driver action")
        return self.session.lifecycle_state

    def shutdown(self):
        self.unblock.set()
        self.pool.shutdown(wait=True)
        self.session.close()


class _StreamDriver:
    """Drive a StreamingAsrSession through its native terminal actions."""

    error = SpeechProviderError

    def __init__(self):
        self.session = StreamingAsrSession(FixedProvider.declaration)

    def apply(self, action):
        assert action in {"finalize", "cancel"}
        getattr(self.session, action)()
        return self.session.lifecycle_state

    def shutdown(self):
        pass


DRIVERS = {RUNTIME_KIND: _RuntimeDriver, STREAM_KIND: _StreamDriver}


def _path_to(kind, state):
    """Return driver actions that leave a fresh session of ``kind`` in ``state``."""
    if state == "open":
        return []
    for sequence in LIFECYCLE_VECTORS["allowed_sequences"]:
        actions = sequence["drivers"].get(kind)
        if actions and state in sequence["states"][1:]:
            return actions[: sequence["states"].index(state)]
    return None


def _reasons_match(error, kind, reason):
    if kind == STREAM_KIND:
        return error.category == reason and str(error) == reason
    return str(error) == reason


def test_lifecycle_vocabulary_agrees_across_module_schema_and_vectors():
    order = list(session_lifecycle.LIFECYCLE_STATES)
    defs = LIFECYCLE_SCHEMA["$defs"]

    assert LIFECYCLE_VECTORS["synthetic"] is True
    assert LIFECYCLE_VECTORS["version"] == 1
    assert LIFECYCLE_VECTORS["lifecycle_schema_version"] == (
        session_lifecycle.SESSION_LIFECYCLE_VERSION
    )
    assert order == ["open", "cancelling", "finalized", "cancelled", "failed", "closed"]
    assert LIFECYCLE_VECTORS["states"] == order == defs["state"]["enum"]
    terminal = {state.value for state in session_lifecycle.TERMINAL_STATES}
    assert set(LIFECYCLE_VECTORS["terminal_states"]) == terminal
    assert set(defs["terminal_state"]["enum"]) == terminal
    assert LIFECYCLE_VECTORS["reason_codes"] == list(
        session_lifecycle.LIFECYCLE_REASON_CODES
    )
    assert sorted(defs["reason_code"]["enum"]) == sorted(
        session_lifecycle.LIFECYCLE_REASON_CODES
    )
    Draft202012Validator.check_schema(LIFECYCLE_SCHEMA)
    validator = Draft202012Validator(LIFECYCLE_SCHEMA)
    for source in order:
        allowed = set(LIFECYCLE_VECTORS["transitions"][source])
        assert allowed == {
            state.value
            for state in session_lifecycle.ALLOWED_TRANSITIONS[
                session_lifecycle.LifecycleState(source)
            ]
        }
        assert (not allowed) is (source in terminal)
        for target in order:
            record = {
                "schema_version": session_lifecycle.SESSION_LIFECYCLE_VERSION,
                "from": source,
                "to": target,
            }
            assert validator.is_valid(record) is (target in allowed)
            reason = session_lifecycle.transition_reason(source, target)
            assert (reason is None) is (target in allowed)


def test_projections_cover_every_native_state():
    projections = LIFECYCLE_VECTORS["projections"]

    assert projections[RUNTIME_KIND]["map"] == {
        native.value: projected.value
        for native, projected in SESSION_LIFECYCLE_PROJECTION.items()
    }
    assert set(projections[RUNTIME_KIND]["map"]) == {s.value for s in SessionState}
    assert projections[STREAM_KIND]["map"] == {
        native.value: projected.value
        for native, projected in STREAM_LIFECYCLE_PROJECTION.items()
    }
    assert set(projections[STREAM_KIND]["map"]) == {s.value for s in StreamState}
    swift_source = (
        Path(__file__).resolve().parents[3]
        / "swift"
        / "OpenMedKit"
        / "Sources"
        / "OpenMedKit"
        / "StreamingSpeechSession.swift"
    ).read_text(encoding="utf-8")
    declared = re.search(
        r"enum SpeechSessionState: String, Sendable \{ case ([a-z, ]+) \}",
        swift_source,
    )
    assert declared is not None
    swift_states = {item.strip() for item in declared.group(1).split(",")}
    assert set(projections["swift.streaming_speech"]["map"]) == swift_states
    for projection in projections.values():
        assert set(projection["map"].values()) <= set(LIFECYCLE_VECTORS["states"])


@pytest.mark.parametrize(
    "sequence",
    LIFECYCLE_VECTORS["allowed_sequences"],
    ids=lambda sequence: sequence["id"],
)
@pytest.mark.parametrize("kind", [RUNTIME_KIND, STREAM_KIND])
def test_allowed_sequences_pass_through_each_session_projection(kind, sequence):
    states = sequence["states"]
    for source, target in zip(states, states[1:]):
        assert session_lifecycle.transition_reason(source, target) is None
    actions = sequence["drivers"].get(kind)
    if not actions:
        # This runtime has no native path to the sequence; the shared
        # transition table checked above is its contract.
        return
    driver = DRIVERS[kind]()
    try:
        assert driver.session.lifecycle_state.value == states[0]
        observed = [states[0]]
        for action, expected in zip(actions, states[1:]):
            driver.session.check_lifecycle_transition(expected)
            observed.append(driver.apply(action).value)
        assert observed == states
    finally:
        driver.shutdown()


@pytest.mark.parametrize(
    "case",
    LIFECYCLE_VECTORS["rejected_transitions"],
    ids=lambda case: case["id"],
)
def test_rejected_transitions_raise_the_existing_error_with_the_reason(case):
    assert case["reason"] in session_lifecycle.LIFECYCLE_REASON_CODES
    assert (
        session_lifecycle.transition_reason(case["from"], case["to"]) == case["reason"]
    )
    for kind, factory in DRIVERS.items():
        path = _path_to(kind, case["from"])
        if path is None:
            continue
        driver = factory()
        try:
            for action in path:
                driver.apply(action)
            before = driver.session.state
            with pytest.raises(driver.error) as raised:
                driver.session.check_lifecycle_transition(case["to"])
            assert _reasons_match(raised.value, kind, case["reason"])
            assert driver.session.state is before
            native = case["native_actions"].get(kind)
            if native is not None:
                with pytest.raises(driver.error) as raised:
                    getattr(driver.session, native)()
                assert _reasons_match(raised.value, kind, case["reason"])
        finally:
            driver.shutdown()


def test_every_reachable_python_state_is_exercised_by_the_vectors():
    sources = {case["from"] for case in LIFECYCLE_VECTORS["rejected_transitions"]}
    for kind, projection in (
        (RUNTIME_KIND, SESSION_LIFECYCLE_PROJECTION),
        (STREAM_KIND, STREAM_LIFECYCLE_PROJECTION),
    ):
        for state in projection.values():
            assert _path_to(kind, state.value) is not None
            assert state.value in sources


def test_lifecycle_projection_leaves_native_enums_and_reports_unchanged():
    assert [state.value for state in SessionState] == [
        "open",
        "cancelling",
        "cancelled",
        "closed",
    ]
    assert [state.value for state in StreamState] == ["open", "finalized", "cancelled"]
    session = MultimodalSession()
    assert session.to_json() == (
        '{"cleanup_failures":0,"events":[],"input_bytes":0,'
        '"requires_network":false,"schema_version":"openmed.multimodal.session.v1",'
        '"state":"open"}'
    )
    session.close()
    assert json.loads(session.to_json())["state"] == "closed"
    assert session.lifecycle_state is session_lifecycle.LifecycleState.CLOSED
    assert "lifecycle" not in session.to_json()
    report = StreamingAsrSession(FixedProvider.declaration).finalize().to_dict()
    assert report["state"] == "finalized"
    assert list(report) == [
        "schema_version",
        "provider_id",
        "provider_revision",
        "model_fingerprint",
        "state",
        "buffered_chunks",
        "buffered_byte_count",
        "buffered_duration_ms",
        "stream_duration_ms",
        "partial_hypotheses",
        "finalized_segments",
        "reason_code",
        "requires_network",
    ]
