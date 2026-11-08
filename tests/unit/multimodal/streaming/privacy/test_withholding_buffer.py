"""Synthetic PHI, revision, deadline and context-loss controls for speech holdback."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.streaming.privacy import (
    BufferState,
    HoldReason,
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
    WithholdingBuffer,
    WithholdingError,
    WithholdingPolicy,
)
from openmed.multimodal.streaming.transcript import RevisionKind, TranscriptToken

SESSION, FIRST, SECOND, THIRD = (f"{value:032x}" for value in range(4))
PRIVATE = "SYNTHETIC-PRIVATE-IDENTIFIER"


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds


class Screener:
    declaration = PhiScreenDeclaration(artifact_digest="a" * 64)

    def __init__(self):
        self.calls = []
        self.resolution = PhiResolution.RESOLVED

    def screen(self, request):
        self.calls.append(request)
        decisions = []
        for segment in request.segments:
            start = segment.text.find(PRIVATE)
            decisions.append(
                SegmentPhiDecision(
                    segment_ref=segment.segment_ref,
                    revision=segment.revision,
                    resolution=self.resolution,
                    redactions=(PhiRedactionSpan(start, start + len(PRIVATE)),)
                    if start >= 0
                    else (),
                )
            )
        return PhiScreenResult(
            request_digest=request.request_digest, decisions=tuple(decisions)
        )


def buffer(*, screener=None, clock=None, **limits):
    return WithholdingBuffer(
        SESSION,
        screener or Screener(),
        clock=clock or Clock(),
        policy=WithholdingPolicy(**limits),
    )


def finalized(target, ref=FIRST, text="Synthetic text"):
    assert target.append(ref, text)
    assert target.finalize(ref, expected_revision=0)


def test_provisional_and_trailing_context_never_release_before_final_screen():
    screener = Screener()
    with buffer(screener=screener) as target:
        target.append(FIRST, PRIVATE)
        assert target.release_ready() == ()
        target.finalize(FIRST, expected_revision=0)
        assert target.release_ready() == () and screener.calls == []
        target.append(SECOND, "No allergy")
        assert target.release_ready() == () and target.reason is HoldReason.PROVISIONAL
        target.finalize(SECOND, expected_revision=0)
        released = target.release_ready()
        assert [item.segment_ref for item in released] == [FIRST]
        assert released[0].text == "[REDACTED]"
        assert target.finish()[0].text == "No allergy"
        assert target.state is BufferState.CLOSED


def test_editing_provisional_text_preserves_negation_and_retires_changed_lineage():
    token = TranscriptToken(token_ref="a" * 32, start=0, end=3, start_ms=0, end_ms=10)
    fresh = TranscriptToken(
        token_ref="b" * 32, start=0, end=len(PRIVATE), start_ms=0, end_ms=10
    )
    with buffer(trailing_segments=0) as target:
        target.append(FIRST, "old has no allergy", (token,))
        assert target.release_ready() == ()
        target.edit(
            FIRST, RevisionKind.REPLACE, 0, 3, PRIVATE, (fresh,), expected_revision=0
        )
        assert target.lineage(FIRST)["operations"][0]["retired_tokens"] == ("a" * 32,)
        target.finalize(FIRST, expected_revision=1)
        result = target.release_ready()[0]
        assert result.text == "[REDACTED] has no allergy"
        assert result.revision == 2 and json.loads(result.lineage_json)["finalized"]
        assert result.mappings[0].redacted
        assert result.mappings[1].source_start == len(PRIVATE)


@pytest.mark.parametrize(
    "resolution,reason",
    [
        (PhiResolution.UNRESOLVED, HoldReason.UNRESOLVED_PHI),
        (PhiResolution.CRITICAL_UNRESOLVED, HoldReason.CRITICAL_IDENTIFIER),
    ],
)
def test_unresolved_identifier_in_any_context_withholds_the_whole_release(
    resolution, reason
):
    screener = Screener()
    screener.resolution = resolution
    with buffer(screener=screener) as target:
        finalized(target)
        finalized(target, SECOND, PRIVATE)
        assert target.release_ready() == () and target.reason is reason
        assert target.finish() == () and target.state is BufferState.ENDING
        screener.resolution = PhiResolution.RESOLVED
        released = target.release_ready()
        assert [item.segment_ref for item in released] == [FIRST, SECOND]
        assert PRIVATE not in " ".join(item.text for item in released)


def test_known_phi_cannot_reappear_when_previous_context_leaves():
    class Forgetful(Screener):
        def screen(self, request):
            result = super().screen(request)
            if len(self.calls) > 1:
                return replace(
                    result,
                    decisions=tuple(
                        replace(d, redactions=()) for d in result.decisions
                    ),
                )
            return result

    with buffer(screener=Forgetful()) as target:
        finalized(target)
        finalized(target, SECOND, PRIVATE)
        assert target.release_ready()[0].segment_ref == FIRST
        finalized(target, THIRD, "Synthetic context")
        released = target.release_ready()
        assert released[0].segment_ref == SECOND and released[0].text == "[REDACTED]"


@pytest.mark.parametrize("mode", ["stale", "missing", "revision", "span", "exception"])
def test_invalid_or_failed_screening_never_releases_or_echoes_payload(mode, caplog):
    class Broken(Screener):
        def screen(self, request):
            result = super().screen(request)
            if mode == "stale":
                return replace(result, request_digest="f" * 64)
            if mode == "missing":
                return replace(result, decisions=())
            if mode == "revision":
                return replace(
                    result, decisions=(replace(result.decisions[0], revision=0),)
                )
            if mode == "span":
                return replace(
                    result,
                    decisions=(
                        replace(
                            result.decisions[0], redactions=(PhiRedactionSpan(0, 1000),)
                        ),
                    ),
                )
            raise RuntimeError(PRIVATE)

    with buffer(screener=Broken(), trailing_segments=0) as target:
        finalized(target, text=PRIVATE)
        assert (
            target.release_ready() == () and target.reason is HoldReason.SCREEN_FAILED
        )
        assert PRIVATE not in target.to_json() + repr(target) + caplog.text


def test_old_decision_cannot_authorize_changed_context():
    class Stale(Screener):
        def screen(self, request):
            if self.calls:
                return self.previous
            self.previous = super().screen(request)
            return self.previous

    screener = Stale()
    with buffer(screener=screener) as target:
        finalized(target)
        finalized(target, SECOND, PRIVATE)
        target.release_ready()
        finalized(target, THIRD, "New context")
        assert (
            target.release_ready() == () and target.reason is HoldReason.SCREEN_FAILED
        )


@pytest.mark.parametrize("operation", ["release", "tick", "edit", "finalize"])
def test_deadline_discards_pending_text_without_release(operation):
    clock = Clock()
    with buffer(clock=clock, trailing_segments=0, max_latency_ms=100) as target:
        target.append(FIRST, PRIVATE)
        clock.seconds = 0.101
        if operation == "release":
            assert target.release_ready() == ()
        elif operation == "tick":
            target.tick()
        elif operation == "finalize":
            assert not target.finalize(FIRST, expected_revision=0)
        else:
            assert not target.edit(
                FIRST, RevisionKind.INSERT, 0, 0, "prefix", expected_revision=0
            )
        assert (
            target.state is BufferState.ABSTAINED
            and target.reason is HoldReason.LATENCY_LIMIT
        )
        report = json.loads(target.to_json())
        assert report["pending"] == [] and report["discarded"] == 1


def test_revisions_do_not_reset_the_deadline_and_slow_detector_cannot_release():
    clock = Clock()
    with buffer(clock=clock, max_latency_ms=100) as target:
        target.append(FIRST, PRIVATE)
        clock.seconds = 0.09
        target.edit(FIRST, RevisionKind.INSERT, 0, 0, "prefix ", expected_revision=0)
        clock.seconds = 0.101
        assert target.tick() is BufferState.ABSTAINED

    class Slow(Screener):
        def screen(self, request):
            clock.seconds += 0.2
            return super().screen(request)

    with buffer(
        screener=Slow(), clock=clock, trailing_segments=0, max_latency_ms=100
    ) as target:
        finalized(target, text=PRIVATE)
        assert (
            target.release_ready() == () and target.reason is HoldReason.LATENCY_LIMIT
        )


@pytest.mark.parametrize(
    "limits,second_text",
    [
        ({"max_segments": 1, "trailing_segments": 0}, "next"),
        ({"max_chars": len(PRIVATE)}, "next"),
    ],
)
def test_overflow_discards_every_pending_segment(limits, second_text):
    with buffer(**limits) as target:
        target.append(FIRST, PRIVATE)
        assert not target.append(SECOND, second_text)
        assert target.reason is HoldReason.BUFFER_LIMIT
        assert json.loads(target.to_json())["pending"] == []
        with pytest.raises(WithholdingError):
            target.release_ready()


def test_operation_limit_and_invalid_clock_fail_closed():
    with buffer(max_operations=1) as target:
        target.append(FIRST, PRIVATE)
        assert target.tick() is BufferState.ABSTAINED
        assert target.reason is HoldReason.OPERATION_LIMIT
    clock = Clock()
    with buffer(clock=clock) as target:
        target.append(FIRST, PRIVATE)
        clock.seconds = -0.1
        assert target.tick() is BufferState.ABSTAINED
        assert target.reason is HoldReason.CLOCK_INVALID


def test_overlapping_redactions_merge_and_offsets_cover_released_text():
    class Overlap(Screener):
        def screen(self, request):
            result = super().screen(request)
            return replace(
                result,
                decisions=(
                    replace(
                        result.decisions[0],
                        redactions=(PhiRedactionSpan(1, 4), PhiRedactionSpan(3, 6)),
                    ),
                ),
            )

    with buffer(screener=Overlap(), trailing_segments=0) as target:
        finalized(target, text="abcdefgh")
        result = target.release_ready()[0]
        assert result.text == "a[REDACTED]gh"
        assert [
            (m.source_start, m.source_end, m.redacted) for m in result.mappings
        ] == [(0, 1, False), (1, 6, True), (6, 8, False)]
        assert result.mappings[-1].output_end == len(result.text)


def test_context_exit_drops_pending_text_and_ended_segments_cannot_be_revised():
    with buffer() as target:
        finalized(target, text=PRIVATE)
        with pytest.raises(ValueError):
            target.edit(
                FIRST, RevisionKind.DELETE, 0, len(PRIVATE), expected_revision=1
            )
    assert target.state is BufferState.CLOSED
    assert json.loads(target.to_json())["pending"] == []


def test_deterministic_replay_and_no_network_or_plaintext_in_reports(
    monkeypatch, caplog
):
    def deny(*args, **kwargs):
        pytest.fail("unexpected network")

    monkeypatch.setattr(socket, "socket", deny)
    outputs = []
    for _ in range(2):
        with buffer(trailing_segments=0) as target:
            finalized(target, text=PRIVATE + " has no allergy")
            result = target.finish()[0]
            assert result.text == "[REDACTED] has no allergy"
            outputs.append((result.to_dict(), target.to_json()))
            assert (
                PRIVATE
                not in repr(result)
                + json.dumps(result.to_dict())
                + target.to_json()
                + caplog.text
            )
    assert outputs[0] == outputs[1]


def test_remote_detector_declaration_is_rejected_before_receiving_text():
    class Remote(Screener):
        @property
        def declaration(self):
            return PhiScreenDeclaration(artifact_digest="a" * 64, requires_network=True)

    with pytest.raises(WithholdingError, match="^invalid speech withholding contract$"):
        buffer(screener=Remote())


def test_expiry_during_redaction_discards_the_entire_prepared_release(monkeypatch):
    import openmed.multimodal.streaming.privacy.withholding_buffer as implementation

    clock = Clock()
    original = implementation._redact

    def advance_while_rendering(*args):
        result = original(*args)
        clock.seconds += 0.2
        return result

    monkeypatch.setattr(implementation, "_redact", advance_while_rendering)
    with buffer(clock=clock, trailing_segments=0, max_latency_ms=100) as target:
        finalized(target, text=PRIVATE)
        assert target.release_ready() == ()
        report = json.loads(target.to_json())
        assert report["released"] == 0 and report["discarded"] == 1
        assert target.reason is HoldReason.LATENCY_LIMIT


def test_stale_edit_cannot_clear_current_text_by_claiming_an_overflow():
    with buffer(max_chars=30, trailing_segments=0) as target:
        target.append(FIRST, "no allergy")
        target.edit(FIRST, RevisionKind.INSERT, 0, 0, "has ", expected_revision=0)
        with pytest.raises(WithholdingError):
            target.edit(FIRST, RevisionKind.INSERT, 0, 0, "x" * 40, expected_revision=0)
        assert (
            target.state is BufferState.OPEN and target.lineage(FIRST)["revision"] == 1
        )
        target.finalize(FIRST, expected_revision=1)
        assert target.finish()[0].text == "has no allergy"
