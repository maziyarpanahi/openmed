"""Synthetic stereo PCM exercises real final screening, muting and destruction."""

import hashlib
import json
import struct
from dataclasses import replace

import pytest

from openmed.multimodal.streaming.privacy import (
    ArtifactKind,
    AudioRetentionError,
    AudioRetentionPolicy,
    AudioRetentionSession,
    DestructionAction,
    ExpectedAudioArtifact,
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
    WithholdingBuffer,
)
from openmed.multimodal.streaming.privacy.audio_muting import (
    AudioMutingError,
    RetainedClipAlignment,
    TimingEvidence,
    mute_retained_clip,
    screened_segment_digest,
)
from openmed.multimodal.streaming.transcript import TranscriptToken

SESSION, SOURCE, DEST, SEG, WORD, OTHER = (f"{i:032x}" for i in range(6))
TEXT = "Say Ivy and May!"
# A deterministic nonzero square wave on both channels, at 44.1 kHz for 100 ms.
PCM = b"".join(
    struct.pack("<hh", (i % 2 * 2 - 1) * 2000, (i % 4 // 2 * 2 - 1) * 1000)
    for i in range(4410)
)


class Screen:
    declaration = PhiScreenDeclaration(artifact_digest="a" * 64)

    def screen(self, request):
        (row,) = request.segments
        return PhiScreenResult(
            request_digest=request.request_digest,
            decisions=(
                SegmentPhiDecision(
                    segment_ref=row.segment_ref,
                    revision=row.revision,
                    resolution=PhiResolution.RESOLVED,
                    redactions=(PhiRedactionSpan(4, 7), PhiRedactionSpan(12, 15)),
                ),
            ),
        )


def released(tokens=None):
    if tokens is None:
        tokens = (
            TranscriptToken(token_ref=WORD, start=4, end=7, start_ms=510, end_ms=520),
            TranscriptToken(
                token_ref=OTHER, start=12, end=15, start_ms=524, end_ms=533
            ),
        )
    with WithholdingBuffer(SESSION, Screen(), clock=lambda: 0.0) as buf:
        assert buf.append(SEG, TEXT, tokens)
        assert buf.finalize(SEG, expected_revision=0)
        (row,) = buf.finish()
    return row


def alignment(row, **changes):
    return replace(
        RetainedClipAlignment(
            session_ref=SESSION,
            source_sha256=hashlib.sha256(PCM).hexdigest(),
            screened_sha256=screened_segment_digest(row),
            source_rate_hz=44100,
            target_rate_hz=16000,
            source_frames=4410,
            channels=2,
            stream_start_ms=500,
            timing=TimingEvidence.TOKENS,
        ),
        **changes,
    )


def retained(**kwargs):
    s = AudioRetentionSession(
        SESSION,
        tuple(ExpectedAudioArtifact(r, ArtifactKind.BUFFER) for r in (SOURCE, DEST)),
        automatic_expiry=False,
        **kwargs,
    )
    s.capture(SOURCE, PCM)
    return s


def test_stereo_resampled_outward_padding_and_bounds_receipts():
    row = released()
    with retained() as session:
        report = mute_retained_clip(
            session, SOURCE, DEST, alignment(row), row, padding_ms=3
        )
        # 10..20 and 24..33 ms, rounded out via 16 kHz then padded by ceil(132.3).
        assert report.ranges == ((441 - 133, 1456 + 133),)
        lo, hi = report.ranges[0]
        assert report.muted_frames == hi - lo and report.span_count == 2
        expected = PCM[: lo * 4] + bytes((hi - lo) * 4) + PCM[hi * 4 :]
        assert session.infer(DEST, bytes) == expected
        assert session.infer(SOURCE, bytes) == PCM
        assert report.output_sha256 == hashlib.sha256(expected).hexdigest()
        public = json.loads(report.to_json())
        assert set(public) == {
            "ranges",
            "span_count",
            "muted_frames",
            "source_sha256",
            "output_sha256",
            "screened_sha256",
        }
        assert "Ivy" not in report.to_json() and "May" not in repr(report)
    audit = session.audit()
    assert audit.passed
    assert {r.artifact_ref for r in audit.receipts} == {SOURCE, DEST}
    assert {r.binding_digest for r in audit.receipts} == {audit.binding_digest}
    assert all(r.action is DestructionAction.BUFFER_OVERWRITTEN for r in audit.receipts)


def test_explicit_whole_segment_alignment_mutes_all_channels_even_without_tokens():
    row = released(())
    with retained() as session:
        report = mute_retained_clip(
            session,
            SOURCE,
            DEST,
            alignment(row, timing=TimingEvidence.SEGMENT),
            row,
            padding_ms=0,
        )
        assert report.ranges == ((0, 4410),)
        assert session.infer(DEST, bytes) == bytes(len(PCM))
        assert session.infer(SOURCE, bytes) == PCM


@pytest.mark.parametrize(
    "broken",
    [
        "uncertain",
        "missing_last",
        "partial_token",
        "zero_time",
        "out_of_clip",
        "provisional",
        "revision",
        "mappings",
        "other_screen",
        "offset",
        "malformed",
    ],
)
def test_entire_derivative_withheld_for_unreliable_or_stale_lineage(broken):
    row = released()
    clip = alignment(row)
    data = json.loads(row.lineage_json)
    if broken in {
        "missing_last",
        "partial_token",
        "zero_time",
        "out_of_clip",
        "provisional",
        "revision",
        "malformed",
    }:
        if broken == "missing_last":
            data["tokens"].pop()
        elif broken == "partial_token":
            data["tokens"][-1]["start"] += 1
        elif broken == "zero_time":
            data["tokens"][-1]["end_ms"] = data["tokens"][-1]["start_ms"]
        elif broken == "out_of_clip":
            data["tokens"][-1]["end_ms"] = 1000
        elif broken == "provisional":
            data["finalized"] = False
        elif broken == "revision":
            data["revision"] += 1
        else:
            data["tokens"] = [{"private": "synthetic private text"}]
        row = replace(row, lineage_json=json.dumps(data))
        clip = alignment(row)
    elif broken == "uncertain":
        clip = replace(clip, timing=TimingEvidence.UNCERTAIN)
    elif broken == "mappings":
        row = replace(row, mappings=row.mappings[:-1])
        clip = alignment(row)
    elif broken == "other_screen":
        clip = replace(clip, screened_sha256="b" * 64)
    else:
        clip = replace(clip, stream_start_ms=600)
    with retained() as session:
        with pytest.raises(AudioMutingError) as error:
            mute_retained_clip(session, SOURCE, DEST, clip, row, padding_ms=1)
        assert (
            str(error.value) == "audio_muting_unverified"
            and error.value.__context__ is None
        )
        with pytest.raises(AudioRetentionError):
            session.infer(DEST, bytes)
        assert session.infer(SOURCE, bytes) == PCM


def test_non_identifier_token_before_capture_invalidates_whole_alignment():
    row = released()
    data = json.loads(row.lineage_json)
    data["tokens"].insert(
        0, dict(token_ref=SEG, start=0, end=3, start_ms=499, end_ms=505)
    )
    row = replace(row, lineage_json=json.dumps(data))
    with retained() as session, pytest.raises(AudioMutingError):
        mute_retained_clip(session, SOURCE, DEST, alignment(row), row, padding_ms=0)


def test_unknown_capture_or_insufficient_capacity_cannot_expose_partial_output():
    row = released()
    for kwargs, changes in (
        ({}, {"source_sha256": "c" * 64}),
        ({"policy": AudioRetentionPolicy(max_bytes=len(PCM))}, {}),
    ):
        with retained(**kwargs) as session:
            with pytest.raises(AudioRetentionError):
                mute_retained_clip(
                    session, SOURCE, DEST, alignment(row, **changes), row, padding_ms=0
                )
            with pytest.raises(AudioRetentionError):
                session.infer(DEST, bytes)


def test_refuses_overwrite_and_does_not_renew_or_extend_deadline():
    clock = [0.0]
    row = released()
    with retained(
        clock=lambda: clock[0], policy=AudioRetentionPolicy(max_lifetime_ms=100)
    ) as session:
        with pytest.raises(AudioRetentionError):
            mute_retained_clip(
                session, SOURCE, SOURCE, alignment(row), row, padding_ms=0
            )
        clock[0] = 0.09
        mute_retained_clip(session, SOURCE, DEST, alignment(row), row, padding_ms=0)
        assert session.infer(SOURCE, bytes) == PCM
        clock[0] = 0.1
        with pytest.raises(AudioRetentionError):
            session.infer(DEST, bytes)
        assert session.audit().passed and len(session.audit().receipts) == 2
