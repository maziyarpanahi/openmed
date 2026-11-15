"""Synthetic endpoint controls; no model output or clinical qualification."""

import math

import pytest

from openmed.multimodal.speech_endpoints import (
    Activity,
    EndpointError,
    SpeechEndpointAdapter,
)


class Detector:
    requires_network = False

    def __init__(self):
        self.owned = None
        self.resets = 0

    def detect(self, samples):
        self.owned = samples
        peak = max(abs(sample) for sample in samples)
        if peak > 0.5:
            return Activity.SPEECH
        return Activity.UNCERTAIN if peak else Activity.SILENCE

    def reset(self):
        self.owned = None
        self.resets += 1


def adapter(detector=None, hangover=3, maximum=10, frame=4):
    return SpeechEndpointAdapter(
        detector or Detector(),
        hangover_samples=hangover,
        max_utterance_samples=maximum,
        max_frame_samples=frame,
    )


def rows(events):
    return [(e.kind, e.start_sample, e.end_sample, e.reason) for e in events]


def test_short_speech_hangover_cuts_inside_frame_and_silence_stays_empty():
    stream = adapter()
    assert stream.push(100, (0.0,) * 4) == ()
    assert rows(stream.push(104, (0.8,) * 2)) == [
        ("speech_start", 104, 104, "detected")
    ]
    assert stream.push(106, (0.0,) * 2) == ()
    assert rows(stream.push(108, (0.0,) * 4)) == [("speech_end", 104, 109, "hangover")]
    for start in range(112, 20_112, 4):
        assert stream.push(start, (0.0,) * 4) == ()
        assert stream.buffered_sample_count == 0
    assert stream.finish() == ()


def test_continuous_speech_forced_splits_conserve_source_coverage():
    stream = adapter()
    events = []
    for start in range(71, 2071, 4):
        result = stream.push(start, (0.8,) * 4)
        assert len(result) <= 4
        events.extend(result)
    events.extend(stream.finish())
    ends = [e for e in events if e.kind == "speech_end"]
    assert [(e.start_sample, e.end_sample) for e in ends] == [
        (start, start + 10) for start in range(71, 2071, 10)
    ]
    assert all(e.reason == "maximum_duration" for e in ends)


def test_unknown_noise_never_starts_speech_and_is_explicit():
    stream = adapter()
    assert rows(stream.push(0, (0.1, -0.1, 0.1, -0.1))) == [
        ("uncertain_activity", 0, 4, "uncertain")
    ]
    stream.push(4, (0.8,) * 4)
    assert rows(stream.push(8, (0.1,) * 4)) == [
        ("uncertain_activity", 8, 12, "uncertain"),
        ("speech_end", 4, 11, "hangover"),
    ]


def test_gap_closes_speech_and_resets_detector_before_next_frame():
    detector = Detector()
    stream = adapter(detector)
    stream.push(30, (0.8,) * 4)
    assert rows(stream.push(40, (0.8,) * 4)) == [
        ("speech_end", 30, 34, "discontinuity"),
        ("discontinuity", 34, 40, "gap"),
        ("speech_start", 40, 40, "detected"),
    ]
    assert detector.resets == 1


def test_gap_without_speech_and_backward_overlap():
    stream = adapter()
    stream.push(5, (0.0,) * 4)
    assert rows(stream.push(13, (0.0,) * 4)) == [("discontinuity", 9, 13, "gap")]
    with pytest.raises(EndpointError, match="endpoint_overlapping_frame"):
        stream.push(16, (0.8,))
    assert rows(stream.push(17, (0.8,))) == [("speech_start", 17, 17, "detected")]


@pytest.mark.parametrize(
    "operation,reason",
    [("reset", "reset"), ("cancel", "cancelled"), ("finish", "stream_end")],
)
def test_release_discards_detector_audio_and_all_stream_state(operation, reason):
    detector = Detector()
    stream = adapter(detector)
    stream.push(50, (0.8,) * 4)
    assert detector.owned is not None
    assert rows(getattr(stream, operation)()) == [("speech_end", 50, 54, reason)]
    assert detector.owned is None
    assert stream.buffered_sample_count == 0
    if operation == "cancel":
        with pytest.raises(EndpointError, match="endpoint_cancelled"):
            stream.push(0, (0.8,))
        with pytest.raises(EndpointError, match="endpoint_cancelled"):
            stream.finish()
        assert stream.reset() == ()
    assert rows(stream.push(0, (0.8,))) == [("speech_start", 0, 0, "detected")]


def test_zero_hangover_and_maximum_trailing_silence_boundary():
    stream = adapter(hangover=0)
    stream.push(0, (0.8,) * 4)
    assert rows(stream.push(4, (0.0,) * 4)) == [("speech_end", 0, 4, "hangover")]
    stream = adapter(hangover=10)
    stream.push(0, (0.8,) * 4)
    stream.push(4, (0.0,) * 4)
    assert rows(stream.push(8, (0.0,) * 4)) == [
        ("speech_end", 0, 10, "maximum_duration")
    ]


def test_speech_resumes_before_hangover_and_resets_silence_count():
    stream = adapter(maximum=20)
    stream.push(0, (0.8,) * 4)
    stream.push(4, (0.0,) * 2)
    assert stream.push(6, (0.8,) * 2) == ()
    assert stream.push(8, (0.0,) * 2) == ()
    assert rows(stream.push(10, (0.0,) * 2)) == [("speech_end", 0, 11, "hangover")]


@pytest.mark.parametrize(
    "start,samples",
    [
        (-1, (0.0,)),
        (True, (0.0,)),
        (2**63 - 1, (0.0,)),
        (0, ()),
        (0, [0.0]),
        (0, (0.0,) * 5),
        (0, (True,)),
        (0, (math.nan,)),
        (0, (math.inf,)),
        (0, (1.1,)),
        (0, (10**1000,)),
        (0, ("synthetic-private-value",)),
    ],
)
def test_invalid_frames_are_value_free(start, samples):
    with pytest.raises(EndpointError) as error:
        adapter().push(start, samples)
    assert str(error.value) == "endpoint_invalid_frame"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum": 0},
        {"maximum": True},
        {"maximum": 10_000_001},
        {"hangover": -1},
        {"hangover": 11},
        {"hangover": False},
        {"frame": 0},
        {"frame": 11},
        {"frame": True},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(EndpointError, match="endpoint_invalid_limits"):
        adapter(**kwargs)


def test_rejects_network_and_missing_local_declaration():
    for value in (True, None, 0, "false"):
        detector = Detector()
        detector.requires_network = value
        with pytest.raises(EndpointError, match="endpoint_detector_not_local"):
            adapter(detector)


def test_detector_failure_sanitized_and_cancels_with_reset():
    class Broken(Detector):
        def detect(self, samples):
            self.owned = samples
            raise RuntimeError("synthetic-private-value")

    detector = Broken()
    stream = adapter(detector)
    with pytest.raises(EndpointError) as error:
        stream.push(0, (0.8,))
    assert str(error.value) == "endpoint_detector_failure"
    assert detector.owned is None
    with pytest.raises(EndpointError, match="endpoint_cancelled"):
        stream.push(1, (0.8,))


def test_invalid_detector_activity_and_reset_failure_fail_closed():
    detector = Detector()
    detector.detect = lambda _: "speech"
    with pytest.raises(EndpointError, match="endpoint_detector_failure"):
        adapter(detector).push(0, (0.8,))

    class BadReset(Detector):
        def reset(self):
            raise RuntimeError("synthetic-private-value")

    stream = adapter(BadReset())
    stream.push(0, (0.8,))
    with pytest.raises(EndpointError, match="endpoint_detector_reset_failure"):
        stream.reset()
    with pytest.raises(EndpointError, match="endpoint_cancelled"):
        stream.push(1, (0.8,))


def test_events_bind_notice_review_and_no_sensitive_payload():
    stream = adapter()
    events = stream.push(0, (0.8123456789,)) + stream.finish()
    for event in events:
        assert event.reviewer_confirmation_required is True
        assert "Non-diagnostic" in event.notice
        assert "no speaker identity, consent or clinical meaning" in event.notice
        assert "0.8123456789" not in repr(event)


def test_source_clock_near_int64_limit():
    stream = adapter()
    start = 2**63 - 5
    stream.push(start, (0.8,) * 4)
    assert rows(stream.finish()) == [("speech_end", start, 2**63 - 1, "stream_end")]
