"""Offline generated waveforms through an injected detector and offset sink."""

import math

import pytest

from openmed.multimodal.speech_endpoints import Activity, SpeechEndpointAdapter


@pytest.mark.integration
def test_generated_waveforms_emit_repeatable_bounded_review_events():
    class SyntheticDetector:
        requires_network = False

        def detect(self, samples):
            energy = sum(s * s for s in samples) / len(samples)
            if energy > 0.1:
                return Activity.SPEECH
            return Activity.UNCERTAIN if energy else Activity.SILENCE

        def reset(self):
            pass

    def run():
        stream = SpeechEndpointAdapter(
            SyntheticDetector(),
            hangover_samples=16,
            max_utterance_samples=64,
            max_frame_samples=16,
        )
        result = []
        for frame in range(20):
            amplitude = 0 if frame < 2 or frame >= 16 else (0.8 if frame < 14 else 0.05)
            waveform = tuple(
                amplitude * math.sin(2 * math.pi * i / 8) for i in range(16)
            )
            # Deliberately dropped frame: retain absolute source coordinates.
            if frame != 7:
                result.extend(stream.push(frame * 16, waveform))
            assert stream.buffered_sample_count == 0
        result.extend(stream.finish())
        return result

    events = run()
    assert events == run()
    ends = [e for e in events if e.kind == "speech_end"]
    assert [(e.start_sample, e.end_sample, e.reason) for e in ends] == [
        (32, 96, "maximum_duration"),
        (96, 112, "discontinuity"),
        (128, 192, "maximum_duration"),
        (192, 240, "hangover"),
    ]
    assert all(0 < e.end_sample - e.start_sample <= 64 for e in ends)
    assert all(e.reviewer_confirmation_required for e in events)
    assert [e.start_sample for e in events if e.kind == "uncertain_activity"] == [
        224,
        240,
    ]
