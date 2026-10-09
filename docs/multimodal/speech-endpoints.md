# Bounded local speech endpoints

`openmed.multimodal.speech_endpoints.SpeechEndpointAdapter` and OpenMedKit's
`SpeechEndpointAdapter` segment caller-owned, normalized mono audio with an
injected local detector. They do not capture audio, invoke ASR, identify speakers,
infer consent or make clinical decisions. No model, weights or dataset is bundled.
The detector's offline declaration is a caller-verified integration contract,
not a sandbox or provider qualification.

## Input and bounds

Supply immutable Python tuples or Swift `[Float]` frames, finite values in
`[-1, 1]`, and absolute source sample indices. Use one fixed sample rate per stream;
convert duration settings to whole samples at that rate. For example, at 16 kHz,
4,800 hangover samples mean 300 ms. Clock changes require `reset()`.

Limits are explicit: `max_utterance_samples`/`maxUtteranceSamples` is between 1
and 10,000,000; hangover is between zero and that cap; frame size is at most
65,536 samples and no larger than the utterance cap. Oversized input is rejected
before detector invocation. The adapters retain only a few counters and offsets:
`buffered_sample_count`/`bufferedSampleCount` is always zero. Returned event lists
are bounded per frame (at most six events); neither audio nor event history is
accumulated internally. The caller owns audio routing and any downstream buffers.

## Events and lifecycle

The detector classifies each entire frame as speech, silence or uncertain. Use
consistent detector frames for reproducibility; arbitrary rechunking may change
detector decisions. No confidence threshold or noise classifier is invented here.

| Event | Source sample bounds | Behavior |
| --- | --- | --- |
| `speech_start` | Equal start/end at the first speech sample | Starts on confirmed speech only |
| `speech_end` | Half-open complete utterance, including hangover | `hangover`, `maximum_duration`, `discontinuity`, `reset`, `cancelled` or `stream_end` reason |
| `uncertain_activity` | Half-open affected input frame | Always explicit; never starts speech |
| `discontinuity` | Half-open missing source interval | Ends active speech at the last received sample and resets the detector before new input |

Silence and uncertainty both count toward consecutive hangover. Confirmed speech
resets that count. A zero hangover ends at the first non-speech frame boundary.
The maximum duration includes trailing silence and cuts exactly at the cap, even
inside a frame. Continuous speech restarts at the same sample, preserving coverage
without overlap or a gap. After a cut, remaining non-speech samples are discarded.
When hangover and maximum limits tie, the end reason is `hangover`.

For gaps, the prior end and gap events precede new frame events. An uncertainty
event precedes any endpoint it causes and may extend beyond that endpoint.
Overlaps, backward indices and invalid frames are rejected without changing stream
state; callers may retry valid input. Detector exceptions are replaced by controlled
codes and cancel the stream. Exceptions and events never echo audio or detector text.

`finish()` ends active speech and releases detector state. `reset()` does the same
with an explicit reset reason and permits a new sample clock. `cancel()` discards
state and rejects input and finish until explicit reset. Detector reset failures
leave the adapter cancelled. The injected detector must discard its own audio on
reset; the adapter cannot erase caller-owned buffers or a misbehaving provider.
Call finish/reset/cancel when leaving a stream, including on application errors.

## Synthetic example

```python
from openmed.multimodal.speech_endpoints import Activity, SpeechEndpointAdapter

class SyntheticDetector:
    requires_network = False

    def detect(self, samples):
        return Activity.SPEECH if max(abs(x) for x in samples) > 0.5 else Activity.SILENCE

    def reset(self):
        pass  # This fixture detector retains no audio.

stream = SpeechEndpointAdapter(
    SyntheticDetector(), hangover_samples=3,
    max_utterance_samples=10, max_frame_samples=4,
)
start = stream.push(100, (0.8, 0.8))
end = stream.push(102, (0.0, 0.0, 0.0, 0.0))
assert [(event.start_sample, event.end_sample) for event in end] == [(100, 105)]
stream.finish()
```

This produces a start at 100 and an end spanning `[100, 105)` with reason
`hangover`. The amplitude rule is a synthetic engineering control, not a VAD model
or clinical validation. Swift uses the same state transitions through
`LocalSpeechActivityDetector.detect(_:)` and `reset()`.

Each returned event binds a non-diagnostic notice and
`reviewer_confirmation_required`/`reviewerConfirmationRequired = true`.
Downstream applications must obtain explicit reviewer confirmation for consequential
outputs; segmentation itself grants no consent or authorization. Keep audio in
caller-controlled memory, apply ASR privacy/stability gates separately, and never
log payloads or add a cloud fallback. This independent slice does not depend on
the pending streaming-ASR contract or qualify an ASR provider.
