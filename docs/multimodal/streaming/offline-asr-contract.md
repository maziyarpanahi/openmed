# Offline Streaming ASR Provider Contract

Ambient dictation has to swap local speech engines without rewriting the
surrounding application. `openmed.multimodal.streaming.speech.providers` fixes
that surface: an engine declares its provenance, languages, sample rates and
buffering limits, and a session accepts typed audio chunks while tracking partial
hypotheses, finalized segments, token times and language confidence.

Importing the contract loads no model, opens no socket and reads no audio. Network-dependent declarations are rejected. This is a declaration and session-accounting boundary, not a network sandbox or an ASR implementation; adapters must independently enforce offline execution, decode audio and own bounded buffers. No real speech engine or clinical workflow is qualified by these synthetic tests.

## Provider declaration

A declaration is immutable and validated when it is built. The provenance carries
the adapter revision and the weights fingerprint, so a session report can name
which engine produced it without naming a patient.

```python
from openmed.multimodal.streaming.speech import (
    ProviderProvenance,
    RegisteredSpeechProvider,
    SpeechProviderDeclaration,
    SpeechStreamLimits,
    register_speech_providers,
)

declaration = SpeechProviderDeclaration(
    provenance=ProviderProvenance(
        provider_id="parakeet-local",
        provider_revision="1.1.0",
        model_fingerprint="sha256:" + "0" * 64,
    ),
    languages=("en",),
    sample_rates_hz=(16000,),
    limits=SpeechStreamLimits(
        max_buffered_chunks=64,
        max_buffered_byte_count=8 * 1024 * 1024,
    ),
)

registrations = register_speech_providers(
    [
        RegisteredSpeechProvider(
            declaration=declaration,
            entrypoint="openmed.multimodal.streaming.speech.local_adapter.build",
        )
    ]
)
```

`register_speech_providers` validates each registration, rejects duplicate
provider identifiers, and returns the registrations sorted by provider id. It
never imports the entry point, so discovery stays offline and side-effect free.

## Session lifecycle

One session drives one declaration. Chunk sequences start at zero and increase by
one, hypothesis revisions increase within an utterance and restart after a finalized segment, segment indexes increase across the stream, and finalization and cancellation are terminal.

All token and segment times are stream-relative half-open millisecond intervals. Tokens have non-decreasing start and end times, finalized tokens must fit inside their segment, and emitted times cannot exceed audio already accepted by the session. Draining does not reset the stream clock.

```python
from openmed.multimodal.streaming.speech import (
    AudioChunk,
    FinalizedSegment,
    PartialHypothesis,
    StreamingAsrSession,
)

session = StreamingAsrSession(declaration)
session.push_chunk(
    AudioChunk(
        sequence=0,
        payload=b"\x00\x00" * 512,
        sample_rate_hz=16000,
        channel_count=1,
        duration_ms=32,
    )
)
session.submit_partial(
    PartialHypothesis(revision=0, text="patient reports", is_final=False)
)
released = session.drain()
report = session.finalize(
    FinalizedSegment(
        segment_index=0,
        text="patient reports",
        start_ms=0,
        end_ms=32,
    )
)
print(report.to_json())
```

`drain()` acknowledges how many chunks the adapter has released and resets the buffered counters without closing the session. The contract retains no audio or transcript payloads; the adapter must actually release its buffers before acknowledging the drain. `cancel()` closes the session with the reason code `session_cancelled`; the adapter must stop inference and release its owned buffers. Session counters remain available for final accounting and may be drained after closure.

| State | Meaning | Further input |
| --- | --- | --- |
| `open` | Chunks, hypotheses and segments are accepted. | Allowed |
| `finalized` | `finalize()` completed the stream. | `session_closed` |
| `cancelled` | `cancel()` ended the stream early. | `session_cancelled` |

## Limits and validation rules

| Field or limit | Contract |
| --- | --- |
| `provider_id` | Lowercase bounded label matching `^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$`. |
| `provider_revision` | At most 64 characters, alphanumeric at both ends. |
| `model_fingerprint` | `sha256:` followed by 64 lowercase hexadecimal characters. |
| `requires_network` | Must be `False`; a network-only engine raises `provider_requires_network`. |
| `languages` | Non-empty sorted unique tuple, at most 64 tags of up to 63 characters, such as `en` or `en-US`. Hypotheses and segments may only report a declared language. |
| `sample_rates_hz` | Non-empty sorted unique tuple between 1 and 768000. Each chunk must use a declared rate. |
| Chunk payload | `bytes` between 1 byte and 8 MiB; the declared sequence must be contiguous from zero. |
| Chunk duration | Between 1 ms and 60000 ms. |
| Buffered chunks, bytes, milliseconds | Provider defaults are 4096, 512 MiB and 3600000 ms. A session override can only tighten provider limits; exceeding a buffer limit raises `buffer_limit_exceeded`. |
| Stream duration | Defaults to 86400000 ms; exceeding it raises `stream_duration_exceeded`. |
| Partial hypotheses | Revision must strictly increase within the current utterance; the session-wide count stays within `max_partial_hypotheses`. |
| Finalized segments | Index must strictly increase, `start_ms` must not go backwards, and the count stays within `max_segments`; otherwise `segment_index_out_of_order`, `segment_times_out_of_order` or `segment_limit_exceeded`. |
| Token times | Non-decreasing stream-relative `start_ms` and `end_ms`, `token_index` strictly increasing, confidence in parts per million. Finalized tokens stay inside their segment and submitted timing stays within received audio. |
| Language confidence | Tag matches the bounded language form, confidence in parts per million. |

## Deterministic output and privacy boundary

`StreamingAsrSessionReport.to_json()` is byte-identical for identical input: the
dictionary order is fixed by the report field tuple and the JSON is emitted with
sorted keys and compact separators.

Audio payloads and recognized text never travel with diagnostics. `AudioChunk`,
`PartialHypothesis`, `FinalizedSegment` and `TokenTime` exclude their payload and
text from `repr`, their `to_dict()` methods report lengths and counts instead of
content, and every failure raises `SpeechProviderError` whose message is a
category string such as `buffer_limit_exceeded`. A session report names the
provider, its revision, model fingerprint, lifecycle state and counters. Provider labels must be static asset identifiers, never patient or site identifiers; syntactic validation does not de-identify caller-supplied labels. Fingerprints are linkage metadata and require appropriate access and retention controls.

## See also

- [Streaming audio windows](../audio-windows.md)
- [ASR audio profiles](../asr-audio-profiles.md)
- [OCR through memory](../ocr-streaming.md)
- Issue [#2812](https://github.com/maziyarpanahi/openmed/issues/2812)
