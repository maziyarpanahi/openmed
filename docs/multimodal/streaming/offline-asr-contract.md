# Offline Streaming ASR Provider Contract

Ambient dictation has to swap local speech engines without rewriting the
surrounding application. `openmed.multimodal.streaming.speech.providers` fixes
that surface: an engine declares its provenance, languages, sample rates and
buffering limits, and a session accepts typed audio chunks while tracking partial
hypotheses, finalized segments, token times and language confidence.

The contract is offline by construction. Importing it loads no model, opens no
socket, and reads no audio, a declaration whose engine needs network access is
rejected outright, and every diagnostic carries counts instead of payload.

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
one, hypothesis revisions and segment indexes are monotonic, and finalization and
cancellation are terminal.

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

`drain()` returns how many buffered chunks were released and resets the buffered
counters, which keeps a long stream inside its memory bound without closing the
session. `cancel()` closes the session with the reason code `session_cancelled`.

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
| `languages` | Non-empty sorted unique tuple, at most 64 tags of the form `en`, `en-US`. |
| `sample_rates_hz` | Non-empty sorted unique tuple between 1 and 768000. |
| Chunk payload | `bytes` between 1 byte and 8 MiB; the declared sequence must be contiguous from zero. |
| Chunk duration | Between 1 ms and 60000 ms. |
| Buffered chunks, bytes, milliseconds | Session limits default to 4096, 512 MiB and 3600000 ms; exceeding one raises `buffer_limit_exceeded`. |
| Stream duration | Defaults to 86400000 ms; exceeding it raises `stream_duration_exceeded`. |
| Partial hypotheses | Revision must strictly increase and stay within `max_partial_hypotheses`; otherwise `hypothesis_revision_out_of_order` or `partial_hypothesis_limit_exceeded`. |
| Finalized segments | Index must strictly increase, `start_ms` must not go backwards, and the count stays within `max_segments`; otherwise `segment_index_out_of_order`, `segment_times_out_of_order` or `segment_limit_exceeded`. |
| Token times | Non-decreasing `end_ms`, `token_index` strictly increasing, confidence in parts per million. |
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
provider, its revision, the lifecycle state and the counters only.

## See also

- [Streaming audio windows](../audio-windows.md)
- [ASR audio profiles](../asr-audio-profiles.md)
- [OCR through memory](../ocr-streaming.md)
- Issue [#2812](https://github.com/maziyarpanahi/openmed/issues/2812)
