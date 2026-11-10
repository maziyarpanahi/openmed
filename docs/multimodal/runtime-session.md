# Local multimodal runtime sessions

`openmed.multimodal.MultimodalSession` composes the existing asset preflight,
image-provider and streaming-ASR contracts for application callers. It supports
declared PNG/JPEG/WebP images, PDFs and a stream of complete PCM WAV chunks.
Providers are injected local adapters. The session downloads no models, chooses
no remote fallback, writes no files and emits no payload logs.

This is Python runtime composition. It does not qualify a model, perform
de-identification, authorize a disclosure or make clinical decisions. Swift
provider mirroring and native reference workflows remain separate milestone
work. Use the existing privacy and human-review gates around protected outputs.

## Operations and admission

| Operation | Provider | Successful protected output |
| --- | --- | --- |
| `infer_image(asset, prompt)` | Existing `EdgeVLMProvider` | Existing `VLMResult`, with its non-diagnostic and review requirements |
| `extract_document(asset)` | `DocumentProvider.extract(data, limits=..., cancellation=...)` | Bounded `ExtractedDocument`, copied spans and metadata |
| `push_audio(asset, sequence=N)` | `SpeechExecutionProvider.push(chunk, cancellation=...)` | Tuple of validated partial hypotheses and finalized segments |
| `finish_audio()` | `SpeechExecutionProvider.finish(cancellation=...)` | Validated final emissions; the audio stream cannot restart |

Each `AssetHandle` carries an immutable `AssetManifest`, a readable binary
source and explicit ownership. The session first checks supported media and
declared resource budgets, then reads a bounded immutable snapshot. It applies
the existing digest, size, signature, metadata and resource checks to that exact
snapshot. The provider receives the same bytes, so changes to the original
source cannot replace the checked input.

Images additionally need matching bounded header geometry. WAV chunks need
matching actual duration and RIFF length; sequence numbers start at zero and
advance only when the streaming contract accepts a chunk. Each WAV chunk must
be a complete file, not an arbitrary fragment from a microphone callback.

PDF manifests cannot express pixel geometry. The session resolves only the two
missing PDF pixel-limit findings using the existing bounded PDF geometry reader
and `pdf_dpi`; any other preflight failure still abstains. Parsed page count must
match the manifest, and estimated per-page and total raster sizes must fit the
profile. Geometry inspection is not full PDF conformance or hostile-decoder
validation. The document adapter receives all `SessionLimits` and must enforce
actual render dimensions, page counts, allocation and output ceilings during
decoding, including renderer-specific scaling. It must release temporary
resources on success, error and cooperative cancellation.

The shared `SessionLimits` default to a 32 MiB encoded snapshot, 1 GiB cumulative
declared input, 1,024 admitted operations, 65,536 output characters, 256 output
items and 150 DPI PDF rendering. Limits can be lowered. Document text, metadata
keys and metadata string values share the character budget; metadata depth and
node counts are also bounded. These are admission ceilings, not measured peak
process memory. Adapters remain responsible for bounded decoder/model buffers.

## Ownership, stream cleanup and cancellation

An owned asset closes on success, abstention, cancellation and rejected calls to
a closed, busy or exhausted session. A borrowed seekable source returns to its
original position; a borrowed nonseekable source is consumed and stays open.
Reads start at the current position. Do not share active handles across callers.

`ProviderLease(provider=..., owned=True, release=..., cancel=...)` owns a reusable
provider until session termination. Its release hook runs exactly once. A
borrowed lease is never closed by the session. To share one resource across
modalities, pass the same lease object; distinct leases for the same object are
rejected. Hook failures are counted without copying exception text, and cleanup
continues for other resources. A failed source close withholds that result.

Speech adapters must release each input payload before returning from `push`,
retain only declaration-bounded stream state, release that state in `finish`,
and implement idempotent `abort()` for stream failure or session termination.
The runtime calls `abort` even for a borrowed adapter: this releases the stream
it started, not the shared model. A failed stream cannot resume; independent
document and image operations can continue. An asset or sequence rejected
before adapter execution does not abort an otherwise healthy stream.

Operations are synchronous, with one in-flight call and no hidden queue.
Successful, abstained and cancelled operations produce ordered events.
Concurrent calls, exhausted operation history and terminal sessions raise a
fixed-message `RuntimeSessionError`; their owned input handles still close.
Use the context manager or explicitly call `close()` on every session.

Another thread can call `cancel()` or `close()`. Both set the cooperative event;
an optional lease `cancel` hook should be nonblocking and may run alongside an
active adapter. Cancellation wins against success at the event-commit lock. A
success already committed before cancellation remains a success; no later
success is committed. An active call that loses that race returns `CANCELLED`
with no protected output.

The state remains `CANCELLING` until active work and cancellation hooks unwind
and owned cleanup finishes. Resources are not closed while a provider still
uses them. Arbitrary Python code or a blocking source cannot be forcibly
interrupted; adapters, sources and hooks are trusted local code and must
cooperate. Provider declarations do not sandbox network access. Qualification
must independently verify offline behavior and disabled telemetry.

## Protected results and audit records

`session.events`, `session.to_json()` and `result.to_json()` expose ordered
operation codes, digests, byte/count metadata, measured durations, typed
abstentions and cleanup-failure counts. They omit asset IDs, private paths,
prompts, document text, transcript text and provider exception messages. The
session does not retain successful protected outputs in its event history.

`result.reviewed_output(reviewer_confirmed=True)` returns protected content for
application processing only after explicit review confirmation. It does not
de-identify that content. Keep the returned document, VLM draft and transcript
out of logs and unprotected persistence. Do not use generic dataclass dumping
or traceback-local capture on these objects. Digests remain linkage metadata
and require access controls. Dropping references is not cryptographic memory
zeroization; caller-held outputs remain the caller's responsibility.

## Runnable synthetic audio example

This exercises admission, execution, finalization and deterministic event order
without an installed speech engine. The adapter holds no audio state.

```python
import hashlib
import io
import wave

from openmed.multimodal import (
    AssetHandle, AssetManifest, MultimodalSession, ProviderLease, SessionOutcome,
)
from openmed.multimodal.streaming.speech.providers import (
    FinalizedSegment, PartialHypothesis, ProviderProvenance,
    SpeechProviderDeclaration,
)


class SyntheticSpeech:
    declaration = SpeechProviderDeclaration(
        provenance=ProviderProvenance(
            provider_id="synthetic", provider_revision="1",
            model_fingerprint="sha256:" + "a" * 64,
        ),
        languages=("en",), sample_rates_hz=(16000,),
    )

    def push(self, chunk, *, cancellation):
        return (PartialHypothesis(revision=chunk.sequence, text="Synthetic audio"),)

    def finish(self, *, cancellation):
        return (FinalizedSegment(
            segment_index=0, text="Synthetic audio", start_ms=0, end_ms=100,
        ),)

    def abort(self):
        pass  # No retained buffers in this fixed provider.


encoded = io.BytesIO()
with wave.open(encoded, "wb") as writer:
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(16000)
    writer.writeframes(b"\0\0" * 1600)
data = encoded.getvalue()
manifest = AssetManifest(
    asset_id="synthetic", media_type="audio/wav", byte_size=len(data),
    sha256=hashlib.sha256(data).hexdigest(), duration_seconds=0.1,
)
handle = AssetHandle(manifest=manifest, source=io.BytesIO(data), owned=True)
with MultimodalSession(speech=ProviderLease(provider=SyntheticSpeech())) as session:
    first = session.push_audio(handle, sequence=0)
    final = session.finish_audio()
    assert first.event.outcome is final.event.outcome is SessionOutcome.SUCCESS
    assert [event.sequence for event in session.events] == [0, 1]
assert handle.source.closed
```

The offline tests use valid synthetic media and fixed providers, including
independent failures, rejected input, ownership, stream cleanup, value-free
reports, denied network access and cancellation races:

```sh
.venv/bin/python -m pytest tests/unit/multimodal/test_runtime_session.py tests/integration/test_multimodal_runtime_session.py -q
```
