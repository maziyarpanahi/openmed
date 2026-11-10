# Speech PHI withholding

`openmed.multimodal.streaming.privacy.WithholdingBuffer` holds provisional speech
in bounded local memory. It releases an ordered prefix only after the pending
context is finalized and an injected local detector resolves its identifiers.
Known identifier spans are replaced with the fixed `[REDACTED]` marker. Nothing
chooses a remote detector or sends audio or transcript text to a service.

This is a release gate, not an ASR engine, detector qualification, compliance
certification or clinical decision system. The caller must qualify the detector
for direct-identifier recall, critical leakage and the languages it processes.
An incorrect detector decision can still miss PHI. Treat released text as
protected candidate output, with the applicable review and disclosure controls.

## Revision and context rules

`append` starts a provisional segment with an opaque 32-character hexadecimal
reference, protected text and optional `TranscriptToken` coordinates. `edit`
delegates insert/delete/replace operations to the existing `RevisionLedger`
while applying the same operation to its protected text. Replacement token
coordinates are fragment-local and their references must be fresh. Stale edits
fail without changing text. `finalize` seals the existing ledger; finalization
alone never releases a segment, and finalized evidence cannot be silently edited.

`release_ready()` constructs a fresh `PhiScreenRequest` covering **all pending
segments**. It fingerprints the session, policy, detector artifact, end-of-stream
state, exact revisions and protected text digests. The detector must return one
ordered decision for every segment, with the matching request fingerprint and
revision. Missing, stale, out-of-range or failed responses hold all text.

All pending context must be finalized before screening can authorize release.
An unresolved or critically unresolved identifier anywhere in that context
withholds the entire candidate prefix. Decisions are never inferred from a
confidence score, a timeout, missing detector output or an empty exception.

The policy keeps the last `trailing_segments` segments as lookahead context
(one by default). Every valid response contributes known redaction spans for
each finalized segment. Those spans remain mandatory when previous context
leaves the buffer: later detector omissions cannot expose a previously located
identifier. Overlapping spans merge before rendering. This bounded lookahead
does not prove that arbitrarily distant future speech cannot change meaning;
choose and qualify the context policy for the capture workflow.

`finish()` marks end of input and removes the lookahead hold. It does not invent
finalization or a safe decision. Unresolved content stays held until a valid
response, expiry or explicit close; provisional segments can still be revised
and finalized while ending, but new segments cannot be appended. A completely
drained ending buffer becomes closed.

## Detector contract

Implement `LocalPhiScreener` with a `PhiScreenDeclaration` and `screen(request)`.
The declaration binds a lowercase SHA-256 covering the exact detector/model
and privacy policy, and rejects `requires_network=True`. No model or network
client is initialized by the buffer. The detector remains caller-owned.

A `SegmentPhiDecision` has a closed `PhiResolution`:

| Resolution | Meaning |
| --- | --- |
| `RESOLVED` | Complete coverage for this revision and context; every detected identifier has complete source redaction spans. An empty span list explicitly asserts that no identifier needs redaction. |
| `UNRESOLVED` | Context or privacy classification remains incomplete; withhold. |
| `CRITICAL_UNRESOLVED` | A critical identifier or fragment remains unresolved; withhold. |

Use half-open Unicode code-point offsets in `PhiRedactionSpan`. Cross-segment
identifiers require complete spans in every affected segment. Do not return
`RESOLVED` for an incomplete name, identifier or uncertain boundary. The runtime
checks shape, bounds, identity and revision binding; it cannot verify detector
recall from those declarations. Adapters are trusted local code, must disable
logging and telemetry, and must not retain protected request objects after use.

## Bounds, latency and cleanup

The default policy permits 16 pending segments, 32,768 held characters, 4,096
operations and a 5,000 ms maximum age from first insertion. Revisions do not
restart that clock. Overflow or deadline expiry closes the buffer as
`ABSTAINED`, discards every pending protected segment and releases nothing.
Previously returned outputs cannot be recalled and remain caller-owned.

Every mutation and release checks a trusted monotonic clock. Screening and
redaction time count toward the deadline; a final check runs before committing
the prepared release. A failing or backwards clock also discards pending text.
Inject a fake clock for deterministic tests, without performance thresholds.

There is no background scheduler. Call `tick()` while capture is idle to enforce
retention deadlines, and always use the context manager or `close()`. A blocked
detector cannot be forcibly interrupted by this synchronous buffer, but its late
result cannot release expired text. The buffer performs no disk writes, keeps
no history of released text, and drops pending references on close. This is not
cryptographic zeroization of Python strings or copies held by the caller.

## Results and diagnostics

Released segments contain protected text, a content-free lineage export,
request/output digests and `RedactionMapping` source-to-output intervals.
Replacement markers change output lengths: use these mappings to connect
released offsets to original finalized transcript coordinates. Do not reuse
original offsets directly against redacted text.

`ReleasedSpeechSegment.to_dict()` and `buffer.to_json()` omit text and exception
payloads. Reports contain closed codes, counts, offsets, opaque references and
digests. Object representations also omit protected text. Do not use generic
dataclass serialization or capture traceback locals; fingerprints remain
sensitive linkage metadata and need access controls.

## Runnable fixed-detector example

This synthetic detector recognizes only one fixture marker. It exercises the
buffer, not real PHI recognition. Replace it with a qualified local detector.

```python
from openmed.multimodal.streaming.privacy import (
    PhiRedactionSpan, PhiResolution, PhiScreenDeclaration, PhiScreenResult,
    SegmentPhiDecision, WithholdingBuffer,
)


class FixtureDetector:
    declaration = PhiScreenDeclaration(artifact_digest="a" * 64)

    def screen(self, request):
        decisions = []
        for segment in request.segments:
            marker = "SYNTHETIC-ID"
            start = segment.text.find(marker)
            decisions.append(SegmentPhiDecision(
                segment_ref=segment.segment_ref, revision=segment.revision,
                resolution=PhiResolution.RESOLVED,
                redactions=(PhiRedactionSpan(start, start + len(marker)),)
                if start >= 0 else (),
            ))
        return PhiScreenResult(
            request_digest=request.request_digest, decisions=tuple(decisions),
        )


with WithholdingBuffer("0" * 32, FixtureDetector()) as held:
    held.append("1" * 32, "SYNTHETIC-ID has no allergy")
    assert held.release_ready() == ()
    held.finalize("1" * 32, expected_revision=0)
    assert held.release_ready() == ()  # Retain trailing context until input ends.
    result = held.finish()[0]
    assert result.text == "[REDACTED] has no allergy"
```

Run the offline controls with:

```sh
.venv/bin/python -m pytest tests/unit/multimodal/streaming/privacy/test_withholding_buffer.py -q
```
