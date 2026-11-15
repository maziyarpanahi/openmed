# Reconciling overlapping ASR windows

`openmed.multimodal.streaming.transcript.window_stitching` reconciles fixed, local ASR provider outputs into one bounded transcript edit plan. It uses the existing audio-window planner, provider declarations, transcript revision ledger, and PHI-withholding buffer. It does not run an ASR model, make a network request, download weights, or make a clinical decision.

A shared phrase is coalesced only when both windows supply the same complete, ordered overlap tokens with identical text and stream-relative start/end times. Repeated phrases outside that shared time interval remain separate. A lost negation, changed medication, timing disagreement, uncovered word, or ambiguous boundary keeps the result provisional and preserves every original hypothesis for review. Provider confidence does not override those checks.

## Source and alignment contract

Create an `AlignedWindow` for each received window. Supply its zero-based planner index, exact audio-source digest, pinned `ProviderProvenance`, original `PartialHypothesis`, explicit character offsets for every `TokenTime`, and a coverage-completeness declaration. Every window must refer to the same source and offline provider/model declaration.

Times are integer **stream-relative milliseconds**, as in the existing provider contract. Convert window-relative timing explicitly before constructing an input. The reconciler does not round, shift, or infer timing. Zero-duration or duplicate timing identities require review. Every token must fit within its planned window and match its exact original substring. All non-whitespace characters, including punctuation and negation, must be covered by those offsets. Supplying only a transcript and guessed word boundaries is insufficient evidence.

The planner output is revalidated against `plan_audio_windows`; edited metadata cannot silently redefine the overlap. This reconciler accepts at most 64 windows, 65,536 total input characters, and 4,096 total input tokens. Existing PHI-buffer limits can impose smaller output and retention budgets. The API operates on fixed, already obtained results, rather than maintaining a new long-lived streaming decoder.

Coverage completeness and provider finality are declarations from the caller's provider integration. They do not prove recognition accuracy, audio authenticity, speaker attribution, or that the provider heard every spoken word. Qualify the provider and its alignment separately. Exact matching is intentionally conservative: real models with jittered or clipped boundary timings may require human review instead of automatic merging.

## Synthetic example

This example supplies fixed outputs with one shared timed token. The first window's negation is preserved and the shared token appears only once.

```python
import hashlib

from openmed.multimodal.audio_windows import plan_audio_windows
from openmed.multimodal.streaming.speech.providers import (
    PartialHypothesis,
    ProviderProvenance,
    TokenTime,
)
from openmed.multimodal.streaming.transcript.window_stitching import (
    AlignedWindow,
    reconcile_windows,
)

plan = plan_audio_windows(3000, window_ms=2000, overlap_ms=1000)
source_digest = hashlib.sha256(b"synthetic fixed-output example").hexdigest()
provider = ProviderProvenance("synthetic-overlap", "v1", "sha256:" + "a" * 64)

first = AlignedWindow(
    window_index=0,
    source_digest=source_digest,
    provider=provider,
    hypothesis=PartialHypothesis(
        revision=0,
        text="No allergy",
        is_final=True,
        token_times=(
            TokenTime(0, 100, 300, 900000, "No"),
            TokenTime(1, 1500, 1700, 900000, "allergy"),
        ),
    ),
    character_offsets=((0, 2), (3, 10)),
    coverage_complete=True,
)
second = AlignedWindow(
    window_index=1,
    source_digest=source_digest,
    provider=provider,
    hypothesis=PartialHypothesis(
        revision=0,
        text="allergy reported",
        is_final=True,
        token_times=(
            TokenTime(0, 1500, 1700, 900000, "allergy"),
            TokenTime(1, 2200, 2500, 900000, "reported"),
        ),
    ),
    character_offsets=((0, 7), (8, 16)),
    coverage_complete=True,
)
result = reconcile_windows(
    plan,
    (first, second),
    session_ref="1" * 32,
    source_digest=source_digest,
    provider=provider,
)
assert not result.review_required
assert "".join(edit.text for edit in result.operations) == "No allergy reported"
assert result == reconcile_windows(
    plan, (first, second), session_ref="1" * 32,
    source_digest=source_digest, provider=provider,
)
```

The assertions above inspect synthetic protected text. In real workflows, keep `operations`, `alternatives`, and provider hypotheses inside the authorized local review boundary. Use only `result.to_json()` for content-free audit output. It includes closed reason codes, counts, time gaps, edit coordinates, and opaque token/segment references. It excludes source transcript text and hypothesis alternatives. Opaque references and timing can still be linkable; apply the workflow's access and retention policy.

## Ambiguity, gaps, and replay

The caller's arrival order is preserved. Missing windows, out-of-order windows, duplicates, and uncovered audio tails produce explicit reason codes and gap intervals where applicable. The reconciler does not silently sort, infer missing audio, or invent continuity. A provider-declared silent overlap can be accepted only when both complete window results agree on its empty token set. Silence is not inferred from a missing window.

On the first unresolved seam, only the known coherent prefix becomes provisional edits; all original windows remain in `alternatives`. No finalization event is emitted while any window or boundary requires review. Non-overlapping word fragments are not glued together when the provider supplied no separator. Existing whitespace is retained, including harmless duplicate whitespace at a seam, rather than normalizing clinical content.

The segment identity binds the opaque session reference, source digest, provider/model declaration, and complete window plan. Replaying the same inputs produces identical edits and token references. Token references also bind the originating window, provider hypothesis revision, token index, original character offsets, and timing; duplicate overlap tokens retain their original lineage. A changed provider/model or source is a distinct segment binding. The same audio decoded again can still be correlated evidence and is not an independent clinical sample.

A corrected provider result must be reconciled again before release. The API does not mutate an earlier finalized transcript or retroactively repair its citations. Preserve any already released evidence and use an explicit higher-level correction/review workflow when necessary.

## Revision and PHI release

Edits use the existing `RevisionKind.INSERT` and `RevisionKind.FINALIZE`, expected revision counters, and fragment-local `TranscriptToken` coordinates. They replay through `RevisionLedger` without introducing another lineage model. Finalization seals the transcript; it does not assert that its content is PHI-free.

`result.release_with(local_screener, policy=..., clock=...)` recomputes the exact reconciliation, refuses altered decision/edit objects, creates a fresh `WithholdingBuffer`, applies the edits, and asks that buffer to finish. It returns existing protected `ReleasedSpeechSegment` objects only when the whole transcript is finalized and local PHI screening resolves every required identifier. Incomplete or ambiguous transcripts remain provisional and return no release. Critical or unresolved PHI, screening failure, retention expiry, or buffer exhaustion also prevent release. The optional clock is for controlled local evaluation; ordinary calls use the existing monotonic retention clock.

The following continuation deliberately withholds even a coherent transcript pending privacy review:

```python
from openmed.multimodal.streaming.privacy import (
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
)

class HoldForPrivacyReview:
    declaration = PhiScreenDeclaration(artifact_digest="b" * 64)

    def screen(self, request):
        return PhiScreenResult(
            request_digest=request.request_digest,
            decisions=tuple(
                SegmentPhiDecision(
                    segment_ref=segment.segment_ref,
                    revision=segment.revision,
                    resolution=PhiResolution.UNRESOLVED,
                )
                for segment in request.segments
            ),
        )

assert result.release_with(HoldForPrivacyReview(), clock=lambda: 0) == ()
```

Supply a qualified local PHI screener for actual use; the library cannot establish detector recall from a declaration. Known-marker fixtures test plumbing only. No identity text, transcript, or source audio is logged or written by this reconciler. Its temporary withholding buffer closes after each call; caller-owned alternatives and returned protected text remain caller-owned and are not erased by that close. Python memory handling does not guarantee secure zeroization.

## Offline regression controls

The fixtures in `tests/fixtures/multimodal/asr_overlap.py` contain fixed synthetic provider outputs, not recordings or model weights. Unit controls cover repeated phrases, seam negation, medication/timing conflict, missing/reordered windows, source/provider mismatch, incomplete alignment, forged finalization, and buffer limits. Integration controls exercise the real ledger and withholding buffer, including a synthetic identifier split across the overlap and sealed citation lineage.

```bash
python -m pytest \
  tests/unit/multimodal/streaming/transcript/test_window_stitching.py \
  tests/integration/multimodal/test_asr_overlap_withholding.py -q
```
