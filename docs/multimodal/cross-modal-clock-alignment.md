# Cross-modal clock alignment

`openmed.multimodal.provenance.clock_alignment` translates actual audio, transcript, ECG and image evidence into a common monotonic clock for human review. It preserves uncertainty, source identity and acquisition breaks. Translation succeeds only when the complete mapped window is covered by the declared calibration, its uncertainty fits the explicit policy, and it has a guaranteed nonempty overlap with the target evidence. Otherwise the result requires review and contains no translated timestamps.

Temporal overlap does not establish clinical agreement or approve evidence. Clock qualification, source timing and acquisition provenance remain host responsibilities. This Python implementation reads no system clock, file, network, model, audio decoder or pixel payload; it is deterministic for the same declarations and current source references. OpenMedKit integration is tracked separately in #3679.

## Source references and continuous bindings

Every adapter takes a `ClockBinding` connecting one continuous asset-time interval to a named monotonic clock and reset epoch. It maps `local_start_ns` to `monotonic_start_ns` with nominal unit slope. `timing_uncertainty_ns` bounds each endpoint, including token alignment, capture latency or source timing error. The entire uncertain source window must be inside the binding. A paused or compressed recording needs separate bindings for its continuous pieces; the library never infers missing time from file order.

| Adapter | Actual source | Time interpretation and required evidence |
| --- | --- | --- |
| `AudioPlanEvidence` | Exact member of an `AudioWindowPlan`, regenerated and compared against the original planner | Integer-millisecond boundaries over a host-declared audio source. The binding supplies its source digest; the planner does not load or authenticate audio. |
| `AudioChunkEvidence` | Actual `AudioChunk` and its `audio_chunk_digest` | Explicit local start plus the provider's declared duration. Unknown start requires review. Encoded bytes and declarations are hashed locally; sequence numbers and byte counts never establish time or PCM encoding. |
| `TranscriptEvidence` | `RevisionLedger` and its exact registered `SpanReference` | Freshly remapped token time in milliseconds. The reference must remain valid and finalized. `transcript_source_digest` identifies the opaque session/segment; the full current ledger state is also bound into the result. No transcript text is accepted. |
| `EcgEvidence` | Actual `EcgEvidenceWindows`, exact member digest and current `EcgWaveform` | Original source offset plus sample index divided by nominal rate. Full waveform fingerprint and source binding must match. Candidate and quality holds remain holds. Sample indices are not timestamps or character offsets. |
| `ImageEvidence` | Source `PAGE`, `PIXELS` or `OCR` `EvidenceNode` with its original geometry | Explicit host-recorded capture/exposure start and end plus capture-manifest digest. Missing time requires review; page numbers, frame ordinals and pixel coordinates do not establish capture time. |

Create ECG windows through [the actual builder or source-verified restoration](ecg-evidence-windows.md), not by manufacturing DTOs. Clock alignment verifies current waveform identity and exact member/candidate linkage; it does not rerun the ECG detector or authenticate saved policies. Use the original protected acquisition and policies when restoring saved evidence. Transcript access must remain serialized by the host. Finalization seals text and timing, and alignment rereads source snapshots before returning a successful result; it is not a concurrency lock.

Source times are nominal. For ECG, the declared rate determines the nominal sample grid, while the clock calibration describes oscillator drift relative to the shared wall reference. Rational sample times are rounded outwards at nanosecond boundaries. The returned window is in the **target monotonic clock**, not target asset offsets or sample indices. The host must explicitly apply the target binding and sample grid if it needs those coordinates.

## Clock epochs, anchors and drift

`ClockEpoch` identifies a clock and one reset epoch using opaque references. It carries a shared `wall_domain_digest`, a host `calibration_digest`, up to 64 strictly increasing `ClockAnchor` observations, and explicit model limits. Each anchor pairs a nominal monotonic reading with a signed wall-reference reading and an absolute `uncertainty_ns` bound. Include timestamp capture latency and shared-reference error in these bounds. Different clock epochs may be translated through qualified anchors; matching numeric readings alone are insufficient.

Monotonic time is not UTC. Python documents that its monotonic clock has an unspecified origin and that differences between readings are meaningful; its integer nanosecond API avoids float precision loss. This library neither samples that clock nor assumes two devices share its origin. [Python monotonic clock documentation](https://docs.python.org/3/library/time.html#time.monotonic)

For each uninterrupted anchor group, the model is:

```text
wall(t) = wall_origin + a * (t - monotonic_origin) + b + e(t)
1 - d <= a <= 1 + d
d = maximum_rate_error_ppb / 1_000_000_000 < 1
abs(e(t)) <= nonlinearity_ns
```

The rate tolerance and uniform nonlinearity bound must be independently qualified for the device, capture path and intended duration. They are not inferred from a small fit residual, and this implementation supplies no clinically recommended defaults. Clock-discipline literature likewise treats oscillator frequency tolerance as a worst-case assumption in uncertainty accounting; this module does not implement NTP or reuse its constants. [RFC 5905, sections 7.2 and 8](https://www.rfc-editor.org/rfc/rfc5905.html)

Each observed anchor constrains the affine prediction within `anchor.uncertainty_ns + nonlinearity_ns`. Exact rational half-plane intersection retains every feasible slope/intercept pair. No outlier anchor is discarded. An empty feasible family requires review. `calibrate_clock` exposes protected `ClockFit` objects: `drift_bounds_ppb` covers the full feasible slope interval, and `drift_estimate_ppb` is the mean vertex slope minus one, expressed in ppb. That descriptive estimate is neither a probability nor the value used to discard uncertainty during translation.

For cross-clock translation, source endpoint envelopes are mapped across every feasible source fit with the uniform nonlinearity bound added. Inversion spans every feasible positive target slope/intercept, also including target nonlinearity. Bounds are rounded outwards and both bindings' timing uncertainty is retained. No uncertainty cancellation or statistical independence benefit is assumed. An inverse outside supported integer resources requires review; bounds are never clipped to make a result pass.

## Discontinuities, coverage and decisions

`ClockBreak` records an inclusive excluded interval or a single excluded point. Supported kinds are pause, dropped chunks, wall-clock step and reset. Breaks partition anchor groups before fitting, and an evidence envelope intersecting a break is refused. After a wall-clock step, each side can still have its own qualified fit. A counter that actually runs backwards needs a new `epoch_ref`; unordered anchors are rejected rather than sorted into apparent continuity. Anchors inside exclusions and overlapping break declarations are invalid.

Cross-clock mapping requires at least two bracketing anchors in the relevant uninterrupted group for both the source window and the target evidence. The entire translated source envelope must also remain inside the target evidence's calibration group. There is no extrapolation across missing coverage. `maximum_anchor_gap_ns` bounds adjacent anchor gaps used by those windows. Sparse calibration, contradictory anchors, mismatched wall references or an ambiguous discontinuity require review.

If source and target already use the identical clock/epoch declaration, their monotonic coordinates can be compared directly without wall anchors. This identity path still checks bindings, source holds, breaks, uncertainty and overlap. It does not establish wall time. Two different declarations for the same clock/epoch are refused instead of mixing calibrations.

`AlignmentPolicy.maximum_uncertainty_ns` is a **full envelope width**, not a plus/minus error: the largest mapped source endpoint width plus the largest target-reference endpoint width must fit the limit. The guaranteed half-open intersection uses the latest possible start and earliest possible end of both windows. Mere possible overlap, an empty interval or exact endpoint contact requires review. All policy and calibration declarations are bound into the result's digests.

## Synthetic example

These synthetic clocks have a known one-second offset. The audio window at source clock 2–3 seconds maps to target clock 3–4 seconds, within the target image's declared exposure interval. The settings below are demonstration assumptions, not device qualification.

```python
from openmed.multimodal.audio_windows import plan_audio_windows
from openmed.multimodal.evidence.fused_graph import (
    EvidenceNode, NodeKind, PageGeometry,
)
from openmed.multimodal.provenance.clock_alignment import (
    AlignmentPolicy, AlignmentState, AudioPlanEvidence, ClockAnchor,
    ClockBinding, ClockEpoch, ImageEvidence, align_evidence,
)

ns = 1_000_000_000
wall = 1_700_000_000_000_000_000

def clock(index, origin):
    return ClockEpoch(
        clock_ref=f"{index:032x}", epoch_ref=f"{index + 100:032x}",
        wall_domain_digest="a" * 64, calibration_digest="b" * 64,
        anchors=(ClockAnchor(0, origin, 0),
                 ClockAnchor(10 * ns, origin + 10 * ns, 0)),
        maximum_rate_error_ppb=1_000_000, nonlinearity_ns=0, breaks=(),
    )

def binding(epoch, digest):
    return ClockBinding(
        source_digest=digest, clock_ref=epoch.clock_ref,
        epoch_ref=epoch.epoch_ref, evidence_digest="c" * 64,
        local_start_ns=0, local_end_ns=10 * ns,
        monotonic_start_ns=0, timing_uncertainty_ns=0,
    )

audio_clock, image_clock = clock(1, wall), clock(2, wall - ns)
plan = plan_audio_windows(10_000, window_ms=1_000)
source = AudioPlanEvidence(plan, plan.windows[2].window_id,
                           binding(audio_clock, "d" * 64))
node = EvidenceNode(
    node_id="1" * 32, kind=NodeKind.PIXELS, source_digest="e" * 64,
    geometry=PageGeometry(page=0, width=100, height=100),
    region=(10, 10, 30, 30),
)
target = ImageEvidence(node, 2 * ns, 5 * ns, "f" * 64,
                       binding(image_clock, "e" * 64))
policy = AlignmentPolicy(
    version="synthetic-v1", evidence_digest="0" * 64,
    maximum_uncertainty_ns=0, maximum_anchor_gap_ns=10 * ns,
)
result = align_evidence(source, target, audio_clock, image_clock, policy)
assert result.state is AlignmentState.ALIGNED
assert result.guaranteed_overlap_ns == (3 * ns, 4 * ns)
print(result.to_report_json())  # Codes and digests only.
```

## Protected output and limitations

`AlignmentResult.target_start_ns`, `target_end_ns`, `guaranteed_overlap_ns` and `uncertainty_ns` are protected review metadata, omitted from `repr` and `to_report_json`. Reports contain only controlled decisions, reasons, digests and the review notice. Calibration reports contain counts and findings, with no anchors or drift values. Source bytes, samples, transcript values, pixel values, paths and raw timestamps are not logged. Do not dump dataclasses or their protected fields into operational logs. Even digests remain potentially linkable provenance, not de-identification guarantees.

Invalid declarations raise fixed value-free `ClockAlignmentError`; valid but unusable evidence returns controlled `ClockReview` reasons. Underlying source-contract validators can independently refuse malformed source objects. Review-required results withhold all translated values. Retain the original source objects, clock declarations and policies in the host's controlled provenance store for deterministic replay; this module does not write files or provide an authenticated serializer.

An anchor fit cannot detect unrecorded clock resets, undisclosed capture pauses, an incorrect oscillator bound or false acquisition metadata. Hashes bind the supplied evidence but do not authenticate it. Temporal agreement must remain separate from clinical interpretation, source integrity, consent, privacy redaction and a clinician's assessment of whether two observations describe the same event.
