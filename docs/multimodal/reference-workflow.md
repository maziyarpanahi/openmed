# Synthetic multimodal reference workflow

Run one offline workflow connecting image and audio preflight, actual decoding, pixel redaction, provider contracts, resource gates, transcript PHI withholding, source coordinates, evidence clocks, review and owned-buffer destruction. The default requires no model weights, microphone, network, patient data or clinical approval.

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python -m examples.multimodal.ambient_reference
uv run --frozen --extra dev pytest tests/integration/multimodal/test_reference_workflow.py -q
```

Run these commands from the repository checkout. The example imports the generated controls in `tests/fixtures/multimodal/reference_workflow.py`; examples and test fixtures are not installed as a production recorder. Orchestration lives in `openmed.multimodal.reference_workflow`. No new dependency is required beyond the development environment's Pillow installation.

## What runs

| Stage | Connected behavior |
| --- | --- |
| Consent and preflight | A synthetic participant grants consent under a digest-bound policy. Both inputs have a 64 KiB admission ceiling. Real preflight checks their media signatures, hashes and declared limits. |
| Decode | Verify the PNG's actual 64 by 48 geometry before pixel inventory. Decode exactly one second of mono PCM16 WAV at 16 kHz; refuse malformed headers, mismatched geometry and truncated samples. |
| Pixel privacy | Inspect the actual bright marker, mask its original rectangle, verify the rewritten PNG and inspect for residual markers. Metadata is excluded. The provider receives this redacted output, not the original image. |
| Visual inference | An explicitly synthetic local provider verifies the actual redacted pixels before returning its known control and box citation. The thermal gate checks fresh resources and elapsed time before and after dispatch. |
| Visual fusion | Compare the original fixture OCR reading with the provider's actual result. Retain both original-coordinate citations and the mask receipt. A disagreement propagates through the fused graph to a review-required claim. |
| Audio and PHI | Own and borrow bounded mutable PCM for the digest-bound replay callback. Validate the speech contract and its original token timing. Release the actual finalized transcript only through the PHI withholding buffer and a separate thermal screening gate. |
| Evidence clock | Align the actual audio chunk and finalized transcript span to the image exposure using an explicitly synthetic common clock. Keep hashes, original transcript offsets, redaction mappings, token times and source boxes. Temporal overlap is not clinical agreement. |
| Review | Return a content-free evidence report with an unknown speaker role and mandatory review. There is no invented role attribution, note, clinician receipt, clinical export or automated clinical write. |
| Cleanup | Close the speech and withholding components, overwrite owned audio, destroy owned image/transcript copies and return actual scoped receipts. An exception in one cleanup path does not skip the other owners. |

The PNG contains a synthetic marker and control region. The WAV is a generated tone, not speech. The visual and speech replay providers verify those inputs before returning predetermined controls. This exercises component integration; it does not measure OCR/ASR accuracy, clinical identifier recall or model quality. The marker detector, timing, memory and confidence thresholds are test controls, not recommended clinical policies. Trusted callbacks execute locally but are not sandboxed by the harness.

## Reading the result

A normal run ends with `status: review_required`, `reached_review: true`, `requires_review: true`, `complete: false` and `clinical_write_allowed: false`. Reaching review means the processing stages ran successfully. It never means the evidence is approved. Unknown speaker identity remains unknown, and no clinical note is generated from that uncertainty.

`review.visual_graph` retains both visual source nodes for each claim. The source hashes refer to the original PNG; the real pixel-redaction receipt connects it to the inference input without changing geometry. `review.transcript` maps released character intervals back to the finalized source. `review.transcript_lineage` retains the actual token and span times. The two clock-alignment receipts bind the audio and transcript to the same image evidence. Recognized strings, image bytes, audio bytes and free-form exception messages are absent from the printed report. Hashes and source metadata remain linkable; content-free does not mean anonymous.

The audio retention audit deliberately reports unresolved external artifacts. Original fixture bytes, immutable decoder/provider objects and any caller copies are outside the mutable-buffer owner. `all_data_erased` therefore remains false, even after every owned mutable buffer is overwritten. This honest limitation must not be changed into a passing all-data destruction claim. The reference does not persist media or transcripts, and does not promise physical RAM erasure, crash recovery or destruction of arbitrary native runtime caches. Production hosts need their own capture, copy inventory, provider shutdown and retention policy.

Cancellation at any stage, expiry, malformed media, incomplete redaction, unresolved PHI, missing token lineage, resource degradation, provider failure or cleanup failure prevents a successful review handoff. Visual disagreement is carried into review with both citations and no selected value. Every result remains incomplete for clinical purposes. The synchronous example checks the original session deadline at each boundary; it does not demonstrate an idle recorder event loop or preemption of arbitrary callbacks.

## Native providers are qualified separately

The [local VLM](edge-vlm-provider.md) and [local ASR](offline-asr-provider.md) have separate opt-in real-model qualification tests. Their optional `edge-vlm` and `edge-asr` extras can be installed together from the combined lockfile on supported Apple silicon hosts. Both unchanged native profiles were also exercised in the combined environment with existing local model assets and operating-system network denial while preparing this reference. That establishes compatibility of those profiles and dependencies, not joint clinical validation of this fixture replay.

Supply your own pinned model assets and review their current terms independently of the Apache-2.0 adapter code. This workflow does not download weights, accept terms, maintain a weight-license allowlist or impose a license-based runtime gate. Replacing a fixture provider with a native provider requires a properly qualified host integration; changing the provider declaration is not a qualification procedure.

## Verification

The integration tests prohibit socket creation and DNS, inspect actual borrowed audio memory after cleanup, and check that reports and captured logs omit fixture identifiers. Controls cover cancellation at every stage, malformed and truncated media, missing masks, memory and thermal pressure before and after inference, provider exceptions, unresolved transcript identifiers, token lineage loss, source citations, disagreement propagation, deadline expiry and cleanup failures. They run offline by default and do not add wall-clock performance assertions to ordinary CI.
