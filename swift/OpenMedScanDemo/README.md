# OpenMed Scan Demo

An iOS SwiftUI demo that shows the full native Apple flow:

- `VNDocumentCameraViewController` for document capture
- Vision OCR for text extraction
- `OpenMedKit` + DeepGrove Maple Preview 2-bit running locally with MLX
- Maple PII removal, entity extraction, and relation extraction
- grounded clinical-note reasoning and chat over the de-identified text
- colorful inline masked labels plus the raw OCR transcript

## Quick Start

1. Open `swift/OpenMedScanDemo/OpenMedScanDemo.xcodeproj` in Xcode.
2. Select the `OpenMedScanDemo` scheme.
3. Run it on a real iPhone or iPad.
4. Tap `Scan Document` to capture pages or `Load Sample Document` to open the bundled clinical note image.
5. Choose **Maple Preview**, download the model, then use the main action button to move through OCR, review, de-identification, clinical extraction, Maple Insights, and the summary.
6. Maple is downloaded from `deepgrove/maple-preview-2bit-mlx` at the revision pinned by `OpenMedMaple.pinnedRevision`. The demo fetches the three exact-head weight shards and deliberately excludes the optional approximate FlashHead weights.
7. The model is cached locally after a successful download. Later runs reuse the cached copy and only fetch files again if an interrupted download left the cache incomplete.
8. In **Maple Insights**, generate a grounded brief or ask document questions. Maple receives only the masked note, and its responses are labeled for clinician review rather than diagnosis or treatment.
9. To test disconnected mode, run the demo once while online so Maple is cached, then disable network access and run the same sample or scan flow again.

## What It Demonstrates

- no Python service
- no remote inference
- native scan and OCR APIs from Apple
- one local Maple runtime for PII removal, entities, and directed relations
- evidence-grounded reasoning and multi-turn chat over masked text
- prompt-injection boundaries that treat scanned document text as untrusted data
- validated Unicode-scalar spans and relations whose endpoints were actually extracted
- a masked document view that replaces detected spans with colorful labels

## Notes

- The scanner UI is iPhone/iPad only because it uses VisionKit's native document camera.
- The local MLX path also expects real Apple hardware; iOS Simulator is useful for UI review, not end-to-end validation.
- Maple Preview 2-bit is a multi-gigabyte model. Use a recent device with enough free storage and memory; the UI shows the download requirement before inference.
- The selected Maple, OpenMed PII, OpenAI Nemotron Privacy Filter 8-bit, and OpenMed Multilingual Privacy Filter 8-bit artifacts are public, so no account setup is required.
- The app now distinguishes between missing, partial, and ready artifact caches so it can resume incomplete downloads without repeatedly re-downloading complete artifacts.
- The demo uses a fixed zero-shot label pack tuned to clinical follow-up documents: symptoms, conditions, medical history, medication, dosage, allergy, treatment, procedure, follow-up plan, care plan, care setting, and work status.
- Model output can be incomplete or wrong. The demo does not make clinical decisions, and consequential results require clinician verification.

## Production Reference

The app is intentionally small and focused. For the underlying Apple integration details, see `docs/swift-openmedkit.md`.
