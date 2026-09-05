# OpenMed Scan Demo

An iOS SwiftUI demo focused entirely on native document intake and specialist
clinical extraction:

- `VNDocumentCameraViewController` document capture
- Vision OCR
- purpose-built OpenMed PII detection and masking
- disease, medication, and anatomy token-classification NER
- local MLX inference with explicit download, load, run, and unload lifecycles
- structured review, model comparison, and privacy-preserving JSON export

Generative models are intentionally outside this app. See
`swift/OpenMedMedicalReasoningDemo` for the separate LFM2.5 clinical chat
example.

## Quick Start

1. Open `swift/OpenMedScanDemo/OpenMedScanDemo.xcodeproj` in Xcode.
2. Select the `OpenMedScanDemo` scheme.
3. Run it on a real iPhone or iPad.
4. Scan a document, paste text, or load a bundled synthetic example.
5. Download and explicitly run one of the three specialist PII models.
6. Download and explicitly run the disease, medication, or anatomy NER model.
7. Review detected spans, compare completed PII engines, or export JSON.

Models are cached locally after a successful download. The app retains at most
one selected PII runtime, replacing it only after an explicit model selection
and run. Every NER run creates one runtime, performs token classification,
releases the runtime, and clears the MLX buffer cache before returning.

## What It Demonstrates

- no Python service or remote inference
- native Apple scan and OCR APIs
- purpose-built PII models used only for de-identification
- purpose-built NER models used only for extraction
- no prompt-based extraction or generative fallback
- validated Unicode-scalar spans with token-classification confidence scores
- colorful masked-document and entity-review surfaces

## Notes

- The scanner is iPhone/iPad-only because it uses VisionKit.
- The MLX path expects real Apple hardware. Simulator builds validate UI and
  integration structure, not model inference.
- The app distinguishes missing, partial, and ready caches so interrupted
  downloads can resume without re-downloading complete files.
- Clinical NER artifacts are pinned to exact revisions.
- Model output can be incomplete or wrong. The demo does not make clinical
  decisions; consequential results require clinician verification.

## Production Reference

See `docs/swift-openmedkit.md` for the underlying OpenMedKit APIs.
