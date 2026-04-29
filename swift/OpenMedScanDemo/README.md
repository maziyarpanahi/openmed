# OpenMed Scan Demo

An iOS SwiftUI demo that shows the full native Apple flow:

- `VNDocumentCameraViewController` for document capture
- Vision OCR for text extraction
- `OpenMedKit` + an on-device OpenMed MLX PII model
- native Swift GLiNER Relex entity extraction over the de-identified note
- colorful inline masked labels plus the raw OCR transcript

## Quick Start

1. Open `swift/OpenMedScanDemo/OpenMedScanDemo.xcodeproj` in Xcode.
2. Select the `OpenMedScanDemo` scheme.
3. Run it on a real iPhone or iPad.
4. Tap `Scan Document` to capture pages or `Load Sample Document` to open the bundled clinical note image.
5. Press the main action button to run OCR, de-identification, and the GLiNER Relex pass in sequence.
6. The app uses `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` or `OpenMed/privacy-filter-nemotron-mlx-8bit` to mask PII and `OpenMed/gliner-relex-base-v1.0-mlx` to extract clinical entities from the masked note.
7. Each artifact is cached locally after a successful download. Later runs reuse the cached copy and only fetch files again if a previous download was interrupted and left the cache incomplete.
8. To test disconnected mode, run the demo once while online so both artifacts are cached, then disable network access and run the same sample or scan flow again.

## What It Demonstrates

- no Python service
- no remote inference
- native scan and OCR APIs from Apple
- local PII extraction and smart merging through `OpenMedKit`
- native Swift GLiNER Relex extraction over a masked note
- a masked document view that replaces detected spans with colorful labels

## Notes

- The scanner UI is iPhone/iPad only because it uses VisionKit's native document camera.
- The local MLX path also expects real Apple hardware; iOS Simulator is useful for UI review, not end-to-end validation.
- The selected OpenMed PII, OpenAI Nemotron Privacy Filter 8-bit, and GLiNER Relex artifacts are public OpenMed MLX artifacts, so no account setup is required.
- The app now distinguishes between missing, partial, and ready artifact caches so it can resume incomplete downloads without repeatedly re-downloading complete artifacts.
- The demo uses a fixed zero-shot label pack tuned to clinical follow-up documents: symptoms, conditions, medical history, medication, dosage, allergy, treatment, procedure, follow-up plan, care plan, care setting, and work status.

## Production Reference

The app is intentionally small and focused. For the underlying Apple integration details, see `docs/swift-openmedkit.md`.
