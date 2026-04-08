# OpenMed PII Demo (SwiftUI)

A SwiftUI demo for testing OpenMed on **macOS** and **iOS** with:

- bundled CoreML models
- downloadable OpenMed MLX artifacts
- a searchable model picker

![Demo Screenshot](screenshot.png)

## What This Demo Now Tests

- **Swift MLX** through `OpenMedKit` on:
  - Apple Silicon macOS
  - real iPhone/iPad hardware
- **CoreML** through `OpenMedKit` when you bundle an Apple model package

iOS Simulator is not a Swift MLX target, so the app will tell you that directly if you select an MLX model there.

## Quick Start

1. Open [`swift/OpenMedDemo/OpenMedDemo.xcodeproj`](/Users/maziyar/Desktop/Work/openmed-release-1.0.0/swift/OpenMedDemo/OpenMedDemo.xcodeproj) in Xcode.
2. Select the `OpenMedDemo` scheme.
3. Choose:
   - `My Mac` for Apple Silicon macOS MLX testing, or
   - a real iPhone/iPad for on-device MLX testing
4. Run the app.
5. Choose one of the published OpenMed MLX models in the picker.
6. Paste a Hugging Face token while the repos remain private.
7. Run inference.

The app downloads the artifact, caches it locally, and then runs it through `OpenMedKit`.

If a private MLX repo was uploaded before the new manifest/tokenizer packaging, the demo still falls back to the legacy layout and uses the source Hugging Face tokenizer reference when needed.

## Published Swift-MLX-Compatible Models

The demo currently hardcodes the supported OpenMed PII models already uploaded in MLX form:

- `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx`
- `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx`
- `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx`
- `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx`
- `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx`

These map to the first Swift MLX rollout families:

- BERT
- DistilBERT
- RoBERTa
- ELECTRA
- XLM-RoBERTa

## Testing Bundled CoreML Models

The demo still supports CoreML bundles too.

### Option 1: Legacy single-model bundle

Add:

1. `OpenMedPII.mlpackage` or `OpenMedPII.mlmodelc`
2. `id2label.json`
3. optional `TokenizerAssets/`

Once those files are part of the app target, the demo discovers them automatically.

### Option 2: Multi-model bundle folders

Create one folder per bundled model and include:

1. `openmed-model.json`
2. the `.mlpackage` or `.mlmodelc`
3. `id2label.json`
4. optional tokenizer assets folder

Example `openmed-model.json`:

```json
{
  "displayName": "ClinicalE5 Small CoreML",
  "sourceModelId": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
  "tokenizerName": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
  "compiledModelName": "OpenMedPII",
  "compiledModelExtension": "mlmodelc",
  "id2labelFileName": "id2label.json",
  "tokenizerFolderName": "TokenizerAssets",
  "note": "Bundled CoreML validation."
}
```

## Notes

- While the MLX repos are private, the demo needs a Hugging Face token.
- Once the repos are public, the same app flow can download them without a token.
- The demo does not route MLX through a Python service and does not use mock inference.
- The app uses `OpenMedKit` directly for both MLX and CoreML runtime paths.

## Production Path

For production apps, use `OpenMedKit` directly:

```swift
import OpenMedKit

let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx",
    authToken: "<token-if-private>"
)

let openmed = try OpenMed(
    backend: .mlx(modelDirectoryURL: modelDirectory)
)

let entities = try openmed.analyzeText("Patient John Doe, SSN 123-45-6789")
```

For the full Apple setup guide, see [OpenMedKit docs](/Users/maziyar/Desktop/Work/openmed-release-1.0.0/docs/swift-openmedkit.md).
