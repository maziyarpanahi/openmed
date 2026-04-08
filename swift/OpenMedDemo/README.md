# OpenMed PII Demo (SwiftUI)

A minimal SwiftUI app demonstrating OpenMed on **macOS** and **iOS** with a searchable model picker for **real bundled CoreML models**.

![Demo Screenshot](screenshot.png)

## Current Status

- Verified locally on macOS: the `OpenMedDemo` Xcode project builds successfully with `xcodebuild`
- The app is wired to **OpenMedKit** for real bundled CoreML inference
- The app includes a searchable model picker for bundled models only
- Uploaded MLX repos are for **Python/macOS** workflows, not direct Swift loading
- Swift apps still need bundled **CoreML** assets plus `id2label.json`

## Quick Start

1. Open `swift/OpenMedDemo/` in Xcode (File > Open)
2. Select the `OpenMedDemo` scheme
3. Choose a target: **My Mac** or **iPhone Simulator**
4. Run (Cmd+R)
5. Add a real OpenMed CoreML bundle before testing inference

If no bundled model is present, the app now blocks inference and tells you exactly which files are missing.

## Adding Real Models

The demo discovers bundled models in two ways.

### Option 1: Legacy single-model bundle

If you want a single default model, add:

1. Drag `OpenMedPII.mlpackage` into the Xcode project navigator
2. Rename `OpenMedPII_id2label.json` to `id2label.json` and add it too
3. Optionally add a `TokenizerAssets/` folder for offline tokenization
4. Ensure all of them have **Target Membership** checked for `OpenMedDemo`

The model picker will show this as `Bundled OpenMedPII`.
`OpenMedKit` compiles `.mlpackage` automatically on first load, so you do not need to precompile it to `.mlmodelc` unless you want to.

### Option 2: Multi-model switcher

For multiple bundled models, create one folder per model in your Xcode project and add these files inside each folder:

1. `openmed-model.json`
2. your compiled model such as `OpenMedPII.mlmodelc`
3. `id2label.json`
4. optional tokenizer assets folder for offline tokenization

Example `openmed-model.json`:

```json
{
  "displayName": "ClinicalE5 Small",
  "sourceModelId": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
  "tokenizerName": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
  "compiledModelName": "OpenMedPII",
  "compiledModelExtension": "mlmodelc",
  "id2labelFileName": "id2label.json",
  "tokenizerFolderName": "TokenizerAssets",
  "note": "Private CoreML packaging for Apple app validation."
}
```

Once those resources are in the app target, the demo discovers them automatically and the searchable picker lets you switch models at runtime.

## Architecture

```
OpenMedDemoApp.swift     — App entry point
ContentView.swift        — Main UI with:
  - TextEditor for clinical note input
  - "Detect PII Entities" button
  - Highlighted text view (color-coded by entity type)
  - Entity list with labels and confidence scores
  - Inference timing display
```

## Entity Color Coding

| Entity Type | Color |
|-------------|-------|
| NAME | Blue |
| DATE | Purple |
| PHONE | Green |
| SSN | Red |
| ADDRESS | Orange |

## Using OpenMedKit (Production)

For production apps, use the `OpenMedKit` Swift package instead of the inline inference code:

```swift
import OpenMedKit

let openmed = try OpenMed(
    modelURL: Bundle.main.url(forResource: "OpenMedPII", withExtension: "mlmodelc")!,
    id2labelURL: Bundle.main.url(forResource: "id2label", withExtension: "json")!
)
let entities = try openmed.analyzeText("Patient John Doe, SSN 123-45-6789")
```

That is also what the demo app now does internally for bundled models.

## macOS vs iOS

- **Python on macOS**: use the private/public MLX repos with `openmed[mlx]`
- **Swift on macOS**: use `OpenMedKit` with bundled CoreML
- **Swift on iOS**: use `OpenMedKit` with bundled CoreML

The MLX artifacts you uploaded are great for Python on Apple Silicon Macs, but they are not loaded directly by Swift or iOS.

Today, the smoothest validated Apple path is a BERT-family OpenMed model such as `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1`, bundled through CoreML plus `id2label.json`.

## Credits

OpenMed is developed by the OpenMed project and community. The demo app in this repository is a lightweight SwiftUI example for validating the OpenMed macOS and iOS integration path.
