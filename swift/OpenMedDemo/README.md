# OpenMed PII Demo (SwiftUI)

A minimal SwiftUI app demonstrating on-device PII detection with OpenMed. It builds on both **macOS** and **iOS** today, and currently runs in **demo mode** unless you bundle a compatible OpenMed CoreML model.

![Demo Screenshot](screenshot.png)

## Current Status

- Verified locally on macOS: the `OpenMedDemo` Xcode project builds successfully with `xcodebuild`
- The app does **not** currently use an uploaded/published model artifact
- The app falls back to mock detections unless `OpenMedPII.mlmodelc` and `id2label.json` are bundled into the app
- The inline SwiftUI demo is **not yet wired** to the `OpenMedKit` package at runtime

## Quick Start (Demo Mode)

The app works immediately in **demo mode** with mock entity detection — no model download needed:

1. Open `swift/OpenMedDemo/` in Xcode (File > Open)
2. Select the `OpenMedDemo` scheme
3. Choose a target: **My Mac** or **iPhone Simulator**
4. Run (Cmd+R)

You'll see highlighted PII entities (names, dates, phone numbers, SSNs) in the sample clinical note.

This is useful for validating the macOS/iOS UI flow, but it is not a real CoreML inference path yet.

## Adding a Real CoreML Model

To switch from demo mode to real on-device inference:

### Step 1: Prepare the app bundle

When you have a compatible OpenMed CoreML package, place these assets into the Xcode project:

1. Drag `OpenMedPII.mlpackage` into the Xcode project navigator
2. Rename `OpenMedPII_id2label.json` to `id2label.json` and add it too
3. Ensure both files have **Target Membership** checked for `OpenMedDemo`

The broader cross-architecture Apple packaging workflow is still in progress, so this README intentionally focuses on the app-side integration contract rather than the converter internals.

### Step 2: Run

The app auto-detects the bundled model and switches from demo mode to real inference.

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

## Credits

OpenMed is developed by the OpenMed project and community. The demo app in this repository is a lightweight SwiftUI example for validating the OpenMed macOS and iOS integration path.
