# OpenMedKit Apple Platform Support

OpenMedKit supports macOS 14+, iOS 17+, watchOS 10+, and visionOS 1+.
The runtime surface depends on the platform so constrained devices do not link
backends that exceed their deployment or memory envelope.

| Platform | Supported backend | Default model ceiling | Resident RAM ceiling | Maximum sequence |
|---|---|---:|---:|---:|
| macOS 14+ | MLX or CoreML | Base | 900 MB | 512 tokens |
| iOS 17+ / iPadOS 17+ | MLX on a physical device, or CoreML | Tiny | 350 MB | 512 tokens |
| watchOS 10+ | CoreML only | Nano, INT8 | 150 MB | 256 tokens |
| visionOS 1+ | CoreML only | Nano, INT8 | 150 MB | 256 tokens |

The watchOS and visionOS limits use OpenMed's canonical Nano sub-tier: 10–30M
parameters, at most 150 MB resident memory, and INT8 CoreML artifacts. These
limits are enforced from model metadata before `MLModel` is opened. A Tiny,
Base, over-budget, or non-INT8 model fails closed instead of being loaded.

## Selecting and loading a constrained CoreML model

Describe the bundled model candidates and let `PlatformModel` select the
highest-capacity candidate that fits the current target:

```swift
import OpenMedKit

let nano = PlatformModelDescriptor(
    identifier: "OpenMed-PII-Nano-INT8",
    modelURL: Bundle.main.url(
        forResource: "OpenMed-PII-Nano-INT8",
        withExtension: "mlmodelc"
    )!,
    id2labelURL: Bundle.main.url(
        forResource: "id2label",
        withExtension: "json"
    )!,
    tier: .nano,
    parameterCount: 24_000_000,
    estimatedResidentMemoryMB: 128,
    isINT8: true
)

let model = try PlatformModel(candidates: [nano])
```

watchOS and visionOS intentionally omit the full MLX and
`swift-transformers` graph. Apps tokenize with the assets bundled beside their
Nano model and pass bounded token IDs, attention masks, and character offsets
to `PlatformModel.predict(...)`. This keeps model and tokenizer access local;
OpenMedKit does not add a cloud fallback.

## Minimal redaction surface

`PlatformModel.redact(...)` applies mask or removal redaction to detected
`EntityPrediction` spans without loading another backend:

```swift
let note = "Patient Ada Lovelace, MRN TEST-123."
let spans = [
    EntityPrediction(
        label: "full_name",
        text: "Ada Lovelace",
        confidence: 0.99,
        start: 8,
        end: 20
    ),
    EntityPrediction(
        label: "medical_record_number",
        text: "TEST-123",
        confidence: 0.99,
        start: 26,
        end: 34
    ),
]

let result = PlatformModel.redact(note, entities: spans)
// Patient [FULL_NAME], MRN [MEDICAL_RECORD_NUMBER].
```

Offsets remain character offsets into the original note. The watchOS and
visionOS simulator tests use the same synthetic note and iOS reference spans,
with a one-character tolerance at each boundary.

## iOS Simulator and MLX

MLX cannot run on the iOS Simulator: mlx-swift aborts the process while it
searches app bundles for its Metal library. OpenMedKit therefore checks one
gate before any MLX entry point reads model files, downloads assets, or
creates an MLX array or container. On the Simulator (and on Intel Macs) these
calls throw `OpenMedMLXRuntimeError.unsupportedPlatform`:

- `OpenMed(backend: .mlx(modelDirectoryURL:))`, including Privacy Filter
  artifacts
- `OpenMedZeroShotNER`, `OpenMedZeroShotClassifier` and
  `OpenMedRelationExtractor`
- `OpenMedMaple(modelDirectoryURL:)`
- both `OpenMedVisionLanguageModel.load` variants (the Hub variant refuses
  before it starts a download)

The error is value-free: it carries no path, model identifier or input text.
Catch it to fall back to the CoreML backend or to tell the user that MLX
features need a physical iPhone or iPad:

```swift
do {
    let maple = try await OpenMedMaple(modelDirectoryURL: modelDirectory)
    // ...
} catch OpenMedMLXRuntimeError.unsupportedPlatform {
    // Simulator or Intel Mac: use CoreML or hide the MLX feature.
}
```

`OpenMed.clearRuntimeMemoryCache()` is a no-op where MLX cannot run, so
CoreML-only callers such as the share extension can call it on the Simulator.

## Build and validation

The Swift workflow performs the normal macOS tests, the iOS simulator build,
an `xcodebuild test` run of `OpenMedKitTests` and `ExtensionTests` on an
available iPhone simulator, and focused parity tests on available watchOS and
visionOS simulators. On the iOS Simulator the CoreML, policy, offset and
extension tests run; MLX-dependent tests are reported as skipped with the
reason "MLX cannot run on the iOS Simulator", and the job lists them in its
step summary. Local checks use the same package scheme:

```bash
cd swift/OpenMedKit
xcodebuild test -scheme OpenMedKit-Package \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:OpenMedKitTests -only-testing:ExtensionTests \
  -skipPackagePluginValidation
xcodebuild build -scheme OpenMedKit \
  -destination 'generic/platform=watchOS Simulator'
xcodebuild build -scheme OpenMedKit \
  -destination 'generic/platform=visionOS Simulator'
```

Pick any iPhone listed by `xcrun simctl list devices available`.

Use only synthetic notes in committed tests and fixtures. OpenMedKit keeps
inference on device and does not log input text or detected span text.
