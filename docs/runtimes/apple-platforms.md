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

## Build and validation

The Swift workflow performs the normal macOS tests, the iOS simulator build,
and focused parity tests on available watchOS and visionOS simulators. Local
checks use the same package scheme:

```bash
cd swift/OpenMedKit
xcodebuild build -scheme OpenMedKit \
  -destination 'generic/platform=watchOS Simulator'
xcodebuild build -scheme OpenMedKit \
  -destination 'generic/platform=visionOS Simulator'
```

Use only synthetic notes in committed tests and fixtures. OpenMedKit keeps
inference on device and does not log input text or detected span text.
