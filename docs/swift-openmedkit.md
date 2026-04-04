# OpenMedKit (Swift Package)

OpenMedKit is the Swift package for running OpenMed models in **iOS** and **macOS** apps via **CoreML**.

If you want Apple Silicon acceleration in **Python on macOS**, use the MLX backend instead. OpenMedKit does **not** load MLX `.npz` artifacts directly.

The package API is public in `1.0.0`. OpenMed's broader cross-architecture model-packaging workflow for Apple platforms is still being hardened, so treat conversion itself as in progress rather than a stable public contract.

## Requirements

- iOS 16+ / macOS 13+
- Xcode 15+
- A compatible OpenMed CoreML bundle plus `id2label.json`

## Apple Platform Matrix

| Use case | Recommended path |
|---|---|
| Python app on Apple Silicon Mac | `pip install "openmed[mlx]"` |
| Swift app on macOS | `OpenMedKit` + CoreML |
| Swift app on iPhone/iPad | `OpenMedKit` + CoreML |
| Direct use of MLX `.npz` in Swift | Not supported in 1.0.0 |

## Installation

### Swift Package Manager

OpenMedKit is exported from the repository root, so you can add the repo directly:

```swift
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "1.0.0"),
]
```

Then add `OpenMedKit` as a dependency to your target:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "OpenMedKit", package: "openmed"),
    ]
)
```

### Xcode

1. File > Add Package Dependencies
2. Enter: `https://github.com/maziyarpanahi/openmed`
3. Select "OpenMedKit" library

## Quick Start

```swift
import OpenMedKit

// 1. Bundle the CoreML model and id2label.json in your app
let modelURL = Bundle.main.url(forResource: "OpenMedPII", withExtension: "mlmodelc")!
let labelsURL = Bundle.main.url(forResource: "id2label", withExtension: "json")!

// 2. Initialize
let openmed = try OpenMed(
    modelURL: modelURL,
    id2labelURL: labelsURL,
    tokenizerName: "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
)

// 3. Analyze text
let entities = try openmed.analyzeText("Patient John Doe, SSN 123-45-6789")
for entity in entities {
    print(entity)
    // [first_name] "John Doe" (8:16) conf=0.95
    // [ssn] "123-45-6789" (22:33) conf=0.92
}
```

## Offline Tokenizer Assets

By default, OpenMedKit loads the tokenizer using the Hugging Face model name you pass in `tokenizerName`.

For fully offline apps, bundle the tokenizer files in your app and initialize OpenMedKit with `tokenizerFolderURL`:

```swift
let openmed = try OpenMed(
    modelURL: modelURL,
    id2labelURL: labelsURL,
    tokenizerFolderURL: Bundle.main.url(forResource: "TokenizerAssets", withExtension: nil)
)
```

Your bundled tokenizer folder should include the tokenizer assets from the source OpenMed model repo, typically:

- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- model-specific vocab assets such as `spm.model`

## API Reference

### `OpenMed`

The main entry point for on-device clinical NLP.

```swift
public class OpenMed {
    /// Initialize with a CoreML model and label mapping.
    public init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        tokenizerFolderURL: URL? = nil,
        maxSeqLength: Int = 512
    ) throws

    /// Run NER on the given text.
    public func analyzeText(
        _ text: String,
        confidenceThreshold: Float = 0.5
    ) throws -> [EntityPrediction]

    /// Run PII detection (alias for analyzeText with a PII model).
    public func extractPII(
        _ text: String,
        confidenceThreshold: Float = 0.5
    ) throws -> [EntityPrediction]
}
```

### `EntityPrediction`

A single detected entity.

```swift
public struct EntityPrediction: Codable, Equatable, Sendable {
    public let label: String       // e.g., "first_name", "ssn", "date_of_birth"
    public let text: String        // The matched text span
    public let confidence: Float   // 0.0 – 1.0
    public let start: Int          // Start character offset
    public let end: Int            // End character offset (exclusive)
}
```

### `PostProcessing`

BIO tag decoding utilities (used internally by `NERPipeline`, also available for custom pipelines).

```swift
public enum PostProcessing {
    public enum AggregationStrategy {
        case first      // Use score of the first token
        case average    // Average scores across tokens
        case max        // Use maximum score
    }

    public static func decodeEntities(
        tokens: [TokenPrediction],
        text: String,
        strategy: AggregationStrategy = .average
    ) -> [EntityPrediction]
}
```

## Preparing Your Model

1. Obtain a compatible OpenMed CoreML bundle from your build or release flow.

2. Compile for your app (optional, Xcode does this automatically):
   ```bash
   xcrun coremlcompiler compile OpenMedPII.mlpackage .
   ```

3. Add `OpenMedPII.mlmodelc` and `id2label.json` to your Xcode project.

4. If you want offline tokenization, also add a tokenizer asset folder and pass it via `tokenizerFolderURL`.

For the current platform status and rollout direction, see [CoreML packaging status](coreml-export.md).

## MLX vs Swift

The OpenMed MLX backend is a **Python/macOS runtime**. It is ideal for local Apple Silicon scripts, notebooks, services, and desktop workflows.

For Swift apps:

- use **CoreML** with OpenMedKit
- do **not** point OpenMedKit at `weights.npz`
- if you need the same model in both Python and Swift, keep an MLX artifact for Python/macOS and a CoreML artifact for Swift/iOS/macOS apps

## Concurrency

`EntityPrediction` is `Sendable`, so results can safely cross actor boundaries. The `OpenMed` class itself should be used from a single thread or wrapped in an actor for concurrent access.
