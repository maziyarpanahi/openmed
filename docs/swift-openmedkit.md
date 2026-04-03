# OpenMedKit (Swift Package)

OpenMedKit is a Swift Package for on-device clinical NER and PII detection on iOS and macOS, powered by CoreML.

## Requirements

- iOS 16+ / macOS 13+
- Xcode 15+
- A CoreML model converted from OpenMed (see [CoreML Export](coreml-export.md))

## Installation

### Swift Package Manager

Add to your `Package.swift`:

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

1. Convert using the Python CLI:
   ```bash
   pip install "openmed[coreml]"
   python -m openmed.coreml.convert \
       --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
       --output ./OpenMedPII.mlpackage
   ```

2. Compile for your app (optional, Xcode does this automatically):
   ```bash
   xcrun coremlcompiler compile OpenMedPII.mlpackage .
   ```

3. Add `OpenMedPII.mlmodelc` and `id2label.json` to your Xcode project.

## Concurrency

`EntityPrediction` is `Sendable`, so results can safely cross actor boundaries. The `OpenMed` class itself should be used from a single thread or wrapped in an actor for concurrent access.
