# OpenMedKit (Swift Package)

OpenMedKit is the Swift package for running OpenMed models in **macOS**, **iOS**, and **iPadOS** apps.

In `1.1.0`, OpenMedKit supports two Apple backends:

- **MLX** for Apple Silicon Macs and real iPhone/iPad devices
- **CoreML** for bundled Apple model packages

The first Swift MLX milestone is intentionally focused on the BERT-family encoder path:

- `bert`
- `distilbert`
- `roberta`
- `xlm-roberta`
- `electra`

DeBERTa-v2/v3, ModernBERT, Longformer, EuroBERT, and Qwen3 are still part of the broader rollout work.

## Requirements

- iOS 17+ / macOS 14+
- Xcode 15+
- For MLX:
  - Apple Silicon Mac, or
  - a real iPhone/iPad device
- For CoreML:
  - a compatible `.mlpackage` or `.mlmodelc` bundle plus `id2label.json`

iOS Simulator is **not** a Swift MLX validation target.

## Apple Platform Matrix

| Use case | Recommended path |
|---|---|
| Python on Apple Silicon Mac | `openmed[mlx]` |
| Swift app on Apple Silicon macOS | `OpenMedKit` + MLX or CoreML |
| Swift app on real iPhone/iPad | `OpenMedKit` + MLX or CoreML |
| Swift app on iOS Simulator | CoreML only |
| Older Apple OS support | CoreML |

## Install OpenMedKit

### Xcode

1. Open your app project in Xcode.
2. Choose `File > Add Package Dependencies...`
3. Enter `https://github.com/maziyarpanahi/openmed`
4. Add the `OpenMedKit` product to your app target

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "1.1.0"),
]
```

Then add the product:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "OpenMedKit", package: "openmed"),
    ]
)
```

## Quick Start: Swift MLX

This is the new on-device path for supported OpenMed MLX artifacts.

```swift
import OpenMedKit

let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx",
    authToken: "<hugging-face-token-while-private>"
)

let openmed = try OpenMed(
    backend: .mlx(modelDirectoryURL: modelDirectory)
)

let entities = try openmed.extractPII(
    "Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"
)
```

That MLX model directory must contain:

- `openmed-mlx.json`
- `config.json`
- `id2label.json`
- tokenizer assets
- `weights.safetensors` preferred, `weights.npz` fallback

The same converted MLX artifact is now intended to work in both:

- Python on Apple Silicon
- Swift via OpenMedKit

OpenMedKit prefers the new self-contained artifact layout above. For older MLX repos that were uploaded before `openmed-mlx.json` and tokenizer asset bundling, it also keeps backward compatibility by falling back to `config.json` plus the source Hugging Face tokenizer reference when available.

## Quick Start: CoreML

CoreML is still supported and remains the right path when you already have an Apple model bundle or need a non-MLX fallback.

```swift
import OpenMedKit

let modelURL = Bundle.main.url(forResource: "OpenMedPII", withExtension: "mlmodelc")!
let labelsURL = Bundle.main.url(forResource: "id2label", withExtension: "json")!

let openmed = try OpenMed(
    backend: .coreML(
        modelURL: modelURL,
        id2labelURL: labelsURL,
        tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        tokenizerFolderURL: nil
    )
)

let entities = try openmed.extractPII("Patient John Doe, SSN 123-45-6789")
```

The convenience initializer remains available:

```swift
let openmed = try OpenMed(
    modelURL: modelURL,
    id2labelURL: labelsURL,
    tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
)
```

## Downloading MLX Models From Hugging Face

OpenMedKit includes a built-in downloader and local cache for Hub-hosted MLX artifacts:

```swift
let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx",
    authToken: "<token-if-private>",
    revision: "main"
)
```

Behavior:

- downloads `openmed-mlx.json` first
- downloads the config, labels, tokenizer assets, and available weight files
- caches the model under the app cache directory
- returns a local directory URL ready for `OpenMedBackend.mlx`

If a repo predates the manifest rollout, OpenMedKit falls back to the legacy layout and downloads the available config, labels, weights, and any bundled tokenizer files it can find.

While the repos remain private, pass a Hugging Face token. Once they are public, `authToken` can be `nil`.

## Offline Tokenizer Assets

For MLX artifacts, tokenizer assets travel with the converted model directory, so Swift can load them locally without going back to the Hub.

For CoreML bundles, you can still bundle tokenizer assets manually and pass `tokenizerFolderURL`:

```swift
let openmed = try OpenMed(
    modelURL: modelURL,
    id2labelURL: labelsURL,
    tokenizerFolderURL: Bundle.main.url(forResource: "TokenizerAssets", withExtension: nil)
)
```

## Public API

### `OpenMedBackend`

```swift
public enum OpenMedBackend: Sendable {
    case coreML(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String,
        tokenizerFolderURL: URL?
    )
    case mlx(modelDirectoryURL: URL)
}
```

### `OpenMed`

```swift
public final class OpenMed {
    public init(
        backend: OpenMedBackend,
        maxSeqLength: Int = 512
    ) throws

    public convenience init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        tokenizerFolderURL: URL? = nil,
        maxSeqLength: Int = 512
    ) throws

    public func analyzeText(
        _ text: String,
        confidenceThreshold: Float = 0.5
    ) throws -> [EntityPrediction]

    public func extractPII(
        _ text: String,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) throws -> [EntityPrediction]
}
```

`extractPII(...)` now follows the Python PII path more closely by applying span repair plus semantic-unit merging for fragmented dates, SSNs, phone numbers, emails, and similar PII spans.

### `OpenMedModelStore`

```swift
public enum OpenMedModelStore {
    public static func downloadMLXModel(
        repoID: String,
        authToken: String? = nil,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) async throws -> URL
}
```

## Supported Swift MLX Families

The current Swift MLX runtime is the BERT-family token-classification path shared across:

- BERT
- DistilBERT
- RoBERTa
- XLM-RoBERTa
- ELECTRA

This is the same first-phase scope as the current public Python MLX BERT-family implementation.

## Demo App

The demo app in [`swift/OpenMedDemo/`](https://github.com/maziyarpanahi/openmed/tree/master/swift/OpenMedDemo) now exposes:

- bundled CoreML models discovered from the app target
- a searchable catalog of Swift-MLX-compatible OpenMed PII models
- Hugging Face token entry for the current private MLX repos

On Apple Silicon macOS or a physical iPhone/iPad, the demo can download a supported MLX artifact and run it locally through OpenMedKit.

## CoreML Status

CoreML remains part of the public Apple story, but it is no longer the only Swift path.

Use CoreML when:

- you already have a bundled Apple model package
- you want an older-OS fallback
- you are validating an app path that already depends on CoreML packaging

For the current CoreML packaging status, see [CoreML packaging status](coreml-export.md).

## Notes On Testing

The Swift MLX runtime is intended for:

- Apple Silicon macOS app builds
- real iPhone/iPad hardware

Command-line `swift test` may skip the MLX execution tests if the local test environment does not package MLX runtime Metal resources. That does not change the supported app runtime targets above.
