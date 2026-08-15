# OpenMedKit (Swift Package)

OpenMedKit is the Swift package for running OpenMed models in **macOS**,
**iOS**, **iPadOS**, **watchOS**, and **visionOS** apps.

OpenMedKit currently supports two Apple backends:

- **MLX** for Apple Silicon Macs and real iPhone/iPad devices
- **CoreML** for bundled Apple model packages, including constrained watchOS
  and visionOS Nano artifacts

Swift MLX supports the first OpenMed artifact families used by the public Apple demos:

- `bert`
- `distilbert`
- `roberta`
- `xlm-roberta`
- `electra`
- `deberta-v2` / DeBERTa-v3-backed experimental GLiNER-family artifacts
- `openai-privacy-filter`
- `privacy-filter-nemotron` / `privacy-filter-multilingual` artifacts through the OpenAI Privacy Filter runtime
- DeepGrove `maple` causal language models through the dedicated local
  `OpenMedMaple` runtime

ModernBERT, Longformer, EuroBERT, Qwen3, and additional architecture families are still part of the broader rollout work.

## Requirements

- iOS 17+ / macOS 14+ / watchOS 10+ / visionOS 1+
- Xcode 15+
- For MLX:
  - Apple Silicon Mac, or
  - a real iPhone/iPad device
- For CoreML:
  - a compatible `.mlpackage` or `.mlmodelc` bundle plus `id2label.json`

iOS Simulator is **not** a Swift MLX validation target.
watchOS and visionOS use the CoreML-only `PlatformModel` surface and require an
INT8 Nano-tier artifact. See [Apple Platform Support](./runtimes/apple-platforms.md)
for selection limits and simulator validation.

Maple Preview 2-bit is a multi-gigabyte generative model. Test it on recent
Apple hardware with sufficient storage and memory; keep a smaller specialized
model available when the product must support constrained devices.

## Apple Platform Matrix

| Use case | Recommended path |
|---|---|
| Python on Apple Silicon Mac | `openmed[mlx]` |
| Swift app on Apple Silicon macOS | `OpenMedKit` + MLX or CoreML |
| Swift app on real iPhone/iPad | `OpenMedKit` + MLX or CoreML |
| Swift app on iOS Simulator | CoreML only |
| Swift app on Apple Watch | `PlatformModel` + Nano INT8 CoreML |
| Swift app on Apple Vision Pro | `PlatformModel` + Nano INT8 CoreML |
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
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "2.2.0"),
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
    repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"
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

## Maple Preview: Local Generative Tasks

`OpenMedMaple` loads the exact-head
[`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)
checkpoint from a complete local directory. It supports de-identification,
entity extraction, relation extraction, evidence-grounded reasoning, and chat.
It never downloads, logs, or persists document text; the host app controls model
acquisition and must pass a local directory.

```swift
import OpenMedKit

let modelDirectory = URL(fileURLWithPath: "/path/to/maple-preview-2bit-mlx")
guard OpenMedMaple.isModelDirectoryReady(modelDirectory) else {
    fatalError("Download Maple's required config, tokenizer, and three weight shards first")
}

let maple = try await OpenMedMaple(modelDirectoryURL: modelDirectory)
let masked = try await maple.complete(
    OpenMedMapleRequest(task: .deidentify, document: scannedText)
)

let clinical = try await maple.complete(
    OpenMedMapleRequest(
        task: .relationExtraction,
        document: masked.redactedText ?? "",
        entityLabels: ["condition", "medication", "dosage", "follow-up plan"],
        relationLabels: ["treated with", "has dosage", "requires follow-up"]
    )
)

let brief = try await maple.complete(
    OpenMedMapleRequest(
        task: .reasoning,
        document: masked.redactedText ?? "",
        question: "What facts, uncertainties, and follow-up items are documented?"
    )
)

var streamedAnswer = ""
let chat = try await maple.complete(
    OpenMedMapleRequest(
        task: .chat,
        document: masked.redactedText ?? "",
        question: "What follow-up is documented?"
    ),
    onFinalAnswerChunk: { chunk in
        await MainActor.run {
            streamedAnswer.append(chunk)
        }
    }
)
```

For multi-turn chat, pass the prior non-system turns through `messages` and the
new prompt through `question`. Keep the `document` de-identified: OpenMedKit's
prompt builder treats it as untrusted input, requests final answers without
chain-of-thought, and adds a clinician-review boundary, but application-level
data minimization remains required.

`onFinalAnswerChunk` is called only for reasoning and chat, and only after Maple
closes its private reasoning segment. De-identification, entity extraction, and
relation extraction remain buffered until OpenMedKit has parsed complete JSON,
validated its label vocabulary, repaired exact source spans, and dropped
relations with unverified endpoints. The returned `chat.answer` is the canonical
final value and can replace the accumulated UI text when generation completes.

For de-identification, the generated rewrite is never trusted directly:
OpenMedKit deterministically masks the validated source spans so unrelated
clinical text cannot be silently rewritten. The runtime pins the public checkpoint revision in
`OpenMedMaple.pinnedRevision`. Required files are exposed through
`OpenMedMaple.requiredModelFiles`; the optional approximate
`model-flashhead.safetensors` is not used. Structured outputs are parsed only
after spans are validated or repaired against the source document, and
relations with unverified endpoints are dropped.

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
    revision: "main"
)
```

Behavior:

- downloads `openmed-mlx.json` first
- downloads the config, labels, tokenizer assets, and available weight files
- caches the model under the app cache directory
- returns a local directory URL ready for `OpenMedBackend.mlx`

If a repo predates the manifest rollout, OpenMedKit falls back to the legacy layout and downloads the available config, labels, weights, and any bundled tokenizer files it can find.

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

## Demo Apps

The demo app in [`swift/OpenMedDemo/`](https://github.com/maziyarpanahi/openmed/tree/master/swift/OpenMedDemo) now exposes:

- bundled CoreML models discovered from the app target
- a searchable catalog of Swift-MLX-compatible OpenMed PII models
- public OpenMed MLX artifact download, local caching, and offline reuse

On Apple Silicon macOS or a physical iPhone/iPad, the demo can download a supported MLX artifact and run it locally through OpenMedKit.

The scanning demo in
[`swift/OpenMedScanDemo/`](https://github.com/maziyarpanahi/openmed/tree/master/swift/OpenMedScanDemo)
shows Maple's complete iOS workflow: VisionKit capture, Vision OCR, PII removal,
entity and relation extraction, a grounded clinical brief, multi-turn document
chat, and JSON export. All generative tasks run after de-identification, and the
app unloads competing MLX runtimes before loading the larger Maple model.

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
