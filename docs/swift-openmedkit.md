# OpenMedKit (Swift Package)

## Local agent review and receipt evidence

OpenMedKit parses content-free agent metadata with strict, bounded JSON:

| Swift value | Existing Python wire contract |
| --- | --- |
| `AgentArtifactReference` | `ArtifactReference`, envelope version `1` |
| `AgentReviewerHandoff` | `openmed.agent.reviewer_handoff.v1` |
| `AgentApprovalReceipt` | `openmed.agent.approval_receipt.v2` |
| `AgentRunEvidence` | `openmed.agent.run_summary.v1` |
| `AgentOMOPPreview` | `openmed.interop.omop.mutation_batch.v1` |
| `AgentApprovalEvidenceResult` | `openmed.agent.approval_evidence.v1` |

The handoff and artifact parsers retain Python's optional default version
fields. Other envelopes require their exact version. Unknown fields, duplicate
keys (including escaped duplicates), non-finite numbers, numeric booleans,
fractional integer fields, unsupported versions and malformed identifiers fail
with fixed `AgentGovernanceError` codes. No rejected values enter diagnostics.
Receipt/report inputs are bounded to 64 KiB; other inputs to 1 MiB, with a
depth limit of 16 and a 200,000-node limit. Handoffs have at most 64 unique
evidence references and whole-second UTC timestamps with exclusive expiry.
Run summaries use the existing Python count, duration and sorted/unique bounds.

`canonicalJSON()` emits Python-compatible sorted compact ASCII JSON. The OMOP
parser checks both the preview digest and the batch's ordered row-digest
commitment. It exposes typed mutation/issue metadata, never staged row values.
It cannot establish the provenance of a supplied snapshot or row digest. This
version supports the native OMOP preview and opaque preview artifact references;
it does not interpret other domain-specific preview envelopes or execute a
mobile workflow.

A v2 receipt contains only its schema version, action/token digests and an `approved`
code. Reviewer role and validity bounds stay in protected local custody. Legacy
role or timestamp fields are rejected. A parsed receipt or report is not approval authority. Configure a local custody
lookup only after the application has authenticated a reviewer and consumed the
original approval in its own trusted store:

```swift
// receiptJSON is previously consumed evidence from application-owned custody.
// localCustody is a protected local lookup established after approval consumption.
// Its AgentReceiptCustody observations bind the exact digest, role and private validity.
let receipt = try AgentApprovalReceipt.parse(receiptJSON)
let verifier = AgentLocalApprovalEvidenceVerifier(authority: { digest in
    guard let custody = localCustody[digest] else { return .unrecognized }
    return .recognized(custody)
})
let observation = try verifier.verify(
    receipt, actionDigest: currentlyReviewedActionDigest,
    reviewerRole: expectedReviewerRole)
// observation.reasonCode records a local evidence check for the review UI.
// observation.authorizesClinicalAction always remains false.
```

Omitting the lookup returns `unsupported_authority`. An unrecognized digest,
failed lookup, expired/future receipt, changed action/role, replay, failed clock
or unavailable replay store remains a typed refusal. Time is checked again
after custody lookup against its private consumption time and exclusive expiry.
The Python lookup returns an actual verifier-created `ApprovalAuthorization`;
a bare recognized flag cannot supply custody. Swift hosts provide a local
`AgentReceiptCustody` observation from their protected approval store. It has no
JSON decoder or serializer and cannot authorize dispatch. Recognized presentations are atomically claimed before
action/role comparison, so a changed-action presentation is burned and cannot
be retried under the original action. The default replay store is thread-safe
and process-local; applications sharing evidence across processes or launches
must inject an atomic application-owned store. It supplements the original
durable approval nonce custody and never replaces dispatch authorization.

The Python counterpart is `openmed.agent.approval_evidence`; the strict native
OMOP projection parser is `openmed.agent.review_previews`. These APIs contain
no signing, receipt issuance, dispatch callback, EHR connection or cloud
fallback. Serialized `verified` observations cannot authorize another caller.
Descriptions contain only controlled codes/counts; explicit canonical JSON
contains opaque references and digests and should remain in local custody.

Both platforms consume versioned synthetic vectors at
`tests/fixtures/agent/governance_parity/v1.json`. Python regenerates the receipts
from test-only signed approval consumption and the other packets from their
native producers. Native Swift tests compare canonical bytes/digests, strict
refusals, changed-action/role burns, replay and expiry. These are offline wire
and local-boundary checks, not clinical validation or a mobile execution claim.

## Guarded clinical briefs

Maple has a structured `.brief` task. Use `OpenMedMaple.brief(source:
originalIdentifiers:evaluate:privacyCheck:)` to buffer generation and validate a
complete local evaluation packet before exposing it. The `evaluate` callback
must execute reviewed-evidence and calibrated NLI verification **on-device**;
it must not manufacture review history or call a cloud service. No verifier or
calibrated checkpoint is implicitly installed by this API.

The native boundary independently enforces source/output digest binding,
identifier leakage, the human-review/non-diagnostic safety envelope, exact
Unicode-scalar citation boundaries and entailment verdicts. `ClinicalBrief`
exposes summary, citations, verdicts and digest. `responseJSON()` is protected
output; `auditJSON()` excludes generated text. Typed failures never contain
prompts, model responses or private exception strings. Missing local model files
fail closed through the existing Maple runtime, with no cloud fallback.

The scan demo now renders verified citations after de-identification. Configure
its `briefEvaluator` and `briefPrivacyCheck` with trusted local integrations;
without them the action displays a review-configuration refusal instead of the
previous unverified reasoning answer. This is an integration boundary, not a
bundled native NLI implementation or clinical readiness claim.

Python and Swift consume `tests/fixtures/clinical/brief_parity/verified.json`.
The Python test regenerates the packet from the composed pipeline with explicit
test-only NLI/privacy doubles. The Swift test checks the native guards and
byte-identical protected response for that fixed output. This proves wire and
guard interoperability, not independent Swift reimplementation of all Python
metrics or model-quality equivalence.

The fixture also tests source-identifier matching, including two-character Han
names, unspaced scripts, recognized Hangul particles and full-width, confusable,
canonical and zero-width variants. Native matching preserves whitespace parts,
its existing punctuation-separated components of at least three characters,
and word boundaries for spaced scripts. The shared fixture records native-only
component matches explicitly; Python retains its whitespace-part contract.
Its confusable mappings mirror the
existing Python detector inventory; the shared controls cover every mapping.
See [summarization](clinical/summarization.md) for the remaining risks from
inflected names and unrecognized suffixes. This guard complements `privacyCheck`.

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
- `cohere-compass` for native-resolution text-and-image generation with North
  Micro Vision
- DeepGrove `maple` causal language models through the dedicated local
  `OpenMedMaple` runtime

ModernBERT, Longformer, EuroBERT, Qwen3, and additional architecture families are still part of the broader rollout work.

## Requirements

- iOS 17+ / macOS 14+ / watchOS 10+ / visionOS 1+
- Xcode 26.4+ (Swift 6.3 toolchain). The pinned `mlx-swift` 0.32.3 manifest
  needs it, and SwiftPM resolves the whole dependency graph, so CoreML-only
  apps need it too.
- For MLX:
  - Apple Silicon Mac, or
  - a real iPhone/iPad device
- For CoreML:
  - a compatible `.mlpackage` or `.mlmodelc` bundle plus `id2label.json`

iOS Simulator is **not** a Swift MLX validation target.
watchOS and visionOS use the CoreML-only `PlatformModel` surface and require an
INT8 Nano-tier artifact. See [Apple Platform Support](./runtimes/apple-platforms.md)
for selection limits and simulator validation.

The Swift packages OpenMedKit links, their licenses, and the files each backend
loads offline are listed in
[Export Toolchains, Artifact Licenses and Redistribution](security/export-redistribution.md).

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
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "4.0.0"),
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

### Privacy Manifests

The `OpenMedKit`, `OpenMedExtensionSupport`, `OpenMedShareExtension` and
`OpenMedActionExtension` products each ship a `PrivacyInfo.xcprivacy` at the
root of their resource bundle. The manifests declare no tracking, no tracking
domains and no collected data types. `OpenMedKit` declares three required-reason
API categories: system boot time (`35F9.1`) for monotonic session and latency
timers, disk space (`E174.1`) for the bundle cache's free-storage check before
it stages or evicts generations, and user defaults (`AC6B.1`) for reading the
managed app configuration that an MDM server installs. The extension targets
use none, so their accessed API lists are empty. The test-support
`OpenMedPrivacySweep` target, which no product exposes, declares system boot
time (`35F9.1`) for its sweep time budget. The scan demo declares user defaults
access with reason `CA92.1` for its saved engine choice and clinical presets.

CI runs an offline scan that maps Apple's required-reason API symbols in the
kit and scan demo Swift sources to their categories: file timestamps, system
boot time, disk space, active keyboards and user defaults, including SwiftUI
`@AppStorage`. Comments and string-literal text are ignored, but code inside
string interpolation is scanned. The scan fails when a category is used without
an approved reason in the covering manifest, when a kit target has no manifest,
or when a manifest declares tracking or collected data:

```bash
python scripts/release/check_privacy_manifests.py
```

### Which manifest SwiftPM resolves

Adding the repository URL resolves the root `Package.swift`. That is the
consumer-facing manifest. `swift/OpenMedKit/Package.swift` is a mirror that the
demos and the iOS, watchOS and visionOS CI lanes build. The two must declare
the same dependencies, version requirements, products, platforms and target
conditions; only target source paths differ. The Swift workflow runs
`scripts/release/check_swift_manifest_drift.py` to compare normalized
`swift package dump-package` output for both. It also runs
`scripts/check_swift_root_consumer.sh` to build a consumer that requires the
root package by version and references `OpenMedMaple` and the Compass runtime
on macOS and for iOS.

Every dependency is a tagged release. `mlx-swift-lm` 3.32.3 is the first tag
that carries the multimodal RoPE state fix that Compass needs. SwiftPM refuses
revision-pinned dependencies inside a package that apps require by version.

Some symbols compile only on some platforms:

| Symbols | Compiled when | Platforms |
|---|---|---|
| `OpenMed`, `OpenMedZeroShotNER`, `OpenMedZeroShotClassifier`, `OpenMedRelationExtractor` | `MLX` and `Tokenizers` are importable | iOS, macOS |
| MLX token-classification, GLiNER and privacy-filter runtimes | `MLX` is importable | iOS, macOS |
| `OpenMedMaple`, including `brief(source:originalIdentifiers:...)` | `MLX`, `MLXNN`, `MLXLMCommon`, `MLXLLM` and `Tokenizers` are importable | iOS, macOS |
| `OpenMedVisionLanguageModel` and `OpenMedVisionLanguageGeneration` (Compass) | `MLXLMCommon` and `Tokenizers` are importable | iOS, macOS |
| `PlatformModel` and the CoreML path | always | iOS, macOS, watchOS, visionOS |

The MLX, `mlx-swift-lm`, `swift-transformers` and ZIPFoundation products are
linked only for iOS and macOS. iOS Simulator builds compile the MLX symbols, but
MLX cannot run there.

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

OpenMedKit prefers the new self-contained artifact layout above. Older MLX repos that were uploaded before `openmed-mlx.json` and tokenizer asset bundling still load from `config.json`, but their tokenizer is no longer fetched implicitly: add the tokenizer files to the directory, or opt into a pinned download as described in [Offline Tokenizer Assets](#offline-tokenizer-assets).

## Quick Start: North Micro Vision

`OpenMedVisionLanguageModel` is the native OpenMedKit Cohere Compass runtime.
It loads the same 4-bit, 5-bit, 6-bit, 8-bit, and bf16 repositories as the
Python OpenMed runtime and performs prompt formatting, native-resolution image
processing, vision encoding, and autoregressive generation on the device:

```swift
import OpenMedKit

let model = try await OpenMedVisionLanguageModel.load(
    modelID: "OpenMed/North-Micro-Vision-Instruct-6bit-mlx"
)

let result = try await model.generate(
    "List the visible medication and dose.",
    imageURL: clinicalDocumentURL,
    maxTokens: 128
)

print(result.text)
```

Text-only generation uses the same loaded model:

```swift
let explanation = try await model.generate(
    "Explain why local processing can improve clinical-document privacy.",
    maxTokens: 96
)
```

For an app-managed or air-gapped artifact, bypass Hub download entirely:

```swift
let model = try await OpenMedVisionLanguageModel.load(
    modelDirectory: bundledOrCachedModelDirectory
)
```

The directory must include `config.json`, `preprocessor_config.json`, tokenizer
and chat-template assets, and the model safetensors payload. OpenMedKit does not
send prompts, images, or generated text to a service and does not provide a
cloud fallback. The initial Hub download is a separate network boundary; cache
or bundle the artifact before processing PHI.

North Micro Vision is useful for document OCR, chart questions, visual
grounding, and structured extraction, but its output is generative. Validate
every consequential field against the source document, apply OpenMedKit privacy
policies before an authorized export, and never auto-trigger diagnosis or
treatment from the result.

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
let tokenizerURL = Bundle.main.url(forResource: "TokenizerAssets", withExtension: nil)!

let openmed = try OpenMed(
    backend: .coreML(
        modelURL: modelURL,
        id2labelURL: labelsURL,
        tokenizerFolderURL: tokenizerURL
    )
)

let entities = try openmed.extractPII("Patient John Doe, SSN 123-45-6789")
```

The convenience initializer remains available:

```swift
let openmed = try OpenMed(
    modelURL: modelURL,
    id2labelURL: labelsURL,
    tokenizerFolderURL: tokenizerURL
)
```

Core ML backends have no default tokenizer. Ship the tokenizer that matches
the model; see [Offline Tokenizer Assets](#offline-tokenizer-assets).

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

The model store validates every manifest path before requesting artifact files,
including all weight candidates and segmenter resources. Paths must be relative
to the snapshot, with no empty, `.` or `..` components, backslashes or escaping
symlinks. The exact tokenizer directory `"."` remains supported as the root
layout used by the OpenMed exporter; tokenizer filenames still require valid
relative paths. Nested directories such as `tokenizer/tokenizer.json` work.

Cache inspection and local artifact loading use the same boundary, including
standard tokenizer files discovered implicitly (config and chat templates). A declared
label map must exist before the snapshot becomes ready. Path rejection throws
`OpenMedModelStoreError.invalidManifestPath` with a controlled reason code and a
fixed message containing no input path, URL or repository content. Rejection
removes a stale ready marker and files newly written by that download where
their parents remain confined; it never follows an escaping link during cleanup.
The downloader repeats the check after each response and before writing. This
does not provide synchronization against another process concurrently replacing
filesystem entries; applications should keep their cache private to the app.

## Task Packs

A task pack picks the model for a job so the app does not name a repository.
OpenMedKit bundles `Resources/task_packs/task_packs.json`, a byte-identical
copy of the shared [task pack manifest](models/task-packs.md), and reads its
`apple` runtime entry: apache-2.0 MLX candidates in rank order.
`OpenMedTaskPackManifest.bundled()` decodes it and fails closed on another
schema or major version, an undeclared field, or a broken rank, license or
notice rule.

Every pack follows the same steps:

- `resolve()` reads the manifest only and returns the candidates this device
  can run, plus the skipped ones with a reason. MLX cannot run on the iOS
  Simulator, so there every candidate is skipped with
  `unsupported_on_simulator` and the status is `unsupportedOnSimulator`.
- `prepare(store:allowDownload:)` loads the best cached candidate through
  `OpenMedModelStore`, falling back in rank order. It downloads only when
  `allowDownload` is `true`; the default `false` throws `not_cached` without
  any network request when nothing is cached.
- Inference runs on device through the existing `OpenMed` APIs. There is no
  cloud fallback, and the packs do not log input text.

`OpenMedTaskPackStore(cacheDirectory:revision:)` selects the cache; the default
`.standard` store is the OpenMed MLX cache. Failures are
`OpenMedTaskPackError` values whose raw value is a fixed reason code.

### `PIIPack`

```swift
let pack = try PIIPack()
try await pack.prepare(allowDownload: userConsentedToDownload)

let found = try pack.extract(note)            // PIIPackExtraction
let redacted = try pack.deidentify(note)      // PIIPackDeidentification
print(redacted.redactedText)
print(found.notice.text)
```

`extract(_:)` runs `OpenMed.extractPII` and returns `PIIPackSpan` values with
the model label, the portable canonical label, Unicode scalar offsets and the
confidence. Spans carry no input text; slice the source with
`span.range(in:)` when you need it. `deidentify(_:policy:)` runs
`OpenMed.deidentify(_:policy:)` under a bundled policy profile
(`hipaa_safe_harbor` by default) and returns the policy result, whose actions
are offset-only. Both results carry the source model id and the pack's
`openmed.notice.pii.review_required` notice, which the app should show with
the output.

### `ClinicalEntityPack`

```swift
let pack = try ClinicalEntityPack()
try await pack.prepare(allowDownload: userConsentedToDownload)

let result = try pack.extract(note)           // ClinicalEntityExtraction
for entity in result.entities {
    print(entity.category, entity.start, entity.end, entity.confidence)
}
print(result.notice.text)
```

`extract(_:)` runs `OpenMed.analyzeText`, normalizes each model label with the
portable OpenMed taxonomy (so `B-PATHOLOGY` and `Disease` normalize like Python's
`normalize_label`), and keeps a mention only when the normalized label is one
of the pack's canonical labels: `CONDITION`, `DISEASE` and `PATHOLOGY` in pack
version 1.0.0. Mentions with any other label are dropped, never assigned a
guessed category. Each `ClinicalEntity` carries the category, the model label,
Unicode scalar offsets, the confidence and the source model id, but no input
text; use `entity.range(in:)` to slice the note. The result carries the
`openmed.notice.clinical_entity.extraction_only` notice: the pack reports
mentions and offsets only and does not diagnose, rank or recommend care.

## Offline Tokenizer Assets

OpenMedKit loads tokenizers from local files only, and makes no network request
unless you explicitly opt in. A tokenizer that drifts or does not match the
model changes token IDs and offsets without any visible error, so a missing
tokenizer fails closed instead of being fetched.

- MLX artifacts carry their tokenizer assets in the model directory.
- Core ML bundles must ship `tokenizer.json` and `tokenizer_config.json` and
  pass the folder as `tokenizerFolderURL`. There is no default
  `tokenizerName`; the name, when given, is only an identity check and never
  triggers a download.

```swift
let openmed = try OpenMed(
    modelURL: modelURL,
    id2labelURL: labelsURL,
    tokenizerFolderURL: Bundle.main.url(forResource: "TokenizerAssets", withExtension: nil)
)
```

When the assets are absent, initialization throws
`OpenMedTokenizerError.missingLocalAssets` before model weights load, with no
request made. Tokenizer errors are value-free: they carry no model path, URL,
repository ID or tokenizer content, and `reasonCode` gives a stable code such
as `tokenizer_assets_missing`.

### Opting into a pinned download

If you cannot ship the tokenizer, opt into network acquisition with
`allowNetworkAccess: true` and an `OpenMedTokenizerPin`. The pin names the
repository, a full 40-character commit SHA (branch names such as `main` are
rejected) and the SHA-256 digest of every file. `allowNetworkAccess: true`
without a pin throws `pinnedRevisionRequired` without making a request.

```swift
let pin = try OpenMedTokenizerPin(
    repositoryID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
    revision: "<40-character commit SHA>",
    files: [
        .init(name: "tokenizer.json", sha256: "<sha256>"),
        .init(name: "tokenizer_config.json", sha256: "<sha256>"),
    ]
)

let openmed = try await OpenMed.load(
    backend: .coreML(
        modelURL: modelURL,
        id2labelURL: labelsURL,
        tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
    ),
    allowNetworkAccess: true,
    tokenizerPin: pin
)
```

With a pin, OpenMedKit:

- requests only `https://huggingface.co/<repo>/resolve/<pinned commit>/<file>`
  for the pinned files;
- checks each file's size (when given) and SHA-256 digest before writing it,
  then installs the snapshot atomically under
  `Caches/OpenMed/TokenizerAssets/`, never under Documents;
- verifies every cached file again before each use, removes a cached copy that
  no longer matches, and reuses a verified copy offline even when
  `allowNetworkAccess` is `false`;
- fails with `identityMismatch` when the pin names a different repository than
  `tokenizerName` or the MLX artifact's declared source model.

### Async loading

`OpenMed.load(backend:maxSeqLength:allowNetworkAccess:tokenizerPin:)` loads the
tokenizer without blocking a thread. The synchronous initializers still work
but block the calling thread on a semaphore while the tokenizer loads; that
blocking bridge is deprecated in favour of `OpenMed.load` and should not be
used from Swift concurrency contexts.

### Migrating from OpenMedKit 3.x

- `allowNetworkAccess` now defaults to `false`. Apps that relied on implicit
  tokenizer downloads must bundle the tokenizer or pass an
  `OpenMedTokenizerPin` with `allowNetworkAccess: true`.
- `OpenMedBackend.coreML` and `OpenMed(modelURL:id2labelURL:)` no longer
  default `tokenizerName` to `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1`.
- The unpinned `resolve/main` download and the `AutoTokenizer.from(pretrained:)`
  fallback, which wrote under Documents, are removed.

## Lifecycle-managed runtime

`OpenMedRuntime` is an actor that owns one on-device text model session and
makes inference cancellable, background-aware and memory-pressure-safe. It
loads the session lazily from local assets, leases it to in-flight requests,
and releases it when idle, under memory pressure, on background expiration or
at shutdown.

```swift
let runtime = OpenMedRuntime(
    backend: .mlx(modelDirectoryURL: modelDirectory),
    // iOS apps opt in to background time; the default never requests it.
    backgroundExecution: OpenMedUIApplicationBackgroundExecution(application: .shared)
)

let task = Task { try await runtime.deidentify(note, policy: Policy.defaultName) }
task.cancel()  // throws OpenMedRuntimeError.cancelled; no partial output

do {
    let entities = try await runtime.extractPII(note)
} catch let error as OpenMedRuntimeError {
    // error.reasonCode, error.retryGuidance — never text, spans or paths
}
await runtime.shutdown()
```

The runtime follows an explicit state machine:

| State | Meaning | Leaves to |
| --- | --- | --- |
| `idle` | No session resident; admission loads one | `loading`, `released` |
| `loading` | The loader is building a session from local assets | `ready`, `idle` (failure, pressure, expiration), `released` |
| `ready` | Session resident, no lease held | `busy(1)`, `idle` (idle release, pressure, expiration), `released` |
| `busy(n)` | `n` inferences hold leases | `busy(n±1)`, `ready`, `draining` |
| `draining` | Admission closed while leases finish or cancel | `idle` (pressure, expiration), `released` (`drain()`, `shutdown()`) |
| `released` | Terminal; no session, no signal observations | — |

**Cancellation.** `analyzeText`, `extractPII` and the three `deidentify`
overloads run tokenization, windowing, model prediction, merging and
serialization behind cooperative checkpoints (`OpenMedInferenceStage`).
Cancelling the calling task, a memory-pressure cancellation, background
expiration or shutdown is observed at the next checkpoint, so the latency to a
terminal state is bounded by one step between checkpoints: one tokenizer pass,
one model window, or one merge or serialization pass. Model work that can
suspend (for example async Core ML prediction) is also interrupted directly.
The runtime always waits for the model call to return before ending its lease,
so a session is never released while in use, and it drops any result produced
after cancellation. `snapshot()` records the stage at which each cancellation
was observed and its monotonic latency.

**Memory pressure.** `OpenMedDispatchMemoryPressureSource` (the default,
backed by `DispatchSource.makeMemoryPressureSource`) feeds warning, critical
and normal levels. Warning and critical refuse new admissions immediately.
By default a warning lets in-flight requests finish (cancelling stragglers
after 2 s) and critical cancels them; both release the session once the last
lease ends, and a session with no lease is released at once. A later request
after `normal` reacquires a session through the loader. Configure this with
`OpenMedRuntimeConfiguration.memoryWarningResponse` and
`memoryCriticalResponse`, or call `handleMemoryPressure(_:)` from your own
signal.

**Background expiration.** `OpenMedUIApplicationBackgroundExecution` begins one
`UIApplication` background task while the runtime is busy and ends it when the
runtime is ready. When the system expires it, in-flight requests stop with
`backgroundExpired`, their output is suppressed, the session is released, and
admission stays closed until `UIApplication.willEnterForegroundNotification`
(or `resumeAfterForeground()`); the next request loads a fresh session rather
than reusing abandoned state. Pass the application explicitly so the library
never touches `UIApplication.shared` from an app extension. macOS, watchOS and
extensions use the default `OpenMedNoBackgroundExecution`, which never expires.
Nothing is persisted to resume work in the background.

**Idle release.** A resident session with no lease is released after
`idleReleaseMilliseconds` (60 s by default; `nil` disables it) or on
`releaseIdleResources()`. All timers use the injectable `OpenMedRuntimeClock`.

**Errors.** Every lifecycle outcome is a value-free `OpenMedRuntimeError`:

| Error | Reason code | Retry guidance |
| --- | --- | --- |
| `cancelled` | `cancelled` | `do_not_retry` |
| `memoryPressure` | `memory_pressure` | `retry_after_memory_pressure_clears` |
| `backgroundExpired` | `background_expired` | `retry_after_foreground` |
| `draining` | `draining` | `retry_with_new_runtime` |
| `released` | `released` | `retry_with_new_runtime` |
| `capacityExceeded` | `capacity_exceeded` | `retry_later` |
| `modelUnavailable` | `model_unavailable` | `check_model_artifact` |
| `inferenceFailed` | `inference_failed` | `do_not_retry` |

Loader and model errors are withheld; `snapshot()` keeps only their Swift type
name. Errors, snapshots and descriptions never contain input text, spans,
labels, model paths or underlying messages.

**Local-only recovery.** `OpenMedRuntime(backend:)` loads with network access
disabled on first use and on every reacquisition, so MLX artifacts must carry
their tokenizer and Core ML bundles need `tokenizerFolderURL`. The runtime never
selects a remote backend. Custom sessions conform to `OpenMedRuntimeModel` and
are supplied through `OpenMedRuntime(loader:)`; the loader must verify its
artifact and build fresh state on every call.

**Synchronous calls.** The existing synchronous `OpenMed` methods keep their
signatures and run the same staged pipeline and checkpoints, so they produce the
same offsets, labels, actions and bytes. They observe only the calling task's
cancellation (and throw `OpenMedRuntimeError.cancelled` when it is cancelled);
they are not admitted, drained, or stopped by memory pressure or background
expiration. Use `OpenMedRuntime` when you need those guarantees.

Device and simulator memory evidence for these paths is not part of the package
test suite; the unit tests drive every signal through deterministic fakes.

## Public API

### `OpenMedBackend`

```swift
public enum OpenMedBackend: Sendable {
    case coreML(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String? = nil,
        tokenizerFolderURL: URL? = nil
    )
    case mlx(modelDirectoryURL: URL)
}
```

### `OpenMed`

```swift
public final class OpenMed {
    public convenience init(
        backend: OpenMedBackend,
        maxSeqLength: Int = 512,
        allowNetworkAccess: Bool = false,
        tokenizerPin: OpenMedTokenizerPin? = nil
    ) throws

    public static func load(
        backend: OpenMedBackend,
        maxSeqLength: Int = 512,
        allowNetworkAccess: Bool = false,
        tokenizerPin: OpenMedTokenizerPin? = nil
    ) async throws -> OpenMed

    public convenience init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String? = nil,
        tokenizerFolderURL: URL? = nil,
        maxSeqLength: Int = 512,
        allowNetworkAccess: Bool = false,
        tokenizerPin: OpenMedTokenizerPin? = nil
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

### `OpenMedTokenizerPin` and `OpenMedTokenizerError`

```swift
public struct OpenMedTokenizerPin: Sendable, Hashable {
    public struct File: Sendable, Hashable {
        public init(name: String, sha256: String, byteCount: Int? = nil)
    }

    public init(repositoryID: String, revision: String, files: [File]) throws
}

public enum OpenMedTokenizerError: Error, Equatable, Sendable {
    case missingLocalAssets
    case pinnedRevisionRequired
    case invalidPin(OpenMedTokenizerPinIssue)
    case identityMismatch
    case downloadFailed(statusCode: Int?)
    case sizeMismatch
    case digestMismatch
    case cacheUnavailable
    case invalidAssets

    public var reasonCode: String { get }
}
```

### `OpenMedRuntime`

```swift
public actor OpenMedRuntime {
    public init(
        configuration: OpenMedRuntimeConfiguration = OpenMedRuntimeConfiguration(),
        memoryPressure: any OpenMedMemoryPressureSource = OpenMedDispatchMemoryPressureSource(),
        backgroundExecution: any OpenMedBackgroundExecutionProvider = OpenMedNoBackgroundExecution(),
        clock: any OpenMedRuntimeClock = OpenMedSystemClock(),
        loader: @escaping @Sendable () async throws -> any OpenMedRuntimeModel
    )
    public init(backend: OpenMedBackend, maxSeqLength: Int = 512, ...)  // iOS and macOS

    public var state: OpenMedRuntimeState { get }
    public func load() async throws
    public func analyzeText(_ text: String, confidenceThreshold: Float = 0.5) async throws -> [EntityPrediction]
    public func extractPII(_ text: String, confidenceThreshold: Float = 0.5, useSmartMerging: Bool = true) async throws -> [EntityPrediction]
    public func deidentify(_ text: String, policy: String, ...) async throws -> PolicyDeidentificationResult
    public func deidentify(_ text: String, policy: Policy, ...) async throws -> PolicyDeidentificationResult
    public func deidentify(_ text: String, method: DeidentificationMethod = .mask, ...) async throws -> DeidentificationResult
    public func releaseIdleResources() -> Bool
    public func handleMemoryPressure(_ level: OpenMedMemoryPressureLevel)
    public func handleBackgroundExpiration()
    public func resumeAfterForeground()
    public func drain() async
    public func shutdown() async
    public func snapshot() -> OpenMedRuntimeSnapshot
}
```

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

### `OpenMedVisionLanguageModel`

```swift
public final class OpenMedVisionLanguageModel: Sendable {
    public static func load(
        modelID: String = "OpenMed/North-Micro-Vision-Instruct-4bit-mlx",
        revision: String = "main",
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> OpenMedVisionLanguageModel

    public static func load(
        modelDirectory: URL
    ) async throws -> OpenMedVisionLanguageModel

    public func generate(
        _ prompt: String,
        imageURL: URL,
        maxTokens: Int = 256,
        temperature: Float = 0
    ) async throws -> OpenMedVisionLanguageGeneration
}
```

The generation result includes decoded text, generated token IDs, prompt and
generation token counts, and prefill/generation timing. Overloads accept
text-only prompts, `CIImage`, or multiple `UserInput.Image` values.

## Concurrency and Thread Safety

OpenMedKit compiles in the Swift 5 language mode, so adding it does not change
your app's language mode. CI builds every first-party target with complete
concurrency checking, the Swift 6 data-race checks reported as warnings, and
fails on any diagnostic in `swift/OpenMedKit/Sources` or
`swift/OpenMedKit/Tests`. The lane builds the root package and its tests on
macOS, and the library and extension targets for the iOS Simulator. To
reproduce it locally:

```bash
python3 scripts/release/check_swift_strict_concurrency.py --platform macos
python3 scripts/release/check_swift_strict_concurrency.py --platform ios
swift test --sanitize=thread --filter 'ConcurrencySafetyTests|SpeechDecoderProgressConcurrencyTests'
```

Public value types are `Sendable`. That covers results, configurations,
policies, limits, snapshots and evidence such as `EntityPrediction`,
`DeidentificationResult`, `Policy`, `OpenMedBackend` and the EDF, WFDB, audio
conversion and voice-embedding value types. You can pass them between tasks and
actors freely.

Runtime types follow one of these contracts:

| Type | `Sendable` | Contract |
|---|---|---|
| `OpenMed`, `NERPipeline`, `PlatformModel`, `OpenMedZeroShotNER`, `OpenMedZeroShotClassifier`, `OpenMedRelationExtractor` | No | Inference is synchronous and has no internal lock. Use each instance from one isolation domain: create and call it from one task or actor, or keep it inside your own actor. Give concurrent workers separate instances. |
| `PIIPack` | No | Same single-isolation-domain rule: prepare and run each pack from one task or actor. |
| `ClinicalEntityPack` | No | Same single-isolation-domain rule as `PIIPack`. |
| `ExtensionRedactionHandler`, `TranscriptLanguageRouter`, `SpeechEndpointAdapter`, `VoiceEmbeddingSession`, `VoiceEmbeddingBuffer`, `VoiceEmbeddingHandle`, `ConvertedAudioSamples`, `WFDBFileSource` | No | Same single-isolation-domain rule. |
| `OpenMedMaple`, `CoreMLSpeechEngine` | Actor | Calls are serialized by the actor. |
| `OpenMedVisionLanguageModel` | Yes | Share one instance. The MLX model container serializes model access. `generate(_:images:)` takes its images as `sending`, so do not keep using them concurrently after the call. |
| `GeneratedTextGuard` | Yes, lock | Share one instance. It runs one detector call at a time. A concurrent caller that finds the detector busy gets `phi_uncertainty` with no text; unscreened text is never released. |
| `StreamingSpeechSession` | Yes, lock | `snapshot()`, `requestCancellation()` and `cancel()` are safe from any task. Appends must still be awaited in sample order. An append made while another is decoding is refused as busy. Every submitted buffer is cleared on every path. |
| `SpeechPCMBuffer`, `CapturePCMBuffer` | Yes, lock | Submission transfers ownership. `clear()` is atomic: a concurrent reader sees the whole buffer or a refusal, never a partially overwritten one. |
| `AmbientSession`, `AmbientAudioCapture` | Yes, lock | One lock guards all state and content. The clock, reader and consumer callbacks run under that lock, so keep them short, synchronous and local. An audio callback that finds the gate busy is dropped and cleared, never queued. |
| `AppleMicrophoneSource` | Yes, lock | Construction and `stop()` are safe from any thread and do not open the microphone. |
| `MultimodalThermalSession` | Yes, lock | Admits one run at a time. Concurrent runs are withheld with reason `resources`. |
| `AgentInMemoryReceiptReplayStore` | Yes, lock | Each digest is claimed at most once across threads. |
| `RedactionShareViewController`, `MultimodalReviewView` | Main actor | Use from the main actor only. `ExtensionItemCodec.plainText(from:)` runs in the caller's isolation domain, so a view controller can pass its extension items directly. |

Every type marked "Yes, lock" is `@unchecked Sendable` and backed by a lock.
`ConcurrencySafetyTests` and `SpeechDecoderProgressConcurrencyTests` call each
of them from many tasks or threads at once, and CI runs those tests under
Thread Sanitizer. A unit test fails when a new `@unchecked Sendable` type
appears in the sources without a matching concurrent-use test.

## Value-Free Errors

OpenMedKit errors are safe to log. `localizedDescription`, `description`,
`String(reflecting:)` and the bridged `NSError` never contain input text,
byte-level token pieces, file or entry names, directory paths, URLs, or strings
read from a manifest or configuration. Error cases carry only stable reason
codes, counts and limits, indices, file roles, and values from OpenMedKit's own
closed vocabularies. Loaders for artifacts, weights, segmenters and the Privacy
Filter tokenizer map Foundation and MLX read failures, which embed the absolute
path, to a role-based case.

| Error | Reason codes and payloads |
|---|---|
| `OpenMedSegmenterError` | `unreadableManifest`, `missingDescriptor`, `missingResource(index:)`, `unreadableResource(index:)`, `invalidDescriptor(DescriptorReason)`. The index points into `resource_files`. `DescriptorReason` raw values: `malformed_manifest`, `unsupported_format_version`, `missing_identity`, `invalid_size_budget`, `unsupported_segmenter_id`, `scripts_or_license_mismatch`, `unexpected_resources`, `invalid_resource_entry`, `resource_role_mismatch`, `resource_escapes_bundle`, `resource_size_mismatch`, `han_dictionary_not_utf8`, `malformed_indic_rules`, `indic_rules_provenance`, `license_notice_not_utf8`, `license_notice_incomplete`, `total_size_mismatch`, `size_budget_exceeded`. |
| `ExtensionRedactionError` | Input cases carry counts and limits. `nonLocalAsset`, `missingAsset` and `invalidAsset` carry an `ExtensionAssetRole`: `bundle_resources`, `model`, `label_map`, `tokenizer_folder`, `tokenizer_file`. `unsupportedModelFormat` has no payload. |
| `PolicyError` | `blankName`, `unknownProfile(allowed:)` (bundled names and aliases; the requested name is not echoed), `missingResource` (a bundled name), `invalidProfile` with `unreadable_resource`, `malformed_resource` or `unsupported_schema_version`. |
| `OpenMedMapleOutputParser.Error` | `missingJSONObject`; `invalidJSONObject` with `type_mismatch`, `value_not_found`, `key_not_found` or `data_corrupted`. |
| `OpenMedMapleRuntimeError` | `missingFiles` (names from `OpenMedMaple.requiredModelFiles`), `unsupportedArchitecture`, `invalidConfiguration` (a fixed OpenMedKit phrase). |
| `OpenMedZeroShotError` | `unsupportedArtifact` with task and family values from the OpenMedKit MLX vocabulary (for example `token-classification` and `bert`), `missingPromptSpec` (a `prompt_spec` field name), `emptyInput`. |
| `NERPipelineError` | `missingOutput("logits")`, `uncompiledModelUnsupported`. |
| `PlatformModelError` | `noCompatibleModel(OpenMedApplePlatform)`, `invalidInputShape`, `sequenceTooLong(actual:maximum:)`. |
| `EDFError`, `OpenMedAECGError`, `BigTiffPreflightError` | A fixed snake_case `code`, `reasonCode` or `category`. |
| String-backed refusals | The raw value is the code: `AgentGovernanceError`, `AmbientAudioCaptureError`, `AmbientSessionError`, `AudioSampleConversionError`, `ClinicalBriefError`, `ClinicalBriefFHIRDocumentError`, `MultimodalContractRefusal`, `MultimodalReviewError`, `OpenMedTaskPackError`, `PreprocessingConformanceError`, `ReviewAdmissionRefusal`, `StreamingSpeechError`, `TranscriptLanguageError`. `SpeechEndpointError` and `MultimodalThermalError` cases have no payload. |

Internal MLX errors follow the same rule and surface through the public
loaders: a Privacy Filter unknown token reports `unknownToken(index:)`, the
position in the encoded sequence, rather than the token piece. Artifact
failures name a file role (`manifest`, `config`, `label_map`), weight failures
a candidate or archive-entry index, and an unrecognized task or family is
reported as `unrecognized` instead of echoing the manifest.

`ValueFreeErrorSentinelTests` and `ExtensionErrorSentinelTests` enforce this.
They enumerate every error type declared under `swift/OpenMedKit/Sources`,
require a sample for every declared case, drive the loaders from a temporary
directory whose name contains a sentinel, and fail if any rendering contains
the sentinel, its byte-level BPE form, the directory or the home path. A new
error type or case without a sample fails the suite. `OpenMedModelStoreError`
is excluded until the model-store work in issue #3644 lands. Errors raised by
dependencies outside these loaders, such as tokenizer loading through
swift-transformers or CoreML compilation, pass through unchanged.

## Supported Swift MLX Families

The current Swift MLX runtime is the BERT-family token-classification path shared across:

- BERT
- DistilBERT
- RoBERTa
- XLM-RoBERTa
- ELECTRA

This is the same first-phase scope as the current public Python MLX BERT-family implementation.

The separate vision-language surface supports Cohere Compass artifacts with
`model_type: cohere_compass`, currently the five OpenMed North Micro Vision
precision variants listed in the Python MLX guide.

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

The Compass integration suite therefore runs through an Xcode test bundle with
Metal resources. It strictly loads every precision and covers deterministic
text extraction, a synthetic clinical-document image, and a synthetic chart.
The ordinary Swift package suite keeps offline processor/configuration tests and
skips multi-gigabyte model execution unless the local artifact environment
variables are explicitly supplied.

## Local speech recognition

The optional `OpenMedSpeech` product provides a pinned Core ML English speech adapter with bounded PCM ownership, partial/final events and cooperative cancellation. See [local Core ML speech](multimodal/apple-speech.md) for model pins, lifecycle rules and offline validation.

## Thermal admission for multimodal work

`MultimodalThermalSession` can gate a local operation against current Apple thermal state, queue age and measured latency. See [Apple session admission](multimodal/thermal-latency-gates.md#apple-session-admission) for the policy table, recovery behavior and value-free evidence.

## Threat model

The [OpenMedKit portable-kit threat model](security/openmedkit-threat-model.md)
covers model downloads, caches, extensions, errors, logs and key handling for
this package. It maps each control to source or tests, or to the issue that
owns the missing work.


## Keychain-backed policy and bundle trust secrets

Policy secrets, such as the surrogate or hash keys a de-identification policy
uses, and portable-bundle trust material, such as verification public keys or
pins, stay out of model bundles, caches and policy files. Code depends on the
small `OpenMedSecretProvider` protocol and passes opaque
`OpenMedSecretReference` values. `OpenMedKeychainSecretProvider` is the bundled
Apple implementation. It mirrors the custody boundary of the Python
[surrogate-key providers](security/surrogate-key-providers.md) and does not
change any surrogate algorithm.

```swift
let configuration = try OpenMedKeychainConfiguration(
    service: "org.example.openmed.secrets",
    accessGroup: .applicationDefault,            // or .named("TEAMID.org.example.shared")
    accessibility: .whenUnlockedThisDeviceOnly,  // the default
    userAuthentication: .notRequired             // or .userPresence, .biometryCurrentSet, ...
)
let provider = OpenMedKeychainSecretProvider(configuration: configuration)
let reference = try OpenMedSecretReference(purpose: .policySecret, label: "study-a.hash-key")

// The caller supplies the bytes; the provider never generates a key.
let stored = try provider.put(callerManagedKey, for: reference)  // version 1
let key = try provider.get(reference, expectedVersion: stored.version)
// Pass key.bytes straight to the local operation; never log, cache or export them.
let rotated = try provider.rotate(reference, to: nextCallerManagedKey, expectedVersion: 1)
```

The configuration is explicit. The access group has no default, and only the
`ThisDeviceOnly` accessibility classes are offered (`whenUnlockedThisDeviceOnly`,
`whenPasscodeSetThisDeviceOnly`, `afterFirstUnlockThisDeviceOnly`). Every call
sends `kSecAttrSynchronizable = false` and `kSecUseDataProtectionKeychain = true`,
including on macOS, so items never sync through iCloud Keychain and never land in
a legacy file keychain. A `userAuthentication` other than `notRequired` stores
the item with a `SecAccessControl` built from the configured class and the
matching `SecAccessControlCreateFlags`, instead of `kSecAttrAccessible`.

A reference hashes its purpose and caller label with SHA-256 and does not keep
the label. The Keychain account is `omk.secret.v1.<digest>`, and descriptions
show only `ref:` plus 16 hexadecimal characters. References encode as
`{"purpose", "digest"}`, so a portable policy can name a secret without holding
it. Use configuration names as labels, never patient or user data.
`OpenMedSecretValue` is not `Codable`, and its description, debug description
and mirror report the reference fingerprint, version and byte count only.
Receipts, metadata and protection reports are `Codable` and value-free.

| Operation | Result |
| --- | --- |
| `put(_:for:)` | Stores version 1. An existing item fails with `secret_duplicate` and is not overwritten. |
| `get(_:expectedVersion:)` | Returns the bytes. With an expected version, a different stored version fails with `secret_rotated` instead of returning it. |
| `metadata(for:)` | Returns the version from attributes only, without reading the bytes. |
| `rotate(_:to:expectedVersion:)` | Replaces the bytes only if the item is still at the expected version. |
| `delete(_:)` | Returns `deleted` or `already_absent`. |
| `protectionReport()` | States the configured protection and whether this device can provide it. |

Rotation is a compare-and-swap: the update matches the metadata the provider
just read, so a concurrent rotation makes it miss and the call fails with
`secret_rotated` rather than overwriting. The Keychain keeps only the current
version. Store earlier material under its own reference first if it must stay
readable.

Every failure is an `OpenMedSecretProviderError` with a fixed `secret_*` code
and no secret bytes, labels, service or account strings, access groups or
platform messages. There is no plaintext, cached or weaker-protection fallback:

- `secret_missing`: no item exists for the reference.
- `secret_locked`: the device is locked or the system did not allow interaction
  (`errSecInteractionNotAllowed`).
- `secret_invalidated`: the item's attributes are readable but its bytes are
  not, for example when a `biometryCurrentSet` item no longer opens after an
  enrollment change, or its stored metadata is malformed or names another
  purpose. Some OS versions report an invalidated item as
  `secret_authentication_failed` instead.
- `secret_rotated`: the stored version is not the one the caller expected.
- `secret_protection_mismatch`: the item's accessibility class, sync flag or
  recorded user-authentication mode differs from the configuration.
- `secret_unsupported`: a write needs a passcode or biometry the device cannot
  provide now, or the platform rejects the configuration. The write is refused
  before any `SecItem` call rather than stored with weaker protection.
- `secret_missing_entitlement` / `secret_unavailable`: the process cannot use
  the data-protection Keychain at all.
- `secret_user_canceled`, `secret_authentication_failed`, `secret_duplicate`,
  `secret_invalid_*` and `secret_unexpected_status` cover the remaining cases.

`protectionReport()` probes `LocalAuthentication` without a prompt and reports
`not_requested`, `available`, `passcode_not_set`, `biometry_not_available`,
`biometry_not_enrolled`, `biometry_lockout`, `unsupported_platform` or
`unknown`. Biometry is `unsupported_platform` on watchOS. The report always
sets `secureHardwareClaimed` to `false`. The items are generic passwords
protected by OS data protection, not Secure Enclave keys, and the adapter makes
no claim that a secret is held in, or never leaves, secure hardware.

Calls can block or show a system authentication prompt, so make them off the
main thread. Tests inject `OpenMedKeychainItemClient`, which wraps
`SecItemAdd`, `SecItemCopyMatching`, `SecItemUpdate` and `SecItemDelete`, and
`OpenMedLocalAuthenticationProbe`. The package tests use an in-memory fake to
cover missing, locked, invalidated, rotated, mismatched and unsupported items.
Set `OPENMED_KEYCHAIN_INTEGRATION=1` to run the optional system-Keychain round
trip. An unsigned `swift test` runner on macOS lacks the data-protection
Keychain entitlement, so that test skips with `secret_missing_entitlement`
there. Locked-device, biometry-enrollment and Secure Enclave behavior needs a
physical device and is not covered by the package tests.
