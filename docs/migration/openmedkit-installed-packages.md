# Installed OpenMedKit packages and model-store migration

Repository parity tests compile OpenMedKit inside its own source tree. They do
not prove that an app which installs a built package receives the resources,
native libraries and API it needs. The installed-consumer gate closes that gap.
It builds each distributable, installs it into a consumer project staged
outside the checkout and runs offline checks against a synthetic contract.

This page describes the gate, then lists what an existing consumer must change
when it moves to the installed packages. It also lists the combinations that
are unsupported or return a typed incompatibility instead of silently changing
behaviour.

## What the gate verifies

| Lane | Artifact | How the consumer receives it | Runner |
| --- | --- | --- | --- |
| `swiftpm` | Root `Package.swift` and its non-test target sources | A tagged throwaway git repository (`.package(url:exact:)`) or, with `--swift-source path`, a path dependency on the staged copy | macOS |
| `gradle` | `org.openmed:openmedkit` AAR from `publishReleasePublicationToMavenLocal` | `mavenLocal()` pointed at an isolated repository with `-Dmaven.repo.local` | Linux |
| `npm-web` | `openmed` tarball from `npm pack` | `npm install --offline <tarball>` | Linux |
| `npm-react-native` | `@openmed/openmedkit-react-native` tarball from `npm pack` | `npm install --offline <tarball>` | Linux |

Every consumer reads the same synthetic contract,
`tests/consumers/fixtures/installed_consumer_contract_v1.json`. Each one checks
the following without network access:

- **Packaged resources.** Swift loads every bundled policy profile through
  `Bundle.module`. Android loads them through the classloader, reads the model
  catalog through `AssetManager` and checks that the APK contains the catalog,
  policy files and the `arm64-v8a` ONNX Runtime and tokenizer libraries. npm
  resolves the packed ESM, CommonJS and type entry points, and the React Native
  tarball ships its Kotlin and Swift module sources.
- **Policy behaviour.** Profile actions for synthetic labels match the Python
  profiles in `openmed/core/policies`. Aliases resolve. An unknown profile
  returns the typed error documented below.
- **Offsets.** Unicode scalar and UTF-16 offsets agree for astral CJK, emoji
  ZWJ and combining-mark cases, and span replacement produces the expected
  text.
- **One synthetic operation.** Swift applies policy actions to fixture spans
  and round-trips the Python-compatible `DeidentificationResult`. Android runs
  `OpenMedKit.deidentify` over a fake classifier. npm runs `deidentify` over a
  fake pipeline and a fake ONNX Runtime Web session. React Native runs it over a
  fake native module.
- **Offline model store.** Swift stages a synthetic legacy MLX snapshot and
  checks the `missing`, `partial` and `ready` states. A cache hit makes no
  request, and an escaping manifest path is rejected. Android downloads a
  synthetic ONNX artifact through an injected client, then serves it from
  `ModelCache` with a client that fails on any call. A tampered file or
  manifest is rejected.
- **Migration compiles.** Each consumer compiles the "after" snippets from this
  page against the installed package: `Migration.swift`, `Migration.kt`,
  `web-migration.ts` and `react-native-migration.ts`.

The JSON report lists lane, step and check ids, artifact names and SHA-256
digests. It holds no paths, fixture text or command output. A lane whose
toolchain is missing is reported as `skipped`, and only a lane that actually
ran and passed appears in `qualified_lanes`. With `--require`, a skipped lane
fails.

### Run it locally

```bash
# Every lane this machine can run; skips are reported, not qualified.
python scripts/release/verify_installed_consumers.py --report consumers.json

# One lane, failing if it cannot run.
python scripts/release/verify_installed_consumers.py --lane gradle --require
```

The npm lanes need the packages' locked dev dependencies (`npm ci` in
`js/openmedkit-web` and `js/openmedkit-react-native`). Pass `--install-js-deps`
to run `npm ci` where `node_modules` is missing. The Gradle lane needs a JDK
and an Android SDK (`ANDROID_HOME` or `android/local.properties`).
`--gradle-offline` reuses a warm Gradle cache. Dependency resolution for
SwiftPM, Gradle and npm may still use the network on a cold cache. The
consumer checks themselves never do. `--work-dir` must be outside the
checkout.

## Swift (SwiftPM)

Depend on the repository root package, as documented in the
[Swift package guide](../swift-openmedkit.md). The installed package carries
its policy profiles as SwiftPM resources. No repository path is read at run
time.

| Before | After |
| --- | --- |
| Call `OpenMedModelStore.downloadMLXModel(repoID:)` at launch and assume it works offline | Download while online. Offline, check `mlxModelCacheState(repoID:revision:cacheDirectory:) == .ready` before loading. A ready snapshot returns from `downloadMLXModel` without a request |
| `OpenMed(backend: .mlx(modelDirectoryURL:))` with the default `allowNetworkAccess: true` | Pass `allowNetworkAccess: false` so missing tokenizer assets fail instead of downloading |
| `OpenMed(modelURL:id2labelURL:tokenizerName:)` for CoreML | Pass `tokenizerFolderURL:` and `allowNetworkAccess: false` |
| `deidentify(_:)` expecting policy behaviour | Call `deidentify(_:policy:)` with `Policy.defaultName` or a named profile. `deidentify(_:method:)` keeps mask redaction and the Python result schema |

```swift
let openmed = try OpenMed(
    backend: .mlx(modelDirectoryURL: modelDirectory),
    allowNetworkAccess: false
)
let result = try openmed.deidentify(text, policy: Policy.defaultName)
```

Typed incompatibilities:

- `OpenMedModelStoreError.invalidManifestPath`: a cached or downloaded
  `openmed-mlx.json` declares a path outside the snapshot. The cache state
  throws and the ready marker is removed. Re-download from a trusted source.
- `OpenMedModelStoreError.missingWeights` and `.missingManifest`: the snapshot
  is incomplete.
- `PolicyError.unknownProfile(_:allowed:)`: the profile name is not bundled.
  The error lists the accepted names and aliases.

Unsupported combinations:

- The root manifest does not depend on `mlx-swift-lm`. The Maple and Compass
  generative runtimes (`canImport(MLXLMCommon)`) are therefore compiled out
  for root-package consumers. The token-classification facade, policies,
  offsets and model store remain available.
- The `OpenMed` facade is compiled out on watchOS and visionOS. `Policy`,
  `PostProcessing` and `OpenMedModelStore` remain available there.

## Android (Gradle)

The release publication is `org.openmed:openmedkit`. JitPack builds the same
publication under its own coordinates, `com.github.maziyarpanahi:openmed`. The
AAR packages the model catalog and policy assets, and the policy profiles as
classloader resources. ONNX Runtime and DJL tokenizer native libraries arrive
through its declared runtime dependencies.

| Before | After |
| --- | --- |
| `OpenMed(backend)`, the Swift-shaped parity class | `OpenMedKit.fromDirectory(modelDirectory, variant = "int8")` or `OpenMedKit(OpenMedBackend(...))`. `OpenMed` ignores its backend and runs a deterministic pattern recognizer, not the model |
| `OpenMedModelStore.downloadMLXModel(...)` | `ModelCatalog.load(context)`, then `ModelDownloader(context).download(entry)`. The downloader verifies the manifest hash, file sizes and SHA-256 before marking the artifact ready |
| `OpenMedModelStore.isMLXModelCached(...)` | `ModelCache(context).isAvailable(entry)`. A cache hit makes no network call |
| Calling `suspend` APIs with only the AAR on the classpath | Add `org.jetbrains.kotlinx:kotlinx-coroutines-core` yourself. The AAR declares it as a runtime dependency only |

On AGP 7.4, D8 cannot dex the logging and annotation jars that DJL 0.33 pulls.
Pin the API-compatible versions from `android/gradle/libs.versions.toml`:

```kotlin
configurations.configureEach {
    resolutionStrategy.eachDependency {
        when (requested.group to requested.name) {
            "org.slf4j" to "slf4j-api" -> useVersion("1.7.30")
            "com.google.errorprone" to "error_prone_annotations" -> useVersion("2.11.0")
        }
    }
}
```

JVM unit tests that call the facade or `DeidentifyEngine` need Robolectric,
because grapheme snapping uses `android.icu`. Policy loading, offsets and the
model store run on the plain JVM.

Typed incompatibilities:

- `ModelIntegrityException`: the manifest hash, a file size or a file SHA-256
  differs from the catalog entry. Nothing is cached.
- `UnsupportedOperationException`: `OpenMedModelStore.downloadMLXModel` is
  called for a model that is not already marked ready. Android never downloads
  MLX snapshots, and `mlxModelCacheState` only inspects a marker file.
- `IllegalArgumentException`: `PolicyProfiles.load` received an unknown
  profile. Android bundles six profiles. `gdpr_art9_health`, `canada_pipeda`
  and `australia_privacy_act` are currently Swift-only and fail here.

## JavaScript (`openmed` on npm)

| Before | After |
| --- | --- |
| `deidentify(text)` or `extractPii(text)` with no `pipeline` | These load `DEFAULT_MODEL_ID`, a Hub id, through the optional `@huggingface/transformers` peer, so they are not offline. Load once with `loadTokenClassificationPipeline(localDir, { localFilesOnly: true, allowRemoteModels: false })` or `loadOrtWebTokenClassificationPipeline({ modelPath, assetPath, ... })`, then pass `pipeline` |
| Comparing `text_hash` values across calls | An omitted `hashSecret` is random per call. Pass a stable key to correlate spans across calls |
| Expecting policy profiles | The npm package has no policy layer. It redacts every detected span and canonicalizes labels into the PII taxonomy, so clinical labels such as `CONDITION` become `OTHER`. Filter `spans` yourself before calling `spansToRedactedText` |

```ts
const pipeline = await loadTokenClassificationPipeline("./models/pii", {
  localFilesOnly: true,
  allowRemoteModels: false,
});
const result = await deidentify(text, { pipeline, hashSecret });
```

A remote `modelPath` or `assetPath` passed to the ONNX Runtime Web loader is
rejected with an error stating that the path must be local or offline. The
ESM (`import`) and CommonJS (`require`) entry points expose the same surface.

## React Native bridge

`@openmed/openmedkit-react-native` is private and is not published to npm.
Consume the `npm pack` tarball. It ships TypeScript sources plus the
`android/` and `ios/` module sources, and your app's bundler compiles them.
Without a linked native module, every call rejects with
`OpenMedKitRN is not installed in the React Native runtime`. `deidentify`
forwards `hipaa_safe_harbor` unless you pass `policy`. Spans keep native Unicode
scalar offsets and gain canonical policy labels.

## Not covered by the gate

The consumers use synthetic fixtures and fake sessions only. The gate does not
run real model inference, use devices or emulators, sign artifacts or publish
anything. Device and real-model evidence belongs to the separate device and
runtime qualification lanes.
