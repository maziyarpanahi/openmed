# Android Integration and Model Catalog

OpenMedKit Android keeps model discovery, model preparation, OCR, and
de-identification on the device. Start with the
[Android Quickstart](android-quickstart.md) for installation and the first text
example; this page covers the complete model-catalog and document-intake
lifecycle.

The Android lifecycle mirrors the Swift `OpenMedModelStore` workflow, but uses
Android-native ONNX/TFLite types:

| Lifecycle step | Android API |
| --- | --- |
| List compatible models | `ModelCatalog.load(context).entries` or `filter(...)` |
| Download and verify a model | `ModelDownloader.download(entry)` |
| Locate the cached directory | `ModelDownloadResult.directory` or `ModelCache.modelDirectory(repoId)` |
| Check whether the cache is ready | `ModelCache.isAvailable(entry)` |
| Inspect size or evict | `sizeBytes`, `totalSizeBytes`, and `evict` |
| Acquire the format the backend opens | `ArtifactAcquisition.acquire(request)` |

Use these catalog APIs, or the
[format-aware artifact acquisition](#format-aware-artifact-acquisition) contract,
for Android-runnable artifacts. Android never runs MLX artifacts: the MLX-named
`OpenMedModelStore` members are deprecated and keep only source compatibility.
See [Migrating from the MLX-named store](#migrating-from-the-mlx-named-store).

## List available models

The Gradle build derives `openmed_model_catalog.jsonl` from OpenMed's canonical
model manifest and bundles it in the AAR. Loading or filtering the catalog is a
local asset read; it does not contact a registry.

```kotlin
import android.content.Context
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.catalog.ModelCatalogEntry

fun availableAndroidModels(context: Context): List<ModelCatalogEntry> {
    val catalog = ModelCatalog.load(context)

    return catalog.entries.filter { entry ->
        entry.formats.any { format ->
            format.startsWith("onnx", ignoreCase = true) ||
                format.startsWith("tflite", ignoreCase = true)
        }
    }
}
```

Each entry exposes its repository ID, runnable formats, languages, tier,
parameter count, license, and reproducibility hash. `ModelCatalog.filter(...)`
can also select an exact format, tier, maximum parameter count, language, or
license, and `byRepoId(...)` resolves a known entry.

## Download with lifecycle progress

`ModelDownloader.download` is blocking, so run it on `Dispatchers.IO`. It emits
`DOWNLOAD_STARTED`, `DOWNLOAD_COMPLETE`, or `CACHE_HIT` events through
`ModelDownloadLogger`. These are lifecycle events, not byte-percentage updates;
show an indeterminate progress indicator between start and completion.

The downloader verifies the catalog reproducibility hash and available file
checksums in a staging directory. It exposes the model as ready only after
integrity checks succeed.

```kotlin
import android.content.Context
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.download.ModelDownloadLogger
import com.openmed.openmedkit.download.ModelDownloadStatus
import com.openmed.openmedkit.download.ModelDownloader
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

enum class ModelPreparationProgress {
    DOWNLOADING,
    READY,
    ALREADY_CACHED,
}

suspend fun prepareModel(
    context: Context,
    entry: ModelCatalogEntry,
    onProgress: (ModelPreparationProgress) -> Unit,
): File = withContext(Dispatchers.IO) {
    val cache = ModelCache(context)
    val downloader = ModelDownloader(
        cache = cache,
        logger = ModelDownloadLogger { event ->
            val progress = when (event.status) {
                ModelDownloadStatus.DOWNLOAD_STARTED ->
                    ModelPreparationProgress.DOWNLOADING
                ModelDownloadStatus.DOWNLOAD_COMPLETE ->
                    ModelPreparationProgress.READY
                ModelDownloadStatus.CACHE_HIT ->
                    ModelPreparationProgress.ALREADY_CACHED
            }
            onProgress(progress)
        },
    )

    downloader.download(entry).directory
}
```

The progress callback runs on the IO dispatcher in this example. Marshal UI
state changes back to the main dispatcher, or collect them through your
ViewModel's state holder. Apps that use the downloader need the standard
network permission for this preparation step:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

Inference itself does not use that permission.

## Locate and check the cache

The default `ModelCache(context)` root is the app-private
`context.noBackupFilesDir/openmed-model-cache` directory. Cache keys are derived
from repository IDs, so do not construct paths yourself.

```kotlin
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import java.io.File

enum class CatalogCacheState {
    MISSING,
    READY,
}

fun cacheState(
    cache: ModelCache,
    entry: ModelCatalogEntry,
): CatalogCacheState = if (cache.isAvailable(entry)) {
    CatalogCacheState.READY
} else {
    CatalogCacheState.MISSING
}

fun cachedModelDirectory(
    cache: ModelCache,
    entry: ModelCatalogEntry,
): File? = cache.modelDirectory(entry.repoId).takeIf {
    cache.isAvailable(entry)
}
```

Android exposes only validated models as ready. Partial downloads stay under a
private staging directory and are removed after a failed download, so callers
normally see `MISSING` or `READY` rather than a usable partial state. Always
check `isAvailable(entry)` before passing `modelDirectory(...)` to
`OpenMedKit.fromDirectory(...)`.

## Format-aware artifact acquisition

`ArtifactAcquisition` resolves the artifact format the configured Android
backend opens. It reuses `ModelDownloader` (manifest digest, per-file SHA-256,
safe paths) and `ModelCache` (staging, atomic promotion, LRU budget) behind one
request and result contract:

- `ArtifactRef` names one immutable artifact: repository, `format` (`ONNX` or
  `TFLITE` on Android), `runtime`, `revision`, `digest` (the catalog
  reproducibility hash), optional `sizeBytes`, and `compatibility` requirements
  such as a minimum runtime version, ONNX opset, or Android API level.
- `ArtifactBackendCapability` describes what the backend can open. A
  requirement the backend does not declare is treated as unsupported.
- `networkConsent` defaults to `DENIED`. Nothing is downloaded unless the
  request carries `ArtifactNetworkConsent.GRANTED`.

```kotlin
import android.content.Context
import android.os.Build
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.artifact.ArtifactAcquisition
import com.openmed.openmedkit.artifact.ArtifactAcquisitionRequest
import com.openmed.openmedkit.artifact.ArtifactAcquisitionResult
import com.openmed.openmedkit.artifact.ArtifactBackendCapability
import com.openmed.openmedkit.artifact.ArtifactFormat
import com.openmed.openmedkit.artifact.ArtifactNetworkConsent
import com.openmed.openmedkit.artifact.ArtifactRef
import com.openmed.openmedkit.artifact.ArtifactRuntime
import com.openmed.openmedkit.catalog.ModelCatalogEntry

suspend fun openPreparedModel(
    context: Context,
    entry: ModelCatalogEntry,
    userApprovedDownload: Boolean,
): OpenMedKit? {
    val request = ArtifactAcquisitionRequest(
        ref = ArtifactRef.fromCatalogEntry(
            entry,
            format = ArtifactFormat.ONNX,
            runtime = ArtifactRuntime.ONNX_RUNTIME,
        ),
        backend = ArtifactBackendCapability.onnxRuntime(
            androidSdkLevel = Build.VERSION.SDK_INT,
        ),
        networkConsent = if (userApprovedDownload) {
            ArtifactNetworkConsent.GRANTED
        } else {
            ArtifactNetworkConsent.DENIED
        },
    )
    return when (val result = ArtifactAcquisition(context).acquire(request)) {
        is ArtifactAcquisitionResult.Ready ->
            OpenMedKit.fromDirectory(result.artifact.directory)
        // Inspect result.state and result.reason?.code; never log the request.
        else -> null
    }
}
```

| State | Meaning |
| --- | --- |
| `CACHED` | Served offline after the recorded digest, file set, and sizes matched. |
| `VERIFIED` | Served offline after every file was re-hashed (`cacheVerification = FULL`). |
| `DOWNLOADED` | Fetched with consent, verified, and promoted atomically. |
| `UNSUPPORTED` | The format, runtime, or declared requirements do not fit the backend, the repository does not publish the format, or the payload lacks its format signature. MLX and Core ML are always unsupported; signed portable bundles stay unsupported until their verifier ships. |
| `CANCELLED` | Stopped by `cancel(ref)`; downloaded bytes are kept for resume. |
| `FAILED` | Missing consent, digest or size mismatch, unsafe manifest path, HTTP status, or an interrupted transfer. |

Every `Ready` artifact (`CACHED`, `VERIFIED`, `DOWNLOADED`) is verified,
compatible with the backend, and safe to open; its payload files are read-only.
Unsupported formats and incompatible requirements are rejected before any
cache or network access, so an incompatible artifact is never fetched.

Interrupted or cancelled transfers keep their bytes in the private staging
area. The next acquisition resumes each file with an HTTP `Range` request and
re-hashes the whole file before promotion; a resumed file that fails the
re-check is discarded and fetched once from the start. Staged bytes are never
reported as ready, and `cacheState(ref)` reports them as `PARTIAL`.

Concurrent callers for the same artifact and cache root share one transfer.
Cancelling one caller's coroutine leaves the transfer running for the others;
it stops once no caller is waiting, or for everyone when `cancel(ref)` is
called. Results, including the ones an `ArtifactAcquisitionLogger` receives,
carry only states, formats, digests, sizes, and value-free reason codes:
`toString()` never includes URLs, signed query values, tokens, cache paths, or
file contents.

### Migrating from the MLX-named store

The Android `OpenMedModelStore` scaffold mirrored Swift's MLX symbol names but
never downloaded anything. Its MLX-named members are now `@Deprecated` with IDE
replacements, and the store gains format-aware entry points that delegate to
`ArtifactAcquisition`:

| Deprecated member | Replacement |
| --- | --- |
| `OpenMedModelStore.downloadMLXModel(repoID, revision, cacheDirectory)` | `OpenMedModelStore.acquireArtifact(request, cacheDirectory)` |
| `OpenMedModelStore.cachedMLXModelDirectory(repoID, revision, cacheDirectory)` | `OpenMedModelStore.cachedArtifact(ref, cacheDirectory)?.directory` |
| `OpenMedModelStore.isMLXModelCached(repoID, revision, cacheDirectory)` | `OpenMedModelStore.artifactCacheState(ref, cacheDirectory) == ArtifactCacheState.READY` |
| `OpenMedModelStore.mlxModelCacheState(repoID, revision, cacheDirectory)` | `OpenMedModelStore.artifactCacheState(ref, cacheDirectory)` |
| `OpenMedMLXModelCacheState` | `ArtifactCacheState` |

The deprecated members keep compiling. `downloadMLXModel` still returns a
directory that already carries the legacy ready marker and otherwise throws
`UnsupportedOperationException`. Legacy scaffold directories that would resolve
outside the cache root, including through a symlink, now raise
`IllegalArgumentException`, and the state queries report them as `missing`.

## Task packs

A task pack picks a model for a job instead of a repository. The Kotlin packs
read `packs/task_packs.json`, a byte-identical copy of the shared
[task pack manifest](models/task-packs.md), and take its Android `onnx`
candidates in rank order. Resolution reads only that file; it never downloads,
opens a socket or loads a model.

`prepare(downloader, allowDownload)` is blocking, so call it off the main
thread. It opens the first cached candidate that matches its catalog
integrity hash. When none is cached it returns
`TaskPackPreparation(failure = NOT_CACHED)` and the downloader's client is
never called, unless `allowDownload` is `true`; then it downloads candidates
in rank order until one passes the manifest and file checks. The prepared
model holds a cache lease until `close()`. Other failures are typed too:
`NO_CANDIDATE`, `DOWNLOAD_FAILED` and `LOAD_FAILED`.

```kotlin
import android.content.Context
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.packs.PiiPack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun redact(context: Context, note: String, userApprovedDownload: Boolean): String? {
    PiiPack(ModelCatalog.load(context)).use { pack ->
        val preparation = withContext(Dispatchers.IO) {
            pack.prepare(ModelDownloader(context), allowDownload = userApprovedDownload)
        }
        if (preparation.failure != null) return null // Show preparation.failure.
        val result = pack.deidentify(note)
        // result.detector, result.modelId and result.notice label the output.
        return result.value.redactedText
    }
}
```

`PiiPack.extract` runs `extractPii` and `PiiPack.deidentify` runs the
policy-driven `deidentify` of `OpenMedKit` on the prepared model. Without a
prepared model both throw `TaskPackException(NOT_PREPARED)`. Pass
`allowLocalFallback = true` to run the deterministic local recognizer of
`OpenMed` instead; the result then reports
`TaskPackDetector.LOCAL_RECOGNIZER` and a `null` model id. Every
`TaskPackResult` carries the pack's notice key and default English text:
detected identifiers are redaction candidates that still need review.
Diagnostics go through `SafeLog` only and never contain input text.

### Clinical entities

`ClinicalEntityPack` prepares a DiseaseDetect candidate the same way and
`extract(text)` runs `analyzeText` on it. A span is kept only when its model
label normalizes, through the portable label taxonomy, to one of the pack's
canonical labels (`CONDITION`, `DISEASE`, `PATHOLOGY`); every other label is
dropped rather than guessed into a category. Each `ClinicalEntity` carries the
`category`, Unicode scalar `start` and `end` offsets, the `confidence` and the
source `modelId`, but no surface text. The pack reports condition mentions
only: it does not diagnose, rank or recommend care, and its result carries
that notice. The synthetic vectors in
`tests/fixtures/parity/clinical_entity_pack_v1.json` fix the expected output
for the Kotlin and Swift pack tests.

## Offline-first deployment

Separate model preparation from document processing. There are two supported
deployment patterns:

1. **Pre-warm the catalog cache.** Call `prepareModel` during an explicit setup
   flow while the device has an approved network connection, before accepting
   clinical input. A later cache hit makes no network calls.
2. **Ship the model with the app.** Package a compatible exported model and its
   tokenizer sidecars as an app asset or managed-install payload, copy it to an
   app-private versioned directory, and open that directory with
   `OpenMedKit.fromDirectory(...)`. See
   [Android ONNX Export](export-onnx-android.md) for the required artifact
   layout.

After either path is ready, OCR, tokenization, model inference, and
de-identification run locally. No model is fetched at inference time, input PHI
is never uploaded or fetched from a service, and there is no cloud fallback.
An offline app that ships its model does not need network permission at all.

The DJL tokenizer library that OpenMedKit uses runs usage telemetry whenever a
tokenizer is created: it sends HTTP requests to the cloud instance-metadata
address unless DJL's `OPT_OUT_TRACKING` switch is set. OpenMedKit sets that JVM
system property before it loads a tokenizer, so `OpenMedKit.fromDirectory(...)`
opens no socket. The property also applies to any other DJL tokenizer created
in the same process.

## Compose OCR, intake, and de-identification

The core `openmedkit` artifact ships no OCR engine. `DocumentIntake` accepts an
image or PDF URI, renders or decodes each page with Android framework APIs,
hands it to the `OcrAdapter` you supply as a platform-neutral `OcrImage`
(a `Bitmap` or encoded bytes plus the clockwise rotation that makes it
upright), and preserves page/token offsets. Image URIs honor their EXIF
orientation, and `DocumentPage.width`/`height` report the upright size.

The ML Kit adapter, `MlKitOcrAdapter`, lives in the optional `openmedkit-mlkit`
artifact. Add it next to the core dependency only when you want ML Kit OCR
(see [Optional ML Kit OCR module](#optional-ml-kit-ocr-module) for what it
adds to your app):

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:v3.2.0")
    // Optional ML Kit OCR adapter. Use the same tag as the core artifact.
    implementation("com.github.maziyarpanahi:openmed-mlkit:v3.2.0")
}
```

Feed the extracted text directly into the local `OpenMedKit` facade. The URI in
this example points to a test document containing only fictional data:

> Patient: Jane Roe (synthetic). MRN 000-00-0000. Contact 555-0100.

```kotlin
import android.content.Context
import android.net.Uri
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.intake.DocumentIntake
import com.openmed.openmedkit.ocr.MlKitOcrAdapter
import java.io.File

suspend fun redactSyntheticDocument(
    context: Context,
    syntheticDocumentUri: Uri,
    readyModelDirectory: File,
): String {
    val intake = MlKitOcrAdapter().use { ocr ->
        DocumentIntake(context = context, ocrAdapter = ocr)
            .extract(syntheticDocumentUri)
    }

    return OpenMedKit.fromDirectory(readyModelDirectory).use { kit ->
        kit.deidentify(intake.text).redactedText
    }
}
```

Keep `DocumentIntakeResult.offsetMap`, `pages`, and `tokens` if the UI needs to
map de-identification spans back to source pages. `DocumentIntakeResult.evidence`
also maps each recognized region back to a box on the original page, with the
page rotation and crop, in the coordinates Swift OpenMedKit uses; see
[Native Document Intake Coordinates](multimodal/native-document-intake.md).
To share a sanitized copy of a redacted document, export it with
`DocumentRedactionExporter`; see
[Native Redaction Export](multimodal/native-redaction-export.md).

`extract` returns only complete intakes and throws `DocumentIntakeException`,
whose message is value-free reason codes, when a page is missing, access is
lost, or coordinates cannot be mapped unambiguously. Call `intake(uri)` to get a
`DocumentIntakeReport` with an explicit `complete`, `partial` or `refused`
outcome instead. OCR adapters report token boxes in the upright image's pixels. Test the full flow with
synthetic fixtures before introducing real clinical documents, and keep human
review in the downstream workflow. In unit tests, pass `FakeOcrAdapter` (in the
core artifact) instead of an engine.

To use another on-device OCR engine, implement `OcrAdapter.recognize(OcrImage)`
and build the result with `buildOcrResult(...)` from `RecognizedOcrBlock`,
`RecognizedOcrLine`, and `RecognizedOcrToken` values. Every adapter then shares
the same block, line, and token ordering and the same offset map.

### Optional ML Kit OCR module

`openmedkit-mlkit` is the only OpenMedKit Android artifact that depends on
Google ML Kit (`com.google.mlkit:text-recognition`). Adding it changes your
app's license and network posture:

- **License.** ML Kit is distributed under Google's ML Kit terms, not the
  Apache License 2.0 that covers OpenMedKit. Review those terms before you ship
  the module.
- **Network permissions and telemetry.** ML Kit's dependency graph includes
  `com.google.android.datatransport`, which ML Kit uses to send usage metrics.
  Its manifests merge `android.permission.INTERNET` and
  `android.permission.ACCESS_NETWORK_STATE` into your app.
- **Auto-starting components.** ML Kit registers `MlKitInitProvider` and
  `MlKitComponentDiscoveryService`, and datatransport adds its scheduler
  service and receiver.

Recognition itself runs on device with the bundled Latin model; OpenMedKit
sends no document content anywhere. The core `openmedkit` artifact carries
none of the items above. Its build runs a dependency-boundary check that fails
if the release POM, the resolved runtime classpath, or any manifest a consumer
would inherit contains ML Kit, Play services, datatransport, or Firebase
artifacts, a network permission, or a dependency-contributed provider,
service, or receiver:

```bash
cd android
./gradlew :openmedkit:verifyDependencyBoundary :openmedkit-mlkit:verifyDependencyBoundary
```

The `:openmedkit-mlkit` task is the negative control. It fails unless the same
scan finds the ML Kit stack, which proves the check can see it.

### Migrating from the bundled ML Kit adapter

**Breaking:** OpenMedKit Android no longer includes ML Kit in the core
artifact, and the OCR contracts no longer use ML Kit types.

1. Add `openmedkit-mlkit` next to the core dependency if you use
   `MlKitOcrAdapter`. Its package, `com.openmed.openmedkit.ocr`, is unchanged.
2. `DocumentIntake` has no default OCR engine. Pass
   `ocrAdapter = MlKitOcrAdapter()` (or your own adapter) explicitly.
3. `OcrAdapter.recognize` now takes an `OcrImage` instead of an ML Kit
   `InputImage`. Wrap bitmaps with `OcrImage.fromBitmap(bitmap, rotation)` and
   encoded captures with `OcrImage.fromEncodedBytes(bytes, rotation)`. If you
   already hold an `InputImage`, for example from `InputImage.fromMediaImage`,
   call the `MlKitOcrAdapter.recognize(InputImage)` overload directly.
4. Custom `OcrAdapter` implementations and test doubles must override
   `recognize(image: OcrImage)`. `FakeOcrAdapter.calls` now records `OcrImage`
   values.
5. `OcrResult`, its token offsets, and `DocumentIntakeResult.offsetMap` are
   unchanged.

## Lifecycle, cancellation, and memory pressure

Long-lived hosts should own one `OpenMedRuntime` at application scope instead
of a bare `OpenMedKit`. The runtime opens sessions lazily through an
`OpenMedSessionFactory`, holds a lease for every operation, and closes a native
session only after its final lease is released. It moves through `IDLE`,
`LOADING`, `READY`, `BUSY`, `DRAINING`, and `RELEASED`, and publishes a
value-free `status` flow.

```kotlin
import com.openmed.openmedkit.OpenMedRuntime
import com.openmed.openmedkit.OpenMedSessionFactory
import com.openmed.openmedkit.OpenMedTrimMemorySource

val runtime = OpenMedRuntime(
    OpenMedSessionFactory.fromCache(cache, entry),
)
runtime.bindLifecycle(OpenMedTrimMemorySource(applicationContext))

// In a coroutine; cancelling it cancels the operation.
val result = runtime.deidentify(syntheticNote)

// When the owner is torn down.
runtime.close()
```

Cancellation follows normal coroutine rules. OpenMedKit checks for cancellation
before tokenization, after native inference, and after decoding, merging, and
policy application, so a cancelled caller resolves after at most one in-flight
native window run and never receives a late result. Outcomes stay
distinguishable: a completed call returns its result, a cancelled caller gets
`CancellationException`, and refusals raise `OpenMedRuntimeException` with a
stable, value-free `reason.code` and `retryGuidance`:

| Reason | When | Retry guidance |
| --- | --- | --- |
| `runtime_closed` | `close()` ran before or during the call | `RECREATE_RUNTIME` |
| `stale_generation` | `replaceModel` ran during the call; its result was discarded | `RETRY_NOW` |
| `admission_paused_background` | the host is in the background | `RETRY_AFTER_FOREGROUND` |
| `admission_paused_memory_pressure` | critical memory pressure | `RETRY_AFTER_BACKOFF` |
| `session_load_failed` | the factory failed to open or verify the model | `RETRY_AFTER_BACKOFF` |
| `session_reload_limit` | consecutive load failures reached the policy bound | `RECREATE_RUNTIME` |

`close()` is idempotent: it stops admission, cancels running work, and releases
each native session once its run returns. `drainAndClose()` instead lets running
work finish with its results. Load failures never carry the underlying message,
which may contain artifact paths; `causeKind` holds only the failing type name.

The default `OpenMedRuntimePolicy` maps trim-memory signals as follows. Active
work is never cancelled by memory pressure or backgrounding, and every
reacquisition re-runs the factory, which re-verifies the cached bundle.

| Signal | Pressure | Default action |
| --- | --- | --- |
| `TRIM_MEMORY_RUNNING_MODERATE`, `TRIM_MEMORY_UI_HIDDEN` | `MODERATE` | none |
| `TRIM_MEMORY_RUNNING_LOW`, `TRIM_MEMORY_BACKGROUND`, `TRIM_MEMORY_MODERATE` | `LOW` | release idle sessions; busy ones close when idle |
| `TRIM_MEMORY_RUNNING_CRITICAL`, `TRIM_MEMORY_COMPLETE`, `onLowMemory` | `CRITICAL` | also refuse new work until 30 s pass, the host returns to the foreground, or a `NONE` signal arrives |

OpenMedKit does not depend on `androidx.lifecycle`. To refuse new work while
the process is in the background and release idle sessions, adapt
`ProcessLifecycleOwner` in the app:

```kotlin
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner
import com.openmed.openmedkit.OpenMedLifecycleSource
import java.io.Closeable

val processLifecycle = OpenMedLifecycleSource { observer ->
    val lifecycle = ProcessLifecycleOwner.get().lifecycle
    val bridge = object : DefaultLifecycleObserver {
        override fun onStart(owner: LifecycleOwner) = observer.onHostForeground()
        override fun onStop(owner: LifecycleOwner) = observer.onHostBackground()
    }
    lifecycle.addObserver(bridge)
    Closeable { lifecycle.removeObserver(bridge) }
}
runtime.bindLifecycle(processLifecycle)
```

Set `OpenMedRuntimePolicy(admitInBackground = true)` for hosts that must keep
working in the background, for example from a foreground service. Bridges such
as the React Native module use `OpenMedHostScope`, which owns the bridge's
coroutine scope and runtime and cancels both when the host is invalidated.

## Storage, eviction, and logging

`ModelCache` defaults to a 2 GiB budget. When a completed download would exceed
the budget, it evicts least-recently-used ready models while protecting the
model just downloaded. Applications can set a smaller budget and provide an
explicit removal control:

```kotlin
val cache = ModelCache(
    context = context,
    cacheBudgetBytes = 512L * 1024L * 1024L,
)

val usedBytes = cache.totalSizeBytes()
val removed = cache.evict(entry.repoId)
```

A model opened through `OpenMedSessionFactory.fromCache` holds a
`ModelCacheLease` while its session is open. Budget trimming skips leased
models, and `evict` returns `false` for them instead of deleting files under a
running session.

The default directory is excluded from Android Auto Backup. Uninstalling the
app or clearing its storage removes the models; otherwise use `evict` when the
user removes a downloaded model or when organizational retention policy
requires deletion.

OpenMedKit does not log model contents, document inputs, OCR text, detected
spans, or redacted output. Download lifecycle events contain only the model
repository ID, status, and size. Application code must preserve that guarantee:
never put input text, document URIs or filenames, OCR tokens, span surface text,
or output text in logs, analytics, crash reports, cache metadata, or temporary
files.

## Shared C++ runtime (libc++_shared.so)

`libdjl_tokenizer.so` from `ai.djl.android:tokenizer-native` 0.33.0 links
against the NDK's shared C++ runtime, `libc++_shared.so`, which that artifact
does not ship. The OpenMedKit AAR therefore packages `libc++_shared.so` from
NDK r28c (28.2.13676358) for `arm64-v8a`, `armeabi-v7a`, `x86` and `x86_64`.
It adds 850,380 to 1,253,544 bytes per ABI to an app. Without it the
tokenizer fails to load with
`dlopen failed: library "libc++_shared.so" not found`.

An app can package only one `libc++_shared.so` per ABI. If another dependency
also ships one (React Native does, and so does any dependency whose native
code is built with `ANDROID_STL=c++_shared`), the build fails in
`mergeNativeLibs` with `2 files found with path 'lib/<abi>/libc++_shared.so'`.
Keep one copy in the app module's `build.gradle.kts`:

```kotlin
android {
    packaging {
        jniLibs {
            pickFirsts += "**/libc++_shared.so"
        }
    }
}
```

React Native apps that apply the React Native Gradle plugin
(`com.facebook.react`, 0.77) already get this rule for the app module.

The NDK caveat: libc++ ABI compatibility across NDK releases is maintained
where possible but not guaranteed, and a library built with a newer NDK can
need symbols that an older runtime lacks. The copy you keep should come from
the newest NDK that built any native library in the app. OpenMedKit's copy is
from r28c, and `libdjl_tokenizer.so` 0.33.0 was built with r27c, so a kept copy
older than r27c can break the tokenizer. Between dependencies, AGP keeps the
copy from the dependency it merges first; a copy from the app module itself
(its `jniLibs` or its own `c++_shared` CMake or ndk-build output) takes
precedence, with a warning in AGP 8.7. After changing the NDK or the
dependency set, load the tokenizer on a device.

libc++ is licensed `Apache-2.0 WITH LLVM-exception`. The OpenMedKit POM
declares only OpenMed's Apache-2.0 license, so a license tool that reads POMs
will not list libc++: add it to your app's open-source notices yourself. See
[Android NDK runtime libraries](security/kit-license-inventory.md#android-ndk-runtime-libraries).

## 16 KB page-size devices

Some Android 15 and later devices use 16 KB memory pages. On those devices the
dynamic linker cannot load a 64-bit native library unless every ELF `PT_LOAD`
segment is aligned to at least 16 KB. Google Play also requires 16 KB support
for new apps and updates that target Android 15 (API 35) or higher. Two separate
conditions must hold for every `arm64-v8a` and `x86_64` library in your APK or
App Bundle:

- **ELF alignment.** Each `.so` must be linked with
  `-Wl,-z,max-page-size=16384`. NDK r28 and newer do this by default; older NDKs
  need the flag.
- **Zip alignment.** An uncompressed `.so` must start on a 16 KB boundary inside
  the APK. Android Gradle Plugin 8.5.1 and newer packages it that way.

The native libraries that OpenMedKit pulls in have this status:

| Library | Dependency | 64-bit ELF alignment |
| --- | --- | --- |
| `libonnxruntime.so`, `libonnxruntime4j_jni.so` | `com.microsoft.onnxruntime:onnxruntime-android:1.25.1` | 16 KB |
| `libc++_shared.so` | NDK r28c, packaged in the OpenMedKit AAR | 16 KB |
| `libmlkit_google_ocr_pipeline.so` | `com.google.mlkit:text-recognition:16.0.1` | 16 KB |
| `libdjl_tokenizer.so` | `ai.djl.android:tokenizer-native:0.33.0` | **4 KB** |

`ai.djl.android:tokenizer-native` 0.33.0 is the newest release on Maven
Central, and no 16 KB-aligned release exists yet
([deepjavalibrary/djl#3815](https://github.com/deepjavalibrary/djl/issues/3815)).
On a 16 KB page-size device the tokenizer library fails to load, so every
model-backed OpenMedKit call fails when the tokenizer is constructed. Devices
with 4 KB pages are not affected. OpenMedKit will move to the first
16 KB-aligned DJL release. Until then, plan 16 KB device support and Play
submissions around this gap.

Check your own release build with the same offline checker the OpenMedKit
build uses. It reads the ELF program headers of every `lib/<abi>/*.so` in an
APK (or `jni/<abi>/*.so` in an AAR) and prints only library names, ABIs, and
alignment values:

```bash
python3 scripts/android/check_elf_page_alignment.py \
  --require-native-libraries app/build/outputs/apk/release/app-release.apk
```

It exits with status 1 when a 64-bit library is aligned below 16 KB. Pair it
with `zipalign -c -P 16 4 app-release.apk` (Build-Tools 35 or newer) to check
zip alignment. The [Android build README](https://github.com/maziyarpanahi/openmed/blob/master/android/README.md#16-kb-elf-alignment)
describes the Gradle gates and the reviewed, expiring allowlist that covers the
DJL tokenizer.

The Flutter plugin builds `libopenmed_ffi.so` with
`-Wl,-z,max-page-size=16384` and binds ONNX Runtime 1.25.1 on Android, so the
plugin's own library and its ONNX Runtime dependency are 16 KB aligned on both
64-bit ABIs. The Flutter CI job runs the checker on a consumer APK.

## Emulator lane

The JVM unit tests run under Robolectric, where the ONNX Runtime and DJL
tokenizer native libraries are never loaded. The emulator lane runs the
instrumented tests in `android/openmedkit/src/androidTest` on an Android
emulator with the real native libraries and a tiny synthetic model:

| Test | What it checks |
| --- | --- |
| `NativeRuntimePackagingTest` | The device has the lane's page size, and `libonnxruntime.so`, `libonnxruntime4j_jni.so`, `libc++_shared.so` and `libdjl_tokenizer.so` load from the APK |
| `OnnxRuntimeParityTest` | A CPU session over the fixture's token IDs gives the same per-token labels as Python ONNX Runtime, with scores within `1e-4` |
| `TokenizerParityTest` | The DJL JNI tokenizer gives the same token IDs and Unicode scalar offsets as Python `tokenizers` |
| `OpenMedKitFromDirectoryTest` | `OpenMedKit.fromDirectory(dir, "fp32")`, with its default execution-provider selection, returns the expected `analyzeText` and `extractPii` spans: the same labels and offsets, confidences within `1e-3` |
| `OfflineGuardTest` | Negative controls for the offline checks below |

The test APK removes the `INTERNET` permission that ML Kit's transport
libraries merge in, so the kernel refuses every IPv4 and IPv6 socket of the test
process. Each lane test also runs under `NoNetworkSocketRule`, which fails the
test when a `java.net` socket is constructed or an IPv4/IPv6 socket descriptor
is open before or after it.

The fixture lives in `android/openmedkit/src/androidTest/assets/emulator-lane/`.
`model/` holds a randomly initialised two-layer BERT token classifier (seeded,
32 hidden units, labels `NAME`, `DATE` and `ID`) exported with the Android
profile of `openmed.onnx.convert`: one `model.onnx` of about 145 KB with every
weight inline, plus `tokenizer.json` and `id2label.json`. `expected.json` holds
synthetic case text, the Python ONNX Runtime token predictions and the derived
spans, with labels, offsets and scores only. One case contains a character
outside the Basic Multilingual Plane, where Unicode scalar and UTF-16 offsets
differ. The `EmulatorLaneFixtureContractTest` JVM test runs the Kotlin decoder
over the recorded token predictions, so the expected spans cannot drift from the
facade's aggregation rules. Regenerate or check the fixture with the `onnx`
extra installed:

```bash
python scripts/android/generate_emulator_lane_fixture.py generate
python scripts/android/generate_emulator_lane_fixture.py check
```

To run the lane locally, start an emulator or connect a device so that
`adb devices` lists exactly one target, then run either the tests alone or the
CI lane script, which also checks the page size and runs the negative control:

```bash
cd android && ./gradlew :openmedkit:connectedDebugAndroidTest
scripts/android/run_emulator_lane.sh --lane local --page-size-kb 4 --negative-control
```

The negative control rebuilds the test APK with
`-PopenmedNativePackagingFault=missing-native-libraries`, which strips the ONNX
Runtime, C++ runtime and tokenizer libraries from every ABI, and requires
`NativeRuntimePackagingTest` to fail with `native_library_unloadable` for
`onnxruntime`, `c++_shared` and `djl_tokenizer`. The property only changes the
instrumented test APK.

Each check writes one JSON record to logcat under the `OpenMedEmulatorLane` tag
(`adb logcat -s OpenMedEmulatorLane:I`). Records contain only the lane name,
case identifiers, check names, labels, offsets, outcomes, reason codes and
timings. The lane script saves them, with the JUnit results, under
`android/openmedkit/build/reports/emulator-lane/<lane>/`.

The `emulator-lane` job in `.github/workflows/android-ci.yml` runs two lanes on
KVM-accelerated `ubuntu-latest` runners:

- `x86_64-api35`: the API 35 Google APIs x86_64 image with 4 KB pages, plus the
  negative control.
- `x86_64-api35-16kb`: the API 35 16 KB page-size Google APIs x86_64 image. DJL
  `tokenizer-native` 0.33.0 ships `libdjl_tokenizer.so` with 4 KB LOAD-segment
  alignment, which a 16 KB kernel refuses to load. This lane passes
  `--known-unloadable djl_tokenizer`: the packaging test requires the load to
  fail, and the tokenizer-dependent tests are skipped. ONNX Runtime and
  `libc++_shared.so`, which are 16 KB aligned, are still checked. When a
  16 KB aligned tokenizer build is adopted, the packaging test fails with
  `known_blocker_resolved` and the argument must be removed.

## Runnable demos and related guides

- [`android/OpenMedDemo`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedDemo)
  shows a Compose text de-identification flow and model status UI.
- [`android/OpenMedScanDemo`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedScanDemo)
  shows on-device capture, ML Kit OCR through the optional `openmedkit-mlkit`
  module, and redaction.
- [Android Quickstart](android-quickstart.md) covers dependency installation and
  the shortest catalog-to-de-identification path.
- [Android ONNX Export](export-onnx-android.md) documents compatible model
  artifacts.
- [Android Span Parity](android-parity.md) defines cross-platform offset and
  label guarantees.
- [OpenMedKit portable-kit threat model](security/openmedkit-threat-model.md)
  maps download, cache, logging and key-handling threats to controls and
  owning issues.

- [Export Toolchains, Artifact Licenses and Redistribution](security/export-redistribution.md)
  lists the runtime libraries the AAR brings into an app, their licenses, and
  the ML Kit posture.
