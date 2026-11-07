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

## Compose OCR, intake, and de-identification

`DocumentIntake` accepts an image or PDF URI, uses the on-device
`MlKitOcrAdapter`, and preserves page/token offsets. Feed its extracted text
directly into the local `OpenMedKit` facade. The URI in this example points to a
test document containing only fictional data:

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
    val intake = DocumentIntake(
        context = context,
        ocrAdapter = MlKitOcrAdapter(),
    ).extract(syntheticDocumentUri)

    return OpenMedKit.fromDirectory(readyModelDirectory).use { kit ->
        kit.deidentify(intake.text).redactedText
    }
}
```

Keep `DocumentIntakeResult.offsetMap`, `pages`, and `tokens` if the UI needs to
map de-identification spans back to source pages. Test the full flow with
synthetic fixtures before introducing real clinical documents, and keep human
review in the downstream workflow.

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

## Runnable demos and related guides

- [`android/OpenMedDemo`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedDemo)
  shows a Compose text de-identification flow and model status UI.
- [`android/OpenMedScanDemo`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedScanDemo)
  shows on-device capture, ML Kit OCR, and redaction.
- [Android Quickstart](android-quickstart.md) covers dependency installation and
  the shortest catalog-to-de-identification path.
- [Android ONNX Export](export-onnx-android.md) documents compatible model
  artifacts.
- [Android Span Parity](android-parity.md) defines cross-platform offset and
  label guarantees.
