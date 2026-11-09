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

Use these catalog APIs for Android-runnable artifacts. The separately named
`OpenMedModelStore` parity object mirrors Swift's MLX symbol shapes; its MLX
download method is not the Android ONNX/TFLite download path.

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
