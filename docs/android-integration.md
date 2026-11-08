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
map de-identification spans back to source pages. Test the full flow with
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
  shows on-device capture, ML Kit OCR through the optional `openmedkit-mlkit`
  module, and redaction.
- [Android Quickstart](android-quickstart.md) covers dependency installation and
  the shortest catalog-to-de-identification path.
- [Android ONNX Export](export-onnx-android.md) documents compatible model
  artifacts.
- [Android Span Parity](android-parity.md) defines cross-platform offset and
  label guarantees.
