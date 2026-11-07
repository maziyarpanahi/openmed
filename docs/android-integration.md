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
