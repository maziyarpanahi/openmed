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

The DJL tokenizer library that OpenMedKit uses runs usage telemetry whenever a
tokenizer is created: it sends HTTP requests to the cloud instance-metadata
address unless DJL's `OPT_OUT_TRACKING` switch is set. OpenMedKit sets that JVM
system property before it loads a tokenizer, so `OpenMedKit.fromDirectory(...)`
opens no socket. The property also applies to any other DJL tokenizer created
in the same process.

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

## Emulator lane

The JVM unit tests run under Robolectric, where the ONNX Runtime and DJL
tokenizer native libraries are never loaded. The emulator lane runs the
instrumented tests in `android/openmedkit/src/androidTest` on an Android
emulator with the real native libraries and a tiny synthetic model:

| Test | What it checks |
| --- | --- |
| `NativeRuntimePackagingTest` | The device has the lane's page size, and `libonnxruntime.so`, `libonnxruntime4j_jni.so` and `libdjl_tokenizer.so` load from the APK |
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
Runtime and tokenizer libraries from every ABI, and requires
`NativeRuntimePackagingTest` to fail with `native_library_unloadable`. The
property only changes the instrumented test APK.

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
  fail, and the tokenizer-dependent tests are skipped. ONNX Runtime, which is
  16 KB aligned, is still checked. When a 16 KB aligned tokenizer build is
  adopted, the packaging test fails with `known_blocker_resolved` and the
  argument must be removed.

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
