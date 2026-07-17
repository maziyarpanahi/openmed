# Android Quickstart (OpenMedKit)

OpenMedKit Android is the on-device counterpart to the
[Swift Package (OpenMedKit)](swift-openmedkit.md). It loads a de-identification
model from the on-device model catalog and redacts clinical text entirely on the
handset — no network access and no cloud fallback for PHI.

This page walks through the first-run path: add the Gradle dependency, load a
model from the catalog, and redact a synthetic clinical note.

## Requirements

| Requirement | Value |
| --- | --- |
| Minimum Android API | 26 (Android 8.0) |
| Compile / target SDK | 33 |
| Build JDK | 11 |
| Language | Kotlin (coroutines) |

The library performs inference on device via ONNX Runtime Mobile. See
[Android ONNX Export](export-onnx-android.md) for producing a compatible model
artifact and [Android Span Parity](android-parity.md) for the cross-platform
span guarantees.

## Install

OpenMedKit Android is published from immutable OpenMed release tags through
JitPack. Add the scoped repository in the consumer application's
`settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven {
            url = uri("https://jitpack.io")
            content { includeGroup("com.github.maziyarpanahi") }
        }
    }
}
```

Then add the release coordinate in the module `build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:v1.9.1")
}
```

JitPack resolves the immutable `v1.9.1` tag and publishes the `openmedkit`
Android release component as an AAR. Public consumers do not need GitHub
credentials.

## Quick Start: Kotlin

The snippet below discovers a model in the on-device catalog, resolves its
cached directory, constructs an `OpenMedKit` facade, and de-identifies a
synthetic, non-PHI clinical note. `deidentify` is a `suspend` function, so call
it from a coroutine.

```kotlin
import android.content.Context
import com.openmed.openmedkit.OpenMedBackend
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalog
import java.io.File

suspend fun redactSyntheticNote(context: Context): String {
    // 1. Load the bundled on-device model catalog and pick an entry.
    val catalog = ModelCatalog.load(context)
    val entry = catalog.entries.first()

    // 2. Resolve the model's on-device directory from the local cache.
    val cache = ModelCache(rootDirectory = File(context.filesDir, "openmed-models"))
    require(cache.isAvailable(entry.repoId)) {
        "Model ${entry.repoId} is not present on device yet."
    }
    val modelDirectory = cache.modelDirectory(entry.repoId)

    // 3. Build the local-first backend and facade.
    val backend = OpenMedBackend(modelDirectory = modelDirectory)

    // 4. De-identify a synthetic (fictional, non-PHI) clinical note.
    val syntheticNote =
        "Patient: Jane Roe (synthetic). MRN 000-00-0000. " +
            "Seen on 2024-01-02 for routine follow-up; contact 555-0100."

    return OpenMedKit(backend).use { kit ->
        val result = kit.deidentify(syntheticNote)
        result.redactedText
    }
}
```

By default `deidentify` applies the bundled `hipaa_safe_harbor` policy profile;
pass `policy = "..."` to select another bundled profile. The returned
`PolicyDeidentificationResult` exposes `redactedText`, the applied `policyName`,
and the per-span `actions`.

## Public API

| Symbol | Purpose |
| --- | --- |
| `OpenMedBackend` | Local-first configuration pointing at an on-device model directory and tokenizer assets. Performs no network access. |
| `OpenMedKit` | `Closeable` facade. `deidentify(text, policy, …)` redacts under a bundled policy; `extractPii(text, …)` and `extractPiiChunked(text, …)` return detected `EntityPrediction` spans. |
| `ModelCatalog` | Reads the bundled `openmed_model_catalog.jsonl` asset into `ModelCatalogEntry` rows (`repoId`, `formats`, `languages`, …). |
| `ModelCache` | Resolves and manages on-device model directories keyed by `repoId`. |

## Local-First Guarantees

OpenMedKit Android is deliberately local-first:

- **On-device only.** All tokenization, inference, and de-identification run on
  the handset. `OpenMedBackend` reads model and tokenizer assets from a local
  directory and performs no network access.
- **No cloud fallback for PHI.** There is no remote inference path; clinical text
  is never transmitted off the device for processing.
- **No PHI in logs.** The library does not log input text, detected spans, or
  redacted output. Do not add application logging that echoes clinical text.
- **Not a clinical decision.** Output is an assistive redaction aid, not medical
  advice or a clinical decision. De-identification is best-effort and must be
  reviewed before any downstream use; it does not guarantee removal of every
  identifier.

## Demo Apps

Two runnable Android demos exercise the on-device path end to end:

- [`android/OpenMedDemo`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedDemo)
  — Compose de-identification demo over a synthetic note.
- [`android/OpenMedScanDemo`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedScanDemo)
  — on-device capture-and-redact demo.

## See Also

- [Swift Package (OpenMedKit)](swift-openmedkit.md) — the Apple-platform counterpart.
- [Swift-Kotlin API Parity](swift-kotlin-parity.md) — cross-platform API alignment.
- [Android ONNX Export](export-onnx-android.md) — producing a compatible on-device model.
- [Model Manifest](model-manifest.md) — the model catalog and reproducibility metadata.
