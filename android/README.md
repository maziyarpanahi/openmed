# OpenMedKit Android

This directory contains the Gradle build for the `:openmedkit` Android library.

## Demos

- `OpenMedScanDemo` demonstrates CameraX capture, on-device OCR, and highlighted
  identifier redaction.
- [`OpenMedMapleDemo`](OpenMedMapleDemo/README.md) is the offline Maple Clinical
  Studio for PII redaction, entity extraction, relation extraction, and
  source-grounded reasoning/chat. It runs without model weights in synthetic
  preview mode and accepts a pinned, checksummed ONNX Runtime Mobile bundle for
  real on-device generation.

## Install OpenMedKit 3.2.0

OpenMedKit Android is published from immutable OpenMed release tags through JitPack.
Add the repository in the consumer application's `settings.gradle.kts`:

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

After v3.2.0 is published, add the `v3.2.0` coordinate. This candidate does not
yet have a published release tag:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:v3.2.0")
}
```

After tag publication, JitPack resolves the immutable `v3.2.0` tag and builds the `openmedkit`
Android release component as an AAR. Public consumers do not need GitHub
credentials. Use a commit coordinate only when intentionally testing an
unreleased build.

## Toolchain Requirements

| Component | Requirement |
| --- | --- |
| Build JDK | 17 (required by Android Gradle Plugin 8) |
| Gradle | 8.11.1 through the checked-in wrapper; `gradle-wrapper.properties` pins the distribution SHA-256 |
| Android Gradle Plugin | 8.7.3 (8.5.1 or newer is required for 16 KB zip alignment) |
| Kotlin | 2.0.21, with the Compose compiler Gradle plugin for the demos |
| Android SDK | Platform 35 (`compileSdk 35`); Build-Tools 35 or newer only for the optional `zipalign` cross-check |
| Library `minSdk` | 26 |
| Demo `targetSdk` | 35, the Google Play target level for new apps and updates |
| Library bytecode | Java 11 |

Plugin and library versions live in `gradle/libs.versions.toml`. The library
sets no `targetSdk` for consumers; its test APK and lint use 35. Applications
that consume OpenMedKit need Kotlin 1.9 or newer, because the AAR carries
Kotlin 2.0 metadata and depends on `kotlin-stdlib` 2.0. The AAR declares no
minimum compile SDK or Android Gradle Plugin version for consumers.

Compared with the earlier AGP 7.4 / Kotlin 1.8 build, the public API and the
packaged consumer rules (`proguard.txt`) are unchanged with two exceptions: the
AAR no longer ships the generated `org.openmed.openmedkit.BuildConfig` class
(AGP 8 stops generating it by default, and OpenMedKit never used it), and public
enums gain the Kotlin 2 `getEntries()` accessor.

## Build Locally

Install JDK 17 and Android SDK Platform 35, then run:

```bash
cd android
./gradlew test
```

The Android CI gate also builds the release AAR and measures offline library
initialization (packaged catalog load plus downloader/cache setup):

```bash
cd android
./gradlew :openmedkit:verifyAndroidBudgets --continue
```

Reviewable ceilings live in `gradle/budgets.properties`. Measurements are
written to `openmedkit/build/reports/budgets/` and published in the GitHub
Actions job summary. Budget increases should include an explicit review of the
packaged AAR or cold-start regression that requires the additional headroom.

Dependency resolution is limited to Google Maven, Maven Central, and the scoped
OpenMed group on JitPack. Dependency and plugin versions are declared in
`gradle/libs.versions.toml`; avoid hardcoding library versions in module build
files.

The library compiles against SDK 35 and sets `minSdk` to 26. Android 8.0 is the
baseline for on-device inference work because it preserves broad device support
while keeping runtime, storage, and execution APIs modern enough for the planned
local-first pipeline.

## 16 KB Zip Alignment

Devices with 16 KB memory pages map uncompressed native libraries straight from
the APK only when each `.so` entry starts on a 16 KB boundary. Android Gradle
Plugin 8.5.1 and newer packages them that way. The build verifies every debug
demo APK and the OpenMedKit instrumented test APK:

```bash
cd android
./gradlew verifyApkZipAlignment
```

Each application variant also has its own task, for example
`:OpenMedMapleDemo:app:verifyReleaseApkZipAlignment`, and the library has
`:openmedkit:verifyDebugAndroidTestApkZipAlignment`. The tasks run
`scripts/android/check_apk_zip_alignment.py`, which applies the same rules as
`zipalign -c -P 16 4`: stored `.so` entries must be 16 KB aligned, other stored
entries 4-byte aligned, and compressed entries are skipped. A task fails when an
APK has no uncompressed native libraries, so the gate cannot pass vacuously. The
Android CI job repeats the check with the SDK's `zipalign` (Build-Tools 35 or
newer). This covers zip alignment only; the ELF segment alignment of the native
libraries themselves is a separate check.

## R8 And ProGuard

The OpenMedKit AAR includes consumer rules for its ONNX Runtime and DJL native
bindings, service-loaded tokenizer providers, and model-catalog boundary. R8 and
ProGuard apply these rules automatically when a consuming application enables
shrinking or obfuscation; consumers do not need additional OpenMedKit keep rules.

The release packaging check builds the AAR and verifies that every rule from
`openmedkit/consumer-rules.pro` is present in its packaged `proguard.txt`:

```bash
cd android
./gradlew :openmedkit:verifyReleaseConsumerRules
```

### Minified consumer verification

`consumer-r8-smoke` is a minimal host app, separate from the demos, that
consumes the release AAR the way integrators ship it: R8 full mode,
`isMinifyEnabled = true`, `isShrinkResources = true`, the default
`proguard-android-optimize.txt`, ABI splits, and no OpenMedKit rules of its own.
Its code loads the bundled catalog and policy profiles, then runs
`OpenMedKit.fromDirectory` over a tiny synthetic ONNX lookup model and
`tokenizer.json` (generated at build time by
`scripts/android/generate_r8_smoke_model.py`), so tokenizer construction, ONNX
Runtime session creation, `extractPii`, `extractPiiChunked` and `deidentify`
are all reachable through R8. It has three flavors: `openmedkit` (the AAR under
test), `baseline` (the same shell without OpenMedKit) and `missingKeepRule` (the
AAR with `-keep class ai.onnxruntime.** { *; }` removed from a copy of its
`proguard.txt`).

```bash
cd android
./gradlew :consumer-r8-smoke:verifyR8Consumer
# Check a published AAR instead of the one built from this checkout:
./gradlew :consumer-r8-smoke:verifyR8Consumer -PopenmedkitAar=/path/to/openmedkit.aar
```

`scripts/android/check_r8_consumer.py` reads the split APKs and R8's
`mapping.txt`, `usage.txt` and `configuration.txt`, and fails when:

- a class that a bundled JNI library (`ai.onnxruntime`, `ai.djl`, `com.openmed`)
  looks up by name is renamed or removed, or loses a field or method;
- a `Java_*` JNI export no longer matches its class or native method name;
- a `META-INF/services` provider is missing while its service survives;
- the merged configuration contains `-ignorewarnings`, or a `-dontwarn` filter
  that is not in `consumer-r8-smoke/r8-dontwarn-allowlist.txt` (R8 itself
  fails the build on any missing class that no rule silences);
- an OpenMedKit consumer rule did not reach R8;
- desktop JNI binaries are packaged as Java resources, or an ABI split lacks
  `libonnxruntime.so`, `libonnxruntime4j_jni.so` or `libdjl_tokenizer.so`;
- the merged manifest requests `android.permission.INTERNET`.

The `missingKeepRule` variant must fail that gate with `jni_class_not_kept` for
ONNX Runtime classes, so the gate is shown to catch a missing keep rule. The
aggregate task also runs 16 KB zip alignment on the release splits, JVM tests of
the value-free smoke report, and the size budget: for every ABI, the
`openmedkit` release APK minus the `baseline` APK must stay within
`consumerApk.<abi>.maxDeltaBytes` in `gradle/budgets.properties`, separate from
`aar.maxBytes`. Native libraries are stored uncompressed, so these APK bytes are
the install footprint. Reports land in
`consumer-r8-smoke/build/reports/r8-consumer/` and the CI job summary.

The release APKs are debug-signed and launch `SmokeActivity`, which runs the
smoke once and logs value-free lines under the `OpenMedR8Smoke` tag (check IDs,
statuses, counts, labels and offsets, ending with `result=PASS` or
`result=FAIL`). The app removes the `INTERNET` permission that ML Kit merges in.
Running that smoke on a device belongs in the Android emulator lane; this build
gate does not start an emulator.

### Packaging desktop JNI resources

DJL's tokenizers jar ships Linux, macOS and Windows `libtokenizers` builds, and
JNA (pulled in by DJL's `api` module) ships desktop `libjnidispatch` builds, as
Java resources. Android never loads them: DJL loads `libdjl_tokenizer.so` from
`lib/<abi>/`. AGP drops the `.so` copies but packages the rest, which adds about
9.6 MB (28 MB uncompressed) to every APK. Host apps should exclude them:

```kotlin
android {
    packaging {
        resources {
            excludes += listOf(
                "native/lib/**",
                "com/sun/jna/*/*jnidispatch*",
            )
        }
    }
}
```

## Optional Maven Central Publishing

JitPack is the public installation path documented above. The separate, optional
Maven Central path runs from `.github/workflows/android-publish.yml` on `v*` tags
or manual dispatch. Tag releases with Android artifact changes always assemble
the library and run its unit tests. They build and upload a signed Central Portal
bundle only when all signing and Sonatype credentials are configured; otherwise
the validated JitPack release remains the supported public artifact. Manual
dispatch always requires the credentials and runs the full publication path after
the explicit upload confirmation.

Repository secrets required only for Central Portal publication:

- `ANDROID_SIGNING_KEY`: ASCII-armored GPG private key for artifact signing.
- `ANDROID_SIGNING_KEY_PASSWORD`: passphrase for the signing key.
- `SONATYPE_USERNAME`: Central Portal user-token username.
- `SONATYPE_PASSWORD`: Central Portal user-token password.
