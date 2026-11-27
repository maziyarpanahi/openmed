# OpenMedKit Android

This directory contains the Gradle build for the OpenMedKit Android libraries:

- `:openmedkit` is the core library. It carries no OCR engine, no Google Play
  services or ML Kit artifacts, no vendor telemetry, and no network permission.
- `:openmedkit-mlkit` is the optional ML Kit Text Recognition adapter
  (`MlKitOcrAdapter`). Adding it brings Google's ML Kit terms, ML Kit's
  auto-initializing components, and the datatransport usage-metrics stack with
  the `INTERNET` and `ACCESS_NETWORK_STATE` permissions into the host app. See
  [Optional ML Kit OCR module](../docs/android-integration.md#optional-ml-kit-ocr-module).

## Demos

- `OpenMedScanDemo` demonstrates CameraX capture, on-device OCR through the
  optional `:openmedkit-mlkit` module, and highlighted identifier redaction.
- [`OpenMedMapleDemo`](OpenMedMapleDemo/README.md) is the offline Maple Clinical
  Studio for PII redaction, entity extraction, relation extraction, and
  source-grounded reasoning/chat. It runs without model weights in synthetic
  preview mode and accepts a pinned, checksummed ONNX Runtime Mobile bundle for
  real on-device generation.

## Install OpenMedKit 4.0.0

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

After v4.0.0 is published, add the `v4.0.0` coordinate. This candidate does not
yet have a published release tag:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:v4.0.0")
}
```

OCR is optional. To use the ML Kit adapter, also add the `openmed-mlkit`
coordinate at the same tag as the core artifact:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed-mlkit:v4.0.0")
}
```

After tag publication, JitPack resolves the immutable `v4.0.0` tag and builds the `openmedkit`
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
| Android NDK | 28.2.13676358 (r28c), pinned as `android-ndk` in `gradle/libs.versions.toml`; `:openmedkit` packages its `libc++_shared.so` (see [C++ runtime](#c-runtime-libc_sharedso)) |
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

Install JDK 17, Android SDK Platform 35 and NDK 28.2.13676358
(`sdkmanager --install "ndk;28.2.13676358"`), then run:

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

The core dependency boundary is a separate gate. It fails when the core
release POM, resolved runtime classpath, or any manifest a consumer would
inherit contains ML Kit, Play services, datatransport, or Firebase artifacts, a
network permission, or a dependency-contributed provider, service, or receiver.
The optional module runs the same scan as a negative control and fails unless
it detects the ML Kit stack:

```bash
cd android
./gradlew :openmedkit:verifyDependencyBoundary :openmedkit-mlkit:verifyDependencyBoundary
```

Reviewable ceilings live in `gradle/budgets.properties`. Measurements are
written to `openmedkit/build/reports/budgets/` and published in the GitHub
Actions job summary. Budget increases should include an explicit review of the
packaged AAR or cold-start regression that requires the additional headroom.

Dependency resolution is limited to Google Maven, Maven Central, and the scoped
OpenMed group on JitPack. Dependency and plugin versions are declared in
`gradle/libs.versions.toml`; avoid hardcoding library versions in module build
files.

Every module's resolved versions are locked in its committed `gradle.lockfile`,
and `gradle/verification-metadata.xml` holds a SHA-256 checksum for every
downloaded artifact. A dependency change that does not regenerate both fails
the build. See
[Android dependency locking](../docs/security/dependency-policy.md#android-dependency-locking-and-verification)
for the regeneration commands.

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
libraries themselves is a separate check, described next.

## 16 KB ELF Alignment

A 64-bit native library can only be mapped on a 16 KB page-size device when
every ELF `PT_LOAD` segment has `p_align` of at least 16384.
`scripts/android/check_elf_page_alignment.py` reads the program headers of each
`.so` in AARs (`jni/<abi>/`), APKs and JARs (`lib/<abi>/`), loose files and
directories. It prints only library names, ABIs, alignment values, and the file
name of the containing archive. 32-bit libraries are reported and skipped. Two
Gradle gates run it:

```bash
cd android
./gradlew :openmedkit:verifyReleaseNativeLibraryAlignment
./gradlew verifyApkElfAlignment
```

- `:openmedkit:verifyReleaseNativeLibraryAlignment` checks the release AAR plus
  every AAR resolved on its release runtime classpath, which is what consumers
  receive. It is part of `:openmedkit:verifyAndroidBudgets`.
- `verifyApkElfAlignment` checks every debug demo APK and the OpenMedKit
  instrumented test APK. Each application variant also has its own task, for
  example `:OpenMedMapleDemo:app:verifyReleaseApkElfAlignment`.

Both gates fail when no 64-bit library is found, so they cannot pass vacuously.

`gradle/native-page-alignment-allowlist.toml` holds reviewed, expiring
exceptions. Each entry names the library, its ABIs, the Maven artifact that
ships it, an owner, an expiry date, a tracking issue, and the upstream fix it
waits for. A covered library is reported as `ALLOW`. After its `expires` date it
fails again. The release AAR gate runs with `--strict-allowlist`, so an entry
that no longer matches a misaligned library also fails and must be removed. A
unit test checks that each entry's artifact version matches
`gradle/libs.versions.toml`, so bumping the dependency means revisiting the
entry.

The only current entry is `libdjl_tokenizer.so` from
`ai.djl.android:tokenizer-native:0.33.0`, which is linked with 4 KB alignment
on `arm64-v8a` and `x86_64`. It is the newest release on Maven Central, and the
upstream fix is tracked in
[deepjavalibrary/djl#3815](https://github.com/deepjavalibrary/djl/issues/3815).
Until a 16 KB-aligned release exists, the DJL tokenizer cannot load on 16 KB
page-size devices. The other libraries in the dependency graph are 16 KB
aligned: ONNX Runtime 1.25.1, the NDK r28c `libc++_shared.so` that OpenMedKit
packages, the ML Kit OCR pipeline, and the CameraX 1.4 libraries in the scan
demo.

## C++ Runtime (libc++_shared.so)

`ai.djl.android:tokenizer-native` 0.33.0 ships `libdjl_tokenizer.so` linked
against the NDK's shared C++ runtime (`DT_NEEDED libc++_shared.so`), but no
dependency ships that runtime. Without it the tokenizer cannot load on a device
(`dlopen failed: library "libc++_shared.so" not found`), so OpenMedKit packages
it:

- `android.ndkVersion` pins the NDK (`android-ndk` in `gradle/libs.versions.toml`,
  currently 28.2.13676358, r28c). The Android CI, publish and installed-consumer
  workflows and `jitpack.yml` install it with `sdkmanager` before building; AGP
  does not download an NDK for this.
- For every variant, `copy<Variant>NdkRuntimeLibraries` copies `libc++_shared.so`
  for `arm64-v8a`, `armeabi-v7a`, `x86` and `x86_64` from
  `toolchains/llvm/prebuilt/<host>/sysroot/usr/lib/<triple>/` of
  `<sdk>/ndk/<revision>` into a generated `jniLibs` directory of the `main`
  sources. It fails when that NDK is missing or reports another revision. AGP
  then strips the copies with the same NDK, so the release AAR is
  byte-for-byte reproducible. No binary is committed. The demos consume
  `project(":openmedkit")` and pin no NDK, so their debug APKs carry the
  unstripped copy.
- The stripped runtime adds 850,380 (`armeabi-v7a`) to 1,253,544 (`arm64-v8a`)
  bytes per ABI to an app and about 1.5 MB to the release AAR. Its 64-bit
  builds are 16 KB aligned.
- The R8 consumer gate requires `libc++_shared.so` in every ABI split, the
  emulator lane's `NativeRuntimePackagingTest` loads it, and the lane's
  negative control strips it.
- libc++ is licensed `Apache-2.0 WITH LLVM-exception`. The kit license gate
  reviews it as an `ndk` package; see
  [Android NDK runtime libraries](../docs/security/kit-license-inventory.md#android-ndk-runtime-libraries).

### Apps that already ship libc++_shared.so

An APK holds one `libc++_shared.so` per ABI. When another dependency also ships
it, `mergeNativeLibs` fails with
`2 files found with path 'lib/arm64-v8a/libc++_shared.so'`. Keep one copy in the
app module:

```kotlin
android {
    packaging {
        jniLibs {
            pickFirsts += "**/libc++_shared.so"
        }
    }
}
```

Which copy is kept matters. The NDK keeps libc++ ABI-compatible across releases
where it can, but does not guarantee it, and a library built with a newer NDK
can need symbols that an older `libc++_shared.so` lacks. Keep the copy from the
newest NDK that built any native library in the app: OpenMedKit's is from r28c,
and `libdjl_tokenizer.so` 0.33.0 was built with r27c. Between dependencies, AGP
keeps the copy from the one it merges first. A copy from the app module
itself (its `jniLibs`, or the output of its own `c++_shared` CMake or
ndk-build build) takes precedence over dependency copies; AGP 8.7 logs a
warning for that case. Load the tokenizer on a device after changing the NDK
or the dependency set.

React Native apps that apply the React Native Gradle plugin
(`com.facebook.react`, 0.77) already get this `pickFirsts` rule for the app
module.

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
  `libonnxruntime.so`, `libonnxruntime4j_jni.so`, `libc++_shared.so` or
  `libdjl_tokenizer.so`;
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
`result=FAIL`). The app removes any `INTERNET` permission that a dependency
merges in. Running that smoke on a device belongs in the Android emulator lane;
this build gate does not start an emulator.

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
