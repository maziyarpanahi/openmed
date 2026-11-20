# OpenMedKit Dependency License Inventory

This is OpenMed's checked-in license review for the dependencies of the
portable kits: OpenMedKit for Swift, the Android library and demo apps, and the
JavaScript, React Native, Electron, Tauri and Flutter wrappers under `js/`. The
Python package has its own [dependency license
inventory](license-inventory.md); both gates share the same permissive
allowlist and the same restricted classes (GPL, LGPL, AGPL, SSPL,
source-available, Commons Clause, proprietary).

The gate is offline. Run it from the repository root:

```bash
python scripts/licenses/kit_inventory.py
python scripts/licenses/kit_inventory.py --report build/kit-license-report.json
```

It parses the committed manifests with the Python standard library, looks each
package up in the tables below, and fails closed. It never resolves packages,
runs SwiftPM, Gradle, npm, Cargo or Flutter, or contacts a registry. CI runs it
in the `repo-policy` job next to the Python dependency license policy.

## What the gate reads

| Kit | Committed manifests |
| --- | --- |
| Swift (root package) | `Package.swift`, `Package.resolved` |
| Swift (nested package) | `swift/OpenMedKit/Package.swift` |
| Swift demo apps | the committed `Package.resolved` of `OpenMedDemo` and `OpenMedScanDemo` |
| Android | `android/settings.gradle.kts`, the root and every included module's build file, `android/gradle/libs.versions.toml` |
| Android NDK runtime libraries | `bundledNdkLibraries` in an Android module's build file |
| Web, Electron, React Native, Tauri | `js/*/package-lock.json` with the sibling `package.json` |
| Electron packaging harness | `js/openmedkit-electron/harness/package-lock.json` with its `package.json` |
| React Native native modules | the bridge's podspec and `android/build.gradle`, and those of the parity probe in `example/probe` |
| React Native parity example | `js/openmedkit-react-native/example/package-lock.json` with its `package.json`, and `example/probe/package.json` |
| Tauri sidecar | `js/openmedkit-tauri/src-tauri/Cargo.lock` with its `Cargo.toml` |
| Flutter | `js/openmedkit-flutter/pubspec.yaml`, the iOS and macOS podspecs, `android/build.gradle` |

The Electron packaging harness and the React Native parity example are private
test apps that never ship in a kit artifact. Every package they install is
treated as development-only: their direct dependencies are reviewed below, and
their transitive lock entries are counted rather than reviewed. The native
modules the parity example builds are gated through their own podspec and
build file. The registry of these test apps is `DEVELOPMENT_HARNESS_MANIFESTS`.

Android modules are discovered from `include(...)` in the settings file, so a
new module such as `:openmedkit-mlkit` is gated without a registry change. The
registry of the other files is `DEFAULT_KIT_MANIFESTS` in
`scripts/licenses/kit_inventory.py`; a unit test fails when a committed kit
manifest is not read by the gate. `swift/OpenMedKit/Package.resolved` and
`js/openmedkit-flutter/pubspec.lock` are git-ignored, so the gate does not read
them; the parsers support both formats once they are committed.

## Runtime and development dependencies

Every dependency occurrence carries a usage class, and the report keeps them
apart:

- **runtime** dependencies ship in, or are linked into, a kit artifact or demo
  app: SwiftPM packages used by non-test targets and all resolved pins, Gradle
  `implementation`, `api`, `compileOnly`, `runtimeOnly` and custom
  configurations, NDK runtime libraries copied into an AAR, npm production,
  optional and peer dependencies, Cargo crates reachable from
  `[dependencies]`, pub `dependencies`, and podspec dependencies. Unknown or
  unreferenced declarations are treated as runtime.
- **development** dependencies never ship: SwiftPM packages used only by test
  targets, Gradle `test*`, `androidTest*`, `debug*`, `kapt`, `ksp`,
  `annotationProcessor`, `lintChecks` and `classpath` configurations and
  plugins, npm `devDependencies`, Cargo `[dev-dependencies]` and
  `[build-dependencies]`, pub `dev_dependencies`, and podspec test or app
  specs.

Every direct declaration needs a review row, whatever its usage. Transitive
lockfile entries that the lockfile itself marks development-only (npm
`"dev": true`, Cargo crates reachable only through development tables) are
counted in the report but not reviewed one by one, because they are build
tooling that never ships. Every other transitive lock entry needs a row.

## Policy

A runtime dependency passes only when every token of its reviewed expression is
in the shared permissive allowlist. `WITH` is accepted only for exceptions that
add permissions (`LLVM-exception` and the Swift runtime library
`Swift-exception`). Compound expressions stay conservative: `MIT OR GPL-3.0`
is restricted. Where upstream offers a copyleft option next to permissive ones,
the row records the permissive licenses OpenMed elects and the review note
keeps the upstream expression.

A development dependency may also carry `EPL-1.0`, `EPL-2.0` or
`LicenseRef-Public-Domain` (for example JUnit 4 and `org.json`), because it is
never redistributed. Strong copyleft, source-available, proprietary and unknown
expressions fail for development dependencies too.

The gate fails, naming only the manifest, ecosystem and package, when:

- a dependency has no review row (`unreviewed`);
- a review row is unknown, restricted, or outside the allowlist for its usage
  and no scoped exception covers it (`denied`);
- a registered manifest is missing or cannot be parsed statically, for example
  a Gradle coordinate built from variables in its group or artifact, or a
  `files(...)` or `fileTree(...)` binary dependency.

The one local binary the gate accepts is first-party. The R8-minified consumer
app in `android/consumer-r8-smoke` verifies the OpenMedKit release AAR itself,
so it consumes that AAR, built from this checkout or passed by path, and a copy
without one keep rule as `files(...)` of build outputs. Only that build file,
listed in `FIRST_PARTY_LOCAL_BINARY_MANIFESTS`, and only `files(...)` of build
variables qualify: a quoted path or `fileTree(...)` still fails there, as does
a local binary in any other module. Every third-party runtime dependency of
that app comes from the reviewed version catalog.

Reports contain ecosystems, package names, repository-relative manifest paths,
usage classes and decision codes. They never contain license text, versions,
URLs or local paths.

## Scoped exceptions

### `mlkit-optional-ocr`: Google ML Kit Text Recognition

`com.google.mlkit:text-recognition` is distributed under Google's proprietary
ML Kit terms, so its row is `LicenseRef-Proprietary-Google-ML-Kit-Terms` and it
is classified as restricted. It is allowed only as the optional on-device OCR
adapter, and only in these build files:

- `android/openmedkit-mlkit/build.gradle.kts`, the optional ML Kit module
  introduced by #4451;
- `android/OpenMedScanDemo/app/build.gradle.kts`, the scan demo that exercises
  the adapter.

The core artifact, `android/openmedkit/build.gradle.kts`, no longer depends on
ML Kit and is not covered. The gate prints every use of the exception, so ML
Kit never passes silently. Declaring it in any other manifest, or changing its
reviewed expression, fails the gate. ML Kit's own transitive Google Play
services artifacts appear only in the Gradle lockfiles, which this gate does
not read.

## Android NDK runtime libraries

`ai.djl.android:tokenizer-native` ships `libdjl_tokenizer.so`, which links
against the NDK's shared C++ runtime (`DT_NEEDED libc++_shared.so`), but no
dependency ships that runtime. The `:openmedkit` build therefore copies
`libc++_shared.so` out of the NDK it pins through `android.ndkVersion`
(`android-ndk` in `android/gradle/libs.versions.toml`, NDK r28c,
28.2.13676358) into the release AAR for `arm64-v8a`, `armeabi-v7a`, `x86` and
`x86_64`, and AGP strips it with the same NDK. No binary is committed. Every
app that uses OpenMedKit Android ships this library.

The library has no Maven coordinate, so the gate reads the
`bundledNdkLibraries` list in the module's build file and reviews each entry
as an `ndk` package. A list that is not a static set of `lib*.so` string
literals fails the gate, and a new entry fails until it has a row.

libc++ is part of the LLVM project, licensed `Apache-2.0 WITH LLVM-exception`.
The NDK's `NOTICE.toolchain` also carries libc++'s legacy license (the
University of Illinois/NCSA and MIT licenses, offered as alternatives) for
code contributed before LLVM relicensed. The LLVM exception covers runtime
portions compiled into an application's own object code. A copy of
`libc++_shared.so` redistributes the library itself, so an app that ships it
should carry the libc++ license text with its other open-source notices.
The OpenMedKit POM declares only OpenMed's own Apache-2.0 license, so license
tools that read POMs do not list libc++.

## Speech recognition (WhisperKit)

`argmax-oss-swift` provides WhisperKit for the `OpenMedSpeech` product. Its
LICENSE at the pinned 1.1.1 tag is MIT, so it is a permissive runtime
dependency of both Swift packages. Speech model weights are downloaded by the
application and are outside this inventory.

## Maintaining the tables

Add a row before adding a dependency to any kit manifest, and keep one row per
ecosystem and package. Record the SPDX expression from the package's own
license file or registry metadata; normalize legacy `MIT/Apache-2.0` notation
to `MIT OR Apache-2.0`. Do not paste URLs, credentials or application data into
the tables. Rows that no manifest uses any more are counted as
`unused_review_rows` in the report; remove them when a dependency is dropped.

## Limits

- Only declared Android dependencies are gated, not their transitive
  closure: the gate does not read the committed `gradle.lockfile` files,
  which the vulnerability scan covers.
- Rows are per package, not per version. A license change in a new release is
  caught only when the row is reviewed again.
- SwiftPM transitive pins of the nested `swift/OpenMedKit` package come from
  the demo apps' committed resolved files; the package's own resolved file is
  not committed.

## Review tables

### SwiftPM

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| swiftpm | `argmax-oss-swift` | `MIT` | WhisperKit for `OpenMedSpeech` (#4428); reviewed LICENSE at 1.1.1 |
| swiftpm | `jinja` | `Apache-2.0` | `johnmai-dev/Jinja`, pinned by the demo apps' resolved graph |
| swiftpm | `mlx-swift` | `MIT` | |
| swiftpm | `mlx-swift-lm` | `MIT` | nested `swift/OpenMedKit` package and scan demo |
| swiftpm | `swift-argument-parser` | `Apache-2.0 WITH Swift-exception` | transitive pin |
| swiftpm | `swift-collections` | `Apache-2.0 WITH Swift-exception` | transitive pin |
| swiftpm | `swift-jinja` | `Apache-2.0` | transitive pin |
| swiftpm | `swift-numerics` | `Apache-2.0 WITH Swift-exception` | transitive pin |
| swiftpm | `swift-syntax` | `Apache-2.0 WITH Swift-exception` | transitive pin (demo resolved graph) |
| swiftpm | `swift-transformers` | `Apache-2.0` | |
| swiftpm | `zipfoundation` | `MIT` | |

### Gradle libraries

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| gradle | `ai.djl.android:tokenizer-native` | `Apache-2.0` | |
| gradle | `ai.djl.huggingface:tokenizers` | `Apache-2.0` | |
| gradle | `androidx.activity:activity-compose` | `Apache-2.0` | demo apps |
| gradle | `androidx.camera:camera-camera2` | `Apache-2.0` | scan demo |
| gradle | `androidx.camera:camera-core` | `Apache-2.0` | scan demo |
| gradle | `androidx.camera:camera-lifecycle` | `Apache-2.0` | scan demo |
| gradle | `androidx.camera:camera-view` | `Apache-2.0` | scan demo |
| gradle | `androidx.compose.foundation:foundation` | `Apache-2.0` | demo apps |
| gradle | `androidx.compose.material3:material3` | `Apache-2.0` | demo apps |
| gradle | `androidx.compose.ui:ui` | `Apache-2.0` | demo apps |
| gradle | `androidx.compose.ui:ui-tooling` | `Apache-2.0` | demo debug builds |
| gradle | `androidx.compose.ui:ui-tooling-preview` | `Apache-2.0` | demo apps |
| gradle | `androidx.compose:compose-bom` | `Apache-2.0` | demo apps |
| gradle | `androidx.core:core-ktx` | `Apache-2.0` | demo apps |
| gradle | `androidx.test:runner` | `Apache-2.0` | instrumented tests only |
| gradle | `com.android.tools.build:gradle` | `Apache-2.0` | Android Gradle Plugin; build tooling |
| gradle | `com.facebook.react:react-android` | `MIT` | React Native for Android; peer of the React Native bridge |
| gradle | `com.github.maziyarpanahi:openmed` | `Apache-2.0` | first-party OpenMedKit Android (JitPack) for the React Native bridge |
| gradle | `com.google.mlkit:text-recognition` | `LicenseRef-Proprietary-Google-ML-Kit-Terms` | restricted; scoped exception `mlkit-optional-ocr` |
| gradle | `com.ibm.icu:icu4j` | `Unicode-3.0` | test parity fixtures |
| gradle | `com.microsoft.onnxruntime:onnxruntime-android` | `MIT` | |
| gradle | `junit:junit` | `EPL-1.0` | unit tests only; never shipped |
| gradle | `org.jetbrains.kotlin:compose-compiler-gradle-plugin` | `Apache-2.0` | Compose compiler Gradle plugin; build tooling |
| gradle | `org.jetbrains.kotlin:kotlin-gradle-plugin` | `Apache-2.0` | build tooling |
| gradle | `org.jetbrains.kotlin:kotlin-stdlib` | `Apache-2.0` | |
| gradle | `org.jetbrains.kotlin:kotlin-test-junit` | `Apache-2.0` | unit tests |
| gradle | `org.jetbrains.kotlinx:kotlinx-coroutines-android` | `Apache-2.0` | demo apps |
| gradle | `org.jetbrains.kotlinx:kotlinx-coroutines-core` | `Apache-2.0` | |
| gradle | `org.jetbrains.kotlinx:kotlinx-coroutines-test` | `Apache-2.0` | unit tests |
| gradle | `org.jetbrains.kotlinx:kotlinx-serialization-json` | `Apache-2.0` | |
| gradle | `org.json:json` | `LicenseRef-Public-Domain` | unit tests; public domain since 20231013 |
| gradle | `org.robolectric:robolectric` | `MIT` | unit tests |

### Gradle plugins

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| gradle-plugin | `com.android.application` | `Apache-2.0` | Android Gradle Plugin |
| gradle-plugin | `com.android.library` | `Apache-2.0` | Android Gradle Plugin |
| gradle-plugin | `org.jetbrains.kotlin.android` | `Apache-2.0` | Kotlin Gradle Plugin |
| gradle-plugin | `org.jetbrains.kotlin.plugin.compose` | `Apache-2.0` | Kotlin Compose compiler plugin for the demo apps |
| gradle-plugin | `org.jetbrains.kotlinx.binary-compatibility-validator` | `Apache-2.0` | public API dump and check; build tooling |

### NDK runtime libraries

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| ndk | `libc++_shared.so` | `Apache-2.0 WITH LLVM-exception` | LLVM libc++ from NDK r28c (28.2.13676358), packaged in the OpenMedKit AAR for four ABIs; legacy NCSA/MIT notice in the NDK `NOTICE.toolchain` |

### npm

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| npm | `@babel/core` | `MIT` | React Native parity example; build tooling |
| npm | `@babel/preset-env` | `MIT` | React Native parity example; build tooling |
| npm | `@babel/runtime` | `MIT` | React Native parity example; build tooling |
| npm | `@electron/fuses` | `MIT` | Electron packaging harness; build tooling |
| npm | `@huggingface/tokenizers` | `Apache-2.0` | web kit tokenizer conformance tests |
| npm | `@microsoft/api-extractor` | `MIT` | web kit declaration API report (`npm run api:check`); build tooling |
| npm | `@huggingface/transformers` | `Apache-2.0` | optional peer of the web kit |
| npm | `@react-native-community/cli` | `MIT` | React Native parity example; build tooling |
| npm | `@react-native-community/cli-platform-android` | `MIT` | React Native parity example; build tooling |
| npm | `@react-native-community/cli-platform-ios` | `MIT` | React Native parity example; build tooling |
| npm | `@react-native/babel-preset` | `MIT` | React Native parity example; build tooling |
| npm | `@react-native/metro-config` | `MIT` | React Native parity example; build tooling |
| npm | `@react-native/typescript-config` | `MIT` | React Native parity example; build tooling |
| npm | `@tauri-apps/api` | `Apache-2.0 OR MIT` | peer of the Tauri kit |
| npm | `@types/node` | `MIT` | type definitions; build tooling |
| npm | `@types/react` | `MIT` | React Native parity example; type definitions |
| npm | `electron` | `MIT` | optional peer of the Electron kit |
| npm | `electron-builder` | `MIT` | Electron packaging harness; build tooling |
| npm | `esbuild` | `MIT` | Electron packaging harness; build tooling |
| npm | `onnxruntime-web` | `MIT` | optional peer of the web kit |
| npm | `openmed` | `Apache-2.0` | first-party web kit (`js/openmedkit-web`) |
| npm | `react` | `MIT` | optional peer of the React Native kit |
| npm | `react-native` | `MIT` | optional peer of the React Native kit |
| npm | `tsup` | `MIT` | build tooling |
| npm | `tsx` | `MIT` | test runner |
| npm | `typescript` | `Apache-2.0` | build tooling |

### Flutter pub

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| pub | `ffi` | `BSD-3-Clause` | |
| pub | `flutter` | `BSD-3-Clause` | Flutter SDK |
| pub | `flutter_test` | `BSD-3-Clause` | Flutter SDK test package |

### CocoaPods

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| cocoapods | `flutter` | `BSD-3-Clause` | Flutter engine pod |
| cocoapods | `fluttermacos` | `BSD-3-Clause` | Flutter macOS engine pod |
| cocoapods | `onnxruntime-c` | `MIT` | ONNX Runtime C pod |
| cocoapods | `react-core` | `MIT` | React Native core pod for the React Native bridge |

### Cargo (Tauri sidecar)

| Ecosystem | Package | License expression | Review note |
| --- | --- | --- | --- |
| cargo | `adler2` | `0BSD OR MIT OR Apache-2.0` | |
| cargo | `aho-corasick` | `Unlicense OR MIT` | |
| cargo | `alloc-no-stdlib` | `BSD-3-Clause` | |
| cargo | `alloc-stdlib` | `BSD-3-Clause` | |
| cargo | `android_system_properties` | `MIT OR Apache-2.0` | |
| cargo | `anyhow` | `MIT OR Apache-2.0` | |
| cargo | `atk` | `MIT` | |
| cargo | `atk-sys` | `MIT` | |
| cargo | `atomic-waker` | `Apache-2.0 OR MIT` | |
| cargo | `autocfg` | `Apache-2.0 OR MIT` | |
| cargo | `base64` | `MIT OR Apache-2.0` | |
| cargo | `bit-set` | `Apache-2.0 OR MIT` | |
| cargo | `bit-vec` | `Apache-2.0 OR MIT` | |
| cargo | `bitflags` | `MIT OR Apache-2.0` | |
| cargo | `block-buffer` | `MIT OR Apache-2.0` | |
| cargo | `block2` | `MIT` | |
| cargo | `brotli` | `BSD-3-Clause AND MIT` | |
| cargo | `brotli-decompressor` | `BSD-3-Clause OR MIT` | |
| cargo | `bumpalo` | `MIT OR Apache-2.0` | |
| cargo | `bytemuck` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `byteorder` | `Unlicense OR MIT` | |
| cargo | `bytes` | `MIT` | |
| cargo | `cairo-rs` | `MIT` | |
| cargo | `cairo-sys-rs` | `MIT` | |
| cargo | `camino` | `MIT OR Apache-2.0` | |
| cargo | `cargo-platform` | `MIT OR Apache-2.0` | |
| cargo | `cargo_metadata` | `MIT` | |
| cargo | `cargo_toml` | `Apache-2.0 OR MIT` | |
| cargo | `cc` | `MIT OR Apache-2.0` | |
| cargo | `cesu8` | `Apache-2.0 OR MIT` | |
| cargo | `cfb` | `MIT` | |
| cargo | `cfg-expr` | `MIT OR Apache-2.0` | |
| cargo | `cfg-if` | `MIT OR Apache-2.0` | |
| cargo | `chrono` | `MIT OR Apache-2.0` | |
| cargo | `combine` | `MIT` | |
| cargo | `cookie` | `MIT OR Apache-2.0` | |
| cargo | `core-foundation` | `MIT OR Apache-2.0` | |
| cargo | `core-foundation-sys` | `MIT OR Apache-2.0` | |
| cargo | `core-graphics` | `MIT OR Apache-2.0` | |
| cargo | `core-graphics-types` | `MIT OR Apache-2.0` | |
| cargo | `cpufeatures` | `MIT OR Apache-2.0` | |
| cargo | `crc32fast` | `MIT OR Apache-2.0` | |
| cargo | `crossbeam-channel` | `MIT OR Apache-2.0` | |
| cargo | `crossbeam-utils` | `MIT OR Apache-2.0` | |
| cargo | `crypto-common` | `MIT OR Apache-2.0` | |
| cargo | `cssparser` | `MPL-2.0` | |
| cargo | `cssparser-macros` | `MPL-2.0` | |
| cargo | `ctor` | `Apache-2.0 OR MIT` | |
| cargo | `ctor-proc-macro` | `Apache-2.0 OR MIT` | |
| cargo | `darling` | `MIT` | |
| cargo | `darling_core` | `MIT` | |
| cargo | `darling_macro` | `MIT` | |
| cargo | `dbus` | `Apache-2.0 OR MIT` | |
| cargo | `deranged` | `MIT OR Apache-2.0` | |
| cargo | `derive_more` | `MIT` | |
| cargo | `derive_more-impl` | `MIT` | |
| cargo | `digest` | `MIT OR Apache-2.0` | |
| cargo | `dirs` | `MIT OR Apache-2.0` | |
| cargo | `dirs-sys` | `MIT OR Apache-2.0` | |
| cargo | `dispatch2` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `displaydoc` | `MIT OR Apache-2.0` | |
| cargo | `dlopen2` | `MIT` | |
| cargo | `dlopen2_derive` | `MIT` | |
| cargo | `dom_query` | `MIT` | |
| cargo | `dpi` | `Apache-2.0 AND MIT` | |
| cargo | `dtoa` | `MIT OR Apache-2.0` | |
| cargo | `dtoa-short` | `MPL-2.0` | |
| cargo | `dtor` | `Apache-2.0 OR MIT` | |
| cargo | `dtor-proc-macro` | `Apache-2.0 OR MIT` | |
| cargo | `dunce` | `CC0-1.0 OR MIT-0 OR Apache-2.0` | |
| cargo | `dyn-clone` | `MIT OR Apache-2.0` | |
| cargo | `embed-resource` | `MIT` | |
| cargo | `embed_plist` | `MIT OR Apache-2.0` | |
| cargo | `equivalent` | `Apache-2.0 OR MIT` | |
| cargo | `erased-serde` | `MIT OR Apache-2.0` | |
| cargo | `errno` | `MIT OR Apache-2.0` | |
| cargo | `fastrand` | `Apache-2.0 OR MIT` | |
| cargo | `fdeflate` | `MIT OR Apache-2.0` | |
| cargo | `field-offset` | `MIT OR Apache-2.0` | |
| cargo | `find-msvc-tools` | `MIT OR Apache-2.0` | |
| cargo | `flate2` | `MIT OR Apache-2.0` | |
| cargo | `fnv` | `Apache-2.0 OR MIT` | |
| cargo | `foldhash` | `Zlib` | |
| cargo | `foreign-types` | `MIT OR Apache-2.0` | |
| cargo | `foreign-types-macros` | `MIT OR Apache-2.0` | |
| cargo | `foreign-types-shared` | `MIT OR Apache-2.0` | |
| cargo | `form_urlencoded` | `MIT OR Apache-2.0` | |
| cargo | `futures-channel` | `MIT OR Apache-2.0` | |
| cargo | `futures-core` | `MIT OR Apache-2.0` | |
| cargo | `futures-executor` | `MIT OR Apache-2.0` | |
| cargo | `futures-io` | `MIT OR Apache-2.0` | |
| cargo | `futures-macro` | `MIT OR Apache-2.0` | |
| cargo | `futures-sink` | `MIT OR Apache-2.0` | |
| cargo | `futures-task` | `MIT OR Apache-2.0` | |
| cargo | `futures-util` | `MIT OR Apache-2.0` | |
| cargo | `gdk` | `MIT` | |
| cargo | `gdk-pixbuf` | `MIT` | |
| cargo | `gdk-pixbuf-sys` | `MIT` | |
| cargo | `gdk-sys` | `MIT` | |
| cargo | `gdkwayland-sys` | `MIT` | |
| cargo | `gdkx11` | `MIT` | |
| cargo | `gdkx11-sys` | `MIT` | |
| cargo | `generic-array` | `MIT` | |
| cargo | `getrandom` | `MIT OR Apache-2.0` | |
| cargo | `gio` | `MIT` | |
| cargo | `gio-sys` | `MIT` | |
| cargo | `glib` | `MIT` | |
| cargo | `glib-macros` | `MIT` | |
| cargo | `glib-sys` | `MIT` | |
| cargo | `glob` | `MIT OR Apache-2.0` | |
| cargo | `gobject-sys` | `MIT` | |
| cargo | `gtk` | `MIT` | |
| cargo | `gtk-sys` | `MIT` | |
| cargo | `gtk3-macros` | `MIT` | |
| cargo | `hashbrown` | `MIT OR Apache-2.0` | |
| cargo | `heck` | `MIT OR Apache-2.0` | |
| cargo | `hex` | `MIT OR Apache-2.0` | |
| cargo | `html5ever` | `MIT OR Apache-2.0` | |
| cargo | `http` | `MIT OR Apache-2.0` | |
| cargo | `http-body` | `MIT` | |
| cargo | `http-body-util` | `MIT` | |
| cargo | `httparse` | `MIT OR Apache-2.0` | |
| cargo | `hyper` | `MIT` | |
| cargo | `hyper-util` | `MIT` | |
| cargo | `iana-time-zone` | `MIT OR Apache-2.0` | |
| cargo | `iana-time-zone-haiku` | `MIT OR Apache-2.0` | |
| cargo | `ico` | `MIT` | |
| cargo | `icu_collections` | `Unicode-3.0` | |
| cargo | `icu_locale_core` | `Unicode-3.0` | |
| cargo | `icu_normalizer` | `Unicode-3.0` | |
| cargo | `icu_normalizer_data` | `Unicode-3.0` | |
| cargo | `icu_properties` | `Unicode-3.0` | |
| cargo | `icu_properties_data` | `Unicode-3.0` | |
| cargo | `icu_provider` | `Unicode-3.0` | |
| cargo | `ident_case` | `MIT OR Apache-2.0` | |
| cargo | `idna` | `MIT OR Apache-2.0` | |
| cargo | `idna_adapter` | `Apache-2.0 OR MIT` | |
| cargo | `indexmap` | `Apache-2.0 OR MIT` | |
| cargo | `infer` | `MIT` | |
| cargo | `ipnet` | `MIT OR Apache-2.0` | |
| cargo | `itoa` | `MIT OR Apache-2.0` | |
| cargo | `javascriptcore-rs` | `MIT` | |
| cargo | `javascriptcore-rs-sys` | `MIT` | |
| cargo | `jni` | `MIT OR Apache-2.0` | |
| cargo | `jni-sys` | `MIT OR Apache-2.0` | |
| cargo | `jni-sys-macros` | `MIT OR Apache-2.0` | |
| cargo | `js-sys` | `MIT OR Apache-2.0` | |
| cargo | `json-patch` | `MIT OR Apache-2.0` | |
| cargo | `jsonptr` | `MIT OR Apache-2.0` | |
| cargo | `keyboard-types` | `MIT OR Apache-2.0` | |
| cargo | `libappindicator` | `Apache-2.0 OR MIT` | |
| cargo | `libappindicator-sys` | `Apache-2.0 OR MIT` | |
| cargo | `libc` | `MIT OR Apache-2.0` | |
| cargo | `libdbus-sys` | `Apache-2.0 OR MIT` | |
| cargo | `libloading` | `ISC` | |
| cargo | `libredox` | `MIT` | |
| cargo | `litemap` | `Unicode-3.0` | |
| cargo | `lock_api` | `MIT OR Apache-2.0` | |
| cargo | `log` | `MIT OR Apache-2.0` | |
| cargo | `markup5ever` | `MIT OR Apache-2.0` | |
| cargo | `memchr` | `Unlicense OR MIT` | |
| cargo | `memoffset` | `MIT` | |
| cargo | `mime` | `MIT OR Apache-2.0` | |
| cargo | `miniz_oxide` | `MIT OR Zlib OR Apache-2.0` | |
| cargo | `mio` | `MIT` | |
| cargo | `muda` | `Apache-2.0 OR MIT` | |
| cargo | `ndk` | `MIT OR Apache-2.0` | |
| cargo | `ndk-sys` | `MIT OR Apache-2.0` | |
| cargo | `new_debug_unreachable` | `MIT` | |
| cargo | `num-conv` | `MIT OR Apache-2.0` | |
| cargo | `num-traits` | `MIT OR Apache-2.0` | |
| cargo | `num_enum` | `BSD-3-Clause OR MIT OR Apache-2.0` | |
| cargo | `num_enum_derive` | `BSD-3-Clause OR MIT OR Apache-2.0` | |
| cargo | `objc2` | `MIT` | |
| cargo | `objc2-app-kit` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-cloud-kit` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-core-data` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-core-foundation` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-core-graphics` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-core-image` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-core-location` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-core-text` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-encode` | `MIT` | |
| cargo | `objc2-exception-helper` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-foundation` | `MIT` | |
| cargo | `objc2-io-surface` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-quartz-core` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-ui-kit` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-user-notifications` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `objc2-web-kit` | `Zlib OR Apache-2.0 OR MIT` | |
| cargo | `once_cell` | `MIT OR Apache-2.0` | |
| cargo | `option-ext` | `MPL-2.0` | |
| cargo | `pango` | `MIT` | |
| cargo | `pango-sys` | `MIT` | |
| cargo | `parking_lot` | `MIT OR Apache-2.0` | |
| cargo | `parking_lot_core` | `MIT OR Apache-2.0` | |
| cargo | `percent-encoding` | `MIT OR Apache-2.0` | |
| cargo | `phf` | `MIT` | |
| cargo | `phf_codegen` | `MIT` | |
| cargo | `phf_generator` | `MIT` | |
| cargo | `phf_macros` | `MIT` | |
| cargo | `phf_shared` | `MIT` | |
| cargo | `pin-project-lite` | `Apache-2.0 OR MIT` | |
| cargo | `pkg-config` | `MIT OR Apache-2.0` | |
| cargo | `plist` | `MIT` | |
| cargo | `png` | `MIT OR Apache-2.0` | |
| cargo | `potential_utf` | `Unicode-3.0` | |
| cargo | `powerfmt` | `MIT OR Apache-2.0` | |
| cargo | `precomputed-hash` | `MIT` | |
| cargo | `proc-macro-crate` | `MIT OR Apache-2.0` | |
| cargo | `proc-macro-error` | `MIT OR Apache-2.0` | |
| cargo | `proc-macro-error-attr` | `MIT OR Apache-2.0` | |
| cargo | `proc-macro2` | `MIT OR Apache-2.0` | |
| cargo | `quick-xml` | `MIT` | |
| cargo | `quote` | `MIT OR Apache-2.0` | |
| cargo | `r-efi` | `MIT OR Apache-2.0` | upstream `MIT OR Apache-2.0 OR LGPL-2.1-or-later`; OpenMed elects `MIT OR Apache-2.0` |
| cargo | `raw-window-handle` | `MIT OR Apache-2.0 OR Zlib` | |
| cargo | `redox_syscall` | `MIT` | |
| cargo | `redox_users` | `MIT` | |
| cargo | `ref-cast` | `MIT OR Apache-2.0` | |
| cargo | `ref-cast-impl` | `MIT OR Apache-2.0` | |
| cargo | `regex` | `MIT OR Apache-2.0` | |
| cargo | `regex-automata` | `MIT OR Apache-2.0` | |
| cargo | `regex-syntax` | `MIT OR Apache-2.0` | |
| cargo | `reqwest` | `MIT OR Apache-2.0` | |
| cargo | `rustc-hash` | `Apache-2.0 OR MIT` | |
| cargo | `rustc_version` | `MIT OR Apache-2.0` | |
| cargo | `rustversion` | `MIT OR Apache-2.0` | |
| cargo | `same-file` | `Unlicense OR MIT` | |
| cargo | `schemars` | `MIT` | |
| cargo | `schemars_derive` | `MIT` | |
| cargo | `scopeguard` | `MIT OR Apache-2.0` | |
| cargo | `selectors` | `MPL-2.0` | |
| cargo | `semver` | `MIT OR Apache-2.0` | |
| cargo | `serde` | `MIT OR Apache-2.0` | |
| cargo | `serde-untagged` | `MIT OR Apache-2.0` | |
| cargo | `serde_core` | `MIT OR Apache-2.0` | |
| cargo | `serde_derive` | `MIT OR Apache-2.0` | |
| cargo | `serde_derive_internals` | `MIT OR Apache-2.0` | |
| cargo | `serde_json` | `MIT OR Apache-2.0` | |
| cargo | `serde_repr` | `MIT OR Apache-2.0` | |
| cargo | `serde_spanned` | `MIT OR Apache-2.0` | |
| cargo | `serde_with` | `MIT OR Apache-2.0` | |
| cargo | `serde_with_macros` | `MIT OR Apache-2.0` | |
| cargo | `serialize-to-javascript` | `MIT OR Apache-2.0` | |
| cargo | `serialize-to-javascript-impl` | `MIT OR Apache-2.0` | |
| cargo | `servo_arc` | `MIT OR Apache-2.0` | |
| cargo | `sha2` | `MIT OR Apache-2.0` | |
| cargo | `shlex` | `MIT OR Apache-2.0` | |
| cargo | `signal-hook-registry` | `MIT OR Apache-2.0` | |
| cargo | `simd-adler32` | `MIT` | |
| cargo | `siphasher` | `MIT OR Apache-2.0` | |
| cargo | `slab` | `MIT` | |
| cargo | `smallvec` | `MIT OR Apache-2.0` | |
| cargo | `socket2` | `MIT OR Apache-2.0` | |
| cargo | `softbuffer` | `MIT OR Apache-2.0` | |
| cargo | `soup3` | `MIT` | |
| cargo | `soup3-sys` | `MIT` | |
| cargo | `stable_deref_trait` | `MIT OR Apache-2.0` | |
| cargo | `string_cache` | `MIT OR Apache-2.0` | |
| cargo | `string_cache_codegen` | `MIT OR Apache-2.0` | |
| cargo | `strsim` | `MIT` | |
| cargo | `swift-rs` | `MIT OR Apache-2.0` | |
| cargo | `syn` | `MIT OR Apache-2.0` | |
| cargo | `sync_wrapper` | `Apache-2.0` | |
| cargo | `synstructure` | `MIT` | |
| cargo | `system-deps` | `MIT OR Apache-2.0` | |
| cargo | `tao` | `Apache-2.0` | |
| cargo | `tao-macros` | `MIT OR Apache-2.0` | |
| cargo | `target-lexicon` | `Apache-2.0 WITH LLVM-exception` | |
| cargo | `tauri` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-build` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-codegen` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-macros` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-runtime` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-runtime-wry` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-utils` | `Apache-2.0 OR MIT` | |
| cargo | `tauri-winres` | `MIT` | |
| cargo | `tendril` | `MIT OR Apache-2.0` | |
| cargo | `thiserror` | `MIT OR Apache-2.0` | |
| cargo | `thiserror-impl` | `MIT OR Apache-2.0` | |
| cargo | `time` | `MIT OR Apache-2.0` | |
| cargo | `time-core` | `MIT OR Apache-2.0` | |
| cargo | `time-macros` | `MIT OR Apache-2.0` | |
| cargo | `tinystr` | `Unicode-3.0` | |
| cargo | `tokio` | `MIT` | |
| cargo | `tokio-util` | `MIT` | |
| cargo | `toml` | `MIT OR Apache-2.0` | |
| cargo | `toml_datetime` | `MIT OR Apache-2.0` | |
| cargo | `toml_edit` | `MIT OR Apache-2.0` | |
| cargo | `toml_parser` | `MIT OR Apache-2.0` | |
| cargo | `toml_writer` | `MIT OR Apache-2.0` | |
| cargo | `tower` | `MIT` | |
| cargo | `tower-http` | `MIT` | |
| cargo | `tower-layer` | `MIT` | |
| cargo | `tower-service` | `MIT` | |
| cargo | `tracing` | `MIT` | |
| cargo | `tracing-core` | `MIT` | |
| cargo | `tray-icon` | `MIT OR Apache-2.0` | |
| cargo | `try-lock` | `MIT` | |
| cargo | `typeid` | `MIT OR Apache-2.0` | |
| cargo | `typenum` | `MIT OR Apache-2.0` | |
| cargo | `unic-char-property` | `MIT OR Apache-2.0` | |
| cargo | `unic-char-range` | `MIT OR Apache-2.0` | |
| cargo | `unic-common` | `MIT OR Apache-2.0` | |
| cargo | `unic-ucd-ident` | `MIT OR Apache-2.0` | |
| cargo | `unic-ucd-version` | `MIT OR Apache-2.0` | |
| cargo | `unicode-ident` | `(MIT OR Apache-2.0) AND Unicode-3.0` | |
| cargo | `unicode-segmentation` | `MIT OR Apache-2.0` | |
| cargo | `url` | `MIT OR Apache-2.0` | |
| cargo | `urlpattern` | `MIT` | |
| cargo | `utf8_iter` | `Apache-2.0 OR MIT` | |
| cargo | `uuid` | `Apache-2.0 OR MIT` | |
| cargo | `version-compare` | `MIT` | |
| cargo | `version_check` | `MIT OR Apache-2.0` | |
| cargo | `vswhom` | `MIT` | |
| cargo | `vswhom-sys` | `MIT` | |
| cargo | `walkdir` | `Unlicense OR MIT` | |
| cargo | `want` | `MIT` | |
| cargo | `wasi` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | |
| cargo | `wasip2` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | |
| cargo | `wasm-bindgen` | `MIT OR Apache-2.0` | |
| cargo | `wasm-bindgen-futures` | `MIT OR Apache-2.0` | |
| cargo | `wasm-bindgen-macro` | `MIT OR Apache-2.0` | |
| cargo | `wasm-bindgen-macro-support` | `MIT OR Apache-2.0` | |
| cargo | `wasm-bindgen-shared` | `MIT OR Apache-2.0` | |
| cargo | `wasm-streams` | `MIT OR Apache-2.0` | |
| cargo | `web-sys` | `MIT OR Apache-2.0` | |
| cargo | `web_atoms` | `MIT OR Apache-2.0` | |
| cargo | `webkit2gtk` | `MIT` | |
| cargo | `webkit2gtk-sys` | `MIT` | |
| cargo | `webview2-com` | `MIT` | |
| cargo | `webview2-com-macros` | `MIT` | |
| cargo | `webview2-com-sys` | `MIT` | |
| cargo | `winapi` | `MIT OR Apache-2.0` | |
| cargo | `winapi-i686-pc-windows-gnu` | `MIT OR Apache-2.0` | |
| cargo | `winapi-util` | `Unlicense OR MIT` | |
| cargo | `winapi-x86_64-pc-windows-gnu` | `MIT OR Apache-2.0` | |
| cargo | `window-vibrancy` | `Apache-2.0 OR MIT` | |
| cargo | `windows` | `MIT OR Apache-2.0` | |
| cargo | `windows-collections` | `MIT OR Apache-2.0` | |
| cargo | `windows-core` | `MIT OR Apache-2.0` | |
| cargo | `windows-future` | `MIT OR Apache-2.0` | |
| cargo | `windows-implement` | `MIT OR Apache-2.0` | |
| cargo | `windows-interface` | `MIT OR Apache-2.0` | |
| cargo | `windows-link` | `MIT OR Apache-2.0` | |
| cargo | `windows-numerics` | `MIT OR Apache-2.0` | |
| cargo | `windows-result` | `MIT OR Apache-2.0` | |
| cargo | `windows-strings` | `MIT OR Apache-2.0` | |
| cargo | `windows-sys` | `MIT OR Apache-2.0` | |
| cargo | `windows-targets` | `MIT OR Apache-2.0` | |
| cargo | `windows-threading` | `MIT OR Apache-2.0` | |
| cargo | `windows-version` | `MIT OR Apache-2.0` | |
| cargo | `windows_aarch64_gnullvm` | `MIT OR Apache-2.0` | |
| cargo | `windows_aarch64_msvc` | `MIT OR Apache-2.0` | |
| cargo | `windows_i686_gnu` | `MIT OR Apache-2.0` | |
| cargo | `windows_i686_gnullvm` | `MIT OR Apache-2.0` | |
| cargo | `windows_i686_msvc` | `MIT OR Apache-2.0` | |
| cargo | `windows_x86_64_gnu` | `MIT OR Apache-2.0` | |
| cargo | `windows_x86_64_gnullvm` | `MIT OR Apache-2.0` | |
| cargo | `windows_x86_64_msvc` | `MIT OR Apache-2.0` | |
| cargo | `winnow` | `MIT` | |
| cargo | `winreg` | `MIT` | |
| cargo | `wit-bindgen` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | |
| cargo | `writeable` | `Unicode-3.0` | |
| cargo | `wry` | `Apache-2.0 OR MIT` | |
| cargo | `x11` | `MIT` | |
| cargo | `x11-dl` | `MIT` | |
| cargo | `yoke` | `Unicode-3.0` | |
| cargo | `yoke-derive` | `Unicode-3.0` | |
| cargo | `zerofrom` | `Unicode-3.0` | |
| cargo | `zerofrom-derive` | `Unicode-3.0` | |
| cargo | `zerotrie` | `Unicode-3.0` | |
| cargo | `zerovec` | `Unicode-3.0` | |
| cargo | `zerovec-derive` | `Unicode-3.0` | |
| cargo | `zmij` | `MIT` | |
