# OpenMedKit React Native

`@openmed/openmedkit-react-native` is a React Native bridge to OpenMedKit for
on-device PII analysis and de-identification. JavaScript calls go to the native
`OpenMedKitRN` module: OpenMedKit (Swift) on iOS and OpenMedKit Android on
Android. Spans returned to JavaScript carry offsets, labels, scores and HMAC
text hashes rather than matched source text.

The package is private and is not published to npm yet. Install it from a
local checkout or a workspace path.

## Requirements

| Surface | Minimum |
| --- | --- |
| React Native | 0.77 (peer dependency `react-native >= 0.77`) |
| iOS deployment target | 17.0 (the OpenMedKit Swift package floor) |
| Android `minSdkVersion` | 26 (the OpenMedKit Android floor) |
| Android build JDK | 17 |

React Native 0.77 is the compile baseline: the Android module pins
`com.facebook.react:react-android:0.77.3` for standalone builds, and the iOS
podspec needs `spm_dependency`, which React Native added in 0.75. When the
module is built on its own it uses AGP 8.7.3, Kotlin 2.0.21 and `compileSdk`
35. Inside an app, the React Native Gradle plugin forces the app's own
`react-android` version and the app's Android Gradle Plugin and Kotlin
versions apply.

## Autolinking

The package ships what the React Native CLI needs to autolink both platforms:

- `openmedkit-react-native.podspec` at the package root, found by CocoaPods
  autolinking.
- `ios/OpenMedKitRN.m`, the Objective-C export file that registers the Swift
  `OpenMedKitRN` class and its four promise methods with `RCT_EXTERN_MODULE`.
- `android/build.gradle` (namespace `com.openmed.openmedkit.reactnative`) and
  `OpenMedKitRnPackage`, which registers `OpenMedKitRnModule` as
  `OpenMedKitRN`.
- `react-native.config.js`, which names the Android package import and
  instance. The CLI loads it with `require()`, so the package is CommonJS
  (no `"type": "module"`).

### iOS

OpenMedKit is a Swift package, not a CocoaPods pod. The podspec calls React
Native's `spm_dependency` helper, which adds the `OpenMedKit` product from
`https://github.com/maziyarpanahi/openmed.git` (up to the next major version
from 3.2.0) to this pod's target during `pod install`. In the app's `Podfile`:

```ruby
platform :ios, '17.0'
```

React Native prints a warning when Swift package dependencies are linked
statically; build with `USE_FRAMEWORKS=dynamic pod install` if static linking
fails. The MLX backend does not run on the iOS Simulator; use a device, or the
Core ML backend in the Simulator.

### Android

OpenMedKit Android is published from immutable OpenMed release tags through
JitPack. The module declares the scoped repository itself. If the app
centralizes repositories with `repositoriesMode`, use `PREFER_SETTINGS` and add
the repository to the app's `settings.gradle`:

```groovy
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven {
            url "https://jitpack.io"
            content { includeGroup "com.github.maziyarpanahi" }
        }
    }
}
```

These `rootProject.ext` properties override the module defaults:

| Property | Default |
| --- | --- |
| `compileSdkVersion` | `35` |
| `reactNativeVersion` | `0.77.3` |
| `openmedkitAndroidDependency` | `com.github.maziyarpanahi:openmed:v3.2.0` |
| `kotlinxCoroutinesVersion` | `1.8.1` |

## New Architecture

`OpenMedKitRN` is a legacy bridge module. The package does not provide a
codegen spec (`codegenConfig`) or a TurboModule implementation. Under the New
Architecture, React Native exposes the module through its automatic interop
layer, and `src/index.ts` resolves it through `TurboModuleRegistry` first, then
`NativeModules`.

## Errors and privacy

When the native module is not linked, every call rejects with
`OpenMedKitRN is not installed in the React Native runtime`.

Native rejections use one of four codes: `openmedkit_load_failed`,
`openmedkit_analyze_failed`, `openmedkit_extract_failed` and
`openmedkit_deidentify_failed`. The message is either a fixed bridge message
(for example `OpenMedKit model is not loaded`) or
`OpenMedKit native call failed`. Input text, file paths, option values and the
underlying native error are never forwarded, and the bridge code writes
nothing to native logs.

## Tests

From this directory:

```bash
npm ci
npm test
```

Packaging and native registration checks run from the repository root:

```bash
python -m pytest tests/mobile/test_rn_packaging.py -q
```
