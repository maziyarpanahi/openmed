# OpenMedKit Support Matrix (v4.0)

This page is the frozen v4.0 support matrix for the portable OpenMedKit
surfaces. It lists every surface named by the portable-kit workstream and, for
each one, which combinations of operating system, coarse device class, ABI,
runtime version, execution provider, artifact format, model family, precision
and device tier are **supported**, **experimental** or **unsupported**.

The roadmap issues still call this release "v3.5". That release was folded
into v4.0, so this matrix is the v4.0 matrix.

The tables below are generated from the machine-readable matrix at
`openmed/eval/data/kit_support_matrix.json`, which sits beside the canonical
tier budgets in `openmed.eval.tiers`. A coherence check,
`scripts/release/check_kit_support_matrix.py`, runs in the unit suite as
`tests/unit/eval/test_kit_support_matrix.py`. It fails when a manifest pin, a
platform floor, the export family list or the tier table changes without a
matrix update, and when this page drifts from the JSON.

## Support levels

- **Supported**: in scope for v4.0, and at least one evidence record runs that
  surface on that platform. A cell can only be supported if its `evidence`
  field names an evidence record.
- **Experimental**: the code path exists, but there is no evidence yet, or the
  combination is deliberately not claimed for v4.0. `"evidence": "pending"`
  always makes a cell experimental.
- **Unsupported**: OpenMed declines the combination. Each one has a fixed
  reason code, for example `runtime-unavailable`, `platform-deprecated`,
  `family-not-exported` or `policy-forbidden`. A runtime selector should return
  an explicit unsupported result for these instead of falling back silently.

At freeze, every evidence record is a repository CI lane. No cell has
real-device qualification or signed release evidence yet:

- The [real-device qualification harness](openmedkit-device-qualification.md)
  (#3716) adds `device-run` records. Only physical-device runs can promote a
  cell; simulator and emulator runs are reported separately.
- The portable-kit release gate (#3932) adds `release-packet` records and
  decides which supported cells a release may claim.

Each evidence record lists what it actually covers (`build`, `contract-parity`,
`runtime-synthetic`, `runtime-model`, `memory-budget`, ...). Read that list
before quoting a cell as qualified.

Because of the evidence rule, only the Python reference contract and Swift
OpenMedKit on Apple Silicon macOS have supported cells at freeze. Android is a
P0 surface, but none of its cells is supported yet. The Android CI lane runs on
the JVM and never executes ONNX Runtime on Android; the emulator lane (#3938)
adds the first such evidence.

## Runtime-version skew

The same runtime is pinned differently across kits. The matrix lists the
difference explicitly and does not change any pin:

- **ONNX Runtime:** Android kit `1.25.1`, Flutter Android and Apple `1.20.0`,
  Flutter Linux and Windows app-supplied with no pin, `onnxruntime-web >=1.20.0`
  and the Python extras `onnxruntime>=1.16`.
- **mlx-swift:** the root `Package.swift` pins `0.31.3`; the nested
  `swift/OpenMedKit/Package.swift` pins `0.31.6` (#3929).

A measured cell must record the runtime version it actually ran.

## Privacy boundary

The matrix holds coarse device classes such as `iphone-ipad`,
`android-device` or `desktop-browser`. It never holds device models,
serial-like values, user data or local paths, and validation rejects values
with those shapes. Evidence references are repository paths. Measurements and
signed packets stay with the harness and the release gate.

## Schema

The JSON document (`schema_version` 1) has these top-level keys:

| Key | Contents |
|---|---|
| `release`, `state`, `frozen_on` | Release name (`v4.0`), `frozen` or `draft`, and the freeze date. |
| `sources` | The family source (`docs/export-matrix.md`) and tier source (`openmed.eval.tiers.TIERS`). |
| `surfaces` | One row per epic surface: `id`, `name`, `epic_scope` (`reference`, `P0`, `P1`, `P2`), `support_claim` (`supported` or `experimental-only`) and optional `builds_on`. |
| `families` | Family ids with the matching `docs/export-matrix.md` row name. |
| `platform_floors` | Deployment floors and SDK bounds with the manifests that declare them. |
| `runtime_pins` | Runtime versions per kit with `pinning` (`exact`, `minimum`, `major`, `unpinned`, `system`) and source manifests. |
| `runtime_skew` | Runtimes whose kit pins differ, with every member pin. |
| `evidence` | Evidence records: `kind` (`ci-lane`, `device-run`, `release-packet`), `covers`, repository `refs` and a note. |
| `cells` | Supported and experimental cells. |
| `unsupported` | Unsupported combinations with a `reason` code. |

A cell has these fields: `surface`, `platform` (`name`, optional `min` and
`max`), `device_class`, `abi`, `runtime` (a pin id), `provider`, `artifact`,
`families`, `precisions`, `tier`, `status` and `evidence`, plus optional
`condition` (such as `script:latin` or `page-size:16kb`), `related` issue
numbers and a short `note`. Unsupported entries use the same fields with
`reason` in place of `evidence`, and may use `*` for "any".

Android platform versions are API levels. The renderer takes each tier's RAM
ceiling from `openmed.eval.tiers.TIERS`; the matrix never restates those
budgets. The ceiling is the target a device run must meet. It is not a
measurement, and no cell claims to meet it until a `device-run` record says so.

To change the matrix, edit the JSON, then regenerate this page and run the
check:

```bash
python scripts/release/check_kit_support_matrix.py --write-docs
python -m pytest tests/unit/eval/test_kit_support_matrix.py -q
```

Downstream work reads the JSON through `openmed.eval.kit_support_matrix`:

- The [real-device qualification harness](openmedkit-device-qualification.md)
  (#3716) enumerates every supported and experimental inference cell and
  proposes `device-run` evidence for the cells it qualifies.
- The release gate (#3932) binds evidence to the supported cells.
- Runtime selection (#3641) turns unsupported entries into explicit
  unsupported results.

<!-- kit-support-matrix:begin -->
<!-- Generated from openmed/eval/data/kit_support_matrix.json by
     scripts/release/check_kit_support_matrix.py --write-docs.
     Edit the JSON, then regenerate; do not edit this block by hand. -->

Matrix `openmedkit-support-matrix` schema 1, release v4.0, state **frozen** on 2026-11-07.

## Surfaces

| Surface | Epic scope | Claim | Supported | Experimental | Unsupported |
|---|---|---|---:|---:|---:|
| Python reference contract (`python`) | reference | supported | 4 | 3 | 3 |
| Swift OpenMedKit (`swift`) | P0 | supported | 1 | 9 | 10 |
| Kotlin and Android OpenMedKit (`android`) | P0 | experimental-only | 0 | 4 | 8 |
| React Native bridge (`react-native`) | P1 | experimental-only | 0 | 3 | 2 |
| Browser inference (`web`) | P1 | experimental-only | 0 | 8 | 5 |
| Shared document intake (`document-intake`) | P1 | experimental-only | 0 | 2 | 3 |
| Platform key-store integration (`key-store`) | P1 | experimental-only | 0 | 4 | 3 |
| Flutter plugin (`flutter`) | P2 | experimental-only | 0 | 5 | 3 |
| Electron integration (`electron`) | P2 | experimental-only | 0 | 3 | 2 |
| Tauri sidecar (`tauri`) | P2 | experimental-only | 0 | 3 | 3 |

## Platform floors and SDK bounds

| Floor | Surface | Bound | Platform | Version | Sources |
|---|---|---|---|---|---|
| `python-requires` | `python` | min | python | 3.11 | `pyproject.toml` |
| `swift-tools` | `swift` | min | swift-tools | 5.10 | `Package.swift`<br>`swift/OpenMedKit/Package.swift` |
| `swift-ios` | `swift` | min | ios | 17 | `Package.swift`<br>`swift/OpenMedKit/Package.swift` |
| `swift-macos` | `swift` | min | macos | 14 | `Package.swift`<br>`swift/OpenMedKit/Package.swift` |
| `swift-watchos` | `swift` | min | watchos | 10 | `Package.swift`<br>`swift/OpenMedKit/Package.swift` |
| `swift-visionos` | `swift` | min | visionos | 1 | `Package.swift`<br>`swift/OpenMedKit/Package.swift` |
| `android-min-sdk` | `android` | min | android | 26 | `android/openmedkit/build.gradle.kts` |
| `android-compile-sdk` | `android` | compile | android | 33 | `android/openmedkit/build.gradle.kts` |
| `android-target-sdk` | `android` | target | android | 33 | `android/openmedkit/build.gradle.kts` |
| `react-native-host` | `react-native` | min | react-native | 0.74 | `js/openmedkit-react-native/package.json` |
| `web-node` | `web` | min | node | 20 | `js/openmedkit-web/package.json` |
| `flutter-android-min-sdk` | `flutter` | min | android | 26 | `js/openmedkit-flutter/android/build.gradle` |
| `flutter-android-compile-sdk` | `flutter` | compile | android | 35 | `js/openmedkit-flutter/android/build.gradle` |
| `flutter-ios` | `flutter` | min | ios | 13.0 | `js/openmedkit-flutter/ios/openmedkit_flutter.podspec` |
| `flutter-macos` | `flutter` | min | macos | 11.0 | `js/openmedkit-flutter/macos/openmedkit_flutter.podspec` |
| `electron-host` | `electron` | min | electron | 30 | `js/openmedkit-electron/package.json` |
| `tauri-host` | `tauri` | min | tauri | 2.0.0 | `js/openmedkit-tauri/package.json`<br>`js/openmedkit-tauri/src-tauri/Cargo.toml` |

## Runtime pins

| Pin | Runtime | Kind | Surface | Version | Pinning | Sources |
|---|---|---|---|---|---|---|
| `torch-python` | torch | inference | `python` | >=2.13.0 | minimum | `pyproject.toml` |
| `onnxruntime-python` | onnxruntime | inference | `python` | >=1.16 | minimum | `pyproject.toml` |
| `mlx-python` | mlx | inference | `python` | >=0.22 | minimum | `pyproject.toml` |
| `mlx-swift` | mlx-swift | inference | `swift` | 0.31.3 | exact | `Package.swift` |
| `mlx-swift-nested` | mlx-swift | inference | `swift` | 0.31.6 | exact | `swift/OpenMedKit/Package.swift` |
| `coreml-system` | coreml | inference | `swift` | system | system | operating system |
| `onnxruntime-android` | onnxruntime | inference | `android` | 1.25.1 | exact | `android/gradle/libs.versions.toml` |
| `mlkit-text-recognition` | mlkit-text-recognition | ocr | `android` | 16.0.1 | exact | `android/gradle/libs.versions.toml` |
| `onnxruntime-web` | onnxruntime | inference | `web` | >=1.20.0 | minimum | `js/openmedkit-web/package.json` |
| `transformers-js` | transformers-js | inference | `web` | >=3.0.0 | minimum | `js/openmedkit-web/package.json` |
| `onnxruntime-flutter-android` | onnxruntime | inference | `flutter` | 1.20.0 | exact | `js/openmedkit-flutter/android/build.gradle` |
| `onnxruntime-flutter-ios` | onnxruntime | inference | `flutter` | 1.20.0 | exact | `js/openmedkit-flutter/ios/openmedkit_flutter.podspec` |
| `onnxruntime-flutter-macos` | onnxruntime | inference | `flutter` | 1.20.0 | exact | `js/openmedkit-flutter/macos/openmedkit_flutter.podspec` |
| `onnxruntime-flutter-desktop` | onnxruntime | inference | `flutter` | unpinned | unpinned | `js/openmedkit-flutter/native/CMakeLists.txt` |
| `apple-keychain` | keychain-services | keystore | `key-store` | system | system | operating system |
| `android-keystore` | android-keystore | keystore | `key-store` | system | system | operating system |

## Runtime-version skew

Pins of the same runtime that differ between kits are listed here explicitly. They are not aligned by this matrix.

| Skew | Runtime | Pins | Related | Note |
|---|---|---|---|---|
| `onnxruntime-cross-kit` | onnxruntime | `onnxruntime-android` 1.25.1; `onnxruntime-flutter-android` 1.20.0; `onnxruntime-flutter-ios` 1.20.0; `onnxruntime-flutter-macos` 1.20.0; `onnxruntime-flutter-desktop` unpinned; `onnxruntime-web` >=1.20.0; `onnxruntime-python` >=1.16 | #3709, #3716 | Kits resolve different ONNX Runtime builds; measured cells must record the version they ran. This matrix flags the skew and changes no pin. |
| `mlx-swift-manifests` | mlx-swift | `mlx-swift` 0.31.3; `mlx-swift-nested` 0.31.6 | #3929 | The root package, which is the documented install path, and the nested package pin different mlx-swift releases. |

## Python reference contract

### Supported

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `python-linux-torch-cpu` | linux | server-or-desktop | x86_64 | torch >=2.13.0 | cpu | pytorch | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | `ci-python-contract` | — | Reference semantics for offsets, labels, merging and policy. |
| `python-macos-torch-cpu` | macos | apple-silicon-mac | arm64 | torch >=2.13.0 | cpu | pytorch | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | `ci-python-contract` | — |  |
| `python-windows-torch-cpu` | windows | server-or-desktop | x86_64 | torch >=2.13.0 | cpu | pytorch | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | `ci-python-contract` | — |  |
| `python-linux-onnx-int8-cpu` | linux | server-or-desktop | x86_64 | onnxruntime >=1.16 | cpu | onnx-android | deberta-v2 | int8 | Tiny (≤ 350 MB) | — | `ci-python-low-resource` | — |  |

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `python-linux-arm64-onnx-int8-cpu` | linux | server-or-desktop | arm64 | onnxruntime >=1.16 | cpu | onnx-android | deberta-v2 | int8 | Tiny (≤ 350 MB) | — | `ci-python-edge-sbc` | #3716 | The aarch64 lane measures footprint with a synthetic graph only. |
| `python-linux-onnx-fp32-cpu` | linux | server-or-desktop | x86_64 | onnxruntime >=1.16 | cpu | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | — |  |
| `python-macos-mlx` | macos | apple-silicon-mac | arm64 | mlx >=0.22 | metal | mlx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2, modernbert, longformer, openai-privacy-filter, gliner-span, gliclass, gliner-token-relex | fp32/int8 | Base (≤ 900 MB) | — | pending | — | The MLX unit lane mocks MLX; no Apple Silicon lane runs it. |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `python-mlx-off-apple-silicon` | linux | server-or-desktop | any | mlx >=0.22 | any | any | any | — | runtime-unavailable | — | OpenMed's MLX backends require Apple Silicon macOS. |
| `python-gguf-token-classification` | any | server-or-desktop | any | any | cpu | any | any | — | runtime-unavailable | — | GGUF export is embedding-only; token-classification heads are rejected. |
| `python-onnx-unexported-families` | any | any | any | onnxruntime >=1.16 | any | openai-privacy-filter, gliner-span, gliclass, gliner-token-relex | any | — | family-not-exported | — |  |

## Swift OpenMedKit

### Supported

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `swift-macos-mlx` | macos 14+ | apple-silicon-mac | arm64 | mlx-swift 0.31.3 | metal | mlx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2, openai-privacy-filter | fp32/int8 | Base (≤ 900 MB) | — | `ci-swift-macos` | #3716 | Real-device recall, latency and memory evidence is pending. |

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `swift-macos-mlx-int4` | macos 14+ | apple-silicon-mac | arm64 | mlx-swift 0.31.3 | metal | mlx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2, openai-privacy-filter | int4 | Base (≤ 900 MB) | — | pending | — | INT4 artifacts are certified only when per-label recall holds. |
| `swift-macos-coreml` | macos 14+ | apple-silicon-mac | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16/int8 | Base (≤ 900 MB) | — | `ci-swift-macos` | — | CI compares recorded spans only; no Core ML model is evaluated. |
| `swift-ios-mlx` | ios 17+ | iphone-ipad | arm64 | mlx-swift 0.31.3 | metal | mlx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2, openai-privacy-filter | fp32/int8 | Tiny (≤ 350 MB) | — | `ci-swift-ios-build` | #3716 |  |
| `swift-ios-coreml` | ios 17+ | iphone-ipad | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16/int8 | Tiny (≤ 350 MB) | — | pending | #3716 |  |
| `swift-ios-simulator-coreml` | ios 17+ | ios-simulator | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16/int8 | Tiny (≤ 350 MB) | — | `ci-swift-ios-build` | #3949 |  |
| `swift-watchos-coreml-nano` | watchos 10+ | apple-watch | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Nano (≤ 150 MB) | — | pending | #3716 |  |
| `swift-watchos-simulator-coreml-nano` | watchos 10+ | watchos-simulator | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Nano (≤ 150 MB) | — | `ci-swift-watchos-simulator` | — |  |
| `swift-visionos-coreml-nano` | visionos 1+ | apple-vision | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Nano (≤ 150 MB) | — | pending | #3716 |  |
| `swift-visionos-simulator-coreml-nano` | visionos 1+ | visionos-simulator | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Nano (≤ 150 MB) | — | `ci-swift-visionos-simulator` | — |  |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `swift-below-ios-17` | ios ≤ 16 | iphone-ipad | any | any | any | any | any | — | below-platform-floor | — |  |
| `swift-below-macos-14` | macos ≤ 13 | any | any | any | any | any | any | — | below-platform-floor | — |  |
| `swift-ios-simulator-mlx` | ios 17+ | ios-simulator | any | mlx-swift 0.31.3 | metal | any | any | — | runtime-unavailable | #3949 | mlx-swift cannot run on the iOS Simulator; use Core ML there. |
| `swift-intel-mac-mlx` | macos 14+ | intel-mac | x86_64 | mlx-swift 0.31.3 | metal | any | any | — | hardware-unavailable | — | MLX requires Apple Silicon. |
| `swift-watchos-mlx` | watchos 10+ | apple-watch | any | mlx-swift 0.31.3 | metal | any | any | — | runtime-not-linked | — | Package.swift links MLX for iOS and macOS only. |
| `swift-visionos-mlx` | visionos 1+ | apple-vision | any | mlx-swift 0.31.3 | metal | any | any | — | runtime-not-linked | — | Package.swift links MLX for iOS and macOS only. |
| `swift-watchos-coreml-non-nano` | watchos 10+ | apple-watch | any | coreml system | coreml | any | any | artifact:non-nano-int8 | budget-exceeded | — | PlatformModel fails closed on anything but an INT8 Nano-tier model. |
| `swift-visionos-coreml-non-nano` | visionos 1+ | apple-vision | any | coreml system | coreml | any | any | artifact:non-nano-int8 | budget-exceeded | — | PlatformModel fails closed on anything but an INT8 Nano-tier model. |
| `swift-coreml-unexported-families` | any | any | any | coreml system | coreml | modernbert, longformer, openai-privacy-filter, gliner-span, gliclass, gliner-token-relex | any | — | family-not-exported | — | The Core ML converter allowlists BERT-family and DeBERTa-v2 only. |
| `swift-mac-catalyst` | mac-catalyst | mac-catalyst | any | any | any | any | any | — | not-qualified | — | Package.swift declares no Mac Catalyst platform and no lane builds it. |

## Kotlin and Android OpenMedKit

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `android-arm64-onnx-cpu` | android 26+ | android-device | arm64-v8a | onnxruntime 1.25.1 | cpu | onnx-android | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32/int8 | Tiny (≤ 350 MB) | — | `ci-android-jvm` | #3938, #3716 | No lane executes ONNX Runtime on Android yet. |
| `android-emulator-onnx-cpu` | android 26+ | android-emulator | x86_64 | onnxruntime 1.25.1 | cpu | onnx-android | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32/int8 | Tiny (≤ 350 MB) | — | pending | #3938 |  |
| `android-arm64-onnx-nnapi` | android 27–34 | android-device | arm64-v8a | onnxruntime 1.25.1 | nnapi | onnx-android | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32/int8 | Tiny (≤ 350 MB) | — | pending | #3716 | CPU stays the deterministic fallback; FP16 relaxation is off by default. |
| `android-arm64-onnx-qnn` | android 26+ | android-device | arm64-v8a | onnxruntime 1.25.1 | qnn | onnx-android | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Tiny (≤ 350 MB) | runtime:app-supplied-qnn | pending | — | Needs an app-supplied QNN-enabled ONNX Runtime and Qualcomm libraries. |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `android-below-min-sdk` | android ≤ 25 | any | any | any | any | any | any | — | below-platform-floor | — |  |
| `android-nnapi-android-15` | android 35+ | android-device | any | onnxruntime 1.25.1 | nnapi | any | any | — | platform-deprecated | — | NNAPI is deprecated from Android 15 (API 35); use the CPU provider. |
| `android-nnapi-api-26` | android 26 | android-device | any | onnxruntime 1.25.1 | nnapi | any | any | — | runtime-unavailable | — | OpenMedKit excludes NNAPI below API 27. |
| `android-qnn-standard-aar` | android 26+ | android-device | arm64-v8a | onnxruntime 1.25.1 | qnn | any | any | runtime:standard-aar | runtime-not-bundled | — | The standard ONNX Runtime AAR has no QNN provider; no Qualcomm SDK ships. |
| `android-16kb-page-size` | android 26+ | android-device | arm64-v8a | onnxruntime 1.25.1 | any | any | any | page-size:16kb | not-qualified | #3927 | Native libraries are not yet verified to load with 16 KB pages. |
| `android-armeabi-v7a` | android 26+ | android-device | armeabi-v7a | onnxruntime 1.25.1 | any | any | any | — | not-qualified | — |  |
| `android-x86` | android 26+ | android-emulator | x86 | onnxruntime 1.25.1 | any | any | any | — | not-qualified | — |  |
| `android-onnx-unexported-families` | android 26+ | any | any | onnxruntime 1.25.1 | any | openai-privacy-filter, gliner-span, gliclass, gliner-token-relex | any | — | family-not-exported | — |  |

## React Native bridge

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `react-native-ios-mlx` | ios 17+ | iphone-ipad | arm64 | mlx-swift 0.31.3 | metal | mlx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2, openai-privacy-filter | fp32/int8 | Tiny (≤ 350 MB) | — | pending | #3936, #3716 | The bridge forwards to Swift OpenMedKit; autolinking is not packaged. |
| `react-native-ios-coreml` | ios 17+ | iphone-ipad | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16/int8 | Tiny (≤ 350 MB) | — | pending | #3936 |  |
| `react-native-android-onnx-cpu` | android 26+ | android-device | arm64-v8a | onnxruntime 1.25.1 | cpu | onnx-android | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32/int8 | Tiny (≤ 350 MB) | — | pending | #3936, #3938 | The bridge forwards to Android OpenMedKit. |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `react-native-ios-simulator-mlx` | ios 17+ | ios-simulator | any | mlx-swift 0.31.3 | metal | any | any | — | runtime-unavailable | #3949 |  |
| `react-native-web` | any | desktop-browser | any | any | any | any | any | — | not-implemented | — | The bridge has no react-native-web implementation. |

## Browser inference

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `web-chromium-webgpu` | chromium | desktop-browser | any | onnxruntime >=1.20.0 | webgpu | onnx-webgpu | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16 | Base (≤ 900 MB) | — | `ci-web-pages` | #3939, #3940 | fp16 WebGPU graphs are not yet gated on logit and span parity. |
| `web-webkit-webgpu` | webkit | desktop-browser | any | onnxruntime >=1.20.0 | webgpu | onnx-webgpu | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16 | Base (≤ 900 MB) | — | pending | #3940 | Not exercised; the loader falls back to WASM when no adapter is returned. |
| `web-gecko-webgpu` | gecko | desktop-browser | any | onnxruntime >=1.20.0 | webgpu | onnx-webgpu | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp16 | Base (≤ 900 MB) | — | pending | #3940 | Not exercised; the loader falls back to WASM when no adapter is returned. |
| `web-chromium-wasm-threads` | chromium | desktop-browser | wasm32 | onnxruntime >=1.20.0 | wasm-simd-threads | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | cross-origin-isolation:required | `ci-web-pages` | #3940 |  |
| `web-chromium-wasm-basic` | chromium | desktop-browser | wasm32 | onnxruntime >=1.20.0 | wasm-basic | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | `ci-web-pages` | #3940 |  |
| `web-webkit-wasm-basic` | webkit | desktop-browser | wasm32 | onnxruntime >=1.20.0 | wasm-basic | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | #3940 |  |
| `web-gecko-wasm-basic` | gecko | desktop-browser | wasm32 | onnxruntime >=1.20.0 | wasm-basic | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | #3940 |  |
| `web-chromium-transformersjs` | chromium | desktop-browser | wasm32 | transformers-js >=3.0.0 | wasm-basic | transformersjs | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Base (≤ 900 MB) | — | pending | — |  |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `web-remote-model-assets` | any | desktop-browser | any | onnxruntime >=1.20.0 | any | any | any | asset-origin:remote | policy-forbidden | — | HTTP(S), protocol-relative, data: and blob: model paths are rejected. |
| `web-webgpu-insecure-context` | any | desktop-browser | any | onnxruntime >=1.20.0 | webgpu | any | any | secure-context:absent | runtime-unavailable | — | WebGPU requires HTTPS or localhost. |
| `web-threads-without-isolation` | any | desktop-browser | wasm32 | onnxruntime >=1.20.0 | wasm-simd-threads | any | any | cross-origin-isolation:absent | runtime-unavailable | — | Without COOP and COEP headers the loader uses single-threaded WASM. |
| `web-mobile-browser` | any | mobile-browser | any | onnxruntime >=1.20.0 | any | any | any | — | not-qualified | — |  |
| `web-unexported-families` | any | desktop-browser | any | onnxruntime >=1.20.0 | any | openai-privacy-filter, gliner-span, gliclass, gliner-token-relex | any | — | family-not-exported | — |  |

## Shared document intake

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `document-intake-android-mlkit-latin` | android 26+ | android-device | arm64-v8a | mlkit-text-recognition 16.0.1 | on-device | — | — | — | — | script:latin | pending | #3713, #3928 | ML Kit OCR ships inside the core Android artifact today. |
| `document-intake-ios-share-extension` | ios 17+ | iphone-ipad | arm64 | coreml system | coreml | coreml | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Nano (≤ 150 MB) | — | pending | #3713 | Share and Action extensions redact selected text with a bundled model. |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `document-intake-android-mlkit-non-latin` | android 26+ | android-device | any | mlkit-text-recognition 16.0.1 | on-device | — | — | script:non-latin | script-unsupported | — | MlKitOcrAdapter uses the bundled Latin recognizer only. |
| `document-intake-apple-ocr` | ios 17+ | iphone-ipad | any | any | any | — | — | input:scanned-image | not-implemented | #3713 | Swift OpenMedKit has no OCR adapter; hosts must extract text first. |
| `document-intake-extension-download` | ios 17+ | iphone-ipad | any | coreml system | coreml | any | any | asset-origin:remote | policy-forbidden | — | Extensions must not download model or tokenizer files. |

## Platform key-store integration

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `key-store-ios-keychain` | ios 17+ | iphone-ipad | arm64 | keychain-services system | secure-enclave | — | — | — | — | — | pending | #3714 |  |
| `key-store-macos-keychain` | macos 14+ | apple-silicon-mac | arm64 | keychain-services system | keychain | — | — | — | — | — | pending | #3714 |  |
| `key-store-android-tee` | android 26+ | android-device | arm64-v8a | android-keystore system | tee | — | — | — | — | — | pending | #3715 |  |
| `key-store-android-strongbox` | android 28+ | android-device | arm64-v8a | android-keystore system | strongbox | — | — | — | — | — | pending | #3715 | StrongBox needs API 28 or newer and device hardware support. |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `key-store-ios-simulator-secure-enclave` | ios 17+ | ios-simulator | any | keychain-services system | secure-enclave | — | — | — | hardware-unavailable | — |  |
| `key-store-android-emulator-strongbox` | android 28+ | android-emulator | any | android-keystore system | strongbox | — | — | — | hardware-unavailable | — |  |
| `key-store-browser` | any | desktop-browser | any | any | any | — | — | — | not-implemented | — | No platform key-store adapter exists for browser inference. |

## Flutter plugin

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flutter-android-onnx-cpu` | android 26+ | android-device | arm64-v8a | onnxruntime 1.20.0 | cpu | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Tiny (≤ 350 MB) | — | `ci-flutter` | #3709 |  |
| `flutter-ios-onnx-cpu` | ios 13.0+ | iphone-ipad | arm64 | onnxruntime 1.20.0 | cpu | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Tiny (≤ 350 MB) | — | pending | #3709 |  |
| `flutter-macos-onnx-cpu` | macos 11.0+ | apple-silicon-mac | arm64 | onnxruntime 1.20.0 | cpu | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | #3709 |  |
| `flutter-linux-onnx-cpu` | linux | desktop-app | x86_64 | onnxruntime unpinned | cpu | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | runtime:app-supplied-onnxruntime | `ci-flutter` | #3709 | Linux and Windows link an app-supplied ONNX Runtime with no pin. |
| `flutter-windows-onnx-cpu` | windows | desktop-app | x86_64 | onnxruntime unpinned | cpu | onnx | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | runtime:app-supplied-onnxruntime | pending | #3709 |  |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `flutter-web` | any | desktop-browser | any | any | any | any | any | — | runtime-not-linked | — | pubspec.yaml declares no web platform for the plugin. |
| `flutter-apple-coreml` | ios 13.0+ | iphone-ipad | any | coreml system | coreml | any | any | — | runtime-not-linked | — | The plugin links ONNX Runtime only; use Swift OpenMedKit for Core ML. |
| `flutter-onnx-unexported-families` | any | any | any | any | any | openai-privacy-filter, gliner-span, gliclass, gliner-token-relex | any | — | family-not-exported | — |  |

## Electron integration

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `electron-macos-transformersjs` | macos | desktop-app | arm64 | transformers-js >=3.0.0 | cpu | transformersjs | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Base (≤ 900 MB) | — | pending | #3711 | Inference runs in one utility process with network entry points blocked. |
| `electron-windows-transformersjs` | windows | desktop-app | x86_64 | transformers-js >=3.0.0 | cpu | transformersjs | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Base (≤ 900 MB) | — | pending | #3711 |  |
| `electron-linux-transformersjs` | linux | desktop-app | x86_64 | transformers-js >=3.0.0 | cpu | transformersjs | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | int8 | Base (≤ 900 MB) | — | pending | #3711 |  |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `electron-renderer-inference` | any | desktop-app | any | any | any | any | any | process:renderer | policy-forbidden | — | Renderers receive span projections only; they never load models. |
| `electron-remote-models` | any | desktop-app | any | transformers-js >=3.0.0 | any | any | any | asset-origin:remote | policy-forbidden | — | The utility process loads local files only and blocks network calls. |

## Tauri sidecar

### Supported

None. No cell on this surface has evidence and an in-scope declaration at freeze.

### Experimental

| Cell | Platform | Device class | ABI | Runtime | Provider | Artifact | Families | Precision | Tier (RAM ceiling) | Condition | Evidence | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tauri-macos-sidecar` | macos | desktop-app | arm64 | torch >=2.13.0 | cpu | pytorch | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | #3712 | A self-contained Python sidecar binary is built per desktop target. |
| `tauri-windows-sidecar` | windows | desktop-app | x86_64 | torch >=2.13.0 | cpu | pytorch | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | #3712 |  |
| `tauri-linux-sidecar` | linux | desktop-app | x86_64 | torch >=2.13.0 | cpu | pytorch | bert, distilbert, electra, roberta, xlm-roberta, deberta-v2 | fp32 | Base (≤ 900 MB) | — | pending | #3712 |  |

### Unsupported combinations

| Entry | Platform | Device class | ABI | Runtime | Provider | Families | Precision | Condition | Reason | Related | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tauri-mobile-ios` | ios | iphone-ipad | any | any | any | any | any | — | not-qualified | — | Mobile targets cannot run the Python sidecar binary. |
| `tauri-mobile-android` | android | android-device | any | any | any | any | any | — | not-qualified | — | Mobile targets cannot run the Python sidecar binary. |
| `tauri-renderer-paths` | any | desktop-app | any | torch >=2.13.0 | any | any | any | process:renderer | policy-forbidden | — | The renderer cannot choose model or filesystem paths. |

## Evidence records

| Evidence | Kind | Covers | References | Note |
|---|---|---|---|---|
| `ci-python-contract` | ci-lane | contract-parity | `.github/workflows/ci.yml`<br>`tests/fixtures/parity` | The sdk-compatibility job runs core privacy and offset contracts on Linux, macOS and Windows; model pipelines are mocked. |
| `ci-python-low-resource` | ci-lane | runtime-model, memory-budget | `.github/workflows/low-resource.yml`<br>`scripts/benchmarks/low_resource_deid.py` | Runs a published DeBERTa-v2 INT8 Android-profile ONNX artifact on CPU over synthetic notes inside a 4 GiB cgroup. |
| `ci-python-edge-sbc` | ci-lane | runtime-synthetic, memory-budget | `.github/workflows/edge-benchmark.yml`<br>`openmed/eval/edge_benchmark.py` | Native aarch64 runner with a synthetic identity ONNX graph and footprint budgets; no model family runs. |
| `ci-swift-macos` | ci-lane | build, contract-parity | `.github/workflows/swift-test.yml`<br>`swift/OpenMedKit/Tests/OpenMedKitTests/OffsetContractParityTests.swift`<br>`swift/OpenMedKit/Tests/OpenMedKitTests/OpenMedMLXTests.swift` | Root package build and tests on a macOS arm64 runner; MLX compute tests skip when the runner exposes no usable Metal device. |
| `ci-swift-ios-build` | ci-lane | build | `.github/workflows/swift-test.yml` | Builds the package for the iOS Simulator SDK; no tests run on iOS. |
| `ci-swift-watchos-simulator` | ci-lane | build, contract-parity | `.github/workflows/swift-test.yml`<br>`swift/OpenMedKit/Tests/OpenMedKitTests/WatchVisionParityTests.swift` | Nano budget, selection and redaction-offset parity tests on a watchOS Simulator; no Core ML model is evaluated. |
| `ci-swift-visionos-simulator` | ci-lane | build, contract-parity | `.github/workflows/swift-test.yml`<br>`swift/OpenMedKit/Tests/OpenMedKitTests/WatchVisionParityTests.swift` | Nano budget, selection and redaction-offset parity tests on a visionOS Simulator; no Core ML model is evaluated. |
| `ci-android-jvm` | ci-lane | build, contract-parity | `.github/workflows/android-ci.yml`<br>`android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity/OffsetContractParityTest.kt` | Assembles the AAR and runs offset-contract parity on the JVM; ONNX Runtime does not execute. |
| `ci-web-pages` | ci-lane | contract-parity, runtime-synthetic | `.github/workflows/pages.yml`<br>`tests/web/test_webgpu_session.spec.ts`<br>`tests/browser/brand/webgpu-token-classification.spec.ts` | Node tests with an injected ONNX Runtime plus a headless Chromium WebGPU dispatch of the WGSL classification head. |
| `ci-flutter` | ci-lane | build, contract-parity | `.github/workflows/flutter-ci.yml`<br>`js/openmedkit-flutter/test/openmedkit_ffi_test.dart` | Dart FFI tests against the synthetic fixture session plus Android and Linux consumer builds; ONNX Runtime does not execute. |

## Model families

| Family | Export matrix row |
|---|---|
| `bert` | BERT |
| `distilbert` | DistilBERT |
| `electra` | Electra |
| `roberta` | RoBERTa |
| `xlm-roberta` | XLM-RoBERTa |
| `deberta-v2` | DeBERTa-v2 |
| `modernbert` | ModernBERT |
| `longformer` | Longformer |
| `openai-privacy-filter` | OpenAI Privacy Filter |
| `gliner-span` | GLiNER (span) |
| `gliclass` | GLiClass |
| `gliner-token-relex` | GLiNER (token-relex) |
<!-- kit-support-matrix:end -->

## See also

- [OpenMedKit Device Qualification](openmedkit-device-qualification.md)
- [Export Format Support Matrix](export-matrix.md)
- [Device Tiers and SLOs](tiers.md)
- [OpenMedKit Apple Platforms](runtimes/apple-platforms.md)
- [Android QNN and NNAPI](runtimes/android-accelerators.md)
- [ONNX Runtime Web Loader](runtimes/onnxruntime-web.md)
- [WebGPU Token-Classification Runtime](runtimes/webgpu.md)
