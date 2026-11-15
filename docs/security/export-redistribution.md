# Export Toolchains, Artifact Licenses and Redistribution

This page maps every OpenMed export command to the converter toolchain that
produces its artifacts, lists the runtime libraries each OpenMedKit kit brings
into an application together with their licenses, explains how a model's
license travels from its model card to an exported artifact and an app bundle,
and lists the files each runtime must find on local storage before it can run
offline. It is engineering documentation for release reviews, not legal advice.

Versions are given as pinned at the time of writing (2026-11-07). The manifest
named in each row is authoritative: when it changes, it wins over this page.
Licenses were read from each dependency's own manifest or `LICENSE` file in the
resolved package caches (SwiftPM checkouts, Gradle POMs, npm, cargo and pub
caches). A license marked **To confirm** could not be verified from a local
manifest; check it against the exact artifact before you ship.

Python dependency licenses are reviewed in the
[Dependency License Inventory](license-inventory.md) and gated offline by
`scripts/licenses/inventory.py`. Swift, Android and JavaScript kit dependencies
have no equivalent gate yet; #3937 tracks it. Recording source revisions, file
digests and toolchain versions inside export manifests is tracked by #3941.

## Export toolchains

Each export command in the export and runtime guides maps to one row. "Locked"
is the version resolved in `uv.lock`; the range before it is the constraint in
`pyproject.toml`. Export runs in a Python environment and copies none of these
converter packages into the artifacts it writes. Python runtimes that ship
inside an app, such as the Tauri sidecar below, are covered with the kits.

| Export command | Guide | Install | Toolchain as pinned at the time of writing | Writes |
| --- | --- | --- | --- | --- |
| `python -m openmed.coreml.convert` | [CoreML Packaging](../coreml-export.md) | `openmed[coreml]` | coremltools `>=8.0` (locked 9.0) for conversion and k-means palettization; torch `>=2.13.0` (2.13.0); transformers `>=4.50` (5.22.1); huggingface-hub `>=0.30` (1.33.0). Minimum deployment target iOS 16. | `.mlpackage`, `_id2label.json`, optional `_int8.mlpackage` |
| `python -m openmed.onnx.convert` (default profile) | [ONNX and WebGPU Export](../export-onnx-webgpu.md) | `openmed[onnx]` | `torch.onnx` exporter from torch `>=2.13.0` (2.13.0) with onnxscript `>=0.1.0` (0.7.0); onnx `>=1.21` (1.22.0); onnxruntime `>=1.16` (1.23.2) for graph optimization, fp16 conversion and parity runs; transformers `>=4.50` (5.22.1). Opset 18. | `model.onnx`, `model.webgpu.onnx`, `model.unoptimized.onnx`, `openmed-onnx.json` |
| `python -m openmed.onnx.convert --include-transformersjs`, `python -m openmed.onnx.transformersjs` | [Transformers.js Export](../export-transformersjs.md) | `openmed[onnx]` | Same as the default profile; onnxruntime dynamic quantization writes the INT8 graph. | `transformersjs/` bundle |
| `python -m openmed.onnx.convert --profile android`, `scripts/onnx/batch_android_convert_publish.py` | [Android ONNX Export](../export-onnx-android.md) | `openmed[onnx,hf]` | Same as the default profile with fixed opset 18; onnxruntime dynamic QInt8 quantization; onnxruntime's ORT-format converter (fixed optimization style, ARM target) when it is installed. | `model.onnx`, `model_fp16.onnx`, `model_int8.onnx`, `model.ort`, `model.required_operators_and_types.config`, `openmed-onnx.json` |
| ONNX Runtime minimal build (`./build.sh --minimal_build`) | [Android ONNX Export](../export-onnx-android.md#ort-mobile-minimal-build) | An ONNX Runtime source checkout outside this repository | ONNX Runtime's own build toolchain; not pinned by OpenMed. | A custom ONNX Runtime Android build |
| `python -m openmed.onnx.convert --profile openvino`, `quantize_openvino_int8` | [OpenVINO Runtime](../runtimes/openvino.md) | `openmed[onnx,openvino]` | openvino `>=2024.0` (2026.2.1); nncf `>=2.10` (3.2.0) for INT8; onnxruntime for the synthetic parity check. | `openvino/model.xml`, `openvino/model.bin` |
| `build_tensorrt_engine` | [TensorRT Runtime](../runtimes/tensorrt.md) | User-installed NVIDIA TensorRT, CUDA-enabled PyTorch and optional NVIDIA Model Optimizer | Not declared or pinned by OpenMed. | A device-specific TensorRT engine |
| `python -m openmed.mlx.convert` (including `--quantize` and `--segmenter`) | [MLX Quantized Export](../export-mlx-quant.md), [On-device Segmenters](../on-device-segmenters.md) | `openmed[mlx,hf]` on Apple Silicon | mlx `>=0.22` (0.32.4); mlx-lm `>=0.31` (0.31.3); safetensors `>=0.4` (0.8.0); tokenizers `>=0.15`; transformers and torch load the source checkpoint. | `openmed-mlx.json`, `config.json`, `id2label.json`, tokenizer assets, `weights.safetensors` |
| `python -m openmed.mlx.maple_export` | [Maple Preview on Device](../maple-on-device.md) | `openmed[mlx]` on Apple Silicon | MLX-LM conversion from mlx-lm `>=0.31` (0.31.3); the source revision is pinned in `openmed/mlx/maple_export.py`. | 4-bit and 8-bit MLX-LM variants with `openmed-maple-export.json` and the upstream `LICENSE` |
| `python -m openmed.onnx.maple_export` (`requirements`, `download`, `export`, `bundle`), `python -m openmed.onnx.maple_qmoe_smoke` | [Maple Preview on Device](../maple-on-device.md) | A separate environment installed from the `requirements` subcommand output | Exact pins in `MAPLE_ONNX_EXPORT_REQUIREMENTS` in `openmed/onnx/maple_export.py`: onnxruntime-genai 0.12.0 model builder, onnxruntime 1.25.1, onnx 1.21.0, onnx-ir 0.2.0, torch 2.13.0, transformers 4.57.1, plus matching accelerate, huggingface-hub, numpy, safetensors, tokenizers and tqdm. | ONNX graph, `openmed-maple-onnx-export.json`, `.ommaple.zip` bundle |
| `python -m openmed.gguf.convert` | [GGUF Embedding Export](../export-gguf.md) | A user-built llama.cpp checkout and its converter requirements | llama.cpp `convert_hf_to_gguf.py`; not declared or pinned by OpenMed. | `model-f16.gguf`, `model-q8_0.gguf`, `openmed-gguf.json` |
| `export_gguf_int4` | [GGUF Grounding Runtime](../runtimes/gguf-grounding.md) | The same llama.cpp checkout plus its `llama-quantize` and `llama-embedding` builds | Not declared or pinned by OpenMed. | `model-q4_k_m.gguf` and its recall certificate |
| `quantize_awq`, `quantize_awq_grounding` | [AWQ Export](../export-awq.md), [AWQ Grounding Embedder](../runtimes/awq-grounding.md) | `openmed[awq]`, Linux only | autoawq `==0.2.9`; transformers `==5.12.1`; torch `>=2.13.0` (2.13.0); datasets `>=5.0.1` (5.0.1). | AWQ checkpoint and `quant_config.json` |
| `quantize_gptq` | [GPTQ Export](../export-gptq.md) | `openmed[gptq]` | auto-gptq `>=0.7.1` (0.7.1); torch `>=2.13.0` (2.13.0); datasets `>=5.0.1` (5.0.1). | GPTQ checkpoint and `quant_config.json` |

Hugging Face Optimum is not part of any export path: the ONNX converter calls
the PyTorch exporter directly, and `pyproject.toml` declares no `optimum`
dependency. Because extras are resolved together, `uv.lock` also records a
second version of a few packages for specific extra combinations, for example
transformers 5.12.1 for the `awq` pin beside 5.22.1, tokenizers 0.22.1 beside
0.23.3, and onnxscript 0.5.0 beside 0.7.0.

Converter licenses, from the dependency license inventory unless noted:

| Package | License | Note |
| --- | --- | --- |
| coremltools | `BSD-3-Clause` | |
| torch | `BSD-3-Clause` | |
| transformers, huggingface-hub, tokenizers, safetensors, datasets | `Apache-2.0` | |
| onnx | `Apache-2.0` | |
| onnxruntime, onnxscript | `MIT` | |
| openvino, nncf | `Apache-2.0` | |
| mlx, mlx-lm | `MIT` | |
| autoawq, auto-gptq | `MIT` | |
| onnxruntime-genai | To confirm | Maple ONNX only, outside `pyproject.toml`. A cached 0.16.0 wheel declares MIT; confirm for the pinned 0.12.0. |
| onnx-ir | To confirm | Maple ONNX only, outside `pyproject.toml`. A cached 1.0.0 wheel declares Apache-2.0; confirm for the pinned 0.2.0. |
| llama.cpp | `MIT` | From the `LICENSE` of a local checkout; the converter is always the user's own checkout. |
| NVIDIA TensorRT, CUDA, Model Optimizer | To confirm | NVIDIA terms; user-installed and never distributed by OpenMed. |

## Kit runtime libraries

These libraries are linked into, or installed beside, the applications that
use OpenMedKit, so an app that ships them redistributes them and must carry
their notices. OpenMed declares them as package dependencies; it does not
vendor their sources or binaries into this repository.

### Swift

The root `Package.swift` (pins in `Package.resolved`) is what SwiftPM
consumers of the repository URL resolve. The nested
`swift/OpenMedKit/Package.swift` currently pins a newer mlx-swift and adds
mlx-swift-lm; #3929 aligns the two manifests.

| Library | Pinned at the time of writing | License | Verified from | Ships in |
| --- | --- | --- | --- | --- |
| swift-transformers | `exact: 1.0.0` | `Apache-2.0` | SwiftPM checkout `LICENSE` | `OpenMedKit` on iOS and macOS |
| mlx-swift | `exact: 0.31.3` (root), `exact: 0.31.6` (nested) | `MIT` | SwiftPM checkout `LICENSE`; vendored MLX core, mlx-c, fmt and nlohmann/json are MIT, metal-cpp is Apache-2.0 | `OpenMedKit` on iOS and macOS |
| mlx-swift-lm | revision `42f08a872075fd07f9f1f40ec1a5e191e6aad86e` (nested only) | `MIT` | SwiftPM checkout `LICENSE` | Maple and Compass generative paths on iOS and macOS |
| ZIPFoundation | `from: 0.9.19` (resolved 0.9.20) | `MIT` | SwiftPM checkout `LICENSE` | `OpenMedKit` on iOS and macOS |
| argmax-oss-swift (WhisperKit) | `exact: 1.1.1` | `MIT` | SwiftPM checkout `LICENSE`; its `NOTICES` file records Apache-2.0 code derived from swift-transformers | Only the optional `OpenMedSpeech` product |
| swift-jinja, swift-collections, swift-numerics, swift-argument-parser, swift-syntax | Resolved transitively | `Apache-2.0` | SwiftPM checkout `LICENSE` | Transitive dependencies of the packages above |

Core ML, Metal and the other Apple system frameworks are part of the operating
system and are not redistributed. watchOS and visionOS builds use the
Core ML-only `PlatformModel` surface without MLX or swift-transformers.

### Android

Versions come from `android/gradle/libs.versions.toml`; the core module
declares them in `android/openmedkit/build.gradle.kts`.

| Library | Pinned at the time of writing | License | Verified from | Ships in |
| --- | --- | --- | --- | --- |
| `com.microsoft.onnxruntime:onnxruntime-android` | 1.25.1 | `MIT` | Gradle cache POM | Every OpenMedKit Android consumer |
| `ai.djl.huggingface:tokenizers`, `ai.djl.android:tokenizer-native` (with `ai.djl:api`) | 0.33.0 | `Apache-2.0` | Gradle cache POMs | Every OpenMedKit Android consumer |
| `org.jetbrains.kotlinx:kotlinx-coroutines-core` | 1.8.1 | `Apache-2.0` | Gradle cache POM | Every OpenMedKit Android consumer |
| `org.jetbrains.kotlinx:kotlinx-serialization-json` | 1.5.1 | `Apache-2.0` | Gradle cache POM | Every OpenMedKit Android consumer |
| `org.jetbrains.kotlin:kotlin-stdlib` | Follows the Kotlin plugin (1.8.22; 1.9.24 in the Flutter plugin) | `Apache-2.0` | Gradle cache POM (1.4.32) | Every Kotlin consumer, including the Flutter Android plugin |
| `com.google.mlkit:text-recognition` | 16.0.1 | Proprietary Google ML Kit terms | The artifact declares no open-source license; the AAR carries `third_party_licenses.txt` for embedded components | Every OpenMedKit Android consumer today; see restricted components |
| Qualcomm QNN libraries | Not a dependency | To confirm (Qualcomm terms) | Not distributed by OpenMed | Only apps that supply their own QNN build |

### JavaScript, desktop and cross-platform kits

| Kit | Library | Pinned at the time of writing | License | Verified from | Ships in |
| --- | --- | --- | --- | --- | --- |
| `js/openmedkit-web` | `onnxruntime-web` | Optional peer `>=1.20.0` | `MIT` | npm cache manifest (1.30.0) | The app's web bundle, with its WASM files |
| `js/openmedkit-web` | `@huggingface/transformers` | Optional peer `>=3.0.0` | `Apache-2.0` | npm cache manifest (4.2.0) | The app's web bundle |
| `js/openmedkit-web` | Transformers.js dependencies: `onnxruntime-node`, `@huggingface/tokenizers`, `@huggingface/jinja`, `sharp` | Resolved by the app | `MIT`, `Apache-2.0`, `MIT`, `Apache-2.0` | npm cache manifests | Node installs, including Electron |
| `js/openmedkit-web` | `@img/sharp-libvips-*` prebuilt binaries pulled by `sharp` | Resolved by the app | `LGPL-3.0-or-later` | npm cache manifest | Node installs that keep `sharp`; review before shipping |
| `js/openmedkit-electron` | `electron` | Optional peer `>=30` | To confirm | Not resolved in any OpenMed lockfile | The app |
| `js/openmedkit-react-native` | `react`, `react-native` | Optional peers `>=18`, `>=0.74` | To confirm | Not resolved in any OpenMed lockfile | The app; native modules link the Swift and Android kits above |
| `js/openmedkit-flutter` | `ffi` | `>=1.1.2 <3.0.0` | `BSD-3-Clause` | pub cache `LICENSE` (2.2.0) | Flutter apps |
| `js/openmedkit-flutter` | `onnxruntime-android` (Gradle), `onnxruntime-c` (CocoaPods) | 1.20.0 | To confirm | Not in the local caches; the ONNX Runtime 1.25.1 POM declares MIT | Flutter Android, iOS and macOS apps |
| `js/openmedkit-tauri` | `@tauri-apps/api` | Peer `>=2.0.0` (lockfile 2.11.1) | `Apache-2.0 OR MIT` | `js/openmedkit-tauri/package-lock.json` | The Tauri front end |
| `js/openmedkit-tauri` | `tauri`, `tauri-plugin-shell` | `2` (locked 2.11.5, 2.3.5) | `Apache-2.0 OR MIT` | cargo registry manifests | The trusted Rust host |
| `js/openmedkit-tauri` | `serde`, `serde_json`, `tokio` | `1` (locked 1.0.229, 1.0.150, 1.53.0) | `MIT OR Apache-2.0`, `MIT OR Apache-2.0`, `MIT` | cargo registry manifests (serde_json read from cached 1.0.132) | The trusted Rust host |
| `js/openmedkit-tauri` | PyInstaller, used to freeze the sidecar in the Tauri example | Not pinned | To confirm | Not in the local caches | The frozen sidecar binary embeds its bootloader; review before distributing |

The Tauri sidecar runs the Python OpenMed runtime with the `hf` extra, so the
frozen binary also redistributes the Python packages listed in the dependency
license inventory for that extra.

## Restricted and vendor components

These components are either proprietary, weak-copyleft, or reach the network.
Each has a default posture and an explicit opt-in path.

| Component | Terms | Default posture | Opt-in path |
| --- | --- | --- | --- |
| Google ML Kit text recognition (Android) | Proprietary Google ML Kit terms | Declared as an `implementation` dependency of the core `android/openmedkit` module, so every consumer inherits it. Its graph adds `com.google.mlkit:common`, which registers an auto-starting `MlKitInitProvider`, and `com.google.android.datatransport:transport-backend-cct`, whose manifest merges `INTERNET` and `ACCESS_NETWORK_STATE` into the host app. | #3928 moves the ML Kit adapter into a separate optional artifact. Until it lands there is no supported way to drop ML Kit from the core AAR. |
| Qualcomm QNN execution provider (Android) | Qualcomm SDK terms, to confirm per SDK release | Not bundled. The standard ONNX Runtime AAR has no QNN provider, so the probe falls through to NNAPI and then CPU. | Supply a QNN-enabled ONNX Runtime build and `libQnnHtp.so`, then configure `AcceleratorConfig`. See [Android QNN and NNAPI](../runtimes/android-accelerators.md). |
| NVIDIA TensorRT, CUDA and Model Optimizer | NVIDIA terms | Not installed, declared or distributed; no prebuilt engines. | Install on the deployment target and build engines locally. See [TensorRT Runtime](../runtimes/tensorrt.md). |
| llama.cpp | `MIT` | Not cloned, built or bundled. | Point the GGUF exporter and grounding runtime at a local build. |
| Hugging Face Hub downloads | Each model's own terms | Export and download happen only on explicit calls (`OpenMedModelStore.downloadMLXModel`, Android `ModelDownloader.download`, a repository ID passed to a loader, `--publish-to-hub`). | Download during a setup step without PHI, then run offline as described below. |
| Transformers.js default WASM location | Not a license issue | Transformers.js 4.2.0 points ONNX Runtime Web at a jsDelivr CDN copy of its WASM files unless the application sets `env.backends.onnx.wasm.wasmPaths`. The OpenMed loader controls remote model loading but does not set this path. | Configure the Transformers.js module with local WASM paths and pass it through the loader's `runtime` option, or use the direct ONNX Runtime Web session, which rejects remote asset paths. |
| `sharp` and libvips in Node installs | `Apache-2.0`; libvips binaries `LGPL-3.0-or-later` | Not used by OpenMed for text; pulled in by Transformers.js in Node and Electron installs. | Review LGPL obligations, or prune the package from a text-only desktop build after testing. |
| Third-party weights with custom licenses | For example the LFM Open License v1.0 for LFM2.5 | Never mirrored under OpenMed. | Users download them under the upstream terms. See [Maple Preview on Device](../maple-on-device.md). |

## Model-license propagation

A model's license follows a fixed chain, and every link inherits the license
of the one before it:

1. **Model card.** The source repository's model card declares the license.
   `scripts/manifest/generate_manifest.py` reads the Hub `license:` tag into the
   `license` field of `models.jsonl`, which is `null` when the card declares
   none. See [Model Manifest](../model-manifest.md).
2. **Exported artifact.** An ONNX graph, MLX weight file, Core ML package, GGUF
   file or AWQ/GPTQ checkpoint is a derivative of the source weights.
   Conversion, quantization and palettization never relax the source license,
   and the derivative needs the same attribution and notices. Tokenizer assets
   copied into the artifact keep the license of the repository they came from.
3. **App bundle.** A bundle carries every license it contains: each model and
   tokenizer, every linked runtime library in the kit tables above, and data
   resources such as the ICU notice that on-device segmenter bundles include
   as a digest-bound `license_notice` resource.

What the exporters record today:

- `openmed-onnx.json` and `openmed-mlx.json` record `source_model_id` for
  provenance. They have no license field and do not copy the source `LICENSE`
  or model card.
- The Core ML converter writes `license = "Apache-2.0"` and `author = "OpenMed"`
  into every package's metadata, whatever the source license
  (`openmed/coreml/convert.py`).
- The Hub publisher writes `"license": "apache-2.0"` into the generated
  `models.jsonl` row, and the generated model card reads that row
  (`openmed/core/hf_publish.py`, `openmed/core/model_card.py`).
- The Maple MLX exporter copies the upstream `LICENSE` and records `MIT` in
  `openmed-maple-export.json`.

The Apache-2.0 defaults are correct only when the source model is itself
Apache-2.0, which holds for most OpenMed checkpoints. Before publishing a
derivative of any other source, correct the Core ML metadata, the registry row
and the model card by hand, and ship the upstream license file beside the
artifact. A registry row whose license is `other` or `null` must not be
redistributed until it has been reviewed. Automatic propagation belongs to
#3941.

## What OpenMed may and may not redistribute

OpenMed may redistribute:

- its own source code and kits under Apache-2.0 (`LICENSE`, `NOTICE`); the
  Android POM, the Flutter podspecs and the web package all declare Apache-2.0
- the third-party data assets listed in `NOTICE`, each with its retained
  notice and provenance record
- OpenMed checkpoints and their exported derivatives when the model card
  declares a permissive license and the base model's license and training-data
  terms allow derivative redistribution
- synthetic fixtures and synthetic evaluation data

OpenMed does not redistribute:

- proprietary SDKs, binaries or engines: Qualcomm QNN, NVIDIA TensorRT and
  CUDA, or prebuilt TensorRT engines. ML Kit is the one proprietary component
  that currently reaches consumers through the Android dependency graph; #3928
  removes it from the core artifact.
- third-party weights under custom, restrictive or undeclared licenses, and
  gated models that require accepting terms; users download these under the
  upstream terms and OpenMed never accepts terms on their behalf
- locally built demo runtimes, such as the patched ONNX Runtime Web build for
  the browser demo, which stay in a git-ignored `vendor/` directory
- anything on the never-bundle list below

## Never bundle

Consistent with the repository rules in `AGENTS.md`, OpenMed packages,
artifacts, fixtures and bundles must never include:

- **GPL-licensed code or binaries** in the in-process runtime. The only
  GPL dependency, `extract-msg` (`GPL-3.0-only`), is limited to the
  `email-msg-gpl` extra's out-of-process bridge.
- **Source-available** code or weights whose license is not open source.
- **Proprietary** SDKs, libraries or model weights. Restricted integrations
  stay behind optional user-supplied keys or out-of-process bridges.
- **DUA-gated data** and corpora: **MIMIC**, **i2b2** and **n2c2**. They are
  evaluation-only, are never committed, and weights trained on them are not
  redistributed when the agreement restricts derivatives.
- **Licensed terminologies**: **UMLS**, **SNOMED CT** and **CPT**.
- Real PHI. Committed fixtures and golden data are synthetic.

## Offline inputs by runtime

Prepare these files on local storage before a runtime has to work offline.
After they exist, inference makes no network request.

| Runtime | Files it loads | Network control |
| --- | --- | --- |
| Python ONNX Runtime (`load_onnx_model`, `OnnxModel`) | One of `model_int8.onnx`, `model.onnx` or `model_fp16.onnx` (tried in that order for `variant="auto"`, also under `onnx/`), `config.json`, `tokenizer.json`, and labels from `id2label.json` or `config.json` | Pass a local directory, or set `OPENMED_OFFLINE=1` after caching the snapshot. See [Offline Installation](../offline-install.md). |
| Python MLX | `openmed-mlx.json`, `config.json`, `id2label.json` when present, the tokenizer files the manifest lists, and `weights.safetensors` or `weights.npz` | Local directory, or `OPENMED_OFFLINE=1` after caching. |
| Python OpenVINO (`OpenVinoTokenClassificationSession`) | `openvino/model.xml` and `openvino/model.bin`; the caller supplies token IDs | Local paths only. |
| Python GGUF grounding (`load_gguf_grounding_embedder`) | The artifact directory with `openmed-gguf.json`, the GGUF file and its passing recall report, plus a locally built `llama-embedding` executable | The bridge rejects remote-model and RPC arguments. |
| Swift Core ML (`OpenMedBackend.coreML`) | A compiled `.mlmodelc` (an `.mlpackage` is compiled at load on iOS and macOS only), `id2label.json`, and a tokenizer folder with `tokenizer.json` and `tokenizer_config.json` | Pass `tokenizerFolderURL` and `allowNetworkAccess: false`. Without a tokenizer folder, the default allows fetching tokenizer assets by `tokenizerName`. |
| Swift MLX (`OpenMedBackend.mlx`) | The same MLX artifact files as Python MLX; Apple Silicon Macs and real iPhone or iPad hardware only | Run `OpenMedModelStore.downloadMLXModel` during setup or ship the directory. |
| Swift `PlatformModel` (watchOS, visionOS) | A precompiled INT8 `.mlmodelc` and `id2label.json`; the app supplies token IDs and offsets | No network path. |
| Swift `OpenMedSpeech` | The pinned `AudioEncoder.mlmodelc/`, `MelSpectrogram.mlmodelc/`, `TextDecoder.mlmodelc/` and `tokenizer/` files listed in [Local Core ML Speech](../multimodal/apple-speech.md) | Verified against built-in digests; never downloads. |
| Android (`OpenMedKit.fromDirectory`, `OpenMedBackend`) | `model_int8.onnx` for `fromDirectory` by default (`model.onnx` for `OpenMedBackend` and `variant = "fp32"`, `model_fp16.onnx` for `"fp16"`), `tokenizer.json`, `tokenizer_config.json` when present, `id2label.json`, and `openmed-onnx.json` when present for accelerator operator coverage. The exported `model.ort` is not loaded by the OpenMedKit session. | Pre-warm the cache with `ModelDownloader.download` during setup, or ship the directory. See [Android Integration](../android-integration.md). |
| Android with QNN | The Android files above, plus an app-supplied QNN-enabled ONNX Runtime build and `libQnnHtp.so` | No network path. |
| Web, direct ONNX Runtime session (`loadWebGpuTokenClassificationSession`) | `model.webgpu.onnx` and `model.onnx`, plus the ONNX Runtime Web `.wasm` and `.mjs` files under `assetPath`; the caller supplies token IDs | HTTP(S), `data:` and `blob:` paths are rejected. See [WebGPU Token-Classification Runtime](../runtimes/webgpu.md). |
| Web, Transformers.js (`loadOnnxModel`) | `config.json`, `tokenizer.json`, `tokenizer_config.json` and `model_int8.onnx` at the repository root (`variant` selects `model.onnx` or `model_fp16.onnx`). A `transformersjs/` bundle read by the Transformers.js `pipeline` uses `onnx/model.onnx` or `onnx/model_quantized.onnx`. | Local references set `localFilesOnly`; also provide local WASM paths as described above. |
| Electron | The Transformers.js files above in a local model directory; inference uses the `onnxruntime-node` native binary | The utility process sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and blocks network APIs. See [Electron In-Process De-identification](../runtimes/electron.md). |
| React Native | iOS: the MLX artifact (default backend), or a Core ML model with `id2LabelPath` and `tokenizerFolderPath`. Android: `model.onnx`, `tokenizer.json` and `id2label.json` or `id2LabelPath`. | Pass `tokenizerFolderPath` for Core ML, otherwise the bridge's default allows a tokenizer fetch. |
| Flutter | `id2label.json`, `tokenizer.json`, and one of `model_int8.onnx`, `model.onnx` or `model_fp16.onnx`; an app-supplied tokenizer adapter; ONNX Runtime 1.20.0 native libraries | The plugin never downloads models. |
| Tauri | The frozen sidecar binary for each target under `src-tauri/binaries/`, and a pre-provisioned Hugging Face model directory pinned by the Rust host or `OPENMED_SIDECAR_MODEL` | The sidecar always enables local-only mode; a missing model fails closed. |

## Related pages

- [Export Format Support Matrix](../export-matrix.md)
- [Dependency License Inventory](license-inventory.md)
- [Multimodal Dependency Licenses](../compliance/multimodal-dependency-licenses.md)
- [Supply Chain Controls](supply-chain.md)
- [Swift Package (OpenMedKit)](../swift-openmedkit.md)
- [Android Integration & Model Catalog](../android-integration.md)
