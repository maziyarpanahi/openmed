# Android ONNX Export

OpenMed can export token-classification checkpoints for Android ONNX Runtime
Mobile with the `android` ONNX profile:

```bash
python -m pip install --upgrade "openmed[onnx,hf]"
```

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model OpenMed/example-token-classifier \
  --output dist/example-android-onnx \
  --profile android
```

The profile writes fp32/fp16 ONNX graphs, an optional dynamic INT8 graph, and,
when ONNX Runtime's ORT conversion tooling is installed, an ORT mobile artifact
plus the minimal-build operator configuration:

```text
dist/example-android-onnx/
  model.onnx
  model_fp16.onnx
  model_int8.onnx
  model.ort
  model.required_operators_and_types.config
  config.json
  id2label.json
  openmed-onnx.json
```

## Run an OpenMed ONNX Model

Python CPU:

```python
from openmed import OnnxModel

model = OnnxModel.from_pretrained("OpenMed/example-v1-onnx-android")
entities = model("Patient Alice Nguyen was seen in cardiology.")
```

WebGPU or WebAssembly:

```typescript
import { loadOnnxModel } from "@openmed/openmedkit-web";

const model = await loadOnnxModel("OpenMed/example-v1-onnx-android");
const entities = await model("Patient Alice Nguyen was seen in cardiology.");
```

OpenMedKit for Android is published directly from OpenMed's `master` branch.
Add the scoped JitPack repository in `settings.gradle.kts`:

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

Add the moving JitPack snapshot to the app module:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:master-SNAPSHOT")
}
```

Then load the repository after it is stored in an app-local model directory:

```kotlin
val model = OpenMedKit.fromDirectory(modelDirectory)
val entities = model.analyzeText("Patient Alice Nguyen was seen in cardiology.")
```

JitPack consumers do not need GitHub credentials. Each runtime reads the
repository's tokenizer and label assets. Python and Web
download once and reuse their local cache; Android performs inference from the
app-local directory and makes no network request during inference.

## Private Batch Rollout

For the full OpenMed PyTorch token-classification inventory, use the resumable
batch runner. It reads `models.jsonl`, selects source PyTorch rows by default,
skips MLX/CoreML/ONNX-derived rows, and writes compact status evidence to
`dist/onnx-android/status.jsonl`. The runner disables Hugging Face Xet transfers
by default for rollout reliability; set `HF_HUB_DISABLE_XET=0` to opt back in:

```bash
.venv/bin/python scripts/onnx/batch_android_convert_publish.py --dry-run
```

To convert and upload privately, set a write token in `HF_WRITE_TOKEN` and run:

```bash
HF_WRITE_TOKEN=... \
.venv/bin/python scripts/onnx/batch_android_convert_publish.py \
  --publish-to-hub \
  --publish-manifest dist/onnx-android/private-models.jsonl
```

Published repositories are created private by the batch runner. There is no
public-upload flag in this path. Keep release promotion and updates to the
public `models.jsonl` as a separate audited step after Android parity,
quantization, and catalog checks pass.

GLiNER checkpoints are intentionally excluded from this batch. Their zero-shot
span contract requires prompt labels, word masks, candidate-span indices, and
span-logit decoding; it is not compatible with OpenMedKit's standard Android
token-classification session. GLiNER remains available through the Python and
MLX runtimes until a dedicated Android session is implemented and certified.

Privacy-filter mixture-of-experts checkpoints are also excluded. Their current
PyTorch forward pass uses data-dependent expert routing that cannot be safely
captured by the legacy ONNX tracer: the exported graph hard-codes routing split
sizes and fails parity execution on different synthetic inputs. These models
remain available through their supported Python and MLX runtimes.

`model.onnx` is the fp32 graph. `model_fp16.onnx` stores fp16 weights while
keeping public input and output tensor types stable for Android callers.
`model_int8.onnx` is produced with ONNX Runtime dynamic QInt8 quantization over
the MatMul/Gemm weights used by transformer token classifiers. This is the ARM
CPU starting point for Android Runtime Mobile because it needs no calibration
corpus and keeps the exported graph callable through the same inputs and
`logits` output.
`model.ort` is the ORT mobile-format model generated from `model.onnx`.
`model.required_operators_and_types.config` is the required-operators/types
file to pass into an ONNX Runtime Android minimal build.

If the optional ONNX Runtime conversion tooling is not available, export still
finishes with the `.onnx` artifacts and logs a dependency-only skip reason. The
message does not include source model text or sample input content.

## Graph Contract

The Android profile validates the exported graph with `onnx.checker` and then
checks the token-classification contract used by OpenMedKit:

- inputs: `input_ids`, `attention_mask`, and optional `token_type_ids`
- input dtype: `int64`
- output: `logits`
- output dtype: `float32`
- input axes: dynamic `batch` and dynamic `sequence`
- output axes: dynamic `batch`, dynamic `sequence`, and static `labels`

The profile fixes the ONNX opset at `18`. A fixed opset is required because the
follow-up ONNX Runtime Mobile `.ort` conversion and minimal-build operator
configuration need a predictable graph contract. The dynamic `batch` and
`sequence` axes are retained so one exported artifact can serve variable
request batches and note lengths on device.

## ORT Mobile Minimal Build

The ORT mobile step uses ONNX Runtime's Python conversion tool with the fixed
optimization style, Android ARM target platform, and type reduction enabled.
The resulting `model.required_operators_and_types.config` can be passed to the
ONNX Runtime build as the include-ops config, for example:

```bash
./build.sh \
  --android \
  --minimal_build \
  --include_ops_by_config dist/example-android-onnx/model.required_operators_and_types.config
```

Building the actual Android AAR is a separate Android-module task; this export
only writes the model and operator/type configuration consumed by that build.

## Mobile Compatibility Metadata

`openmed-onnx.json` records the Android artifacts with the format string
`onnx-android` for fp32/fp16 and `onnx-int8` for the dynamic INT8 graph. When
ORT conversion succeeds, the manifest also records an `ort-android` artifact.
Each artifact entry includes profile metadata with the opset, tensor contract,
execution-provider hints (`NNAPI`, `XNNPACK`), graph operators, and any
operators that are outside OpenMed's Android mobile safe set. The INT8 artifact
metadata records the dynamic quantization recipe and target `model_int8.onnx`
path. The ORT artifact metadata records the `model.ort` path, the
required-operators/types config path, the fixed optimization style, Android ARM
target platform, type-reduction status, opset, and graph operators.

Unsupported operators are reported as warnings and recorded in artifact
metadata. The converter does not delete, rewrite, or silently drop graph nodes;
the warnings are used by the `.ort` and Android runtime tasks to decide whether
a model needs a fallback or exporter adjustment.

## INT8 Recall Certification

When an eval suite is available, pass it to the Android export:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model OpenMed/example-token-classifier \
  --output dist/example-android-onnx \
  --profile android \
  --eval-suite openmed/eval/golden/fixtures/checksum_ids.json
```

The converter writes `recall_delta.json` next to the ONNX artifacts. The report
compares the Hugging Face full-precision parent with `model_int8.onnx` on the
same fixtures and records:

- `format`: `onnx-int8`
- `quantization`: dynamic QInt8 metadata for the ARM CPU artifact
- `fp_parent_per_label_recall`
- `candidate_per_label_recall`
- `per_label`, with fp32 recall, INT8 recall, recall delta, and aggregate gold
  span/character counts
- `quant_recall_delta`, the single max recall-loss figure consumed by G4
- `limit`, currently `0.005` for INT8
- `certified`, true only when the measured delta is below the INT8 limit

The report is aggregate-only. It does not serialize fixture text, record IDs,
span text, offsets, gold spans, or predicted spans. Use synthetic or approved
eval fixtures and keep any DUA-governed corpus outside committed artifacts.

The artifact `config.json` and `openmed-onnx.json` mirror the certification
fields under `quant_recall_delta`, `certified`, `recall_delta_path`, and
`quantization`. An over-budget INT8 export still writes `model_int8.onnx` and
the report, but records `certified: false` so release gates can block the
format.
