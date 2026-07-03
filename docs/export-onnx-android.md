# Android ONNX Export

OpenMed can export token-classification checkpoints for Android ONNX Runtime
Mobile with the `android` ONNX profile:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model OpenMed/example-token-classifier \
  --output dist/example-android-onnx \
  --profile android
```

The profile writes two ONNX graphs and, when ONNX Runtime's ORT conversion
tooling is installed, an ORT mobile artifact plus the minimal-build operator
configuration:

```text
dist/example-android-onnx/
  model.onnx
  model_fp16.onnx
  model.ort
  model.required_operators_and_types.config
  config.json
  id2label.json
  openmed-onnx.json
```

`model.onnx` is the fp32 graph. `model_fp16.onnx` stores fp16 weights while
keeping public input and output tensor types stable for Android callers.
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
`onnx-android`. When ORT conversion succeeds, the manifest also records an
`ort-android` artifact. Its metadata includes the `model.ort` path, the
required-operators/types config path, the fixed optimization style, Android ARM
target platform, type-reduction status, opset, and graph operators.

Unsupported operators are reported as warnings and recorded in artifact
metadata. The converter does not delete, rewrite, or silently drop graph nodes;
the warnings are used by the Android runtime tasks to decide whether a model
needs a fallback or exporter adjustment.
