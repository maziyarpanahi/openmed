# Android ONNX Export

OpenMed can export token-classification checkpoints for Android ONNX Runtime
Mobile with the `android` ONNX profile:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model OpenMed/example-token-classifier \
  --output dist/example-android-onnx \
  --profile android
```

The profile writes two ONNX graphs:

```text
dist/example-android-onnx/
  model.onnx
  model_fp16.onnx
  config.json
  id2label.json
  openmed-onnx.json
```

`model.onnx` is the fp32 graph. `model_fp16.onnx` stores fp16 weights while
keeping public input and output tensor types stable for Android callers.

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

## Mobile Compatibility Metadata

`openmed-onnx.json` records the Android artifacts with the format string
`onnx-android`. Each artifact entry includes profile metadata with the opset,
tensor contract, execution-provider hints (`NNAPI`, `XNNPACK`), graph
operators, and any operators that are outside OpenMed's Android mobile safe
set.

Unsupported operators are reported as warnings and recorded in artifact
metadata. The converter does not delete, rewrite, or silently drop graph nodes;
the warnings are used by the later `.ort` and Android runtime tasks to decide
whether a model needs a fallback or exporter adjustment.
