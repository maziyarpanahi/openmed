# CoreML Export

Use CoreML when you need a bundled Apple model package for Swift, iOS, or
macOS app integration. If you want the shared OpenMed MLX artifact path, see
the [MLX backend](mlx-backend.md) and
[OpenMedKit Swift guide](swift-openmedkit.md).

The CoreML converter is a local packaging path for Hugging Face
token-classification checkpoints. After model download, conversion runs
locally and writes `.mlpackage` artifacts plus label sidecars for app bundles.

## Converter Contract

```bash
python -m openmed.coreml.convert \
  --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
  --output ./OpenMedPIISmall.mlpackage \
  --precision float16 \
  --compute-units cpuAndNeuralEngine \
  --quantize int8
```

The command writes:

- `OpenMedPIISmall.mlpackage`: the float16 CoreML package
- `OpenMedPIISmall_id2label.json`: label sidecar for the float package
- `OpenMedPIISmall_int8.mlpackage`: the INT8-palettized package when
  `--quantize int8` is set
- `OpenMedPIISmall_int8_id2label.json`: label sidecar for the INT8 package

Use `--quantized-output <path>` when the default `_int8.mlpackage` sibling name
does not fit your release layout.

## Supported Families

The converter accepts these Hugging Face token-classification source families:

- `bert`
- `distilbert`
- `electra`
- `roberta`
- `xlm-roberta`
- `deberta-v2`

Unsupported architectures fail before tracing with an error that lists the
supported families.

## Compute Units

Use `--compute-units` to declare the Core ML runtime target:

- `all`: Core ML may use all available compute units
- `cpuAndNeuralEngine`: prefer CPU plus Apple Neural Engine
- `cpuOnly`: CPU-only package execution

The selected value is passed to Core ML conversion and stored in package
metadata.

## Metadata

Each `.mlpackage` carries the following `user_defined_metadata` fields:

- `id2label`
- `num_labels`
- `max_seq_length`
- `source_model`
- `source_model_type`
- `compute_precision`
- `compute_units`
- `quantization`

The float package records `quantization=none`; the INT8 package records
`quantization=int8`.

## Manual CoreML Integration

If you already have a compatible CoreML model and prefer not to use
OpenMedKit, you can integrate it directly:

```swift
import CoreML

let model = try MLModel(contentsOf: modelURL)

let inputIds = try MLMultiArray(shape: [1, seqLen], dataType: .int32)
let mask = try MLMultiArray(shape: [1, seqLen], dataType: .int32)

let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": MLFeatureValue(multiArray: inputIds),
    "attention_mask": MLFeatureValue(multiArray: mask),
])

let output = try model.prediction(from: input)
let logits = output.featureValue(for: "logits")!.multiArrayValue!
```

For most apps, `OpenMedKit` is the simpler route.
