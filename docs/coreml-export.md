# CoreML Export (iOS & macOS)

OpenMed can export NER and PII models to Apple's [CoreML](https://developer.apple.com/documentation/coreml) format for deployment in iOS and macOS applications.

## Installation

```bash
pip install "openmed[coreml]"
```

This installs `coremltools`, `torch`, and `transformers`.

## Converting a Model

```bash
python -m openmed.coreml.convert \
    --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
    --output ./OpenMedPII.mlpackage
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | Required | HuggingFace model ID |
| `--output` | Required | Output `.mlpackage` path |
| `--max-seq-length` | 512 | Maximum token sequence length |
| `--precision` | float16 | `float16` (Neural Engine) or `float32` (CPU) |
| `--cache-dir` | None | HuggingFace model cache directory |

### Output Files

The converter produces:
- `OpenMedPII.mlpackage` — CoreML model package
- `OpenMedPII_id2label.json` — Label mapping (include this in your app bundle)

### Model Metadata

The CoreML model includes embedded metadata:
- `id2label` — JSON string mapping label IDs to names
- `num_labels` — Number of entity labels
- `max_seq_length` — Maximum input length
- `source_model` — Original HuggingFace model ID

## Using in Swift

See the [OpenMedKit Swift Package](swift-openmedkit.md) documentation for using the converted model in iOS/macOS apps.

### Manual Integration

If you prefer not to use OpenMedKit, you can integrate the CoreML model directly:

```swift
import CoreML

let model = try MLModel(contentsOf: modelURL)

// Prepare input
let inputIds = try MLMultiArray(shape: [1, seqLen], dataType: .int32)
let mask = try MLMultiArray(shape: [1, seqLen], dataType: .int32)
// ... fill with token IDs from your tokenizer ...

let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": MLFeatureValue(multiArray: inputIds),
    "attention_mask": MLFeatureValue(multiArray: mask),
])

// Run inference
let output = try model.prediction(from: input)
let logits = output.featureValue(for: "logits")!.multiArrayValue!
// ... apply softmax and decode BIO tags ...
```

## GitHub Actions

Use the `convert-models.yml` workflow to convert models in CI:

1. Go to Actions > "Convert Models"
2. Click "Run workflow"
3. Enter the model ID and desired formats
4. Download the converted model from the workflow artifacts
