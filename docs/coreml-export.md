# CoreML Packaging Status (iOS & macOS)

Use CoreML when you need a bundled Apple model package for **Swift/iOS/macOS app integration**. If you want the shared OpenMed MLX artifact path, see the [MLX backend](mlx-backend.md) and [OpenMedKit Swift guide](swift-openmedkit.md).

OpenMedKit is the public Swift runtime in `1.0.0`, and now supports both MLX and CoreML backends. The universal OpenMed-to-CoreML packaging workflow is still being generalized across the model collection, so conversion should be treated as **active platform work**, not a stable public release surface yet.

## Current Status

As of April 4, 2026:

- the `OpenMedKit` Swift package builds and tests successfully
- the `OpenMedDemo` Xcode project builds and launches on macOS
- Swift MLX is the forward Apple Silicon path for supported BERT-family artifacts
- MLX artifacts such as `weights.safetensors` or `weights.npz` are still separate from CoreML app bundles
- a fresh DeBERTa-v2 pilot export is **not** yet release-ready in the current arm64 CoreML environment

## What To Ship Today

When you already have a compatible CoreML bundle, the app-facing packaging contract is:

- `YourModel.mlmodelc` or `.mlpackage`
- `id2label.json`
- tokenizer assets if the app must run offline

That is the stable surface consumed by [OpenMedKit](swift-openmedkit.md).

## Architecture Rollout

OpenMed is actively working toward a universal Apple packaging path for:

- BERT
- DistilBERT
- RoBERTa
- XLM-RoBERTa
- Longformer
- ModernBERT
- EuroBERT
- Qwen3

The goal is one repeatable packaging story across the collection rather than a one-off converter for a single checkpoint.

## Manual CoreML Integration

If you already have a compatible CoreML model and prefer not to use OpenMedKit, you can integrate it directly:

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

For most apps, though, `OpenMedKit` is the simpler route.
