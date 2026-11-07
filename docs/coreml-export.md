# CoreML Export

Use CoreML when you need a bundled Apple model package for Swift, iOS, or
macOS app integration. If you want the shared OpenMed MLX artifact path, see
the [MLX backend](mlx-backend.md) and
[OpenMedKit Swift guide](swift-openmedkit.md). For the cross-backend view
of which architectures reach which runtimes, see the
[Export Format Support Matrix](export-matrix.md).

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
- `modernbert` (qualification-gated, see below)
- `longformer` (qualification-gated, bounded buckets, see below)

Unsupported architectures fail before tracing with an error that lists the
supported families and carries the stable `unsupported_family` reason code.

## ModernBERT Qualification

ModernBERT is traced through the family adapter in
`openmed.torch.export_adapters`, always with one static
`(1, max_seq_length)` sequence bucket. Dynamic shapes
(`--disable-ane-static-shapes`), sequences longer than
`max_position_embeddings`, and Transformers 4.x are rejected before any
weights load. After conversion, every package (float and each `--quantize`
variant) is qualified against the PyTorch reference checkpoint on the
committed synthetic PHI suite: logit parity, span parity, direct-identifier
recall, critical leakage, and the INT8/INT4 recall-delta limits. The report is
written to `<stem>_qualification.json`, summarized under `qualification` in
the conversion manifest, and packages also carry `export_family`,
`export_adapter_version`, `sequence_shape_mode`, and `sequence_buckets`
metadata. A failing gate raises after the evidence is written, and
`--publish-to-hub` refuses unqualified packages.

Qualification runs Core ML predictions, so it needs macOS. Pass `--qualify`
to require it, or `--no-qualify` to skip it; elsewhere the manifest records
`status: not_run`. Gate tolerances, reason codes, and the offline reproduction
command are in the
[Export Format Support Matrix](export-matrix.md#modernbert-runtime-support).

## Longformer Qualification

Longformer packages are traced through the same family adapter and the same
qualification harness. Each package has one static `(1, max_seq_length)`
bucket and three inputs: `input_ids`, `attention_mask`, and an explicit
`global_attention_mask`. `--max-seq-length` must be a multiple of the largest
attention window, no longer than the position table allows, and its estimated
attention working set must fit the Core ML memory ceiling; otherwise the
request fails before weights load with `bucket_unavailable`,
`unsupported_sequence_length`, or `memory_budget_exceeded`. Export one package
per bucket you ship.

`--global-attention none` (the default, matching the stock
token-classification forward) or `--global-attention first_token` names the
fill policy that qualification uses and that packages record as
`global_attention_policy` metadata, next to `attention_windows`,
`memory_ceiling_bytes`, and `estimated_attention_bytes`. On top of the
ModernBERT gates, qualification sweeps minimum, typical, attention-window
boundary, and maximum lengths (with a first/middle/last global-attention
probe) and compares windowed long documents, with document-global offsets,
against the windowed reference. Bounds, the memory estimate, and the windowing
contract are in the
[Export Format Support Matrix](export-matrix.md#longformer-runtime-support).

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
