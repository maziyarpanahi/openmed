# ONNX and WebGPU Export

The default OpenMed ONNX profile converts a Hugging Face
token-classification checkpoint or local model directory into:

- an fp32 ONNX graph for ONNX Runtime
- an fp16 ONNX graph intended for WebGPU, emitted only when it passes the
  [fp16 WebGPU parity gate](#fp16-webgpu-parity-gate)
- an optional Transformers.js bundle for browser pipelines

For the cross-backend view of which architectures reach ONNX (default,
Android, or Transformers.js), see the
[Export Format Support Matrix](export-matrix.md).

Android uses a separate, fixed-opset profile documented in
[Android ONNX Export](./export-onnx-android.md).

## Install the Export Dependencies

```bash
python -m pip install --upgrade "openmed[onnx]"
```

The `onnx` extra includes PyTorch, Transformers, ONNX, ONNX Runtime, and
ONNX Script. The source checkpoint must expose a token-classification model
with a `logits` output and label metadata in its configuration.

## Export the Artifacts

Run the converter without a profile flag to use the default ONNX/WebGPU
profile:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model dslim/bert-base-NER \
  --output dist/example-onnx
```

Add `--include-transformersjs` when the same output repository also needs a
Transformers.js bundle:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model dslim/bert-base-NER \
  --output dist/example-onnx \
  --include-transformersjs
```

The combined export has this layout:

```text
dist/example-onnx/
  config.json
  id2label.json
  tokenizer.json
  tokenizer_config.json
  model.unoptimized.onnx
  model.onnx
  model.webgpu.onnx
  openmed-onnx.json
  transformersjs/
    config.json
    tokenizer.json
    tokenizer_config.json
    quantize_config.json
    transformersjs-contract.json
    onnx/
      model.onnx
      model_quantized.onnx
```

`model.unoptimized.onnx` is the validation baseline produced when graph
optimization is enabled. It is not a runtime choice in the manifest.
Tokenizer filenames vary by tokenizer implementation, and `id2label.json` is
written when the source configuration contains a label map. OpenMed's
high-level runtime loaders require saved tokenizer assets and label metadata
from either `id2label.json` or `config.json`.

Use `--no-webgpu` to omit `model.webgpu.onnx`, or
`--no-onnx-optimization` to omit the optimization pass and its
optimized-vs-unoptimized parity check. Skipping optimization is useful for
diagnosis, but is not recommended for a release artifact. It does not skip the
fp16 WebGPU parity gate: whenever `model.webgpu.onnx` is produced, the gate
runs against the fp32 graph it was converted from.

## Manifest Format Mapping

`openmed-onnx.json` is the artifact manifest. Its top-level
`"format": "openmed-onnx"` identifies the manifest schema; it is not a runtime
format. The `formats[]` values are deduplicated from `artifacts[]`, whose
entries provide the exact path and precision:

| `formats[]` value | Artifact path | Precision | Intended consumer |
| --- | --- | --- | --- |
| `onnx` | `model.onnx` | fp32 | Python ONNX Runtime or browser fallback |
| `webgpu` | `model.webgpu.onnx` | fp16 | Direct WebGPU-capable ONNX runtime |
| `transformersjs` | `transformersjs/` | int8 bundle | Transformers.js browser pipeline |

For example:

```json
{
  "format": "openmed-onnx",
  "formats": ["onnx", "webgpu", "transformersjs"],
  "artifacts": [
    {"format": "onnx", "path": "model.onnx", "precision": "float32"},
    {
      "format": "webgpu",
      "path": "model.webgpu.onnx",
      "precision": "float16"
    },
    {
      "format": "transformersjs",
      "path": "transformersjs",
      "precision": "int8"
    }
  ]
}
```

When `--publish-to-hub` is used, the converter forwards the same format names
to the published `models.jsonl` row. Consumers should use `artifacts[].path`
from `openmed-onnx.json` to resolve files rather than deriving filenames from
the format name.

## fp16 WebGPU Parity Gate

fp16 overflow and underflow in attention, softmax and layer normalization are a
known failure mode of transformer encoders. Before `model.webgpu.onnx` is
listed in `openmed-onnx.json`, the converter compares it with the fp32 graph it
was converted from (`model.onnx`) on ONNX Runtime CPU. The gate runs whether or
not graph optimization is enabled.

The gate evaluates:

- deterministic synthetic token inputs at every `--validation-lengths` value up
  to the model's maximum sequence length, with one full row and one padded row
  per length; longer lengths are recorded as skipped
- the synthetic `--sample-text` note when a tokenizer is available, and every
  `--validation-text`; the gate fails if validation texts cannot be tokenized
- an activation probe that reads every intermediate tensor the fp16 artifact
  stores in fp16 from the fp32 graph, at the shortest and longest evaluated
  length up to 512

The probe exists because ONNX Runtime CPU runs fp16 nodes that lack a CPU fp16
kernel in fp32. CPU logits can match while the same graph overflows on a WebGPU
device, so CPU parity alone cannot pass an overflowing artifact.

| Check | Default limit |
| --- | --- |
| Logit delta | `abs(fp32 - fp16) <= 0.05 + 0.02 * abs(fp32)` for every logit |
| Non-finite logits | none |
| Token argmax disagreement | at most 0.1% of evaluated tokens |
| Span recall loss on G1/G2 labels | at most 0.003 per label, spans matched exactly |
| fp16-stored activations | magnitude at most 65504 |

Values at or beyond `1e30` in the fp32 graph, such as `finfo(float32).min`
attention-mask fills, are counted as masking sentinels instead of overflow
because the fp16 converter clamps those constants.

### Operators kept in fp32

The converter keeps ONNX Runtime's default fp16 block list in fp32 and adds
these operator types: `LayerNormalization`, `SimplifiedLayerNormalization`,
`SkipLayerNormalization`, `SkipSimplifiedLayerNormalization`, `Softmax`,
`ReduceMean`, `ReduceSum`, `ReduceSumSquare`, `ReduceL2`, `ReduceLogSumExp`,
`Pow` and `Sqrt`. When two kept nodes are adjacent, for example
`Pow -> ReduceMean` inside a decomposed layer norm, the value between them
stays in fp32 instead of round-tripping through fp16. Add operator types or
node names with repeatable flags:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model dslim/bert-base-NER \
  --output dist/example-onnx \
  --webgpu-keep-fp32-op Erf \
  --webgpu-keep-fp32-node /bert/encoder/layer.0/output/Add
```

From Python, pass `WebGpuFp16GateConfig(keep_fp32_ops=..., keep_fp32_nodes=...)`
as `webgpu_gate_config` to `openmed.onnx.convert.convert`; an empty
`keep_fp32_ops` keeps only the ONNX Runtime defaults. The keep-list is stamped
into the artifact's ONNX metadata and copied into the gate record.

### Failure policy

`--webgpu-gate-policy fail` is the default. A failed gate deletes
`model.webgpu.onnx` and exits with `WebGpuFp16GateError` before the manifest is
written or anything is published. `--webgpu-gate-policy omit` deletes the
artifact, writes the manifest without the `webgpu` format and records why:

```json
{
  "formats": ["onnx"],
  "webgpu_fp16_gate": {
    "passed": false,
    "action": "omitted",
    "policy": "omit",
    "reasons": ["fp16_activation_overflow"]
  }
}
```

Reason codes are `fp16_activation_overflow`, `non_finite_logits`,
`logit_tolerance_exceeded`, `token_mismatch_rate_exceeded`,
`critical_recall_delta_exceeded`, `fp16_runtime_error`,
`reference_runtime_error`, `activation_probe_failed`,
`validation_corpus_unavailable` and `no_evaluation_inputs`.

### Gate record

Every manifest that lists a `webgpu` artifact carries a passing
`webgpu_fp16_gate` record. It holds the tolerances, evaluated and skipped
lengths, the corpus example count, aggregate logit, token and span deltas,
per-label span recall deltas, the activation-probe summary, the fp32 keep-list,
the ONNX Runtime version, and SHA-256 digests of both graphs and of the
synthetic inputs. It never stores input text, token ids, logits, file paths or
exception messages.

`write_export_manifest`, the converter's `--publish-to-hub` path and
`python -m openmed.core.hf_publish` all refuse a `webgpu` artifact without a
passing record whose `artifact_sha256` matches the file. Re-run the converter
to publish a bundle exported before the gate existed.

The gate is a CPU proxy. It does not run WebGPU kernels, so browser-side
parity checks such as `certifyWebGpuReference` in the
[WebGPU Token-Classification Runtime](./runtimes/webgpu.md) still apply.
Random token inputs rarely produce entity spans for a trained model, so span
and recall evidence comes mainly from the sample note and validation texts.

## Load with Python ONNX Runtime

Install only the inference dependencies when export tooling is not needed:

```bash
python -m pip install --upgrade "openmed[onnx-runtime]"
```

Load the fp32 graph from a local export directory or a Hugging Face repository:

```python
from openmed import load_onnx_model

model = load_onnx_model("dist/example-onnx", variant="fp32")
entities = model("Patient Alice Nguyen was seen in cardiology.")
```

The loader resolves `variant="fp32"` to `model.onnx` and loads the colocated
configuration, label map, and tokenizer assets. Passing a repository ID uses a
Hugging Face snapshot; inference performs no network requests after the
snapshot is available locally.

## Load in a Browser

### ONNX Runtime Web

Install the OpenMed browser package and ONNX Runtime Web:

```bash
npm install openmed onnxruntime-web
```

Serve the export and the ONNX Runtime Web WASM files from application-local
paths. Select the fp16 graph when WebGPU is available and the fp32 graph for
the WebAssembly fallback:

```typescript
import { loadWebGpuTokenClassificationSession } from "openmed";

const session = await loadWebGpuTokenClassificationSession({
  modelPath: {
    webgpu: "/models/example-onnx/model.webgpu.onnx",
    wasm: "/models/example-onnx/model.onnx",
  },
  assetPath: "/onnxruntime/",
});

const logits = await session.run({
  inputIds,
  attentionMask,
  batchSize: 1,
  sequenceLength: inputIds.length,
});

await session.dispose();
```

The typed session probes a real adapter, chooses WebGPU first, and then selects
threaded or basic WASM with the dedicated fp32 artifact. It rejects remote model
and runtime asset URLs so PHI processing does not silently depend on a CDN. See
the [WebGPU Token-Classification Runtime](./runtimes/webgpu.md) for capability,
benchmark, parity/recall, and offline-inference details. The lower-level
[ONNX Runtime Web Loader](./runtimes/onnxruntime-web.md) remains available for
custom feed and decoding adapters.

### Transformers.js

Install Transformers.js and export with `--include-transformersjs`:

```bash
npm install @huggingface/transformers
```

```javascript
import { pipeline } from "@huggingface/transformers";

const detector = await pipeline(
  "token-classification",
  "/models/example-onnx/transformersjs",
  { device: "webgpu" },
);

const entities = await detector("Patient Alice Nguyen was seen in cardiology.");
```

This path loads models from `transformersjs/onnx/`; it does not select the
root-level `model.webgpu.onnx`. See
[Transformers.js Export](./export-transformersjs.md) for the bundle contract
and standalone exporter.

## Validate Before Publishing

The default converter completes these checks before its built-in Hub publisher
runs:

1. ONNX checker validation for the exported and fp16 graphs.
2. Dynamic sequence-shape and optimized-vs-unoptimized logits parity checks.
3. The configured optimization latency threshold, which defaults to a 20%
   improvement on CPU.
4. The [fp16 WebGPU parity gate](#fp16-webgpu-parity-gate) for
   `model.webgpu.onnx`, including when optimization is disabled.
5. Transformers.js file, label, opset, input, output, and dynamic-axis checks
   when `--include-transformersjs` is enabled.

A failed check exits conversion before upload. Before publishing, also confirm
that the output contains the tokenizer and label assets required by the target
runtime. The default profile warns, rather than fails, when tokenizer assets
cannot be saved.

Run the focused regression tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/onnx/test_onnx_convert.py \
  tests/unit/onnx/test_webgpu_fp16_gate.py \
  tests/unit/onnx/test_transformersjs_export.py \
  -q
```

If publication updates the repository catalog, validate it separately:

```bash
.venv/bin/python scripts/manifest/validate_manifest.py \
  --manifest models.jsonl
```

Publish only after those checks pass:

```bash
HF_WRITE_TOKEN=... \
.venv/bin/python -m openmed.onnx.convert \
  --model dslim/bert-base-NER \
  --output dist/example-onnx \
  --include-transformersjs \
  --publish-to-hub \
  --publish-repo-id OpenMed/example-onnx \
  --publish-manifest models.jsonl
```
