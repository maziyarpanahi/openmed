# ONNX and WebGPU Export

The default OpenMed ONNX profile converts a Hugging Face
token-classification checkpoint or local model directory into:

- an fp32 ONNX graph for ONNX Runtime
- an fp16 ONNX graph intended for WebGPU
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
`--no-onnx-optimization` to omit the optimization and parity pass. Skipping
those checks is useful for diagnosis, but is not recommended for a release
artifact.

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
4. Transformers.js file, label, opset, input, output, and dynamic-axis checks
   when `--include-transformersjs` is enabled.

A failed check exits conversion before upload. Before publishing, also confirm
that the output contains the tokenizer and label assets required by the target
runtime. The default profile warns, rather than fails, when tokenizer assets
cannot be saved.

Run the focused regression tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/onnx/test_onnx_convert.py \
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
