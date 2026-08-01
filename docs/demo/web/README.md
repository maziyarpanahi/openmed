# Browser PII inference and load benchmark

This static page is an offline-safe configuration shell for comparing WASM
and WebGPU PII inference with synthetic text. It deliberately does **not**
import Transformers.js from a CDN or select a remote model by default. Initial
page load makes no runtime or model request.

Before inference can run, the user must provide:

- a same-origin ES module that exports
  `createOpenMedPipeline(options)`; and
- a same-origin URL for a locally published Transformers.js model bundle.

Missing, cross-origin, or invalid configuration fails before inference. The
page never uploads input text. A supplied runtime adapter must also keep remote
model access disabled.

The timing panel records:

- **Model load**, including local runtime initialization and construction of a
  backend-specific pipeline; and
- **First inference**, measuring only the first token-classification call after
  that pipeline has loaded.

## Run the shell locally

No application server is required. Start a static file server from the
repository root, then open the demo:

```bash
.venv/bin/python -m http.server 8000
```

```text
http://localhost:8000/docs/demo/web/
```

Opening `index.html` through `file://` is unsupported because browser module
requests require an HTTP origin. WebGPU also requires a browser and device that
expose `navigator.gpu`; the page reports an unavailable backend instead of
silently changing it.

## Supply a local runtime adapter

Place an audited, repository-owned or locally generated runtime and adapter
under the same static origin. The adapter contract is intentionally small:

```javascript
import { env, pipeline } from "./transformers.bundle.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;

export async function createOpenMedPipeline({
  backend,
  dtype,
  modelUrl,
  task,
}) {
  return pipeline(task, modelUrl, {
    device: backend,
    dtype,
  });
}
```

The example import is relative; `transformers.bundle.js` must also be local.
The production shell does not ship this optional runtime bundle. Do not replace
the relative import with a CDN URL.

Enter paths such as these in the page:

```text
Runtime adapter: ./vendor/openmed-transformers-runtime.js
Model bundle:    ./models/openmed-pii/
```

The same values can be provided in the query string:

```text
http://localhost:8000/docs/demo/web/?runtime=./vendor/openmed-transformers-runtime.js&model=./models/openmed-pii/
```

Both URLs are resolved against the page and rejected unless they share its
origin.

## Prepare a local model bundle

Use `models.jsonl` to select a token-classification entry whose `formats`
include `transformersjs`, then export or mirror its approved artifacts under
the local static origin. A manifest `repo_id` is evidence for selecting the
source model; it is not entered directly into this offline-safe page.

The model directory follows the contract produced by
`openmed.onnx.transformersjs`:

```text
config.json
tokenizer.json
tokenizer_config.json
quantize_config.json
transformersjs-contract.json
onnx/model.onnx
onnx/model_quantized.onnx
```

`config.json` must contain `id2label`, and the ONNX graph must accept
`input_ids` and `attention_mask` and return `logits`. The adapter should map
the WebGPU request to a compatible half-precision artifact and the WASM request
to the q8 `onnx/model_quantized.onnx` artifact. Keep access tokens out of the
adapter, URL, browser storage, and committed files.

## Read the benchmark

Each backend gets its own pipeline so timing rows do not accidentally share a
runtime session. **Reset benchmarks** disposes those in-memory pipelines but
does not clear the browser's persistent model cache. For a true uncached
comparison, clear site data or use a fresh browser profile.

WebGPU support and performance vary by browser, operating system, driver, and
model operators. The benchmark is a local diagnostic, not a cross-device
performance claim. Detection output is demonstrative and must not be treated
as proof that a document is fully de-identified.
