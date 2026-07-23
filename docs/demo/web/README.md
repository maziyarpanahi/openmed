# Browser PII inference and load benchmark

This static demo loads a published Transformers.js token-classification bundle,
runs PII detection entirely in the browser, and records two cold-path timings
for both WASM and WebGPU:

- **Model load** measures construction of a backend-specific pipeline, including
  model download or browser-cache access and runtime session initialization.
- **First inference** measures only the first token-classification call after
  that pipeline has loaded.

The default, `onnx-community/multilang-pii-ner-ONNX`, is a public PII model with
the browser bundle layout needed to run the page without credentials. The page
does not upload input text, but it downloads the JavaScript runtime and model
artifacts from their configured hosts. Use synthetic text only.

## Run it locally

No application build is required. Start any static file server from the
repository root, then open the demo:

```bash
.venv/bin/python -m http.server 8000
```

```text
http://localhost:8000/docs/demo/web/
```

Opening `index.html` through `file://` is not supported because browser module
and model requests require an HTTP origin. A basic static server is sufficient;
WASM does not require a server-side inference process.

WebGPU requires a browser and device that expose `navigator.gpu`. When WebGPU is
not available, select WASM. The page reports that WebGPU is unavailable rather
than silently changing the selected backend.

## Target a manifest `repo_id`

The model field is a Hugging Face repository id, not a URL. To use a model from
the canonical [`models.jsonl`](../../../models.jsonl) manifest:

1. Select a token-classification row whose `formats` include
   `transformersjs`.
2. Copy that row's `repo_id`, such as `OpenMed/example-pii-transformersjs`,
   into the model field.
3. Reset the benchmarks and run one backend, or use **Benchmark both**.

The same value can be supplied in the page URL:

```text
http://localhost:8000/docs/demo/web/?repo_id=OpenMed/example-pii-transformersjs
```

The referenced model repository must be browser-accessible and use the bundle
contract produced by `openmed.onnx.transformersjs`:

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
`input_ids` and `attention_mask` and return `logits`. The demo uses the
half-precision `onnx/model_fp16.onnx` artifact for WebGPU and the q8
`onnx/model_quantized.onnx` artifact for WASM through Transformers.js 4.2.0.

Private or gated repositories require the user to arrange browser
authentication separately. Do not put access tokens in `app.js`, the URL, or a
committed file.

## Read the benchmark

Each backend gets its own pipeline so the timing rows do not accidentally share
a runtime session. **Reset benchmarks** disposes those in-memory pipelines but
does not clear the browser's persistent model cache. For a true uncached
download comparison, clear site data or use a fresh browser profile before
loading the page.

WebGPU support and performance vary by browser, operating system, driver, and
model operators. The benchmark is a local diagnostic, not a cross-device
performance claim. Detection output is demonstrative and must not be treated as
proof that a document is fully de-identified.
