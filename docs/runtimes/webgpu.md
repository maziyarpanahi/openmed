# WebGPU Token-Classification Runtime

OpenMed's browser package exposes a typed token-classification session for
local ONNX artifacts. It probes a real WebGPU adapter, selects the WebGPU
execution provider when usable, and falls back to WebAssembly without changing
the `run(tokens) -> logits` contract.

The runtime consumes either the root ONNX/WebGPU export or the ONNX files inside
a Transformers.js bundle. It does not download models from a registry.

## Install

```bash
npm install openmed onnxruntime-web
```

Serve ONNX Runtime Web's WASM files and the exported model from the same local
application bundle. WebGPU requires HTTPS or `localhost`; threaded WASM also
requires cross-origin isolation.

## Load local model variants

An OpenMed ONNX export normally contains a float16 WebGPU graph and a float32
fallback graph:

```text
/models/openmed-pii/
  model.webgpu.onnx
  model.onnx
/runtime/onnxruntime/
  ort-wasm-simd-threaded.wasm
  ...
```

Pass both local paths so fallback never tries to reinterpret a missing or
remote artifact:

```ts
import { loadWebGpuTokenClassificationSession } from "openmed";

const session = await loadWebGpuTokenClassificationSession({
  modelPath: {
    webgpu: "/models/openmed-pii/model.webgpu.onnx",
    wasm: "/models/openmed-pii/model.onnx",
  },
  assetPath: "/runtime/onnxruntime/",
  labelCount: 17,
  modelName: "OpenMed/local-pii",
  deviceName: "browser",
  tier: "base",
  canonicalTier: "Base",
});

const logits = await session.run({
  inputIds: [101, 2023, 2003, 1037, 3231, 102],
  attentionMask: [1, 1, 1, 1, 1, 1],
  batchSize: 1,
  sequenceLength: 6,
});

console.log(logits.dims); // [1, 6, 17]
await session.dispose();
```

The input names default to `input_ids`, `attention_mask`, and optional
`token_type_ids`. Override `inputNames` and `outputName` for an exported graph
with different names. Inputs are validated and converted to ONNX int64 tensors;
the returned logits are an owned `Float32Array` with shape
`[batch, sequence, labels]`.

Each typed session uses a private ORT cache by default, so `dispose()` can
release its resources safely. Pass an explicit `cache` only to share an ORT
session across wrappers; an individual wrapper will not release a caller-owned
shared cache.

For a Transformers.js bundle, point both variants at its local `onnx/`
directory:

```ts
modelPath: {
  webgpu: "/models/openmed-pii/transformersjs/onnx/model.onnx",
  wasm: "/models/openmed-pii/transformersjs/onnx/model_quantized.onnx",
}
```

## Capability and fallback behavior

`probeOrtWebCapabilities()` distinguishes API exposure from an adapter that can
actually be requested. `WebGpuTokenClassificationSession.capabilityProbe`
records the probe result, while `backend` reports one of `webgpu`,
`wasm-simd-threads`, or `wasm-basic`.

Selection is deterministic:

1. Use WebGPU only when `navigator.gpu.requestAdapter()` returns an adapter.
2. Otherwise use SIMD plus threaded WASM when WebAssembly,
   `SharedArrayBuffer`, cross-origin isolation, and at least two logical cores
   are available.
3. Otherwise use single-threaded WASM.

If WebGPU session creation fails after a successful probe, the loader retries
once with the dedicated WASM model. Set `fallbackOnWebGpuError: false` only when
the application must fail instead. Configure these response headers to enable
threaded WASM:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Direct WGSL classification head

ONNX Runtime Web's WebGPU provider already uses shader-backed matrix
multiplication for a complete token-classification graph. For an encoder that
exports hidden states separately, OpenMed also ships the audited
`web/runtime/kernels/classify.wgsl` head and a small wrapper:

```ts
import { createWebGpuClassificationHead } from "openmed";

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("WebGPU is unavailable");
const device = await adapter.requestDevice();

const head = await createWebGpuClassificationHead(device, {
  weights, // row-major [hiddenSize, labelCount]
  bias,    // [labelCount]
  hiddenSize: 768,
  labelCount: 17,
});

const logits = await head.run(hiddenStates, batchSize, sequenceLength);
head.dispose();
```

The kernel dispatches the sequence and label axes in 8-by-8 workgroups and the
batch on the third dispatch axis. Weights and bias remain resident on the GPU;
per-run hidden-state and logits buffers are destroyed after readback.

## Parity and recall gates

`certifyWebGpuReference()` fails closed unless all of these conditions hold:

- candidate and Python-reference shapes match;
- every logit is finite and the maximum absolute delta is within the fixed
  tolerance;
- decoded BIO/BILOU token spans match within the configured token-offset
  tolerance;
- recall loss stays within `maxRecallDelta`; and
- no label listed in `criticalLabels` is missed.

```ts
import { certifyWebGpuReference } from "openmed";

const evidence = certifyWebGpuReference({
  referenceLogits: pythonReference,
  candidateLogits: logits,
  id2label,
  attentionMask,
  tolerance: 1e-3,
  maxRecallDelta: 0,
  criticalLabels: ["PERSON", "EMAIL", "PHONE", "ID_NUM"],
});
```

Certification evidence contains tensor deltas, token offsets, labels, and
aggregate recall only. It never copies note text or identifier surfaces.

## Per-device benchmark record

Call `benchmark()` before unrelated inference when cold-start evidence is
required. The session measures model load, first inference, warm p50/p95 and
mean latency, and active tokens per second:

```ts
const report = await session.benchmark(tokens, {
  warmupIterations: 1,
  iterations: 10,
});

await saveLocally(JSON.stringify(report, null, 2));
```

The result uses the shared `BenchmarkReport` envelope:

```json
{
  "suite": "webgpu-token-classification-runtime",
  "model_name": "OpenMed/local-pii",
  "device": "webgpu:browser",
  "fixture_count": 1,
  "generated_at": null,
  "metadata": {
    "runtime": "onnxruntime-web",
    "local_files_only": true,
    "tier": "base",
    "canonical_tier": "Base",
    "warmup_iterations": 1,
    "token_count": 6,
    "fallback_used": false
  },
  "metrics": {
    "devices": {
      "browser": {
        "backend": "webgpu",
        "precision": "float32",
        "batch_size": 1,
        "sequence_length": 6,
        "latency": {
          "cold_load_ms": 25.1,
          "first_inference_ms": 7.4,
          "cold_ms": 32.5,
          "warm_ms": 2.2,
          "p50_ms": 2.1,
          "p95_ms": 2.6,
          "count": 10
        },
        "throughput": { "tokens_per_second": 2727.3 }
      }
    }
  }
}
```

No benchmark is transmitted automatically. The optional `benchmarkSink`
callback is local application code, and the built-in record omits model paths,
tokens, note text, browser user agents, and adapter fingerprints.

## Offline and privacy boundary

- Model and WASM asset paths must be relative, root-relative, Windows-local, or
  `file://` paths. HTTP(S), protocol-relative, `data:`, and `blob:` references
  are rejected before session construction.
- A browser may make same-origin requests while constructing the session to
  read those application-local files. Once construction completes, `run()`
  performs no fetch and sends no inference data anywhere.
- The package enables no telemetry, remote model fallback, console logging, or
  raw-PHI benchmark fields.
- Model licensing remains the deployer's responsibility. OpenMed bundles no
  restricted model weights or clinical datasets.

## Validation

The committed fixture is synthetic. Node tests cover capability profiles,
offline paths, typed tensors, fallback, benchmarking, parity, and recall. A
Playwright Chromium gate dispatches the production WGSL kernel on a real
headless WebGPU context and compares it with the Python float32 fixture.

```bash
cd js/openmedkit-web
npm ci
npm test

cd ../../tests/browser/brand
npm ci
npx playwright test webgpu-token-classification.spec.ts --project chromium
```

Training, the demo UI, OCR, and other multimodal workloads remain outside this
runtime.
