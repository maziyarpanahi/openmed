# OpenMed for Web

Run medical named-entity recognition and de-identification in browsers or
Node.js with the same OpenMed models used by Python, Swift, and Android. Text is
processed locally; loading a Hugging Face model may download its artifacts once,
but inference does not send clinical text to a hosted API.

## Install

For OpenMed models loaded through Transformers.js:

```bash
npm install openmed @huggingface/transformers
```

For direct ONNX Runtime Web sessions:

```bash
npm install openmed onnxruntime-web
```

## Run an OpenMed ONNX Model

```ts
import { deidentify, loadOnnxModel } from "openmed";

const model = await loadOnnxModel("OpenMed/<model>-onnx-android");
const result = await deidentify(
  "Patient Alice Nguyen was seen in cardiology.",
  { pipeline: model },
);

console.log(result.deidentifiedText);
console.log(result.spans);
```

`loadOnnxModel()` selects the root INT8 model by default. Pass
`{ variant: "fp32" }` or `{ variant: "fp16" }` when a different published
variant is appropriate.

## Local Browser Runtime

The lower-level ONNX Runtime Web loader keeps model and runtime assets on local
paths and selects the strongest execution path available:

1. WebGPU
2. WebAssembly with SIMD and threads
3. Single-threaded WebAssembly

```ts
import {
  deidentify,
  loadOrtWebTokenClassificationPipeline,
} from "openmed";

const pipeline = await loadOrtWebTokenClassificationPipeline({
  modelPath: "/models/openmed/model.onnx",
  assetPath: "/models/openmed/onnxruntime/",
  tokenize: tokenizeClinicalNote,
  decode: decodeTokenClassificationOutputs,
});

const result = await deidentify(clinicalNote, {
  pipeline,
  detector: "ort-web",
});
```

Threaded WebAssembly requires cross-origin isolation. Serve the application with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`; otherwise OpenMed falls back to a
single WebAssembly thread.

For a typed `run(tokens) -> logits` contract with a real adapter probe and
separate WebGPU/WASM artifacts:

```ts
import { loadWebGpuTokenClassificationSession } from "openmed";

const session = await loadWebGpuTokenClassificationSession({
  modelPath: {
    webgpu: "/models/openmed/model.webgpu.onnx",
    wasm: "/models/openmed/model.onnx",
  },
  assetPath: "/models/openmed/onnxruntime/",
});

const logits = await session.run({
  inputIds,
  attentionMask,
  batchSize: 1,
  sequenceLength: inputIds.length,
});

await session.dispose();
```

The session also emits local-only warm/cold per-device benchmark records and
provides fail-closed Python-reference parity and critical-label recall gates.

## Privacy and Safety

- No telemetry is enabled by default.
- Local model and runtime paths reject remote URL schemes.
- Span records contain hashes and offsets rather than raw identifier text.
- OpenMed is not a medical device and must not autonomously make clinical
  decisions.

OpenMed is Apache-2.0 licensed. See the
[main repository](https://github.com/maziyarpanahi/openmed) for models,
documentation, and the Python, Swift, and Android runtimes.
