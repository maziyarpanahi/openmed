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
import { deidentify } from "openmed";

const result = await deidentify(
  "Patient Alice Nguyen was seen in cardiology.",
);

console.log(result.deidentifiedText);
console.log(result.spans);
```

Without a `model` or `pipeline` option, `deidentify()` and `extractPii()` load
`OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android` (exported as
`DEFAULT_MODEL_ID`): the root INT8 artifact of a 33M-parameter clinical PII
model, about 70 MB, served through Transformers.js. Pass `model` to pick any
other public `OpenMed/<model>-onnx-android` repository, or load one yourself:

```ts
import { deidentify, loadOnnxModel } from "openmed";

const model = await loadOnnxModel("OpenMed/<model>-onnx-android");
const result = await deidentify(
  "Patient Alice Nguyen was seen in cardiology.",
  { pipeline: model },
);
```

`loadOnnxModel()` selects the root INT8 model by default. Pass
`{ variant: "fp32" }` or `{ variant: "fp16" }` when a different published
variant is appropriate.

Transformers.js token-classification output carries no character offsets.
OpenMed aligns every token back to the source text before decoding spans
(case- and accent-insensitive, WordPiece `##`, SentencePiece `▁`, and
byte-level `Ġ` markers handled), so `result.spans` always carry `start` and
`end` into the original string. `alignTokenOffsets()` is exported for custom
pipelines that need the same alignment. Offsets use JavaScript UTF-16 indices;
decomposed accents remain attached to their source character. Unknown or
unalignable tokens produce a content-free error instead of incomplete
redaction. Custom tokenizers and filtered output should supply exact source
offsets; treat an alignment error as a failed scan, not a PII-free document.

Existing `TokenClassificationEntity` and `TokenClassificationPipeline` outputs
retain required numeric offsets. Use the additive `RawTokenClassificationEntity`,
`RawTokenClassificationPipeline`, and `RawTransformersRuntime` input types for
offset-less runtimes. Model loaders and `alignTokenOffsets()` return aligned
entities, preserving the v2.2 typed-consumer contract.

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

## Inference Workers

Run tokenization and inference in a dedicated worker so long notes do not block
the UI thread. The worker serves a token-classification pipeline; the page
routes calls through a bounded, cancellable client and keeps hashing and
redaction on the host.

```ts
// openmed.worker.ts, bundled as a module worker
import { loadOnnxModel, serveInferenceWorker } from "openmed";

serveInferenceWorker({
  async load(spec) {
    const pipeline = await loadOnnxModel(spec.modelRef, {
      localFilesOnly: spec.localOnly,
    });
    return { pipeline, backend: "wasm", localOnly: spec.localOnly };
  },
});
```

```ts
import { createInferenceWorkerClient, deidentify } from "openmed";

const client = createInferenceWorkerClient({
  createWorker: () =>
    new Worker(new URL("./openmed.worker.ts", import.meta.url), {
      type: "module",
    }),
  maxQueueSize: 8,
});
await client.load({
  modelRef: "/models/openmed/pii",
  allowedBackends: ["webgpu", "wasm"],
});

const controller = new AbortController();
const result = await deidentify(clinicalNote, {
  pipeline: client.pipeline({ signal: controller.signal }),
});
```

- **Bounded admission.** Up to `maxQueueSize` requests (default 16) wait on
  the host and `maxInFlight` (default 1) run in the worker. Overflow rejects
  at once with an `InferenceWorkerError` whose `code` is `queue_full`. The
  worker server keeps its own bounded queue as well.
- **Cancellation.** Aborting the signal drops a queued request before its
  text is sent, or tells the worker, which aborts the `context.signal` passed
  as the pipeline's third argument. The promise rejects with `cancelled`
  immediately. A cancelled request keeps its in-flight slot until the worker
  acknowledges it, the worker posts no result for it, and any late result is
  dropped and counted in `stats().suppressedLateResults`.
- **Model generations.** Each `load()` starts a new generation. Outstanding
  requests from earlier generations reject with `superseded`, and a pipeline
  returned by `client.pipeline()` rejects with `stale_generation` once a later
  `load()` replaces its model.
- **Device loss and restarts.** Return the `lost` promise of the runtime's
  `GPUDevice` as `deviceLost`. On loss the worker releases the runtime and the
  client becomes `invalidated`: calls reject with `device_lost` or
  `plan_invalidated` until `recover()` succeeds. A worker `error`,
  `messageerror` or malformed message makes the client `failed`, and
  `recover()` starts a fresh worker. Call `invalidate()` when the page itself
  sees device loss or replaces the runtime. Recovery reloads with
  `localOnly: true` and must return a plan with the same model reference, the
  same `modelDigest` when one was reported, and an allowed backend; otherwise
  it fails with `plan_incompatible`. A verified recovery keeps the generation,
  so existing pipelines resume. A new load waits for the running request to
  stop, so pass `{ restart: true }` to `load()` or `recover()` to replace a
  worker whose runtime cannot be interrupted.
- **Local-only loads.** `load()` defaults to `localOnly: true`, rejects
  remote URL model references, and requires the worker to report a
  local-only runtime.
- **Transfer and privacy.** Input text crosses as a transferred UTF-16 buffer,
  so offsets match the source exactly. Results come back as transferred
  typed-array columns holding only labels, scores, token indices and offsets;
  token text is dropped after alignment in the worker. Errors carry a fixed
  message and a code from `INFERENCE_WORKER_ERROR_CODES`, never input text or
  an underlying error message. The client cancels worker `error` events so the
  browser does not report their messages to the console, and neither side
  logs, caches or stores input text.
- **Teardown.** `dispose()` rejects outstanding work with `disposed`, asks the
  worker to release its runtime and terminates it.

## Privacy and Safety

- No telemetry is enabled by default.
- Local model and runtime paths reject remote URL schemes.
- Span records contain hashes and offsets rather than raw identifier text.
- By default, span hashes use a fresh random key per call. For stable hashes
  across calls, supply a non-empty `hashSecret` from your application's secret
  store. Use a high-entropy key, keep it private, and do not persist it beside
  the span output. Explicit keys preserve deterministic cross-platform hashes.
- OpenMed is not a medical device and must not autonomously make clinical
  decisions.

OpenMed is Apache-2.0 licensed. See the
[main repository](https://github.com/maziyarpanahi/openmed) for models,
documentation, and the Python, Swift, and Android runtimes.
