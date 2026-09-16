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

## Privacy and Safety

- No telemetry is enabled by default.
- Local model and runtime paths reject remote URL schemes.
- Span records contain hashes and offsets rather than raw identifier text.
- OpenMed is not a medical device and must not autonomously make clinical
  decisions.

OpenMed is Apache-2.0 licensed. See the
[main repository](https://github.com/maziyarpanahi/openmed) for models,
documentation, and the Python, Swift, and Android runtimes.
