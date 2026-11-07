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
`DEFAULT_MODEL_ID`) at the commit pinned by `DEFAULT_MODEL_REVISION`: the root
INT8 artifact of a 33M-parameter clinical PII model, about 70 MB, served
through Transformers.js. Pass `model` to pick any other public
`OpenMed/<model>-onnx-android` repository, or load one yourself:

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

### Model Acquisition and Offline-Only Mode

`DEFAULT_MODEL_REVISION` is the full Hub commit
(`79f7db205869b1be4be23ac4f42aa95bdedc5aee`) published with the default
model's current model card, so a later push to that repository does not change
existing deployments. On first use Transformers.js downloads that revision from
the Hugging Face Hub and caches it. Set `loaderOptions.revision` to request
another commit. Other Hub models load the revision you pass, or `main` if you
pass none, so pin one in production.

Explicit local paths (`/`, `./`, `../`, `~`, `file://` or a Windows drive)
always load with `local_files_only` and `allowRemoteModels: false`, whatever the
loader options say. A path that only looks local but names another host, such
as `//host/...`, is refused.

Offline-only mode makes every OpenMed loader refuse remote model references and
remote runtime asset locations before anything is fetched; absolute, relative
and same-origin paths count as local. Turn it on for the whole package, or per
call with `offlineOnly: true` on `extractPii()`, `deidentify()`,
`loadOnnxModel()` or `loadTokenClassificationPipeline()`. A call cannot opt out
of the package-wide setting. A custom `modelLoader` still receives only local
references, but what it or a supplied `pipeline` fetches is up to your code.

```ts
import { deidentify, setOfflineOnly } from "openmed";

setOfflineOnly(true);

const result = await deidentify(clinicalNote, {
  model: "/models/openmed-pii-clinicale5-small",
  loaderOptions: { wasmPaths: "/assets/onnxruntime/" },
});
```

In offline-only mode:

- Hub IDs and URLs, including the default model, are refused before any loader,
  runtime import or fetch runs. Serve a local copy and pass its path as `model`.
- Transformers.js loads with `allowRemoteModels: false` and
  `local_files_only: true`.
- ONNX Runtime WebAssembly files must be local. In browsers Transformers.js
  points `env.backends.onnx.wasm.wasmPaths` at jsDelivr by default, so pass
  `wasmPaths` (a directory prefix or `{ mjs, wasm }` file paths) or the load is
  refused. The value stays set, because ONNX Runtime reads it once, when its
  WebAssembly backend starts. A remote `env.localModelPath` is refused too.
- Missing or unreadable local assets fail closed.

The ONNX Runtime Web loader and the WebGPU session accept only local model and
runtime paths in every mode. With the package-wide setting on, their load
failures also surface as `OpenMedOfflineError`.

Refusals from `extractPii()`, `deidentify()` and the Transformers.js loaders,
and offline-only load failures from every loader, throw `OpenMedOfflineError`.
Its `code` is `remote_model_refused`, `remote_asset_refused` or
`local_assets_unavailable`, and its message is fixed. It carries no input text,
path, URL or underlying runtime error (there is no `cause`), so it is safe to
log.

To prepare a local copy, download the pinned revision once and copy the
`onnxruntime-web` `dist/*.wasm` and `dist/*.mjs` files next to your app:

```bash
hf download OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android \
  --revision 79f7db205869b1be4be23ac4f42aa95bdedc5aee \
  --local-dir public/models/openmed-pii-clinicale5-small
```

Downloading the default model implicitly on the first call is deprecated. A
future major release will require explicit preparation: a local `model` path or
a `pipeline` you load yourself. Prepare models that way now, and turn on
offline-only mode to catch any download that remains.

### Token Offsets

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
- By default, span hashes use a fresh random key per call. For stable hashes
  across calls, supply a non-empty `hashSecret` from your application's secret
  store. Use a high-entropy key, keep it private, and do not persist it beside
  the span output. Explicit keys preserve deterministic cross-platform hashes.
- OpenMed is not a medical device and must not autonomously make clinical
  decisions.

OpenMed is Apache-2.0 licensed. See the
[main repository](https://github.com/maziyarpanahi/openmed) for models,
documentation, and the Python, Swift, and Android runtimes.
