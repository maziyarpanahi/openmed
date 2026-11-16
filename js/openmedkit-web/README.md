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

## Model Catalog

The `openmed/catalog` subpath lists the OpenMed ONNX models this release
knows about, so an application can pick a model by language, size or license
without querying the Hub. The rows are bundled in the package and listing them
makes no network request; the default `openmed` entry does not include them.

```ts
import { getModel, listModels } from "openmed/catalog";

const spanish = listModels({ language: "es", maxParamCount: 100_000_000 });
const entry = getModel("OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android");
```

Each row has the same fields as the Android catalog: `repo_id`, `formats`
(ONNX format tokens only), `tier`, `param_count`, `languages`, `license` and
`reproducibility_hash`. Only permissively licensed models with at least one
ONNX format are included. `listModels()` filters work like the Android
`ModelCatalog.filter()`:

| Filter | Matches |
| --- | --- |
| `format` | any of the row's formats, case-insensitive |
| `tier` | the row's tier, case-insensitive; rows without a tier never match |
| `maxParamCount` | `param_count <= maxParamCount`; rows without a count never match |
| `language` | any of the row's languages, case-insensitive |
| `license` | the row's license, case-insensitive |

Every given filter must match, and results stay sorted by repo id. An unknown
filter field, an empty token or a negative `maxParamCount` throws
`ModelCatalogError` with code `invalid_filter`. `getModel()` matches the repo
id exactly and returns `undefined` for ids outside the catalog.
`createModelCatalog(rows)` and `parseModelCatalogJsonl(text)` build the same
queryable catalog over your own rows, for example an Android catalog JSONL
asset.

The rows are generated from the repository's `models.jsonl` into
`src/catalog.generated.ts`; never edit that file by hand. After a manifest
change, regenerate it from the repository root with
`python scripts/web/build_web_catalog.py`; CI runs the same command with
`--check` and fails when the committed file is stale. `MODEL_CATALOG_DIGEST`
is the SHA-256 of the rows as compact JSONL in Android field order, so it
equals the digest of an Android catalog asset that holds the same rows.

## Node.js Example

A runnable end-to-end script lives at
[`examples/web/node_deidentify.mjs`](../../examples/web/node_deidentify.mjs).
It de-identifies a synthetic note and prints the redacted text and span
labels. From the repository root, install the optional Transformers.js peer
in the web workspace and build the package before running the example:

```bash
npm --prefix js/openmedkit-web install --no-save --package-lock=false @huggingface/transformers
npm --prefix js/openmedkit-web run build
```

```bash
node examples/web/node_deidentify.mjs
```

Running it directly downloads the default model once; the test suite
(`npm test`) exercises the same `run()` entry point offline with a stub
pipeline.

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

### PII Task Pack

`piiPack` picks the PII model by job instead of by repository id, with the
same ordered candidates as the Python, Swift and Android packs:

```ts
import { piiPack } from "openmed";

const pack = await piiPack.load({ wasmPaths: "/ort/" }); // local files only
const result = await pack.deidentify(text);
showNotice(pack.notice); // your UI: show the review notice with the output
```

`piiPack.resolve()` lists the candidates without loading anything. `load()`
tries them in rank order from local files only and fails with a
`TaskPackError` (`local_model_unavailable`) rather than fetch; pass
`allowRemoteModels: true` to let it download a candidate. Pass `pipeline` to
run the pack over your own pipeline.

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

### Shared runtime environment

Transformers.js keeps `allowLocalModels` and `allowRemoteModels` on the module
object shared by every caller, so `loadTokenClassificationPipeline()` and
`loadOnnxModel()` change that environment only while a pipeline is created and
then put the previous state back, including deleting keys the host never set.
Overlapping loads that request the same settings share one window; a load that
requests different settings waits for the open window to close, so a local-only
load never observes `allowRemoteModels: true` during creation. Injected runtimes
are coordinated per environment object, and pipeline-creation errors propagate
unchanged with the environment restored.

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

## Content-Security-Policy profile

A `connect-src` restriction is the browser's own enforcement of the no-upload
invariant, but a policy that is too strict breaks threaded WebAssembly or
workers without an obvious error. This minimal profile is exercised by a synthetic WebAssembly fixture (`tests/browser/csp-profile.spec.ts`):

```http
Content-Security-Policy: default-src 'self'; base-uri 'none'; connect-src 'self' blob:; font-src 'self'; form-action 'none'; img-src 'self' data:; object-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self' blob:
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

| Directive | Why inference needs it |
|---|---|
| `script-src 'self' 'wasm-unsafe-eval'` | Same-origin runtime scripts. `'wasm-unsafe-eval'` is required to compile WebAssembly (ONNX Runtime Web WASM and WebGPU builds). Without it, compilation throws `CompileError` in every engine; Chromium and Firefox also report a `script-src` violation, while WebKit (Safari) dispatches none. There is no silent fallback. |
| `worker-src 'self' blob:` | Same-origin workers and the blob workers ONNX Runtime Web uses for threads. |
| `connect-src 'self' blob:` | Model and `.wasm` fetches from your own origin; any cross-origin request (including to a hosted API) is refused. Add the model host here only if you deliberately load models remotely. |
| `default-src 'self'`, `object-src 'none'`, `base-uri 'none'`, `form-action 'none'` | Deny everything else by default. |
| COOP `same-origin` + COEP `require-corp` | Cross-origin isolation, needed for `SharedArrayBuffer` and threaded WASM. Every subresource must then be same-origin or carry CORP/CORS headers. |

The demo at `docs/demo/web/index.html` uses the same script, connection and
worker directives; it additionally allows inline styles for its UI. COOP/COEP
must be configured as HTTP response headers by the host.

The automated check uses a small WebAssembly addition module on the main thread
and in a same-origin worker. It validates browser policy enforcement, not an
ONNX Runtime model, blob workers, or threaded inference; WGSL shaders are passed to the browser as strings rather than evaluated
as script, so WebGPU is not expected to need an extra directive, but that path
is not covered by the CSP test (it needs a WebGPU-capable runner).
Violations are observable through the `securitypolicyviolation` event; record
only `effectiveDirective` and counts, never `blockedURI`, `sample` or request
payloads, which can contain text or paths. Do not rely on that event alone to
detect a missing `'wasm-unsafe-eval'`: WebKit reports blocked requests such as
`connect-src` but not blocked WebAssembly compilation, so also surface the
`CompileError`.

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

## Resource ownership

Calls to `extractPii` or `deidentify` without a caller-supplied `pipeline` reuse
one package-owned pipeline for each model, loader, runtime identity and loader
option combination. Call `await disposeDefaultPipelines()` when closing the
application or dropping these models. It removes cached entries immediately,
waits for existing operations and pending loads, and releases each owned
pipeline once. A later call acquires a fresh pipeline. Failed package-owned
operations evict and dispose their pipeline; supplied pipelines remain owned
by the caller and are never disposed by these entry points.

`await clearOrtWebSessionCache(cache)` removes cache entries immediately and
releases both settled sessions and sessions still loading. Active native
`run` calls drain before release; subsequent runs reject with a controlled
disposed-session error. Await clearing before application shutdown. Pipelines
created by `loadOrtWebTokenClassificationPipeline` also expose an optional
`dispose()` method. Each returned handle owns a lease on its cached session:
disposing one handle drains its admitted operations without invalidating other
handles. The final handle removes and releases the session exactly once,
including when cache clearing already released it. Disposed handles reject new
calls; explicitly clearing the session cache invalidates every handle for those
sessions after active native runs drain. Release failures use
controlled messages without incorporating the underlying error or its cause.

Cache option keys distinguish scalar types, signed zero, arrays and plain objects,
with sorted object keys; runtime, loader, function and opaque object identities
remain distinct. Symbol option values are unsupported and reject before loading.
