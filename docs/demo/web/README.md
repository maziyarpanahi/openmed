# Maple clinical studio for WebGPU

This static example presents three structured workflows powered by local
Maple-Preview and a persistent chat with a selectable local conversation model:

- direct-identifier detection and reviewable PII redaction;
- clinical entity extraction;
- evidence-linked relation extraction.

The **Ask about this note** composer remains below PII/NER output. Choose Maple
Preview or LiquidAI LFM2.5 2.6B for conversation while keeping the completed
extraction visible. Real generated deltas stream into the assistant turn. Final
answers include exact source evidence and uncertainty; private reasoning is
neither displayed nor stored.

The page is a runtime shell, not a hosted inference client. It makes no network
request on initial load, bundles no model weights, and has no cloud fallback.
The user explicitly supplies same-origin model packs, then loads a model
locally. A restrictive Content Security Policy prevents the demo from
connecting to a different origin.

The **Preview the interface** button uses deterministic synthetic fixtures. It
is useful for exploring and testing the UI, but it is clearly labelled and does
not pretend to run Maple.

## Model and device expectations

The source model is
[`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview), a
custom 20B-parameter / 1B-active reasoning MoE. The portable browser path starts
from the pinned BF16 ONNX export, then losslessly replaces all 96 attention
projection tensors and all 24 MoE expert tensors with the published
[`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)
ternary weights. The result is one coherent `qmoe-2bit-ternary-rowwise` ONNX
graph for ONNX Runtime Web. Codes `0/1/2` in the MLX checkpoint map exactly to
`-alpha/0/+alpha`; the exporter shifts them to ONNX Runtime's two-bit zero point
without requantizing values. Embeddings and the language-model head retain the
audited four-bit base-export representation; routers remain FP32 and norms
remain FP16. The complete local model pack is approximately 5.0 GB. Peak GPU
memory also includes runtime allocations and the selected KV-cache context, so
measure it in every target browser.

See the repository-wide [Maple on-device guide](../../maple-on-device.md) for
the pinned source revisions and the shared MLX/ONNX release gates.

Maple uses custom ternary MoE operators. A stock Transformers.js
`text-generation` pipeline cannot load the original checkpoint by itself. Use
the pinned OpenMed ONNX Runtime Web build described below. Stock ONNX Runtime
1.26 supports two-bit `MatMulNBits`, but its QMoE wrapper only admits four- and
eight-bit experts; OpenMed's narrow patch enables two-bit routing and the correct
four-values-per-byte packing ratio. Do not silently substitute a remote
inference endpoint.

## Run locally

Serve the repository over HTTP. `file://` cannot provide the module and WebGPU
security model this example requires.

```bash
.venv/bin/python -m http.server 8000
```

Open:

```text
http://localhost:8000/docs/demo/web/
```

For a UI-only tour, open:

```text
http://localhost:8000/docs/demo/web/?preview=1
```

The `models/` and `vendor/` directories beside this README are ignored by Git
on purpose. They are local runtime inputs, not source artifacts. Place an
audited browser bundle at paths such as:

```text
docs/demo/web/vendor/ort.webgpu.min.mjs
docs/demo/web/vendor/ort-wasm-simd-threaded.asyncify.mjs
docs/demo/web/vendor/ort-wasm-simd-threaded.asyncify.wasm
docs/demo/web/vendor/transformers.web.min.js
docs/demo/web/maple-tokenizer.mjs
docs/demo/web/models/maple-preview-2bit-webgpu-coherent/maple-bundle.json
docs/demo/web/models/maple-preview-2bit-webgpu-coherent/model.onnx
docs/demo/web/models/maple-preview-2bit-webgpu-coherent/model.onnx.data.000
docs/demo/web/models/maple-preview-2bit-webgpu-coherent/model.onnx.data.001
...
docs/demo/web/models/maple-preview-2bit-webgpu-coherent/model.onnx.data.012
docs/demo/web/models/maple-preview-2bit-webgpu-coherent/tokenizer.json
```

Then enter these values in **Local asset settings**:

```text
Runtime adapter: ./maple-ort-web-adapter.mjs
Model pack:      ./models/maple-preview-2bit-webgpu-coherent/
```

They can also be supplied without loading anything via the query string:

```text
http://localhost:8000/docs/demo/web/?runtime=./maple-ort-web-adapter.mjs&model=./models/maple-preview-2bit-webgpu-coherent/
```

The demo rejects cross-origin URLs, credentials, query strings, and fragments.
Keep Hugging Face access tokens out of paths, source code, browser storage, and
committed files. Mirror or export public assets before serving the demo; the
page will not fetch a model directly from the Hub.

## Optional LFM2.5 conversation model

The conversation-model picker at the bottom can use the official LiquidAI
LFM2.5 2.6B Q4F16 ONNX export. OpenMed pins revision
`66826372fd4fa166f53be0371c9315745c07cace`. Acquire only these files from that
revision:

```bash
hf download LiquidAI/LFM2.5-2.6B-ONNX \
  chat_template.jinja config.json generation_config.json LICENSE \
  tokenizer.json tokenizer_config.json \
  onnx/model_q4f16.onnx onnx/model_q4f16.onnx_data \
  onnx/model_q4f16.onnx_data_1 \
  --revision 66826372fd4fa166f53be0371c9315745c07cace \
  --local-dir docs/demo/web/models/lfm2.5-2.6b-onnx-q4f16
```

The three graph files total exactly 1,534,153,680 bytes. The complete local pack
above is 1,552,083,193 bytes and has this layout:

```text
models/lfm2.5-2.6b-onnx-q4f16/
├── chat_template.jinja
├── config.json
├── generation_config.json
├── LICENSE
├── tokenizer.json
├── tokenizer_config.json
└── onnx/
    ├── model_q4f16.onnx
    ├── model_q4f16.onnx_data
    └── model_q4f16.onnx_data_1
```

Select **LFM2.5 · 2.6B Q4F16**, verify the path, and click **Load LFM2.5
locally**. The equivalent query string is:

```text
http://localhost:8000/docs/demo/web/?chat_model=lfm25&lfm_model=./models/lfm2.5-2.6b-onnx-q4f16/
```

The adapter loads the pack in a dedicated module worker with the vendored
Transformers.js 4.2.0 and ONNX Runtime Web modules. Remote model loading is
disabled: all runtime and model requests must stay on the page's origin. Model
asset caching follows the explicit **Cache model locally** toggle; prompts and
outputs are never persisted. Loading LFM2.5 releases Maple's GPU session to
avoid holding both multi-gigabyte models at once, but preserves the completed
extraction. Reload Maple before running another structured task.

LFM2.5's real token deltas stream from the worker as they are generated. The
page validates the final grounded-answer schema and exact evidence quotes before
presenting them as trusted structure; plain text remains visibly marked for
review. The adapter closes the model template's initial reasoning segment and
rejects `<think>` markers in visible output.

**License warning:** LFM2.5 uses the custom **LFM Open License v1.0**, not a
permissive open-source license. Its commercial-use grant applies only while the
user or Legal Entity is below US$10 million in annual revenue. Redistribution
requires supplying the license, marking modifications, and retaining applicable
copyright, patent, trademark, and attribution notices. Review the pinned
`LICENSE` before use or redistribution; do not mirror or publish this pack as an
OpenMed artifact without an explicit license review.

## Build the exact 2-bit browser stack

Start from the completed pinned four-bit WebGPU export and the verified
`deepgrove/maple-preview-2bit-mlx` snapshot. This command rewrites all 24 expert
tensors, all 96 attention projection tensors, and their quantization
attributes; it refuses unverified source revisions, reserved ternary code `3`,
changed artifact hashes, or a mismatched graph:

```bash
python -m openmed.onnx.maple_export repack-2bit-webgpu \
  /path/to/maple-preview-4bit-onnx-webgpu \
  /path/to/maple-preview-2bit-mlx \
  docs/demo/web/models/maple-preview-2bit-webgpu-coherent

python -m openmed.onnx.maple_export write-web-manifest \
  docs/demo/web/models/maple-preview-2bit-webgpu-coherent
```

Build the pinned ONNX Runtime fork from exact revision
`8c546c37b43caaca1fa25db430dab94b901cf277` (ONNX Runtime v1.26.0). The reviewed
patch is [`scripts/onnxruntime/patches/maple-qmoe-2bit-webgpu-v1.patch`](../../../scripts/onnxruntime/patches/maple-qmoe-2bit-webgpu-v1.patch),
and the wrapper requires its `openmed-qmoe2-webgpu-v1` marker before creating a
session. The build uses ONNX Runtime's native WebGPU execution provider and its
Asyncify WASM support files; it does not use the JSEP build:

```bash
git clone --branch v1.26.0 --depth 1 --recurse-submodules \
  https://github.com/microsoft/onnxruntime.git /tmp/openmed-ort-qmoe2

python scripts/onnxruntime/build_maple_web_runtime.py \
  /tmp/openmed-ort-qmoe2 docs/demo/web/vendor

python scripts/onnxruntime/build_maple_tokenizer_runtime.py \
  docs/demo/web/vendor
```

The tokenizer build pins Transformers.js 4.2.0, disables remote models and
raw-text persistence, loads `tokenizer.json` and `tokenizer_config.json` only
from the selected same-origin pack, and applies Maple's own chat template with
the assistant generation prompt. Model-asset caching is opt-in. Keep
`maple-qmoe2-runtime.mjs` and
`maple-tokenizer.mjs` beside the demo; the generated dependencies remain under
the ignored `vendor/` directory.

## Runtime adapter contract

The same-origin ES module must export this factory:

```javascript
export async function createOpenMedMapleRuntime({
  cache,
  contextTokens,
  device,
  modelUrl,
  networkPolicy,
  onProgress,
  signal,
}) {
  // Construct an audited WebGPU or ONNX Runtime Web implementation here.
  // `modelUrl` is an absolute same-origin URL ending at the local pack.
  // Honor `signal`, and never send prompts or generated text over a request.
  return {
    async *generate(messages, {
      maxNewTokens,
      minP,
      signal,
      temperature,
    }) {
      // Yield { delta: "...", index: 0 }, a string, or cumulative { text }.
    },
    details() {
      return {
        device: "Apple M-series WebGPU",
        weights: "Maple coherent 2-bit attention + QMoE ONNX",
        cache: cache ? "OPFS" : "session only",
      };
    },
    async clearCache() {
      // Remove only this model's adapter-owned persistent entries.
    },
    dispose() {
      // Destroy GPU buffers, sessions, workers, and prompt/KV memory.
    },
  };
}
```

The factory receives `networkPolicy: "same-origin-model-assets-only"`. Treat
that as a required invariant, not a hint. The adapter may fetch runtime and
model files from `modelUrl`, but inference must be local and it must not put
note text in URLs, headers, request bodies, telemetry, exceptions, console
messages, or persistent storage.

Progress events drive the download/cache UI:

```javascript
onProgress({
  phase: "Creating local ONNX Runtime Web session",
  loaded: 0,
  total: bundleBytes,
  detail: "model.onnx",
});
```

An adapter may additionally export
`clearOpenMedMapleCache({ modelUrl })`. The page uses it when the runtime is no
longer resident. Cache deletion must be scoped to Maple; never clear all data
for the origin.

### Checked-in ONNX Runtime Web reference adapter

[`maple-ort-web-adapter.mjs`](maple-ort-web-adapter.mjs) is a source-only
reference implementation of the page contract. It supplies generation and
ephemeral KV-cache plumbing, but deliberately does **not** supply any of these
external inputs:

- `vendor/ort.webgpu.min.mjs` and its same-directory ONNX Runtime Web WASM
  support files;
- `vendor/transformers.web.min.js`; or
- the model directory, graph, external weights, tokenizer data, or
  `maple-bundle.json`.

Acquire and audit those files separately, then serve them from the page's exact
origin. Do not replace either module with a CDN import. The adapter independently
rejects a different scheme, hostname, or port; credentials; query strings;
fragments; encoded paths; directory traversal; and manifest files that resolve
outside `modelUrl`.

The checked-in [`maple-tokenizer.mjs`](maple-tokenizer.mjs) module exports:

```javascript
export async function createOpenMedMapleTokenizer({
  modelUrl,
  networkPolicy,
  signal,
  tokenizerUrl,
}) {
  return {
    // Apply Maple's pinned chat template and include its generation prompt.
    async encodeMessages(messages, { addGenerationPrompt, signal }) {
      return [/* integer token IDs; batch 1, no padding */];
    },
    async decode(tokenIds, { skipSpecialTokens }) {
      return "cumulative decoded text";
    },
    dispose() {},
  };
}
```

Its token IDs and chat template must reproduce the bundle's `tokenizer.json`;
do not invent a padding token. The adapter uses the manifest EOS IDs (currently
`151645`) and never assumes a pad ID.

The required portable graph is one unified `model.onnx`, declared as both
`graphs.prefill_path` and `graphs.decode_path`. Its exact contract is:

- `input_ids`: INT64 `[1, sequence]`;
- `attention_mask`: INT64 `[1, total_sequence]`, all ones for this unpadded
  batch-1 path;
- `past_key_values.{0..23}.{key|value}`: FLOAT16
  `[1, 4, past_sequence, 128]`;
- `logits`: FLOAT16 `[1, sequence, 151936]`; and
- `present.{0..23}.{key|value}`: FLOAT16
  `[1, 4, total_sequence, 128]`.

Prefill passes zero-length caches. Decode passes one new token, the full-length
mask, and every preceding `present` tensor under its matching
`past_key_values` name. Cache outputs are requested in GPU buffers and released
after generation; only final-row logits are sampled on the CPU. The current
graph has no `position_ids` input. The manifest retains its historical name, so
the adapter only supplies sequential positions if a received session explicitly
advertises that input.

Abort signals set the active ONNX run's termination flag. Runtime disposal also
aborts generation, releases the session and tokenizer, and discards all prompt,
output, and KV references. ONNX Runtime's model/HTTP caching remains
implementation- and browser-defined; this adapter does not claim ownership of
a persistent multi-gigabyte cache and therefore does not expose a misleading
global cache-clearing operation. Browser module imports and ONNX session
creation are not interruptible mid-call; when a load is cancelled during either
operation, the adapter releases any resulting session as soon as creation
returns.

The complete 2-bit graph has passed ONNX checking, Maple graph-contract
validation, and exact ternary source parity for every value in all 96 attention
projections and all 24 expert blocks. The 5.0 GB pack contains `model.onnx` and
13 external-data shards named `model.onnx.data.000` through
`model.onnx.data.012`. Automated tests cover the adapter's cached decoder,
streaming, cancellation, manifest, same-origin, and structured-output plumbing.

A development run in actual Chrome 151 used the native WebGPU provider reported
as `apple · metal-3`. Direct streamed PII returned three exact identifier
surfaces in 12.9 seconds. A prompted entity run over “Metformin 500 mg treats
type 2 diabetes. Nausea followed the evening dose.” returned four exact
medication, dosage, condition, and symptom surfaces in 19.1 seconds (125 tokens
at 7.0 tokens/second), after which the page derived offsets locally. Grounded
chat exposed its first answer token in 946 ms and completed 108 tokens in 16.2
seconds at 7.1 tokens/second with two exact evidence quotes and uncertainty.

Those measurements establish a real Chrome load, WebGPU prefill/decode,
streaming, and coherence check on that development machine. They are not a
production performance promise. Peak-memory measurement, the complete privacy
and coherence suite, and equivalent testing on every supported browser/device
remain release gates.

### Wrapping an existing Maple runtime

If a locally bundled runtime already exposes `MapleRuntime.load()` and
`runtime.generate()`, the adapter is intentionally small:

```javascript
import { MapleRuntime } from "./maple-runtime.bundle.js";

export async function createOpenMedMapleRuntime(options) {
  const runtime = await MapleRuntime.load({
    cache: options.cache,
    maxContext: options.contextTokens,
    onProgress: options.onProgress,
    signal: options.signal,
    weightBase: options.modelUrl,
  });

  return {
    generate: (messages, generation) => runtime.generate(messages, generation),
    details: () => runtime.details(),
    clearCache: () => runtime.clearCache?.(),
    dispose: () => runtime.destroy(),
  };
}
```

Confirm that the runtime actually supports a caller-provided local
`weightBase`. A build with hard-coded remote weight or tokenizer URLs is not
compatible with this demo's privacy policy or CSP.

## Task schemas

All task instructions treat the note as untrusted data and ask Maple for JSON:

- PII: exact unmodified `text` plus `type` for every `span`, and `warnings`;
- entities: exact source `text`, `type`, optional normalization, and extraction
  confidence;
- relations: entity IDs plus source/type/target edges and short evidence quotes;
- chat: a concise `answer`, evidence quotes, `uncertainty`, and a safety note.

The model is not asked to calculate PII or entity offsets. The UI exact-matches
each returned source surface, rejects missing or ambiguous surfaces, derives
UTF-16 offsets, and constructs deterministic PII replacements and redacted text
locally. It parses every structure defensively and always writes model content
with `textContent`; it never injects model HTML. If JSON is malformed, the
response is shown as an explicit schema-review state.

Maple's chat template normally starts implicit reasoning. For these browser
tasks, the same-origin tokenizer closes that empty reasoning segment inside the
prompt and the runtime receives `reasoning: false`. Answer JSON can therefore
stream from the first generated token. The adapter withholds any partial
`</think>` suffix and stops if Maple emits that delimiter instead of streaming
a repeated answer. Compatible custom runtimes must honor the same direct-mode
contract. The chat workflow requests concise evidence and uncertainty rather
than private chain-of-thought.

## Privacy and safety properties

- Initial page load uses only repository-owned, same-origin HTML, CSS, fonts,
  and JavaScript.
- Runtime and model paths must resolve to the page's origin.
- CSP limits scripts, workers, and connections to the same origin (plus local
  `blob:` workers).
- Note and question fields use no persistence. The app clears them on explicit
  reset and when the page is discarded.
- The runtime receives note text only in memory after an explicit run.
- Cache APIs are for model artifacts only. Prompts, outputs, and KV state must
  remain ephemeral.
- Exact-match offsets and PII redaction are constructed locally rather than
  trusting model-generated indices or rewritten notes.
- Synthetic examples use reserved `example.test` addresses and fictional
  identifiers.
- Output cannot diagnose, prescribe, autonomously decide care, or certify
  de-identification. Human review is required.

A same-origin module still executes with the page's authority. Review and pin
its source and dependencies before processing any sensitive data. Browser
privacy controls do not turn a research model into a compliant clinical
workflow by themselves.

## Validation

Run the focused structural checks:

```bash
.venv/bin/python -m pytest tests/unit/test_web_demo_layout.py -q
```

Run the adapter's mocked Node contract tests:

```bash
node --test tests/web/test_maple_ort_web_adapter.mjs
```

Build the exact documentation artifact:

```bash
make docs-build
```

Run the focused browser interactions and privacy test:

```bash
npm --prefix tests/browser/brand test -- \
  --project=chromium \
  -g "Maple ORT Web adapter|Maple browser demo|browser demo"
```

The complete cross-browser publication gate is:

```bash
make docs-browser-test
```

The automated runtime fixture is synthetic and same-origin. It exercises model
progress, streamed generation, three structured result views plus persistent
chat, keyboard tabs, cache cleanup, responsive layout, accessibility, and the rule that note text never
crosses a request boundary.
