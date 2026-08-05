# Maple clinical studio for WebGPU

This static example presents four workflows powered by a single local
Maple-Preview causal model:

- direct-identifier detection and reviewable PII redaction;
- clinical entity extraction;
- evidence-linked relation extraction; and
- note-grounded questions with an answer, quoted evidence, and uncertainty.

The page is a runtime shell, not a hosted inference client. It makes no network
request on initial load, bundles no model weights, and has no cloud fallback.
The user explicitly supplies a same-origin runtime module and model pack, then
clicks **Load Maple locally**. A restrictive Content Security Policy prevents
the demo from connecting to a different origin.

The **Preview the interface** button uses deterministic synthetic fixtures. It
is useful for exploring and testing the UI, but it is clearly labelled and does
not pretend to run Maple.

## Model and device expectations

The source model is
[`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview), a
custom 20B-parameter / 1B-active reasoning MoE. The portable browser path starts
from that pinned BF16 source and exports one
`qmoe-4bit-blockwise-128` ONNX graph for ONNX Runtime Web. The
[`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)
checkpoint is an MLX reference for Python and Apple-platform work; it is not an
ONNX graph and this browser adapter cannot load it. Bundle size and peak GPU
memory depend on the actual 4-bit export and context length. Measure both on the
target browser rather than applying the 5.31 GB MLX figure to ONNX.

See the repository-wide [Maple on-device guide](../../maple-on-device.md) for
the pinned source revisions and the shared MLX/ONNX release gates.

Maple uses custom ternary MoE operators. A stock Transformers.js
`text-generation` pipeline cannot load the original checkpoint by itself. Use
an audited Maple WebGPU runtime or a compatible exported ONNX Runtime Web
implementation behind the adapter contract below. Do not silently substitute a
remote inference endpoint.

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
docs/demo/web/vendor/ort-wasm-simd-threaded.jsep.wasm
docs/demo/web/vendor/maple-tokenizer.mjs
docs/demo/web/models/maple-preview-webgpu/maple-bundle.json
docs/demo/web/models/maple-preview-webgpu/model.onnx
docs/demo/web/models/maple-preview-webgpu/model.onnx.data
docs/demo/web/models/maple-preview-webgpu/tokenizer.json
```

Then enter these values in **Local asset settings**:

```text
Runtime adapter: ./maple-ort-web-adapter.mjs
Model pack:      ./models/maple-preview-webgpu/
```

They can also be supplied without loading anything via the query string:

```text
http://localhost:8000/docs/demo/web/?runtime=./maple-ort-web-adapter.mjs&model=./models/maple-preview-webgpu/
```

The demo rejects cross-origin URLs, credentials, query strings, and fragments.
Keep Hugging Face access tokens out of paths, source code, browser storage, and
committed files. Mirror or export public assets before serving the demo; the
page will not fetch a model directly from the Hub.

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
        weights: "Maple 4-bit block-128 QMoE ONNX",
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
- `vendor/maple-tokenizer.mjs`; or
- the model directory, graph, external weights, tokenizer data, or
  `maple-bundle.json`.

Acquire and audit those files separately, then serve them from the page's exact
origin. Do not replace either module with a CDN import. The adapter independently
rejects a different scheme, hostname, or port; credentials; query strings;
fragments; encoded paths; directory traversal; and manifest files that resolve
outside `modelUrl`.

The tokenizer module must export:

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

This is an integration reference, not evidence of working Maple inference in a
real browser. Automated browser tests use mocked ONNX Runtime and tokenizer
modules. The complete pinned 4-bit FLOAT16 graph has passed ONNX validation and
raw ONNX Runtime 1.25.1 CPU prefill plus cached decode with finite logits; the
smaller fused-QMoE CPU probe passes too. Maple execution through the WebGPU
provider, source parity, target-browser memory use, and end-to-end clinical
quality remain unvalidated release gates.

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

Maple's chat template starts implicit reasoning. Generated text stays hidden
until a complete `</think>` delimiter arrives, after which only the final
payload is used. A complete direct JSON object is also accepted for adapters or
synthetic fixtures configured without reasoning. Partial prose is never exposed
as a rationale. The chat workflow requests concise evidence and uncertainty
rather than private chain-of-thought.

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
progress, streamed generation, all four result views, keyboard tabs, cache
cleanup, responsive layout, accessibility, and the rule that note text never
crosses a request boundary.
