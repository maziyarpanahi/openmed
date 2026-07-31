# ONNX Runtime Web Loader

OpenMedKit web includes an ONNX Runtime Web loader for local token-classification
artifacts. It selects the strongest available execution path at runtime:

1. WebGPU when `navigator.gpu` is available
2. wasm with SIMD and threads when cross-origin isolation and
   `SharedArrayBuffer` are available
3. single-threaded wasm as the conservative fallback

The threaded wasm path requires the application to serve pages with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. Without those headers, browsers
hide `SharedArrayBuffer`; the loader falls back to single-threaded wasm instead
of reaching for a remote runtime.

```bash
npm install openmed onnxruntime-web
```

```ts
import {
  loadOrtWebTokenClassificationPipeline,
  deidentify,
} from "openmed";

const pipeline = await loadOrtWebTokenClassificationPipeline({
  modelPath: "/models/openmed-pii/onnx/model.onnx",
  assetPath: "/models/openmed-pii/onnxruntime/",
  tokenize: tokenizeClinicalNote,
  decode: decodeTokenClassificationOutputs,
});

const result = await deidentify(
  "Patient Casey Example called 212-555-0198.",
  { pipeline, detector: "ort-web" },
);
```

`modelPath` and `assetPath` must point to local or offline paths such as
`/models/...`, `./models/...`, `../models/...`, or `file://...`. HTTP(S),
protocol-relative, `data:`, and `blob:` paths are rejected so tests and browser
bundles do not silently fetch PHI-processing assets from a CDN.

Session construction is cached by model path, wasm asset path, backend choice,
and session options. Repeated calls reuse the same `InferenceSession`, while
callers that need isolation can pass their own `cache: new Map()`.

For tests, pass a small runtime object through `runtime` and explicit
`capabilities`. That lets Node exercise WebGPU, threaded wasm, and basic wasm
selection without installing browser APIs or downloading ONNX Runtime assets.
