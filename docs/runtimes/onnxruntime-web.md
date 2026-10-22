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
session options, and external-data shard paths and locations. Repeated calls
reuse the same `InferenceSession`, while callers that need isolation can pass
their own `cache: new Map()`.

For tests, pass a small runtime object through `runtime` and explicit
`capabilities`. That lets Node exercise WebGPU, threaded wasm, and basic wasm
selection without installing browser APIs or downloading ONNX Runtime assets.

Applications that need typed token tensors, a `run(tokens) -> logits` contract,
separate WebGPU and WASM model artifacts, parity/recall certification, and
per-device timing should use the production
[WebGPU Token-Classification Runtime](webgpu.md).

## Sharded external data

Browsers cap a single `ArrayBuffer` allocation near 2 GB, so larger graphs
store their weights in several external-data shards. Declare each shard through
the typed `externalData` option:

```ts
const session = await loadOrtWebSession({
  modelPath: "/models/openmed-large/model.onnx",
  assetPath: "/models/openmed-large/onnxruntime/",
  externalData: {
    shards: [
      {
        path: "model.onnx.data.000",
        location: "/models/openmed-large/model.onnx.data.000",
        byteLength: 402_653_184,
      },
      {
        path: "model.onnx.data.001",
        location: "/models/openmed-large/model.onnx.data.001",
        byteLength: 398_458_880,
      },
    ],
    maxTotalBytes: 6_000_000_000,
  },
});
```

`path` is the file name the ONNX graph references and `location` is where ONNX
Runtime Web loads the shard from. Both follow the offline path rule above and
must not include a query string. The loader checks every declared
`byteLength` against `maxShardBytes` (2 GiB minus one byte by default)
and the optional `maxTotalBytes` before it imports the runtime, so an oversized
shard fails before any allocation. The limits apply to the declared sizes;
serve shard files whose sizes match their declarations.

`sessionOptions.externalData` is rejected so shards cannot bypass these checks.
Validation errors contain only the shard index and the failed rule, never the
path, location, or file content.
