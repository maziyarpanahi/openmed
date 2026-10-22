import assert from "node:assert/strict";
import test from "node:test";

import {
  assertOfflineAssetPath,
  deidentify,
  loadOrtWebSession,
  loadOrtWebTokenClassificationPipeline,
  selectOrtWebBackend,
  type OpenMedSpan,
  type OrtExecutionProvider,
  type OrtFeeds,
  type OrtInferenceSession,
  type OrtResults,
  type OrtSessionCreateOptions,
  type OrtWebCapabilityProfile,
  type OrtWebExternalDataOptions,
  type OrtWebExternalDataShard,
  type OrtWebRuntime,
  type TokenClassificationOutput,
} from "../../js/openmedkit-web/src/index";

const threadedWasmProfile: OrtWebCapabilityProfile = {
  webgpu: false,
  wasm: true,
  wasmSimd: true,
  sharedArrayBuffer: true,
  crossOriginIsolated: true,
  hardwareConcurrency: 8,
};

const webGpuProfile: OrtWebCapabilityProfile = {
  ...threadedWasmProfile,
  webgpu: true,
};

const basicWasmProfile: OrtWebCapabilityProfile = {
  webgpu: false,
  wasm: true,
  wasmSimd: true,
  sharedArrayBuffer: false,
  crossOriginIsolated: false,
  hardwareConcurrency: 8,
};

test("selects the highest-capability backend from simulated profiles", () => {
  assert.equal(selectOrtWebBackend(webGpuProfile).backend, "webgpu");
  assert.equal(
    selectOrtWebBackend(threadedWasmProfile).backend,
    "wasm-simd-threads",
  );
  assert.equal(selectOrtWebBackend(basicWasmProfile).backend, "wasm-basic");
});

test("reuses cached sessions and configures wasm from local assets", async () => {
  const cache = new Map();
  const runtime = createRuntime(["wasm"]);

  const first = await loadOrtWebSession({
    modelPath: "/models/openmed/token-classifier/model.onnx",
    assetPath: "/models/openmed/ort/",
    runtime,
    capabilities: threadedWasmProfile,
    cache,
  });
  const second = await loadOrtWebSession({
    modelPath: "/models/openmed/token-classifier/model.onnx",
    assetPath: "/models/openmed/ort/",
    runtime,
    capabilities: threadedWasmProfile,
    cache,
  });

  assert.equal(first.session, second.session);
  assert.equal(runtime.createCalls.length, 1);
  assert.equal(runtime.env?.wasm?.wasmPaths, "/models/openmed/ort/");
  assert.equal(runtime.env?.wasm?.simd, true);
  assert.equal(runtime.env?.wasm?.numThreads, 4);
  assert.deepEqual(runtime.createCalls[0]?.options.executionProviders, ["wasm"]);
});

test("rejects remote model and wasm asset paths", async () => {
  const runtime = createRuntime(["wasm"]);
  await assert.rejects(
    () =>
      loadOrtWebSession({
        modelPath: "https://cdn.example.invalid/model.onnx",
        assetPath: "/models/openmed/ort/",
        runtime,
        capabilities: basicWasmProfile,
        cache: new Map(),
      }),
    /local\/offline/,
  );
  await assert.rejects(
    () =>
      loadOrtWebSession({
        modelPath: "/models/openmed/model.onnx",
        assetPath: "https://cdn.example.invalid/ort/",
        runtime,
        capabilities: basicWasmProfile,
        cache: new Map(),
      }),
    /local\/offline/,
  );
});

test("rejects UNC and remote-host file asset paths", () => {
  for (const path of [
    "\\\\models.example.invalid\\share\\model.onnx",
    "file://models.example.invalid/share/model.onnx",
    "file:////models.example.invalid/share/model.onnx",
  ]) {
    assert.throws(() => assertOfflineAssetPath(path), /local\/offline/);
  }

  assert.equal(
    assertOfflineAssetPath("file:///models/openmed/model.onnx"),
    "file:///models/openmed/model.onnx",
  );
});

test("forwards local external-data shards and keys the cache by shard", async () => {
  const cache = new Map();
  const runtime = createRuntime(["wasm"]);
  const load = (shards: OrtWebExternalDataShard[]) =>
    loadOrtWebSession({
      modelPath: "/models/openmed/model.onnx",
      assetPath: "/models/openmed/ort/",
      runtime,
      capabilities: basicWasmProfile,
      externalData: { shards },
      cache,
    });
  const localShards = [
    externalDataShard(0, "/models/openmed/model.onnx.data.000"),
    externalDataShard(1, "./model.onnx.data.001"),
  ];

  const first = await load(localShards);
  const otherShards = await load([
    localShards[0]!,
    externalDataShard(1, "./other/model.onnx.data.001"),
  ]);
  const repeated = await load(localShards);

  assert.deepEqual(runtime.createCalls[0]?.options.externalData, [
    { path: "model.onnx.data.000", data: "/models/openmed/model.onnx.data.000" },
    { path: "model.onnx.data.001", data: "./model.onnx.data.001" },
  ]);
  assert.notEqual(first.cacheKey, otherShards.cacheKey);
  assert.notEqual(first.session, otherShards.session);
  assert.equal(first.session, repeated.session);
  assert.equal(runtime.createCalls.length, 2);
});

test("rejects remote and query-bearing shards before importing the runtime", async () => {
  let imports = 0;
  for (const location of [
    "https://cdn.example.invalid/model.onnx.data.000",
    "//cdn.example.invalid/model.onnx.data.000",
    "/\\cdn.example.invalid/model.onnx.data.000",
    "/\t/cdn.example.invalid/model.onnx.data.000",
    "ht\ttps://cdn.example.invalid/model.onnx.data.000",
    "https://user:secret@cdn.example.invalid/model.onnx.data.000",
    "file://user:secret@localhost/models/model.onnx.data.000",
    "/models/openmed/model.onnx.data.000?token=secret",
  ]) {
    await assert.rejects(
      () =>
        loadOrtWebSession({
          modelPath: "/models/openmed/model.onnx",
          assetPath: "/models/openmed/ort/",
          runtime: () => {
            imports += 1;
            return createRuntime(["wasm"]);
          },
          capabilities: basicWasmProfile,
          externalData: { shards: [externalDataShard(0, location)] },
          cache: new Map(),
        }),
      valueFreeError(/external data shard 0 location/),
    );
  }
  assert.equal(imports, 0);
});

test("rejects untyped sessionOptions external data with a value-free error", async () => {
  const runtime = createRuntime(["wasm"]);
  await assert.rejects(
    () =>
      loadOrtWebSession({
        modelPath: "/models/openmed/model.onnx",
        assetPath: "/models/openmed/ort/",
        runtime,
        capabilities: basicWasmProfile,
        sessionOptions: {
          externalData: [
            {
              path: "model.onnx.data",
              data: "https://cdn.example.invalid/model.onnx.data",
            },
          ],
        },
        cache: new Map(),
      }),
    valueFreeError(/must use the typed externalData option/),
  );
  assert.equal(runtime.createCalls.length, 0);
});

test("enforces declared shard and total byte limits before session creation", async () => {
  const runtime = createRuntime(["wasm"]);
  const load = (externalData: OrtWebExternalDataOptions) =>
    loadOrtWebSession({
      modelPath: "/models/openmed/model.onnx",
      assetPath: "/models/openmed/ort/",
      runtime,
      capabilities: basicWasmProfile,
      externalData,
      cache: new Map(),
    });
  const shard = (index: number, byteLength: number) => ({
    ...externalDataShard(index, `/models/openmed/model.onnx.data.00${index}`),
    byteLength,
  });

  await assert.rejects(
    () => load({ shards: [shard(0, 2 ** 31)] }),
    valueFreeError(/shard 0 exceeds the per-shard byte limit/),
  );
  await assert.rejects(
    () => load({ shards: [shard(0, 100), shard(1, 101)], maxShardBytes: 100 }),
    valueFreeError(/shard 1 exceeds the per-shard byte limit/),
  );
  await assert.rejects(
    () => load({ shards: [shard(0, 100), shard(1, 51)], maxTotalBytes: 150 }),
    valueFreeError(/exceeds the total byte limit/),
  );
  await assert.rejects(
    () => load({ shards: [shard(0, -1)] }),
    valueFreeError(/byteLength must be a positive integer/),
  );
  assert.equal(runtime.createCalls.length, 0);

  await load({
    shards: [shard(0, 100), shard(1, 50)],
    maxShardBytes: 100,
    maxTotalBytes: 150,
  });
  assert.equal(runtime.createCalls.length, 1);
});

test("does not fetch when loading local offline assets", async () => {
  const runtime = createRuntime(["wasm"]);
  const originalFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = (() => {
    fetchCalled = true;
    throw new Error("network fetch is not allowed in this test");
  }) as typeof fetch;

  try {
    await loadOrtWebSession({
      modelPath: "/models/openmed/model.onnx",
      assetPath: "/models/openmed/ort/",
      runtime,
      capabilities: basicWasmProfile,
      cache: new Map(),
    });
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(fetchCalled, false);
  assert.equal(runtime.createCalls[0]?.modelPath, "/models/openmed/model.onnx");
});

test("synthetic note spans are identical across wasm and WebGPU loaders", async () => {
  const note = "Patient Alice Nguyen emailed alice@example.org.";
  const tokenizer = (text: string): OrtFeeds => ({
    input_ids: { data: new BigInt64Array([101n, 102n]), dims: [1, 2] },
    attention_mask: { data: new BigInt64Array([1n, 1n]), dims: [1, 2] },
    text,
  });
  const decode = ({ text, outputs }: { text: string; outputs: OrtResults }) =>
    decodeSyntheticOutputs(text, outputs);

  const wasmPipeline = await loadOrtWebTokenClassificationPipeline({
    modelPath: "/models/openmed/model.onnx",
    assetPath: "/models/openmed/ort/",
    runtime: createRuntime(["wasm"]),
    capabilities: threadedWasmProfile,
    cache: new Map(),
    tokenize: tokenizer,
    decode,
  });
  const webGpuPipeline = await loadOrtWebTokenClassificationPipeline({
    modelPath: "/models/openmed/model.onnx",
    assetPath: "/models/openmed/ort/",
    runtime: createRuntime(["webgpu", "wasm"]),
    capabilities: webGpuProfile,
    cache: new Map(),
    tokenize: tokenizer,
    decode,
  });

  const wasmResult = await deidentify(note, {
    pipeline: wasmPipeline,
    docId: "synthetic-note",
    hashSecret: "ort-web-test",
    detector: "ort-web",
  });
  const webGpuResult = await deidentify(note, {
    pipeline: webGpuPipeline,
    docId: "synthetic-note",
    hashSecret: "ort-web-test",
    detector: "ort-web",
  });

  assert.equal(wasmResult.deidentifiedText, webGpuResult.deidentifiedText);
  assertSpansClose(wasmResult.spans, webGpuResult.spans);
});

function createRuntime(expectedProviders: OrtExecutionProvider[]) {
  const createCalls: {
    modelPath: string;
    options: OrtSessionCreateOptions;
  }[] = [];
  const runtime: OrtWebRuntime & { createCalls: typeof createCalls } = {
    env: {},
    createCalls,
    InferenceSession: {
      create: (modelPath, options = {}) => {
        createCalls.push({ modelPath, options });
        assert.deepEqual(options.executionProviders, expectedProviders);
        return createSyntheticSession();
      },
    },
  };
  return runtime;
}

function externalDataShard(
  index: number,
  location: string,
): OrtWebExternalDataShard {
  return { path: `model.onnx.data.00${index}`, location, byteLength: 1024 };
}

function valueFreeError(pattern: RegExp) {
  return (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, pattern);
    assert.doesNotMatch(error.message, /example|secret|token|model\.onnx|\/models/);
    return true;
  };
}

function createSyntheticSession(): OrtInferenceSession {
  return {
    run: () => ({
      logits: {
        data: [0.99, 0.97, 0.98],
        dims: [1, 3, 1],
      },
    }),
  };
}

function decodeSyntheticOutputs(
  text: string,
  outputs: OrtResults,
): TokenClassificationOutput {
  const logits = outputs.logits as { data: number[] };
  const aliceStart = text.indexOf("Alice");
  const nguyenStart = text.indexOf("Nguyen");
  const emailStart = text.indexOf("alice@example.org");
  return [
    {
      entity: "B-PERSON",
      score: logits.data[0] ?? 0,
      word: "Alice",
      start: aliceStart,
      end: aliceStart + "Alice".length,
      index: 0,
    },
    {
      entity: "E-PERSON",
      score: logits.data[1] ?? 0,
      word: "Nguyen",
      start: nguyenStart,
      end: nguyenStart + "Nguyen".length,
      index: 1,
    },
    {
      entity: "S-EMAIL",
      score: logits.data[2] ?? 0,
      word: "alice@example.org",
      start: emailStart,
      end: emailStart + "alice@example.org".length,
      index: 2,
    },
  ];
}

function assertSpansClose(actual: OpenMedSpan[], expected: OpenMedSpan[]) {
  assert.equal(actual.length, expected.length);
  for (const [index, actualSpan] of actual.entries()) {
    const expectedSpan = expected[index];
    assert.ok(expectedSpan, `missing expected span at ${index}`);
    const { score: actualScore, ...actualRest } = actualSpan;
    const { score: expectedScore, ...expectedRest } = expectedSpan;
    assert.deepEqual(actualRest, expectedRest);
    assert.ok(actualScore !== null);
    assert.ok(expectedScore !== null);
    assert.ok(Math.abs(actualScore - expectedScore) <= 1e-6);
  }
}
