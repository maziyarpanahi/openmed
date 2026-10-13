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
  type OrtWebCapabilityProfile,
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
    options: { executionProviders?: OrtExecutionProvider[] };
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

test("cache clearing releases settled and pending sessions exactly once", async () => {
  const { clearOrtWebSessionCache } = await import("../../js/openmedkit-web/src/index");
  const cache = new Map();
  let releaseCount = 0;
  let finish!: (value: OrtInferenceSession) => void;
  const pending = new Promise<OrtInferenceSession>(resolve => { finish = resolve; });
  const runtime = { InferenceSession: { create: () => pending } };
  const loading = loadOrtWebSession({ modelPath: "/model.onnx", assetPath: "/ort/", runtime, capabilities: basicWasmProfile, cache });
  const clearing = clearOrtWebSessionCache(cache);
  assert.equal(cache.size, 0);
  finish({ run: () => ({}), release() { releaseCount++; } });
  const loaded = await loading;
  await clearing;
  await loaded.session.release?.();
  assert.equal(releaseCount, 1);
  await assert.rejects(loaded.session.run({}), /session is disposed/);
});

test("cache clearing waits for active execution and disposable pipelines recreate", async () => {
  const { clearOrtWebSessionCache } = await import("../../js/openmedkit-web/src/index");
  const cache = new Map();
  let releaseCount = 0;
  let created = 0;
  let finish!: () => void;
  const running = new Promise<void>(resolve => { finish = resolve; });
  const runtime = { InferenceSession: { create() { created++; return {
    async run() { await running; return {}; },
    release() { releaseCount++; },
  }; } } };
  const options = { modelPath: "/active.onnx", assetPath: "/ort/", runtime, capabilities: basicWasmProfile, cache, tokenize: () => ({}), decode: () => [] };
  const pipeline = await loadOrtWebTokenClassificationPipeline(options);
  const inference = pipeline("synthetic");
  await new Promise(resolve => setImmediate(resolve));
  const clearing = clearOrtWebSessionCache(cache);
  assert.equal(releaseCount, 0);
  finish();
  await Promise.all([inference, clearing]);
  assert.equal(releaseCount, 1);
  await pipeline.dispose?.();
  assert.equal(releaseCount, 1);
  const replacement = await loadOrtWebTokenClassificationPipeline(options);
  assert.equal(created, 2);
  await replacement.dispose?.();
  assert.equal(cache.size, 0);
  assert.equal(releaseCount, 2);
});

test("session release errors are controlled and clear still releases remaining sessions", async () => {
  const { clearOrtWebSessionCache } = await import("../../js/openmedkit-web/src/index");
  const cache = new Map();
  let released = 0;
  const runtime = { InferenceSession: { create(model: string) { return {
    run: () => ({}),
    release() { released++; if (model.includes("bad")) throw new Error("synthetic-private-path-token"); },
  }; } } };
  for (const modelPath of ["/bad.onnx", "/good.onnx"]) {
    await loadOrtWebSession({ modelPath, assetPath: "/ort/", runtime, capabilities: basicWasmProfile, cache });
  }
  await assert.rejects(clearOrtWebSessionCache(cache), error => {
    assert.equal(String(error), "Error: OpenMed session release failed");
    assert.equal((error as Error).cause, undefined);
    return true;
  });
  assert.equal(released, 2);
  assert.equal(cache.size, 0);
});

test("failed pending creations do not prevent clearing another session", async () => {
  const { clearOrtWebSessionCache } = await import("../../js/openmedkit-web/src/index");
  const cache = new Map();
  let released = 0;
  let reject!: (error: Error) => void;
  const pending = new Promise<OrtInferenceSession>((_resolve, fail) => { reject = fail; });
  const runtime = { InferenceSession: { create(model: string) {
    return model.includes("pending") ? pending : { run: () => ({}), release() { released++; } };
  } } };
  await loadOrtWebSession({ modelPath: "/ready.onnx", assetPath: "/ort/", runtime, capabilities: basicWasmProfile, cache });
  const loading = loadOrtWebSession({ modelPath: "/pending.onnx", assetPath: "/ort/", runtime, capabilities: basicWasmProfile, cache });
  const failure = assert.rejects(loading, /synthetic load failure/);
  const clearing = clearOrtWebSessionCache(cache);
  reject(new Error("synthetic load failure"));
  await Promise.all([failure, clearing]);
  assert.equal(released, 1);
  assert.equal(cache.size, 0);
});

test("disposing one pipeline preserves another handle sharing its session", async () => {
  const cache = new Map();
  let releases = 0;
  let creates = 0;
  const runtime = { InferenceSession: { create() { creates++; return {
    run: () => ({}), release() { releases++; },
  }; } } };
  const options = { modelPath: "/shared.onnx", assetPath: "/ort/", runtime, cache, capabilities: basicWasmProfile, tokenize: () => ({}), decode: () => [] };
  const first = await loadOrtWebTokenClassificationPipeline(options);
  const second = await loadOrtWebTokenClassificationPipeline(options);
  await first.dispose?.();
  await assert.rejects(first("synthetic"), /pipeline is disposed/);
  assert.deepEqual(await second("synthetic"), []);
  assert.equal(creates, 1);
  assert.equal(releases, 0);
  await second.dispose?.();
  assert.equal(releases, 1);
  assert.equal(cache.size, 0);
});

test("default teardown cannot invalidate a replacement sharing an active native session", async () => {
  const { extractPii, disposeDefaultPipelines } = await import("../../js/openmedkit-web/src/index");
  const cache = new Map();
  let releases = 0;
  let finish!: () => void;
  const gate = new Promise<void>(resolve => { finish = resolve; });
  let runs = 0;
  const runtime = { InferenceSession: { create() { return {
    async run() { if (++runs === 1) await gate; return {}; },
    release() { releases++; },
  }; } } };
  const modelLoader = async () => loadOrtWebTokenClassificationPipeline({ modelPath: "/generation.onnx", assetPath: "/ort/", runtime, cache, capabilities: basicWasmProfile, tokenize: () => ({}), decode: () => [] });
  const options = { model: "/generation", modelLoader, hashSecret: "synthetic-secret" };
  const original = extractPii("synthetic", options);
  await new Promise(resolve => setImmediate(resolve));
  const clearing = disposeDefaultPipelines();
  await extractPii("synthetic", options);
  finish();
  await Promise.all([original, clearing]);
  await extractPii("synthetic", options);
  assert.equal(releases, 0);
  await disposeDefaultPipelines();
  assert.equal(releases, 1);
});

test("clearing a pending session does not close or remove a new cache generation", async () => {
  const { clearOrtWebSessionCache } = await import("../../js/openmedkit-web/src/index");
  const cache = new Map();
  let complete!: (session: OrtInferenceSession) => void;
  const pending = new Promise<OrtInferenceSession>(resolve => { complete = resolve; });
  let creates = 0;
  let oldReleased = 0;
  let newReleased = 0;
  const runtime = { InferenceSession: { create() {
    return ++creates === 1 ? pending : { run: () => ({}), release() { newReleased++; } };
  } } };
  const options = { modelPath: "/same.onnx", assetPath: "/ort/", runtime, cache, capabilities: basicWasmProfile };
  const oldLoading = loadOrtWebSession(options);
  const clearing = clearOrtWebSessionCache(cache);
  const replacement = await loadOrtWebSession(options);
  complete({ run: () => ({}), release() { oldReleased++; } });
  await Promise.all([oldLoading, clearing]);
  assert.deepEqual(await replacement.session.run({}), {});
  assert.equal(await loadOrtWebSession(options), replacement);
  assert.equal(oldReleased, 1);
  assert.equal(newReleased, 0);
  await clearOrtWebSessionCache(cache);
  assert.equal(newReleased, 1);
});
