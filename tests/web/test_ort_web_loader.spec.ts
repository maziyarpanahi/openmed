import assert from "node:assert/strict";
import test from "node:test";

import {
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
