import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  CLASSIFY_WGSL_SOURCE,
  WebGpuVerificationError,
  certifyWebGpuReference,
  decodeWebGpuTokenSpans,
  evaluateWebGpuRecallGate,
  loadWebGpuTokenClassificationSession,
  probeOrtWebCapabilities,
  type OrtExecutionProvider,
  type OrtInferenceSession,
  type OrtResults,
  type OrtSessionCreateOptions,
  type OrtTensorLike,
  type OrtWebRuntime,
  type TokenClassificationLogits,
  type WebGpuBenchmarkReport,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const fixturePath = join(
  rootDir,
  "tests",
  "web",
  "fixtures",
  "webgpu_token_classification_golden.json",
);
const shaderPath = join(
  rootDir,
  "web",
  "runtime",
  "kernels",
  "classify.wgsl",
);

const basicWasmProfile = {
  webgpu: false,
  wasm: true,
  wasmSimd: true,
  sharedArrayBuffer: false,
  crossOriginIsolated: false,
  hardwareConcurrency: 8,
};

const webGpuProfile = {
  ...basicWasmProfile,
  webgpu: true,
};

test("capability probe treats API exposure without an adapter as unavailable", async () => {
  const noAdapter = await probeOrtWebCapabilities({
    globalScope: {
      WebAssembly,
      navigator: {
        gpu: { requestAdapter: async () => null },
        hardwareConcurrency: 8,
      },
    },
  });
  assert.equal(noAdapter.profile.webgpu, false);
  assert.equal(noAdapter.adapterAvailable, false);
  assert.match(noAdapter.reason, /no compatible adapter/);

  const failedProbe = await probeOrtWebCapabilities({
    globalScope: {
      WebAssembly,
      navigator: {
        gpu: {
          requestAdapter: async () => {
            throw new Error("synthetic adapter failure");
          },
        },
        hardwareConcurrency: 8,
      },
    },
  });
  assert.equal(failedProbe.profile.webgpu, false);
  assert.equal(failedProbe.adapterAvailable, false);
  assert.equal(failedProbe.reason, "navigator.gpu adapter probing failed");
});

test("capability probe deterministically selects local wasm when WebGPU is disabled", async () => {
  const fixture = await loadFixture();
  const runtime = createRuntime(fixture.reference_logits);
  const session = await loadWebGpuTokenClassificationSession({
    modelPath: {
      webgpu: "/models/synthetic/model.webgpu.onnx",
      wasm: "/models/synthetic/model.onnx",
    },
    assetPath: "/runtime/ort/",
    runtime,
    globalScope: {
      WebAssembly,
      crossOriginIsolated: false,
      navigator: { hardwareConcurrency: 8 },
    },
    labelCount: fixture.classification_head.label_count,
  });

  assert.equal(session.backend, "wasm-basic");
  assert.equal(session.capabilityProbe.adapterAvailable, false);
  assert.equal(session.modelPath, "/models/synthetic/model.onnx");
  assert.deepEqual(runtime.createCalls[0], {
    modelPath: "/models/synthetic/model.onnx",
    providers: ["wasm"],
  });

  const originalFetch = globalThis.fetch;
  let fetchCalled = false;
  globalThis.fetch = (() => {
    fetchCalled = true;
    throw new Error("network access is forbidden during inference");
  }) as typeof fetch;
  try {
    const logits = await session.run(tokenBatch(fixture));
    assert.deepEqual([...logits.data], fixture.reference_logits);
    assert.deepEqual(logits.dims, [1, 4, 3]);
  } finally {
    globalThis.fetch = originalFetch;
    await session.dispose();
  }
  assert.equal(fetchCalled, false);
  assert.equal(runtime.releases(), 1);
  assertTypedFeeds(runtime.runFeeds[0], fixture.tokens.input_ids);
});

test("WebGPU creation failure retries the dedicated local wasm model", async () => {
  const fixture = await loadFixture();
  const runtime = createRuntime(fixture.reference_logits, {
    failWebGpuCreate: true,
  });
  const session = await loadWebGpuTokenClassificationSession({
    modelPath: {
      webgpu: "/models/synthetic/model.webgpu.onnx",
      wasm: "/models/synthetic/model.onnx",
    },
    assetPath: "/runtime/ort/",
    runtime,
    capabilities: webGpuProfile,
    cache: new Map(),
  });

  assert.equal(session.backend, "wasm-basic");
  assert.equal(
    session.fallbackReason,
    "WebGPU session creation failed; wasm fallback selected",
  );
  assert.deepEqual(runtime.createCalls, [
    {
      modelPath: "/models/synthetic/model.webgpu.onnx",
      providers: ["webgpu", "wasm"],
    },
    {
      modelPath: "/models/synthetic/model.onnx",
      providers: ["wasm"],
    },
  ]);
  await session.dispose();
});

test("typed session selects the local WebGPU graph when the probe passes", async () => {
  const fixture = await loadFixture();
  const runtime = createRuntime(fixture.reference_logits);
  const session = await loadWebGpuTokenClassificationSession({
    modelPath: {
      webgpu: "/models/synthetic/model.webgpu.onnx",
      wasm: "/models/synthetic/model.onnx",
    },
    assetPath: "/runtime/ort/",
    runtime,
    capabilities: webGpuProfile,
  });

  const logits = await session.run(tokenBatch(fixture));
  assert.equal(session.backend, "webgpu");
  assert.equal(session.modelPath, "/models/synthetic/model.webgpu.onnx");
  assert.deepEqual(runtime.createCalls[0], {
    modelPath: "/models/synthetic/model.webgpu.onnx",
    providers: ["webgpu", "wasm"],
  });
  assert.deepEqual([...logits.data], fixture.reference_logits);
  await session.dispose();
  assert.equal(runtime.releases(), 1);
});

test("rejects remote model variants before runtime creation", async () => {
  const fixture = await loadFixture();
  const runtime = createRuntime(fixture.reference_logits);
  await assert.rejects(
    () =>
      loadWebGpuTokenClassificationSession({
        modelPath: {
          webgpu: "https://models.example.invalid/model.onnx",
          wasm: "/models/synthetic/model.onnx",
        },
        assetPath: "/runtime/ort/",
        runtime,
        capabilities: webGpuProfile,
      }),
    /local\/offline/,
  );
  assert.equal(runtime.createCalls.length, 0);
});

test("Python reference logits pass fixed-tolerance span and recall gates", async () => {
  const fixture = await loadFixture();
  const reference = fixtureLogits(fixture);
  const candidate: TokenClassificationLogits = {
    data: Float32Array.from(fixture.reference_logits, (value) => value + 1e-7),
    dims: reference.dims,
    outputName: "logits",
  };

  const certification = certifyWebGpuReference({
    referenceLogits: reference,
    candidateLogits: candidate,
    id2label: fixture.id2label,
    attentionMask: fixture.tokens.attention_mask,
    tolerance: fixture.logit_tolerance,
    maxRecallDelta: fixture.max_recall_delta,
    criticalLabels: fixture.critical_labels,
  });

  assert.equal(certification.passed, true);
  assert.equal(certification.recall_gate.recall, 1);
  assert.equal(certification.recall_gate.critical_missed_count, 0);
  assert.deepEqual(
    certification.candidate_token_spans,
    fixture.reference_token_spans,
  );

  const missedCritical = evaluateWebGpuRecallGate(
    certification.reference_token_spans,
    [],
    { maxRecallDelta: 1, criticalLabels: fixture.critical_labels },
  );
  assert.equal(missedCritical.passed, false);
  assert.equal(missedCritical.critical_missed_count, 1);
});

test("parity certification fails closed when decoded spans diverge", async () => {
  const fixture = await loadFixture();
  const candidate = fixtureLogits(fixture);
  candidate.data.set([4, 0, -4], 3);

  assert.throws(
    () =>
      certifyWebGpuReference({
        referenceLogits: fixtureLogits(fixture),
        candidateLogits: candidate,
        id2label: fixture.id2label,
        attentionMask: fixture.tokens.attention_mask,
        tolerance: 10,
        criticalLabels: fixture.critical_labels,
      }),
    (error: unknown) =>
      error instanceof WebGpuVerificationError &&
      /decoded token spans/.test(error.message),
  );
});

test("emits warm and cold timing in the shared per-device benchmark schema", async () => {
  const fixture = await loadFixture();
  const runtime = createRuntime(fixture.reference_logits);
  const emitted: WebGpuBenchmarkReport[] = [];
  const session = await loadWebGpuTokenClassificationSession({
    modelPath: "/models/synthetic/model.onnx",
    assetPath: "/runtime/ort/",
    runtime,
    capabilities: basicWasmProfile,
    cache: new Map(),
    modelName: "OpenMed/synthetic-webgpu-fixture",
    deviceName: "headless-browser",
    tier: "base",
    canonicalTier: "Base",
    generatedAt: "2026-08-18T00:00:00Z",
    clock: sequenceClock([0, 5, 10, 14, 20, 22, 30, 34]),
    benchmarkSink: (report) => emitted.push(report),
  });

  const report = await session.benchmark(tokenBatch(fixture), {
    iterations: 2,
    warmupIterations: 1,
  });
  const metrics = report.metrics.devices["headless-browser"];
  assert.ok(metrics);
  assert.equal(report.suite, "webgpu-token-classification-runtime");
  assert.equal(report.device, "wasm-basic:headless-browser");
  assert.equal(metrics.latency.cold_load_ms, 5);
  assert.equal(metrics.latency.first_inference_ms, 4);
  assert.equal(metrics.latency.cold_ms, 9);
  assert.equal(metrics.latency.warm_ms, 3);
  assert.equal(metrics.latency.p50_ms, 2);
  assert.equal(metrics.latency.p95_ms, 4);
  assert.equal(metrics.throughput.tokens_per_second, 8000 / 6);
  assert.deepEqual(emitted, [report]);
  const serialized = JSON.stringify(report);
  assert.doesNotMatch(serialized, /Synthetic patient/);
  assert.doesNotMatch(serialized, /models\/synthetic/);
  await session.dispose();
});

test("the shipped WGSL file is the exact batched kernel embedded in the runtime", async () => {
  const fileSource = await readFile(shaderPath, "utf8");
  const normalizedFileSource = fileSource.replace(/\r\n/g, "\n");
  assert.equal(normalizedFileSource.trim(), CLASSIFY_WGSL_SOURCE.trim());
  assert.match(fileSource, /@workgroup_size\(8, 8, 1\)/);
  assert.match(fileSource, /batch = id\.z/);
  assert.match(fileSource, /weights\[weight_index\]/);
});

interface WebGpuFixture {
  reference_runtime: string;
  note: string;
  tokens: {
    input_ids: number[];
    attention_mask: number[];
    batch_size: number;
    sequence_length: number;
  };
  classification_head: {
    hidden_size: number;
    label_count: number;
    weights: number[];
    bias: number[];
  };
  hidden_states: number[];
  reference_logits: number[];
  id2label: Record<string, string>;
  reference_token_spans: Array<{
    batch_index: number;
    label: string;
    start_token: number;
    end_token: number;
  }>;
  critical_labels: string[];
  logit_tolerance: number;
  max_recall_delta: number;
}

async function loadFixture(): Promise<WebGpuFixture> {
  return JSON.parse(await readFile(fixturePath, "utf8")) as WebGpuFixture;
}

function fixtureLogits(fixture: WebGpuFixture): TokenClassificationLogits {
  return {
    data: Float32Array.from(fixture.reference_logits),
    dims: [
      fixture.tokens.batch_size,
      fixture.tokens.sequence_length,
      fixture.classification_head.label_count,
    ],
    outputName: "logits",
  };
}

function tokenBatch(fixture: WebGpuFixture) {
  return {
    inputIds: fixture.tokens.input_ids,
    attentionMask: fixture.tokens.attention_mask,
    batchSize: fixture.tokens.batch_size,
    sequenceLength: fixture.tokens.sequence_length,
  };
}

function createRuntime(
  logits: number[],
  options: { failWebGpuCreate?: boolean } = {},
) {
  const createCalls: Array<{
    modelPath: string;
    providers: OrtExecutionProvider[];
  }> = [];
  const runFeeds: Array<Record<string, OrtTensorLike | unknown>> = [];
  let releases = 0;

  class Tensor implements OrtTensorLike {
    disposed = false;

    constructor(
      readonly type: string,
      readonly data: unknown,
      readonly dims: readonly number[],
    ) {}

    dispose() {
      this.disposed = true;
    }
  }

  const runtime: OrtWebRuntime & {
    createCalls: typeof createCalls;
    runFeeds: typeof runFeeds;
    releases: () => number;
  } = {
    env: {},
    Tensor,
    createCalls,
    runFeeds,
    releases: () => releases,
    InferenceSession: {
      create: async (modelPath, sessionOptions: OrtSessionCreateOptions = {}) => {
        const providers = sessionOptions.executionProviders ?? [];
        createCalls.push({ modelPath, providers });
        if (options.failWebGpuCreate && providers[0] === "webgpu") {
          throw new Error("synthetic WebGPU initialization failure");
        }
        const session: OrtInferenceSession = {
          run: (feeds) => {
            runFeeds.push(feeds);
            const input = feeds.input_ids as OrtTensorLike;
            const dims = input.dims ?? [];
            const output: OrtResults = {
              logits: new Tensor("float32", Float32Array.from(logits), [
                dims[0] ?? 1,
                dims[1] ?? 4,
                3,
              ]),
            };
            return output;
          },
          release: () => {
            releases += 1;
          },
        };
        return session;
      },
    },
  };
  return runtime;
}

function assertTypedFeeds(
  feeds: Record<string, OrtTensorLike | unknown> | undefined,
  expectedInputIds: number[],
) {
  assert.ok(feeds);
  const inputIds = feeds.input_ids as OrtTensorLike;
  const attentionMask = feeds.attention_mask as OrtTensorLike;
  assert.equal(inputIds.type, "int64");
  assert.deepEqual(inputIds.dims, [1, 4]);
  assert.deepEqual(
    [...(inputIds.data as BigInt64Array)],
    expectedInputIds.map(BigInt),
  );
  assert.deepEqual([...(attentionMask.data as BigInt64Array)], [1n, 1n, 1n, 1n]);
}

function sequenceClock(values: number[]) {
  const iterator = values[Symbol.iterator]();
  let last = values.at(-1) ?? 0;
  return () => {
    const next = iterator.next();
    if (!next.done) last = next.value;
    return last;
  };
}
