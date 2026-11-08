// Offline checks for the packed `openmed` npm tarball, installed outside the
// source tree. Failures name fixture case ids, never fixture text.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import * as openmed from "openmed";

const consumerRoot = new URL("../", import.meta.url);
const contract = JSON.parse(
  readFileSync(new URL("consumer_contract.json", consumerRoot), "utf8"),
);
// The npm package has no policy layer and canonicalizes into the PII label
// taxonomy, so it redacts every span and maps clinical labels to OTHER. The
// contract records that divergence instead of hiding it.
const divergence = contract.kit_divergences.npm_web;
const hipaaCase = contract.redaction_cases.find(
  (redactionCase) => redactionCase.id === divergence.redaction_case,
);
const webExpectedRedacted = divergence.expected_redacted;
const basicWasmProfile = {
  webgpu: false,
  wasm: true,
  wasmSimd: true,
  sharedArrayBuffer: false,
  crossOriginIsolated: false,
  hardwareConcurrency: 4,
};

const originalFetch = globalThis.fetch;
let fetchCalls = 0;
globalThis.fetch = () => {
  fetchCalls += 1;
  throw new Error("network is disabled for the installed consumer");
};

function fixtureTokens(redactionCase) {
  return redactionCase.entities.map((entity, index) => ({
    entity: `B-${entity.label}`,
    word: redactionCase.text.slice(entity.utf16_start, entity.utf16_end),
    start: entity.utf16_start,
    end: entity.utf16_end,
    score: entity.confidence,
    index,
  }));
}

test("resolves openmed from the consumer install, not the source tree", () => {
  const installUrl = new URL("node_modules/openmed/", consumerRoot);
  const resolved = fileURLToPath(import.meta.resolve("openmed"));
  assert.ok(resolved.startsWith(fileURLToPath(installUrl)), "resolved_outside_install");
  assert.ok(resolved.endsWith("index.js"), "esm_entry");
  for (const file of [
    "package.json",
    "LICENSE",
    "dist/index.js",
    "dist/index.cjs",
    "dist/index.d.ts",
    "dist/index.d.cts",
  ]) {
    assert.ok(existsSync(new URL(file, installUrl)), `missing:${file}`);
  }
});

test("installed public surface matches the committed snapshot", (t) => {
  const snapshotUrl = new URL("public-api.json", consumerRoot);
  if (!existsSync(snapshotUrl)) {
    t.skip("public API snapshot was not staged");
    return;
  }
  const snapshot = JSON.parse(readFileSync(snapshotUrl, "utf8"));
  assert.deepEqual(Object.keys(openmed).sort(), snapshot.exports);
  const installed = JSON.parse(
    readFileSync(new URL("node_modules/openmed/package.json", consumerRoot), "utf8"),
  );
  assert.deepEqual(installed.exports, snapshot.packageExports);
});

test("deidentify redacts a synthetic session with stable keyed hashes", async () => {
  const pipeline = () => fixtureTokens(hipaaCase);
  const options = { pipeline, hashSecret: "fixture-secret", docId: "consumer" };
  const first = await openmed.deidentify(hipaaCase.text, options);
  const second = await openmed.deidentify(hipaaCase.text, options);

  assert.equal(first.deidentifiedText, webExpectedRedacted, hipaaCase.id);
  assert.notEqual(webExpectedRedacted, hipaaCase.expected_redacted, "divergence_resolved");
  assert.deepEqual(
    first.spans.map((span) => [span.start, span.end, span.entity_type]),
    hipaaCase.entities.map((entity) => [entity.utf16_start, entity.utf16_end, entity.label]),
  );
  for (const span of first.spans) {
    assert.equal(span.schema_version, openmed.OPENMED_SPAN_SCHEMA_VERSION);
    assert.match(span.text_hash, /^hmac-sha256:[0-9a-f]{64}$/);
    assert.equal(span.action, "redact");
    const surface = hipaaCase.text.slice(span.start, span.end);
    assert.ok(!JSON.stringify(span).includes(surface), "span_leaks_surface");
  }
  assert.deepEqual(
    second.spans.map((span) => span.text_hash),
    first.spans.map((span) => span.text_hash),
  );
});

test("UTF-16 span offsets redact every contract case", () => {
  for (const offsetCase of contract.offset_cases) {
    assert.equal(offsetCase.text.length, offsetCase.utf16_length, `${offsetCase.id}:utf16_length`);
    const redacted = openmed.spansToRedactedText(offsetCase.text, [
      {
        start: offsetCase.utf16_start,
        end: offsetCase.utf16_end,
        replacement: offsetCase.replacement,
        canonical_label: "OTHER",
      },
    ]);
    assert.equal(redacted, offsetCase.expected_redacted, `${offsetCase.id}:redaction`);
  }
});

test("ORT Web loader runs from local assets and refuses remote ones", async () => {
  const createCalls = [];
  const runtime = {
    env: {},
    InferenceSession: {
      create: async (modelPath, options) => {
        createCalls.push({ modelPath, providers: options?.executionProviders });
        return { run: async () => ({ logits: { data: new Float32Array(0), dims: [0] } }) };
      },
    },
  };
  const pipeline = await openmed.loadOrtWebTokenClassificationPipeline({
    modelPath: "/models/installed-consumer/model.onnx",
    assetPath: "/models/installed-consumer/ort",
    runtime,
    capabilities: basicWasmProfile,
    cache: new Map(),
    tokenize: (text) => ({ text }),
    decode: ({ text }) =>
      fixtureTokens({ ...hipaaCase, text }).map(({ entity, word, start, end, score }) => ({
        entity,
        word,
        start,
        end,
        score,
      })),
  });
  const result = await openmed.deidentify(hipaaCase.text, {
    pipeline,
    hashSecret: "fixture-secret",
  });

  assert.equal(result.deidentifiedText, webExpectedRedacted);
  assert.equal(runtime.env.wasm.wasmPaths, "/models/installed-consumer/ort/");
  assert.deepEqual(createCalls.map((call) => call.providers), [["wasm"]]);
  await assert.rejects(
    openmed.loadOrtWebSession({
      modelPath: "https://models.invalid/installed-consumer/model.onnx",
      assetPath: "/models/installed-consumer/ort/",
      runtime,
      capabilities: basicWasmProfile,
      cache: new Map(),
    }),
    /local\/offline/,
  );
  assert.throws(
    () => openmed.assertOfflineAssetPath("https://models.invalid/ort/"),
    /local\/offline/,
  );
  assert.equal(createCalls.length, 1);
});

test("local model references load with local files only", async () => {
  const calls = [];
  const runtime = {
    env: {},
    pipeline: async (task, model, options) => {
      calls.push({ task, options: { ...options }, allowRemote: runtime.env.allowRemoteModels });
      return async () => fixtureTokens(hipaaCase);
    },
  };
  const pipeline = await openmed.loadTokenClassificationPipeline(
    "./models/installed-consumer",
    { runtime },
  );
  const spans = await openmed.extractPii(hipaaCase.text, {
    pipeline,
    hashSecret: "fixture-secret",
  });

  assert.equal(calls[0].task, "token-classification");
  assert.equal(calls[0].options.local_files_only, true);
  assert.equal(calls[0].allowRemote, false);
  assert.equal(spans.length, hipaaCase.entities.length);
  assert.equal(openmed.isLocalModelReference("./models/installed-consumer"), true);
  assert.equal(openmed.isLocalModelReference(openmed.DEFAULT_MODEL_ID), false);
});

test("installed package made no network requests", () => {
  globalThis.fetch = originalFetch;
  assert.equal(fetchCalls, 0);
});
