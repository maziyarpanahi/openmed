import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import type {
  OpenMedDeidentifyResult,
  OpenMedSpan,
  TokenClassificationPipeline,
  TransformersRuntime,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const packageDir = join(rootDir, "js", "openmedkit-web");
const distUrl = pathToFileURL(join(packageDir, "dist", "index.js")).href;

const fixturePath = join(
  rootDir,
  "tests",
  "web",
  "fixtures",
  "npm_deidentify_golden.json",
);
const publicSurfaceSnapshotPath = join(
  rootDir,
  "tests",
  "web",
  "__snapshots__",
  "openmedkit-web-public-api.json",
);

test("deidentify returns OpenMedSpan records matching the Python golden", async () => {
  const api = await loadApi();
  const expected = JSON.parse(
    await readFile(fixturePath, "utf8"),
  ) as OpenMedDeidentifyResult;

  const result = (await api.deidentify(expected.text, {
    pipeline: fixturePipeline,
    docId: "web-fixture",
    hashSecret: "test-secret",
    detector: "fixture-token-classifier",
    metadata: { fixture: "python-generated" },
  })) as OpenMedDeidentifyResult;

  assert.equal(result.text, expected.text);
  assert.equal(result.deidentifiedText, expected.deidentifiedText);
  assertSpansClose(result.spans, expected.spans);
});

test("public runtime surface is snapshot-tested", async () => {
  const api = await loadApi();
  const snapshot = JSON.parse(
    await readFile(publicSurfaceSnapshotPath, "utf8"),
  ) as { exports: string[]; packageExports: unknown };
  const packageJson = JSON.parse(
    await readFile(join(packageDir, "package.json"), "utf8"),
  ) as { exports: unknown };

  assert.deepEqual(Object.keys(api).sort(), snapshot.exports);
  assert.deepEqual(packageJson.exports, snapshot.packageExports);
});

test("local model loading is offline-only by default", async () => {
  const api = await loadApi();
  const runtime: TransformersRuntime = {
    env: {
      allowLocalModels: false,
      allowRemoteModels: true,
    },
    pipeline: async (_task, model, options) => {
      assert.equal(model, "/models/openmed-transformersjs");
      assert.equal(runtime.env?.allowRemoteModels, false);
      assert.equal(runtime.env?.allowLocalModels, true);
      assert.equal(options?.local_files_only, true);
      assert.equal(options?.localFilesOnly, true);
      return fixturePipeline;
    },
  };

  const loaded = (await api.loadTokenClassificationPipeline(
    "/models/openmed-transformersjs",
    { runtime },
  )) as TokenClassificationPipeline;

  assert.equal(loaded, fixturePipeline);
  assert.equal(runtime.env?.allowRemoteModels, true);
  assert.equal(runtime.env?.allowLocalModels, false);
});

test("OpenMed ONNX loading selects the root INT8 artifact", async () => {
  const api = await loadApi();
  const runtime: TransformersRuntime = {
    pipeline: async (task, model, options) => {
      assert.equal(task, "token-classification");
      assert.equal(model, "OpenMed/example-v1-onnx-android");
      assert.equal(options?.subfolder, "");
      assert.equal(options?.model_file_name, "model_int8");
      assert.equal(options?.quantized, false);
      return fixturePipeline;
    },
  };

  const loaded = (await api.loadOnnxModel(
    "OpenMed/example-v1-onnx-android",
    { runtime },
  )) as TokenClassificationPipeline;

  assert.equal(loaded, fixturePipeline);
});

async function loadApi() {
  return import(distUrl);
}

const fixturePipeline: TokenClassificationPipeline = (text) => {
  const aliceStart = text.indexOf("Alice");
  const nguyenStart = text.indexOf("Nguyen");
  const dobStart = text.indexOf("1979-04-12");
  const emailStart = text.indexOf("alice@example.org");
  return [
    {
      entity: "B-NAME",
      score: 0.99,
      word: "Alice",
      start: aliceStart,
      end: aliceStart + "Alice".length,
      index: 0,
    },
    {
      entity: "E-NAME",
      score: 0.97,
      word: "Nguyen",
      start: nguyenStart,
      end: nguyenStart + "Nguyen".length,
      index: 1,
    },
    {
      entity: "S-DATE_OF_BIRTH",
      score: 0.96,
      word: "1979-04-12",
      start: dobStart,
      end: dobStart + "1979-04-12".length,
      index: 2,
    },
    {
      entity: "S-EMAIL",
      score: 0.98,
      word: "alice@example.org",
      start: emailStart,
      end: emailStart + "alice@example.org".length,
      index: 3,
    },
  ];
};

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
    assert.ok(Math.abs(actualScore - expectedScore) <= 1e-12);
  }
}
