import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { promisify } from "node:util";

import type {
  OpenMedDeidentifyResult,
  OpenMedSpan,
  RawTokenClassificationEntity,
  RawTokenClassificationPipeline,
  RawTransformersRuntime,
  TokenClassificationEntity,
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

test("default span hashes are private per call and consistent within it", async () => {
  const api = await loadApi();
  const pipeline = () => [
    { entity: "B-NAME", word: "Synthetic", start: 0, end: 9, score: 1 },
    { entity: "B-NAME", word: "Synthetic", start: 14, end: 23, score: 1 },
  ];
  const text = "Synthetic met Synthetic";
  const first = await api.extractPii(text, { pipeline });
  const second = await api.extractPii(text, { pipeline });
  const redacted = await api.deidentify(text, { pipeline });
  assert.equal(first.length, 2);
  assert.equal(first[0].text_hash, first[1].text_hash);
  assert.notEqual(first[0].text_hash, second[0].text_hash);
  assert.notEqual(
    first[0].text_hash,
    await api.hmacTextHash("Synthetic", "openmedkit-web"),
  );
  assert.equal(redacted.deidentifiedText, "[PERSON] met [PERSON]");
  assert.notEqual(first[0].text_hash, redacted.spans[0].text_hash);
});

test("explicit hash keys remain deterministic and byte keys are snapshotted", async () => {
  const api = await loadApi();
  const secret = "synthetic-private-fixture-key";
  const bytes = new TextEncoder().encode(secret);
  const pipeline = async () => {
    bytes.fill(0);
    return [{ entity: "B-NAME", word: "Synthetic", start: 0, end: 9, score: 1 }];
  };
  const first = await api.extractPii("Synthetic", { pipeline, hashSecret: bytes });
  const second = await api.extractPii("Synthetic", {
    pipeline,
    hashSecret: secret,
  });
  assert.equal(first[0].text_hash, second[0].text_hash);
  assert.equal(first[0].text_hash, await api.hmacTextHash("Synthetic", secret));
});

test("empty hash keys fail before inference even with no detections", async () => {
  const api = await loadApi();
  for (const secret of ["", new Uint8Array()]) {
    await assert.rejects(
      api.hmacTextHash("Synthetic", secret),
      /hashSecret must not be empty/,
    );
    await assert.rejects(
      api.extractPii("", {
        hashSecret: secret,
        pipeline: () => {
          throw new Error("inference must not run");
        },
      }),
      /hashSecret must not be empty/,
    );
  }
});

test("Node crypto fallback uses random keys and rejects empty keys", async () => {
  const script = `
    import assert from 'node:assert/strict';
    Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true });
    const api = await import(${JSON.stringify(distUrl)});
    const pipeline = () => [{ entity: 'B-NAME', word: 'Synthetic', start: 0, end: 9, score: 1 }];
    const first = await api.extractPii('Synthetic', { pipeline });
    const second = await api.extractPii('Synthetic', { pipeline });
    assert.notEqual(first[0].text_hash, second[0].text_hash);
    for (const key of ['', new Uint8Array()]) {
      await assert.rejects(api.hmacTextHash('Synthetic', key), /hashSecret must not be empty/);
    }
    const { createHmac } = await import('node:crypto');
    assert.equal(await api.hmacTextHash('Synthetic', 'fixture-key'),
      'hmac-sha256:' + createHmac('sha256', 'fixture-key').update('Synthetic').digest('hex'));
  `;
  await promisify(execFile)(process.execPath, ["--input-type=module", "-e", script]);
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

  assert.deepEqual(await loaded(goldenText), fixturePipeline(goldenText));
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

  assert.deepEqual(await loaded(goldenText), fixturePipeline(goldenText));
});

test("default model is a public -onnx-android repo loaded through loadOnnxModel", async () => {
  const api = await loadApi();
  assert.equal(
    api.DEFAULT_MODEL_ID,
    "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android",
  );

  let seenOptions: Record<string, unknown> | undefined;
  const runtime: RawTransformersRuntime = {
    pipeline: async (task, model, options) => {
      assert.equal(task, "token-classification");
      assert.equal(model, api.DEFAULT_MODEL_ID);
      seenOptions = options;
      return transformersJsFixturePipeline;
    },
  };

  const result = (await api.deidentify(goldenText, {
    loaderOptions: { runtime },
  })) as OpenMedDeidentifyResult;

  assert.equal(seenOptions?.subfolder, "");
  assert.equal(seenOptions?.model_file_name, "model_int8");
  assert.equal(seenOptions?.quantized, false);
  assert.equal(result.deidentifiedText, goldenRedactedText);
  assert.equal(result.spans[0]?.metadata.model, api.DEFAULT_MODEL_ID);
});

test("default path requests only the pinned model revision", async () => {
  const api = await loadApi();
  assert.match(api.DEFAULT_MODEL_REVISION, /^[0-9a-f]{40}$/);

  const requests: Array<{ model: string; revision: unknown }> = [];
  const runtime: RawTransformersRuntime = {
    pipeline: async (_task, model, options) => {
      requests.push({ model, revision: options?.revision });
      return transformersJsFixturePipeline;
    },
  };
  await api.deidentify(goldenText, { loaderOptions: { runtime } });
  await api.extractPii(goldenText, {
    model: api.DEFAULT_MODEL_ID,
    loaderOptions: { runtime },
  });
  await api.extractPii(goldenText, {
    loaderOptions: { runtime, revision: "a".repeat(40) },
  });
  await api.extractPii(goldenText, {
    model: "OpenMed/example-v1-onnx-android",
    loaderOptions: { runtime },
  });
  assert.deepEqual(requests, [
    { model: api.DEFAULT_MODEL_ID, revision: api.DEFAULT_MODEL_REVISION },
    { model: api.DEFAULT_MODEL_ID, revision: api.DEFAULT_MODEL_REVISION },
    { model: api.DEFAULT_MODEL_ID, revision: "a".repeat(40) },
    { model: "OpenMed/example-v1-onnx-android", revision: undefined },
  ]);

  let loaderRevision: unknown;
  await api.extractPii(goldenText, {
    modelLoader: async (_model: string, options?: { revision?: string }) => {
      loaderRevision = options?.revision;
      return () => [];
    },
  });
  assert.equal(loaderRevision, api.DEFAULT_MODEL_REVISION);
});

test("offline-only mode refuses Hub IDs and URLs before any loader runs", async () => {
  const api = await loadApi();
  const recorder = recordingRuntime();
  const remoteModels = [
    undefined,
    api.DEFAULT_MODEL_ID,
    "OpenMed/example-v1-onnx-android",
    "https://models.example.invalid/openmed",
  ];
  const attempts = (offlineOnly: boolean | undefined) =>
    remoteModels.flatMap((model) => [
      () =>
        api.deidentify(goldenText, {
          ...(model === undefined ? {} : { model }),
          ...(offlineOnly === undefined ? {} : { offlineOnly }),
          modelLoader: recorder.modelLoader,
        }),
      () =>
        api.extractPii(goldenText, {
          ...(model === undefined ? {} : { model }),
          loaderOptions: {
            runtime: recorder.runtimeProvider,
            ...(offlineOnly === undefined ? {} : { offlineOnly }),
          },
        }),
      () =>
        api.loadOnnxModel(model ?? api.DEFAULT_MODEL_ID, {
          runtime: recorder.runtimeProvider,
          ...(offlineOnly === undefined ? {} : { offlineOnly }),
        }),
      () =>
        api.loadTokenClassificationPipeline(model ?? api.DEFAULT_MODEL_ID, {
          runtime: recorder.runtimeProvider,
          ...(offlineOnly === undefined ? {} : { offlineOnly }),
        }),
    ]);

  for (const attempt of attempts(true)) {
    await assertOfflineError(api, attempt(), "remote_model_refused");
  }
  assert.equal(api.isOfflineOnly(), false);
  api.setOfflineOnly(true);
  try {
    assert.equal(api.isOfflineOnly(), true);
    // A call cannot opt back out of the package-wide setting.
    for (const attempt of [...attempts(undefined), ...attempts(false)]) {
      await assertOfflineError(api, attempt(), "remote_model_refused");
    }
  } finally {
    api.setOfflineOnly(false);
  }
  assert.deepEqual(recorder.calls, []);
});

test("offline-only mode refuses remote WASM and model-path settings", async () => {
  const api = await loadApi();
  const cdnWasmPaths = {
    mjs: "https://cdn.example.invalid/ort/ort-wasm-simd-threaded.asyncify.mjs",
    wasm: "https://cdn.example.invalid/ort/ort-wasm-simd-threaded.asyncify.wasm",
  };
  const remoteSettings = [
    { env: { backends: { onnx: { wasm: { wasmPaths: cdnWasmPaths } } } } },
    { env: { backends: { onnx: { wasm: { wasmPaths: "//cdn.example.invalid/" } } } } },
    {
      env: {
        backends: {
          onnx: { wasm: { wasmPaths: new URL("https://cdn.example.invalid/") } },
        },
      },
    },
    { env: { localModelPath: "https://models.example.invalid/" } },
    { env: {}, wasmPaths: "https://cdn.example.invalid/ort/" },
    { env: {}, wasmPaths: "/\\cdn.example.invalid/ort/" },
    { env: {}, wasmPaths: { wasm: "https://cdn.example.invalid/ort.wasm" } },
  ];
  for (const { env, wasmPaths } of remoteSettings as Array<{
    env: Record<string, unknown>;
    wasmPaths?: unknown;
  }>) {
    const recorder = recordingRuntime(env);
    await assertOfflineError(
      api,
      api.loadOnnxModel("/models/openmed", {
        runtime: recorder.runtime,
        offlineOnly: true,
        ...(wasmPaths === undefined ? {} : { wasmPaths }),
      }),
      "remote_asset_refused",
    );
    assert.deepEqual(recorder.calls, []);
  }

  // A remote override is refused even when offline-only mode is off.
  await assertOfflineError(
    api,
    api.loadTokenClassificationPipeline("/models/openmed", {
      runtime: recordingRuntime().runtime,
      wasmPaths: "https://cdn.example.invalid/ort/",
    }),
    "remote_asset_refused",
  );

  // Local WebAssembly assets replace the CDN default before the runtime loads.
  for (const [wasmPaths, expected] of [
    ["/assets/onnxruntime", "/assets/onnxruntime/"],
    [
      { mjs: "/assets/ort.mjs", wasm: "/assets/ort.wasm" },
      { mjs: "/assets/ort.mjs", wasm: "/assets/ort.wasm" },
    ],
  ] as const) {
    const recorder = recordingRuntime({
      backends: { onnx: { wasm: { wasmPaths: cdnWasmPaths } } },
    });
    await api.loadOnnxModel("/models/openmed", {
      runtime: recorder.runtime,
      offlineOnly: true,
      wasmPaths,
    });
    assert.deepEqual(recorder.calls, [
      {
        model: "/models/openmed",
        localFilesOnly: true,
        allowRemoteModels: false,
        wasmPaths: expected,
      },
    ]);
  }
});

test("explicit local paths stay local-only in every mode", async () => {
  const api = await loadApi();
  const recorder = recordingRuntime();
  for (const offlineOnly of [false, true]) {
    await api.loadTokenClassificationPipeline("/models/openmed", {
      runtime: recorder.runtime,
      localFilesOnly: false,
      allowRemoteModels: true,
      offlineOnly,
    });
    await api.deidentify(goldenText, {
      model: "./models/openmed",
      offlineOnly,
      loaderOptions: {
        runtime: recorder.runtime,
        localFilesOnly: false,
        allowRemoteModels: true,
      },
    });
  }
  assert.deepEqual(
    recorder.calls.map(({ localFilesOnly, allowRemoteModels }) => ({
      localFilesOnly,
      allowRemoteModels,
    })),
    Array(4).fill({ localFilesOnly: true, allowRemoteModels: false }),
  );

  // Paths that only look local but resolve to another host are refused.
  for (const model of [
    "//models.example.invalid/openmed",
    "/\\models.example.invalid/openmed",
    "file://models.example.invalid/openmed",
  ]) {
    await assertOfflineError(
      api,
      api.loadTokenClassificationPipeline(model, { runtime: recorder.runtime }),
      "remote_model_refused",
    );
  }
  assert.equal(recorder.calls.length, 4);
});

test("offline-only loads fail closed with value-free errors on missing assets", async () => {
  const api = await loadApi();
  const secretPath = "/private/synthetic-patient-dir/model_int8.onnx";
  const failingRuntime: RawTransformersRuntime = {
    env: {},
    pipeline: async () => {
      throw new Error(`file was not found locally at "${secretPath}"`);
    },
  };
  await assert.rejects(
    api.loadOnnxModel("/models/openmed", { runtime: failingRuntime }),
    (error: Error) => error.message.includes(secretPath),
  );
  await assertOfflineError(
    api,
    api.loadOnnxModel("/models/openmed", {
      runtime: failingRuntime,
      offlineOnly: true,
    }),
    "local_assets_unavailable",
    [secretPath, "synthetic-patient-dir"],
  );
  await assertOfflineError(
    api,
    api.deidentify("Patient Synthetic Person", {
      model: "/models/openmed",
      offlineOnly: true,
      loaderOptions: { runtime: failingRuntime },
    }),
    "local_assets_unavailable",
    [secretPath, "Synthetic Person"],
  );

  // The ONNX Runtime Web loader is local-only in every mode; offline-only mode
  // also replaces its load failures with the typed error.
  const createCalls: string[] = [];
  const ortRuntime = {
    env: {},
    InferenceSession: {
      create: (modelPath: string) => {
        createCalls.push(modelPath);
        throw new Error(`failed to load model from ${secretPath}`);
      },
    },
  };
  const ortOptions = {
    modelPath: secretPath,
    assetPath: "/assets/onnxruntime/",
    runtime: ortRuntime,
    capabilities: {
      webgpu: false,
      wasm: true,
      wasmSimd: true,
      sharedArrayBuffer: false,
      crossOriginIsolated: false,
      hardwareConcurrency: 1,
    },
  };
  const originalFetch = globalThis.fetch;
  let fetchCalls = 0;
  globalThis.fetch = (() => {
    fetchCalls += 1;
    throw new Error("network fetch is not allowed in this test");
  }) as typeof fetch;
  api.setOfflineOnly(true);
  try {
    await assertOfflineError(
      api,
      api.loadOrtWebSession({ ...ortOptions, cache: new Map() }),
      "local_assets_unavailable",
      [secretPath, "synthetic-patient-dir"],
    );
    await assertOfflineError(
      api,
      api.loadWebGpuTokenClassificationSession({
        ...ortOptions,
        cache: new Map(),
        probeAdapter: false,
      }),
      "local_assets_unavailable",
      [secretPath, "synthetic-patient-dir"],
    );
    await assert.rejects(
      api.loadOrtWebSession({
        ...ortOptions,
        modelPath: "https://models.example.invalid/model.onnx",
        cache: new Map(),
      }),
      /local\/offline/,
    );
  } finally {
    api.setOfflineOnly(false);
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(createCalls, [secretPath, secretPath]);
  assert.equal(fetchCalls, 0);
  await assert.rejects(
    api.loadOrtWebSession({ ...ortOptions, cache: new Map() }),
    (error: Error) => error.message.includes(secretPath),
  );
});

test("extractPii keeps O tokens so offset alignment sees the full sequence", async () => {
  const api = await loadApi();
  let seenOptions: Record<string, unknown> | undefined;
  const pipeline: TokenClassificationPipeline = (_text, options) => {
    seenOptions = options;
    return [];
  };

  await api.extractPii(goldenText, { pipeline });

  assert.equal(seenOptions?.aggregation_strategy, "none");
  assert.deepEqual(seenOptions?.ignore_labels, []);
});

test("offset-less Transformers.js output is aligned to character offsets", async () => {
  const api = await loadApi();
  const result = (await api.deidentify(goldenText, {
    pipeline: transformersJsFixturePipeline,
  })) as OpenMedDeidentifyResult;

  assert.equal(result.deidentifiedText, goldenRedactedText);
  assert.deepEqual(
    result.spans.map((span) => [span.start, span.end, span.canonical_label]),
    [
      [8, 20, "PERSON"],
      [26, 36, "DATE_OF_BIRTH"],
      [44, 61, "EMAIL"],
    ],
  );
  assert.deepEqual(
    result.spans.map((span) => span.evidence.token_count),
    [4, 5, 5],
  );
});

test("word-initial tokens are not matched inside earlier words when O tokens were dropped", async () => {
  const api = await loadApi();
  const text = "Consent obtained by phone. dcarter@email.com is the mother.";
  // Transformers.js default ignore_labels=["O"] drops every O token, so the
  // first email piece "d" arrives with a large index gap after [CLS].
  const output = transformersJsTokens([
    [6, "d", "B-email"],
    [7, "##carter", "I-email"],
    [8, "@", "I-email"],
    [9, "email", "I-email"],
    [10, ".", "I-email"],
    [11, "com", "I-email"],
  ]);

  const spans = (await api.extractPii(text, {
    pipeline: () => output,
  })) as OpenMedSpan[];

  assert.deepEqual(
    spans.map((span) => [span.start, span.end, span.canonical_label]),
    [[27, 44, "EMAIL"]],
  );
});

test("SentencePiece and byte-level markers, case, and accents align to the source text", async () => {
  const api = await loadApi();
  const text = "Patient Nguyễn Văn An was seen.";

  const sentencePiece = transformersJsTokens([
    [1, "▁Patient", "O"],
    [2, "▁Nguyễn", "B-PERSON"],
    [3, "▁Văn", "I-PERSON"],
    [4, "▁An", "I-PERSON"],
    [5, "▁was", "O"],
    [6, "▁seen", "O"],
    [7, ".", "O"],
  ]);
  const byteLevelLowercased = transformersJsTokens([
    [1, "patient", "O"],
    [2, "Ġnguyen", "B-PERSON"],
    [3, "Ġvan", "I-PERSON"],
    [4, "Ġan", "I-PERSON"],
    [5, "Ġwas", "O"],
    [6, "Ġseen", "O"],
    [7, ".", "O"],
  ]);

  for (const output of [sentencePiece, byteLevelLowercased]) {
    const spans = (await api.extractPii(text, {
      pipeline: () => output,
    })) as OpenMedSpan[];
    assert.deepEqual(
      spans.map((span) => [span.start, span.end, span.canonical_label]),
      [[8, 21, "PERSON"]],
    );
    assert.equal(text.slice(8, 21), "Nguyễn Văn An");
  }
});

test("alignTokenOffsets keeps supplied offsets, drops special tokens, and fills the rest", async () => {
  const api = await loadApi();
  const text = "Email alice@example.org today.";
  const aligned = api.alignTokenOffsets(text, [
    { entity: "O", index: 0, word: "[CLS]" },
    { entity: "O", index: 1, word: "email", start: 0, end: 5 },
    { entity: "B-EMAIL", index: 2, word: "alice" },
    { entity: "I-EMAIL", index: 3, word: "@" },
    { entity: "I-EMAIL", index: 4, word: "example" },
    { entity: "I-EMAIL", index: 5, word: "." },
    { entity: "I-EMAIL", index: 6, word: "org" },
    { entity: "O", index: 7, word: "today" },
    { entity: "O", index: 8, word: "." },
    { entity: "O", index: 9, word: "[SEP]" },
  ]) as TokenClassificationEntity[];

  assert.deepEqual(
    aligned.map((token) => [token.word, token.start, token.end]),
    [
      ["email", 0, 5],
      ["alice", 6, 11],
      ["@", 11, 12],
      ["example", 12, 19],
      [".", 19, 20],
      ["org", 20, 23],
      ["today", 24, 29],
      [".", 29, 30],
    ],
  );
});

test("alignment preserves decomposed accents and supplementary Unicode letters", async () => {
  const api = await loadApi();
  for (const [name, word] of [["Jose\u0301", "jose"], ["𐐀", "𐐨"]]) {
    const text = `Name ${name}`;
    const result = await api.deidentify(text, {
      pipeline: () => transformersJsTokens([
        [1, "name", "O"],
        [2, word!, "B-PERSON"],
      ]),
    });
    assert.equal(result.deidentifiedText, "Name [PERSON]");
    assert.deepEqual(result.spans.map((span) => [span.start, span.end]), [
      [5, text.length],
    ]);
  }
});

test("unalignable classified tokens fail without exposing source text", async () => {
  const api = await loadApi();
  for (const word of ["[UNK]", "not-in-source"]) {
    await assert.rejects(
      api.deidentify("Name Synthetic", {
        pipeline: () => transformersJsTokens([
          [1, "name", "O"],
          [2, word, "B-PERSON"],
        ]),
      }),
      { message: "Token offset alignment failed; provide source offsets." },
    );
  }
});

test("model loaders return aligned offsets for flat and nested runtime output", async () => {
  const api = await loadApi();
  const rawTokens = transformersJsTokens([[1, "alice", "B-PERSON"]]);
  for (const nested of [false, true]) {
    const runtime: RawTransformersRuntime = {
      pipeline: () => () => nested ? [rawTokens] : rawTokens,
    };
    const loaded = await api.loadTokenClassificationPipeline("/models/local", { runtime });
    const output = await loaded("Alice");
    const expected = [{ ...rawTokens[0], start: 0, end: 5 }];
    assert.deepEqual(output, nested ? [expected] : expected);
  }
});

test("v2.2 numeric-offset public types remain compatible with raw runtime inputs", async () => {
  const directory = await mkdtemp(join(tmpdir(), "openmed-types-"));
  try {
    const sourcePath = join(directory, "compatibility.ts");
    const importPath = JSON.stringify(join(packageDir, "dist", "index.js").replaceAll("\\", "/"));
    await writeFile(sourcePath, `
import {
  alignTokenOffsets, extractPii, flattenTokenClassificationOutput,
  loadTokenClassificationPipeline,
} from ${importPath};
import type {
  TokenClassificationEntity, TokenClassificationPipeline, TransformersRuntime,
  RawTokenClassificationEntity, RawTokenClassificationPipeline, RawTransformersRuntime,
} from ${importPath};

function numericOffsets(entity: TokenClassificationEntity): number {
  return entity.end - entity.start;
}
const aligned: TokenClassificationPipeline = () => [{ start: 0, end: 5 }];
const legacyRuntime: TransformersRuntime = { pipeline: () => aligned };
const loaded: TokenClassificationPipeline = await loadTokenClassificationPipeline(
  "/models/local", { runtime: legacyRuntime },
);
flattenTokenClassificationOutput(await loaded("Alice")).map(numericOffsets);
const legacyPipeline = await legacyRuntime.pipeline("token-classification", "/models/local");
flattenTokenClassificationOutput(await legacyPipeline("Alice")).map(numericOffsets);

const rawTokens: RawTokenClassificationEntity[] = [{ word: "alice", entity: "B-PERSON" }];
const raw: RawTokenClassificationPipeline = () => rawTokens;
const runtime: RawTransformersRuntime = { pipeline: () => raw };
alignTokenOffsets("Alice", rawTokens).map(numericOffsets);
await extractPii("Alice", { pipeline: raw });
await extractPii("Alice", { loaderOptions: { runtime } });
const normalized: TokenClassificationPipeline = await loadTokenClassificationPipeline(
  "/models/local", { runtime },
);
flattenTokenClassificationOutput(await normalized("Alice")).map(numericOffsets);
`);
    await promisify(execFile)(process.execPath, [
      join(packageDir, "node_modules", "typescript", "bin", "tsc"),
      "--noEmit", "--strict", "--skipLibCheck", "--target", "ES2022",
      "--module", "ESNext", "--moduleResolution", "Bundler",
      "--types", "node", "--typeRoots", join(packageDir, "node_modules", "@types"),
      sourcePath,
    ]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("aligned model loading preserves runtime metadata, disposal, and bound calls", async () => {
  const api = await loadApi();
  let disposed = false;
  const pipeline = Object.assign(
    () => transformersJsTokens([[1, "alice", "B-PERSON"]]),
    {
      model: { id: "synthetic" },
      async dispose() {
        assert.equal(this, pipeline);
        disposed = true;
      },
    },
  );
  const loaded = await api.loadTokenClassificationPipeline("/models/local", {
    runtime: { pipeline: () => pipeline },
  });
  assert.equal(loaded.model, pipeline.model);
  const output = await loaded.bind(null)("Alice");
  assert.equal(output[0].start, 0);
  assert.equal(output[0].end, 5);
  await loaded.dispose();
  assert.equal(disposed, true);
});

async function loadApi() {
  return import(distUrl);
}

// Records every runtime, loader and pipeline request a load makes.
function recordingRuntime(env: Record<string, unknown> = {}) {
  const calls: Array<Record<string, unknown>> = [];
  const runtime: RawTransformersRuntime = {
    env: env as RawTransformersRuntime["env"],
    pipeline: async (_task, model, options) => {
      const backends = env.backends as
        | { onnx?: { wasm?: { wasmPaths?: unknown } } }
        | undefined;
      calls.push({
        model,
        localFilesOnly: options?.local_files_only,
        allowRemoteModels: env.allowRemoteModels,
        wasmPaths: backends?.onnx?.wasm?.wasmPaths,
      });
      return transformersJsFixturePipeline;
    },
  };
  return {
    calls,
    runtime,
    runtimeProvider: async () => {
      calls.push({ runtimeProvider: true });
      return runtime;
    },
    modelLoader: async (model: string) => {
      calls.push({ modelLoader: model });
      return transformersJsFixturePipeline;
    },
  };
}

async function assertOfflineError(
  api: Awaited<ReturnType<typeof loadApi>>,
  promise: Promise<unknown>,
  code: string,
  forbidden: string[] = [],
) {
  await assert.rejects(promise, (error: Error & { code?: string }) => {
    assert.ok(error instanceof api.OpenMedOfflineError);
    assert.equal(error.name, "OpenMedOfflineError");
    assert.equal(error.code, code);
    assert.equal("cause" in error, false);
    // Controlled messages carry no paths, URLs, schemes or source text.
    assert.doesNotMatch(error.message, /[\\/:]/);
    for (const value of forbidden) {
      assert.equal(String(error).includes(value), false);
    }
    return true;
  });
}

const goldenText =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";
const goldenRedactedText =
  "Patient [PERSON], DOB [DATE_OF_BIRTH], email [EMAIL].";

function transformersJsTokens(
  rows: Array<[number, string, string]>,
): RawTokenClassificationEntity[] {
  return rows.map(([index, word, entity]) => ({
    entity,
    score: 0.9,
    index,
    word,
  }));
}

// Shape emitted by the Transformers.js token-classification pipeline with
// aggregation_strategy "none": lowercased WordPiece words, no start/end.
const transformersJsFixturePipeline: RawTokenClassificationPipeline = () =>
  transformersJsTokens([
    [1, "patient", "O"],
    [2, "alice", "B-NAME"],
    [3, "ng", "I-NAME"],
    [4, "##uy", "I-NAME"],
    [5, "##en", "I-NAME"],
    [6, ",", "O"],
    [7, "do", "O"],
    [8, "##b", "O"],
    [9, "1979", "B-DATE_OF_BIRTH"],
    [10, "-", "I-DATE_OF_BIRTH"],
    [11, "04", "I-DATE_OF_BIRTH"],
    [12, "-", "I-DATE_OF_BIRTH"],
    [13, "12", "I-DATE_OF_BIRTH"],
    [14, ",", "O"],
    [15, "email", "O"],
    [16, "alice", "B-EMAIL"],
    [17, "@", "I-EMAIL"],
    [18, "example", "I-EMAIL"],
    [19, ".", "I-EMAIL"],
    [20, "org", "I-EMAIL"],
    [21, ".", "O"],
  ]);

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
