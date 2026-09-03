import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import type {
  OpenMedDeidentifyResult,
  OpenMedSpan,
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

test("default model is a public -onnx-android repo loaded through loadOnnxModel", async () => {
  const api = await loadApi();
  assert.equal(
    api.DEFAULT_MODEL_ID,
    "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android",
  );

  let seenOptions: Record<string, unknown> | undefined;
  const runtime: TransformersRuntime = {
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

async function loadApi() {
  return import(distUrl);
}

const goldenText =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";
const goldenRedactedText =
  "Patient [PERSON], DOB [DATE_OF_BIRTH], email [EMAIL].";

function transformersJsTokens(
  rows: Array<[number, string, string]>,
): TokenClassificationEntity[] {
  return rows.map(([index, word, entity]) => ({
    entity,
    score: 0.9,
    index,
    word,
  }));
}

// Shape emitted by the Transformers.js token-classification pipeline with
// aggregation_strategy "none": lowercased WordPiece words, no start/end.
const transformersJsFixturePipeline: TokenClassificationPipeline = () =>
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
