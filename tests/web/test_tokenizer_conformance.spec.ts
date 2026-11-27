// Portable tokenizer conformance for the web runtime.
//
// Runs the Transformers.js tokenizers module (@huggingface/tokenizers) and the
// kit's alignTokenOffsets adapter against the shared synthetic vectors in
// tests/fixtures/parity/tokenizer before any inference comparison. Diagnostics
// carry case ids, field names, indices and reason codes only.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  CONTRACT_VERSION,
  RUNTIME,
  describe,
  loadTokenizerClass,
  makeRunner,
  verify,
  type RuntimeSupport,
  type Vectors,
} from "./tokenizer_conformance_support";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const fixtureDir = join(rootDir, "tests", "fixtures", "parity", "tokenizer");

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

async function loadFixtures() {
  const vectors = await readJson<Vectors>(
    join(fixtureDir, "tokenizer_conformance_v1.json"),
  );
  const support = await readJson<RuntimeSupport>(
    join(fixtureDir, "tokenizer_runtime_support_v1.json"),
  );
  const tokenizerDir = join(fixtureDir, vectors.tokenizer.directory);
  const tokenizerBytes = await readFile(join(tokenizerDir, "tokenizer.json"));
  const tokenizerConfig = await readJson<Record<string, unknown>>(
    join(tokenizerDir, "tokenizer_config.json"),
  );
  return { vectors, support, tokenizerBytes, tokenizerConfig };
}

function sha256(data: Buffer): string {
  return createHash("sha256").update(data).digest("hex");
}

test("tokenizer conformance assets match the vectors and manifest digests", async () => {
  const { vectors, tokenizerBytes } = await loadFixtures();
  const manifest = await readJson<{
    contract_version: string;
    files: { path: string; sha256: string }[];
  }>(join(fixtureDir, "manifest.json"));

  assert.equal(manifest.contract_version, CONTRACT_VERSION);
  assert.equal(sha256(tokenizerBytes), vectors.tokenizer.sha256);
  for (const file of manifest.files) {
    const digest = sha256(await readFile(join(fixtureDir, file.path)));
    assert.equal(digest, file.sha256, `TOKENIZER_ASSET_DIGEST_MISMATCH role=${file.path}`);
  }
});

test("tokenizers-js matches the shared tokenizer vectors before inference", async () => {
  const Tokenizer = await loadTokenizerClass();
  const { vectors, support, tokenizerBytes, tokenizerConfig } = await loadFixtures();
  const tokenizer = new Tokenizer(JSON.parse(tokenizerBytes.toString("utf8")), tokenizerConfig);

  const report = verify(vectors, support, makeRunner(tokenizer));
  console.log(
    `TOKENIZER_CONFORMANCE runtime=${RUNTIME} status=${
      report.mismatches.length === 0 ? "pass" : "fail"
    } checked=${report.checked} unsupported=${report.unsupported
      .map((item) => `${item.field}:${item.reason}`)
      .join(",")} declared_divergences=${report.declared.length}`,
  );

  assert.equal(report.mismatches.length, 0, describe(report));
  assert.ok(report.checked > 0);
  assert.deepEqual(
    report.unsupported.map((item) => item.field),
    ["special_tokens_mask", "truncated_input_ids", "window_ranges", "window_input_ids"],
  );
});

test("special-token insertion and normalization changes fail before model execution", async () => {
  const Tokenizer = await loadTokenizerClass();
  const { vectors, support, tokenizerBytes, tokenizerConfig } = await loadFixtures();
  let modelCalls = 0;
  const runModel = () => {
    modelCalls += 1;
  };

  const withoutCls = JSON.parse(tokenizerBytes.toString("utf8"));
  withoutCls.post_processor.single = withoutCls.post_processor.single.filter(
    (item: { SpecialToken?: { id: string } }) => item.SpecialToken?.id !== "[CLS]",
  );
  const casedNormalizer = JSON.parse(tokenizerBytes.toString("utf8"));
  casedNormalizer.normalizer.lowercase = false;
  casedNormalizer.normalizer.strip_accents = false;

  for (const [mutated, field] of [
    [withoutCls, "input_ids"],
    [casedNormalizer, "normalized"],
  ] as const) {
    const report = verify(
      vectors,
      support,
      makeRunner(new Tokenizer(mutated, tokenizerConfig)),
    );
    if (report.mismatches.length === 0) {
      runModel();
    }
    assert.ok(
      report.mismatches.some(
        (item) => item.caseId === "ascii-basic-01" && item.field === field && item.index === 0,
      ),
      describe(report),
    );
    const diagnostics = describe(report);
    for (const vectorCase of vectors.cases) {
      if (vectorCase.text.trim().length > 3) {
        assert.ok(!diagnostics.includes(vectorCase.text), "diagnostics must be value-free");
      }
    }
  }
  assert.equal(modelCalls, 0);
});
