// Portable tokenizer conformance for the web runtime.
//
// Runs the Transformers.js tokenizers module (@huggingface/tokenizers) and the
// kit's alignTokenOffsets adapter against the shared synthetic vectors in
// tests/fixtures/parity/tokenizer before any inference comparison. Diagnostics
// carry case ids, field names, indices and reason codes only.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import { alignTokenOffsets } from "../../js/openmedkit-web/src/offsets";
import type { RawTokenClassificationEntity } from "../../js/openmedkit-web/src/types";

const RUNTIME = "tokenizers-js";
const CONTRACT_VERSION = "openmed.tokenizer-conformance.v1";
const FIELDS = [
  "normalized",
  "tokens",
  "input_ids",
  "attention_mask",
  "token_type_ids",
  "special_tokens_mask",
  "offsets",
  "truncated_input_ids",
  "window_ranges",
  "window_input_ids",
] as const;
type Field = (typeof FIELDS)[number];
type RuntimeEncoding = Partial<Record<Field, unknown>>;

interface VectorCase {
  id: string;
  features: string[];
  text: string;
  window?: { max_length: number; stride: number };
  expected: Partial<Record<Field, unknown>>;
}

interface Vectors {
  contract_version: string;
  tokenizer: { directory: string; sha256: string };
  cases: VectorCase[];
}

interface FieldStatus {
  status: "supported" | "unsupported";
  reason?: string;
}

interface RuntimeSupport {
  contract_version: string;
  runtimes: Record<
    string,
    {
      fields: Record<Field, FieldStatus>;
      divergences: { feature: string; fields: Field[]; reason: string }[];
    }
  >;
}

interface Mismatch {
  caseId: string;
  field: string;
  index: number | null;
  code: string;
  feature?: string;
}

interface Report {
  checked: number;
  unsupported: { field: Field; reason: string }[];
  declared: { caseId: string; field: Field; reason: string }[];
  mismatches: Mismatch[];
}

interface JsTokenizer {
  normalizer: ((text: string) => string) | null;
  encode(
    text: string,
    options: { add_special_tokens: boolean; return_token_type_ids: boolean },
  ): {
    ids: number[];
    tokens: string[];
    attention_mask: number[];
    token_type_ids?: number[];
  };
}

type TokenizerConstructor = new (
  tokenizerJson: object,
  tokenizerConfig: object,
) => JsTokenizer;

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const fixtureDir = join(rootDir, "tests", "fixtures", "parity", "tokenizer");
const packageDir = join(rootDir, "js", "openmedkit-web");

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

async function loadTokenizerClass(): Promise<TokenizerConstructor> {
  // Resolve from the web package so the pinned devDependency is the one used.
  const require = createRequire(join(packageDir, "package.json"));
  const resolved = require.resolve("@huggingface/tokenizers");
  const module = (await import(pathToFileURL(resolved).href)) as {
    Tokenizer?: TokenizerConstructor;
    default?: { Tokenizer?: TokenizerConstructor };
  };
  const Tokenizer = module.Tokenizer ?? module.default?.Tokenizer;
  assert.ok(Tokenizer, "TOKENIZER_RUNTIME_UNAVAILABLE runtime=tokenizers-js");
  return Tokenizer;
}

function utf16ToScalar(text: string, utf16Offset: number): number {
  let scalar = 0;
  let unit = 0;
  for (const character of text) {
    if (unit >= utf16Offset) {
      break;
    }
    unit += character.length;
    scalar += 1;
  }
  return scalar;
}

// Kit adapter: Transformers.js gives words without offsets, so the web kit
// aligns them with alignTokenOffsets (UTF-16) and the contract compares scalar
// offsets. A refused alignment yields no offsets at all.
function kitOffsets(text: string, tokens: string[]): number[][] {
  const raw: RawTokenClassificationEntity[] = tokens.map((word, index) => ({
    entity: "B-SYNTH",
    score: 1,
    index,
    word,
  }));
  let aligned;
  try {
    aligned = alignTokenOffsets(text, raw);
  } catch {
    return [];
  }
  const spans = new Map<number, [number, number]>();
  for (const entity of aligned) {
    if (typeof entity.index === "number") {
      spans.set(entity.index, [entity.start, entity.end]);
    }
  }
  return tokens.map((_, index) => {
    const span = spans.get(index);
    return span === undefined
      ? [0, 0]
      : [utf16ToScalar(text, span[0]), utf16ToScalar(text, span[1])];
  });
}

function makeRunner(tokenizer: JsTokenizer) {
  return (vectorCase: VectorCase): RuntimeEncoding => {
    const encoded = tokenizer.encode(vectorCase.text, {
      add_special_tokens: true,
      return_token_type_ids: true,
    });
    return {
      normalized: tokenizer.normalizer
        ? tokenizer.normalizer(vectorCase.text)
        : vectorCase.text,
      tokens: encoded.tokens,
      input_ids: encoded.ids,
      attention_mask: encoded.attention_mask,
      token_type_ids: encoded.token_type_ids,
      offsets: kitOffsets(vectorCase.text, encoded.tokens),
    };
  };
}

function compare(
  expected: unknown,
  actual: unknown,
): { code: string; index: number | null } | null {
  const left = typeof expected === "string" ? Array.from(expected) : (expected as unknown[]);
  if (typeof expected === "string" && typeof actual !== "string") {
    return { code: "TOKENIZER_FIELD_VALUE_MISMATCH", index: null };
  }
  const right = typeof actual === "string" ? Array.from(actual) : (actual as unknown[]);
  if (!Array.isArray(right)) {
    return { code: "TOKENIZER_FIELD_VALUE_MISMATCH", index: null };
  }
  const shared = Math.min(left.length, right.length);
  for (let index = 0; index < shared; index += 1) {
    if (JSON.stringify(left[index]) !== JSON.stringify(right[index])) {
      return { code: "TOKENIZER_FIELD_VALUE_MISMATCH", index };
    }
  }
  if (left.length !== right.length) {
    return { code: "TOKENIZER_FIELD_LENGTH_MISMATCH", index: shared };
  }
  return null;
}

function verify(
  vectors: Vectors,
  support: RuntimeSupport,
  runner: (vectorCase: VectorCase) => RuntimeEncoding,
): Report {
  const report: Report = { checked: 0, unsupported: [], declared: [], mismatches: [] };
  const entry = support.runtimes[RUNTIME];
  if (
    vectors.contract_version !== CONTRACT_VERSION ||
    support.contract_version !== CONTRACT_VERSION
  ) {
    report.mismatches.push({
      caseId: "*",
      field: "runtime",
      index: null,
      code: "TOKENIZER_CONTRACT_VERSION_MISMATCH",
    });
    return report;
  }
  if (entry === undefined) {
    report.mismatches.push({
      caseId: "*",
      field: "runtime",
      index: null,
      code: "TOKENIZER_RUNTIME_UNDECLARED",
    });
    return report;
  }
  for (const field of FIELDS) {
    const status = entry.fields[field];
    if (status?.status !== "supported") {
      report.unsupported.push({ field, reason: status?.reason ?? "undeclared" });
    }
  }
  const unsupported = new Set(report.unsupported.map((item) => item.field));
  const observed = new Set<string>();
  for (const vectorCase of vectors.cases) {
    let produced: RuntimeEncoding;
    try {
      produced = runner(vectorCase);
    } catch {
      report.mismatches.push({
        caseId: vectorCase.id,
        field: "runtime",
        index: null,
        code: "TOKENIZER_RUNTIME_ERROR",
      });
      continue;
    }
    const features = new Set(vectorCase.features);
    for (const field of FIELDS) {
      if (!(field in vectorCase.expected) || unsupported.has(field)) {
        continue;
      }
      const actual = produced[field];
      if (actual === undefined || actual === null) {
        report.mismatches.push({
          caseId: vectorCase.id,
          field,
          index: null,
          code: "TOKENIZER_FIELD_MISSING",
        });
        continue;
      }
      const finding = compare(vectorCase.expected[field], actual);
      if (finding === null) {
        report.checked += 1;
        continue;
      }
      const matching = entry.divergences.filter(
        (item) => features.has(item.feature) && item.fields.includes(field),
      );
      if (matching.length > 0) {
        for (const item of matching) {
          observed.add(`${item.feature}\u0000${field}`);
        }
        report.declared.push({
          caseId: vectorCase.id,
          field,
          reason: matching[0]?.reason ?? "undeclared",
        });
        continue;
      }
      report.checked += 1;
      report.mismatches.push({ caseId: vectorCase.id, field, ...finding });
    }
  }
  for (const item of entry.divergences) {
    for (const field of item.fields) {
      if (!observed.has(`${item.feature}\u0000${field}`)) {
        report.mismatches.push({
          caseId: "*",
          field,
          index: null,
          code: "TOKENIZER_DECLARED_DIVERGENCE_STALE",
          feature: item.feature,
        });
      }
    }
  }
  return report;
}

function describe(report: Report): string {
  return report.mismatches
    .map(
      (item) =>
        `${item.code} case=${item.caseId} field=${item.field} index=${item.index ?? "-"}` +
        (item.feature ? ` feature=${item.feature}` : ""),
    )
    .join("\n");
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
