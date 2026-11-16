// Shared harness for the portable tokenizer conformance vectors in
// tests/fixtures/parity/tokenizer: runs the Transformers.js tokenizers module
// (@huggingface/tokenizers) and the kit's alignTokenOffsets adapter and
// reports value-free mismatches (case ids, fields, indices, reason codes).
// Used by test_tokenizer_conformance.spec.ts and the contract-gate reporter.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { alignTokenOffsets } from "../../js/openmedkit-web/src/offsets";
import type { RawTokenClassificationEntity } from "../../js/openmedkit-web/src/types";

const packageDir = fileURLToPath(new URL("../../js/openmedkit-web", import.meta.url));

export const RUNTIME = "tokenizers-js";
export const CONTRACT_VERSION = "openmed.tokenizer-conformance.v1";
export const FIELDS = [
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
export type Field = (typeof FIELDS)[number];
export type RuntimeEncoding = Partial<Record<Field, unknown>>;

export interface VectorCase {
  id: string;
  features: string[];
  text: string;
  window?: { max_length: number; stride: number };
  expected: Partial<Record<Field, unknown>>;
}

export interface Vectors {
  contract_version: string;
  tokenizer: { directory: string; sha256: string };
  cases: VectorCase[];
}

export interface FieldStatus {
  status: "supported" | "unsupported";
  reason?: string;
}

export interface RuntimeSupport {
  contract_version: string;
  runtimes: Record<
    string,
    {
      fields: Record<Field, FieldStatus>;
      divergences: { feature: string; fields: Field[]; reason: string }[];
    }
  >;
}

export interface Mismatch {
  caseId: string;
  field: string;
  index: number | null;
  code: string;
  feature?: string;
}

export interface Report {
  checked: number;
  unsupported: { field: Field; reason: string }[];
  declared: { caseId: string; field: Field; reason: string }[];
  mismatches: Mismatch[];
}

export interface JsTokenizer {
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

export type TokenizerConstructor = new (
  tokenizerJson: object,
  tokenizerConfig: object,
) => JsTokenizer;

export async function loadTokenizerClass(): Promise<TokenizerConstructor> {
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

export function utf16ToScalar(text: string, utf16Offset: number): number {
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
export function kitOffsets(text: string, tokens: string[]): number[][] {
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

export function makeRunner(tokenizer: JsTokenizer) {
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

export function compare(
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

export function verify(
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

export function describe(report: Report): string {
  return report.mismatches
    .map(
      (item) =>
        `${item.code} case=${item.caseId} field=${item.field} index=${item.index ?? "-"}` +
        (item.feature ? ` feature=${item.feature}` : ""),
    )
    .join("\n");
}
