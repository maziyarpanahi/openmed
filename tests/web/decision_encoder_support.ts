// Shared harness for tests/fixtures/parity/decision_encoder_v1.json: a
// word-level tokenizer and a tiny deterministic encoder session that replay
// the Python GLiClass ONNX reference through the web decision runtime. Used by
// test_decision_runtime.spec.ts and the contract-gate reporter.
import assert from "node:assert/strict";

import type {
  DecisionAccessPolicy,
  DecisionBackendKind,
  DecisionCalibrationProfile,
  DecisionTokenizer,
  GLiClassDecisionConfiguration,
  OrtInferenceSession,
  OrtTensorConstructor,
  OrtTensorLike,
} from "../../js/openmedkit-web/src/index";

export interface Fixture {
  schema_version: string;
  synthetic: boolean;
  decision_schema_version: string;
  tolerance: { logits: number; scores: number };
  tokenizer: TokenizerSpec;
  encoder: EncoderSpec;
  backend: {
    backend_id: string;
    runtime: string;
    model_id: string;
    revision: string;
    license_id: string;
  };
  runtimes: Record<string, RuntimeSettings>;
  cases: {
    id: string;
    runtime: string;
    request: unknown;
    calibration_profiles: DecisionCalibrationProfile[] | null;
    policy: (DecisionAccessPolicy & { allowed_backend_kinds: DecisionBackendKind[] }) | null;
    expected: {
      encoder_calls: { input_ids: number[]; logits: number[] | null }[];
      result: unknown;
    };
  }[];
  invalid_requests: { id: string; request: unknown; error: string }[];
}

export interface TokenizerSpec {
  vocab: string[];
  markers: string[];
  word_pattern: string;
  unk_token: string;
  cls_token: string;
  sep_token: string;
}

export interface EncoderSpec {
  hidden_size: number;
  embeddings: number[][];
  projection: number[][];
  projection_bias: number[];
  logit_scale: number;
  logit_bias: number;
  label_token_id: number;
  separator_token_id: number;
  ignored_token_ids: number[];
}

export interface RuntimeSettings {
  prompt: string | null;
  examples: { text: string; labels: string[] }[];
  score_levels: string[];
  prompt_first: boolean;
  max_length: number;
  fault: string | null;
}

export class WordLevelTokenizer implements DecisionTokenizer {
  readonly #ids: Map<string, number>;
  readonly #pattern: RegExp;
  readonly #spec: TokenizerSpec;

  constructor(spec: TokenizerSpec) {
    this.#spec = spec;
    this.#ids = new Map(spec.vocab.map((token, index) => [token, index]));
    this.#pattern = new RegExp(spec.word_pattern, "g");
  }

  tokenToId(token: string): number | undefined {
    return this.#ids.get(token);
  }

  encode(text: string): number[] {
    const id = (token: string) => this.#ids.get(token) ?? this.#ids.get(this.#spec.unk_token)!;
    const result = [id(this.#spec.cls_token)];
    let rest = text;
    for (;;) {
      let next: { index: number; marker: string } | undefined;
      for (const marker of this.#spec.markers) {
        const index = rest.indexOf(marker);
        if (index >= 0 && (next === undefined || index < next.index)) {
          next = { index, marker };
        }
      }
      const piece = next === undefined ? rest : rest.slice(0, next.index);
      const lowered = piece.replace(/[A-Z]/g, (char) => char.toLowerCase());
      for (const match of lowered.matchAll(this.#pattern)) {
        result.push(id(match[0]));
      }
      if (next === undefined) {
        break;
      }
      result.push(id(next.marker));
      rest = rest.slice(next.index + next.marker.length);
    }
    result.push(id(this.#spec.sep_token));
    return result;
  }
}

export function tinyLogits(spec: EncoderSpec, ids: number[], mask: number[]): number[] {
  const size = spec.hidden_size;
  const hidden = ids.map((token) =>
    Array.from({ length: size }, (_, column) => {
      let total = 0;
      for (let inner = 0; inner < size; inner += 1) {
        total += spec.embeddings[token]![inner]! * spec.projection[inner]![column]!;
      }
      return Math.tanh(total + spec.projection_bias[column]!);
    }),
  );
  const spans: number[][] = [];
  ids.forEach((token, position) => {
    if (mask[position] !== 1 || token !== spec.label_token_id) {
      return;
    }
    let end = position + 1;
    while (end < ids.length && ids[end] !== spec.label_token_id && ids[end] !== spec.separator_token_id) {
      end += 1;
    }
    spans.push(Array.from({ length: end - position }, (_, offset) => position + offset));
  });
  const inLabel = new Set(spans.flat());
  const context = ids
    .map((_, position) => position)
    .filter(
      (position) =>
        mask[position] === 1 &&
        !inLabel.has(position) &&
        !spec.ignored_token_ids.includes(ids[position]!),
    );
  const mean = (positions: number[]) => {
    const totals = new Array<number>(size).fill(0);
    for (const position of positions) {
      for (let column = 0; column < size; column += 1) {
        totals[column]! += hidden[position]![column]!;
      }
    }
    return positions.length === 0 ? totals : totals.map((value) => value / positions.length);
  };
  const text = mean(context);
  return spans.map((span) => {
    const label = mean(span.filter((position) => mask[position] === 1));
    let dot = 0;
    for (let column = 0; column < size; column += 1) {
      dot += label[column]! * text[column]!;
    }
    return spec.logit_scale * dot + spec.logit_bias;
  });
}

export class FakeTensor implements OrtTensorLike {
  [key: string]: unknown;
  constructor(
    readonly type: string,
    readonly data: unknown,
    readonly dims: readonly number[],
  ) {}
}

export const Tensor = FakeTensor as unknown as OrtTensorConstructor;

export function tinySession(
  spec: EncoderSpec,
  fault: string | null,
): OrtInferenceSession & { calls: { ids: number[]; logits: number[] | null; type: string }[] } {
  const calls: { ids: number[]; logits: number[] | null; type: string }[] = [];
  return {
    calls,
    async run(feeds) {
      const inputIds = feeds.input_ids as FakeTensor;
      const attention = feeds.attention_mask as FakeTensor;
      assert.deepEqual(inputIds.dims, attention.dims);
      const ids = Array.from(inputIds.data as ArrayLike<bigint | number>, Number);
      const mask = Array.from(attention.data as ArrayLike<bigint | number>, Number);
      if (fault === "raise") {
        calls.push({ ids, logits: null, type: inputIds.type });
        throw new Error("synthetic encoder fault");
      }
      let logits = Float32Array.from(tinyLogits(spec, ids, mask));
      if (fault === "short") {
        logits = logits.slice(0, -1);
      }
      if (fault === "non_finite") {
        logits[0] = Number.NaN;
      }
      calls.push({
        ids,
        logits: fault === "non_finite" ? null : Array.from(logits),
        type: inputIds.type,
      });
      if (fault === "missing") {
        return {};
      }
      const dims = fault === "rank3" ? [1, 1, logits.length] : [1, logits.length];
      return { logits: new FakeTensor("float32", logits, dims) };
    },
  };
}

export function configuration(settings: RuntimeSettings): GLiClassDecisionConfiguration {
  return {
    prompt: settings.prompt,
    examples: settings.examples,
    scoreLevels: settings.score_levels,
    promptFirst: settings.prompt_first,
    maxLength: settings.max_length,
  };
}

export function assertClose(actual: unknown, expected: unknown, tolerance: number, path: string): void {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", path);
    assert.ok(Math.abs((actual as number) - expected) <= tolerance, `${path}: ${String(actual)}`);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), path);
    assert.equal(actual.length, expected.length, path);
    expected.forEach((item, index) => assertClose(actual[index], item, tolerance, `${path}[${index}]`));
  } else if (expected !== null && typeof expected === "object") {
    assert.ok(actual !== null && typeof actual === "object" && !Array.isArray(actual), path);
    assert.deepEqual(Object.keys(actual as object).sort(), Object.keys(expected).sort(), path);
    for (const [key, value] of Object.entries(expected)) {
      assertClose((actual as Record<string, unknown>)[key], value, tolerance, `${path}.${key}`);
    }
  } else {
    assert.equal(actual, expected, path);
  }
}
