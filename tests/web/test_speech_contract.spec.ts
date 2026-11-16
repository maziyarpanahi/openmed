// Shared streaming speech contract: the synthetic fixture that Python and
// OpenMedKit Swift both check (tests/fixtures/parity/speech_contract_v1.json).
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { inspect } from "node:util";

import {
  SPEECH_ALIGNMENT_STATUSES,
  SPEECH_CONTRACT_ERROR_CODES,
  SPEECH_CONTRACT_LIMITS,
  SpeechContractError,
  SpeechEngineProvenance,
  SpeechFinalizedSegment,
  SpeechPartialHypothesis,
  SpeechRecognition,
  SpeechTokenTiming,
  scalarCount,
  type SpeechEngineProvenanceInit,
  type SpeechTokenTimingInit,
} from "../../js/openmedkit-web/src/speech/index";
import * as mainEntry from "../../js/openmedkit-web/src/index";

interface TokenRow {
  token_index: number;
  start_ms: number;
  end_ms: number;
  confidence_ppm: number;
  token: string;
}

interface Fixture {
  synthetic_only: boolean;
  version: number;
  provenance: Record<"provider_id" | "provider_revision" | "model_fingerprint" | "language", string>;
  invalid_provenance: Array<{ field: string; value: string }>;
  segment: {
    segment_index: number;
    text: string;
    start_ms: number;
    end_ms: number;
    token_times: TokenRow[];
  };
  invalid_tokens: Array<Partial<Record<keyof TokenRow, number>>>;
}

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const fixturePath = join(rootDir, "tests", "fixtures", "parity", "speech_contract_v1.json");
const distDir = join(rootDir, "js", "openmedkit-web", "dist");

// Fixture (Python) field names and their web counterparts.
const TOKEN_FIELDS: Record<string, keyof SpeechTokenTimingInit> = {
  token_index: "tokenIndex",
  start_ms: "startMilliseconds",
  end_ms: "endMilliseconds",
  confidence_ppm: "confidencePPM",
  token: "text",
};
const PROVENANCE_FIELDS: Record<string, keyof SpeechEngineProvenanceInit> = {
  provider_id: "providerId",
  provider_revision: "providerRevision",
  model_fingerprint: "modelFingerprint",
  language: "language",
};

let fixtureCache: Promise<Fixture> | undefined;

function fixture(): Promise<Fixture> {
  fixtureCache ??= readFile(fixturePath, "utf8").then((text) => JSON.parse(text) as Fixture);
  return fixtureCache;
}

function tokenInit(row: Record<string, unknown>): SpeechTokenTimingInit {
  return Object.fromEntries(
    Object.entries(row).map(([name, value]) => [TOKEN_FIELDS[name] ?? name, value]),
  ) as unknown as SpeechTokenTimingInit;
}

function provenanceInit(row: Record<string, string>): SpeechEngineProvenanceInit {
  return Object.fromEntries(
    Object.entries(row).map(([name, value]) => [PROVENANCE_FIELDS[name] ?? name, value]),
  ) as unknown as SpeechEngineProvenanceInit;
}

async function fixtureTokens(): Promise<SpeechTokenTiming[]> {
  const { segment } = await fixture();
  return segment.token_times.map((row) => new SpeechTokenTiming(tokenInit({ ...row })));
}

function contractError(code: string, field: string, ...forbidden: string[]) {
  return (error: unknown) => {
    assert.ok(error instanceof SpeechContractError, field);
    assert.equal(error.code, code, field);
    assert.equal(error.field, field);
    assert.equal(error.message, `${code} at ${field}`);
    for (const value of forbidden) {
      assert.equal(error.message.includes(value), false, `${field}: message leaks a value`);
    }
    return true;
  };
}

test("fixture is the synthetic v1 corpus", async () => {
  const corpus = await fixture();
  assert.equal(corpus.version, 1);
  assert.equal(corpus.synthetic_only, true);
});

test("fixture tokens construct with shared units and protected text", async () => {
  const { segment } = await fixture();
  const tokens = await fixtureTokens();
  const recognition = new SpeechRecognition({ text: segment.text, tokenTimes: tokens });

  // Code points, not UTF-16 units: the emoji is one scalar but two code units.
  assert.equal(segment.text.length, 7);
  assert.equal(recognition.textLength, 6);
  assert.equal(scalarCount(segment.text), 6);
  assert.deepEqual(
    recognition.tokenTimes.map((token) => token.startMilliseconds),
    [500, 650],
  );
  assert.deepEqual(
    recognition.tokenTimes.map((token) => token.confidencePPM),
    [750_000, 250_000],
  );
  assert.deepEqual(
    recognition.tokenTimes.map((token) => token.textLength),
    [4, 2],
  );
  assert.equal(recognition.alignmentStatus, "aligned");
  assert.equal(recognition.text, segment.text);
  assert.equal(recognition.tokenTimes[0]?.text, segment.token_times[0]?.token);
});

test("fixture segment matches Python's FinalizedSegment and PartialHypothesis", async () => {
  const { segment } = await fixture();
  const tokens = await fixtureTokens();
  const finalized = new SpeechFinalizedSegment({
    segmentIndex: segment.segment_index,
    text: segment.text,
    startMilliseconds: segment.start_ms,
    endMilliseconds: segment.end_ms,
    tokenTimes: tokens,
  });
  assert.equal(finalized.durationMilliseconds, 400);
  assert.equal(finalized.toJSON().text_length, 6);
  assert.equal(finalized.toJSON().token_count, 2);
  assert.equal(finalized.allowsAutomaticClinicalFinalization, false);

  const partial = new SpeechPartialHypothesis({
    revision: 2,
    text: segment.text,
    tokenTimes: tokens,
    windowStartMilliseconds: segment.start_ms,
    windowEndMilliseconds: segment.end_ms,
  });
  assert.equal(partial.toJSON().token_count, 2);
  assert.equal(partial.isFinal, false);

  for (const [field, init] of [
    ["tokenTimes[0]", { startMilliseconds: 501 }],
    ["tokenTimes[1]", { endMilliseconds: 899 }],
  ] as const) {
    assert.throws(
      () =>
        new SpeechFinalizedSegment({
          segmentIndex: 0,
          text: segment.text,
          startMilliseconds: init.startMilliseconds ?? 500,
          endMilliseconds: init.endMilliseconds ?? 900,
          tokenTimes: tokens,
        }),
      contractError("speech_hypothesis_invalid", field, segment.text),
    );
    assert.throws(
      () =>
        new SpeechPartialHypothesis({
          revision: 0,
          text: segment.text,
          tokenTimes: tokens,
          windowStartMilliseconds: init.startMilliseconds ?? 500,
          windowEndMilliseconds: init.endMilliseconds ?? 900,
        }),
      contractError("speech_hypothesis_invalid", field, segment.text),
    );
  }
});

test("every invalid fixture token throws a coded error for its field", async () => {
  const { segment, invalid_tokens: patches } = await fixture();
  const [first] = segment.token_times;
  assert.ok(first);
  assert.ok(patches.length > 0);
  for (const patch of patches) {
    const [name] = Object.keys(patch);
    assert.ok(name);
    const field = TOKEN_FIELDS[name];
    assert.ok(field, name);
    assert.throws(
      () => new SpeechTokenTiming(tokenInit({ ...first, ...patch })),
      contractError(
        "speech_hypothesis_invalid",
        field,
        first.token,
        String(patch[name as keyof TokenRow]),
      ),
    );
  }
});

test("fixture provenance validates and every invalid field is refused", async () => {
  const { provenance, invalid_provenance: patches } = await fixture();
  const valid = new SpeechEngineProvenance(provenanceInit(provenance));
  assert.equal(valid.providerId, provenance.provider_id);
  assert.equal(valid.language, provenance.language);
  assert.ok(patches.length > 0);
  for (const patch of patches) {
    const field = PROVENANCE_FIELDS[patch.field];
    assert.ok(field, patch.field);
    assert.throws(
      () =>
        new SpeechEngineProvenance(
          provenanceInit({ ...provenance, [patch.field]: patch.value }),
        ),
      contractError("speech_artifact_unverified", field, patch.value),
    );
  }
});

test("recognition enforces token order, counts and alignment status", async () => {
  const tokens = await fixtureTokens();
  const [first, second] = tokens;
  assert.ok(first && second);
  const token = (init: Partial<SpeechTokenTimingInit>) =>
    new SpeechTokenTiming({
      tokenIndex: 1,
      startMilliseconds: 600,
      endMilliseconds: 700,
      confidencePPM: 0,
      text: "",
      ...init,
    });
  const invalid: Array<[string, () => unknown]> = [
    ["tokenTimes[1]", () => new SpeechRecognition({ text: "", tokenTimes: [second, first] })],
    [
      "tokenTimes[1]",
      () => new SpeechRecognition({ text: "", tokenTimes: [first, token({ tokenIndex: 0 })] }),
    ],
    [
      "tokenTimes[1]",
      () =>
        new SpeechRecognition({
          text: "",
          tokenTimes: [first, token({ startMilliseconds: 499, endMilliseconds: 700 })],
        }),
    ],
    [
      "tokenTimes[1]",
      () =>
        new SpeechRecognition({
          text: "",
          tokenTimes: [first, token({ startMilliseconds: 600, endMilliseconds: 600 })],
        }),
    ],
    [
      "tokenTimes[0]",
      () =>
        new SpeechRecognition({
          text: "",
          tokenTimes: [{ ...first } as unknown as SpeechTokenTiming],
        }),
    ],
    [
      "tokenTimes",
      () =>
        new SpeechRecognition({
          text: "",
          tokenTimes: Array.from({ length: SPEECH_CONTRACT_LIMITS.maxTokenTimes + 1 }, () => first),
        }),
    ],
    ["alignmentStatus", () => new SpeechRecognition({ text: "", alignmentStatus: "aligned" })],
    [
      "alignmentStatus",
      () =>
        new SpeechRecognition({ text: "", tokenTimes: tokens, alignmentStatus: "outsideWindow" }),
    ],
    [
      "alignmentStatus",
      () =>
        new SpeechRecognition({
          text: "",
          alignmentStatus: "estimated" as unknown as "aligned",
        }),
    ],
    [
      "decoderConfidencePPM",
      () => new SpeechRecognition({ text: "", decoderConfidencePPM: 1_000_001 }),
    ],
    [
      "text",
      () => new SpeechRecognition({ text: "a".repeat(SPEECH_CONTRACT_LIMITS.maxTextScalars + 1) }),
    ],
  ];
  for (const [field, build] of invalid) {
    assert.throws(build, contractError("speech_hypothesis_invalid", field));
  }

  // 8,192 astral code points are 16,384 UTF-16 units and still within limits.
  const wide = new SpeechRecognition({ text: "\u{1F600}".repeat(8192) });
  assert.equal(wide.textLength, 8192);
  assert.equal(wide.alignmentStatus, "unavailable");
  for (const status of ["unavailable", "outsideWindow", "unverifiedTokenization"] as const) {
    assert.equal(new SpeechRecognition({ text: "", alignmentStatus: status }).alignmentStatus, status);
  }
  assert.deepEqual([...SPEECH_ALIGNMENT_STATUSES], [
    "unavailable",
    "aligned",
    "outsideWindow",
    "unverifiedTokenization",
  ]);
});

test("numeric fields accept only bounded integers", async () => {
  const { segment } = await fixture();
  const base = tokenInit({ ...segment.token_times[0] });
  for (const [field, value] of [
    ["tokenIndex", 4097],
    ["tokenIndex", -1],
    ["tokenIndex", 1.5],
    ["startMilliseconds", Number.NaN],
    ["startMilliseconds", "500"],
    ["endMilliseconds", Number.POSITIVE_INFINITY],
    ["confidencePPM", true],
    ["confidencePPM", -1],
  ] as const) {
    assert.throws(
      () => new SpeechTokenTiming({ ...base, [field]: value } as SpeechTokenTimingInit),
      contractError("speech_hypothesis_invalid", field),
    );
  }
  assert.throws(
    () => new SpeechTokenTiming({ ...base, text: 7 as unknown as string }),
    contractError("speech_hypothesis_invalid", "text"),
  );
  const boundary = new SpeechTokenTiming({
    ...base,
    tokenIndex: 4096,
    startMilliseconds: 86_400_000,
    endMilliseconds: 86_400_000,
    confidencePPM: 1_000_000,
  });
  assert.equal(boundary.endMilliseconds, SPEECH_CONTRACT_LIMITS.maxStreamMilliseconds);
});

test("shifting a token keeps it inside the 24 hour stream", async () => {
  const [first] = await fixtureTokens();
  assert.ok(first);
  const moved = first.shifted(100);
  assert.deepEqual([moved.startMilliseconds, moved.endMilliseconds], [600, 750]);
  assert.equal(moved.text, first.text);
  assert.throws(
    () => first.shifted(86_400_000 - 649),
    contractError("speech_hypothesis_invalid", "shift"),
  );
  assert.throws(() => first.shifted(-1), contractError("speech_hypothesis_invalid", "shift"));
});

test("transcript text never appears in errors, JSON, strings or inspection", async () => {
  const { segment } = await fixture();
  const tokens = await fixtureTokens();
  const values = [
    ...tokens,
    new SpeechRecognition({ text: segment.text, tokenTimes: tokens }),
    new SpeechFinalizedSegment({
      segmentIndex: 0,
      text: segment.text,
      startMilliseconds: 500,
      endMilliseconds: 900,
      tokenTimes: tokens,
    }),
    new SpeechPartialHypothesis({
      revision: 0,
      text: segment.text,
      windowStartMilliseconds: 500,
      windowEndMilliseconds: 900,
    }),
  ];
  const secrets = [segment.text, ...segment.token_times.map((row) => row.token), "Caf"];
  for (const value of values) {
    for (const rendered of [
      JSON.stringify(value),
      String(value),
      `${value}`,
      inspect(value, { depth: 4, showHidden: true }),
      inspect([value]),
      Object.keys(value).join(","),
    ]) {
      for (const secret of secrets) {
        assert.equal(rendered.includes(secret), false, value.constructor.name);
      }
    }
    assert.ok(Object.isFrozen(value));
  }
  assert.throws(
    () =>
      new SpeechRecognition({
        text: segment.text,
        tokenTimes: [...tokens].reverse(),
      }),
    contractError("speech_hypothesis_invalid", "tokenTimes[1]", ...secrets),
  );
  assert.deepEqual([...SPEECH_CONTRACT_ERROR_CODES], [
    "speech_hypothesis_invalid",
    "speech_artifact_unverified",
  ]);
});

test("the speech contract stays off the main entry and builds as a subpath", async () => {
  assert.equal(Object.hasOwn(mainEntry, "SpeechTokenTiming"), false);
  assert.equal(Object.hasOwn(mainEntry, "SpeechRecognition"), false);
  const built = await import(pathToFileURL(join(distDir, "speech", "index.js")).href);
  const token = new built.SpeechTokenTiming({
    tokenIndex: 0,
    startMilliseconds: 0,
    endMilliseconds: 1,
    confidencePPM: 0,
    text: "",
  });
  assert.equal(token.endMilliseconds, 1);
  const mainBundle = await readFile(join(distDir, "index.js"), "utf8");
  assert.equal(mainBundle.includes("speech_hypothesis_invalid"), false);
  assert.equal(mainBundle.includes("SpeechTokenTiming"), false);
});
