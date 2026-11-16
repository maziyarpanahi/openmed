// Patient-keyed date-shift offsets in the web package, replayed from the
// shared synthetic vectors in tests/fixtures/parity/date_shift_v1.json
// (Python reference: openmed.core.date_shift). Mismatches are reported by
// vector id only.
import assert from "node:assert/strict";
import { createHash, createHmac } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import {
  DEFAULT_DATE_SHIFT_MAX_DAYS,
  DateShiftError,
  shiftIsoDate,
  stableDateShiftOffset,
  stableDateShiftOffsetFromSeed,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "date_shift_v1.json");
const SOURCE_PATH = join(rootDir, "js", "openmedkit-web", "src", "date-shift.ts");
const DIST_DIR = join(rootDir, "js", "openmedkit-web", "dist");

interface Secret {
  id: string;
  test_only: boolean;
  utf8?: string;
  hex: string;
}

interface PatientVector {
  id: string;
  patient_key: string;
  secret_id: string;
  max_days: number;
  digest_sha256_hex: string;
  offset_days: number;
}

interface SeedVector {
  id: string;
  seed?: number;
  seed_decimal: string;
  max_days: number;
  digest_sha256_hex: string;
  offset_days: number;
}

/** A date vector takes its offset from a patient key or from a seed. */
interface DateVector {
  id: string;
  patient_key?: string;
  secret_id?: string;
  seed_decimal?: string;
  max_days: number;
  offset_days: number;
  source_date: string;
  shifted_date: string;
}

interface Vectors {
  synthetic_only: boolean;
  version: number;
  algorithm: { seed_domain_hex: string };
  secrets: Secret[];
  max_days_values: number[];
  patient_key_offsets: PatientVector[];
  seed_offsets: SeedVector[];
  date_shifts: DateVector[];
}

let cached: Promise<Vectors> | undefined;
function vectors(): Promise<Vectors> {
  cached ??= readFile(VECTORS_PATH, "utf8").then((text) => JSON.parse(text) as Vectors);
  return cached;
}

function secretBytes(fixture: Vectors, id: string): Uint8Array {
  const secret = fixture.secrets.find((item) => item.id === id);
  assert.ok(secret?.test_only, id);
  return new Uint8Array(Buffer.from(secret.hex, "hex"));
}

/** Every form a caller may pass the secret in: bytes, and text when UTF-8. */
function secretForms(fixture: Vectors, id: string): (string | Uint8Array)[] {
  const secret = fixture.secrets.find((item) => item.id === id)!;
  return secret.utf8 === undefined
    ? [secretBytes(fixture, id)]
    : [secret.utf8, secretBytes(fixture, id)];
}

async function codeOf(action: () => unknown): Promise<string> {
  try {
    await action();
  } catch (error) {
    assert.ok(error instanceof DateShiftError);
    assert.equal(error.message, `OpenMed date shift failed: ${error.code}`);
    return error.code;
  }
  return "none";
}

test("the vectors are synthetic and use the Python seed domain", async () => {
  const fixture = await vectors();
  assert.equal(fixture.synthetic_only, true);
  assert.equal(fixture.version, 1);
  assert.equal(
    Buffer.from(fixture.algorithm.seed_domain_hex, "hex").toString("latin1"),
    "openmed-date-shift-seed-v1\u0000",
  );
  assert.ok(fixture.secrets.every((secret) => secret.test_only));
  assert.equal(DEFAULT_DATE_SHIFT_MAX_DAYS, 365);
});

test("every patient-keyed vector matches the Python offset", async () => {
  const fixture = await vectors();
  const mismatches: string[] = [];
  for (const vector of fixture.patient_key_offsets) {
    // The recorded digest is the HMAC the offset is reduced from.
    const digest = createHmac("sha256", secretBytes(fixture, vector.secret_id))
      .update(Buffer.from(vector.patient_key, "utf8"))
      .digest("hex");
    assert.equal(digest, vector.digest_sha256_hex, vector.id);
    for (const secret of secretForms(fixture, vector.secret_id)) {
      for (const patientKey of [vector.patient_key, new TextEncoder().encode(vector.patient_key)]) {
        const offset = await stableDateShiftOffset({
          patientKey,
          secret,
          maxDays: vector.max_days,
        });
        if (offset !== vector.offset_days) {
          mismatches.push(vector.id);
        }
      }
    }
  }
  assert.deepEqual([...new Set(mismatches)], []);
  assert.ok(fixture.patient_key_offsets.length >= 40);
});

test("every seeded vector matches, including seeds beyond 2^53", async () => {
  const fixture = await vectors();
  const mismatches: string[] = [];
  let beyondSafe = 0;
  for (const vector of fixture.seed_offsets) {
    const digest = createHash("sha256")
      .update(Buffer.from(fixture.algorithm.seed_domain_hex, "hex"))
      .update(vector.seed_decimal, "ascii")
      .digest("hex");
    assert.equal(digest, vector.digest_sha256_hex, vector.id);
    const forms: (number | bigint | string)[] = [vector.seed_decimal, BigInt(vector.seed_decimal)];
    if (vector.seed === undefined) {
      beyondSafe += 1;
    } else {
      forms.push(vector.seed);
    }
    for (const seed of forms) {
      if ((await stableDateShiftOffsetFromSeed(seed, vector.max_days)) !== vector.offset_days) {
        mismatches.push(vector.id);
      }
    }
  }
  assert.deepEqual([...new Set(mismatches)], []);
  assert.ok(beyondSafe > 0, "vectors include seeds beyond the safe-integer range");
});

test("every date vector shifts by its patient or seed offset to the expected ISO date", async () => {
  const fixture = await vectors();
  const mismatches: string[] = [];
  let seeded = 0;
  for (const vector of fixture.date_shifts) {
    let offset: number;
    if (vector.seed_decimal !== undefined) {
      seeded += 1;
      offset = await stableDateShiftOffsetFromSeed(vector.seed_decimal, vector.max_days);
    } else {
      offset = await stableDateShiftOffset({
        patientKey: vector.patient_key!,
        secret: secretBytes(fixture, vector.secret_id!),
        maxDays: vector.max_days,
      });
    }
    if (offset !== vector.offset_days || shiftIsoDate(vector.source_date, offset) !== vector.shifted_date) {
      mismatches.push(vector.id);
    }
    // Shifting back restores the source date.
    assert.equal(shiftIsoDate(vector.shifted_date, -vector.offset_days), vector.source_date, vector.id);
  }
  assert.deepEqual(mismatches, []);
  assert.ok(seeded > 0 && seeded < fixture.date_shifts.length);
});

test("ISO date arithmetic follows the proleptic Gregorian calendar in UTC", () => {
  for (const [date, days, expected] of [
    ["2024-02-28", 1, "2024-02-29"],
    ["2023-02-28", 1, "2023-03-01"],
    ["1900-02-28", 1, "1900-03-01"],
    ["2000-02-28", 1, "2000-02-29"],
    ["0001-01-01", 0, "0001-01-01"],
    ["0001-01-02", -1, "0001-01-01"],
    ["9999-12-30", 1, "9999-12-31"],
    ["1970-01-01", -1, "1969-12-31"],
    ["2026-03-29", 1, "2026-03-30"],
    ["2024-01-31", 365, "2025-01-30"],
    ["2024-12-31", -366, "2023-12-31"],
  ] as const) {
    assert.equal(shiftIsoDate(date, days), expected, `${date}${days}`);
  }
});

test("invalid input fails with codes that never echo keys, secrets, seeds or dates", async () => {
  const sentinel = "Quorble-Zyxwick-0042";
  const cases: [() => unknown, string][] = [
    [() => stableDateShiftOffset({ patientKey: sentinel, secret: sentinel, maxDays: 0 }), "max_days_invalid"],
    [() => stableDateShiftOffset({ patientKey: sentinel, secret: sentinel, maxDays: -3 }), "max_days_invalid"],
    [() => stableDateShiftOffset({ patientKey: sentinel, secret: sentinel, maxDays: 1.5 }), "max_days_invalid"],
    [
      () => stableDateShiftOffset({ patientKey: sentinel, secret: sentinel, maxDays: "365" as never }),
      "max_days_invalid",
    ],
    [() => stableDateShiftOffset({ patientKey: sentinel, secret: sentinel, maxDays: true as never }), "max_days_invalid"],
    [() => stableDateShiftOffset({ patientKey: "", secret: sentinel }), "patient_key_invalid"],
    [() => stableDateShiftOffset({ patientKey: new Uint8Array(), secret: sentinel }), "patient_key_invalid"],
    [() => stableDateShiftOffset({ patientKey: 42 as never, secret: sentinel }), "patient_key_invalid"],
    [() => stableDateShiftOffset({ patientKey: sentinel, secret: "" }), "secret_invalid"],
    [() => stableDateShiftOffset({ patientKey: sentinel, secret: null as never }), "secret_invalid"],
    [() => stableDateShiftOffset(undefined as never), "patient_key_invalid"],
    [() => stableDateShiftOffsetFromSeed(2 ** 53), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed(1.5), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed(Number.NaN), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed("01"), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed("-0"), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed("+1"), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed("1e3"), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed(` ${sentinel}`), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed(true as never), "seed_invalid"],
    [() => stableDateShiftOffsetFromSeed(7, 0), "max_days_invalid"],
    [() => shiftIsoDate("2024-02-30", 1), "date_invalid"],
    [() => shiftIsoDate("2023-02-29", 1), "date_invalid"],
    [() => shiftIsoDate("0000-01-01", 1), "date_invalid"],
    [() => shiftIsoDate("2024-1-01", 1), "date_invalid"],
    [() => shiftIsoDate("2024-01-01T00:00:00Z", 1), "date_invalid"],
    [() => shiftIsoDate(`${sentinel}`, 1), "date_invalid"],
    [() => shiftIsoDate(20240101 as never, 1), "date_invalid"],
    [() => shiftIsoDate("2024-01-01", 0.5), "days_invalid"],
    [() => shiftIsoDate("2024-01-01", Number.POSITIVE_INFINITY), "days_invalid"],
    [() => shiftIsoDate("9999-12-31", 1), "date_out_of_range"],
    [() => shiftIsoDate("0001-01-01", -1), "date_out_of_range"],
  ];
  for (const [action, expected] of cases) {
    let caught: unknown;
    try {
      await action();
    } catch (error) {
      caught = error;
    }
    assert.ok(caught instanceof DateShiftError, expected);
    assert.equal(caught.code, expected);
    assert.equal(String(caught).includes(sentinel), false);
    assert.equal(JSON.stringify(caught).includes(sentinel), false);
  }
  // The caller's key and secret bytes are left as they were.
  const secret = new TextEncoder().encode("synthetic-secret");
  const copy = Uint8Array.from(secret);
  await stableDateShiftOffset({ patientKey: "SYNTH-PATIENT-0001", secret });
  assert.deepEqual(secret, copy);
});

test("a missing WebCrypto fails with a code", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  assert.ok(descriptor?.configurable, "Node exposes globalThis.crypto as configurable");
  Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true });
  try {
    assert.equal(
      await codeOf(() => stableDateShiftOffset({ patientKey: "SYNTH-PATIENT-0001", secret: "s" })),
      "crypto_unavailable",
    );
    assert.equal(await codeOf(() => stableDateShiftOffsetFromSeed(1)), "crypto_unavailable");
  } finally {
    Object.defineProperty(globalThis, "crypto", descriptor);
  }
});

test("the helpers work from the ESM and CJS builds and import nothing from Node", async () => {
  const fixture = await vectors();
  const patient = fixture.patient_key_offsets[0]!;
  const seed = fixture.seed_offsets.at(-1)!;
  const date = fixture.date_shifts.find((item) => item.patient_key !== undefined)!;
  const esm = await import(pathToFileURL(join(DIST_DIR, "index.js")).href);
  const cjs = createRequire(import.meta.url)(join(DIST_DIR, "index.cjs"));
  for (const build of [esm, cjs]) {
    assert.equal(
      await build.stableDateShiftOffset({
        patientKey: patient.patient_key,
        secret: secretBytes(fixture, patient.secret_id),
        maxDays: patient.max_days,
      }),
      patient.offset_days,
    );
    assert.equal(
      await build.stableDateShiftOffsetFromSeed(seed.seed_decimal, seed.max_days),
      seed.offset_days,
    );
    assert.equal(build.shiftIsoDate(date.source_date, date.offset_days), date.shifted_date);
    assert.throws(() => build.shiftIsoDate("2024-02-30", 1), {
      name: "DateShiftError",
      code: "date_invalid",
    });
  }
  const source = await readFile(SOURCE_PATH, "utf8");
  assert.doesNotMatch(source, /from\s+["']node:|require\(|import\(["']node:/);
});
