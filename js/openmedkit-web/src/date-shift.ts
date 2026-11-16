// Patient-keyed date-shift offsets, ported from openmed/core/date_shift.py.
//
// A patient's offset is HMAC-SHA256(secret, patient key) reduced modulo
// 2 * maxDays with BigInt, then mapped to a non-zero signed day count in
// [-maxDays, maxDays]. A seeded offset hashes a domain-separated decimal seed
// with SHA-256 instead. Hashing uses WebCrypto (`crypto.subtle`) only, so the
// helpers run in browsers, workers and Node without Node-only imports. ISO
// dates are shifted with exact proleptic-Gregorian day arithmetic in UTC.
// Errors carry a code only: never a key, secret, seed or date.

/** Default largest absolute offset, as in Python. */
export const DEFAULT_DATE_SHIFT_MAX_DAYS = 365;

const SEED_DOMAIN = "openmed-date-shift-seed-v1\u0000";
const CANONICAL_SEED = /^-?(?:0|[1-9][0-9]*)$/;
const ISO_DATE = /^([0-9]{4})-([0-9]{2})-([0-9]{2})$/;
const MIN_YEAR = 1;
const MAX_YEAR = 9999;

export type DateShiftErrorCode =
  | "max_days_invalid"
  | "patient_key_invalid"
  | "secret_invalid"
  | "seed_invalid"
  | "date_invalid"
  | "days_invalid"
  | "date_out_of_range"
  | "crypto_unavailable";

/** A value-free date-shift failure: the message names the code only. */
export class DateShiftError extends Error {
  readonly code: DateShiftErrorCode;

  constructor(code: DateShiftErrorCode) {
    super(`OpenMed date shift failed: ${code}`);
    this.name = "DateShiftError";
    this.code = code;
  }
}

export interface StableDateShiftOffsetOptions {
  /** The caller's stable patient key, as text (UTF-8) or bytes. Non-empty. */
  patientKey: string | Uint8Array;
  /** The non-empty HMAC secret, as text (UTF-8) or bytes. */
  secret: string | Uint8Array;
  /** Largest absolute offset in days, a positive integer; default 365. */
  maxDays?: number;
}

function subtle(): SubtleCrypto {
  const candidate = (globalThis as { crypto?: Crypto }).crypto?.subtle;
  if (candidate === undefined) {
    throw new DateShiftError("crypto_unavailable");
  }
  return candidate;
}

/** Copy bytes into a fresh ArrayBuffer that this module owns and can zero. */
function ownedBytes(source: Uint8Array): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(source.byteLength));
  bytes.set(source);
  return bytes;
}

function nonEmptyBytes(
  value: unknown,
  code: "patient_key_invalid" | "secret_invalid",
): Uint8Array<ArrayBuffer> {
  let bytes: Uint8Array<ArrayBuffer> | undefined;
  if (typeof value === "string") {
    bytes = ownedBytes(new TextEncoder().encode(value));
  } else if (value instanceof Uint8Array) {
    bytes = ownedBytes(value);
  }
  if (bytes === undefined || bytes.byteLength === 0) {
    throw new DateShiftError(code);
  }
  return bytes;
}

function validateMaxDays(maxDays: unknown): number {
  if (typeof maxDays !== "number" || !Number.isSafeInteger(maxDays) || maxDays <= 0) {
    throw new DateShiftError("max_days_invalid");
  }
  return maxDays;
}

function offsetFromDigest(digest: ArrayBuffer, maxDays: number): number {
  let value = 0n;
  for (const byte of new Uint8Array(digest)) {
    value = (value << 8n) | BigInt(byte);
  }
  const bucket = Number(value % BigInt(maxDays * 2));
  return bucket < maxDays ? bucket - maxDays : bucket - maxDays + 1;
}

async function guarded<T>(operation: () => Promise<T>): Promise<T> {
  try {
    return await operation();
  } catch {
    // A WebCrypto failure may describe its inputs; report the code only.
    throw new DateShiftError("crypto_unavailable");
  }
}

/**
 * Return a deterministic non-zero signed day offset for a patient key.
 *
 * Matches Python's `stable_offset_for(patient_key, max_days=..., secret=...)`.
 * The patient key is used only as the HMAC message; nothing is stored.
 *
 * Args:
 *   options: The patient key, the secret and the largest absolute offset.
 *
 * Returns:
 *   An integer in `[-maxDays, -1]` or `[1, maxDays]`.
 *
 * Throws:
 *   DateShiftError: `max_days_invalid`, `patient_key_invalid`,
 *   `secret_invalid` or `crypto_unavailable`.
 */
export async function stableDateShiftOffset(
  options: StableDateShiftOffsetOptions,
): Promise<number> {
  const maxDays = validateMaxDays(options?.maxDays ?? DEFAULT_DATE_SHIFT_MAX_DAYS);
  const message = nonEmptyBytes(options?.patientKey, "patient_key_invalid");
  const secret = nonEmptyBytes(options?.secret, "secret_invalid");
  const crypto = subtle();
  try {
    const digest = await guarded(async () => {
      const key = await crypto.importKey(
        "raw",
        secret,
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["sign"],
      );
      return crypto.sign("HMAC", key, message);
    });
    return offsetFromDigest(digest, maxDays);
  } finally {
    secret.fill(0);
    message.fill(0);
  }
}

function canonicalSeed(seed: unknown): string {
  if (typeof seed === "bigint") {
    return seed.toString();
  }
  if (typeof seed === "number") {
    // Only integers a double holds exactly have one decimal form.
    if (!Number.isSafeInteger(seed)) {
      throw new DateShiftError("seed_invalid");
    }
    return String(seed);
  }
  if (typeof seed === "string" && CANONICAL_SEED.test(seed) && seed !== "-0") {
    return seed;
  }
  throw new DateShiftError("seed_invalid");
}

/**
 * Return a deterministic non-zero day offset for a request seed.
 *
 * Matches Python's `stable_offset_from_seed(seed, max_days=...)`: SHA-256 of
 * the seed domain followed by the seed's canonical decimal text.
 *
 * Args:
 *   seed: An integer seed: a safe-integer number, a bigint, or its canonical
 *     decimal text (no leading zeros, no `+`, no `-0`), which keeps seeds
 *     beyond 2^53 exact.
 *   maxDays: Largest absolute offset in days; default 365.
 *
 * Returns:
 *   An integer in `[-maxDays, -1]` or `[1, maxDays]`.
 *
 * Throws:
 *   DateShiftError: `seed_invalid`, `max_days_invalid` or
 *   `crypto_unavailable`.
 */
export async function stableDateShiftOffsetFromSeed(
  seed: number | bigint | string,
  maxDays: number = DEFAULT_DATE_SHIFT_MAX_DAYS,
): Promise<number> {
  const decimal = canonicalSeed(seed);
  const limit = validateMaxDays(maxDays);
  const crypto = subtle();
  const digest = await guarded(async () =>
    crypto.digest("SHA-256", ownedBytes(new TextEncoder().encode(`${SEED_DOMAIN}${decimal}`))),
  );
  return offsetFromDigest(digest, limit);
}

function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

function daysInMonth(year: number, month: number): number {
  const lengths = [31, isLeapYear(year) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return lengths[month - 1]!;
}

// Days since 1970-01-01 in the proleptic Gregorian calendar (H. Hinnant).
function daysFromCivil(year: number, month: number, day: number): number {
  const y = month <= 2 ? year - 1 : year;
  const era = Math.floor(y / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

function civilFromDays(days: number): [number, number, number] {
  const z = days + 719468;
  const era = Math.floor(z / 146097);
  const doe = z - era * 146097;
  const yoe = Math.floor(
    (doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365,
  );
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100));
  const mp = Math.floor((5 * doy + 2) / 153);
  const day = doy - Math.floor((153 * mp + 2) / 5) + 1;
  const month = mp < 10 ? mp + 3 : mp - 9;
  const year = yoe + era * 400 + (month <= 2 ? 1 : 0);
  return [year, month, day];
}

/**
 * Shift an ISO `YYYY-MM-DD` date by a whole number of days, in UTC.
 *
 * Uses the proleptic Gregorian calendar, like Python's `date + timedelta`,
 * so leap days, month ends and year boundaries match. No time zone or
 * daylight-saving rule is involved.
 *
 * Args:
 *   date: A calendar date `YYYY-MM-DD` between `0001-01-01` and `9999-12-31`.
 *   days: A safe integer number of days, for example a date-shift offset.
 *
 * Returns:
 *   The shifted date as `YYYY-MM-DD`.
 *
 * Throws:
 *   DateShiftError: `date_invalid`, `days_invalid` or `date_out_of_range`.
 */
export function shiftIsoDate(date: string, days: number): string {
  const match = typeof date === "string" ? ISO_DATE.exec(date) : null;
  if (match === null) {
    throw new DateShiftError("date_invalid");
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (
    year < MIN_YEAR ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > daysInMonth(year, month)
  ) {
    throw new DateShiftError("date_invalid");
  }
  if (typeof days !== "number" || !Number.isSafeInteger(days)) {
    throw new DateShiftError("days_invalid");
  }
  const target = daysFromCivil(year, month, day) + days;
  if (
    target < daysFromCivil(MIN_YEAR, 1, 1) ||
    target > daysFromCivil(MAX_YEAR, 12, 31)
  ) {
    throw new DateShiftError("date_out_of_range");
  }
  const [shiftedYear, shiftedMonth, shiftedDay] = civilFromDays(target);
  return [
    String(shiftedYear).padStart(4, "0"),
    String(shiftedMonth).padStart(2, "0"),
    String(shiftedDay).padStart(2, "0"),
  ].join("-");
}
