// Verification of Python-signed OpenMed audit reports (AuditReport.sign()).
//
// A report carries a `repro_hash` (SHA-256 of its canonical JSON without
// `repro_hash` and `signature`) and an HMAC-SHA256 `signature` over its
// canonical JSON without `signature`. Canonical JSON is Python's
// `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True,
// allow_nan=False)`: members ordered by Unicode code point, every character
// outside printable ASCII escaped as lowercase `\uXXXX` (astral characters as
// surrogate pairs), and floats in Python `repr` form (`1e-07`, `1.0`). Plain
// `JSON.parse` and `JSON.stringify` lose the difference between `1` and `1.0`
// and print `1e-7`, so this module parses the report text itself and keeps
// every number literal. Verification uses WebCrypto (`crypto.subtle`) only;
// the module imports nothing from Node and never signs.

/** The only signature algorithm a report may carry. */
export const AUDIT_REPORT_SIGNATURE_ALGORITHM = "HMAC-SHA256";

/** Reason codes shared with the Swift and Kotlin receipt verifiers. */
export const AUDIT_REPORT_VERIFICATION_REASONS = Object.freeze([
  "valid",
  "hash_mismatch",
  "signature_mismatch",
  "unsupported_algorithm",
  "malformed",
] as const);

export type AuditReportVerificationReason =
  (typeof AUDIT_REPORT_VERIFICATION_REASONS)[number];

/** `sorted` is the deterministic span order; `stored` is the legacy order. */
export type AuditReportSpanOrder = "sorted" | "stored";

export interface AuditReportVerification {
  /** True only when the repro hash and the HMAC-SHA256 signature verify. */
  result: boolean;
  /** `valid` when `result` is true, otherwise why the report did not verify. */
  reason: AuditReportVerificationReason;
  /** The span order that verified, or null. */
  spanOrder: AuditReportSpanOrder | null;
}

/** Canonical payload bytes for one span order. */
export interface AuditReportPayload {
  spanOrder: AuditReportSpanOrder;
  /** Canonical bytes whose SHA-256 is the `repro_hash`. */
  repro: Uint8Array;
  /** Canonical bytes the HMAC-SHA256 signature covers. */
  signing: Uint8Array;
}

/** WebCrypto operations the verifier uses; `crypto.subtle` by default. */
export type AuditReportCrypto = Pick<SubtleCrypto, "digest" | "importKey" | "verify">;

export interface VerifyAuditReportOptions {
  /** Override `globalThis.crypto.subtle`, for example in a worker shim. */
  crypto?: AuditReportCrypto;
}

export type AuditReceiptErrorCode = "key_invalid" | "malformed" | "verifier_unavailable";

/** A value-free verifier failure. Messages carry the code only. */
export class AuditReceiptError extends Error {
  readonly code: AuditReceiptErrorCode;

  constructor(code: AuditReceiptErrorCode) {
    super(code);
    this.name = "AuditReceiptError";
    this.code = code;
  }
}

const MAX_DEPTH = 256;
const HEX_DIGITS = "0123456789abcdef";
const SIGNATURE_VALUE = /^[0-9a-f]{64}$/;
const HMAC_SHA256 = { name: "HMAC", hash: "SHA-256" } as const;
const NUMBER = /-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?/y;
const SIMPLE_ESCAPES: Readonly<Record<string, string>> = {
  '"': '"',
  "\\": "\\",
  "/": "/",
  b: "\b",
  f: "\f",
  n: "\n",
  r: "\r",
  t: "\t",
};

/** A JSON number with the literal it was written as. */
class JsonNumber {
  constructor(
    readonly literal: string,
    readonly isFloat: boolean,
  ) {}
}

type JsonValue = null | boolean | string | JsonNumber | JsonValue[] | Map<string, JsonValue>;

class Malformed extends Error {}

/** Strict JSON parser that keeps number literals (Python `json.loads` grammar). */
class Parser {
  private index = 0;

  constructor(private readonly text: string) {}

  parse(): JsonValue {
    const value = this.value(0);
    this.whitespace();
    if (this.index !== this.text.length) {
      throw new Malformed();
    }
    return value;
  }

  private whitespace(): void {
    while (this.index < this.text.length) {
      const char = this.text.charCodeAt(this.index);
      if (char !== 0x20 && char !== 0x09 && char !== 0x0a && char !== 0x0d) {
        return;
      }
      this.index += 1;
    }
  }

  private value(depth: number): JsonValue {
    if (depth > MAX_DEPTH) {
      throw new Malformed();
    }
    this.whitespace();
    const char = this.text[this.index];
    if (char === "{") {
      return this.object(depth);
    }
    if (char === "[") {
      return this.array(depth);
    }
    if (char === '"') {
      return this.string();
    }
    for (const [word, value] of [
      ["true", true],
      ["false", false],
      ["null", null],
    ] as const) {
      if (this.text.startsWith(word, this.index)) {
        this.index += word.length;
        return value;
      }
    }
    NUMBER.lastIndex = this.index;
    const match = NUMBER.exec(this.text);
    if (match === null) {
      throw new Malformed();
    }
    this.index += match[0].length;
    return new JsonNumber(match[0], match[1] !== undefined || match[2] !== undefined);
  }

  private object(depth: number): Map<string, JsonValue> {
    const members = new Map<string, JsonValue>();
    this.index += 1;
    this.whitespace();
    if (this.text[this.index] === "}") {
      this.index += 1;
      return members;
    }
    for (;;) {
      this.whitespace();
      if (this.text[this.index] !== '"') {
        throw new Malformed();
      }
      const key = this.string();
      this.whitespace();
      if (this.text[this.index] !== ":") {
        throw new Malformed();
      }
      this.index += 1;
      // A repeated member keeps its last value, as Python's json.loads does.
      members.set(key, this.value(depth + 1));
      this.whitespace();
      const next = this.text[this.index];
      this.index += 1;
      if (next === "}") {
        return members;
      }
      if (next !== ",") {
        throw new Malformed();
      }
    }
  }

  private array(depth: number): JsonValue[] {
    const items: JsonValue[] = [];
    this.index += 1;
    this.whitespace();
    if (this.text[this.index] === "]") {
      this.index += 1;
      return items;
    }
    for (;;) {
      items.push(this.value(depth + 1));
      this.whitespace();
      const next = this.text[this.index];
      this.index += 1;
      if (next === "]") {
        return items;
      }
      if (next !== ",") {
        throw new Malformed();
      }
    }
  }

  private string(): string {
    this.index += 1;
    let output = "";
    for (;;) {
      if (this.index >= this.text.length) {
        throw new Malformed();
      }
      const code = this.text.charCodeAt(this.index);
      if (code === 0x22) {
        this.index += 1;
        return output;
      }
      if (code < 0x20) {
        throw new Malformed();
      }
      if (code !== 0x5c) {
        output += this.text[this.index];
        this.index += 1;
        continue;
      }
      const escape = this.text[this.index + 1];
      this.index += 2;
      if (escape !== undefined && Object.hasOwn(SIMPLE_ESCAPES, escape)) {
        output += SIMPLE_ESCAPES[escape]!;
      } else if (escape === "u") {
        const hex = this.text.slice(this.index, this.index + 4);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) {
          throw new Malformed();
        }
        output += String.fromCharCode(Number.parseInt(hex, 16));
        this.index += 4;
      } else {
        throw new Malformed();
      }
    }
  }
}

function parseReport(json: string | Uint8Array): JsonValue {
  let text: string;
  if (typeof json === "string") {
    text = json;
  } else if (json instanceof Uint8Array) {
    try {
      // Keep a byte order mark so it fails to parse, as it does for text input
      // and in Python's json.loads.
      text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(json);
    } catch {
      throw new Malformed();
    }
  } else {
    throw new Malformed();
  }
  return new Parser(text).parse();
}

function compareCodePoints(left: string, right: string): number {
  const a = left[Symbol.iterator]();
  const b = right[Symbol.iterator]();
  for (;;) {
    const x = a.next();
    const y = b.next();
    if (x.done || y.done) {
      return x.done && y.done ? 0 : x.done ? -1 : 1;
    }
    const difference = x.value.codePointAt(0)! - y.value.codePointAt(0)!;
    if (difference !== 0) {
      return difference;
    }
  }
}

function quote(value: string): string {
  let output = '"';
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code === 0x22) {
      output += '\\"';
    } else if (code === 0x5c) {
      output += "\\\\";
    } else if (code >= 0x20 && code <= 0x7e) {
      output += value[index];
    } else if (code === 0x0a) {
      output += "\\n";
    } else if (code === 0x0d) {
      output += "\\r";
    } else if (code === 0x09) {
      output += "\\t";
    } else if (code === 0x08) {
      output += "\\b";
    } else if (code === 0x0c) {
      output += "\\f";
    } else {
      // Every other UTF-16 unit, including each half of an astral pair.
      output += "\\u";
      for (let shift = 12; shift >= 0; shift -= 4) {
        output += HEX_DIGITS[(code >> shift) & 0xf];
      }
    }
  }
  return `${output}"`;
}

/** Format a finite double exactly as Python's `repr(float)` does. */
function pythonFloat(value: number): string {
  if (!Number.isFinite(value)) {
    throw new Malformed();
  }
  if (value === 0) {
    return Object.is(value, -0) ? "-0.0" : "0.0";
  }
  const sign = value < 0 ? "-" : "";
  // toExponential() without an argument yields the shortest round-trip digits.
  const [mantissa = "", exponentText = "0"] = Math.abs(value).toExponential().split("e");
  const digits = mantissa.replace(".", "");
  const point = Number(exponentText) + 1;
  if (point <= -4 || point > 16) {
    const exponent = point - 1;
    const body = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const magnitude = String(Math.abs(exponent)).padStart(2, "0");
    return `${sign}${body}e${exponent < 0 ? "-" : "+"}${magnitude}`;
  }
  if (point <= 0) {
    return `${sign}0.${"0".repeat(-point)}${digits}`;
  }
  if (point >= digits.length) {
    return `${sign}${digits}${"0".repeat(point - digits.length)}.0`;
  }
  return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
}

function encode(value: JsonValue): string {
  if (value === null) {
    return "null";
  }
  if (value === true) {
    return "true";
  }
  if (value === false) {
    return "false";
  }
  if (typeof value === "string") {
    return quote(value);
  }
  if (value instanceof JsonNumber) {
    if (value.isFloat) {
      return pythonFloat(Number(value.literal));
    }
    // Python reads an integer literal as an exact int; `-0` prints as `0`.
    return BigInt(value.literal).toString();
  }
  if (Array.isArray(value)) {
    return `[${value.map(encode).join(",")}]`;
  }
  const keys = [...value.keys()].sort(compareCodePoints);
  return `{${keys.map((key) => `${quote(key)}:${encode(value.get(key)!)}`).join(",")}}`;
}

/**
 * Re-encode JSON text as Python's canonical audit JSON.
 *
 * Args:
 *   json: JSON text or UTF-8 bytes.
 *
 * Returns:
 *   The canonical JSON text (ASCII only).
 *
 * Throws:
 *   AuditReceiptError: `malformed` when the input is not strict JSON or holds
 *   a number Python cannot serialise (an overflowing float).
 */
export function canonicalAuditJson(json: string | Uint8Array): string {
  try {
    return encode(parseReport(json));
  } catch (error) {
    if (error instanceof Malformed) {
      throw new AuditReceiptError("malformed");
    }
    throw error;
  }
}

interface ParsedReport {
  report: Map<string, JsonValue>;
  spans: Map<string, JsonValue>[];
  sortedSpans: Map<string, JsonValue>[];
  reproHash: string;
  signature: JsonValue | undefined;
}

function integer(value: JsonValue | undefined): bigint {
  if (!(value instanceof JsonNumber) || value.isFloat) {
    throw new Malformed();
  }
  return BigInt(value.literal);
}

function text(value: JsonValue | undefined): string {
  if (typeof value !== "string") {
    throw new Malformed();
  }
  return value;
}

/** Validate what verification needs, as AuditReport's own checks do. */
function structure(json: string | Uint8Array): ParsedReport {
  const report = parseReport(json);
  if (!(report instanceof Map)) {
    throw new Malformed();
  }
  const documentLength = integer(report.get("document_length"));
  const rawSpans = report.get("spans");
  if (documentLength < 0n || !Array.isArray(rawSpans)) {
    throw new Malformed();
  }
  const keyed = rawSpans.map((span) => {
    if (!(span instanceof Map)) {
      throw new Malformed();
    }
    const start = integer(span.get("start"));
    const end = integer(span.get("end"));
    if (start < 0n || end < start || end > documentLength) {
      throw new Malformed();
    }
    return {
      span,
      start,
      end,
      label: text(span.get("canonical_label")),
      action: text(span.get("action")),
      canonical: encode(span),
    };
  });
  const sortedSpans = [...keyed]
    .sort(
      (left, right) =>
        (left.start < right.start ? -1 : left.start > right.start ? 1 : 0) ||
        (left.end < right.end ? -1 : left.end > right.end ? 1 : 0) ||
        compareCodePoints(left.label, right.label) ||
        compareCodePoints(left.action, right.action) ||
        compareCodePoints(left.canonical, right.canonical),
    )
    .map((item) => item.span);
  const reproHash = report.get("repro_hash");
  if (typeof reproHash !== "string" || reproHash === "") {
    throw new Malformed();
  }
  return {
    report,
    spans: rawSpans as Map<string, JsonValue>[],
    sortedSpans,
    reproHash,
    signature: report.get("signature"),
  };
}

function payloads(parsed: ParsedReport): AuditReportPayload[] {
  const encoder = new TextEncoder();
  const build = (spans: Map<string, JsonValue>[], omit: readonly string[]) => {
    const copy = new Map(parsed.report);
    for (const key of omit) {
      copy.delete(key);
    }
    copy.set("spans", spans);
    return encoder.encode(encode(copy));
  };
  const candidate = (
    spanOrder: AuditReportSpanOrder,
    spans: Map<string, JsonValue>[],
  ): AuditReportPayload => ({
    spanOrder,
    repro: build(spans, ["repro_hash", "signature"]),
    signing: build(spans, ["signature"]),
  });
  const sorted = candidate("sorted", parsed.sortedSpans);
  const stored = candidate("stored", parsed.spans);
  return sameBytes(sorted.repro, stored.repro) ? [sorted] : [sorted, stored];
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  return left.length === right.length && left.every((byte, index) => byte === right[index]);
}

/**
 * Return the canonical repro and signing payloads a verifier checks.
 *
 * The `sorted` span order comes first; the `stored` order follows only when
 * its bytes differ. These are the bytes `receipt_verification_v1.json`
 * records for each candidate.
 *
 * Args:
 *   json: The report as JSON text or UTF-8 bytes, as Python wrote it.
 *
 * Returns:
 *   One payload pair per span order to try.
 *
 * Throws:
 *   AuditReceiptError: `malformed` when the report cannot be checked.
 */
export function auditReportPayloads(json: string | Uint8Array): AuditReportPayload[] {
  try {
    return payloads(structure(json));
  } catch (error) {
    if (error instanceof Malformed) {
      throw new AuditReceiptError("malformed");
    }
    throw error;
  }
}

function buffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer;
}

function hexBytes(value: string): Uint8Array {
  const bytes = new Uint8Array(value.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(value.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

function keyBytes(key: string | Uint8Array): Uint8Array {
  if (typeof key === "string") {
    if (key.length > 0) {
      return new TextEncoder().encode(key);
    }
  } else if (key instanceof Uint8Array && key.byteLength > 0) {
    return new Uint8Array(key);
  }
  throw new AuditReceiptError("key_invalid");
}

function resolveCrypto(options: VerifyAuditReportOptions): AuditReportCrypto {
  const subtle = options.crypto ?? (globalThis as { crypto?: Crypto }).crypto?.subtle;
  if (
    subtle === undefined ||
    typeof subtle.digest !== "function" ||
    typeof subtle.importKey !== "function" ||
    typeof subtle.verify !== "function"
  ) {
    throw new AuditReceiptError("verifier_unavailable");
  }
  return subtle;
}

async function guarded<T>(operation: () => Promise<T>): Promise<T> {
  try {
    return await operation();
  } catch {
    // Never forward a WebCrypto error: it may describe the key.
    throw new AuditReceiptError("verifier_unavailable");
  }
}

/**
 * Verify a Python-signed audit report with its HMAC key.
 *
 * Mirrors `AuditReport.verify()`: the signature must be an object whose
 * algorithm is `HMAC-SHA256`, and some span order, tried `sorted` first and
 * then `stored`, must give the stored `repro_hash` and a signing payload whose
 * HMAC equals `signature.value`. The HMAC is checked with
 * `crypto.subtle.verify`. Every member other than `repro_hash` and
 * `signature` is covered, so a member added after signing fails the hash.
 * An unsigned report gives `signature_mismatch`.
 *
 * Args:
 *   json: The report as JSON text or UTF-8 bytes, as Python wrote it. Do not
 *     round-trip it through `JSON.parse`: that loses float literals.
 *   key: The non-empty HMAC key, as text (UTF-8) or bytes.
 *   options: Optional WebCrypto override.
 *
 * Returns:
 *   `{ result, reason, spanOrder }`; a malformed report gives
 *   `{ result: false, reason: "malformed", spanOrder: null }`.
 *
 * Throws:
 *   AuditReceiptError: `key_invalid` for an empty key, `verifier_unavailable`
 *   when WebCrypto is missing or fails. Errors never carry key material.
 */
export async function verifyAuditReport(
  json: string | Uint8Array,
  key: string | Uint8Array,
  options: VerifyAuditReportOptions = {},
): Promise<AuditReportVerification> {
  const secret = keyBytes(key);
  try {
    const subtle = resolveCrypto(options);
    const rejected = (reason: AuditReportVerificationReason): AuditReportVerification => ({
      result: false,
      reason,
      spanOrder: null,
    });
    let parsed: ParsedReport;
    let candidates: AuditReportPayload[];
    try {
      parsed = structure(json);
      candidates = payloads(parsed);
    } catch (error) {
      if (error instanceof Malformed) {
        return rejected("malformed");
      }
      throw error;
    }
    const signature = parsed.signature;
    if (signature === undefined || signature === null) {
      return rejected("signature_mismatch");
    }
    const value = signature instanceof Map ? signature.get("value") : undefined;
    if (!(signature instanceof Map) || typeof value !== "string") {
      // Not an object, or no string value: there is nothing to check.
      return rejected("malformed");
    }
    if (signature.get("algorithm") !== AUDIT_REPORT_SIGNATURE_ALGORITHM) {
      return rejected("unsupported_algorithm");
    }
    const expected = SIGNATURE_VALUE.test(value) ? hexBytes(value) : null;
    let cryptoKey: CryptoKey | undefined;
    let hashMatched = false;
    for (const candidate of candidates) {
      const digest = await guarded(() => subtle.digest("SHA-256", buffer(candidate.repro)));
      const reproHash = `sha256:${Array.from(new Uint8Array(digest), (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("")}`;
      if (reproHash !== parsed.reproHash) {
        continue;
      }
      hashMatched = true;
      if (expected === null) {
        continue;
      }
      cryptoKey ??= await guarded(async () => {
        const raw = new Uint8Array(secret);
        try {
          return await subtle.importKey("raw", raw.buffer, HMAC_SHA256, false, ["verify"]);
        } finally {
          raw.fill(0);
        }
      });
      const key = cryptoKey;
      const verified = await guarded(() =>
        subtle.verify("HMAC", key, buffer(expected), buffer(candidate.signing)),
      );
      if (verified) {
        return { result: true, reason: "valid", spanOrder: candidate.spanOrder };
      }
    }
    return rejected(hashMatched ? "signature_mismatch" : "hash_mismatch");
  } finally {
    secret.fill(0);
  }
}
