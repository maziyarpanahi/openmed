// Canonical JSON, Unicode-scalar helpers, and value-free fingerprints.

import { sha256Hex } from "./sha256";

const FINGERPRINT_DOMAIN = "openmed.rn.parity.v1\n";

/** Serialize JSON-compatible data with sorted object keys and no whitespace. */
export function canonicalJson(value: unknown): string {
  if (value === undefined || value === null) {
    return "null";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? JSON.stringify(value) : "null";
  }
  if (typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record)
    .filter((key) => record[key] !== undefined)
    .sort();
  return `{${keys
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
    .join(",")}}`;
}

/** Digest of canonical JSON, used to pin committed corpora. */
export function canonicalDigest(value: unknown): string {
  return `sha256:${sha256Hex(canonicalJson(value))}`;
}

/** Domain-separated digest used in mismatch diagnostics instead of values. */
export function fingerprintDigest(value: unknown): string {
  return `sha256:${sha256Hex(FINGERPRINT_DOMAIN + canonicalJson(value))}`;
}

/** Count Unicode scalar values (code points), the OpenMed offset unit. */
export function scalarLength(text: string): number {
  let count = 0;
  for (const _scalar of text) {
    count += 1;
  }
  return count;
}

/** Slice text by half-open Unicode scalar offsets. */
export function scalarSlice(text: string, start: number, end: number): string {
  return Array.from(text).slice(start, end).join("");
}

/** Replace a half-open Unicode scalar range. */
export function replaceScalarRange(
  text: string,
  start: number,
  end: number,
  replacement: string,
): string {
  const scalars = Array.from(text);
  return [
    ...scalars.slice(0, start),
    replacement,
    ...scalars.slice(end),
  ].join("");
}
