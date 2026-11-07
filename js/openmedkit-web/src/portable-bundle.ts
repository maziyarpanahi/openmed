/**
 * Offline verifier for signed OpenMed portable model bundles.
 *
 * This is the browser and Node.js reader for the contract defined by
 * `openmed/core/portable_bundle.py`. It runs the same checks in the same order
 * and returns the same value-free result for the golden vectors in
 * `tests/fixtures/portable_bundle/vectors.json`. It uses WebCrypto (SHA-256 and
 * Ed25519) only: no network access, no bundled-code execution, and no external
 * references. Trust roots are always supplied by the caller.
 */

export const PORTABLE_BUNDLE_SCHEMA = "openmed.portable_bundle";
export const PORTABLE_BUNDLE_SCHEMA_VERSION = 1;
export const PORTABLE_BUNDLE_ENVELOPE_SCHEMA = "openmed.portable_bundle.signatures";
export const PORTABLE_BUNDLE_TRUST_POLICY_SCHEMA = "openmed.portable_bundle.trust_policy";
export const PORTABLE_BUNDLE_VERIFICATION_SCHEMA = "openmed.portable_bundle.verification";
export const PORTABLE_BUNDLE_MANIFEST_FILENAME = "openmed-bundle.json";
export const PORTABLE_BUNDLE_ENVELOPE_FILENAME = "openmed-bundle.sig.json";
export const PORTABLE_BUNDLE_PAYLOAD_TYPE = "application/vnd.openmed.portable-bundle.v1+json";

/** Stable reason codes, in evaluation order. */
export const PORTABLE_BUNDLE_REASONS = [
  "manifest_too_large",
  "envelope_too_large",
  "envelope_malformed",
  "envelope_version_unsupported",
  "content_id_mismatch",
  "bundle_revoked",
  "signature_missing",
  "signature_algorithm_unsupported",
  "untrusted_key",
  "key_revoked",
  "key_not_yet_valid",
  "key_expired",
  "signature_invalid",
  "signature_verifier_unavailable",
  "signature_threshold_not_met",
  "manifest_not_canonical",
  "schema_unsupported",
  "schema_version_unsupported",
  "unknown_critical_field",
  "manifest_malformed",
  "path_unsafe",
  "executable_file_forbidden",
  "runtime_combination_unsupported",
  "duplicate_path",
  "path_conflict",
  "file_order_invalid",
  "file_reference_undeclared",
  "fallback_invalid",
  "unknown_critical_extension",
  "size_limit_exceeded",
  "bundle_not_yet_valid",
  "bundle_expired",
  "rollback_rejected",
  "sdk_incompatible",
  "policy_incompatible",
  "platform_incompatible",
  "file_missing",
  "file_size_mismatch",
  "file_digest_mismatch",
  "file_undeclared",
  "trust_policy_invalid",
  "context_invalid",
  "signing_key_invalid",
] as const;

export type PortableBundleReason = (typeof PORTABLE_BUNDLE_REASONS)[number];

/** Error carrying only a controlled reason code. */
export class PortableBundleError extends Error {
  readonly reason: PortableBundleReason;

  constructor(reason: PortableBundleReason) {
    super(reason);
    this.name = "PortableBundleError";
    this.reason = reason;
  }
}

export interface PortableBundleTrustedKey {
  readonly kid: string;
  readonly publicKey: Uint8Array;
  readonly notBefore: string;
  readonly notAfter: string;
  readonly revoked: boolean;
}

export interface PortableBundleTrustPolicy {
  readonly keys: readonly PortableBundleTrustedKey[];
  readonly threshold: number;
  readonly revokedContentIds: ReadonlySet<string>;
}

export interface PortableBundleTarget {
  readonly platform: string;
  readonly sdkName: string;
  readonly sdkVersion: string;
  readonly runtimes: readonly string[];
  readonly osVersion?: string | null;
  readonly policyVersion?: number | null;
  readonly memoryBytes?: number | null;
  readonly storageBytes?: number | null;
}

export interface PortableBundleContext {
  /** Verification instant as `YYYY-MM-DDTHH:MM:SSZ`. */
  readonly now: string;
  readonly target?: PortableBundleTarget | null;
  readonly sequenceFloors?: Readonly<Record<string, number>>;
  readonly understoodExtensions?: readonly string[];
  readonly maxManifestBytes?: number;
  readonly maxEnvelopeBytes?: number;
  readonly maxTotalBytes?: number;
  readonly maxFiles?: number;
}

export interface PortableBundleVerification {
  readonly ok: boolean;
  readonly reason: PortableBundleReason | null;
  readonly contentId: string | null;
  readonly validSignatures: number;
  readonly eligibleRuntimes: readonly number[];
  readonly filesChecked: number;
  readonly filesVerified: boolean;
}

export interface VerifyPortableBundleInput {
  readonly manifest: Uint8Array;
  readonly envelope: Uint8Array;
  readonly trustPolicy: PortableBundleTrustPolicy;
  readonly context: PortableBundleContext;
  /** Bundle files by relative path; omit to verify manifest and signatures only. */
  readonly files?: ReadonlyMap<string, Uint8Array> | null;
}

export const PORTABLE_BUNDLE_DEFAULT_LIMITS = {
  maxManifestBytes: 1 << 20,
  maxEnvelopeBytes: 64 << 10,
  maxTotalBytes: 64 * 1024 * 1024 * 1024,
  maxFiles: 4096,
} as const;

const MAX_SAFE = Number.MAX_SAFE_INTEGER;
const MAX_SIGNATURES = 16;
const MAX_DEPTH = 32;

const FORMAT_PLATFORMS: Readonly<Record<string, readonly string[]>> = {
  coreml: ["ios", "macos", "watchos", "visionos"],
  mlx: ["ios", "macos"],
  onnx: ["android", "ios", "macos", "linux", "windows", "web"],
  ort: ["android", "ios"],
  transformersjs: ["web"],
  pytorch: ["linux", "macos", "windows"],
};
const FORMAT_PRECISIONS: Readonly<Record<string, readonly string[]>> = {
  coreml: ["fp32", "fp16", "int8", "int4"],
  mlx: ["fp32", "fp16", "bf16", "int8", "int4", "int2"],
  onnx: ["fp32", "fp16", "int8", "int4"],
  ort: ["fp32", "fp16", "int8"],
  transformersjs: ["fp32", "fp16", "int8", "int4"],
  pytorch: ["fp32", "fp16", "bf16"],
};
const PLATFORMS = ["ios", "macos", "watchos", "visionos", "android", "web", "linux", "windows"];
const PRECISIONS = ["fp32", "fp16", "bf16", "int8", "int4", "int2"];
const FILE_ROLES = [
  "calibration",
  "config",
  "decoder",
  "evidence",
  "graph",
  "labels",
  "license",
  "metadata",
  "policy",
  "provenance",
  "segmenter",
  "tokenizer",
  "weights",
];
const OFFSET_UNITS = ["unicode_scalar", "utf16_code_unit", "utf8_byte"];
const RESERVED_PATHS = [PORTABLE_BUNDLE_MANIFEST_FILENAME, PORTABLE_BUNDLE_ENVELOPE_FILENAME];

const whole = (pattern: string): RegExp => new RegExp(`^(?:${pattern})$`);
const TOKEN = whole("[a-z0-9][a-z0-9._-]{0,63}");
const BUNDLE_ID = whole("[a-z0-9][a-z0-9._-]{0,127}");
const MODEL_ID = whole("[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}");
const REVISION = whole("[0-9a-f]{40}");
const LANGUAGE = whole("[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}");
const LICENSE = whole("[a-z0-9][a-z0-9.+-]{0,63}");
const SHA256 = whole("sha256:[0-9a-f]{64}");
const MEDIA_TYPE = whole("[a-z]+/[a-z0-9][a-z0-9.+-]{0,126}");
const EXTENSION = whole("[a-z0-9][a-z0-9-]{0,31}(?:\\.[a-z0-9][a-z0-9_-]{0,31}){1,7}");
const SEMVER = whole("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){2}");
const VERSION = whole("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){0,3}");
const TIMESTAMP = whole(
  "[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
);
const KID = whole("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
const PATH_SEGMENT = whole("[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}");
const B64URL = whole("[A-Za-z0-9_-]+");
const LONE_SURROGATE = /[\uD800-\uDFFF]/u;

const TOP_KEYS = [
  "schema",
  "schema_version",
  "bundle",
  "model",
  "tokenizer",
  "labels",
  "offsets",
  "decoder",
  "segmenter",
  "policy",
  "calibration",
  "compatibility",
  "runtimes",
  "licenses",
  "provenance",
  "evidence",
  "files",
  "extensions",
  "critical_extensions",
];

class Refusal extends Error {
  readonly reason: PortableBundleReason;

  constructor(reason: PortableBundleReason) {
    super(reason);
    this.reason = reason;
  }
}

const fail = (reason: PortableBundleReason): Refusal => new Refusal(reason);

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type JsonObject = { [key: string]: Json };

// ---------------------------------------------------------------------------
// Canonical JSON (RFC 8785 subset)
// ---------------------------------------------------------------------------

/**
 * Return the RFC 8785 canonical encoding of a JSON value. Non-integer numbers,
 * integers outside the IEEE 754 safe range, and unpaired surrogates are
 * rejected with `manifest_not_canonical`.
 */
export function canonicalJson(value: unknown): string {
  try {
    return encode(value, 0);
  } catch (error) {
    if (error instanceof Refusal) {
      throw new PortableBundleError("manifest_not_canonical");
    }
    throw error;
  }
}

function encode(value: unknown, depth: number): string {
  if (depth > MAX_DEPTH) throw fail("manifest_not_canonical");
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw fail("manifest_not_canonical");
    return String(value);
  }
  if (typeof value === "string") {
    if (LONE_SURROGATE.test(value)) throw fail("manifest_not_canonical");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => encode(item, depth + 1)).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    // Default string ordering compares UTF-16 code units, as RFC 8785 requires.
    const keys = Object.keys(record).sort();
    return `{${keys
      .map((key) => `${encode(key, depth + 1)}:${encode(record[key], depth + 1)}`)
      .join(",")}}`;
  }
  throw fail("manifest_not_canonical");
}

function parseCanonical(data: Uint8Array): Json {
  let value: Json;
  try {
    const text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(data);
    value = JSON.parse(text) as Json;
  } catch {
    throw fail("manifest_not_canonical");
  }
  let encoded: Uint8Array;
  try {
    encoded = new TextEncoder().encode(encode(value, 0));
  } catch {
    throw fail("manifest_not_canonical");
  }
  if (!bytesEqual(encoded, data)) throw fail("manifest_not_canonical");
  return value;
}

function bytesEqual(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Crypto helpers
// ---------------------------------------------------------------------------

function subtle(): SubtleCrypto | null {
  const candidate = (globalThis as { crypto?: Crypto }).crypto;
  return candidate?.subtle ?? null;
}

function hex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function buffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const crypto = subtle();
  if (crypto === null) throw fail("signature_verifier_unavailable");
  return `sha256:${hex(await crypto.digest("SHA-256", buffer(bytes)))}`;
}

/** Return the `sha256:<hex>` content identity of canonical manifest bytes. */
export async function portableBundleContentId(manifest: Uint8Array): Promise<string> {
  return sha256(manifest);
}

/** Return the DSSE v1 pre-authentication encoding that signatures cover. */
export function portableBundleSigningInput(manifest: Uint8Array): Uint8Array {
  const header = new TextEncoder().encode(
    `DSSEv1 ${PORTABLE_BUNDLE_PAYLOAD_TYPE.length} ${PORTABLE_BUNDLE_PAYLOAD_TYPE} ${manifest.length} `,
  );
  const output = new Uint8Array(header.length + manifest.length);
  output.set(header, 0);
  output.set(manifest, header.length);
  return output;
}

const B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

function base64UrlEncode(bytes: Uint8Array): string {
  let output = "";
  for (let index = 0; index < bytes.length; index += 3) {
    const a = bytes[index] ?? 0;
    const b = bytes[index + 1] ?? 0;
    const c = bytes[index + 2] ?? 0;
    const triple = (a << 16) | (b << 8) | c;
    const remaining = bytes.length - index;
    output += B64_ALPHABET[(triple >> 18) & 63];
    output += B64_ALPHABET[(triple >> 12) & 63];
    if (remaining > 1) output += B64_ALPHABET[(triple >> 6) & 63];
    if (remaining > 2) output += B64_ALPHABET[triple & 63];
  }
  return output;
}

function base64UrlDecode(text: unknown, length: number): Uint8Array | null {
  if (typeof text !== "string" || !B64URL.test(text) || text.length % 4 === 1) return null;
  const bytes: number[] = [];
  let bits = 0;
  let accumulator = 0;
  for (const char of text) {
    accumulator = (accumulator << 6) | B64_ALPHABET.indexOf(char);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push((accumulator >> bits) & 0xff);
    }
  }
  const raw = Uint8Array.from(bytes);
  if (raw.length !== length || base64UrlEncode(raw) !== text) return null;
  return raw;
}

async function ed25519Verify(
  publicKey: Uint8Array,
  signature: Uint8Array,
  message: Uint8Array,
): Promise<boolean> {
  const crypto = subtle();
  if (crypto === null) throw fail("signature_verifier_unavailable");
  let key: CryptoKey;
  try {
    key = await crypto.importKey("raw", buffer(publicKey), { name: "Ed25519" }, false, ["verify"]);
  } catch (error) {
    if (error instanceof Error && error.name === "NotSupportedError") {
      throw fail("signature_verifier_unavailable");
    }
    return false;
  }
  try {
    return await crypto.verify({ name: "Ed25519" }, key, buffer(signature), buffer(message));
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Trust policy
// ---------------------------------------------------------------------------

function isTimestamp(value: unknown): value is string {
  return typeof value === "string" && TIMESTAMP.test(value);
}

function isVersion(value: unknown): value is string {
  return typeof value === "string" && VERSION.test(value);
}

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: JsonObject, keys: readonly string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

/** Parse a JSON trust-policy document (`portable_bundle_trust_policy`). */
export function parsePortableBundleTrustPolicy(value: unknown): PortableBundleTrustPolicy {
  const invalid = () => new PortableBundleError("trust_policy_invalid");
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["schema", "schema_version", "threshold", "keys", "revoked_content_ids"])
  ) {
    throw invalid();
  }
  const { keys, threshold } = value;
  const revoked = value.revoked_content_ids;
  if (
    value.schema !== PORTABLE_BUNDLE_TRUST_POLICY_SCHEMA ||
    value.schema_version !== 1 ||
    typeof threshold !== "number" ||
    !Number.isInteger(threshold) ||
    threshold < 1 ||
    threshold > MAX_SIGNATURES ||
    !Array.isArray(keys) ||
    keys.length === 0 ||
    keys.length > 64 ||
    !Array.isArray(revoked) ||
    !revoked.every((item) => typeof item === "string" && SHA256.test(item))
  ) {
    throw invalid();
  }
  const parsed: PortableBundleTrustedKey[] = [];
  for (const entry of keys) {
    if (
      !isRecord(entry) ||
      !hasExactKeys(entry, ["kid", "kty", "crv", "x", "not_before", "not_after", "status"])
    ) {
      throw invalid();
    }
    const publicKey = base64UrlDecode(entry.x, 32);
    if (
      typeof entry.kid !== "string" ||
      !KID.test(entry.kid) ||
      entry.kty !== "OKP" ||
      entry.crv !== "Ed25519" ||
      publicKey === null ||
      !isTimestamp(entry.not_before) ||
      !isTimestamp(entry.not_after) ||
      entry.not_before >= entry.not_after ||
      (entry.status !== "active" && entry.status !== "revoked")
    ) {
      throw invalid();
    }
    parsed.push({
      kid: entry.kid,
      publicKey,
      notBefore: entry.not_before,
      notAfter: entry.not_after,
      revoked: entry.status === "revoked",
    });
  }
  if (new Set(parsed.map((key) => key.kid)).size !== parsed.length) throw invalid();
  return { keys: parsed, threshold, revokedContentIds: new Set(revoked as string[]) };
}

// ---------------------------------------------------------------------------
// Manifest structure
// ---------------------------------------------------------------------------

interface ValidatedManifest {
  readonly bundleId: string;
  readonly sequence: number;
  readonly issuedAt: string;
  readonly expiresAt: string;
  readonly sdks: readonly (readonly [string, string, string | null])[];
  readonly policyRange: readonly [number, number];
  readonly runtimes: readonly JsonObject[];
  readonly files: readonly JsonObject[];
}

function object(value: Json | undefined, keys: readonly string[]): JsonObject {
  if (!isRecord(value)) throw fail("manifest_malformed");
  const actual = Object.keys(value);
  if (!actual.every((key) => keys.includes(key))) throw fail("unknown_critical_field");
  if (actual.length !== keys.length) throw fail("manifest_malformed");
  return value;
}

function string(value: Json | undefined, pattern: RegExp): string {
  if (typeof value !== "string" || !pattern.test(value)) throw fail("manifest_malformed");
  return value;
}

function choice(value: Json | undefined, choices: readonly string[]): string {
  if (typeof value !== "string" || !choices.includes(value)) throw fail("manifest_malformed");
  return value;
}

function uint(value: Json | undefined, minimum = 0): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) {
    throw fail("manifest_malformed");
  }
  return value;
}

function array(value: Json | undefined, minimum: number, maximum: number): Json[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) {
    throw fail("manifest_malformed");
  }
  return value;
}

function unique(items: readonly unknown[]): void {
  if (new Set(items).size !== items.length) throw fail("manifest_malformed");
}

function path(value: Json | undefined): string {
  if (typeof value !== "string") throw fail("manifest_malformed");
  const segments = value.split("/");
  if (
    value.length === 0 ||
    value.length > 512 ||
    segments.length > 16 ||
    !segments.every((segment) => PATH_SEGMENT.test(segment)) ||
    RESERVED_PATHS.includes(value.toLowerCase())
  ) {
    throw fail("path_unsafe");
  }
  return value;
}

function pathList(value: Json | undefined, minimum: number, maximum: number): string[] {
  const items = array(value, minimum, maximum).map(path);
  unique(items);
  return items;
}

function version(value: Json | undefined): string {
  if (!isVersion(value)) throw fail("manifest_malformed");
  return value;
}

function versionKey(value: string): number[] {
  const parts = value.split(".").map((part) => Number(part));
  while (parts.length < 4) parts.push(0);
  return parts;
}

function compareVersions(left: string, right: string): number {
  const a = versionKey(left);
  const b = versionKey(right);
  for (let index = 0; index < 4; index += 1) {
    const difference = (a[index] ?? 0) - (b[index] ?? 0);
    if (difference !== 0) return difference < 0 ? -1 : 1;
  }
  return 0;
}

function utf8Less(left: string, right: string): boolean {
  // Portable paths are ASCII, so UTF-16 and UTF-8 orders coincide.
  return left < right;
}

function validateManifest(value: Json, understood: ReadonlySet<string>): ValidatedManifest {
  if (!isRecord(value)) throw fail("manifest_malformed");
  if (value.schema !== PORTABLE_BUNDLE_SCHEMA) throw fail("schema_unsupported");
  const schemaVersion = value.schema_version;
  if (typeof schemaVersion !== "number") throw fail("manifest_malformed");
  if (schemaVersion !== PORTABLE_BUNDLE_SCHEMA_VERSION) throw fail("schema_version_unsupported");
  const top = object(value, TOP_KEYS);
  const references: string[] = [];

  const bundle = object(top.bundle, ["id", "version", "sequence", "issued_at", "expires_at"]);
  const bundleId = string(bundle.id, BUNDLE_ID);
  string(bundle.version, SEMVER);
  const sequence = uint(bundle.sequence);
  const issuedAt = string(bundle.issued_at, TIMESTAMP);
  const expiresAt = string(bundle.expires_at, TIMESTAMP);
  if (issuedAt >= expiresAt) throw fail("manifest_malformed");

  const model = object(top.model, ["id", "revision", "family", "architecture", "task", "languages"]);
  string(model.id, MODEL_ID);
  if (model.revision !== null) string(model.revision, REVISION);
  string(model.family, TOKEN);
  string(model.architecture, TOKEN);
  string(model.task, TOKEN);
  unique(array(model.languages, 1, 32).map((item) => string(item, LANGUAGE)));

  if (top.tokenizer !== null) {
    const tokenizer = object(top.tokenizer, ["kind", "contract_version", "files"]);
    string(tokenizer.kind, TOKEN);
    uint(tokenizer.contract_version, 1);
    references.push(...pathList(tokenizer.files, 1, 64));
  }
  if (top.labels !== null) {
    const labels = object(top.labels, ["path", "count", "scheme"]);
    references.push(path(labels.path));
    uint(labels.count, 1);
    string(labels.scheme, TOKEN);
  }
  const offsets = object(top.offsets, ["unit", "contract_version"]);
  choice(offsets.unit, OFFSET_UNITS);
  uint(offsets.contract_version, 1);
  const decoder = object(top.decoder, ["kind", "contract_version"]);
  string(decoder.kind, TOKEN);
  uint(decoder.contract_version, 1);
  if (top.segmenter !== null) {
    const segmenter = object(top.segmenter, ["id", "files"]);
    string(segmenter.id, TOKEN);
    references.push(...pathList(segmenter.files, 1, 64));
  }
  const policy = object(top.policy, ["min_version", "max_version_exclusive", "path"]);
  const policyMin = uint(policy.min_version, 1);
  const policyMax = uint(policy.max_version_exclusive, 1);
  if (policyMin >= policyMax) throw fail("manifest_malformed");
  if (policy.path !== null) references.push(path(policy.path));
  if (top.calibration !== null) {
    const calibration = object(top.calibration, ["profile_version", "path"]);
    uint(calibration.profile_version, 1);
    references.push(path(calibration.path));
  }

  const compatibility = object(top.compatibility, ["sdks"]);
  const sdks: (readonly [string, string, string | null])[] = [];
  for (const entry of array(compatibility.sdks, 1, 16)) {
    const sdk = object(entry, ["name", "min", "max_exclusive"]);
    const name = string(sdk.name, TOKEN);
    const minimum = version(sdk.min);
    const maximum = sdk.max_exclusive === null ? null : version(sdk.max_exclusive);
    if (maximum !== null && compareVersions(minimum, maximum) >= 0) {
      throw fail("manifest_malformed");
    }
    sdks.push([name, minimum, maximum]);
  }
  unique(sdks.map((item) => item[0]));

  const runtimes: JsonObject[] = [];
  for (const entry of array(top.runtimes, 1, 32)) {
    const runtime = object(entry, [
      "id",
      "format",
      "precision",
      "entrypoint",
      "files",
      "platforms",
      "min_memory_bytes",
      "min_storage_bytes",
      "fallback",
    ]);
    string(runtime.id, TOKEN);
    const format = choice(runtime.format, Object.keys(FORMAT_PLATFORMS));
    const precision = choice(runtime.precision, PRECISIONS);
    const entrypoint = path(runtime.entrypoint);
    const files = pathList(runtime.files, 1, 1024);
    if (!files.includes(entrypoint)) throw fail("manifest_malformed");
    references.push(...files);
    const platforms: string[] = [];
    for (const item of array(runtime.platforms, 1, 16)) {
      const platform = object(item, ["platform", "min_os", "max_os_exclusive"]);
      const name = choice(platform.platform, PLATFORMS);
      const minOs = platform.min_os === null ? null : version(platform.min_os);
      const maxOs = platform.max_os_exclusive === null ? null : version(platform.max_os_exclusive);
      if (minOs !== null && maxOs !== null && compareVersions(minOs, maxOs) >= 0) {
        throw fail("manifest_malformed");
      }
      platforms.push(name);
    }
    unique(platforms);
    uint(runtime.min_memory_bytes);
    uint(runtime.min_storage_bytes);
    if (runtime.fallback !== null) string(runtime.fallback, TOKEN);
    if (
      !(FORMAT_PRECISIONS[format] ?? []).includes(precision) ||
      !platforms.every((item) => (FORMAT_PLATFORMS[format] ?? []).includes(item))
    ) {
      throw fail("runtime_combination_unsupported");
    }
    runtimes.push(runtime);
  }

  for (const entry of array(top.licenses, 1, 16)) {
    const license = object(entry, ["id", "files"]);
    string(license.id, LICENSE);
    references.push(...pathList(license.files, 0, 64));
  }
  const provenance = object(top.provenance, ["source_digest", "export_record"]);
  if (provenance.source_digest !== null) string(provenance.source_digest, SHA256);
  if (provenance.export_record !== null) references.push(path(provenance.export_record));
  for (const entry of array(top.evidence, 0, 64)) {
    const evidence = object(entry, ["kind", "path"]);
    string(evidence.kind, TOKEN);
    references.push(path(evidence.path));
  }

  const files: JsonObject[] = [];
  for (const entry of array(top.files, 1, 65536)) {
    const item = object(entry, ["path", "media_type", "size_bytes", "sha256", "role", "executable"]);
    path(item.path);
    string(item.media_type, MEDIA_TYPE);
    uint(item.size_bytes);
    string(item.sha256, SHA256);
    choice(item.role, FILE_ROLES);
    if (typeof item.executable !== "boolean") throw fail("manifest_malformed");
    if (item.executable) throw fail("executable_file_forbidden");
    files.push(item);
  }

  const extensions = top.extensions;
  if (!isRecord(extensions) || Object.keys(extensions).length > 64) {
    throw fail("manifest_malformed");
  }
  for (const name of Object.keys(extensions)) string(name, EXTENSION);
  const critical = array(top.critical_extensions, 0, 64).map((item) => string(item, EXTENSION));
  unique(critical);
  if (!critical.every((name) => Object.hasOwn(extensions, name))) {
    throw fail("manifest_malformed");
  }

  const paths = files.map((item) => item.path as string);
  const folded = paths.map((item) => item.toLowerCase());
  if (new Set(folded).size !== folded.length) throw fail("duplicate_path");
  const foldedSet = new Set(folded);
  for (const item of folded) {
    const segments = item.split("/");
    for (let index = 1; index < segments.length; index += 1) {
      if (foldedSet.has(segments.slice(0, index).join("/"))) throw fail("path_conflict");
    }
  }
  for (let index = 0; index + 1 < paths.length; index += 1) {
    if (!utf8Less(paths[index] as string, paths[index + 1] as string)) {
      throw fail("file_order_invalid");
    }
  }
  const declared = new Set(paths);
  if (!references.every((reference) => declared.has(reference))) {
    throw fail("file_reference_undeclared");
  }

  const runtimeIds = runtimes.map((item) => item.id as string);
  unique(runtimeIds);
  const byId = new Map(runtimes.map((item) => [item.id as string, item]));
  for (const runtime of runtimes) {
    let current = runtime.fallback as string | null;
    if (current === null) continue;
    if (!byId.has(current) || current === runtime.id) throw fail("fallback_invalid");
    const seen = new Set<string>([runtime.id as string]);
    while (current !== null) {
      if (seen.has(current)) throw fail("fallback_invalid");
      seen.add(current);
      const next: JsonObject | undefined = byId.get(current);
      if (next === undefined) throw fail("fallback_invalid");
      current = next.fallback as string | null;
    }
  }

  if (!critical.every((name) => understood.has(name))) throw fail("unknown_critical_extension");

  return {
    bundleId,
    sequence,
    issuedAt,
    expiresAt,
    sdks,
    policyRange: [policyMin, policyMax],
    runtimes,
    files,
  };
}

// ---------------------------------------------------------------------------
// Envelope and signatures
// ---------------------------------------------------------------------------

function parseEnvelope(data: Uint8Array): { contentId: string; signatures: JsonObject[] } {
  let value: Json;
  try {
    value = parseCanonical(data);
  } catch {
    throw fail("envelope_malformed");
  }
  if (!isRecord(value) || value.schema !== PORTABLE_BUNDLE_ENVELOPE_SCHEMA) {
    throw fail("envelope_malformed");
  }
  if (typeof value.schema_version !== "number") throw fail("envelope_malformed");
  if (value.schema_version !== 1) throw fail("envelope_version_unsupported");
  if (!hasExactKeys(value, ["schema", "schema_version", "content_id", "signatures"])) {
    throw fail("envelope_malformed");
  }
  const contentId = value.content_id;
  const signatures = value.signatures;
  if (
    typeof contentId !== "string" ||
    !SHA256.test(contentId) ||
    !Array.isArray(signatures) ||
    signatures.length > MAX_SIGNATURES
  ) {
    throw fail("envelope_malformed");
  }
  for (const entry of signatures) {
    if (
      !isRecord(entry) ||
      !hasExactKeys(entry, ["kid", "alg", "sig"]) ||
      typeof entry.kid !== "string" ||
      !KID.test(entry.kid) ||
      typeof entry.alg !== "string" ||
      typeof entry.sig !== "string"
    ) {
      throw fail("envelope_malformed");
    }
  }
  return { contentId, signatures: signatures as JsonObject[] };
}

async function checkSignatures(
  signatures: readonly JsonObject[],
  manifest: Uint8Array,
  policy: PortableBundleTrustPolicy,
  now: string,
): Promise<number> {
  if (signatures.length === 0) throw fail("signature_missing");
  const message = portableBundleSigningInput(manifest);
  const valid = new Set<string>();
  let firstFailure: PortableBundleReason | null = null;
  for (const entry of signatures) {
    const kid = entry.kid as string;
    if (valid.has(kid)) continue;
    const reason = await signatureFailure(entry, kid, message, policy, now);
    if (reason === null) valid.add(kid);
    else if (firstFailure === null) firstFailure = reason;
  }
  if (valid.size >= policy.threshold) return valid.size;
  throw fail(firstFailure ?? "signature_threshold_not_met");
}

async function signatureFailure(
  entry: JsonObject,
  kid: string,
  message: Uint8Array,
  policy: PortableBundleTrustPolicy,
  now: string,
): Promise<PortableBundleReason | null> {
  if (entry.alg !== "Ed25519") return "signature_algorithm_unsupported";
  const key = policy.keys.find((candidate) => candidate.kid === kid);
  if (key === undefined) return "untrusted_key";
  if (key.revoked) return "key_revoked";
  if (now < key.notBefore) return "key_not_yet_valid";
  if (now >= key.notAfter) return "key_expired";
  const signature = base64UrlDecode(entry.sig, 64);
  if (signature === null || !(await ed25519Verify(key.publicKey, signature, message))) {
    return "signature_invalid";
  }
  return null;
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

function validateContext(context: PortableBundleContext): void {
  const invalid = () => new PortableBundleError("context_invalid");
  const limits = [
    context.maxManifestBytes,
    context.maxEnvelopeBytes,
    context.maxTotalBytes,
    context.maxFiles,
  ];
  if (
    !isTimestamp(context.now) ||
    !limits.every((limit) => limit === undefined || (Number.isSafeInteger(limit) && limit > 0)) ||
    !(context.understoodExtensions ?? []).every((item) => EXTENSION.test(item)) ||
    !Object.values(context.sequenceFloors ?? {}).every(
      (floor) => Number.isSafeInteger(floor) && floor >= 0,
    )
  ) {
    throw invalid();
  }
  const target = context.target;
  if (target === undefined || target === null) return;
  const optionalUint = (item: number | null | undefined) =>
    item === undefined || item === null || (Number.isSafeInteger(item) && item >= 0);
  if (
    !PLATFORMS.includes(target.platform) ||
    !TOKEN.test(target.sdkName) ||
    !isVersion(target.sdkVersion) ||
    !target.runtimes.every((item) => Object.hasOwn(FORMAT_PLATFORMS, item)) ||
    !(target.osVersion === undefined || target.osVersion === null || isVersion(target.osVersion)) ||
    !optionalUint(target.policyVersion) ||
    !optionalUint(target.memoryBytes) ||
    !optionalUint(target.storageBytes)
  ) {
    throw invalid();
  }
}

/**
 * Verify a portable bundle offline. Checks run in the normative order and the
 * first failure is reported. `filesVerified` is true only when `files` was
 * supplied and matched the manifest exactly.
 */
export async function verifyPortableBundle(
  input: VerifyPortableBundleInput,
): Promise<PortableBundleVerification> {
  const { manifest, envelope, trustPolicy, context } = input;
  validateContext(context);
  const limits = { ...PORTABLE_BUNDLE_DEFAULT_LIMITS, ...definedLimits(context) };
  const now = context.now;
  let contentId: string | null = null;
  let validSignatures = 0;
  let eligibleRuntimes: number[] = [];
  let filesChecked = 0;
  try {
    if (manifest.length > limits.maxManifestBytes) throw fail("manifest_too_large");
    contentId = await sha256(manifest);
    if (envelope.length > limits.maxEnvelopeBytes) throw fail("envelope_too_large");
    const parsed = parseEnvelope(envelope);
    if (parsed.contentId !== contentId) throw fail("content_id_mismatch");
    if (trustPolicy.revokedContentIds.has(contentId)) throw fail("bundle_revoked");
    validSignatures = await checkSignatures(parsed.signatures, manifest, trustPolicy, now);
    const value = parseCanonical(manifest);
    const validated = validateManifest(value, new Set(context.understoodExtensions ?? []));
    const declaredBytes = validated.files.reduce(
      (total, item) => total + (item.size_bytes as number),
      0,
    );
    if (validated.files.length > limits.maxFiles || declaredBytes > limits.maxTotalBytes) {
      throw fail("size_limit_exceeded");
    }
    if (now < validated.issuedAt) throw fail("bundle_not_yet_valid");
    if (now >= validated.expiresAt) throw fail("bundle_expired");
    const floor = context.sequenceFloors?.[validated.bundleId];
    if (floor !== undefined && validated.sequence < floor) throw fail("rollback_rejected");
    eligibleRuntimes = eligible(validated, context.target ?? null);
    const files = input.files;
    if (files !== undefined && files !== null) {
      for (const item of validated.files) {
        const data = files.get(item.path as string);
        if (data === undefined) throw fail("file_missing");
        if (data.length !== item.size_bytes) throw fail("file_size_mismatch");
        if ((await sha256(data)) !== item.sha256) throw fail("file_digest_mismatch");
        filesChecked += 1;
      }
      const declared = new Set(validated.files.map((item) => item.path as string));
      for (const candidate of files.keys()) {
        if (!declared.has(candidate) && !RESERVED_PATHS.includes(candidate)) {
          throw fail("file_undeclared");
        }
      }
    }
  } catch (error) {
    if (error instanceof Refusal) {
      return {
        ok: false,
        reason: error.reason,
        contentId,
        validSignatures,
        eligibleRuntimes,
        filesChecked,
        filesVerified: false,
      };
    }
    throw error;
  }
  return {
    ok: true,
    reason: null,
    contentId,
    validSignatures,
    eligibleRuntimes,
    filesChecked,
    filesVerified: input.files !== undefined && input.files !== null,
  };
}

function definedLimits(context: PortableBundleContext): Partial<
  Record<keyof typeof PORTABLE_BUNDLE_DEFAULT_LIMITS, number>
> {
  const output: Partial<Record<keyof typeof PORTABLE_BUNDLE_DEFAULT_LIMITS, number>> = {};
  if (context.maxManifestBytes !== undefined) output.maxManifestBytes = context.maxManifestBytes;
  if (context.maxEnvelopeBytes !== undefined) output.maxEnvelopeBytes = context.maxEnvelopeBytes;
  if (context.maxTotalBytes !== undefined) output.maxTotalBytes = context.maxTotalBytes;
  if (context.maxFiles !== undefined) output.maxFiles = context.maxFiles;
  return output;
}

function eligible(manifest: ValidatedManifest, target: PortableBundleTarget | null): number[] {
  if (target === null) return manifest.runtimes.map((_runtime, index) => index);
  const sdk = manifest.sdks.find((item) => item[0] === target.sdkName);
  if (
    sdk === undefined ||
    compareVersions(target.sdkVersion, sdk[1]) < 0 ||
    (sdk[2] !== null && compareVersions(target.sdkVersion, sdk[2]) >= 0)
  ) {
    throw fail("sdk_incompatible");
  }
  const [low, high] = manifest.policyRange;
  const policyVersion = target.policyVersion;
  if (
    policyVersion !== undefined &&
    policyVersion !== null &&
    !(low <= policyVersion && policyVersion < high)
  ) {
    throw fail("policy_incompatible");
  }
  const indexes: number[] = [];
  manifest.runtimes.forEach((runtime, index) => {
    if (runtimeFits(runtime, target)) indexes.push(index);
  });
  if (indexes.length === 0) throw fail("platform_incompatible");
  return indexes;
}

function runtimeFits(runtime: JsonObject, target: PortableBundleTarget): boolean {
  if (!target.runtimes.includes(runtime.format as string)) return false;
  if (
    target.memoryBytes !== undefined &&
    target.memoryBytes !== null &&
    (runtime.min_memory_bytes as number) > target.memoryBytes
  ) {
    return false;
  }
  if (
    target.storageBytes !== undefined &&
    target.storageBytes !== null &&
    (runtime.min_storage_bytes as number) > target.storageBytes
  ) {
    return false;
  }
  for (const entry of runtime.platforms as JsonObject[]) {
    if (entry.platform !== target.platform) continue;
    const low = entry.min_os as string | null;
    const high = entry.max_os_exclusive as string | null;
    if (low === null && high === null) return true;
    const osVersion = target.osVersion;
    if (osVersion === undefined || osVersion === null) return false;
    return (
      (low === null || compareVersions(osVersion, low) >= 0) &&
      (high === null || compareVersions(osVersion, high) < 0)
    );
  }
  return false;
}

/**
 * Parse and validate canonical manifest bytes without checking signatures.
 * Returns the parsed value; unknown non-critical extensions are preserved, so
 * `canonicalJson(value)` reproduces the input bytes exactly.
 */
export function parsePortableBundleManifest(
  manifest: Uint8Array,
  understoodExtensions: readonly string[] = [],
): Record<string, unknown> {
  try {
    const value = parseCanonical(manifest);
    validateManifest(value, new Set(understoodExtensions));
    return value as Record<string, unknown>;
  } catch (error) {
    if (error instanceof Refusal) throw new PortableBundleError(error.reason);
    throw error;
  }
}
