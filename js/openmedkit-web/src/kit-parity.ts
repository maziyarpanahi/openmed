import type { OpenMedDeidentifyResult, OpenMedSpan } from "./types";

/**
 * OpenMedKit kit-parity result profile (`openmed.kit_parity_result` v1).
 *
 * Every OpenMedKit surface projects its native de-identification result into
 * this value-free record and serializes it to the same canonical JSON bytes.
 * The shared golden vectors live in
 * `tests/fixtures/parity/kit_parity_vectors.json`.
 *
 * Offsets are half-open Unicode scalar (code point) offsets. JavaScript string
 * indices are UTF-16 code units, so the web package keeps UTF-16 offsets on
 * `OpenMedSpan` and converts them only through the helpers below.
 */
export const KIT_PARITY_PROFILE = "openmed.kit_parity_result" as const;
export const KIT_PARITY_PROFILE_VERSION = 1 as const;
export const KIT_PARITY_OFFSET_UNIT = "unicode_scalar" as const;
export const KIT_PARITY_CONFIDENCE_SCALE = 1_000_000 as const;
export const KIT_PARITY_OFFSET_CONTRACT_VERSIONS: readonly number[] = [1];
export const KIT_PARITY_DECODER_CONTRACT_VERSIONS: readonly number[] = [1];

export const KIT_PARITY_ACTIONS = [
  "hash",
  "keep",
  "mask",
  "redact",
  "remove",
  "replace",
] as const;

export type KitParityAction = (typeof KIT_PARITY_ACTIONS)[number];

export const KIT_PARITY_CANONICAL_LABELS = [
  "ACCOUNT_NUMBER",
  "AGE",
  "AMOUNT",
  "ANTIBIOTIC",
  "API_KEY",
  "BIC",
  "BITCOIN_ADDRESS",
  "BODY_SITE",
  "BUILDING_NUMBER",
  "CONDITION",
  "CREDIT_CARD",
  "CREDIT_CARD_ISSUER",
  "CURRENCY",
  "CVV",
  "DATE",
  "DATE_OF_BIRTH",
  "DIET_TYPE",
  "EMAIL",
  "ETHEREUM_ADDRESS",
  "EYE_COLOR",
  "FEEDING_ROUTE",
  "FIRST_NAME",
  "GENDER",
  "GPS_COORDINATES",
  "HEIGHT",
  "IBAN",
  "ID_NUM",
  "IMEI",
  "IP_ADDRESS",
  "JOB_DEPARTMENT",
  "JOB_TITLE",
  "LAB_TEST",
  "LAST_NAME",
  "LITECOIN_ADDRESS",
  "LOCATION",
  "MAC_ADDRESS",
  "MASKED_NUMBER",
  "MEDICATION",
  "MICROORGANISM",
  "MIDDLE_NAME",
  "NUTRITIONAL_STATUS",
  "NUTRITION_TARGET",
  "OCCUPATION",
  "ORDINAL_DIRECTION",
  "ORGANIZATION",
  "OTHER",
  "PASSWORD",
  "PERSON",
  "PHONE",
  "PIN",
  "PREFIX",
  "PROCEDURE",
  "SSN",
  "STREET_ADDRESS",
  "SUSCEPTIBILITY",
  "TIME",
  "URL",
  "USERNAME",
  "USER_AGENT",
  "VEHICLE_REGISTRATION",
  "VIN",
  "ZIPCODE",
] as const;

/** One applied span; offsets are Unicode scalar offsets into the source. */
export interface KitParitySpan {
  label: string;
  canonical_label: string;
  action: string;
  start: number;
  end: number;
  /** Confidence in parts per million; omitted when the surface has no score. */
  confidence_ppm?: number;
  /** Replacement text; omitted for `keep`. */
  replacement?: string;
}

export interface KitParityResult {
  deidentified_text: string;
  spans: KitParitySpan[];
}

export interface KitParityBundleContract {
  offsetContractVersion: number;
  decoderContractVersion: number;
}

/** Value-free profile violation: a reason code and a JSON field path only. */
export class KitParityError extends Error {
  readonly code: string;
  readonly fieldPath: string;

  constructor(code: string, fieldPath = "$") {
    super(`${code} at ${fieldPath}`);
    this.name = "KitParityError";
    this.code = code;
    this.fieldPath = fieldPath;
  }
}

const LABEL_PATTERN = /^[A-Za-z][A-Za-z0-9_:-]{0,79}$/;
const ACTION_SET = new Set<string>(KIT_PARITY_ACTIONS);
const LABEL_SET = new Set<string>(KIT_PARITY_CANONICAL_LABELS);
const TOP_LEVEL_KEYS = new Set([
  "deidentified_text",
  "offset_unit",
  "profile",
  "profile_version",
  "spans",
]);
const SPAN_KEYS = new Set([
  "action",
  "canonical_label",
  "confidence_ppm",
  "end",
  "label",
  "replacement",
  "start",
]);

type FieldKind = "string" | "integer" | "array";

/** Return the number of Unicode scalar values in `text`. */
export function scalarLength(text: string): number {
  let count = 0;
  for (const _character of text) {
    count += 1;
  }
  return count;
}

/** Convert a Unicode scalar offset into a JavaScript UTF-16 string index. */
export function scalarToUtf16Offset(text: string, scalarOffset: number): number {
  return scalarToNative(text, scalarOffset, (character) => character.length);
}

/** Convert a Unicode scalar offset into a UTF-8 byte offset. */
export function scalarToUtf8Offset(text: string, scalarOffset: number): number {
  return scalarToNative(text, scalarOffset, utf8Width);
}

/**
 * Convert a JavaScript UTF-16 string index into a Unicode scalar offset.
 *
 * An index between the halves of a surrogate pair is rejected with
 * `offset_not_scalar_boundary`.
 */
export function utf16ToScalarOffset(text: string, utf16Offset: number): number {
  return nativeToScalar(text, utf16Offset, (character) => character.length);
}

/**
 * Convert a UTF-8 byte offset into a Unicode scalar offset.
 *
 * An offset inside a multi-byte sequence is rejected with
 * `offset_not_scalar_boundary`.
 */
export function utf8ToScalarOffset(text: string, utf8Offset: number): number {
  return nativeToScalar(text, utf8Offset, utf8Width);
}

/** Encode a confidence in `[0, 1]` as `floor(confidence * 1e6 + 0.5)`. */
export function confidenceToPpm(confidence: number): number {
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new KitParityError("confidence_invalid", "$.confidence");
  }
  return Math.floor(confidence * KIT_PARITY_CONFIDENCE_SCALE + 0.5);
}

/**
 * Reject bundle contracts the profile cannot interpret. Call this before any
 * result produced under the bundle is consumed.
 */
export function checkKitParityCompatibility(bundle: KitParityBundleContract): void {
  if (!KIT_PARITY_OFFSET_CONTRACT_VERSIONS.includes(bundle.offsetContractVersion)) {
    throw new KitParityError(
      "offset_contract_version_unsupported",
      "$.offset_contract_version",
    );
  }
  if (!KIT_PARITY_DECODER_CONTRACT_VERSIONS.includes(bundle.decoderContractVersion)) {
    throw new KitParityError(
      "decoder_contract_version_unsupported",
      "$.decoder_contract_version",
    );
  }
}

/** Validate a result against the profile vocabulary and ordering rules. */
export function validateKitParityResult(result: KitParityResult): void {
  if (typeof result.deidentified_text !== "string") {
    throw new KitParityError("field_type_invalid", "$.deidentified_text");
  }
  let previous: KitParitySpan | undefined;
  result.spans.forEach((span, index) => {
    const path = `$.spans[${index}]`;
    checkSpan(span, path);
    checkOrder(span, previous, path);
    previous = span;
  });
}

/** Return the canonical JSON text of a result (UTF-8 encode it for bytes). */
export function serializeKitParityResult(result: KitParityResult): string {
  validateKitParityResult(result);
  const spans = result.spans.map((span) => {
    let record = `{"action":${JSON.stringify(span.action)}`;
    record += `,"canonical_label":${JSON.stringify(span.canonical_label)}`;
    if (span.confidence_ppm !== undefined) {
      record += `,"confidence_ppm":${span.confidence_ppm}`;
    }
    record += `,"end":${span.end},"label":${JSON.stringify(span.label)}`;
    if (span.replacement !== undefined) {
      record += `,"replacement":${JSON.stringify(span.replacement)}`;
    }
    return `${record},"start":${span.start}}`;
  });
  return (
    `{"deidentified_text":${JSON.stringify(result.deidentified_text)}` +
    `,"offset_unit":"${KIT_PARITY_OFFSET_UNIT}"` +
    `,"profile":"${KIT_PARITY_PROFILE}"` +
    `,"profile_version":${KIT_PARITY_PROFILE_VERSION}` +
    `,"spans":[${spans.join(",")}]}`
  );
}

/**
 * Parse and strictly validate canonical kit-parity result JSON.
 *
 * Checks run in the order shared by every surface: JSON syntax, object shape,
 * profile, profile version, unknown top-level keys, offset unit, de-identified
 * text, each span, span ordering, and finally the canonical bytes.
 */
export function parseKitParityResult(json: string | Uint8Array): KitParityResult {
  let text: string;
  if (typeof json === "string") {
    text = json;
  } else {
    try {
      text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(json);
    } catch {
      throw new KitParityError("json_invalid");
    }
  }
  let document: unknown;
  try {
    document = JSON.parse(text);
  } catch {
    throw new KitParityError("json_invalid");
  }
  if (!isRecord(document)) {
    throw new KitParityError("result_not_object");
  }
  if (required(document, "profile", "$", "string") !== KIT_PARITY_PROFILE) {
    throw new KitParityError("profile_unknown", "$.profile");
  }
  if (
    required(document, "profile_version", "$", "integer") !==
    KIT_PARITY_PROFILE_VERSION
  ) {
    throw new KitParityError("profile_version_unsupported", "$.profile_version");
  }
  if (Object.keys(document).some((key) => !TOP_LEVEL_KEYS.has(key))) {
    throw new KitParityError("field_unknown", "$");
  }
  if (required(document, "offset_unit", "$", "string") !== KIT_PARITY_OFFSET_UNIT) {
    throw new KitParityError("offset_unit_unsupported", "$.offset_unit");
  }
  const deidentifiedText = required(document, "deidentified_text", "$", "string");
  const rawSpans = required(document, "spans", "$", "array");

  const spans: KitParitySpan[] = [];
  let previous: KitParitySpan | undefined;
  rawSpans.forEach((raw, index) => {
    const path = `$.spans[${index}]`;
    const span = parseSpan(raw, path);
    checkOrder(span, previous, path);
    spans.push(span);
    previous = span;
  });

  const result: KitParityResult = { deidentified_text: deidentifiedText, spans };
  if (serializeKitParityResult(result) !== text) {
    throw new KitParityError("result_not_canonical");
  }
  return result;
}

/** Gate a bundle's contracts, then parse a result produced under it. */
export function acceptKitParityResult(
  json: string | Uint8Array,
  bundle: KitParityBundleContract,
): KitParityResult {
  checkKitParityCompatibility(bundle);
  return parseKitParityResult(json);
}

/** Apply span replacements right to left at Unicode scalar offsets. */
export function applyKitParitySpans(
  sourceText: string,
  spans: readonly KitParitySpan[],
): string {
  validateKitParityResult({ deidentified_text: "", spans: [...spans] });
  const length = scalarLength(sourceText);
  spans.forEach((span, index) => {
    if (span.end > length) {
      throw new KitParityError("offset_out_of_bounds", `$.spans[${index}].end`);
    }
  });
  let result = sourceText;
  for (const span of [...spans].reverse()) {
    if (span.replacement !== undefined) {
      const start = scalarToUtf16Offset(result, span.start);
      const end = scalarToUtf16Offset(result, span.end);
      result = result.slice(0, start) + span.replacement + result.slice(end);
    }
  }
  return result;
}

/** Check that a result's offsets and UTF-8 bytes agree with its source text. */
export function verifyKitParityResult(
  result: KitParityResult,
  sourceText: string,
): void {
  validateKitParityResult(result);
  const expected = new TextEncoder().encode(applyKitParitySpans(sourceText, result.spans));
  const actual = new TextEncoder().encode(result.deidentified_text);
  if (
    expected.length !== actual.length ||
    expected.some((byte, index) => byte !== actual[index])
  ) {
    throw new KitParityError("deidentified_text_mismatch", "$.deidentified_text");
  }
}

/**
 * Project a web de-identification result into the kit-parity profile.
 *
 * Span offsets are converted from JavaScript UTF-16 indices to Unicode scalar
 * offsets against `result.text`; `entity_type` becomes `label`; a `null` score
 * omits `confidence_ppm`. Actions outside the profile, such as
 * `format_preserve`, are rejected rather than mapped.
 */
export function toKitParityResult(result: OpenMedDeidentifyResult): KitParityResult {
  const ordered = [...result.spans].sort(
    (left, right) => left.start - right.start || left.end - right.end,
  );
  const spans = ordered.map((span, index) =>
    projectSpan(result.text, span, `$.spans[${index}]`),
  );
  const projected: KitParityResult = {
    deidentified_text: result.deidentifiedText,
    spans,
  };
  validateKitParityResult(projected);
  return projected;
}

function projectSpan(text: string, span: OpenMedSpan, path: string): KitParitySpan {
  const start = convertNative(text, span.start, `${path}.start`);
  const end = convertNative(text, span.end, `${path}.end`);
  if (!ACTION_SET.has(span.action)) {
    throw new KitParityError("action_unknown", `${path}.action`);
  }
  const projected: KitParitySpan = {
    label: span.entity_type,
    canonical_label: span.canonical_label,
    action: span.action,
    start,
    end,
  };
  if (span.score !== null) {
    projected.confidence_ppm = convertConfidence(span.score, path);
  }
  if (span.action !== "keep") {
    if (span.replacement === null) {
      throw new KitParityError("field_missing", `${path}.replacement`);
    }
    projected.replacement = span.replacement;
  }
  return projected;
}

function convertNative(text: string, offset: number, path: string): number {
  try {
    return utf16ToScalarOffset(text, offset);
  } catch (error) {
    if (error instanceof KitParityError) {
      throw new KitParityError(error.code, path);
    }
    throw error;
  }
}

function convertConfidence(score: number, path: string): number {
  try {
    return confidenceToPpm(score);
  } catch {
    throw new KitParityError("confidence_invalid", `${path}.confidence_ppm`);
  }
}

function parseSpan(raw: unknown, path: string): KitParitySpan {
  if (!isRecord(raw)) {
    throw new KitParityError("field_type_invalid", path);
  }
  if (Object.keys(raw).some((key) => !SPAN_KEYS.has(key))) {
    throw new KitParityError("field_unknown", path);
  }
  const label = required(raw, "label", path, "string");
  checkLabel(label, path);
  const canonicalLabel = required(raw, "canonical_label", path, "string");
  checkCanonicalLabel(canonicalLabel, path);
  const action = required(raw, "action", path, "string");
  checkAction(action, path);
  const start = required(raw, "start", path, "integer");
  checkStart(start, path);
  const end = required(raw, "end", path, "integer");
  checkEnd(start, end, path);
  const confidence = optional(raw, "confidence_ppm", path, "integer");
  checkConfidence(confidence, path);
  const replacement = optional(raw, "replacement", path, "string");
  checkReplacement(action, replacement, path);

  const span: KitParitySpan = { label, canonical_label: canonicalLabel, action, start, end };
  if (confidence !== undefined) {
    span.confidence_ppm = confidence;
  }
  if (replacement !== undefined) {
    span.replacement = replacement;
  }
  return span;
}

function checkSpan(span: KitParitySpan, path: string): void {
  typed(span.label, `${path}.label`, "string");
  checkLabel(span.label, path);
  typed(span.canonical_label, `${path}.canonical_label`, "string");
  checkCanonicalLabel(span.canonical_label, path);
  typed(span.action, `${path}.action`, "string");
  checkAction(span.action, path);
  typed(span.start, `${path}.start`, "integer");
  checkStart(span.start, path);
  typed(span.end, `${path}.end`, "integer");
  checkEnd(span.start, span.end, path);
  if (span.confidence_ppm !== undefined) {
    typed(span.confidence_ppm, `${path}.confidence_ppm`, "integer");
  }
  checkConfidence(span.confidence_ppm, path);
  if (span.replacement !== undefined) {
    typed(span.replacement, `${path}.replacement`, "string");
  }
  checkReplacement(span.action, span.replacement, path);
}

function checkLabel(label: string, path: string): void {
  if (!LABEL_PATTERN.test(label)) {
    throw new KitParityError("label_invalid", `${path}.label`);
  }
}

function checkCanonicalLabel(label: string, path: string): void {
  if (!LABEL_SET.has(label)) {
    throw new KitParityError("label_unknown", `${path}.canonical_label`);
  }
}

function checkAction(action: string, path: string): void {
  if (!ACTION_SET.has(action)) {
    throw new KitParityError("action_unknown", `${path}.action`);
  }
}

function checkStart(start: number, path: string): void {
  if (start < 0) {
    throw new KitParityError("offset_invalid", `${path}.start`);
  }
}

function checkEnd(start: number, end: number, path: string): void {
  if (end <= start) {
    throw new KitParityError("offset_invalid", `${path}.end`);
  }
}

function checkConfidence(confidence: number | undefined, path: string): void {
  if (
    confidence !== undefined &&
    (confidence < 0 || confidence > KIT_PARITY_CONFIDENCE_SCALE)
  ) {
    throw new KitParityError("confidence_invalid", `${path}.confidence_ppm`);
  }
}

function checkReplacement(
  action: string,
  replacement: string | undefined,
  path: string,
): void {
  if (action === "keep") {
    if (replacement !== undefined) {
      throw new KitParityError("replacement_unexpected", `${path}.replacement`);
    }
  } else if (replacement === undefined) {
    throw new KitParityError("field_missing", `${path}.replacement`);
  }
}

function checkOrder(
  span: KitParitySpan,
  previous: KitParitySpan | undefined,
  path: string,
): void {
  if (previous === undefined) {
    return;
  }
  if (
    span.start < previous.start ||
    (span.start === previous.start && span.end <= previous.end)
  ) {
    throw new KitParityError("span_order_invalid", path);
  }
  if (span.start < previous.end) {
    throw new KitParityError("span_overlap", path);
  }
}

function required(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: "string",
): string;
function required(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: "integer",
): number;
function required(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: "array",
): unknown[];
function required(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: FieldKind,
): unknown {
  if (!Object.prototype.hasOwnProperty.call(record, key)) {
    throw new KitParityError("field_missing", `${path}.${key}`);
  }
  return typed(record[key], `${path}.${key}`, kind);
}

function optional(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: "string",
): string | undefined;
function optional(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: "integer",
): number | undefined;
function optional(
  record: Record<string, unknown>,
  key: string,
  path: string,
  kind: FieldKind,
): unknown {
  if (!Object.prototype.hasOwnProperty.call(record, key)) {
    return undefined;
  }
  return typed(record[key], `${path}.${key}`, kind);
}

function typed(value: unknown, path: string, kind: FieldKind): unknown {
  if (value === null) {
    throw new KitParityError("field_null", path);
  }
  const valid =
    kind === "string"
      ? typeof value === "string"
      : kind === "integer"
        ? typeof value === "number" && Number.isSafeInteger(value)
        : Array.isArray(value);
  if (!valid) {
    throw new KitParityError("field_type_invalid", path);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function utf8Width(character: string): number {
  const codePoint = character.codePointAt(0) ?? 0;
  if (codePoint < 0x80) {
    return 1;
  }
  if (codePoint < 0x800) {
    return 2;
  }
  return codePoint < 0x10000 ? 3 : 4;
}

function scalarToNative(
  text: string,
  scalarOffset: number,
  width: (character: string) => number,
): number {
  if (!Number.isSafeInteger(scalarOffset) || scalarOffset < 0) {
    throw new KitParityError("offset_out_of_bounds", "$.offset");
  }
  let scalar = 0;
  let native = 0;
  for (const character of text) {
    if (scalar === scalarOffset) {
      return native;
    }
    scalar += 1;
    native += width(character);
  }
  if (scalar === scalarOffset) {
    return native;
  }
  throw new KitParityError("offset_out_of_bounds", "$.offset");
}

function nativeToScalar(
  text: string,
  nativeOffset: number,
  width: (character: string) => number,
): number {
  if (!Number.isSafeInteger(nativeOffset) || nativeOffset < 0) {
    throw new KitParityError("offset_out_of_bounds", "$.offset");
  }
  let scalar = 0;
  let native = 0;
  for (const character of text) {
    if (native === nativeOffset) {
      return scalar;
    }
    native += width(character);
    scalar += 1;
    if (native > nativeOffset) {
      throw new KitParityError("offset_not_scalar_boundary", "$.offset");
    }
  }
  if (native === nativeOffset) {
    return scalar;
  }
  throw new KitParityError("offset_out_of_bounds", "$.offset");
}
