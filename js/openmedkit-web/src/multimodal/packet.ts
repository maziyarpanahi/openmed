/**
 * Admission of `openmed.multimodal.apple.v1` evidence packets.
 *
 * This is the browser and Node port of `openmed.multimodal.apple_contract`
 * (Python) and `MultimodalPacket.fromJSON` (OpenMedKit Swift). Checks run in
 * the Python order, so a packet with several faults is refused with the same
 * code on every platform. See docs/multimodal/openmedkit-contracts.md.
 */

import {
  JsonNumber,
  canonicalJson,
  isJsonArray,
  isJsonObject,
  parseStrictJson,
  type JsonObject,
  type JsonValue,
} from "./json";

/** Envelope schema id produced by Python and OpenMedKit. */
export const MULTIMODAL_PACKET_SCHEMA = "openmed.multimodal.apple.v1";

/** Every envelope schema id this package admits. */
export const MULTIMODAL_PACKET_SCHEMAS: readonly string[] = Object.freeze([
  MULTIMODAL_PACKET_SCHEMA,
]);

/** Largest admitted wire packet, in bytes. */
export const MULTIMODAL_MAX_WIRE_BYTES = 65536;

/** Closed refusal vocabulary shared with Python and Swift. */
export const MULTIMODAL_REFUSALS = Object.freeze([
  "invalid_multimodal_contract",
  "unsupported_multimodal_schema",
  "unsupported_multimodal_capability",
  "invalid_multimodal_transform",
  "incomplete_multimodal_provenance",
] as const);

export type MultimodalRefusal = (typeof MULTIMODAL_REFUSALS)[number];

export const MULTIMODAL_PROJECTION_KINDS = Object.freeze([
  "asset",
  "visual",
  "transcript",
  "provider",
  "clock",
] as const);

export type MultimodalProjectionKind = (typeof MULTIMODAL_PROJECTION_KINDS)[number];

export const MULTIMODAL_PROVIDER_OUTCOMES = Object.freeze([
  "success",
  "abstention",
  "provider_unavailable",
  "validation_failure",
] as const);

export type MultimodalProviderOutcome = (typeof MULTIMODAL_PROVIDER_OUTCOMES)[number];

export const MULTIMODAL_ABSTENTION_CODES = Object.freeze([
  "unsupported_media",
  "malformed_media",
  "resource_limit",
  "low_quality",
  "phi_uncertainty",
  "speaker_uncertainty",
  "temporal_instability",
  "consent_unavailable",
  "consent_withdrawn",
  "capture_interrupted",
  "language_unsupported",
  "language_uncertain",
  "policy_refused",
] as const);

export type MultimodalAbstentionCode = (typeof MULTIMODAL_ABSTENTION_CODES)[number];

export const MULTIMODAL_CLOCK_REASONS = Object.freeze([
  "insufficient-anchors",
  "inconsistent-clock-anchors",
  "clock-coverage-missing",
  "clock-discontinuity",
  "anchor-gap-exceeds-policy",
  "clock-epoch-mismatch",
  "clock-declaration-mismatch",
  "wall-reference-mismatch",
  "source-timing-missing",
  "source-binding-mismatch",
  "invalid-source-reference",
  "transcript-not-finalized",
  "source-requires-review",
  "alignment-uncertainty-exceeds-policy",
  "evidence-does-not-overlap",
  "overlap-is-uncertain",
  "source-changed-during-alignment",
] as const);

export type MultimodalClockReason = (typeof MULTIMODAL_CLOCK_REASONS)[number];

export type MultimodalClockState = "aligned-for-review" | "review-required";
export type MultimodalVisualKind = "page_geometry" | "pixels" | "ocr_span";
export type MultimodalSpatialUnit = "pixel" | "point";
export type MultimodalRotation = 0 | 90 | 180 | 270;
export type MultimodalSpanInvalidation = "cited_text_changed" | "cited_token_changed";
export type MultimodalProviderSchema =
  | "openmed.multimodal.provider_result.v1"
  | "openmed.multimodal.provider_result.v2";

/** Asset manifest metadata. The original asset id is only an opaque hash. */
export interface MultimodalAsset {
  readonly sourceSchema: 1;
  readonly manifestDigest: string;
  readonly assetRef: string;
  readonly mediaType: string;
  /** SHA-256 of the asset bytes, owned by the application. */
  readonly sourceDigest: string;
  /** Up to 2^63 - 1, so kept as a `bigint`. */
  readonly byteSize: bigint;
  readonly pages: number | null;
  readonly width: number | null;
  readonly height: number | null;
  readonly frames: number | null;
  readonly durationSeconds: number | null;
}

/** Top-left, half-open source rectangle in pixels or PDF points. */
export interface MultimodalRectangle {
  readonly left: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
}

/** Half-open interval; the parent field defines its unit. */
export interface MultimodalInterval {
  readonly start: number;
  readonly end: number;
}

/** Source coordinates kept separate from the checked display rotation. */
export interface MultimodalVisual {
  readonly sourceSchema: "openmed.multimodal.fused-evidence.v1";
  readonly nodeRef: string;
  readonly sourceDigest: string;
  readonly kind: MultimodalVisualKind;
  readonly page: number;
  readonly width: number;
  readonly height: number;
  readonly unit: MultimodalSpatialUnit;
  readonly sourceBox: MultimodalRectangle | null;
  /** Unicode code point offsets into the caller's OCR text, never UTF-16. */
  readonly scalarSpan: MultimodalInterval | null;
  readonly rotation: MultimodalRotation;
  readonly displayWidth: number;
  readonly displayHeight: number;
  readonly displayBox: MultimodalRectangle | null;
}

/** Transcript citation. Invalidated citations have no current coordinates. */
export interface MultimodalTranscript {
  readonly sourceSchema: "openmed.transcript.revisions.v1";
  readonly snapshotDigest: string;
  readonly sessionRef: string;
  readonly segmentRef: string;
  readonly spanRef: string;
  readonly originalRevision: number;
  readonly revision: number;
  /** Unicode code point offsets, never UTF-16 or UTF-8 offsets. */
  readonly originalScalarSpan: MultimodalInterval;
  readonly tokenRefs: readonly string[];
  readonly scalarLength: number;
  readonly finalized: boolean;
  readonly invalidation: MultimodalSpanInvalidation | null;
  readonly currentScalarSpan: MultimodalInterval | null;
  readonly relativeMilliseconds: MultimodalInterval | null;
  readonly sampleRateHz: number;
  /** Outward conversion to a declared grid, not acoustic alignment evidence. */
  readonly sampleBounds: MultimodalInterval | null;
}

export interface MultimodalProviderResult {
  readonly schemaVersion: MultimodalProviderSchema;
  readonly providerId: string;
  readonly modelId: string;
  readonly inputDigest: string;
  readonly outputDigest: string | null;
  readonly outcome: MultimodalProviderOutcome;
  readonly abstention: MultimodalAbstentionCode | null;
  readonly durationMilliseconds: number;
  /** Closed counters, each up to 2^63 - 1. */
  readonly counts: Readonly<Record<string, bigint>>;
}

/** Inclusive uncertainty envelope on a monotonic clock, never wall time. */
export interface MultimodalNanosecondBound {
  readonly lower: bigint;
  readonly upper: bigint;
}

/** Half-open nanosecond interval. */
export interface MultimodalNanosecondInterval {
  readonly start: bigint;
  readonly end: bigint;
}

export interface MultimodalClockAlignment {
  readonly sourceSchema: "openmed.multimodal.clock-alignment.v1";
  readonly sourceResultDigest: string;
  readonly state: MultimodalClockState;
  readonly reasons: readonly MultimodalClockReason[];
  readonly sourceReferenceDigest: string;
  readonly targetReferenceDigest: string;
  readonly sourceClockDigest: string;
  readonly targetClockDigest: string;
  readonly policyDigest: string;
  readonly targetStart: MultimodalNanosecondBound | null;
  readonly targetEnd: MultimodalNanosecondBound | null;
  readonly guaranteedOverlap: MultimodalNanosecondInterval | null;
  readonly uncertaintyNanoseconds: bigint | null;
}

export type MultimodalProjection =
  | { readonly kind: "asset"; readonly asset: MultimodalAsset }
  | { readonly kind: "visual"; readonly visual: MultimodalVisual }
  | { readonly kind: "transcript"; readonly transcript: MultimodalTranscript }
  | { readonly kind: "provider"; readonly provider: MultimodalProviderResult }
  | { readonly kind: "clock"; readonly clock: MultimodalClockAlignment };

/**
 * A refused packet. The message is the refusal code only; rejected values,
 * file names and source content are never included.
 */
export class MultimodalContractError extends Error {
  readonly code: MultimodalRefusal;

  constructor(code: MultimodalRefusal = "invalid_multimodal_contract") {
    super(code);
    this.name = "MultimodalContractError";
    this.code = code;
  }
}

/** Raised when no WebCrypto SHA-256 is available to verify a digest. */
export class MultimodalDigestUnavailableError extends Error {
  constructor() {
    super("webcrypto_digest_unavailable");
    this.name = "MultimodalDigestUnavailableError";
  }
}

/** The part of `SubtleCrypto` that admission uses. */
export interface MultimodalDigestProvider {
  digest(algorithm: "SHA-256", data: Uint8Array): Promise<ArrayBuffer>;
}

export interface AdmitMultimodalPacketOptions {
  /**
   * SHA-256 provider. Defaults to `globalThis.crypto.subtle`, which browsers
   * expose in secure contexts and Node.js 20+ exposes globally.
   */
  readonly subtle?: MultimodalDigestProvider;
}

// ---------------------------------------------------------------------------
// Field checks. Each mirrors a helper in apple_contract.py.
// ---------------------------------------------------------------------------

const INVALID: MultimodalRefusal = "invalid_multimodal_contract";
const SCHEMA: MultimodalRefusal = "unsupported_multimodal_schema";
const CAPABILITY: MultimodalRefusal = "unsupported_multimodal_capability";
const TRANSFORM: MultimodalRefusal = "invalid_multimodal_transform";
const PROVENANCE: MultimodalRefusal = "incomplete_multimodal_provenance";

const MAX_INTEGER = (1n << 63n) - 1n;
const MAX_COORDINATE = 2 ** 31 - 1;
const CLOCK_LIMIT = 9_000_000_000_000_000_000n;
const FUSED_EVIDENCE_SCHEMA = "openmed.multimodal.fused-evidence.v1";
const TRANSCRIPT_SCHEMA = "openmed.transcript.revisions.v1";
const CLOCK_SCHEMA = "openmed.multimodal.clock-alignment.v1";
const PROVIDER_V1 = "openmed.multimodal.provider_result.v1";
const PROVIDER_V2 = "openmed.multimodal.provider_result.v2";
const V2_ABSTENTIONS = new Set<string>([
  "consent_unavailable",
  "consent_withdrawn",
  "capture_interrupted",
  "language_unsupported",
  "language_uncertain",
  "policy_refused",
]);
const COUNT_FIELDS = new Set([
  "detection_count",
  "frame_count",
  "input_bytes",
  "input_items",
  "output_items",
  "page_count",
  "sample_count",
  "segment_count",
  "token_count",
]);
const FORBIDDEN_IDENTIFIER_PARTS = new Set([
  "bearer",
  "credential",
  "mrn",
  "password",
  "patient",
  "prompt",
  "secret",
  "token",
]);
const IDENTIFIER = /^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$/;
const MEDIA_TYPE = /^[a-z0-9][a-z0-9.+-]*\/[a-z0-9][a-z0-9.+-]*$/;
const EXACT_MEDIA_TYPES = new Set([
  "video/mp4",
  "application/dicom",
  "application/pdf",
  "application/dicom+json",
  "application/edf",
  "application/wfdb",
]);

function check(
  condition: boolean,
  code: MultimodalRefusal = INVALID,
): asserts condition {
  if (!condition) {
    throw new MultimodalContractError(code);
  }
}

function keys(value: JsonValue | undefined, names: string): JsonObject {
  check(isJsonObject(value));
  const expected = names.split(" ");
  check(value.size === expected.length && expected.every((name) => value.has(name)));
  return value;
}

function field(object: JsonObject, name: string): JsonValue {
  return object.get(name) as JsonValue;
}

/** Python `type(value) is int and minimum <= value <= maximum`. */
function integer(
  value: JsonValue,
  minimum: bigint = 0n,
  maximum: bigint = MAX_INTEGER,
): bigint {
  check(value instanceof JsonNumber && value.isInteger);
  const result = BigInt(value.text);
  check(minimum <= result && result <= maximum);
  return result;
}

function smallInteger(value: JsonValue, minimum: number, maximum: number): number {
  return Number(integer(value, BigInt(minimum), BigInt(maximum)));
}

/** Python `value == 1`, where `True` and `1.0` also compare equal. */
function equalsOne(value: JsonValue): boolean {
  if (value === true) {
    return true;
  }
  if (!(value instanceof JsonNumber)) {
    return false;
  }
  return value.isInteger ? BigInt(value.text) === 1n : Number(value.text) === 1;
}

function hex(value: JsonValue, width = 64): string {
  check(
    typeof value === "string" && value.length === width && /^[0-9a-f]+$/.test(value),
    PROVENANCE,
  );
  return value;
}

/** Big-endian IEEE-754 binary64 bits as lowercase hex; zero is always `+0`. */
function bits(value: number): string {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value === 0 ? 0 : value, false);
  let result = "";
  for (let offset = 0; offset < 8; offset += 1) {
    result += view.getUint8(offset).toString(16).padStart(2, "0");
  }
  return result;
}

/** Decode a sixteen-digit hex double, refusing NaN, infinities and `-0`. */
function hexDouble(value: JsonValue, minimum = 0, maximum = MAX_COORDINATE): number {
  check(typeof value === "string" && /^[0-9a-f]{16}$/.test(value));
  const view = new DataView(new ArrayBuffer(8));
  for (let offset = 0; offset < 8; offset += 1) {
    view.setUint8(offset, Number.parseInt(value.slice(offset * 2, offset * 2 + 2), 16));
  }
  const result = view.getFloat64(0, false);
  check(Number.isFinite(result) && minimum <= result && result <= maximum);
  check(bits(result) === value);
  return result;
}

function pair(value: JsonValue, maximum: number, allowEmpty = false): MultimodalInterval {
  check(isJsonArray(value) && value.length === 2);
  const start = smallInteger(value[0] as JsonValue, 0, maximum);
  const end = smallInteger(value[1] as JsonValue, 0, maximum);
  check(allowEmpty ? start <= end : start < end);
  return Object.freeze({ start, end });
}

function rectangle(values: readonly number[]): MultimodalRectangle {
  return Object.freeze({
    left: values[0] as number,
    top: values[1] as number,
    right: values[2] as number,
    bottom: values[3] as number,
  });
}

function optionalCount(value: JsonValue): number | null {
  return value === null ? null : smallInteger(value, 1, MAX_COORDINATE);
}

// ---------------------------------------------------------------------------
// Projection validators.
// ---------------------------------------------------------------------------

interface Validated {
  readonly projection: MultimodalProjection;
  /** A source digest to verify after the synchronous checks, as in Python. */
  readonly pendingDigest?: { readonly canonical: string; readonly expected: string };
}

function validateAsset(payload: JsonObject): Validated {
  keys(
    payload,
    "source_schema manifest_digest asset_ref media_type sha256 byte_size pages width height frames duration_f64",
  );
  check(equalsOne(field(payload, "source_schema")), SCHEMA);
  const manifestDigest = hex(field(payload, "manifest_digest"));
  const assetRef = hex(field(payload, "asset_ref"), 32);
  const sourceDigest = hex(field(payload, "sha256"));
  const rawDuration = field(payload, "duration_f64");
  const durationSeconds = rawDuration === null ? null : hexDouble(rawDuration);
  // AssetManifest(...) checks, all reported as invalid_multimodal_contract.
  const mediaType = field(payload, "media_type");
  check(typeof mediaType === "string" && MEDIA_TYPE.test(mediaType));
  check(
    EXACT_MEDIA_TYPES.has(mediaType) ||
      mediaType.startsWith("image/") ||
      mediaType.startsWith("audio/"),
  );
  const byteSize = integer(field(payload, "byte_size"), 1n, MAX_INTEGER);
  const pages = optionalCount(field(payload, "pages"));
  const width = optionalCount(field(payload, "width"));
  const height = optionalCount(field(payload, "height"));
  const frames = optionalCount(field(payload, "frames"));
  check(durationSeconds === null || durationSeconds > 0);
  integer(field(payload, "source_schema"), 1n, 1n);
  return {
    projection: Object.freeze({
      kind: "asset",
      asset: Object.freeze({
        sourceSchema: 1,
        manifestDigest,
        assetRef,
        mediaType,
        sourceDigest,
        byteSize,
        pages,
        width,
        height,
        frames,
        durationSeconds,
      }),
    }),
  };
}

function canonicalValue(value: number): number {
  return value === 0 ? 0 : value;
}

/** PageTransform.bbox: rotate the four corners clockwise, then take extents. */
function rotateBox(
  box: readonly number[],
  width: number,
  height: number,
  rotation: MultimodalRotation,
): number[] {
  const [x0, y0, x1, y1] = box as [number, number, number, number];
  const point = (x: number, y: number): [number, number] => {
    if (rotation === 0) return [x, y];
    if (rotation === 90) return [canonicalValue(height - y), x];
    if (rotation === 180) return [canonicalValue(width - x), canonicalValue(height - y)];
    return [y, canonicalValue(width - x)];
  };
  const corners = [point(x0, y0), point(x0, y1), point(x1, y0), point(x1, y1)];
  const xs = corners.map((corner) => corner[0]);
  const ys = corners.map((corner) => corner[1]);
  return [
    canonicalValue(Math.min(...xs)),
    canonicalValue(Math.min(...ys)),
    canonicalValue(Math.max(...xs)),
    canonicalValue(Math.max(...ys)),
  ];
}

function validateVisual(payload: JsonObject): Validated {
  keys(
    payload,
    "source_schema node_ref source_digest kind page width_f64 height_f64 unit box_f64 span rotation display_width_f64 display_height_f64 display_box_f64",
  );
  check(field(payload, "source_schema") === FUSED_EVIDENCE_SCHEMA, SCHEMA);
  const nodeRef = hex(field(payload, "node_ref"), 32);
  const sourceDigest = hex(field(payload, "source_digest"));
  const kind = field(payload, "kind");
  // Python's set membership raises TypeError (an invalid contract) for an
  // unhashable list or object before it can report an unsupported kind.
  check(!isJsonArray(kind) && !isJsonObject(kind));
  check(
    kind === "page_geometry" || kind === "pixels" || kind === "ocr_span",
    CAPABILITY,
  );
  const width = hexDouble(field(payload, "width_f64"));
  const height = hexDouble(field(payload, "height_f64"));
  const rawBox = field(payload, "box_f64");
  let box: number[] | null = null;
  if (rawBox !== null) {
    check(isJsonArray(rawBox) && rawBox.length === 4);
    box = rawBox.map((value) => hexDouble(value));
  }
  const rawSpan = field(payload, "span");
  const scalarSpan = rawSpan === null ? null : pair(rawSpan, MAX_COORDINATE);
  // EvidenceNode and PageGeometry checks.
  const page = smallInteger(field(payload, "page"), 0, MAX_COORDINATE);
  check(width > 0 && height > 0);
  const unit = field(payload, "unit");
  check(unit === "pixel" || unit === "point");
  if (kind === "page_geometry") {
    check(box === null);
  } else {
    check(box !== null);
    const [x0, y0, x1, y1] = box as [number, number, number, number];
    check(0 <= x0 && x0 < x1 && x1 <= width);
    check(0 <= y0 && y0 < y1 && y1 <= height);
  }
  check(kind === "ocr_span" ? scalarSpan !== null : scalarSpan === null);
  const rawRotation = field(payload, "rotation");
  check(
    rawRotation instanceof JsonNumber &&
      rawRotation.isInteger &&
      ["0", "90", "180", "270"].includes(BigInt(rawRotation.text).toString()),
    TRANSFORM,
  );
  const rotation = Number(BigInt(rawRotation.text)) as MultimodalRotation;
  const quarter = rotation === 90 || rotation === 270;
  const displayWidth = quarter ? height : width;
  const displayHeight = quarter ? width : height;
  let displayBox: number[] | null = null;
  if (box !== null) {
    displayBox = rotateBox(box, width, height, rotation);
    check(
      (displayBox[0] as number) < (displayBox[2] as number) &&
        (displayBox[1] as number) < (displayBox[3] as number),
      TRANSFORM,
    );
  }
  const rawDisplayBox = field(payload, "display_box_f64");
  check(
    field(payload, "display_width_f64") === bits(displayWidth) &&
      field(payload, "display_height_f64") === bits(displayHeight) &&
      (displayBox === null
        ? rawDisplayBox === null
        : isJsonArray(rawDisplayBox) &&
          rawDisplayBox.length === 4 &&
          displayBox.every((value, index) => rawDisplayBox[index] === bits(value))),
    TRANSFORM,
  );
  return {
    projection: Object.freeze({
      kind: "visual",
      visual: Object.freeze({
        sourceSchema: FUSED_EVIDENCE_SCHEMA,
        nodeRef,
        sourceDigest,
        kind,
        page,
        width,
        height,
        unit,
        sourceBox: box === null ? null : rectangle(box),
        scalarSpan,
        rotation,
        displayWidth,
        displayHeight,
        displayBox: displayBox === null ? null : rectangle(displayBox),
      }),
    }),
  };
}

function validateTranscript(payload: JsonObject): Validated {
  keys(
    payload,
    "source_schema snapshot_digest session_ref segment_ref span_ref original_revision revision original_span token_refs length finalized invalidation current_span time_ms sample_rate_hz sample_bounds",
  );
  check(field(payload, "source_schema") === TRANSCRIPT_SCHEMA, SCHEMA);
  const snapshotDigest = hex(field(payload, "snapshot_digest"));
  const sessionRef = hex(field(payload, "session_ref"), 32);
  const segmentRef = hex(field(payload, "segment_ref"), 32);
  const spanRef = hex(field(payload, "span_ref"), 32);
  const revision = smallInteger(field(payload, "revision"), 0, 4096);
  const originalRevision = smallInteger(field(payload, "original_revision"), 0, revision);
  const originalScalarSpan = pair(field(payload, "original_span"), 65536);
  const rawRefs = field(payload, "token_refs");
  check(isJsonArray(rawRefs) && rawRefs.length > 0 && rawRefs.length <= 4096);
  const tokenRefs = rawRefs.map((ref) => hex(ref, 32));
  check(new Set(tokenRefs).size === tokenRefs.length);
  const scalarLength = smallInteger(field(payload, "length"), 0, 65536);
  const finalized = field(payload, "finalized");
  check(typeof finalized === "boolean");
  const sampleRateHz = smallInteger(field(payload, "sample_rate_hz"), 1, 384000);
  const rawInvalidation = field(payload, "invalidation");
  let invalidation: MultimodalSpanInvalidation | null = null;
  let currentScalarSpan: MultimodalInterval | null = null;
  let relativeMilliseconds: MultimodalInterval | null = null;
  let sampleBounds: MultimodalInterval | null = null;
  if (rawInvalidation !== null) {
    check(
      rawInvalidation === "cited_text_changed" || rawInvalidation === "cited_token_changed",
    );
    invalidation = rawInvalidation;
    check(
      ["current_span", "time_ms", "sample_bounds"].every(
        (name) => field(payload, name) === null,
      ),
    );
  } else {
    currentScalarSpan = pair(field(payload, "current_span"), scalarLength);
    relativeMilliseconds = pair(field(payload, "time_ms"), 86_400_000, true);
    // A declared target sample grid, not a new acoustic alignment claim.
    const rate = BigInt(sampleRateHz);
    const expectedStart = (BigInt(relativeMilliseconds.start) * rate) / 1000n;
    const expectedEnd = (BigInt(relativeMilliseconds.end) * rate + 999n) / 1000n;
    sampleBounds = pair(field(payload, "sample_bounds"), (86_400_000 * 384000) / 1000, true);
    check(
      BigInt(sampleBounds.start) === expectedStart && BigInt(sampleBounds.end) === expectedEnd,
    );
  }
  return {
    projection: Object.freeze({
      kind: "transcript",
      transcript: Object.freeze({
        sourceSchema: TRANSCRIPT_SCHEMA,
        snapshotDigest,
        sessionRef,
        segmentRef,
        spanRef,
        originalRevision,
        revision,
        originalScalarSpan,
        tokenRefs: Object.freeze(tokenRefs),
        scalarLength,
        finalized,
        invalidation,
        currentScalarSpan,
        relativeMilliseconds,
        sampleRateHz,
        sampleBounds,
      }),
    }),
  };
}

function identifier(value: JsonValue): string {
  check(typeof value === "string" && IDENTIFIER.test(value));
  check(!value.split(/[._-]+/).some((part) => FORBIDDEN_IDENTIFIER_PARTS.has(part)));
  return value;
}

function validateProvider(payload: JsonObject): Validated {
  keys(
    payload,
    "schema_version provider_id model_id input_digest output_digest outcome abstention_code duration_f64 count_metadata",
  );
  const durationMilliseconds = hexDouble(field(payload, "duration_f64"), 0, 86_400_000);
  const version = field(payload, "schema_version");
  check(version === PROVIDER_V1 || version === PROVIDER_V2, SCHEMA);
  const code = field(payload, "abstention_code");
  let expectedVersion = PROVIDER_V1;
  if (typeof code === "string") {
    check((MULTIMODAL_ABSTENTION_CODES as readonly string[]).includes(code));
    expectedVersion = V2_ABSTENTIONS.has(code) ? PROVIDER_V2 : PROVIDER_V1;
  }
  check(version === expectedVersion, SCHEMA);
  const inputDigest = hex(field(payload, "input_digest"));
  const rawOutput = field(payload, "output_digest");
  const outputDigest = rawOutput === null ? null : hex(rawOutput);
  // ProviderResultEnvelope.from_dict checks.
  const providerId = identifier(field(payload, "provider_id"));
  const modelId = identifier(field(payload, "model_id"));
  const outcome = field(payload, "outcome");
  check(
    typeof outcome === "string" &&
      (MULTIMODAL_PROVIDER_OUTCOMES as readonly string[]).includes(outcome),
  );
  const rawCounts = field(payload, "count_metadata");
  check(isJsonObject(rawCounts));
  const counts: Record<string, bigint> = {};
  for (const [name, value] of [...rawCounts.entries()].sort(([a], [b]) =>
    a < b ? -1 : a > b ? 1 : 0,
  )) {
    check(COUNT_FIELDS.has(name));
    counts[name] = integer(value, 0n, MAX_INTEGER);
  }
  check(code === null || typeof code === "string");
  check(outcome === "success" ? outputDigest !== null : outputDigest === null);
  check(outcome === "abstention" ? code !== null : code === null);
  return {
    projection: Object.freeze({
      kind: "provider",
      provider: Object.freeze({
        schemaVersion: version,
        providerId,
        modelId,
        inputDigest,
        outputDigest,
        outcome: outcome as MultimodalProviderOutcome,
        abstention: code as MultimodalAbstentionCode | null,
        durationMilliseconds,
        counts: Object.freeze(counts),
      }),
    }),
  };
}

function validateClock(payload: JsonObject): Validated {
  keys(
    payload,
    "source_schema source_result_digest state reasons source_reference_digest target_reference_digest source_clock_digest target_clock_digest policy_digest target_start_ns target_end_ns guaranteed_overlap_ns uncertainty_ns",
  );
  check(field(payload, "source_schema") === CLOCK_SCHEMA, SCHEMA);
  const sourceResultDigest = hex(field(payload, "source_result_digest"));
  const sourceReferenceDigest = hex(field(payload, "source_reference_digest"));
  const targetReferenceDigest = hex(field(payload, "target_reference_digest"));
  const sourceClockDigest = hex(field(payload, "source_clock_digest"));
  const targetClockDigest = hex(field(payload, "target_clock_digest"));
  const policyDigest = hex(field(payload, "policy_digest"));
  const reasons = field(payload, "reasons");
  check(isJsonArray(reasons) && reasons.length <= MULTIMODAL_CLOCK_REASONS.length);
  check(
    reasons.every(
      (reason) =>
        typeof reason === "string" &&
        (MULTIMODAL_CLOCK_REASONS as readonly string[]).includes(reason),
    ),
  );
  check(new Set(reasons).size === reasons.length);
  const state = field(payload, "state");
  check(state === "aligned-for-review" || state === "review-required");
  const timing = [
    "target_start_ns",
    "target_end_ns",
    "guaranteed_overlap_ns",
    "uncertainty_ns",
  ];
  let targetStart: MultimodalNanosecondBound | null = null;
  let targetEnd: MultimodalNanosecondBound | null = null;
  let guaranteedOverlap: MultimodalNanosecondInterval | null = null;
  let uncertaintyNanoseconds: bigint | null = null;
  if (state === "review-required") {
    check(reasons.length > 0 && timing.every((name) => field(payload, name) === null));
  } else {
    check(reasons.length === 0);
    const bound = (name: string): MultimodalNanosecondBound => {
      const value = keys(field(payload, name), "lower upper");
      const lower = integer(field(value, "lower"), -CLOCK_LIMIT, CLOCK_LIMIT);
      const upper = integer(field(value, "upper"), lower, CLOCK_LIMIT);
      return Object.freeze({ lower, upper });
    };
    targetStart = bound("target_start_ns");
    targetEnd = bound("target_end_ns");
    const overlap = field(payload, "guaranteed_overlap_ns");
    check(isJsonArray(overlap) && overlap.length === 2);
    const start = integer(overlap[0] as JsonValue, targetStart.upper, targetEnd.lower);
    const end = integer(overlap[1] as JsonValue, start + 1n, targetEnd.lower);
    check(start < end);
    guaranteedOverlap = Object.freeze({ start, end });
    const widest = [targetStart, targetEnd]
      .map(({ lower, upper }) => upper - lower)
      .reduce((left, right) => (left > right ? left : right));
    uncertaintyNanoseconds = integer(field(payload, "uncertainty_ns"), widest, CLOCK_LIMIT);
  }
  const original = new Map(
    [...payload.entries()].filter(
      ([name]) => name !== "source_schema" && name !== "source_result_digest",
    ),
  );
  const sourceResult = new Map<string, JsonValue>([
    ["algorithm", CLOCK_SCHEMA],
    ["result", original],
  ]);
  return {
    projection: Object.freeze({
      kind: "clock",
      clock: Object.freeze({
        sourceSchema: CLOCK_SCHEMA,
        sourceResultDigest,
        state,
        reasons: Object.freeze([...(reasons as readonly MultimodalClockReason[])]),
        sourceReferenceDigest,
        targetReferenceDigest,
        sourceClockDigest,
        targetClockDigest,
        policyDigest,
        targetStart,
        targetEnd,
        guaranteedOverlap,
        uncertaintyNanoseconds,
      }),
    }),
    pendingDigest: { canonical: canonicalJson(sourceResult), expected: sourceResultDigest },
  };
}

const VALIDATORS: Readonly<Record<MultimodalProjectionKind, (payload: JsonObject) => Validated>> =
  {
    asset: validateAsset,
    visual: validateVisual,
    transcript: validateTranscript,
    provider: validateProvider,
    clock: validateClock,
  };

// ---------------------------------------------------------------------------
// Admission.
// ---------------------------------------------------------------------------

const ASCII_ENCODER = new TextEncoder();

function digestProvider(options: AdmitMultimodalPacketOptions | undefined) {
  const provider =
    options?.subtle ?? (globalThis as { crypto?: { subtle?: MultimodalDigestProvider } }).crypto?.subtle;
  if (provider === undefined || typeof provider.digest !== "function") {
    throw new MultimodalDigestUnavailableError();
  }
  return provider;
}

async function sha256Hex(provider: MultimodalDigestProvider, text: string): Promise<string> {
  const digest = new Uint8Array(await provider.digest("SHA-256", ASCII_ENCODER.encode(text)));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function wireBytes(wire: Uint8Array | string): Uint8Array {
  if (typeof wire === "string") {
    return ASCII_ENCODER.encode(wire);
  }
  check(wire instanceof Uint8Array);
  return wire.slice();
}

function decodeWire(bytes: Uint8Array): string {
  try {
    // UTF-8 like Python's json.loads on bytes; a leading BOM is dropped.
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new MultimodalContractError(INVALID);
  }
}

const ADMISSION = Symbol("openmed.multimodal.admission");

/**
 * An admitted multimodal evidence packet. Create one with
 * {@link admitMultimodalPacket}; there is no unchecked constructor.
 *
 * Admission is structural: it proves the packet is canonical, within limits
 * and self-consistent. It does not authenticate who produced it, approve
 * disclosure or make any clinical decision.
 */
export class MultimodalPacket {
  /** Envelope schema id of the admitted packet. */
  readonly schemaVersion: string;
  /** Lowercase SHA-256 of the canonical envelope without its digest field. */
  readonly evidenceDigest: string;
  /** Typed, frozen projection. */
  readonly projection: MultimodalProjection;
  readonly #wire: Uint8Array;

  private constructor(
    token: symbol,
    wire: Uint8Array,
    schemaVersion: string,
    evidenceDigest: string,
    projection: MultimodalProjection,
  ) {
    if (token !== ADMISSION) {
      throw new MultimodalContractError(INVALID);
    }
    this.#wire = wire;
    this.schemaVersion = schemaVersion;
    this.evidenceDigest = evidenceDigest;
    this.projection = projection;
    Object.freeze(this);
  }

  /** Projection kind of the admitted packet. */
  get kind(): MultimodalProjectionKind {
    return this.projection.kind;
  }

  /** A fresh copy of the exact admitted canonical bytes. */
  toBytes(): Uint8Array {
    return this.#wire.slice();
  }

  /** The exact admitted canonical JSON text (always ASCII). */
  toText(): string {
    return new TextDecoder().decode(this.#wire);
  }

  /** Admit a wire packet; see {@link admitMultimodalPacket}. */
  static async admit(
    wire: Uint8Array | string,
    options?: AdmitMultimodalPacketOptions,
  ): Promise<MultimodalPacket> {
    const provider = digestProvider(options);
    const bytes = wireBytes(wire);
    check(bytes.length > 0 && bytes.length <= MULTIMODAL_MAX_WIRE_BYTES);
    let root: JsonValue;
    try {
      root = parseStrictJson(decodeWire(bytes));
    } catch (error) {
      if (error instanceof MultimodalContractError) throw error;
      throw new MultimodalContractError(INVALID);
    }
    const envelope = keys(root, "schema_version kind payload evidence_digest");
    const schemaVersion = field(envelope, "schema_version");
    check(typeof schemaVersion === "string" && MULTIMODAL_PACKET_SCHEMAS.includes(schemaVersion), SCHEMA);
    const kind = field(envelope, "kind");
    check(
      typeof kind === "string" &&
        (MULTIMODAL_PROJECTION_KINDS as readonly string[]).includes(kind),
      CAPABILITY,
    );
    const evidenceDigest = hex(field(envelope, "evidence_digest"));
    const payload = field(envelope, "payload");
    check(isJsonObject(payload));
    let validated: Validated;
    let identity: string;
    let canonical: string;
    try {
      validated = VALIDATORS[kind as MultimodalProjectionKind](payload);
      identity = canonicalJson(
        new Map([...envelope.entries()].filter(([name]) => name !== "evidence_digest")),
      );
      canonical = canonicalJson(envelope);
    } catch (error) {
      // JsonSyntaxError (a non-integer number reaching the canonical form) and
      // any unexpected failure are invalid contracts, never echoed values.
      if (error instanceof MultimodalContractError) throw error;
      throw new MultimodalContractError(INVALID);
    }
    if (validated.pendingDigest !== undefined) {
      const actual = await sha256Hex(provider, validated.pendingDigest.canonical);
      check(actual === validated.pendingDigest.expected, PROVENANCE);
    }
    check((await sha256Hex(provider, identity)) === evidenceDigest, PROVENANCE);
    const expected = ASCII_ENCODER.encode(canonical);
    check(
      expected.length === bytes.length && expected.every((byte, index) => byte === bytes[index]),
    );
    return new MultimodalPacket(
      ADMISSION,
      bytes,
      schemaVersion,
      evidenceDigest,
      validated.projection,
    );
  }
}

/**
 * Admit a canonical multimodal evidence packet.
 *
 * The packet must be the exact canonical ASCII JSON that Python's projection
 * factories and OpenMedKit write: compact, keys sorted, ASCII-escaped, at most
 * 64 KiB, with every floating field as sixteen hex digits of a big-endian
 * IEEE-754 double and every integer within the contract's bounds. The
 * evidence digest, and a clock projection's source result digest, are checked
 * with WebCrypto SHA-256.
 *
 * @param wire Packet bytes, or its text (encoded as UTF-8).
 * @param options Optional digest provider for hosts without WebCrypto.
 * @returns The admitted packet with a typed, frozen projection.
 * @throws MultimodalContractError with one of {@link MULTIMODAL_REFUSALS}.
 * @throws MultimodalDigestUnavailableError when no SHA-256 provider exists.
 */
export function admitMultimodalPacket(
  wire: Uint8Array | string,
  options?: AdmitMultimodalPacketOptions,
): Promise<MultimodalPacket> {
  return MultimodalPacket.admit(wire, options);
}
