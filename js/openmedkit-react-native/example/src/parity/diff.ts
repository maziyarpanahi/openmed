// Field-by-field comparison of bridge results against direct native results.
//
// Diagnostics name JSON paths and carry digests, types and lengths only.
// They never carry the compared values, so a mismatch on a span replacement or
// a de-identified text cannot leak source text into evidence.

import { fingerprintDigest, scalarLength } from "./canonical";

export type ValueType =
  | "string"
  | "number"
  | "boolean"
  | "null"
  | "array"
  | "object"
  | "absent";

export interface ValueFingerprint {
  type: ValueType;
  /** `sha256:` digest of the canonical value; absent values have none. */
  digest: string | null;
  /** Unicode scalar length for strings, element count for arrays. */
  length?: number;
}

export type MismatchKind =
  | "value"
  | "type"
  | "length"
  | "missing_in_bridge"
  | "missing_in_direct"
  | "score_tolerance"
  | "order"
  | "outcome";

export interface FieldMismatch {
  path: string;
  kind: MismatchKind;
  bridge: ValueFingerprint;
  direct: ValueFingerprint;
}

export interface DiffPolicy {
  /** Absolute tolerance applied to numeric fields named `score`. */
  scoreAbsTolerance: number;
}

/** Fingerprint a value without retaining it. */
export function fingerprint(value: unknown): ValueFingerprint {
  const type = valueType(value);
  if (type === "absent") {
    return { type, digest: null };
  }
  const result: ValueFingerprint = { type, digest: fingerprintDigest(value) };
  if (typeof value === "string") {
    result.length = scalarLength(value);
  } else if (Array.isArray(value)) {
    result.length = value.length;
  }
  return result;
}

/** Compare two JSON-compatible values and list every differing field path. */
export function diffValues(
  bridge: unknown,
  direct: unknown,
  policy: DiffPolicy,
  path = "$",
): FieldMismatch[] {
  const mismatches: FieldMismatch[] = [];
  walk(bridge, direct, policy, path, mismatches);
  return mismatches;
}

/**
 * Compare span lists after checking the shared ordering contract.
 *
 * Bridge spans are already sorted by `(start, end)`. When the direct list is
 * not, an `order` mismatch is recorded and the fields are compared against the
 * direct list sorted the same way, so one ordering bug does not cascade into a
 * field mismatch on every span.
 */
export function diffSpanLists(
  bridgeSpans: unknown[],
  directSpans: unknown[],
  policy: DiffPolicy,
  path: string,
): FieldMismatch[] {
  const mismatches: FieldMismatch[] = [];
  const sortedDirect = [...directSpans].sort(compareSpanOffsets);
  if (sortedDirect.some((span, index) => span !== directSpans[index])) {
    mismatches.push({
      path,
      kind: "order",
      bridge: fingerprint(bridgeSpans.map(spanOffsets)),
      direct: fingerprint(directSpans.map(spanOffsets)),
    });
  }
  walk(bridgeSpans, sortedDirect, policy, path, mismatches);
  return mismatches;
}

function walk(
  bridge: unknown,
  direct: unknown,
  policy: DiffPolicy,
  path: string,
  mismatches: FieldMismatch[],
): void {
  const bridgeType = valueType(bridge);
  const directType = valueType(direct);

  if (bridgeType === "absent" || directType === "absent") {
    if (bridgeType !== directType) {
      mismatches.push({
        path,
        kind: bridgeType === "absent" ? "missing_in_bridge" : "missing_in_direct",
        bridge: fingerprint(bridge),
        direct: fingerprint(direct),
      });
    }
    return;
  }

  if (bridgeType !== directType) {
    mismatches.push({
      path,
      kind: "type",
      bridge: fingerprint(bridge),
      direct: fingerprint(direct),
    });
    return;
  }

  if (Array.isArray(bridge) && Array.isArray(direct)) {
    if (bridge.length !== direct.length) {
      mismatches.push({
        path,
        kind: "length",
        bridge: fingerprint(bridge),
        direct: fingerprint(direct),
      });
    }
    const longest = Math.max(bridge.length, direct.length);
    for (let index = 0; index < longest; index += 1) {
      walk(
        index < bridge.length ? bridge[index] : ABSENT,
        index < direct.length ? direct[index] : ABSENT,
        policy,
        `${path}[${index}]`,
        mismatches,
      );
    }
    return;
  }

  if (bridgeType === "object") {
    const bridgeRecord = bridge as Record<string, unknown>;
    const directRecord = direct as Record<string, unknown>;
    const keys = new Set([
      ...Object.keys(bridgeRecord).filter((key) => bridgeRecord[key] !== undefined),
      ...Object.keys(directRecord).filter((key) => directRecord[key] !== undefined),
    ]);
    for (const key of [...keys].sort()) {
      walk(
        has(bridgeRecord, key) ? bridgeRecord[key] : ABSENT,
        has(directRecord, key) ? directRecord[key] : ABSENT,
        policy,
        `${path}.${key}`,
        mismatches,
      );
    }
    return;
  }

  if (typeof bridge === "number" && typeof direct === "number") {
    if (isScorePath(path)) {
      if (Math.abs(bridge - direct) > policy.scoreAbsTolerance) {
        mismatches.push({
          path,
          kind: "score_tolerance",
          bridge: fingerprint(bridge),
          direct: fingerprint(direct),
        });
      }
      return;
    }
  }

  if (!Object.is(bridge, direct)) {
    mismatches.push({
      path,
      kind: "value",
      bridge: fingerprint(bridge),
      direct: fingerprint(direct),
    });
  }
}

const ABSENT: unique symbol = Symbol("absent");

function valueType(value: unknown): ValueType {
  if (value === ABSENT || value === undefined) {
    return "absent";
  }
  if (value === null) {
    return "null";
  }
  if (Array.isArray(value)) {
    return "array";
  }
  switch (typeof value) {
    case "string":
      return "string";
    case "number":
      return "number";
    case "boolean":
      return "boolean";
    default:
      return "object";
  }
}

function has(record: Record<string, unknown>, key: string): boolean {
  return (
    Object.prototype.hasOwnProperty.call(record, key) && record[key] !== undefined
  );
}

function isScorePath(path: string): boolean {
  return path.endsWith(".score");
}

function spanOffsets(span: unknown): [number, number] {
  const record =
    span !== null && typeof span === "object" ? (span as Record<string, unknown>) : {};
  return [
    typeof record.start === "number" ? record.start : -1,
    typeof record.end === "number" ? record.end : -1,
  ];
}

function compareSpanOffsets(left: unknown, right: unknown): number {
  const [leftStart, leftEnd] = spanOffsets(left);
  const [rightStart, rightEnd] = spanOffsets(right);
  return leftStart - rightStart || leftEnd - rightEnd;
}
