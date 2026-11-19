import { assertOfflineAssetPath } from "../runtime/ort-web-loader";
import type { TokenClassificationEntity } from "../types";

/** Version of the host/worker message protocol; both sides must agree. */
export const INFERENCE_WORKER_PROTOCOL_VERSION = 1 as const;

/** Default number of requests that may wait on the host before admission fails. */
export const DEFAULT_INFERENCE_WORKER_MAX_QUEUE_SIZE = 16;

/** Default number of requests the host keeps outstanding in the worker. */
export const DEFAULT_INFERENCE_WORKER_MAX_IN_FLIGHT = 1;

/** Default per-request input bound, in UTF-16 code units. */
export const DEFAULT_INFERENCE_WORKER_MAX_INPUT_CHARS = 1_000_000;

/**
 * Value-free failure codes shared by the worker client and server.
 *
 * Errors carry one of these codes and a fixed message. They never include
 * input text, model output, file paths, or the message of an underlying error.
 */
export const INFERENCE_WORKER_ERROR_CODES = [
  "alignment_failed",
  "cancelled",
  "device_lost",
  "disposed",
  "inference_failed",
  "input_too_large",
  "invalid_request",
  "load_failed",
  "not_loaded",
  "plan_incompatible",
  "plan_invalidated",
  "protocol_error",
  "queue_full",
  "stale_generation",
  "superseded",
  "worker_failed",
] as const;

export type InferenceWorkerErrorCode =
  (typeof INFERENCE_WORKER_ERROR_CODES)[number];

const ERROR_MESSAGES: Readonly<Record<InferenceWorkerErrorCode, string>> = {
  alignment_failed:
    "Inference worker could not align token offsets; provide source offsets.",
  cancelled: "Inference worker request was cancelled.",
  device_lost: "Inference worker lost its GPU device; recover before reuse.",
  disposed: "Inference worker client was disposed.",
  inference_failed: "Inference worker could not complete the request.",
  input_too_large: "Inference worker input exceeds the configured size limit.",
  invalid_request: "Inference worker request is invalid.",
  load_failed: "Inference worker could not load the model.",
  not_loaded: "Inference worker has no verified execution plan.",
  plan_incompatible: "Inference worker execution plan failed verification.",
  plan_invalidated:
    "Inference worker execution plan was invalidated; recover before reuse.",
  protocol_error: "Inference worker sent a malformed message.",
  queue_full: "Inference worker queue is full.",
  stale_generation:
    "Inference worker request targets a model generation that is no longer loaded.",
  superseded: "Inference worker request was superseded by a newer model load.",
  worker_failed: "Inference worker failed; recover before reuse.",
};

const ERROR_CODE_SET: ReadonlySet<string> = new Set(INFERENCE_WORKER_ERROR_CODES);

/**
 * Typed, value-free failure raised by the inference worker client or server.
 *
 * The message is fixed per `code`, so it is safe to log. No `cause` is attached.
 */
export class InferenceWorkerError extends Error {
  /** Stable machine-readable reason. */
  readonly code: InferenceWorkerErrorCode;

  /**
   * Creates an error for a controlled failure code.
   *
   * Args:
   *   code: The value-free reason for the failure.
   */
  constructor(code: InferenceWorkerErrorCode) {
    super(ERROR_MESSAGES[code]);
    this.name = "InferenceWorkerError";
    this.code = code;
  }
}

/** Returns whether `value` is one of the controlled worker error codes. */
export function isInferenceWorkerErrorCode(
  value: unknown,
): value is InferenceWorkerErrorCode {
  return typeof value === "string" && ERROR_CODE_SET.has(value);
}

/** Why the worker is being asked to load a runtime. */
export type InferenceWorkerLoadReason = "load" | "recover";

/** Structured-clone-safe description of the model the worker should load. */
export interface InferenceWorkerLoadSpec {
  /** Model id or local path passed to the worker's `load` handler. */
  modelRef: string;
  /** Require a local-only runtime (no remote model fetches). Defaults to true. */
  localOnly?: boolean;
  /** Backends the host accepts, for example `["webgpu", "wasm-basic"]`. */
  allowedBackends?: readonly string[];
  /** Expected model digest; the reported plan must match when set. */
  modelDigest?: string;
  /** Handler-specific plain data (strings, numbers, booleans, arrays, objects). */
  options?: Readonly<Record<string, unknown>>;
}

/** Normalised load spec as it crosses the worker boundary. */
export interface NormalizedInferenceWorkerLoadSpec {
  modelRef: string;
  localOnly: boolean;
  allowedBackends: string[] | null;
  modelDigest: string | null;
  options: Record<string, unknown>;
}

/** Verified execution plan reported by the worker for one load. */
export interface InferenceWorkerPlan {
  protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
  /** Model generation: changes on every `load()`, kept across `recover()`. */
  generation: number;
  /** Plan epoch: changes on every load and recovery. */
  epoch: number;
  modelRef: string;
  backend: string;
  localOnly: boolean;
  modelDigest: string | null;
}

/** Columnar, transferable encoding of aligned token-classification output. */
export interface EncodedInferenceTokens {
  count: number;
  labels: string[];
  entity: Int32Array;
  entityGroup: Int32Array;
  start: Float64Array;
  end: Float64Array;
  score: Float64Array;
  index: Float64Array;
  flags: Uint8Array;
}

export type HostToWorkerMessage =
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "load";
      id: number;
      generation: number;
      epoch: number;
      reason: InferenceWorkerLoadReason;
      spec: NormalizedInferenceWorkerLoadSpec;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "infer";
      id: number;
      epoch: number;
      input: ArrayBuffer;
      callOptions: Record<string, unknown>;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "cancel";
      id: number;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "dispose";
    };

export type WorkerToHostMessage =
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "ready";
      id: number;
      epoch: number;
      plan: InferenceWorkerPlan;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "result";
      id: number;
      epoch: number;
      tokens: EncodedInferenceTokens;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "error";
      id: number;
      epoch: number;
      code: InferenceWorkerErrorCode;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "cancelled";
      id: number;
      epoch: number;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "device_lost";
      epoch: number;
    }
  | {
      protocol: typeof INFERENCE_WORKER_PROTOCOL_VERSION;
      type: "disposed";
    };

const FLAG_SCORE = 1;
const FLAG_INDEX = 2;
const MAX_PLAIN_DATA_DEPTH = 32;
const DECODE_CHUNK = 8192;

/** Copies text into a fresh UTF-16 code-unit buffer so it can be transferred. */
export function encodeTextUnits(text: string): ArrayBuffer {
  const units = new Uint16Array(text.length);
  for (let offset = 0; offset < text.length; offset += 1) {
    units[offset] = text.charCodeAt(offset);
  }
  return units.buffer;
}

/** Rebuilds the exact source string, lone surrogates included. */
export function decodeTextUnits(buffer: ArrayBuffer): string {
  const units = new Uint16Array(buffer);
  let text = "";
  for (let offset = 0; offset < units.length; offset += DECODE_CHUNK) {
    text += String.fromCharCode(...units.subarray(offset, offset + DECODE_CHUNK));
  }
  return text;
}

/**
 * Packs aligned tokens into typed-array columns; token surface text is dropped.
 *
 * Only labels, scores, token indices, and character offsets cross the boundary.
 */
export function encodeInferenceTokens(tokens: readonly TokenClassificationEntity[]): {
  tokens: EncodedInferenceTokens;
  transfer: ArrayBuffer[];
} {
  const count = tokens.length;
  const labels: string[] = [];
  const labelIds = new Map<string, number>();
  const labelId = (label: unknown): number => {
    if (typeof label !== "string") {
      return -1;
    }
    let id = labelIds.get(label);
    if (id === undefined) {
      id = labels.length;
      labels.push(label);
      labelIds.set(label, id);
    }
    return id;
  };
  const entity = new Int32Array(count);
  const entityGroup = new Int32Array(count);
  const start = new Float64Array(count);
  const end = new Float64Array(count);
  const score = new Float64Array(count);
  const index = new Float64Array(count);
  const flags = new Uint8Array(count);
  tokens.forEach((token, position) => {
    entity[position] = labelId(token.entity);
    entityGroup[position] = labelId(token.entity_group);
    start[position] = Number(token.start);
    end[position] = Number(token.end);
    let tokenFlags = 0;
    if (typeof token.score === "number") {
      score[position] = token.score;
      tokenFlags |= FLAG_SCORE;
    }
    if (typeof token.index === "number") {
      index[position] = token.index;
      tokenFlags |= FLAG_INDEX;
    }
    flags[position] = tokenFlags;
  });
  return {
    tokens: { count, labels, entity, entityGroup, start, end, score, index, flags },
    transfer: [
      entity.buffer,
      entityGroup.buffer,
      start.buffer,
      end.buffer,
      score.buffer,
      index.buffer,
      flags.buffer,
    ],
  };
}

/** Rebuilds token entities from columns, or returns null for malformed data. */
export function decodeInferenceTokens(
  value: unknown,
): TokenClassificationEntity[] | null {
  if (!isRecord(value)) {
    return null;
  }
  const { count, labels } = value;
  if (!isCount(count) || !Array.isArray(labels)) {
    return null;
  }
  if (!labels.every((label) => typeof label === "string")) {
    return null;
  }
  const entity = value.entity;
  const entityGroup = value.entityGroup;
  const start = value.start;
  const end = value.end;
  const score = value.score;
  const index = value.index;
  const flags = value.flags;
  if (
    !(entity instanceof Int32Array) ||
    !(entityGroup instanceof Int32Array) ||
    !(start instanceof Float64Array) ||
    !(end instanceof Float64Array) ||
    !(score instanceof Float64Array) ||
    !(index instanceof Float64Array) ||
    !(flags instanceof Uint8Array) ||
    [entity, entityGroup, start, end, score, index, flags].some(
      (column) => column.length !== count,
    )
  ) {
    return null;
  }
  const labelAt = (id: number): string | undefined | null => {
    if (id === -1) {
      return undefined;
    }
    const label = labels[id] as string | undefined;
    return label === undefined ? null : label;
  };
  const tokens: TokenClassificationEntity[] = [];
  for (let position = 0; position < count; position += 1) {
    const entityLabel = labelAt(entity[position] as number);
    const groupLabel = labelAt(entityGroup[position] as number);
    if (entityLabel === null || groupLabel === null) {
      return null;
    }
    const token: TokenClassificationEntity = {
      start: start[position] as number,
      end: end[position] as number,
    };
    if (entityLabel !== undefined) {
      token.entity = entityLabel;
    }
    if (groupLabel !== undefined) {
      token.entity_group = groupLabel;
    }
    const tokenFlags = flags[position] as number;
    if ((tokenFlags & FLAG_SCORE) !== 0) {
      token.score = score[position] as number;
    }
    if ((tokenFlags & FLAG_INDEX) !== 0) {
      token.index = index[position] as number;
    }
    tokens.push(token);
  }
  return tokens;
}

/** Returns whether a value is a plain, structured-clone-safe data tree. */
export function isPlainData(value: unknown): boolean {
  return isPlainDataAt(value, new Set<object>(), 0);
}

function isPlainDataAt(value: unknown, seen: Set<object>, depth: number): boolean {
  if (value === null) {
    return true;
  }
  switch (typeof value) {
    case "string":
    case "number":
    case "boolean":
    case "bigint":
    case "undefined":
      return true;
    case "object":
      break;
    default:
      return false;
  }
  const node = value as object;
  if (depth >= MAX_PLAIN_DATA_DEPTH || seen.has(node)) {
    return false;
  }
  seen.add(node);
  let plain: boolean;
  if (Array.isArray(node)) {
    plain = node.every((entry) => isPlainDataAt(entry, seen, depth + 1));
  } else {
    const prototype = Object.getPrototypeOf(node) as unknown;
    plain =
      (prototype === Object.prototype || prototype === null) &&
      Object.values(node).every((entry) => isPlainDataAt(entry, seen, depth + 1));
  }
  seen.delete(node);
  return plain;
}

/** Returns whether a model reference avoids remote URL schemes. */
export function isOfflineModelRef(modelRef: string): boolean {
  try {
    assertOfflineAssetPath(modelRef, "model reference");
    return true;
  } catch {
    return false;
  }
}

/** Returns whether a value is a non-null, non-array object. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Returns whether a value is a positive safe integer message id or epoch. */
export function isPositiveId(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isCount(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

/** Validates a positive integer bound, falling back to a default. */
export function resolveBound(
  value: number | undefined,
  fallback: number,
): number {
  if (value === undefined) {
    return fallback;
  }
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError("Inference worker bounds must be positive integers.");
  }
  return value;
}
