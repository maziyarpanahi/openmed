/** Strict consumers of the existing workflow HTTP and agent evidence contracts. */
import type {
  FetchLike, GovernedWorkflowReference, GovernedWorkflowMutation,
  GovernedWorkflowReview,
} from "./index.js";

export type WorkflowPhase = "queued" | "preflight" | "ready" | "running" |
  "waiting-review" | "completed" | "aborted";
export interface WorkflowEffect {
  readonly ordinal: number;
  readonly action_id: string;
  readonly tool_id: string;
  readonly kind: "local_tool" | "fhir_write" | "omop_batch";
  readonly operation_digest: string;
  readonly idempotency_key: string;
  readonly approval_required: boolean;
  readonly compensation_limit: "none" | "propose_only";
  readonly state: "pending" | "committed";
  readonly commit_evidence_digest: string | null;
}
export interface WorkflowOutcome {
  readonly schema_version: "openmed.agent.outcome.v1";
  readonly outcome_class: "success" | "abstained" | "review_required" | "policy_denied" | "failed";
  readonly reason_code: string;
}
export interface WorkflowSnapshot {
  readonly schema_version: "openmed.service.workflow_response.v1";
  readonly run_id: string;
  readonly workflow_id: string;
  readonly action_digest: string;
  readonly state_digest: string;
  readonly phase: WorkflowPhase;
  readonly effects: readonly WorkflowEffect[];
  readonly preview_digest: string;
  readonly proposed_effect_count: number;
  readonly committed_effect_count: number;
  readonly outcome: WorkflowOutcome | null;
  readonly receipt_digest: string | null;
  readonly cancellation_requested: boolean;
}
export interface WorkflowReadOptions {
  maxAttempts?: number;
  retryDelayMs?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}
export interface WorkflowMutationOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}
export interface WorkflowPollOptions {
  maxRequests?: number;
  timeoutMs?: number;
  intervalMs?: number;
  signal?: AbortSignal;
  /** Monotonic milliseconds; injectable for offline tests. */
  clock?: () => number;
}
export type WorkflowMutationOutcome = "not_applicable" | "not_attempted" | "refused" | "unknown";

const MAX_RESPONSE_BYTES = 262144;
const MAX_REQUEST_BYTES = 65536;
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const REQUEST_ID = /^req_[0-9a-f]{32}$/;
const SAFE_CORRELATION = /^(?:req_[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/;
const RESPONSE_FIELDS = ["schema_version", "run_id", "workflow_id", "action_digest",
  "state_digest", "phase", "effects", "preview_digest", "proposed_effect_count",
  "committed_effect_count", "outcome", "receipt_digest", "cancellation_requested"];
const REFERENCE_FIELDS = ["schema_version", "run_id", "workflow_id", "action_digest", "expected_state_digest", "request_id"];
const EFFECT_FIELDS = ["ordinal", "action_id", "tool_id", "kind", "operation_digest",
  "idempotency_key", "approval_required", "compensation_limit", "state", "commit_evidence_digest"];
const ERROR_STATUSES: Readonly<Record<string, number>> = Object.freeze({
  workflow_authentication_required: 401, workflow_forbidden: 403, workflow_disabled: 503,
  workflow_invalid_input: 422, workflow_request_too_large: 413, workflow_conflict: 409,
  workflow_receipt_unverified: 403, workflow_receipt_expired: 409, workflow_receipt_future: 409,
  workflow_terminal: 409, workflow_unavailable: 503, workflow_service_failed: 503,
  workflow_invalid_result: 502, workflow_mutation_unknown: 503,
});
const CLIENT_CODES = new Set(["workflow_invalid_request", "workflow_unsupported_version",
  "workflow_malformed_response", "workflow_response_too_large", "workflow_binding_mismatch",
  "workflow_transport_failed", "workflow_server_refused", "workflow_poll_cancelled",
  "workflow_poll_timeout", "workflow_clock_failed"]);
const RETRY_CODES = new Set(["workflow_unavailable", "workflow_service_failed", "workflow_transport_failed"]);
const UNCERTAIN_CODES = new Set(["workflow_unavailable", "workflow_service_failed", "workflow_invalid_result", "workflow_mutation_unknown"]);
const READ_PATHS = new Set(["/v1/workflows/preflight", "/v1/workflows/preview", "/v1/workflows/status"]);
const MUTATION_PATHS = new Set(["/v1/workflows/review-receipts", "/v1/workflows/cancel"]);
const VALIDATED_SNAPSHOTS = new WeakSet<object>();

/** Fixed diagnostics; raw envelopes, credentials and protected content are discarded. */
export class WorkflowClientError extends Error {
  readonly code: string;
  readonly status: number | undefined;
  readonly mutationOutcome: WorkflowMutationOutcome;
  readonly requestId: string | undefined;
  readonly lastView: WorkflowSnapshot | undefined;
  constructor(code: string, options: {
    status?: number; mutationOutcome?: WorkflowMutationOutcome;
    requestId?: string | null; lastView?: WorkflowSnapshot;
  } = {}) {
    const safeCode = typeof code === "string" &&
      (CLIENT_CODES.has(code) || Object.hasOwnProperty.call(ERROR_STATUSES, code))
      ? code : "workflow_server_refused";
    super(`${safeCode}: governed workflow request did not complete.`);
    this.name = "WorkflowClientError";
    this.code = safeCode;
    this.status = typeof options.status === "number" && Number.isSafeInteger(options.status) &&
      options.status >= 100 && options.status <= 599 ? options.status : undefined;
    const outcome = options.mutationOutcome ?? "not_applicable";
    this.mutationOutcome = ["not_applicable", "not_attempted", "refused", "unknown"].includes(outcome)
      ? outcome : "unknown";
    this.requestId = typeof options.requestId === "string" && text(options.requestId, SAFE_CORRELATION)
      ? options.requestId : undefined;
    this.lastView = options.lastView !== undefined && VALIDATED_SNAPSHOTS.has(options.lastView)
      ? options.lastView : undefined;
  }
}
function fail(code = "workflow_malformed_response"): never { throw new WorkflowClientError(code); }
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function exact(value: unknown, fields: readonly string[]): asserts value is Record<string, unknown> {
  if (!record(value) || Object.keys(value).length !== fields.length ||
      fields.some(key => !Object.prototype.hasOwnProperty.call(value, key))) fail();
}
function text(value: unknown, regex: RegExp): value is string { return typeof value === "string" && regex.exec(value)?.[0] === value; }
function identifier(value: unknown, kind: string, role = false): boolean {
  if (typeof value !== "string" || (!role && value.length > 512)) return false;
  const match = /^(workflow|tool|role):([^/]+)\/([a-z][a-z0-9-]{0,63})(?:@((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)))?$/.exec(value);
  if (!match || match[0] !== value || match[1] !== kind || (!role && match[2].length > 253)) return false;
  const labels = match[2].split(".");
  return labels.length >= 2 && labels.every(label => text(label, /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/));
}
function finite(value: unknown, maximum: number, zero = false): value is number {
  return typeof value === "number" && Number.isFinite(value) && (zero ? value >= 0 : value > 0) && value <= maximum;
}
function integer(value: unknown, maximum: number): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 && value <= maximum;
}

// Lexical JSON keeps integer tokens distinct from 1.0/1e0 and rejects duplicate
// keys before information is lost. No eval, normalization or prototype writes.
class Fraction { constructor(readonly value: number) {} }
function parseJson(raw: Uint8Array): unknown {
  if (raw.byteLength > MAX_RESPONSE_BYTES) fail("workflow_response_too_large");
  let source: string;
  try { source = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(raw); }
  catch { return fail(); }
  let position = 0, nodes = 0;
  const space = () => { while (/[\x20\t\r\n]/.test(source[position] ?? "x")) position++; };
  const string = (): string => {
    const start = position++;
    while (position < source.length) {
      const char = source[position++];
      if (char === '"') {
        try { return JSON.parse(source.slice(start, position)) as string; } catch { return fail(); }
      }
      if (char === "\\") position++;
    }
    return fail();
  };
  const value = (depth: number): unknown => {
    if (++nodes > 4096 || depth > 8) fail();
    space();
    if (source[position] === '"') return string();
    if (source[position] === "{") {
      position++; space();
      const result: Record<string, unknown> = Object.create(null);
      if (source[position] === "}") { position++; return result; }
      for (;;) {
        if (source[position] !== '"') fail();
        const key = string(); space();
        if (Object.prototype.hasOwnProperty.call(result, key) || source[position++] !== ":") fail();
        result[key] = value(depth + 1); space();
        const end = source[position++];
        if (end === "}") return result;
        if (end !== ",") fail();
        space();
      }
    }
    if (source[position] === "[") {
      position++; space();
      const result: unknown[] = [];
      if (source[position] === "]") { position++; return result; }
      for (;;) {
        result.push(value(depth + 1)); space();
        const end = source[position++];
        if (end === "]") return result;
        if (end !== ",") fail();
      }
    }
    for (const [token, literal] of [["null", null], ["true", true], ["false", false]] as const) {
      if (source.startsWith(token, position)) { position += token.length; return literal; }
    }
    const match = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/.exec(source.slice(position));
    if (!match) fail();
    position += match[0].length;
    if (/[.eE]/.test(match[0])) {
      const number = Number(match[0]); if (!Number.isFinite(number)) fail();
      return new Fraction(number);
    }
    const number = BigInt(match[0]);
    return number >= BigInt(Number.MIN_SAFE_INTEGER) && number <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(number) : number;
  };
  const result = value(0); space();
  if (position !== source.length) fail();
  return result;
}
function asciiString(value: string): string {
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g,
    char => "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0"));
}
function canonical(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return asciiString(value);
  if (typeof value === "boolean" || typeof value === "bigint") return String(value);
  if (typeof value === "number" && Number.isSafeInteger(value)) return String(value);
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (record(value)) return "{" + Object.keys(value).sort().map(key => asciiString(key) + ":" + canonical(value[key])).join(",") + "}";
  return fail("workflow_invalid_request");
}
async function hash(value: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

/** Parse metadata and recompute native preview/run-bound effect commitments. */
export async function parseWorkflowResponse(raw: Uint8Array, reference: GovernedWorkflowReference): Promise<WorkflowSnapshot> {
  try { return await parseWorkflowSnapshot(raw, reference); }
  catch (error) {
    if (error instanceof WorkflowClientError) throw error;
    return fail();
  }
}
async function parseWorkflowSnapshot(raw: Uint8Array, reference: GovernedWorkflowReference): Promise<WorkflowSnapshot> {
  const v = parseJson(raw); exact(v, RESPONSE_FIELDS);
  if (v.schema_version !== "openmed.service.workflow_response.v1") fail("workflow_unsupported_version");
  if (!text(v.run_id, /^run_[0-9a-f]{32}$/) || !identifier(v.workflow_id, "workflow") || !text(v.action_digest, DIGEST)) fail();
  if (v.run_id !== reference.run_id || v.workflow_id !== reference.workflow_id || v.action_digest !== reference.action_digest) fail("workflow_binding_mismatch");
  if (!text(v.state_digest, DIGEST) || !text(v.preview_digest, DIGEST) ||
      (v.receipt_digest !== null && !text(v.receipt_digest, DIGEST)) ||
      typeof v.cancellation_requested !== "boolean" ||
      !["queued", "preflight", "ready", "running", "waiting-review", "completed", "aborted"].includes(v.phase as string) ||
      !Array.isArray(v.effects) || v.effects.length > 128 || !integer(v.proposed_effect_count, 128) || !integer(v.committed_effect_count, 128)) fail();
  const actions = new Set<string>(), keys = new Set<string>();
  let committed = 0;
  for (const [ordinal, effect] of v.effects.entries()) {
    exact(effect, EFFECT_FIELDS);
    if (!integer(effect.ordinal, 127) || effect.ordinal !== ordinal || !text(effect.action_id, /^act_[0-9a-f]{32}$/) ||
        !identifier(effect.tool_id, "tool") || !["local_tool", "fhir_write", "omop_batch"].includes(effect.kind as string) ||
        !text(effect.operation_digest, DIGEST) || !text(effect.idempotency_key, /^idem_[0-9a-f]{64}$/) ||
        typeof effect.approval_required !== "boolean" || !["none", "propose_only"].includes(effect.compensation_limit as string) ||
        !["pending", "committed"].includes(effect.state as string) ||
        (effect.state === "pending" ? effect.commit_evidence_digest !== null : !text(effect.commit_evidence_digest, DIGEST))) fail();
    if (actions.has(effect.action_id) || keys.has(effect.idempotency_key)) fail();
    const expected = "idem_" + await hash(canonical({ action_id: effect.action_id, kind: effect.kind,
      operation_digest: effect.operation_digest, run_id: v.run_id, tool_id: effect.tool_id }));
    if (effect.idempotency_key !== expected) fail();
    actions.add(effect.action_id); keys.add(effect.idempotency_key);
    if (effect.state === "committed") committed++;
    Object.freeze(effect);
  }
  if (v.proposed_effect_count !== v.effects.length || v.committed_effect_count !== committed ||
      (v.phase === "completed" && committed !== v.effects.length) ||
      v.preview_digest !== "sha256:" + await hash("openmed.service.workflow_preview.v1\0" + canonical(v.effects))) fail();
  if (v.outcome !== null) {
    exact(v.outcome, ["schema_version", "outcome_class", "reason_code"]);
    const reasons: Record<string, readonly string[]> = {
      success: ["completed"], abstained: ["insufficient_evidence", "out_of_scope", "low_confidence"],
      review_required: ["conflicting_evidence", "safety_review", "human_gate"],
      policy_denied: ["consent_required", "purpose_mismatch", "phi_policy"], failed: ["tool_error", "timeout", "invalid_input"],
    };
    if (v.outcome.schema_version !== "openmed.agent.outcome.v1" ||
        typeof v.outcome.outcome_class !== "string" || !Object.hasOwnProperty.call(reasons, v.outcome.outcome_class) ||
        !reasons[v.outcome.outcome_class].includes(v.outcome.reason_code as string)) fail();
    Object.freeze(v.outcome);
  }
  Object.freeze(v.effects);
  const snapshot = Object.freeze(v) as unknown as WorkflowSnapshot;
  VALIDATED_SNAPSHOTS.add(snapshot);
  return snapshot;
}

function payload(request: GovernedWorkflowReference, mutation: boolean, receipt: boolean): string {
  try {
    exact(request, receipt ? [...REFERENCE_FIELDS, "receipt"] : REFERENCE_FIELDS);
    if (request.schema_version !== "openmed.service.workflow_request.v1" || !text(request.run_id, /^run_[0-9a-f]{32}$/) ||
        !identifier(request.workflow_id, "workflow") || !text(request.action_digest, DIGEST) ||
        (request.expected_state_digest !== null && !text(request.expected_state_digest, DIGEST)) ||
        (request.request_id !== null && !text(request.request_id, REQUEST_ID)) ||
        (mutation && (request.expected_state_digest === null || request.request_id === null))) fail();
    if (receipt) {
      const r = (request as unknown as GovernedWorkflowReview).receipt;
      exact(r, ["schema_version", "action_digest", "token_digest", "code"]);
      if (r.schema_version !== "openmed.agent.approval_receipt.v2" || r.action_digest !== request.action_digest ||
          !text(r.token_digest, DIGEST) || r.code !== "approved") fail();
    }
    const body = canonical(request);
    if (new TextEncoder().encode(body).byteLength > MAX_REQUEST_BYTES) fail();
    return body;
  } catch {
    throw new WorkflowClientError("workflow_invalid_request", { mutationOutcome: mutation ? "not_attempted" : "not_applicable" });
  }
}
function validSignal(value: unknown): value is AbortSignal | undefined {
  return value === undefined || (typeof AbortSignal !== "undefined" && value instanceof AbortSignal);
}
async function delay(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) fail("workflow_poll_cancelled");
  await new Promise<void>((resolve, reject) => {
    const finish = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); resolve(); };
    const abort = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); reject(new WorkflowClientError("workflow_poll_cancelled")); };
    const timer = setTimeout(finish, ms);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
  });
}

/** Inspection retries are explicit; receipt and cancellation writes have one attempt. */
export class GovernedWorkflowClient {
  constructor(private readonly baseUrl: string, private readonly fetchImpl: FetchLike) {}

  async read(path: string, request: GovernedWorkflowReference, options: WorkflowReadOptions = {}): Promise<WorkflowSnapshot> {
    if (!READ_PATHS.has(path) || (options === null || typeof options !== "object")) fail("workflow_invalid_request");
    const attempts = options.maxAttempts ?? 1, retryDelay = options.retryDelayMs ?? 0;
    const timeout = options.timeoutMs ?? 30000;
    if (!integer(attempts, 3) || attempts === 0 || !finite(retryDelay, 5000, true) || !finite(timeout, 30000) || !validSignal(options.signal)) fail("workflow_invalid_request");
    const body = payload(request, false, false);
    const binding = Object.freeze({ ...request });
    for (let attempt = 0; attempt < attempts; attempt++) {
      try { return await this.send(path, body, binding, timeout, options.signal, false); }
      catch (error) {
        if (!(error instanceof WorkflowClientError) || attempt + 1 === attempts || !RETRY_CODES.has(error.code)) throw error;
        await delay(retryDelay, options.signal);
      }
    }
    return fail("workflow_transport_failed");
  }
  async mutate(path: string, request: GovernedWorkflowMutation | GovernedWorkflowReview, options: WorkflowMutationOptions = {}): Promise<WorkflowSnapshot> {
    if (!MUTATION_PATHS.has(path) || (options === null || typeof options !== "object"))
      throw new WorkflowClientError("workflow_invalid_request", { mutationOutcome: "not_attempted" });
    const timeout = options.timeoutMs ?? 30000;
    if (!finite(timeout, 30000) || !validSignal(options.signal))
      throw new WorkflowClientError("workflow_invalid_request", { mutationOutcome: "not_attempted" });
    const receipt = path.endsWith("review-receipts");
    const body = payload(request, true, receipt);
    const binding = Object.freeze({ ...request });
    let expectedReceiptDigest: string | undefined;
    if (receipt) {
      try { expectedReceiptDigest = "sha256:" + await hash(canonical(JSON.parse(body).receipt)); }
      catch { throw new WorkflowClientError("workflow_invalid_request", { mutationOutcome: "not_attempted" }); }
    }
    return this.send(path, body, binding, timeout, options.signal, true, expectedReceiptDigest);
  }
  async poll(request: GovernedWorkflowReference, options: WorkflowPollOptions = {}): Promise<WorkflowSnapshot> {
    if ((options === null || typeof options !== "object")) fail("workflow_invalid_request");
    const maximum = options.maxRequests ?? 20, timeout = options.timeoutMs ?? 60000, interval = options.intervalMs ?? 1000;
    if (!integer(maximum, 100) || maximum === 0 || !finite(timeout, 300000) || !finite(interval, 30000, true) || !validSignal(options.signal)) fail("workflow_invalid_request");
    payload(request, false, false);
    request = Object.freeze({ ...request });
    const clock = options.clock ?? (() => performance.now());
    const now = (): number => { try { const n = clock(); if (typeof n !== "number" || !Number.isFinite(n)) fail(); return n; } catch { return fail("workflow_clock_failed"); } };
    const start = now(); let previous = start, last: WorkflowSnapshot | undefined;
    const stopped = (code: string): never => { throw new WorkflowClientError(code, { lastView: last }); };
    for (let index = 0; index < maximum; index++) {
      if (options.signal?.aborted) stopped("workflow_poll_cancelled");
      const current = now();
      if (current < previous) stopped("workflow_clock_failed");
      const remaining = timeout - (current - start);
      if (remaining <= 0) stopped("workflow_poll_timeout");
      try {
        last = await this.read("/v1/workflows/status", request, { timeoutMs: Math.min(30000, remaining), signal: options.signal });
      } catch (error) {
        if (error instanceof WorkflowClientError)
          throw new WorkflowClientError(error.code, { status: error.status, requestId: error.requestId, lastView: last });
        return stopped("workflow_transport_failed");
      }
      const after = now();
      if (after < current) stopped("workflow_clock_failed");
      if (options.signal?.aborted) stopped("workflow_poll_cancelled");
      if (after - start >= timeout) stopped("workflow_poll_timeout");
      if (["waiting-review", "completed", "aborted"].includes(last.phase)) return last;
      if (index + 1 === maximum) break;
      previous = after;
      try { await delay(Math.min(interval, timeout - (after - start)), options.signal); }
      catch { return stopped("workflow_poll_cancelled"); }
    }
    return stopped("workflow_poll_timeout");
  }
  private async send(path: string, body: string, request: GovernedWorkflowReference, timeout: number, signal: AbortSignal | undefined, mutation: boolean, expectedReceiptDigest?: string): Promise<WorkflowSnapshot> {
    const before = mutation ? "not_attempted" : "not_applicable";
    const uncertain = mutation ? "unknown" : "not_applicable";
    if (signal?.aborted) throw new WorkflowClientError("workflow_poll_cancelled", { mutationOutcome: before });
    const controller = new AbortController();
    let status: number | undefined, correlation: string | null = null;
    let validatedRefusal = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let rejectBound: (reason: unknown) => void = () => {};
    const bound = new Promise<never>((_, reject) => { rejectBound = reject; });
    const stop = (code: string) => {
      controller.abort();
      rejectBound(new WorkflowClientError(code, { mutationOutcome: uncertain }));
    };
    const abort = () => stop("workflow_poll_cancelled");
    signal?.addEventListener("abort", abort, { once: true });
    timer = setTimeout(() => stop(mutation ? "workflow_mutation_unknown" : "workflow_transport_failed"), timeout);
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    try {
      const work = (async () => {
        if (signal?.aborted) fail("workflow_poll_cancelled");
        const response = await this.fetchImpl(this.baseUrl + path, {
          method: "POST", redirect: "manual", signal: controller.signal,
          headers: { "Accept": "application/json", "Content-Type": "application/json" }, body,
        });
        if (controller.signal.aborted) {
          if (response.body) void response.body.cancel().catch(() => {});
          fail("workflow_poll_cancelled");
        }
        status = response.status; correlation = response.headers.get("X-Request-ID");
        if (response.headers.get("Content-Type")?.split(";", 1)[0].trim().toLowerCase() !== "application/json") fail();
        if (!response.body) fail();
        reader = response.body.getReader();
        const chunks: Uint8Array[] = []; let length = 0;
        for (;;) {
          if (controller.signal.aborted) fail("workflow_poll_cancelled");
          const part = await reader.read(); if (part.done) break;
          length += part.value.byteLength;
          if (length > MAX_RESPONSE_BYTES) fail("workflow_response_too_large");
          chunks.push(part.value);
        }
        const raw = new Uint8Array(length); let offset = 0;
        for (const chunk of chunks) { raw.set(chunk, offset); offset += chunk.byteLength; }
        if (status < 200 || status >= 300) {
          const values = parseJson(raw); let code = "workflow_server_refused";
          if (record(values) && Object.keys(values).length === 1 && record(values.error)) {
            const error = values.error, fields = Object.keys(error);
            if (["code", "message", "details"].every(key => fields.includes(key)) &&
                fields.every(key => ["code", "message", "details", "request_id"].includes(key)) &&
                typeof error.code === "string" && Object.hasOwnProperty.call(ERROR_STATUSES, error.code) && ERROR_STATUSES[error.code] === status) code = error.code;
          }
          validatedRefusal = code !== "workflow_server_refused" && !UNCERTAIN_CODES.has(code);
          throw new WorkflowClientError(code);
        }
        const snapshot = await parseWorkflowResponse(raw, request);
        if (expectedReceiptDigest !== undefined && snapshot.receipt_digest !== expectedReceiptDigest)
          fail("workflow_binding_mismatch");
        if (path.endsWith("/cancel") && !snapshot.cancellation_requested && snapshot.phase !== "aborted")
          fail();
        return snapshot;
      })();
      return await Promise.race([work, bound]);
    } catch (error) {
      let ownCode: unknown;
      try {
        if (error !== null && typeof error === "object" &&
            Object.getPrototypeOf(error) === WorkflowClientError.prototype)
          ownCode = Object.getOwnPropertyDescriptor(error, "code")?.value;
      } catch { /* Foreign exception objects cannot supply diagnostic metadata. */ }
      const code = typeof ownCode === "string" ? ownCode : mutation ? "workflow_mutation_unknown" : "workflow_transport_failed";
      const outcome = mutation && validatedRefusal ? "refused" : uncertain;
      throw new WorkflowClientError(code, { status, mutationOutcome: outcome, requestId: correlation });
    } finally {
      clearTimeout(timer); signal?.removeEventListener("abort", abort);
      controller.abort();
      // Custom transports may ignore AbortSignal; do not await their cancellation.
      if (reader) void reader.cancel().catch(() => {});
    }
  }
}
