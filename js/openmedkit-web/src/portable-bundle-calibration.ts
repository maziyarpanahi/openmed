/**
 * Reader for the `org.openmed.decision_calibration` portable-bundle extension.
 *
 * Mirrors `openmed/core/portable_bundle_calibration.py`: the same checks run in
 * the same order and produce the same result for
 * `tests/fixtures/portable_bundle_calibration`. It never opens a network
 * connection, never runs a model, and reports only reason codes, identifiers,
 * and digests. Digests use WebCrypto, so every entry point is asynchronous.
 */

import {
  canonicalJson,
  parsePortableBundleManifest,
  portableBundleContentId,
  type PortableBundleVerification,
} from "./portable-bundle";

export const DECISION_CALIBRATION_EXTENSION = "org.openmed.decision_calibration";
export const DECISION_CALIBRATION_EXTENSION_VERSION = 1;
export const DECISION_CALIBRATION_PPM_SCALE = 1_000_000;

/** Stable, value-free reason codes shared by every calibration reader. */
export const DECISION_CALIBRATION_REASONS = [
  "calibration_bundle_unverified",
  "calibration_extension_missing",
  "calibration_malformed",
  "calibration_version_unsupported",
  "calibration_digest_mismatch",
  "calibration_model_mismatch",
  "calibration_runtime_mismatch",
  "calibration_scope_mismatch",
  "calibration_evidence_mismatch",
  "calibration_profile_missing",
  "calibration_runtime_unbound",
  "calibration_schema_incompatible",
  "calibration_pin_mismatch",
  "calibration_request_invalid",
] as const;

export type DecisionCalibrationReason = (typeof DECISION_CALIBRATION_REASONS)[number];

/** Error that carries only a controlled calibration reason code. */
export class DecisionCalibrationError extends Error {
  readonly reason: DecisionCalibrationReason;

  constructor(reason: DecisionCalibrationReason) {
    super(reason);
    this.name = "DecisionCalibrationError";
    this.reason = reason;
  }
}

/** Calibration method with its integer parameters (parts per million). */
export type DecisionCalibrationMethod =
  | { readonly kind: "deterministic_margin" }
  | { readonly kind: "temperature_scaling"; readonly temperaturePpm: number }
  | { readonly kind: "platt_scaling"; readonly slopePpm: number; readonly interceptPpm: number }
  | {
      readonly kind: "isotonic_regression";
      readonly points: readonly { readonly inputPpm: number; readonly outputPpm: number }[];
    };

/** One runtime variant a profile was calibrated on. */
export interface DecisionCalibrationRuntimeBinding {
  readonly runtimeId: string;
  readonly digest: string;
}

/** A validated calibration profile read from a signed bundle. */
export interface BundleCalibrationProfile {
  readonly calibrationId: string;
  readonly version: string;
  readonly method: DecisionCalibrationMethod;
  readonly minimumConfidencePpm: number;
  readonly minimumMarginPpm: number;
  readonly multiLabelThresholdPpm: number;
  readonly scopeKind: "labels" | "options" | "questions";
  readonly scopeDigest: string;
  readonly modelId: string;
  readonly modelRevision: string | null;
  readonly runtimes: readonly DecisionCalibrationRuntimeBinding[];
  readonly decisionSchemaMin: string;
  readonly decisionSchemaMaxExclusive: string;
  readonly datasetDigest: string;
  readonly inDomainDigest: string;
  readonly outOfDomainDigest: string;
  readonly evaluationPath: string | null;
  readonly evaluationSha256: string | null;
  readonly profileDigest: string;
}

/** Value-free result of reading calibration profiles from a bundle. */
export interface DecisionCalibrationRead {
  readonly ok: boolean;
  readonly reason: DecisionCalibrationReason | null;
  readonly contentId: string;
  readonly profiles: readonly BundleCalibrationProfile[];
}

/** What the caller loaded and needs a calibration profile for. */
export interface DecisionCalibrationRequest {
  readonly calibrationId: string;
  readonly runtimeId: string;
  readonly decisionSchemaVersion: string;
  readonly modelId?: string | null;
  readonly modelRevision?: string | null;
  readonly runtimeDigest?: string | null;
  readonly scopeKind?: string | null;
  readonly scopeDigest?: string | null;
  readonly profileDigest?: string | null;
}

/** Value-free result of resolving the profile for a loaded runtime. */
export interface DecisionCalibrationSelection {
  readonly ok: boolean;
  readonly reason: DecisionCalibrationReason | null;
  readonly contentId: string;
  readonly profile: BundleCalibrationProfile | null;
}

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type JsonObject = { [key: string]: Json };

const whole = (pattern: string): RegExp => new RegExp(`^(?:${pattern})$`);
const CONTROLLED = whole("[a-z][a-z0-9_.:/-]{0,127}");
const TOKEN = whole("[a-z0-9][a-z0-9._-]{0,63}");
const MODEL_ID = whole("[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}");
const REVISION = whole("[0-9a-f]{40}");
const SHA256 = whole("sha256:[0-9a-f]{64}");
const SEMVER = whole("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){2}");
const EVALUATION_PATH = whole("[A-Za-z0-9_][A-Za-z0-9._+/-]{0,511}");
const METHODS = ["deterministic_margin", "isotonic_regression", "platt_scaling", "temperature_scaling"];
const SCOPE_KINDS = ["labels", "options", "questions"];
const PROFILE_KEYS = [
  "calibration_id",
  "decision_schema",
  "method",
  "model",
  "parameters",
  "profile_digest",
  "provenance",
  "scope",
  "thresholds",
  "version",
];
const MAX_PROFILES = 16;
const MAX_RUNTIMES = 32;
const MAX_POINTS = 64;
const MAX_TEMPERATURE = 100_000_000;
const MAX_PLATT = 100_000_000;
const PPM = DECISION_CALIBRATION_PPM_SCALE;

class Malformed extends Error {}

const encoder = new TextEncoder();

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function object(value: Json | undefined, keys: readonly string[]): JsonObject {
  if (!isRecord(value)) throw new Malformed();
  const own = Object.keys(value);
  if (own.length !== keys.length || !keys.every((key) => Object.hasOwn(value, key))) {
    throw new Malformed();
  }
  return value;
}

function text(value: Json | undefined, pattern: RegExp): string {
  if (typeof value !== "string" || !pattern.test(value)) throw new Malformed();
  return value;
}

function integer(value: Json | undefined, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Malformed();
  }
  return value;
}

function semverKey(value: string): number[] {
  return value.split(".").map(Number);
}

function compare(left: readonly number[], right: readonly number[]): number {
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] as number;
    const b = right[index] as number;
    if (a !== b) return a < b ? -1 : 1;
  }
  return 0;
}

async function digestOf(value: unknown): Promise<string> {
  return portableBundleContentId(encoder.encode(canonicalJson(value)));
}

function parseMethod(name: Json | undefined, value: Json | undefined): DecisionCalibrationMethod {
  if (typeof name !== "string" || !METHODS.includes(name)) throw new Malformed();
  if (name === "temperature_scaling") {
    const parameters = object(value, ["temperature_ppm"]);
    return {
      kind: name,
      temperaturePpm: integer(parameters.temperature_ppm, 1, MAX_TEMPERATURE),
    };
  }
  if (name === "platt_scaling") {
    const parameters = object(value, ["intercept_ppm", "slope_ppm"]);
    const slopePpm = integer(parameters.slope_ppm, -MAX_PLATT, MAX_PLATT);
    if (slopePpm === 0) throw new Malformed();
    return {
      kind: name,
      slopePpm,
      interceptPpm: integer(parameters.intercept_ppm, -MAX_PLATT, MAX_PLATT),
    };
  }
  if (name === "isotonic_regression") {
    const raw = object(value, ["points"]).points;
    if (!Array.isArray(raw) || raw.length < 2 || raw.length > MAX_POINTS) throw new Malformed();
    const points: { inputPpm: number; outputPpm: number }[] = [];
    for (const item of raw) {
      const point = object(item, ["input_ppm", "output_ppm"]);
      const next = {
        inputPpm: integer(point.input_ppm, 0, PPM),
        outputPpm: integer(point.output_ppm, 0, PPM),
      };
      const last = points[points.length - 1];
      if (last !== undefined && (next.inputPpm <= last.inputPpm || next.outputPpm < last.outputPpm)) {
        throw new Malformed();
      }
      points.push(next);
    }
    return { kind: name, points };
  }
  object(value, []);
  return { kind: "deterministic_margin" };
}

function parseProfile(value: Json): BundleCalibrationProfile {
  const profile = object(value, PROFILE_KEYS);
  const calibrationId = text(profile.calibration_id, CONTROLLED);
  const version = text(profile.version, SEMVER);
  const method = parseMethod(profile.method, profile.parameters);

  const thresholds = object(profile.thresholds, [
    "minimum_confidence_ppm",
    "minimum_margin_ppm",
    "multi_label_threshold_ppm",
  ]);
  const minimumConfidencePpm = integer(thresholds.minimum_confidence_ppm, 0, PPM);
  const minimumMarginPpm = integer(thresholds.minimum_margin_ppm, 0, PPM);
  const multiLabelThresholdPpm = integer(thresholds.multi_label_threshold_ppm, 0, PPM);
  if (minimumMarginPpm > minimumConfidencePpm) throw new Malformed();

  const scope = object(profile.scope, ["digest", "kind"]);
  const scopeKind = scope.kind;
  if (typeof scopeKind !== "string" || !SCOPE_KINDS.includes(scopeKind)) throw new Malformed();
  const scopeDigest = text(scope.digest, SHA256);

  const model = object(profile.model, ["id", "revision", "runtimes"]);
  const modelId = text(model.id, MODEL_ID);
  const modelRevision = model.revision === null ? null : text(model.revision, REVISION);
  const rawRuntimes = model.runtimes;
  if (!Array.isArray(rawRuntimes) || rawRuntimes.length < 1 || rawRuntimes.length > MAX_RUNTIMES) {
    throw new Malformed();
  }
  const runtimes = rawRuntimes.map((item) => {
    const binding = object(item, ["digest", "id"]);
    return { runtimeId: text(binding.id, TOKEN), digest: text(binding.digest, SHA256) };
  });
  if (new Set(runtimes.map((item) => item.runtimeId)).size !== runtimes.length) {
    throw new Malformed();
  }

  const range = object(profile.decision_schema, ["max_exclusive", "min"]);
  const decisionSchemaMin = text(range.min, SEMVER);
  const decisionSchemaMaxExclusive = text(range.max_exclusive, SEMVER);
  if (compare(semverKey(decisionSchemaMin), semverKey(decisionSchemaMaxExclusive)) >= 0) {
    throw new Malformed();
  }

  const provenance = object(profile.provenance, [
    "dataset_digest",
    "evaluation",
    "in_domain_digest",
    "out_of_domain_digest",
  ]);
  const datasetDigest = text(provenance.dataset_digest, SHA256);
  const inDomainDigest = text(provenance.in_domain_digest, SHA256);
  const outOfDomainDigest = text(provenance.out_of_domain_digest, SHA256);
  let evaluationPath: string | null = null;
  let evaluationSha256: string | null = null;
  if (provenance.evaluation !== null) {
    const evaluation = object(provenance.evaluation, ["path", "sha256"]);
    evaluationPath = text(evaluation.path, EVALUATION_PATH);
    evaluationSha256 = text(evaluation.sha256, SHA256);
  }
  return {
    calibrationId,
    version,
    method,
    minimumConfidencePpm,
    minimumMarginPpm,
    multiLabelThresholdPpm,
    scopeKind: scopeKind as BundleCalibrationProfile["scopeKind"],
    scopeDigest,
    modelId,
    modelRevision,
    runtimes,
    decisionSchemaMin,
    decisionSchemaMaxExclusive,
    datasetDigest,
    inDomainDigest,
    outOfDomainDigest,
    evaluationPath,
    evaluationSha256,
    profileDigest: text(profile.profile_digest, SHA256),
  };
}

function fileDigests(manifest: JsonObject): Map<string, string> {
  const digests = new Map<string, string>();
  for (const item of (manifest.files as JsonObject[] | undefined) ?? []) {
    digests.set(item.path as string, item.sha256 as string);
  }
  return digests;
}

async function runtimeDigest(manifest: JsonObject, runtimeId: string): Promise<string | null> {
  const runtime = ((manifest.runtimes as JsonObject[] | undefined) ?? []).find(
    (item) => item.id === runtimeId,
  );
  if (runtime === undefined) return null;
  const digests = fileDigests(manifest);
  const files = [];
  for (const path of runtime.files as string[]) {
    const digest = digests.get(path);
    if (digest === undefined) return null;
    files.push({ path, sha256: digest });
  }
  return digestOf({
    entrypoint: runtime.entrypoint,
    files,
    format: runtime.format,
    id: runtimeId,
    precision: runtime.precision,
  });
}

function profileJson(profile: BundleCalibrationProfile): JsonObject {
  const method = profile.method;
  let parameters: JsonObject = {};
  if (method.kind === "temperature_scaling") {
    parameters = { temperature_ppm: method.temperaturePpm };
  } else if (method.kind === "platt_scaling") {
    parameters = { intercept_ppm: method.interceptPpm, slope_ppm: method.slopePpm };
  } else if (method.kind === "isotonic_regression") {
    parameters = {
      points: method.points.map((point) => ({
        input_ppm: point.inputPpm,
        output_ppm: point.outputPpm,
      })),
    };
  }
  return {
    calibration_id: profile.calibrationId,
    decision_schema: {
      max_exclusive: profile.decisionSchemaMaxExclusive,
      min: profile.decisionSchemaMin,
    },
    method: method.kind,
    model: {
      id: profile.modelId,
      revision: profile.modelRevision,
      runtimes: profile.runtimes.map((item) => ({ digest: item.digest, id: item.runtimeId })),
    },
    parameters,
    profile_digest: profile.profileDigest,
    provenance: {
      dataset_digest: profile.datasetDigest,
      evaluation:
        profile.evaluationPath === null
          ? null
          : { path: profile.evaluationPath, sha256: profile.evaluationSha256 },
      in_domain_digest: profile.inDomainDigest,
      out_of_domain_digest: profile.outOfDomainDigest,
    },
    scope: { digest: profile.scopeDigest, kind: profile.scopeKind },
    thresholds: {
      minimum_confidence_ppm: profile.minimumConfidencePpm,
      minimum_margin_ppm: profile.minimumMarginPpm,
      multi_label_threshold_ppm: profile.multiLabelThresholdPpm,
    },
    version: profile.version,
  };
}

/** Return the canonical JSON of the extension profile object, rebuilt from typed fields. */
export function bundleCalibrationProfileJson(profile: BundleCalibrationProfile): string {
  return canonicalJson(profileJson(profile));
}

/**
 * Validate the calibration extension of a structurally valid manifest value in
 * the normative order. Throws {@link DecisionCalibrationError} on failure.
 */
export async function validateDecisionCalibrationExtension(
  manifest: Record<string, unknown>,
): Promise<BundleCalibrationProfile[]> {
  const root = manifest as JsonObject;
  const raw = (root.extensions as JsonObject)[DECISION_CALIBRATION_EXTENSION];
  let items: Json[];
  let profiles: BundleCalibrationProfile[];
  try {
    if (!isRecord(raw) || typeof raw.schema_version !== "number") throw new Malformed();
    if (raw.schema_version !== DECISION_CALIBRATION_EXTENSION_VERSION) {
      throw new DecisionCalibrationError("calibration_version_unsupported");
    }
    object(raw, ["profiles", "schema_version"]);
    const list = raw.profiles;
    if (!Array.isArray(list) || list.length < 1 || list.length > MAX_PROFILES) throw new Malformed();
    items = list;
    profiles = list.map(parseProfile);
    const bindings = profiles.flatMap((profile) =>
      profile.runtimes.map((item) => `${profile.calibrationId}\n${item.runtimeId}`),
    );
    if (new Set(bindings).size !== bindings.length) throw new Malformed();
  } catch (error) {
    if (error instanceof Malformed) throw new DecisionCalibrationError("calibration_malformed");
    throw error;
  }

  const model = root.model as JsonObject;
  const runtimeIds = new Set(((root.runtimes as JsonObject[]) ?? []).map((item) => item.id));
  const digests = fileDigests(root);
  const labels = root.labels as JsonObject | null;
  const labelsDigest = labels === null ? undefined : digests.get(labels.path as string);
  for (let index = 0; index < profiles.length; index += 1) {
    const profile = profiles[index] as BundleCalibrationProfile;
    const body = { ...(items[index] as JsonObject) };
    delete body.profile_digest;
    if ((await digestOf(body)) !== profile.profileDigest) {
      throw new DecisionCalibrationError("calibration_digest_mismatch");
    }
    if (profile.modelId !== model.id || profile.modelRevision !== model.revision) {
      throw new DecisionCalibrationError("calibration_model_mismatch");
    }
    for (const binding of profile.runtimes) {
      if (
        !runtimeIds.has(binding.runtimeId) ||
        (await runtimeDigest(root, binding.runtimeId)) !== binding.digest
      ) {
        throw new DecisionCalibrationError("calibration_runtime_mismatch");
      }
    }
    if (
      profile.scopeKind === "labels" &&
      (labelsDigest === undefined || labelsDigest !== profile.scopeDigest)
    ) {
      throw new DecisionCalibrationError("calibration_scope_mismatch");
    }
    if (
      profile.evaluationPath !== null &&
      digests.get(profile.evaluationPath) !== profile.evaluationSha256
    ) {
      throw new DecisionCalibrationError("calibration_evidence_mismatch");
    }
  }
  return profiles;
}

function structural(manifest: Uint8Array): Record<string, unknown> | null {
  try {
    const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(manifest)) as unknown;
    const critical = isRecord(value) ? value.critical_extensions : undefined;
    const understood = Array.isArray(critical)
      ? critical.filter((item): item is string => typeof item === "string")
      : [];
    return parsePortableBundleManifest(manifest, understood);
  } catch {
    return null;
  }
}

async function read(
  manifest: Uint8Array,
  verification: PortableBundleVerification,
  requireFiles: boolean,
): Promise<DecisionCalibrationRead> {
  const contentId = await portableBundleContentId(manifest);
  const refused = (reason: DecisionCalibrationReason): DecisionCalibrationRead => ({
    ok: false,
    reason,
    contentId,
    profiles: [],
  });
  if (
    !verification.ok ||
    verification.contentId !== contentId ||
    (requireFiles && !verification.filesVerified)
  ) {
    return refused("calibration_bundle_unverified");
  }
  const value = structural(manifest);
  if (value === null) return refused("calibration_bundle_unverified");
  const extensions = value.extensions as Record<string, unknown>;
  if (!Object.hasOwn(extensions, DECISION_CALIBRATION_EXTENSION)) {
    return refused("calibration_extension_missing");
  }
  try {
    const profiles = await validateDecisionCalibrationExtension(value);
    return { ok: true, reason: null, contentId, profiles };
  } catch (error) {
    if (error instanceof DecisionCalibrationError) return refused(error.reason);
    throw error;
  }
}

/** Read and validate every calibration profile of a verified bundle. */
export async function readDecisionCalibration(
  manifest: Uint8Array,
  verification: PortableBundleVerification,
): Promise<DecisionCalibrationRead> {
  return read(manifest, verification, false);
}

function optional(value: string | null | undefined, pattern: RegExp): boolean {
  return value === undefined || value === null || (typeof value === "string" && pattern.test(value));
}

function absent(value: unknown): value is null | undefined {
  return value === undefined || value === null;
}

/**
 * Validate a resolution request. Throws `calibration_request_invalid` when a
 * field is malformed.
 */
export function validateDecisionCalibrationRequest(request: DecisionCalibrationRequest): void {
  const valid =
    typeof request.calibrationId === "string" &&
    CONTROLLED.test(request.calibrationId) &&
    typeof request.runtimeId === "string" &&
    TOKEN.test(request.runtimeId) &&
    typeof request.decisionSchemaVersion === "string" &&
    SEMVER.test(request.decisionSchemaVersion) &&
    optional(request.modelId, MODEL_ID) &&
    optional(request.modelRevision, REVISION) &&
    optional(request.runtimeDigest, SHA256) &&
    optional(request.profileDigest, SHA256) &&
    absent(request.scopeKind) === absent(request.scopeDigest) &&
    (absent(request.scopeKind) ||
      (typeof request.scopeKind === "string" && SCOPE_KINDS.includes(request.scopeKind))) &&
    optional(request.scopeDigest, SHA256);
  if (!valid) throw new DecisionCalibrationError("calibration_request_invalid");
}

/**
 * Select the profile for the runtime a caller loaded, failing closed. The
 * bundle must have verified with its file tree; any loaded model identity,
 * decision schema version, scope, and pinned profile digest the caller
 * supplies must then match the selected profile.
 */
export async function resolveDecisionCalibration(
  manifest: Uint8Array,
  verification: PortableBundleVerification,
  request: DecisionCalibrationRequest,
): Promise<DecisionCalibrationSelection> {
  validateDecisionCalibrationRequest(request);
  const result = await read(manifest, verification, true);
  const refused = (reason: DecisionCalibrationReason | null): DecisionCalibrationSelection => ({
    ok: false,
    reason,
    contentId: result.contentId,
    profile: null,
  });
  if (!result.ok) return refused(result.reason);
  const candidates = result.profiles.filter((item) => item.calibrationId === request.calibrationId);
  if (candidates.length === 0) return refused("calibration_profile_missing");
  const bound = (profile: BundleCalibrationProfile): string | undefined =>
    profile.runtimes.find((item) => item.runtimeId === request.runtimeId)?.digest;
  const profile = candidates.find((item) => bound(item) !== undefined);
  if (profile === undefined) return refused("calibration_runtime_unbound");
  if (
    (!absent(request.modelId) && request.modelId !== profile.modelId) ||
    (!absent(request.modelRevision) && request.modelRevision !== profile.modelRevision) ||
    (!absent(request.runtimeDigest) && request.runtimeDigest !== bound(profile))
  ) {
    return refused("calibration_model_mismatch");
  }
  const version = semverKey(request.decisionSchemaVersion);
  if (
    compare(version, semverKey(profile.decisionSchemaMin)) < 0 ||
    compare(version, semverKey(profile.decisionSchemaMaxExclusive)) >= 0
  ) {
    return refused("calibration_schema_incompatible");
  }
  if (
    !absent(request.scopeKind) &&
    (request.scopeKind !== profile.scopeKind || request.scopeDigest !== profile.scopeDigest)
  ) {
    return refused("calibration_scope_mismatch");
  }
  if (!absent(request.profileDigest) && request.profileDigest !== profile.profileDigest) {
    return refused("calibration_pin_mismatch");
  }
  return { ok: true, reason: null, contentId: result.contentId, profile };
}

/**
 * Return the digest that binds a profile to one runtime of canonical manifest
 * bytes, or `null` when the manifest or runtime is invalid.
 */
export async function decisionCalibrationRuntimeDigest(
  manifest: Uint8Array,
  runtimeId: string,
): Promise<string | null> {
  const value = structural(manifest);
  if (value === null) return null;
  return runtimeDigest(value as JsonObject, runtimeId);
}
