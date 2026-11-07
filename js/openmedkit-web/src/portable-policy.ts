import {
  KIT_PARITY_ACTIONS,
  KIT_PARITY_CONFIDENCE_SCALE,
  applyKitParitySpans,
  scalarLength,
  scalarToUtf16Offset,
  validateKitParityResult,
  type KitParityResult,
  type KitParitySpan,
} from "./kit-parity";
import { PORTABLE_POLICY_RESOURCES } from "./portable-policy-resources";

/**
 * Portable OpenMedKit policy subset (`openmed.portable_policy` v1).
 *
 * The TypeScript twin of `openmed.core.portable_policy`. Profiles are the
 * generated, byte-identical resources in `portable-policy-resources.ts`;
 * loading verifies the manifest, profile and taxonomy digests, the IR
 * version, and the required capabilities, and fails closed with a
 * value-free {@link PortablePolicyError}. {@link decidePortablePolicy}
 * reproduces the shared decision vectors in
 * `tests/fixtures/parity/portable_policy_vectors.json` byte for byte.
 */
export const PORTABLE_POLICY_IR_SCHEMA = "openmed.portable_policy" as const;
export const PORTABLE_POLICY_IR_VERSION = 1 as const;
export const PORTABLE_POLICY_MANIFEST_FILE = "portable_policy_manifest.json" as const;
export const PORTABLE_POLICY_TAXONOMY_FILE = "portable_policy_taxonomy.json" as const;
export const PORTABLE_POLICY_DEFAULT_PROFILE = "hipaa_safe_harbor" as const;
export const PORTABLE_POLICY_SURROGATE_KEY_MIN_BYTES = 16;
export const PORTABLE_POLICY_SURROGATE_KEY_MAX_BYTES = 64;

/** Capabilities this runtime implements. */
export const PORTABLE_POLICY_SUPPORTED_CAPABILITIES: readonly string[] = [
  "actions.v1",
  "labels.portable_taxonomy.v1",
  "merge.cluster_union.v1",
  "merge.semantic_units.v1",
  "strict_no_leak.v1",
  "surrogate.hmac_sha256.v1",
  "thresholds.keep_floor.v1",
];

const MANIFEST_SCHEMA = "openmed.portable_policy_manifest";
const MANIFEST_VERSION = 1;
const TAXONOMY_SCHEMA = "openmed.portable_taxonomy";
const TAXONOMY_VERSION = 1;
const MERGE_ALGORITHM = "openmed.portable_merge";
const MERGE_VERSION = 1;
const SURROGATE_ALGORITHM = "openmed.portable_surrogate.hmac_sha256";
const SURROGATE_VERSION = 1;
const SURROGATE_DOMAIN = "openmed.portable_surrogate.v1";
const LABEL_PATTERN = /^[A-Za-z][A-Za-z0-9_:-]{0,79}$/;
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ACTIONS = new Set<string>(KIT_PARITY_ACTIONS);
const ACTION_STRENGTH: Readonly<Record<string, number>> = {
  keep: 0,
  replace: 1,
  hash: 2,
  mask: 3,
  redact: 4,
  remove: 5,
};

/** Value-free policy failure: a reason code and a JSON field path only. */
export class PortablePolicyError extends Error {
  readonly code: string;
  readonly fieldPath: string;

  constructor(code: string, fieldPath = "$") {
    super(`${code} at ${fieldPath}`);
    this.name = "PortablePolicyError";
    this.code = code;
    this.fieldPath = fieldPath;
  }
}

export interface PortablePolicyThreshold {
  keepFloorPpm: number;
  escalateBelowPpm: number;
}

export interface PortablePolicyTaxonomy {
  labels: readonly string[];
  policyLabels: Readonly<Record<string, string>>;
  specificity: Readonly<Record<string, number>>;
  aliases: Readonly<Record<string, string>>;
  unknownLabel: string;
  digest: string;
}

export interface PortablePolicyManifest {
  defaultProfile: string;
  profiles: Readonly<Record<string, string>>;
  excludedProfiles: Readonly<Record<string, string>>;
  profileAliases: Readonly<Record<string, string>>;
  taxonomyDigest: string;
}

export interface PortablePolicy {
  name: string;
  digest: string;
  taxonomy: PortablePolicyTaxonomy;
  actions: Readonly<Record<string, string>>;
  defaultAction: string;
  arbitrationMode: "balanced" | "high_recall_union";
  strictNoLeak: boolean;
  thresholdProfile: string;
  defaultThreshold: PortablePolicyThreshold;
  labelThresholds: Readonly<Record<string, PortablePolicyThreshold>>;
  requiredCapabilities: readonly string[];
  /** Python-only features of the source profile the portable subset omits. */
  excludedFeatures: readonly string[];
  forcedCascadeTiers: readonly string[];
}

/** One normalized detector span; offsets are Unicode scalar offsets. */
export interface PortablePolicyDetection {
  label: string;
  start: number;
  end: number;
  confidence_ppm: number;
  /** `model` (default) or `semantic` for pattern units. */
  source?: string;
  /** For semantic units, whether the pattern validator passed (default true). */
  validated?: boolean;
}

/** Value-free decision record for one candidate span. */
export interface PortablePolicyDecision {
  inputs: number[];
  kind: "model" | "merged" | "semantic_only";
  label: string;
  canonical_label: string;
  recognized: boolean;
  start: number;
  end: number;
  confidence_ppm: number;
  keep_floor_ppm: number;
  escalate: boolean;
  action: string;
  strict_upgrade: boolean;
  outcome: "applied" | "overlap_loser" | "duplicate" | "below_threshold";
  span?: number;
}

export interface PortablePolicyDecisionResult {
  policyName: string;
  policyDigest: string;
  result: KitParityResult;
  decisions: PortablePolicyDecision[];
}

export interface PortablePolicyDecideOptions {
  /** Canonical label to action overrides. */
  overrides?: Readonly<Record<string, string>>;
  /** Caller-supplied 16-64 byte key for keyed `replace` surrogates. */
  surrogateKey?: Uint8Array;
}

export interface LoadPortablePolicyOptions {
  /** Pin a `sha256:` digest; a different bundled profile fails closed. */
  expectedDigest?: string;
  /** Resource text by file name; defaults to the bundled resources. */
  resources?: Readonly<Record<string, string>>;
  supportedCapabilities?: readonly string[];
}

/** Return the bundled portable profile names, sorted. */
export function portablePolicyNames(): string[] {
  return Object.keys(bundledManifest().profiles).sort();
}

/** Load and verify a portable profile by name or alias. */
export function loadPortablePolicy(
  name: string,
  options: LoadPortablePolicyOptions = {},
): PortablePolicy {
  const resources = options.resources ?? PORTABLE_POLICY_RESOURCES;
  const manifestText = resources[PORTABLE_POLICY_MANIFEST_FILE];
  if (manifestText === undefined) {
    throw new PortablePolicyError("resource_missing", "$.manifest");
  }
  const manifest = decodePortablePolicyManifest(manifestText);
  const canonical = resolvePortablePolicyName(name, manifest);
  const digest = manifest.profiles[canonical] as string;
  if (options.expectedDigest !== undefined && options.expectedDigest !== digest) {
    throw new PortablePolicyError("digest_mismatch", "$.digest");
  }
  const text = resources[`${canonical}.json`];
  if (text === undefined) {
    throw new PortablePolicyError("resource_missing", "$.profiles");
  }
  if (sha256Digest(text) !== digest) {
    throw new PortablePolicyError("resource_digest_mismatch", "$.profiles");
  }
  const taxonomyText = resources[PORTABLE_POLICY_TAXONOMY_FILE];
  if (taxonomyText === undefined) {
    throw new PortablePolicyError("resource_missing", "$.taxonomy_digest");
  }
  const taxonomyDigest = sha256Digest(taxonomyText);
  if (taxonomyDigest !== manifest.taxonomyDigest) {
    throw new PortablePolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest");
  }
  const taxonomy = decodePortablePolicyTaxonomy(taxonomyText, taxonomyDigest);
  return decodePortablePolicyProfile(parseObject(text, "profile_invalid"), {
    name: canonical,
    digest,
    taxonomy,
    ...(options.supportedCapabilities
      ? { supportedCapabilities: options.supportedCapabilities }
      : {}),
  });
}

/** Decode and validate manifest text. */
export function decodePortablePolicyManifest(text: string): PortablePolicyManifest {
  const document = parseObject(text, "manifest_invalid");
  if (document.schema !== MANIFEST_SCHEMA) {
    throw new PortablePolicyError("manifest_invalid", "$.schema");
  }
  if (document.schema_version !== MANIFEST_VERSION) {
    throw new PortablePolicyError("manifest_version_unsupported", "$.schema_version");
  }
  if (document.ir_schema !== PORTABLE_POLICY_IR_SCHEMA) {
    throw new PortablePolicyError("manifest_invalid", "$.ir_schema");
  }
  if (document.ir_version !== PORTABLE_POLICY_IR_VERSION) {
    throw new PortablePolicyError("ir_version_unsupported", "$.ir_version");
  }
  const profiles = stringRecord(document.profiles);
  if (!profiles) throw new PortablePolicyError("manifest_invalid", "$.profiles");
  const excluded = stringRecord(document.excluded_profiles);
  if (!excluded) throw new PortablePolicyError("manifest_invalid", "$.excluded_profiles");
  const aliases = stringRecord(document.profile_aliases);
  if (!aliases) throw new PortablePolicyError("manifest_invalid", "$.profile_aliases");
  if (!Object.values(profiles).every((value) => DIGEST_PATTERN.test(value))) {
    throw new PortablePolicyError("manifest_invalid", "$.profiles");
  }
  const taxonomyDigest = document.taxonomy_digest;
  if (typeof taxonomyDigest !== "string" || !DIGEST_PATTERN.test(taxonomyDigest)) {
    throw new PortablePolicyError("manifest_invalid", "$.taxonomy_digest");
  }
  const defaultProfile = document.default_profile;
  if (typeof defaultProfile !== "string" || !hasOwn(profiles, defaultProfile)) {
    throw new PortablePolicyError("manifest_invalid", "$.default_profile");
  }
  return {
    defaultProfile,
    profiles,
    excludedProfiles: excluded,
    profileAliases: aliases,
    taxonomyDigest,
  };
}

/** Resolve a profile name or alias against a verified manifest. */
export function resolvePortablePolicyName(
  name: string,
  manifest: PortablePolicyManifest,
): string {
  const trimmed = typeof name === "string" ? name.trim() : "";
  if (trimmed.length === 0) {
    throw new PortablePolicyError("profile_name_invalid", "$.name");
  }
  const normalized = trimmed.toLowerCase().replaceAll("-", "_");
  const canonical = hasOwn(manifest.profileAliases, normalized)
    ? (manifest.profileAliases[normalized] as string)
    : normalized;
  if (hasOwn(manifest.profiles, canonical)) {
    return canonical;
  }
  if (hasOwn(manifest.excludedProfiles, canonical)) {
    throw new PortablePolicyError("profile_not_portable", "$.name");
  }
  throw new PortablePolicyError("profile_unknown", "$.name");
}

/** Decode and validate taxonomy text. */
export function decodePortablePolicyTaxonomy(
  text: string,
  digest: string,
): PortablePolicyTaxonomy {
  const document = parseObject(text, "taxonomy_invalid");
  if (document.name !== TAXONOMY_SCHEMA) {
    throw new PortablePolicyError("taxonomy_invalid", "$.name");
  }
  if (document.version !== TAXONOMY_VERSION) {
    throw new PortablePolicyError("taxonomy_invalid", "$.version");
  }
  const rawLabels = document.labels;
  if (!isRecord(rawLabels) || Object.keys(rawLabels).length === 0) {
    throw new PortablePolicyError("taxonomy_invalid", "$.labels");
  }
  const policyLabels: Record<string, string> = {};
  const specificity: Record<string, number> = {};
  for (const [label, entry] of Object.entries(rawLabels)) {
    if (
      !isRecord(entry) ||
      typeof entry.policy_label !== "string" ||
      !Number.isInteger(entry.specificity)
    ) {
      throw new PortablePolicyError("taxonomy_invalid", "$.labels");
    }
    policyLabels[label] = entry.policy_label;
    specificity[label] = entry.specificity as number;
  }
  const aliases = stringRecord(document.aliases);
  if (!aliases || !Object.values(aliases).every((value) => hasOwn(policyLabels, value))) {
    throw new PortablePolicyError("taxonomy_invalid", "$.aliases");
  }
  const unknown = document.unknown_label;
  if (typeof unknown !== "string" || !hasOwn(policyLabels, unknown)) {
    throw new PortablePolicyError("taxonomy_invalid", "$.unknown_label");
  }
  return {
    labels: Object.keys(policyLabels).sort(),
    policyLabels,
    specificity,
    aliases,
    unknownLabel: unknown,
    digest,
  };
}

/** Validate a decoded profile document in the order every kit shares. */
export function decodePortablePolicyProfile(
  document: Readonly<Record<string, unknown>>,
  context: {
    name: string;
    digest: string;
    taxonomy: PortablePolicyTaxonomy;
    supportedCapabilities?: readonly string[];
  },
): PortablePolicy {
  const { name, digest, taxonomy } = context;
  const supported = new Set(
    context.supportedCapabilities ?? PORTABLE_POLICY_SUPPORTED_CAPABILITIES,
  );
  if (document.ir_schema !== PORTABLE_POLICY_IR_SCHEMA) {
    throw new PortablePolicyError("profile_invalid", "$.ir_schema");
  }
  if (document.ir_version !== PORTABLE_POLICY_IR_VERSION) {
    throw new PortablePolicyError("ir_version_unsupported", "$.ir_version");
  }
  if (document.name !== name) {
    throw new PortablePolicyError("profile_invalid", "$.name");
  }
  if (document.taxonomy_digest !== taxonomy.digest) {
    throw new PortablePolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest");
  }
  const required = stringArray(document.required_capabilities);
  if (!required) {
    throw new PortablePolicyError("profile_invalid", "$.required_capabilities");
  }
  if (!required.every((item) => supported.has(item))) {
    throw new PortablePolicyError("capability_missing", "$.required_capabilities");
  }
  const merge = document.merge;
  if (!isRecord(merge) || merge.algorithm !== MERGE_ALGORITHM || merge.version !== MERGE_VERSION) {
    throw new PortablePolicyError("merge_algorithm_unsupported", "$.merge");
  }
  const surrogate = document.surrogate;
  if (
    !isRecord(surrogate) ||
    surrogate.algorithm !== SURROGATE_ALGORITHM ||
    surrogate.version !== SURROGATE_VERSION
  ) {
    throw new PortablePolicyError("surrogate_algorithm_unsupported", "$.surrogate");
  }
  const mode = document.arbitration_mode;
  if (mode !== "balanced" && mode !== "high_recall_union") {
    throw new PortablePolicyError("profile_invalid", "$.arbitration_mode");
  }
  const strict = document.strict_no_leak;
  if (typeof strict !== "boolean") {
    throw new PortablePolicyError("profile_invalid", "$.strict_no_leak");
  }
  const actions = stringRecord(document.actions);
  const labelSet = new Set(taxonomy.labels);
  if (
    !actions ||
    Object.keys(actions).length !== labelSet.size ||
    !Object.keys(actions).every((label) => labelSet.has(label)) ||
    !Object.values(actions).every((action) => ACTIONS.has(action))
  ) {
    throw new PortablePolicyError("profile_invalid", "$.actions");
  }
  const defaultAction = document.default_action;
  if (typeof defaultAction !== "string" || !ACTIONS.has(defaultAction)) {
    throw new PortablePolicyError("profile_invalid", "$.default_action");
  }
  const thresholds = document.thresholds;
  if (!isRecord(thresholds)) {
    throw new PortablePolicyError("profile_invalid", "$.thresholds");
  }
  const defaultThreshold = decodeThreshold(thresholds.default);
  const rawLabelThresholds = thresholds.labels;
  if (
    !isRecord(rawLabelThresholds) ||
    !Object.keys(rawLabelThresholds).every((label) => labelSet.has(label))
  ) {
    throw new PortablePolicyError("profile_invalid", "$.thresholds");
  }
  const labelThresholds: Record<string, PortablePolicyThreshold> = {};
  for (const [label, entry] of Object.entries(rawLabelThresholds)) {
    labelThresholds[label] = decodeThreshold(entry);
  }
  return {
    name,
    digest,
    taxonomy,
    actions,
    defaultAction,
    arbitrationMode: mode,
    strictNoLeak: strict,
    thresholdProfile: typeof document.threshold_profile === "string" ? document.threshold_profile : "",
    defaultThreshold,
    labelThresholds,
    requiredCapabilities: required,
    excludedFeatures: stringArray(document.excluded_features) ?? [],
    forcedCascadeTiers: stringArray(document.forced_cascade_tiers) ?? [],
  };
}

/** Return the normalization key of a detector label. */
export function portableLabelKey(label: string): string {
  const trimmed = label.trim();
  const stripped = /^[BIES]-/.test(trimmed) ? trimmed.slice(2) : trimmed;
  return stripped.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** Normalize a detector label through the portable taxonomy. */
export function normalizePortableLabel(
  taxonomy: PortablePolicyTaxonomy,
  label: string,
): { canonicalLabel: string; recognized: boolean } {
  const key = portableLabelKey(label);
  if (hasOwn(taxonomy.aliases, key)) {
    return { canonicalLabel: taxonomy.aliases[key] as string, recognized: true };
  }
  return { canonicalLabel: taxonomy.unknownLabel, recognized: false };
}

interface Candidate {
  inputs: number[];
  kind: PortablePolicyDecision["kind"];
  label: string;
  canonicalLabel: string;
  recognized: boolean;
  start: number;
  end: number;
  confidencePpm: number;
  keepFloorPpm: number;
  escalate: boolean;
  action: string;
  strictUpgrade: boolean;
  outcome: PortablePolicyDecision["outcome"] | "retained";
  span: number | undefined;
}

interface Normalized {
  index: number;
  detection: PortablePolicyDetection;
  canonicalLabel: string;
  recognized: boolean;
}

/**
 * Apply a portable policy to normalized detections.
 *
 * Stages: validation, label normalization, semantic-unit merging, integer
 * ppm thresholds, action lookup with overrides and the strict no-leak
 * upgrade, duplicate collapse, overlap clustering into union spans, and
 * right-to-left rendering at Unicode scalar offsets.
 */
export function decidePortablePolicy(
  policy: PortablePolicy,
  text: string,
  detections: readonly PortablePolicyDetection[],
  options: PortablePolicyDecideOptions = {},
): PortablePolicyDecisionResult {
  validateDetections(detections, scalarLength(text));
  const overrides = validateOverrides(options.overrides, policy.taxonomy);
  const key = options.surrogateKey;
  if (
    key !== undefined &&
    (key.byteLength < PORTABLE_POLICY_SURROGATE_KEY_MIN_BYTES ||
      key.byteLength > PORTABLE_POLICY_SURROGATE_KEY_MAX_BYTES)
  ) {
    throw new PortablePolicyError("surrogate_key_invalid", "$.surrogate_key");
  }

  const candidates = buildCandidates(policy.taxonomy, detections);
  for (const candidate of candidates) {
    const threshold = thresholdFor(policy, candidate.canonicalLabel);
    candidate.keepFloorPpm = threshold.keepFloorPpm;
    const configured = overrides[candidate.canonicalLabel] ?? configuredAction(policy, candidate.canonicalLabel);
    if (policy.strictNoLeak && configured === "keep") {
      candidate.action = "mask";
      candidate.strictUpgrade = true;
    } else {
      candidate.action = configured;
    }
    if (candidate.confidencePpm >= threshold.keepFloorPpm) {
      candidate.outcome = "retained";
      candidate.escalate = candidate.confidencePpm < threshold.escalateBelowPpm;
    }
  }

  const retained = candidates.filter((item) => item.outcome === "retained");
  const unique = collapseDuplicates(retained, policy);
  const spans: KitParitySpan[] = [];
  for (const cluster of clusters(unique)) {
    const winner = representative(cluster, policy);
    const start = Math.min(...cluster.map((item) => item.start));
    const end = Math.max(...cluster.map((item) => item.end));
    for (const member of cluster) {
      member.outcome = member === winner ? "applied" : "overlap_loser";
      member.span = spans.length;
    }
    const surface = text.slice(scalarToUtf16Offset(text, start), scalarToUtf16Offset(text, end));
    const replacement = portablePolicyReplacement(winner.action, winner.canonicalLabel, surface, key);
    spans.push({
      label: winner.label,
      canonical_label: winner.canonicalLabel,
      action: winner.action,
      start,
      end,
      confidence_ppm: winner.confidencePpm,
      ...(replacement === undefined ? {} : { replacement }),
    });
  }
  for (const item of retained) {
    if (item.outcome === "duplicate") {
      const match = unique.find((candidate) => duplicateKey(candidate) === duplicateKey(item));
      item.span = match?.span;
    }
  }

  const deidentifiedText = applyKitParitySpans(text, spans);
  const result: KitParityResult = { deidentified_text: deidentifiedText, spans };
  validateKitParityResult(result);
  const decisions = candidates.map((item): PortablePolicyDecision => ({
    inputs: item.inputs,
    kind: item.kind,
    label: item.label,
    canonical_label: item.canonicalLabel,
    recognized: item.recognized,
    start: item.start,
    end: item.end,
    confidence_ppm: item.confidencePpm,
    keep_floor_ppm: item.keepFloorPpm,
    escalate: item.escalate,
    action: item.action,
    strict_upgrade: item.strictUpgrade,
    outcome: item.outcome as PortablePolicyDecision["outcome"],
    ...(item.span === undefined ? {} : { span: item.span }),
  }));
  return { policyName: policy.name, policyDigest: policy.digest, result, decisions };
}

/** Replacement text for one output span; `undefined` for `keep`. */
export function portablePolicyReplacement(
  action: string,
  canonicalLabel: string,
  surface: string,
  surrogateKey?: Uint8Array,
): string | undefined {
  switch (action) {
    case "keep":
      return undefined;
    case "mask":
    case "redact":
      return `[${canonicalLabel}]`;
    case "remove":
      return "";
    case "replace":
      return surrogateKey === undefined
        ? `[${canonicalLabel}_REPLACED]`
        : `[${canonicalLabel}_${portableSurrogateToken(surrogateKey, canonicalLabel, surface)}]`;
    case "hash":
      return `${canonicalLabel}_${toHex(sha256(utf8(surface)).subarray(0, 12))}`;
    default:
      throw new PortablePolicyError("action_unknown", "$.action");
  }
}

/**
 * Deterministic keyed surrogate token: HMAC-SHA256 over
 * `"openmed.portable_surrogate.v1" 0x00 label 0x00 surface`, first 8 bytes as
 * lowercase hex. The key never leaves the caller.
 */
export function portableSurrogateToken(
  key: Uint8Array,
  canonicalLabel: string,
  surface: string,
): string {
  const message = concat([utf8(SURROGATE_DOMAIN), Uint8Array.of(0), utf8(canonicalLabel), Uint8Array.of(0), utf8(surface)]);
  return toHex(hmacSha256(key, message).subarray(0, 8));
}

/** `sha256:` digest of UTF-8 resource text. */
export function sha256Digest(text: string): string {
  return `sha256:${toHex(sha256(utf8(text)))}`;
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

let manifestCache: PortablePolicyManifest | undefined;

function bundledManifest(): PortablePolicyManifest {
  if (manifestCache === undefined) {
    const text = PORTABLE_POLICY_RESOURCES[PORTABLE_POLICY_MANIFEST_FILE];
    if (text === undefined) {
      throw new PortablePolicyError("resource_missing", "$.manifest");
    }
    manifestCache = decodePortablePolicyManifest(text);
  }
  return manifestCache;
}

function thresholdFor(policy: PortablePolicy, label: string): PortablePolicyThreshold {
  return hasOwn(policy.labelThresholds, label)
    ? (policy.labelThresholds[label] as PortablePolicyThreshold)
    : policy.defaultThreshold;
}

function configuredAction(policy: PortablePolicy, label: string): string {
  if (hasOwn(policy.actions, label)) {
    return policy.actions[label] as string;
  }
  return policy.defaultAction;
}

function decodeThreshold(entry: unknown): PortablePolicyThreshold {
  if (!isRecord(entry) || !isPpm(entry.keep_floor_ppm) || !isPpm(entry.escalate_below_ppm)) {
    throw new PortablePolicyError("profile_invalid", "$.thresholds");
  }
  return { keepFloorPpm: entry.keep_floor_ppm, escalateBelowPpm: entry.escalate_below_ppm };
}

function isPpm(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0 && (value as number) <= KIT_PARITY_CONFIDENCE_SCALE;
}

function validateDetections(detections: readonly PortablePolicyDetection[], length: number): void {
  detections.forEach((detection, index) => {
    const path = `$.detections[${index}]`;
    if (typeof detection.label !== "string" || !LABEL_PATTERN.test(detection.label)) {
      throw new PortablePolicyError("detection_label_invalid", `${path}.label`);
    }
    if (!Number.isInteger(detection.start) || detection.start < 0) {
      throw new PortablePolicyError("detection_offset_invalid", `${path}.start`);
    }
    if (!Number.isInteger(detection.end) || detection.end <= detection.start) {
      throw new PortablePolicyError("detection_offset_invalid", `${path}.end`);
    }
    if (detection.end > length) {
      throw new PortablePolicyError("detection_offset_out_of_bounds", `${path}.end`);
    }
    if (!isPpm(detection.confidence_ppm)) {
      throw new PortablePolicyError("detection_confidence_invalid", `${path}.confidence_ppm`);
    }
    const source = detection.source ?? "model";
    if (source !== "model" && source !== "semantic") {
      throw new PortablePolicyError("detection_source_invalid", `${path}.source`);
    }
    if (detection.validated !== undefined && typeof detection.validated !== "boolean") {
      throw new PortablePolicyError("detection_source_invalid", `${path}.validated`);
    }
  });
}

function validateOverrides(
  overrides: Readonly<Record<string, string>> | undefined,
  taxonomy: PortablePolicyTaxonomy,
): Record<string, string> {
  const resolved: Record<string, string> = {};
  if (!overrides) {
    return resolved;
  }
  for (const label of Object.keys(overrides).sort(asciiCompare)) {
    if (!hasOwn(taxonomy.policyLabels, label)) {
      throw new PortablePolicyError("override_label_unknown", "$.overrides");
    }
    const action = overrides[label];
    if (action === undefined || !ACTIONS.has(action)) {
      throw new PortablePolicyError("override_action_unknown", "$.overrides");
    }
    resolved[label] = action;
  }
  return resolved;
}

function asciiCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareNumbers(left: number, right: number): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function detectionOrder(left: Normalized, right: Normalized): number {
  return (
    compareNumbers(left.detection.start, right.detection.start) ||
    compareNumbers(left.detection.end, right.detection.end) ||
    asciiCompare(left.canonicalLabel, right.canonicalLabel) ||
    asciiCompare(left.detection.label, right.detection.label) ||
    compareNumbers(left.index, right.index)
  );
}

function candidateOrder(left: Candidate, right: Candidate): number {
  return (
    compareNumbers(left.start, right.start) ||
    compareNumbers(left.end, right.end) ||
    asciiCompare(left.canonicalLabel, right.canonicalLabel) ||
    asciiCompare(left.label, right.label) ||
    compareNumbers(left.inputs[0] as number, right.inputs[0] as number)
  );
}

function newCandidate(
  fields: Pick<Candidate, "inputs" | "kind" | "label" | "canonicalLabel" | "recognized" | "start" | "end" | "confidencePpm">,
): Candidate {
  return {
    ...fields,
    keepFloorPpm: 0,
    escalate: false,
    action: "keep",
    strictUpgrade: false,
    outcome: "below_threshold",
    span: undefined,
  };
}

function buildCandidates(
  taxonomy: PortablePolicyTaxonomy,
  detections: readonly PortablePolicyDetection[],
): Candidate[] {
  const normalized: Normalized[] = detections.map((detection, index) => {
    const resolved = normalizePortableLabel(taxonomy, detection.label);
    return { index, detection, canonicalLabel: resolved.canonicalLabel, recognized: resolved.recognized };
  });
  const sourceOf = (entry: Normalized) => entry.detection.source ?? "model";
  const models = normalized.filter((entry) => sourceOf(entry) === "model").sort(detectionOrder);
  const units = normalized.filter((entry) => sourceOf(entry) === "semantic").sort(detectionOrder);
  const consumed = new Set<number>();
  const candidates: Candidate[] = [];
  for (const unit of units) {
    const overlapping = models.filter(
      (entry) => entry.detection.start < unit.detection.end && entry.detection.end > unit.detection.start,
    );
    if (overlapping.length === 0) {
      candidates.push(
        newCandidate({
          inputs: [unit.index],
          kind: "semantic_only",
          label: unit.detection.label,
          canonicalLabel: unit.canonicalLabel,
          recognized: unit.recognized,
          start: unit.detection.start,
          end: unit.detection.end,
          confidencePpm: unit.detection.confidence_ppm,
        }),
      );
      continue;
    }
    const dominant = dominantLabel(overlapping);
    const modelLabels = new Set(overlapping.map((entry) => entry.canonicalLabel));
    const unitRank = taxonomy.specificity[unit.canonicalLabel] ?? 1;
    const dominantRank = taxonomy.specificity[dominant] ?? 1;
    const final = modelLabels.has(unit.canonicalLabel) || dominantRank > unitRank ? dominant : unit.canonicalLabel;
    let carrier: Normalized | undefined;
    for (const entry of overlapping) {
      if (
        entry.canonicalLabel === final &&
        (carrier === undefined || entry.detection.confidence_ppm > carrier.detection.confidence_ppm)
      ) {
        carrier = entry;
      }
    }
    const total = overlapping.reduce((sum, entry) => sum + entry.detection.confidence_ppm, 0);
    const average = Math.floor(total / overlapping.length);
    const [modelWeight, unitWeight] = unit.detection.validated === false ? [9, 1] : [6, 4];
    const confidence = Math.floor((modelWeight * average + unitWeight * unit.detection.confidence_ppm + 5) / 10);
    for (const entry of overlapping) consumed.add(entry.index);
    candidates.push(
      newCandidate({
        inputs: [unit.index, ...overlapping.map((entry) => entry.index)].sort(compareNumbers),
        kind: "merged",
        label: carrier?.detection.label ?? unit.detection.label,
        canonicalLabel: final,
        recognized: carrier?.recognized ?? unit.recognized,
        start: Math.min(unit.detection.start, ...overlapping.map((entry) => entry.detection.start)),
        end: Math.max(unit.detection.end, ...overlapping.map((entry) => entry.detection.end)),
        confidencePpm: confidence,
      }),
    );
  }
  for (const entry of models) {
    if (consumed.has(entry.index)) continue;
    candidates.push(
      newCandidate({
        inputs: [entry.index],
        kind: "model",
        label: entry.detection.label,
        canonicalLabel: entry.canonicalLabel,
        recognized: entry.recognized,
        start: entry.detection.start,
        end: entry.detection.end,
        confidencePpm: entry.detection.confidence_ppm,
      }),
    );
  }
  return candidates.sort(candidateOrder);
}

function dominantLabel(entries: readonly Normalized[]): string {
  const counts = new Map<string, number>();
  const sums = new Map<string, number>();
  for (const entry of entries) {
    counts.set(entry.canonicalLabel, (counts.get(entry.canonicalLabel) ?? 0) + 1);
    sums.set(entry.canonicalLabel, (sums.get(entry.canonicalLabel) ?? 0) + entry.detection.confidence_ppm);
  }
  let best = "";
  for (const label of [...counts.keys()].sort(asciiCompare)) {
    if (best === "") {
      best = label;
      continue;
    }
    const count = counts.get(label) as number;
    const bestCount = counts.get(best) as number;
    if (count !== bestCount) {
      if (count > bestCount) best = label;
      continue;
    }
    if ((sums.get(label) as number) * bestCount > (sums.get(best) as number) * count) {
      best = label;
    }
  }
  return best;
}

function winnerOrder(policy: PortablePolicy) {
  return (left: Candidate, right: Candidate): number =>
    compareNumbers(left.action !== "keep" ? 0 : 1, right.action !== "keep" ? 0 : 1) ||
    compareNumbers(
      -(policy.taxonomy.specificity[left.canonicalLabel] ?? 1),
      -(policy.taxonomy.specificity[right.canonicalLabel] ?? 1),
    ) ||
    compareNumbers(-(left.end - left.start), -(right.end - right.start)) ||
    compareNumbers(-left.confidencePpm, -right.confidencePpm) ||
    compareNumbers(left.start, right.start) ||
    asciiCompare(left.canonicalLabel, right.canonicalLabel) ||
    asciiCompare(left.label, right.label) ||
    compareNumbers(left.inputs[0] as number, right.inputs[0] as number);
}

function best(members: readonly Candidate[], policy: PortablePolicy): Candidate {
  const order = winnerOrder(policy);
  let winner = members[0] as Candidate;
  for (const member of members.slice(1)) {
    if (order(member, winner) < 0) winner = member;
  }
  return winner;
}

function duplicateKey(candidate: Candidate): string {
  return `${candidate.start}:${candidate.end}:${candidate.canonicalLabel}`;
}

function collapseDuplicates(retained: readonly Candidate[], policy: PortablePolicy): Candidate[] {
  const groups = new Map<string, Candidate[]>();
  for (const item of retained) {
    const key = duplicateKey(item);
    const group = groups.get(key);
    if (group) group.push(item);
    else groups.set(key, [item]);
  }
  const unique: Candidate[] = [];
  for (const members of groups.values()) {
    const winner = best(members, policy);
    for (const member of members) {
      if (member !== winner) member.outcome = "duplicate";
    }
    unique.push(winner);
  }
  return unique.sort(candidateOrder);
}

function clusters(items: readonly Candidate[]): Candidate[][] {
  const result: Candidate[][] = [];
  let current: Candidate[] = [];
  let currentEnd = -1;
  for (const item of items) {
    if (current.length > 0 && item.start < currentEnd) {
      current.push(item);
      currentEnd = Math.max(currentEnd, item.end);
      continue;
    }
    if (current.length > 0) result.push(current);
    current = [item];
    currentEnd = item.end;
  }
  if (current.length > 0) result.push(current);
  return result;
}

function representative(cluster: readonly Candidate[], policy: PortablePolicy): Candidate {
  let members = [...cluster];
  if (policy.arbitrationMode === "high_recall_union") {
    const strongest = Math.max(...members.map((item) => ACTION_STRENGTH[item.action] ?? 0));
    members = members.filter((item) => (ACTION_STRENGTH[item.action] ?? 0) === strongest);
  }
  return best(members, policy);
}

function parseObject(text: string, code: string): Record<string, unknown> {
  let document: unknown;
  try {
    document = JSON.parse(text);
  } catch {
    throw new PortablePolicyError(code);
  }
  if (!isRecord(document)) {
    throw new PortablePolicyError(code);
  }
  return document;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOwn(record: Readonly<Record<string, unknown>>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function stringRecord(value: unknown): Record<string, string> | undefined {
  if (!isRecord(value)) return undefined;
  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== "string") return undefined;
    result[key] = item;
  }
  return result;
}

function stringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    return undefined;
  }
  return [...(value as string[])];
}

function utf8(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function concat(parts: readonly Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.byteLength, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  return out;
}

function toHex(bytes: Uint8Array): string {
  let hex = "";
  for (const byte of bytes) hex += byte.toString(16).padStart(2, "0");
  return hex;
}

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** Synchronous SHA-256 (FIPS 180-4); verified against node:crypto in tests. */
function sha256(data: Uint8Array): Uint8Array {
  const state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const bitLength = data.byteLength * 8;
  const padded = new Uint8Array((((data.byteLength + 9 + 63) >> 6) << 6));
  padded.set(data);
  padded[data.byteLength] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.byteLength - 8, Math.floor(bitLength / 0x1_0000_0000));
  view.setUint32(padded.byteLength - 4, bitLength >>> 0);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < padded.byteLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4);
    }
    for (let index = 16; index < 64; index += 1) {
      const w15 = words[index - 15] as number;
      const w2 = words[index - 2] as number;
      const s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >>> 3);
      const s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >>> 10);
      words[index] = ((words[index - 16] as number) + s0 + (words[index - 7] as number) + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = state as unknown as number[];
    for (let index = 0; index < 64; index += 1) {
      const s1 = rotr(e as number, 6) ^ rotr(e as number, 11) ^ rotr(e as number, 25);
      const choice = ((e as number) & (f as number)) ^ (~(e as number) & (g as number));
      const temp1 = ((h as number) + s1 + choice + (SHA256_K[index] as number) + (words[index] as number)) >>> 0;
      const s0 = rotr(a as number, 2) ^ rotr(a as number, 13) ^ rotr(a as number, 22);
      const majority = ((a as number) & (b as number)) ^ ((a as number) & (c as number)) ^ ((b as number) & (c as number));
      const temp2 = (s0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = ((d as number) + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    const updates = [a, b, c, d, e, f, g, h] as number[];
    for (let index = 0; index < 8; index += 1) {
      state[index] = ((state[index] as number) + (updates[index] as number)) >>> 0;
    }
  }
  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);
  for (let index = 0; index < 8; index += 1) {
    outView.setUint32(index * 4, state[index] as number);
  }
  return out;
}

function rotr(value: number, bits: number): number {
  return ((value >>> bits) | (value << (32 - bits))) >>> 0;
}

/** HMAC-SHA256 (RFC 2104). */
function hmacSha256(key: Uint8Array, message: Uint8Array): Uint8Array {
  const block = new Uint8Array(64);
  block.set(key.byteLength > 64 ? sha256(key) : key);
  const inner = new Uint8Array(64);
  const outer = new Uint8Array(64);
  for (let index = 0; index < 64; index += 1) {
    inner[index] = (block[index] as number) ^ 0x36;
    outer[index] = (block[index] as number) ^ 0x5c;
  }
  return sha256(concat([outer, sha256(concat([inner, message]))]));
}
