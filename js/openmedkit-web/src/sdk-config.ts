// Portable SDK configuration (docs/configuration/sdk-config.md).
//
// One small JSON document that Python and every OpenMedKit port read with the
// same six keys, types, bounds and defaults
// (openmed/core/schemas/json/sdk_config.schema.json). Validation fails closed
// and reports value-free issues, in the order the shared vectors in
// tests/fixtures/parity/sdk_config_v1.json pin. The mapping helpers turn a
// validated document into the web package's existing option objects.
import type { ExtractPiiOptions, LoadOnnxModelOptions } from "./types";

export const SDK_CONFIG_SCHEMA_ID = "https://openmed.life/schemas/sdk_config.schema.json";
export const SDK_CONFIG_VERSION = 1;

/** The portable keys, in the order issues are reported. */
export const SDK_CONFIG_FIELDS = Object.freeze([
  "confidence_threshold",
  "local_only",
  "model_id",
  "model_revision",
  "onnx_variant",
  "policy_profile",
] as const);

export const SDK_CONFIG_ONNX_VARIANTS = Object.freeze(["auto", "int8", "fp16", "fp32"] as const);

/** Canonical portable policy profile names accepted by version 1. */
export const SDK_CONFIG_POLICY_PROFILES = Object.freeze([
  "australia_privacy_act",
  "canada_pipeda",
  "clinical_minimal_redaction",
  "gdpr_art9_health",
  "gdpr_pseudonymization",
  "hipaa_expert_review_assist",
  "hipaa_safe_harbor",
  "research_limited_dataset",
  "strict_no_leak",
] as const);

/** The closed set of issue codes. */
export const SDK_CONFIG_ISSUE_CODES = Object.freeze([
  "config_not_object",
  "invalid_format",
  "missing_dependency",
  "not_allowed_value",
  "out_of_range",
  "unknown_field",
  "wrong_type",
] as const);

export type SdkConfigField = (typeof SDK_CONFIG_FIELDS)[number];
export type SdkConfigOnnxVariant = (typeof SDK_CONFIG_ONNX_VARIANTS)[number];
export type SdkConfigPolicyProfile = (typeof SDK_CONFIG_POLICY_PROFILES)[number];
export type SdkConfigIssueCode = (typeof SDK_CONFIG_ISSUE_CODES)[number];

/** A resolved portable SDK configuration: every key present. */
export interface OpenMedSdkConfig {
  /** Minimum detection confidence in `[0, 1]`; default `0.5`. */
  readonly confidence_threshold: number;
  /** Cache-only, no-egress mode; default `true`. */
  readonly local_only: boolean;
  /** Hub repository id, or `null` for the kit default model. */
  readonly model_id: string | null;
  /** Pinned revision of `model_id`, or `null`. */
  readonly model_revision: string | null;
  /** ONNX weight variant; `auto` keeps the loader's default. */
  readonly onnx_variant: SdkConfigOnnxVariant;
  /** Portable policy profile for `loadPortablePolicy()`. */
  readonly policy_profile: SdkConfigPolicyProfile;
}

/** Values used for absent keys. `local_only` is on by default. */
export const SDK_CONFIG_DEFAULTS: Readonly<OpenMedSdkConfig> = Object.freeze({
  confidence_threshold: 0.5,
  local_only: true,
  model_id: null,
  model_revision: null,
  onnx_variant: "auto",
  policy_profile: "hipaa_safe_harbor",
});

/** One value-free validation problem. */
export interface SdkConfigIssue {
  readonly code: SdkConfigIssueCode;
  /** The key the issue concerns, or `null` for document-level issues. */
  readonly field: SdkConfigField | null;
}

/** Thrown by `parseSdkConfig()` with every issue; never carries input values. */
export class SdkConfigError extends Error {
  readonly code = "sdk_config_invalid";
  readonly issues: readonly SdkConfigIssue[];

  constructor(issues: readonly SdkConfigIssue[]) {
    const summary = issues
      .map((issue) => (issue.field === null ? issue.code : `${issue.field}:${issue.code}`))
      .join(", ");
    super(`invalid SDK configuration: ${summary}`);
    this.name = "SdkConfigError";
    this.issues = Object.freeze(issues.map((issue) => Object.freeze({ ...issue })));
  }
}

const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:\/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$/;
const MODEL_REVISION = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const FIELD_SET: ReadonlySet<string> = new Set(SDK_CONFIG_FIELDS);

function isDocument(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function fieldIssue(field: SdkConfigField, value: unknown): SdkConfigIssueCode | null {
  switch (field) {
    case "confidence_threshold":
      if (typeof value !== "number") {
        return "wrong_type";
      }
      return Number.isFinite(value) && value >= 0 && value <= 1 ? null : "out_of_range";
    case "local_only":
      return typeof value === "boolean" ? null : "wrong_type";
    case "model_id":
    case "model_revision": {
      if (value === null) {
        return null;
      }
      if (typeof value !== "string") {
        return "wrong_type";
      }
      const pattern = field === "model_id" ? MODEL_ID : MODEL_REVISION;
      return pattern.test(value) && !value.includes("..") ? null : "invalid_format";
    }
    case "onnx_variant":
    case "policy_profile": {
      if (typeof value !== "string") {
        return "wrong_type";
      }
      const allowed: readonly string[] =
        field === "onnx_variant" ? SDK_CONFIG_ONNX_VARIANTS : SDK_CONFIG_POLICY_PROFILES;
      return allowed.includes(value) ? null : "not_allowed_value";
    }
  }
}

/**
 * Return every issue in a portable SDK configuration document.
 *
 * Issues are ordered as Python's `validate_sdk_config()` orders them:
 * `config_not_object` alone when the value is not a plain object; otherwise
 * at most one `unknown_field`, then at most one issue per key in
 * `SDK_CONFIG_FIELDS` order.
 *
 * Args:
 *   value: A parsed JSON value (use `JSON.parse` for text).
 *
 * Returns:
 *   The issues; empty when the document is valid.
 */
export function validateSdkConfig(value: unknown): readonly SdkConfigIssue[] {
  if (!isDocument(value)) {
    return [{ code: "config_not_object", field: null }];
  }
  const issues: SdkConfigIssue[] = [];
  if (Object.keys(value).some((key) => !FIELD_SET.has(key))) {
    issues.push({ code: "unknown_field", field: null });
  }
  for (const field of SDK_CONFIG_FIELDS) {
    if (!Object.hasOwn(value, field)) {
      continue;
    }
    let code = fieldIssue(field, value[field]);
    const modelId = Object.hasOwn(value, "model_id") ? value.model_id : null;
    if (code === null && field === "model_revision" && value[field] !== null && modelId === null) {
      // A revision only pins something when a model id is set.
      code = "missing_dependency";
    }
    if (code !== null) {
      issues.push({ code, field });
    }
  }
  return issues;
}

/**
 * Validate a portable SDK configuration and fill absent keys with defaults.
 *
 * Args:
 *   value: A parsed JSON value (use `JSON.parse` for text).
 *
 * Returns:
 *   A frozen configuration with all six keys.
 *
 * Throws:
 *   SdkConfigError: With every issue when the document is invalid.
 */
export function parseSdkConfig(value: unknown): Readonly<OpenMedSdkConfig> {
  const issues = validateSdkConfig(value);
  if (issues.length > 0) {
    throw new SdkConfigError(issues);
  }
  const document = value as Partial<OpenMedSdkConfig>;
  const pick = <K extends SdkConfigField>(field: K): OpenMedSdkConfig[K] =>
    Object.hasOwn(document, field)
      ? (document[field] as OpenMedSdkConfig[K])
      : SDK_CONFIG_DEFAULTS[field];
  return Object.freeze({
    confidence_threshold: pick("confidence_threshold"),
    local_only: pick("local_only"),
    model_id: pick("model_id"),
    model_revision: pick("model_revision"),
    onnx_variant: pick("onnx_variant"),
    policy_profile: pick("policy_profile"),
  });
}

/**
 * Map a portable configuration onto model loader options.
 *
 * `local_only: true` becomes `localFilesOnly: true, allowRemoteModels: false`
 * (cache or local files only, no Hub download); `false` opts in to remote
 * models. `model_revision` becomes `revision`, and an `onnx_variant` other than
 * `auto` becomes the ONNX `variant`. The configuration is revalidated first.
 *
 * Args:
 *   config: A configuration from `parseSdkConfig()`.
 *
 * Returns:
 *   Options for `loadOnnxModel()` or `loadTokenClassificationPipeline()`.
 *
 * Throws:
 *   SdkConfigError: When `config` is not a valid configuration.
 */
export function toLoadModelOptions(config: OpenMedSdkConfig): LoadOnnxModelOptions {
  const resolved = parseSdkConfig(config);
  return {
    localFilesOnly: resolved.local_only,
    allowRemoteModels: !resolved.local_only,
    ...(resolved.model_revision === null ? {} : { revision: resolved.model_revision }),
    ...(resolved.onnx_variant === "auto"
      ? {}
      : { variant: resolved.onnx_variant }),
  };
}

/**
 * Map a portable configuration onto `extractPii()` and `deidentify()` options.
 *
 * `confidence_threshold` becomes `threshold`, a non-null `model_id` becomes
 * `model` (null keeps the package default), and `loaderOptions` come from
 * `toLoadModelOptions()`. `policy_profile` is not an extraction option: pass
 * it to `loadPortablePolicy()`.
 *
 * Args:
 *   config: A configuration from `parseSdkConfig()`.
 *
 * Returns:
 *   Extraction options.
 *
 * Throws:
 *   SdkConfigError: When `config` is not a valid configuration.
 */
export function toExtractPiiOptions(config: OpenMedSdkConfig): ExtractPiiOptions {
  const resolved = parseSdkConfig(config);
  return {
    threshold: resolved.confidence_threshold,
    ...(resolved.model_id === null ? {} : { model: resolved.model_id }),
    loaderOptions: toLoadModelOptions(resolved),
  };
}
