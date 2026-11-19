import type { OpenMedDeidentifyResult, OpenMedSpan } from "./index";
import { PORTABLE_TAXONOMY_LABELS } from "./portable-taxonomy-labels";

/**
 * React Native projection of the OpenMedKit kit-parity result profile
 * (`openmed.kit_parity_result` v1).
 *
 * Native Swift and Kotlin modules already report half-open Unicode scalar
 * offsets, so the bridge projects spans without index conversion. The shared
 * golden vectors live in `tests/fixtures/parity/kit_parity_vectors.json`.
 */
export const KIT_PARITY_PROFILE = "openmed.kit_parity_result" as const;
export const KIT_PARITY_PROFILE_VERSION = 1 as const;

export const KIT_PARITY_ACTIONS = [
  "hash",
  "keep",
  "mask",
  "redact",
  "remove",
  "replace",
] as const;

/**
 * Canonical labels a kit result may carry: the portable policy taxonomy,
 * generated from `openmed/core/labels.py` by
 * `scripts/parity/generate_portable_policies.py`.
 */
export const KIT_PARITY_CANONICAL_LABELS: readonly string[] = PORTABLE_TAXONOMY_LABELS;

export interface KitParitySpan {
  label: string;
  canonical_label: string;
  action: string;
  start: number;
  end: number;
  confidence_ppm?: number;
  replacement?: string;
}

export interface KitParityResult {
  deidentified_text: string;
  spans: KitParitySpan[];
}

/** Value-free bridge contract violation: a reason code and a field path. */
export class OpenMedKitParityError extends Error {
  readonly code: string;
  readonly fieldPath: string;

  constructor(code: string, fieldPath = "$") {
    super(`${code} at ${fieldPath}`);
    this.name = "OpenMedKitParityError";
    this.code = code;
    this.fieldPath = fieldPath;
  }
}

const LABEL_PATTERN = /^[A-Za-z][A-Za-z0-9_:-]{0,79}$/;
const ACTION_SET = new Set<string>(KIT_PARITY_ACTIONS);
const LABEL_SET = new Set<string>(KIT_PARITY_CANONICAL_LABELS);
const CONFIDENCE_SCALE = 1_000_000;

/**
 * Project a bridge de-identification result into the kit-parity profile.
 *
 * `entity_type` becomes `label`, a `null` score omits `confidence_ppm`, and
 * `keep` spans omit `replacement`. Labels, actions, and offsets outside the
 * profile fail with {@link OpenMedKitParityError}.
 */
export function toKitParityResult(result: OpenMedDeidentifyResult): KitParityResult {
  const spans = [...result.spans]
    .sort((left, right) => left.start - right.start || left.end - right.end)
    .map((span, index) => projectSpan(span, `$.spans[${index}]`));
  const projected = { deidentified_text: result.deidentifiedText, spans };
  validateKitParityResult(projected);
  return projected;
}

/** Return the canonical JSON text of a validated kit-parity result. */
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
    `,"offset_unit":"unicode_scalar","profile":"${KIT_PARITY_PROFILE}"` +
    `,"profile_version":${KIT_PARITY_PROFILE_VERSION}` +
    `,"spans":[${spans.join(",")}]}`
  );
}

function projectSpan(span: OpenMedSpan, path: string): KitParitySpan {
  const projected: KitParitySpan = {
    label: span.entity_type,
    canonical_label: span.canonical_label,
    action: span.action,
    start: span.start,
    end: span.end,
  };
  if (span.score !== null) {
    if (!Number.isFinite(span.score) || span.score < 0 || span.score > 1) {
      throw new OpenMedKitParityError("confidence_invalid", `${path}.confidence_ppm`);
    }
    projected.confidence_ppm = Math.floor(span.score * CONFIDENCE_SCALE + 0.5);
  }
  if (span.action !== "keep") {
    if (span.replacement === null) {
      throw new OpenMedKitParityError("field_missing", `${path}.replacement`);
    }
    projected.replacement = span.replacement;
  }
  return projected;
}

function validateKitParityResult(result: KitParityResult): void {
  if (typeof result.deidentified_text !== "string") {
    throw new OpenMedKitParityError("field_type_invalid", "$.deidentified_text");
  }
  let previous: KitParitySpan | undefined;
  result.spans.forEach((span, index) => {
    const path = `$.spans[${index}]`;
    if (!LABEL_PATTERN.test(span.label)) {
      throw new OpenMedKitParityError("label_invalid", `${path}.label`);
    }
    if (!LABEL_SET.has(span.canonical_label)) {
      throw new OpenMedKitParityError("label_unknown", `${path}.canonical_label`);
    }
    if (!ACTION_SET.has(span.action)) {
      throw new OpenMedKitParityError("action_unknown", `${path}.action`);
    }
    if (!Number.isSafeInteger(span.start)) {
      throw new OpenMedKitParityError("field_type_invalid", `${path}.start`);
    }
    if (span.start < 0) {
      throw new OpenMedKitParityError("offset_invalid", `${path}.start`);
    }
    if (!Number.isSafeInteger(span.end)) {
      throw new OpenMedKitParityError("field_type_invalid", `${path}.end`);
    }
    if (span.end <= span.start) {
      throw new OpenMedKitParityError("offset_invalid", `${path}.end`);
    }
    if (
      span.confidence_ppm !== undefined &&
      (!Number.isSafeInteger(span.confidence_ppm) ||
        span.confidence_ppm < 0 ||
        span.confidence_ppm > CONFIDENCE_SCALE)
    ) {
      throw new OpenMedKitParityError("confidence_invalid", `${path}.confidence_ppm`);
    }
    if (span.action === "keep" && span.replacement !== undefined) {
      throw new OpenMedKitParityError("replacement_unexpected", `${path}.replacement`);
    }
    if (span.action !== "keep" && typeof span.replacement !== "string") {
      throw new OpenMedKitParityError("field_missing", `${path}.replacement`);
    }
    if (previous !== undefined) {
      if (
        span.start < previous.start ||
        (span.start === previous.start && span.end <= previous.end)
      ) {
        throw new OpenMedKitParityError("span_order_invalid", path);
      }
      if (span.start < previous.end) {
        throw new OpenMedKitParityError("span_overlap", path);
      }
    }
    previous = span;
  });
}
