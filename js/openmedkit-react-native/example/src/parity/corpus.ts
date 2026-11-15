// Parity corpus manifest: which shared synthetic vectors the harness replays.
//
// Each corpus is a committed JSON file under tests/fixtures/parity/. The
// manifest pins its canonical digest and says where the cases, ids, texts and
// optional policy names live, so newer vector files can be added by declaring
// selectors instead of writing a new loader.

import { canonicalDigest } from "./canonical";

export const CORPUS_MANIFEST_SCHEMA = "openmed.rn.parity.corpus_manifest";

export type ParityOperation = "analyzeText" | "extractPii" | "deidentify";

export const PARITY_OPERATIONS: readonly ParityOperation[] = [
  "analyzeText",
  "extractPii",
  "deidentify",
];

/** Dotted paths into one corpus file. */
export interface CorpusSelector {
  /** Path to the case array, for example `cases` or `vectors`. */
  cases: string;
  /** Path inside each case to its stable, non-PHI identifier. */
  id: string;
  /** Path inside each case to its synthetic input text. */
  text: string;
  /** Optional path inside each case to a policy profile name. */
  policy?: string;
}

export interface CorpusEntry {
  id: string;
  /** Repository-relative path of the committed vector file. */
  path: string;
  /** `sha256:` digest of the file's canonical JSON. */
  canonical_sha256: string;
  select: CorpusSelector;
  /** Top-level `offset_unit` the file must declare, when it declares one. */
  offset_unit?: "unicode_scalar";
}

/** A vector file that is expected to land later and how to select it. */
export interface PendingCorpusEntry {
  id: string;
  path: string;
  issue: number;
  select: CorpusSelector;
}

export interface CorpusDefaults {
  policy: string;
  confidence_threshold: number;
  use_smart_merging: boolean;
  operations: ParityOperation[];
}

export interface CorpusManifest {
  schema: typeof CORPUS_MANIFEST_SCHEMA;
  version: 1;
  synthetic_only: true;
  defaults: CorpusDefaults;
  tolerances: { score_abs: number };
  corpora: CorpusEntry[];
  pending: PendingCorpusEntry[];
}

export interface ParityCase {
  corpusId: string;
  caseId: string;
  text: string;
  policy: string;
}

export interface LoadedCorpus {
  id: string;
  digest: string;
  cases: ParityCase[];
}

export type CorpusErrorCode =
  | "corpus_manifest_invalid"
  | "corpus_missing"
  | "corpus_digest_mismatch"
  | "corpus_offset_unit_mismatch"
  | "corpus_selector_invalid"
  | "corpus_case_invalid"
  | "corpus_case_duplicate"
  | "corpus_empty";

/** Value-free corpus failure; `field` names a manifest field or corpus id. */
export class CorpusError extends Error {
  readonly code: CorpusErrorCode;
  readonly field: string;

  constructor(code: CorpusErrorCode, field: string) {
    super(`${code}: ${field}`);
    this.name = "CorpusError";
    this.code = code;
    this.field = field;
  }
}

/** Validate the parsed manifest and return it typed. */
export function parseCorpusManifest(value: unknown): CorpusManifest {
  const root = requireRecord(value, "$");
  if (root.schema !== CORPUS_MANIFEST_SCHEMA || root.version !== 1) {
    throw new CorpusError("corpus_manifest_invalid", "$.schema");
  }
  if (root.synthetic_only !== true) {
    throw new CorpusError("corpus_manifest_invalid", "$.synthetic_only");
  }
  const defaults = requireRecord(root.defaults, "$.defaults");
  const operations = defaults.operations;
  if (
    !Array.isArray(operations) ||
    operations.length === 0 ||
    !operations.every((op) => PARITY_OPERATIONS.includes(op as ParityOperation))
  ) {
    throw new CorpusError("corpus_manifest_invalid", "$.defaults.operations");
  }
  if (
    typeof defaults.policy !== "string" ||
    typeof defaults.confidence_threshold !== "number" ||
    typeof defaults.use_smart_merging !== "boolean"
  ) {
    throw new CorpusError("corpus_manifest_invalid", "$.defaults");
  }
  const tolerances = requireRecord(root.tolerances, "$.tolerances");
  if (
    typeof tolerances.score_abs !== "number" ||
    !(tolerances.score_abs >= 0 && tolerances.score_abs < 0.01)
  ) {
    throw new CorpusError("corpus_manifest_invalid", "$.tolerances.score_abs");
  }
  if (!Array.isArray(root.corpora) || root.corpora.length === 0) {
    throw new CorpusError("corpus_manifest_invalid", "$.corpora");
  }
  const seen = new Set<string>();
  root.corpora.forEach((entry, index) => {
    const record = requireRecord(entry, `$.corpora[${index}]`);
    requireString(record.id, `$.corpora[${index}].id`);
    requireString(record.path, `$.corpora[${index}].path`);
    if (
      typeof record.canonical_sha256 !== "string" ||
      !/^sha256:[0-9a-f]{64}$/.test(record.canonical_sha256)
    ) {
      throw new CorpusError(
        "corpus_manifest_invalid",
        `$.corpora[${index}].canonical_sha256`,
      );
    }
    requireSelector(record.select, `$.corpora[${index}].select`);
    if (seen.has(record.id as string)) {
      throw new CorpusError("corpus_manifest_invalid", `$.corpora[${index}].id`);
    }
    seen.add(record.id as string);
  });
  if (!Array.isArray(root.pending)) {
    throw new CorpusError("corpus_manifest_invalid", "$.pending");
  }
  root.pending.forEach((entry, index) => {
    const record = requireRecord(entry, `$.pending[${index}]`);
    requireString(record.id, `$.pending[${index}].id`);
    requireString(record.path, `$.pending[${index}].path`);
    if (typeof record.issue !== "number") {
      throw new CorpusError("corpus_manifest_invalid", `$.pending[${index}].issue`);
    }
    requireSelector(record.select, `$.pending[${index}].select`);
  });
  return root as unknown as CorpusManifest;
}

/** Extract parity cases from one parsed vector file using its selector. */
export function extractCases(
  corpusId: string,
  select: CorpusSelector,
  json: unknown,
  defaults: Pick<CorpusDefaults, "policy">,
): ParityCase[] {
  const rawCases = readPath(json, select.cases);
  if (!Array.isArray(rawCases)) {
    throw new CorpusError("corpus_selector_invalid", `${corpusId}.cases`);
  }
  const ids = new Set<string>();
  const cases = rawCases.map((rawCase, index) => {
    const caseId = readPath(rawCase, select.id);
    const text = readPath(rawCase, select.text);
    if (typeof caseId !== "string" || caseId.length === 0) {
      throw new CorpusError("corpus_case_invalid", `${corpusId}[${index}].id`);
    }
    if (typeof text !== "string" || text.length === 0) {
      throw new CorpusError("corpus_case_invalid", `${corpusId}:${caseId}.text`);
    }
    if (ids.has(caseId)) {
      throw new CorpusError("corpus_case_duplicate", `${corpusId}:${caseId}`);
    }
    ids.add(caseId);
    const policy =
      select.policy === undefined ? undefined : readPath(rawCase, select.policy);
    if (policy !== undefined && typeof policy !== "string") {
      throw new CorpusError("corpus_case_invalid", `${corpusId}:${caseId}.policy`);
    }
    return {
      corpusId,
      caseId,
      text,
      policy: policy ?? defaults.policy,
    };
  });
  if (cases.length === 0) {
    throw new CorpusError("corpus_empty", corpusId);
  }
  return cases;
}

/**
 * Resolve every manifest corpus from a registry of parsed JSON files.
 *
 * Fails closed: a missing file, a digest drift, or an offset-unit change
 * stops the run instead of silently replaying a different corpus.
 */
export function loadCorpora(
  manifest: CorpusManifest,
  registry: Readonly<Record<string, unknown>>,
): LoadedCorpus[] {
  return manifest.corpora.map((entry) => {
    if (!Object.prototype.hasOwnProperty.call(registry, entry.id)) {
      throw new CorpusError("corpus_missing", entry.id);
    }
    const json = registry[entry.id];
    const digest = canonicalDigest(json);
    if (digest !== entry.canonical_sha256) {
      throw new CorpusError("corpus_digest_mismatch", entry.id);
    }
    if (entry.offset_unit !== undefined) {
      const declared = readPath(json, "offset_unit");
      if (declared !== entry.offset_unit) {
        throw new CorpusError("corpus_offset_unit_mismatch", entry.id);
      }
    }
    return {
      id: entry.id,
      digest,
      cases: extractCases(entry.id, entry.select, json, manifest.defaults),
    };
  });
}

/** Read a dotted path such as `engine.policy`; returns undefined when absent. */
export function readPath(value: unknown, path: string): unknown {
  let current: unknown = value;
  for (const segment of path.split(".")) {
    if (current === null || typeof current !== "object" || Array.isArray(current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[segment];
  }
  return current;
}

function requireSelector(value: unknown, field: string): void {
  const record = requireRecord(value, field);
  for (const key of ["cases", "id", "text"] as const) {
    if (typeof record[key] !== "string" || !SELECTOR_PATH.test(record[key])) {
      throw new CorpusError("corpus_manifest_invalid", `${field}.${key}`);
    }
  }
  if (
    record.policy !== undefined &&
    (typeof record.policy !== "string" || !SELECTOR_PATH.test(record.policy))
  ) {
    throw new CorpusError("corpus_manifest_invalid", `${field}.policy`);
  }
}

const SELECTOR_PATH = /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$/;

function requireRecord(value: unknown, field: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new CorpusError("corpus_manifest_invalid", field);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, field: string): void {
  if (typeof value !== "string" || value.length === 0) {
    throw new CorpusError("corpus_manifest_invalid", field);
  }
}
