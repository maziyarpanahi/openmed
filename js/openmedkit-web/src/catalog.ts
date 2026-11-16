/**
 * Typed ONNX model catalog for the `openmed/catalog` subpath.
 *
 * The bundled rows live in `catalog.generated.ts`, which
 * `scripts/web/build_web_catalog.py` derives from the canonical `models.jsonl`
 * manifest: permissively licensed models with at least one ONNX format, using
 * the Android catalog row fields. Nothing here performs I/O; listing or
 * looking up a model never touches the network.
 *
 * Filter semantics match the Android `ModelCatalog.filter()`:
 * - `format`: case-insensitive equality with any of the row's formats.
 * - `tier` and `license`: case-insensitive equality; a row whose value is
 *   `null` never matches a filter on that field.
 * - `maxParamCount`: `param_count <= maxParamCount`; a row with a `null`
 *   parameter count never matches.
 * - `language`: case-insensitive membership in the row's languages.
 * - Every given filter must match (logical AND). No filter returns all rows.
 *
 * `getModel()` matches `repo_id` exactly, as the Android `byRepoId()` does.
 */

import {
  MODEL_CATALOG_DIGEST,
  MODEL_CATALOG_FORMAT,
  MODEL_CATALOG_ROWS,
} from "./catalog.generated";

export { MODEL_CATALOG_DIGEST, MODEL_CATALOG_FORMAT };

/** Row fields, in order. Identical to the Android catalog JSONL fields. */
export const MODEL_CATALOG_FIELDS = Object.freeze([
  "repo_id",
  "formats",
  "tier",
  "param_count",
  "languages",
  "license",
  "reproducibility_hash",
] as const);

/** One catalog row. Field names match the Android catalog JSONL. */
export interface ModelCatalogEntry {
  /** Hugging Face repository id, for example `OpenMed/<model>-onnx-android`. */
  readonly repo_id: string;
  /** ONNX format tokens published by the repository (for example `onnx`). */
  readonly formats: readonly string[];
  /** Size tier from the manifest, or `null` when the manifest has none. */
  readonly tier: string | null;
  /** Parameter count, or `null` when the manifest does not record it. */
  readonly param_count: number | null;
  /** Language codes the model covers. */
  readonly languages: readonly string[];
  /**
   * License identifier as written in the manifest. Bundled rows always carry a
   * permissive license; rows passed to {@link createModelCatalog} may be `null`.
   */
  readonly license: string | null;
  /** `sha256:` reproducibility digest from the manifest. */
  readonly reproducibility_hash: string;
}

/** Filter for {@link listModels}. Omitted fields do not constrain the result. */
export interface ModelCatalogFilter {
  /** Keep rows that publish this format (case-insensitive). */
  readonly format?: string;
  /** Keep rows in this tier (case-insensitive). */
  readonly tier?: string;
  /** Keep rows whose known parameter count is at most this value. */
  readonly maxParamCount?: number;
  /** Keep rows that cover this language code (case-insensitive). */
  readonly language?: string;
  /** Keep rows under this license (case-insensitive). */
  readonly license?: string;
}

/** An immutable, queryable set of catalog rows. */
export interface ModelCatalog {
  /** Frozen rows, in the order they were supplied. */
  readonly entries: readonly ModelCatalogEntry[];
  /** Return the rows that match every given filter field. */
  listModels(filter?: ModelCatalogFilter): ModelCatalogEntry[];
  /** Return the row whose `repo_id` equals `repoId`, if any. */
  getModel(repoId: string): ModelCatalogEntry | undefined;
}

/** Stable reasons a catalog row set or filter is refused. */
export type ModelCatalogErrorCode =
  | "invalid_json"
  | "invalid_row"
  | "duplicate_repo_id"
  | "invalid_filter";

/**
 * Raised for a malformed catalog row or filter. The message names the code
 * and the row index or filter field only, never row contents.
 */
export class ModelCatalogError extends Error {
  readonly code: ModelCatalogErrorCode;
  readonly fieldPath: string;

  constructor(code: ModelCatalogErrorCode, fieldPath = "$") {
    super(`${code} at ${fieldPath}`);
    this.name = "ModelCatalogError";
    this.code = code;
    this.fieldPath = fieldPath;
  }
}

const FILTER_KEYS = new Set<string>([
  "format",
  "tier",
  "maxParamCount",
  "language",
  "license",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringList(value: unknown, path: string): readonly string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new ModelCatalogError("invalid_row", path);
  }
  return Object.freeze([...(value as string[])]);
}

function nullableString(value: unknown, path: string): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "string") {
    throw new ModelCatalogError("invalid_row", path);
  }
  return value;
}

function requiredString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelCatalogError("invalid_row", path);
  }
  return value;
}

function nullableCount(value: unknown, path: string): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new ModelCatalogError("invalid_row", path);
  }
  return value;
}

function normalizeEntry(row: unknown, index: number): ModelCatalogEntry {
  const base = `$[${index}]`;
  if (!isRecord(row)) {
    throw new ModelCatalogError("invalid_row", base);
  }
  return Object.freeze({
    repo_id: requiredString(row.repo_id, `${base}.repo_id`),
    formats: stringList(row.formats ?? [], `${base}.formats`),
    tier: nullableString(row.tier, `${base}.tier`),
    param_count: nullableCount(row.param_count, `${base}.param_count`),
    languages: stringList(row.languages ?? [], `${base}.languages`),
    license: nullableString(row.license, `${base}.license`),
    reproducibility_hash: requiredString(
      row.reproducibility_hash,
      `${base}.reproducibility_hash`,
    ),
  });
}

function optionalToken(
  filter: Record<string, unknown>,
  key: string,
): string | undefined {
  const value = filter[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new ModelCatalogError("invalid_filter", key);
  }
  return value.toLowerCase();
}

interface NormalizedFilter {
  readonly format: string | undefined;
  readonly tier: string | undefined;
  readonly maxParamCount: number | undefined;
  readonly language: string | undefined;
  readonly license: string | undefined;
}

function normalizeFilter(filter: unknown): NormalizedFilter {
  if (filter === undefined) {
    return {
      format: undefined,
      tier: undefined,
      maxParamCount: undefined,
      language: undefined,
      license: undefined,
    };
  }
  if (!isRecord(filter)) {
    throw new ModelCatalogError("invalid_filter");
  }
  for (const key of Object.keys(filter)) {
    if (!FILTER_KEYS.has(key)) {
      throw new ModelCatalogError("invalid_filter", key);
    }
  }
  const maxParamCount = filter.maxParamCount;
  if (
    maxParamCount !== undefined &&
    (typeof maxParamCount !== "number" ||
      Number.isNaN(maxParamCount) ||
      maxParamCount < 0)
  ) {
    throw new ModelCatalogError("invalid_filter", "maxParamCount");
  }
  return {
    format: optionalToken(filter, "format"),
    tier: optionalToken(filter, "tier"),
    maxParamCount: maxParamCount as number | undefined,
    language: optionalToken(filter, "language"),
    license: optionalToken(filter, "license"),
  };
}

function matchesToken(actual: string | null, expected: string | undefined) {
  return expected === undefined || actual?.toLowerCase() === expected;
}

function containsToken(values: readonly string[], expected: string | undefined) {
  return (
    expected === undefined ||
    values.some((value) => value.toLowerCase() === expected)
  );
}

function matches(entry: ModelCatalogEntry, filter: NormalizedFilter): boolean {
  return (
    containsToken(entry.formats, filter.format) &&
    matchesToken(entry.tier, filter.tier) &&
    (filter.maxParamCount === undefined ||
      (entry.param_count !== null &&
        entry.param_count <= filter.maxParamCount)) &&
    containsToken(entry.languages, filter.language) &&
    matchesToken(entry.license, filter.license)
  );
}

/**
 * Validate catalog rows and return a queryable catalog over them.
 *
 * Use this for rows that did not come from the bundled catalog, such as a
 * catalog JSONL file shipped next to an application or a synthetic test set.
 *
 * Rows are read like the Android catalog reads its JSONL asset: `repo_id`
 * and `reproducibility_hash` are required; `formats` and `languages` default
 * to empty lists; `tier`, `param_count` and `license` default to `null`;
 * fields outside {@link MODEL_CATALOG_FIELDS} are ignored and not copied.
 *
 * @param rows Catalog rows.
 * @returns A catalog whose rows are frozen copies of the input.
 * @throws ModelCatalogError `invalid_row` for a malformed row,
 *   `duplicate_repo_id` when two rows share a `repo_id`.
 */
export function createModelCatalog(rows: Iterable<unknown>): ModelCatalog {
  const entries: ModelCatalogEntry[] = [];
  const byRepoId = new Map<string, ModelCatalogEntry>();
  let index = 0;
  for (const row of rows) {
    const entry = normalizeEntry(row, index);
    if (byRepoId.has(entry.repo_id)) {
      throw new ModelCatalogError("duplicate_repo_id", `$[${index}].repo_id`);
    }
    byRepoId.set(entry.repo_id, entry);
    entries.push(entry);
    index += 1;
  }
  const frozen = Object.freeze(entries);
  return Object.freeze({
    entries: frozen,
    listModels(filter?: ModelCatalogFilter): ModelCatalogEntry[] {
      const normalized = normalizeFilter(filter);
      return frozen.filter((entry) => matches(entry, normalized));
    },
    getModel(repoId: string): ModelCatalogEntry | undefined {
      return typeof repoId === "string" ? byRepoId.get(repoId) : undefined;
    },
  });
}

/**
 * Parse catalog JSONL (one row per line, the Android asset format) into a
 * catalog. Blank lines are ignored.
 *
 * @param text JSONL text.
 * @returns A catalog over the parsed rows.
 * @throws ModelCatalogError `invalid_json` naming the 1-based line, or any
 *   error {@link createModelCatalog} raises.
 */
export function parseModelCatalogJsonl(text: string): ModelCatalog {
  const rows: unknown[] = [];
  const lines = text.split(/\r?\n/);
  for (const [index, line] of lines.entries()) {
    const trimmed = line.trim();
    if (trimmed.length === 0) {
      continue;
    }
    try {
      rows.push(JSON.parse(trimmed));
    } catch {
      throw new ModelCatalogError("invalid_json", `line ${index + 1}`);
    }
  }
  return createModelCatalog(rows);
}

let bundledCatalog: ModelCatalog | undefined;

/** Return the catalog over the rows bundled with this package. */
export function getModelCatalog(): ModelCatalog {
  bundledCatalog ??= createModelCatalog(MODEL_CATALOG_ROWS);
  return bundledCatalog;
}

/**
 * List bundled ONNX models that match every given filter field.
 *
 * @example
 * ```ts
 * import { listModels } from "openmed/catalog";
 *
 * const spanish = listModels({ language: "es", maxParamCount: 100_000_000 });
 * ```
 *
 * @param filter Optional format, tier, maximum parameter count, language and
 *   license constraints.
 * @returns Matching rows, sorted by case-folded repo id.
 * @throws ModelCatalogError `invalid_filter` for an unknown filter field, an
 *   empty token, or a negative or non-numeric `maxParamCount`.
 */
export function listModels(filter?: ModelCatalogFilter): ModelCatalogEntry[] {
  return getModelCatalog().listModels(filter);
}

/**
 * Look up one bundled model by its exact repository id.
 *
 * @param repoId Hugging Face repository id.
 * @returns The catalog row, or `undefined` when the id is not in the catalog.
 */
export function getModel(repoId: string): ModelCatalogEntry | undefined {
  return getModelCatalog().getModel(repoId);
}
