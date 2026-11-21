import bundled from "./crisis-resources.json" with { type: "json" };

export interface CrisisResource {
  readonly id: string;
  readonly label: string;
  readonly url: string;
  readonly source_url: string;
  readonly verified_on: string;
}

export interface CrisisNotice {
  readonly text: string;
  readonly mandatory: true;
  readonly matchedLocale: string | null;
  readonly resources: readonly CrisisResource[];
}

const genericNotice = "If someone is in immediate danger, contact local emergency services now.";

function localeKeys(tag: string): string[] {
  if (typeof tag !== "string" || !/^[a-zA-Z0-9_-]+$/.test(tag)) return [];
  const match = /^([a-z]{2,3})(?:-([a-z]{4}))?(?:-([a-z]{2}|[0-9]{3}))?$/.exec(
    tag.replaceAll("_", "-").toLowerCase(),
  );
  if (!match) return [];
  const [, language, rawScript, rawRegion] = match;
  const script = rawScript && rawScript[0]!.toUpperCase() + rawScript.slice(1);
  const region = rawRegion?.toUpperCase();
  const keys = [[language, script, region].filter(Boolean).join("-")];
  if (script && region) keys.push(language + "-" + region, language + "-" + script);
  if (language) keys.push(language);
  return keys;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function invalid(): never {
  throw new Error("invalid_crisis_table");
}

function link(value: unknown): string {
  if (typeof value !== "string" || value.length > 2048 ||
      !/^https:\/\//i.test(value) || /[^\x21-\x7e]|\\/.test(value)) return invalid();
  const parsed = new URL(value);
  if (!parsed.hostname || parsed.username || parsed.password || parsed.port === "0") return invalid();
  return value;
}

function resource(row: unknown): CrisisResource {
  if (!record(row) || Object.keys(row).sort().join() !== "id,label,source_url,url,verified_on") return invalid();
  const { id, label, verified_on } = row;
  if (typeof id !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(id) ||
      typeof label !== "string" || !label.trim() || Array.from(label).length > 160 ||
      /[\p{C}\p{Zl}\p{Zp}]/u.test(label) ||
      typeof verified_on !== "string" || !/^(?!0000)\d{4}-\d{2}-\d{2}$/.test(verified_on) ||
      new Date(verified_on + "T00:00:00Z").toISOString().slice(0, 10) !== verified_on) return invalid();
  return Object.freeze({ id, label, url: link(row.url), source_url: link(row.source_url), verified_on });
}

function table(value: unknown): Map<string, readonly CrisisResource[]> {
  if (!record(value) || Object.keys(value).sort().join() !== "locales,schema_version" ||
      value.schema_version !== 1 || !record(value.locales)) return invalid();
  const entries = new Map<string, readonly CrisisResource[]>();
  for (const [tag, rows] of Object.entries(value.locales)) {
    const [key] = localeKeys(tag);
    if (!key || entries.has(key) || !Array.isArray(rows) || !rows.length) return invalid();
    const resources = rows.map(resource);
    if (new Set(resources.map((r) => r.id)).size !== resources.length) return invalid();
    entries.set(key, Object.freeze(resources));
  }
  return entries;
}

/** Resolve host signals locally and return mandatory display data for a positive signal. */
export class CrisisRouting {
  readonly #entries: Map<string, readonly CrisisResource[]>;

  constructor(override?: unknown) {
    try {
      this.#entries = table(bundled);
      if (override !== undefined) {
        for (const [key, resources] of table(override)) this.#entries.set(key, resources);
      }
    } catch {
      throw new Error("invalid_crisis_table");
    }
  }

  route(locale: string, signal: boolean): CrisisNotice | null {
    if (typeof signal !== "boolean") throw new Error("invalid_crisis_signal");
    if (!signal) return null;
    const matchedLocale = localeKeys(locale).find((key) => this.#entries.has(key)) ?? null;
    return Object.freeze({
      text: genericNotice, mandatory: true, matchedLocale,
      resources: matchedLocale ? this.#entries.get(matchedLocale)! : Object.freeze([]),
    });
  }
}
