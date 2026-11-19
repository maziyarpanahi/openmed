// Privacy scans and the JavaScript-side network-denial guard.

import { scalarLength, scalarSlice } from "./canonical";

/** Shortest matched surface (in Unicode scalars) that the scans look for. */
export const MIN_SENTINEL_SCALARS = 2;

const DIGEST_VALUE = /^(sha256|hmac-sha256):[0-9a-f]{64}$/;
const HEX_RUN = /[0-9a-f]{12,}/g;

/**
 * Find string fields of bridge spans that contain a matched source surface.
 *
 * Spans are allowed to carry offsets, labels, digests and policy replacements,
 * never the matched substring itself. Returns the JSON paths that leaked.
 */
export function findSurfaceLeaks(
  spans: readonly unknown[],
  text: string,
  path = "$.spans",
): string[] {
  const surfaces = spanSurfaces(spans, text);
  if (surfaces.length === 0) {
    return [];
  }
  const leaks: string[] = [];
  spans.forEach((span, index) => {
    for (const [fieldPath, value] of collectStrings(span, `${path}[${index}]`)) {
      if (DIGEST_VALUE.test(value)) {
        continue;
      }
      // Hash-action replacements embed a truncated hex digest; short numeric
      // surfaces would otherwise match it by chance.
      const scanned = value.replace(HEX_RUN, "");
      if (surfaces.some((surface) => scanned.includes(surface))) {
        leaks.push(fieldPath);
      }
    }
  });
  return leaks;
}

/** Matched surfaces long enough to scan for without false positives. */
export function spanSurfaces(spans: readonly unknown[], text: string): string[] {
  const textLength = scalarLength(text);
  const surfaces = new Set<string>();
  for (const span of spans) {
    if (span === null || typeof span !== "object") {
      continue;
    }
    const { start, end } = span as { start?: unknown; end?: unknown };
    if (
      typeof start !== "number" ||
      typeof end !== "number" ||
      !Number.isInteger(start) ||
      !Number.isInteger(end) ||
      start < 0 ||
      end <= start ||
      end > textLength
    ) {
      continue;
    }
    const surface = scalarSlice(text, start, end).trim();
    if (scalarLength(surface) >= MIN_SENTINEL_SCALARS) {
      surfaces.add(surface);
    }
  }
  return [...surfaces];
}

/**
 * Return the paths of report strings that contain any sentinel value.
 *
 * Structural fields (`case_id`, `corpus_id`, `path`, codes and digests) come
 * from committed, non-PHI fixture metadata and are skipped.
 */
export function findReportLeaks(
  report: unknown,
  sentinels: readonly string[],
): string[] {
  const needles = sentinels.filter(
    (sentinel) => scalarLength(sentinel) >= MIN_SENTINEL_SCALARS,
  );
  const leaks: string[] = [];
  for (const [path, value] of collectStrings(report, "$")) {
    const key = path.slice(path.lastIndexOf(".") + 1).replace(/(\[\d+\])+$/, "");
    if (STRUCTURAL_KEYS.has(key) || DIGEST_VALUE.test(value)) {
      continue;
    }
    if (needles.some((needle) => value.includes(needle))) {
      leaks.push(path);
    }
  }
  return leaks;
}

const STRUCTURAL_KEYS = new Set([
  "case_id",
  "corpus_id",
  "path",
  "mismatched_paths",
  "failure_reasons",
  "code",
  "reason",
  "digest",
  "schema",
]);

function collectStrings(value: unknown, path: string): Array<[string, string]> {
  if (typeof value === "string") {
    return [[path, value]];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => collectStrings(item, `${path}[${index}]`));
  }
  if (value !== null && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(([key, item]) =>
      collectStrings(item, `${path}.${key}`),
    );
  }
  return [];
}

export interface EgressGuard {
  /** Number of blocked network attempts since the guard was installed. */
  attempts(): number;
  /** Put the original globals back. */
  restore(): void;
}

const GUARDED_GLOBALS = ["fetch", "XMLHttpRequest", "WebSocket", "EventSource"];

/**
 * Replace JavaScript network entry points with stubs that refuse and count.
 *
 * This proves that the bridge's JavaScript layer makes no request during the
 * run. Native egress is denied separately by the CI device (airplane mode on
 * the Android emulator) and by loading only pinned local artifacts.
 */
export function installEgressGuard(
  scope: Record<string, unknown> = globalThis as unknown as Record<string, unknown>,
): EgressGuard {
  let attempts = 0;
  const originals = new Map<string, PropertyDescriptor | undefined>();
  const refuse = () => {
    attempts += 1;
    throw new Error("network egress is blocked during the parity run");
  };

  for (const name of GUARDED_GLOBALS) {
    originals.set(name, Object.getOwnPropertyDescriptor(scope, name));
    const stub =
      name === "fetch"
        ? function blockedFetch() {
            attempts += 1;
            return Promise.reject(
              new Error("network egress is blocked during the parity run"),
            );
          }
        : function BlockedConstructor() {
            refuse();
          };
    Object.defineProperty(scope, name, {
      configurable: true,
      writable: true,
      value: stub,
    });
  }

  return {
    attempts: () => attempts,
    restore() {
      for (const [name, descriptor] of originals) {
        if (descriptor === undefined) {
          delete scope[name];
        } else {
          Object.defineProperty(scope, name, descriptor);
        }
      }
    },
  };
}
