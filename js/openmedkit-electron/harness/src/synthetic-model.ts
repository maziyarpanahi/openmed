/**
 * Deterministic synthetic token classifier for the packaged harness.
 *
 * The qualification bundles are the shared portable-bundle golden vectors,
 * whose model files are placeholders, so no real network runs here. This
 * tagger stands in for one: it emits each shared reference case's span at the
 * contract offsets, tags "Patient <Given> <Family>" names by rule, and honours
 * a few control markers the harness uses to order and fault requests. The
 * same function runs in the utility process (packaged path) and in the main
 * process (direct reference path), so any difference between the two results
 * comes from the Electron boundary, not from the model.
 */
import type {
  OpenMedDeidentifyResult,
  TokenClassificationEntity,
  TokenClassificationPipeline,
} from "openmed";

import npmGolden from "../../../../tests/web/fixtures/npm_deidentify_golden.json";
import offsetContract from "../../../../tests/fixtures/parity/offset_contract.json";

export const SLOW_MARKER = "[[synthetic-slow]]";
export const CRASH_MARKER = "[[synthetic-crash]]";
export const FAIL_MARKER = "[[synthetic-fail]]";
export const NETWORK_PROBE_MARKER = "[[synthetic-network-probe]]";
export const SLOW_DELAY_MS = 750;

export interface OffsetContractCase {
  id: string;
  category: string;
  text: string;
  expected_start: number;
  expected_end: number;
  replacement: string;
  expected_redacted: string;
}

export const OFFSET_CASES = (offsetContract as { cases: OffsetContractCase[] }).cases;
export const NPM_GOLDEN = npmGolden as unknown as OpenMedDeidentifyResult;

/** Convert a Unicode-scalar offset into a JavaScript (UTF-16) index. */
export function scalarToUtf16(text: string, scalarOffset: number): number {
  return Array.from(text).slice(0, scalarOffset).join("").length;
}

/** Convert a JavaScript (UTF-16) index into a Unicode-scalar offset. */
export function utf16ToScalar(text: string, utf16Offset: number): number {
  return Array.from(text.slice(0, utf16Offset)).length;
}

const REFERENCE = new Map<string, TokenClassificationEntity[]>();
for (const item of OFFSET_CASES) {
  REFERENCE.set(item.text, [
    {
      entity: "B-NAME",
      score: 0.99,
      index: 1,
      start: scalarToUtf16(item.text, item.expected_start),
      end: scalarToUtf16(item.text, item.expected_end),
    },
  ]);
}
REFERENCE.set(NPM_GOLDEN.text, goldenTokens(NPM_GOLDEN.text));

function goldenTokens(text: string): TokenClassificationEntity[] {
  const token = (entity: string, surface: string, score: number, index: number) => {
    const start = text.indexOf(surface);
    return { entity, score, index, start, end: start + surface.length };
  };
  return [
    token("B-NAME", "Alice", 0.99, 1),
    token("E-NAME", "Nguyen", 0.97, 2),
    token("S-DATE_OF_BIRTH", "1979-04-12", 0.96, 3),
    token("S-EMAIL", "alice@example.org", 0.98, 4),
  ];
}

function ruleTokens(text: string): TokenClassificationEntity[] {
  const match = /Patient (\p{Lu}\p{Ll}+) (\p{Lu}\p{Ll}+)/u.exec(text);
  if (match === null || match[1] === undefined || match[2] === undefined) {
    return [];
  }
  const first = match.index + "Patient ".length;
  const second = first + match[1].length + 1;
  return [
    { entity: "B-NAME", score: 0.99, index: 1, start: first, end: first + match[1].length },
    {
      entity: "I-NAME",
      score: 0.97,
      index: 2,
      start: second,
      end: second + match[2].length,
    },
  ];
}

export interface SyntheticHooks {
  /** Called for the crash marker (the utility process crashes itself). */
  crash?: () => never;
  /** Called for the network probe; must throw unless every probe is blocked. */
  probeNetwork?: () => void;
}

/** Create the synthetic pipeline. */
export function createSyntheticPipeline(hooks: SyntheticHooks = {}): TokenClassificationPipeline {
  return async (text: string) => {
    if (text.includes(CRASH_MARKER)) {
      if (hooks.crash === undefined) {
        throw new Error("synthetic crash unavailable");
      }
      hooks.crash();
    }
    if (text.includes(FAIL_MARKER)) {
      throw new Error("synthetic failure");
    }
    if (text.includes(NETWORK_PROBE_MARKER)) {
      hooks.probeNetwork?.();
    }
    if (text.includes(SLOW_MARKER)) {
      await new Promise<void>((resolve) => setTimeout(resolve, SLOW_DELAY_MS));
    }
    return (REFERENCE.get(text) ?? ruleTokens(text)).map((token) => ({ ...token }));
  };
}
