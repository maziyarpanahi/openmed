// Validate a parity report pulled from the simulator or emulator.
//
//   tsx scripts/check-report.ts --report FILE --platform ios|android [--summary FILE]
//
// Exits non-zero unless the report is release evidence: real bridge, release
// bundle, every manifest corpus replayed with matching digests, no mismatch,
// no network attempt, and no source text anywhere in the report.

import { appendFile, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { loadCorpora, parseCorpusManifest } from "../src/parity/corpus";
import { CORPUS_MANIFEST, CORPUS_REGISTRY } from "../src/parity/corpusRegistry";
import { checkEvidenceReport, summarizeEvidence } from "../src/parity/evidence";
import type { ParityReport } from "../src/parity/runner";

function option(args: readonly string[], name: string): string | undefined {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : undefined;
}

async function main(args: readonly string[]): Promise<number> {
  const reportPath = option(args, "report");
  const platform = option(args, "platform");
  if (reportPath === undefined || (platform !== "ios" && platform !== "android")) {
    console.error("usage: check-report.ts --report FILE --platform ios|android");
    return 2;
  }
  let report: unknown;
  try {
    report = JSON.parse(await readFile(resolve(reportPath), "utf8"));
  } catch {
    console.error("parity report is missing or is not JSON");
    return 1;
  }
  const manifest = parseCorpusManifest(CORPUS_MANIFEST);
  const corpora = loadCorpora(manifest, CORPUS_REGISTRY);
  const check = checkEvidenceReport(report, manifest, corpora, platform);

  const summaryPath = option(args, "summary");
  if (summaryPath !== undefined && check.problems[0] !== "report_not_object") {
    try {
      await appendFile(
        summaryPath,
        summarizeEvidence(report as ParityReport, check),
        "utf8",
      );
    } catch {
      console.error("parity summary could not be rendered");
    }
  }
  if (!check.ok) {
    for (const problem of check.problems) {
      console.error(`parity evidence problem: ${problem}`);
    }
    return 1;
  }
  console.log(`parity evidence accepted for ${platform}`);
  return 0;
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    () => {
      console.error("check-report failed");
      process.exitCode = 1;
    },
  );
}
