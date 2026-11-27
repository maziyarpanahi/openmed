// Validation of a parity report pulled off a simulator or emulator.
//
// CI runs this against the report the app wrote. A report counts as release
// evidence only when it came from a release bundle, crossed the real bridge,
// covered every manifest corpus with matching digests, and stays value-free.

import { canonicalDigest } from "./canonical";
import type { CorpusManifest, LoadedCorpus } from "./corpus";
import { findReportLeaks } from "./privacy";
import { PARITY_REPORT_SCHEMA, type ParityReport } from "./runner";

export const REQUIRED_LIFECYCLE_CHECKS = [
  "release_guard",
  "analyze_before_load_rejects",
  "model_load_initial",
  "model_load_reuse",
  "model_replacement",
  "model_replacement_reuse",
] as const;

export interface EvidenceCheck {
  ok: boolean;
  problems: string[];
}

/** Check that a pulled report is complete, passing, and value-free. */
export function checkEvidenceReport(
  value: unknown,
  manifest: CorpusManifest,
  corpora: readonly LoadedCorpus[],
  expectedPlatform: "ios" | "android",
): EvidenceCheck {
  const problems: string[] = [];
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return { ok: false, problems: ["report_not_object"] };
  }
  const report = value as ParityReport;

  if (report.schema !== PARITY_REPORT_SCHEMA || report.version !== 1) {
    problems.push("report_schema_unsupported");
  }
  const { report_digest: digest, ...unsigned } = report;
  if (digest !== canonicalDigest(unsigned)) {
    problems.push("report_digest_mismatch");
  }
  if (report.status !== "pass") {
    problems.push("report_status_not_pass");
  }
  if (report.evidence_eligible !== true) {
    problems.push("report_not_evidence_eligible");
  }
  if (report.platform !== expectedPlatform) {
    problems.push("report_platform_unexpected");
  }
  if (report.js_mode !== "release" || report.build_configuration !== "release") {
    problems.push("report_not_release_build");
  }
  if (report.bridge_source !== "turbo-module" && report.bridge_source !== "bridge-module") {
    problems.push("report_bridge_not_native");
  }
  if (report.egress?.blocked_attempts !== 0) {
    problems.push("report_network_egress");
  }
  if (
    report.sentinel?.bridge_surface_leaks !== 0 ||
    report.sentinel?.report_leaks !== 0 ||
    report.diagnostics_redacted !== false
  ) {
    problems.push("report_sentinel_leak");
  }
  if (typeof report.bundle?.digest !== "string" || !(report.bundle.file_count > 0)) {
    problems.push("report_bundle_unverified");
  }

  const expectedCorpora = corpora.map((corpus) => ({
    corpus_id: corpus.id,
    digest: corpus.digest,
    cases: corpus.cases.length,
  }));
  if (
    canonicalDigest(report.corpora) !== canonicalDigest(expectedCorpora) ||
    manifest.corpora.length !== corpora.length
  ) {
    problems.push("report_corpus_coverage_mismatch");
  }
  const expectedCases = corpora.reduce((total, corpus) => total + corpus.cases.length, 0);
  if (!(report.cases?.total >= expectedCases) || report.cases?.mismatched !== 0) {
    problems.push("report_case_coverage_incomplete");
  }
  for (const operation of manifest.defaults.operations) {
    const stats = report.operations?.[operation];
    if (!stats || stats.mismatched !== 0 || stats.compared - stats.error_parity <= 0) {
      problems.push(`report_operation_incomplete_${operation.toLowerCase()}`);
    }
  }
  for (const check of REQUIRED_LIFECYCLE_CHECKS) {
    const entry = report.lifecycle?.find((item) => item.check === check);
    if (entry?.outcome !== "pass") {
      problems.push(`report_lifecycle_${check}_not_pass`);
    }
  }

  const sentinels = corpora.flatMap((corpus) => corpus.cases.map((item) => item.text));
  if (findReportLeaks(report, sentinels).length > 0) {
    problems.push("report_contains_source_text");
  }

  return { ok: problems.length === 0, problems };
}

/** Render a short, value-free Markdown summary for the CI job page. */
export function summarizeEvidence(report: ParityReport, check: EvidenceCheck): string {
  const lines = [
    `### React Native bridge parity (${report.platform})`,
    "",
    "| Field | Value |",
    "| --- | --- |",
    `| Status | ${report.status} |`,
    `| Evidence eligible | ${check.ok ? "yes" : "no"} |`,
    `| Runtime | ${report.runtime_backend} on ${report.device_class} (${report.os_version}) |`,
    `| Build | js ${report.js_mode}, native ${report.build_configuration} |`,
    `| Bridge source | ${report.bridge_source} |`,
    `| Bundle digest | ${report.bundle.digest ?? "none"} (${report.bundle.file_count} files) |`,
    `| Cases | ${report.cases.matched}/${report.cases.total} matched |`,
    `| Network attempts blocked | ${report.egress.blocked_attempts} |`,
    `| Report digest | ${report.report_digest ?? "none"} |`,
    "",
    "| Lifecycle check | Outcome |",
    "| --- | --- |",
    ...report.lifecycle.map((item) => `| ${item.check} | ${item.outcome} |`),
  ];
  if (check.problems.length > 0) {
    lines.push("", "Problems:", ...check.problems.map((problem) => `- ${problem}`));
  }
  return `${lines.join("\n")}\n`;
}
