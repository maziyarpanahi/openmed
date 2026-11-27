#!/usr/bin/env node
// Check or update the committed API reports of the openmed web package.
//
//   npm run api:check    rebuild dist/ and fail when a report differs
//   npm run api:update   rebuild dist/ and rewrite the reports under api/
//
// Every entry in package.json "exports" has its own report, derived from the
// declaration file its "import" condition points at: "." is api/openmed.api.md
// and "./<name>" is api/openmed-<name>.api.md. A new subpath therefore needs
// no extra configuration, only `npm run api:update`. All settings except the
// entry point and report name come from api-extractor.json. This script runs
// offline and never ships in the package.
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Extractor, ExtractorConfig } from "@microsoft/api-extractor";

const REPORT_SUFFIX = ".api.md";
const DEFAULT_PACKAGE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * List the API report entries for a package's "exports" map.
 *
 * @param {string} packageDir Directory holding package.json.
 * @returns {{ subpath: string, types: string, reportName: string }[]}
 */
export function apiReportEntries(packageDir) {
  const packageJson = JSON.parse(
    readFileSync(join(packageDir, "package.json"), "utf8"),
  );
  const baseName = String(packageJson.name).replace(/^@[^/]+\//, "");
  const exportsMap = packageJson.exports ?? {};
  return Object.entries(exportsMap)
    .map(([subpath, conditions]) => {
      const types = conditions?.import?.types;
      if (typeof types !== "string" || !types.endsWith(".d.ts")) {
        throw new Error(`export ${subpath} has no import types declaration`);
      }
      const suffix = subpath === "." ? "" : `-${subpath.replace(/^\.\//, "").replaceAll("/", "-")}`;
      return { subpath, types, reportName: `${baseName}${suffix}` };
    })
    .sort((left, right) => left.reportName.localeCompare(right.reportName));
}

function reportFolder(packageDir, configObject) {
  const folder = configObject.apiReport?.reportFolder ?? "<projectFolder>/api/";
  return resolve(packageDir, folder.replace("<projectFolder>", "."));
}

/**
 * Run API Extractor for every export of the package.
 *
 * @param {object} [options]
 * @param {string} [options.packageDir] Package directory (defaults to this package).
 * @param {boolean} [options.update] Rewrite reports instead of comparing them.
 * @param {boolean} [options.quiet] Suppress API Extractor console output.
 * @returns {{ ok: boolean, changed: string[], missing: string[], orphaned: string[], failed: string[] }}
 *   Subpaths whose report changed, report files that were missing, report
 *   files no export produces, and subpaths whose analysis failed.
 */
export function runApiReports({
  packageDir = DEFAULT_PACKAGE_DIR,
  update = false,
  quiet = false,
} = {}) {
  const configPath = join(packageDir, "api-extractor.json");
  const packageJsonPath = join(packageDir, "package.json");
  const baseConfig = ExtractorConfig.loadFile(configPath);
  const folder = reportFolder(packageDir, baseConfig);
  const entries = apiReportEntries(packageDir);
  const expected = new Set(entries.map((entry) => `${entry.reportName}${REPORT_SUFFIX}`));
  const outcome = { ok: true, changed: [], missing: [], orphaned: [], failed: [] };
  if (update) {
    mkdirSync(folder, { recursive: true });
  }

  for (const entry of entries) {
    const reportFile = join(folder, `${entry.reportName}${REPORT_SUFFIX}`);
    const hadReport = existsSync(reportFile);
    const configObject = {
      ...baseConfig,
      mainEntryPointFilePath: `<projectFolder>/${entry.types.replace(/^\.\//, "")}`,
      apiReport: { ...baseConfig.apiReport, enabled: true, reportFileName: entry.reportName },
    };
    const config = ExtractorConfig.prepare({
      configObject,
      configObjectFullPath: configPath,
      packageJsonFullPath: packageJsonPath,
    });
    let result;
    try {
      result = Extractor.invoke(config, {
        localBuild: update,
        printApiReportDiff: !update && !quiet,
        ...(quiet
          ? {
              messageCallback: (message) => {
                message.handled = true;
              },
            }
          : {}),
      });
    } catch (error) {
      // API Extractor throws on declarations it cannot analyze, for example
      // a reference to a type that does not exist.
      if (!quiet) {
        console.error(`API analysis failed for ${entry.subpath}: ${error?.message ?? error}`);
      }
      outcome.failed.push(entry.subpath);
      continue;
    }
    if (!hadReport) {
      outcome.missing.push(entry.subpath);
    } else if (result.apiReportChanged) {
      outcome.changed.push(entry.subpath);
    }
    // A changed report is the only warning a check expects; any other error
    // or warning (for example a compiler diagnostic) fails the analysis.
    const reportOnlyFailure = !update && result.apiReportChanged;
    if (result.errorCount > 0 || (!result.succeeded && !reportOnlyFailure)) {
      outcome.failed.push(entry.subpath);
    }
  }

  const present = existsSync(folder)
    ? readdirSync(folder).filter((name) => name.endsWith(REPORT_SUFFIX))
    : [];
  for (const name of present.sort()) {
    if (!expected.has(name)) {
      outcome.orphaned.push(name);
      if (update) {
        rmSync(join(folder, name));
      }
    }
  }

  outcome.ok = update
    ? outcome.failed.length === 0
    : outcome.failed.length === 0 &&
      outcome.changed.length === 0 &&
      outcome.missing.length === 0 &&
      outcome.orphaned.length === 0;
  return outcome;
}

function main(argv) {
  const update = argv.includes("--update");
  if (!update && !argv.includes("--check")) {
    console.error("usage: api-report.mjs --check | --update");
    return 2;
  }
  const outcome = runApiReports({ update });
  if (update) {
    const touched = [...outcome.changed, ...outcome.missing];
    console.log(
      outcome.ok
        ? `API reports updated (${touched.length ? touched.join(", ") : "no changes"}` +
            `${outcome.orphaned.length ? `; removed ${outcome.orphaned.join(", ")}` : ""}).`
        : `API report update failed for ${outcome.failed.join(", ")}.`,
    );
    return outcome.ok ? 0 : 1;
  }
  if (outcome.ok) {
    console.log("API reports match the committed baseline.");
    return 0;
  }
  const problems = [
    outcome.changed.length ? `changed: ${outcome.changed.join(", ")}` : "",
    outcome.missing.length ? `missing report: ${outcome.missing.join(", ")}` : "",
    outcome.orphaned.length ? `no matching export: ${outcome.orphaned.join(", ")}` : "",
    outcome.failed.length ? `analysis failed: ${outcome.failed.join(", ")}` : "",
  ].filter(Boolean);
  console.error(
    `The public API differs from js/openmedkit-web/api/ (${problems.join("; ")}). ` +
      "Review the change, then run `npm run api:update` in js/openmedkit-web " +
      "and commit the updated report.",
  );
  return 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
