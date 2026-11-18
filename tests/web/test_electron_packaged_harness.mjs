/**
 * Runs the OpenMedKit Electron qualification harness and checks its report.
 *
 * Default mode launches the electron-builder `--dir` output (an unsigned,
 * local packaged app). Set OPENMED_ELECTRON_HARNESS_MODE=unpackaged to run the
 * staged app with the pinned Electron binary instead, or
 * OPENMED_ELECTRON_HARNESS_APP to point at a packaged executable.
 *
 * Besides the harness's own checks, this runner plants a random synthetic
 * sentinel name, then scans stdout, stderr, the report, the app's user-data
 * directory, logs and crash dumps for it (UTF-8 and UTF-16LE).
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, readdirSync } from "node:fs";
import {
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const harnessDir = fileURLToPath(
  new URL("../../js/openmedkit-electron/harness/", import.meta.url),
);
const mode = process.env.OPENMED_ELECTRON_HARNESS_MODE ?? "packaged";
const LAUNCH_TIMEOUT_MS = 300_000;

const REQUIRED_CHECKS = [
  "app_packaging",
  "bundle_verified_in_utility",
  "window_hardening",
  "parity_offset_contract",
  "parity_npm_golden",
  "parity_sentinel_note",
  "utility_network_blocked",
  "ipc_capabilities_constrained",
  "cancellation",
  "bundle_generation_change",
  "bundle_rejection_fails_closed",
  "renderer_teardown",
  "utility_crash_restart",
  "relaxed_window_refused",
  "remote_navigation_blocked",
  "diagnostics_value_free",
  "service_diagnostics",
];

/** Synthetic fixture identifiers that must never reach logs or caches either. */
const FIXTURE_IDENTIFIERS = ["Alice Nguyen", "alice@example.org"];

function packagedExecutable() {
  if (process.env.OPENMED_ELECTRON_HARNESS_APP) {
    return process.env.OPENMED_ELECTRON_HARNESS_APP;
  }
  const dist = join(harnessDir, "dist");
  const candidates = [];
  if (process.platform === "linux") {
    candidates.push(join(dist, "linux-unpacked", "openmedkit-electron-harness"));
    candidates.push(join(dist, "linux-arm64-unpacked", "openmedkit-electron-harness"));
  } else if (process.platform === "darwin") {
    for (const name of existsSync(dist) ? readdirSync(dist) : []) {
      if (name.startsWith("mac")) {
        candidates.push(
          join(
            dist,
            name,
            "OpenMedKit Electron Harness.app",
            "Contents",
            "MacOS",
            "OpenMedKit Electron Harness",
          ),
        );
      }
    }
  } else if (process.platform === "win32") {
    candidates.push(join(dist, "win-unpacked", "OpenMedKit Electron Harness.exe"));
  }
  const found = candidates.find((candidate) => existsSync(candidate));
  assert.ok(found, "package the harness first: npm --prefix js/openmedkit-electron/harness run package");
  return found;
}

function fuseTarget(executable) {
  // macOS keeps the fuse wire in the framework; @electron/fuses resolves it
  // from the .app bundle path.
  return process.platform === "darwin" ? join(executable, "..", "..", "..") : executable;
}

function launchCommand() {
  if (mode === "unpackaged") {
    const require = createRequire(join(harnessDir, "package.json"));
    const electron = require("electron");
    return { command: electron, args: [join(harnessDir, "build", "app")] };
  }
  return { command: packagedExecutable(), args: [] };
}

function syntheticName() {
  const letters = "abcdefghijklmnopqrstuvwxyz";
  const bytes = randomBytes(18);
  const word = (offset) =>
    Array.from(bytes.subarray(offset, offset + 9), (value) => letters[value % 26]).join("");
  const given = `Zq${word(0)}`;
  const family = `Xv${word(9)}`;
  return { given, family };
}

async function* walk(root) {
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    const path = join(root, entry.name);
    const info = await lstat(path);
    if (info.isDirectory()) {
      yield* walk(path);
    } else if (info.isFile()) {
      yield path;
    }
  }
}

function encodings(token) {
  return [Buffer.from(token, "utf8"), Buffer.from(token, "utf16le")];
}

async function run(args, env) {
  const { command, args: baseArgs } = launchCommand();
  return new Promise((resolve, reject) => {
    const child = spawn(command, [...baseArgs, ...args], {
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("harness did not exit"));
    }, LAUNCH_TIMEOUT_MS);
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      resolve({
        code,
        signal,
        stdout: Buffer.concat(stdout),
        stderr: Buffer.concat(stderr),
      });
    });
  });
}

test(`OpenMedKit Electron harness qualifies (${mode})`, { timeout: LAUNCH_TIMEOUT_MS + 30_000 }, async () => {
  const scratch = await mkdtemp(join(tmpdir(), "openmed-electron-harness-"));
  const userData = join(scratch, "user-data");
  const reportPath = join(scratch, "report.json");
  const inputPath = join(scratch, "input.json");
  const sentinel = syntheticName();
  await writeFile(inputPath, JSON.stringify(sentinel));
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.ELECTRON_ENABLE_LOGGING;
  try {
    const result = await run(
      [
        `--harness-report=${reportPath}`,
        `--harness-input=${inputPath}`,
        `--harness-user-data=${userData}`,
        "--enable-logging=stderr",
      ],
      env,
    );
    assert.ok(existsSync(reportPath), `harness wrote no report (exit ${result.code})`);
    const reportBytes = await readFile(reportPath);
    const report = JSON.parse(reportBytes.toString("utf8"));
    const failed = report.checks.filter((item) => !item.ok);
    assert.deepEqual(
      failed.map((item) => ({ name: item.name, details: item.details })),
      [],
      "harness checks failed",
    );
    assert.equal(report.ok, true);
    assert.equal(result.code, 0);
    assert.deepEqual(
      REQUIRED_CHECKS.filter((name) => !report.checks.some((item) => item.name === name)),
      [],
      "required checks missing from the report",
    );
    assert.equal(report.packaged, mode !== "unpackaged");
    assert.ok(report.event_counts.utility_process_exited >= 1, "crash path exercised");

    const tokens = [sentinel.given, sentinel.family, ...FIXTURE_IDENTIFIERS];
    const sources = [
      ["stdout", result.stdout],
      ["stderr", result.stderr],
      ["report", reportBytes],
    ];
    let scanned = 0;
    for await (const path of walk(userData)) {
      sources.push([path.slice(userData.length) || "user-data", await readFile(path)]);
      scanned += 1;
    }
    assert.ok(scanned > 0, "the app user-data directory was not created");
    for (const [label, bytes] of sources) {
      for (const token of tokens) {
        for (const encoded of encodings(token)) {
          assert.equal(bytes.includes(encoded), false, `${label} contains a planted identifier`);
        }
      }
    }
  } finally {
    const evidence = process.env.OPENMED_ELECTRON_HARNESS_EVIDENCE;
    if (evidence && existsSync(reportPath)) {
      // The report is value-free (codes, counts, offsets, digests) by design.
      await mkdir(evidence, { recursive: true });
      await copyFile(reportPath, join(evidence, `report-${mode}.json`));
    }
    await rm(scratch, { recursive: true, force: true });
  }
});

test(
  "packaged harness disables the Electron fuses that widen local attack surface",
  { skip: mode === "unpackaged" ? "fuses are applied only by packaging" : false },
  async () => {
    const require = createRequire(join(harnessDir, "package.json"));
    const { getCurrentFuseWire } = require("@electron/fuses");
    const wire = await getCurrentFuseWire(fuseTarget(packagedExecutable()));
    const DISABLED = 48;
    const ENABLED = 49;
    // Indexes follow @electron/fuses FuseV1Options.
    const expected = {
      0: DISABLED, // RunAsNode
      1: ENABLED, // EnableCookieEncryption
      2: DISABLED, // EnableNodeOptionsEnvironmentVariable
      3: DISABLED, // EnableNodeCliInspectArguments
      5: ENABLED, // OnlyLoadAppFromAsar
      7: DISABLED, // GrantFileProtocolExtraPrivileges
    };
    for (const [index, state] of Object.entries(expected)) {
      assert.equal(wire[index], state, `fuse ${index}`);
    }
    if (process.platform === "darwin" || process.platform === "win32") {
      assert.equal(wire[4], ENABLED, "embedded asar integrity validation");
    }
  },
);
