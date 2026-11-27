// Checks the declaration API gate of the openmed web package
// (js/openmedkit-web/scripts/api-report.mjs with the package's real
// api-extractor.json). Gate behavior runs against small synthetic packages in
// a temporary directory, so these tests never depend on the current exports.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

interface ApiReportOutcome {
  ok: boolean;
  changed: string[];
  missing: string[];
  orphaned: string[];
  failed: string[];
}

interface ApiReportModule {
  apiReportEntries(packageDir: string): Array<{
    subpath: string;
    types: string;
    reportName: string;
  }>;
  runApiReports(options: {
    packageDir: string;
    update?: boolean;
    quiet?: boolean;
  }): ApiReportOutcome;
}

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const packageDir = join(rootDir, "js", "openmedkit-web");
let gatePromise: Promise<ApiReportModule> | undefined;

function loadGate(): Promise<ApiReportModule> {
  gatePromise ??= import(
    pathToFileURL(join(packageDir, "scripts", "api-report.mjs")).href
  ) as Promise<ApiReportModule>;
  return gatePromise;
}

const BASE_DECLARATIONS = `/** Options for loading a synthetic model. */
export interface LoadOptions {
  readonly model?: string;
  readonly variant?: "int8" | "fp32";
}
/** Load a synthetic model. */
export declare function load(options?: LoadOptions): Promise<void>;
/** Release a synthetic model. */
export declare function unload(): void;
`;

const EXTRA_DECLARATIONS = `/** A synthetic subpath export. */
export declare function extra(value: number): number;
`;

interface Fixture {
  dir: string;
  check(): ApiReportOutcome;
  update(): ApiReportOutcome;
  writeDeclarations(file: string, text: string): Promise<void>;
  setExports(subpaths: Record<string, string>): Promise<void>;
  report(name: string): Promise<string>;
}

async function fixture(t: { after(fn: () => Promise<void>): void }) {
  const gate = await loadGate();
  const dir = await mkdtemp(join(tmpdir(), "openmed-api-gate-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await mkdir(join(dir, "dist"));
  await copyFile(
    join(packageDir, "api-extractor.json"),
    join(dir, "api-extractor.json"),
  );
  await writeFile(
    join(dir, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: { strict: true, lib: ["ES2022"], types: [] },
    }),
  );
  const fixtureApi: Fixture = {
    dir,
    check: () => gate.runApiReports({ packageDir: dir, quiet: true }),
    update: () =>
      gate.runApiReports({ packageDir: dir, update: true, quiet: true }),
    writeDeclarations: (file, text) => writeFile(join(dir, "dist", file), text),
    async setExports(subpaths) {
      const exportsMap = Object.fromEntries(
        Object.entries(subpaths).map(([subpath, file]) => [
          subpath,
          { import: { types: `./dist/${file}`, default: "./dist/index.js" } },
        ]),
      );
      await writeFile(
        join(dir, "package.json"),
        JSON.stringify({
          name: "@synthetic/api-gate",
          version: "0.0.0",
          exports: exportsMap,
        }),
      );
    },
    report: (name) => readFile(join(dir, "api", name), "utf8"),
  };
  await fixtureApi.writeDeclarations("index.d.ts", BASE_DECLARATIONS);
  await fixtureApi.writeDeclarations("extra.d.ts", EXTRA_DECLARATIONS);
  await fixtureApi.setExports({ ".": "index.d.ts", "./extra": "extra.d.ts" });
  return fixtureApi;
}

function assertFails(
  outcome: ApiReportOutcome,
  expected: Partial<ApiReportOutcome>,
) {
  assert.equal(outcome.ok, false);
  assert.deepEqual(
    {
      changed: outcome.changed,
      missing: outcome.missing,
      orphaned: outcome.orphaned,
      failed: outcome.failed,
    },
    { changed: [], missing: [], orphaned: [], failed: [], ...expected },
  );
}

test("update writes one report per export and the check then passes", async (t) => {
  const pkg = await fixture(t);

  assertFails(pkg.check(), { missing: [".", "./extra"] });
  assert.equal(existsSync(join(pkg.dir, "api")), false);

  const updated = pkg.update();
  assert.equal(updated.ok, true);
  assert.deepEqual(updated.missing, [".", "./extra"]);
  assert.deepEqual((await readdir(join(pkg.dir, "api"))).sort(), [
    "api-gate-extra.api.md",
    "api-gate.api.md",
  ]);
  const main = await pkg.report("api-gate.api.md");
  assert.match(main, /export function load\(options\?: LoadOptions\): Promise<void>;/);
  assert.match(main, /readonly variant\?: "int8" \| "fp32";/);
  assert.match(await pkg.report("api-gate-extra.api.md"), /export function extra/);

  assert.deepEqual(pkg.check(), {
    ok: true,
    changed: [],
    missing: [],
    orphaned: [],
    failed: [],
  });
});

test("removing an exported function fails the check without touching the report", async (t) => {
  const pkg = await fixture(t);
  pkg.update();
  const before = await pkg.report("api-gate.api.md");

  await pkg.writeDeclarations(
    "index.d.ts",
    BASE_DECLARATIONS.replace(/\/\*\* Release[^]*$/, ""),
  );

  assertFails(pkg.check(), { changed: ["."] });
  assert.equal(await pkg.report("api-gate.api.md"), before);
});

test("changing an option type fails the check", async (t) => {
  const pkg = await fixture(t);
  pkg.update();

  await pkg.writeDeclarations(
    "index.d.ts",
    BASE_DECLARATIONS.replace('"int8" | "fp32"', "string"),
  );

  assertFails(pkg.check(), { changed: ["."] });
});

test("adding an export changes the report and passes after update", async (t) => {
  const pkg = await fixture(t);
  pkg.update();

  await pkg.writeDeclarations(
    "extra.d.ts",
    `${EXTRA_DECLARATIONS}/** A synthetic constant. */\nexport declare const EXTRA_FLAG = true;\n`,
  );

  assertFails(pkg.check(), { changed: ["./extra"] });
  assert.doesNotMatch(await pkg.report("api-gate-extra.api.md"), /EXTRA_FLAG/);
  const updated = pkg.update();
  assert.equal(updated.ok, true);
  assert.deepEqual(updated.changed, ["./extra"]);
  assert.match(await pkg.report("api-gate-extra.api.md"), /EXTRA_FLAG = true/);
  assert.equal(pkg.check().ok, true);
});

test("a new subpath needs a report and a removed one leaves no orphan", async (t) => {
  const pkg = await fixture(t);
  pkg.update();

  await pkg.writeDeclarations(
    "more.d.ts",
    "export declare const MORE: number;\n",
  );
  await pkg.setExports({
    ".": "index.d.ts",
    "./extra": "extra.d.ts",
    "./more": "more.d.ts",
  });
  assertFails(pkg.check(), { missing: ["./more"] });
  assert.equal(pkg.update().ok, true);
  assert.equal(pkg.check().ok, true);

  await pkg.setExports({ ".": "index.d.ts", "./extra": "extra.d.ts" });
  assertFails(pkg.check(), { orphaned: ["api-gate-more.api.md"] });
  assert.deepEqual(pkg.update().orphaned, ["api-gate-more.api.md"]);
  assert.equal(existsSync(join(pkg.dir, "api", "api-gate-more.api.md")), false);
  assert.equal(pkg.check().ok, true);
});

test("update refuses declarations that do not compile", async (t) => {
  const pkg = await fixture(t);
  pkg.update();

  await pkg.writeDeclarations(
    "extra.d.ts",
    "export declare function extra(value: MissingType): number;\n",
  );

  assert.equal(pkg.check().ok, false);
  const updated = pkg.update();
  assert.equal(updated.ok, false);
  assert.deepEqual(updated.failed, ["./extra"]);
});

test("the committed baseline covers exactly the package exports", async () => {
  const gate = await loadGate();
  const expected = gate
    .apiReportEntries(packageDir)
    .map((entry) => `${entry.reportName}.api.md`)
    .sort();
  const committed = (await readdir(join(packageDir, "api")))
    .filter((name) => name.endsWith(".api.md"))
    .sort();
  assert.deepEqual(committed, expected);
  assert.ok(expected.includes("openmed.api.md"));

  const ci = await readFile(join(rootDir, ".github", "workflows", "ci.yml"), "utf8");
  assert.ok(
    ci.includes("npm --prefix js/openmedkit-web run api:check"),
    "ci.yml must run the web API check",
  );
});
