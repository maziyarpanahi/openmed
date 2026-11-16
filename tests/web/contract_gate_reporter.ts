// Web reporter for the cross-runtime contract gate (docs/contract-gate.md).
//
// Reads tests/fixtures/parity/contract_gate.json, checks every case of each
// fixture whose `web` cell is `required` against the openmed web package, and
// writes a value-free `web.json` result: fixture ids, the SHA-256 of the
// fixture bytes read, case ids and snake_case reason codes. Known failures are
// recorded as failed cases with a reason code, never skipped; the gate, not
// this reporter, decides what blocks a release.
//
//   npm --prefix js/openmedkit-web run contract-gate -- --results-dir <dir>
//
// Options mirror `contract_gate.py run-python`: --results-dir (default
// $OPENMED_CONTRACT_GATE_RESULTS_DIR), --output, --fixture (repeatable) and
// --check. Output lines carry ids and counts only.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  ContractCaseFailure,
  WEB_CONTRACT_CHECKERS,
  type FixtureChecker,
  type FixturePlan,
  type FixtureReader,
} from "./contract_gate_checks";

export const WEB_RUNTIME = "web";
export const DEFAULT_MANIFEST = "tests/fixtures/parity/contract_gate.json";
export const RESULTS_DIR_ENV = "OPENMED_CONTRACT_GATE_RESULTS_DIR";
export const RESULT_SCHEMA = "openmed.contract_gate.result";
export const RESULT_SCHEMA_VERSION = 1;

/** Reason codes shared with the Python runner (scripts/release/contract_gate.py). */
export const REASON_CONSUMER_ERROR = "consumer_error";
export const REASON_CONSUMER_MISSING = "consumer_missing";
export const REASON_CASE_NOT_EXERCISED = "case_not_exercised";
export const REASON_FIXTURE_UNREADABLE = "fixture_unreadable";

export const REASON_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;
const CASE_PART_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const REPO_ROOT = fileURLToPath(new URL("../..", import.meta.url));

export interface CaseOutcome {
  case_id: string;
  reason: string;
}

export interface FixtureResult {
  fixture_id: string;
  fixture_sha256: string;
  passed: string[];
  failed: CaseOutcome[];
  skipped: CaseOutcome[];
  errors?: string[];
  tests?: { passed: number; failed: number; skipped: number };
}

export interface ContractGateResult {
  schema: typeof RESULT_SCHEMA;
  schema_version: typeof RESULT_SCHEMA_VERSION;
  runtime: string;
  manifest_sha256: string;
  fixtures: FixtureResult[];
}

export interface ManifestCaseGroup {
  key: string;
  id_field?: string;
}

export interface ManifestFixture {
  id: string;
  path: string;
  sha256: string;
  version: number;
  case_groups: ManifestCaseGroup[];
  companions: { path: string; sha256: string }[];
  runtimes: Record<string, { status: string; consumers?: string[]; reason?: string }>;
}

export interface ContractGateManifest {
  result_schema: { path: string; sha256: string };
  runtimes: Record<string, { reporter: string; command?: string; tracking?: string }>;
  fixtures: ManifestFixture[];
}

/** A value-free reporter failure, such as an unreadable manifest. */
export class ContractGateError extends Error {
  constructor(readonly code: string) {
    super(code);
    this.name = "ContractGateError";
  }
}

export interface RunWebContractGateOptions {
  /** Repository root; defaults to the checkout that holds this file. */
  root?: string;
  /** Repository-relative manifest path. */
  manifest?: string;
  /** Per-fixture checkers; defaults to the web package checkers. */
  checkers?: Readonly<Record<string, FixtureChecker>>;
  /** Only report these fixture ids. */
  fixtureIds?: readonly string[];
  /** Reads repository-relative files; defaults to the file system under root. */
  read?: FixtureReader;
}

interface DerivedCase {
  caseId: string;
  group: string;
  item: unknown;
  index: number;
}

export function sha256Hex(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function fileReader(root: string): FixtureReader {
  return async (path) => new Uint8Array(await readFile(join(root, path)));
}

function decodeJson(bytes: Uint8Array): unknown {
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Load the manifest and the digest of the bytes read. */
export async function loadContractGateManifest(
  read: FixtureReader,
  manifest = DEFAULT_MANIFEST,
): Promise<{ manifest: ContractGateManifest; sha256: string }> {
  let bytes: Uint8Array;
  let document: unknown;
  try {
    bytes = await read(manifest);
    document = decodeJson(bytes);
  } catch {
    throw new ContractGateError("manifest_unreadable");
  }
  if (
    !isRecord(document) ||
    document.schema !== "openmed.contract_gate.manifest" ||
    !Array.isArray(document.fixtures) ||
    !isRecord(document.runtimes)
  ) {
    throw new ContractGateError("manifest_invalid");
  }
  return { manifest: document as unknown as ContractGateManifest, sha256: sha256Hex(bytes) };
}

/**
 * Derive a fixture's case ids from its content, exactly as
 * `contract_gate.py` does: `<key>/<item id>` for a list group with an
 * `id_field`, `<key>/<index>` for one without, `<key>` for an object group.
 */
export function expectedCaseIds(entry: ManifestFixture, document: unknown): string[] {
  return deriveCases(entry, document).map((item) => item.caseId);
}

function deriveCases(entry: ManifestFixture, document: unknown): DerivedCase[] {
  if (!isRecord(document)) {
    throw new ContractGateError("case_group_missing");
  }
  const cases: DerivedCase[] = [];
  for (const group of entry.case_groups) {
    const value = document[group.key];
    if (isRecord(value)) {
      cases.push({ caseId: group.key, group: group.key, item: value, index: 0 });
      continue;
    }
    if (!Array.isArray(value)) {
      throw new ContractGateError("case_group_missing");
    }
    value.forEach((item, index) => {
      let part = String(index);
      if (group.id_field !== undefined) {
        const raw = isRecord(item) ? item[group.id_field] : undefined;
        if (typeof raw !== "string" || !CASE_PART_PATTERN.test(raw)) {
          throw new ContractGateError("case_id_invalid");
        }
        part = raw;
      }
      cases.push({ caseId: `${group.key}/${part}`, group: group.key, item, index });
    });
  }
  if (new Set(cases.map((item) => item.caseId)).size !== cases.length) {
    throw new ContractGateError("case_id_duplicate");
  }
  return cases;
}

function reasonFor(error: unknown): string {
  return error instanceof ContractCaseFailure && REASON_PATTERN.test(error.reason)
    ? error.reason
    : REASON_CONSUMER_ERROR;
}

async function fixtureResult(
  entry: ManifestFixture,
  checkers: Readonly<Record<string, FixtureChecker>>,
  read: FixtureReader,
): Promise<FixtureResult> {
  const result: FixtureResult = {
    fixture_id: entry.id,
    fixture_sha256: "0".repeat(64),
    passed: [],
    failed: [],
    skipped: [],
  };
  let document: unknown;
  let cases: DerivedCase[];
  try {
    const bytes = await read(entry.path);
    result.fixture_sha256 = sha256Hex(bytes);
    document = decodeJson(bytes);
    cases = deriveCases(entry, document);
  } catch {
    result.errors = [REASON_FIXTURE_UNREADABLE];
    return result;
  }
  const fail = (reason: string) =>
    cases.map((item) => ({ case_id: item.caseId, reason }));
  const checker = Object.hasOwn(checkers, entry.id) ? checkers[entry.id] : undefined;
  if (checker === undefined) {
    result.failed = fail(REASON_CONSUMER_MISSING);
    return result;
  }
  let plan: FixturePlan;
  try {
    plan = await checker(document, { fixturePath: entry.path, read });
  } catch (error) {
    result.failed = fail(reasonFor(error));
    result.tests = { passed: 0, failed: 1, skipped: 0 };
    return result;
  }
  const tests = { passed: 0, failed: 0, skipped: 0 };
  for (const item of cases) {
    const check = Object.hasOwn(plan.groups, item.group) ? plan.groups[item.group] : undefined;
    if (check === undefined) {
      result.failed.push({ case_id: item.caseId, reason: REASON_CASE_NOT_EXERCISED });
      continue;
    }
    try {
      await check(item.item, item.index);
      result.passed.push(item.caseId);
      tests.passed += 1;
    } catch (error) {
      result.failed.push({ case_id: item.caseId, reason: reasonFor(error) });
      tests.failed += 1;
    }
  }
  const errors = [...new Set(plan.errors ?? [])].map((code) =>
    REASON_PATTERN.test(code) ? code : REASON_CONSUMER_ERROR,
  );
  if (errors.length > 0) {
    result.errors = [...new Set(errors)];
  }
  result.tests = tests;
  return result;
}

/**
 * Check every web-required fixture of the contract-gate manifest against the
 * web package and build a value-free result document.
 */
export async function runWebContractGate(
  options: RunWebContractGateOptions = {},
): Promise<ContractGateResult> {
  const read = options.read ?? fileReader(options.root ?? REPO_ROOT);
  const checkers = options.checkers ?? WEB_CONTRACT_CHECKERS;
  const { manifest, sha256 } = await loadContractGateManifest(read, options.manifest);
  const selected = manifest.fixtures.filter(
    (entry) =>
      entry.runtimes?.[WEB_RUNTIME]?.status === "required" &&
      (options.fixtureIds === undefined || options.fixtureIds.includes(entry.id)),
  );
  const fixtures: FixtureResult[] = [];
  for (const entry of selected) {
    fixtures.push(await fixtureResult(entry, checkers, read));
  }
  return {
    schema: RESULT_SCHEMA,
    schema_version: RESULT_SCHEMA_VERSION,
    runtime: WEB_RUNTIME,
    manifest_sha256: sha256,
    fixtures,
  };
}

/** Serialise a result the way `contract_gate.py` commits JSON. */
export function dumpResult(result: ContractGateResult): string {
  return `${JSON.stringify(result, null, 2)}\n`.replace(
    /[\u007f-￿]/g,
    (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

type SchemaNode = Record<string, unknown>;

const SCHEMA_ANNOTATIONS = new Set([
  "$schema",
  "$id",
  "$defs",
  "title",
  "description",
  "schema_version",
]);
const SCHEMA_KEYWORDS = new Set([
  "type",
  "const",
  "pattern",
  "required",
  "properties",
  "additionalProperties",
  "items",
  "maxItems",
  "uniqueItems",
  "minimum",
  "maximum",
  "$ref",
]);

/**
 * Validate a result document against the committed result JSON Schema.
 *
 * Implements the Draft 2020-12 keywords that schema uses and fails closed on
 * any other keyword. Findings carry a JSON path and keyword, never values.
 */
export function validateContractGateResult(document: unknown, schema: SchemaNode): string[] {
  const findings: string[] = [];
  const resolveRef = (ref: string): SchemaNode => {
    const match = /^#\/\$defs\/([A-Za-z0-9_]+)$/.exec(ref);
    const defs = schema.$defs as Record<string, SchemaNode> | undefined;
    const target = match === null ? undefined : defs?.[match[1]!];
    if (target === undefined) {
      throw new ContractGateError("result_schema_unsupported");
    }
    return target;
  };
  const visit = (value: unknown, node: SchemaNode, path: string): void => {
    for (const keyword of Object.keys(node)) {
      if (!SCHEMA_KEYWORDS.has(keyword) && !SCHEMA_ANNOTATIONS.has(keyword)) {
        throw new ContractGateError("result_schema_unsupported");
      }
    }
    const report = (keyword: string) =>
      findings.push(`result_schema_violation path=${path} keyword=${keyword}`);
    if (typeof node.$ref === "string") {
      visit(value, resolveRef(node.$ref), path);
    }
    if ("const" in node && value !== node.const) {
      report("const");
    }
    const type = node.type;
    if (type === "object") {
      if (!isRecord(value)) {
        report("type");
        return;
      }
      for (const key of (node.required as string[] | undefined) ?? []) {
        if (!Object.hasOwn(value, key)) {
          report("required");
        }
      }
      const properties = (node.properties as Record<string, SchemaNode> | undefined) ?? {};
      for (const [key, child] of Object.entries(value)) {
        if (Object.hasOwn(properties, key)) {
          visit(child, properties[key]!, `${path}.${key}`);
        } else if (node.additionalProperties === false) {
          report("additionalProperties");
        }
      }
    } else if (type === "array") {
      if (!Array.isArray(value)) {
        report("type");
        return;
      }
      if (typeof node.maxItems === "number" && value.length > node.maxItems) {
        report("maxItems");
      }
      if (node.uniqueItems === true) {
        const keys = value.map((item) => JSON.stringify(item));
        if (new Set(keys).size !== keys.length) {
          report("uniqueItems");
        }
      }
      if (isRecord(node.items)) {
        value.forEach((item, index) => visit(item, node.items as SchemaNode, `${path}[${index}]`));
      }
    } else if (type === "string") {
      if (typeof value !== "string") {
        report("type");
        return;
      }
      if (typeof node.pattern === "string" && !new RegExp(node.pattern, "u").test(value)) {
        report("pattern");
      }
    } else if (type === "integer") {
      if (typeof value !== "number" || !Number.isInteger(value)) {
        report("type");
        return;
      }
      if (typeof node.minimum === "number" && value < node.minimum) {
        report("minimum");
      }
      if (typeof node.maximum === "number" && value > node.maximum) {
        report("maximum");
      }
    } else if (type !== undefined) {
      throw new ContractGateError("result_schema_unsupported");
    }
  };
  visit(document, schema, "$");
  if (findings.length > 0 || !isRecord(document) || !Array.isArray(document.fixtures)) {
    return findings;
  }
  const seenFixtures = new Set<string>();
  document.fixtures.forEach((item: FixtureResult, index: number) => {
    if (seenFixtures.has(item.fixture_id)) {
      findings.push(`fixture_reported_twice path=$.fixtures[${index}]`);
    }
    seenFixtures.add(item.fixture_id);
    const reported = [
      ...item.passed,
      ...item.failed.map((row) => row.case_id),
      ...item.skipped.map((row) => row.case_id),
    ];
    if (new Set(reported).size !== reported.length) {
      findings.push(`case_reported_twice path=$.fixtures[${index}]`);
    }
  });
  return findings;
}

/** Load the result JSON Schema the manifest pins. */
export async function loadResultSchema(
  read: FixtureReader = fileReader(REPO_ROOT),
  manifest = DEFAULT_MANIFEST,
): Promise<SchemaNode> {
  const { manifest: parsed } = await loadContractGateManifest(read, manifest);
  try {
    return decodeJson(await read(parsed.result_schema.path)) as SchemaNode;
  } catch {
    throw new ContractGateError("result_schema_unreadable");
  }
}

/** Resolve where `web.json` goes, or `undefined` when nothing names a target. */
export function resultPath(
  options: { output?: string; resultsDir?: string },
  env: Readonly<Record<string, string | undefined>> = process.env,
  cwd = process.cwd(),
): string | undefined {
  if (options.output !== undefined) {
    return resolve(cwd, options.output);
  }
  const directory = options.resultsDir ?? env[RESULTS_DIR_ENV];
  if (directory === undefined || directory === "") {
    return undefined;
  }
  return resolve(cwd, directory, `${WEB_RUNTIME}.json`);
}

/** Write a result file, creating its directory. */
export async function writeContractGateResult(
  result: ContractGateResult,
  path: string,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, dumpResult(result), "utf8");
}

interface CliArguments {
  output?: string;
  resultsDir?: string;
  fixtureIds: string[];
  check: boolean;
}

function parseArguments(argv: readonly string[]): CliArguments | undefined {
  const parsed: CliArguments = { fixtureIds: [], check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (flag === "--check") {
      parsed.check = true;
      continue;
    }
    if (value === undefined || value.startsWith("--")) {
      return undefined;
    }
    if (flag === "--output" && parsed.resultsDir === undefined) {
      parsed.output = value;
    } else if (flag === "--results-dir" && parsed.output === undefined) {
      parsed.resultsDir = value;
    } else if (flag === "--fixture") {
      parsed.fixtureIds.push(value);
    } else {
      return undefined;
    }
    index += 1;
  }
  return parsed;
}

export interface CliIo {
  env?: Readonly<Record<string, string | undefined>>;
  cwd?: string;
  out?: (line: string) => void;
  err?: (line: string) => void;
  run?: RunWebContractGateOptions;
}

/**
 * Run the reporter command line and return its exit code: 0 once a
 * schema-valid result is written, 1 with --check when a case did not pass,
 * 2 on a usage error or an unreadable manifest or schema.
 */
export async function main(argv: readonly string[], io: CliIo = {}): Promise<number> {
  const out = io.out ?? ((line: string) => console.log(line));
  const err = io.err ?? ((line: string) => console.error(line));
  const prefix = "contract_gate web";
  const args = parseArguments(argv);
  if (args === undefined) {
    err(`${prefix}: usage: [--results-dir DIR | --output FILE] [--fixture ID]... [--check]`);
    return 2;
  }
  // npm runs package scripts in the package directory; INIT_CWD is where the
  // command was typed, so relative paths keep their meaning.
  const env = io.env ?? process.env;
  const cwd = io.cwd ?? env.INIT_CWD ?? process.cwd();
  const output = resultPath(args, env, isAbsolute(cwd) ? cwd : resolve(cwd));
  if (output === undefined) {
    err(`${prefix}: pass --output or --results-dir, or set ${RESULTS_DIR_ENV}`);
    return 2;
  }
  try {
    const run = {
      ...io.run,
      ...(args.fixtureIds.length > 0 ? { fixtureIds: args.fixtureIds } : {}),
    };
    const result = await runWebContractGate(run);
    const schemaReader = run.read ?? fileReader(run.root ?? REPO_ROOT);
    const findings = validateContractGateResult(
      result,
      await loadResultSchema(schemaReader, run.manifest),
    );
    if (findings.length > 0) {
      for (const finding of findings) {
        err(`${prefix}: ${finding}`);
      }
      return 2;
    }
    await writeContractGateResult(result, output);
    let incomplete = 0;
    for (const item of result.fixtures) {
      out(
        `${prefix}: fixture=${item.fixture_id} passed=${item.passed.length} ` +
          `failed=${item.failed.length} skipped=${item.skipped.length}`,
      );
      if (item.failed.length > 0 || item.skipped.length > 0 || item.errors !== undefined) {
        incomplete += 1;
      }
    }
    out(`${prefix}: wrote ${WEB_RUNTIME}.json`);
    return args.check && incomplete > 0 ? 1 : 0;
  } catch (error) {
    err(`${prefix}: ${error instanceof ContractGateError ? error.code : "reporter_error"}`);
    return 2;
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  void main(process.argv.slice(2)).then((code) => {
    process.exitCode = code;
  });
}
