// Contract-gate reporter for the web package (docs/contract-gate.md).
//
// Runs every fixture whose `web` cell in tests/fixtures/parity/contract_gate.json
// is `required` against the package, checks that the value-free result
// validates against the committed result schema, and writes `web.json` into
// $OPENMED_CONTRACT_GATE_RESULTS_DIR when it is set. Failing cases are
// recorded, not asserted away: the gate decides what blocks a release.
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { scalarToUtf16Offset } from "../../js/openmedkit-web/src/index";
import {
  ContractCaseFailure,
  OFFSET_CONTRACT_REASONS,
  WEB_CONTRACT_CHECKERS,
  type FixtureReader,
} from "./contract_gate_checks";
import {
  DEFAULT_MANIFEST,
  RESULTS_DIR_ENV,
  WEB_RUNTIME,
  dumpResult,
  expectedCaseIds,
  loadContractGateManifest,
  loadResultSchema,
  main,
  resultPath,
  runWebContractGate,
  sha256Hex,
  validateContractGateResult,
  writeContractGateResult,
  type ContractGateResult,
  type FixtureResult,
  type ManifestFixture,
} from "./contract_gate_reporter";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const SPEC = "tests/web/test_contract_gate.spec.ts";

const diskReader: FixtureReader = async (path) =>
  new Uint8Array(await readFile(join(rootDir, path)));

function patchedReader(
  target: string,
  patch: (document: any) => void,
  base: FixtureReader = diskReader,
): FixtureReader {
  return async (path) => {
    const bytes = await base(path);
    if (path !== target) {
      return bytes;
    }
    const document = JSON.parse(new TextDecoder().decode(bytes));
    patch(document);
    return new TextEncoder().encode(JSON.stringify(document));
  };
}

async function manifest() {
  return loadContractGateManifest(diskReader);
}

async function webFixtures(): Promise<ManifestFixture[]> {
  const { manifest: parsed } = await manifest();
  return parsed.fixtures.filter((entry) => entry.runtimes[WEB_RUNTIME]?.status === "required");
}

async function fixtureEntry(id: string): Promise<ManifestFixture> {
  const entry = (await manifest()).manifest.fixtures.find((item) => item.id === id);
  assert.ok(entry, id);
  return entry;
}

function only(result: ContractGateResult, id: string): FixtureResult {
  const item = result.fixtures.find((row) => row.fixture_id === id);
  assert.ok(item, id);
  return item;
}

function reasons(item: FixtureResult): Map<string, string> {
  return new Map(item.failed.map((row) => [row.case_id, row.reason]));
}

let cachedResult: Promise<ContractGateResult> | undefined;
function committedResult(): Promise<ContractGateResult> {
  cachedResult ??= runWebContractGate();
  return cachedResult;
}

test("the web result covers every web-required fixture and validates against the schema", async () => {
  const { manifest: parsed, sha256 } = await manifest();
  const result = await committedResult();
  const schema = await loadResultSchema();
  const required = await webFixtures();

  assert.deepEqual(validateContractGateResult(result, schema), []);
  assert.equal(result.runtime, WEB_RUNTIME);
  assert.equal(result.manifest_sha256, sha256);
  assert.deepEqual(
    result.fixtures.map((item) => item.fixture_id),
    required.map((entry) => entry.id),
  );
  assert.ok(required.length >= 5);
  for (const entry of required) {
    const item = only(result, entry.id);
    const document = JSON.parse(await readFile(join(rootDir, entry.path), "utf8"));
    const reported = [...item.passed, ...item.failed.map((row) => row.case_id)];

    assert.equal(item.fixture_sha256, entry.sha256, entry.id);
    assert.deepEqual(reported.sort(), expectedCaseIds(entry, document).sort(), entry.id);
    assert.deepEqual(item.skipped, [], entry.id);
    assert.equal(item.errors, undefined, entry.id);
    assert.ok(
      entry.runtimes[WEB_RUNTIME]!.consumers!.includes(SPEC),
      `${entry.id} names the web reporter spec as a consumer`,
    );
    assert.ok(
      Object.hasOwn(WEB_CONTRACT_CHECKERS, entry.id),
      `${entry.id} has a web checker`,
    );
  }
  assert.ok(parsed.runtimes[WEB_RUNTIME]?.command?.includes("contract-gate"));

  // The committed bytes round-trip through the schema validator.
  const scratch = await mkdtemp(join(tmpdir(), "openmed-contract-gate-"));
  try {
    const path = join(scratch, "web.json");
    await writeContractGateResult(result, path);
    const written = await readFile(path, "utf8");
    assert.equal(written, dumpResult(result));
    assert.match(written, /^[\x20-\x7e\n]*$/);
    assert.deepEqual(validateContractGateResult(JSON.parse(written), schema), []);
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
  const directory = process.env[RESULTS_DIR_ENV];
  if (directory) {
    await writeContractGateResult(result, join(directory, `${WEB_RUNTIME}.json`));
  }
});

test("offset cases that fail today are recorded as failed with a reason code", async () => {
  const entry = await fixtureEntry("offset_contract");
  const fixture = JSON.parse(await readFile(join(rootDir, entry.path), "utf8")) as {
    cases: {
      id: string;
      text: string;
      input_start: number;
      input_end: number;
      expected_start: number;
      expected_end: number;
    }[];
  };
  const item = only(await committedResult(), "offset_contract");
  const failed = reasons(item);

  for (const reason of failed.values()) {
    assert.ok((OFFSET_CONTRACT_REASONS as readonly string[]).includes(reason), reason);
  }
  // The web package neither snaps spans to grapheme clusters nor reports
  // Unicode scalar offsets yet: a case that needs snapping fails on its
  // boundary, a supplementary-plane case on its offset unit, and the rest
  // pass. Update this expectation in the change that fixes web offsets.
  for (const testCase of fixture.cases) {
    const caseId = `cases/${testCase.id}`;
    const snaps =
      testCase.input_start !== testCase.expected_start ||
      testCase.input_end !== testCase.expected_end;
    const shifts =
      scalarToUtf16Offset(testCase.text, testCase.expected_start) !== testCase.expected_start ||
      scalarToUtf16Offset(testCase.text, testCase.expected_end) !== testCase.expected_end;
    const expected = snaps
      ? "grapheme_boundary_mismatch"
      : shifts
        ? "offset_unit_mismatch"
        : undefined;
    assert.equal(failed.get(caseId), expected, caseId);
    assert.equal(item.passed.includes(caseId), expected === undefined, caseId);
  }
  assert.ok(failed.size > 0);
  assert.ok(item.passed.length > 0);
});

test("the result file carries ids, digests and codes but no fixture text", async () => {
  const result = await committedResult();
  // Compact JSON: indentation is layout, not content.
  const serialized = JSON.stringify(result);
  const derivedIds = new Set<string>();
  const vectorIds = new Set<string>();
  const fixtureStrings = new Set<string>();
  const collect = (value: unknown) => {
    if (typeof value === "string") {
      fixtureStrings.add(value);
    } else if (Array.isArray(value)) {
      value.forEach(collect);
    } else if (value !== null && typeof value === "object") {
      Object.values(value).forEach(collect);
    }
  };
  for (const entry of await webFixtures()) {
    const document = JSON.parse(await readFile(join(rootDir, entry.path), "utf8"));
    for (const caseId of expectedCaseIds(entry, document)) {
      derivedIds.add(caseId);
      vectorIds.add(caseId.slice(caseId.indexOf("/") + 1));
    }
    collect(document);
  }

  // Case ids are the fixture's own vector ids, nothing else.
  for (const item of result.fixtures) {
    for (const caseId of [...item.passed, ...item.failed.map((row) => row.case_id)]) {
      assert.ok(derivedIds.has(caseId), caseId);
    }
  }
  // Every other string is a schema constant, runtime id, digest or code.
  const others = new Set<string>([result.schema, result.runtime, result.manifest_sha256]);
  for (const item of result.fixtures) {
    others.add(item.fixture_id).add(item.fixture_sha256);
    item.failed.forEach((row) => others.add(row.reason));
    (item.errors ?? []).forEach((code) => others.add(code));
  }
  const controlled = /^(?:[a-z][a-z0-9_]{0,63}|[0-9a-f]{64}|openmed\.contract_gate\.result)$/;
  for (const value of others) {
    assert.match(value, controlled);
  }
  let checked = 0;
  for (const value of fixtureStrings) {
    if (value.length < 4 || /^[a-z][a-z0-9_]*$/.test(value) || vectorIds.has(value)) {
      continue;
    }
    for (const other of others) {
      assert.equal(other.includes(value), false, "a result code or digest echoes fixture text");
    }
    if (/\s|[^\x20-\x7e]/u.test(value) || value.length >= 24) {
      assert.equal(serialized.includes(value), false, "the result echoes fixture text");
    }
    checked += 1;
  }
  assert.ok(checked > 100);
});

test("altered fixture bytes fail their case with a specific code and a new digest", async () => {
  const run = async (id: string, patch: (document: any) => void) => {
    const entry = await fixtureEntry(id);
    const document = JSON.parse(await readFile(join(rootDir, entry.path), "utf8"));
    const result = await runWebContractGate({
      read: patchedReader(entry.path, patch),
      fixtureIds: [id],
    });
    const item = only(result, id);
    const baseline = reasons(only(await committedResult(), id));
    assert.notEqual(item.fixture_sha256, entry.sha256, id);
    // Only the cases the patch touched change outcome.
    const changed = new Map(
      [...reasons(item)].filter(([caseId, reason]) => baseline.get(caseId) !== reason),
    );
    return { document, changed };
  };

  const offset = await run("offset_contract", (document) => {
    document.cases[0].expected_redacted += "!";
  });
  assert.deepEqual([...offset.changed], [
    [`cases/${offset.document.cases[0].id}`, "redaction_mismatch"],
  ]);

  const kit = await run("kit_parity_vectors", (document) => {
    document.vectors[0].result_json += " ";
    document.rejections[0].reason = "label_unknown_synthetic";
    document.offset_rejections[0].offset = 0;
  });
  assert.deepEqual(Object.fromEntries(kit.changed), {
    [`vectors/${kit.document.vectors[0].id}`]: "canonical_bytes_mismatch",
    [`rejections/${kit.document.rejections[0].id}`]: "reason_mismatch",
    [`offset_rejections/${kit.document.offset_rejections[0].id}`]: "rejection_not_raised",
  });

  const policy = await run("portable_policy_vectors", (document) => {
    document.decision_cases[0].expected.policy_digest = `sha256:${"0".repeat(64)}`;
    document.error_cases[0].expected.code = "synthetic_code";
  });
  assert.deepEqual(Object.fromEntries(policy.changed), {
    [`decision_cases/${policy.document.decision_cases[0].id}`]: "policy_digest_mismatch",
    [`error_cases/${policy.document.error_cases[0].id}`]: "reason_mismatch",
  });

  const decision = await run("decision_encoder_v1", (document) => {
    document.cases[0].expected.result.state = "synthetic_state";
    document.invalid_requests[0].error = "synthetic_kind";
  });
  assert.deepEqual(Object.fromEntries(decision.changed), {
    [`cases/${decision.document.cases[0].id}`]: "result_mismatch",
    [`invalid_requests/${decision.document.invalid_requests[0].id}`]: "reason_mismatch",
  });

  const tokenizer = await run("tokenizer_conformance_v1", (document) => {
    document.cases[0].expected.input_ids[0] += 1;
  });
  assert.deepEqual(Object.fromEntries(tokenizer.changed), {
    [`cases/${tokenizer.document.cases[0].id}`]: "tokenizer_field_value_mismatch",
  });
});

test("unreadable fixtures, missing checkers and broken checks are failures, not skips", async () => {
  const kit = await fixtureEntry("kit_parity_vectors");
  const failing: FixtureReader = async (path) => {
    if (path === kit.path) {
      throw new Error("synthetic read failure");
    }
    return diskReader(path);
  };
  const unreadable = only(
    await runWebContractGate({ read: failing, fixtureIds: [kit.id] }),
    kit.id,
  );
  assert.deepEqual(unreadable.errors, ["fixture_unreadable"]);
  assert.deepEqual([unreadable.passed, unreadable.failed, unreadable.skipped], [[], [], []]);
  assert.equal(unreadable.fixture_sha256, "0".repeat(64));

  const garbled = only(
    await runWebContractGate({
      read: async (path) =>
        path === kit.path ? new TextEncoder().encode("{not json") : diskReader(path),
      fixtureIds: [kit.id],
    }),
    kit.id,
  );
  assert.deepEqual(garbled.errors, ["fixture_unreadable"]);
  assert.equal(garbled.fixture_sha256, sha256Hex("{not json"));

  const missing = only(await runWebContractGate({ checkers: {}, fixtureIds: [kit.id] }), kit.id);
  assert.ok(missing.failed.length > 0);
  assert.ok(missing.failed.every((row) => row.reason === "consumer_missing"));

  const broken = only(
    await runWebContractGate({
      checkers: {
        kit_parity_vectors: () => ({
          groups: {
            vectors: () => {
              throw new Error("synthetic failure with a value");
            },
            rejections: () => {
              throw new ContractCaseFailure("Not A Code");
            },
            source_rejections: () => {
              throw new ContractCaseFailure("synthetic_reason");
            },
          },
        }),
      },
      fixtureIds: [kit.id],
    }),
    kit.id,
  );
  const brokenReasons = new Map<string, Set<string>>();
  for (const row of broken.failed) {
    const group = row.case_id.split("/")[0]!;
    brokenReasons.set(group, (brokenReasons.get(group) ?? new Set()).add(row.reason));
  }
  assert.deepEqual(
    Object.fromEntries([...brokenReasons].map(([group, codes]) => [group, [...codes]])),
    {
      vectors: ["consumer_error"],
      rejections: ["consumer_error"],
      source_rejections: ["synthetic_reason"],
      offset_rejections: ["case_not_exercised"],
      compatibility: ["case_not_exercised"],
    },
  );
  assert.deepEqual(broken.passed, []);

  const unprepared = only(
    await runWebContractGate({
      checkers: {
        kit_parity_vectors: () => {
          throw new Error("synthetic preparation failure");
        },
      },
      fixtureIds: [kit.id],
    }),
    kit.id,
  );
  assert.ok(unprepared.failed.every((row) => row.reason === "consumer_error"));

  const combined: ContractGateResult = {
    ...(await committedResult()),
    fixtures: [unreadable, garbled, missing, broken, unprepared].map((item, index) => ({
      ...item,
      fixture_id: `fixture_${index}`,
    })),
  };
  assert.deepEqual(validateContractGateResult(combined, await loadResultSchema()), []);
});

test("the schema validator rejects values, extra fields and unknown keywords", async () => {
  const schema = await loadResultSchema();
  const result = await committedResult();

  const withText = structuredClone(result);
  withText.fixtures[0]!.failed = [
    { case_id: "cases/Patient José", reason: "consumer_failed" },
    { case_id: "cases/x", reason: "Offset differs at 3" },
  ];
  assert.deepEqual(validateContractGateResult(withText, schema), [
    "result_schema_violation path=$.fixtures[0].failed[0].case_id keyword=pattern",
    "result_schema_violation path=$.fixtures[0].failed[1].reason keyword=pattern",
  ]);

  const extra = structuredClone(result) as unknown as Record<string, any>;
  extra.fixtures[0].message = "assertion text";
  extra.generated_at = "2026-11-16";
  assert.deepEqual(validateContractGateResult(extra, schema).sort(), [
    "result_schema_violation path=$ keyword=additionalProperties",
    "result_schema_violation path=$.fixtures[0] keyword=additionalProperties",
  ]);

  const repeatedFixture = structuredClone(result);
  repeatedFixture.fixtures.push(structuredClone(result.fixtures[0]!));
  assert.deepEqual(validateContractGateResult(repeatedFixture, schema), [
    `fixture_reported_twice path=$.fixtures[${result.fixtures.length}]`,
  ]);

  const repeatedCase = structuredClone(result);
  const first = repeatedCase.fixtures[0]!;
  first.failed = [{ case_id: first.passed[0]!, reason: "consumer_failed" }];
  assert.deepEqual(validateContractGateResult(repeatedCase, schema), [
    "case_reported_twice path=$.fixtures[0]",
  ]);

  assert.throws(() => validateContractGateResult(result, { ...schema, oneOf: [] }), {
    name: "ContractGateError",
    code: "result_schema_unsupported",
  });
});

test("the reporter command writes web.json where it is told and nowhere else", async () => {
  const scratch = await mkdtemp(join(tmpdir(), "openmed-contract-gate-cli-"));
  const lines: string[] = [];
  const errors: string[] = [];
  const io = (env: Record<string, string | undefined>) => ({
    env,
    cwd: scratch,
    out: (line: string) => lines.push(line),
    err: (line: string) => errors.push(line),
  });
  try {
    assert.equal(await main([], io({})), 2);
    assert.match(errors.at(-1)!, new RegExp(RESULTS_DIR_ENV));
    assert.equal(await main(["--unknown", "x"], io({})), 2);
    assert.equal(await main(["--results-dir"], io({})), 2);
    assert.equal(resultPath({}, {}), undefined);
    assert.equal(resultPath({ resultsDir: "out" }, {}, scratch), join(scratch, "out", "web.json"));
    assert.equal(resultPath({}, { [RESULTS_DIR_ENV]: "env" }, scratch), join(scratch, "env", "web.json"));
    assert.equal(resultPath({ output: "x.json" }, { [RESULTS_DIR_ENV]: "env" }, scratch), join(scratch, "x.json"));

    const fixture = ["--fixture", "kit_parity_vectors"];
    assert.equal(await main([...fixture, "--results-dir", "relative", "--check"], io({})), 0);
    const relative = JSON.parse(await readFile(join(scratch, "relative", "web.json"), "utf8"));
    assert.deepEqual(relative.fixtures.map((item: FixtureResult) => item.fixture_id), [
      "kit_parity_vectors",
    ]);

    assert.equal(await main(fixture, io({ [RESULTS_DIR_ENV]: join(scratch, "env") })), 0);
    await readFile(join(scratch, "env", "web.json"), "utf8");

    const missing = { ...io({}), run: { checkers: {} } };
    assert.equal(await main([...fixture, "--output", "out.json", "--check"], missing), 1);
    assert.equal(await main([...fixture, "--output", "out.json"], missing), 0);

    const noManifest = { ...io({}), run: { manifest: "tests/fixtures/parity/absent.json" } };
    assert.equal(await main(["--output", "absent.json"], noManifest), 2);
    assert.equal(errors.at(-1), "contract_gate web: manifest_unreadable");

    for (const line of [...lines, ...errors]) {
      assert.match(
        line,
        /^contract_gate web: (?:fixture=[a-z0-9_]+ passed=\d+ failed=\d+ skipped=\d+|wrote web\.json|[a-z_]+|pass --output .*|usage: .*)$/,
      );
    }
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
});

test("the manifest pins this reporter as the web consumer of each required fixture", async () => {
  const { manifest: parsed } = await manifest();
  assert.equal(DEFAULT_MANIFEST, "tests/fixtures/parity/contract_gate.json");
  for (const entry of parsed.fixtures) {
    const cell = entry.runtimes[WEB_RUNTIME];
    assert.ok(cell, entry.id);
    if (cell.status === "required") {
      assert.ok(cell.consumers?.includes(SPEC), entry.id);
    } else {
      assert.equal(Object.hasOwn(WEB_CONTRACT_CHECKERS, entry.id), false, entry.id);
    }
  }
});
