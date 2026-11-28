import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
  BASELINE_PATH,
  PACKAGE_DIR,
  compareApi,
  renderApi,
} from "../scripts/api-baseline.mjs";

const INDEX = "src/index.ts";
const GENERATED = "src/journey-workflows.generated.ts";
const baseline = readFileSync(BASELINE_PATH, "utf8");
const source = (path) => readFileSync(join(PACKAGE_DIR, path), "utf8");

/** Replace `from` with `to` inside one exported interface of `text`. */
function inInterface(text, name, from, to) {
  const start = text.indexOf(`export interface ${name} {`);
  assert.ok(start >= 0, `${name} is missing`);
  const end = text.indexOf("\n}\n", start);
  const body = text.slice(start, end);
  assert.ok(body.includes(from), `${name} has no ${from}`);
  return text.slice(0, start) + body.replace(from, to) + text.slice(end);
}

test("the committed baseline matches the public declarations", () => {
  assert.equal(renderApi(), baseline);
  assert.deepEqual(compareApi(renderApi()), { ok: true, line: null });
});

test("the baseline records every declaration file, including the Journey types", () => {
  for (const file of [
    "index.d.ts",
    "journey-workflows.generated.d.ts",
    "workflow-client.d.ts",
  ]) {
    assert.ok(baseline.includes(`\n// ${file}\n`), file);
  }
  assert.match(baseline, /export type JourneyWorkflowName = /);
  assert.match(baseline, /export declare class OpenMedClient \{/);
  assert.match(baseline, /export interface AnalyzeRequest \{\n {4}text: string;/);
});

test("changing a request field type fails the check", () => {
  const rendered = renderApi({
    overrides: {
      [INDEX]: inInterface(source(INDEX), "AnalyzeRequest", "  text: string;", "  text: number;"),
    },
  });

  assert.notEqual(rendered, baseline);
  const result = compareApi(rendered);
  assert.equal(result.ok, false);
  assert.ok(rendered.split("\n")[result.line - 1].includes("text: number;"));
});

test("changing a response field type or removing a field fails the check", () => {
  const changed = inInterface(
    source(INDEX),
    "PIIDeidentifyResponse",
    "  num_entities_redacted: number;",
    "  num_entities_redacted: string;",
  );
  const removed = inInterface(
    source(INDEX),
    "PIIDeidentifyResponse",
    "\n  mapping?: Record<string, string>;",
    "",
  );

  for (const text of [changed, removed]) {
    assert.equal(compareApi(renderApi({ overrides: { [INDEX]: text } })).ok, false);
  }
});

test("a generated Journey type change fails the check", () => {
  const text = source(GENERATED);
  const changed = text.replace(/export type JourneyWorkflowName =/, "export type JourneyWorkflowNameRenamed =");
  assert.notEqual(changed, text);

  // An alias keeps the client that imports the old name type-checking, so the
  // only difference is the renamed generated declaration.
  const rendered = renderApi({
    overrides: {
      [GENERATED]: `${changed}\nexport type JourneyWorkflowName = JourneyWorkflowNameRenamed;\n`,
    },
  });
  assert.equal(compareApi(rendered).ok, false);
});

test("comment-only edits keep the baseline stable", () => {
  const commented = source(INDEX).replace(
    "export interface AnalyzeRequest {",
    "/** Synthetic doc comment that must not change the API baseline. */\nexport interface AnalyzeRequest {",
  );

  assert.equal(renderApi({ overrides: { [INDEX]: commented } }), baseline);
});
