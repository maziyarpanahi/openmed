// CommonJS consumers must receive the same packed build as ESM consumers.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const openmed = require("openmed");

const consumerRoot = path.resolve(__dirname, "..");
const contract = JSON.parse(
  readFileSync(path.join(consumerRoot, "consumer_contract.json"), "utf8"),
);

test("require resolves the packed CommonJS entry point", () => {
  const resolved = require.resolve("openmed");
  const installRoot = path.join(consumerRoot, "node_modules", "openmed") + path.sep;
  assert.ok(resolved.startsWith(installRoot), "resolved_outside_install");
  assert.ok(resolved.endsWith("index.cjs"), "cjs_entry");
});

test("CommonJS and ESM expose the same public surface", async () => {
  const esm = await import("openmed");
  assert.deepEqual(
    Object.keys(openmed).sort(),
    Object.keys(esm)
      .filter((key) => key !== "default")
      .sort(),
  );
  assert.equal(openmed.DEFAULT_MODEL_ID, esm.DEFAULT_MODEL_ID);
  assert.equal(openmed.OPENMED_SPAN_SCHEMA_VERSION, esm.OPENMED_SPAN_SCHEMA_VERSION);
});

test("CommonJS deidentify redacts the synthetic session", async () => {
  const divergence = contract.kit_divergences.npm_web;
  const redactionCase = contract.redaction_cases.find(
    (candidate) => candidate.id === divergence.redaction_case,
  );
  const pipeline = () =>
    redactionCase.entities.map((entity, index) => ({
      entity: `B-${entity.label}`,
      word: redactionCase.text.slice(entity.utf16_start, entity.utf16_end),
      start: entity.utf16_start,
      end: entity.utf16_end,
      score: entity.confidence,
      index,
    }));
  const result = await openmed.deidentify(redactionCase.text, {
    pipeline,
    hashSecret: "fixture-secret",
  });
  assert.equal(result.deidentifiedText, divergence.expected_redacted, redactionCase.id);
});
