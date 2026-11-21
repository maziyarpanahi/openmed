import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { effectiveRegulatoryTier, getIntendedUse, regulatoryObligations } from "../../js/openmedkit-web/src/index";

test("bundled declarations match every reviewed Python capability offline", () => {
  const source = JSON.parse(readFileSync(new URL("../../openmed/core/data/intended_use.json", import.meta.url), "utf8"));
  const copy = JSON.parse(readFileSync(new URL("../../js/openmedkit-web/src/intended-use.json", import.meta.url), "utf8"));
  assert.deepEqual(Object.keys(copy).sort(), Object.keys(source).sort());
  for (const code of Object.keys(source)) {
    assert.deepEqual(copy[code], source[code].declaration);
    const declaration = getIntendedUse(code);
    assert.ok(declaration);
    const { obligations, ...wire } = declaration;
    assert.deepEqual(wire, source[code].declaration);
    assert.deepEqual(obligations, regulatoryObligations(wire.regulatory_tier));
  }
  assert.equal(getIntendedUse("constructor"), undefined);
  assert.equal(getIntendedUse(""), undefined);
});

test("unfamiliar tier codes require all restricted obligations", () => {
  assert.equal(effectiveRegulatoryTier("future"), "restricted");
  assert.deepEqual(regulatoryObligations("future"), [
    "notice_required", "acknowledgement_required", "clinician_mode_required", "crisis_routing_required",
  ]);
  assert.deepEqual(regulatoryObligations(undefined), getIntendedUse("ecg")?.obligations);
});
