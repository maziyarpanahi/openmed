import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  acknowledgeNotice, CapabilityReviewError, deidentify, disclaimerById, disclaimerForCapability,
  disclaimerForTier, regulatoryObligations, requireCapabilityAccess,
} from "../../js/openmedkit-web/src/index";

const vectors = JSON.parse(readFileSync(new URL("../fixtures/parity/safety_primitives_v1.json", import.meta.url), "utf8"));
const instant = new Date(vectors.acknowledged_at);

test("shared notice, obligation, alias and receipt vectors match Python", () => {
  for (const row of vectors.tiers) {
    assert.equal(disclaimerForTier(row.tier).id, row.notice_id);
    assert.deepEqual([...regulatoryObligations(row.tier)].sort(), row.obligations);
  }
  for (const row of vectors.notices) {
    const notice = disclaimerById(row.id);
    assert.ok(notice);
    const { acknowledgement, ...expected } = row;
    assert.deepEqual(notice, expected);
    assert.deepEqual(acknowledgeNotice(notice, instant), acknowledgement);
    assert.ok(Object.isFrozen(notice));
  }
  for (const [capability, id] of Object.entries({ ...vectors.capability_notices, ...vectors.entry_point_overrides })) {
    assert.equal(disclaimerForCapability(capability).id, id);
  }
});

test("restricted access checks a matching current notice and explicit mode", () => {
  const notice = disclaimerForCapability("decisions");
  const ack = acknowledgeNotice(notice, instant)!;
  const denied = (code: string) => (error: unknown) => error instanceof CapabilityReviewError && error.code === code;
  assert.throws(() => requireCapabilityAccess("decisions", { regulatoryTier: "non_clinical", acknowledgement: ack }), denied("clinician_mode_required"));
  assert.throws(() => requireCapabilityAccess("decisions", { clinicianMode: true }), denied("acknowledgement_required"));
  for (const invalid of [{ ...ack, version_digest: "stale" }, { ...ack, notice_id: "other" }, { ...ack, acknowledged_at: "invalid" }]) {
    assert.throws(() => requireCapabilityAccess("decisions", { clinicianMode: true, acknowledgement: invalid }), denied("invalid_acknowledgement"));
  }
  assert.equal(requireCapabilityAccess("decisions", { clinicianMode: true, acknowledgement: ack }), notice);
  assert.equal(disclaimerForCapability("future_capability").tier, "restricted");
});

test("deidentify refuses before inference and returns notices without accessing storage", async () => {
  const old = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { configurable: true, get() { throw Error("storage accessed"); } });
  try {
    let calls = 0;
    const pipeline = async () => { calls++; return []; };
    await assert.rejects(deidentify("synthetic", { pipeline, review: { regulatoryTier: "restricted" } }), { code: "clinician_mode_required" });
    assert.equal(calls, 0);
    const acknowledgement = acknowledgeNotice(disclaimerForTier("restricted"), instant);
    const result = await deidentify("synthetic", { pipeline, review: { regulatoryTier: "restricted", clinicianMode: true, acknowledgement } });
    assert.equal(result.notice.tier, "restricted");
    assert.equal(calls, 1);
  } finally {
    if (old) Object.defineProperty(globalThis, "localStorage", old);
    else Reflect.deleteProperty(globalThis, "localStorage");
  }
});
