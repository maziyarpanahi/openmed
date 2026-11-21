import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { CrisisRouting } from "../../js/openmedkit-web/src/index";

function file(path: string) {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

test("all Python locale vectors return identical guidance without fetching", (t) => {
  t.mock.method(globalThis, "fetch", () => assert.fail("unexpected network request"));
  const vectors = JSON.parse(file("../fixtures/parity/crisis_routing_v1.json"));
  assert.equal(file("../../js/openmedkit-web/src/crisis-resources.json"), file("../../openmed/clinical/data/crisis_resources.json"));
  const router = new CrisisRouting();
  for (const row of vectors.cases) {
    const notice = router.route(row.locale, true);
    assert.ok(notice?.mandatory);
    assert.equal(notice.text, vectors.generic_notice);
    assert.equal(notice.matchedLocale, row.matched);
    assert.deepEqual(notice.resources.map(r => r.id), row.ids);
  }
  assert.equal(router.route("en-GB", false), null);
});

test("host overrides apply by locale and retain verified source metadata", () => {
  const row = {id:"site", label:"Site resource", url:"https://example.org/info", source_url:"https://example.org/source", verified_on:"2025-01-12"};
  const table = {schema_version:1, locales:{"en-AU":[row], fr:[row]}};
  const router = new CrisisRouting(table);
  assert.equal(router.route("fr-CA", true)?.matchedLocale, "fr");
  assert.deepEqual(router.route("en-AU", true)?.resources[0], row);
  assert.equal(router.route("en-GB", true)?.resources[0]?.id, "nhs_emergency_departments");
  for (const change of [{url:"tel:private"}, {source_url:null}, {verified_on:"2025-02-30"}]) {
    assert.throws(() => new CrisisRouting({schema_version:1,locales:{en:[{...row,...change}]}}), /^Error: invalid_crisis_table$/);
  }
});
