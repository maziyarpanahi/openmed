import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { test } from "node:test";
import { webcrypto } from "node:crypto";

globalThis.crypto ??= webcrypto;
const { OpenMedClient, OpenMedApiError } = await import(pathToFileURL(resolve(process.env.OPENMED_WORKFLOW_CLIENT_DIST ?? fileURLToPath(new URL("../dist", import.meta.url)), "index.js")));
const fixture = JSON.parse(await readFile(new URL("../../../tests/fixtures/service/decision-negotiation-v1.json", import.meta.url), "utf8"));

// The shared POST /v1/decisions exchanges also replayed by the Python and Go clients.
for (const vector of fixture.cases) test(vector.name, async () => {
  const calls = [];
  const client = new OpenMedClient({
    baseUrl: "http://test.example",
    fetch: async (url, init) => {
      calls.push({ url, init });
      const body = vector.status === 200 ? vector.response : { error: { ...vector.error, message: "Synthetic refusal" } };
      return new Response(JSON.stringify(body), { status: vector.status, headers: { "Content-Type": "application/json" } });
    },
  });
  const method = vector.shape === "openai_decisions" ? "openaiDecisions" : "decision";
  if (vector.status === 200) {
    assert.deepEqual(await client[method](vector.request), vector.response);
  } else {
    await assert.rejects(client[method](vector.request), (error) => {
      assert.ok(error instanceof OpenMedApiError);
      assert.equal(error.status, vector.status);
      assert.equal(error.code, vector.error.code);
      assert.deepEqual(error.details, vector.error.details);
      return true;
    });
  }
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://test.example/v1/decisions");
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].init.body), vector.request);
});

test("every negotiation outcome is replayed", () => {
  const outcomes = new Set(fixture.cases.map((vector) => vector.status === 200 ? vector.shape : vector.error.code));
  for (const outcome of ["openmed", "openai_decisions", "decision_request_ambiguous", "decision_media_unsupported", "model_not_served"]) {
    assert.ok(outcomes.has(outcome), outcome);
  }
});
