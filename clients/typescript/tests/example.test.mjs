import assert from "node:assert/strict";
import { test } from "node:test";

import { run } from "../examples/deidentify.mjs";

const BASE_URL = "http://test.example";
const SYNTHETIC_NOTE = "Patient Jane Doe, MRN 0000000, was seen on 2020-01-01.";

function stubFetch(status, body, contentType = "application/json") {
  const calls = [];
  const fetch = async (url, init) => {
    calls.push({ url, init });
    return new Response(typeof body === "string" ? body : JSON.stringify(body), {
      status,
      headers: { "Content-Type": contentType },
    });
  };
  return { calls, fetch };
}

function errorBody(code, message) {
  return { error: { code, message, details: [] } };
}

test("success path returns the redacted text from the stubbed service", async () => {
  const { calls, fetch } = stubFetch(200, {
    original_text: SYNTHETIC_NOTE,
    deidentified_text: "Patient [NAME], MRN [ID], was seen on [DATE].",
    pii_entities: [],
    method: "mask",
    timestamp: "2020-01-01T00:00:00Z",
    num_entities_redacted: 3,
    metadata: {},
    audit_report: null,
  });

  const outcome = await run({ fetch, baseUrl: BASE_URL, text: SYNTHETIC_NOTE });

  assert.deepEqual(outcome, {
    ok: true,
    deidentifiedText: "Patient [NAME], MRN [ID], was seen on [DATE].",
    redactedCount: 3,
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${BASE_URL}/pii/deidentify`);
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    text: SYNTHETIC_NOTE,
    method: "mask",
  });
});

test("error path reports a validation error by code", async () => {
  const { calls, fetch } = stubFetch(
    422,
    errorBody("validation_error", "Request validation failed"),
  );

  const outcome = await run({ fetch, baseUrl: BASE_URL, text: "   " });

  assert.deepEqual(outcome, {
    ok: false,
    code: "validation_error",
    status: 422,
    retry: false,
  });
  assert.equal(calls.length, 1);
});

test("retryable service codes are marked for retry", async () => {
  for (const [status, code] of [
    [429, "rate_limited"],
    [503, "service_busy"],
    [504, "timeout"],
  ]) {
    const { fetch } = stubFetch(status, errorBody(code, "Synthetic failure"));
    assert.deepEqual(await run({ fetch, baseUrl: BASE_URL }), {
      ok: false,
      code,
      status,
      retry: true,
    });
  }
});

test("the example reads the error code, not the message", async () => {
  // Negative control: the message names a retryable condition, the code does not.
  const { fetch } = stubFetch(
    422,
    errorBody("validation_error", "rate_limited service_busy timeout"),
  );
  assert.equal((await run({ fetch, baseUrl: BASE_URL })).retry, false);

  const reversed = stubFetch(503, errorBody("service_busy", "validation_error"));
  assert.equal((await run({ fetch: reversed.fetch, baseUrl: BASE_URL })).retry, true);
});

test("a non-envelope error body falls back to the http_error code", async () => {
  const { fetch } = stubFetch(502, "Bad Gateway", "text/plain");
  assert.deepEqual(await run({ fetch, baseUrl: BASE_URL }), {
    ok: false,
    code: "http_error",
    status: 502,
    retry: false,
  });
});

test("transport failures that are not service errors are rethrown", async () => {
  const failure = new TypeError("synthetic transport failure");
  await assert.rejects(
    run({
      fetch: async () => {
        throw failure;
      },
      baseUrl: BASE_URL,
    }),
    (error) => error === failure,
  );
});
