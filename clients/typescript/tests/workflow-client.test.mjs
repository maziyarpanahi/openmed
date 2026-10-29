import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { test } from "node:test";
import { webcrypto } from "node:crypto";

globalThis.crypto ??= webcrypto;
const { OpenMedClient, WorkflowClientError, parseWorkflowResponse } = await import(pathToFileURL(resolve(process.env.OPENMED_WORKFLOW_CLIENT_DIST ?? fileURLToPath(new URL("../dist", import.meta.url)), "index.js")));
const vectors = JSON.parse(await readFile(new URL("../../../tests/fixtures/service/workflow-clients-v1.json", import.meta.url), "utf8"));
const reference = vectors.cases[0].reference;
const good = vectors.receipt_response_json;
const response = (body = good, status = 200) => new Response(body, { status, headers: { "Content-Type": "application/json", "X-Request-ID": "req_" + "9".repeat(32) } });

for (const vector of vectors.cases) test(vector.name, async () => {
  const calls = [];
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async (url, init) => { calls.push({ url, init }); return response(vector.body, vector.status); } });
  const operation = vector.operation === "cancel" ? "workflowCancel" : "workflowStatus";
  if (vector.code) {
    await assert.rejects(client[operation](vector.reference), error => {
      assert.ok(error instanceof WorkflowClientError);
      assert.equal(error.code, vector.code);
      assert.equal(error.status, vector.status);
      assert.equal(error.mutationOutcome, vector.mutation_outcome);
      assert.equal(error.requestId, "req_" + "9".repeat(32));
      assert.ok(!String(error).includes("PRIVATE-CANARY"));
      assert.ok(!JSON.stringify(error).includes("PRIVATE-CANARY"));
      assert.ok(!("details" in error) && !("envelope" in error) && !("cause" in error));
      return true;
    });
  } else {
    const value = await client[operation](vector.reference);
    assert.deepEqual(JSON.parse(JSON.stringify(value)), vector.expected);
    assert.ok(Object.isFrozen(value) && Object.isFrozen(value.effects));
    assert.deepEqual(JSON.parse(JSON.stringify(await parseWorkflowResponse(new TextEncoder().encode(vector.body), vector.reference))), vector.expected);
  }
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://test.example/v1/workflows/" + vector.operation);
  assert.deepEqual(JSON.parse(calls[0].init.body), vector.reference);
  assert.equal(calls[0].init.redirect, "manual");
});

// v2 receipt metadata carries no role or timestamp authority.
const review = JSON.parse(vectors.receipt_request_json);
const { receipt, ...mutation } = review;
test("existing receipt retains exact v2 metadata", async () => {
  const calls = [];
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async (_, init) => { calls.push(init); return response(); } });
  await client.workflowSubmitReceipt(review);
  assert.equal(calls[0].body, vectors.receipt_request_json);
  await assert.rejects(client.workflowSubmitReceipt({ ...review, receipt: { ...receipt, expires_at: BigInt("9223372036854775807") } }), error => error.code === "workflow_invalid_request" && error.mutationOutcome === "not_attempted");
  await assert.rejects(client.workflowCancel(reference), error => error.mutationOutcome === "not_attempted");
  assert.equal(calls.length, 1);
});
test("only explicit inspection retries", async () => {
  for (const code of ["workflow_unavailable", "workflow_service_failed", "workflow_conflict", "workflow_forbidden", "workflow_mutation_unknown"]) {
    let calls = 0;
    const vector = vectors.cases.find(v => v.name === code);
    const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => ++calls === 1 ? response(vector.body, vector.status) : response() });
    if (["workflow_unavailable", "workflow_service_failed"].includes(code)) {
      await client.workflowStatus(reference, { maxAttempts: 3 }); assert.equal(calls, 2);
    } else { await assert.rejects(client.workflowStatus(reference, { maxAttempts: 3 })); assert.equal(calls, 1); }
    calls = 0; await assert.rejects(client.workflowCancel(mutation)); assert.equal(calls, 1);
  }
});
test("transport failure preserves unknown mutation", async () => {
  let calls = 0;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => { calls++; throw new Error("PRIVATE-CANARY"); } });
  await assert.rejects(client.workflowCancel(mutation), error => error.code === "workflow_mutation_unknown" && error.mutationOutcome === "unknown" && !String(error).includes("PRIVATE-CANARY"));
  assert.equal(calls, 1); calls = 0;
  await assert.rejects(client.workflowStatus(reference, { maxAttempts: 3 })); assert.equal(calls, 3);
});
test("bounded polling stops for review and terminal phases", async () => {
  for (const phase of ["waiting-review", "completed", "aborted"]) {
    const bodies = ["running", phase].map(p => vectors.cases.find(v => v.name === "phase-" + p).body);
    const calls = [];
    const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async url => { calls.push(url); return response(bodies.shift()); } });
    assert.equal((await client.pollWorkflow(reference, { maxRequests: 3, intervalMs: 0, clock: () => 0 })).phase, phase);
    assert.deepEqual(calls, Array(2).fill("http://test.example/v1/workflows/status"));
  }
});
test("poll request limit retains final snapshot without unnecessary delay", async () => {
  let calls = 0;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => { calls++; return response(vectors.cases.find(v => v.name === "phase-running").body); } });
  await assert.rejects(client.pollWorkflow(reference, { maxRequests: 1, intervalMs: 30000, clock: () => 0 }), error => error.code === "workflow_poll_timeout" && error.lastView.phase === "running");
  assert.equal(calls, 1);
});
test("cancel before dispatch and cancel an injected fetch that ignores AbortSignal", async () => {
  const before = new AbortController(); before.abort(); let calls = 0;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: () => { calls++; return new Promise(() => {}); } });
  await assert.rejects(client.workflowCancel(mutation, { signal: before.signal }), error => error.mutationOutcome === "not_attempted");
  assert.equal(calls, 0);
  const during = new AbortController();
  const pending = client.pollWorkflow(reference, { signal: during.signal });
  during.abort();
  await assert.rejects(pending, error => error.code === "workflow_poll_cancelled");
  assert.equal(calls, 1);
  await assert.rejects(client.workflowCancel(mutation, { timeoutMs: 5 }), error => error.code === "workflow_mutation_unknown" && error.mutationOutcome === "unknown");
  assert.equal(calls, 2);
});
test("clock deadline, rollback and invalid policy fail before transport", async () => {
  let calls = 0;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => { calls++; return response(); } });
  for (const [times, code] of [[[0, 100000], "workflow_poll_timeout"], [[1, 0], "workflow_clock_failed"], [[NaN], "workflow_clock_failed"]])
    await assert.rejects(client.pollWorkflow(reference, { clock: () => times.shift() }), error => error.code === code);
  for (const options of [{ maxRequests: 101 }, { timeoutMs: 300001 }, { intervalMs: -1 }]) await assert.rejects(client.pollWorkflow(reference, options), error => error.code === "workflow_invalid_request");
  assert.equal(calls, 0);
});
test("byte, UTF8, node, and depth bounds", async () => {
  for (const raw of [new Uint8Array(262145), new Uint8Array([255]), new TextEncoder().encode(JSON.stringify({ value: Array(5000).fill(0) }))])
    await assert.rejects(parseWorkflowResponse(raw, reference), error => error instanceof WorkflowClientError);
});
test("reference is captured before an asynchronous transport runs", async () => {
  const request = { ...reference };
  let release;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: () => new Promise(resolve => { release = resolve; }) });
  const pending = client.workflowStatus(request);
  request.action_digest = "sha256:" + "a".repeat(64);
  release(response());
  assert.equal((await pending).action_digest, reference.action_digest);
});
test("stream byte bound and cancellation do not leave a poll running", async () => {
  const stop = new AbortController(); let calls = 0;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => {
    calls++; return new Response(new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(262145)); } }), { headers: { "Content-Type": "application/json" } });
  } });
  await assert.rejects(client.workflowCancel(mutation), error => error.code === "workflow_response_too_large" && error.mutationOutcome === "unknown");
  assert.equal(calls, 1);
  const pendingClient = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => {
    calls++; return new Response(new ReadableStream({ start() { setTimeout(() => stop.abort(), 1); } }), { headers: { "Content-Type": "application/json" } });
  } });
  await assert.rejects(pendingClient.pollWorkflow(reference, { signal: stop.signal }), error => error.code === "workflow_poll_cancelled");
  assert.equal(calls, 2);
});
test("poll transport refusal preserves the last validated phase", async () => {
  let calls = 0;
  const vector = vectors.cases.find(v => v.name === "workflow_conflict");
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => ++calls === 1 ? response(vectors.cases.find(v => v.name === "phase-running").body) : response(vector.body, vector.status) });
  await assert.rejects(client.pollWorkflow(reference, { intervalMs: 0, clock: () => 0 }), error => error.code === "workflow_conflict" && error.lastView.phase === "running");
  assert.equal(calls, 2);
});

for (const operation of ["receipt", "cancel"]) test("successful " + operation + " reply acknowledges the requested mutation", async () => {
    const body=JSON.parse(good); body.receipt_digest="sha256:"+"0".repeat(64);
    let calls=0;
    const client=new OpenMedClient({baseUrl:"http://test.example",fetch:async()=>{calls++;return response(JSON.stringify(body));}});
    await assert.rejects(operation === "receipt" ? client.workflowSubmitReceipt(review) : client.workflowCancel(mutation),error=>error instanceof WorkflowClientError && error.mutationOutcome === "unknown");
    assert.equal(calls,1);
});

test("transport diagnostic cannot declare a mutation refused", async () => {
  let calls=0;
  const client=new OpenMedClient({baseUrl:"http://test.example",fetch:async()=>{calls++;throw new WorkflowClientError("workflow_forbidden",{mutationOutcome:"refused"});}});
  await assert.rejects(client.workflowCancel(mutation),error=>error.mutationOutcome === "unknown");
  assert.equal(calls,1);
});

test("public error metadata drops private values",()=>{
  const error=new WorkflowClientError("workflow_invalid_request",{status:"PRIVATE-CANARY",mutationOutcome:"PRIVATE-CANARY",lastView:{clinical:"PRIVATE-CANARY"}});
  assert.ok(!JSON.stringify(error).includes("PRIVATE-CANARY"));
});


test("transport error accessors cannot leak private diagnostics",async()=>{
  const error=new WorkflowClientError("workflow_transport_failed");
  Object.defineProperty(error,"code",{get(){throw new Error("PRIVATE-CANARY");}});
  let calls=0;
  const client=new OpenMedClient({baseUrl:"http://test.example",fetch:async()=>{calls++;throw error;}});
  await assert.rejects(client.workflowCancel(mutation),failure=>failure instanceof WorkflowClientError && failure.mutationOutcome === "unknown" && !String(failure).includes("PRIVATE-CANARY"));
  assert.equal(calls,1);
});

test("transport exception proxies cannot expose private diagnostics", async () => {
  const privateError = new Proxy({}, { getPrototypeOf() { throw new Error("PRIVATE-CANARY"); } });
  let calls = 0;
  const client = new OpenMedClient({ baseUrl: "http://test.example", fetch: async () => { calls++; throw privateError; } });
  await assert.rejects(client.workflowCancel(mutation), error => error instanceof WorkflowClientError && error.mutationOutcome === "unknown" && !String(error).includes("PRIVATE-CANARY"));
  assert.equal(calls, 1);
});
