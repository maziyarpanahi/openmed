import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { Worker as NodeWorker } from "node:worker_threads";

import {
  INFERENCE_WORKER_ERROR_CODES,
  InferenceWorkerError,
  createInferenceWorkerClient,
  deidentify,
  extractPii,
  serveInferenceWorker,
  type InferenceWorkerClient,
  type InferenceWorkerEndpoint,
  type InferenceWorkerErrorCode,
  type InferenceWorkerEventListener,
  type InferenceWorkerLoadSpec,
  type InferenceWorkerPipeline,
  type InferenceWorkerRuntime,
  type InferenceWorkerScope,
  type InferenceWorkerServer,
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
  type RawTokenClassificationOutput,
  type ServeInferenceWorkerOptions,
  type TokenClassificationEntity,
} from "../../js/openmedkit-web/src/index";
import {
  decodeTextUnits,
  encodeInferenceTokens,
} from "../../js/openmedkit-web/src/worker/protocol";

// Synthetic sentinel: it must never appear in errors, events, or worker output.
const SENTINEL = "Zyxwv Sentinel-PHI 4242";
const CRASH_MARKER = "[crash-worker]";
const LOCAL_SPEC: InferenceWorkerLoadSpec = {
  modelRef: "/models/openmed/synthetic-token-classifier",
  allowedBackends: ["webgpu", "wasm-basic"],
};
const DIGEST = "sha256:synthetic-model";

interface ParityVector {
  name: string;
  text: string;
  tokens?: Record<string, unknown>[];
  batches?: Record<string, unknown>[][];
}

const fixturesDir = fileURLToPath(new URL("./fixtures/", import.meta.url));
const parity = JSON.parse(
  readFileSync(`${fixturesDir}inference_worker_parity_vectors.json`, "utf8"),
) as { hashSecret: string; vectors: ParityVector[] };
const pythonGolden = JSON.parse(
  readFileSync(`${fixturesDir}npm_deidentify_golden.json`, "utf8"),
) as OpenMedDeidentifyResult;

const vectorPipeline: InferenceWorkerPipeline = (text) => {
  const vector = parity.vectors.find((candidate) => candidate.text === text);
  return structuredClone(
    vector?.batches ?? vector?.tokens ?? [],
  ) as RawTokenClassificationOutput;
};

test("admission is bounded and overflow rejects with a typed queue_full code", async () => {
  const worker = new FakeWorker();
  const client = createInferenceWorkerClient({
    createWorker: () => worker,
    maxQueueSize: 2,
    maxInFlight: 1,
  });
  try {
    await loadFake(client, worker);
    const first = client.run(`${SENTINEL} one`);
    const second = client.run(`${SENTINEL} two`);
    const third = client.run(`${SENTINEL} three`);
    const overflow = await rejection(client.run(`${SENTINEL} four`));

    assertWorkerError(overflow, "queue_full");
    assert.deepEqual(
      pick(client.stats(), ["queued", "inFlight", "rejectedQueueFull"]),
      { queued: 2, inFlight: 1, rejectedQueueFull: 1 },
    );
    assert.equal(worker.messages("infer").length, 1);

    for (const pending of [first, second, third]) {
      const infer = worker.lastMessage("infer");
      worker.result(infer, [{ entity: "O", start: 0, end: 4, score: 1 }]);
      assert.deepEqual(await pending, [{ entity: "O", start: 0, end: 4, score: 1 }]);
    }
    assert.equal(worker.messages("infer").length, 3);
    assert.equal(client.stats().completed, 3);
  } finally {
    await client.dispose();
  }
});

test("cancelling a queued request never sends its text to the worker", async () => {
  const worker = new FakeWorker();
  const client = createInferenceWorkerClient({ createWorker: () => worker });
  try {
    await loadFake(client, worker);
    const active = client.run("synthetic active request");
    const controller = new AbortController();
    const queued = client.run(`${SENTINEL} queued`, { signal: controller.signal });
    const next = client.run("synthetic next request");

    controller.abort();
    assertWorkerError(await rejection(queued), "cancelled");
    assert.equal(client.stats().queued, 1);

    worker.result(worker.lastMessage("infer"), []);
    assert.deepEqual(await active, []);
    worker.result(worker.lastMessage("infer"), []);
    assert.deepEqual(await next, []);

    const sentTexts = worker
      .messages("infer")
      .map((message) => decodeTextUnits(message.input as ArrayBuffer));
    assert.deepEqual(sentTexts, ["synthetic active request", "synthetic next request"]);
    assert.equal(worker.messages("cancel").length, 0);
  } finally {
    await client.dispose();
  }
});

test("in-flight cancellation reaches the worker and late results are suppressed", async () => {
  const worker = new FakeWorker();
  const client = createInferenceWorkerClient({ createWorker: () => worker });
  try {
    await loadFake(client, worker);
    const controller = new AbortController();
    let delivered = false;
    const cancelled = client
      .run("synthetic in-flight request", { signal: controller.signal })
      .then((tokens) => {
        delivered = true;
        return tokens;
      });
    const waiting = client.run("synthetic waiting request");
    const cancelledInfer = worker.lastMessage("infer");

    controller.abort();
    assertWorkerError(await rejection(cancelled), "cancelled");
    assert.deepEqual(worker.lastMessage("cancel").id, cancelledInfer.id);
    // The cancelled request keeps its slot until the worker settles it.
    assert.deepEqual(pick(client.stats(), ["inFlight", "draining", "queued"]), {
      inFlight: 1,
      draining: 1,
      queued: 1,
    });
    assert.equal(worker.messages("infer").length, 1);

    worker.result(cancelledInfer, [{ entity: "B-NAME", start: 0, end: 9, score: 1 }]);
    await flush();
    assert.equal(delivered, false);
    assert.equal(client.stats().suppressedLateResults, 1);
    assert.equal(worker.messages("infer").length, 2);

    worker.result(worker.lastMessage("infer"), []);
    assert.deepEqual(await waiting, []);
  } finally {
    await client.dispose();
  }
});

test("the worker aborts a running pipeline and posts cancelled instead of a result", async () => {
  const started = deferred<AbortSignal>();
  const release = deferred<void>();
  const channel = connectWorker({
    load: () =>
      runtime(async (_text, _options, context) => {
        started.resolve(context?.signal as AbortSignal);
        await release.promise;
        return [{ entity: "B-NAME", start: 0, end: 4, score: 1 }];
      }),
  });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  try {
    await client.load(LOCAL_SPEC);
    const controller = new AbortController();
    const request = client.run("synthetic running request", {
      signal: controller.signal,
    });
    const signal = await started.promise;
    assert.equal(signal.aborted, false);

    controller.abort();
    assertWorkerError(await rejection(request), "cancelled");
    await waitFor(() => signal.aborted);
    release.resolve();
    await waitFor(() => channel.posted.some((message) => message.type === "cancelled"));

    assert.equal(countPosted(channel, "result"), 0);
    assert.equal(channel.server.stats().cancelled, 1);
    await waitFor(() => client.stats().inFlight === 0);
    assert.equal(client.stats().suppressedLateResults, 0);
  } finally {
    await client.dispose();
    channel.close();
  }
});

test("a new model generation supersedes in-flight work and stales old pipelines", async () => {
  const started = deferred<void>();
  let loads = 0;
  const channel = connectWorker({
    load: () => {
      loads += 1;
      if (loads === 1) {
        return runtime((_text, _options, context) => {
          started.resolve();
          return abortable(context);
        });
      }
      return runtime(vectorPipeline);
    },
  });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  try {
    const firstPlan = await client.load(LOCAL_SPEC);
    const firstPipeline = client.pipeline();
    const inFlight = rejection(client.run("synthetic first-generation request"));
    await started.promise;

    const secondPlan = await client.load({ ...LOCAL_SPEC, modelDigest: DIGEST });
    assertWorkerError(await inFlight, "superseded");
    assert.equal(firstPlan.generation, 1);
    assert.equal(secondPlan.generation, 2);
    assertWorkerError(
      await rejection(Promise.resolve(firstPipeline(parity.vectors[0]!.text))),
      "stale_generation",
    );

    const tokens = await client.pipeline()(parity.vectors[0]!.text);
    assert.equal((tokens as TokenClassificationEntity[]).length, 4);
    assert.equal(countPosted(channel, "result"), 1);
  } finally {
    await client.dispose();
    channel.close();
  }
});

test("GPU device loss invalidates the plan until a verified local-only recovery", async () => {
  const lost = deferred<void>();
  const started = deferred<void>();
  const loadCalls: Array<{ localOnly: boolean; reason: string }> = [];
  let disposals = 0;
  let recoveryDigest = DIGEST;
  let recoveryLocalOnly = true;
  const channel = connectWorker({
    load: (spec, context) => {
      loadCalls.push({ localOnly: spec.localOnly, reason: context.reason });
      if (context.reason === "load") {
        return {
          ...runtime((text, _options, runContext) => {
            if (text === "synthetic stalled request") {
              started.resolve();
              return abortable(runContext);
            }
            return vectorPipeline(text);
          }, "webgpu"),
          localOnly: false,
          modelDigest: DIGEST,
          deviceLost: lost.promise,
          dispose: () => {
            disposals += 1;
          },
        };
      }
      return {
        ...runtime(vectorPipeline, "wasm-basic"),
        localOnly: recoveryLocalOnly,
        modelDigest: recoveryDigest,
      };
    },
  });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  try {
    // A first load may download artifacts; recovery must never widen access.
    const plan = await client.load({ ...LOCAL_SPEC, localOnly: false });
    assert.deepEqual(pick(plan, ["backend", "localOnly", "modelDigest"]), {
      backend: "webgpu",
      localOnly: false,
      modelDigest: DIGEST,
    });
    const pipeline = client.pipeline();
    const stalled = client.run("synthetic stalled request");
    await started.promise;

    lost.resolve();
    assertWorkerError(await rejection(stalled), "device_lost");
    assert.equal(client.state, "invalidated");
    assertWorkerError(
      await rejection(client.run(parity.vectors[0]!.text)),
      "plan_invalidated",
    );
    await waitFor(() => disposals === 1);
    assert.equal(channel.server.stats().loaded, false);

    recoveryDigest = "sha256:different-model";
    assertWorkerError(await rejection(client.recover()), "plan_incompatible");
    assert.equal(client.state, "invalidated");
    recoveryDigest = DIGEST;
    recoveryLocalOnly = false;
    assertWorkerError(await rejection(client.recover()), "plan_incompatible");
    assert.equal(client.state, "invalidated");

    recoveryLocalOnly = true;
    const recovered = await client.recover();
    assert.deepEqual(
      pick(recovered, ["generation", "backend", "localOnly", "modelDigest"]),
      { generation: 1, backend: "wasm-basic", localOnly: true, modelDigest: DIGEST },
    );
    assert.deepEqual(loadCalls, [
      { localOnly: false, reason: "load" },
      { localOnly: true, reason: "recover" },
      { localOnly: true, reason: "recover" },
      { localOnly: true, reason: "recover" },
    ]);
    // The generation survived a verified compatible recovery, so its pipeline works.
    const spans = await extractPii(parity.vectors[0]!.text, {
      pipeline,
      hashSecret: parity.hashSecret,
    });
    assert.equal(spans.length, 3);
  } finally {
    await client.dispose();
    channel.close();
  }
});

test("host-side invalidation requires recovery before reuse", async () => {
  const channel = connectWorker({ load: () => runtime(vectorPipeline) });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  try {
    await client.load({ ...LOCAL_SPEC, modelDigest: DIGEST });
    const runtimeReplacedPlan = client.plan;
    client.invalidate();
    assert.equal(client.plan, null);
    assertWorkerError(
      await rejection(client.run(parity.vectors[0]!.text)),
      "plan_invalidated",
    );
    const recovered = await client.recover();
    assert.equal(recovered.generation, runtimeReplacedPlan?.generation);
    assert.ok(recovered.epoch > (runtimeReplacedPlan?.epoch ?? 0));
    assert.equal((await client.run(parity.vectors[0]!.text)).length, 4);
  } finally {
    await client.dispose();
    channel.close();
  }
});

test("worker failure hides the worker error and recovery restarts a verified worker", async () => {
  const workers: FakeWorker[] = [];
  const client = createInferenceWorkerClient({
    createWorker: () => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker;
    },
  });
  try {
    const loading = client.load({ ...LOCAL_SPEC, modelDigest: DIGEST });
    workers[0]!.ready(workers[0]!.lastMessage("load"), { modelDigest: DIGEST });
    await loading;
    const outcomes = [
      client.run(`${SENTINEL} crash`),
      client.run(`${SENTINEL} queued`),
    ].map(rejection);
    let prevented = false;
    workers[0]!.emit("error", {
      message: `Uncaught Error: ${SENTINEL}`,
      preventDefault: () => {
        prevented = true;
      },
    } as never);

    for (const outcome of outcomes) {
      assertWorkerError(await outcome, "worker_failed");
    }
    assert.equal(prevented, true);
    assert.equal(workers[0]!.terminated, true);
    assert.equal(client.state, "failed");
    assertWorkerError(await rejection(client.run("synthetic")), "worker_failed");

    const recovery = client.recover();
    assert.equal(workers.length, 2);
    const load = workers[1]!.lastMessage("load");
    assert.equal(load.reason, "recover");
    assert.equal((load.spec as { localOnly: boolean }).localOnly, true);
    assert.equal((load.spec as { modelDigest: string }).modelDigest, DIGEST);
    workers[1]!.ready(load, { modelRef: "/models/other" });
    assertWorkerError(await rejection(recovery), "plan_incompatible");

    const retry = client.recover({ restart: true });
    assert.equal(workers.length, 3);
    workers[2]!.ready(workers[2]!.lastMessage("load"), { modelDigest: DIGEST });
    assert.equal((await retry).generation, 1);
    const tokens = client.run("synthetic after restart");
    workers[2]!.result(workers[2]!.lastMessage("infer"), []);
    assert.deepEqual(await tokens, []);

    // A stuck runtime can be replaced outright when loading the next model.
    const reloading = client.load(
      { ...LOCAL_SPEC, modelDigest: DIGEST },
      { restart: true },
    );
    assert.equal(workers.length, 4);
    assert.equal(workers[2]!.terminated, true);
    workers[3]!.ready(workers[3]!.lastMessage("load"), { modelDigest: DIGEST });
    assert.equal((await reloading).generation, 2);
    assert.equal(client.stats().workerStarts, 4);
  } finally {
    await client.dispose();
  }
});

test("dispose tears down queued and in-flight work and releases the runtime", async () => {
  const started = deferred<void>();
  let disposals = 0;
  const channel = connectWorker({
    load: () => ({
      ...runtime((_text, _options, context) => {
        started.resolve();
        return abortable(context);
      }),
      dispose: () => {
        disposals += 1;
      },
    }),
  });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  await client.load(LOCAL_SPEC);
  const outcomes = [
    client.run("synthetic in-flight request"),
    client.run("synthetic queued request"),
  ].map(rejection);
  await started.promise;

  await client.dispose();
  for (const outcome of outcomes) {
    assertWorkerError(await outcome, "disposed");
  }
  assert.equal(channel.terminated, true);
  await waitFor(() => disposals === 1 && countPosted(channel, "disposed") === 1);
  assert.deepEqual(pick(channel.server.stats(), ["loaded", "queued", "running"]), {
    loaded: false,
    queued: 0,
    running: false,
  });
  assertWorkerError(await rejection(client.run("synthetic")), "disposed");
  assertWorkerError(await rejection(client.load(LOCAL_SPEC)), "disposed");
  assert.equal(client.state, "disposed");
  channel.close();
});

test("worker errors, events and output never carry input text", async () => {
  const consoleCalls = spyOnConsole();
  const storageAccess = guardStorage();
  const channel = connectWorker({
    load: () =>
      runtime((text) => {
        if (text.includes("throw")) {
          throw new Error(`pipeline failed on ${text}`);
        }
        if (text.includes("misaligned")) {
          return [{ entity: "B-NAME", word: "absent-token", score: 1 }];
        }
        return [
          {
            entity: "B-NAME",
            word: SENTINEL,
            start: text.indexOf(SENTINEL),
            end: text.indexOf(SENTINEL) + SENTINEL.length,
            score: 0.9,
            index: 0,
          },
        ];
      }),
  });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  try {
    await client.load(LOCAL_SPEC);
    const failure = await rejection(client.run(`throw ${SENTINEL}`));
    assertWorkerError(failure, "inference_failed");
    const misaligned = await rejection(client.run(`misaligned ${SENTINEL}`));
    assertWorkerError(misaligned, "alignment_failed");
    const tooLarge = await rejection(
      createInferenceWorkerClient({
        createWorker: () => new FakeWorker(),
        maxInputChars: 8,
      }).run(SENTINEL),
    );
    assertWorkerError(tooLarge, "input_too_large");

    const tokens = await client.run(`Patient ${SENTINEL} seen.`);
    assert.deepEqual(tokens, [
      { entity: "B-NAME", start: 8, end: 8 + SENTINEL.length, score: 0.9, index: 0 },
    ]);
    // Results carry offsets, labels and scores; token surface text stays in the worker.
    assert.equal(channel.hostTransfers.length, 3);
    assert.ok(channel.hostTransfers.every((buffer) => buffer.byteLength === 0));

    for (const error of [failure, misaligned, tooLarge]) {
      assertNoSentinel(error);
    }
    assertNoSentinel(channel.posted);
    assertNoSentinel(channel.server.stats());
    assertNoSentinel(client.stats());
    assert.deepEqual(consoleCalls.calls, []);
    assert.deepEqual(storageAccess.accessed, []);
  } finally {
    consoleCalls.restore();
    storageAccess.restore();
    await client.dispose();
    channel.close();
  }
});

test("direct and worker-routed extraction return identical spans on the parity vectors", async () => {
  const workerTexts: string[] = [];
  const channel = connectWorker({
    load: () =>
      runtime((text, options, context) => {
        workerTexts.push(text);
        return vectorPipeline(text, options, context);
      }),
  });
  const client = createInferenceWorkerClient({ createWorker: () => channel.endpoint });
  try {
    await client.load(LOCAL_SPEC);
    assert.ok(parity.vectors.length >= 5);
    for (const vector of parity.vectors) {
      const options = { hashSecret: parity.hashSecret, docId: vector.name };
      const direct = await extractPii(vector.text, {
        ...options,
        pipeline: vectorPipeline,
      });
      const routed = await extractPii(vector.text, {
        ...options,
        pipeline: client.pipeline(),
      });
      assert.ok(direct.length > 0, vector.name);
      assert.deepEqual(routed, direct, vector.name);
    }
    // The UTF-16 transfer is lossless, lone surrogates included.
    assert.deepEqual(
      workerTexts,
      parity.vectors.map((vector) => vector.text),
    );

    const golden = parity.vectors[0]!;
    const goldenOptions = {
      docId: "web-fixture",
      hashSecret: "test-secret",
      detector: "fixture-token-classifier",
      metadata: { fixture: "python-generated" },
    };
    const routed = await deidentify(golden.text, {
      ...goldenOptions,
      pipeline: client.pipeline(),
    });
    const direct = await deidentify(golden.text, {
      ...goldenOptions,
      pipeline: vectorPipeline,
    });
    assert.deepEqual(routed, direct);
    assert.equal(routed.deidentifiedText, pythonGolden.deidentifiedText);
    assertSpansClose(routed.spans, pythonGolden.spans);
  } finally {
    await client.dispose();
    channel.close();
  }
});

test("a real worker thread keeps parity, isolates crashes and restarts on recovery", async () => {
  const threads: NodeWorker[] = [];
  const client = createInferenceWorkerClient({
    createWorker: () => {
      const thread = new NodeWorker(
        new URL("./fixtures/inference_worker_thread.ts", import.meta.url),
      );
      threads.push(thread);
      return nodeWorkerEndpoint(thread);
    },
  });
  try {
    const plan = await client.load({
      ...LOCAL_SPEC,
      modelDigest: "sha256:synthetic-thread-fixture",
    });
    assert.equal(plan.localOnly, true);
    await assertParity(client);

    const crash = await rejection(client.run(`${CRASH_MARKER} ${SENTINEL}`));
    assertWorkerError(crash, "worker_failed");
    assertNoSentinel(crash);
    await once(threads[0]!, "exit");

    const recovered = await client.recover();
    assert.equal(recovered.generation, plan.generation);
    assert.equal(threads.length, 2);
    await assertParity(client);

    const exited = once(threads[1]!, "exit");
    await client.dispose();
    await exited;
  } finally {
    await client.dispose();
    for (const thread of threads) {
      await thread.terminate();
    }
  }
});

test("malformed traffic and requests fail closed with controlled codes", async () => {
  const worker = new FakeWorker();
  const client = createInferenceWorkerClient({ createWorker: () => worker });
  try {
    assertWorkerError(await rejection(client.run("synthetic")), "not_loaded");
    assertWorkerError(
      await rejection(client.load({ modelRef: "https://models.example.invalid/m" })),
      "invalid_request",
    );
    assertWorkerError(
      await rejection(client.load({ ...LOCAL_SPEC, options: { hook: () => 1 } })),
      "invalid_request",
    );
    assert.equal(worker.posted.length, 0);
    assertWorkerError(await rejection(client.recover()), "not_loaded");

    await loadFake(client, worker);
    assertWorkerError(
      await rejection(
        client.run("synthetic", {
          callOptions: { signal: new AbortController().signal },
        }),
      ),
      "invalid_request",
    );
    const inFlight = client.run("synthetic");
    worker.reply({ protocol: 99, type: "result" });
    assertWorkerError(await rejection(inFlight), "protocol_error");
    assert.equal(client.state, "failed");
    assert.equal(worker.terminated, true);
    assert.deepEqual(
      [...INFERENCE_WORKER_ERROR_CODES].sort(),
      [...INFERENCE_WORKER_ERROR_CODES],
    );
  } finally {
    await client.dispose();
  }
});

test("the worker server bounds its own queue and rejects stale epochs", async () => {
  const releases: Array<() => void> = [];
  const channel = connectWorker({
    maxQueueSize: 1,
    maxInputChars: 32,
    load: () =>
      runtime(
        () =>
          new Promise<RawTokenClassificationOutput>((resolve) => {
            releases.push(() => resolve([]));
          }),
      ),
  });
  const client = createInferenceWorkerClient({
    createWorker: () => channel.endpoint,
    maxInFlight: 3,
  });
  try {
    await client.load(LOCAL_SPEC);
    const running = client.run("synthetic running");
    const waiting = client.run("synthetic waiting");
    const overflow = client.run("synthetic overflow");
    assertWorkerError(await rejection(overflow), "queue_full");
    await waitFor(() => releases.length === 1);
    releases.shift()?.();
    assert.deepEqual(await running, []);
    await waitFor(() => releases.length === 1);
    releases.shift()?.();
    assert.deepEqual(await waiting, []);

    // The worker enforces its own input bound even when the host allows more.
    assertWorkerError(
      await rejection(client.run(`${SENTINEL} ${"x".repeat(32)}`)),
      "input_too_large",
    );
    assert.equal(client.stats().inFlight, 0);

    channel.endpoint.postMessage(
      {
        protocol: 1,
        type: "infer",
        id: 999,
        epoch: 99,
        input: new ArrayBuffer(2),
        callOptions: {},
      },
      [],
    );
    await waitFor(() => channel.posted.some((message) => message.id === 999));
    assert.equal(
      channel.posted.find((message) => message.id === 999)?.code,
      "stale_generation",
    );
  } finally {
    await client.dispose();
    channel.close();
  }
});

// ---------------------------------------------------------------------------
// Helpers

async function assertParity(client: InferenceWorkerClient): Promise<void> {
  for (const vector of parity.vectors) {
    const options = { hashSecret: parity.hashSecret, docId: vector.name };
    const direct = await extractPii(vector.text, {
      ...options,
      pipeline: vectorPipeline,
    });
    const routed = await extractPii(vector.text, {
      ...options,
      pipeline: client.pipeline(),
    });
    assert.deepEqual(routed, direct, vector.name);
  }
}

function runtime(
  pipeline: InferenceWorkerPipeline,
  backend = "wasm-basic",
): InferenceWorkerRuntime {
  return { pipeline, backend, localOnly: true, modelDigest: DIGEST };
}

function abortable(
  context: { signal: AbortSignal } | undefined,
): Promise<RawTokenClassificationOutput> {
  return new Promise((_resolve, reject) => {
    context?.signal.addEventListener("abort", () => reject(new Error("aborted")), {
      once: true,
    });
  });
}

type PostedMessage = Record<string, unknown>;

class FakeWorker implements InferenceWorkerEndpoint {
  readonly posted: PostedMessage[] = [];
  terminated = false;
  private readonly listeners = new Map<string, Set<InferenceWorkerEventListener>>();

  postMessage(message: unknown): void {
    this.posted.push(message as PostedMessage);
  }

  addEventListener(type: string, listener: InferenceWorkerEventListener): void {
    const set = this.listeners.get(type) ?? new Set();
    set.add(listener);
    this.listeners.set(type, set);
  }

  removeEventListener(type: string, listener: InferenceWorkerEventListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(type: string, event: Parameters<InferenceWorkerEventListener>[0]): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener(event);
    }
  }

  reply(data: unknown): void {
    this.emit("message", { data });
  }

  messages(type: string): PostedMessage[] {
    return this.posted.filter((message) => message.type === type);
  }

  lastMessage(type: string): PostedMessage {
    const messages = this.messages(type);
    const last = messages[messages.length - 1];
    assert.ok(last, `no ${type} message posted`);
    return last;
  }

  ready(load: PostedMessage, overrides: Record<string, unknown> = {}): void {
    const spec = load.spec as Record<string, unknown>;
    this.reply({
      protocol: 1,
      type: "ready",
      id: load.id,
      epoch: load.epoch,
      plan: {
        protocol: 1,
        generation: load.generation,
        epoch: load.epoch,
        modelRef: spec.modelRef,
        backend: "wasm-basic",
        localOnly: true,
        modelDigest: null,
        ...overrides,
      },
    });
  }

  result(infer: PostedMessage, tokens: TokenClassificationEntity[]): void {
    this.reply({
      protocol: 1,
      type: "result",
      id: infer.id,
      epoch: infer.epoch,
      tokens: encodeInferenceTokens(tokens).tokens,
    });
  }
}

async function loadFake(
  client: InferenceWorkerClient,
  worker: FakeWorker,
  overrides: Record<string, unknown> = {},
): Promise<void> {
  const loading = client.load({ ...LOCAL_SPEC, ...overrides });
  worker.ready(worker.lastMessage("load"), overrides);
  await loading;
}

interface ConnectedWorker {
  endpoint: InferenceWorkerEndpoint;
  server: InferenceWorkerServer;
  /** Worker-to-host messages, serialised before transfer for inspection. */
  posted: PostedMessage[];
  /** Input buffers the host transferred to the worker. */
  hostTransfers: ArrayBuffer[];
  terminated: boolean;
  close(): void;
}

/** Runs the real worker server behind a MessageChannel (clone and transfer). */
function connectWorker(
  options: Omit<ServeInferenceWorkerOptions, "scope">,
): ConnectedWorker {
  const { port1, port2 } = new MessageChannel();
  const connected: ConnectedWorker = {
    posted: [],
    hostTransfers: [],
    terminated: false,
    endpoint: {
      postMessage(message, transfer) {
        for (const item of transfer) {
          if (item instanceof ArrayBuffer) {
            connected.hostTransfers.push(item);
          }
        }
        port1.postMessage(message, transfer as never);
      },
      addEventListener: (type, listener) =>
        port1.addEventListener(type, listener as never),
      removeEventListener: (type, listener) =>
        port1.removeEventListener(type, listener as never),
      terminate() {
        connected.terminated = true;
      },
    },
    server: undefined as unknown as InferenceWorkerServer,
    close() {
      port1.close();
      port2.close();
    },
  };
  const scope: InferenceWorkerScope = {
    postMessage(message, transfer) {
      connected.posted.push(JSON.parse(JSON.stringify(message, typedArrayReplacer)));
      port2.postMessage(message, transfer as never);
    },
    addEventListener: (type, listener) =>
      port2.addEventListener(type, listener as never),
    removeEventListener: (type, listener) =>
      port2.removeEventListener(type, listener as never),
    close: () => port2.close(),
  };
  connected.server = serveInferenceWorker({ ...options, scope });
  port1.start();
  port2.start();
  return connected;
}

function countPosted(channel: ConnectedWorker, type: string): number {
  return channel.posted.filter((message) => message.type === type).length;
}

function nodeWorkerEndpoint(thread: NodeWorker): InferenceWorkerEndpoint {
  const handlers = new Map<InferenceWorkerEventListener, (value: unknown) => void>();
  return {
    postMessage: (message, transfer) => thread.postMessage(message, transfer as never),
    addEventListener(type, listener) {
      const handler =
        type === "message"
          ? (data: unknown) => listener({ data })
          : (error: unknown) =>
              listener({ data: error, preventDefault: () => undefined });
      handlers.set(listener, handler);
      thread.on(type, handler);
    },
    removeEventListener(type, listener) {
      const handler = handlers.get(listener);
      if (handler !== undefined) {
        thread.off(type, handler);
      }
    },
    terminate: () => void thread.terminate(),
  };
}

function typedArrayReplacer(_key: string, value: unknown): unknown {
  return ArrayBuffer.isView(value)
    ? Array.from(value as unknown as ArrayLike<number>)
    : value;
}

function assertWorkerError(error: unknown, code: InferenceWorkerErrorCode): void {
  assert.ok(error instanceof InferenceWorkerError, `expected ${code}`);
  assert.equal(error.code, code);
  assert.equal(error.name, "InferenceWorkerError");
  assert.equal(error.cause, undefined);
  assertNoSentinel(error);
}

function assertNoSentinel(value: unknown): void {
  const rendered =
    value instanceof Error
      ? [value.name, value.message, value.stack ?? "", String(value.cause ?? "")].join(
          "\n",
        )
      : JSON.stringify(value, typedArrayReplacer);
  assert.ok(!rendered.includes(SENTINEL), "input text leaked");
  assert.ok(!rendered.includes("Zyxwv"), "input text leaked");
}

function spyOnConsole(): { calls: unknown[][]; restore(): void } {
  const calls: unknown[][] = [];
  const methods = ["log", "info", "warn", "error", "debug", "trace"] as const;
  const originals = methods.map((method) => console[method]);
  for (const method of methods) {
    console[method] = (...args: unknown[]) => {
      calls.push([method, ...args]);
    };
  }
  return {
    calls,
    restore() {
      methods.forEach((method, index) => {
        console[method] = originals[index] as (typeof console)[typeof method];
      });
    },
  };
}

/** Negative control: no browser storage API may be touched by the worker path. */
function guardStorage(): { accessed: string[]; restore(): void } {
  const accessed: string[] = [];
  const names = ["localStorage", "sessionStorage", "indexedDB", "caches"];
  const originals = names.map((name) =>
    Object.getOwnPropertyDescriptor(globalThis, name),
  );
  for (const name of names) {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      get() {
        accessed.push(name);
        return undefined;
      },
    });
  }
  return {
    accessed,
    restore() {
      names.forEach((name, index) => {
        const original = originals[index];
        if (original === undefined) {
          delete (globalThis as Record<string, unknown>)[name];
        } else {
          Object.defineProperty(globalThis, name, original);
        }
      });
    },
  };
}

function assertSpansClose(actual: OpenMedSpan[], expected: OpenMedSpan[]): void {
  assert.equal(actual.length, expected.length);
  for (const [index, actualSpan] of actual.entries()) {
    const expectedSpan = expected[index];
    assert.ok(expectedSpan, `missing expected span at ${index}`);
    const { score: actualScore, ...actualRest } = actualSpan;
    const { score: expectedScore, ...expectedRest } = expectedSpan;
    assert.deepEqual(actualRest, expectedRest);
    assert.ok(actualScore !== null && expectedScore !== null);
    assert.ok(Math.abs(actualScore - expectedScore) <= 1e-12);
  }
}

function pick<T extends object, K extends keyof T>(value: T, keys: K[]): Pick<T, K> {
  return Object.fromEntries(keys.map((key) => [key, value[key]])) as Pick<T, K>;
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  assert.fail("expected the promise to reject");
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

function once(thread: NodeWorker, event: "exit"): Promise<void> {
  return new Promise((resolve) => thread.once(event, () => resolve()));
}

async function flush(): Promise<void> {
  for (let turn = 0; turn < 5; turn += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

/** Polls by event-loop turns (no wall-clock thresholds). */
async function waitFor(predicate: () => boolean, turns = 500): Promise<void> {
  for (let turn = 0; turn < turns; turn += 1) {
    if (predicate()) {
      return;
    }
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail("condition not reached");
}
