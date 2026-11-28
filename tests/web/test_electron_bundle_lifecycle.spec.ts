/**
 * Verified-bundle lifecycle of the OpenMedKit Electron integration: bundle
 * verification inside the utility process, generation changes, cancellation,
 * utility-process crash and restart, renderer teardown and content-free
 * diagnostics. Uses the shared portable-bundle golden vectors (TEST-ONLY keys)
 * and a synthetic pipeline; no model weights and no network.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  deidentify,
  parsePortableBundleTrustPolicy,
  type TokenClassificationPipeline,
} from "../../js/openmedkit-web/src/index";
import {
  ElectronBundleActivationError,
  ElectronDeidentifyError,
  ElectronDeidentifyService,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  createElectronDeidentifyClient,
  registerElectronDeidentifyIpc,
  verifyPortableBundleDirectory,
  type ElectronBundleSource,
  type ElectronDeidentifyServiceEvent,
  type RendererOpenMedSpan,
} from "../../js/openmedkit-electron/src/index";
import {
  createUtilityDeidentifyHandler,
  type VerifiedElectronBundle,
} from "../../js/openmedkit-electron/src/utility-runtime";
import {
  FakeIpcMain,
  FakeOfflineSession,
  FakeSender,
  FakeUtilityProcessModule,
  deferred,
  expectedVerification,
  materializeBundle,
  until,
  vectorCase,
  vectorTarget,
  vectorsPath,
  workerPath,
  type PortableVectors,
} from "./electron_test_support";

const vectors = JSON.parse(readFileSync(vectorsPath, "utf8")) as PortableVectors;
const WEB_CONTEXT = vectors.contexts.valid_web;
assert.ok(WEB_CONTEXT);
const GOLDEN_ID = vectorCase(vectors, "valid_web").expected.content_id as string;
const EXTENSION_ID = vectorCase(vectors, "valid_critical_extension_understood").expected
  .content_id as string;
const HOLD = "[[synthetic-hold]]";
const NOTE = "Patient Qorvane Tilsby arrived for review.";

let scratch = "";
let goldenRoot = "";
let extensionRoot = "";
let tamperedRoot = "";
test.before(async () => {
  scratch = await mkdtemp(join(tmpdir(), "openmed-electron-lifecycle-"));
  goldenRoot = await materializeBundle(vectors, vectorCase(vectors, "valid_web"), scratch);
  extensionRoot = await materializeBundle(
    vectors,
    vectorCase(vectors, "valid_critical_extension_understood"),
    scratch,
  );
  tamperedRoot = await materializeBundle(
    vectors,
    vectorCase(vectors, "file_modified"),
    scratch,
  );
});
test.after(async () => {
  await rm(scratch, { recursive: true, force: true });
});

function source(root: string, overrides: Partial<ElectronBundleSource> = {}): ElectronBundleSource {
  return {
    root,
    trustPolicy: vectors.trust_policies.default,
    target: vectorTarget(WEB_CONTEXT as NonNullable<typeof WEB_CONTEXT>),
    ...(root === extensionRoot ? { understoodExtensions: ["org.openmed.test.note"] } : {}),
    ...overrides,
  };
}

/** Synthetic tagger: names after "Patient", gated by an optional hold marker. */
class SyntheticModel {
  readonly gates: Array<{ promise: Promise<void>; resolve(): void }> = [];
  readonly inferredTexts: number[] = [];
  readonly loadedBundles: string[] = [];

  hold(): () => void {
    const gate = deferred();
    this.gates.push(gate);
    return () => gate.resolve();
  }

  pipeline(): TokenClassificationPipeline {
    return async (text) => {
      this.inferredTexts.push(text.length);
      if (text.includes(HOLD)) {
        const gate = this.gates.shift();
        assert.ok(gate, "a hold gate must be registered");
        await gate.promise;
      }
      return tagNames(text);
    };
  }

  async load(bundle: VerifiedElectronBundle): Promise<TokenClassificationPipeline> {
    assert.ok(bundle.files.get("id2label.json"), "verified label map present");
    assert.ok(bundle.files.get(bundle.runtime.entrypoint), "verified entrypoint present");
    this.loadedBundles.push(bundle.contentId);
    return this.pipeline();
  }
}

function tagNames(text: string) {
  const match = /Patient (\p{Lu}\p{Ll}+) (\p{Lu}\p{Ll}+)/u.exec(text);
  if (match === null || match[1] === undefined || match[2] === undefined) {
    return [];
  }
  const first = match.index + "Patient ".length;
  const second = first + match[1].length + 1;
  return [
    { entity: "B-NAME", score: 0.99, start: first, end: first + match[1].length, index: 1 },
    { entity: "I-NAME", score: 0.97, start: second, end: second + match[2].length, index: 2 },
  ];
}

function setup(options: { roots?: string; logger?: ElectronDeidentifyServiceEvent[] } = {}) {
  const model = new SyntheticModel();
  const terminal: Array<{ child: number; requestId: string }> = [];
  const utilityProcess = new FakeUtilityProcessModule(() => {
    const childIndex = utilityProcess.children.length;
    const handler = createUtilityDeidentifyHandler({
      loadBundlePipeline: (bundle) => model.load(bundle),
    });
    return async (message) => {
      const response = await handler(message);
      if (response?.type === "deidentify-result") {
        terminal.push({ child: childIndex, requestId: response.requestId });
      }
      return response;
    };
  });
  const events = options.logger ?? [];
  const service = new ElectronDeidentifyService({
    utilityProcess,
    offlineSession: new FakeOfflineSession(),
    workerPath,
    bundle: source(options.roots ?? goldenRoot),
    now: () => new Date(WEB_CONTEXT?.now as string),
    logger: (event) => events.push(event),
  });
  return { model, utilityProcess, service, events, terminal };
}

function request(requestId: string, text: string) {
  return { schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION, requestId, text } as const;
}

async function typedRejection(promise: Promise<unknown>): Promise<ElectronDeidentifyError> {
  try {
    await promise;
  } catch (error) {
    assert.ok(error instanceof ElectronDeidentifyError);
    return error;
  }
  return assert.fail("expected a typed rejection");
}

async function directReference(text: string): Promise<RendererOpenMedSpan[]> {
  const result = await deidentify(text, {
    pipeline: async (input) => tagNames(input),
    docId: "electron-document",
    detector: "electron-utility-process",
  });
  return result.spans.map((span) => ({
    schema_version: span.schema_version,
    start: span.start,
    end: span.end,
    entity_type: span.entity_type,
    canonical_label: span.canonical_label,
    policy_label: span.policy_label,
    score: span.score,
  }));
}

test("directory reader reproduces every shared file-tree vector", async () => {
  const cases = vectors.cases.filter((item) => item.tree !== null);
  assert.ok(cases.length >= 40);
  for (const item of cases) {
    const root = await materializeBundle(vectors, item, join(scratch, "vectors"));
    const raw = vectors.contexts[item.context];
    assert.ok(raw, item.context);
    const result = await verifyPortableBundleDirectory(root, {
      trustPolicy: parsePortableBundleTrustPolicy(vectors.trust_policies[item.trust_policy]),
      context: {
        now: raw.now,
        target: vectorTarget(raw),
        sequenceFloors: raw.sequence_floors,
        understoodExtensions: raw.understood_extensions,
        maxManifestBytes: raw.max_manifest_bytes,
        maxEnvelopeBytes: raw.max_envelope_bytes,
        maxTotalBytes: raw.max_total_bytes,
        maxFiles: raw.max_files,
      },
    });
    assert.deepEqual(
      { ...result.verification, eligibleRuntimes: [...result.verification.eligibleRuntimes] },
      expectedVerification(item),
      item.id,
    );
    assert.equal(result.files !== null, item.expected.ok, item.id);
  }
});

test("directory reader refuses symlinks, special roots and missing documents", async () => {
  const policy = parsePortableBundleTrustPolicy(vectors.trust_policies.default);
  const context = {
    now: WEB_CONTEXT?.now as string,
    target: vectorTarget(WEB_CONTEXT as NonNullable<typeof WEB_CONTEXT>),
  };
  const linked = await materializeBundle(
    vectors,
    { ...vectorCase(vectors, "valid_web"), id: "symlinked" },
    join(scratch, "negative"),
  );
  await rm(join(linked, "vocab.txt"));
  await symlink(join(goldenRoot, "vocab.txt"), join(linked, "vocab.txt"));
  const linkedResult = await verifyPortableBundleDirectory(linked, { trustPolicy: policy, context });
  assert.equal(linkedResult.verification.reason, "path_unsafe");
  assert.equal(linkedResult.files, null);

  const rootLink = join(scratch, "negative", "root-link");
  await symlink(goldenRoot, rootLink);
  const rootResult = await verifyPortableBundleDirectory(rootLink, { trustPolicy: policy, context });
  assert.equal(rootResult.verification.reason, "file_missing");

  const missing = await verifyPortableBundleDirectory(join(scratch, "absent"), {
    trustPolicy: policy,
    context,
  });
  assert.equal(missing.verification.reason, "file_missing");
  assert.equal(missing.verification.contentId, null);

  const large = await materializeBundle(
    vectors,
    { ...vectorCase(vectors, "valid_web"), id: "large-manifest" },
    join(scratch, "negative"),
  );
  await writeFile(join(large, "openmed-bundle.json"), "x".repeat(64));
  const largeResult = await verifyPortableBundleDirectory(large, {
    trustPolicy: policy,
    context: { ...context, maxManifestBytes: 32 },
  });
  assert.equal(largeResult.verification.reason, "manifest_too_large");
});

test("bundle mode verifies inside the utility process before serving", async () => {
  const { service, model, utilityProcess } = setup();
  try {
    const handle = await service.whenReady();
    assert.equal(handle.generation, 1);
    assert.equal(handle.contentId, GOLDEN_ID);
    assert.equal(handle.runtimeId, "onnx-int8");
    assert.deepEqual(
      { ...handle.verification, eligibleRuntimes: [...handle.verification.eligibleRuntimes] },
      expectedVerification(vectorCase(vectors, "valid_web")),
    );
    assert.deepEqual(model.loadedBundles, [GOLDEN_ID]);
    const posted = utilityProcess.children[0]?.posted[0] as Record<string, unknown>;
    assert.equal(posted.type, "load-bundle");
    assert.equal(posted.now, WEB_CONTEXT?.now);

    const response = await service.deidentify(request("first", NOTE));
    assert.deepEqual(response.bundle, { generation: 1, contentId: GOLDEN_ID });
    assert.deepEqual(response.spans, await directReference(NOTE));
    assert.equal(response.spans.length, 1);
    const deidentifyMessage = utilityProcess.children[0]?.posted[1] as Record<string, unknown>;
    assert.deepEqual(Object.keys(deidentifyMessage).sort(), [
      "generation",
      "requestId",
      "schemaVersion",
      "text",
      "type",
    ]);
    assert.equal(service.diagnostics().activeBundle?.contentId, GOLDEN_ID);
  } finally {
    service.dispose();
  }
});

test("generation change pins in-flight work and serves new work on the new bundle", async () => {
  const { service, model } = setup();
  try {
    await service.whenReady();
    const release = model.hold();
    const inFlight = service.deidentify(request("in-flight", `${HOLD} ${NOTE}`));
    await until(() => model.inferredTexts.length === 1, "first job running");

    const activation = service.activateBundle(source(extensionRoot));
    const next = service.deidentify(request("after-change", NOTE));
    assert.equal(service.diagnostics().generation, 2);
    release();

    const [first, handle, second] = await Promise.all([inFlight, activation, next]);
    assert.deepEqual(first.bundle, { generation: 1, contentId: GOLDEN_ID });
    assert.equal(handle.generation, 2);
    assert.equal(handle.contentId, EXTENSION_ID);
    assert.deepEqual(second.bundle, { generation: 2, contentId: EXTENSION_ID });
    assert.deepEqual(model.loadedBundles, [GOLDEN_ID, EXTENSION_ID]);
    assert.deepEqual(first.spans, await directReference(`${HOLD} ${NOTE}`));
    assert.deepEqual(second.spans, await directReference(NOTE));
  } finally {
    service.dispose();
  }
});

test("a rejected generation fails closed with a value-free reason", async () => {
  const events: ElectronDeidentifyServiceEvent[] = [];
  const { service } = setup({ logger: events });
  try {
    await service.whenReady();
    const rejectedActivation = service.activateBundle(source(tamperedRoot));
    const blocked = service.deidentify(request("blocked", NOTE));
    const error = await rejectedActivation.then(
      () => assert.fail("tampered bundle must not activate"),
      (reason: unknown) => reason,
    );
    assert.ok(error instanceof ElectronBundleActivationError);
    assert.equal(error.code, "BUNDLE_REJECTED");
    assert.equal(error.reason, "file_digest_mismatch");
    assert.deepEqual(
      error.verification && {
        ...error.verification,
        eligibleRuntimes: [...error.verification.eligibleRuntimes],
      },
      expectedVerification(vectorCase(vectors, "file_modified")),
    );
    assert.equal((await typedRejection(blocked)).code, "BUNDLE_UNAVAILABLE");
    assert.equal(service.diagnostics().activeBundle, null);
    const notReady = await service.whenReady().then(
      () => assert.fail("a rejected generation is never ready"),
      (value: unknown) => value,
    );
    assert.ok(notReady instanceof ElectronBundleActivationError);
    assert.equal(notReady.reason, "file_digest_mismatch");

    for (const [overrides, reason] of [
      [{ sequenceFloors: { "openmed.synthetic.tiny-pii": 8 } }, "rollback_rejected"],
      [{ expectedContentId: `sha256:${"0".repeat(64)}` }, "content_id_unexpected"],
      [{ trustPolicy: vectors.trust_policies.key_revoked }, "key_revoked"],
      [{ trustPolicy: { schema: "synthetic" } }, "trust_policy_invalid"],
      [{ target: { ...vectorTarget(WEB_CONTEXT as NonNullable<typeof WEB_CONTEXT>)!, runtimes: ["coreml"] } }, "platform_incompatible"],
    ] as const) {
      const failure = await service.activateBundle(source(goldenRoot, overrides)).then(
        () => assert.fail(reason),
        (value: unknown) => value,
      );
      assert.ok(failure instanceof ElectronBundleActivationError, reason);
      assert.equal(failure.reason, reason);
    }

    const recovered = await service.activateBundle(
      source(goldenRoot, { expectedContentId: GOLDEN_ID }),
    );
    assert.equal(recovered.contentId, GOLDEN_ID);
    const response = await service.deidentify(request("recovered", NOTE));
    assert.deepEqual(response.bundle, { generation: recovered.generation, contentId: GOLDEN_ID });
  } finally {
    service.dispose();
  }
  const rejections = events.filter((event) => event.event === "bundle_rejected");
  assert.equal(rejections.length, 6);
  assert.doesNotMatch(JSON.stringify(events), /Qorvane|Tilsby|\/|\\/);
});

test("a utility-process crash rejects in-flight work once and restarts verified", async () => {
  const events: ElectronDeidentifyServiceEvent[] = [];
  const { service, model, utilityProcess, terminal } = setup({ logger: events });
  try {
    await service.whenReady();
    model.hold();
    let settlements = 0;
    const doomed = service.deidentify(request("doomed", `${HOLD} ${NOTE}`)).finally(() => {
      settlements += 1;
    });
    const queued = service.deidentify(request("queued", NOTE)).finally(() => {
      settlements += 1;
    });
    await until(() => model.inferredTexts.length === 1, "doomed job running");
    const crashed = utilityProcess.children[0];
    assert.ok(crashed);
    crashed.crash();

    assert.equal((await typedRejection(doomed)).code, "UTILITY_PROCESS_EXITED");
    assert.equal((await typedRejection(queued)).code, "UTILITY_PROCESS_EXITED");
    assert.equal(settlements, 2);
    assert.equal(service.diagnostics().pendingRequests, 0);
    assert.equal(service.diagnostics().activeBundle, null);

    crashed.inject({
      type: "deidentify-result",
      requestId: "job-1",
      ok: true,
      spans: [],
      generation: 1,
    });
    assert.equal(service.diagnostics().utilityProcessRunning, false);

    const response = await service.deidentify(request("after-restart", NOTE));
    assert.deepEqual(response.bundle, { generation: 1, contentId: GOLDEN_ID });
    assert.equal(utilityProcess.forkCount, 2);
    assert.equal(service.diagnostics().utilityProcessStarts, 2);
    assert.deepEqual(model.loadedBundles, [GOLDEN_ID, GOLDEN_ID]);
    assert.equal((utilityProcess.children[1]?.posted[0] as { type: string }).type, "load-bundle");
    assert.equal(settlements, 2, "no request settles twice");
    assert.equal(terminal.filter((item) => item.child === 0).length, 0);
  } finally {
    service.dispose();
  }
  assert.equal(events.filter((event) => event.event === "bundle_verified").length, 2);
  assert.equal(events.filter((event) => event.event === "utility_process_exited").length, 1);
});

test("cancellation skips queued work and discards running output", async () => {
  const { service, model, utilityProcess, terminal } = setup();
  try {
    await service.whenReady();
    const release = model.hold();
    const running = service.deidentify(request("running", `${HOLD} ${NOTE}`));
    const queued = service.deidentify(request("queued", NOTE));
    const kept = service.deidentify(request("kept", NOTE));
    await until(() => model.inferredTexts.length === 1, "running job started");

    assert.equal(service.cancel("queued"), true);
    assert.equal(service.cancel("queued"), false);
    assert.equal(service.cancel("unknown"), false);
    assert.equal(service.cancel("running"), true);
    assert.equal((await typedRejection(queued)).code, "CANCELLED");
    assert.equal((await typedRejection(running)).code, "CANCELLED");
    // Let the asynchronous cancel messages reach the utility process first.
    await new Promise<void>((resolve) => setImmediate(resolve));
    await new Promise<void>((resolve) => setImmediate(resolve));
    release();

    const response = await kept;
    assert.deepEqual(response.bundle, { generation: 1, contentId: GOLDEN_ID });
    await until(() => service.diagnostics().cancellingRequests === 0, "cancel acks");
    assert.equal(model.inferredTexts.length, 2, "the queued job never ran");
    assert.equal(utilityProcess.children[0]?.killed, false);
    assert.equal(utilityProcess.forkCount, 1);
    const ids = terminal.map((item) => item.requestId);
    assert.equal(ids.length, new Set(ids).size, "exactly one terminal message per job");
    assert.equal(ids.length, 3);
  } finally {
    service.dispose();
  }
});

test("renderer teardown cancels only that renderer's work", async () => {
  const { service, model } = setup();
  const ipcMain = new FakeIpcMain();
  const windowA = new FakeSender(1);
  const windowB = new FakeSender(2);
  const attacker = new FakeSender(3);
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => [windowA, windowB, attacker].includes(event.sender as FakeSender),
  });
  try {
    await service.whenReady();
    const clientA = createElectronDeidentifyClient(ipcMain.createRenderer(windowA));
    const clientB = createElectronDeidentifyClient(ipcMain.createRenderer(windowB));
    const raw = ipcMain.createRenderer(attacker);

    const release = model.hold();
    const taskB = clientB.submit(`${HOLD} ${NOTE}`);
    const taskA = clientA.submit(NOTE);
    await until(() => service.diagnostics().pendingRequests === 2, "both pending");

    const stolen = (await raw.invoke("openmed:deidentify:cancel", {
      schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
      requestId: taskA.requestId,
    })) as { cancelled: boolean };
    assert.equal(stolen.cancelled, false, "another renderer cannot cancel this request");

    windowB.destroy();
    assert.equal((await typedRejection(taskB.result)).code, "CANCELLED");
    assert.equal(service.diagnostics().pendingRequests, 1);
    release();
    const responseA = await taskA.result;
    assert.deepEqual(responseA.bundle, { generation: 1, contentId: GOLDEN_ID });
    assert.equal(await taskA.cancel(), false, "a settled task cannot be cancelled");

    const release2 = model.hold();
    const crashedTask = clientA.submit(`${HOLD} ${NOTE}`);
    await until(() => service.diagnostics().pendingRequests === 1, "pending again");
    windowA.crashRenderer();
    assert.equal((await typedRejection(crashedTask.result)).code, "CANCELLED");
    release2();
    await until(() => service.diagnostics().cancellingRequests === 0, "cancel acks");
    assert.equal(service.diagnostics().utilityProcessStarts, 1);
  } finally {
    unregister();
    service.dispose();
  }
});

test("explicit cancellation through the renderer client is acknowledged", async () => {
  const { service, model } = setup();
  const ipcMain = new FakeIpcMain();
  const window = new FakeSender(7);
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => event.sender === window,
  });
  try {
    await service.whenReady();
    const client = createElectronDeidentifyClient(ipcMain.createRenderer(window));
    const release = model.hold();
    const task = client.submit(`${HOLD} ${NOTE}`);
    await until(() => service.diagnostics().pendingRequests === 1, "pending");
    assert.equal(await task.cancel(), true);
    assert.equal((await typedRejection(task.result)).code, "CANCELLED");
    release();
    const after = await client.deidentify(NOTE);
    assert.equal(after.spans.length, 1);
  } finally {
    unregister();
    service.dispose();
  }
});

test("service diagnostics stay value-free across the lifecycle", async () => {
  const events: ElectronDeidentifyServiceEvent[] = [];
  const { service, utilityProcess } = setup({ logger: events });
  try {
    await service.whenReady();
    await service.deidentify(request("one", NOTE));
    utilityProcess.children[0]?.crash();
    await service.deidentify(request("two", NOTE));
    await service.activateBundle(source(tamperedRoot)).catch(() => undefined);
  } finally {
    service.dispose();
  }
  const allowed: Record<string, string[]> = {
    request_started: [],
    request_completed: ["spanCount"],
    request_failed: ["errorCode"],
    request_cancelled: [],
    utility_process_started: [],
    utility_process_exited: [],
    bundle_verified: ["generation", "contentId"],
    bundle_rejected: ["generation", "reason"],
  };
  for (const event of events) {
    const { event: name, ...fields } = event as { event: string } & Record<string, unknown>;
    assert.ok(name in allowed, name);
    assert.deepEqual(Object.keys(fields).sort(), [...(allowed[name] as string[])].sort());
    for (const value of Object.values(fields)) {
      assert.ok(
        typeof value === "number" ||
          (typeof value === "string" && /^(?:[A-Za-z_]+|sha256:[0-9a-f]{64})$/.test(value)),
      );
    }
  }
  assert.doesNotMatch(JSON.stringify(events), /Qorvane|Tilsby|Patient/);
});

test("bundle-mode service refuses ambiguous or relaxed configuration", () => {
  const base = {
    utilityProcess: new FakeUtilityProcessModule(() => async () => null),
    offlineSession: new FakeOfflineSession(),
    workerPath,
  };
  assert.throws(() => new ElectronDeidentifyService(base), TypeError);
  assert.throws(
    () =>
      new ElectronDeidentifyService({
        ...base,
        bundle: source(goldenRoot),
        modelPath: goldenRoot,
      }),
    TypeError,
  );
  assert.throws(
    () => new ElectronDeidentifyService({ ...base, bundle: source("relative/bundle") }),
    TypeError,
  );
  assert.throws(
    () =>
      new ElectronDeidentifyService({
        ...base,
        bundle: { ...source(goldenRoot), expectedContentId: "sha256:short" },
      }),
    TypeError,
  );
});
