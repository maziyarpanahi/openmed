import assert from "node:assert/strict";
import childProcess from "node:child_process";
import dgram from "node:dgram";
import dns from "node:dns";
import dnsPromises from "node:dns/promises";
import { readFile } from "node:fs/promises";
import http from "node:http";
import http2 from "node:http2";
import https from "node:https";
import net from "node:net";
import { join } from "node:path";
import tls from "node:tls";
import test from "node:test";

import {
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
  type TokenClassificationPipeline,
} from "../../js/openmedkit-web/src/index";
import {
  ELECTRON_DEIDENTIFY_ERROR_CODES,
  ElectronDeidentifyError,
  ElectronDeidentifyService,
  ElectronHardeningError,
  OPENMED_CANCEL_CHANNEL,
  OPENMED_DEIDENTIFY_CHANNEL,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  OPENMED_HARDENED_WEB_PREFERENCES,
  OPENMED_RENDERER_CSP,
  WEB_PREFERENCE_RELAXATIONS,
  assertHardenedWebPreferences,
  createElectronDeidentifyClient,
  createHardenedWebPreferences,
  findWebPreferenceRelaxations,
  hardenOpenMedSession,
  hardenOpenMedWebContents,
  isAllowedOpenMedOrigin,
  redactTextWithSpans,
  registerElectronDeidentifyIpc,
  type ElectronDeidentifyRequest,
  type RendererOpenMedSpan,
  type UtilityProcessModuleLike,
} from "../../js/openmedkit-electron/src/index";
import {
  createUtilityDeidentifyHandler,
  installOfflineNetworkGuard,
} from "../../js/openmedkit-electron/src/utility-process";
import {
  FakeIpcMain,
  FakeOfflineSession,
  FakeSender,
  FakeUtilityProcessModule,
  ManualUtilityProcessModule,
  legacyModelPath,
  rootDir,
  until,
  workerPath,
} from "./electron_test_support";

const fixturePath = join(
  rootDir,
  "tests",
  "web",
  "fixtures",
  "npm_deidentify_golden.json",
);

test("Electron IPC returns only renderer-safe spans matching the golden", async () => {
  const golden = JSON.parse(
    await readFile(fixturePath, "utf8"),
  ) as OpenMedDeidentifyResult;
  const mainLogs: unknown[] = [];
  const rendererLogs: unknown[] = [];
  let modelLoadCount = 0;
  const handler = createUtilityDeidentifyHandler({
    loadPipeline: async () => {
      modelLoadCount += 1;
      return fixturePipeline;
    },
  });
  const utilityProcess = new FakeUtilityProcessModule(() => async (message) => {
    const response = await handler(message);
    if (response === null || response.type !== "deidentify-result" || !response.ok) {
      return response;
    }
    return {
      ...response,
      spans: response.spans.map((span) => ({ ...span, rawText: golden.text })),
    };
  });
  const offlineSession = new FakeOfflineSession();
  const service = new ElectronDeidentifyService({
    utilityProcess,
    offlineSession,
    workerPath,
    modelPath: legacyModelPath,
    logger: (entry) => mainLogs.push(entry),
  });
  const ipcMain = new FakeIpcMain();
  const trusted = new Set([new FakeSender(1), new FakeSender(2)]);
  const [firstSender, secondSender] = [...trusted];
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => trusted.has(event.sender as FakeSender),
  });
  const firstWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(firstSender as FakeSender, rendererLogs),
  );
  const secondWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(secondSender as FakeSender, rendererLogs),
  );
  const restoreNetwork = installOfflineNetworkGuard();
  const restoreConsole = captureConsole(rendererLogs);

  try {
    assert.throws(
      () => globalThis.fetch("https://example.invalid/model.onnx"),
      /Network access is disabled/,
    );
    assert.throws(
      () => http.request("http://example.invalid/model.onnx"),
      /Network access is disabled/,
    );
    assert.throws(
      () => net.connect(443, "example.invalid"),
      /Network access is disabled/,
    );
    assert.throws(
      () => https.request("https://example.invalid/model.onnx"),
      /Network access is disabled/,
    );
    assert.throws(
      () => tls.connect(443, "example.invalid"),
      /Network access is disabled/,
    );
    assert.throws(
      () => http2.connect("https://example.invalid"),
      /Network access is disabled/,
    );
    assert.throws(
      () => dgram.createSocket("udp4"),
      /Network access is disabled/,
    );
    assert.throws(
      () => dns.lookup("example.invalid", () => undefined),
      /Network access is disabled/,
    );
    assert.throws(
      () => dnsPromises.lookup("example.invalid"),
      /Network access is disabled/,
    );
    assert.throws(
      () => childProcess.spawn("curl", ["https://example.invalid"]),
      /Network access is disabled/,
    );

    const first = await firstWindow.deidentify(golden.text);
    const second = await secondWindow.deidentify(golden.text);
    assertRendererSpansClose(first.spans, golden.spans.map(projectGoldenSpan));
    assertRendererSpansClose(second.spans, golden.spans.map(projectGoldenSpan));
    assert.equal(first.bundle, null, "legacy model paths are not verified bundles");
    assert.equal(redactTextWithSpans(golden.text, first.spans), golden.deidentifiedText);
    assert.equal(utilityProcess.forkCount, 1);
    assert.equal(utilityProcess.lastFork?.workerPath.endsWith("utility-process.js"), true);
    assert.equal(utilityProcess.lastFork?.options?.stdio, "ignore");
    assert.equal(utilityProcess.lastFork?.options?.session, offlineSession);
    assert.deepEqual(utilityProcess.lastFork?.options?.env, {
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
    });
    assert.equal(JSON.stringify(utilityProcess.lastFork).includes("Alice"), false);
    assert.equal(modelLoadCount, 1, "model cache must be shared across windows");
    assert.deepEqual(offlineSession.simulateRequest(), { cancel: true });
    assert.deepEqual(ipcMain.channels(), [
      OPENMED_DEIDENTIFY_CHANNEL,
      OPENMED_CANCEL_CHANNEL,
    ].sort());

    for (const span of first.spans) {
      assert.deepEqual(Object.keys(span).sort(), [
        "canonical_label",
        "end",
        "entity_type",
        "policy_label",
        "schema_version",
        "score",
        "start",
      ]);
    }
  } finally {
    restoreConsole();
    restoreNetwork();
    unregister();
    service.dispose();
  }
  assert.deepEqual(ipcMain.channels(), []);

  const combinedLogs = JSON.stringify({ mainLogs, rendererLogs });
  for (const phi of ["Alice", "Nguyen", "1979-04-12", "alice@example.org"]) {
    assert.equal(combinedLogs.includes(phi), false, `logs leaked ${phi}`);
  }
});

test("Electron IPC rejects untrusted senders before inference", async () => {
  const utilityProcess = new FakeUtilityProcessModule(() => async () => {
    throw new Error("must not run");
  });
  const service = createService(utilityProcess);
  const ipcMain = new FakeIpcMain();
  const trusted = new FakeSender(1);
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => event.sender === trusted,
  });

  try {
    const client = createElectronDeidentifyClient(
      ipcMain.createRenderer(new FakeSender(2)),
    );
    const error = await rejection(client.deidentify("Synthetic patient note"));
    assert.equal(error.code, "UNTRUSTED_SENDER");
    assert.match(error.message, /Unauthorized OpenMed Electron IPC sender/);
    assert.equal(utilityProcess.forkCount, 0);
  } finally {
    unregister();
    service.dispose();
  }
});

test("IPC refuses trusted senders whose web preferences are relaxed", async () => {
  const utilityProcess = new FakeUtilityProcessModule(() => async () => {
    throw new Error("must not run");
  });
  const service = createService(utilityProcess);
  const ipcMain = new FakeIpcMain();
  const relaxedPreferences: Array<Record<string, unknown>> = [
    { contextIsolation: false },
    { sandbox: false },
    { nodeIntegration: true },
    { nodeIntegrationInWorker: true },
    { nodeIntegrationInSubFrames: true },
    { webSecurity: false },
    { allowRunningInsecureContent: true },
    { webviewTag: true },
    { experimentalFeatures: true },
    { enableBlinkFeatures: "SyntheticFeature" },
  ];
  const senders = [
    ...relaxedPreferences.map(
      (override, index) =>
        new FakeSender(index + 10, { ...OPENMED_HARDENED_WEB_PREFERENCES, ...override }),
    ),
    new FakeSender(40, { contextIsolation: true, nodeIntegration: false }),
    new FakeSender(41, null),
  ];
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: () => true,
  });

  try {
    for (const sender of [...senders, { id: 50 }]) {
      const client = createElectronDeidentifyClient(ipcMain.createRenderer(sender));
      const error = await rejection(client.deidentify("Synthetic patient note"));
      assert.equal(error.code, "UNTRUSTED_SENDER");
    }
    assert.equal(utilityProcess.forkCount, 0);
  } finally {
    unregister();
    service.dispose();
  }
});

test("IPC refuses renderer-chosen model paths, bundles, runtimes and policies", async () => {
  const utilityProcess = new FakeUtilityProcessModule(() => async () => {
    throw new Error("must not run");
  });
  const service = createService(utilityProcess);
  const ipcMain = new FakeIpcMain();
  const sender = new FakeSender(1);
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => event.sender === sender,
  });
  const raw = ipcMain.createRenderer(sender);
  const base = {
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId: "attack-1",
    text: "Synthetic patient note",
  };

  try {
    for (const extra of [
      { modelPath: "/synthetic/elsewhere" },
      { bundle: { root: "/synthetic/elsewhere" } },
      { bundleRoot: "/synthetic/elsewhere" },
      { trustPolicy: { keys: [] } },
      { generation: 99 },
      { runtime: "pytorch" },
      { policy: { allow: "all" } },
      { type: "load-bundle" },
    ]) {
      const response = (await raw.invoke(OPENMED_DEIDENTIFY_CHANNEL, {
        ...base,
        ...extra,
      })) as Record<string, unknown>;
      assert.deepEqual(response, {
        schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
        requestId: "attack-1",
        ok: false,
        errorCode: "INVALID_REQUEST",
      });
    }
    for (const malformed of [null, [], "text", { ...base, schemaVersion: 2 }]) {
      const response = (await raw.invoke(OPENMED_DEIDENTIFY_CHANNEL, malformed)) as {
        errorCode: string;
      };
      assert.equal(response.errorCode, "INVALID_REQUEST");
    }
    await assert.rejects(raw.invoke("openmed:activate-bundle", base), /No handler/);
    const cancel = (await raw.invoke(OPENMED_CANCEL_CHANNEL, {
      ...base,
    })) as { errorCode: string };
    assert.equal(cancel.errorCode, "INVALID_REQUEST");
    assert.equal(utilityProcess.forkCount, 0);
  } finally {
    unregister();
    service.dispose();
  }
});

test("error codes are value-free and cover the IPC failure contract", () => {
  for (const code of ELECTRON_DEIDENTIFY_ERROR_CODES) {
    const error = new ElectronDeidentifyError(code);
    assert.equal(error.code, code);
    assert.doesNotMatch(error.message, /[/\\]|Alice|sha256/);
  }
});

test("renderer entrypoint contains no Node-only imports", async () => {
  const distDir = join(rootDir, "js", "openmedkit-electron", "dist");
  const rendererEntry = await readFile(join(distDir, "ipc.js"), "utf8");
  const localImports = [
    ...rendererEntry.matchAll(/from ["']\.\/([^"']+)["']/g),
  ].map((match) => match[1]);
  const rendererBundle = [
    rendererEntry,
    ...(await Promise.all(
      localImports.map(async (filename) => {
        assert.ok(filename);
        return readFile(join(distDir, filename), "utf8");
      }),
    )),
  ].join("\n");

  assert.doesNotMatch(rendererBundle, /\bnode:/);
  assert.doesNotMatch(rendererBundle, /from ["'](?:fs|path|http|net|tls)["']/);
  assert.doesNotMatch(rendererBundle, /\brequire\s*\(/);
});

test("renderer span validation rejects unsafe bounds and overlap", () => {
  const text = "Synthetic note";
  const first = rendererSpan(0, 4);
  const second = rendererSpan(3, 6);

  assert.throws(
    () => redactTextWithSpans(text, [first, second]),
    /invalid OpenMed Electron response spans/i,
  );
  assert.throws(
    () => redactTextWithSpans(text, [rendererSpan(0, text.length + 1)]),
    /invalid OpenMed Electron response spans/i,
  );
  assert.throws(
    () =>
      redactTextWithSpans(text, [
        {
          ...first,
          canonical_label: "SYNTHETIC_VALUE",
        } as unknown as RendererOpenMedSpan,
      ]),
    /invalid OpenMed Electron response spans/i,
  );
});

test("service rejects corrupt utility output and restarts safely", async () => {
  const utilityProcess = new FakeUtilityProcessModule(() => async (message) => {
    const request = message as ElectronDeidentifyRequest;
    return utilitySuccess(request.requestId, [rendererSpan(0, request.text.length + 1)]);
  });
  const service = createService(utilityProcess);

  try {
    const error = await rejection(
      service.deidentify(electronRequest("invalid-output", "Synthetic note")),
    );
    assert.equal(error.code, "INVALID_UTILITY_RESPONSE");
    assert.match(error.message, /invalid response/);
    assert.equal(utilityProcess.children[0]?.killed, true);
  } finally {
    service.dispose();
  }
});

test("timeout retires the worker and ignores its stale response", async () => {
  const utilityProcess = new ManualUtilityProcessModule();
  const service = createService(utilityProcess, { requestTimeoutMs: 20 });
  const request = electronRequest("reused-after-timeout", "Note");

  try {
    const firstResult = service.deidentify(request);
    const firstChild = utilityProcess.children[0];
    assert.ok(firstChild);
    const firstError = await rejection(firstResult);
    assert.equal(firstError.code, "REQUEST_TIMEOUT");
    assert.match(firstError.message, /timed out/);
    assert.equal(firstChild.killed, true);

    const secondResult = service.deidentify(request);
    const secondChild = utilityProcess.children[1];
    assert.ok(secondChild);
    firstChild.reply(utilitySuccess(jobIdOf(firstChild.messages[0]), [rendererSpan(0, 4)]));
    secondChild.reply(
      utilitySuccess(jobIdOf(secondChild.messages[0]), [rendererSpan(0, 4)]),
    );

    const response = await secondResult;
    assert.deepEqual(response.spans, [rendererSpan(0, 4)]);
    assert.equal(response.requestId, request.requestId);
  } finally {
    service.dispose();
  }
});

test("service bounds pending work and isolates logger failures", async () => {
  const utilityProcess = new ManualUtilityProcessModule();
  const service = createService(utilityProcess, {
    maxPendingRequests: 1,
    logger: () => {
      throw new Error("logger failure");
    },
  });
  const firstRequest = electronRequest("first", "Note");

  try {
    const firstResult = service.deidentify(firstRequest);
    const full = await rejection(service.deidentify(electronRequest("second", "Note")));
    assert.equal(full.code, "QUEUE_FULL");
    assert.match(full.message, /queue is full/);
    const duplicate = await rejection(service.deidentify(firstRequest));
    assert.equal(duplicate.code, "DUPLICATE_REQUEST");
    const child = utilityProcess.children[0];
    assert.ok(child);
    child.reply(utilitySuccess(jobIdOf(child.messages[0]), [rendererSpan(0, 4)]));
    assert.deepEqual((await firstResult).spans, [rendererSpan(0, 4)]);
  } finally {
    service.dispose();
  }
});

test("request ids are scoped per renderer and may repeat across renderers", async () => {
  const utilityProcess = new ManualUtilityProcessModule();
  const service = createService(utilityProcess);
  const request = electronRequest("shared-id", "Note");

  try {
    const first = service.deidentify(request, { owner: "window-a" });
    const second = service.deidentify(request, { owner: "window-b" });
    const duplicate = await rejection(service.deidentify(request, { owner: "window-a" }));
    assert.equal(duplicate.code, "DUPLICATE_REQUEST");
    const child = utilityProcess.children[0];
    assert.ok(child);
    const [firstJob, secondJob] = child.messages.map(jobIdOf);
    assert.notEqual(firstJob, secondJob);
    child.reply(utilitySuccess(secondJob as string, [rendererSpan(0, 2)]));
    child.reply(utilitySuccess(firstJob as string, [rendererSpan(0, 4)]));
    assert.deepEqual((await first).spans, [rendererSpan(0, 4)]);
    assert.deepEqual((await second).spans, [rendererSpan(0, 2)]);
  } finally {
    service.dispose();
  }
});

test("utility inference is serialized while its pipeline cache is shared", async () => {
  let active = 0;
  let maximumActive = 0;
  let modelLoadCount = 0;
  const golden = JSON.parse(
    await readFile(fixturePath, "utf8"),
  ) as OpenMedDeidentifyResult;
  const handler = createUtilityDeidentifyHandler({
    loadPipeline: async () => {
      modelLoadCount += 1;
      return async (text) => {
        active += 1;
        maximumActive = Math.max(maximumActive, active);
        await new Promise<void>((resolve) => setTimeout(resolve, 5));
        active -= 1;
        return fixturePipeline(text);
      };
    },
  });

  const [first, second] = await Promise.all([
    handler(utilityRequest("parallel-first", golden.text)),
    handler(utilityRequest("parallel-second", golden.text)),
  ]);
  assert.equal(first?.type === "deidentify-result" && first.ok, true);
  assert.equal(second?.type === "deidentify-result" && second.ok, true);
  assert.equal(modelLoadCount, 1);
  assert.equal(maximumActive, 1);
});

test("hardened web preferences refuse every relaxation", () => {
  const hardened = createHardenedWebPreferences({ preload: "/synthetic/preload.cjs" });
  assert.deepEqual(findWebPreferenceRelaxations(hardened), []);
  assert.equal(hardened.contextIsolation, true);
  assert.equal(hardened.sandbox, true);
  assert.equal(hardened.nodeIntegration, false);
  assert.equal(hardened.webSecurity, true);
  assertHardenedWebPreferences(hardened);
  assert.throws(() => createHardenedWebPreferences({ preload: "" }), TypeError);

  const cases: Array<[Record<string, unknown>, string]> = [
    [{ contextIsolation: false }, "context_isolation_disabled"],
    [{ contextIsolation: undefined }, "context_isolation_disabled"],
    [{ sandbox: false }, "sandbox_disabled"],
    [{ sandbox: undefined }, "sandbox_disabled"],
    [{ nodeIntegration: true }, "node_integration_enabled"],
    [{ nodeIntegration: undefined }, "node_integration_enabled"],
    [{ nodeIntegrationInWorker: true }, "node_integration_in_worker_enabled"],
    [{ nodeIntegrationInSubFrames: true }, "node_integration_in_subframes_enabled"],
    [{ webSecurity: false }, "web_security_disabled"],
    [{ allowRunningInsecureContent: true }, "insecure_content_allowed"],
    [{ webviewTag: true }, "webview_tag_enabled"],
    [{ experimentalFeatures: true }, "experimental_features_enabled"],
    [{ enableBlinkFeatures: "SyntheticFeature" }, "blink_features_enabled"],
  ];
  const seen = new Set<string>();
  for (const [override, relaxation] of cases) {
    const preferences = { ...hardened, ...override };
    assert.deepEqual(findWebPreferenceRelaxations(preferences), [relaxation]);
    assert.throws(
      () => assertHardenedWebPreferences(preferences),
      (error: unknown) =>
        error instanceof ElectronHardeningError &&
        error.relaxations.length === 1 &&
        error.relaxations[0] === relaxation,
    );
    seen.add(relaxation);
  }
  assert.deepEqual(findWebPreferenceRelaxations(null), ["web_preferences_invalid"]);
  seen.add("web_preferences_invalid");
  assert.deepEqual([...seen].sort(), [...WEB_PREFERENCE_RELAXATIONS].sort());
});

test("renderer CSP refuses inline code, eval, network and frames", () => {
  const directives = new Map(
    OPENMED_RENDERER_CSP.split(";").map((directive) => {
      const [name, ...values] = directive.trim().split(/\s+/);
      return [name as string, values];
    }),
  );
  assert.deepEqual(directives.get("default-src"), ["'none'"]);
  assert.deepEqual(directives.get("script-src"), ["'self'"]);
  assert.deepEqual(directives.get("connect-src"), ["'none'"]);
  assert.deepEqual(directives.get("object-src"), ["'none'"]);
  assert.deepEqual(directives.get("frame-src"), ["'none'"]);
  assert.deepEqual(directives.get("frame-ancestors"), ["'none'"]);
  assert.deepEqual(directives.get("base-uri"), ["'none'"]);
  assert.deepEqual(directives.get("form-action"), ["'none'"]);
  assert.deepEqual(directives.get("worker-src"), ["'none'"]);
  for (const values of directives.values()) {
    for (const value of values) {
      assert.ok(
        value === "'self'" || value === "'none'",
        "renderer CSP may only use 'self' or 'none'",
      );
    }
  }
});

test("window guard blocks off-origin navigation, popups and webviews", () => {
  const events = new Map<string, (...args: unknown[]) => void>();
  let openHandler: ((details: { url: string }) => { action: "deny" }) | undefined;
  const blocked: string[] = [];
  const contents = {
    on(event: string, listener: (...args: never[]) => void) {
      events.set(event, listener as (...args: unknown[]) => void);
      return contents;
    },
    setWindowOpenHandler(handler: (details: { url: string }) => { action: "deny" }) {
      openHandler = handler;
    },
    getLastWebPreferences: () => ({ ...OPENMED_HARDENED_WEB_PREFERENCES }),
  };
  hardenOpenMedWebContents(contents, {
    allowedOrigins: ["app://openmed"],
    onBlocked: (event) => blocked.push(event),
  });

  const navigate = (name: string, url: string) => {
    let prevented = false;
    events.get(name)?.({ url, preventDefault: () => (prevented = true) });
    return prevented;
  };
  for (const name of ["will-navigate", "will-redirect", "will-frame-navigate"]) {
    assert.equal(navigate(name, "https://example.invalid/"), true);
    assert.equal(navigate(name, "file:///synthetic/index.html"), true);
    assert.equal(navigate(name, "app://openmed.example.invalid/"), true);
    assert.equal(navigate(name, "app://openmed/index.html"), false);
  }
  assert.deepEqual(openHandler?.({ url: "app://openmed/index.html" }), { action: "deny" });
  let webviewPrevented = false;
  events.get("will-attach-webview")?.({ preventDefault: () => (webviewPrevented = true) });
  assert.equal(webviewPrevented, true);
  assert.ok(blocked.includes("navigation_blocked"));
  assert.ok(blocked.includes("window_open_blocked"));
  assert.ok(blocked.includes("webview_blocked"));

  const relaxed = { ...contents, getLastWebPreferences: () => ({ sandbox: false }) };
  assert.throws(
    () => hardenOpenMedWebContents(relaxed, { allowedOrigins: ["app://openmed"] }),
    ElectronHardeningError,
  );
  for (const origins of [[], ["https://example.invalid"], ["file://"], ["http://localhost"]]) {
    assert.throws(
      () => hardenOpenMedWebContents(contents, { allowedOrigins: origins }),
      TypeError,
    );
  }
  assert.equal(isAllowedOpenMedOrigin("app://user:pw@openmed/", ["app://openmed"]), false);
  assert.equal(isAllowedOpenMedOrigin("not a url", ["app://openmed"]), false);
});

test("session guard denies permissions, off-origin requests and weaker CSP", () => {
  let permissionHandler:
    | ((contents: unknown, permission: string, callback: (granted: boolean) => void) => void)
    | null = null;
  let checkHandler: (() => boolean) | null = null;
  let beforeRequest:
    | ((details: { url: string }, callback: (response: { cancel: boolean }) => void) => void)
    | undefined;
  let headersReceived:
    | ((
        details: { url: string; responseHeaders?: Record<string, string[]> },
        callback: (response: {
          cancel?: boolean;
          responseHeaders?: Record<string, string[]>;
        }) => void,
      ) => void)
    | undefined;
  hardenOpenMedSession(
    {
      setPermissionRequestHandler: (handler) => {
        permissionHandler = handler;
      },
      setPermissionCheckHandler: (handler) => {
        checkHandler = handler;
      },
      webRequest: {
        onBeforeRequest: (listener) => {
          beforeRequest = listener;
        },
        onHeadersReceived: (listener) => {
          headersReceived = listener;
        },
      },
    },
    { allowedOrigins: ["app://openmed"] },
  );
  let granted: boolean | undefined;
  (permissionHandler as unknown as (c: unknown, p: string, cb: (g: boolean) => void) => void)(
    {},
    "media",
    (value) => (granted = value),
  );
  assert.equal(granted, false);
  assert.equal((checkHandler as unknown as () => boolean)(), false);

  const decide = (url: string) => {
    let decision: { cancel: boolean } | undefined;
    beforeRequest?.({ url }, (value) => (decision = value));
    return decision;
  };
  assert.deepEqual(decide("https://example.invalid/model.onnx"), { cancel: true });
  assert.deepEqual(decide("app://openmed/renderer.js"), { cancel: false });

  let headers: Record<string, string[]> | undefined;
  headersReceived?.(
    {
      url: "app://openmed/index.html",
      responseHeaders: {
        "content-security-policy": ["default-src *"],
        "Content-Type": ["text/html"],
      },
    },
    (value) => (headers = value.responseHeaders),
  );
  assert.deepEqual(headers, {
    "Content-Type": ["text/html"],
    "Content-Security-Policy": [OPENMED_RENDERER_CSP],
  });
});

test("renderer clients surface typed errors for unexpected main responses", async () => {
  const responses: unknown[] = [
    { schemaVersion: 1, requestId: "other", ok: true, spans: [], bundle: null },
    { schemaVersion: 1, ok: false, errorCode: "SYNTHETIC_CODE" },
    "not-an-object",
  ];
  const client = createElectronDeidentifyClient({
    invoke: async (_channel, request) => {
      const next = responses.shift();
      if (isRecord(next) && next.requestId === undefined) {
        return { ...next, requestId: (request as { requestId: string }).requestId };
      }
      return next;
    },
  });
  for (let index = 0; index < 3; index += 1) {
    const error = await rejection(client.deidentify("Synthetic patient note"));
    assert.equal(error.code, "INVALID_RESPONSE");
  }
  const failing = createElectronDeidentifyClient({
    invoke: async () => {
      throw new Error("Error invoking remote method: synthetic detail");
    },
  });
  const error = await rejection(failing.deidentify("Synthetic patient note"));
  assert.equal(error.code, "INTERNAL_ERROR");
  assert.doesNotMatch(error.message, /synthetic detail/);
  const invalid = await rejection(failing.deidentify(""));
  assert.equal(invalid.code, "INVALID_REQUEST");
});

test("an aborted signal cancels before and during a request", async () => {
  const ipcMain = new FakeIpcMain();
  const utilityProcess = new ManualUtilityProcessModule();
  const service = createService(utilityProcess);
  const sender = new FakeSender(1);
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => event.sender === sender,
  });
  const client = createElectronDeidentifyClient(ipcMain.createRenderer(sender));
  try {
    const before = new AbortController();
    before.abort();
    assert.equal(
      (await rejection(client.deidentify("Note", { signal: before.signal }))).code,
      "CANCELLED",
    );
    const during = new AbortController();
    const pending = client.deidentify("Note", { signal: during.signal });
    await until(() => service.diagnostics().pendingRequests === 1, "request pending");
    during.abort();
    assert.equal((await rejection(pending)).code, "CANCELLED");
    await until(() => service.diagnostics().pendingRequests === 0, "request dropped");
    assert.equal(service.diagnostics().cancellingRequests, 1);
    const child = utilityProcess.children[0];
    assert.ok(child);
    assert.deepEqual(child.messages.at(-1), {
      type: "cancel",
      requestId: jobIdOf(child.messages[0]),
    });
    child.reply(utilitySuccess(jobIdOf(child.messages[0]), [rendererSpan(0, 4)]));
    assert.equal(service.diagnostics().cancellingRequests, 0);
    assert.equal(child.killed, false, "a late result of a cancelled job is dropped");
  } finally {
    unregister();
    service.dispose();
  }
});

class Rejection extends Error {}

async function rejection(promise: Promise<unknown>): Promise<ElectronDeidentifyError> {
  try {
    await promise;
  } catch (error) {
    assert.ok(error instanceof ElectronDeidentifyError, "expected a typed error");
    return error;
  }
  throw new Rejection("expected a rejection");
}

function createService(
  utilityProcess: UtilityProcessModuleLike,
  overrides: Partial<{
    requestTimeoutMs: number;
    maxTextLength: number;
    maxPendingRequests: number;
    logger: (entry: unknown) => void;
  }> = {},
): ElectronDeidentifyService {
  return new ElectronDeidentifyService({
    utilityProcess,
    offlineSession: new FakeOfflineSession(),
    workerPath,
    modelPath: legacyModelPath,
    ...overrides,
  });
}

function electronRequest(
  requestId: string,
  text: string,
): ElectronDeidentifyRequest {
  return {
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId,
    text,
  };
}

function utilityRequest(requestId: string, text: string): unknown {
  return {
    ...electronRequest(requestId, text),
    type: "deidentify",
    modelPath: legacyModelPath,
  };
}

function jobIdOf(message: unknown): string {
  const requestId = (message as { requestId?: unknown }).requestId;
  assert.equal(typeof requestId, "string");
  return requestId as string;
}

function rendererSpan(start: number, end: number): RendererOpenMedSpan {
  return {
    schema_version: 1,
    start,
    end,
    entity_type: "NAME",
    canonical_label: "PERSON",
    policy_label: "DIRECT_IDENTIFIER",
    score: 0.99,
  };
}

function utilitySuccess(requestId: string, spans: RendererOpenMedSpan[]): unknown {
  return {
    type: "deidentify-result",
    requestId,
    ok: true,
    spans,
    generation: null,
  };
}

function projectGoldenSpan(span: OpenMedSpan): RendererOpenMedSpan {
  return {
    schema_version: span.schema_version,
    start: span.start,
    end: span.end,
    entity_type: span.entity_type,
    canonical_label: span.canonical_label,
    policy_label: span.policy_label,
    score: span.score,
  };
}

function assertRendererSpansClose(
  actual: RendererOpenMedSpan[],
  expected: RendererOpenMedSpan[],
): void {
  assert.equal(actual.length, expected.length);
  for (const [index, actualSpan] of actual.entries()) {
    const expectedSpan = expected[index];
    assert.ok(expectedSpan, `missing expected span at ${index}`);
    const { score: actualScore, ...actualRest } = actualSpan;
    const { score: expectedScore, ...expectedRest } = expectedSpan;
    assert.deepEqual(actualRest, expectedRest);
    assert.ok(actualScore !== null);
    assert.ok(expectedScore !== null);
    assert.ok(Math.abs(actualScore - expectedScore) <= 1e-12);
  }
}

const fixturePipeline: TokenClassificationPipeline = (text) => {
  const aliceStart = text.indexOf("Alice");
  const nguyenStart = text.indexOf("Nguyen");
  const dobStart = text.indexOf("1979-04-12");
  const emailStart = text.indexOf("alice@example.org");
  return [
    {
      entity: "B-NAME",
      score: 0.99,
      start: aliceStart,
      end: aliceStart + "Alice".length,
    },
    {
      entity: "E-NAME",
      score: 0.97,
      start: nguyenStart,
      end: nguyenStart + "Nguyen".length,
    },
    {
      entity: "S-DATE_OF_BIRTH",
      score: 0.96,
      start: dobStart,
      end: dobStart + "1979-04-12".length,
    },
    {
      entity: "S-EMAIL",
      score: 0.98,
      start: emailStart,
      end: emailStart + "alice@example.org".length,
    },
  ];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function captureConsole(logs: unknown[]): () => void {
  const originalLog = console.log;
  const originalWarn = console.warn;
  const originalError = console.error;
  console.log = (...args: unknown[]) => logs.push({ level: "log", args });
  console.warn = (...args: unknown[]) => logs.push({ level: "warn", args });
  console.error = (...args: unknown[]) => logs.push({ level: "error", args });
  return () => {
    console.log = originalLog;
    console.warn = originalWarn;
    console.error = originalError;
  };
}
