import assert from "node:assert/strict";
import childProcess from "node:child_process";
import dgram from "node:dgram";
import dns from "node:dns";
import dnsPromises from "node:dns/promises";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import http from "node:http";
import http2 from "node:http2";
import https from "node:https";
import net from "node:net";
import { join } from "node:path";
import tls from "node:tls";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  deidentify,
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
  type TokenClassificationPipeline,
} from "../../js/openmedkit-web/src/index";
import {
  ElectronDeidentifyService,
  OPENMED_DEIDENTIFY_CHANNEL,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  createElectronDeidentifyClient,
  redactTextWithSpans,
  registerElectronDeidentifyIpc,
  type ElectronDeidentifyRequest,
  type ElectronOfflineSessionLike,
  type IpcMainLike,
  type RendererOpenMedSpan,
  type UtilityProcessLike,
  type UtilityProcessModuleLike,
} from "../../js/openmedkit-electron/src/index";
import {
  createUtilityDeidentifyHandler,
  installOfflineNetworkGuard,
} from "../../js/openmedkit-electron/src/utility-process";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
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
  const utilityProcess = new FakeUtilityProcessModule(async (message) => {
    const response = await handler(message);
    if (!response.ok) {
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
    workerPath: join(
      rootDir,
      "js",
      "openmedkit-electron",
      "dist",
      "utility-process.js",
    ),
    modelPath: join(rootDir, "tests", "web", "fixtures", "local-model-cache"),
    logger: (entry) => mainLogs.push(entry),
  });
  const ipcMain = new FakeIpcMain();
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => event.sender === "trusted-window",
  });
  const firstWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(rendererLogs),
  );
  const secondWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(rendererLogs),
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
    assertRendererSpansClose(
      first.spans,
      golden.spans.map(projectGoldenSpan),
    );
    assertRendererSpansClose(
      second.spans,
      golden.spans.map(projectGoldenSpan),
    );
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

  const combinedLogs = JSON.stringify({ mainLogs, rendererLogs });
  for (const phi of ["Alice", "Nguyen", "1979-04-12", "alice@example.org"]) {
    assert.equal(combinedLogs.includes(phi), false, `logs leaked ${phi}`);
  }
});

test("Electron IPC rejects untrusted senders before inference", async () => {
  const utilityProcess = new FakeUtilityProcessModule(async () => {
    throw new Error("must not run");
  });
  const service = createService(utilityProcess);
  const ipcMain = new FakeIpcMain();
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    isTrustedSender: (event) => event.sender === "trusted-window",
  });

  try {
    const client = createElectronDeidentifyClient(
      ipcMain.createRenderer([], "untrusted-window"),
    );
    await assert.rejects(
      client.deidentify("Synthetic patient note"),
      /Unauthorized OpenMed Electron IPC sender/,
    );
    assert.equal(utilityProcess.forkCount, 0);
  } finally {
    unregister();
    service.dispose();
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
  const utilityProcess = new FakeUtilityProcessModule(async (message) => {
    const request = message as ElectronDeidentifyRequest;
    return utilitySuccess(request.requestId, [rendererSpan(0, request.text.length + 1)]);
  });
  const service = createService(utilityProcess);

  try {
    await assert.rejects(
      service.deidentify(electronRequest("invalid-output", "Synthetic note")),
      /invalid response/,
    );
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
    await assert.rejects(firstResult, /timed out/);
    assert.equal(firstChild.killed, true);

    const secondResult = service.deidentify(request);
    const secondChild = utilityProcess.children[1];
    assert.ok(secondChild);
    firstChild.reply(utilitySuccess(request.requestId, [rendererSpan(0, 4)]));
    secondChild.reply(utilitySuccess(request.requestId, [rendererSpan(0, 4)]));

    const response = await secondResult;
    assert.deepEqual(response.spans, [rendererSpan(0, 4)]);
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
    await assert.rejects(
      service.deidentify(electronRequest("second", "Note")),
      /queue is full/,
    );
    utilityProcess.children[0]?.reply(
      utilitySuccess(firstRequest.requestId, [rendererSpan(0, 4)]),
    );
    assert.deepEqual((await firstResult).spans, [rendererSpan(0, 4)]);
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
  assert.equal(first.ok, true);
  assert.equal(second.ok, true);
  assert.equal(modelLoadCount, 1);
  assert.equal(maximumActive, 1);
});

class FakeUtilityProcess extends EventEmitter implements UtilityProcessLike {
  killed = false;

  constructor(
    private readonly handler: (message: unknown) => Promise<unknown>,
  ) {
    super();
  }

  postMessage(message: unknown): void {
    void this.handler(message).then((response) => this.emit("message", response));
  }

  kill(): boolean {
    this.killed = true;
    this.emit("exit", 0);
    return true;
  }
}

class FakeUtilityProcessModule implements UtilityProcessModuleLike {
  forkCount = 0;
  readonly children: FakeUtilityProcess[] = [];
  lastFork:
    | {
        workerPath: string;
        args: string[];
        options?: {
          env?: Record<string, string>;
          session?: ElectronOfflineSessionLike;
          serviceName?: string;
          stdio?: "ignore";
        };
      }
    | undefined;

  constructor(
    private readonly handler: (message: unknown) => Promise<unknown>,
  ) {}

  fork(
    workerPath: string,
    args: string[] = [],
    options?: {
      env?: Record<string, string>;
      session?: ElectronOfflineSessionLike;
      serviceName?: string;
      stdio?: "ignore";
    },
  ): UtilityProcessLike {
    this.forkCount += 1;
    this.lastFork = options
      ? { workerPath, args, options }
      : { workerPath, args };
    const child = new FakeUtilityProcess(this.handler);
    this.children.push(child);
    return child;
  }
}

class ManualUtilityProcess extends EventEmitter implements UtilityProcessLike {
  readonly messages: unknown[] = [];
  killed = false;

  postMessage(message: unknown): void {
    this.messages.push(message);
  }

  reply(message: unknown): void {
    this.emit("message", message);
  }

  kill(): boolean {
    this.killed = true;
    this.emit("exit", 0);
    return true;
  }
}

class ManualUtilityProcessModule implements UtilityProcessModuleLike {
  readonly children: ManualUtilityProcess[] = [];

  fork(): UtilityProcessLike {
    const child = new ManualUtilityProcess();
    this.children.push(child);
    return child;
  }
}

interface FakeIpcEvent {
  sender: string;
}

class FakeIpcMain implements IpcMainLike<FakeIpcEvent> {
  private readonly handlers = new Map<
    string,
    (event: FakeIpcEvent, request: unknown) => Promise<unknown> | unknown
  >();

  handle(
    channel: string,
    listener: (
      event: FakeIpcEvent,
      request: unknown,
    ) => Promise<unknown> | unknown,
  ): void {
    this.handlers.set(channel, listener);
  }

  removeHandler(channel: string): void {
    this.handlers.delete(channel);
  }

  createRenderer(logs: unknown[], sender = "trusted-window") {
    return {
      invoke: async (channel: string, request: unknown): Promise<unknown> => {
        assert.equal(channel, OPENMED_DEIDENTIFY_CHANNEL);
        const handler = this.handlers.get(channel);
        assert.ok(handler);
        const response = await handler({ sender }, request);
        logs.push({ event: "deidentify_completed" });
        return response;
      },
    };
  }
}

class FakeOfflineSession implements ElectronOfflineSessionLike {
  private listener:
    | ((
        details: unknown,
        callback: (response: { cancel: boolean }) => void,
      ) => void)
    | null = null;

  readonly webRequest = {
    onBeforeRequest: (
      listener:
        | ((
            details: unknown,
            callback: (response: { cancel: boolean }) => void,
          ) => void)
        | null,
    ): void => {
      this.listener = listener;
    },
  };

  simulateRequest(): { cancel: boolean } | undefined {
    let response: { cancel: boolean } | undefined;
    this.listener?.({}, (value) => {
      response = value;
    });
    return response;
  }
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
    workerPath: join(
      rootDir,
      "js",
      "openmedkit-electron",
      "dist",
      "utility-process.js",
    ),
    modelPath: join(rootDir, "tests", "web", "fixtures", "local-model-cache"),
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
    modelPath: join(rootDir, "tests", "web", "fixtures", "local-model-cache"),
  };
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
