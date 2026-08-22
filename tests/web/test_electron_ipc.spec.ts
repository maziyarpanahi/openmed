import assert from "node:assert/strict";
import dgram from "node:dgram";
import dns from "node:dns";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import http from "node:http";
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
  OPENMED_ELECTRON_MAX_TEXT_LENGTH,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  createElectronDeidentifyClient,
  redactTextWithSpans,
  registerElectronDeidentifyIpc,
  type IpcMainLike,
  type ElectronDeidentifyRequest,
  type ElectronDeidentifyServiceOptions,
  type RendererOpenMedSpan,
  type UtilityProcessLike,
  type UtilityProcessModuleLike,
} from "../../js/openmedkit-electron/src/index";
import {
  createUtilityDeidentifyHandler,
  installOfflineNetworkGuard,
  modelPathFromUtilityArguments,
} from "../../js/openmedkit-electron/src/utility-process";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const fixturePath = join(
  rootDir,
  "tests",
  "web",
  "fixtures",
  "npm_deidentify_golden.json",
);
const syntheticNote =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";

test("Electron renderer export is free of Node and model-runtime imports", async () => {
  const packageDir = join(rootDir, "js", "openmedkit-electron");
  const manifest = JSON.parse(
    await readFile(join(packageDir, "package.json"), "utf8"),
  ) as { exports?: Record<string, unknown> };
  assert.ok(manifest.exports?.["./renderer"]);

  for (const filename of ["ipc.js", "ipc.cjs"]) {
    const bundle = await readFile(join(packageDir, "dist", filename), "utf8");
    assert.doesNotMatch(
      bundle,
      /from ["'](?:node:|openmed["'])|require\(["'](?:node:|openmed["'])/,
    );
  }
});

test("Electron IPC returns only renderer-safe spans matching the golden", async () => {
  const golden = JSON.parse(
    await readFile(fixturePath, "utf8"),
  ) as OpenMedDeidentifyResult;
  const mainLogs: unknown[] = [];
  const rendererLogs: unknown[] = [];
  let modelLoadCount = 0;
  const modelPath = join(
    rootDir,
    "tests",
    "web",
    "fixtures",
    "local-model-cache",
  );
  const handler = createUtilityDeidentifyHandler({
    modelPath,
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
  const service = new ElectronDeidentifyService({
    utilityProcess,
    workerPath: join(
      rootDir,
      "js",
      "openmedkit-electron",
      "dist",
      "utility-process.js",
    ),
    modelPath,
    logger: (entry) => mainLogs.push(entry),
  });
  const ipcMain = new FakeIpcMain();
  const allowedSenderIds = new Set([101, 102]);
  const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
    authorizeSender: (event) => allowedSenderIds.has(event.sender.id),
  });
  const firstWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(rendererLogs, 101),
  );
  const secondWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(rendererLogs, 102),
  );
  const unauthorizedWindow = createElectronDeidentifyClient(
    ipcMain.createRenderer(rendererLogs, 999),
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
      () => dns.lookup("example.invalid", () => undefined),
      /Network access is disabled/,
    );
    assert.throws(
      () => dgram.createSocket("udp4"),
      /Network access is disabled/,
    );
    assert.throws(
      () => dns.promises.lookup("example.invalid"),
      /Network access is disabled/,
    );

    await assert.rejects(
      unauthorizedWindow.deidentify(golden.text),
      /Unauthorized OpenMed Electron IPC sender/,
    );
    assert.equal(utilityProcess.forkCount, 0);

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
    assert.deepEqual(utilityProcess.lastFork?.args, [
      "--openmed-model-path",
      modelPath,
    ]);
    assert.deepEqual(utilityProcess.lastFork?.options?.env, {
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
    });
    assert.equal(JSON.stringify(utilityProcess.lastFork).includes("Alice"), false);
    assert.equal(
      JSON.stringify(utilityProcess.lastMessage).includes(modelPath),
      false,
      "the model cache path must not be renderer-controlled message data",
    );
    assert.equal(modelLoadCount, 1, "model cache must be shared across windows");

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

test("Electron IPC rejects unauthorized, oversized, and overlapping data", async () => {
  let invokeCount = 0;
  const oversizedClient = createElectronDeidentifyClient({
    invoke: async () => {
      invokeCount += 1;
      return undefined;
    },
  });
  await assert.rejects(
    oversizedClient.deidentify("x".repeat(OPENMED_ELECTRON_MAX_TEXT_LENGTH + 1)),
    /Invalid OpenMed Electron request text/,
  );
  assert.equal(invokeCount, 0, "oversized text must not cross IPC");

  const invalidResponseClient = createElectronDeidentifyClient({
    invoke: async (_channel, request) => {
      const typedRequest = request as ElectronDeidentifyRequest;
      return {
        schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
        requestId: typedRequest.requestId,
        spans: [rendererSpan(0, 99)],
      };
    },
  });
  await assert.rejects(
    invalidResponseClient.deidentify("short"),
    /Invalid OpenMed Electron renderer spans/,
  );

  assert.throws(
    () => redactTextWithSpans("abcdef", [rendererSpan(0, 4), rendererSpan(3, 6)]),
    /Invalid OpenMed Electron renderer spans/,
  );
});

test("Electron service bounds pending work and replaces timed-out workers", async () => {
  const utilityProcess = new FakeUtilityProcessModule(
    async () => new Promise<never>(() => undefined),
  );
  const service = createService(utilityProcess, {
    requestTimeoutMs: 10,
    maxPendingRequests: 1,
  });

  try {
    const first = service.deidentify(request("first", "synthetic one"));
    await assert.rejects(
      service.deidentify(request("second", "synthetic two")),
      /service is busy/,
    );
    await assert.rejects(first, /timed out/);
    assert.equal(utilityProcess.forkCount, 1);
    assert.equal(utilityProcess.killCount, 1);
  } finally {
    service.dispose();
  }
});

test("a stale utility exit cannot reject a replacement worker request", async () => {
  const modelPath = join(rootDir, "tests", "web", "fixtures", "local-model-cache");
  const realHandler = createUtilityDeidentifyHandler({
    modelPath,
    loadPipeline: async () => fixturePipeline,
  });
  let releaseReplacement: (() => void) | undefined;
  const replacementGate = new Promise<void>((resolve) => {
    releaseReplacement = resolve;
  });
  const utilityProcess = new FakeUtilityProcessModule(async (message) => {
    const requestId = (message as { requestId?: unknown }).requestId;
    if (requestId === "first") {
      return new Promise<never>(() => undefined);
    }
    await replacementGate;
    return realHandler(message);
  });
  utilityProcess.suppressNextKillExit();
  const service = createService(utilityProcess, { requestTimeoutMs: 25 });

  try {
    await assert.rejects(
      service.deidentify(request("first", syntheticNote)),
      /timed out/,
    );
    const replacement = service.deidentify(request("second", syntheticNote));
    assert.equal(utilityProcess.children.length, 2);
    utilityProcess.children[0]?.emit("exit", 0);
    releaseReplacement?.();
    assert.equal((await replacement).requestId, "second");
  } finally {
    service.dispose();
  }
});

test("Electron service cleans up a synchronous utility send failure", async () => {
  const modelPath = join(rootDir, "tests", "web", "fixtures", "local-model-cache");
  const handler = createUtilityDeidentifyHandler({
    modelPath,
    loadPipeline: async () => fixturePipeline,
  });
  const utilityProcess = new FakeUtilityProcessModule(handler);
  utilityProcess.failNextPost();
  const service = createService(utilityProcess);

  try {
    await assert.rejects(
      service.deidentify(request("first", syntheticNote)),
      /communication failed/,
    );
    const response = await service.deidentify(request("second", syntheticNote));
    assert.equal(response.requestId, "second");
    assert.equal(utilityProcess.forkCount, 2);
  } finally {
    service.dispose();
  }
});

test("Electron utility pins one local model and serializes inference", async () => {
  const modelPath = join(rootDir, "tests", "web", "fixtures", "local-model-cache");
  let loadedPath = "";
  let loadCount = 0;
  let active = 0;
  let maximumActive = 0;
  const handler = createUtilityDeidentifyHandler({
    modelPath,
    loadPipeline: async (requestedPath) => {
      loadedPath = requestedPath;
      loadCount += 1;
      return async (text) => {
        active += 1;
        maximumActive = Math.max(maximumActive, active);
        await new Promise((resolve) => setTimeout(resolve, 5));
        const result = fixturePipeline(text);
        active -= 1;
        return result;
      };
    },
  });

  const firstRequest = {
    ...request("first", syntheticNote),
    type: "deidentify" as const,
    modelPath: "/renderer-controlled/path",
  };
  const secondRequest = {
    ...request("second", syntheticNote),
    type: "deidentify" as const,
  };
  const [first, second] = await Promise.all([
    handler(firstRequest),
    handler(secondRequest),
  ]);

  assert.equal(first.ok, true);
  assert.equal(second.ok, true);
  assert.equal(loadedPath, modelPath);
  assert.equal(loadCount, 1);
  assert.equal(maximumActive, 1);
  assert.equal(
    modelPathFromUtilityArguments([
      "electron-helper",
      "utility-process.js",
      "--openmed-model-path",
      modelPath,
    ]),
    modelPath,
  );
  assert.equal(
    modelPathFromUtilityArguments([
      "--openmed-model-path",
      modelPath,
      "--openmed-model-path",
      "/unexpected",
    ]),
    "",
  );

  const invalid = await handler({
    type: "deidentify",
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId: "Alice Nguyen",
    text: "synthetic",
  });
  assert.equal(invalid.ok, false);
  assert.equal(invalid.requestId, "invalid-request");
});

class FakeUtilityProcess extends EventEmitter implements UtilityProcessLike {
  constructor(
    private readonly handler: (message: unknown) => Promise<unknown>,
    private readonly onPost: (message: unknown) => void,
    private readonly onKill: () => void,
    private readonly emitExitOnKill: boolean,
  ) {
    super();
  }

  postMessage(message: unknown): void {
    this.onPost(message);
    void this.handler(message).then((response) => this.emit("message", response));
  }

  kill(): boolean {
    this.onKill();
    if (this.emitExitOnKill) {
      this.emit("exit", 0);
    }
    return true;
  }
}

class FakeUtilityProcessModule implements UtilityProcessModuleLike {
  forkCount = 0;
  killCount = 0;
  lastMessage: unknown;
  readonly children: FakeUtilityProcess[] = [];
  private postFailures = 0;
  private suppressedKillExits = 0;
  lastFork:
    | {
        workerPath: string;
        args: string[];
        options?: {
          env?: Record<string, string>;
          serviceName?: string;
          stdio?: "ignore";
        };
      }
    | undefined;

  constructor(
    private readonly handler: (message: unknown) => Promise<unknown>,
  ) {}

  failNextPost(): void {
    this.postFailures += 1;
  }

  suppressNextKillExit(): void {
    this.suppressedKillExits += 1;
  }

  fork(
    workerPath: string,
    args: string[] = [],
    options?: {
      env?: Record<string, string>;
      serviceName?: string;
      stdio?: "ignore";
    },
  ): UtilityProcessLike {
    this.forkCount += 1;
    this.lastFork = options
      ? { workerPath, args, options }
      : { workerPath, args };
    const emitExitOnKill = this.suppressedKillExits === 0;
    if (!emitExitOnKill) {
      this.suppressedKillExits -= 1;
    }
    const child = new FakeUtilityProcess(
      this.handler,
      (message) => {
        if (this.postFailures > 0) {
          this.postFailures -= 1;
          throw new Error("synthetic post failure");
        }
        this.lastMessage = message;
      },
      () => {
        this.killCount += 1;
      },
      emitExitOnKill,
    );
    this.children.push(child);
    return child;
  }
}

class FakeIpcMain implements IpcMainLike {
  private readonly handlers = new Map<
    string,
    (event: unknown, request: unknown) => Promise<unknown> | unknown
  >();

  handle(
    channel: string,
    listener: (event: unknown, request: unknown) => Promise<unknown> | unknown,
  ): void {
    this.handlers.set(channel, listener);
  }

  removeHandler(channel: string): void {
    this.handlers.delete(channel);
  }

  createRenderer(logs: unknown[], senderId: number) {
    return {
      invoke: async (channel: string, request: unknown): Promise<unknown> => {
        assert.equal(channel, OPENMED_DEIDENTIFY_CHANNEL);
        const handler = this.handlers.get(channel);
        assert.ok(handler);
        const response = await handler({ sender: { id: senderId } }, request);
        logs.push({ event: "deidentify_completed" });
        return response;
      },
    };
  }
}

function createService(
  utilityProcess: UtilityProcessModuleLike,
  options: Pick<
    ElectronDeidentifyServiceOptions,
    "requestTimeoutMs" | "maxPendingRequests"
  > = {},
): ElectronDeidentifyService {
  return new ElectronDeidentifyService({
    utilityProcess,
    workerPath: join(
      rootDir,
      "js",
      "openmedkit-electron",
      "dist",
      "utility-process.js",
    ),
    modelPath: join(
      rootDir,
      "tests",
      "web",
      "fixtures",
      "local-model-cache",
    ),
    ...options,
  });
}

function request(requestId: string, text: string): ElectronDeidentifyRequest {
  return {
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId,
    text,
  };
}

function rendererSpan(start: number, end: number): RendererOpenMedSpan {
  return {
    schema_version: 1,
    start,
    end,
    entity_type: "B-NAME",
    canonical_label: "PERSON",
    policy_label: "DIRECT_IDENTIFIER",
    score: 0.99,
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
