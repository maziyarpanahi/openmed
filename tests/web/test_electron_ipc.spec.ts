import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import http from "node:http";
import net from "node:net";
import { join } from "node:path";
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
  createElectronDeidentifyClient,
  redactTextWithSpans,
  registerElectronDeidentifyIpc,
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
  const service = new ElectronDeidentifyService({
    utilityProcess,
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
  const unregister = registerElectronDeidentifyIpc(ipcMain, service);
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
    assert.deepEqual(utilityProcess.lastFork?.options?.env, {
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
    });
    assert.equal(JSON.stringify(utilityProcess.lastFork).includes("Alice"), false);
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

class FakeUtilityProcess extends EventEmitter implements UtilityProcessLike {
  constructor(
    private readonly handler: (message: unknown) => Promise<unknown>,
  ) {
    super();
  }

  postMessage(message: unknown): void {
    void this.handler(message).then((response) => this.emit("message", response));
  }

  kill(): boolean {
    this.emit("exit", 0);
    return true;
  }
}

class FakeUtilityProcessModule implements UtilityProcessModuleLike {
  forkCount = 0;
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
    return new FakeUtilityProcess(this.handler);
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

  createRenderer(logs: unknown[]) {
    return {
      invoke: async (channel: string, request: unknown): Promise<unknown> => {
        assert.equal(channel, OPENMED_DEIDENTIFY_CHANNEL);
        const handler = this.handlers.get(channel);
        assert.ok(handler);
        const response = await handler({ sender: "synthetic-window" }, request);
        logs.push({ event: "deidentify_completed" });
        return response;
      },
    };
  }
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
