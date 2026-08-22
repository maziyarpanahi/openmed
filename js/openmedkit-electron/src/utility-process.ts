import childProcess from "node:child_process";
import dgram from "node:dgram";
import dns from "node:dns";
import dnsPromises from "node:dns/promises";
import http from "node:http";
import http2 from "node:http2";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";

import {
  deidentify,
  loadTokenClassificationPipeline,
  type TokenClassificationPipeline,
} from "openmed";

import {
  assertRendererSpans,
  toRendererOpenMedSpan,
  type UtilityDeidentifyRequest,
  type UtilityDeidentifyResponse,
} from "./ipc";
import { inferenceFailure, isUtilityDeidentifyMessage } from "./main-service";

export interface UtilityMessageEvent {
  data: unknown;
}

export interface UtilityParentPortLike {
  on(event: "message", listener: (event: UtilityMessageEvent) => void): this;
  postMessage(message: unknown): void;
}

export interface UtilityHandlerOptions {
  loadPipeline?: (modelPath: string) => Promise<TokenClassificationPipeline>;
}

export function createUtilityDeidentifyHandler(
  options: UtilityHandlerOptions = {},
): (message: unknown) => Promise<UtilityDeidentifyResponse> {
  const modelCache = new Map<string, Promise<TokenClassificationPipeline>>();
  let inferenceQueue: Promise<void> = Promise.resolve();
  const loadPipeline =
    options.loadPipeline ??
    ((modelPath: string) =>
      loadTokenClassificationPipeline(modelPath, {
        allowRemoteModels: false,
        localFilesOnly: true,
      }));

  const run = async (
    message: UtilityDeidentifyRequest,
  ): Promise<UtilityDeidentifyResponse> => {
    try {
      let pipeline = modelCache.get(message.modelPath);
      if (!pipeline) {
        pipeline = loadPipeline(message.modelPath);
        modelCache.set(message.modelPath, pipeline);
      }
      const result = await deidentify(message.text, {
        pipeline: await pipeline,
        docId: "electron-document",
        detector: "electron-utility-process",
      });
      const spans = result.spans.map(toRendererOpenMedSpan);
      assertRendererSpans(spans, message.text.length);
      return {
        type: "deidentify-result",
        requestId: message.requestId,
        ok: true,
        spans,
      };
    } catch {
      modelCache.delete(message.modelPath);
      return inferenceFailure(message.requestId, "INFERENCE_FAILED");
    }
  };

  return async (message: unknown): Promise<UtilityDeidentifyResponse> => {
    if (!isUtilityDeidentifyMessage(message)) {
      return inferenceFailure(requestIdFrom(message), "INVALID_REQUEST");
    }
    const response = inferenceQueue.then(() => run(message));
    inferenceQueue = response.then(
      () => undefined,
      () => undefined,
    );
    return response;
  };
}

export function installOfflineNetworkGuard(): () => void {
  const restorers: Array<() => void> = [];
  const blocked = (): never => {
    throw new Error("Network access is disabled in the OpenMed utility process.");
  };

  blockMethods(globalThis, ["fetch", "WebSocket", "EventSource"], blocked, restorers);
  blockMethods(http, ["request", "get"], blocked, restorers);
  blockMethods(https, ["request", "get"], blocked, restorers);
  blockMethods(http2, ["connect"], blocked, restorers);
  blockMethods(net, ["connect", "createConnection"], blocked, restorers);
  blockMethods(tls, ["connect"], blocked, restorers);
  blockMethods(dgram, ["createSocket"], blocked, restorers);
  blockMethods(
    dns,
    [
      "lookup",
      "lookupService",
      "resolve",
      "resolve4",
      "resolve6",
      "resolveAny",
      "resolveCaa",
      "resolveCname",
      "resolveMx",
      "resolveNaptr",
      "resolveNs",
      "resolvePtr",
      "resolveSoa",
      "resolveSrv",
      "resolveTxt",
      "reverse",
    ],
    blocked,
    restorers,
  );
  blockMethods(
    dnsPromises,
    [
      "lookup",
      "lookupService",
      "resolve",
      "resolve4",
      "resolve6",
      "resolveAny",
      "resolveCaa",
      "resolveCname",
      "resolveMx",
      "resolveNaptr",
      "resolveNs",
      "resolvePtr",
      "resolveSoa",
      "resolveSrv",
      "resolveTxt",
      "reverse",
    ],
    blocked,
    restorers,
  );
  blockMethods(
    childProcess,
    ["exec", "execFile", "execFileSync", "execSync", "fork", "spawn", "spawnSync"],
    blocked,
    restorers,
  );

  return () => {
    for (const restore of restorers.reverse()) {
      restore();
    }
  };
}

export function startUtilityProcess(parentPort: UtilityParentPortLike): void {
  installOfflineNetworkGuard();
  const handleMessage = createUtilityDeidentifyHandler();
  parentPort.on("message", (event) => {
    void handleMessage(event.data).then((response) => {
      try {
        parentPort.postMessage(response);
      } catch {
        // The main process timeout owns recovery; never serialize process errors.
      }
    });
  });
}

function blockMethods(
  target: object,
  methodNames: readonly string[],
  blocked: () => never,
  restorers: Array<() => void>,
): void {
  const mutableTarget = target as Record<string, unknown>;
  for (const methodName of methodNames) {
    const original = mutableTarget[methodName];
    if (typeof original !== "function") {
      continue;
    }
    mutableTarget[methodName] = blocked;
    restorers.push(() => {
      mutableTarget[methodName] = original;
    });
  }
}

function requestIdFrom(message: unknown): string {
  if (typeof message === "object" && message !== null) {
    const requestId = (message as Record<string, unknown>).requestId;
    if (
      typeof requestId === "string" &&
      requestId.length > 0 &&
      requestId.length <= 80 &&
      /^[A-Za-z0-9_-]+$/.test(requestId)
    ) {
      return requestId;
    }
  }
  return "invalid-request";
}

const electronProcess = process as NodeJS.Process & {
  parentPort?: UtilityParentPortLike | null;
};
if (electronProcess.parentPort) {
  startUtilityProcess(electronProcess.parentPort);
}
