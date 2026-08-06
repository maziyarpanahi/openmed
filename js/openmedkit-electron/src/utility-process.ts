import http from "node:http";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";

import {
  deidentify,
  loadTokenClassificationPipeline,
  type TokenClassificationPipeline,
} from "openmed";

import { toRendererOpenMedSpan, type UtilityDeidentifyResponse } from "./ipc";
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
  const loadPipeline =
    options.loadPipeline ??
    ((modelPath: string) =>
      loadTokenClassificationPipeline(modelPath, {
        allowRemoteModels: false,
        localFilesOnly: true,
      }));

  return async (message: unknown): Promise<UtilityDeidentifyResponse> => {
    if (!isUtilityDeidentifyMessage(message)) {
      return inferenceFailure(requestIdFrom(message), "INVALID_REQUEST");
    }
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
      return {
        type: "deidentify-result",
        requestId: message.requestId,
        ok: true,
        spans: result.spans.map(toRendererOpenMedSpan),
      };
    } catch {
      modelCache.delete(message.modelPath);
      return inferenceFailure(message.requestId, "INFERENCE_FAILED");
    }
  };
}

export function installOfflineNetworkGuard(): () => void {
  const originalFetch = globalThis.fetch;
  const originalHttpRequest = http.request;
  const originalHttpGet = http.get;
  const originalHttpsRequest = https.request;
  const originalHttpsGet = https.get;
  const originalNetConnect = net.connect;
  const originalNetCreateConnection = net.createConnection;
  const originalTlsConnect = tls.connect;
  const blocked = (): never => {
    throw new Error("Network access is disabled in the OpenMed utility process.");
  };

  globalThis.fetch = blocked as typeof globalThis.fetch;
  http.request = blocked as typeof http.request;
  http.get = blocked as typeof http.get;
  https.request = blocked as typeof https.request;
  https.get = blocked as typeof https.get;
  net.connect = blocked as typeof net.connect;
  net.createConnection = blocked as typeof net.createConnection;
  tls.connect = blocked as typeof tls.connect;

  return () => {
    globalThis.fetch = originalFetch;
    http.request = originalHttpRequest;
    http.get = originalHttpGet;
    https.request = originalHttpsRequest;
    https.get = originalHttpsGet;
    net.connect = originalNetConnect;
    net.createConnection = originalNetCreateConnection;
    tls.connect = originalTlsConnect;
  };
}

export function startUtilityProcess(parentPort: UtilityParentPortLike): void {
  installOfflineNetworkGuard();
  const handleMessage = createUtilityDeidentifyHandler();
  parentPort.on("message", (event) => {
    void handleMessage(event.data).then((response) => {
      parentPort.postMessage(response);
    });
  });
}

function requestIdFrom(message: unknown): string {
  if (typeof message === "object" && message !== null) {
    const requestId = (message as Record<string, unknown>).requestId;
    if (typeof requestId === "string") {
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
