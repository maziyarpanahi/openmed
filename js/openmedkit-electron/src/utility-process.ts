import dgram from "node:dgram";
import dns from "node:dns";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import { isAbsolute } from "node:path";
import tls from "node:tls";

import {
  deidentify,
  loadTokenClassificationPipeline,
  type TokenClassificationPipeline,
} from "openmed";

import {
  OPENMED_UTILITY_MODEL_PATH_ARGUMENT,
  assertRendererOpenMedSpans,
  toRendererOpenMedSpan,
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
  modelPath: string;
  loadPipeline?: (modelPath: string) => Promise<TokenClassificationPipeline>;
}

export function createUtilityDeidentifyHandler(
  options: UtilityHandlerOptions,
): (message: unknown) => Promise<UtilityDeidentifyResponse> {
  if (!isAbsolute(options.modelPath)) {
    throw new TypeError("The OpenMed model cache path must be absolute.");
  }
  let pipeline: Promise<TokenClassificationPipeline> | undefined;
  let workQueue = Promise.resolve();
  const loadPipeline =
    options.loadPipeline ??
    ((modelPath: string) =>
      loadTokenClassificationPipeline(modelPath, {
        allowRemoteModels: false,
        localFilesOnly: true,
      }));

  return (message: unknown): Promise<UtilityDeidentifyResponse> => {
    if (!isUtilityDeidentifyMessage(message)) {
      return Promise.resolve(
        inferenceFailure(requestIdFrom(message), "INVALID_REQUEST"),
      );
    }

    const execute = async (): Promise<UtilityDeidentifyResponse> => {
      let currentPipeline = pipeline;
      if (!currentPipeline) {
        currentPipeline = loadPipeline(options.modelPath);
        pipeline = currentPipeline;
      }
      try {
        const result = await deidentify(message.text, {
          pipeline: await currentPipeline,
          docId: "electron-document",
          detector: "electron-utility-process",
        });
        const spans = result.spans.map(toRendererOpenMedSpan);
        assertRendererOpenMedSpans(spans, message.text.length);
        return {
          type: "deidentify-result",
          schemaVersion: message.schemaVersion,
          requestId: message.requestId,
          ok: true,
          spans,
        };
      } catch {
        if (pipeline === currentPipeline) {
          pipeline = undefined;
        }
        return inferenceFailure(message.requestId, "INFERENCE_FAILED");
      }
    };

    const response = workQueue.then(execute);
    workQueue = response.then(
      () => undefined,
      () => undefined,
    );
    return response;
  };
}

export function installOfflineNetworkGuard(): () => void {
  const originalFetch = globalThis.fetch;
  const originalDgramCreateSocket = dgram.createSocket;
  const originalDnsLookup = dns.lookup;
  const originalDnsResolve = dns.resolve;
  const originalDnsResolve4 = dns.resolve4;
  const originalDnsResolve6 = dns.resolve6;
  const originalDnsReverse = dns.reverse;
  const originalDnsPromisesLookup = dns.promises.lookup;
  const originalDnsPromisesResolve = dns.promises.resolve;
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
  dgram.createSocket = blocked as typeof dgram.createSocket;
  dns.lookup = blocked as unknown as typeof dns.lookup;
  dns.resolve = blocked as unknown as typeof dns.resolve;
  dns.resolve4 = blocked as unknown as typeof dns.resolve4;
  dns.resolve6 = blocked as unknown as typeof dns.resolve6;
  dns.reverse = blocked as typeof dns.reverse;
  dns.promises.lookup = blocked as typeof dns.promises.lookup;
  dns.promises.resolve = blocked as typeof dns.promises.resolve;
  http.request = blocked as typeof http.request;
  http.get = blocked as typeof http.get;
  https.request = blocked as typeof https.request;
  https.get = blocked as typeof https.get;
  net.connect = blocked as typeof net.connect;
  net.createConnection = blocked as typeof net.createConnection;
  tls.connect = blocked as typeof tls.connect;

  let restored = false;
  return () => {
    if (restored) {
      return;
    }
    restored = true;
    globalThis.fetch = originalFetch;
    dgram.createSocket = originalDgramCreateSocket;
    dns.lookup = originalDnsLookup;
    dns.resolve = originalDnsResolve;
    dns.resolve4 = originalDnsResolve4;
    dns.resolve6 = originalDnsResolve6;
    dns.reverse = originalDnsReverse;
    dns.promises.lookup = originalDnsPromisesLookup;
    dns.promises.resolve = originalDnsPromisesResolve;
    http.request = originalHttpRequest;
    http.get = originalHttpGet;
    https.request = originalHttpsRequest;
    https.get = originalHttpsGet;
    net.connect = originalNetConnect;
    net.createConnection = originalNetCreateConnection;
    tls.connect = originalTlsConnect;
  };
}

export function startUtilityProcess(
  parentPort: UtilityParentPortLike,
  modelPath: string,
): void {
  installOfflineNetworkGuard();
  const handleMessage = createUtilityDeidentifyHandler({ modelPath });
  parentPort.on("message", (event) => {
    void handleMessage(event.data).then(
      (response) => safePostMessage(parentPort, response),
      () =>
        safePostMessage(
          parentPort,
          inferenceFailure(requestIdFrom(event.data), "INFERENCE_FAILED"),
        ),
    );
  });
}

export function modelPathFromUtilityArguments(args: readonly string[]): string {
  const matches = args.flatMap((argument, index) =>
    argument === OPENMED_UTILITY_MODEL_PATH_ARGUMENT ? [index] : [],
  );
  if (matches.length !== 1) {
    return "";
  }
  return args[(matches[0] as number) + 1] ?? "";
}

function safePostMessage(
  parentPort: UtilityParentPortLike,
  message: UtilityDeidentifyResponse,
): void {
  try {
    parentPort.postMessage(message);
  } catch {
    // The main process owns timeout and worker replacement behavior.
  }
}

function requestIdFrom(message: unknown): unknown {
  if (typeof message === "object" && message !== null) {
    return (message as Record<string, unknown>).requestId;
  }
  return undefined;
}

const electronProcess = process as NodeJS.Process & {
  parentPort?: UtilityParentPortLike | null;
};
if (electronProcess.parentPort) {
  startUtilityProcess(
    electronProcess.parentPort,
    modelPathFromUtilityArguments(process.argv),
  );
}
