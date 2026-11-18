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
  PortableBundleError,
  deidentify,
  loadTokenClassificationPipeline,
  parsePortableBundleTrustPolicy,
  type PortableBundleContext,
  type TokenClassificationPipeline,
} from "openmed";

import { verifyPortableBundleDirectory } from "./bundle-directory";
import { assertRendererSpans, isValidRequestId, toRendererOpenMedSpan } from "./ipc";
import {
  isUtilityCancelMessage,
  isUtilityDeidentifyMessage,
  isUtilityLoadBundleMessage,
  utilityFailure,
  type ElectronBundleReason,
  type UtilityBundleRejectedMessage,
  type UtilityBundleVerification,
  type UtilityDeidentifyRequest,
  type UtilityLoadBundleMessage,
  type UtilityOutboundMessage,
} from "./utility-protocol";

export interface UtilityMessageEvent {
  data: unknown;
}

export interface UtilityParentPortLike {
  on(event: "message", listener: (event: UtilityMessageEvent) => void): this;
  postMessage(message: unknown): void;
}

/** The runtime variant chosen from a verified bundle. */
export interface VerifiedBundleRuntime {
  readonly index: number;
  readonly id: string;
  readonly format: string;
  readonly precision: string;
  readonly entrypoint: string;
}

/**
 * A portable bundle that passed signature, policy, compatibility and file-tree
 * verification inside the utility process.
 */
export interface VerifiedElectronBundle {
  readonly root: string;
  readonly generation: number;
  readonly contentId: string;
  readonly manifest: Readonly<Record<string, unknown>>;
  readonly runtime: VerifiedBundleRuntime;
  /** The exact verified bytes, by relative path. */
  readonly files: ReadonlyMap<string, Uint8Array>;
}

export interface UtilityHandlerOptions {
  /** Loader for legacy, unverified model directories. */
  loadPipeline?: (modelPath: string) => Promise<TokenClassificationPipeline>;
  /** Loader for a verified bundle's selected runtime. */
  loadBundlePipeline?: (
    bundle: VerifiedElectronBundle,
  ) => Promise<TokenClassificationPipeline>;
  /** Runtime formats the bundle loader can run, in preference order. */
  supportedRuntimeFormats?: readonly string[];
}

/**
 * Handles one inbound message and resolves its response, or `null` for a
 * message that has none (cancellation).
 */
export type UtilityMessageHandler = (
  message: unknown,
) => Promise<UtilityOutboundMessage | null>;

interface ActiveBundle {
  generation: number;
  contentId: string;
  pipeline: TokenClassificationPipeline;
}

const DEFAULT_RUNTIME_FORMATS = ["onnx", "transformersjs"] as const;

/**
 * Default bundle loader: the `openmed` Transformers.js loader with remote
 * models disabled, pointed at the selected runtime's entrypoint.
 *
 * Transformers.js reads the files from disk again after verification, so the
 * bundle directory must be app-owned and read-only (for example inside the
 * app's resources). Loaders that need a stronger guarantee should build the
 * pipeline from `bundle.files`.
 */
export async function loadVerifiedBundlePipeline(
  bundle: VerifiedElectronBundle,
): Promise<TokenClassificationPipeline> {
  const entrypoint = bundle.runtime.entrypoint;
  const slash = entrypoint.lastIndexOf("/");
  return loadTokenClassificationPipeline(bundle.root, {
    allowRemoteModels: false,
    localFilesOnly: true,
    quantized: false,
    pipelineOptions: {
      subfolder: slash < 0 ? "" : entrypoint.slice(0, slash),
      model_file_name: entrypoint.slice(slash + 1).replace(/\.onnx$/, ""),
    },
  });
}

/**
 * Create the utility-process message handler.
 *
 * All inference and bundle loads run through one serialized queue, so a
 * request queued before a bundle change runs on the generation it was
 * submitted for, and a request queued after it runs only once the new
 * bundle has verified. Cancellation is applied immediately: a queued job is
 * skipped and a running job's output is discarded. Every job produces
 * exactly one result message.
 *
 * Args:
 *   options: Optional legacy and bundle pipeline loaders.
 *
 * Returns:
 *   The message handler.
 */
export function createUtilityDeidentifyHandler(
  options: UtilityHandlerOptions = {},
): UtilityMessageHandler {
  const modelCache = new Map<string, Promise<TokenClassificationPipeline>>();
  const known = new Set<string>();
  const cancelled = new Set<string>();
  let queue: Promise<void> = Promise.resolve();
  let active: ActiveBundle | undefined;
  const loadPipeline =
    options.loadPipeline ??
    ((modelPath: string) =>
      loadTokenClassificationPipeline(modelPath, {
        allowRemoteModels: false,
        localFilesOnly: true,
      }));
  const loadBundlePipeline = options.loadBundlePipeline ?? loadVerifiedBundlePipeline;
  const formats = [...(options.supportedRuntimeFormats ?? DEFAULT_RUNTIME_FORMATS)];

  const enqueue = <T>(work: () => Promise<T>): Promise<T> => {
    const result = queue.then(work);
    queue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  };

  const finish = (requestId: string): boolean => {
    known.delete(requestId);
    return cancelled.delete(requestId);
  };

  const loadBundle = async (
    message: UtilityLoadBundleMessage,
  ): Promise<UtilityOutboundMessage> => {
    const previous = active;
    active = undefined;
    disposePipeline(previous?.pipeline);
    const reject = (
      reason: ElectronBundleReason,
      verification: UtilityBundleVerification | null,
    ): UtilityBundleRejectedMessage => ({
      type: "bundle-rejected",
      generation: message.generation,
      reason,
      verification,
    });
    const { source } = message;
    let result: Awaited<ReturnType<typeof verifyPortableBundleDirectory>>;
    try {
      const context: PortableBundleContext = {
        now: message.now,
        target: source.target,
        sequenceFloors: source.sequenceFloors,
        understoodExtensions: source.understoodExtensions,
        ...source.limits,
      };
      result = await verifyPortableBundleDirectory(source.root, {
        trustPolicy: parsePortableBundleTrustPolicy(source.trustPolicy),
        context,
      });
    } catch (error) {
      return reject(
        error instanceof PortableBundleError ? error.reason : "bundle_source_invalid",
        null,
      );
    }
    const verification = summarize(result.verification);
    if (!result.verification.ok || result.manifest === null || result.files === null) {
      return reject(
        (result.verification.reason ?? "bundle_source_invalid") as ElectronBundleReason,
        verification,
      );
    }
    const contentId = result.verification.contentId as string;
    if (source.expectedContentId !== null && source.expectedContentId !== contentId) {
      return reject("content_id_unexpected", verification);
    }
    const runtimes = (result.manifest.runtimes ?? []) as Array<Record<string, unknown>>;
    const candidates = result.verification.eligibleRuntimes
      .map((index) => ({ index, runtime: runtimes[index] }))
      .filter(
        (item): item is { index: number; runtime: Record<string, unknown> } =>
          item.runtime !== undefined && formats.includes(item.runtime.format as string),
      )
      .sort(
        (left, right) =>
          formats.indexOf(left.runtime.format as string) -
            formats.indexOf(right.runtime.format as string) || left.index - right.index,
      );
    if (candidates.length === 0) {
      return reject("runtime_unsupported", verification);
    }
    for (const candidate of candidates) {
      const runtime: VerifiedBundleRuntime = {
        index: candidate.index,
        id: candidate.runtime.id as string,
        format: candidate.runtime.format as string,
        precision: candidate.runtime.precision as string,
        entrypoint: candidate.runtime.entrypoint as string,
      };
      try {
        const pipeline = await loadBundlePipeline({
          root: source.root,
          generation: message.generation,
          contentId,
          manifest: result.manifest,
          runtime,
          files: result.files,
        });
        active = { generation: message.generation, contentId, pipeline };
        return {
          type: "bundle-ready",
          generation: message.generation,
          contentId,
          runtimeId: runtime.id,
          verification,
        };
      } catch {
        // Try the next eligible runtime; failures stay value-free.
      }
    }
    return reject("pipeline_load_failed", verification);
  };

  const run = async (
    message: UtilityDeidentifyRequest,
  ): Promise<UtilityOutboundMessage> => {
    const generation = message.generation ?? null;
    if (cancelled.has(message.requestId)) {
      finish(message.requestId);
      return utilityFailure(message.requestId, "CANCELLED", generation);
    }
    let pipeline: TokenClassificationPipeline;
    let modelPath: string | undefined;
    if (generation !== null) {
      if (active === undefined || active.generation !== generation) {
        finish(message.requestId);
        return utilityFailure(message.requestId, "BUNDLE_UNAVAILABLE", generation);
      }
      pipeline = active.pipeline;
    } else {
      modelPath = message.modelPath as string;
      let cached = modelCache.get(modelPath);
      if (!cached) {
        cached = loadPipeline(modelPath);
        modelCache.set(modelPath, cached);
      }
      try {
        pipeline = await cached;
      } catch {
        modelCache.delete(modelPath);
        return utilityFailure(
          message.requestId,
          finish(message.requestId) ? "CANCELLED" : "INFERENCE_FAILED",
          generation,
        );
      }
    }
    try {
      const result = await deidentify(message.text, {
        pipeline,
        docId: "electron-document",
        detector: "electron-utility-process",
      });
      const spans = result.spans.map(toRendererOpenMedSpan);
      assertRendererSpans(spans, message.text.length);
      if (finish(message.requestId)) {
        return utilityFailure(message.requestId, "CANCELLED", generation);
      }
      return {
        type: "deidentify-result",
        requestId: message.requestId,
        ok: true,
        spans,
        generation,
      };
    } catch {
      if (modelPath !== undefined) {
        modelCache.delete(modelPath);
      }
      return utilityFailure(
        message.requestId,
        finish(message.requestId) ? "CANCELLED" : "INFERENCE_FAILED",
        generation,
      );
    }
  };

  return async (message: unknown): Promise<UtilityOutboundMessage | null> => {
    if (isUtilityCancelMessage(message)) {
      if (known.has(message.requestId)) {
        cancelled.add(message.requestId);
      }
      return null;
    }
    if (isUtilityLoadBundleMessage(message)) {
      return enqueue(() => loadBundle(message));
    }
    if (isUtilityDeidentifyMessage(message) && !known.has(message.requestId)) {
      known.add(message.requestId);
      return enqueue(() => run(message));
    }
    return utilityFailure(requestIdFrom(message), "INVALID_REQUEST");
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
  blockMethods(dns, DNS_METHODS, blocked, restorers);
  blockMethods(dnsPromises, DNS_METHODS, blocked, restorers);
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

/**
 * Start the OpenMed utility-process loop on Electron's `parentPort`.
 *
 * Installs the offline network guard, then answers each message in queue
 * order. Responses never include exception text.
 *
 * Args:
 *   parentPort: `process.parentPort` inside an Electron utility process.
 *   options: Optional pipeline loaders (for example a synthetic test loader).
 */
export function startUtilityProcess(
  parentPort: UtilityParentPortLike,
  options: UtilityHandlerOptions = {},
): void {
  installOfflineNetworkGuard();
  const handleMessage = createUtilityDeidentifyHandler(options);
  parentPort.on("message", (event) => {
    void handleMessage(event?.data).then(
      (response) => {
        if (response === null) {
          return;
        }
        try {
          parentPort.postMessage(response);
        } catch {
          // The main process timeout owns recovery; never serialize process errors.
        }
      },
      () => undefined,
    );
  });
}

const DNS_METHODS = [
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
] as const;

function summarize(
  verification: Awaited<ReturnType<typeof verifyPortableBundleDirectory>>["verification"],
): UtilityBundleVerification {
  return {
    ok: verification.ok,
    reason: verification.reason,
    contentId: verification.contentId,
    validSignatures: verification.validSignatures,
    eligibleRuntimes: [...verification.eligibleRuntimes],
    filesChecked: verification.filesChecked,
    filesVerified: verification.filesVerified,
  };
}

function disposePipeline(pipeline: TokenClassificationPipeline | undefined): void {
  const dispose = (pipeline as { dispose?: () => unknown } | undefined)?.dispose;
  if (typeof dispose === "function") {
    try {
      void Promise.resolve(dispose.call(pipeline)).catch(() => undefined);
    } catch {
      // Disposal is best-effort; the retired pipeline is never used again.
    }
  }
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
    if (isValidRequestId(requestId)) {
      return requestId;
    }
  }
  return "invalid-request";
}
