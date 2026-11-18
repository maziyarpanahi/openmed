/**
 * In-process fakes for the OpenMedKit Electron tests.
 *
 * The fakes model the Electron behaviour the integration depends on: IPC
 * handlers with a sender `webContents`, asynchronous utility-process message
 * delivery, process exit and crash, and renderer teardown events.
 */
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  OPENMED_HARDENED_WEB_PREFERENCES,
  type ElectronOfflineSessionLike,
  type IpcMainLike,
  type UtilityProcessLike,
  type UtilityProcessModuleLike,
} from "../../js/openmedkit-electron/src/index";

export const rootDir = fileURLToPath(new URL("../..", import.meta.url));
export const workerPath = join(
  rootDir,
  "js",
  "openmedkit-electron",
  "dist",
  "utility-process.js",
);
export const legacyModelPath = join(
  rootDir,
  "tests",
  "web",
  "fixtures",
  "local-model-cache",
);
export const vectorsPath = join(
  rootDir,
  "tests",
  "fixtures",
  "portable_bundle",
  "vectors.json",
);

export type ChildFactory = () => (message: unknown) => Promise<unknown>;

/** Utility process that answers asynchronously, like Electron's parentPort. */
export class AsyncUtilityProcess extends EventEmitter implements UtilityProcessLike {
  readonly posted: unknown[] = [];
  killed = false;
  exited = false;

  constructor(private readonly handler: (message: unknown) => Promise<unknown>) {
    super();
  }

  postMessage(message: unknown): void {
    if (this.exited) {
      throw new Error("process gone");
    }
    this.posted.push(structuredClone(message));
    const copy = structuredClone(message);
    setImmediate(() => {
      void this.handler(copy).then(
        (response) => {
          if (response !== null && response !== undefined && !this.exited) {
            this.emit("message", structuredClone(response));
          }
        },
        () => undefined,
      );
    });
  }

  /** Deliver an arbitrary message as if the child had sent it. */
  inject(message: unknown): void {
    this.emit("message", message);
  }

  kill(): boolean {
    this.killed = true;
    this.exit(0);
    return true;
  }

  /** Simulate an abrupt crash of the utility process. */
  crash(): void {
    this.exit(1);
  }

  private exit(code: number): void {
    if (this.exited) {
      return;
    }
    this.exited = true;
    this.emit("exit", code);
  }
}

export class FakeUtilityProcessModule implements UtilityProcessModuleLike {
  readonly children: AsyncUtilityProcess[] = [];
  readonly forks: Array<{
    workerPath: string;
    args: string[];
    options?: {
      env?: Record<string, string>;
      session?: ElectronOfflineSessionLike;
      serviceName?: string;
      stdio?: "ignore";
    };
  }> = [];

  constructor(private readonly factory: ChildFactory) {}

  get forkCount(): number {
    return this.children.length;
  }

  get lastFork() {
    return this.forks.at(-1);
  }

  fork(
    modulePath: string,
    args: string[] = [],
    options?: {
      env?: Record<string, string>;
      session?: ElectronOfflineSessionLike;
      serviceName?: string;
      stdio?: "ignore";
    },
  ): UtilityProcessLike {
    this.forks.push(
      options ? { workerPath: modulePath, args, options } : { workerPath: modulePath, args },
    );
    const child = new AsyncUtilityProcess(this.factory());
    this.children.push(child);
    return child;
  }
}

export class ManualUtilityProcess extends EventEmitter implements UtilityProcessLike {
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

export class ManualUtilityProcessModule implements UtilityProcessModuleLike {
  readonly children: ManualUtilityProcess[] = [];

  fork(): UtilityProcessLike {
    const child = new ManualUtilityProcess();
    this.children.push(child);
    return child;
  }
}

/** A renderer `webContents` with lifecycle events and web preferences. */
export class FakeSender extends EventEmitter {
  constructor(
    readonly id: number,
    private readonly preferences: unknown = { ...OPENMED_HARDENED_WEB_PREFERENCES },
  ) {
    super();
  }

  getLastWebPreferences(): unknown {
    return this.preferences;
  }

  destroy(): void {
    this.emit("destroyed");
  }

  crashRenderer(): void {
    this.emit("render-process-gone");
  }
}

export interface FakeIpcEvent {
  sender: FakeSender | Record<string, unknown>;
}

export class FakeIpcMain implements IpcMainLike<FakeIpcEvent> {
  private readonly handlers = new Map<
    string,
    (event: FakeIpcEvent, request: unknown) => Promise<unknown> | unknown
  >();

  handle(
    channel: string,
    listener: (event: FakeIpcEvent, request: unknown) => Promise<unknown> | unknown,
  ): void {
    this.handlers.set(channel, listener);
  }

  removeHandler(channel: string): void {
    this.handlers.delete(channel);
  }

  channels(): string[] {
    return [...this.handlers.keys()].sort();
  }

  createRenderer(
    sender: FakeIpcEvent["sender"],
    logs: unknown[] = [],
  ): { invoke(channel: string, request: unknown): Promise<unknown> } {
    return {
      invoke: async (channel: string, request: unknown): Promise<unknown> => {
        const handler = this.handlers.get(channel);
        if (!handler) {
          throw new Error("No handler registered");
        }
        const response = await handler({ sender }, structuredClone(request));
        logs.push({ event: "ipc_completed" });
        return structuredClone(response);
      },
    };
  }
}

export class FakeOfflineSession implements ElectronOfflineSessionLike {
  private listener:
    | ((details: unknown, callback: (response: { cancel: boolean }) => void) => void)
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

/** A promise with external resolution, for ordering tests without timers. */
export function deferred<T = void>(): {
  promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

export async function until(
  predicate: () => boolean,
  label: string,
  attempts = 2_000,
): Promise<void> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (predicate()) {
      return;
    }
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  assert.fail(`condition not reached: ${label}`);
}

// ---------------------------------------------------------------------------
// Portable-bundle vectors
// ---------------------------------------------------------------------------

export interface PortableVectors {
  manifests: Record<
    string,
    { text?: string; base?: string; at?: number; delete?: number; insert?: string }
  >;
  envelopes: Record<string, string>;
  contexts: Record<string, RawVectorContext>;
  trust_policies: Record<string, unknown>;
  trees: Record<string, { path: string; text: string }[]>;
  cases: RawVectorCase[];
}

export interface RawVectorContext {
  now: string;
  target: {
    platform: string;
    sdk_name: string;
    sdk_version: string;
    runtimes: string[];
    os_version: string | null;
    policy_version: number | null;
    memory_bytes: number | null;
    storage_bytes: number | null;
  } | null;
  sequence_floors: Record<string, number>;
  understood_extensions: string[];
  max_manifest_bytes: number;
  max_envelope_bytes: number;
  max_total_bytes: number;
  max_files: number;
}

export interface RawVectorCase {
  id: string;
  manifest: string;
  envelope: string;
  context: string;
  trust_policy: string;
  tree: string | null;
  tree_remove: string[];
  tree_set: { path: string; text: string }[];
  expected: {
    ok: boolean;
    reason: string | null;
    content_id: string | null;
    valid_signatures: number;
    eligible_runtimes: number[];
    files_checked: number;
    files_verified: boolean;
  };
}

export function manifestText(vectors: PortableVectors, key: string): string {
  const entry = vectors.manifests[key];
  assert.ok(entry, key);
  if (entry.text !== undefined) {
    return entry.text;
  }
  const base = manifestText(vectors, entry.base as string);
  const at = entry.at as number;
  return base.slice(0, at) + (entry.insert as string) + base.slice(at + (entry.delete as number));
}

export function vectorCase(vectors: PortableVectors, id: string): RawVectorCase {
  const item = vectors.cases.find((candidate) => candidate.id === id);
  assert.ok(item, id);
  return item;
}

export function vectorTarget(raw: RawVectorContext) {
  return raw.target === null
    ? null
    : {
        platform: raw.target.platform,
        sdkName: raw.target.sdk_name,
        sdkVersion: raw.target.sdk_version,
        runtimes: raw.target.runtimes,
        osVersion: raw.target.os_version,
        policyVersion: raw.target.policy_version,
        memoryBytes: raw.target.memory_bytes,
        storageBytes: raw.target.storage_bytes,
      };
}

/** Write a vector case's manifest, envelope and tree into a new directory. */
export async function materializeBundle(
  vectors: PortableVectors,
  item: RawVectorCase,
  parent?: string,
): Promise<string> {
  const base = parent ?? (await mkdtemp(join(tmpdir(), "openmed-electron-bundle-")));
  const root = parent === undefined ? join(base, "bundle") : join(base, item.id);
  await mkdir(root, { recursive: true });
  await writeFile(join(root, "openmed-bundle.json"), manifestText(vectors, item.manifest));
  await writeFile(
    join(root, "openmed-bundle.sig.json"),
    vectors.envelopes[item.envelope] as string,
  );
  const files = new Map<string, string>();
  for (const entry of item.tree === null ? [] : (vectors.trees[item.tree] ?? [])) {
    files.set(entry.path, entry.text);
  }
  for (const path of item.tree_remove) {
    files.delete(path);
  }
  for (const entry of item.tree_set) {
    files.set(entry.path, entry.text);
  }
  for (const [path, text] of files) {
    const target = join(root, ...path.split("/"));
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, text);
  }
  return root;
}

export function expectedVerification(item: RawVectorCase) {
  return {
    ok: item.expected.ok,
    reason: item.expected.reason,
    contentId: item.expected.content_id,
    validSignatures: item.expected.valid_signatures,
    eligibleRuntimes: item.expected.eligible_runtimes,
    filesChecked: item.expected.files_checked,
    filesVerified: item.expected.files_verified,
  };
}
