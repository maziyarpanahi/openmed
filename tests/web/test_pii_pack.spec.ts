// The web PII task pack: candidate resolution from the bundled task pack
// manifest and ONNX catalog, local-first loading with injected runtimes, and
// parity with a direct extractPii() call. Every model id below comes from the
// committed manifest; runtimes and pipelines are synthetic.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { getModel } from "../../js/openmedkit-web/src/catalog";
import {
  DEFAULT_MODEL_ID,
  DEFAULT_MODEL_REVISION,
  TaskPackError,
  deidentify,
  extractPii,
  piiPack,
  setOfflineOnly,
  type RawTokenClassificationEntity,
  type RawTokenClassificationPipeline,
  type RawTransformersRuntime,
  type TaskPackResolution,
} from "../../js/openmedkit-web/src/index";
import { createTaskPack, type WebTaskPackData } from "../../js/openmedkit-web/src/packs";
import {
  TASK_PACKS_SCHEMA,
  TASK_PACKS_SCHEMA_VERSION,
  TASK_PACKS_SOURCE_DIGEST,
  WEB_TASK_PACKS,
} from "../../js/openmedkit-web/src/task-packs.generated";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const MANIFEST_PATH = join(rootDir, "openmed", "core", "data", "task_packs.json");
const TEXT = "Synthetic Person met on 2026-01-02";

interface ManifestPack {
  pack_id: string;
  version: string;
  task: string;
  canonical_labels: string[];
  notice_key: string;
  runtimes: Record<
    string,
    {
      format: string;
      status: string;
      candidates: { rank: number; repo_id: string; param_count: number; license: string }[];
    }
  >;
}

interface Manifest {
  schema: string;
  schema_version: string;
  notices: Record<string, string>;
  packs: ManifestPack[];
}

async function manifest(): Promise<{ bytes: Buffer; document: Manifest }> {
  const bytes = await readFile(MANIFEST_PATH);
  return { bytes, document: JSON.parse(bytes.toString("utf8")) as Manifest };
}

/** What openmed.core.task_packs.resolve(pack_id, "web") returns, as JSON. */
function pythonResolution(pack: ManifestPack) {
  const entry = pack.runtimes.web!;
  return {
    pack_id: pack.pack_id,
    pack_version: pack.version,
    task: pack.task,
    runtime: "web",
    format: entry.format,
    status: entry.status,
    candidates: entry.candidates.map(({ rank, repo_id, param_count, license }) => ({
      rank,
      repo_id,
      param_count,
      license,
    })),
    canonical_labels: pack.canonical_labels,
    notice_key: pack.notice_key,
  };
}

function pythonShape(resolution: TaskPackResolution) {
  return {
    pack_id: resolution.pack_id,
    pack_version: resolution.pack_version,
    task: resolution.task,
    runtime: resolution.runtime,
    format: resolution.format,
    status: resolution.status,
    candidates: resolution.candidates.map(({ rank, repo_id, param_count, license }) => ({
      rank,
      repo_id,
      param_count,
      license,
    })),
    canonical_labels: [...resolution.canonical_labels],
    notice_key: resolution.notice_key,
  };
}

const TOKENS: RawTokenClassificationEntity[] = [
  { entity: "B-PERSON", word: "Synthetic", score: 0.98, start: 0, end: 9 },
  { entity: "I-PERSON", word: "Person", score: 0.97, start: 10, end: 16 },
  { entity: "B-DATE", word: "2026-01-02", score: 0.91, start: 24, end: 34 },
];

/**
 * A synthetic Transformers.js runtime. `local` lists the repo ids that load
 * from local files; any other load with `local_files_only: false` "downloads"
 * through the global fetch, and the rest fail as a missing local cache would.
 */
function fakeRuntime(local: readonly string[] = [], env: Record<string, unknown> = {}) {
  const loads: {
    model: string;
    localFilesOnly: unknown;
    allowRemoteModels: unknown;
    revision: unknown;
  }[] = [];
  let disposed = 0;
  const runtime: RawTransformersRuntime = {
    env: { allowLocalModels: false, allowRemoteModels: true, ...env },
    pipeline: async (_task, model, options) => {
      loads.push({
        model,
        localFilesOnly: options?.local_files_only,
        allowRemoteModels: runtime.env?.allowRemoteModels,
        revision: options?.revision,
      });
      if (!local.includes(model)) {
        if (options?.local_files_only !== false) {
          throw new Error("synthetic: not in the local cache");
        }
        const response = await fetch(`https://hub.invalid/${model}`);
        if (!response.ok) {
          throw new Error("synthetic: download failed");
        }
      }
      return Object.assign((() => TOKENS) as RawTokenClassificationPipeline, {
        dispose: () => {
          disposed += 1;
        },
      });
    },
  };
  return { runtime, loads, disposals: () => disposed };
}

/** Replace the global fetch with a recorder for the duration of `body`. */
async function withFetch<T>(ok: boolean, body: (calls: string[]) => Promise<T>): Promise<T> {
  const calls: string[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: unknown) => {
    calls.push(String(input));
    return new Response("{}", { status: ok ? 200 : 503 });
  }) as typeof fetch;
  try {
    return await body(calls);
  } finally {
    globalThis.fetch = original;
  }
}

async function codeOf(promise: Promise<unknown>): Promise<string> {
  try {
    await promise;
  } catch (error) {
    assert.ok(error instanceof TaskPackError);
    assert.equal(error.message, `OpenMed task pack failed: ${error.code}`);
    return error.code;
  }
  assert.fail("expected a TaskPackError");
}

test("the pack resolves the same candidates, in order, as the Python resolver for web", async () => {
  const { document } = await manifest();
  const pack = document.packs.find((item) => item.pack_id === "pii")!;
  const resolution = piiPack.resolve();

  assert.equal(piiPack.id, "pii");
  assert.deepEqual(pythonShape(resolution), pythonResolution(pack));
  assert.equal(resolution.status, "resolved");
  assert.ok(resolution.candidates.length > 0);
  assert.deepEqual(
    resolution.candidates.map((candidate) => candidate.rank),
    resolution.candidates.map((_, index) => index + 1),
  );
  assert.equal(resolution.notice, document.notices[resolution.notice_key]);
  assert.ok(Object.isFrozen(resolution) && Object.isFrozen(resolution.candidates));
  // resolve() is pure: it reads bundled data and returns equal results.
  assert.deepEqual(piiPack.resolve(), resolution);
});

test("bundled pack data equal the manifest and every candidate is an ONNX catalog row", async () => {
  const { bytes, document } = await manifest();
  assert.equal(TASK_PACKS_SCHEMA, document.schema);
  assert.equal(TASK_PACKS_SCHEMA_VERSION, document.schema_version);
  assert.equal(
    TASK_PACKS_SOURCE_DIGEST,
    `sha256:${createHash("sha256").update(bytes).digest("hex")}`,
  );
  assert.deepEqual(
    WEB_TASK_PACKS.map((pack) => ({
      ...pack,
      candidates: pack.candidates.map(({ reproducibility_hash: _, ...rest }) => rest),
    })),
    [...document.packs].sort((a, b) => (a.pack_id < b.pack_id ? -1 : 1)).map(pythonResolution),
  );
  for (const pack of WEB_TASK_PACKS) {
    for (const candidate of pack.candidates) {
      const row = getModel(candidate.repo_id);
      assert.ok(row, "candidate missing from the web catalog");
      assert.ok(row.formats.some((format) => format.toLowerCase().startsWith("onnx")));
      assert.equal(row.license, candidate.license);
      assert.equal(row.param_count, candidate.param_count);
      assert.equal(row.reproducibility_hash, candidate.reproducibility_hash);
    }
  }
});

test("with an injected pipeline, extract equals a direct extractPii call", async () => {
  const pipeline: RawTokenClassificationPipeline & { dispose?: () => void } = () => TOKENS;
  let disposed = 0;
  pipeline.dispose = () => {
    disposed += 1;
  };
  const options = { hashSecret: "synthetic-pack-key", docId: "pack-doc", threshold: 0.5 };
  const pack = await withFetch(true, async (calls) => {
    const loaded = await piiPack.load({ pipeline });
    assert.deepEqual(calls, []);
    return loaded;
  });

  assert.equal(pack.candidate, null);
  assert.equal(pack.notice_key, "openmed.notice.pii.review_required");
  assert.equal(pack.notice, piiPack.resolve().notice);
  assert.deepEqual(await pack.extract(TEXT, options), await extractPii(TEXT, { ...options, pipeline }));
  assert.deepEqual(
    await pack.deidentify(TEXT, options),
    await deidentify(TEXT, { ...options, pipeline }),
  );
  // The caller owns a supplied pipeline.
  await pack.dispose();
  assert.equal(disposed, 0);
});

test("without remote consent no fetch is attempted and loading fails with a code", async () => {
  const candidates = piiPack.resolve().candidates.map((candidate) => candidate.repo_id);
  const { runtime, loads } = fakeRuntime();
  const code = await withFetch(true, async (calls) => {
    const result = await codeOf(piiPack.load({ runtime }));
    assert.equal(calls.length, 0);
    return result;
  });

  assert.equal(code, "local_model_unavailable");
  assert.deepEqual(
    loads.map((load) => load.model),
    candidates,
  );
  assert.ok(loads.every((load) => load.localFilesOnly === true && load.allowRemoteModels === false));
  // The host's runtime environment is restored afterwards.
  assert.equal(runtime.env?.allowRemoteModels, true);
});

test("a later candidate that is local loads without consent and without a fetch", async () => {
  const resolution = piiPack.resolve();
  const second = resolution.candidates[1] ?? resolution.candidates[0]!;
  const { runtime, loads } = fakeRuntime([second.repo_id]);
  await withFetch(true, async (calls) => {
    const pack = await piiPack.load({ runtime });
    assert.equal(calls.length, 0);
    assert.deepEqual(pack.candidate, second);
    assert.equal(loads.at(-1)!.model, second.repo_id);
    const spans = await pack.extract(TEXT, { hashSecret: "synthetic-pack-key" });
    assert.deepEqual(
      spans.map((span) => [span.canonical_label, span.start, span.end, span.metadata.model]),
      [
        ["PERSON", 0, 16, second.repo_id],
        ["DATE", 24, 34, second.repo_id],
      ],
    );
  });
});

test("allowRemoteModels: true is the only way a candidate is fetched", async () => {
  const first = piiPack.resolve().candidates[0]!;
  const { runtime, loads, disposals } = fakeRuntime();
  await withFetch(true, async (calls) => {
    const pack = await piiPack.load({ runtime, allowRemoteModels: true });
    assert.equal(pack.candidate?.repo_id, first.repo_id);
    assert.deepEqual(calls, [`https://hub.invalid/${first.repo_id}`]);
    assert.deepEqual(
      [loads[0]!.localFilesOnly, loads[0]!.allowRemoteModels],
      [false, true],
    );
    await pack.dispose();
    await pack.dispose();
    assert.equal(disposals(), 1);
  });

  const failing = fakeRuntime();
  await withFetch(false, async (calls) => {
    assert.equal(
      await codeOf(piiPack.load({ runtime: failing.runtime, allowRemoteModels: true })),
      "candidates_unavailable",
    );
    assert.equal(calls.length, piiPack.resolve().candidates.length);
  });
});

test("the default model candidate requests its pinned revision", async () => {
  const candidates = piiPack.resolve().candidates;
  assert.ok(candidates.some((candidate) => candidate.repo_id === DEFAULT_MODEL_ID));
  const { runtime, loads } = fakeRuntime([DEFAULT_MODEL_ID]);
  const pack = await piiPack.load({ runtime });
  assert.equal(pack.candidate?.repo_id, DEFAULT_MODEL_ID);
  for (const load of loads) {
    assert.equal(load.revision, load.model === DEFAULT_MODEL_ID ? DEFAULT_MODEL_REVISION : undefined);
  }
});

test("remote runtime assets and offline-only mode stop the search before any load", async () => {
  const remoteWasm = { backends: { onnx: { wasm: { wasmPaths: "https://cdn.invalid/ort/" } } } };
  const remote = fakeRuntime([DEFAULT_MODEL_ID], remoteWasm);
  await withFetch(true, async (calls) => {
    assert.equal(await codeOf(piiPack.load({ runtime: remote.runtime })), "remote_asset_refused");
    assert.equal(remote.loads.length, 0);
    assert.equal(calls.length, 0);
  });
  // Local WebAssembly assets satisfy the check.
  const local = fakeRuntime(
    piiPack.resolve().candidates.map((candidate) => candidate.repo_id),
    remoteWasm,
  );
  const pack = await piiPack.load({ runtime: local.runtime, wasmPaths: "/ort/" });
  assert.equal(pack.candidate?.rank, 1);

  // With consent the caller has accepted remote assets too.
  const consented = fakeRuntime([], remoteWasm);
  await withFetch(true, async () => {
    const loaded = await piiPack.load({ runtime: consented.runtime, allowRemoteModels: true });
    assert.equal(loaded.candidate?.rank, 1);
  });

  setOfflineOnly(true);
  try {
    const offline = fakeRuntime([DEFAULT_MODEL_ID]);
    assert.equal(await codeOf(piiPack.load({ runtime: offline.runtime })), "remote_model_refused");
    assert.equal(offline.loads.length, 0);
  } finally {
    setOfflineOnly(false);
  }
});

test("the pack logs nothing and its errors carry codes only", async () => {
  const methods = ["log", "info", "warn", "error", "debug", "trace"] as const;
  const originals = methods.map((name) => console[name]);
  let calls = 0;
  for (const name of methods) {
    console[name] = () => {
      calls += 1;
    };
  }
  try {
    const sentinel = "Quorble Zyxwick MRN-PLANTED-0042";
    const failing = fakeRuntime();
    await assert.rejects(piiPack.load({ runtime: failing.runtime }), (error: unknown) => {
      assert.ok(error instanceof TaskPackError);
      for (const candidate of piiPack.resolve().candidates) {
        assert.equal(error.message.includes(candidate.repo_id), false);
      }
      return true;
    });
    const working = fakeRuntime([DEFAULT_MODEL_ID]);
    const pack = await piiPack.load({ runtime: working.runtime });
    await pack.extract(sentinel, { hashSecret: "synthetic-pack-key" });
    await pack.deidentify(sentinel, { hashSecret: "synthetic-pack-key" });
  } finally {
    methods.forEach((name, index) => {
      console[name] = originals[index]!;
    });
  }
  assert.equal(calls, 0);
});

test("bundled data that disagree fail closed", async () => {
  const runtime = { extractPii, deidentify };
  const good: WebTaskPackData = WEB_TASK_PACKS.find((pack) => pack.pack_id === "pii")!;
  const notices = { [good.notice_key]: "synthetic notice" };
  const resolveWith = (packs: WebTaskPackData[], noticeText = notices) =>
    createTaskPack("pii", runtime, { packs, notices: noticeText }).resolve();
  const codeFrom = (action: () => unknown) => {
    try {
      action();
    } catch (error) {
      assert.ok(error instanceof TaskPackError);
      return error.code;
    }
    return "none";
  };

  assert.equal(codeFrom(() => createTaskPack("medication_dosing", runtime).resolve()), "unknown_pack");
  assert.equal(
    codeFrom(() =>
      resolveWith([{ ...good, candidates: good.candidates.map((item) => ({ ...item, rank: item.rank + 1 })) }]),
    ),
    "catalog_mismatch",
  );
  assert.equal(
    codeFrom(() =>
      resolveWith([
        { ...good, candidates: [{ ...good.candidates[0]!, reproducibility_hash: "sha256:x" }] },
      ]),
    ),
    "catalog_mismatch",
  );
  assert.equal(codeFrom(() => resolveWith([good], {})), "catalog_mismatch");
  assert.equal(codeFrom(() => resolveWith([{ ...good, candidates: [] }])), "catalog_mismatch");
  assert.equal(codeFrom(() => resolveWith([{ ...good, runtime: "android" }])), "catalog_mismatch");

  const empty = { ...good, status: "no_candidate", candidates: [] };
  const pack = createTaskPack("pii", runtime, { packs: [empty], notices });
  assert.equal(pack.resolve().status, "no_candidate");
  assert.equal(await codeOf(pack.load()), "no_candidate");

  // Synthetic loaders: the local one is used without consent, the remote one with it.
  const used: string[] = [];
  const loader = (name: string) => async () => {
    used.push(name);
    return (() => TOKENS) as never;
  };
  const injected = createTaskPack("pii", runtime, {
    packs: [good],
    notices,
    loadLocal: loader("local"),
    loadRemote: loader("remote"),
  });
  await injected.load();
  await injected.load({ allowRemoteModels: true });
  assert.deepEqual(used, ["local", "remote"]);
});
