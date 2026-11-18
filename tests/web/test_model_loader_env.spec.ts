import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join } from "node:path";
import test from "node:test";

import type {
  RawTokenClassificationEntity,
  RawTokenClassificationPipeline,
  RawTransformersRuntime,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const packageDir = join(rootDir, "js", "openmedkit-web");
const distUrl = pathToFileURL(join(packageDir, "dist", "index.js")).href;

const LOCAL_MODEL = "/models/openmed-transformersjs";
const HUB_MODEL = "OpenMed/synthetic-token-classifier";

type RuntimeEnvironment = NonNullable<RawTransformersRuntime["env"]>;

const fixtureTokens: RawTokenClassificationEntity[] = [
  { entity: "B-PERSON", word: "Alice", score: 0.99 },
];

const fixturePipeline: RawTokenClassificationPipeline = () => fixtureTokens;

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

/** Let every pending microtask and timer callback run before asserting. */
async function settle(turns = 5): Promise<void> {
  for (let index = 0; index < turns; index += 1) {
    await new Promise<void>((resolve) => {
      setImmediate(resolve);
    });
  }
}

test("overlapping local and remote loads leave an absent-key environment absent", async () => {
  const api = await loadApi();
  const env: RuntimeEnvironment = {};
  const localCreation = deferred<RawTokenClassificationPipeline>();
  const remoteCreation = deferred<RawTokenClassificationPipeline>();
  const created: string[] = [];
  const runtime: RawTransformersRuntime = {
    env,
    pipeline: (_task, model) => {
      created.push(model);
      if (model === LOCAL_MODEL) {
        assert.equal(env.allowRemoteModels, false);
        assert.equal(env.allowLocalModels, true);
        return localCreation.promise;
      }
      assert.equal(env.allowRemoteModels, true);
      assert.equal(env.allowLocalModels, true);
      return remoteCreation.promise;
    },
  };

  const localLoad = api.loadTokenClassificationPipeline(LOCAL_MODEL, { runtime });
  await settle();
  assert.deepEqual(created, [LOCAL_MODEL]);

  const remoteLoad = api.loadTokenClassificationPipeline(HUB_MODEL, { runtime });
  await settle();
  assert.deepEqual(created, [LOCAL_MODEL]);
  assert.equal(env.allowRemoteModels, false);

  localCreation.resolve(fixturePipeline);
  await localLoad;
  await settle();
  assert.deepEqual(created, [LOCAL_MODEL, HUB_MODEL]);
  assert.equal(env.allowRemoteModels, true);
  assert.equal(env.allowLocalModels, true);

  remoteCreation.resolve(fixturePipeline);
  await remoteLoad;
  assert.deepEqual(Object.keys(env), []);
});

test("a local-only load never observes remote models enabled during creation", async () => {
  const api = await loadApi();
  const env: RuntimeEnvironment = {};
  const remoteCreation = deferred<RawTokenClassificationPipeline>();
  const localCreation = deferred<RawTokenClassificationPipeline>();
  const observed: Array<boolean | undefined> = [];
  const runtime: RawTransformersRuntime = {
    env,
    pipeline: (_task, model) => {
      observed.push(env.allowRemoteModels);
      return model === HUB_MODEL ? remoteCreation.promise : localCreation.promise;
    },
  };

  const remoteLoad = api.loadTokenClassificationPipeline(HUB_MODEL, { runtime });
  await settle();
  assert.deepEqual(observed, [true]);

  const localLoad = api.loadTokenClassificationPipeline(LOCAL_MODEL, { runtime });
  await settle();
  assert.deepEqual(observed, [true]);
  assert.equal(env.allowRemoteModels, true);

  remoteCreation.resolve(fixturePipeline);
  await remoteLoad;
  await settle();
  assert.deepEqual(observed, [true, false]);

  localCreation.resolve(fixturePipeline);
  await localLoad;
  assert.deepEqual(Object.keys(env), []);
});

test("loads that request the same settings share one environment window", async () => {
  const api = await loadApi();
  const env: RuntimeEnvironment = {};
  const firstCreation = deferred<RawTokenClassificationPipeline>();
  const secondCreation = deferred<RawTokenClassificationPipeline>();
  const created: string[] = [];
  const runtime: RawTransformersRuntime = {
    env,
    pipeline: (_task, model) => {
      created.push(model);
      assert.equal(env.allowRemoteModels, false);
      return model === LOCAL_MODEL ? firstCreation.promise : secondCreation.promise;
    },
  };

  const firstLoad = api.loadTokenClassificationPipeline(LOCAL_MODEL, { runtime });
  const secondLoad = api.loadTokenClassificationPipeline(
    `${LOCAL_MODEL}-second`,
    { runtime },
  );
  await settle();
  assert.deepEqual(created, [LOCAL_MODEL, `${LOCAL_MODEL}-second`]);
  assert.equal(env.allowRemoteModels, false);

  firstCreation.resolve(fixturePipeline);
  await firstLoad;
  assert.equal(env.allowRemoteModels, false);
  assert.equal(env.allowLocalModels, true);

  secondCreation.resolve(fixturePipeline);
  await secondLoad;
  assert.deepEqual(Object.keys(env), []);
});

test("a load restores host settings and unrelated runtime keys", async () => {
  const api = await loadApi();
  const env: RuntimeEnvironment = {
    allowLocalModels: false,
    allowRemoteModels: true,
    cacheDir: "/models/cache",
  };
  const runtime: RawTransformersRuntime = {
    env,
    pipeline: () => {
      assert.equal(env.allowRemoteModels, false);
      assert.equal(env.allowLocalModels, true);
      return fixturePipeline;
    },
  };

  await api.loadTokenClassificationPipeline(HUB_MODEL, {
    runtime,
    allowRemoteModels: false,
  });

  assert.deepEqual(env, {
    allowLocalModels: false,
    allowRemoteModels: true,
    cacheDir: "/models/cache",
  });
});

test("a rejected load releases the window for queued loads without leaking paths", async () => {
  const api = await loadApi();
  const env: RuntimeEnvironment = {};
  const created: string[] = [];
  const runtime: RawTransformersRuntime = {
    env,
    pipeline: (_task, model) => {
      created.push(model);
      if (model === LOCAL_MODEL) {
        return Promise.reject(new Error("synthetic pipeline creation failed"));
      }
      return fixturePipeline;
    },
  };

  const failingLoad = api.loadTokenClassificationPipeline(LOCAL_MODEL, { runtime });
  const queuedLoad = api.loadTokenClassificationPipeline(HUB_MODEL, { runtime });

  await assert.rejects(failingLoad, (error: Error) => {
    assert.equal(error.message, "synthetic pipeline creation failed");
    assert.doesNotMatch(error.message, /\/models\//);
    return true;
  });
  await queuedLoad;
  assert.deepEqual(created, [LOCAL_MODEL, HUB_MODEL]);
  assert.deepEqual(Object.keys(env), []);
});

test("sequential loads keep a per-call environment window", async () => {
  const api = await loadApi();
  const env: RuntimeEnvironment = {};
  const seen: Array<[boolean | undefined, boolean | undefined]> = [];
  const runtime: RawTransformersRuntime = {
    env,
    pipeline: () => {
      seen.push([env.allowLocalModels, env.allowRemoteModels]);
      return fixturePipeline;
    },
  };

  await api.loadTokenClassificationPipeline(LOCAL_MODEL, { runtime });
  assert.deepEqual(Object.keys(env), []);
  await api.loadTokenClassificationPipeline(HUB_MODEL, { runtime });
  assert.deepEqual(Object.keys(env), []);
  await api.loadOnnxModel(HUB_MODEL, { runtime });
  assert.deepEqual(Object.keys(env), []);

  assert.deepEqual(seen, [
    [true, false],
    [true, true],
    [true, true],
  ]);
});

async function loadApi() {
  return import(distUrl);
}
