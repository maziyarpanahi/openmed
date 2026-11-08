import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  analyzeText,
  deidentify,
  extractPii,
  getOpenMedKitNativeModuleSource,
  loadModel,
  OpenMedKitBridgeError,
  setOpenMedKitNativeModuleForTests,
  type OpenMedKitBridgeErrorCode,
  type OpenMedKitNativeModule,
  type OpenMedSpan,
} from "../../js/openmedkit-react-native/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const packageDir = join(rootDir, "js", "openmedkit-react-native");
const publicSurfaceSnapshotPath = join(
  rootDir,
  "tests",
  "mobile",
  "__snapshots__",
  "openmedkit-react-native-public-api.json",
);

const fixtureText =
  "Patient Alice Nguyen was born on 1979-04-12. Email alice@example.org.";
const piiFragments = ["Alice", "Nguyen", "1979-04-12", "alice@example.org"];

test.afterEach(() => {
  setOpenMedKitNativeModuleForTests(null);
});

test("public runtime surface is snapshot-tested", async () => {
  const api = await import("../../js/openmedkit-react-native/src/index.ts");
  const snapshot = JSON.parse(
    await readFile(publicSurfaceSnapshotPath, "utf8"),
  ) as {
    exports: string[];
    packagePeerDependencies: unknown;
    packagePeerDependenciesMeta: unknown;
    packageScripts: unknown;
  };
  const packageJson = JSON.parse(
    await readFile(join(packageDir, "package.json"), "utf8"),
  ) as {
    peerDependencies: unknown;
    peerDependenciesMeta: unknown;
    scripts: unknown;
  };

  assert.deepEqual(Object.keys(api).sort(), snapshot.exports);
  assert.deepEqual(
    packageJson.peerDependencies,
    snapshot.packagePeerDependencies,
  );
  assert.deepEqual(
    packageJson.peerDependenciesMeta,
    snapshot.packagePeerDependenciesMeta,
  );
  assert.deepEqual(packageJson.scripts, snapshot.packageScripts);
});

test("empty hash keys are rejected before reaching native inference", async () => {
  const native = createNativeFixture("ios");
  const fail = async () => { throw new Error("native inference must not run"); };
  setOpenMedKitNativeModuleForTests({
    ...native,
    analyzeText: fail,
    extractPii: fail,
    deidentify: fail,
  });
  for (const call of [analyzeText, extractPii, deidentify]) {
    await assert.rejects(call(fixtureText, { hashSecret: "" }), /hashSecret must not be empty/);
  }
});

for (const platform of ["ios", "android"] as const) {
  test(`${platform} bridge returns spans matching the shared native fixture`, async () => {
    const native = createNativeFixture(platform);
    setOpenMedKitNativeModuleForTests(native);

    const backend = platform === "android" ? "onnx" : "mlx";
    const loadResult = await loadModel({
      modelPath: `/app/models/openmed-${platform}`,
      backend,
      cacheKey: `fixture-${platform}`,
    });
    const secondLoadResult = await loadModel({
      modelPath: `/app/models/openmed-${platform}`,
      backend,
      cacheKey: `fixture-${platform}`,
    });

    assert.deepEqual(loadResult, {
      cacheKey: `fixture-${platform}`,
      modelPath: `/app/models/openmed-${platform}`,
      backend,
      platform,
      loaded: true,
    });
    assert.equal(secondLoadResult.loaded, false);

    const analyzeSpans = await analyzeText(fixtureText, fixtureOptions);
    const extractSpans = await extractPii(fixtureText, fixtureOptions);

    assert.deepEqual(analyzeSpans, sharedKeepSpans);
    assert.deepEqual(extractSpans, sharedKeepSpans);
    assertNoPhiInSpans(analyzeSpans);
  });
}

test("deidentify returns redacted text and never logs raw PHI", async () => {
  const native = createNativeFixture("ios");
  const consoleCalls: string[] = [];
  const originalConsole = {
    error: console.error,
    info: console.info,
    log: console.log,
    warn: console.warn,
  };

  console.error = (...args: unknown[]) => {
    consoleCalls.push(args.join(" "));
  };
  console.info = (...args: unknown[]) => {
    consoleCalls.push(args.join(" "));
  };
  console.log = (...args: unknown[]) => {
    consoleCalls.push(args.join(" "));
  };
  console.warn = (...args: unknown[]) => {
    consoleCalls.push(args.join(" "));
  };

  try {
    setOpenMedKitNativeModuleForTests(native);
    await loadModel({
      modelPath: "/app/models/openmed-ios",
      backend: "mlx",
      cacheKey: "fixture-ios",
    });

    const result = await deidentify(fixtureText, {
      ...fixtureOptions,
      policy: "hipaa_safe_harbor",
    });

    assert.equal(
      result.deidentifiedText,
      "Patient [PERSON] was born on [DATE_OF_BIRTH]. Email [EMAIL].",
    );
    assert.deepEqual(result.spans, sharedRedactedSpans);
    assertNoPhiInSpans(result.spans);
    assert.equal(consoleCalls.length, 0);
  } finally {
    console.error = originalConsole.error;
    console.info = originalConsole.info;
    console.log = originalConsole.log;
    console.warn = originalConsole.warn;
  }
});

test("deidentify fails closed when native omits de-identified text", async () => {
  const native = createNativeFixture("ios");
  const malformedResponses: unknown[] = [
    { text: fixtureText, spans: sharedRedactedSpans },
    { text: fixtureText, deidentifiedText: null, spans: sharedRedactedSpans },
    { text: fixtureText, deidentifiedText: 42, spans: sharedRedactedSpans },
    { originalText: fixtureText },
    {},
    null,
  ];

  for (const response of malformedResponses) {
    setOpenMedKitNativeModuleForTests({
      ...native,
      deidentify: async () => response as never,
    });
    await assertBridgeError(
      deidentify(fixtureText, fixtureOptions),
      "openmedkit_deidentify_output_missing",
    );
  }
});

test("deidentify rejects unchanged text when native rewrote spans", async () => {
  const native = createNativeFixture("ios");
  setOpenMedKitNativeModuleForTests({
    ...native,
    deidentify: async () => ({
      text: fixtureText,
      deidentifiedText: fixtureText,
      spans: sharedRedactedSpans,
    }),
  });

  await assertBridgeError(
    deidentify(fixtureText, fixtureOptions),
    "openmedkit_deidentify_output_unredacted",
  );
});

test("deidentify accepts the redactedText alias and keep-only results", async () => {
  const native = createNativeFixture("android");
  setOpenMedKitNativeModuleForTests({
    ...native,
    deidentify: async () => ({
      redactedText: "Patient [PERSON] was born on [DATE_OF_BIRTH]. Email [EMAIL].",
      spans: sharedRedactedSpans,
    }),
  });
  const aliased = await deidentify(fixtureText, fixtureOptions);
  assert.equal(
    aliased.deidentifiedText,
    "Patient [PERSON] was born on [DATE_OF_BIRTH]. Email [EMAIL].",
  );
  assert.equal(aliased.text, fixtureText);

  setOpenMedKitNativeModuleForTests({
    ...native,
    deidentify: async () => ({
      text: fixtureText,
      deidentifiedText: fixtureText,
      spans: sharedKeepSpans,
    }),
  });
  const keepOnly = await deidentify(fixtureText, fixtureOptions);
  assert.equal(keepOnly.deidentifiedText, fixtureText);
  assert.deepEqual(keepOnly.spans, sharedKeepSpans);
});

test("test doubles are rejected in release and unrecognised runtimes", async () => {
  const native = createNativeFixture("ios");
  const releaseRuntimes: RuntimeFlags[] = [
    { dev: false, nodeEnv: undefined, testContext: "child-v8" },
    { dev: undefined, nodeEnv: "production", testContext: "child-v8" },
    { dev: true, nodeEnv: "production", testContext: undefined },
    { dev: undefined, nodeEnv: undefined, testContext: undefined },
  ];

  for (const flags of releaseRuntimes) {
    withRuntimeFlags(flags, () => {
      assert.throws(
        () => setOpenMedKitNativeModuleForTests(native),
        (error: unknown) =>
          isBridgeError(error, "openmedkit_test_double_forbidden"),
      );
      // Clearing a double must always work, including in release bundles.
      setOpenMedKitNativeModuleForTests(null);
    });
    assert.equal(getOpenMedKitNativeModuleSource(), "unavailable");
  }
});

test("test doubles are accepted in development and test runtimes", () => {
  const native = createNativeFixture("ios");
  const allowedRuntimes: RuntimeFlags[] = [
    { dev: true, nodeEnv: undefined, testContext: undefined },
    { dev: undefined, nodeEnv: "test", testContext: undefined },
    { dev: undefined, nodeEnv: "development", testContext: undefined },
    { dev: undefined, nodeEnv: undefined, testContext: "child-v8" },
  ];

  for (const flags of allowedRuntimes) {
    withRuntimeFlags(flags, () => {
      setOpenMedKitNativeModuleForTests(native);
      assert.equal(getOpenMedKitNativeModuleSource(), "test-double");
      setOpenMedKitNativeModuleForTests(null);
    });
  }
});

test("an installed double stops answering once the runtime is release", async () => {
  const native = createNativeFixture("ios");
  let nativeCalls = 0;
  setOpenMedKitNativeModuleForTests({
    ...native,
    async loadModel(source) {
      nativeCalls += 1;
      return native.loadModel(source);
    },
    async analyzeText() {
      nativeCalls += 1;
      return sharedKeepSpans;
    },
  });

  await withRuntimeFlagsAsync(
    { dev: false, nodeEnv: "production", testContext: undefined },
    async () => {
      await assertBridgeError(
        loadModel({ modelPath: "/app/models/openmed-ios", backend: "mlx" }),
        "openmedkit_test_double_forbidden",
      );
      await assertBridgeError(
        analyzeText(fixtureText, fixtureOptions),
        "openmedkit_test_double_forbidden",
      );
    },
  );
  assert.equal(nativeCalls, 0);
});

test("the bridge reports when no native module is linked", async () => {
  assert.equal(getOpenMedKitNativeModuleSource(), "unavailable");
  await assert.rejects(
    loadModel({ modelPath: "/app/models/openmed-ios", backend: "mlx" }),
    /OpenMedKitRN is not installed in the React Native runtime/,
  );
});

interface RuntimeFlags {
  dev: boolean | undefined;
  nodeEnv: string | undefined;
  testContext: string | undefined;
}

function withRuntimeFlags(flags: RuntimeFlags, body: () => void): void {
  const restore = applyRuntimeFlags(flags);
  try {
    body();
  } finally {
    restore();
  }
}

async function withRuntimeFlagsAsync(
  flags: RuntimeFlags,
  body: () => Promise<void>,
): Promise<void> {
  const restore = applyRuntimeFlags(flags);
  try {
    await body();
  } finally {
    restore();
  }
}

function applyRuntimeFlags(flags: RuntimeFlags): () => void {
  const scope = globalThis as { __DEV__?: boolean };
  const previous = {
    dev: Object.prototype.hasOwnProperty.call(scope, "__DEV__")
      ? scope.__DEV__
      : undefined,
    hadDev: Object.prototype.hasOwnProperty.call(scope, "__DEV__"),
    nodeEnv: process.env.NODE_ENV,
    testContext: process.env.NODE_TEST_CONTEXT,
  };
  setOrDelete(scope, "__DEV__", flags.dev);
  setEnv("NODE_ENV", flags.nodeEnv);
  setEnv("NODE_TEST_CONTEXT", flags.testContext);
  return () => {
    if (previous.hadDev) {
      scope.__DEV__ = previous.dev;
    } else {
      delete scope.__DEV__;
    }
    setEnv("NODE_ENV", previous.nodeEnv);
    setEnv("NODE_TEST_CONTEXT", previous.testContext);
  };
}

function setOrDelete(
  scope: { __DEV__?: boolean },
  key: "__DEV__",
  value: boolean | undefined,
): void {
  if (value === undefined) {
    delete scope[key];
  } else {
    scope[key] = value;
  }
}

function setEnv(key: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[key];
  } else {
    process.env[key] = value;
  }
}

function isBridgeError(error: unknown, code: OpenMedKitBridgeErrorCode): boolean {
  assert.ok(error instanceof OpenMedKitBridgeError);
  assert.equal(error.code, code);
  for (const fragment of piiFragments) {
    assert.equal(error.message.includes(fragment), false);
  }
  return true;
}

async function assertBridgeError(
  promise: Promise<unknown>,
  code: OpenMedKitBridgeErrorCode,
): Promise<void> {
  await assert.rejects(promise, (error: unknown) => isBridgeError(error, code));
}

const fixtureOptions = {
  docId: "rn-fixture",
  hashSecret: "fixture-secret",
  detector: "openmedkit-native-fixture",
  metadata: { fixture: "simulator-parity" },
};

const sharedKeepSpans: OpenMedSpan[] = [
  span({
    start: 8,
    end: 20,
    text_hash:
      "hmac-sha256:7f42b1f8ec1bbff19ef8cf25ef95dc45d9486182b94a0b3fb4f3ca930bbd5a59",
    entity_type: "PERSON",
    canonical_label: "PERSON",
    score: 0.99,
  }),
  span({
    start: 33,
    end: 43,
    text_hash:
      "hmac-sha256:8c385628c5b09d88ab813bb27f305c793f37f3ef9a1b2c9f74ccb7e8479da24f",
    entity_type: "DATE_OF_BIRTH",
    canonical_label: "DATE_OF_BIRTH",
    score: 0.98,
  }),
  span({
    start: 51,
    end: 68,
    text_hash:
      "hmac-sha256:bd72f3d754422ae0d14d40a7fd16b7200eb2e427f404915cb8d5c41bf96ded39",
    entity_type: "EMAIL",
    canonical_label: "EMAIL",
    score: 0.97,
  }),
];

const sharedRedactedSpans: OpenMedSpan[] = [
  { ...sharedKeepSpans[0]!, action: "mask", replacement: "[PERSON]" },
  {
    ...sharedKeepSpans[1]!,
    action: "mask",
    replacement: "[DATE_OF_BIRTH]",
  },
  { ...sharedKeepSpans[2]!, action: "mask", replacement: "[EMAIL]" },
];

function createNativeFixture(platform: "ios" | "android"): OpenMedKitNativeModule {
  let loadedCacheKey: string | null = null;
  return {
    async loadModel(source) {
      const loaded = loadedCacheKey !== source.cacheKey;
      loadedCacheKey = source.cacheKey;
      return {
        cacheKey: source.cacheKey,
        modelPath: source.modelPath,
        backend: source.backend,
        platform,
        loaded,
      };
    },
    async analyzeText() {
      assert.ok(loadedCacheKey);
      return sharedKeepSpans;
    },
    async extractPii() {
      assert.ok(loadedCacheKey);
      return { spans: sharedKeepSpans };
    },
    async deidentify() {
      assert.ok(loadedCacheKey);
      return {
        text: fixtureText,
        deidentifiedText:
          "Patient [PERSON] was born on [DATE_OF_BIRTH]. Email [EMAIL].",
        spans: sharedRedactedSpans,
      };
    },
  };
}

function span(input: {
  start: number;
  end: number;
  text_hash: OpenMedSpan["text_hash"];
  entity_type: string;
  canonical_label: string;
  score: number;
}): OpenMedSpan {
  return {
    schema_version: 1,
    doc_id: "rn-fixture",
    start: input.start,
    end: input.end,
    text_hash: input.text_hash,
    entity_type: input.entity_type,
    canonical_label: input.canonical_label,
    policy_label: "DIRECT_IDENTIFIER",
    regulatory_tags: [],
    score: input.score,
    detector: "openmedkit-native-fixture",
    evidence: {},
    action: "keep",
    replacement: null,
    reversible_id: null,
    section: null,
    metadata: { fixture: "simulator-parity" },
  };
}

function assertNoPhiInSpans(spans: OpenMedSpan[]) {
  const encoded = JSON.stringify(spans);
  for (const fragment of piiFragments) {
    assert.equal(encoded.includes(fragment), false);
  }
}
