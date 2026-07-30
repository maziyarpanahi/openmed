import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  analyzeText,
  deidentify,
  extractPii,
  loadModel,
  setOpenMedKitNativeModuleForTests,
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
