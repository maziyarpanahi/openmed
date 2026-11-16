// Portable SDK configuration for the web package, replayed from the shared
// vectors in tests/fixtures/parity/sdk_config_v1.json (Python reference:
// openmed.core.sdk_config.validate_sdk_config).
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  DEFAULT_MODEL_ID,
  SDK_CONFIG_DEFAULTS,
  SDK_CONFIG_FIELDS,
  SDK_CONFIG_ISSUE_CODES,
  SDK_CONFIG_ONNX_VARIANTS,
  SDK_CONFIG_POLICY_PROFILES,
  SDK_CONFIG_SCHEMA_ID,
  SDK_CONFIG_VERSION,
  SdkConfigError,
  disposeDefaultPipelines,
  extractPii,
  loadOnnxModel,
  loadPortablePolicy,
  loadTokenClassificationPipeline,
  parseSdkConfig,
  portablePolicyNames,
  toExtractPiiOptions,
  toLoadModelOptions,
  validateSdkConfig,
  type OpenMedSdkConfig,
  type RawTokenClassificationEntity,
  type RawTransformersRuntime,
  type SdkConfigIssue,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "sdk_config_v1.json");
const SCHEMA_PATH = join(rootDir, "openmed", "core", "schemas", "json", "sdk_config.schema.json");
const DTS_PATH = join(rootDir, "js", "openmedkit-web", "dist", "index.d.ts");
const HUB_MODEL = "synthetic-org/synthetic-pii-model-onnx";

interface Vectors {
  schema: string;
  version: number;
  synthetic: boolean;
  sdk_config_schema: string;
  sdk_config_version: number;
  fields: string[];
  defaults: Record<string, unknown>;
  issue_codes: string[];
  valid: { id: string; config: unknown; resolved: Record<string, unknown> }[];
  invalid: { id: string; config: unknown; errors: SdkConfigIssue[] }[];
}

let cached: Promise<Vectors> | undefined;
function vectors(): Promise<Vectors> {
  cached ??= readFile(VECTORS_PATH, "utf8").then((text) => JSON.parse(text) as Vectors);
  return cached;
}

/** A fake Transformers.js runtime that records what each load observed. */
function recordingRuntime() {
  const tokens: RawTokenClassificationEntity[] = [
    { entity: "B-PERSON", word: "Synthetic", score: 0.6, start: 0, end: 9 },
    { entity: "B-DATE", word: "today", score: 0.4, start: 14, end: 19 },
  ];
  const loads: {
    model: string;
    allowRemoteModels: unknown;
    allowLocalModels: unknown;
    options: Record<string, unknown>;
  }[] = [];
  const runtime: RawTransformersRuntime = {
    env: { allowLocalModels: false, allowRemoteModels: true },
    pipeline: async (_task, model, options) => {
      loads.push({
        model,
        allowRemoteModels: runtime.env?.allowRemoteModels,
        allowLocalModels: runtime.env?.allowLocalModels,
        options: { ...options },
      });
      return () => tokens;
    },
  };
  return { runtime, loads };
}

test("the shared vectors, the schema and the web constants agree", async () => {
  const fixture = await vectors();
  const schema = JSON.parse(await readFile(SCHEMA_PATH, "utf8")) as {
    $id: string;
    "x-sdk-config-version": number;
    properties: Record<string, { default: unknown; enum?: string[] }>;
  };

  assert.equal(fixture.schema, "openmed.tests.sdk_config_parity");
  assert.equal(fixture.version, 1);
  assert.equal(fixture.synthetic, true);
  assert.equal(fixture.sdk_config_schema, SDK_CONFIG_SCHEMA_ID);
  assert.equal(schema.$id, SDK_CONFIG_SCHEMA_ID);
  assert.equal(fixture.sdk_config_version, SDK_CONFIG_VERSION);
  assert.equal(schema["x-sdk-config-version"], SDK_CONFIG_VERSION);
  assert.deepEqual(fixture.fields, [...SDK_CONFIG_FIELDS]);
  assert.deepEqual(Object.keys(schema.properties).sort(), [...SDK_CONFIG_FIELDS].sort());
  assert.deepEqual(fixture.defaults, { ...SDK_CONFIG_DEFAULTS });
  assert.deepEqual(fixture.issue_codes, [...SDK_CONFIG_ISSUE_CODES]);
  for (const field of SDK_CONFIG_FIELDS) {
    assert.deepEqual(schema.properties[field]!.default, SDK_CONFIG_DEFAULTS[field], field);
  }
  assert.deepEqual(schema.properties.onnx_variant!.enum, [...SDK_CONFIG_ONNX_VARIANTS]);
  assert.deepEqual(schema.properties.policy_profile!.enum, [...SDK_CONFIG_POLICY_PROFILES]);
  // Every portable profile name is bundled, so loadPortablePolicy() accepts it.
  assert.deepEqual([...SDK_CONFIG_POLICY_PROFILES], portablePolicyNames());
  assert.equal(SDK_CONFIG_DEFAULTS.local_only, true);
  assert.ok(Object.isFrozen(SDK_CONFIG_DEFAULTS));
});

test("valid vectors parse to their resolved form", async () => {
  const fixture = await vectors();
  assert.ok(fixture.valid.length >= 10);
  for (const item of fixture.valid) {
    assert.deepEqual(validateSdkConfig(item.config), [], item.id);
    const config = parseSdkConfig(item.config);
    assert.deepEqual({ ...config }, item.resolved, item.id);
    assert.deepEqual(Object.keys(config), [...SDK_CONFIG_FIELDS], item.id);
    assert.ok(Object.isFrozen(config), item.id);
    // A resolved document is itself valid and resolves to itself.
    assert.deepEqual(parseSdkConfig(JSON.parse(JSON.stringify(config))), config, item.id);
    assert.equal(loadPortablePolicy(config.policy_profile).name, config.policy_profile, item.id);
  }
});

test("invalid vectors fail with exactly the fixture's ordered issues", async () => {
  const fixture = await vectors();
  assert.ok(fixture.invalid.length >= 30);
  for (const item of fixture.invalid) {
    assert.deepEqual(validateSdkConfig(item.config), item.errors, item.id);
    assert.throws(
      () => parseSdkConfig(item.config),
      (error: unknown) => {
        assert.ok(error instanceof SdkConfigError, item.id);
        assert.equal(error.code, "sdk_config_invalid", item.id);
        assert.deepEqual(error.issues, item.errors, item.id);
        assert.ok(Object.isFrozen(error.issues), item.id);
        // Messages name codes and known fields only, never values or keys.
        const values =
          item.config !== null && typeof item.config === "object"
            ? Object.entries(item.config).flatMap(([key, value]) => [key, String(value)])
            : [String(item.config)];
        for (const value of values) {
          if (
            value.length >= 3 &&
            !(SDK_CONFIG_FIELDS as readonly string[]).includes(value) &&
            !(SDK_CONFIG_ISSUE_CODES as readonly string[]).includes(value)
          ) {
            assert.equal(error.message.includes(value), false, item.id);
          }
        }
        return true;
      },
      item.id,
    );
  }
});

test("only plain JSON objects are configuration documents", () => {
  const notObject = [{ code: "config_not_object", field: null }];
  for (const value of [new Map(), new Date(0), Object.create({ local_only: true }), undefined, () => {}]) {
    assert.deepEqual(validateSdkConfig(value), notObject);
  }
  assert.deepEqual(validateSdkConfig(Object.create(null)), []);
  const inherited = Object.create(null);
  inherited.local_only = false;
  assert.equal(parseSdkConfig(inherited).local_only, false);
  // JSON.parse creates an own "__proto__" member: it is an unknown key.
  assert.deepEqual(validateSdkConfig(JSON.parse('{"__proto__": {"local_only": false}}')), [
    { code: "unknown_field", field: null },
  ]);
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.deepEqual(validateSdkConfig({ confidence_threshold: value }), [
      { code: "out_of_range", field: "confidence_threshold" },
    ]);
  }
  assert.deepEqual(validateSdkConfig({ model_id: undefined }), [
    { code: "wrong_type", field: "model_id" },
  ]);
});

test("local_only maps to local files only, and false is an explicit opt-in", () => {
  assert.deepEqual(toLoadModelOptions(parseSdkConfig({})), {
    localFilesOnly: true,
    allowRemoteModels: false,
  });
  assert.deepEqual(toLoadModelOptions(parseSdkConfig({ local_only: false })), {
    localFilesOnly: false,
    allowRemoteModels: true,
  });
  const pinned = parseSdkConfig({
    model_id: HUB_MODEL,
    model_revision: "0123456789abcdef0123456789abcdef01234567",
    onnx_variant: "fp16",
    confidence_threshold: 0.75,
  });
  assert.deepEqual(toLoadModelOptions(pinned), {
    localFilesOnly: true,
    allowRemoteModels: false,
    revision: "0123456789abcdef0123456789abcdef01234567",
    variant: "fp16",
  });
  assert.deepEqual(toExtractPiiOptions(pinned), {
    threshold: 0.75,
    model: HUB_MODEL,
    loaderOptions: toLoadModelOptions(pinned),
  });
  // A null model keeps the package default model and its pinned revision.
  assert.deepEqual(toExtractPiiOptions(parseSdkConfig({})), {
    threshold: 0.5,
    loaderOptions: { localFilesOnly: true, allowRemoteModels: false },
  });

  // The helpers revalidate, so a hand-built object cannot bypass the schema.
  const forged = { ...SDK_CONFIG_DEFAULTS, local_only: "no" } as unknown as OpenMedSdkConfig;
  assert.throws(() => toLoadModelOptions(forged), SdkConfigError);
  assert.throws(() => toExtractPiiOptions({ ...forged, extra: 1 } as OpenMedSdkConfig), {
    issues: [
      { code: "unknown_field", field: null },
      { code: "wrong_type", field: "local_only" },
    ],
  });
});

test("local_only: true refuses remote models in the existing loaders", async () => {
  const offline = toLoadModelOptions(parseSdkConfig({ model_id: HUB_MODEL }));
  const online = toLoadModelOptions(parseSdkConfig({ model_id: HUB_MODEL, local_only: false }));

  const first = recordingRuntime();
  await loadTokenClassificationPipeline(HUB_MODEL, { ...offline, runtime: first.runtime });
  assert.deepEqual(first.loads.map((load) => [load.allowRemoteModels, load.allowLocalModels]), [
    [false, true],
  ]);
  assert.equal(first.loads[0]!.options.local_files_only, true);
  assert.equal(first.loads[0]!.options.localFilesOnly, true);
  // The host's environment is restored after the load.
  assert.equal(first.runtime.env?.allowRemoteModels, true);

  const second = recordingRuntime();
  await loadTokenClassificationPipeline(HUB_MODEL, { ...online, runtime: second.runtime });
  assert.equal(second.loads[0]!.allowRemoteModels, true);
  assert.equal(second.loads[0]!.options.local_files_only, false);

  const third = recordingRuntime();
  const variant = toLoadModelOptions(parseSdkConfig({ onnx_variant: "fp16" }));
  await loadOnnxModel(DEFAULT_MODEL_ID, { ...variant, runtime: third.runtime });
  assert.equal(third.loads[0]!.allowRemoteModels, false);
  assert.equal(third.loads[0]!.options.model_file_name, "model_fp16");
});

test("extraction options carry the threshold, model and offline loader", async () => {
  const config = parseSdkConfig({ model_id: HUB_MODEL, confidence_threshold: 0.5 });
  const options = toExtractPiiOptions(config);
  const recorded = recordingRuntime();
  try {
    const spans = await extractPii("Synthetic met today", {
      ...options,
      hashSecret: "synthetic-sdk-config-key",
      loaderOptions: { ...options.loaderOptions, runtime: recorded.runtime },
    });
    assert.deepEqual(recorded.loads.map((load) => [load.model, load.allowRemoteModels]), [
      [HUB_MODEL, false],
    ]);
    // The 0.4 DATE token is below the configured threshold.
    assert.deepEqual(spans.map((span) => span.canonical_label), ["PERSON"]);
  } finally {
    await disposeDefaultPipelines();
  }
});

test("the new exports reach the built declarations", async () => {
  // npm test builds dist/ before this runs, as test_npm_deidentify.spec.ts needs.
  const declarations = await readFile(DTS_PATH, "utf8");
  for (const name of [
    "OpenMedSdkConfig",
    "SdkConfigIssue",
    "SdkConfigError",
    "parseSdkConfig",
    "validateSdkConfig",
    "toLoadModelOptions",
    "toExtractPiiOptions",
    "SDK_CONFIG_DEFAULTS",
  ]) {
    assert.match(declarations, new RegExp(`\\b${name}\\b`), name);
  }
  assert.match(
    declarations,
    /declare function toLoadModelOptions\(config: OpenMedSdkConfig\): LoadOnnxModelOptions;/,
  );
  assert.match(
    declarations,
    /declare function toExtractPiiOptions\(config: OpenMedSdkConfig\): ExtractPiiOptions;/,
  );
});
