// Shared error-code vectors for the web package
// (tests/fixtures/parity/error_codes_v1.json, built from the R2 registry in
// openmed/core/error_codes.json). Every scenario whose `web` cell is
// `required` runs with the fixture's synthetic inputs and must throw an
// OpenMedError whose `errorCode` is the scenario's code. Only the code is
// part of the contract; messages are not, but they must stay value-free.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  OPENMED_ERROR_CODES,
  OPENMED_ERROR_CODE_RETRYABLE,
  OpenMedError,
  OpenMedOfflineError,
  PortablePolicyError,
  alignTokenOffsets,
  assertOfflineAssetPath,
  decidePortablePolicy,
  deidentify,
  extractPii,
  hmacTextHash,
  loadOnnxModel,
  loadPortablePolicy,
  loadTokenClassificationPipeline,
  openMedErrorCode,
  selectOrtWebBackend,
  setOfflineOnly,
  type OpenMedErrorCode,
  type RawTransformersRuntime,
} from "../../js/openmedkit-web/src/index";
import { portablePolicyErrorCode } from "../../js/openmedkit-web/src/portable-policy";
import { assertOfflineExternalData } from "../../js/openmedkit-web/src/runtime/ort-web-loader";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "error_codes_v1.json");
const ERRORS_SOURCE = join(rootDir, "js", "openmedkit-web", "src", "errors.ts");

interface Cell {
  status: "required" | "pending" | "not_applicable";
  entry?: string[];
  today?: string;
  reason?: string;
}

interface Scenario {
  id: string;
  summary: string;
  code: OpenMedErrorCode;
  inputs: Record<string, unknown>;
  runtimes: Record<string, Cell>;
}

interface Vectors {
  schema: string;
  schema_version: number;
  synthetic: boolean;
  codes: { code: string; retryable: boolean; scenarios: string[] }[];
  scenarios: Scenario[];
}

let cached: Promise<Vectors> | undefined;
function vectors(): Promise<Vectors> {
  cached ??= readFile(VECTORS_PATH, "utf8").then((text) => JSON.parse(text) as Vectors);
  return cached;
}

/** A Transformers.js runtime whose local model files are all missing. */
function emptyCacheRuntime(): RawTransformersRuntime {
  return {
    env: { allowLocalModels: false, allowRemoteModels: true },
    pipeline: async () => {
      throw new Error("synthetic: no local model files");
    },
  };
}

type Trigger = (inputs: Record<string, any>) => unknown;

/** Web triggers, one list per scenario id; each must throw the scenario code. */
const WEB_TRIGGERS: Readonly<Record<string, readonly Trigger[]>> = {
  empty_hash_secret: [
    (inputs) =>
      extractPii("Synthetic text", { pipeline: () => [], hashSecret: inputs.secret }),
    (inputs) =>
      deidentify("Synthetic text", { pipeline: () => [], hashSecret: inputs.secret }),
    (inputs) => hmacTextHash("Synthetic text", inputs.secret),
  ],
  invalid_offsets: [
    (inputs) =>
      decidePortablePolicy(loadPortablePolicy("hipaa_safe_harbor"), "x".repeat(inputs.text_length), [
        { label: "PERSON", start: inputs.start, end: inputs.end, confidence: 0.9 },
      ]),
  ],
  unknown_policy_profile: [(inputs) => loadPortablePolicy(inputs.profile)],
  model_missing_offline: [
    // A local path whose files are missing, as the vector describes.
    async (inputs) => {
      setOfflineOnly(inputs.offline);
      try {
        await loadTokenClassificationPipeline(`/models/${inputs.model_id}`, {
          runtime: emptyCacheRuntime(),
        });
      } finally {
        setOfflineOnly(false);
      }
    },
    // The Hub id itself: offline-only mode refuses it before any lookup.
    async (inputs) => {
      setOfflineOnly(inputs.offline);
      try {
        await loadOnnxModel(inputs.model_id, { runtime: emptyCacheRuntime() });
      } finally {
        setOfflineOnly(false);
      }
    },
  ],
};

async function thrown(trigger: () => unknown): Promise<unknown> {
  try {
    await trigger();
  } catch (error) {
    return error;
  }
  assert.fail("expected the trigger to throw");
}

function inputValues(inputs: Record<string, unknown>): string[] {
  return Object.values(inputs)
    .filter((value): value is string => typeof value === "string" && value.length >= 3);
}

test("the vectors carry the registry's portable codes and retryable flags", async () => {
  const fixture = await vectors();
  assert.equal(fixture.schema, "openmed.error_code_vectors");
  assert.equal(fixture.schema_version, 1);
  assert.equal(fixture.synthetic, true);
  assert.deepEqual(
    fixture.codes.map((item) => item.code).sort(),
    Object.values(OPENMED_ERROR_CODES).sort(),
  );
  for (const item of fixture.codes) {
    assert.equal(
      OPENMED_ERROR_CODE_RETRYABLE[item.code as OpenMedErrorCode],
      item.retryable,
      item.code,
    );
  }
});

test("every required web scenario throws an OpenMedError with its code", async () => {
  const fixture = await vectors();
  const required = fixture.scenarios.filter((item) => item.runtimes.web?.status === "required");
  assert.ok(required.length >= 4);
  assert.deepEqual(
    Object.keys(WEB_TRIGGERS).sort(),
    required.map((item) => item.id).sort(),
    "every required web scenario has a trigger and no trigger is stale",
  );
  for (const scenario of required) {
    for (const [index, trigger] of WEB_TRIGGERS[scenario.id]!.entries()) {
      const tag = `scenario=${scenario.id} trigger=${index}`;
      const error = await thrown(() => trigger(scenario.inputs));
      assert.ok(error instanceof Error, tag);
      assert.ok(error instanceof OpenMedError, tag);
      assert.equal(error.errorCode, scenario.code, tag);
      assert.equal(openMedErrorCode(error), scenario.code, tag);
      assert.equal(error.retryable, OPENMED_ERROR_CODE_RETRYABLE[scenario.code], tag);
      for (const value of inputValues(scenario.inputs)) {
        assert.equal(error.message.includes(value), false, `${tag} message echoes an input`);
        assert.equal(String(error).includes(value), false, `${tag} string echoes an input`);
      }
    }
  }
});

test("pending and not-applicable web cells give their reason and stay untriggered", async () => {
  const fixture = await vectors();
  for (const scenario of fixture.scenarios) {
    const cell = scenario.runtimes.web;
    assert.ok(cell, scenario.id);
    if (cell.status !== "required") {
      assert.ok(cell.reason && cell.reason.length > 0, scenario.id);
      assert.equal(Object.hasOwn(WEB_TRIGGERS, scenario.id), false, scenario.id);
    }
  }
  // The web SDK configuration loader is #5389; the cell flips when it lands.
  const sdk = fixture.scenarios.find((item) => item.id === "invalid_sdk_config")!;
  assert.equal(sdk.runtimes.web!.status, "pending");
  assert.match(sdk.runtimes.web!.reason!, /#5389/);
});

test("errors that had reason codes keep them beside the registry code", async () => {
  const unknown = await thrown(() => loadPortablePolicy("synthetic_unknown_profile"));
  assert.ok(unknown instanceof PortablePolicyError && unknown instanceof OpenMedError);
  assert.deepEqual(
    [unknown.name, unknown.code, unknown.errorCode, unknown.fieldPath, unknown.message],
    ["PortablePolicyError", "profile_unknown", "policy_error", "$.name", "profile_unknown at $.name"],
  );
  const offsets = await thrown(() =>
    decidePortablePolicy(loadPortablePolicy("hipaa_safe_harbor"), "x".repeat(14), [
      { label: "PERSON", start: 5, end: 2, confidence: 0.9 },
    ]),
  );
  assert.ok(offsets instanceof PortablePolicyError);
  assert.deepEqual([offsets.code, offsets.errorCode], ["detection_offset_invalid", "input_error"]);
  // Override problems are policy errors, as in Swift and Kotlin.
  for (const [overrides, reason] of [
    [{ NOT_A_LABEL: "keep" }, "override_label_unknown"],
    [{ PERSON: "obliterate" }, "override_action_unknown"],
  ] as const) {
    const override = await thrown(() =>
      decidePortablePolicy(loadPortablePolicy("hipaa_safe_harbor"), "x".repeat(14), [], {
        overrides,
      }),
    );
    assert.ok(override instanceof PortablePolicyError);
    assert.deepEqual(
      [override.code, override.errorCode, override.fieldPath],
      [reason, "policy_error", "$.overrides"],
    );
  }
  for (const [reason, expected] of [
    ["detection_offset_invalid", "input_error"],
    ["detection_label_invalid", "input_error"],
    ["override_label_unknown", "policy_error"],
    ["override_action_unknown", "policy_error"],
    ["surrogate_key_invalid", "configuration_error"],
    ["capability_missing", "capability_error"],
    ["profile_unknown", "policy_error"],
    ["resource_digest_mismatch", "policy_error"],
    ["taxonomy_invalid", "policy_error"],
  ] as const) {
    assert.equal(portablePolicyErrorCode(reason), expected, reason);
    assert.equal(new PortablePolicyError(reason).errorCode, expected, reason);
  }
  for (const [reason, expected] of [
    ["remote_model_refused", "model_load_error"],
    ["local_assets_unavailable", "model_load_error"],
    ["remote_asset_refused", "configuration_error"],
  ] as const) {
    const error = new OpenMedOfflineError(reason);
    assert.ok(error instanceof OpenMedError);
    assert.deepEqual([error.name, error.code, error.errorCode], ["OpenMedOfflineError", reason, expected]);
  }
});

test("former plain Error sites throw coded errors with unchanged messages", async () => {
  const cases: [() => unknown, OpenMedErrorCode, string, string?][] = [
    [() => hmacTextHash("Synthetic", ""), "configuration_error", "hashSecret must not be empty"],
    [
      () => alignTokenOffsets("Synthetic text", [{ entity: "B-NAME", word: "absent" }]),
      "inference_error",
      "Token offset alignment failed; provide source offsets.",
    ],
    [
      () => assertOfflineAssetPath("https://cdn.invalid/synthetic/ort/", "wasm asset path"),
      "configuration_error",
      "ONNX Runtime Web wasm asset path must be local/offline, not remote.",
      "remote_asset_refused",
    ],
    [
      () => assertOfflineAssetPath("   ", "wasm asset path"),
      "configuration_error",
      "ONNX Runtime Web wasm asset path must not be empty.",
    ],
    [
      () =>
        selectOrtWebBackend({
          webgpu: false,
          wasm: false,
          wasmSimd: false,
          sharedArrayBuffer: false,
          crossOriginIsolated: false,
          hardwareConcurrency: 1,
        }),
      "capability_error",
      "ONNX Runtime Web requires WebAssembly when WebGPU is absent.",
    ],
    [
      () => assertOfflineExternalData({ sessionOptions: { externalData: [] } as never }),
      "configuration_error",
      "ONNX Runtime Web external data must use the typed externalData option.",
    ],
    [
      () =>
        assertOfflineExternalData({
          externalData: {
            maxTotalBytes: 1,
            shards: [{ path: "model.onnx.data", location: "/models/model.onnx.data", byteLength: 2 }],
          },
        }),
      "budget_exceeded",
      "ONNX Runtime Web external data exceeds the total byte limit.",
    ],
    [
      () =>
        extractPii("Synthetic", {
          modelLoader: async () => () => [],
          loaderOptions: { pipelineOptions: { [Symbol("synthetic")]: true } } as never,
        }),
      "configuration_error",
      "OpenMed loader options do not support symbol keys",
    ],
  ];
  for (const [trigger, errorCode, message, code] of cases) {
    const error = await thrown(trigger);
    assert.ok(error instanceof OpenMedError, message);
    assert.equal(error.errorCode, errorCode, message);
    assert.equal(error.code, code ?? errorCode, message);
    assert.equal(error.message, message);
    // The name stays "Error", so String(error) reads as it did before.
    assert.equal(String(error), `Error: ${message}`);
  }
  // The refused URL never reaches the message.
  const remote = await thrown(() =>
    assertOfflineAssetPath("https://cdn.invalid/synthetic-secret-path/ort/"),
  );
  assert.equal(String(remote).includes("synthetic-secret-path"), false);
});

test("an unknown registry code degrades to internal_error and errors.ts stays browser-safe", async () => {
  const error = new OpenMedError("not_a_code" as OpenMedErrorCode, "synthetic");
  assert.equal(error.errorCode, "internal_error");
  assert.equal(openMedErrorCode(new Error("plain")), null);
  const cause = new Error("synthetic cause");
  assert.equal(new OpenMedError("capability_error", "x", { cause }).cause, cause);
  assert.equal("cause" in new OpenMedError("capability_error", "x"), false);

  const source = await readFile(ERRORS_SOURCE, "utf8");
  assert.doesNotMatch(source, /from\s+["']node:|require\(/);
});
