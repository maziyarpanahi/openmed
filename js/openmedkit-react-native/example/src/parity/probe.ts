// Typed access to the example app's direct-native probe module.
//
// `OpenMedKitParityProbe` is a separate native module compiled only into the
// example app. It calls OpenMedKit (Swift or Kotlin) directly, without the
// `OpenMedKitRN` bridge, and returns its results as a JSON string so the
// direct path never goes through React Native's value conversion.

import type { OpenMedBackendKind } from "../../../src/index";
import type { ParityOperation } from "./corpus";

export const PARITY_PROBE_MODULE_NAME = "OpenMedKitParityProbe" as const;

export interface ProbeEnvironment {
  platform: "ios" | "android";
  runtime_backend: OpenMedBackendKind;
  device_class: "simulator" | "emulator" | "device";
  os_version: string;
  build_configuration: "debug" | "release";
}

/** A pinned local model bundle whose digests the probe verified natively. */
export interface ProbeModelFixture {
  modelPath: string;
  backend: OpenMedBackendKind;
  id2LabelPath?: string;
  tokenizerFolderPath?: string;
  tokenizerName?: string;
  /** `sha256:` digest over the verified manifest file list. */
  bundleDigest: string;
  fileCount: number;
}

export interface DirectProbe {
  describeEnvironment(): Promise<ProbeEnvironment>;
  prepareModelFixture(): Promise<ProbeModelFixture>;
  /** 32 random bytes, hex encoded, generated natively and never persisted. */
  createEphemeralHashSecret(): Promise<string>;
  runDirect(requestJson: string): Promise<string>;
  writeReport(reportJson: string): Promise<boolean>;
}

export interface DirectModel {
  cacheKey: string;
  modelPath: string;
  backend: OpenMedBackendKind;
  id2LabelPath?: string;
  tokenizerFolderPath?: string;
  tokenizerName?: string;
}

export interface DirectOptions {
  docId: string;
  hashSecret: string;
  metadata: Record<string, string>;
  confidenceThreshold: number;
  useSmartMerging: boolean;
  policy?: string;
}

export interface DirectRequest {
  operation: ParityOperation;
  text: string;
  options: DirectOptions;
  model: DirectModel;
}

export interface DirectResult {
  spans: unknown[];
  text?: string;
  deidentifiedText?: string;
}

export type DirectResponse =
  | { ok: true; result: DirectResult }
  | { ok: false; error: { code: string } };

const PROBE_METHODS = [
  "describeEnvironment",
  "prepareModelFixture",
  "createEphemeralHashSecret",
  "runDirect",
  "writeReport",
] as const;

/** Resolve the probe from `NativeModules`; there is no injection hook. */
export function resolveNativeProbe(
  nativeModules: Record<string, unknown> | undefined,
): DirectProbe | null {
  const candidate = nativeModules?.[PARITY_PROBE_MODULE_NAME];
  if (candidate === null || typeof candidate !== "object") {
    return null;
  }
  const record = candidate as Record<string, unknown>;
  return PROBE_METHODS.every((method) => typeof record[method] === "function")
    ? (candidate as DirectProbe)
    : null;
}

/** Validate the environment record the probe reports. */
export function parseProbeEnvironment(value: unknown): ProbeEnvironment {
  const record = asRecord(value);
  const platform = record.platform;
  const backend = record.runtime_backend;
  const deviceClass = record.device_class;
  const configuration = record.build_configuration;
  if (
    (platform !== "ios" && platform !== "android") ||
    (backend !== "coreml" && backend !== "mlx" && backend !== "onnx") ||
    (deviceClass !== "simulator" && deviceClass !== "emulator" && deviceClass !== "device") ||
    (configuration !== "debug" && configuration !== "release") ||
    typeof record.os_version !== "string" ||
    !/^[0-9A-Za-z .()_-]{1,64}$/.test(record.os_version)
  ) {
    throw new Error("parity_probe_environment_invalid");
  }
  return {
    platform,
    runtime_backend: backend,
    device_class: deviceClass,
    os_version: record.os_version,
    build_configuration: configuration,
  };
}

/** Validate the verified model fixture record the probe reports. */
export function parseModelFixture(value: unknown): ProbeModelFixture {
  const record = asRecord(value);
  const backend = record.backend;
  if (
    typeof record.modelPath !== "string" ||
    record.modelPath.length === 0 ||
    (backend !== "coreml" && backend !== "mlx" && backend !== "onnx") ||
    typeof record.bundleDigest !== "string" ||
    !/^sha256:[0-9a-f]{64}$/.test(record.bundleDigest) ||
    typeof record.fileCount !== "number" ||
    !Number.isInteger(record.fileCount) ||
    record.fileCount < 1
  ) {
    throw new Error("parity_model_fixture_invalid");
  }
  const fixture: ProbeModelFixture = {
    modelPath: record.modelPath,
    backend,
    bundleDigest: record.bundleDigest,
    fileCount: record.fileCount,
  };
  for (const key of ["id2LabelPath", "tokenizerFolderPath", "tokenizerName"] as const) {
    const field = record[key];
    if (typeof field === "string" && field.length > 0) {
      fixture[key] = field;
    }
  }
  return fixture;
}

/** Parse a probe response; malformed output becomes a value-free error. */
export function parseDirectResponse(json: string): DirectResponse {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return { ok: false, error: { code: "direct_response_invalid" } };
  }
  const record = asRecord(parsed);
  if (record.ok === false) {
    const code = asRecord(record.error).code;
    return {
      ok: false,
      error: { code: typeof code === "string" ? safeCode(code) : "direct_failed" },
    };
  }
  const result = asRecord(record.result);
  if (record.ok !== true || !Array.isArray(result.spans)) {
    return { ok: false, error: { code: "direct_response_invalid" } };
  }
  const direct: DirectResult = { spans: result.spans };
  if (typeof result.text === "string") {
    direct.text = result.text;
  }
  if (typeof result.deidentifiedText === "string") {
    direct.deidentifiedText = result.deidentifiedText;
  }
  return { ok: true, result: direct };
}

/** Keep only short snake_case codes; anything else collapses to a constant. */
export function safeCode(value: unknown, fallback = "unknown_error"): string {
  return typeof value === "string" && /^[a-z][a-z0-9_]{0,63}$/.test(value)
    ? value
    : fallback;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
