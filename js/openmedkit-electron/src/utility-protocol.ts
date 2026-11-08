/**
 * Messages exchanged between the main process and the OpenMed utility
 * process. Main and utility code only; renderers never see these shapes.
 *
 * Requests reach the utility process with a main-process job id instead of
 * the renderer's request id. Bundle-mode requests name a bundle generation,
 * never a path: the bundle root, trust policy and verification target come
 * only from the host through `load-bundle`.
 */
import { isAbsolute } from "node:path";

import { PORTABLE_BUNDLE_REASONS, type PortableBundleReason } from "openmed";

import {
  OPENMED_ELECTRON_SCHEMA_VERSION,
  assertElectronDeidentifyRequest,
  isBundleContentId,
  isRendererOpenMedSpan,
  isValidRequestId,
  type RendererOpenMedSpan,
} from "./ipc";

const MAX_PATH_LENGTH = 4_096;
const RUNTIME_ID_PATTERN = /^[a-z0-9][a-z0-9._-]{0,63}$/;

/** Bundle reasons beyond the portable verifier's own reason codes. */
export const ELECTRON_BUNDLE_EXTRA_REASONS = [
  "bundle_source_invalid",
  "content_id_unexpected",
  "runtime_unsupported",
  "pipeline_load_failed",
] as const;

export type ElectronBundleReason =
  | PortableBundleReason
  | (typeof ELECTRON_BUNDLE_EXTRA_REASONS)[number];

const BUNDLE_REASON_SET = new Set<string>([
  ...PORTABLE_BUNDLE_REASONS,
  ...ELECTRON_BUNDLE_EXTRA_REASONS,
]);

export function isElectronBundleReason(value: unknown): value is ElectronBundleReason {
  return typeof value === "string" && BUNDLE_REASON_SET.has(value);
}

/** Verification target, mirroring `PortableBundleTarget` as plain data. */
export interface UtilityBundleTarget {
  platform: string;
  sdkName: string;
  sdkVersion: string;
  runtimes: string[];
  osVersion: string | null;
  policyVersion: number | null;
  memoryBytes: number | null;
  storageBytes: number | null;
}

export interface UtilityBundleLimits {
  maxManifestBytes?: number;
  maxEnvelopeBytes?: number;
  maxTotalBytes?: number;
  maxFiles?: number;
}

/** Host-pinned bundle source. Structured-clone safe. */
export interface UtilityBundleSource {
  root: string;
  trustPolicy: unknown;
  target: UtilityBundleTarget | null;
  sequenceFloors: Record<string, number>;
  understoodExtensions: string[];
  expectedContentId: string | null;
  limits: UtilityBundleLimits;
}

export interface UtilityLoadBundleMessage {
  type: "load-bundle";
  generation: number;
  /** Verification instant, `YYYY-MM-DDTHH:MM:SSZ`. */
  now: string;
  source: UtilityBundleSource;
}

export interface UtilityDeidentifyRequest {
  type: "deidentify";
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  /** Main-process job id. */
  requestId: string;
  text: string;
  /** Legacy, unverified model directory. Exactly one of `modelPath`/`generation`. */
  modelPath?: string;
  /** Verified bundle generation this job must run against. */
  generation?: number;
}

export interface UtilityCancelMessage {
  type: "cancel";
  requestId: string;
}

export type UtilityInboundMessage =
  | UtilityLoadBundleMessage
  | UtilityDeidentifyRequest
  | UtilityCancelMessage;

/** Value-free summary of a portable-bundle verification. */
export interface UtilityBundleVerification {
  ok: boolean;
  reason: string | null;
  contentId: string | null;
  validSignatures: number;
  eligibleRuntimes: number[];
  filesChecked: number;
  filesVerified: boolean;
}

export interface UtilityBundleReadyMessage {
  type: "bundle-ready";
  generation: number;
  contentId: string;
  runtimeId: string;
  verification: UtilityBundleVerification;
}

export interface UtilityBundleRejectedMessage {
  type: "bundle-rejected";
  generation: number;
  reason: ElectronBundleReason;
  verification: UtilityBundleVerification | null;
}

export type UtilityErrorCode =
  | "INVALID_REQUEST"
  | "INFERENCE_FAILED"
  | "CANCELLED"
  | "BUNDLE_UNAVAILABLE";

export interface UtilityDeidentifySuccess {
  type: "deidentify-result";
  requestId: string;
  ok: true;
  spans: RendererOpenMedSpan[];
  generation: number | null;
}

export interface UtilityDeidentifyFailure {
  type: "deidentify-result";
  requestId: string;
  ok: false;
  errorCode: UtilityErrorCode;
  generation: number | null;
}

export type UtilityDeidentifyResponse =
  | UtilityDeidentifySuccess
  | UtilityDeidentifyFailure;

export type UtilityOutboundMessage =
  | UtilityBundleReadyMessage
  | UtilityBundleRejectedMessage
  | UtilityDeidentifyResponse;

const UTILITY_ERROR_CODES = new Set<string>([
  "INVALID_REQUEST",
  "INFERENCE_FAILED",
  "CANCELLED",
  "BUNDLE_UNAVAILABLE",
]);

export function isSafeAbsolutePath(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_PATH_LENGTH &&
    !value.includes("\0") &&
    isAbsolute(value)
  );
}

export function isGeneration(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 1;
}

export function isUtilityDeidentifyMessage(
  message: unknown,
): message is UtilityDeidentifyRequest {
  if (!isRecord(message) || message.type !== "deidentify") {
    return false;
  }
  const hasPath = message.modelPath !== undefined;
  const hasGeneration = message.generation !== undefined;
  if (hasPath === hasGeneration) {
    return false;
  }
  if (hasPath ? !isSafeAbsolutePath(message.modelPath) : !isGeneration(message.generation)) {
    return false;
  }
  try {
    assertElectronDeidentifyRequest({
      schemaVersion: message.schemaVersion,
      requestId: message.requestId,
      text: message.text,
    });
  } catch {
    return false;
  }
  return true;
}

export function isUtilityCancelMessage(message: unknown): message is UtilityCancelMessage {
  return (
    isRecord(message) && message.type === "cancel" && isValidRequestId(message.requestId)
  );
}

export function isUtilityLoadBundleMessage(
  message: unknown,
): message is UtilityLoadBundleMessage {
  if (
    !isRecord(message) ||
    message.type !== "load-bundle" ||
    !isGeneration(message.generation) ||
    typeof message.now !== "string" ||
    !isRecord(message.source)
  ) {
    return false;
  }
  const source = message.source;
  return (
    isSafeAbsolutePath(source.root) &&
    (source.target === null || isRecord(source.target)) &&
    isRecord(source.sequenceFloors) &&
    Array.isArray(source.understoodExtensions) &&
    (source.expectedContentId === null || isBundleContentId(source.expectedContentId)) &&
    isRecord(source.limits)
  );
}

function isVerification(value: unknown): value is UtilityBundleVerification {
  return (
    isRecord(value) &&
    typeof value.ok === "boolean" &&
    (value.reason === null || isElectronBundleReason(value.reason)) &&
    (value.contentId === null || isBundleContentId(value.contentId)) &&
    Number.isSafeInteger(value.validSignatures) &&
    Array.isArray(value.eligibleRuntimes) &&
    value.eligibleRuntimes.every((index) => Number.isSafeInteger(index)) &&
    Number.isSafeInteger(value.filesChecked) &&
    typeof value.filesVerified === "boolean"
  );
}

export function isUtilityBundleReadyMessage(
  message: unknown,
): message is UtilityBundleReadyMessage {
  return (
    isRecord(message) &&
    message.type === "bundle-ready" &&
    isGeneration(message.generation) &&
    isBundleContentId(message.contentId) &&
    typeof message.runtimeId === "string" &&
    RUNTIME_ID_PATTERN.test(message.runtimeId) &&
    isVerification(message.verification) &&
    message.verification.ok === true &&
    message.verification.filesVerified === true &&
    message.verification.contentId === message.contentId
  );
}

export function isUtilityBundleRejectedMessage(
  message: unknown,
): message is UtilityBundleRejectedMessage {
  return (
    isRecord(message) &&
    message.type === "bundle-rejected" &&
    isGeneration(message.generation) &&
    isElectronBundleReason(message.reason) &&
    (message.verification === null || isVerification(message.verification))
  );
}

export function isUtilityDeidentifyResponse(
  response: unknown,
): response is UtilityDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.type !== "deidentify-result" ||
    !isValidRequestId(response.requestId) ||
    typeof response.ok !== "boolean" ||
    !(response.generation === null || isGeneration(response.generation))
  ) {
    return false;
  }
  return response.ok
    ? Array.isArray(response.spans) && response.spans.every(isRendererOpenMedSpan)
    : typeof response.errorCode === "string" && UTILITY_ERROR_CODES.has(response.errorCode);
}

export function utilityFailure(
  requestId: string,
  errorCode: UtilityErrorCode,
  generation: number | null = null,
): UtilityDeidentifyFailure {
  return {
    type: "deidentify-result",
    requestId,
    ok: false,
    errorCode,
    generation,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
