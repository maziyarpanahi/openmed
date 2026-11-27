import { OpenMedError } from "./errors";
import type { OpenMedErrorCode } from "./error-codes";

/** Value-free reason codes carried by {@link OpenMedOfflineError}. */
export type OpenMedOfflineErrorCode =
  | "remote_model_refused"
  | "remote_asset_refused"
  | "local_assets_unavailable";

const OFFLINE_ERROR_MESSAGES: Record<OpenMedOfflineErrorCode, string> = {
  remote_model_refused: "OpenMed refused a model reference that is not local.",
  remote_asset_refused:
    "OpenMed refused a runtime asset location that is not local.",
  local_assets_unavailable:
    "OpenMed could not load the local model or runtime assets.",
};

// A Hub id refused in offline-only mode cannot load without leaving the
// device, which Python reports as a model load failure; a remote runtime
// asset location is configuration that offline-only mode does not accept.
const OFFLINE_REGISTRY_CODES: Record<OpenMedOfflineErrorCode, OpenMedErrorCode> = {
  remote_model_refused: "model_load_error",
  remote_asset_refused: "configuration_error",
  local_assets_unavailable: "model_load_error",
};

/**
 * Raised when a load would leave the device or local assets cannot be loaded
 * in offline-only mode. The message and `code` never include input text,
 * paths, URLs or the underlying runtime error. `errorCode` is the portable
 * registry code: `model_load_error` for `remote_model_refused` and
 * `local_assets_unavailable`, `configuration_error` for
 * `remote_asset_refused`.
 */
export class OpenMedOfflineError extends OpenMedError {
  declare readonly code: OpenMedOfflineErrorCode;

  constructor(code: OpenMedOfflineErrorCode) {
    super(OFFLINE_REGISTRY_CODES[code], OFFLINE_ERROR_MESSAGES[code], { code });
    this.name = "OpenMedOfflineError";
  }
}

let packageOfflineOnly = false;

/**
 * Turn package-wide offline-only loading on or off. While it is on, every
 * loader refuses Hub IDs and remote asset locations before any fetch, and a
 * call option cannot opt back out.
 */
export function setOfflineOnly(enabled: boolean): void {
  packageOfflineOnly = enabled === true;
}

/** Report whether package-wide offline-only loading is on. */
export function isOfflineOnly(): boolean {
  return packageOfflineOnly;
}

export function resolveOfflineOnly(option: boolean | undefined): boolean {
  return packageOfflineOnly || option === true;
}

/** Run a load and, in offline-only mode, replace any failure with a typed one. */
export async function failClosedWhenOffline<T>(
  offlineOnly: boolean,
  load: () => T | Promise<T>,
): Promise<T> {
  if (!offlineOnly) {
    return load();
  }
  try {
    return await load();
  } catch (error) {
    if (error instanceof OpenMedOfflineError) {
      throw error;
    }
    throw new OpenMedOfflineError("local_assets_unavailable");
  }
}
