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

/**
 * Raised when a load would leave the device or local assets cannot be loaded
 * in offline-only mode. The message and `code` never include input text,
 * paths, URLs or the underlying runtime error.
 */
export class OpenMedOfflineError extends Error {
  readonly code: OpenMedOfflineErrorCode;

  constructor(code: OpenMedOfflineErrorCode) {
    super(OFFLINE_ERROR_MESSAGES[code]);
    this.name = "OpenMedOfflineError";
    this.code = code;
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
