/**
 * Renderer hardening for OpenMed Electron windows.
 *
 * This module has no Node or Electron imports. It is shared by the main
 * process (window and session setup) and by the IPC registration, which
 * refuses senders whose resolved web preferences are relaxed. All errors and
 * callbacks carry fixed codes only; URLs, paths and page content are never
 * reported.
 */

/**
 * Content Security Policy for OpenMed renderer documents.
 *
 * Scripts, styles, images and fonts may load only from the app's own origin.
 * Inline scripts, `eval`, network connections, frames, workers, plugins, form
 * submission and base-URL changes are all refused.
 */
export const OPENMED_RENDERER_CSP = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self'",
  "font-src 'self'",
  "connect-src 'none'",
  "media-src 'none'",
  "object-src 'none'",
  "frame-src 'none'",
  "child-src 'none'",
  "worker-src 'none'",
  "manifest-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join("; ");

/** Web preferences every window that can reach OpenMed IPC must use. */
export const OPENMED_HARDENED_WEB_PREFERENCES = Object.freeze({
  contextIsolation: true,
  sandbox: true,
  nodeIntegration: false,
  nodeIntegrationInWorker: false,
  nodeIntegrationInSubFrames: false,
  webSecurity: true,
  allowRunningInsecureContent: false,
  webviewTag: false,
  experimentalFeatures: false,
  navigateOnDragDrop: false,
  safeDialogs: true,
} as const);

/** Value-free names for each relaxed web preference. */
export const WEB_PREFERENCE_RELAXATIONS = [
  "web_preferences_invalid",
  "context_isolation_disabled",
  "sandbox_disabled",
  "node_integration_enabled",
  "node_integration_in_worker_enabled",
  "node_integration_in_subframes_enabled",
  "web_security_disabled",
  "insecure_content_allowed",
  "webview_tag_enabled",
  "experimental_features_enabled",
  "blink_features_enabled",
] as const;

export type WebPreferenceRelaxation = (typeof WEB_PREFERENCE_RELAXATIONS)[number];

/** Value-free events reported when a hardened window blocks an action. */
export type ElectronHardeningEvent =
  | "navigation_blocked"
  | "window_open_blocked"
  | "webview_blocked"
  | "permission_denied"
  | "request_blocked";

/** Raised when a window or sender does not use hardened web preferences. */
export class ElectronHardeningError extends Error {
  readonly relaxations: readonly WebPreferenceRelaxation[];

  constructor(relaxations: readonly WebPreferenceRelaxation[]) {
    super(`OpenMed Electron window is not hardened: ${relaxations.join(",")}.`);
    this.name = "ElectronHardeningError";
    this.relaxations = Object.freeze([...relaxations]);
  }
}

/** Options for {@link createHardenedWebPreferences}. */
export interface HardenedWebPreferencesOptions {
  /** Absolute path of the bundled, sandbox-compatible preload script. */
  preload: string;
  /** Optional dedicated session partition for the window. */
  partition?: string;
}

/**
 * Return OpenMed's hardened web preferences with the app's preload script.
 *
 * Args:
 *   options: The preload path and optional session partition.
 *
 * Returns:
 *   A fresh web-preferences object suitable for `new BrowserWindow`.
 */
export function createHardenedWebPreferences(
  options: HardenedWebPreferencesOptions,
): typeof OPENMED_HARDENED_WEB_PREFERENCES & {
  preload: string;
  partition?: string;
} {
  if (typeof options?.preload !== "string" || options.preload.length === 0) {
    throw new TypeError("A bundled OpenMed preload script is required.");
  }
  return {
    ...OPENMED_HARDENED_WEB_PREFERENCES,
    preload: options.preload,
    ...(options.partition === undefined ? {} : { partition: options.partition }),
  };
}

/**
 * List every way a web-preferences object relaxes OpenMed's hardening.
 *
 * `contextIsolation`, `sandbox`, `nodeIntegration` and `webSecurity` must be
 * set explicitly, so a change in Electron defaults cannot silently relax a
 * window. The remaining switches must be absent or set to the safe value.
 *
 * Args:
 *   preferences: A `webPreferences` object or the value returned by
 *     `webContents.getLastWebPreferences()`.
 *
 * Returns:
 *   The relaxations found, in a fixed order; empty when hardened.
 */
export function findWebPreferenceRelaxations(
  preferences: unknown,
): WebPreferenceRelaxation[] {
  if (typeof preferences !== "object" || preferences === null) {
    return ["web_preferences_invalid"];
  }
  const value = preferences as Record<string, unknown>;
  const relaxations: WebPreferenceRelaxation[] = [];
  const optionalFalse = (key: string) =>
    value[key] === undefined || value[key] === false;
  if (value.contextIsolation !== true) {
    relaxations.push("context_isolation_disabled");
  }
  if (value.sandbox !== true) {
    relaxations.push("sandbox_disabled");
  }
  if (value.nodeIntegration !== false) {
    relaxations.push("node_integration_enabled");
  }
  if (!optionalFalse("nodeIntegrationInWorker")) {
    relaxations.push("node_integration_in_worker_enabled");
  }
  if (!optionalFalse("nodeIntegrationInSubFrames")) {
    relaxations.push("node_integration_in_subframes_enabled");
  }
  if (value.webSecurity !== true) {
    relaxations.push("web_security_disabled");
  }
  if (!optionalFalse("allowRunningInsecureContent")) {
    relaxations.push("insecure_content_allowed");
  }
  if (!optionalFalse("webviewTag")) {
    relaxations.push("webview_tag_enabled");
  }
  if (!optionalFalse("experimentalFeatures")) {
    relaxations.push("experimental_features_enabled");
  }
  if (value.enableBlinkFeatures !== undefined && value.enableBlinkFeatures !== "") {
    relaxations.push("blink_features_enabled");
  }
  return relaxations;
}

/**
 * Throw unless a web-preferences object is hardened.
 *
 * Args:
 *   preferences: A `webPreferences` object or resolved preferences.
 *
 * Raises:
 *   ElectronHardeningError: With the value-free list of relaxations.
 */
export function assertHardenedWebPreferences(preferences: unknown): void {
  const relaxations = findWebPreferenceRelaxations(preferences);
  if (relaxations.length > 0) {
    throw new ElectronHardeningError(relaxations);
  }
}

/**
 * Return whether `url` belongs to one of the allowed app origins.
 *
 * Origins are compared as `scheme://host[:port]`, which also works for custom
 * standard schemes such as `app://openmed`. Malformed URLs are refused.
 *
 * Args:
 *   url: The navigation or request URL.
 *   allowedOrigins: Exact origins the app serves its renderer from.
 *
 * Returns:
 *   True only for an exact origin match.
 */
export function isAllowedOpenMedOrigin(
  url: unknown,
  allowedOrigins: readonly string[],
): boolean {
  if (typeof url !== "string" || url.length === 0 || url.length > 8_192) {
    return false;
  }
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.username !== "" || parsed.password !== "") {
    return false;
  }
  const origin = `${parsed.protocol}//${parsed.host}`;
  return allowedOrigins.includes(origin);
}

/** The subset of Electron's `WebContents` used for hardening. */
export interface HardenableWebContentsLike {
  on(event: string, listener: (...args: never[]) => void): unknown;
  setWindowOpenHandler(
    handler: (details: { url: string }) => { action: "deny" },
  ): void;
  getLastWebPreferences?(): unknown;
}

/** Options for {@link hardenOpenMedWebContents}. */
export interface HardenWebContentsOptions {
  /** Exact origins (`scheme://host`) the renderer may stay on. */
  allowedOrigins: readonly string[];
  /** Observer for value-free block events. */
  onBlocked?: (event: ElectronHardeningEvent) => void;
}

/**
 * Lock a window's `webContents` to the app origins.
 *
 * Verifies the resolved web preferences, denies every `window.open`, refuses
 * `<webview>` attachment, and cancels any navigation, redirect or frame
 * navigation that leaves the allowed origins.
 *
 * Args:
 *   webContents: The window's `webContents`.
 *   options: Allowed origins and an optional block observer.
 *
 * Raises:
 *   ElectronHardeningError: When the window's preferences are relaxed.
 */
export function hardenOpenMedWebContents(
  webContents: HardenableWebContentsLike,
  options: HardenWebContentsOptions,
): void {
  const allowedOrigins = validateOrigins(options?.allowedOrigins);
  if (typeof webContents?.getLastWebPreferences !== "function") {
    throw new ElectronHardeningError(["web_preferences_invalid"]);
  }
  assertHardenedWebPreferences(webContents.getLastWebPreferences());
  const report = (event: ElectronHardeningEvent) => {
    try {
      options.onBlocked?.(event);
    } catch {
      // Observers are diagnostic only and must not alter enforcement.
    }
  };
  webContents.setWindowOpenHandler(() => {
    report("window_open_blocked");
    return { action: "deny" };
  });
  const guard = (event: unknown, url: unknown) => {
    const target =
      typeof url === "string"
        ? url
        : (event as { url?: unknown } | null | undefined)?.url;
    if (!isAllowedOpenMedOrigin(target, allowedOrigins)) {
      (event as { preventDefault(): void }).preventDefault();
      report("navigation_blocked");
    }
  };
  for (const name of ["will-navigate", "will-redirect", "will-frame-navigate"]) {
    webContents.on(name, guard as (...args: never[]) => void);
  }
  webContents.on("will-attach-webview", ((event: { preventDefault(): void }) => {
    event.preventDefault();
    report("webview_blocked");
  }) as (...args: never[]) => void);
}

/** The subset of Electron's `Session` used for hardening. */
export interface HardenableSessionLike {
  setPermissionRequestHandler(
    handler:
      | ((
          webContents: unknown,
          permission: string,
          callback: (granted: boolean) => void,
        ) => void)
      | null,
  ): void;
  setPermissionCheckHandler(handler: (() => boolean) | null): void;
  webRequest: {
    onBeforeRequest(
      listener: (
        details: { url: string },
        callback: (response: { cancel: boolean }) => void,
      ) => void,
    ): void;
    onHeadersReceived(
      listener: (
        details: { url: string; responseHeaders?: Record<string, string[]> },
        callback: (response: {
          cancel?: boolean;
          responseHeaders?: Record<string, string[]>;
        }) => void,
      ) => void,
    ): void;
  };
}

/** Options for {@link hardenOpenMedSession}. */
export interface HardenSessionOptions {
  /** Exact origins (`scheme://host`) the renderer is served from. */
  allowedOrigins: readonly string[];
  /** Policy to attach to app responses; defaults to {@link OPENMED_RENDERER_CSP}. */
  contentSecurityPolicy?: string;
  /** Observer for value-free block events. */
  onBlocked?: (event: ElectronHardeningEvent) => void;
}

/**
 * Deny permissions and off-origin requests for a renderer session.
 *
 * Every permission request and check is refused, every request outside the
 * allowed origins is cancelled, and app responses carry the OpenMed CSP in
 * place of any policy they declared.
 *
 * Args:
 *   session: A dedicated renderer session.
 *   options: Allowed origins, an optional policy and a block observer.
 */
export function hardenOpenMedSession(
  session: HardenableSessionLike,
  options: HardenSessionOptions,
): void {
  const allowedOrigins = validateOrigins(options?.allowedOrigins);
  const policy = options.contentSecurityPolicy ?? OPENMED_RENDERER_CSP;
  if (typeof policy !== "string" || policy.length === 0) {
    throw new TypeError("Invalid OpenMed renderer content security policy.");
  }
  const report = (event: ElectronHardeningEvent) => {
    try {
      options.onBlocked?.(event);
    } catch {
      // Observers are diagnostic only and must not alter enforcement.
    }
  };
  session.setPermissionRequestHandler((_webContents, _permission, callback) => {
    report("permission_denied");
    callback(false);
  });
  session.setPermissionCheckHandler(() => false);
  session.webRequest.onBeforeRequest((details, callback) => {
    const allowed = isAllowedOpenMedOrigin(details?.url, allowedOrigins);
    if (!allowed) {
      report("request_blocked");
    }
    callback({ cancel: !allowed });
  });
  session.webRequest.onHeadersReceived((details, callback) => {
    if (!isAllowedOpenMedOrigin(details?.url, allowedOrigins)) {
      callback({ cancel: true });
      return;
    }
    const headers: Record<string, string[]> = {};
    for (const [name, value] of Object.entries(details.responseHeaders ?? {})) {
      if (name.toLowerCase() !== "content-security-policy") {
        headers[name] = value;
      }
    }
    headers["Content-Security-Policy"] = [policy];
    callback({ responseHeaders: headers });
  });
}

const NON_APP_SCHEMES = new Set([
  "about",
  "blob",
  "chrome",
  "data",
  "devtools",
  "file",
  "ftp",
  "http",
  "https",
  "javascript",
  "ws",
  "wss",
]);

function validateOrigins(origins: unknown): readonly string[] {
  if (
    !Array.isArray(origins) ||
    origins.length === 0 ||
    origins.length > 8 ||
    !origins.every((origin) => {
      if (typeof origin !== "string") {
        return false;
      }
      const match = /^([a-z][a-z0-9+.-]{0,31}):\/\/[a-z0-9.-]{1,253}$/.exec(origin);
      return match !== null && !NON_APP_SCHEMES.has(match[1] as string);
    })
  ) {
    throw new TypeError(
      "OpenMed allowed origins must be exact local app-scheme origins.",
    );
  }
  return [...origins];
}
