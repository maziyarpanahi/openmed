import {
  deidentify,
  spansToRedactedText,
  type OpenMedSpan,
} from "openmed";

import { bundledPhiPipeline } from "./detector";
import {
  DEFAULT_POLICY,
  applyPolicy,
  isPolicyName,
  type PolicyName,
} from "./policy";

interface SiteSettings {
  enabled: boolean;
  policy: PolicyName;
}

type ExtensionRequest =
  | { type: "openmed:get-settings" }
  | { type: "openmed:set-enabled"; enabled: boolean }
  | { type: "openmed:set-policy"; policy: string }
  | { type: "openmed:scan"; text: string };

interface ScanResponse {
  ok: true;
  enabled: boolean;
  policy: PolicyName;
  text: string;
  deidentifiedText: string;
  spans: OpenMedSpan[];
}

const MAX_SCAN_TEXT_CHARS = 1_000_000;
const MAX_SCAN_TEXT_BYTES = 4_000_000;

const extensionApi =
  (globalThis as typeof globalThis & { browser?: typeof chrome }).browser ??
  chrome;
const workerHashSecret = crypto.randomUUID();

extensionApi.runtime.onMessage.addListener(
  (
    message: unknown,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response: unknown) => void,
  ) => {
    void handleMessage(message, sender)
      .then(sendResponse)
      .catch(() => {
        sendResponse({
          ok: false,
          error: "Detection failed safely.",
        });
      });
    return true;
  },
);

async function handleMessage(
  rawMessage: unknown,
  sender: chrome.runtime.MessageSender,
): Promise<unknown> {
  if (sender.id !== extensionApi.runtime.id) {
    throw new Error("Unexpected extension message sender");
  }
  const message = parseRequest(rawMessage);
  const origin = senderOrigin(sender);
  const settings = await readSettings(origin);

  if (message.type === "openmed:get-settings") {
    return { ok: true, ...settings };
  }

  if (message.type === "openmed:set-enabled") {
    const updated = { ...settings, enabled: message.enabled };
    await writeSettings(origin, updated);
    return { ok: true, ...updated };
  }

  if (message.type === "openmed:set-policy") {
    if (!isPolicyName(message.policy)) {
      throw new Error(`Unknown policy profile: ${message.policy}`);
    }
    const updated = { ...settings, policy: message.policy };
    await writeSettings(origin, updated);
    return { ok: true, ...updated };
  }

  if (message.type === "openmed:scan") {
    if (!settings.enabled) {
      return emptyScan(message.text, settings);
    }
    const result = await deidentify(message.text, {
      pipeline: bundledPhiPipeline,
      detector: "browser-extension-local",
      hashSecret: workerHashSecret,
      metadata: { runtime: "extension-background" },
    });
    const spans = applyPolicy(result.spans, settings.policy);
    return {
      ok: true,
      enabled: true,
      policy: settings.policy,
      text: message.text,
      deidentifiedText: spansToRedactedText(message.text, spans),
      spans,
    } satisfies ScanResponse;
  }

  throw new Error("Unsupported extension request");
}

function emptyScan(text: string, settings: SiteSettings): ScanResponse {
  return {
    ok: true,
    enabled: false,
    policy: settings.policy,
    text,
    deidentifiedText: text,
    spans: [],
  };
}

function senderOrigin(sender: chrome.runtime.MessageSender): string {
  const senderUrl = sender.url ?? sender.tab?.url;
  if (senderUrl === undefined) {
    throw new Error("Cannot determine the requesting site");
  }
  const parsed = new URL(senderUrl);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Unsupported requesting site");
  }
  return parsed.origin;
}

function settingsKey(origin: string): string {
  return `openmed:site:${origin}`;
}

async function readSettings(origin: string): Promise<SiteSettings> {
  const key = settingsKey(origin);
  const stored = await extensionApi.storage.local.get(key);
  const candidate = stored[key] as Partial<SiteSettings> | undefined;
  return {
    enabled:
      typeof candidate?.enabled === "boolean" ? candidate.enabled : true,
    policy: isPolicyName(candidate?.policy) ? candidate.policy : DEFAULT_POLICY,
  };
}

async function writeSettings(
  origin: string,
  settings: SiteSettings,
): Promise<void> {
  await extensionApi.storage.local.set({ [settingsKey(origin)]: settings });
}

function parseRequest(value: unknown): ExtensionRequest {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Invalid extension request");
  }
  const candidate = value as Record<string, unknown>;
  if (candidate.type === "openmed:get-settings") {
    requireKeys(candidate, ["type"]);
    return { type: candidate.type };
  }
  if (candidate.type === "openmed:set-enabled") {
    requireKeys(candidate, ["enabled", "type"]);
    if (typeof candidate.enabled !== "boolean") {
      throw new Error("Invalid enabled setting");
    }
    return { type: candidate.type, enabled: candidate.enabled };
  }
  if (candidate.type === "openmed:set-policy") {
    requireKeys(candidate, ["policy", "type"]);
    if (typeof candidate.policy !== "string") {
      throw new Error("Invalid policy setting");
    }
    return { type: candidate.type, policy: candidate.policy };
  }
  if (candidate.type === "openmed:scan") {
    requireKeys(candidate, ["text", "type"]);
    if (
      typeof candidate.text !== "string" ||
      candidate.text.length === 0 ||
      candidate.text.length > MAX_SCAN_TEXT_CHARS ||
      new TextEncoder().encode(candidate.text).byteLength > MAX_SCAN_TEXT_BYTES
    ) {
      throw new Error("Invalid scan text");
    }
    return { type: candidate.type, text: candidate.text };
  }
  throw new Error("Unsupported extension request");
}

function requireKeys(
  candidate: Record<string, unknown>,
  expected: string[],
): void {
  const actual = Object.keys(candidate).sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error("Extension request contains unsupported fields");
  }
}
