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

const extensionApi =
  (globalThis as typeof globalThis & { browser?: typeof chrome }).browser ??
  chrome;
const workerHashSecret = crypto.randomUUID();

extensionApi.runtime.onMessage.addListener(
  (
    message: ExtensionRequest,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response: unknown) => void,
  ) => {
    void handleMessage(message, sender)
      .then(sendResponse)
      .catch((error: unknown) => {
        sendResponse({
          ok: false,
          error: error instanceof Error ? error.message : "Detection failed",
        });
      });
    return true;
  },
);

async function handleMessage(
  message: ExtensionRequest,
  sender: chrome.runtime.MessageSender,
): Promise<unknown> {
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
  const senderUrl = sender.tab?.url ?? sender.url;
  if (senderUrl === undefined) {
    throw new Error("Cannot determine the requesting site");
  }
  return new URL(senderUrl).origin;
}

function settingsKey(origin: string): string {
  return `openmed:site:${origin}`;
}

async function readSettings(origin: string): Promise<SiteSettings> {
  const key = settingsKey(origin);
  const stored = await extensionApi.storage.local.get(key);
  const candidate = stored[key] as Partial<SiteSettings> | undefined;
  return {
    enabled: candidate?.enabled ?? true,
    policy: isPolicyName(candidate?.policy) ? candidate.policy : DEFAULT_POLICY,
  };
}

async function writeSettings(
  origin: string,
  settings: SiteSettings,
): Promise<void> {
  await extensionApi.storage.local.set({ [settingsKey(origin)]: settings });
}
