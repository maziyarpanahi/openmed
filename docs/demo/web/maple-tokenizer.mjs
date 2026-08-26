/*
 * Same-origin Maple tokenizer adapter around pinned Transformers.js.
 * The generated vendor module is local-only; remote Hub access is disabled.
 */

import { AutoTokenizer, env } from "./vendor/transformers.web.min.js";

const NETWORK_POLICY = "same-origin-model-assets-only";

export async function createOpenMedMapleTokenizer(options = {}) {
  const { chatTemplateUrl, modelUrl, networkPolicy, signal, tokenizerUrl } = options;
  if (networkPolicy !== NETWORK_POLICY) {
    throw new Error(`networkPolicy must be ${NETWORK_POLICY}`);
  }
  throwIfAborted(signal);

  const page = new URL(globalThis.location?.href ?? "about:blank");
  const model = requireSameOriginDirectory(modelUrl, page, "modelUrl");
  const tokenizer = requireSameOriginFile(tokenizerUrl, page, "tokenizerUrl");
  const chatTemplate = requireSameOriginFile(
    chatTemplateUrl,
    page,
    "chatTemplateUrl",
  );
  if (!tokenizer.pathname.startsWith(model.pathname)) {
    throw new Error("tokenizerUrl must remain inside modelUrl");
  }
  if (!chatTemplate.pathname.startsWith(model.pathname)) {
    throw new Error("chatTemplateUrl must remain inside modelUrl");
  }

  const chatTemplateText = await loadChatTemplate(chatTemplate, signal);

  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = "/";
  env.useBrowserCache = false;
  env.useFSCache = false;
  env.useCustomCache = false;
  if ("logLevel" in env) env.logLevel = 5;

  const implementation = await AutoTokenizer.from_pretrained(model.pathname, {
    local_files_only: true,
  });
  throwIfAborted(signal);

  return {
    async encodeMessages(
      messages,
      { addGenerationPrompt, reasoning = true, signal } = {},
    ) {
      throwIfAborted(signal);
      if (!Array.isArray(messages) || messages.length === 0) {
        throw new TypeError("messages must be a non-empty array");
      }
      const rendered = implementation.apply_chat_template(messages, {
        add_generation_prompt: Boolean(addGenerationPrompt),
        chat_template: chatTemplateText,
        return_dict: false,
        return_tensor: false,
        tokenize: false,
      });
      const prompt = reasoning
        ? rendered
        : closeTrailingReasoningPrompt(rendered, addGenerationPrompt);
      const encoded = implementation.encode(prompt, {
        add_special_tokens: false,
      });
      throwIfAborted(signal);
      return normalizeTokenIds(encoded);
    },
    async decode(tokenIds, { skipSpecialTokens = true } = {}) {
      return implementation.decode(normalizeTokenIds(tokenIds), {
        clean_up_tokenization_spaces: false,
        skip_special_tokens: Boolean(skipSpecialTokens),
      });
    },
    dispose() {},
  };
}

function closeTrailingReasoningPrompt(rendered, addGenerationPrompt) {
  if (typeof rendered !== "string") {
    throw new TypeError("Maple tokenizer returned an unsupported rendered prompt");
  }
  if (!addGenerationPrompt) return rendered;
  const suffix = "<think>\n";
  if (!rendered.endsWith(suffix)) {
    throw new Error("Maple chat template is missing its reasoning prompt suffix");
  }
  return `${rendered}</think>\n`;
}

async function loadChatTemplate(url, signal) {
  const response = await fetch(url.href, {
    cache: "no-store",
    credentials: "omit",
    method: "GET",
    redirect: "error",
    referrerPolicy: "no-referrer",
    signal,
  });
  if (!response.ok) {
    throw new Error(`Unable to load local chat template: HTTP ${response.status}`);
  }
  const template = await response.text();
  if (!template || template.length > 65_536) {
    throw new Error("The local chat template must be between 1 byte and 64 KiB");
  }
  return template;
}

function normalizeTokenIds(value) {
  const source = value?.input_ids ?? value?.inputIds ?? value;
  const unbatched =
    Array.isArray(source) && source.length === 1 && Array.isArray(source[0])
      ? source[0]
      : source?.tolist?.() ?? source;
  if (!Array.isArray(unbatched) && !ArrayBuffer.isView(unbatched)) {
    throw new TypeError("Maple tokenizer returned an unsupported token array");
  }
  return Array.from(unbatched, (token) => {
    const normalized = typeof token === "bigint" ? Number(token) : Number(token);
    if (!Number.isSafeInteger(normalized) || normalized < 0) {
      throw new RangeError("Maple tokenizer returned an invalid token id");
    }
    return normalized;
  });
}

function requireSameOriginDirectory(value, page, label) {
  const url = requireSameOriginFile(value, page, label);
  if (!url.pathname.endsWith("/")) throw new Error(`${label} must end with a slash`);
  return url;
}

function requireSameOriginFile(value, page, label) {
  const url = new URL(value, page);
  if (url.origin !== page.origin) throw new Error(`${label} must use this page's origin`);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error(`${label} must use HTTP or HTTPS`);
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error(`${label} must not contain credentials, query data, or a fragment`);
  }
  return url;
}

function throwIfAborted(signal) {
  if (!signal?.aborted) return;
  throw new DOMException(String(signal.reason ?? "Operation aborted"), "AbortError");
}
