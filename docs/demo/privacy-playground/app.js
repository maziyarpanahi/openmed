const BUILTIN_RULES = [
  {
    label: "EMAIL",
    pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,
  },
  {
    label: "PHONE",
    pattern: /(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b/g,
  },
  {
    label: "DATE",
    pattern: /\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b/g,
  },
  {
    label: "ID",
    pattern: /\b(?:MRN|ID)[-_][A-Z0-9-]{3,}\b/gi,
  },
];

const runtimeInput = document.querySelector("#runtime-module");
const modelInput = document.querySelector("#model-url");
const textInput = document.querySelector("#input-text");
const redactButton = document.querySelector("#redact-button");
const resetButton = document.querySelector("#reset-button");
const processingStatus = document.querySelector("#processing-status");
const networkStatus = document.querySelector("#network-status");
const uploadCount = document.querySelector("#upload-count");
const runtimeStatus = document.querySelector("#runtime-status");
const status = document.querySelector("#status");
const redactedOutput = document.querySelector("#redacted-output");
const redactionCounts = document.querySelector("#redaction-counts");
const totalCount = document.querySelector("#total-count");

let blockedUploadAttempts = 0;
let activeConfiguration = "";
let detector = null;
let detectorMode = "builtin";

installLocalOnlyNetworkPolicy();
initializeFromQuery();
renderNetworkStatus();

redactButton.addEventListener("click", () => redactLocally());
resetButton.addEventListener("click", () => clearResult());
for (const input of [runtimeInput, modelInput]) {
  input.addEventListener("change", () => {
    activeConfiguration = "";
    detector = null;
    detectorMode = "builtin";
    runtimeStatus.textContent = "Adapter configuration changed.";
  });
}
window.addEventListener("online", renderNetworkStatus);
window.addEventListener("offline", renderNetworkStatus);

async function redactLocally() {
  setBusy(true);
  try {
    const input = textInput.value;
    if (!input.trim()) {
      throw new Error("empty-input");
    }

    const localDetector = await loadDetector();
    setStatus("Processing in this browser tab…");
    const output = await localDetector(input, {
      aggregation_strategy: "simple",
    });
    const spans = normalizeOutput(output, input);
    renderResult(input, spans);
    setStatus(
      `${spans.length} redaction${spans.length === 1 ? "" : "s"} completed locally.`,
    );
  } catch (error) {
    clearResult({ keepStatus: true });
    setStatus(publicErrorMessage(error), "error");
  } finally {
    setBusy(false);
  }
}

async function loadDetector() {
  const runtimeValue = runtimeInput.value.trim();
  const modelValue = modelInput.value.trim();
  if (!runtimeValue && !modelValue) {
    detectorMode = "builtin";
    runtimeStatus.textContent = "Deterministic local rules selected.";
    return builtinDetector;
  }
  if (!runtimeValue || !modelValue) {
    throw new Error("incomplete-adapter");
  }

  const configuration = localConfiguration(runtimeValue, modelValue);
  if (detector && activeConfiguration === configuration.key) {
    return detector;
  }

  runtimeStatus.textContent = "Loading same-origin adapter…";
  const runtime = await import(configuration.runtimeUrl.href);
  if (typeof runtime.createOpenMedPipeline !== "function") {
    throw new Error("invalid-adapter");
  }
  const candidate = await runtime.createOpenMedPipeline({
    backend: "wasm",
    dtype: "q8",
    modelUrl: configuration.modelUrl.href,
    task: "token-classification",
  });
  if (typeof candidate !== "function") {
    throw new Error("invalid-detector");
  }

  activeConfiguration = configuration.key;
  detector = candidate;
  detectorMode = "adapter";
  runtimeStatus.textContent = "Same-origin local adapter ready.";
  return detector;
}

function localConfiguration(runtimeValue, modelValue) {
  const runtimeUrl = sameOriginUrl(runtimeValue);
  const modelUrl = sameOriginUrl(modelValue);
  return {
    key: `${runtimeUrl.href}\n${modelUrl.href}`,
    modelUrl,
    runtimeUrl,
  };
}

function sameOriginUrl(value) {
  const resolved = new URL(value, window.location.href);
  if (resolved.origin !== window.location.origin) {
    throw new Error("cross-origin-adapter");
  }
  return resolved;
}

async function builtinDetector(text) {
  const spans = [];
  for (const rule of BUILTIN_RULES) {
    rule.pattern.lastIndex = 0;
    for (const match of text.matchAll(rule.pattern)) {
      spans.push({
        end: match.index + match[0].length,
        entity_group: rule.label,
        score: 1,
        start: match.index,
      });
    }
  }
  return spans;
}

function normalizeOutput(output, text) {
  const flat = Array.isArray(output?.[0]) ? output.flat() : output;
  if (!Array.isArray(flat)) {
    return [];
  }

  const located = [];
  let searchCursor = 0;
  for (const item of flat) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const rawLabel = item.entity_group ?? item.entity ?? item.label ?? "PII";
    const label = normalizeLabel(rawLabel);
    if (label === "O") {
      continue;
    }

    let start = Number(item.start);
    let end = Number(item.end);
    if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) {
      const match = locateWord(text, item.word, searchCursor);
      if (!match) {
        continue;
      }
      ({ start, end } = match);
    }
    if (start < 0 || end > text.length || end <= start) {
      continue;
    }
    searchCursor = end;
    located.push({
      end,
      label,
      score: Number(item.score ?? 0),
      start,
    });
  }

  return mergeBioSpans(located, text).sort(
    (left, right) => left.start - right.start || right.end - left.end,
  );
}

function locateWord(text, modelWord, searchCursor) {
  const cleaned = String(modelWord ?? "")
    .replace(/^##/, "")
    .replace(/^[▁Ġ]+/, "")
    .trim();
  if (!cleaned) {
    return null;
  }
  const start = text.toLocaleLowerCase().indexOf(
    cleaned.toLocaleLowerCase(),
    searchCursor,
  );
  return start === -1 ? null : { end: start + cleaned.length, start };
}

function mergeBioSpans(spans, text) {
  const merged = [];
  for (const span of spans) {
    const match = /^([BIES])-([\s\S]+)$/.exec(span.label);
    const prefix = match?.[1] ?? null;
    const label = match?.[2] ?? span.label;
    const previous = merged.at(-1);
    const gap = previous ? text.slice(previous.end, span.start) : "";
    const continuesPrevious =
      previous &&
      previous.label === label &&
      (((prefix === "I" || prefix === "E") &&
        /^[\s.,@+:/()\-]*$/.test(gap)) ||
        (prefix === null && gap === ""));

    if (continuesPrevious) {
      previous.end = span.end;
      previous.score = (previous.score + span.score) / 2;
      continue;
    }
    merged.push({ ...span, label });
  }
  return merged;
}

function renderResult(text, spans) {
  const accepted = nonOverlappingSpans(spans, text.length);
  redactedOutput.textContent = redactText(text, accepted);
  totalCount.textContent = String(accepted.length);
  while (redactionCounts.children.length > 1) {
    redactionCounts.lastElementChild.remove();
  }

  const counts = new Map();
  for (const span of accepted) {
    counts.set(span.label, (counts.get(span.label) ?? 0) + 1);
  }
  for (const [label, count] of [...counts.entries()].sort()) {
    redactionCounts.append(createCountRow(label, count));
  }
  processingStatus.textContent =
    detectorMode === "adapter"
      ? "Same-origin adapter in this tab"
      : "Deterministic rules in this tab";
}

function createCountRow(label, count) {
  const row = document.createElement("tr");
  const name = document.createElement("th");
  name.scope = "row";
  name.textContent = label;
  const value = document.createElement("td");
  value.className = "numeric";
  value.textContent = String(count);
  row.append(name, value);
  return row;
}

function redactText(text, spans) {
  if (spans.length === 0) {
    return "No sensitive spans detected locally.";
  }
  let cursor = 0;
  let output = "";
  for (const span of spans) {
    output += text.slice(cursor, span.start);
    output += `[${span.label}]`;
    cursor = span.end;
  }
  return output + text.slice(cursor);
}

function nonOverlappingSpans(spans, textLength) {
  const accepted = [];
  let cursor = 0;
  for (const span of spans) {
    const start = Math.max(0, Math.min(span.start, textLength));
    const end = Math.max(start, Math.min(span.end, textLength));
    if (start < cursor || end === start) {
      continue;
    }
    accepted.push({ ...span, start, end });
    cursor = end;
  }
  return accepted;
}

function normalizeLabel(label) {
  return String(label).toUpperCase();
}

function clearResult({ keepStatus = false } = {}) {
  redactedOutput.textContent = "No local result yet.";
  totalCount.textContent = "0";
  while (redactionCounts.children.length > 1) {
    redactionCounts.lastElementChild.remove();
  }
  processingStatus.textContent = "This browser tab";
  if (!keepStatus) {
    setStatus("Result cleared. Input remains in this tab only.");
  }
}

function setBusy(busy) {
  redactButton.disabled = busy;
  resetButton.disabled = busy;
}

function setStatus(message, kind = "info") {
  status.textContent = message;
  status.dataset.kind = kind;
}

function publicErrorMessage(error) {
  switch (error?.message) {
    case "empty-input":
      return "Enter synthetic text before running local redaction.";
    case "incomplete-adapter":
      return "Provide both local adapter fields, or leave both empty.";
    case "cross-origin-adapter":
      return "Local adapter and model URLs must use this page's origin.";
    case "invalid-adapter":
    case "invalid-detector":
      return "The local adapter does not implement the OpenMed browser contract.";
    default:
      return "Local redaction failed. Check the supplied adapter and model bundle.";
  }
}

function initializeFromQuery() {
  const query = new URLSearchParams(window.location.search);
  runtimeInput.value = query.get("runtime") ?? "";
  modelInput.value = query.get("model") ?? query.get("model_url") ?? "";
  if (runtimeInput.value || modelInput.value) {
    runtimeStatus.textContent = "Local adapter configuration loaded from the URL.";
  }
}

function installLocalOnlyNetworkPolicy() {
  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const requestUrl = input instanceof Request ? input.url : input;
    const method = String(
      init.method ?? (input instanceof Request ? input.method : "GET"),
    ).toUpperCase();
    if (!allowedAssetRequest(requestUrl, method) || init.body != null) {
      recordBlockedUpload();
      return Promise.reject(new Error("local-only-network-policy"));
    }
    return nativeFetch(input, init);
  };

  const xhrOpen = XMLHttpRequest.prototype.open;
  const xhrSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__openmedLocalOnly = allowedAssetRequest(url, method);
    this.__openmedBlocked = !this.__openmedLocalOnly;
    if (!this.__openmedLocalOnly) {
      recordBlockedUpload();
      return undefined;
    }
    return xhrOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (body) {
    if (this.__openmedBlocked) {
      return undefined;
    }
    if (body != null) {
      recordBlockedUpload();
      this.__openmedBlocked = true;
      return undefined;
    }
    return xhrSend.call(this, body);
  };

  try {
    navigator.sendBeacon = () => {
      recordBlockedUpload();
      return false;
    };
  } catch {
    // A hardened browser may expose sendBeacon as read-only; CSP still applies.
  }
  try {
    window.WebSocket = class LocalOnlyWebSocket {
      constructor() {
        recordBlockedUpload();
        throw new Error("local-only-network-policy");
      }
    };
  } catch {
    // A hardened browser may expose WebSocket as read-only; CSP still applies.
  }
}

function allowedAssetRequest(value, method) {
  try {
    const url = new URL(value, window.location.href);
    return url.origin === window.location.origin && ["GET", "HEAD"].includes(
      String(method).toUpperCase(),
    );
  } catch {
    return false;
  }
}

function recordBlockedUpload() {
  blockedUploadAttempts += 1;
  uploadCount.textContent = `${blockedUploadAttempts} blocked`;
}

function renderNetworkStatus() {
  const connection = navigator.onLine
    ? "Browser reports online"
    : "Browser reports offline";
  networkStatus.textContent = `${connection}; uploads blocked`;
}
