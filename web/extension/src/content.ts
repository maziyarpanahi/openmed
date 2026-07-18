import type { OpenMedSpan } from "openmed";

import {
  DEFAULT_POLICY,
  POLICY_OPTIONS,
  isPolicyName,
  type PolicyName,
} from "./policy";

type EditableElement = HTMLInputElement | HTMLTextAreaElement | HTMLElement;

interface SiteSettingsResponse {
  ok: boolean;
  enabled?: boolean;
  policy?: string;
  error?: string;
}

interface ScanResponse extends SiteSettingsResponse {
  text?: string;
  deidentifiedText?: string;
  spans?: OpenMedSpan[];
}

interface UiElements {
  host: HTMLElement;
  status: HTMLElement;
  policy: HTMLSelectElement;
  toggle: HTMLButtonElement;
  mask: HTMLButtonElement;
}

const extensionApi =
  (globalThis as typeof globalThis & { browser?: typeof chrome }).browser ??
  chrome;
const editableResults = new Map<EditableElement, ScanResponse>();
const debounceTimers = new Map<EditableElement, number>();
const textRedactions = new Map<HTMLElement, string>();
const ui = createUi();
let enabled = true;
let currentPolicy: PolicyName = DEFAULT_POLICY;
let activeEditable: EditableElement | null = null;

void initialize();

async function initialize(): Promise<void> {
  const settings = await sendMessage<SiteSettingsResponse>({
    type: "openmed:get-settings",
  });
  if (!settings.ok) {
    showError(settings.error ?? "Unable to load extension settings");
    return;
  }
  enabled = settings.enabled ?? true;
  currentPolicy = isPolicyName(settings.policy)
    ? settings.policy
    : DEFAULT_POLICY;
  ui.policy.value = currentPolicy;
  renderEnabledState();
  installPageListeners();
  if (enabled) {
    scanExistingPage();
  }
}

function installPageListeners(): void {
  document.addEventListener("input", handleEditableEvent, true);
  document.addEventListener("focusin", handleEditableEvent, true);
  document.addEventListener("submit", handleSubmit, true);
  ui.toggle.addEventListener("click", () => void toggleSite());
  ui.mask.addEventListener("click", maskActiveEditable);
  ui.policy.addEventListener("change", () => void changePolicy());

  const observer = new MutationObserver((records) => {
    if (!enabled) {
      return;
    }
    for (const record of records) {
      for (const node of record.addedNodes) {
        scanNodeTree(node);
      }
    }
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
}

function handleEditableEvent(event: Event): void {
  const target = editableFromTarget(event.target);
  if (target === null) {
    return;
  }
  activeEditable = target;
  if (event.type === "focusin") {
    renderEditableResult(target);
  }
  scheduleEditableScan(target);
}

function handleSubmit(event: Event): void {
  if (!enabled || !(event.target instanceof HTMLFormElement)) {
    return;
  }
  const riskyEntry = [...editableResults.entries()].find(
    ([element, result]) =>
      event.target instanceof HTMLFormElement &&
      event.target.contains(element) &&
      result.text === editableText(element) &&
      (result.spans?.length ?? 0) > 0,
  );
  if (riskyEntry === undefined) {
    return;
  }
  event.preventDefault();
  event.stopImmediatePropagation();
  activeEditable = riskyEntry[0];
  ui.mask.disabled = false;
  setStatus("PHI detected. Mask it before submitting.", "warning");
}

function scheduleEditableScan(target: EditableElement): void {
  const pending = debounceTimers.get(target);
  if (pending !== undefined) {
    window.clearTimeout(pending);
  }
  if (!enabled) {
    clearEditableResult(target);
    return;
  }
  const timer = window.setTimeout(() => {
    debounceTimers.delete(target);
    void scanEditable(target);
  }, 150);
  debounceTimers.set(target, timer);
}

async function scanEditable(target: EditableElement): Promise<void> {
  const text = editableText(target);
  if (!enabled || text.trim().length === 0) {
    clearEditableResult(target);
    return;
  }
  const response = await scanText(text);
  if (!enabled || editableText(target) !== text) {
    return;
  }
  if (!response.ok) {
    showError(response.error ?? "Detection failed");
    return;
  }
  editableResults.set(target, response);
  const spans = response.spans ?? [];
  target.dataset.openmedPhiCount = String(spans.length);
  target.dataset.openmedPhiLabels = spans
    .map((span) => span.canonical_label)
    .join(",");
  if (activeEditable === target) {
    renderEditableResult(target);
  }
}

function renderEditableResult(target: EditableElement): void {
  const response = editableResults.get(target);
  const count = response?.spans?.length ?? 0;
  ui.mask.disabled = count === 0 || !enabled;
  if (count === 0) {
    setStatus("No PHI detected in the active field.", "safe");
    return;
  }
  setStatus(
    `${count} PHI span${count === 1 ? "" : "s"} detected on-device.`,
    "warning",
  );
}

function clearEditableResult(target: EditableElement): void {
  editableResults.delete(target);
  delete target.dataset.openmedPhiCount;
  delete target.dataset.openmedPhiLabels;
  if (activeEditable === target) {
    ui.mask.disabled = true;
  }
}

function maskActiveEditable(): void {
  if (!enabled || activeEditable === null) {
    return;
  }
  const result = editableResults.get(activeEditable);
  if (result?.deidentifiedText === undefined || (result.spans?.length ?? 0) === 0) {
    return;
  }
  setEditableText(activeEditable, result.deidentifiedText);
  clearEditableResult(activeEditable);
  activeEditable.dispatchEvent(new Event("input", { bubbles: true }));
  setStatus("PHI masked on-device. Review the text before sharing.", "safe");
}

async function toggleSite(): Promise<void> {
  const nextEnabled = !enabled;
  const response = await sendMessage<SiteSettingsResponse>({
    type: "openmed:set-enabled",
    enabled: nextEnabled,
  });
  if (!response.ok) {
    showError(response.error ?? "Unable to update this site");
    return;
  }
  enabled = nextEnabled;
  if (!enabled) {
    clearAllPageState();
  }
  renderEnabledState();
  if (enabled) {
    scanExistingPage();
  }
}

async function changePolicy(): Promise<void> {
  const selected = ui.policy.value;
  if (!isPolicyName(selected)) {
    showError("Unknown policy profile");
    return;
  }
  const response = await sendMessage<SiteSettingsResponse>({
    type: "openmed:set-policy",
    policy: selected,
  });
  if (!response.ok) {
    showError(response.error ?? "Unable to update the policy profile");
    return;
  }
  currentPolicy = selected;
  restoreTextRedactions();
  for (const editable of editableResults.keys()) {
    clearEditableResult(editable);
    scheduleEditableScan(editable);
  }
  setStatus(`Using ${selected.replaceAll("_", " ")}.`, "ready");
  scanDocumentText();
}

function scanExistingPage(): void {
  for (const element of document.querySelectorAll("input, textarea, [contenteditable='true']")) {
    const editable = editableFromTarget(element);
    if (editable !== null && editableText(editable).trim().length > 0) {
      scheduleEditableScan(editable);
    }
  }
  scanDocumentText();
}

function scanDocumentText(): void {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  while (walker.nextNode()) {
    if (walker.currentNode instanceof Text) {
      nodes.push(walker.currentNode);
    }
  }
  for (const node of nodes) {
    void scanTextNode(node);
  }
}

function scanNodeTree(node: Node): void {
  if (node instanceof Text) {
    void scanTextNode(node);
    return;
  }
  if (!(node instanceof Element) || shouldSkipElement(node)) {
    return;
  }
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    if (walker.currentNode instanceof Text) {
      void scanTextNode(walker.currentNode);
    }
  }
}

async function scanTextNode(node: Text): Promise<void> {
  const parent = node.parentElement;
  const text = node.data;
  if (
    !enabled ||
    parent === null ||
    shouldSkipElement(parent) ||
    text.trim().length < 3 ||
    !/[\d@]|\b(?:Patient|Doctor|Dr\.)\b/.test(text)
  ) {
    return;
  }
  const response = await scanText(text);
  if (!enabled || !node.isConnected || node.data !== text || !response.ok) {
    return;
  }
  const spans = response.spans ?? [];
  if (spans.length === 0) {
    return;
  }
  const wrapper = document.createElement("span");
  wrapper.dataset.openmedTextRedaction = "true";
  let cursor = 0;
  for (const span of spans.sort((left, right) => left.start - right.start)) {
    wrapper.append(text.slice(cursor, span.start));
    const mark = document.createElement("mark");
    mark.dataset.openmedPhi = span.canonical_label;
    mark.title = `OpenMed masked ${span.canonical_label}`;
    mark.textContent = `[${span.canonical_label}]`;
    mark.style.background = "#ffe2a8";
    mark.style.color = "#532f00";
    mark.style.borderRadius = "0.2em";
    mark.style.padding = "0 0.15em";
    wrapper.append(mark);
    cursor = span.end;
  }
  wrapper.append(text.slice(cursor));
  textRedactions.set(wrapper, text);
  node.replaceWith(wrapper);
}

function shouldSkipElement(element: Element): boolean {
  return (
    element.closest(
      "#openmed-phi-guard, [data-openmed-text-redaction], script, style, noscript, textarea, input, [contenteditable='true']",
    ) !== null
  );
}

async function scanText(text: string): Promise<ScanResponse> {
  return sendMessage<ScanResponse>({ type: "openmed:scan", text });
}

function clearAllPageState(): void {
  for (const timer of debounceTimers.values()) {
    window.clearTimeout(timer);
  }
  debounceTimers.clear();
  for (const editable of editableResults.keys()) {
    clearEditableResult(editable);
  }
  restoreTextRedactions();
  ui.mask.disabled = true;
}

function restoreTextRedactions(): void {
  for (const [wrapper, original] of textRedactions) {
    if (wrapper.isConnected) {
      wrapper.replaceWith(document.createTextNode(original));
    }
  }
  textRedactions.clear();
}

function renderEnabledState(): void {
  ui.toggle.textContent = enabled ? "Disable on this site" : "Enable on this site";
  ui.toggle.dataset.enabled = String(enabled);
  ui.policy.disabled = !enabled;
  const activeCount =
    activeEditable === null
      ? 0
      : (editableResults.get(activeEditable)?.spans?.length ?? 0);
  ui.mask.disabled = !enabled || activeCount === 0;
  setStatus(
    enabled
      ? "OpenMed PHI Guard is ready."
      : "OpenMed PHI Guard is disabled on this site.",
    enabled ? "ready" : "disabled",
  );
}

function showError(message: string): void {
  setStatus(message, "error");
}

function setStatus(message: string, state: string): void {
  ui.status.textContent = message;
  ui.status.dataset.state = state;
}

function editableFromTarget(target: EventTarget | null): EditableElement | null {
  if (target instanceof HTMLTextAreaElement) {
    return target;
  }
  if (target instanceof HTMLInputElement) {
    const supportedTypes = new Set(["", "email", "search", "tel", "text", "url"]);
    return supportedTypes.has(target.type) ? target : null;
  }
  if (target instanceof HTMLElement && target.isContentEditable) {
    return target;
  }
  return null;
}

function editableText(target: EditableElement): string {
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) {
    return target.value;
  }
  return target.textContent ?? "";
}

function setEditableText(target: EditableElement, text: string): void {
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) {
    const prototype =
      target instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
    setter?.call(target, text);
    return;
  }
  target.textContent = text;
}

async function sendMessage<T>(message: object): Promise<T> {
  try {
    return (await extensionApi.runtime.sendMessage(message)) as T;
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Extension worker unavailable",
    } as T;
  }
}

function createUi(): UiElements {
  const host = document.createElement("aside");
  host.id = "openmed-phi-guard";
  host.setAttribute("aria-label", "OpenMed PHI Guard");
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; }
    .panel {
      position: fixed;
      right: 16px;
      bottom: 16px;
      z-index: 2147483647;
      width: 280px;
      box-sizing: border-box;
      padding: 14px;
      border: 1px solid #b9c8c2;
      border-radius: 12px;
      background: #f8fffc;
      color: #15352d;
      box-shadow: 0 8px 28px rgb(20 53 45 / 22%);
      font: 13px/1.4 system-ui, sans-serif;
    }
    strong { display: block; margin-bottom: 6px; font-size: 14px; }
    [data-state="warning"] { color: #8a3d00; }
    [data-state="error"] { color: #a01818; }
    label { display: block; margin-top: 10px; font-size: 12px; }
    select, button {
      width: 100%;
      box-sizing: border-box;
      margin-top: 5px;
      padding: 7px 9px;
      border: 1px solid #91aaa1;
      border-radius: 7px;
      background: white;
      color: #15352d;
      font: inherit;
    }
    button { cursor: pointer; }
    button:disabled { cursor: default; opacity: 0.55; }
    .primary { background: #176b54; color: white; border-color: #176b54; }
  `;
  const panel = document.createElement("section");
  panel.className = "panel";

  const title = document.createElement("strong");
  title.textContent = "OpenMed PHI Guard";
  const status = document.createElement("div");
  status.dataset.testid = "openmed-status";
  status.setAttribute("aria-live", "polite");

  const policyLabel = document.createElement("label");
  policyLabel.textContent = "Policy profile";
  const policy = document.createElement("select");
  policy.dataset.testid = "openmed-policy";
  for (const option of POLICY_OPTIONS) {
    const element = document.createElement("option");
    element.value = option.name;
    element.textContent = option.label;
    policy.append(element);
  }
  policyLabel.append(policy);

  const mask = document.createElement("button");
  mask.type = "button";
  mask.className = "primary";
  mask.dataset.testid = "openmed-mask";
  mask.textContent = "Mask PHI";
  mask.disabled = true;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.dataset.testid = "openmed-site-toggle";

  panel.append(title, status, policyLabel, mask, toggle);
  shadow.append(style, panel);
  document.documentElement.append(host);
  return { host, status, policy, toggle, mask };
}
