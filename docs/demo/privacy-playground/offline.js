// Explicit installable offline profile controls for the privacy playground.
//
// Nothing is registered or stored until the user selects "Prepare offline
// use". The page never fetches assets itself: the service worker downloads,
// verifies and serves the versioned files, and this module only renders its
// state. Messages exchanged with the worker carry fixed codes, counts and the
// generation digest, never text, results or request data.

const MESSAGE_SOURCE = "openmed-offline";
const SCOPE_URL = new URL("./", import.meta.url);
const WORKER_URL = new URL("./offline-sw.js", import.meta.url);
const REPLY_TIMEOUT_MS = 15_000;
const REPAIR_TIMEOUT_MS = 300_000;

const STATES = {
  failed: { actions: ["prepare"], kind: "error", label: "Preparation failed" },
  "not-prepared": { actions: ["prepare"], kind: "info", label: "Not prepared" },
  preparing: { actions: [], kind: "info", label: "Preparing" },
  ready: {
    actions: ["update", "remove"],
    kind: "info",
    label: "Running from the verified offline copy",
  },
  "ready-reopen": {
    actions: ["update", "remove"],
    kind: "info",
    label: "Verified; reopen to use",
  },
  repair: { actions: ["repair", "remove"], kind: "error", label: "Repair needed" },
  unsupported: { actions: [], kind: "error", label: "Unsupported" },
  "update-failed": {
    actions: ["update", "remove"],
    kind: "error",
    label: "Update rejected; current copy kept",
  },
  "update-ready": {
    actions: ["switch", "remove"],
    kind: "info",
    label: "Verified update waiting",
  },
  updating: { actions: [], kind: "info", label: "Verifying update" },
};

const MESSAGES = {
  "asset-digest-mismatch":
    "A file did not match its published SHA-256 digest, so it was not used.",
  "asset-fetch-failed":
    "A file could not be downloaded completely. Nothing partial was kept.",
  "asset-missing": "A stored file is missing from the offline copy.",
  "asset-size-mismatch": "A file had an unexpected size, so it was not used.",
  "cache-storage-unavailable": "This browser does not provide Cache Storage here.",
  "cache-write-failed": "The browser refused to store the offline copy.",
  "digest-unavailable": "This browser cannot compute SHA-256 digests here.",
  "generation-mismatch":
    "The published files belong to different versions. Try again after the site update completes.",
  "insecure-context": "Offline install needs an HTTPS or localhost page.",
  "insufficient-storage":
    "The browser does not have enough free storage for a verified copy.",
  "manifest-invalid": "The published asset manifest is invalid.",
  "manifest-unavailable": "The published asset manifest could not be downloaded.",
  "offline-copy-evicted":
    "The browser removed the offline copy, for example to free storage space.",
  "offline-copy-incomplete": "The offline copy was not completely prepared.",
  "other-tabs-open":
    "Close the other playground tabs first; the update then switches without mixing versions.",
  "profile-invalid": "The offline profile failed its integrity check.",
  "profile-unbuilt": "This build has no versioned offline profile.",
  "quota-exceeded": "The browser storage quota was exceeded.",
  "registration-failed": "The browser did not register the offline worker.",
  "service-worker-timeout": "The offline worker did not answer.",
  "service-worker-unavailable": "This browser does not support service workers here.",
  "unexpected-error": "The offline copy could not be verified.",
  "update-check-failed":
    "Checking for an update needs a network connection; the verified copy stays in use.",
};

const card = document.querySelector("#offline-card");
const stateCell = document.querySelector("#offline-state");
const generationCell = document.querySelector("#offline-generation");
const storageCell = document.querySelector("#offline-storage");
const statusLine = document.querySelector("#offline-status");
const buttons = {
  prepare: document.querySelector("#offline-prepare"),
  remove: document.querySelector("#offline-remove"),
  repair: document.querySelector("#offline-repair"),
  switch: document.querySelector("#offline-switch"),
  update: document.querySelector("#offline-update"),
};

const tracked = new WeakSet();
let registration = null;
let currentState = "not-prepared";
let lastFailureCode = "";
let preparedGeneration = "";
const startedControlled = Boolean(navigator.serviceWorker?.controller);

if (card) {
  buttons.prepare.addEventListener("click", () => prepare());
  buttons.update.addEventListener("click", () => checkForUpdate());
  buttons.switch.addEventListener("click", () => switchToUpdate());
  buttons.repair.addEventListener("click", () => repair());
  buttons.remove.addEventListener("click", () => remove());
  initialise().catch(() => render("unsupported", { code: "unexpected-error" }));
}

async function initialise() {
  const unsupported = unsupportedReason();
  if (unsupported) {
    render("unsupported", { code: unsupported });
    return;
  }
  navigator.serviceWorker.addEventListener("message", onWorkerMessage);
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    // A generation switch must never leave this page running files from two
    // generations, so reload into the newly verified one.
    if (startedControlled) {
      location.reload();
    }
  });
  let existing;
  try {
    existing = await navigator.serviceWorker.getRegistration(SCOPE_URL.href);
  } catch {
    existing = undefined;
  }
  if (!existing) {
    render("not-prepared");
    return;
  }
  adopt(existing);
}

function unsupportedReason() {
  if (!window.isSecureContext) {
    return "insecure-context";
  }
  if (!("serviceWorker" in navigator)) {
    return "service-worker-unavailable";
  }
  if (!("caches" in window)) {
    return "cache-storage-unavailable";
  }
  if (!window.crypto?.subtle) {
    return "digest-unavailable";
  }
  return "";
}

function adopt(candidate) {
  registration = candidate;
  candidate.addEventListener("updatefound", () => track(candidate.installing));
  if (candidate.installing) {
    track(candidate.installing);
  } else if (candidate.waiting && candidate.active) {
    render("update-ready");
  } else if (navigator.serviceWorker.controller) {
    refreshStatus(navigator.serviceWorker.controller, "ready");
  } else if (candidate.active) {
    refreshStatus(candidate.active, "ready-reopen");
  } else {
    render("not-prepared");
  }
}

function track(worker) {
  if (!worker || tracked.has(worker)) {
    return;
  }
  tracked.add(worker);
  const isUpdate = Boolean(registration?.active && registration.active !== worker);
  lastFailureCode = "";
  render(isUpdate ? "updating" : "preparing");
  let installed = false;
  worker.addEventListener("statechange", () => {
    if (worker.state === "installed") {
      installed = true;
      if (isUpdate) {
        render("update-ready", { generation: preparedGeneration });
      }
    } else if (worker.state === "activated" && !isUpdate) {
      render(navigator.serviceWorker.controller ? "ready" : "ready-reopen", {
        generation: preparedGeneration,
      });
      refreshStorage();
    } else if (worker.state === "redundant" && !installed) {
      render(isUpdate ? "update-failed" : "failed", {
        code: lastFailureCode || "unexpected-error",
      });
    }
  });
}

function onWorkerMessage(event) {
  const data = event.data;
  if (!data || data.source !== MESSAGE_SOURCE) {
    return;
  }
  const code = knownCode(data.code);
  switch (data.event) {
    case "progress":
      if (currentState === "preparing" || currentState === "updating") {
        statusLine.textContent = `Verified ${Number(data.done) || 0} of ${
          Number(data.total) || 0
        } files.`;
      }
      break;
    case "prepared":
      preparedGeneration = safeGeneration(data.generation);
      if (currentState === "preparing" || currentState === "updating") {
        statusLine.textContent = "Every file matched its published digest.";
      }
      break;
    case "prepare-failed":
      lastFailureCode = code;
      if (currentState === "failed" || currentState === "update-failed") {
        render(currentState, { code });
      }
      break;
    case "repair-required":
      render("repair", { code });
      break;
    default:
      break;
  }
}

async function prepare() {
  render("preparing");
  await requestPersistence();
  let created;
  try {
    created = await navigator.serviceWorker.register(WORKER_URL.href, {
      scope: SCOPE_URL.href,
      updateViaCache: "none",
    });
  } catch {
    created = undefined;
  }
  if (!created) {
    render("failed", { code: "registration-failed" });
    return;
  }
  adopt(created);
}

async function checkForUpdate() {
  if (!registration) {
    render("not-prepared");
    return;
  }
  const stable = currentState;
  statusLine.textContent = "Checking for a published update.";
  try {
    await registration.update();
  } catch {
    if (currentState === stable) {
      render(stable, { detail: MESSAGES["update-check-failed"] });
    }
    return;
  }
  if (!registration.installing && !registration.waiting && currentState === stable) {
    render(stable, {
      detail: "No newer version is published; the verified copy stays in use.",
    });
  }
}

async function switchToUpdate() {
  const waiting = registration?.waiting;
  if (!waiting) {
    return;
  }
  setBusy(true);
  const reply = await ask(waiting, "openmed-offline:activate", REPLY_TIMEOUT_MS);
  if (!reply.ok) {
    setBusy(false);
    render("update-ready", { code: knownCode(reply.code) });
    return;
  }
  statusLine.textContent = "Switching to the verified update.";
  if (!startedControlled) {
    location.reload();
  }
}

async function repair() {
  const worker = navigator.serviceWorker.controller ?? registration?.active;
  if (!worker) {
    render("repair", { code: "service-worker-unavailable" });
    return;
  }
  setBusy(true);
  statusLine.textContent = "Downloading and verifying the offline copy.";
  const reply = await ask(worker, "openmed-offline:repair", REPAIR_TIMEOUT_MS);
  if (reply.ok) {
    location.reload();
    return;
  }
  setBusy(false);
  render("repair", { code: knownCode(reply.code) });
}

async function remove() {
  const target = navigator.serviceWorker.controller ?? registration?.active;
  setBusy(true);
  let removed = false;
  if (target) {
    removed = Boolean((await ask(target, "openmed-offline:remove", REPLY_TIMEOUT_MS)).ok);
  } else if (registration) {
    removed = await registration.unregister().catch(() => false);
  }
  setBusy(false);
  if (!removed) {
    render(currentState, { detail: "The offline copy could not be removed." });
    return;
  }
  registration = null;
  render("not-prepared", {
    detail: "Offline copy removed. This page keeps working while online.",
  });
  if (startedControlled) {
    location.reload();
  }
}

async function refreshStatus(worker, readyState) {
  const reply = await ask(worker, "openmed-offline:status", REPLY_TIMEOUT_MS);
  if (reply.ok && reply.state === "ready") {
    render(readyState, { generation: safeGeneration(reply.generation) });
  } else {
    render("repair", {
      code: knownCode(reply.code),
      generation: safeGeneration(reply.generation),
    });
  }
  refreshStorage();
}

async function requestPersistence() {
  try {
    await navigator.storage?.persist?.();
  } catch {
    // Persistence is a best-effort request; eviction is still handled.
  }
  await refreshStorage();
}

async function refreshStorage() {
  if (!storageCell) {
    return;
  }
  try {
    const persisted = await navigator.storage?.persisted?.();
    storageCell.textContent = persisted
      ? "Persistent storage granted"
      : "Best effort; the browser may evict it";
  } catch {
    storageCell.textContent = "Not reported by this browser";
  }
}

function ask(worker, type, timeoutMs) {
  return new Promise((resolve) => {
    if (!worker) {
      resolve({ code: "service-worker-unavailable", ok: false });
      return;
    }
    const channel = new MessageChannel();
    const timer = setTimeout(() => {
      resolve({ code: "service-worker-timeout", ok: false });
    }, timeoutMs);
    channel.port1.onmessage = (event) => {
      clearTimeout(timer);
      resolve(event.data && typeof event.data === "object" ? event.data : {});
    };
    try {
      worker.postMessage({ type }, [channel.port2]);
    } catch {
      clearTimeout(timer);
      resolve({ code: "service-worker-unavailable", ok: false });
    }
  });
}

function render(state, { code = "", detail = "", generation = "" } = {}) {
  const definition = STATES[state] ?? STATES.unsupported;
  currentState = state;
  card.dataset.offlineState = state;
  if (code) {
    card.dataset.offlineCode = code;
  } else {
    delete card.dataset.offlineCode;
  }
  if (generation) {
    card.dataset.offlineGeneration = generation;
    generationCell.textContent = generation.slice(0, 12);
  } else if (state === "not-prepared" || state === "unsupported") {
    delete card.dataset.offlineGeneration;
    generationCell.textContent = "None";
  }
  stateCell.textContent = definition.label;
  for (const [name, button] of Object.entries(buttons)) {
    button.hidden = !definition.actions.includes(name);
    button.disabled = false;
  }
  statusLine.dataset.kind = definition.kind;
  statusLine.textContent = detail || (code ? MESSAGES[code] : defaultText(state));
  if (state === "repair" && navigator.serviceWorker?.controller) {
    lockApplication();
  }
}

function defaultText(state) {
  switch (state) {
    case "not-prepared":
      return "Offline use is off until you prepare it.";
    case "preparing":
      return "Downloading and verifying each file.";
    case "ready":
      return "Every file matched its digest. The playground works offline.";
    case "ready-reopen":
      return "Verified offline copy stored. Reopen this page to run from it.";
    case "updating":
      return "A new version is being verified; the current copy keeps serving.";
    case "update-ready":
      return "A verified update is waiting. Switch now, or it activates after every playground tab closes.";
    default:
      return "";
  }
}

function lockApplication() {
  // An unverified generation must not run: stop the redaction controls until
  // the copy is repaired or removed.
  for (const section of document.querySelectorAll("main .grid > section")) {
    if (section !== card) {
      section.inert = true;
    }
  }
  const redactButton = document.querySelector("#redact-button");
  if (redactButton) {
    redactButton.disabled = true;
  }
}

function setBusy(busy) {
  for (const button of Object.values(buttons)) {
    button.disabled = busy;
  }
}

function knownCode(value) {
  return typeof value === "string" && Object.hasOwn(MESSAGES, value)
    ? value
    : "unexpected-error";
}

function safeGeneration(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value) ? value : "";
}
