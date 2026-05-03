// OpenMed Privacy Filter Stream — frontend.

const STATE = {
  scenarios: [],
  activeScenarioId: null,
  source: null,
  running: false,
  mode: "replace",
  rawChars: 0,
  sharedChars: 0,
  entityCount: 0,
};

const els = {
  scenarioStrip: document.querySelector("#scenarioStrip"),
  delayRange: document.querySelector("#delayRange"),
  delayValue: document.querySelector("#delayValue"),
  speedRange: document.querySelector("#speedRange"),
  speedValue: document.querySelector("#speedValue"),
  rawChars: document.querySelector("#rawChars"),
  sharedChars: document.querySelector("#sharedChars"),
  entityCount: document.querySelector("#entityCount"),
  lagLabel: document.querySelector("#lagLabel"),
  statusPill: document.querySelector("#statusPill"),
  rawStream: document.querySelector("#rawStream"),
  safeStream: document.querySelector("#safeStream"),
  rawLiveDot: document.querySelector("#rawLiveDot"),
  safeLiveDot: document.querySelector("#safeLiveDot"),
  startButton: document.querySelector("#startButton"),
  resetButton: document.querySelector("#resetButton"),
  modeButtons: document.querySelectorAll(".mode-button"),
};

const MAX_OUTPUT_CHARS = 42000;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(kind, label) {
  els.statusPill.className = "status-pill";
  if (kind) els.statusPill.classList.add(kind);
  els.statusPill.textContent = label;
}

function setLive(rawLive, safeLive) {
  els.rawLiveDot.classList.toggle("is-live", rawLive);
  els.safeLiveDot.classList.toggle("is-live", safeLive);
}

function formatDelay(ms) {
  return `${(Number(ms) / 1000).toFixed(1)}s`;
}

function updateRangeLabels() {
  els.delayValue.textContent = formatDelay(els.delayRange.value);
  els.lagLabel.textContent = formatDelay(els.delayRange.value);
  els.speedValue.textContent = `${els.speedRange.value}ms`;
}

function renderScenarios() {
  els.scenarioStrip.innerHTML = STATE.scenarios.map((scenario) => {
    const active = scenario.id === STATE.activeScenarioId ? " is-active" : "";
    return `<button type="button" class="scenario-chip${active}" data-id="${escapeHtml(scenario.id)}">
      <i data-lucide="scroll-text"></i>
      <span>${escapeHtml(scenario.title)}</span>
      <small>${escapeHtml(String(scenario.lineCount))} lines</small>
    </button>`;
  }).join("");

  els.scenarioStrip.querySelectorAll(".scenario-chip").forEach((button) => {
    button.addEventListener("click", () => {
      STATE.activeScenarioId = button.dataset.id;
      renderScenarios();
      startStream();
    });
  });

  if (window.lucide) window.lucide.createIcons();
}

function resetOutput() {
  stopStream();
  STATE.rawChars = 0;
  STATE.sharedChars = 0;
  STATE.entityCount = 0;
  els.rawChars.textContent = "0";
  els.sharedChars.textContent = "0";
  els.entityCount.textContent = "0";
  els.rawStream.innerHTML = "";
  els.safeStream.innerHTML = "";
  setLive(false, false);
  setStatus("ready", "Ready");
  els.startButton.querySelector("span").textContent = "Start Stream";
}

function appendRaw(chunk) {
  els.rawStream.textContent += chunk;
  trimOutput(els.rawStream);
  scrollToBottom(els.rawStream);
}

function renderSegments(segments) {
  return segments.map((segment) => {
    if (segment.kind !== "entity") {
      return escapeHtml(segment.text);
    }
    const label = escapeHtml(segment.labelTitle || segment.label || "PII");
    const group = escapeHtml(segment.group || "other");
    const original = escapeHtml(segment.original || "");
    return `<span class="redacted-entity" data-group="${group}" data-tooltip="${label}: ${original}">${escapeHtml(segment.text)}<span>${label}</span></span>`;
  }).join("");
}

function appendSafe(payload) {
  els.safeStream.innerHTML += renderSegments(payload.segments || [{ kind: "plain", text: payload.redacted || "" }]);
  trimOutput(els.safeStream);
  scrollToBottom(els.safeStream);
}

function scrollToBottom(node) {
  window.requestAnimationFrame(() => {
    node.scrollTop = node.scrollHeight;
  });
}

function trimOutput(node) {
  const text = node.textContent || "";
  if (text.length <= MAX_OUTPUT_CHARS) return;
  const trimBy = text.length - MAX_OUTPUT_CHARS;
  if (node === els.rawStream) {
    node.textContent = text.slice(trimBy);
  } else {
    while ((node.textContent || "").length > MAX_OUTPUT_CHARS && node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }
}

function setMode(mode) {
  STATE.mode = mode;
  els.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  startStream();
}

function stopStream() {
  if (STATE.source) {
    STATE.source.close();
    STATE.source = null;
  }
  STATE.running = false;
  setLive(false, false);
}

function startStream() {
  stopStream();
  STATE.rawChars = 0;
  STATE.sharedChars = 0;
  STATE.entityCount = 0;
  els.rawStream.innerHTML = "";
  els.safeStream.innerHTML = "";
  els.rawChars.textContent = "0";
  els.sharedChars.textContent = "0";
  els.entityCount.textContent = "0";

  const params = new URLSearchParams({
    scenario: STATE.activeScenarioId || "",
    mode: STATE.mode,
    delay_ms: els.delayRange.value,
    speed_ms: els.speedRange.value,
  });
  const source = new EventSource(`/api/stream?${params.toString()}`);
  STATE.source = source;
  STATE.running = true;
  els.startButton.querySelector("span").textContent = "Restart";
  setStatus("running", "Streaming");
  setLive(true, false);

  source.addEventListener("meta", () => {
    setStatus("running", "Streaming");
  });

  source.addEventListener("source", (event) => {
    const payload = JSON.parse(event.data);
    appendRaw(payload.chunk || "");
    STATE.rawChars = payload.rawChars || STATE.rawChars;
    els.rawChars.textContent = String(STATE.rawChars);
    setLive(true, false);
  });

  source.addEventListener("redacted", (event) => {
    const payload = JSON.parse(event.data);
    appendSafe(payload);
    STATE.sharedChars = payload.sharedChars || STATE.sharedChars;
    STATE.entityCount = payload.entityCount || STATE.entityCount;
    els.sharedChars.textContent = String(STATE.sharedChars);
    els.entityCount.textContent = String(STATE.entityCount);
    setLive(true, true);
  });

  source.addEventListener("done", (event) => {
    const payload = JSON.parse(event.data);
    els.rawChars.textContent = String(payload.rawChars || STATE.rawChars);
    els.sharedChars.textContent = String(payload.sharedChars || STATE.sharedChars);
    els.entityCount.textContent = String(payload.entityCount || STATE.entityCount);
    setStatus("done", "Complete");
    setLive(false, false);
    stopStream();
  });

  source.onerror = () => {
    if (!STATE.running) return;
    setStatus("error", "Disconnected");
    setLive(false, false);
    stopStream();
  };
}

async function boot() {
  updateRangeLabels();

  els.startButton.addEventListener("click", startStream);
  els.resetButton.addEventListener("click", resetOutput);
  els.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  els.delayRange.addEventListener("input", updateRangeLabels);
  els.speedRange.addEventListener("input", updateRangeLabels);
  els.delayRange.addEventListener("change", startStream);
  els.speedRange.addEventListener("change", startStream);

  try {
    const response = await fetch("/api/scenarios");
    const payload = await response.json();
    STATE.scenarios = payload.scenarios || [];
    STATE.activeScenarioId = payload.defaults?.scenario || STATE.scenarios[0]?.id;
    STATE.mode = payload.defaults?.mode || STATE.mode;
    els.delayRange.value = String(payload.defaults?.delayMs || els.delayRange.value);
    els.speedRange.value = String(payload.defaults?.speedMs || els.speedRange.value);
  } catch (error) {
    console.warn("Failed to load scenarios", error);
    setStatus("error", "Load error");
  }

  updateRangeLabels();
  renderScenarios();
  els.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === STATE.mode);
  });
  if (window.lucide) window.lucide.createIcons();
  if (STATE.scenarios.length > 0) {
    window.setTimeout(startStream, 300);
  }
}

window.addEventListener("beforeunload", stopStream);

boot();
