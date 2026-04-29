// OpenMed Privacy Filter Studio — frontend.

const STATE = {
  examples: [],
  activeExampleId: null,
  busy: false,
  mode: "mask",
  theme: "dark",
  modelId: "--",
  lastResult: null,
};

const els = {
  inputText:       document.querySelector("#inputText"),
  resultText:      document.querySelector("#resultText"),
  exampleChips:    document.querySelector("#exampleChips"),
  runButton:       document.querySelector("#runButton"),
  themeToggle:     document.querySelector("#themeToggle"),
  allowDownloads:  document.querySelector("#allowDownloads"),
  clearButton:     document.querySelector("#clearButton"),
  charCount:       document.querySelector("#charCount"),
  tokenCount:      document.querySelector("#tokenCount"),
  entityCount:     document.querySelector("#entityCount"),
  latencyLabel:    document.querySelector("#latencyLabel"),
  modelLabel:      document.querySelector("#modelLabel"),
  backendLabel:    document.querySelector("#backendLabel"),
  statusPill:      document.querySelector("#statusPill"),
  legendList:      document.querySelector("#legendList"),
  modeButtons:     document.querySelectorAll(".mode-button"),
};

// ---------------------------------------------------------------------------
// Label → color group mapping. Mirrors the palette in styles.css.
// Covers the 55 fine-grained Nemotron-PII labels plus the OpenAI baseline's
// 8 coarse classes; anything unrecognised falls through to "other".
// ---------------------------------------------------------------------------

const LABEL_GROUPS = {
  // identity (warm red/coral)
  first_name: "identity", last_name: "identity", user_name: "identity",
  age: "identity", gender: "identity", marital_status: "identity",
  nationality: "identity", language: "identity", biometric_identifier: "identity",
  blood_type: "identity", private_person: "identity", name: "identity",
  patient: "identity", doctor: "identity",

  // contact (blue)
  email: "contact", phone_number: "contact", fax_number: "contact",
  url: "contact", private_email: "contact", private_phone: "contact",
  private_url: "contact",

  // address (green)
  street_address: "address", city: "address", county: "address",
  state: "address", country: "address", postcode: "address",
  coordinate: "address", private_address: "address",

  // dates / time (purple)
  date: "dates", date_of_birth: "dates", date_time: "dates", time: "dates",
  private_date: "dates",

  // government IDs (magenta-red)
  ssn: "govid", national_id: "govid", tax_id: "govid",
  certificate_license_number: "govid", passport_number: "govid",

  // financial (gold)
  account_number: "financial", bank_routing_number: "financial",
  swift_bic: "financial", credit_debit_card: "financial",
  cvv: "financial", pin: "financial", password: "financial",
  account: "financial",

  // healthcare (teal)
  medical_record_number: "healthcare",
  health_plan_beneficiary_number: "healthcare",

  // enterprise / workplace (slate)
  customer_id: "workplace", employee_id: "workplace", unique_id: "workplace",
  company_name: "workplace", occupation: "workplace",
  employment_status: "workplace", education_level: "workplace",

  // online (orange)
  ipv4: "online", ipv6: "online", mac_address: "online",
  device_identifier: "online", api_key: "online", http_cookie: "online",
  secret: "online",

  // demographic (magenta)
  race_ethnicity: "demographic", religious_belief: "demographic",
  political_view: "demographic", sexuality: "demographic",

  // vehicle (orange)
  license_plate: "vehicle", vehicle_identifier: "vehicle",
};

function groupForLabel(label) {
  if (!label) return "other";
  const norm = String(label).toLowerCase().replace(/[\s-]/g, "_");
  return LABEL_GROUPS[norm] || "other";
}

function prettyLabel(label) {
  return String(label || "").replace(/_/g, " ");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function getSavedTheme() {
  try {
    const saved = window.localStorage.getItem("privacy-filter-studio-theme");
    return saved === "light" || saved === "dark" ? saved : "dark";
  } catch {
    return "dark";
  }
}

function setTheme(theme) {
  STATE.theme = theme;
  document.documentElement.dataset.theme = theme;
  try { window.localStorage.setItem("privacy-filter-studio-theme", theme); } catch {}
  const next = theme === "dark" ? "light" : "dark";
  els.themeToggle.setAttribute("aria-label", `Switch to ${next} theme`);
  els.themeToggle.setAttribute("title", `Switch to ${next} theme`);
  els.themeToggle.innerHTML = `<i data-lucide="${theme === "dark" ? "sun" : "moon"}"></i>`;
  if (window.lucide) window.lucide.createIcons();
}

function toggleTheme() {
  setTheme(STATE.theme === "dark" ? "light" : "dark");
}

// ---------------------------------------------------------------------------
// Examples strip
// ---------------------------------------------------------------------------

function renderExampleChips() {
  els.exampleChips.innerHTML = STATE.examples
    .map((ex) => {
      const tooltip = escapeHtml(ex.blurb || "");
      const cls = ex.id === STATE.activeExampleId ? "example-chip is-active" : "example-chip";
      return `<button type="button" class="${cls}" data-id="${escapeHtml(ex.id)}" data-tooltip="${tooltip}">
        <i data-lucide="file-text"></i>
        <span>${escapeHtml(ex.title)}</span>
      </button>`;
    })
    .join("");

  els.exampleChips.querySelectorAll("button.example-chip").forEach((btn) => {
    btn.addEventListener("click", () => loadExample(btn.dataset.id));
  });

  if (window.lucide) window.lucide.createIcons();
}

function loadExample(id) {
  const example = STATE.examples.find((ex) => ex.id === id);
  if (!example) return;
  STATE.activeExampleId = id;
  els.inputText.value = example.text;
  updateInputCounters();
  renderExampleChips();
  // Reset output so user sees the impact of the next run
  STATE.lastResult = null;
  renderResult(null);
  setStatus("ready", "Ready");
}

// ---------------------------------------------------------------------------
// Input counters
// ---------------------------------------------------------------------------

function updateInputCounters() {
  const text = els.inputText.value;
  els.charCount.textContent = String(text.length);
  els.tokenCount.textContent = String((text.match(/\S+/g) || []).length);
}

// ---------------------------------------------------------------------------
// Status pill
// ---------------------------------------------------------------------------

function setStatus(kind, label) {
  els.statusPill.className = "status-pill";
  if (kind === "running") els.statusPill.classList.add("running");
  else if (kind === "live") els.statusPill.classList.add("live");
  else if (kind === "error") els.statusPill.classList.add("error");
  els.statusPill.textContent = label;
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------

function renderResult(result) {
  if (!result) {
    els.resultText.innerHTML = `
      <p class="placeholder">
        Press <kbd>⌘</kbd> + <kbd>Enter</kbd> or click
        <strong>Run De-identification</strong> to redact entities.
        Hover over a tag in the output to see the original value.
      </p>`;
    els.entityCount.textContent = "0";
    els.latencyLabel.textContent = "--";
    els.legendList.innerHTML = "";
    return;
  }

  const text = els.inputText.value;
  const entities = result.entities || [];

  if (result.status === "error") {
    els.resultText.innerHTML = `
      <p class="error-note">${escapeHtml(result.note || "Something went wrong.")}</p>`;
    els.entityCount.textContent = "0";
    els.latencyLabel.textContent = "--";
    els.legendList.innerHTML = "";
    return;
  }

  if (entities.length === 0) {
    els.resultText.innerHTML = `
      <p class="empty-note">No PII detected in this text.</p>`;
    els.entityCount.textContent = "0";
    els.latencyLabel.textContent = formatMs(result.stats?.inferenceMs);
    els.legendList.innerHTML = "";
    return;
  }

  els.resultText.innerHTML = renderEntities(text, entities, STATE.mode);
  els.entityCount.textContent = String(entities.length);
  els.latencyLabel.textContent = formatMs(result.stats?.inferenceMs);
  renderLegend(entities);
}

function renderEntities(text, entities, mode) {
  const sorted = [...entities].sort((a, b) => a.start - b.start || b.end - a.end);
  let cursor = 0;
  let html = "";

  for (const ent of sorted) {
    if (ent.start < cursor || ent.end <= ent.start) continue;
    html += escapeHtml(text.slice(cursor, ent.start));
    const original = text.slice(ent.start, ent.end);
    const group = groupForLabel(ent.label);
    const tagText = prettyLabel(ent.label);
    const tooltip = mode === "mask"
      ? `${tagText}: "${original}"`
      : `${tagText}: "${original}" → "${ent.surrogate || ""}"`;

    let display;
    if (mode === "mask") {
      display = `<span class="entity-mask">[${tagText.toUpperCase()}]</span>`;
    } else {
      display = escapeHtml(ent.surrogate || original);
    }

    html += `<span class="entity" data-group="${group}" data-tooltip="${escapeHtml(tooltip)}" tabindex="0">`
         + display
         + `<span class="entity-tag">${escapeHtml(tagText)}</span>`
         + `</span>`;
    cursor = ent.end;
  }

  html += escapeHtml(text.slice(cursor));
  return html;
}

function renderLegend(entities) {
  const counts = new Map();
  for (const ent of entities) {
    const group = groupForLabel(ent.label);
    counts.set(group, (counts.get(group) || 0) + 1);
  }
  const order = [
    "identity", "contact", "address", "dates", "govid", "financial",
    "healthcare", "workplace", "online", "demographic", "vehicle", "other",
  ];
  els.legendList.innerHTML = order
    .filter((g) => counts.has(g))
    .map((g) => {
      const label = g.charAt(0).toUpperCase() + g.slice(1);
      return `<li class="legend-item" data-group="${g}">
        <span>${label}</span>
        <span class="legend-count">${counts.get(g)}</span>
      </li>`;
    })
    .join("");
}

function formatMs(ms) {
  if (!Number.isFinite(ms)) return "--";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// ---------------------------------------------------------------------------
// Mode switch
// ---------------------------------------------------------------------------

function setMode(mode) {
  if (STATE.mode === mode) return;
  STATE.mode = mode;
  els.modeButtons.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.mode === mode);
  });
  // Re-render last result with the new mode (if any)
  if (STATE.lastResult) renderResult(STATE.lastResult);
}

// ---------------------------------------------------------------------------
// Run de-identification
// ---------------------------------------------------------------------------

function setBusy(isBusy) {
  STATE.busy = isBusy;
  els.runButton.disabled = isBusy;
  els.runButton.querySelector("span").textContent =
    isBusy ? "Running" : "Run De-identification";
}

function startScan() {
  const targets = document.querySelectorAll(".paper-frame");
  targets.forEach((node) => {
    node.classList.remove("scan-active");
    void node.offsetWidth;  // restart animation
    node.classList.add("scan-active");
  });
}

async function runDeidentification() {
  if (STATE.busy) return;
  const text = els.inputText.value.trim();
  if (!text) {
    setStatus("error", "No input");
    return;
  }

  setBusy(true);
  setStatus("running", "Running");
  startScan();

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: els.inputText.value,
        mode: STATE.mode,
        seed: 42,
        download: els.allowDownloads.checked,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    STATE.lastResult = payload;
    if (payload.modelId) {
      els.modelLabel.textContent = payload.modelId;
      STATE.modelId = payload.modelId;
    }
    if (payload.backend) {
      els.backendLabel.textContent = payload.backend.toUpperCase();
    }

    // Wait for the scan-line animation to feel complete before revealing
    window.setTimeout(() => {
      if (payload.status === "live") {
        renderResult(payload);
        setStatus("live", "Live");
      } else if (payload.status === "empty") {
        renderResult(null);
        setStatus("ready", "Ready");
      } else {
        renderResult(payload);
        setStatus("error", "Error");
      }
      setBusy(false);
    }, 760);
  } catch (error) {
    console.error(error);
    setStatus("error", "Error");
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  setTheme(getSavedTheme());

  els.inputText.addEventListener("input", () => {
    updateInputCounters();
    if (STATE.activeExampleId) {
      STATE.activeExampleId = null;
      renderExampleChips();
    }
  });
  els.clearButton.addEventListener("click", () => {
    els.inputText.value = "";
    STATE.activeExampleId = null;
    STATE.lastResult = null;
    updateInputCounters();
    renderExampleChips();
    renderResult(null);
    setStatus("ready", "Ready");
    els.inputText.focus();
  });
  els.runButton.addEventListener("click", runDeidentification);
  els.themeToggle.addEventListener("click", toggleTheme);
  els.modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runDeidentification();
    }
  });

  try {
    const response = await fetch("/api/examples");
    const payload = await response.json();
    STATE.examples = payload.examples || [];
    STATE.modelId = payload.model;
    els.modelLabel.textContent = payload.model || "--";
  } catch (e) {
    console.warn("Failed to load examples", e);
  }

  renderExampleChips();
  if (STATE.examples.length > 0) {
    loadExample(STATE.examples[0].id);
  } else {
    updateInputCounters();
  }

  if (window.lucide) window.lucide.createIcons();
}

boot();
