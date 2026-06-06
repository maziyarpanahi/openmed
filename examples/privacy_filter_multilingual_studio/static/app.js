// OpenMed Privacy Filter — Multilingual Comparison Studio
// Two-model side-by-side comparison with 16 language tabs.

const STATE = {
  mode: "mask",
  examples: {}, // { lang: [{id,title,blurb,text}, ...] }
  languages: [],
  models: {},
  activeLang: "en",
  activeExampleId: null,
  lastResult: null,
  download: false,
  inputRevision: 0,
  runToken: 0,
};

const PALETTE = {
  // Identity / contact / address / dates / financial / digital / etc.
  // Map upper-cased label fragments to a colour role; unmatched falls back to "default".
  default: { role: "default" },
  identity: ["FIRSTNAME", "MIDDLENAME", "LASTNAME", "PREFIX", "USERNAME", "USER_NAME", "AGE", "GENDER", "SEX", "EYECOLOR", "HEIGHT", "OCCUPATION", "JOBTITLE", "JOBDEPARTMENT", "ORGANIZATION", "USERAGENT", "PRIVATE_PERSON", "FIRST_NAME", "LAST_NAME"],
  contact: ["EMAIL", "PHONE", "URL", "PHONE_NUMBER", "PRIVATE_EMAIL", "PRIVATE_PHONE_NUMBER"],
  address: ["STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "COUNTY", "STATE", "ZIPCODE", "GPSCOORDINATES", "ORDINALDIRECTION", "STREET_ADDRESS"],
  date: ["DATE", "DATEOFBIRTH", "TIME", "DATE_TIME", "DATE_OF_BIRTH"],
  govid: ["SSN"],
  financial: ["ACCOUNTNAME", "BANKACCOUNT", "IBAN", "BIC", "CREDITCARD", "CREDITCARDISSUER", "CVV", "PIN", "MASKEDNUMBER", "AMOUNT", "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL", "PRIVATE_FINANCE"],
  crypto: ["BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS"],
  vehicle: ["VIN", "VRM"],
  digital: ["IPADDRESS", "MACADDRESS", "IMEI", "PRIVATE_IDENTIFIER", "DEVICE_IDENTIFIER"],
  auth: ["PASSWORD"],
};

function roleForLabel(rawLabel) {
  const L = String(rawLabel || "").toUpperCase();
  for (const [role, list] of Object.entries(PALETTE)) {
    if (Array.isArray(list) && list.includes(L)) return role;
  }
  // Heuristic fallback for upstream's lowercase coarse labels.
  if (L.includes("EMAIL")) return "contact";
  if (L.includes("PHONE") || L.includes("FAX")) return "contact";
  if (L.includes("ADDRESS")) return "address";
  if (L.includes("PERSON") || L.includes("NAME")) return "identity";
  if (L.includes("DATE") || L.includes("TIME")) return "date";
  if (L.includes("SSN") || L.includes("ID")) return "govid";
  if (L.includes("FINANCE") || L.includes("CARD") || L.includes("BANK")) return "financial";
  if (L.includes("IP") || L.includes("MAC")) return "digital";
  return "default";
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) window.lucide.createIcons();
  attachToolbarHandlers();
  attachEditorHandlers();
  loadExamples();
});

function attachToolbarHandlers() {
  document.querySelectorAll(".mode-button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-button").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      STATE.mode = btn.dataset.mode || "mask";
      if (STATE.lastResult) renderBothPanes(STATE.lastResult);
    });
  });

  document.getElementById("themeToggle").addEventListener("click", () => {
    const root = document.documentElement;
    root.classList.toggle("theme-dark");
    if (window.lucide) window.lucide.createIcons();
  });

  document.getElementById("allowDownloads").addEventListener("change", (e) => {
    STATE.download = !!e.target.checked;
  });

  document.getElementById("runButton").addEventListener("click", runComparison);

  document.getElementById("clearButton").addEventListener("click", () => {
    document.getElementById("inputText").value = "";
    updateInputMeta();
    STATE.activeExampleId = null;
    syncChipSelection();
    invalidateCurrentResults("Pick a language and an example to begin.");
  });
}

function attachEditorHandlers() {
  document.getElementById("inputText").addEventListener("input", () => {
    updateInputMeta();
    STATE.activeExampleId = null;
    syncChipSelection();
    invalidateCurrentResults("Input changed. Run Comparison to refresh both model windows.");
  });
}

async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const data = await res.json();
    STATE.examples = data.examples || {};
    STATE.languages = data.languages || [];
    STATE.models = data.models || {};
    document.getElementById("openmedModelLabel").textContent = STATE.models?.openmed?.label || "OpenMed";
    document.getElementById("baselineModelLabel").textContent = STATE.models?.baseline?.label || "Baseline";
    renderLanguageTabs();
    selectLanguage(STATE.activeLang);
  } catch (err) {
    console.error("Failed to load examples", err);
    setComparisonNote(`Failed to load examples: ${err.message || err}`);
  }
}

// ---------------------------------------------------------------------------
// Language tabs + example chips
// ---------------------------------------------------------------------------

function renderLanguageTabs() {
  const host = document.getElementById("langTabs");
  host.innerHTML = "";
  STATE.languages.forEach((lang) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lang-tab";
    btn.dataset.lang = lang.code;
    btn.setAttribute("role", "tab");
    btn.innerHTML = `
      <span class="lang-code">${lang.code}</span>
      <span class="lang-native">${lang.native}</span>
    `;
    btn.addEventListener("click", () => selectLanguage(lang.code));
    host.appendChild(btn);
  });
}

function selectLanguage(code) {
  STATE.activeLang = code;
  document.querySelectorAll(".lang-tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.lang === code);
  });
  renderExampleChips();
  // Auto-load the first example for the language for instant feedback.
  const examples = STATE.examples[code] || [];
  if (examples.length > 0) loadExample(examples[0]);
}

function renderExampleChips() {
  const host = document.getElementById("exampleChips");
  host.innerHTML = "";
  const examples = STATE.examples[STATE.activeLang] || [];
  examples.forEach((ex) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.dataset.id = ex.id;
    chip.innerHTML = `
      <span class="chip-title">${escapeHtml(ex.title)}</span>
      <span class="chip-blurb">${escapeHtml(ex.blurb)}</span>
    `;
    chip.addEventListener("click", () => loadExample(ex));
    host.appendChild(chip);
  });
  syncChipSelection();
}

function loadExample(example) {
  document.getElementById("inputText").value = example.text;
  STATE.activeExampleId = example.id;
  updateInputMeta();
  syncChipSelection();
  invalidateCurrentResults("Loaded a new sample. Run Comparison to populate both model windows.");
}

function syncChipSelection() {
  document.querySelectorAll(".example-chip").forEach((chip) => {
    chip.classList.toggle("is-active", chip.dataset.id === STATE.activeExampleId);
  });
}

// ---------------------------------------------------------------------------
// Run inference + render
// ---------------------------------------------------------------------------

async function runComparison() {
  const text = document.getElementById("inputText").value.trim();
  if (!text) {
    setComparisonNote("Empty input — pick an example or paste text.");
    return;
  }

  const runToken = STATE.runToken + 1;
  STATE.runToken = runToken;
  const inputRevision = STATE.inputRevision;
  setBusy(true);
  setComparisonNote("Running both models…");

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        mode: STATE.mode,
        seed: 42,
        download: STATE.download,
      }),
    });
    const data = await res.json();
    if (runToken !== STATE.runToken || inputRevision !== STATE.inputRevision) return;
    STATE.lastResult = data;
    if (data.status === "live") {
      setComparisonNote("Redacting line by line…");
      await animateBothPanes(data, inputRevision);
    } else {
      renderBothPanes(data);
    }
    if (runToken !== STATE.runToken || inputRevision !== STATE.inputRevision) return;
    setComparisonNote(data.note || "");
  } catch (err) {
    console.error(err);
    if (runToken !== STATE.runToken || inputRevision !== STATE.inputRevision) return;
    setComparisonNote(`Error: ${err.message || err}`);
  } finally {
    if (runToken === STATE.runToken) setBusy(false);
  }
}

function renderBothPanes(data) {
  renderSide("openmed", data.openmed, data.status);
  renderSide("baseline", data.baseline, data.status);
}

function renderSide(side, payload, status) {
  if (!payload) return;
  const resultHost = document.getElementById(`${side}ResultText`);
  const entityCount = document.getElementById(`${side}EntityCount`);
  const latency = document.getElementById(`${side}Latency`);
  const statusPill = document.getElementById(`${side}StatusPill`);

  const text = STATE.mode === "randomize" ? payload.randomized : payload.masked;
  const baseText = document.getElementById("inputText").value;
  const fragment = renderHighlighted(baseText, payload.entities, STATE.mode);
  resultHost.innerHTML = "";
  resultHost.appendChild(fragment);

  entityCount.textContent = payload.stats?.entities ?? 0;
  latency.textContent = `${payload.stats?.inferenceMs ?? "--"} ms`;
  statusPill.textContent = status === "live" ? "Live" : status === "error" ? "Error" : "Ready";
  statusPill.dataset.state = status || "ready";
}

async function animateBothPanes(data, inputRevision) {
  const baseText = document.getElementById("inputText").value;
  const ranges = chunkRanges(baseText);
  prepareSideForReveal("openmed", data.openmed, data.status, baseText, ranges);
  prepareSideForReveal("baseline", data.baseline, data.status, baseText, ranges);
  beginScan();
  await Promise.all([
    revealSide("openmed", data.openmed, baseText, ranges, inputRevision),
    revealSide("baseline", data.baseline, baseText, ranges, inputRevision),
  ]);
  if (inputRevision !== STATE.inputRevision) return;
  endScan();
}

function prepareSideForReveal(side, payload, status, text, ranges) {
  if (!payload) return;
  const resultHost = document.getElementById(`${side}ResultText`);
  const entityCount = document.getElementById(`${side}EntityCount`);
  const latency = document.getElementById(`${side}Latency`);
  const statusPill = document.getElementById(`${side}StatusPill`);

  resultHost.innerHTML = "";
  ranges.forEach((range, index) => {
    const span = document.createElement("span");
    span.className = "redaction-line";
    span.dataset.index = String(index);
    span.textContent = text.slice(range.start, range.end);
    resultHost.appendChild(span);
  });

  entityCount.textContent = payload.stats?.entities ?? 0;
  latency.textContent = `${payload.stats?.inferenceMs ?? "--"} ms`;
  statusPill.textContent = "Scanning";
  statusPill.dataset.state = "running";
}

async function revealSide(side, payload, text, ranges, inputRevision) {
  if (!payload) return;
  const resultHost = document.getElementById(`${side}ResultText`);
  const entities = payload.entities || [];
  const delay = Math.max(42, Math.min(120, Math.round(1100 / Math.max(ranges.length, 1))));

  for (let i = 0; i < ranges.length; i += 1) {
    if (inputRevision !== STATE.inputRevision) return;
    const line = resultHost.querySelector(`.redaction-line[data-index="${i}"]`);
    if (!line) continue;
    line.innerHTML = "";
    line.appendChild(renderHighlightedRange(text, entities, STATE.mode, ranges[i].start, ranges[i].end));
    line.classList.add("is-redacted");
    await sleep(delay);
  }

  if (inputRevision !== STATE.inputRevision) return;
  const statusPill = document.getElementById(`${side}StatusPill`);
  statusPill.textContent = "Live";
  statusPill.dataset.state = "live";
}

function renderHighlightedRange(text, entities, mode, start, end) {
  const frag = document.createDocumentFragment();
  const scoped = (entities || [])
    .filter((ent) => ent.start >= start && ent.end <= end)
    .map((ent) => ({ ...ent, start: ent.start - start, end: ent.end - start }));
  frag.appendChild(renderHighlighted(text.slice(start, end), scoped, mode));
  return frag;
}

function chunkRanges(text) {
  if (!text) return [{ start: 0, end: 0 }];
  const ranges = [];
  const pattern = /[^\n.!?؟。]+(?:[.!?؟。]+|\n+|$)|\n+/g;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    ranges.push({ start: match.index, end: match.index + match[0].length });
  }
  if (ranges.length === 0) ranges.push({ start: 0, end: text.length });
  return ranges;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function renderHighlighted(text, entities, mode) {
  // Build a DOM fragment that shows the original text with each entity span
  // replaced by either a [LABEL] pill (mask) or its surrogate (randomize).
  const frag = document.createDocumentFragment();
  if (!entities || entities.length === 0) {
    const span = document.createElement("span");
    span.textContent = text;
    frag.appendChild(span);
    return frag;
  }
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  let cursor = 0;
  for (const ent of sorted) {
    if (ent.start < cursor) continue; // skip overlap
    if (ent.start > cursor) {
      frag.appendChild(document.createTextNode(text.slice(cursor, ent.start)));
    }
    const pill = document.createElement("span");
    pill.className = `pii-pill role-${roleForLabel(ent.label)}`;
    pill.dataset.label = ent.label;
    if (mode === "randomize" && ent.surrogate) {
      pill.textContent = ent.surrogate;
    } else {
      pill.textContent = `[${String(ent.label).toUpperCase()}]`;
    }
    pill.title = `${ent.label} · ${ent.text}`;
    frag.appendChild(pill);
    cursor = ent.end;
  }
  if (cursor < text.length) {
    frag.appendChild(document.createTextNode(text.slice(cursor)));
  }
  return frag;
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function updateInputMeta() {
  const text = document.getElementById("inputText").value;
  document.getElementById("charCount").textContent = text.length;
  document.getElementById("tokenCount").textContent = (text.match(/\S+/g) || []).length;
}

function setBusy(busy) {
  const btn = document.getElementById("runButton");
  btn.disabled = !!busy;
  btn.classList.toggle("is-busy", !!busy);
}

function invalidateCurrentResults(note) {
  STATE.inputRevision += 1;
  STATE.runToken += 1;
  STATE.lastResult = null;
  endScan();
  setBusy(false);
  clearResultPanes();
  setComparisonNote(note);
}

function clearResultPanes() {
  ["openmed", "baseline"].forEach((side) => {
    const resultHost = document.getElementById(`${side}ResultText`);
    const entityCount = document.getElementById(`${side}EntityCount`);
    const latency = document.getElementById(`${side}Latency`);
    const statusPill = document.getElementById(`${side}StatusPill`);
    resultHost.innerHTML = '<p class="result-empty">Press <strong>Run Comparison</strong> to populate this side.</p>';
    entityCount.textContent = "0";
    latency.textContent = "--";
    statusPill.textContent = "Ready";
    statusPill.dataset.state = "ready";
  });
}

function beginScan() {
  document.querySelectorAll(".paper-frame--result").forEach((frame) => {
    const el = frame.querySelector(".scan-line");
    if (!el) return;
    el.classList.remove("is-scanning");
    el.style.setProperty("--scan-distance", `${Math.max(frame.clientHeight - 3, 220)}px`);
    void el.offsetWidth;
    el.classList.add("is-scanning");
  });
}
function endScan() {
  document.querySelectorAll(".paper-frame--result .scan-line").forEach((el) => {
    el.classList.remove("is-scanning");
  });
}

function setComparisonNote(text) {
  document.getElementById("comparisonNote").textContent = text || "";
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
