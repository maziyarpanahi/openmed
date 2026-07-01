import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const TRANSFORMERS_VERSION = "4.2.0";
const MODEL = {
  id: "Xenova/bert-base-NER",
  revision: "24c7e5aba9ae350923357a6f0b92571be34037ec",
  dtype: "q8",
};

const SAMPLES = [
  {
    id: "discharge",
    label: "Discharge note",
    text:
      "Patient Sarah Park visited OpenMed Clinic in Palo Alto on 2026-06-19.\n" +
      "MRN OM-88421. Phone (415) 555-0198. Email sarah.park@example.test.\n" +
      "Follow-up with Dr. Neil Chen at Stanford Health next Friday.",
  },
  {
    id: "referral",
    label: "Referral",
    text:
      "Referral for Miguel Alvarez from North Bay Cardiology in Oakland.\n" +
      "DOB 1979-11-04. Medical Record Number NB-557013.\n" +
      "Contact miguel.alvarez@example.test or 510-555-0144 for scheduling.",
  },
  {
    id: "lab",
    label: "Lab result",
    text:
      "Lab callback for Priya Raman at Harbor Medical Group, Seattle.\n" +
      "Collected 2026-05-28. MRN HM-004219.\n" +
      "Reach the care coordinator at coordinator@example.test.",
  },
];

const LABEL_MAP = new Map([
  ["PER", "PERSON"],
  ["PERSON", "PERSON"],
  ["ORG", "ORGANIZATION"],
  ["LOC", "LOCATION"],
  ["MISC", "MISC"],
]);

const KIND_MAP = new Map([
  ["PERSON", "person"],
  ["ORGANIZATION", "org"],
  ["LOCATION", "location"],
  ["EMAIL", "email"],
  ["PHONE", "phone"],
  ["DATE", "date"],
  ["DATE_OF_BIRTH", "date"],
  ["MEDICAL_RECORD_NUMBER", "id"],
  ["ADDRESS", "address"],
  ["MISC", "misc"],
]);

const els = {
  loadModelButton: document.querySelector("#loadModelButton"),
  runButton: document.querySelector("#runButton"),
  clearButton: document.querySelector("#clearButton"),
  inputText: document.querySelector("#inputText"),
  outputText: document.querySelector("#outputText"),
  sampleButtons: document.querySelector("#sampleButtons"),
  entityRows: document.querySelector("#entityRows"),
  statusDot: document.querySelector("#statusDot"),
  statusLabel: document.querySelector("#statusLabel"),
  statusDetail: document.querySelector("#statusDetail"),
  progressLabel: document.querySelector("#progressLabel"),
  progressFill: document.querySelector("#progressFill"),
  modelValue: document.querySelector("#modelValue"),
  runtimeValue: document.querySelector("#runtimeValue"),
  packageValue: document.querySelector("#packageValue"),
  charCount: document.querySelector("#charCount"),
  wordCount: document.querySelector("#wordCount"),
  entityCount: document.querySelector("#entityCount"),
  latencyValue: document.querySelector("#latencyValue"),
  modeButtons: document.querySelectorAll(".mode-button"),
};

const state = {
  analyzer: null,
  loadingPromise: null,
  busy: false,
  activeSampleId: SAMPLES[0].id,
  mode: "mask",
  entities: [],
  lastText: "",
};

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useBrowserCache = true;
env.cacheKey = "openmed-browser-transformersjs-demo";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "0%";
  return `${Math.max(0, Math.min(100, value)).toFixed(0)}%`;
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatMs(value) {
  if (!Number.isFinite(value)) return "--";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  els.loadModelButton.disabled = isBusy;
  els.runButton.disabled = isBusy;
}

function setStatus(kind, label, detail = "") {
  els.statusDot.className = "status-dot";
  if (kind === "loading") els.statusDot.classList.add("is-loading");
  if (kind === "error") els.statusDot.classList.add("is-error");
  els.statusLabel.textContent = label;
  els.statusDetail.textContent = detail;
}

function setProgress(progress, detail = "") {
  els.progressFill.style.width = formatPercent(progress);
  els.progressLabel.textContent = detail || formatPercent(progress);
}

function handleProgress(info) {
  if (info.status === "progress_total" || info.status === "progress") {
    const progress = Number(info.progress ?? 0);
    const loaded = formatBytes(info.loaded);
    const total = formatBytes(info.total);
    setProgress(progress, loaded && total ? `${formatPercent(progress)} ${loaded}/${total}` : "");
    setStatus("loading", "Loading model assets", info.file || info.name || "");
    return;
  }

  if (info.status === "download" || info.status === "initiate") {
    setStatus("loading", "Preparing model asset", info.file || info.name || "");
  }

  if (info.status === "done") {
    setStatus("loading", "Cached model asset", info.file || "");
  }
}

async function loadAnalyzer() {
  if (state.analyzer) return state.analyzer;
  if (state.loadingPromise) return state.loadingPromise;

  setBusy(true);
  setProgress(0);
  setStatus("loading", "Loading model", MODEL.id);

  state.loadingPromise = pipeline("token-classification", MODEL.id, {
    dtype: MODEL.dtype,
    revision: MODEL.revision,
    progress_callback: handleProgress,
  })
    .then((analyzer) => {
      state.analyzer = analyzer;
      setProgress(100, "100%");
      setStatus(
        "ready",
        "Model ready",
        "Inference and deterministic matching run locally in this browser.",
      );
      return analyzer;
    })
    .catch((error) => {
      state.loadingPromise = null;
      setStatus("error", "Model load failed", error.message);
      renderError(error);
      throw error;
    })
    .finally(() => {
      setBusy(false);
    });

  return state.loadingPromise;
}

function renderSamples() {
  els.sampleButtons.innerHTML = SAMPLES.map((sample) => {
    const active = sample.id === state.activeSampleId ? " is-active" : "";
    return `<button class="${active}" data-sample="${sample.id}" type="button">
      ${escapeHtml(sample.label)}
    </button>`;
  }).join("");

  els.sampleButtons.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      const sample = SAMPLES.find((item) => item.id === button.dataset.sample);
      if (!sample) return;
      state.activeSampleId = sample.id;
      els.inputText.value = sample.text;
      state.entities = [];
      state.lastText = "";
      renderSamples();
      renderOutput();
      updateCounters();
      setStatus("ready", "Ready", "Synthetic sample loaded.");
    });
  });
}

function updateCounters() {
  const text = els.inputText.value;
  els.charCount.textContent = String(text.length);
  els.wordCount.textContent = String((text.match(/\S+/g) || []).length);
}

function normalizeModelLabel(rawLabel) {
  const stripped = String(rawLabel || "")
    .replace(/^[BI]-/, "")
    .toUpperCase();
  return LABEL_MAP.get(stripped) || stripped || "MISC";
}

function findWordSpan(text, word, fromIndex) {
  const cleaned = String(word || "")
    .replace(/\s*##\s*/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) return null;

  const exact = text.indexOf(cleaned, fromIndex);
  if (exact >= 0) {
    return { start: exact, end: exact + cleaned.length };
  }

  const lowerText = text.toLowerCase();
  const lowerWord = cleaned.toLowerCase();
  const folded = lowerText.indexOf(lowerWord, fromIndex);
  if (folded >= 0) {
    return { start: folded, end: folded + cleaned.length };
  }

  return null;
}

function normalizeTransformerEntity(item, text, fromIndex) {
  let start = Number(item.start);
  let end = Number(item.end);
  const score = Number(item.score ?? 0);

  if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) {
    const span = findWordSpan(text, item.word, fromIndex);
    if (!span) return null;
    start = span.start;
    end = span.end;
  }

  if (score < 0.45) return null;

  return {
    start,
    end,
    label: normalizeModelLabel(item.entity_group || item.entity || item.label),
    source: "transformer",
    score,
    priority: 50,
  };
}

function normalizeTransformerEntities(raw, text) {
  let cursor = 0;
  const entities = [];

  for (const item of raw) {
    const entity = normalizeTransformerEntity(item, text, cursor);
    if (!entity) continue;
    entities.push(entity);
    cursor = entity.end;
  }

  return entities;
}

function regexEntities(text) {
  const patterns = [
    {
      label: "EMAIL",
      priority: 90,
      pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,
    },
    {
      label: "PHONE",
      priority: 90,
      pattern:
        /(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b/g,
    },
    {
      label: "DATE_OF_BIRTH",
      priority: 95,
      pattern:
        /\b(?:DOB|date of birth)\s*[:#-]?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{2,4}|[A-Z][a-z]+ \d{1,2}, \d{4})\b/g,
    },
    {
      label: "MEDICAL_RECORD_NUMBER",
      priority: 95,
      pattern:
        /\b(?:MRN|Medical Record(?: Number)?)\s*[:#-]?\s*[A-Z]{0,4}-?\d{4,}\b/gi,
    },
    {
      label: "DATE",
      priority: 70,
      pattern:
        /\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}\/\d{1,2}\/(?:19|20)?\d{2}\b/g,
    },
  ];

  return patterns.flatMap(({ label, priority, pattern }) => {
    pattern.lastIndex = 0;
    return Array.from(text.matchAll(pattern)).map((match) => ({
      start: match.index,
      end: match.index + match[0].length,
      label,
      source: "deterministic",
      score: 1,
      priority,
    }));
  });
}

function overlaps(left, right) {
  return left.start < right.end && right.start < left.end;
}

function mergeEntities(entities) {
  const cleaned = entities
    .filter((entity) => entity && entity.end > entity.start)
    .sort((a, b) => {
      const lengthA = a.end - a.start;
      const lengthB = b.end - b.start;
      return b.priority - a.priority || lengthB - lengthA || a.start - b.start;
    });

  const selected = [];
  for (const entity of cleaned) {
    if (!selected.some((existing) => overlaps(entity, existing))) {
      selected.push(entity);
    }
  }

  return selected.sort((a, b) => a.start - b.start || a.end - b.end);
}

async function runAnalysis() {
  const text = els.inputText.value.trim();
  if (!text) {
    state.entities = [];
    state.lastText = "";
    renderOutput();
    setStatus("error", "No input text", "Load a sample or enter synthetic text.");
    return;
  }

  setBusy(true);
  setStatus("loading", "Running analysis", "Token classification in progress.");
  try {
    const analyzer = await loadAnalyzer();
    const startedAt = performance.now();
    const raw = await analyzer(text, { aggregation_strategy: "simple" });
    const latency = performance.now() - startedAt;
    const modelEntities = normalizeTransformerEntities(raw, text);
    state.entities = mergeEntities([...regexEntities(text), ...modelEntities]);
    state.lastText = text;
    renderOutput(latency);
    setStatus(
      "ready",
      "Analysis complete",
      `${state.entities.length} entities found in ${formatMs(latency)}.`,
    );
  } catch (error) {
    renderError(error);
  } finally {
    setBusy(false);
  }
}

function kindForLabel(label) {
  return KIND_MAP.get(label) || "misc";
}

function labelText(label) {
  return String(label).replaceAll("_", " ");
}

function renderRedactedText(text, entities, mode) {
  if (!entities.length) {
    return `<p class="empty-state">No supported entities detected.</p>`;
  }

  let html = "";
  let cursor = 0;

  for (const entity of entities) {
    html += escapeHtml(text.slice(cursor, entity.start));
    const original = text.slice(entity.start, entity.end);
    const label = labelText(entity.label);
    const kind = kindForLabel(entity.label);
    if (mode === "review") {
      html += `<span class="review-token" title="${escapeHtml(label)}">${escapeHtml(original)}</span>`;
    } else {
      html += `<span class="entity-token" data-kind="${kind}" title="${escapeHtml(original)}">[${escapeHtml(label)}]</span>`;
    }
    cursor = entity.end;
  }

  html += escapeHtml(text.slice(cursor));
  return html;
}

function renderRows(text, entities) {
  if (!entities.length) {
    els.entityRows.innerHTML = `
      <tr>
        <td colspan="5" class="empty-row">No entities detected.</td>
      </tr>`;
    return;
  }

  els.entityRows.innerHTML = entities
    .map((entity) => {
      const source = entity.source;
      const confidence =
        source === "deterministic" ? "rule" : `${(entity.score * 100).toFixed(1)}%`;
      return `<tr>
        <td><span class="label-pill">${escapeHtml(labelText(entity.label))}</span></td>
        <td class="entity-text">${escapeHtml(text.slice(entity.start, entity.end))}</td>
        <td><span class="source-pill" data-source="${escapeHtml(source)}">${escapeHtml(source)}</span></td>
        <td>${escapeHtml(confidence)}</td>
        <td><code>${entity.start}-${entity.end}</code></td>
      </tr>`;
    })
    .join("");
}

function renderOutput(latency) {
  const text = state.lastText || els.inputText.value;
  els.outputText.innerHTML = state.lastText
    ? renderRedactedText(text, state.entities, state.mode)
    : `<p class="empty-state">Run analysis to create a browser-local redaction preview.</p>`;
  els.entityCount.textContent = String(state.entities.length);
  els.latencyValue.textContent = latency === undefined ? "--" : formatMs(latency);
  renderRows(text, state.entities);
}

function renderError(error) {
  els.outputText.innerHTML = `
    <p class="error-state">${escapeHtml(error.message || "Analysis failed.")}</p>`;
}

function init() {
  els.modelValue.textContent = `${MODEL.id} @ ${MODEL.revision.slice(0, 8)}`;
  els.packageValue.textContent = `@huggingface/transformers ${TRANSFORMERS_VERSION}`;
  els.runtimeValue.textContent = navigator.gpu ? "Browser WASM; WebGPU available" : "Browser WASM";

  renderSamples();
  els.inputText.value = SAMPLES[0].text;
  updateCounters();
  renderOutput();

  els.loadModelButton.addEventListener("click", () => {
    loadAnalyzer().catch(() => {});
  });
  els.runButton.addEventListener("click", runAnalysis);
  els.clearButton.addEventListener("click", () => {
    els.inputText.value = "";
    state.entities = [];
    state.lastText = "";
    renderOutput();
    updateCounters();
    setStatus("ready", "Ready", "Input cleared.");
  });
  els.inputText.addEventListener("input", () => {
    state.lastText = "";
    state.entities = [];
    updateCounters();
    renderOutput();
  });
  els.inputText.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runAnalysis();
    }
  });

  els.modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      els.modeButtons.forEach((item) => {
        item.classList.toggle("is-active", item === button);
      });
      renderOutput();
    });
  });
}

window.addEventListener("pagehide", () => {
  if (state.analyzer?.dispose) {
    state.analyzer.dispose();
  }
});

init();
