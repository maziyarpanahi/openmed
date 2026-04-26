const state = {
  documents: [],
  current: 0,
  latestResults: null,
  redacted: false,
  busy: false,
};

const els = {
  book: document.querySelector("#book"),
  prev: document.querySelector("#prevPage"),
  next: document.querySelector("#nextPage"),
  run: document.querySelector("#runButton"),
  allowDownloads: document.querySelector("#allowDownloads"),
  folioNow: document.querySelector("#folioNow"),
  folioTotal: document.querySelector("#folioTotal"),
  mlxCopy: document.querySelector("#mlxCopy"),
  cpuCopy: document.querySelector("#cpuCopy"),
  mlxStats: document.querySelector("#mlxStats"),
  cpuStats: document.querySelector("#cpuStats"),
  mlxPage: document.querySelector("#mlxPage"),
  cpuPage: document.querySelector("#cpuPage"),
  mlxModel: document.querySelector("#mlxModel"),
  cpuModel: document.querySelector("#cpuModel"),
};

const docFields = {
  mlx: {
    code: document.querySelector("#docCodeMlx"),
    date: document.querySelector("#docDateMlx"),
    className: document.querySelector("#docClassMlx"),
    title: document.querySelector("#docTitleMlx"),
    lang: document.querySelector("#docLangMlx"),
  },
  cpu: {
    code: document.querySelector("#docCodeCpu"),
    date: document.querySelector("#docDateCpu"),
    className: document.querySelector("#docClassCpu"),
    title: document.querySelector("#docTitleCpu"),
    lang: document.querySelector("#docLangCpu"),
  },
};

const formatNumber = (value) => {
  if (!Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
};

const escapeHtml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

function normalizeLabel(label) {
  return label.replaceAll("_", " ");
}

function renderText(text, entities = [], reveal = false) {
  if (!reveal) {
    return escapeHtml(text);
  }

  const sorted = [...entities].sort((a, b) => a.start - b.start || b.end - a.end);
  let html = "";
  let cursor = 0;

  for (const entity of sorted) {
    if (entity.start < cursor || entity.end <= entity.start) continue;
    html += escapeHtml(text.slice(cursor, entity.start));
    const original = text.slice(entity.start, entity.end);
    const label = normalizeLabel(entity.label || "private");
    html += `<span class="redaction" data-label="${escapeHtml(label)}" title="${escapeHtml(label)}: ${escapeHtml(original)}"><span>${escapeHtml(original)}</span></span>`;
    cursor = entity.end;
  }

  html += escapeHtml(text.slice(cursor));
  return html;
}

function currentDocument() {
  return state.documents[state.current];
}

function setDocumentChrome(documentData) {
  for (const side of Object.values(docFields)) {
    side.code.textContent = documentData.codename;
    side.date.textContent = documentData.date;
    side.className.textContent = documentData.className;
    side.title.textContent = documentData.title;
    side.lang.textContent = documentData.languages;
  }

  els.folioNow.textContent = String(state.current + 1);
  els.folioTotal.textContent = String(state.documents.length);
}

function resetStats() {
  renderStats(els.mlxStats, null, "Ready");
  renderStats(els.cpuStats, null, "Ready");
}

function renderStats(container, result, fallbackLabel = "Ready") {
  const values = container.querySelectorAll("strong");
  const pill = container.querySelector(".status-pill");

  if (!result) {
    values[0].textContent = "--";
    pill.textContent = fallbackLabel;
    pill.className = `status-pill ${fallbackLabel === "Running" ? "running" : "pending"}`;
    return;
  }

  values[0].textContent = formatNumber(result.stats.tokensPerSecond);
  pill.textContent = result.status === "live" ? "Live" : "Demo";
  pill.className = `status-pill ${result.status === "live" ? "live" : "fallback"}`;
  container.dataset.note = result.note || "";
}

function renderPages({ reveal = state.redacted } = {}) {
  const documentData = currentDocument();
  setDocumentChrome(documentData);

  const mlxEntities = state.latestResults?.results?.mlx?.entities || [];
  const cpuEntities = state.latestResults?.results?.cpu?.entities || [];
  els.mlxCopy.innerHTML = renderText(documentData.text, mlxEntities, reveal);
  els.cpuCopy.innerHTML = renderText(documentData.text, cpuEntities, reveal);

  els.book.classList.toggle("has-redactions", reveal);
}

function setBusy(isBusy) {
  state.busy = isBusy;
  els.run.disabled = isBusy;
  els.prev.disabled = isBusy;
  els.next.disabled = isBusy;
  els.run.querySelector("span").textContent = isBusy ? "Running" : "Run Redaction";
}

function startScan() {
  els.mlxPage.classList.remove("scan-active");
  els.cpuPage.classList.remove("scan-active");
  void els.mlxPage.offsetWidth;
  els.mlxPage.classList.add("scan-active");
  els.cpuPage.classList.add("scan-active");
}

function revealAfterScan(results) {
  state.latestResults = results;
  renderStats(els.mlxStats, results.results.mlx);
  renderStats(els.cpuStats, results.results.cpu);

  window.setTimeout(() => {
    state.redacted = true;
    renderPages({ reveal: true });
    setBusy(false);
  }, 940);
}

async function runRedaction() {
  if (state.busy) return;
  setBusy(true);
  state.redacted = false;
  renderPages({ reveal: false });
  renderStats(els.mlxStats, null, "Running");
  renderStats(els.cpuStats, null, "Running");
  startScan();

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        page_id: currentDocument().id,
        download: els.allowDownloads.checked,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    revealAfterScan(payload);
  } catch (error) {
    console.error(error);
    renderStats(els.mlxStats, null, "Error");
    renderStats(els.cpuStats, null, "Error");
    setBusy(false);
  }
}

function turnPage(direction) {
  if (state.busy || state.documents.length === 0) return;
  const nextIndex = state.current + direction;
  if (nextIndex < 0 || nextIndex >= state.documents.length) return;

  els.book.classList.remove("turn-next", "turn-prev");
  void els.book.offsetWidth;
  els.book.classList.add(direction > 0 ? "turn-next" : "turn-prev");

  window.setTimeout(() => {
    state.current = nextIndex;
    state.latestResults = null;
    state.redacted = false;
    resetStats();
    renderPages({ reveal: false });
  }, 180);

  window.setTimeout(() => {
    els.book.classList.remove("turn-next", "turn-prev");
  }, 640);
}

async function boot() {
  const response = await fetch("/api/documents");
  const payload = await response.json();
  state.documents = payload.documents;
  els.mlxModel.textContent = payload.models.mlx;
  els.cpuModel.textContent = payload.models.cpu;
  resetStats();
  renderPages({ reveal: false });

  els.prev.addEventListener("click", () => turnPage(-1));
  els.next.addEventListener("click", () => turnPage(1));
  els.run.addEventListener("click", runRedaction);

  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") turnPage(-1);
    if (event.key === "ArrowRight") turnPage(1);
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runRedaction();
  });

  if (window.lucide) {
    window.lucide.createIcons();
  }

  runRedaction();
}

boot();
