const BACKENDS = ["wasm", "webgpu"];
const PIPELINE_DTYPES = { wasm: "q8", webgpu: "fp16" };

const runtimeInput = document.querySelector("#runtime-module");
const repoInput = document.querySelector("#repo-id");
const backendSelect = document.querySelector("#backend");
const textInput = document.querySelector("#input-text");
const runButton = document.querySelector("#run-selected");
const benchmarkButton = document.querySelector("#benchmark-both");
const resetButton = document.querySelector("#reset");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const entities = document.querySelector("#entities");
const webGpuSupport = document.querySelector("#webgpu-support");

let activeConfiguration = "";
let runtimeFactory = null;
let runtimeFactoryUrl = "";
const detectors = new Map();
const timings = createTimings();

const query = new URLSearchParams(window.location.search);
runtimeInput.value = query.get("runtime") ?? "";
repoInput.value = query.get("model") ?? query.get("repo_id") ?? "";
webGpuSupport.textContent = navigator.gpu
  ? "WebGPU is available. Runtime and model assets must be supplied locally."
  : "WebGPU is unavailable; a locally supplied WASM runtime remains usable.";

runButton.addEventListener("click", () => runSelectedBackend());
benchmarkButton.addEventListener("click", () => benchmarkBothBackends());
resetButton.addEventListener("click", () => resetBenchmarks());
for (const input of [runtimeInput, repoInput]) {
  input.addEventListener("change", () => {
    try {
      resetForConfiguration(localConfiguration());
    } catch (error) {
      setStatus(errorMessage(error), "error");
    }
  });
}

async function runSelectedBackend() {
  try {
    await withBusyState(async () => {
      const backend = backendSelect.value;
      const output = await runBackend(backend);
      renderOutput(textInput.value, output, backend);
    });
  } catch (error) {
    setStatus(errorMessage(error), "error");
  }
}

async function benchmarkBothBackends() {
  await withBusyState(async () => {
    let completed = 0;
    for (const backend of BACKENDS) {
      try {
        const output = await runBackend(backend);
        renderOutput(textInput.value, output, backend);
        completed += 1;
      } catch (error) {
        setTimingError(backend);
        setStatus(`${backendLabel(backend)} failed: ${errorMessage(error)}`, "error");
      }
    }
    if (completed === BACKENDS.length) {
      setStatus("WASM and WebGPU benchmarks completed.");
    } else if (completed > 0) {
      setStatus("Benchmark completed for the available backend; see timing panel.");
    }
  });
}

async function runBackend(backend) {
  if (!BACKENDS.includes(backend)) {
    throw new Error(`Unsupported backend: ${backend}`);
  }
  if (backend === "webgpu" && !navigator.gpu) {
    throw new Error("this browser does not expose navigator.gpu");
  }

  const configuration = localConfiguration();
  const input = textInput.value.trim();
  if (!input) {
    throw new Error("enter synthetic text before running inference");
  }
  resetForConfiguration(configuration);
  const detector = await loadDetector(configuration, backend);

  setStatus(`Running first ${backendLabel(backend)} inference…`);
  const startedAt = performance.now();
  const output = await detector(input, { aggregation_strategy: "simple" });
  const elapsed = performance.now() - startedAt;
  if (timings[backend].firstInferenceMs === null) {
    timings[backend].firstInferenceMs = elapsed;
  }
  renderTimings();
  setStatus(`${backendLabel(backend)} inference completed in ${formatMs(elapsed)}.`);
  return normalizeOutput(output, input);
}

async function loadDetector(configuration, backend) {
  if (detectors.has(backend)) {
    return detectors.get(backend);
  }

  setStatus(
    `Loading the local model for ${backendLabel(backend)} from ` +
      `${configuration.modelUrl.pathname}…`,
  );
  const startedAt = performance.now();
  const createPipeline = await loadRuntimeFactory(configuration.runtimeUrl);
  const detector = await createPipeline({
    backend,
    dtype: PIPELINE_DTYPES[backend],
    modelUrl: configuration.modelUrl.href,
    task: "token-classification",
  });
  if (typeof detector !== "function") {
    throw new Error(
      "The local runtime adapter did not return a callable detector.",
    );
  }
  timings[backend].loadMs = performance.now() - startedAt;
  detectors.set(backend, detector);
  renderTimings();
  return detector;
}

async function loadRuntimeFactory(runtimeUrl) {
  if (runtimeFactory && runtimeFactoryUrl === runtimeUrl.href) {
    return runtimeFactory;
  }
  setStatus(`Loading local runtime adapter from ${runtimeUrl.pathname}…`);
  const runtime = await import(runtimeUrl.href);
  if (typeof runtime.createOpenMedPipeline !== "function") {
    throw new Error(
      "The local runtime module must export createOpenMedPipeline(options).",
    );
  }
  runtimeFactory = runtime.createOpenMedPipeline;
  runtimeFactoryUrl = runtimeUrl.href;
  return runtimeFactory;
}

function resetForConfiguration(configuration) {
  if (configuration.key === activeConfiguration) {
    return;
  }
  activeConfiguration = configuration.key;
  resetBenchmarks({ keepStatus: true });
  setStatus("Local runtime or model changed; benchmark state reset.");
}

function resetBenchmarks({ keepStatus = false } = {}) {
  for (const detector of detectors.values()) {
    detector.dispose?.();
  }
  detectors.clear();
  Object.assign(timings, createTimings());
  renderTimings();
  if (!keepStatus) {
    setStatus("Benchmark state reset. The browser asset cache is unchanged.");
  }
}

function createTimings() {
  return Object.fromEntries(
    BACKENDS.map((backend) => [
      backend,
      { loadMs: null, firstInferenceMs: null, error: false },
    ]),
  );
}

function renderTimings() {
  for (const backend of BACKENDS) {
    const timing = timings[backend];
    document.querySelector(`#${backend}-load`).textContent = timing.error
      ? "Unavailable"
      : formatMs(timing.loadMs);
    document.querySelector(`#${backend}-first`).textContent = timing.error
      ? "Unavailable"
      : formatMs(timing.firstInferenceMs);
  }
}

function setTimingError(backend) {
  timings[backend].error = true;
  renderTimings();
}

function normalizeOutput(output, text) {
  const flat = Array.isArray(output?.[0]) ? output.flat() : output;
  if (!Array.isArray(flat)) {
    return [];
  }

  const located = [];
  let searchCursor = 0;
  for (const item of flat) {
    const label = item.entity_group ?? item.entity ?? "PII";
    if (label === "O") {
      continue;
    }

    let start = Number(item.start);
    let end = Number(item.end);
    if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) {
      const match = locateWord(text, item.word ?? "", searchCursor);
      if (!match) {
        continue;
      }
      ({ start, end } = match);
    }
    searchCursor = end;
    located.push({
      label,
      score: Number(item.score ?? 0),
      start,
      end,
      word: item.word ?? "",
    });
  }

  return mergeBioSpans(located, text);
}

function locateWord(text, modelWord, searchCursor) {
  const cleaned = String(modelWord)
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
  return start === -1 ? null : { start, end: start + cleaned.length };
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

function renderOutput(text, spans, backend) {
  results.replaceChildren();
  entities.replaceChildren();

  if (spans.length === 0) {
    results.textContent = text;
    const empty = document.createElement("li");
    empty.textContent = `No spans returned by ${backendLabel(backend)}.`;
    entities.append(empty);
    return;
  }

  let cursor = 0;
  for (const span of nonOverlappingSpans(spans, text.length)) {
    results.append(document.createTextNode(text.slice(cursor, span.start)));
    const mark = document.createElement("mark");
    mark.textContent = text.slice(span.start, span.end);
    mark.title = `${span.label} · ${(span.score * 100).toFixed(1)}%`;
    results.append(mark);
    cursor = span.end;

    const item = document.createElement("li");
    item.textContent = `${span.label}: ${text.slice(span.start, span.end)} (${(
      span.score * 100
    ).toFixed(1)}%)`;
    entities.append(item);
  }
  results.append(document.createTextNode(text.slice(cursor)));
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

async function withBusyState(callback) {
  setButtonsDisabled(true);
  try {
    await callback();
  } catch (error) {
    setStatus(errorMessage(error), "error");
  } finally {
    setButtonsDisabled(false);
  }
}

function setButtonsDisabled(disabled) {
  runButton.disabled = disabled;
  benchmarkButton.disabled = disabled;
  resetButton.disabled = disabled;
}

function localConfiguration() {
  const runtimeValue = runtimeInput.value.trim();
  const modelValue = repoInput.value.trim();
  if (!runtimeValue || !modelValue) {
    throw new Error(
      "Supply a same-origin runtime module and model bundle before running.",
    );
  }
  const runtimeUrl = sameOriginUrl(runtimeValue, "runtime module");
  const modelUrl = sameOriginUrl(modelValue, "model bundle");
  return {
    key: `${runtimeUrl.href}\n${modelUrl.href}`,
    modelUrl,
    runtimeUrl,
  };
}

function sameOriginUrl(value, label) {
  const resolved = new URL(value, window.location.href);
  if (resolved.origin !== window.location.origin) {
    throw new Error(`${label} must use this page's origin`);
  }
  return resolved;
}

function setStatus(message, kind = "info") {
  status.textContent = message;
  status.dataset.kind = kind;
}

function backendLabel(backend) {
  return backend === "webgpu" ? "WebGPU" : "WASM";
}

function formatMs(value) {
  return value === null ? "—" : `${value.toFixed(1)} ms`;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
