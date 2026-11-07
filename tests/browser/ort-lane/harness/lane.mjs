// Browser half of the ONNX Runtime Web token-classification lane.
//
// `runLaneCase(caseId)` loads one exported graph from the same origin with the
// OpenMed browser runtime and the locally served ONNX Runtime Web bundle, runs
// the synthetic feeds, and compares the logits and decoded spans with the
// Python ONNX Runtime reference from `lane-fixture.json`. It returns a
// value-free record: controlled codes, counts, measured deltas, versions and
// SHA-256 digests. Logits, token ids, URLs and runtime log lines stay inside
// the page; only operator types and node counts are derived from the logs.

const SDK_URL = "/openmedkit/index.js";
const ORT_ASSET_PATH = "/ort/";
const ORT_BUNDLE_URLS = {
  wasm: "/ort/ort.wasm.min.mjs",
  webgpu: "/ort/ort.webgpu.min.mjs",
};
const FIXTURE_BASE = "/fixture/";
const FIXTURE_URL = `${FIXTURE_BASE}lane-fixture.json`;
const RECORD_FORMAT = "openmed-ort-browser-lane-record";
const LANE_ID = "ort-web-token-classification";
const EXPECTED_BACKENDS = {
  webgpu: "webgpu",
  "wasm-single": "wasm-basic",
  "wasm-threaded": "wasm-simd-threads",
};
const MISSING_KERNEL =
  /kernel not found in registries for Op type: ([A-Za-z][A-Za-z0-9_]*) node name: (.*)$/;
const PLACEMENT =
  /(?:All nodes|Node\(s\)) placed on \[([A-Za-z]+)\]\. Number of nodes: (\d+)/;
const PLACEMENT_KEYS = {
  CPUExecutionProvider: "cpu",
  JsExecutionProvider: "webgpu-jsep",
  WebGpuExecutionProvider: "webgpu",
};

export async function runLaneCase(caseId) {
  const capture = captureConsole();
  let record;
  try {
    const fixture = await fetchJson(FIXTURE_URL);
    const laneCase = fixture.cases.find((item) => item.case_id === caseId);
    if (laneCase === undefined) throw new Error("unknown lane case");
    record = baseRecord(fixture, laneCase);
    await runCase(fixture, laneCase, record, capture);
  } catch {
    if (record === undefined) throw new Error("lane fixture could not be loaded");
    if (record.outcomes.status === "pending") fail(record, "harness_error");
  } finally {
    capture.restore();
  }
  if (record.outcomes.status === "pending") fail(record, "harness_error");
  return record;
}

async function runCase(fixture, laneCase, record, capture) {
  const provider = laneCase.provider;
  const graph = fixture.graphs[laneCase.graph];
  const bundle = provider === "webgpu" ? "webgpu" : "wasm";
  const ort = await import(ORT_BUNDLE_URLS[bundle]);
  record.provider.runtime_version = String(
    ort.env?.versions?.web ?? ort.env?.versions?.common ?? "0.0.0",
  );
  const sdk = await import(SDK_URL);

  // Hash exactly what the browser receives before anything else, so every
  // record (including unavailable ones) carries browser-computed digests.
  const graphUrl = `${FIXTURE_BASE}${graph.file}`;
  const graphDigest = await sha256Hex(await fetchBytes(graphUrl));
  const externalDigests = [];
  for (const item of graph.external_data) {
    externalDigests.push(await sha256Hex(await fetchBytes(`${FIXTURE_BASE}${item.file}`)));
  }
  const feeds = fixture.feeds;
  const reference = fixture.reference;
  record.digests = {
    external_data_sha256: await combineDigests(externalDigests),
    feeds_sha256: await sha256Hex(int64Bytes(feeds)),
    graph_sha256: graphDigest,
    reference_sha256: await sha256Hex(Float32Array.from(reference.logits).buffer),
  };
  if (
    record.digests.graph_sha256 !== graph.sha256 ||
    externalDigests.some((value, index) => value !== graph.external_data[index].sha256) ||
    record.digests.feeds_sha256 !== feeds.sha256 ||
    record.digests.reference_sha256 !== reference.sha256
  ) {
    fail(record, "fixture_digest_mismatch");
    return;
  }

  if (provider === "webgpu") {
    if (navigator.gpu === undefined || navigator.gpu === null) {
      unavailable(record, "webgpu_api_unavailable");
      return;
    }
    const probe = await sdk.probeOrtWebCapabilities({
      powerPreference: "high-performance",
    });
    if (!probe.adapterAvailable) {
      unavailable(record, "webgpu_adapter_unavailable");
      return;
    }
    if (graph.precision === "float16" && !(await adapterSupportsShaderF16())) {
      // ORT's WebGPU kernels for float16 tensors need the shader-f16 feature;
      // without it (for example on SwiftShader) the graph cannot run here.
      unavailable(record, "webgpu_shader_f16_unavailable");
      return;
    }
  }
  if (provider === "wasm-threaded") {
    if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") {
      unavailable(record, "cross_origin_isolation_unavailable");
      return;
    }
    if ((navigator.hardwareConcurrency ?? 1) < 2) {
      unavailable(record, "wasm_threads_unavailable");
      return;
    }
  }

  if (ort.env !== undefined) ort.env.logLevel = "verbose";
  const labelCount = Object.keys(fixture.id2label).length;
  let session;
  try {
    session = await sdk.loadWebGpuTokenClassificationSession({
      assetPath: ORT_ASSET_PATH,
      capabilities: provider === "webgpu" ? undefined : { webgpu: false },
      fallbackOnWebGpuError: false,
      labelCount,
      modelName: "synthetic-token-classifier",
      modelPath: { wasm: graphUrl, webgpu: graphUrl },
      precision: graph.precision,
      runtime: ort,
      sessionOptions: {
        externalData: graph.external_data.map((item) => ({
          data: `${FIXTURE_BASE}${item.file}`,
          path: item.location,
        })),
        graphOptimizationLevel: "all",
        logSeverityLevel: 0,
        logVerbosityLevel: 0,
      },
    });
  } catch {
    applyFallbackEvidence(record, capture.lines, provider);
    fail(record, "session_create_failed");
    return;
  }

  try {
    record.provider.backend = session.backend;
    record.provider.execution_providers =
      session.backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
    record.provider.wasm_threads = Number(ort.env?.wasm?.numThreads ?? 0);
    applyFallbackEvidence(record, capture.lines, provider);
    if (session.backend !== EXPECTED_BACKENDS[provider]) {
      fail(record, "backend_mismatch");
      return;
    }
    if (
      (provider === "wasm-threaded" && record.provider.wasm_threads < 2) ||
      (provider === "wasm-single" && record.provider.wasm_threads !== 1)
    ) {
      fail(record, "thread_count_mismatch");
      return;
    }

    const tokens = {
      attentionMask: feeds.attention_mask,
      batchSize: feeds.batch_size,
      inputIds: feeds.input_ids,
      sequenceLength: feeds.sequence_length,
    };
    if (feeds.input_names.includes("token_type_ids")) {
      tokens.tokenTypeIds = feeds.token_type_ids;
    }
    let logits;
    try {
      logits = await session.run(tokens);
    } catch {
      fail(record, "inference_failed");
      return;
    }
    compare(sdk, fixture, laneCase, record, logits);
  } finally {
    await session.dispose();
  }
}

function compare(sdk, fixture, laneCase, record, logits) {
  const reference = {
    data: Float32Array.from(fixture.reference.logits),
    dims: fixture.reference.dims,
    outputName: "logits",
  };
  const mask = fixture.feeds.attention_mask;
  const tolerance = laneCase.tolerances;
  const sameDims =
    logits.dims.length === reference.dims.length &&
    logits.dims.every((value, index) => value === reference.dims[index]);
  if (!sameDims || logits.data.length !== reference.data.length) {
    fail(record, "logit_shape_mismatch");
    return;
  }
  if (!logits.data.every((value) => Number.isFinite(value))) {
    fail(record, "non_finite_logits");
    return;
  }
  let maxAbs = 0;
  for (let index = 0; index < reference.data.length; index += 1) {
    maxAbs = Math.max(maxAbs, Math.abs(reference.data[index] - logits.data[index]));
  }
  const referenceSpans = sdk.decodeWebGpuTokenSpans(reference, fixture.id2label, mask);
  const candidateSpans = sdk.decodeWebGpuTokenSpans(logits, fixture.id2label, mask);
  const gate = sdk.evaluateWebGpuRecallGate(referenceSpans, candidateSpans, {
    maxRecallDelta: tolerance.max_recall_delta,
    spanTolerance: tolerance.span_tolerance_tokens,
  });
  const spanParity =
    referenceSpans.length === candidateSpans.length &&
    gate.matched_count === referenceSpans.length;
  Object.assign(record.outcomes, {
    candidate_span_count: candidateSpans.length,
    logits_within_tolerance: maxAbs <= tolerance.max_abs_logit_delta,
    matched_span_count: gate.matched_count,
    max_abs_logit_delta: maxAbs,
    recall: gate.recall,
    reference_span_count: referenceSpans.length,
    span_parity: spanParity,
  });

  if (spanKey(referenceSpans) !== spanKey(fixture.reference.spans)) {
    fail(record, "reference_decode_mismatch");
  } else if (!record.outcomes.logits_within_tolerance) {
    fail(record, "logit_tolerance_exceeded");
  } else if (!spanParity) {
    fail(record, "span_mismatch");
  } else if (!gate.passed) {
    fail(record, "recall_gate_failed");
  } else if (!certifies(sdk, fixture, tolerance, reference, logits)) {
    fail(record, "certification_rejected");
  } else if (record.outcomes.webgpu_support === "unknown") {
    fail(record, "fallback_evidence_missing");
  } else if (
    laneCase.expected_webgpu_support !== null &&
    record.outcomes.webgpu_support !== laneCase.expected_webgpu_support
  ) {
    fail(record, "unexpected_webgpu_support");
  } else {
    record.outcomes.status = "passed";
    record.outcomes.reason_code = "ok";
  }
}

function spanKey(spans) {
  return spans
    .map((span) =>
      [span.batch_index, span.label, span.start_token, span.end_token].join(":"),
    )
    .join("|");
}

function certifies(sdk, fixture, tolerance, reference, logits) {
  try {
    sdk.certifyWebGpuReference({
      attentionMask: fixture.feeds.attention_mask,
      candidateLogits: logits,
      id2label: fixture.id2label,
      maxRecallDelta: tolerance.max_recall_delta,
      referenceLogits: reference,
      spanTolerance: tolerance.span_tolerance_tokens,
      tolerance: tolerance.max_abs_logit_delta,
    });
    return true;
  } catch {
    return false;
  }
}

function applyFallbackEvidence(record, lines, provider) {
  const missingNodes = new Set();
  const missingOps = new Set();
  const placement = {};
  for (const line of lines) {
    const missing = MISSING_KERNEL.exec(line);
    if (missing !== null) {
      missingOps.add(missing[1]);
      missingNodes.add(`${missing[1]}\u0000${missing[2]}`);
    }
    const placed = PLACEMENT.exec(line);
    if (placed !== null) {
      const key = PLACEMENT_KEYS[placed[1]] ?? "other";
      placement[key] = Number(placed[2]);
    }
  }
  record.outcomes.fallback_op_types = [...missingOps].sort();
  record.outcomes.fallback_node_count = missingNodes.size;
  record.outcomes.node_placement = placement;
  if (provider !== "webgpu") {
    record.outcomes.webgpu_support = "not_applicable";
  } else if (missingOps.size > 0) {
    record.outcomes.webgpu_support = "partial";
  } else if ((placement.webgpu ?? 0) + (placement["webgpu-jsep"] ?? 0) > 0) {
    record.outcomes.webgpu_support = "full";
  } else {
    record.outcomes.webgpu_support = "unknown";
  }
}

function baseRecord(fixture, laneCase) {
  const graph = fixture.graphs[laneCase.graph];
  return {
    browser: { name: "chromium", version: "0" },
    case_id: laneCase.case_id,
    digests: {
      external_data_sha256: null,
      feeds_sha256: fixture.feeds.sha256,
      graph_sha256: graph.sha256,
      reference_sha256: fixture.reference.sha256,
    },
    format: RECORD_FORMAT,
    format_version: 1,
    graph: laneCase.graph,
    lane: LANE_ID,
    outcomes: {
      candidate_span_count: null,
      cross_origin_request_count: 0,
      fallback_node_count: 0,
      fallback_op_types: [],
      logits_within_tolerance: null,
      matched_span_count: null,
      max_abs_logit_delta: null,
      node_placement: {},
      reason_code: "pending",
      recall: null,
      reference_span_count: null,
      same_origin_request_count: 0,
      span_parity: null,
      status: "pending",
      webgpu_support: laneCase.provider === "webgpu" ? "unknown" : "not_applicable",
    },
    precision: graph.precision,
    provider: {
      backend: null,
      cross_origin_isolated: globalThis.crossOriginIsolated === true,
      execution_providers: [],
      id: laneCase.provider,
      runtime: "onnxruntime-web",
      runtime_bundle: laneCase.provider === "webgpu" ? "webgpu" : "wasm",
      runtime_version: "0.0.0",
      wasm_threads: 0,
      worker_count: 0,
    },
    tolerances: { ...laneCase.tolerances },
  };
}

function unavailable(record, reason) {
  record.outcomes.status = "unavailable";
  record.outcomes.reason_code = reason;
}

function fail(record, reason) {
  record.outcomes.status = "failed";
  record.outcomes.reason_code = reason;
}

async function adapterSupportsShaderF16() {
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    return adapter !== null && adapter.features.has("shader-f16");
  } catch {
    return false;
  }
}

function captureConsole() {
  const lines = [];
  const methods = ["debug", "error", "info", "log", "warn"];
  const originals = {};
  for (const method of methods) {
    originals[method] = console[method];
    console[method] = (...args) => {
      lines.push(args.map((value) => String(value)).join(" "));
    };
  }
  return {
    lines,
    restore() {
      for (const method of methods) console[method] = originals[method];
    },
  };
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("fixture request failed");
  return response.json();
}

async function fetchBytes(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("artifact request failed");
  return response.arrayBuffer();
}

function int64Bytes(feeds) {
  const values = [];
  for (const name of feeds.input_names) {
    for (const value of feeds[name]) values.push(BigInt(value));
  }
  return BigInt64Array.from(values).buffer;
}

async function combineDigests(digests) {
  if (digests.length === 0) return null;
  if (digests.length === 1) return digests[0];
  return sha256Hex(new TextEncoder().encode(digests.join("")));
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
