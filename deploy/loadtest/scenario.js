import http from "k6/http";
import { check } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

const SYNTHETIC_NOTE =
  "Taylor Reed, born 1981-02-03, visited Example Clinic for a routine " +
  "follow-up. Call the fictional records desk at 555-0100.";

function envValue(names, fallback) {
  for (const name of names) {
    const value = __ENV[name];
    if (value !== undefined && value !== "") {
      return value;
    }
  }
  return fallback;
}

function envNumber(names, fallback, minimum, maximum) {
  const raw = envValue(names, String(fallback));
  const value = Number(raw);
  if (
    !Number.isFinite(value) ||
    value < minimum ||
    (maximum !== undefined && value > maximum)
  ) {
    throw new Error(
      `${names[0]} must be a number between ${minimum} and ${maximum === undefined ? "infinity" : maximum}`,
    );
  }
  return value;
}

function envInteger(names, fallback, minimum) {
  const value = envNumber(names, fallback, minimum);
  if (!Number.isInteger(value)) {
    throw new Error(`${names[0]} must be an integer`);
  }
  return value;
}

const BASE_URL = envValue(
  ["LOADTEST_BASE_URL", "BASE_URL"],
  "http://127.0.0.1:8080",
).replace(/\/+$/, "");
const DURATION_SECONDS = envInteger(
  ["LOADTEST_DURATION_SECONDS", "LOADTEST_DURATION"],
  30,
  1,
);
const ARRIVAL_RATE = envInteger(
  ["LOADTEST_RATE", "LOADTEST_RATE_RPS", "LOADTEST_ARRIVAL_RATE"],
  2,
  1,
);
const PRE_ALLOCATED_VUS = envInteger(
  ["LOADTEST_CONCURRENCY"],
  4,
  1,
);
const MAX_VUS = envInteger(
  ["LOADTEST_MAX_VUS"],
  Math.max(PRE_ALLOCATED_VUS, PRE_ALLOCATED_VUS * 2),
  PRE_ALLOCATED_VUS,
);

const SLO_P95_MS = envNumber(
  ["LOADTEST_SLO_P95_MS", "SLO_P95_MS"],
  30000,
  1,
);
const SLO_P99_MS = envNumber(
  ["LOADTEST_SLO_P99_MS", "SLO_P99_MS"],
  60000,
  1,
);
const SLO_ERROR_RATE = envNumber(
  ["LOADTEST_SLO_ERROR_RATE", "SLO_ERROR_RATE"],
  0.05,
  0,
  1,
);
const SLO_MIN_THROUGHPUT_RPS = envNumber(
  ["LOADTEST_SLO_MIN_THROUGHPUT_RPS", "SLO_MIN_THROUGHPUT_RPS"],
  0.5,
  0.001,
);

const requestLatency = new Trend("loadtest_latency_ms", true);
const requestErrors = new Rate("loadtest_errors");
const requestCount = new Counter("loadtest_requests");

export const options = {
  discardResponseBodies: true,
  scenarios: {
    mixed_traffic: {
      executor: "constant-arrival-rate",
      rate: ARRIVAL_RATE,
      timeUnit: "1s",
      duration: `${DURATION_SECONDS}s`,
      preAllocatedVUs: PRE_ALLOCATED_VUS,
      maxVUs: MAX_VUS,
    },
  },
  thresholds: {
    // Keep both the custom metric and the built-in metric visible in the k6
    // summary so the gate remains useful when the workload is edited.
    loadtest_latency_ms: [`p(95)<${SLO_P95_MS}`, `p(99)<${SLO_P99_MS}`],
    http_req_duration: [`p(95)<${SLO_P95_MS}`, `p(99)<${SLO_P99_MS}`],
    loadtest_errors: [`rate<${SLO_ERROR_RATE}`],
    http_req_failed: [`rate<${SLO_ERROR_RATE}`],
    http_reqs: [`rate>=${SLO_MIN_THROUGHPUT_RPS}`],
  },
};

const jsonParams = {
  headers: {
    Accept: "application/json",
    "Content-Type": "application/json",
  },
};

function recordResponse(endpoint, response) {
  const succeeded = response.status >= 200 && response.status < 300;
  const tags = { endpoint };
  requestLatency.add(response.timings.duration, tags);
  requestErrors.add(succeeded ? 0 : 1, tags);
  requestCount.add(1, tags);
  check(response, {
    [`${endpoint} returns 2xx`]: () => succeeded,
  });
}

function post(endpoint, payload) {
  const response = http.post(
    `${BASE_URL}${endpoint}`,
    JSON.stringify(payload),
    { headers: jsonParams.headers, tags: { endpoint } },
  );
  recordResponse(endpoint, response);
}

export default function mixedTraffic() {
  // A fixed mix makes a run reproducible enough to compare while retaining
  // concurrent traffic across all three service surfaces.
  const bucket = (__VU + __ITER) % 10;
  if (bucket < 4) {
    post("/analyze", {
      text: SYNTHETIC_NOTE,
      model_name: "disease_detection_superclinical",
      confidence_threshold: 0,
    });
    return;
  }

  if (bucket < 8) {
    post("/pii/deidentify", {
      text: SYNTHETIC_NOTE,
      method: "mask",
      model_name: "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
      confidence_threshold: 0,
    });
    return;
  }

  post("/pii/extract/stream", {
    text: SYNTHETIC_NOTE,
    model_name: "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    confidence_threshold: 0,
    chunk_size: 128,
    window_chars: 256,
    tokenizer_context_chars: 64,
    max_entity_chars: 128,
    include_text: false,
  });
}

function metricValues(data, name) {
  const metric = data.metrics[name];
  return metric && metric.values ? metric.values : {};
}

function rounded(value) {
  return Math.round(value * 100) / 100;
}

export function handleSummary(data) {
  const latency = metricValues(data, "loadtest_latency_ms");
  const httpLatency = metricValues(data, "http_req_duration");
  const errors = metricValues(data, "loadtest_errors");
  const requests = metricValues(data, "loadtest_requests");
  const httpRequests = metricValues(data, "http_reqs");
  const p95 = latency["p(95)"] || httpLatency["p(95)"] || 0;
  const p99 = latency["p(99)"] || httpLatency["p(99)"] || 0;
  const errorRate = errors.rate || 0;
  const throughput = httpRequests.rate || 0;
  const count = requests.count || httpRequests.count || 0;
  const passed =
    p95 < SLO_P95_MS &&
    p99 < SLO_P99_MS &&
    errorRate < SLO_ERROR_RATE &&
    throughput >= SLO_MIN_THROUGHPUT_RPS;

  const report = {
    schema_version: 1,
    workload: {
      name: "synthetic-mixed-service-traffic",
      endpoints: ["/analyze", "/pii/deidentify", "/pii/extract/stream"],
      weights: { analyze: 0.4, deidentify: 0.4, stream: 0.2 },
    },
    requests: count,
    throughput_rps: rounded(throughput),
    p95_ms: rounded(p95),
    p99_ms: rounded(p99),
    error_rate: rounded(errorRate),
    slo: {
      p95_ms: SLO_P95_MS,
      p99_ms: SLO_P99_MS,
      max_error_rate: SLO_ERROR_RATE,
      min_throughput_rps: SLO_MIN_THROUGHPUT_RPS,
    },
    passed,
  };

  const output = {
    stdout: [
      "Load-test SLO summary",
      `requests: ${report.requests}`,
      `throughput: ${report.throughput_rps} req/s (minimum ${SLO_MIN_THROUGHPUT_RPS})`,
      `p95 latency: ${report.p95_ms} ms (maximum ${SLO_P95_MS} ms)`,
      `p99 latency: ${report.p99_ms} ms (maximum ${SLO_P99_MS} ms)`,
      `error rate: ${report.error_rate} (maximum ${SLO_ERROR_RATE})`,
      `SLO status: ${passed ? "PASS" : "FAIL; k6 will exit non-zero"}`,
      "",
    ].join("\n"),
  };
  const resultFile = envValue(
    ["LOADTEST_RESULT_FILE", "K6_SUMMARY_FILE"],
    "",
  );
  if (resultFile) {
    output[resultFile] = JSON.stringify(report, null, 2);
  }
  return output;
}
