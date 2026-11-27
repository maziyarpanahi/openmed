import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  expect,
  test,
  type Browser,
  type BrowserContext,
  type Page,
} from "@playwright/test";

// Runs every case in the generated lane fixture on its ONNX Runtime Web
// provider and writes one value-free record per case. Generate the fixture
// first with `python -m openmed.onnx.browser_lane prepare`.

const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const fixtureDir = resolve(
  repoRoot,
  process.env.OPENMED_ORT_LANE_FIXTURE_DIR ?? "output/playwright/ort-lane/fixture",
);
const recordsDir = resolve(
  repoRoot,
  process.env.OPENMED_ORT_LANE_RECORDS_DIR ?? "output/playwright/ort-lane/records",
);
const fixturePath = join(fixtureDir, "lane-fixture.json");
// Reason codes for which an `unavailable` record is accepted, for example
// `webgpu_shader_f16_unavailable` on a SwiftShader adapter. The record is
// still written and annotated; every other unavailable record fails.
const allowedUnavailable = new Set(
  (process.env.OPENMED_ORT_LANE_ALLOW_UNAVAILABLE ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0),
);

const RECORD_KEYS = [
  "browser",
  "case_id",
  "digests",
  "format",
  "format_version",
  "graph",
  "lane",
  "outcomes",
  "precision",
  "provider",
  "tolerances",
];
const SAFE_STRING = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

interface LaneCase {
  case_id: string;
  provider: "webgpu" | "wasm-threaded" | "wasm-single";
  graph: "fp32" | "fp16" | "fallback-control";
  isolation: "cross-origin-isolated" | "none";
  expected_webgpu_support: string | null;
}

interface LaneRecord {
  case_id: string;
  graph: string;
  browser: { name: string; version: string };
  provider: {
    id: string;
    backend: string | null;
    cross_origin_isolated: boolean;
    wasm_threads: number;
    worker_count: number;
    [key: string]: unknown;
  };
  digests: Record<string, string | null>;
  outcomes: {
    status: "passed" | "failed" | "unavailable";
    reason_code: string;
    webgpu_support: string;
    fallback_op_types: string[];
    same_origin_request_count: number;
    cross_origin_request_count: number;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

interface LaneFixture {
  cases: LaneCase[];
  graphs: Record<string, { sha256: string; control_op_type?: string }>;
}

const fixture: LaneFixture | null = existsSync(fixturePath)
  ? (JSON.parse(readFileSync(fixturePath, "utf8")) as LaneFixture)
  : null;

test.beforeAll(() => {
  mkdirSync(recordsDir, { recursive: true });
});

if (fixture === null) {
  test("lane fixture is prepared", () => {
    throw new Error(
      "Missing lane fixture. Run `python -m openmed.onnx.browser_lane prepare` first.",
    );
  });
} else {
  for (const laneCase of fixture.cases) {
    test(`${laneCase.case_id} runs the exported ${laneCase.graph} graph on ${laneCase.provider}`, async ({
      browser,
      baseURL,
    }, testInfo) => {
      const record = await runLaneCase(browser, baseURL, laneCase);
      assertValueFree(record);
      writeFileSync(
        join(recordsDir, `${laneCase.case_id}.json`),
        `${JSON.stringify(record, null, 2)}\n`,
      );
      await testInfo.attach(`${laneCase.case_id}.json`, {
        body: JSON.stringify(record, null, 2),
        contentType: "application/json",
      });
      testInfo.annotations.push({
        type: "lane-record",
        description:
          `${record.outcomes.status}:${record.outcomes.reason_code}:` +
          `${record.outcomes.webgpu_support}`,
      });

      expect(record.outcomes.cross_origin_request_count).toBe(0);
      expect(record.outcomes.same_origin_request_count).toBeGreaterThan(0);
      if (record.outcomes.status === "unavailable") {
        expect(
          allowedUnavailable.has(record.outcomes.reason_code),
          `${laneCase.case_id} is unavailable (${record.outcomes.reason_code}); ` +
            "list the reason in OPENMED_ORT_LANE_ALLOW_UNAVAILABLE to accept it",
        ).toBe(true);
        return;
      }
      expect(record.outcomes.status, record.outcomes.reason_code).toBe("passed");
      expect(record.digests.graph_sha256).toBe(
        fixture.graphs[laneCase.graph].sha256,
      );
      if (laneCase.expected_webgpu_support !== null) {
        expect(record.outcomes.webgpu_support).toBe(
          laneCase.expected_webgpu_support,
        );
      }
      const controlOp = fixture.graphs[laneCase.graph].control_op_type;
      if (controlOp !== undefined) {
        expect(record.outcomes.fallback_op_types).toContain(controlOp);
      }
      if (laneCase.provider === "wasm-threaded") {
        expect(record.provider.cross_origin_isolated).toBe(true);
        expect(record.provider.wasm_threads).toBeGreaterThan(1);
        expect(record.provider.worker_count).toBeGreaterThan(0);
      }
      if (laneCase.provider === "wasm-single") {
        expect(record.provider.cross_origin_isolated).toBe(false);
        expect(record.provider.wasm_threads).toBe(1);
        expect(record.provider.worker_count).toBe(0);
      }
    });
  }

  test("a graph that does not match its fixture digest fails closed", async ({
    browser,
    baseURL,
  }) => {
    const laneCase = caseFor(fixture, "wasm-single");
    const record = await runLaneCase(browser, baseURL, laneCase, {
      tamperGraphDigest: true,
    });

    assertValueFree(record);
    expect(record.outcomes.status).toBe("failed");
    expect(record.outcomes.reason_code).toBe("fixture_digest_mismatch");
    expect(record.outcomes.max_abs_logit_delta).toBeNull();
  });

  test("threaded WASM without cross-origin isolation is recorded as unavailable", async ({
    browser,
    baseURL,
  }) => {
    const laneCase = caseFor(fixture, "wasm-threaded");
    const record = await runLaneCase(browser, baseURL, laneCase, {
      isolation: "none",
    });

    assertValueFree(record);
    expect(record.provider.cross_origin_isolated).toBe(false);
    expect(record.outcomes.status).toBe("unavailable");
    expect(record.outcomes.reason_code).toBe("cross_origin_isolation_unavailable");
    expect(record.outcomes.max_abs_logit_delta).toBeNull();
  });

  test("request interception blocks and counts a cross-origin attempt", async ({
    browser,
    baseURL,
  }) => {
    const lane = await openLanePage(browser, baseURL, "none");
    try {
      const outcome = await lane.page.evaluate(async () => {
        try {
          await fetch(
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort-wasm-simd-threaded.wasm",
          );
          return "fetched";
        } catch {
          return "blocked";
        }
      });
      expect(outcome).toBe("blocked");
      expect(lane.crossOriginRequests()).toBe(1);
    } finally {
      await lane.context.close();
    }
  });
}

interface LaneRunOptions {
  isolation?: LaneCase["isolation"];
  tamperGraphDigest?: boolean;
}

function caseFor(laneFixture: LaneFixture, provider: LaneCase["provider"]): LaneCase {
  const laneCase = laneFixture.cases.find((item) => item.provider === provider);
  if (laneCase === undefined) throw new Error(`fixture has no ${provider} case`);
  return laneCase;
}

async function runLaneCase(
  browser: Browser,
  baseURL: string | undefined,
  laneCase: LaneCase,
  options: LaneRunOptions = {},
): Promise<LaneRecord> {
  const lane = await openLanePage(
    browser,
    baseURL,
    options.isolation ?? laneCase.isolation,
  );
  if (options.tamperGraphDigest) {
    // Negative control: the served fixture claims a different graph digest.
    await lane.context.route("**/fixture/lane-fixture.json", async (route) => {
      const response = await route.fetch();
      const served = (await response.json()) as LaneFixture;
      served.graphs[laneCase.graph].sha256 = "0".repeat(64);
      await route.fulfill({ response, json: served });
    });
  }
  try {
    const record = (await lane.page.evaluate(async (caseId) => {
      // Served by server.mjs on the lane origin; resolved inside the page.
      const harnessUrl = "/lane/lane.mjs";
      const harness = await import(harnessUrl);
      return harness.runLaneCase(caseId);
    }, laneCase.case_id)) as LaneRecord;
    record.browser = {
      name: browser.browserType().name(),
      version: browser.version(),
    };
    record.provider.worker_count = lane.workerCount();
    record.outcomes.same_origin_request_count = lane.sameOriginRequests();
    record.outcomes.cross_origin_request_count = lane.crossOriginRequests();
    if (
      record.outcomes.cross_origin_request_count > 0 &&
      record.outcomes.status !== "failed"
    ) {
      record.outcomes.status = "failed";
      record.outcomes.reason_code = "cross_origin_request";
    }
    return record;
  } finally {
    await lane.context.close();
  }
}

async function openLanePage(
  browser: Browser,
  baseURL: string | undefined,
  isolation: LaneCase["isolation"],
): Promise<{
  context: BrowserContext;
  page: Page;
  crossOriginRequests: () => number;
  sameOriginRequests: () => number;
  workerCount: () => number;
}> {
  const origin = new URL(baseURL ?? "http://127.0.0.1:4174").origin;
  const context = await browser.newContext({ serviceWorkers: "block" });
  let crossOrigin = 0;
  let sameOrigin = 0;
  let workers = 0;
  // Every issued request is counted (URLs are never stored), and routing
  // aborts anything that is not this origin before it reaches the network.
  context.on("request", (request) => {
    if (new URL(request.url()).origin === origin) sameOrigin += 1;
    else crossOrigin += 1;
  });
  await context.route("**/*", async (route) => {
    if (new URL(route.request().url()).origin === origin) {
      await route.continue();
    } else {
      await route.abort("blockedbyclient");
    }
  });
  const page = await context.newPage();
  page.on("worker", () => {
    workers += 1;
  });
  const harness = isolation === "cross-origin-isolated" ? "isolated" : "plain";
  await page.goto(`${origin}/lane/${harness}.html`, {
    waitUntil: "domcontentloaded",
  });
  return {
    context,
    page,
    crossOriginRequests: () => crossOrigin,
    sameOriginRequests: () => sameOrigin,
    workerCount: () => workers,
  };
}

function assertValueFree(record: LaneRecord): void {
  expect(Object.keys(record).sort()).toEqual(RECORD_KEYS);
  const visit = (value: unknown, path: string): void => {
    if (typeof value === "string") {
      expect(SAFE_STRING.test(value), `${path} must be a controlled token`).toBe(
        true,
      );
    } else if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, `${path}[${index}]`));
    } else if (value !== null && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        visit(item, `${path}.${key}`);
      }
    }
  };
  visit(record, "record");
}
