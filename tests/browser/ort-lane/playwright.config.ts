import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.OPENMED_ORT_LANE_PORT ?? "4174");
const baseURL = `http://127.0.0.1:${port}`;
// Optional installed browser channel (for example "chrome") for local runs
// where Playwright's pinned Chromium build is not installed.
const channel = process.env.OPENMED_ORT_LANE_CHANNEL;

export default defineConfig({
  testDir: ".",
  testMatch: ["*.spec.ts"],
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  // One attempt per case: a retry would hide a flaky provider record.
  retries: 0,
  workers: 1,
  timeout: 120_000,
  outputDir: "../../../output/playwright/ort-lane/test-results",
  reporter: [
    ["line"],
    [
      "json",
      { outputFile: "../../../output/playwright/ort-lane/results.json" },
    ],
  ],
  use: {
    baseURL,
    serviceWorkers: "block",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "node server.mjs",
    url: `${baseURL}/healthz`,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(channel ? { channel } : {}),
        launchOptions: { args: webGpuLaunchArgs() },
      },
    },
  ],
});

function webGpuLaunchArgs(): string[] {
  const common = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
  if (process.platform === "darwin") {
    return [
      ...common,
      "--enable-features=WebGPUDeveloperFeatures",
      "--use-angle=metal",
    ];
  }
  if (process.platform === "win32") {
    return [
      ...common,
      "--enable-features=WebGPUDeveloperFeatures",
      "--use-angle=d3d11",
    ];
  }
  return [
    ...common,
    "--enable-features=Vulkan,WebGPUDeveloperFeatures",
    "--enable-unsafe-swiftshader",
    "--use-angle=swiftshader",
    "--use-vulkan=swiftshader",
  ];
}
