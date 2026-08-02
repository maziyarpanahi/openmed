import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.OPENMED_PREVIEW_PORT ?? "4173");
const baseURL = process.env.OPENMED_PREVIEW_URL ?? `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: ".",
  testMatch: ["**/*.spec.ts"],
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 3 : undefined,
  timeout: 45_000,
  snapshotPathTemplate:
    "{testDir}/snapshots/{platform}/{projectName}/{arg}{ext}",
  expect: {
    timeout: 7_500,
    toHaveScreenshot: {
      animations: "disabled",
      maxDiffPixelRatio: 0.002,
    },
  },
  outputDir: "../../../output/playwright/test-results",
  reporter: [
    ["line"],
    [
      "html",
      {
        open: "never",
        outputFolder: "../../../output/playwright/report",
      },
    ],
    [
      "json",
      {
        outputFile: "../../../output/playwright/results.json",
      },
    ],
  ],
  use: {
    baseURL,
    colorScheme: "light",
    reducedMotion: "no-preference",
    screenshot: "only-on-failure",
    serviceWorkers: "block",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  webServer: {
    command:
      `python3 -m http.server ${port} --bind 127.0.0.1 ` +
      "--directory ../../../site",
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
  ],
});
