import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "../../tests/web",
  testMatch: "test_extension_redaction.spec.ts",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  outputDir: "dist/test-results",
  use: {
    trace: "retain-on-failure",
  },
});
