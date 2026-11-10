import { copyFile, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const extensionDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryDir = resolve(extensionDir, "../..");
const outputDir = resolve(extensionDir, "dist");
const openmedEntry = resolve(
  repositoryDir,
  "js/openmedkit-web/src/index.ts",
);

const browserCryptoShim = {
  name: "browser-crypto-shim",
  setup(builder) {
    builder.onResolve({ filter: /^node:crypto$/ }, () => ({
      path: "node:crypto",
      namespace: "openmed-browser",
    }));
    builder.onLoad(
      { filter: /.*/, namespace: "openmed-browser" },
      () => ({
        contents:
          "export function createHmac() { throw new Error('Web Crypto is required'); }",
        loader: "js",
      }),
    );
  },
};

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });

const shared = {
  bundle: true,
  platform: "browser",
  target: ["chrome121", "firefox140"],
  sourcemap: true,
  legalComments: "none",
  alias: {
    openmed: openmedEntry,
  },
  plugins: [browserCryptoShim],
};

await Promise.all([
  build({
    ...shared,
    entryPoints: [resolve(extensionDir, "src/content.ts")],
    outfile: resolve(outputDir, "content.js"),
    format: "iife",
  }),
  build({
    ...shared,
    entryPoints: [resolve(extensionDir, "src/worker.ts")],
    outfile: resolve(outputDir, "worker.js"),
    format: "esm",
  }),
  copyFile(
    resolve(extensionDir, "manifest.json"),
    resolve(outputDir, "manifest.json"),
  ),
  copyFile(
    resolve(extensionDir, "PRIVACY.md"),
    resolve(outputDir, "PRIVACY.md"),
  ),
]);
