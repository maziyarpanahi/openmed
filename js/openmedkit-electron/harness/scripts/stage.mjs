#!/usr/bin/env node
/**
 * Stage the packaged harness app.
 *
 * Bundles the harness and the OpenMedKit Electron sources with esbuild into
 * build/app, and materializes the shared TEST-ONLY portable-bundle vectors
 * into build/resources/openmed-harness. Everything is local: no network, no
 * signing.
 */
import { mkdir, readFile, rm, writeFile, copyFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const harnessDir = fileURLToPath(new URL("..", import.meta.url));
const repoRoot = join(harnessDir, "..", "..", "..");
const buildDir = join(harnessDir, "build");
const appDir = join(buildDir, "app");
const resourcesDir = join(buildDir, "resources", "openmed-harness");
const vectorsPath = join(repoRoot, "tests", "fixtures", "portable_bundle", "vectors.json");
const openmedSource = join(repoRoot, "js", "openmedkit-web", "src", "index.ts");
// Renderer-side bundles need only the label tables, never the Node loaders.
const openmedLabels = join(repoRoot, "js", "openmedkit-web", "src", "types.ts");

const BUNDLES = {
  golden: { caseId: "valid_web", understoodExtensions: [] },
  extension: {
    caseId: "valid_critical_extension_understood",
    understoodExtensions: ["org.openmed.test.note"],
  },
  tampered: { caseId: "file_modified", understoodExtensions: [] },
};

function manifestText(vectors, key) {
  const entry = vectors.manifests[key];
  if (entry.text !== undefined) return entry.text;
  const base = manifestText(vectors, entry.base);
  return base.slice(0, entry.at) + entry.insert + base.slice(entry.at + entry.delete);
}

function caseById(vectors, id) {
  const item = vectors.cases.find((candidate) => candidate.id === id);
  if (item === undefined) throw new Error(`missing vector case ${id}`);
  return item;
}

async function materialize(vectors, item, root) {
  await mkdir(root, { recursive: true });
  await writeFile(join(root, "openmed-bundle.json"), manifestText(vectors, item.manifest));
  await writeFile(join(root, "openmed-bundle.sig.json"), vectors.envelopes[item.envelope]);
  const files = new Map(vectors.trees[item.tree].map((entry) => [entry.path, entry.text]));
  for (const path of item.tree_remove) files.delete(path);
  for (const entry of item.tree_set) files.set(entry.path, entry.text);
  for (const [path, text] of files) {
    const target = join(root, ...path.split("/"));
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, text);
  }
}

function verification(item) {
  return {
    ok: item.expected.ok,
    reason: item.expected.reason,
    contentId: item.expected.content_id,
    validSignatures: item.expected.valid_signatures,
    eligibleRuntimes: item.expected.eligible_runtimes,
    filesChecked: item.expected.files_checked,
    filesVerified: item.expected.files_verified,
  };
}

async function bundleSources() {
  const common = {
    bundle: true,
    sourcemap: false,
    minify: false,
    legalComments: "none",
    logLevel: "warning",
    alias: { openmed: openmedSource },
    tsconfig: join(harnessDir, "tsconfig.json"),
  };
  const node = { ...common, platform: "node", format: "cjs", target: "node22", external: ["electron"] };
  await build({ ...node, entryPoints: [join(harnessDir, "src", "main.ts")], outfile: join(appDir, "main.cjs") });
  await build({ ...node, entryPoints: [join(harnessDir, "src", "worker.ts")], outfile: join(appDir, "worker.cjs") });
  const preload = {
    ...common,
    alias: { openmed: openmedLabels },
    platform: "browser",
    format: "cjs",
    target: "chrome120",
    external: ["electron"],
  };
  await build({ ...preload, entryPoints: [join(harnessDir, "src", "preload.ts")], outfile: join(appDir, "preload.cjs") });
  await build({
    ...preload,
    entryPoints: [join(harnessDir, "src", "attack-preload.ts")],
    outfile: join(appDir, "attack-preload.cjs"),
  });
  await build({
    ...common,
    alias: { openmed: openmedLabels },
    platform: "browser",
    format: "iife",
    target: "chrome120",
    entryPoints: [join(harnessDir, "src", "renderer.ts")],
    outfile: join(appDir, "renderer", "renderer.js"),
  });
  await copyFile(join(harnessDir, "static", "index.html"), join(appDir, "renderer", "index.html"));
  const manifest = JSON.parse(await readFile(join(harnessDir, "package.json"), "utf8"));
  await writeFile(
    join(appDir, "package.json"),
    `${JSON.stringify(
      {
        name: manifest.name,
        productName: manifest.productName,
        version: manifest.version,
        description: manifest.description,
        author: manifest.author,
        license: manifest.license,
        private: true,
        main: "main.cjs",
      },
      null,
      2,
    )}\n`,
  );
}

async function stageResources() {
  const vectors = JSON.parse(await readFile(vectorsPath, "utf8"));
  if (vectors.synthetic_only !== true || vectors.test_only_keys !== true) {
    throw new Error("portable-bundle vectors must be synthetic and TEST-ONLY");
  }
  const webContext = vectors.contexts.valid_web;
  const bundles = {};
  for (const [name, spec] of Object.entries(BUNDLES)) {
    const item = caseById(vectors, spec.caseId);
    await materialize(vectors, item, join(resourcesDir, "bundles", name));
    bundles[name] = {
      directory: name,
      understoodExtensions: spec.understoodExtensions,
      expected: verification(item),
    };
  }
  const target = webContext.target;
  const qualification = {
    notice: vectors.notice,
    now: webContext.now,
    trustPolicy: vectors.trust_policies.default,
    target: {
      platform: target.platform,
      sdkName: target.sdk_name,
      sdkVersion: target.sdk_version,
      runtimes: target.runtimes,
      osVersion: target.os_version,
      policyVersion: target.policy_version,
      memoryBytes: target.memory_bytes,
      storageBytes: target.storage_bytes,
    },
    bundles,
  };
  await writeFile(
    join(resourcesDir, "qualification.json"),
    `${JSON.stringify(qualification, null, 2)}\n`,
  );
}

await rm(buildDir, { recursive: true, force: true });
await bundleSources();
await stageResources();
console.log("Staged the OpenMedKit Electron harness in build/.");
