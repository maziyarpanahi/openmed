// Stage the pinned local model fixture that the parity app loads.
//
// CI runs this before building the app. It only downloads artifacts that are
// already published, by immutable revision, and checks every file against the
// sha256 and size pinned in parity/model-sources.json. It never converts or
// publishes a model (docs/security/hf-token-policy.md). The app never
// downloads: the probe verifies the staged bundle natively and refuses drift.
//
//   tsx scripts/stage-model-fixture.ts download --platform android|ios --cache DIR
//   tsx scripts/stage-model-fixture.ts assemble --platform android|ios --source DIR --out DIR
//   tsx scripts/stage-model-fixture.ts verify --fixture DIR
//
// For iOS, `download` places the tokenizer files under DIR/tokenizer and the
// pinned Core ML export under DIR/coreml, and fails closed while no Core ML
// export is pinned.

import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

import {
  bundleDigest,
  FIXTURE_MANIFEST_FILE,
  FIXTURE_MANIFEST_SCHEMA,
  validateFixtureManifest,
  type FixtureFile,
  type FixtureManifest,
} from "../src/parity/fixture";

const EXAMPLE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCES_PATH = join(EXAMPLE_DIR, "parity", "model-sources.json");

interface PinnedFile {
  path: string;
  sha256: string;
  bytes: number;
}

interface PinnedRepo {
  repo: string;
  revision: string;
  files: PinnedFile[];
}

interface PinnedCoreML extends PinnedRepo {
  /** Directory of the `.mlpackage` inside the repo, e.g. `model.mlpackage`. */
  package_dir: string;
  /** Path of the id2label JSON inside the repo. */
  id2label: string;
}

export interface ModelSources {
  android: PinnedRepo & { backend: "onnx" };
  ios: {
    backend: "coreml";
    tokenizer_name: string;
    tokenizer: PinnedRepo;
    /** Null until a maintainer publishes and pins a Core ML export. */
    coreml: PinnedCoreML | null;
  };
}

/** Thrown while no Core ML export is pinned for the iOS fixture. */
export const COREML_NOT_PINNED =
  "no pinned Core ML export: GitHub Actions must not convert models, so a " +
  "maintainer must convert and publish it locally (docs/security/hf-token-policy.md) " +
  "and pin it in parity/model-sources.json";

export async function loadModelSources(): Promise<ModelSources> {
  const sources = JSON.parse(await readFile(SOURCES_PATH, "utf8")) as ModelSources;
  const repos: PinnedRepo[] = [sources.android, sources.ios.tokenizer];
  if (sources.ios.coreml !== null) {
    repos.push(sources.ios.coreml);
  }
  for (const repo of repos) {
    if (!/^[0-9a-f]{40}$/.test(repo.revision)) {
      throw new Error("model source revision must be an immutable commit sha");
    }
    for (const file of repo.files) {
      if (
        !/^[0-9a-f]{64}$/.test(file.sha256) ||
        !(file.bytes > 0) ||
        file.path.split("/").some((segment) => segment === ".." || segment === "")
      ) {
        throw new Error("model source files must pin a safe path, sha256 and bytes");
      }
    }
  }
  return sources;
}

/** Hash a file and return its sha256 and size. */
export async function hashFile(path: string): Promise<{ sha256: string; bytes: number }> {
  const hash = createHash("sha256");
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    const buffer = chunk as Buffer;
    hash.update(buffer);
    bytes += buffer.length;
  }
  return { sha256: hash.digest("hex"), bytes };
}

async function download(repo: PinnedRepo, cacheDir: string): Promise<void> {
  await mkdir(cacheDir, { recursive: true });
  for (const file of repo.files) {
    const target = join(cacheDir, file.path);
    if (await matchesPin(target, file)) {
      continue;
    }
    const url = `https://huggingface.co/${repo.repo}/resolve/${repo.revision}/${file.path}`;
    const response = await fetch(url, { redirect: "follow" });
    if (!response.ok || response.body === null) {
      throw new Error(`download failed for pinned file ${file.path}: HTTP ${response.status}`);
    }
    await mkdir(dirname(target), { recursive: true });
    await pipeline(
      Readable.fromWeb(response.body as import("node:stream/web").ReadableStream),
      createWriteStream(target),
    );
    if (!(await matchesPin(target, file))) {
      await rm(target, { force: true });
      throw new Error(`pinned digest mismatch for ${file.path}`);
    }
  }
}

async function matchesPin(path: string, pin: PinnedFile): Promise<boolean> {
  try {
    const actual = await hashFile(path);
    return actual.sha256 === pin.sha256 && actual.bytes === pin.bytes;
  } catch {
    return false;
  }
}

async function requirePinned(sourceDir: string, files: readonly PinnedFile[]): Promise<void> {
  for (const file of files) {
    if (!(await matchesPin(join(sourceDir, file.path), file))) {
      throw new Error(`source file does not match its pin: ${file.path}`);
    }
  }
}

/** List every regular file below `root` as `/`-separated relative paths. */
export async function listFiles(root: string): Promise<string[]> {
  const entries = await readdir(root, { recursive: true, withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    if (entry.isSymbolicLink()) {
      throw new Error("fixture trees must not contain symbolic links");
    }
    if (entry.isFile()) {
      const parent = (entry as { parentPath?: string; path?: string }).parentPath ??
        (entry as { path?: string }).path ??
        root;
      files.push(relative(root, join(parent, entry.name)).split(sep).join("/"));
    }
  }
  return files.sort();
}

async function copyTree(source: string, target: string): Promise<void> {
  for (const file of await listFiles(source)) {
    await mkdir(dirname(join(target, file)), { recursive: true });
    await copyFile(join(source, file), join(target, file));
  }
}

/** Write the manifest for a staged fixture directory and return it. */
export async function writeFixtureManifest(
  outDir: string,
  fields: Omit<FixtureManifest, "schema" | "version" | "files" | "bundle_digest">,
): Promise<FixtureManifest> {
  const files: FixtureFile[] = [];
  for (const path of await listFiles(join(outDir, "model"))) {
    const { sha256, bytes } = await hashFile(join(outDir, "model", path));
    files.push({ path: `model/${path}`, sha256, bytes });
  }
  const manifest: FixtureManifest = {
    schema: FIXTURE_MANIFEST_SCHEMA,
    version: 1,
    ...fields,
    files,
    bundle_digest: bundleDigest(files),
  };
  validateFixtureManifest(manifest);
  await writeFile(
    join(outDir, FIXTURE_MANIFEST_FILE),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );
  return manifest;
}

/** Re-verify a staged fixture the same way the native probes do. */
export async function verifyFixture(fixtureDir: string): Promise<FixtureManifest> {
  const manifest = validateFixtureManifest(
    JSON.parse(await readFile(join(fixtureDir, FIXTURE_MANIFEST_FILE), "utf8")),
  );
  const listed = manifest.files.map((file) => file.path).sort();
  const found = (await listFiles(join(fixtureDir, "model"))).map((path) => `model/${path}`);
  if (JSON.stringify(listed) !== JSON.stringify(found)) {
    throw new Error("fixture file set does not match its manifest");
  }
  for (const file of manifest.files) {
    const actual = await hashFile(join(fixtureDir, file.path));
    if (actual.sha256 !== file.sha256 || actual.bytes !== file.bytes) {
      throw new Error("fixture file digest does not match its manifest");
    }
  }
  return manifest;
}

async function assembleAndroid(sourceDir: string, outDir: string): Promise<FixtureManifest> {
  const sources = await loadModelSources();
  await requirePinned(sourceDir, sources.android.files);
  await rm(outDir, { recursive: true, force: true });
  for (const file of sources.android.files) {
    await mkdir(dirname(join(outDir, "model", file.path)), { recursive: true });
    await copyFile(join(sourceDir, file.path), join(outDir, "model", file.path));
  }
  return writeFixtureManifest(outDir, {
    platform: "android",
    backend: "onnx",
    model_path: "model",
    id2label_path: "model/id2label.json",
    tokenizer_folder_path: null,
    tokenizer_name: null,
    source: { repo: sources.android.repo, revision: sources.android.revision },
  });
}

async function assembleIos(sourceDir: string, outDir: string): Promise<FixtureManifest> {
  const sources = await loadModelSources();
  const coreml = sources.ios.coreml;
  if (coreml === null) {
    throw new Error(COREML_NOT_PINNED);
  }
  const tokenizerDir = join(sourceDir, "tokenizer");
  const coremlDir = join(sourceDir, "coreml");
  await requirePinned(tokenizerDir, sources.ios.tokenizer.files);
  await requirePinned(coremlDir, coreml.files);
  if (!(await stat(join(coremlDir, coreml.package_dir))).isDirectory()) {
    throw new Error("the pinned Core ML package_dir is not a directory");
  }
  await rm(outDir, { recursive: true, force: true });
  await copyTree(join(coremlDir, coreml.package_dir), join(outDir, "model", "model.mlpackage"));
  await mkdir(join(outDir, "model", "tokenizer"), { recursive: true });
  await copyFile(join(coremlDir, coreml.id2label), join(outDir, "model", "id2label.json"));
  for (const file of sources.ios.tokenizer.files) {
    await copyFile(join(tokenizerDir, file.path), join(outDir, "model", "tokenizer", file.path));
  }
  return writeFixtureManifest(outDir, {
    platform: "ios",
    backend: "coreml",
    model_path: "model/model.mlpackage",
    id2label_path: "model/id2label.json",
    tokenizer_folder_path: "model/tokenizer",
    tokenizer_name: sources.ios.tokenizer_name,
    source: { repo: coreml.repo, revision: coreml.revision },
  });
}

function option(args: readonly string[], name: string): string {
  const index = args.indexOf(`--${name}`);
  const value = index >= 0 ? args[index + 1] : undefined;
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`missing --${name}`);
  }
  return value;
}

async function main(args: readonly string[]): Promise<void> {
  const [command] = args;
  switch (command) {
    case "download": {
      const sources = await loadModelSources();
      const platform = option(args, "platform");
      const cache = resolve(option(args, "cache"));
      if (platform === "android") {
        await download(sources.android, cache);
        return;
      }
      if (sources.ios.coreml === null) {
        throw new Error(COREML_NOT_PINNED);
      }
      await download(sources.ios.tokenizer, join(cache, "tokenizer"));
      await download(sources.ios.coreml, join(cache, "coreml"));
      return;
    }
    case "assemble": {
      const platform = option(args, "platform");
      const source = resolve(option(args, "source"));
      const out = resolve(option(args, "out"));
      const manifest =
        platform === "android"
          ? await assembleAndroid(source, out)
          : await assembleIos(source, out);
      console.log(`staged ${manifest.platform} fixture ${manifest.bundle_digest}`);
      return;
    }
    case "verify": {
      const manifest = await verifyFixture(resolve(option(args, "fixture")));
      console.log(`verified ${manifest.platform} fixture ${manifest.bundle_digest}`);
      return;
    }
    default:
      throw new Error("usage: stage-model-fixture.ts download|assemble|verify ...");
  }
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : "stage-model-fixture failed");
    process.exitCode = 1;
  });
}
